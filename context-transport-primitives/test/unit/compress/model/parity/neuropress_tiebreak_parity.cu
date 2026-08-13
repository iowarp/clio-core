/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file neuropress_tiebreak_parity.cu
 * @brief Differential test: which candidate wins when several score equal.
 *
 * Ties are not a corner case here, they are routine. Both sides clamp hard
 * before ranking -- compression and decompression times floor at 1 ms and
 * the ratio caps at 100x (nn_gpu.cu:227-230) -- so on compressible data a
 * whole group of candidates saturates to a bit-identical cost. Which one
 * then wins is decided purely by position, and the two sides arrive at
 * position differently:
 *
 *   upstream  a bitonic ranking network over the 32 configs (nn_gpu.cu:344,
 *             :497) built from strict comparators, so it never swaps equal
 *             keys and the LOWEST action index survives.
 *   here      std::stable_sort over the candidate vector (predictor.h:382),
 *             so the FIRST ENUMERATED survives.
 *
 * Those agree only if Clio enumerates in ascending action-index order.
 * decodeAction (internal.hpp:167-172) numbers an action
 *
 *     algo + 8*quant + 16*shuffle
 *
 * which makes shuffle the outermost loop, quant the middle and algo the
 * innermost -- the order neuropress_bridge.cc:180-193 builds.
 *
 * Three phases:
 *
 *  1. UPSTREAM'S RULE, on real ties from the real model. Read the per-config
 *     costs out of the ranking kernel (out_predicted_costs), find every
 *     config sharing the winner's cost, and check upstream returned the
 *     minimum action index among them. This is the claim Clio's ordering
 *     rests on, checked against upstream directly and without reference to
 *     Clio. Clio is then required to select the same action.
 *
 *  2. CLIO'S RULE, on ties chosen to be ORDER-DISCRIMINATING. Phase 1 alone
 *     is not enough and the reason is worth recording: the real model's ties
 *     turn out to cluster on a low algorithm index (action 11 dominates
 *     them), and both the correct enumeration and an inverted one put such
 *     an action first, so they resolve identically either way -- reversing
 *     the order leaves phase 1 at zero failures. Ties only discriminate when
 *     the lowest ACTION index sits at a high ALGORITHM index, e.g. {7, 16}:
 *     ascending order picks 7, algo-outermost order reaches algo 0 first and
 *     picks 16. A stub predictor supplies exactly those tie sets through the
 *     real Rank(), so the ordering is measured rather than assumed.
 *
 *  3. GUARD. Count the order-discriminating cases and fail if there are
 *     none. A tie-break test run on inputs that cannot distinguish two
 *     orderings passes against both, which is precisely how the earlier
 *     harnesses missed this.
 *
 * Built only when the NeuroPress source tree is present (see CMakeLists.txt);
 * this is a cross-project check, not part of the default build.
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <set>
#include <string>
#include <vector>

#include "clio_ctp/compress/model/neuropress_nn_predictor.h"
#include "clio_ctp/compress/model/predictor.h"

// --- Native NeuroPress (compiled from its own source, namespace gpucompress).
#include "api/internal.hpp"

extern cudaStream_t g_sgd_stream;
extern cudaEvent_t g_sgd_done;

using ctp::compress::model::CandidateConfig;
using ctp::compress::model::CompressionFeatures;
using ctp::compress::model::CompressionPredictor;
using ctp::compress::model::CompressionPrediction;
using ctp::compress::model::DataFeatures;
using ctp::compress::model::ModelType;
using ctp::compress::model::NeuroPressNNPredictor;
using ctp::compress::model::RankedPrediction;
using ctp::compress::model::RankingWeights;

namespace {

int g_failures = 0;
int g_checks = 0;
int g_tied_cases = 0;           // real cases whose winner's cost was shared
int g_discriminating_cases = 0; // ... of which the two orders disagree
int g_max_tie_width = 0;
long g_total_cases = 0;

void Check(bool cond, const std::string &what) {
  ++g_checks;
  if (!cond) {
    ++g_failures;
    std::printf("  [FAIL] %s\n", what.c_str());
  }
}

/** NeuroPress action index (0-7) -> Clio ML base_id. Inverse of
 *  NeuroPressAlgoIdForBaseId() in neuropress_nn_predictor.cc. */
int BaseIdForAlgoIdx(int algo_idx) {
  switch (algo_idx) {
    case 0: return 13;  // nvcomp-lz4
    case 1: return 14;  // nvcomp-snappy
    case 2: return 17;  // nvcomp-deflate
    case 3: return 16;  // nvcomp-gdeflate
    case 4: return 15;  // nvcomp-zstd
    case 5: return 18;  // nvcomp-ans
    case 6: return 23;  // nvcomp-cascaded
    default: return 24; // nvcomp-bitcomp
  }
}

int AlgoIdxForBaseId(int base_id) {
  for (int i = 0; i < 8; ++i) {
    if (BaseIdForAlgoIdx(i) == base_id) return i;
  }
  return -1;
}

int ActionOf(const CandidateConfig &c) {
  return AlgoIdxForBaseId(c.base_id) + 8 * (c.quantize ? 1 : 0) +
         16 * (c.byte_shuffle ? 1 : 0);
}

/**
 * Candidate enumeration in upstream's action order: shuffle outermost, then
 * quant, then algo, so position in this vector is ascending action index.
 * Mirrors neuropress_bridge.cc:180-193.
 *
 * `reverse_order` inverts it to algo-outermost -- the same candidates,
 * permuted. It is the falsification lever for the whole harness: with it
 * set, every order-discriminating tie must resolve differently.
 */
std::vector<CandidateConfig> BuildCandidates(double error_bound,
                                             bool reverse_order) {
  std::vector<CandidateConfig> cands;
  auto push = [&](int shuffle, int quant, int algo) {
    if (quant == 1 && !(error_bound > 0.0)) return;
    CandidateConfig c;
    c.base_id = BaseIdForAlgoIdx(algo);
    c.preset_id = 2;  // BALANCED; preset is not an NN input
    c.byte_shuffle = (shuffle != 0);
    c.quantize = (quant != 0);
    c.error_bound = (quant != 0) ? error_bound : 0.0;
    cands.push_back(c);
  };
  if (!reverse_order) {
    for (int shuffle = 0; shuffle <= 1; ++shuffle) {
      for (int quant = 0; quant <= 1; ++quant) {
        for (int algo = 0; algo < 8; ++algo) push(shuffle, quant, algo);
      }
    }
  } else {
    for (int algo = 0; algo < 8; ++algo) {
      for (int quant = 0; quant <= 1; ++quant) {
        for (int shuffle = 0; shuffle <= 1; ++shuffle) {
          push(shuffle, quant, algo);
        }
      }
    }
  }
  return cands;
}

/** Which member of `tied` a given enumeration order reaches first. */
int WinnerUnderOrder(const std::set<int> &tied, double error_bound,
                     bool reverse_order) {
  for (const auto &c : BuildCandidates(error_bound, reverse_order)) {
    if (tied.count(ActionOf(c))) return ActionOf(c);
  }
  return -1;
}

/** True when the correct and inverted enumerations disagree on this tie --
 *  i.e. when the case can actually detect an ordering mistake. */
bool IsOrderDiscriminating(const std::set<int> &tied, double error_bound) {
  return WinnerUnderOrder(tied, error_bound, false) !=
         WinnerUnderOrder(tied, error_bound, true);
}

/**
 * Predictor that saturates a chosen set of actions to an identical score and
 * leaves every other candidate strictly worse.
 *
 * Deliberately drives the REAL Rank() (this class does not override it) so
 * the stable_sort and the candidate ordering under test are the shipped
 * ones, not a copy. Winners return the clamped extremes -- 1 ms times, 100x
 * ratio -- so their costs are equal by construction rather than by luck,
 * exactly as the policy clamps make them equal on real data.
 */
class TiePredictor : public CompressionPredictor {
 public:
  explicit TiePredictor(std::set<int> winners) : winners_(std::move(winners)) {}

  bool Load(const std::string &) override { return true; }
  bool Save(const std::string &) override { return true; }
  bool IsReady() const override { return true; }
  ModelType Type() const override { return ModelType::kQTable; }

  CompressionPrediction Predict(const CompressionFeatures &f) override {
    const int algo = AlgoIdxForBaseId(static_cast<int>(f.library_config_id) / 10);
    const int action = algo + 8 * (f.quantize > 0.5 ? 1 : 0) +
                       16 * (f.byte_shuffle > 0.5 ? 1 : 0);
    CompressionPrediction p;
    if (winners_.count(action)) {
      p.compression_ratio = 100.0;      // at the cap
      p.compression_time_ms = 1.0;      // at the floor
      p.decompression_time_ms = 1.0;    // at the floor
    } else {
      p.compression_ratio = 2.0;
      p.compression_time_ms = 50.0;
      p.decompression_time_ms = 50.0;
    }
    p.psnr_db = 0.0;
    return p;
  }

 private:
  std::set<int> winners_;
};

struct Chunk {
  std::string label;
  double entropy;
  double mad;
  double deriv;
  size_t size_bytes;
};

/** Grid sweep. Ties come from the policy clamps, so they cluster where the
 *  model predicts a saturating ratio or sub-millisecond times; rather than
 *  hand-pick, sweep and let the counters report what was reached. */
std::vector<Chunk> BuildCorpus() {
  const double entropies[] = {0.05, 0.10, 0.20, 0.35, 0.60, 0.90, 1.30,
                              1.80, 2.50, 3.40, 4.50, 5.60, 6.80, 7.90};
  const size_t sizes[] = {256u << 10, 1u << 20, 4u << 20, 16u << 20,
                          64u << 20};
  std::vector<Chunk> out;
  for (double e : entropies) {
    for (size_t sz : sizes) {
      Chunk c;
      c.label = "e" + std::to_string(e).substr(0, 4) + "/" +
                std::to_string(sz >> 10) + "K";
      c.entropy = e;
      // MAD and curvature track entropy: smooth data has small deviations.
      c.mad = e * 0.12;
      c.deriv = e * 0.06;
      c.size_bytes = sz;
      out.push_back(c);
    }
  }
  // Second family, with MAD and curvature far below what the entropy alone
  // would suggest. This is where the policy clamps actually bind and real
  // ties appear -- the coupled family above never reaches them, and a sweep
  // built only from it reports zero ties over hundreds of cases.
  const double flat_mads[] = {0.001, 0.005, 0.010, 0.020};
  for (double e : {0.05, 0.10, 0.20, 0.30, 0.60, 1.00}) {
    for (double m : flat_mads) {
      for (size_t sz : sizes) {
        Chunk c;
        c.label = "flat-e" + std::to_string(e).substr(0, 4) + "/m" +
                  std::to_string(m).substr(0, 5) + "/" +
                  std::to_string(sz >> 10) + "K";
        c.entropy = e;
        c.mad = m;
        c.deriv = m * 0.5;
        c.size_bytes = sz;
        out.push_back(c);
      }
    }
  }
  return out;
}

/** The upstream inference context: only the buffers the fused kernel reads. */
struct NativeInfer {
  CompContext ctx{};
  AutoStatsGPU *d_stats = nullptr;

  bool Init() {
    if (cudaMalloc(&d_stats, sizeof(AutoStatsGPU)) != cudaSuccess) return false;
    if (cudaMalloc(&ctx.d_fused_infer_output,
                   sizeof(NNInferenceOutput)) != cudaSuccess) return false;
    if (cudaMalloc(&ctx.d_fused_top_actions,
                   NN_NUM_CONFIGS * sizeof(int)) != cudaSuccess) return false;
    if (cudaMalloc(&ctx.d_fused_costs,
                   NN_NUM_CONFIGS * sizeof(float)) != cudaSuccess) return false;
    return true;
  }

  void Free() {
    cudaFree(d_stats);
    cudaFree(ctx.d_fused_infer_output);
    cudaFree(ctx.d_fused_top_actions);
    cudaFree(ctx.d_fused_costs);
  }

  /** @return selected action, or -1 on failure. costs[] is per-config. */
  int Infer(const Chunk &c, double error_bound, cudaStream_t stream,
            float *costs) {
    AutoStatsGPU h_stats{};
    h_stats.entropy = c.entropy;
    h_stats.mad_normalized = c.mad;
    h_stats.deriv_normalized = c.deriv;
    h_stats.num_elements = c.size_bytes / sizeof(float);
    cudaMemcpy(d_stats, &h_stats, sizeof(h_stats), cudaMemcpyHostToDevice);

    int action = -1;
    float ratio = 0, ct = 0, dt = 0, psnr = 0;
    int top[NN_NUM_CONFIGS] = {0};
    int rc = gpucompress::runNNFusedInferenceCtx(
        d_stats, c.size_bytes, error_bound, stream, &ctx, &action, &ratio,
        &ct, &dt, &psnr, top, costs);
    // runNNFusedInferenceCtx RETURNS the action (nn_gpu.cu:2265), so a
    // negative return is the only failure signal; a zero return is action 0.
    return (rc < 0) ? -1 : action;
  }
};

}  // namespace

int main(int argc, char **argv) {
  const std::string weights_dir =
      (argc > 1 && argv[1][0] != '-') ? argv[1]
                                      : std::string(CLIO_CTP_NEUROPRESS_WEIGHTS_DIR);
  const std::string nnwt = weights_dir + "/model.nnwt";
  bool reverse_order = false;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--reverse") reverse_order = true;
  }

  int dev_count = 0;
  if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) {
    std::printf("No CUDA device -- skipping tie-break parity check.\n");
    return 77;  // ctest SKIP_RETURN_CODE
  }

  cudaStreamCreate(&g_sgd_stream);
  cudaEventCreate(&g_sgd_done);

  if (!gpucompress::loadNNFromBinary(nnwt.c_str())) {
    std::printf("FATAL: native NeuroPress failed to load %s\n", nnwt.c_str());
    return 1;
  }
  NeuroPressNNPredictor clio;
  if (!clio.Load(weights_dir) || !clio.IsReady()) {
    std::printf("FATAL: Clio predictor failed to load %s\n",
                weights_dir.c_str());
    return 1;
  }
  std::printf("Both sides loaded %s\n", nnwt.c_str());
  if (reverse_order) {
    std::printf("*** --reverse: enumerating algo-outermost (falsification)\n");
  }

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  NativeInfer native;
  if (!native.Init()) {
    std::printf("FATAL: could not allocate native inference context\n");
    return 1;
  }

  const std::vector<Chunk> corpus = BuildCorpus();
  const double kBounds[] = {0.0, 1e-6, 1e-3, 1e-2};
  const size_t kNumBounds = sizeof(kBounds) / sizeof(kBounds[0]);

  RankingWeights weights;
  weights.use_cost_model = true;  // same policy the bridge opts into

  // ---- Phase 1: real ties from the real model. ----
  std::printf("\n[phase 1] real-model ties over %zu chunks x %zu bounds\n",
              corpus.size(), kNumBounds);
  for (size_t bi = 0; bi < kNumBounds; ++bi) {
    const double eb = kBounds[bi];
    int tied_here = 0, disc_here = 0;
    for (const auto &c : corpus) {
      ++g_total_cases;
      float costs[NN_NUM_CONFIGS];
      for (int i = 0; i < NN_NUM_CONFIGS; ++i) costs[i] = INFINITY;
      const int n_action = native.Infer(c, eb, stream, costs);
      const std::string where = c.label + " eb=" + std::to_string(eb);
      if (n_action < 0 || n_action >= NN_NUM_CONFIGS) {
        std::printf("  [FAIL] %s: native inference failed (action=%d)\n",
                    where.c_str(), n_action);
        ++g_failures;
        ++g_checks;
        continue;
      }

      // Every config sharing the winner's cost, exactly. Both values come
      // from the same kernel, so equality is the right test -- a tolerance
      // would invent ties the ranking network never saw.
      const float win_cost = costs[n_action];
      std::set<int> tied;
      for (int i = 0; i < NN_NUM_CONFIGS; ++i) {
        if (std::isfinite(costs[i]) && costs[i] == win_cost) tied.insert(i);
      }
      if (tied.size() > 1) {
        ++g_tied_cases;
        ++tied_here;
        g_max_tie_width =
            std::max(g_max_tie_width, static_cast<int>(tied.size()));
        if (IsOrderDiscriminating(tied, eb)) {
          ++g_discriminating_cases;
          ++disc_here;
        }
      }

      // Upstream's own rule, independent of Clio: lowest tied action wins.
      const int lowest_tied = tied.empty() ? n_action : *tied.begin();
      Check(n_action == lowest_tied,
            where + ": upstream returned the lowest tied action (got " +
                std::to_string(n_action) + ", lowest tied " +
                std::to_string(lowest_tied) + ")");

      // Clio must select the same action.
      DataFeatures data;
      data.chunk_size_bytes = static_cast<double>(c.size_bytes);
      data.shannon_entropy = c.entropy;
      data.mad = c.mad;
      data.second_derivative_mean = c.deriv;
      data.data_type_float = 1.0;
      data.data_type_char = 0.0;

      std::vector<CandidateConfig> cands = BuildCandidates(eb, reverse_order);
      std::vector<RankedPrediction> ranked = clio.Rank(data, cands, weights);
      if (ranked.empty()) {
        std::printf("  [FAIL] %s: Clio produced no ranking\n", where.c_str());
        ++g_failures;
        ++g_checks;
        continue;
      }
      const int c_action = ActionOf(ranked.front().candidate);
      Check(c_action == n_action,
            where + ": selected action (native " + std::to_string(n_action) +
                ", clio " + std::to_string(c_action) + ", tie width " +
                std::to_string(tied.size()) + ")");
    }
    std::printf("  eb=%-8g %3d tied, %3d of those order-discriminating\n", eb,
                tied_here, disc_here);
  }

  // ---- Phase 2: controlled, order-discriminating ties. ----
  // Each set is fed to the real Rank() through a stub predictor, and the
  // winner must be the lowest ACTION index -- upstream's rule as measured in
  // phase 1. The sets marked discriminating below resolve differently under
  // an algo-outermost enumeration, so they are what actually pins the order.
  std::printf("\n[phase 2] controlled ties through the real Rank()\n");
  struct TieCase { double eb; std::set<int> tied; };
  const std::vector<TieCase> tie_cases = {
      // Lossless (16 reachable configs: 0-7 and 16-23).
      {0.0, {7, 16}},        // discriminating: algo7 vs algo0
      {0.0, {6, 16, 22}},    // discriminating
      {0.0, {3, 19}},        // control: same algo, ascending wins either way
      {0.0, {0, 23}},        // control: action 0 is first under both
      // With a bound, all 32 configs are reachable.
      {1e-3, {1, 8}},        // discriminating: algo1 plain vs algo0 quantized
      {1e-3, {15, 16}},      // discriminating: algo7 quantized vs algo0 shuffled
      {1e-3, {7, 24}},       // discriminating
      {1e-3, {11, 12, 27}},  // control: algo3 reached first either way
      // Control: one algorithm's whole column. Algo-outermost reaches algo5
      // as a block and still yields 5, so a same-algo tie cannot detect an
      // ordering mistake however wide it is -- which is exactly the shape
      // the real model's ties take (see phase 1).
      {1e-3, {5, 13, 21, 29}},
  };
  for (const auto &tc : tie_cases) {
    TiePredictor stub(tc.tied);
    DataFeatures data;
    data.chunk_size_bytes = 4.0 * 1024 * 1024;
    data.shannon_entropy = 1.0;
    data.mad = 0.1;
    data.second_derivative_mean = 0.05;
    data.data_type_float = 1.0;
    data.data_type_char = 0.0;

    std::vector<CandidateConfig> cands = BuildCandidates(tc.eb, reverse_order);
    std::vector<RankedPrediction> ranked = stub.Rank(data, cands, weights);
    const int expected = *tc.tied.begin();  // lowest action index
    const bool disc = IsOrderDiscriminating(tc.tied, tc.eb);
    if (disc) ++g_discriminating_cases;

    std::string set_str;
    for (int a : tc.tied) set_str += " " + std::to_string(a);
    const int got = ranked.empty() ? -1 : ActionOf(ranked.front().candidate);
    Check(got == expected,
          "tie {" + set_str + " } eb=" + std::to_string(tc.eb) +
              (disc ? " [discriminating]" : " [control]") +
              ": winner should be lowest action " + std::to_string(expected) +
              ", got " + std::to_string(got));
  }

  // ---- Phase 3: the harness has to be able to fail. ----
  std::printf("\n[phase 3] discrimination guard\n");
  ++g_checks;
  if (g_discriminating_cases == 0) {
    ++g_failures;
    std::printf("  [FAIL] not one order-discriminating tie in %ld cases -- "
                "this harness would pass against either enumeration\n",
                g_total_cases);
  } else {
    std::printf("  %d order-discriminating cases (%d real ties over %ld "
                "cases, widest tie %d configs)\n",
                g_discriminating_cases, g_tied_cases, g_total_cases,
                g_max_tie_width);
  }

  native.Free();
  cudaStreamDestroy(stream);
  cudaStreamDestroy(g_sgd_stream);
  cudaEventDestroy(g_sgd_done);

  std::printf("\n===== %d checks, %d failures =====\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
