/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file neuropress_device_path_parity.cu
 * @brief Cross-examines Clio's DEVICE-RESIDENT selection path, stage by stage,
 *        against the upstream functions it is modelled on.
 *
 * The other harnesses check that the two projects compute the same NUMBERS.
 * This one checks that Clio reaches them the same WAY: upstream's inference
 * phase never brings a chunk's statistics to the host --
 * runStatsKernelsNoSync returns an AutoStatsGPU* and gpucompress_infer_gpu
 * passes that device pointer to the network ("Stats remain on GPU -- NN
 * inference reads d_stats_ptr directly on device",
 * gpucompress_compress.cpp:281). Clio's original path returned host doubles,
 * so it copied the histogram and scalars down, summed the entropy in a host
 * loop, and uploaded a feature matrix back.
 *
 * Function-by-function correspondence being checked here:
 *
 *   upstream                            Clio
 *   ------------------------------      ------------------------------------
 *   runStatsKernelsNoSync               ComputeDeviceStatsResident
 *     statsPass1Kernel                    StatsPass1Kernel
 *     launchEntropyKernelsAsync           EntropyFromHistKernel
 *       (writes &d_stats->entropy)          (writes &d_stats->entropy)
 *     madPass2Kernel                      StatsPass2DevKernel
 *       (mean read on-device)               (mean read on-device)
 *     finalizeStatsOnlyKernel             FinalizeFeatureStatsKernel
 *   -> AutoStatsGPU*  (device)          -> DeviceFeatureStats* (device)
 *
 *   runNNFusedInferenceCtx              NeuroPressGpuInferBatchDeviceStats
 *     nnFusedInferenceKernel(d_stats)     InferKernelDeviceStats(d_stats)
 *     cudaStreamSynchronize(stream)       cudaStreamSynchronize(stream)
 *
 * Three things are asserted:
 *
 *   1. STATISTICS AGREE with upstream's, computed by upstream's own kernels
 *      on the same device bytes.
 *   2. THE DEVICE PATH AGREES WITH CLIO'S HOST PATH, so routing a chunk
 *      through the new path cannot change a selection. Predictions are
 *      compared per candidate, not just the winner.
 *   3. NOTHING SYNCHRONIZES IN THE MIDDLE. The stats call is required to
 *      leave work still pending on its stream -- if it has already completed
 *      when it returns, some host round trip crept back in.
 *
 * Built only when the NeuroPress source tree is present (see CMakeLists.txt).
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <set>
#include <string>
#include <vector>

#include "clio_ctp/compress/model/neuropress_nn_gpu_kernels.h"
#include "clio_ctp/compress/model/neuropress_nn_predictor.h"
#include "clio_ctp/compress/model/ranking.h"
#include "clio_ctp/compress/preprocess/data_stats_gpu.h"

// --- Native NeuroPress, compiled from its own source.
#include "stats/auto_stats_gpu.h"

namespace gpucompress {
AutoStatsGPU *runStatsKernelsNoSync(const void *d_input, size_t input_size,
                                    cudaStream_t stream);
}

using ctp::compress::model::CandidateConfig;
using ctp::compress::model::CompressionFeatures;
using ctp::compress::model::CompressionPrediction;
using ctp::compress::model::DataFeatures;
using ctp::compress::model::DefaultCandidates;
using ctp::compress::model::MakeCompressionFeatures;
using ctp::compress::model::NeuroPressNNPredictor;

namespace {

const std::set<int> kTrainedBaseIds = {13, 14, 15, 16, 17, 18, 23, 24};

int g_failures = 0;
long g_checks = 0;
long g_tied_pairs = 0;  // equal-score adjacent pairs seen in the ranking

void Check(bool cond, const std::string &what) {
  ++g_checks;
  if (!cond) {
    ++g_failures;
    std::printf("  [FAIL] %s\n", what.c_str());
  }
}

/** Relative difference, with an absolute floor for values near zero. */
double RelDiff(double a, double b) {
  const double denom = std::max({std::fabs(a), std::fabs(b), 1e-12});
  return std::fabs(a - b) / denom;
}

void CheckClose(double a, double b, double tol, const std::string &what) {
  ++g_checks;
  const double d = RelDiff(a, b);
  if (!(d <= tol)) {
    ++g_failures;
    std::printf("  [FAIL] %s: %.17g vs %.17g (rel %.3e > %.1e)\n", what.c_str(),
                a, b, d, tol);
  }
}

/** Deterministic float chunk with a few different statistical regimes. */
std::vector<float> MakeChunk(size_t n, int regime) {
  std::vector<float> v(n);
  uint32_t s = 0x9e3779b9u ^ static_cast<uint32_t>(regime * 2654435761u);
  for (size_t i = 0; i < n; ++i) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    const double u = static_cast<double>(s) / 4294967296.0;
    switch (regime) {
      case 0: v[i] = static_cast<float>(u * 2.0 - 1.0); break;
      case 1: v[i] = static_cast<float>(std::sin(i * 0.0005) + 0.02 * u); break;
      case 2: v[i] = static_cast<float>(i % 97) * 0.25f; break;
      case 3: v[i] = (i % 1024 < 512) ? 0.0f : static_cast<float>(u); break;
      default: v[i] = static_cast<float>(u * 1e5 - 5e4); break;
    }
  }
  return v;
}

}  // namespace

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::printf("No CUDA device -- skipping.\n");
    return 77;
  }

  NeuroPressNNPredictor predictor;
  if (!predictor.Load(CLIO_CTP_NEUROPRESS_WEIGHTS_DIR)) {
    std::printf("Could not load NeuroPress weights -- skipping.\n");
    return 77;
  }

  const size_t kElems = 1u << 20;  // 4 MiB of float32, a production chunk
  const size_t kBytes = kElems * sizeof(float);

  std::printf("=== Device stats: upstream kernels vs Clio's device path ===\n");

  for (int regime = 0; regime < 5; ++regime) {
    const std::vector<float> host = MakeChunk(kElems, regime);
    float *d_data = nullptr;
    if (cudaMalloc(&d_data, kBytes) != cudaSuccess) {
      std::printf("  [SKIP] regime %d: cudaMalloc failed\n", regime);
      continue;
    }
    cudaMemcpy(d_data, host.data(), kBytes, cudaMemcpyHostToDevice);

    char tag[64];
    std::snprintf(tag, sizeof(tag), "regime %d", regime);

    // ---- upstream, through its own kernels on its own stream.
    cudaStream_t up_stream = nullptr;
    cudaStreamCreate(&up_stream);
    AutoStatsGPU *d_up = gpucompress::runStatsKernelsNoSync(d_data, kBytes,
                                                            up_stream);
    Check(d_up != nullptr, std::string(tag) + ": upstream stats ran");
    AutoStatsGPU h_up{};
    if (d_up) {
      cudaStreamSynchronize(up_stream);
      cudaMemcpy(&h_up, d_up, sizeof(h_up), cudaMemcpyDeviceToHost);
    }

    // ---- Clio, device-resident path.
    void *stream = ctp::DeviceStatsStream();
    const void *d_clio = ctp::ComputeDeviceStatsResident(
        d_data, kElems, ctp::DataType::FLOAT32, stream);
    Check(d_clio != nullptr, std::string(tag) + ": Clio device stats ran");

    // (3) The call must NOT have synchronized. cudaStreamQuery reporting
    // cudaSuccess here would mean every kernel had already finished by the
    // time it returned, i.e. something waited -- which is exactly the
    // property this path exists to avoid. Checked before the read below,
    // since that one does synchronize (deliberately, and afterwards).
    const cudaError_t pending = cudaStreamQuery(
        static_cast<cudaStream_t>(stream));
    Check(pending == cudaErrorNotReady,
          std::string(tag) +
              ": stats left work pending (no mid-pipeline sync); query=" +
              std::to_string(static_cast<int>(pending)));

    double c_entropy = 0, c_mad = 0, c_deriv = 0;
    Check(ctp::ReadDeviceFeatureStats(d_clio, &c_entropy, &c_mad, &c_deriv,
                                      stream),
          std::string(tag) + ": read device stats back");

    // ---- Clio's original host-returning path, as the third opinion.
    double h_entropy = 0, h_mad = 0, h_deriv = 0;
    Check(ctp::ComputeDeviceStats(d_data, kElems, ctp::DataType::FLOAT32,
                                  &h_entropy, &h_mad, &h_deriv),
          std::string(tag) + ": Clio host-returning stats ran");

    if (d_up && d_clio) {
      // (1) Against upstream. Tolerance rather than bit equality: the two
      // reduce in different orders (upstream's MAD is a warp shuffle tree,
      // Clio's a shared-memory tree), which is a floating-point difference
      // the dataset harness already characterizes -- what matters here is
      // that moving entropy and the mean onto the device did not shift them.
      CheckClose(h_up.entropy, c_entropy, 1e-12,
                 std::string(tag) + ": entropy vs upstream");
      CheckClose(h_up.mad_normalized, c_mad, 1e-9,
                 std::string(tag) + ": MAD vs upstream");
      CheckClose(h_up.deriv_normalized, c_deriv, 1e-9,
                 std::string(tag) + ": second derivative vs upstream");

      // (2) Against Clio's own host path -- this must be much tighter, since
      // a routing change between the two would be a Clio-internal regression.
      CheckClose(h_entropy, c_entropy, 1e-14,
                 std::string(tag) + ": entropy, device vs host path");
      CheckClose(h_mad, c_mad, 1e-12,
                 std::string(tag) + ": MAD, device vs host path");
      CheckClose(h_deriv, c_deriv, 1e-12,
                 std::string(tag) + ": second derivative, device vs host path");

      std::printf("  %s: entropy %.12f  mad %.9f  d2 %.9f\n", tag, c_entropy,
                  c_mad, c_deriv);
    }

    // ---- Inference: device-stats kernel vs the host-matrix kernel.
    DataFeatures data;
    data.chunk_size_bytes = static_cast<double>(kBytes);
    data.shannon_entropy = c_entropy;
    data.mad = c_mad;
    data.second_derivative_mean = c_deriv;
    data.data_type_char = 0.0;
    data.data_type_float = 1.0;

    // The TRAINED eight only, as the bridge does. Anything outside the action
    // space maps through NeuroPressAlgoIdForBaseId's fallback and collides.
    std::vector<CandidateConfig> cands;
    for (const auto &c : DefaultCandidates(/*include_gpu=*/true, {2}, false,
                                           1e-3, /*include_cpu=*/false)) {
      if (kTrainedBaseIds.count(c.base_id)) cands.push_back(c);
    }
    // Ordered by ACTION INDEX, as the bridge now orders it. This matters for
    // the comparison below: the kernel breaks ties on the action index
    // (upstream's rule), while ScoreAndSort breaks them on position. The two
    // rules coincide only when position IS the action index -- which is the
    // whole reason the bridge sorts. Feeding an unsorted list here makes them
    // disagree on exactly the zstd/deflate ties this fix was about.
    std::stable_sort(cands.begin(), cands.end(),
                     [](const CandidateConfig &a, const CandidateConfig &b) {
                       return ctp::compress::model::NeuroPressAlgoIdForBaseId(
                                  a.base_id) <
                              ctp::compress::model::NeuroPressAlgoIdForBaseId(
                                  b.base_id);
                     });
    std::vector<CompressionFeatures> feats;
    feats.reserve(cands.size());
    for (const auto &c : cands) feats.push_back(MakeCompressionFeatures(data, c));

    const std::vector<CompressionPrediction> host_preds =
        predictor.PredictBatch(feats);
    const std::vector<CompressionPrediction> dev_preds =
        predictor.PredictBatchDeviceStats(d_clio, feats, stream);

    Check(dev_preds.size() == host_preds.size(),
          std::string(tag) + ": device inference returned a full batch");
    if (dev_preds.size() == host_preds.size()) {
      for (size_t i = 0; i < dev_preds.size(); ++i) {
        const std::string w = std::string(tag) + " cand " + std::to_string(i);
        // The two kernels share NeuroPressForwardShared, and the only
        // difference is where the eight inputs are read from -- so these
        // should agree exactly, modulo the float32 round trip that the
        // device-stats path applies to the three statistics.
        CheckClose(host_preds[i].compression_ratio,
                   dev_preds[i].compression_ratio, 1e-5, w + ": ratio");
        CheckClose(host_preds[i].compression_time_ms,
                   dev_preds[i].compression_time_ms, 1e-5, w + ": comp time");
        CheckClose(host_preds[i].decompression_time_ms,
                   dev_preds[i].decompression_time_ms, 1e-5,
                   w + ": decomp time");
        CheckClose(host_preds[i].psnr_db, dev_preds[i].psnr_db, 1e-5,
                   w + ": psnr");
      }
    }

    // ---- Ranking: the GPU order must equal the host ScoreAndSort order.
    //
    // This moved from the host to a kernel, so the thing to prove is that it
    // decides the SAME thing -- upstream ranks in-kernel (nn_gpu.cu:499-532)
    // and Clio now does too, but a reordering here would silently change which
    // algorithm every chunk is compressed with.
    {
      ctp::compress::model::RankingWeights rw;
      rw.use_cost_model = true;
      std::vector<int> gpu_order;
      const auto ranked_preds = predictor.PredictBatchDeviceStats(
          d_clio, feats, stream, &rw, &gpu_order);

      Check(gpu_order.size() == cands.size(),
            std::string(tag) + ": GPU ranking returned a full permutation");

      const auto host_ranked =
          ctp::compress::model::CompressionPredictor::ScoreAndSort(
              data, cands, ranked_preds, rw);

      if (gpu_order.size() == cands.size() &&
          host_ranked.size() == cands.size()) {
        // Compare candidate identity slot by slot, not just the winner: an
        // ordering that agrees on first place and disagrees below it still
        // changes what the runtime logs and what training later joins against.
        bool same = true;
        for (size_t i = 0; i < host_ranked.size(); ++i) {
          const auto &h = host_ranked[i].candidate;
          const auto &g = cands[static_cast<size_t>(gpu_order[i])];
          if (h.base_id != g.base_id || h.preset_id != g.preset_id ||
              h.byte_shuffle != g.byte_shuffle || h.quantize != g.quantize) {
            same = false;
            std::printf(
                "  [FAIL] %s: rank %zu differs -- host base_id %d shuf %d, "
                "gpu base_id %d shuf %d\n",
                tag, i, h.base_id, static_cast<int>(h.byte_shuffle), g.base_id,
                static_cast<int>(g.byte_shuffle));
            break;
          }
        }
        Check(same, std::string(tag) + ": GPU order == host ScoreAndSort order");

        // And the permutation must be one: every slot exactly once. A bitonic
        // network that drops or duplicates a lane would still produce a
        // plausible-looking winner.
        std::vector<int> seen(cands.size(), 0);
        for (int slot : gpu_order) {
          if (slot >= 0 && static_cast<size_t>(slot) < seen.size()) ++seen[slot];
        }
        bool perm = true;
        for (int c : seen) if (c != 1) perm = false;
        Check(perm, std::string(tag) + ": GPU order is a permutation");

        // Count the ties this input actually produced. Without this the check
        // above is weaker than it looks: on a candidate set with no equal
        // costs, ANY correct sort agrees, so the comparison would pass whether
        // or not the two sides break ties the same way. An earlier tie-break
        // harness on this project passed for exactly that reason.
        for (size_t i = 0; i + 1 < host_ranked.size(); ++i) {
          if (host_ranked[i].score == host_ranked[i + 1].score) ++g_tied_pairs;
        }
      }
    }

    cudaStreamDestroy(up_stream);
    cudaFree(d_data);
  }

  // ---- Upstream's tie rule, on an enumeration that DISAGREES with it.
  //
  // The check above compares the kernel against ScoreAndSort on an
  // action-ordered list, where "lowest action" and "first enumerated" happen
  // to be the same answer -- so it cannot tell the two rules apart. Feed the
  // candidates in REVERSE action order and they diverge: first-enumerated
  // picks the highest action, upstream picks the lowest. This is the test the
  // old tie-break harness was missing, and the reason the zstd/deflate
  // divergence survived 1,530 passing checks.
  std::printf("\n=== Tie rule: lowest ACTION index wins ===\n");
  {
    const std::vector<float> host = MakeChunk(kElems, 2);
    float *d_data = nullptr;
    cudaMalloc(&d_data, kBytes);
    cudaMemcpy(d_data, host.data(), kBytes, cudaMemcpyHostToDevice);
    void *stream = ctp::DeviceStatsStream();
    const void *d_stats = ctp::ComputeDeviceStatsResident(
        d_data, kElems, ctp::DataType::FLOAT32, stream);

    DataFeatures data;
    data.chunk_size_bytes = static_cast<double>(kBytes);
    data.data_type_float = 1.0;

    std::vector<CandidateConfig> rev;
    for (const auto &c : DefaultCandidates(true, {2}, false, 1e-3, false)) {
      if (kTrainedBaseIds.count(c.base_id)) rev.push_back(c);
    }
    // DESCENDING action index -- the adversarial order.
    std::stable_sort(rev.begin(), rev.end(),
                     [](const CandidateConfig &a, const CandidateConfig &b) {
                       return ctp::compress::model::NeuroPressAlgoIdForBaseId(
                                  a.base_id) >
                              ctp::compress::model::NeuroPressAlgoIdForBaseId(
                                  b.base_id);
                     });
    std::vector<CompressionFeatures> rf;
    for (const auto &c : rev) rf.push_back(MakeCompressionFeatures(data, c));

    ctp::compress::model::RankingWeights rw;
    rw.use_cost_model = true;
    std::vector<int> order;
    std::vector<double> scores;
    const auto preds = predictor.PredictBatchDeviceStats(
        d_stats, rf, stream, &rw, &order, 0.0, &scores);

    long ties = 0, correct = 0;
    if (order.size() == rev.size()) {
      for (size_t i = 0; i + 1 < order.size(); ++i) {
        if (scores[i] != scores[i + 1]) continue;
        ++ties;
        const int a = ctp::compress::model::NeuroPressAlgoIdForBaseId(
            rev[static_cast<size_t>(order[i])].base_id);
        const int b = ctp::compress::model::NeuroPressAlgoIdForBaseId(
            rev[static_cast<size_t>(order[i + 1])].base_id);
        if (a < b) ++correct;
      }
    }
    std::printf("  %ld tied pairs on a reversed list, %ld resolved to the "
                "lower action\n", ties, correct);
    Check(ties > 0,
          "tie rule: the reversed list actually produced ties (otherwise this "
          "check proves nothing)");
    Check(ties == correct,
          "tie rule: every tie resolved to the LOWER action index, despite "
          "enumeration order saying the opposite");

    cudaFree(d_data);
  }

  // ---- The two -INFINITY masks, applied in-kernel as nn_gpu.cu:238-239 does.
  //
  // Upstream evaluates all 32 configs every time and masks rather than
  // omitting, so the properties to check are: a masked action never outranks
  // an unmasked one, and the set is still fully ranked (nothing dropped).
  std::printf("\n=== In-kernel masking ===\n");
  {
    const std::vector<float> host = MakeChunk(kElems, 1);
    float *d_data = nullptr;
    cudaMalloc(&d_data, kBytes);
    cudaMemcpy(d_data, host.data(), kBytes, cudaMemcpyHostToDevice);
    void *stream = ctp::DeviceStatsStream();
    const void *d_stats = ctp::ComputeDeviceStatsResident(
        d_data, kElems, ctp::DataType::FLOAT32, stream);

    // Full 32-action space, enumerated the way the bridge does it.
    DataFeatures data;
    data.chunk_size_bytes = static_cast<double>(kBytes);
    data.data_type_float = 1.0;
    // Restricted to the 8 TRAINED algorithms first, exactly as the bridge
    // does. Without that filter the set is 12 x 4 = 48, which overflows the
    // ranking warp -- 32 is upstream's NN_NUM_CONFIGS and the hard ceiling on
    // what the action space can be.
    std::vector<CandidateConfig> plain;
    for (const auto &c : DefaultCandidates(true, {2}, false, 1e-3, false)) {
      if (kTrainedBaseIds.count(c.base_id)) plain.push_back(c);
    }
    std::vector<CandidateConfig> full;
    for (int shuffle = 0; shuffle <= 1; ++shuffle) {
      for (int quant = 0; quant <= 1; ++quant) {
        for (const auto &b : plain) {
          CandidateConfig c = b;
          c.byte_shuffle = (shuffle != 0);
          c.quantize = (quant != 0);
          c.error_bound = 0.0;  // LOSSLESS: every quantize action is masked
          full.push_back(c);
        }
      }
    }
    std::vector<CompressionFeatures> feats;
    for (const auto &c : full) feats.push_back(MakeCompressionFeatures(data, c));

    ctp::compress::model::RankingWeights rw;
    rw.use_cost_model = true;
    std::vector<int> order;
    const auto preds =
        predictor.PredictBatchDeviceStats(d_stats, feats, stream, &rw, &order,
                                          /*min_psnr=*/0.0);

    Check(order.size() == full.size(),
          "quantize mask: full 32-action space is ranked, not truncated");
    if (order.size() == full.size()) {
      // Every unmasked (lossless) action must come before every masked
      // (quantize-at-eb-0) one. That is what -INFINITY buys: the action is
      // still present and still ordered, just never preferred.
      size_t first_quant = full.size(), last_plain = 0;
      for (size_t i = 0; i < order.size(); ++i) {
        if (full[static_cast<size_t>(order[i])].quantize) {
          if (i < first_quant) first_quant = i;
        } else {
          last_plain = i;
        }
      }
      Check(first_quant > last_plain,
            "quantize mask: no masked action outranks an unmasked one");
      Check(!full[static_cast<size_t>(order[0])].quantize,
            "quantize mask: the winner is a lossless action");
    }

    // LOSSY: with a positive bound the quantize half must NOT be masked.
    //
    // The mask input is the chunk's bound, and candidates carry it only when
    // they quantize -- candidate 0 never does. Reading the bound off candidate
    // 0 therefore reported 0 on every run and masked the quantize half even on
    // a lossy write, which no eb=0 test can see. This is that test.
    {
      std::vector<CandidateConfig> lossy = full;
      for (auto &c : lossy) c.error_bound = c.quantize ? 1e-3 : 0.0;
      std::vector<CompressionFeatures> lf;
      for (const auto &c : lossy) lf.push_back(MakeCompressionFeatures(data, c));

      std::vector<int> lossy_order;
      const auto lossy_preds = predictor.PredictBatchDeviceStats(
          d_stats, lf, stream, &rw, &lossy_order, /*min_psnr=*/0.0);
      Check(lossy_order.size() == lossy.size(),
            "lossy: full action space ranked");
      if (lossy_order.size() == lossy.size()) {
        // Not "a quantize action must win" -- the model decides that. The
        // claim is only that quantize actions are still IN the running, i.e.
        // not all shoved to the bottom by a mask that should not have fired.
        size_t first_quant = lossy.size(), last_plain = 0;
        for (size_t i = 0; i < lossy_order.size(); ++i) {
          if (lossy[static_cast<size_t>(lossy_order[i])].quantize) {
            if (i < first_quant) first_quant = i;
          } else {
            last_plain = i;
          }
        }
        Check(!(first_quant > last_plain),
              "lossy: quantize actions are NOT all masked below the lossless "
              "ones (the eb>0 mask must not fire)");
      }
    }

    // PSNR floor. Set it above every prediction so EVERY action is masked --
    // the case where masking and filtering genuinely differ. Upstream still
    // returns an action here; a host-side filter would have returned none.
    std::vector<int> order_all_masked;
    const auto preds2 = predictor.PredictBatchDeviceStats(
        d_stats, feats, stream, &rw, &order_all_masked, /*min_psnr=*/1e9);
    Check(order_all_masked.size() == full.size(),
          "psnr mask: everything masked still yields a full ranking");
    Check(!preds2.empty(),
          "psnr mask: an action is still returned when all are masked");

    cudaFree(d_data);
  }

  // ---- The no-silent-host-fallback rule.
  //
  // Upstream never answers from the host: a null d_stats_ptr or a failed
  // inference both end the call with GPUCOMPRESS_ERROR_NN_NOT_LOADED
  // (gpucompress_compress.cpp:285 and :208-212), and there is no CPU
  // implementation of the network in that project to fall back to. The device
  // entry points here must therefore REFUSE rather than quietly produce an
  // answer some other way -- a host-computed ranking would be indistinguishable
  // from a real one at the call site.
  std::printf("\n=== No silent fallback to the host ===\n");
  {
    DataFeatures data;
    data.chunk_size_bytes = static_cast<double>(kBytes);
    data.data_type_float = 1.0;
    std::vector<CandidateConfig> cands =
        DefaultCandidates(true, {2}, false, 1e-3, false);
    std::vector<CompressionFeatures> feats;
    for (const auto &c : cands) feats.push_back(MakeCompressionFeatures(data, c));

    Check(predictor.PredictBatchDeviceStats(nullptr, feats,
                                            ctp::DeviceStatsStream())
              .empty(),
          "null device stats -> empty, not a host-computed batch");
    Check(predictor
              .PredictBatchDeviceStats(reinterpret_cast<const void *>(0x1),
                                       {}, ctp::DeviceStatsStream())
              .empty(),
          "empty candidate set -> empty");

    // A zero-element chunk: upstream refuses it outright rather than
    // producing statistics for it (gpucompress_compress.cpp:274).
    float *d_tiny = nullptr;
    cudaMalloc(&d_tiny, sizeof(float));
    Check(ctp::ComputeDeviceStatsResident(d_tiny, 0, ctp::DataType::FLOAT32,
                                          ctp::DeviceStatsStream()) == nullptr,
          "zero-element chunk -> refused, no host substitute");
    Check(ctp::ComputeDeviceStatsResident(nullptr, 16, ctp::DataType::FLOAT32,
                                          ctp::DeviceStatsStream()) == nullptr,
          "null chunk -> refused");
    cudaFree(d_tiny);

    // THE INVARIANT: on a GPU-capable build, ready implies GPU. Load() now
    // refuses rather than returning a host-only predictor, so there is no
    // state in which this model answers from the CPU. That is what makes
    // "no silent CPU execution" a property of the object rather than a
    // promise about which branch callers happen to take -- and it is the
    // thing to assert, because the alternative (a ready predictor with no
    // device weights) is exactly what used to fall through to the host loop.
    Check(predictor.IsReady() == predictor.GpuInferenceActive(),
          "ready <=> GPU active: no ready-but-host-only state exists");
    Check(predictor.GpuInferenceActive(),
          "GpuInferenceActive() true on a device build with a device present");

    // A second predictor loaded from the same weights must reach the same
    // state -- Load() failing on one and succeeding on another would mean the
    // refusal depends on something other than device availability.
    NeuroPressNNPredictor second;
    const bool loaded = second.Load(CLIO_CTP_NEUROPRESS_WEIGHTS_DIR);
    Check(loaded && second.GpuInferenceActive(),
          "a second Load() also lands on the GPU, not the host port");
  }

  std::printf("\n  ranking saw %ld tied (equal-cost) adjacent pairs\n", g_tied_pairs);
  std::printf("\n===== %ld checks, %d failures =====\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
