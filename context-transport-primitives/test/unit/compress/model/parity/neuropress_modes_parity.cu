/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file neuropress_modes_parity.cu
 * @brief 1 GiB / 16 MiB chunks, both projects, in all three operating modes.
 *
 * The sibling dataset harness sweeps in INFERENCE only, where frozen weights
 * mean a divergence cannot compound. Learning and exploration mutate the model
 * between chunks, where it can. Each mode is its own pass over the dataset.
 * The label is shared, since a real measurement comes from nvcomp on both
 * sides and cannot cause a divergence.
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "clio_ctp/compress/model/neuropress_nn_predictor.h"
#include "clio_ctp/compress/model/predictor.h"
#include "clio_ctp/compress/model/ranking.h"
#include "clio_ctp/compress/preprocess/data_stats_gpu.h"

#include "auto_stats_gpu.h"
#include "nn_weights.h"

namespace gpucompress {
bool loadNNFromBinary(const char *filepath);
void cleanupNN();
int runNNInference(double entropy, double mad_norm, double deriv_norm,
                   size_t data_size, double error_bound, cudaStream_t stream,
                   float *out_predicted_ratio, float *out_predicted_comp_time,
                   float *out_predicted_decomp_time, float *out_predicted_psnr,
                   int *out_top_actions);
int runNNSGD(const AutoStatsGPU *d_stats, const SGDSample *samples,
             int num_samples, size_t data_size, double error_bound,
             float learning_rate, cudaStream_t stream, float *out_grad_norm,
             int *out_clipped, int *out_count);
AutoStatsGPU *runStatsKernelsNoSync(const void *d_input, size_t input_size,
                                    cudaStream_t stream);
void freeStatsWorkspace();
}  // namespace gpucompress

extern cudaStream_t g_sgd_stream;
extern cudaEvent_t g_sgd_done;
extern float g_rank_w0;
extern float g_rank_w1;
extern float g_rank_w2;
extern float g_measured_bw_bytes_per_ms;

using ctp::compress::model::CandidateConfig;
using ctp::compress::model::CompressionFeatures;
using ctp::compress::model::DataFeatures;
using ctp::compress::model::MakeCompressionFeatures;
using ctp::compress::model::NeuroPressNNPredictor;
using ctp::compress::model::TrainingLabels;

namespace {

constexpr size_t kTotalBytes = 1ull << 30;                // 1 GiB
constexpr size_t kChunkBytes = 16ull << 20;               // 16 MiB
constexpr size_t kNumChunks = kTotalBytes / kChunkBytes;  // 64

/* At 0 upstream masks every quantize action; a positive bound frees them. */
constexpr double kBounds[] = {0.0, 1e-3};
constexpr size_t kNumBounds = sizeof(kBounds) / sizeof(kBounds[0]);

/* Upstream's own defaults (gpucompress_api.cpp): X1 = 0.30 fires SGD,
   X2 = 0.50 fires exploration, and K = 3 alternatives. */
constexpr double kSgdMape = 0.30;
constexpr double kExploreMape = 0.50;
constexpr int kExploreK = 3;
constexpr float kLearningRate = 0.01f;

enum Mode { kInference = 0, kLearning = 1, kExploration = 2 };
const char *ModeName(int m) {
  switch (m) {
    case kInference: return "inference";
    case kLearning: return "learning";
    default: return "exploration";
  }
}

int g_failures = 0;
long g_checks = 0;

struct Worst {
  double entropy = 0, mad = 0, deriv = 0;
  double ratio = 0, ct = 0, dt = 0;
  int selection_mismatch = 0;
  int explore_set_mismatch = 0;
  int trigger_mismatch = 0;
};

double Rel(double a, double b) {
  const double denom = std::max(1e-12, std::max(std::fabs(a), std::fabs(b)));
  return std::fabs(a - b) / denom;
}

void Expect(double a, double b, double tol, double *worst, const char *what,
            const char *mode, size_t chunk, double eb) {
  ++g_checks;
  const double rel = Rel(a, b);
  if (rel > *worst) *worst = rel;
  if (rel > tol) {
    if (g_failures < 25) {
      std::printf("  [FAIL] %s chunk %zu eb=%g %s: native=%.17g clio=%.17g "
                  "(rel %.3g)\n",
                  mode, chunk, eb, what, a, b, rel);
    }
    ++g_failures;
  }
}

const char *AlgoName(int a) {
  switch (a) {
    case 0: return "nvcomp-lz4";      case 1: return "nvcomp-snappy";
    case 2: return "nvcomp-deflate";  case 3: return "nvcomp-gdeflate";
    case 4: return "nvcomp-zstd";     case 5: return "nvcomp-ans";
    case 6: return "nvcomp-cascaded"; default: return "nvcomp-bitcomp";
  }
}

int BaseIdForAlgoIdx(int a) {
  switch (a) {
    case 0: return 13; case 1: return 14; case 2: return 17; case 3: return 16;
    case 4: return 15; case 5: return 18; case 6: return 23; default: return 24;
  }
}

int AlgoIdxForBaseId(int base_id) {
  for (int a = 0; a < 8; ++a) {
    if (BaseIdForAlgoIdx(a) == base_id) return a;
  }
  return -1;
}

/* Ascending action order -- shuffle outer, quant middle, algo inner -- so a
   tie resolves to the lowest index on both sides. */
std::vector<CandidateConfig> BuildCandidates(double eb) {
  std::vector<CandidateConfig> cands;
  for (int shuffle = 0; shuffle <= 1; ++shuffle) {
    for (int quant = 0; quant <= 1; ++quant) {
      if (quant == 1 && !(eb > 0.0)) continue;
      for (int algo = 0; algo < 8; ++algo) {
        CandidateConfig c;
        c.base_id = BaseIdForAlgoIdx(algo);
        c.preset_id = 2;
        c.byte_shuffle = (shuffle != 0);
        c.quantize = (quant != 0);
        c.error_bound = (quant != 0) ? eb : 0.0;
        cands.push_back(c);
      }
    }
  }
  return cands;
}

/* INFERENCE: a lossless candidate carries no bound and the 1e-7 sentinel is
   substituted, as nnInferenceKernel does. */
CandidateConfig CandidateForAction(int action, double eb) {
  CandidateConfig c;
  c.base_id = BaseIdForAlgoIdx(action % 8);
  c.preset_id = 2;
  c.quantize = ((action / 8) % 2) != 0;
  c.byte_shuffle = ((action / 16) % 2) != 0;
  c.error_bound = c.quantize ? eb : 0.0;
  return c;
}

/* SGD uses a DIFFERENT convention: nnSGDKernel writes raw[3] with no quant
   test, so a lossless action sees the configured bound, not 0 or 1e-7. */
CandidateConfig SgdCandidateForAction(int action, double eb) {
  CandidateConfig c = CandidateForAction(action, eb);
  c.error_bound = eb;
  return c;
}

/* w0*ct + w1*dt + w2*ds/(ratio*bw), clamps first: ct/dt floored 1 ms, ratio
   capped 100x (nnForwardPass, nn_gpu.cu). */
double Cost(double ct, double dt, double ratio, double ds) {
  const double c = std::max(1.0, ct);
  const double d = std::max(1.0, dt);
  const double r = std::min(100.0, ratio);
  const double bw = static_cast<double>(g_measured_bw_bytes_per_ms);
  return static_cast<double>(g_rank_w0) * c +
         static_cast<double>(g_rank_w1) * d +
         ((r > 0.0) ? static_cast<double>(g_rank_w2) * ds / (r * bw) : 0.0);
}

/** Deterministic 1 GiB over several regimes; one buffer feeds both sides. */
__global__ void FillDataset(float *buf, size_t num_elems,
                            size_t elems_per_chunk) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= num_elems) return;
  size_t chunk = i / elems_per_chunk;
  size_t local = i % elems_per_chunk;
  switch (chunk % 5) {
    case 0:
      buf[i] = 42.0f;
      break;
    case 1: {
      float p = 6.28318f * static_cast<float>(local) /
                static_cast<float>(elems_per_chunk);
      buf[i] = sinf(p * 4.0f);
      break;
    }
    case 2:
      buf[i] = static_cast<float>((local / 512) % 1024) * 0.25f;
      break;
    case 3: {
      uint32_t x = static_cast<uint32_t>(i) * 747796405u + 2891336453u;
      x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15;
      float n = static_cast<float>(x & 0xFFFF) / 65535.0f - 0.5f;
      float p = 6.28318f * static_cast<float>(local) /
                static_cast<float>(elems_per_chunk);
      buf[i] = sinf(p * 2.0f) + 0.05f * n;
      break;
    }
    default: {
      uint32_t x = static_cast<uint32_t>(i) * 2654435761u + 0x9E3779B9u;
      x ^= x >> 15; x *= 0x85EBCA6Bu; x ^= x >> 13;
      buf[i] = static_cast<float>(x) / 4294967295.0f;
      break;
    }
  }
}

}  // namespace

int main(int argc, char **argv) {
  const std::string weights_dir =
      (argc > 1) ? argv[1] : std::string(CLIO_CTP_NEUROPRESS_WEIGHTS_DIR);
  const std::string nnwt = weights_dir + "/model.nnwt";

  int dev_count = 0;
  if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) {
    std::printf("No CUDA device -- skipping modes parity.\n");
    return 77;
  }

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cudaStreamCreate(&g_sgd_stream);
  cudaEventCreate(&g_sgd_done);

  float *d_data = nullptr;
  if (cudaMalloc(&d_data, kTotalBytes) != cudaSuccess) {
    std::printf("FATAL: cudaMalloc %zu bytes failed\n", kTotalBytes);
    return 1;
  }
  const size_t num_elems = kTotalBytes / sizeof(float);
  const size_t elems_per_chunk = kChunkBytes / sizeof(float);
  {
    const int threads = 256;
    const size_t blocks = (num_elems + threads - 1) / threads;
    FillDataset<<<static_cast<int>(blocks), threads>>>(d_data, num_elems,
                                                       elems_per_chunk);
    cudaDeviceSynchronize();
  }

  std::printf("Dataset: %zu MiB in %zu chunks of %zu MiB\n", kTotalBytes >> 20,
              kNumChunks, kChunkBytes >> 20);
  std::printf("Modes: inference, learning, exploration | bounds: 0.0, 1e-3\n");
  std::printf("Cost model: w0=%g w1=%g w2=%g bw=%g bytes/ms\n\n", g_rank_w0,
              g_rank_w1, g_rank_w2, g_measured_bw_bytes_per_ms);

  std::FILE *csv = std::fopen("neuropress_modes_parity.csv", "w");
  if (csv) {
    std::fprintf(csv,
                 "mode,error_bound,chunk,"
                 "entropy_native,entropy_clio,mad_native,mad_clio,"
                 "deriv_native,deriv_clio,"
                 "action_native,action_clio,algo_native,algo_clio,"
                 "quant_native,quant_clio,shuffle_native,shuffle_clio,"
                 "ratio_native,ratio_clio,ct_native,ct_clio,dt_native,dt_clio,"
                 "label_ratio,label_ct,cost_mape,sgd_fired,explored_k,"
                 "explore_set_native,explore_set_clio,selection_match\n");
  }

  Worst worst[3];

  for (int mode = kInference; mode <= kExploration; ++mode) {
    for (size_t bi = 0; bi < kNumBounds; ++bi) {
      const double eb = kBounds[bi];

      /* Fresh weights per pass, so the modes are independent trials. */
      if (!gpucompress::loadNNFromBinary(nnwt.c_str())) {
        std::printf("FATAL: native failed to load %s\n", nnwt.c_str());
        return 1;
      }
      NeuroPressNNPredictor clio;
      if (!clio.Load(weights_dir) || !clio.IsReady()) {
        std::printf("FATAL: Clio failed to load %s\n", weights_dir.c_str());
        return 1;
      }
      clio.SetLearningRate(kLearningRate);

      int sgd_fires = 0, explore_fires = 0, sel_mismatch = 0;
      std::printf("===== mode=%s error_bound=%g =====\n", ModeName(mode), eb);

      for (size_t ci = 0; ci < kNumChunks; ++ci) {
        const char *d_chunk =
            reinterpret_cast<const char *>(d_data) + ci * kChunkBytes;

        /* ---- (1) statistics, each side with its own kernels ---- */
        AutoStatsGPU *d_native_stats = gpucompress::runStatsKernelsNoSync(
            d_chunk, kChunkBytes, stream);
        cudaStreamSynchronize(stream);
        ++g_checks;
        if (d_native_stats == nullptr) {
          std::printf("  [FAIL] %s chunk %zu: native stats failed\n",
                      ModeName(mode), ci);
          ++g_failures;
          continue;
        }
        AutoStatsGPU h_native{};
        cudaMemcpy(&h_native, d_native_stats, sizeof(AutoStatsGPU),
                   cudaMemcpyDeviceToHost);
        const double n_ent = h_native.entropy;
        const double n_mad = h_native.mad_normalized;
        const double n_der = h_native.deriv_normalized;

        void *clio_stream = ctp::DeviceStatsStream();
        const void *d_clio_stats = ctp::ComputeDeviceStatsResident(
            d_chunk, kChunkBytes / sizeof(float), ctp::DataType::FLOAT32,
            clio_stream);
        ++g_checks;
        if (d_clio_stats == nullptr) {
          std::printf("  [FAIL] %s chunk %zu: Clio stats failed\n",
                      ModeName(mode), ci);
          ++g_failures;
          continue;
        }
        double c_ent = 0, c_mad = 0, c_der = 0;
        ctp::ReadDeviceFeatureStats(d_clio_stats, &c_ent, &c_mad, &c_der,
                                    clio_stream);

        Expect(n_ent, c_ent, 1e-9, &worst[mode].entropy, "entropy",
               ModeName(mode), ci, eb);
        Expect(n_mad, c_mad, 1e-9, &worst[mode].mad, "mad", ModeName(mode), ci,
               eb);
        Expect(n_der, c_der, 1e-9, &worst[mode].deriv, "second_derivative",
               ModeName(mode), ci, eb);

        /* ---- (2) native inference: predictions + full ranked order ---- */
        float n_ratio = 0, n_ct = 0, n_dt = 0, n_psnr = 0;
        int top[NN_NUM_CONFIGS] = {0};
        const int n_action = gpucompress::runNNInference(
            n_ent, n_mad, n_der, kChunkBytes, eb, stream, &n_ratio, &n_ct,
            &n_dt, &n_psnr, top);
        cudaStreamSynchronize(stream);
        ++g_checks;
        if (n_action < 0) {
          std::printf("  [FAIL] %s chunk %zu: native inference failed\n",
                      ModeName(mode), ci);
          ++g_failures;
          continue;
        }
        const int n_algo = n_action % 8;
        const int n_quant = (n_action / 8) % 2;
        const int n_shuf = (n_action / 16) % 2;

        /* ---- (3) Clio: rank its own candidates from its OWN statistics ---- */
        DataFeatures data;
        data.chunk_size_bytes = static_cast<double>(kChunkBytes);
        data.shannon_entropy = c_ent;
        data.mad = c_mad;
        data.second_derivative_mean = c_der;
        data.data_type_float = 1.0;

        ctp::compress::model::RankingWeights rw;
        rw.use_cost_model = true;
        const std::vector<CandidateConfig> cands = BuildCandidates(eb);
        auto ranked = clio.Rank(data, cands, rw);
        ++g_checks;
        if (ranked.empty()) {
          std::printf("  [FAIL] %s chunk %zu: Clio produced no ranking\n",
                      ModeName(mode), ci);
          ++g_failures;
          continue;
        }
        const int c_algo = AlgoIdxForBaseId(ranked.front().candidate.base_id);
        const int c_quant = ranked.front().candidate.quantize ? 1 : 0;
        const int c_shuf = ranked.front().candidate.byte_shuffle ? 1 : 0;
        const int c_action = c_algo + 8 * c_quant + 16 * c_shuf;

        ++g_checks;
        const bool sel_match =
            (c_algo == n_algo && c_quant == n_quant && c_shuf == n_shuf);
        if (!sel_match) {
          ++sel_mismatch;
          ++worst[mode].selection_mismatch;
          if (worst[mode].selection_mismatch <= 10) {
            std::printf("  [DIFF] %s chunk %zu eb=%g selection: native "
                        "action=%d (%s q=%d s=%d), clio action=%d (%s q=%d "
                        "s=%d)\n",
                        ModeName(mode), ci, eb, n_action, AlgoName(n_algo),
                        n_quant, n_shuf, c_action,
                        (c_algo >= 0 ? AlgoName(c_algo) : "none"), c_quant,
                        c_shuf);
          }
          ++g_failures;
        }

        /* ---- (4) prediction parity for the action native chose ---- */
        CompressionFeatures chosen_f =
            MakeCompressionFeatures(data, CandidateForAction(n_action, eb));
        auto p = clio.Predict(chosen_f);
        Expect(n_ratio, p.compression_ratio, 1e-3, &worst[mode].ratio, "ratio",
               ModeName(mode), ci, eb);
        Expect(n_ct, p.compression_time_ms, 1e-3, &worst[mode].ct, "comp_time",
               ModeName(mode), ci, eb);
        Expect(n_dt, p.decompression_time_ms, 1e-3, &worst[mode].dt,
               "decomp_time", ModeName(mode), ci, eb);

        /* ---- (5) the shared label, and the trigger both sides compute ---- */
        /* Swings across the thresholds, so both branches get exercised. */
        const double swing = 0.25 + 0.65 * static_cast<double>(ci % 7) / 6.0;
        const double label_ratio = std::max(0.5, n_ratio * swing);
        const double label_ct = std::max(0.01, n_ct * (2.0 - swing));
        const double label_dt = std::max(0.01, n_dt * (2.0 - swing));
        const double ds = static_cast<double>(kChunkBytes);
        const double pred_cost = Cost(n_ct, n_dt, n_ratio, ds);
        const double act_cost = Cost(label_ct, label_dt, label_ratio, ds);
        const double mape =
            (act_cost > 0.0) ? std::fabs(act_cost - pred_cost) / act_cost : 0.0;
        const bool fire_sgd = (mode >= kLearning) && (mape > kSgdMape);
        const bool fire_explore =
            (mode == kExploration) && (mape > kExploreMape);

        std::string explore_native = "-", explore_clio = "-";
        int explored_k = 0;

        if (fire_sgd) {
          ++sgd_fires;
          SGDSample s{};
          s.action = n_action;
          s.actual_ratio = static_cast<float>(label_ratio);
          s.actual_comp_time = static_cast<float>(label_ct);
          s.actual_decomp_time = static_cast<float>(label_dt);
          /* Negative is upstream's "skip the PSNR head" sentinel. */
          s.actual_psnr = (n_quant != 0) ? 120.0f : -1.0f;
          float gn = 0; int gc = 0, gs = 0;
          const int rc = gpucompress::runNNSGD(d_native_stats, &s, 1,
                                               kChunkBytes, eb, kLearningRate,
                                               g_sgd_stream, &gn, &gc, &gs);
          ++g_checks;
          if (rc != 0) {
            std::printf("  [FAIL] %s chunk %zu: native SGD rc=%d\n",
                        ModeName(mode), ci, rc);
            ++g_failures;
          }

          std::vector<CompressionFeatures> feats{MakeCompressionFeatures(
              data, SgdCandidateForAction(n_action, eb))};
          std::vector<TrainingLabels> labels{TrainingLabels(
              static_cast<float>(label_ratio),
              (n_quant != 0) ? 120.0f : -1.0f,
              static_cast<float>(label_ct), static_cast<float>(label_dt))};
          ++g_checks;
          if (!clio.TrainDeviceStats(feats, labels, d_clio_stats)) {
            std::printf("  [FAIL] %s chunk %zu: Clio SGD refused\n",
                        ModeName(mode), ci);
            ++g_failures;
          }
          /* Fire-and-forget; land them before the next chunk reads. */
          cudaStreamSynchronize(g_sgd_stream);
          cudaDeviceSynchronize();
        }

        if (fire_explore) {
          ++explore_fires;
          explored_k = kExploreK;
          /* Both walk their ranked list from index 1, skipping the primary;
             the orders must agree over the explored prefix. */
          char nb[64] = {0}, cb[64] = {0};
          int npos = 0, cpos = 0;
          int taken = 0;
          for (int k = 1; k < NN_NUM_CONFIGS && taken < kExploreK; ++k) {
            if (top[k] == n_action) continue;
            npos += std::snprintf(nb + npos, sizeof(nb) - npos, "%s%d",
                                  taken ? "|" : "", top[k]);
            ++taken;
          }
          taken = 0;
          for (size_t k = 1; k < ranked.size() && taken < kExploreK; ++k) {
            const auto &cc = ranked[k].candidate;
            const int act = AlgoIdxForBaseId(cc.base_id) +
                            8 * (cc.quantize ? 1 : 0) +
                            16 * (cc.byte_shuffle ? 1 : 0);
            if (act == c_action) continue;
            cpos += std::snprintf(cb + cpos, sizeof(cb) - cpos, "%s%d",
                                  taken ? "|" : "", act);
            ++taken;
          }
          explore_native = nb;
          explore_clio = cb;
          ++g_checks;
          if (explore_native != explore_clio) {
            ++worst[mode].explore_set_mismatch;
            if (worst[mode].explore_set_mismatch <= 10) {
              std::printf("  [DIFF] %s chunk %zu eb=%g explore set: native=%s "
                          "clio=%s\n",
                          ModeName(mode), ci, eb, nb, cb);
            }
            ++g_failures;
          }
        }

        if (csv) {
          std::fprintf(csv,
                       "%s,%g,%zu,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
                       "%d,%d,%s,%s,%d,%d,%d,%d,"
                       "%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,"
                       "%.9g,%.9g,%.6g,%d,%d,%s,%s,%s\n",
                       ModeName(mode), eb, ci, n_ent, c_ent, n_mad, c_mad,
                       n_der, c_der, n_action, c_action, AlgoName(n_algo),
                       (c_algo >= 0 ? AlgoName(c_algo) : "none"), n_quant,
                       c_quant, n_shuf, c_shuf, n_ratio, p.compression_ratio,
                       n_ct, p.compression_time_ms, n_dt,
                       p.decompression_time_ms, label_ratio, label_ct, mape,
                       fire_sgd ? 1 : 0, explored_k, explore_native.c_str(),
                       explore_clio.c_str(), sel_match ? "YES" : "NO");
        }

        if (ci < 3 || (ci % 16) == 0 || !sel_match) {
          std::printf("  chunk %-3zu H=%.6f mad=%.6f d2=%.6f | native=%-16s "
                      "clio=%-16s q=%d/%d s=%d/%d | mape=%.3f %s%s\n",
                      ci, n_ent, n_mad, n_der, AlgoName(n_algo),
                      (c_algo >= 0 ? AlgoName(c_algo) : "none"), n_quant,
                      c_quant, n_shuf, c_shuf, mape,
                      fire_sgd ? "SGD " : "", fire_explore ? "EXPLORE" : "");
        }
      }

      std::printf("  -- %s eb=%g: %d SGD step(s), %d exploration(s), "
                  "%d selection mismatch(es)\n\n",
                  ModeName(mode), eb, sgd_fires, explore_fires, sel_mismatch);

      gpucompress::cleanupNN();
    }
  }

  if (csv) std::fclose(csv);

  std::printf("=========== worst-case relative deviation, per mode ===========\n");
  std::printf("%-12s %-12s %-12s %-12s %-10s %-10s %-10s %s\n", "mode",
              "entropy", "mad", "second_d", "ratio", "comp_t", "decomp_t",
              "sel/expl mismatch");
  for (int m = kInference; m <= kExploration; ++m) {
    std::printf("%-12s %-12.3g %-12.3g %-12.3g %-10.3g %-10.3g %-10.3g %d/%d\n",
                ModeName(m), worst[m].entropy, worst[m].mad, worst[m].deriv,
                worst[m].ratio, worst[m].ct, worst[m].dt,
                worst[m].selection_mismatch, worst[m].explore_set_mismatch);
  }
  std::printf("\nPer-chunk detail: neuropress_modes_parity.csv\n");
  std::printf("\n===== %ld checks, %d failures =====\n", g_checks, g_failures);

  gpucompress::freeStatsWorkspace();
  cudaFree(d_data);
  return g_failures == 0 ? 0 : 1;
}
