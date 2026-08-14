/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file neuropress_parity.cu
 * @brief Differential test: native NeuroPress vs Clio's ported predictor.
 *
 * Both sides load the SAME model.nnwt and are fed the SAME per-chunk feature
 * values. This is the only check that can catch a porting error the source
 * review cannot -- reading two implementations and concluding they agree is
 * not the same as running them on identical inputs and diffing the numbers.
 *
 * Two things are compared:
 *
 *  1. INFERENCE. For each synthetic chunk, native predicts (ratio, comp_time,
 *     decomp_time, psnr) and reports the action it selected. Clio is then
 *     asked to predict for THAT SAME action, so the comparison isolates the
 *     model from the selection policy (the two deliberately differ in policy:
 *     native ranks 32 configs incl. quantize/byte-shuffle, Clio ranks the 8
 *     nvcomp algorithms -- see neuropress_bridge.cc).
 *
 *  2. LEARNING. After each SGD step driven by identical samples, the full
 *     weight vector is pulled from both and diffed element-wise, so a
 *     divergence in any hyperparameter, in the gradient math, or in which
 *     weights a pass is allowed to touch shows up immediately.
 *
 * Built only when the NeuroPress source tree is present (see CMakeLists.txt);
 * this is a cross-project check, not part of the default build.
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "clio_ctp/compress/model/neuropress_nn_predictor.h"

// --- Native NeuroPress (compiled from its own source, namespace gpucompress).
#include "nn_weights.h"
#include "auto_stats_gpu.h"

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
int runBatchedDecompSGD(const DeferredDecompSample *samples, int num_samples,
                        float learning_rate, float *out_grad_norm);
}  // namespace gpucompress

extern "C" void *gpucompress_nn_get_device_ptr_impl(void);

extern cudaStream_t g_sgd_stream;
extern cudaEvent_t g_sgd_done;

using ctp::compress::model::CompressionFeatures;
using ctp::compress::model::NeuroPressNNPredictor;
using ctp::compress::model::TrainingLabels;

namespace {

int g_failures = 0;
int g_checks = 0;
double g_max_rel_pred = 0.0;      // worst relative error, predictions
std::string g_max_rel_what;
double g_max_abs_weight = 0.0;    // worst absolute diff, weights
const char *g_max_abs_layer = "";
std::string g_max_abs_phase;

void Check(bool cond, const std::string &what) {
  ++g_checks;
  if (!cond) {
    ++g_failures;
    std::printf("  [FAIL] %s\n", what.c_str());
  }
}

void CheckClose(double a, double b, double rel_tol, const std::string &what) {
  ++g_checks;
  double denom = std::max(1e-9, std::max(std::fabs(a), std::fabs(b)));
  double rel = std::fabs(a - b) / denom;
  if (rel > g_max_rel_pred) { g_max_rel_pred = rel; g_max_rel_what = what; }
  if (!(rel <= rel_tol)) {
    ++g_failures;
    std::printf("  [FAIL] %s: native=%.9g clio=%.9g rel_err=%.3g\n",
                what.c_str(), a, b, rel);
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

/** Pull the full native weight set off the device. */
bool SnapshotNative(NNWeightsGPU *out) {
  void *d = gpucompress_nn_get_device_ptr_impl();
  if (!d) return false;
  return cudaMemcpy(out, d, sizeof(NNWeightsGPU), cudaMemcpyDeviceToHost) ==
         cudaSuccess;
}

/** Flatten Clio's weights into native's NNWeightsGPU layout for diffing.
 *  Clio stores weights_ as [L0..L4] concatenated and biases_ separately, in
 *  the same per-layer order and the same row-major shape. */
void FlattenClio(const std::vector<float> &w, const std::vector<float> &b,
                 NNWeightsGPU *out) {
  size_t wo = 0, bo = 0;
  auto copy_w = [&](float *dst, size_t n) {
    std::memcpy(dst, w.data() + wo, n * sizeof(float));
    wo += n;
  };
  auto copy_b = [&](float *dst, size_t n) {
    std::memcpy(dst, b.data() + bo, n * sizeof(float));
    bo += n;
  };
  copy_w(out->w1, NN_HIDDEN_DIM * NN_INPUT_DIM);
  copy_w(out->w2, NN_HIDDEN_DIM * NN_HIDDEN_DIM);
  copy_w(out->w3, NN_HIDDEN_DIM * NN_HIDDEN_DIM);
  copy_w(out->w4, NN_HIDDEN_DIM * NN_HIDDEN_DIM);
  copy_w(out->w5, NN_OUTPUT_DIM * NN_HIDDEN_DIM);
  copy_b(out->b1, NN_HIDDEN_DIM);
  copy_b(out->b2, NN_HIDDEN_DIM);
  copy_b(out->b3, NN_HIDDEN_DIM);
  copy_b(out->b4, NN_HIDDEN_DIM);
  copy_b(out->b5, NN_OUTPUT_DIM);
}

/** Element-wise weight diff, reported per layer so a divergence is localized
 *  rather than just "something moved". */
void CompareWeights(const NNWeightsGPU &nat, const NNWeightsGPU &clio,
                    const std::string &phase, double tol) {
  struct Layer { const char *name; const float *a; const float *b; size_t n; };
  const Layer layers[] = {
      {"w1", nat.w1, clio.w1, NN_HIDDEN_DIM * NN_INPUT_DIM},
      {"b1", nat.b1, clio.b1, NN_HIDDEN_DIM},
      {"w2", nat.w2, clio.w2, NN_HIDDEN_DIM * NN_HIDDEN_DIM},
      {"b2", nat.b2, clio.b2, NN_HIDDEN_DIM},
      {"w3", nat.w3, clio.w3, NN_HIDDEN_DIM * NN_HIDDEN_DIM},
      {"b3", nat.b3, clio.b3, NN_HIDDEN_DIM},
      {"w4", nat.w4, clio.w4, NN_HIDDEN_DIM * NN_HIDDEN_DIM},
      {"b4", nat.b4, clio.b4, NN_HIDDEN_DIM},
      {"w5", nat.w5, clio.w5, NN_OUTPUT_DIM * NN_HIDDEN_DIM},
      {"b5", nat.b5, clio.b5, NN_OUTPUT_DIM},
  };
  for (const auto &L : layers) {
    double max_abs = 0.0;
    size_t worst = 0;
    for (size_t i = 0; i < L.n; ++i) {
      double d = std::fabs(static_cast<double>(L.a[i]) - L.b[i]);
      if (d > max_abs) { max_abs = d; worst = i; }
    }
    ++g_checks;
    if (max_abs > g_max_abs_weight) {
      g_max_abs_weight = max_abs; g_max_abs_layer = L.name;
      g_max_abs_phase = phase;
    }
    if (max_abs > tol) {
      ++g_failures;
      std::printf("  [FAIL] %s %s: max|diff|=%.6g at idx %zu "
                  "(native=%.9g clio=%.9g)\n",
                  phase.c_str(), L.name, max_abs, worst,
                  static_cast<double>(L.a[worst]), L.b[worst]);
    }
  }
}

struct Chunk {
  const char *label;
  double entropy;
  double mad;
  double deriv;
  size_t size_bytes;
};

}  // namespace

int main(int argc, char **argv) {
  const std::string weights_dir =
      (argc > 1) ? argv[1] : std::string(CLIO_CTP_NEUROPRESS_WEIGHTS_DIR);
  const std::string nnwt = weights_dir + "/model.nnwt";

  int dev_count = 0;
  if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) {
    std::printf("No CUDA device -- skipping parity check.\n");
    return 77;  // ctest SKIP_RETURN_CODE
  }

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
  std::printf("Both sides loaded %s\n\n", nnwt.c_str());

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  // gpucompress_init() normally creates these; the harness skips it (that
  // path drags in nvcomp/cuSZ/ndzip), so stand them up here. Leaving them
  // null makes runNNSGD's event record fail and the update silently no-op.
  cudaStreamCreate(&g_sgd_stream);
  cudaEventCreate(&g_sgd_done);

  // ---- Phase 0: identical weights before anything runs. ----
  {
    NNWeightsGPU nat{}, cl{};
    Check(SnapshotNative(&nat), "snapshot native weights");
    FlattenClio(clio.DebugWeights(), clio.DebugBiases(), &cl);
    CompareWeights(nat, cl, "load", 0.0);  // must be bit-identical
    std::printf("[phase 0] post-load weight parity checked\n");
  }

  const std::vector<Chunk> chunks = {
      {"near-zero entropy", 0.20, 0.010, 0.005, 2u << 20},
      {"low entropy",       1.50, 0.050, 0.020, 2u << 20},
      {"mid entropy",       4.00, 0.200, 0.100, 1u << 20},
      {"high entropy",      6.50, 0.600, 0.400, 4u << 20},
      {"max entropy",       7.90, 0.950, 0.900, 512u << 10},
      {"tiny chunk",        3.20, 0.150, 0.080, 64u << 10},
      {"large chunk",       2.10, 0.090, 0.030, 16u << 20},
  };

  // ---- Phase 1: per-chunk inference parity. ----
  std::printf("\n[phase 1] inference parity (native winner vs Clio, same action)\n");
  for (const auto &c : chunks) {
    float n_ratio = 0, n_ct = 0, n_dt = 0, n_psnr = 0;
    int top[NN_NUM_CONFIGS] = {0};
    // NOTE: runNNInference returns the SELECTED ACTION, not a status code.
    int action = gpucompress::runNNInference(
        c.entropy, c.mad, c.deriv, c.size_bytes, /*error_bound=*/0.0, stream,
        &n_ratio, &n_ct, &n_dt, &n_psnr, top);
    cudaStreamSynchronize(stream);
    ++g_checks;
    if (action < 0) {
      std::printf("  [FAIL] native inference failed for %s (cuda=%s)\n",
                  c.label, cudaGetErrorString(cudaGetLastError()));
      ++g_failures;
      continue;
    }
    const int algo_idx = action % 8;
    const int quant = (action / 8) % 2;
    const int shuffle = (action / 16) % 2;

    // Ask Clio to predict for the SAME action, isolating model from policy.
    CompressionFeatures f;
    f.chunk_size_bytes = static_cast<double>(c.size_bytes);
    f.shannon_entropy = c.entropy;
    f.mad = c.mad;
    f.second_derivative_mean = c.deriv;
    f.data_type_float = 1.0;
    f.library_config_id = BaseIdForAlgoIdx(algo_idx) * 10 + 2;
    f.config_balanced = 1.0;
    f.quantize = quant ? 1.0 : 0.0;
    f.byte_shuffle = shuffle ? 1.0 : 0.0;
    f.error_bound = 0.0;

    auto p = clio.Predict(f);

    std::printf("  %-18s action=%2d (algo=%d q=%d s=%d)  "
                "ratio n/c=%.4f/%.4f  ct=%.4f/%.4f  dt=%.4f/%.4f\n",
                c.label, action, algo_idx, quant, shuffle, n_ratio,
                p.compression_ratio, n_ct, p.compression_time_ms, n_dt,
                p.decompression_time_ms);

    const double kTol = 1e-4;  // float32 kernels vs float32 host math
    CheckClose(n_ratio, p.compression_ratio, kTol,
               std::string(c.label) + " ratio");
    CheckClose(n_ct, p.compression_time_ms, kTol,
               std::string(c.label) + " comp_time");
    CheckClose(n_dt, p.decompression_time_ms, kTol,
               std::string(c.label) + " decomp_time");
    CheckClose(n_psnr, p.psnr_db, kTol, std::string(c.label) + " psnr");
  }

  // ---- Phase 1b: inference parity at NONZERO error bounds. ----
  // Phase 1 runs entirely at error_bound = 0, which leaves two things
  // untested. First, nnInferenceKernel masks every quantize config when the
  // bound is zero -- "if (quant == 1 && error_bound <= 0.0) rank_val =
  // -INFINITY" (nn_gpu.cu) -- so the winner is always quant=0 and half
  // the 32-config space is unreachable. Second, input 3 is
  // "(quant == 0) ? 1e-7f : eb_enc" (nn_gpu.cu), so at a zero bound the
  // sentinel branch is taken for every candidate and the raw-bound branch
  // never runs. A port that fed the sentinel unconditionally, or fed the raw
  // bound unconditionally, would agree with upstream on every phase-1 case.
  //
  // Sweeping real bounds makes quantize configs selectable and exercises
  // both branches. kQuantSeen below is the discriminating assertion: if no
  // quant=1 winner ever appears, this phase proved nothing and must fail
  // rather than pass vacuously.
  std::printf("\n[phase 1b] inference parity at nonzero error bounds\n");
  int quant_winners = 0;
  {
    const double kBounds[] = {1e-6, 1e-4, 1e-3, 1e-2};
    for (double eb : kBounds) {
      for (const auto &c : chunks) {
        float n_ratio = 0, n_ct = 0, n_dt = 0, n_psnr = 0;
        int top[NN_NUM_CONFIGS] = {0};
        int action = gpucompress::runNNInference(c.entropy, c.mad, c.deriv,
                                                 c.size_bytes, eb, stream,
                                                 &n_ratio, &n_ct, &n_dt,
                                                 &n_psnr, top);
        cudaStreamSynchronize(stream);
        ++g_checks;
        if (action < 0) {
          std::printf("  [FAIL] native inference failed for %s at eb=%g\n",
                      c.label, eb);
          ++g_failures;
          continue;
        }
        const int algo_idx = action % 8;
        const int quant = (action / 8) % 2;
        const int shuffle = (action / 16) % 2;
        if (quant) ++quant_winners;

        CompressionFeatures f;
        f.chunk_size_bytes = static_cast<double>(c.size_bytes);
        f.shannon_entropy = c.entropy;
        f.mad = c.mad;
        f.second_derivative_mean = c.deriv;
        f.data_type_float = 1.0;
        f.library_config_id = BaseIdForAlgoIdx(algo_idx) * 10 + 2;
        f.config_balanced = 1.0;
        f.quantize = quant ? 1.0 : 0.0;
        f.byte_shuffle = shuffle ? 1.0 : 0.0;
        // The bound Clio is handed is the same one native ranked under.
        // Clio re-applies the sentinel itself when quant == 0, exactly as
        // the kernel does, so this must NOT be pre-substituted here.
        f.error_bound = eb;

        auto p = clio.Predict(f);

        const std::string tag =
            std::string(c.label) + " @eb=" + std::to_string(eb) +
            (quant ? " [quant]" : " [lossless]");
        const double kTol = 1e-4;
        CheckClose(n_ratio, p.compression_ratio, kTol, tag + " ratio");
        CheckClose(n_ct, p.compression_time_ms, kTol, tag + " comp_time");
        CheckClose(n_dt, p.decompression_time_ms, kTol, tag + " decomp_time");
        CheckClose(n_psnr, p.psnr_db, kTol, tag + " psnr");
      }
      std::printf("  eb=%-8g swept %zu chunks\n", eb, chunks.size());
    }
    // Discriminating: a sweep that never selects a quantize config has not
    // exercised the raw-bound branch at all.
    Check(quant_winners > 0,
          "at least one quantize config selected across the bound sweep");
    std::printf("  quantize configs selected: %d of %zu cases\n",
                quant_winners, chunks.size() * 4);
  }

  // ---- Phase 2: SGD parity, weights diffed after every step. ----
  std::printf("\n[phase 2] online SGD parity (weights diffed after each step)\n");
  AutoStatsGPU *d_stats = nullptr;
  cudaMalloc(&d_stats, sizeof(AutoStatsGPU));

  for (size_t step = 0; step < chunks.size(); ++step) {
    const auto &c = chunks[step];

    // Same observed outcome fed to both sides.
    const float actual_ratio = 2.0f + 0.5f * static_cast<float>(step);
    const float actual_ct = 12.0f + 3.0f * static_cast<float>(step);
    const float actual_dt = 7.0f + 2.0f * static_cast<float>(step);
    const float actual_psnr = 0.0f;  // lossless
    const int algo_idx = static_cast<int>(step % 8);

    // Native: stats struct carries the features; sample carries the outcome.
    AutoStatsGPU h_stats{};
    h_stats.entropy = c.entropy;
    h_stats.mad_normalized = c.mad;
    h_stats.deriv_normalized = c.deriv;
    h_stats.num_elements = c.size_bytes / sizeof(float);
    cudaMemcpy(d_stats, &h_stats, sizeof(h_stats), cudaMemcpyHostToDevice);

    SGDSample s{};
    s.action = algo_idx;  // quant=0, shuffle=0
    s.actual_ratio = actual_ratio;
    s.actual_comp_time = actual_ct;
    s.actual_decomp_time = actual_dt;
    s.actual_psnr = actual_psnr;

    float gn = 0; int clipped = 0, cnt = 0;
    int rc = gpucompress::runNNSGD(d_stats, &s, 1, c.size_bytes,
                                   /*error_bound=*/0.0, /*lr=*/0.01f, stream,
                                   &gn, &clipped, &cnt);
    cudaStreamSynchronize(stream);
    if (rc != 0) {
      std::printf("  [FAIL] native SGD rc=%d at step %zu (cuda=%s)\n", rc, step,
                  cudaGetErrorString(cudaGetLastError()));
      ++g_failures;
    }
    ++g_checks;

    // Clio: same features, same outcome.
    CompressionFeatures f;
    f.chunk_size_bytes = static_cast<double>(c.size_bytes);
    f.shannon_entropy = c.entropy;
    f.mad = c.mad;
    f.second_derivative_mean = c.deriv;
    f.data_type_float = 1.0;
    f.library_config_id = BaseIdForAlgoIdx(algo_idx) * 10 + 2;
    f.config_balanced = 1.0;
    // Identical label on both sides. actual_dt matters: native feeds output
    // 1's error into the shared trunk (only its HEAD weights are withheld),
    // so withholding the label from Clio alone would make the trunks diverge
    // for a reason that has nothing to do with the port.
    std::vector<TrainingLabels> labels = {
        TrainingLabels(actual_ratio, actual_psnr, actual_ct, actual_dt)};
    clio.Train({f}, labels);

    NNWeightsGPU nat{}, cl{};
    SnapshotNative(&nat);
    FlattenClio(clio.DebugWeights(), clio.DebugBiases(), &cl);
    // Tolerance grows slowly: both sides accumulate float32 rounding in a
    // different order (warp reductions vs sequential host loops).
    CompareWeights(nat, cl, "sgd step " + std::to_string(step), 2e-5);
    std::printf("  step %zu: weights compared (native grad_norm=%.6g)\n", step,
                gn);
  }

  // ---- Phase 2b: SGD parity with a REVERSING gradient direction. ----
  // Phase 2's labels rise monotonically, so the combined gradient keeps a
  // stable sign, dot_ema stays positive on both sides, and the anti-flip
  // damping branch is never taken asymmetrically -- it agrees by never
  // firing. That hid a real divergence: NeuroPress accumulates the
  // current-vs-EMA dot over the TRUNK ONLY (nn_gpu.cu stops at
  // DB4), while the port summed all 13576 parameters including the
  // unprojected, sign-stable W5 block, biasing the dot positive and damping
  // less often than upstream.
  //
  // Alternating far-over and far-under labels past the warmup threshold
  // (sgd_call_count > 3) force the dot negative on some steps and positive
  // on others, so a scope mismatch shows up as a factor-of-two step
  // difference in the weights.
  std::printf("\n[phase 2b] SGD parity with alternating gradient direction\n");
  for (size_t step = 0; step < 12; ++step) {
    const auto &c = chunks[step % chunks.size()];
    const bool over = (step % 2) == 0;
    const float actual_ratio = over ? 9.0f : 1.1f;
    const float actual_ct = over ? 40.0f : 2.0f;
    const float actual_dt = over ? 25.0f : 1.5f;
    const int algo_idx = static_cast<int>(step % 8);

    AutoStatsGPU h_stats{};
    h_stats.entropy = c.entropy;
    h_stats.mad_normalized = c.mad;
    h_stats.deriv_normalized = c.deriv;
    h_stats.num_elements = c.size_bytes / sizeof(float);
    cudaMemcpy(d_stats, &h_stats, sizeof(h_stats), cudaMemcpyHostToDevice);

    SGDSample s{};
    s.action = algo_idx;
    s.actual_ratio = actual_ratio;
    s.actual_comp_time = actual_ct;
    s.actual_decomp_time = actual_dt;
    s.actual_psnr = 0.0f;

    float gn = 0; int clipped = 0, cnt = 0;
    int rc = gpucompress::runNNSGD(d_stats, &s, 1, c.size_bytes,
                                   /*error_bound=*/0.0, /*lr=*/0.01f, stream,
                                   &gn, &clipped, &cnt);
    cudaStreamSynchronize(stream);
    if (rc != 0) {
      std::printf("  [FAIL] native SGD rc=%d at 2b step %zu\n", rc, step);
      ++g_failures;
    }
    ++g_checks;

    CompressionFeatures f;
    f.chunk_size_bytes = static_cast<double>(c.size_bytes);
    f.shannon_entropy = c.entropy;
    f.mad = c.mad;
    f.second_derivative_mean = c.deriv;
    f.data_type_float = 1.0;
    f.library_config_id = BaseIdForAlgoIdx(algo_idx) * 10 + 2;
    f.config_balanced = 1.0;
    std::vector<TrainingLabels> labels = {
        TrainingLabels(actual_ratio, 0.0f, actual_ct, actual_dt)};
    clio.Train({f}, labels);

    NNWeightsGPU nat{}, cl{};
    SnapshotNative(&nat);
    FlattenClio(clio.DebugWeights(), clio.DebugBiases(), &cl);
    CompareWeights(nat, cl, "sgd-alt step " + std::to_string(step), 2e-5);
    std::printf("  alt step %2zu (%s): weights compared (grad_norm=%.6g)\n",
                step, over ? "over " : "under", gn);
  }

  // ---- Phase 2c: SGD parity at NONZERO error bounds. ----
  // Both SGD kernels feed the RAW bound into input 3 -- nnSGDKernel under
  // "Use raw values for error_bound and data_size (no log encoding)"
  // (nn_gpu.cu) and nnBatchedDecompSGDKernel via
  // "raw[3] = samp.error_bound_enc" (nn_gpu.cu). Neither applies the
  // 1e-7 inference sentinel.
  //
  // Every other SGD phase trains at error_bound = 0, where the raw bound and
  // the sentinel differ by only 1e-7 and a port that wrongly applied the
  // sentinel during training still matched to ~4e-6 -- small enough to read
  // as float32 noise rather than a divergence. Training at bounds that are
  // orders of magnitude above the sentinel makes that same mistake a large,
  // unambiguous weight mismatch instead of a numeric drift.
  // Both quant values are swept at every bound, and the LOSSLESS-at-nonzero-
  // bound combination is the one that actually discriminates. The kernel's
  // error_bound is a per-call argument, independent of the action's quant
  // bit, so a quant=0 action trained under a nonzero bound is reachable --
  // phase 1b selects exactly that in several cases. Upstream feeds the raw
  // bound there; a port that applied the inference sentinel during training
  // would feed 1e-7 instead, which at eb=0.1 is a six-order-of-magnitude
  // error in input 3. Sweeping only quant=1 would hide this entirely, since
  // the sentinel never applies when quant=1 on either side.
  std::printf("\n[phase 2c] SGD parity at nonzero error bounds\n");
  {
    const double kBounds[] = {1e-6, 1e-3, 1e-1};
    size_t step = 0;
    for (double eb : kBounds) {
      for (int quant = 0; quant <= 1; ++quant) {
        for (size_t i = 0; i < chunks.size(); ++i, ++step) {
          const auto &c = chunks[i];
          const float actual_ratio = 3.0f + 0.4f * static_cast<float>(step);
          const float actual_ct = 10.0f + 2.0f * static_cast<float>(step);
          const float actual_dt = 6.0f + 1.5f * static_cast<float>(step);
          // Lossy configs report a measured PSNR; lossless uses the 0.0
          // sentinel, matching how each is labelled upstream.
          const float actual_psnr =
              quant ? 55.0f + 2.0f * static_cast<float>(i) : 0.0f;
          const int algo_idx = static_cast<int>(step % 8);
          const int action = algo_idx + 8 * quant;

          AutoStatsGPU h_stats{};
          h_stats.entropy = c.entropy;
          h_stats.mad_normalized = c.mad;
          h_stats.deriv_normalized = c.deriv;
          h_stats.num_elements = c.size_bytes / sizeof(float);
          cudaMemcpy(d_stats, &h_stats, sizeof(h_stats),
                     cudaMemcpyHostToDevice);

          SGDSample s{};
          s.action = action;
          s.actual_ratio = actual_ratio;
          s.actual_comp_time = actual_ct;
          s.actual_decomp_time = actual_dt;
          s.actual_psnr = actual_psnr;

          float gn = 0; int clipped = 0, cnt = 0;
          int rc = gpucompress::runNNSGD(d_stats, &s, 1, c.size_bytes, eb,
                                         /*lr=*/0.01f, stream, &gn, &clipped,
                                         &cnt);
          cudaStreamSynchronize(stream);
          if (rc != 0) {
            std::printf("  [FAIL] native SGD rc=%d at 2c step %zu\n", rc, step);
            ++g_failures;
          }
          ++g_checks;

          CompressionFeatures f;
          f.chunk_size_bytes = static_cast<double>(c.size_bytes);
          f.shannon_entropy = c.entropy;
          f.mad = c.mad;
          f.second_derivative_mean = c.deriv;
          f.data_type_float = 1.0;
          f.library_config_id = BaseIdForAlgoIdx(algo_idx) * 10 + 2;
          f.config_balanced = 1.0;
          f.quantize = quant ? 1.0 : 0.0;
          f.error_bound = eb;
          std::vector<TrainingLabels> labels = {
              TrainingLabels(actual_ratio, actual_psnr, actual_ct, actual_dt)};
          clio.Train({f}, labels);

          NNWeightsGPU nat{}, cl{};
          SnapshotNative(&nat);
          FlattenClio(clio.DebugWeights(), clio.DebugBiases(), &cl);
          CompareWeights(nat, cl,
                         "sgd-eb " + std::to_string(eb) + " q" +
                             std::to_string(quant) + " step " +
                             std::to_string(i),
                         2e-5);
        }
        std::printf("  eb=%-8g quant=%d trained %zu steps\n", eb, quant,
                    chunks.size());
      }
    }
  }

  // ---- Phase 3: deferred decompression-head parity. ----
  std::printf("\n[phase 3] deferred decomp-head SGD parity\n");
  {
    std::vector<DeferredDecompSample> nat_batch;
    std::vector<CompressionFeatures> clio_feats;
    std::vector<double> clio_times;
    for (size_t i = 0; i < chunks.size(); ++i) {
      const auto &c = chunks[i];
      const int algo_idx = static_cast<int>(i % 8);
      const float measured = 5.0f + 4.0f * static_cast<float>(i);

      DeferredDecompSample s{};
      s.action = algo_idx;
      s.entropy = static_cast<float>(c.entropy);
      s.mad_normalized = static_cast<float>(c.mad);
      s.deriv_normalized = static_cast<float>(c.deriv);
      s.error_bound_enc = 0.0f;
      s.data_size_enc = static_cast<float>(c.size_bytes);
      s.actual_decomp_ms = measured;
      nat_batch.push_back(s);

      CompressionFeatures f;
      f.chunk_size_bytes = static_cast<double>(c.size_bytes);
      f.shannon_entropy = c.entropy;
      f.mad = c.mad;
      f.second_derivative_mean = c.deriv;
      f.data_type_float = 1.0;
      f.library_config_id = BaseIdForAlgoIdx(algo_idx) * 10 + 2;
      f.config_balanced = 1.0;
      clio_feats.push_back(f);
      clio_times.push_back(measured);
    }

    float gn = 0;
    int rc = gpucompress::runBatchedDecompSGD(
        nat_batch.data(), static_cast<int>(nat_batch.size()), 0.01f, &gn);
    cudaDeviceSynchronize();
    Check(rc == 0, "native batched decomp SGD rc");

    clio.TrainDecompHead(clio_feats, clio_times);

    NNWeightsGPU nat{}, cl{};
    SnapshotNative(&nat);
    FlattenClio(clio.DebugWeights(), clio.DebugBiases(), &cl);
    CompareWeights(nat, cl, "decomp-head", 2e-5);
    std::printf("  decomp-head batch of %zu compared (native grad_norm=%.6g)\n",
                nat_batch.size(), gn);
  }

  // ---- Phase 3b: decomp head under REPLAYED, growing batches. ----
  // LearnDecompTime no longer consumes a record once trained: upstream's
  // gpucompress_batched_decomp_sgd() re-sweeps its whole diagnostics store on
  // every read and never erases entries, so a chunk read earlier is replayed
  // into every later batch and the trust region (0.15 * mean|err|) is
  // computed over that growing population. Drive exactly that shape here --
  // batches of 2, then 4, then 7 over the same records -- and include
  // sub-millisecond measurements, which the previous phase never produced.
  std::printf("\n[phase 3b] decomp head, replayed growing batches\n");
  for (size_t take : {size_t(2), size_t(4), size_t(7)}) {
    std::vector<DeferredDecompSample> nat_batch;
    std::vector<CompressionFeatures> clio_feats;
    std::vector<double> clio_times;
    for (size_t i = 0; i < take; ++i) {
      const auto &c = chunks[i % chunks.size()];
      const int algo_idx = static_cast<int>(i % 8);
      // Deliberately spans the sub-millisecond range. Both sides receive the
      // SAME value here: the 1 ms policy floor lives at the call site (Clio's
      // LearnDecompTime, upstream's DiagnosticsStore::recordDecompMs), not in
      // the kernel, so this isolates the kernel's own [0.01, 5000] clamp.
      const float measured = (i % 2 == 0) ? 0.35f : 6.0f + 3.0f * i;

      DeferredDecompSample s{};
      s.action = algo_idx;
      s.entropy = static_cast<float>(c.entropy);
      s.mad_normalized = static_cast<float>(c.mad);
      s.deriv_normalized = static_cast<float>(c.deriv);
      s.error_bound_enc = 0.0f;
      s.data_size_enc = static_cast<float>(c.size_bytes);
      s.actual_decomp_ms = measured;
      nat_batch.push_back(s);

      CompressionFeatures f;
      f.chunk_size_bytes = static_cast<double>(c.size_bytes);
      f.shannon_entropy = c.entropy;
      f.mad = c.mad;
      f.second_derivative_mean = c.deriv;
      f.data_type_float = 1.0;
      f.library_config_id = BaseIdForAlgoIdx(algo_idx) * 10 + 2;
      f.config_balanced = 1.0;
      clio_feats.push_back(f);
      clio_times.push_back(measured);
    }

    float gn = 0;
    int rc = gpucompress::runBatchedDecompSGD(
        nat_batch.data(), static_cast<int>(nat_batch.size()), 0.01f, &gn);
    cudaDeviceSynchronize();
    Check(rc == 0, "native batched decomp SGD rc (replay)");

    clio.TrainDecompHead(clio_feats, clio_times);

    NNWeightsGPU nat{}, cl{};
    SnapshotNative(&nat);
    FlattenClio(clio.DebugWeights(), clio.DebugBiases(), &cl);
    CompareWeights(nat, cl, "decomp-head replay n=" + std::to_string(take),
                   2e-5);
    std::printf("  replay batch of %zu compared (native grad_norm=%.6g)\n",
                take, gn);
  }

  // ---- Phase 3c: decomp head at NONZERO error bounds. ----
  // DeferredDecompSample carries its own error_bound_enc, which
  // nnBatchedDecompSGDKernel copies straight into raw[3] (nn_gpu.cu).
  // Both earlier decomp phases pin it to 0.0f, so the value Clio derives for
  // that slot has never actually been varied against upstream's.
  // As in phase 2c, quant=0 at a nonzero bound is the discriminating case:
  // the sample's error_bound_enc reaches raw[3] untouched on both sides only
  // if the port keeps the sentinel out of its training path.
  std::printf("\n[phase 3c] decomp head at nonzero error bounds\n");
  for (double eb : {1e-5, 1e-2}) {
    for (int quant = 0; quant <= 1; ++quant) {
      std::vector<DeferredDecompSample> nat_batch;
      std::vector<CompressionFeatures> clio_feats;
      std::vector<double> clio_times;
      for (size_t i = 0; i < chunks.size(); ++i) {
        const auto &c = chunks[i];
        const int algo_idx = static_cast<int>(i % 8);
        const float measured = 4.0f + 3.0f * static_cast<float>(i);

        DeferredDecompSample s{};
        s.action = algo_idx + 8 * quant;
        s.entropy = static_cast<float>(c.entropy);
        s.mad_normalized = static_cast<float>(c.mad);
        s.deriv_normalized = static_cast<float>(c.deriv);
        s.error_bound_enc = static_cast<float>(eb);
        s.data_size_enc = static_cast<float>(c.size_bytes);
        s.actual_decomp_ms = measured;
        nat_batch.push_back(s);

        CompressionFeatures f;
        f.chunk_size_bytes = static_cast<double>(c.size_bytes);
        f.shannon_entropy = c.entropy;
        f.mad = c.mad;
        f.second_derivative_mean = c.deriv;
        f.data_type_float = 1.0;
        f.library_config_id = BaseIdForAlgoIdx(algo_idx) * 10 + 2;
        f.config_balanced = 1.0;
        f.quantize = quant ? 1.0 : 0.0;
        f.error_bound = eb;
        clio_feats.push_back(f);
        clio_times.push_back(measured);
      }

      float gn = 0;
      int rc = gpucompress::runBatchedDecompSGD(
          nat_batch.data(), static_cast<int>(nat_batch.size()), 0.01f, &gn);
      cudaDeviceSynchronize();
      Check(rc == 0, "native batched decomp SGD rc (nonzero eb)");

      clio.TrainDecompHead(clio_feats, clio_times);

      NNWeightsGPU nat{}, cl{};
      SnapshotNative(&nat);
      FlattenClio(clio.DebugWeights(), clio.DebugBiases(), &cl);
      CompareWeights(nat, cl,
                     "decomp-head eb=" + std::to_string(eb) + " q" +
                         std::to_string(quant),
                     2e-5);
      std::printf("  eb=%-8g quant=%d batch of %zu compared (grad_norm=%.6g)\n",
                  eb, quant, nat_batch.size(), gn);
    }
  }

  cudaFree(d_stats);
  cudaStreamDestroy(stream);
  gpucompress::cleanupNN();

  std::printf("\n----- observed worst-case deviation -----\n");
  std::printf("  predictions : max rel err = %.6g  (%s)\n", g_max_rel_pred,
              g_max_rel_what.c_str());
  std::printf("  weights     : max abs err = %.6g  (%s %s)\n",
              g_max_abs_weight, g_max_abs_phase.c_str(), g_max_abs_layer);
  std::printf("\n===== %d checks, %d failures =====\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
