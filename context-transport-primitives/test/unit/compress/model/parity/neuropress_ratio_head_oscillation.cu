/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file neuropress_ratio_head_oscillation.cu
 * @brief Why the ratio head fails to converge on some regimes.
 *
 * The ratio head ping-pongs between the 0.1 forward-clamp and the 100x cap on
 * the bimodal and heavy-tail regimes, firing SGD on every chunk without ever
 * converging. This drives NeuroPress's OWN nnSGDKernel on its own AutoStatsGPU
 * against a fixed outcome, snapshots the weights around each step, and replays
 * the forward pass on the host so the UNCLAMPED output is visible -- the value
 * runNNInference returns has already been through max(0.1, min(ratio, 1e5))
 * then min(100, ratio), which hides exactly the quantity in question.
 *
 * The cause is not the SGD at all. mad and deriv are absolute-magnitude
 * features standardized against constants baked into the weights file
 * (mad mean 0.1892 std 0.1062, deriv mean 0.2389 std 0.2113), so a chunk's
 * AMPLITUDE, not its distribution shape, decides whether it lands inside the
 * training distribution. Bimodal data at +-100 sits ~941 sigma out, driving the
 * last hidden layer to activations near 229 instead of ~3. The trust region
 * bounds the step in WEIGHT space (|dW| ~ 0.01); the output moves by roughly
 * |dW5| * |h4|, so the same bounded step swings the output by tens of units and
 * lands on a clamp every time.
 *
 * The sweep below holds the distribution fixed and varies only amplitude, which
 * is what separates the two explanations: at amplitude 0.2 (in distribution)
 * the head converges and never touches a clamp; at 100 it oscillates. A
 * mid-range ratio-30 target runs at each amplitude as a control.
 *
 * This is upstream's behaviour, faithfully ported -- not a clio defect.
 */

#include <cuda_runtime.h>
#include <algorithm>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <set>
#include <string>
#include <vector>

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

/* Host-visible copy of the live device weights. Declared rather than pulled in
   via gpucompress.h so this harness does not need the whole public API. */
extern "C" int gpucompress_nn_save_snapshot(void *dst);

extern cudaStream_t g_sgd_stream;
extern cudaEvent_t g_sgd_done;

namespace {

constexpr size_t kElems = 1u << 20;          // 4 MiB of float32
constexpr size_t kBytes = kElems * sizeof(float);
constexpr int kSteps = 40;
constexpr double kErrorBound = 0.0;          // lossless

int g_failures = 0;
void Check(bool c, const std::string &w) {
  if (c) return;
  ++g_failures;
  std::printf("  [FAIL] %s\n", w.c_str());
}
/** Data amplitude. The shipped model normalizes MAD with mean 0.1892 and std
 *  0.1062, so amplitude is what decides whether a chunk lands inside the
 *  training distribution -- the distribution's SHAPE does not. */
float g_amp = 100.0f;

/** The bimodal generator from the regime harness: two clusters, ratio ~1. */
__global__ void FillBimodal(float *b, size_t n, unsigned seed, float amp) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t x = static_cast<uint32_t>(i) * 2654435761u + seed;
  x ^= x >> 15; x *= 0x85EBCA6Bu; x ^= x >> 13; x *= 0xC2B2AE35u; x ^= x >> 16;
  const float u = static_cast<float>(x) / 4294967295.0f;
  b[i] = amp * ((u < 0.5f ? -1.0f : 1.0f) + 0.005f * (u - 0.5f));
}

/** Upstream's forward pass, on the host, from a weight snapshot.
 *
 * Reproduces nnForwardPass (nn_gpu.cu) exactly so the UNCLAMPED network output
 * is visible. runNNInference only returns the value after
 * `max(0.1, min(ratio, 1e5))` then `min(100, ratio)`, which is precisely the
 * information needed to tell "the head is oscillating" from "the head is fine
 * and the clamps are showing their edges".
 */
struct Fwd { float y2; float raw_ratio; float h4max; float xz[NN_INPUT_DIM]; };
Fwd HostForward(const NNWeightsGPU &w, int action, double entropy, double mad,
                double deriv, double ds, double eb) {
  const int algo = action % 8, quant = (action / 8) % 2, shuf = (action / 16) % 2;
  float raw[NN_INPUT_DIM] = {
      static_cast<float>(algo), static_cast<float>(quant),
      static_cast<float>(shuf), (quant == 0) ? 1e-7f : static_cast<float>(eb),
      static_cast<float>(ds), static_cast<float>(entropy),
      static_cast<float>(mad), static_cast<float>(deriv)};
  float x[NN_INPUT_DIM];
  for (int i = 0; i < NN_INPUT_DIM; ++i) {
    float sd = w.x_stds[i]; if (sd < 1e-8f) sd = 1e-8f;
    x[i] = (raw[i] - w.x_means[i]) / sd;
  }
  float h1[NN_HIDDEN_DIM], h2[NN_HIDDEN_DIM], h3[NN_HIDDEN_DIM], h4[NN_HIDDEN_DIM];
  auto relu_layer = [](const float *W, const float *B, const float *in, int fan,
                       float *out) {
    for (int j = 0; j < NN_HIDDEN_DIM; ++j) {
      float s = B[j];
      for (int i = 0; i < fan; ++i) s += W[j * fan + i] * in[i];
      out[j] = s > 0.0f ? s : 0.0f;
    }
  };
  relu_layer(w.w1, w.b1, x,  NN_INPUT_DIM,  h1);
  relu_layer(w.w2, w.b2, h1, NN_HIDDEN_DIM, h2);
  relu_layer(w.w3, w.b3, h2, NN_HIDDEN_DIM, h3);
  relu_layer(w.w4, w.b4, h3, NN_HIDDEN_DIM, h4);
  float y2 = w.b5[2];
  for (int i = 0; i < NN_HIDDEN_DIM; ++i) y2 += w.w5[2 * NN_HIDDEN_DIM + i] * h4[i];
  Fwd f{y2, std::expm1(y2 * w.y_stds[2] + w.y_means[2]), 0.0f, {}};
  for (int i = 0; i < NN_HIDDEN_DIM; ++i) f.h4max = std::max(f.h4max, h4[i]);
  for (int i = 0; i < NN_INPUT_DIM; ++i) f.xz[i] = x[i];
  return f;
}

double L2(const float *a, const float *b, int n) {
  double s = 0.0;
  for (int i = 0; i < n; ++i) { const double d = a[i] - b[i]; s += d * d; }
  return std::sqrt(s);
}

/** One convergence run: same action, same statistics, same measured outcome,
 *  fed to upstream's kernel `kSteps` times. Returns the predicted ratio the
 *  inference kernel reports before each step. */
std::vector<float> Run(const char *nnwt, AutoStatsGPU *d_stats, int action,
                       float target_ratio, float lr, cudaStream_t stream) {
  std::vector<float> trace;
  if (!gpucompress::loadNNFromBinary(nnwt)) return trace;  // fresh weights
  double h_ent = 0, h_mad = 0, h_der = 0;
  {
    AutoStatsGPU h{};
    cudaMemcpy(&h, d_stats, sizeof(AutoStatsGPU), cudaMemcpyDeviceToHost);
    h_ent = h.entropy; h_mad = h.mad_normalized; h_der = h.deriv_normalized;
  }
  NNWeightsGPU wb{}, wa{};
  const float y2_target =
      (std::log1p(std::max(0.5f, target_ratio)) - 0.0f);  // filled below
  for (int s = 0; s < kSteps; ++s) {
    float r = 0, ct = 0, dt = 0, ps = 0;
    const int a = gpucompress::runNNInference(h_ent, h_mad, h_der, kBytes,
                                              kErrorBound, stream,
                                              &r, &ct, &dt, &ps, nullptr);
    cudaStreamSynchronize(stream);
    if (a < 0) break;
    trace.push_back(r);

    gpucompress_nn_save_snapshot(&wb);
    const Fwd f = HostForward(wb, action, h_ent, h_mad, h_der, kBytes, kErrorBound);
    const float tgt = (std::log1p(std::max(0.5f, target_ratio)) - wb.y_means[2]) /
                      std::max(wb.y_stds[2], 1e-8f);
    (void)y2_target;

    SGDSample smp{};
    smp.action = action;
    smp.actual_ratio = target_ratio;
    // Measured comp time held at the prediction so the ratio head is the only
    // one under pressure -- otherwise a time miss drives the shared layers and
    // the ratio trace is not attributable.
    smp.actual_comp_time = ct;
    smp.actual_decomp_time = 0.0f;   // <=0 -> upstream skips that head
    smp.actual_psnr = -1.0f;         // <0  -> upstream skips the PSNR head
    float gn = 0; int gc = 0, gs = 0;
    gpucompress::runNNSGD(d_stats, &smp, 1, kBytes, kErrorBound, lr,
                          g_sgd_stream, &gn, &gc, &gs);
    cudaStreamSynchronize(g_sgd_stream);
    cudaDeviceSynchronize();
    gpucompress_nn_save_snapshot(&wa);

    if (s == 0) {
      std::printf("      normalized inputs (z-scores vs training set):\n"
                  "        algo %+.2f quant %+.2f shuf %+.2f eb %+.2f "
                  "size %+.2f entropy %+.2f mad %+.2f deriv %+.2f\n"
                  "        max h4 activation %.1f | y_mean[2] %.3f y_std[2] %.3f\n",
                  f.xz[0], f.xz[1], f.xz[2], f.xz[3], f.xz[4], f.xz[5], f.xz[6],
                  f.xz[7], f.h4max, wb.y_means[2], wb.y_stds[2]);
    }
    if (s < 12) {
      const double d_trunk =
          L2(wb.w1, wa.w1, NN_HIDDEN_DIM * NN_INPUT_DIM) +
          L2(wb.w2, wa.w2, NN_HIDDEN_DIM * NN_HIDDEN_DIM) +
          L2(wb.w3, wa.w3, NN_HIDDEN_DIM * NN_HIDDEN_DIM) +
          L2(wb.w4, wa.w4, NN_HIDDEN_DIM * NN_HIDDEN_DIM);
      const double d_head =
          L2(wb.w5 + 2 * NN_HIDDEN_DIM, wa.w5 + 2 * NN_HIDDEN_DIM, NN_HIDDEN_DIM);
      std::printf("      step %2d  y2 %8.3f  target %8.3f  err %8.3f  "
                  "raw_ratio %12.3f  clamped %8.3f  |dW_trunk| %.5f  "
                  "|dW5row2| %.5f  db5[2] %+.5f\n",
                  s, f.y2, tgt, f.y2 - tgt, f.raw_ratio, r, d_trunk, d_head,
                  wa.b5[2] - wb.b5[2]);
    }
  }
  return trace;
}

/** Steps that landed on the forward clamps. */
int Clamped(const std::vector<float> &t) {
  int n = 0;
  for (float v : t) if (v <= 0.1001f || v >= 99.999f) ++n;
  return n;
}

void Report(const char *label, const std::vector<float> &t, float target) {
  if (t.empty()) { std::printf("  %s: no trace\n", label); return; }
  std::set<int> distinct;
  int at_floor = 0, at_cap = 0;
  for (float v : t) {
    distinct.insert(static_cast<int>(v * 1000.0f));
    if (v <= 0.1001f) ++at_floor;
    if (v >= 99.999f) ++at_cap;
  }
  std::printf("  %-22s target %7.3f | first 8:", label, target);
  for (int i = 0; i < 8 && i < static_cast<int>(t.size()); ++i)
    std::printf(" %.3f", t[i]);
  std::printf("\n  %-22s last  4:", "");
  for (size_t i = (t.size() > 4 ? t.size() - 4 : 0); i < t.size(); ++i)
    std::printf(" %.3f", t[i]);
  std::printf("\n  %-22s distinct %zu | at floor %d | at cap %d | final %.3f\n\n",
              "", distinct.size(), at_floor, at_cap, t.back());
}

}  // namespace

int main(int argc, char **argv) {
  const std::string dir =
      (argc > 1) ? argv[1] : std::string(CLIO_CTP_NEUROPRESS_WEIGHTS_DIR);
  const std::string nnwt = dir + "/model.nnwt";

  int devs = 0;
  if (cudaGetDeviceCount(&devs) != cudaSuccess || devs == 0) {
    std::printf("No CUDA device -- skipping.\n");
    return 77;
  }

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cudaStreamCreate(&g_sgd_stream);
  cudaEventCreate(&g_sgd_done);

  float *d = nullptr;
  if (cudaMalloc(&d, kBytes) != cudaSuccess) return 1;

  /* Same bimodal SHAPE at three amplitudes. The model's MAD normalization was
     fit with mean 0.1892 / std 0.1062, so amplitude alone decides whether a
     chunk lands inside the training distribution. */
  for (float amp : {100.0f, 1.0f, 0.2f}) {
    g_amp = amp;
    FillBimodal<<<static_cast<int>((kElems + 255) / 256), 256>>>(
        d, kElems, 0x9E3779B9u, g_amp);
    cudaDeviceSynchronize();

    AutoStatsGPU *d_stats =
        gpucompress::runStatsKernelsNoSync(d, kBytes, stream);
    cudaStreamSynchronize(stream);
    Check(d_stats != nullptr, "upstream stats ran");
    if (!d_stats) return 1;

    AutoStatsGPU h{};
    cudaMemcpy(&h, d_stats, sizeof(AutoStatsGPU), cudaMemcpyDeviceToHost);
    std::printf("\n================ amplitude %g ================\n"
                "entropy %.4f  mad %.4f  deriv %.4f   "
                "(mad is %+.1f sigma from the training mean)\n\n",
                amp, h.entropy, h.mad_normalized, h.deriv_normalized,
                (h.mad_normalized - 0.1892) / 0.1062);

    char lbl[64];
    std::snprintf(lbl, sizeof(lbl), "amp %-5g ratio 1.0  lr=0.10", amp);
    const std::vector<float> t1 = Run(nnwt.c_str(), d_stats, 0, 1.0f, 0.1f, stream);
    Report(lbl, t1, 1.0f);
    std::snprintf(lbl, sizeof(lbl), "amp %-5g ratio 30.0 lr=0.10", amp);
    const std::vector<float> t30 = Run(nnwt.c_str(), d_stats, 0, 30.0f, 0.1f, stream);
    Report(lbl, t30, 30.0f);

    if (amp <= 0.5f) {
      /* In distribution: the machinery works. Nothing may touch a clamp, and
         the head must land near the target it was trained toward. */
      Check(Clamped(t1) == 0 && Clamped(t30) == 0,
            "in-distribution run touched a clamp");
      Check(!t1.empty() && t1.back() > 0.5f && t1.back() < 2.0f,
            "in-distribution ratio-1 target did not converge");
      Check(!t30.empty() && t30.back() > t30.front() * 3.0f,
            "in-distribution ratio-30 control did not climb");
    } else if (amp >= 50.0f) {
      /* Far out of distribution: upstream oscillates on the clamps. Asserted so
         a future change to the normalization or trust region shows up here. */
      Check(Clamped(t1) > static_cast<int>(t1.size()) / 2,
            "far-OOD run unexpectedly stayed off the clamps");

      /* Proposed fix, measured through upstream's UNMODIFIED kernel.
         Clamping the standardized input at +-kZ is arithmetically identical to
         feeding raw stats already clamped to mean +- kZ*std, which is a thing
         this harness can do without touching upstream. If the head converges
         here, an input-side clamp is enough to restore online adaptation on a
         workload the model has never seen. */
      constexpr float kZ = 3.0f;
      AutoStatsGPU c = h;
      c.mad_normalized = std::min<double>(h.mad_normalized, 0.1892 + kZ * 0.1062);
      c.deriv_normalized = std::min<double>(h.deriv_normalized, 0.2389 + kZ * 0.2113);
      AutoStatsGPU *d_clamped = nullptr;
      cudaMalloc(&d_clamped, sizeof(AutoStatsGPU));
      cudaMemcpy(d_clamped, &c, sizeof(AutoStatsGPU), cudaMemcpyHostToDevice);
      std::printf("---- same chunk, standardized inputs clamped to +-%.0f sigma "
                  "(mad %.4f -> %.4f, deriv %.4f -> %.4f) ----\n\n",
                  kZ, h.mad_normalized, c.mad_normalized, h.deriv_normalized,
                  c.deriv_normalized);
      const std::vector<float> z1 = Run(nnwt.c_str(), d_clamped, 0, 1.0f, 0.1f, stream);
      Report("z-clamp ratio 1.0", z1, 1.0f);
      const std::vector<float> z30 = Run(nnwt.c_str(), d_clamped, 0, 30.0f, 0.1f, stream);
      Report("z-clamp ratio 30.0", z30, 30.0f);
      Check(Clamped(z1) < Clamped(t1),
            "input z-clamp did not reduce clamp hits");
      cudaFree(d_clamped);
    }
  }

  gpucompress::cleanupNN();
  gpucompress::freeStatsWorkspace();
  cudaFree(d);
  std::printf("===== %s =====\n", g_failures ? "FAILURES" : "0 failures");
  return g_failures ? 1 : 0;
}
