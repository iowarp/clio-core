/** Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology All rights rese... */


#include "clio_ctp/compress/model/neuropress_nn_gpu_kernels.h"
#include "clio_ctp/compress/preprocess/prediction_reuse_gpu.h"
// For ctp::DeviceFeatureStats -- the device-resident feature triple the device-stats inference k...
#include "clio_ctp/compress/preprocess/data_stats_gpu.h"

#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>
#include <algorithm>
#include <cstddef>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

namespace ctp::compress::model::gpu {

namespace {
constexpr int kInputDim = 8;
constexpr int kHiddenDim = 64;
constexpr int kOutputDim = 8;
constexpr int kMaxSamples = 8;  // NeuroPress's NN_MAX_SGD_SAMPLES

constexpr int kW1 = kHiddenDim * kInputDim;   // 512
constexpr int kW234 = kHiddenDim * kHiddenDim;  // 4096 each (W2,W3,W4)
constexpr int kW5 = kOutputDim * kHiddenDim;   // 512
constexpr int kParamCount =
    kW1 + kHiddenDim +          // w1,b1
    kW234 + kHiddenDim +        // w2,b2
    kW234 + kHiddenDim +        // w3,b3
    kW234 + kHiddenDim +        // w4,b4
    kW5 + kOutputDim;           // w5,b5  == 13576, matches NeuroPress's SGD_REGION

// Offsets into a flat kParamCount-length weights-then-biases buffer.
constexpr int kOffW1 = 0;
constexpr int kOffB1 = kOffW1 + kW1;
constexpr int kOffW2 = kOffB1 + kHiddenDim;
constexpr int kOffB2 = kOffW2 + kW234;
constexpr int kOffW3 = kOffB2 + kHiddenDim;
constexpr int kOffB3 = kOffW3 + kW234;
constexpr int kOffW4 = kOffB3 + kHiddenDim;
constexpr int kOffB4 = kOffW4 + kW234;
constexpr int kOffW5 = kOffB4 + kHiddenDim;
constexpr int kOffB5 = kOffW5 + kW5;
static_assert(kOffB5 + kOutputDim == kParamCount, "offset layout mismatch");

/** Env-gated deviations from the ported rule. */
struct AdaptOptions {
  bool adaptive = false;      // CLIO_NEUROPRESS_SGD_RULE=adaptive
  float lr = 0.0f;            // CLIO_NEUROPRESS_SGD_LR (<=0: caller's rate)
  int head_steps = 32;        // CLIO_NEUROPRESS_SGD_HEAD_STEPS
  float trunk_scale = 0.1f;   // CLIO_NEUROPRESS_SGD_TRUNK_SCALE
  float decay_tau = 100.0f;   // CLIO_NEUROPRESS_SGD_DECAY_TAU (0: no decay)
  float max_step = 0.02f;     // CLIO_NEUROPRESS_SGD_MAX_STEP
  float grad_clip = 1.0f;     // CLIO_NEUROPRESS_SGD_GRAD_CLIP (<=0: off)
  float momentum = 0.85f;     // CLIO_NEUROPRESS_SGD_MOMENTUM
  // Output-space trust region for the SHIPPED rule, in standardised output units: one SGD call m...
  float out_delta = 0.5f;     // CLIO_NEUROPRESS_SGD_OUT_DELTA
  float ratio_target_cap = 100.0f;  // CLIO_NEUROPRESS_SGD_RATIO_TARGET_CAP
};
}  // namespace

/** @brief Device-resident state. Persists for the handle's lifetime. */
struct NeuroPressGpuWeights {
  float params[kParamCount];        // w1,b1,w2,b2,w3,b3,w4,b4,w5,b5
  float x_means[kInputDim];
  float x_stds[kInputDim];
  float y_means[kOutputDim];
  float y_stds[kOutputDim];
  // Feature bounds in STANDARDISED units, (x_min - mean)/std and (x_max - mean)/std, consumed by...
  float x_los[kInputDim];
  float x_his[kInputDim];

  // Online-learning state (never serialized to .nnwt, matches upstream).
  float log_var[kOutputDim];
  // NOTE: the EMA gradient does NOT live here.
  int sgd_call_count;

  // Persistent scratch for Train() -- avoids per-call cudaMalloc.
  float act_x[kMaxSamples][kInputDim];
  // Only the post-ReLU activations are kept.
  float act_h1[kMaxSamples][kHiddenDim];
  float act_h2[kMaxSamples][kHiddenDim];
  float act_h3[kMaxSamples][kHiddenDim];
  float act_h4[kMaxSamples][kHiddenDim];
  float act_y[kMaxSamples][kOutputDim];
  float d5_clamped[kMaxSamples][kOutputDim];
  float d5_raw[kMaxSamples][kOutputDim];
  float combined[kParamCount];
  // ONE GRADIENT SLICE PER OUTPUT HEAD, so the eight backward passes can run as eight blocks ins...
  float out_grad[kOutputDim][kParamCount];
  /** Each head's clipped step size, handed to phase C so the scale and the accumulate stay ONE ... */
  float lr_head[kOutputDim];
  /** Scalars and per-sample maxima carried between the phases of the update step, which used to... */
  float sgd_step;
  float sgd_g_norm;
  float dy_sample[kMaxSamples];
  // One slot per SAMPLE, not one shared slot rebuilt per target output: see the hoist in SGDKernel.
  float dz4_all[kMaxSamples][kOutputDim][kHiddenDim];
};

namespace {

/** Standardise raw input i and soft-bound it to the training range. */
__device__ __forceinline__ float NeuroPressStandardize(
    const NeuroPressGpuWeights *__restrict__ w, int i, float raw) {
  float sd = w->x_stds[i];
  if (sd < 1e-8f) sd = 1e-8f;
  const float centred = raw - w->x_means[i];
  return NeuroPressSoftBoundSigma(centred / sd, w->x_los[i], w->x_his[i]);
}

/** The options, parsed ONCE per process on first use. */
const AdaptOptions &Opts() {
  static const AdaptOptions o = [] {
    AdaptOptions a;
    auto num = [](const char *name, float dflt) {
      const char *v = std::getenv(name);
      return (v != nullptr && *v != '\0') ? std::strtof(v, nullptr) : dflt;
    };
    const char *rule = std::getenv("CLIO_NEUROPRESS_SGD_RULE");
    a.adaptive = (rule != nullptr && std::strcmp(rule, "adaptive") == 0);
    a.lr = num("CLIO_NEUROPRESS_SGD_LR", 0.0f);
    const char *hs = std::getenv("CLIO_NEUROPRESS_SGD_HEAD_STEPS");
    a.head_steps = (hs != nullptr && *hs != '\0') ? std::atoi(hs) : 32;
    a.trunk_scale = num("CLIO_NEUROPRESS_SGD_TRUNK_SCALE", 0.1f);
    a.decay_tau = num("CLIO_NEUROPRESS_SGD_DECAY_TAU", 100.0f);
    a.max_step = num("CLIO_NEUROPRESS_SGD_MAX_STEP", 0.02f);
    a.grad_clip = num("CLIO_NEUROPRESS_SGD_GRAD_CLIP", 1.0f);
    a.momentum = num("CLIO_NEUROPRESS_SGD_MOMENTUM", 0.85f);
    a.out_delta = num("CLIO_NEUROPRESS_SGD_OUT_DELTA", 0.5f);
    // Upstream trains toward min(ratio, 10000) (nn_gpu.cu:773); 100, the
    // ranking cap, keeps the ratio head from saturating on VPIC.
    a.ratio_target_cap =
        num("CLIO_NEUROPRESS_SGD_RATIO_TARGET_CAP", 100.0f);
    if (!(a.ratio_target_cap > 0.0f)) a.ratio_target_cap = 100.0f;
    return a;
  }();
  return o;
}

/** Per-flow EMA gradient buffer. */
struct EmaRegistry {
  std::mutex mutex;
  std::vector<float *> buffers;
};

/** The registry is allocated on first use and DELIBERATELY NEVER DESTROYED. */
EmaRegistry &Registry() {
  static EmaRegistry *r = new EmaRegistry();
  return *r;
}

/** SGD stream, completion event, and "has SGD ever run" flag. */
struct SgdSync {
  cudaStream_t stream = nullptr;
  cudaEvent_t done = nullptr;
  std::atomic<bool> ever_fired{false};
  bool ok = false;
};

SgdSync &Sgd() {
  /** Initialized in place: SgdSync holds an atomic and so is not copyable, and returning one fr... */
  static SgdSync s;
  static const bool once = [] {
    /** Non-blocking: this stream must not implicitly synchronize with the legacy default stream... */
    s.ok = cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking) ==
               cudaSuccess &&
           cudaEventCreateWithFlags(&s.done, cudaEventDisableTiming) ==
               cudaSuccess;
    return true;
  }();
  (void)once;
  return s;
}

/** Persistent per-thread SGD sample buffer, grown rather than reallocated. */
struct SgdScratch {
  void *d_samples = nullptr;
  size_t bytes = 0;
};

SgdScratch &SgdSamples() {
  static thread_local SgdScratch s;
  return s;
}

/** SGD-owned copy of the chunk's statistics: the resident buffer is reused by the next chunk wh... */
ctp::DeviceFeatureStats *SgdStatsSnapshot() {
  static thread_local ctp::DeviceFeatureStats *p = [] {
    ctp::DeviceFeatureStats *q = nullptr;
    if (cudaMalloc(&q, sizeof(ctp::DeviceFeatureStats)) != cudaSuccess) {
      return static_cast<ctp::DeviceFeatureStats *>(nullptr);
    }
    return q;
  }();
  return p;
}

bool EnsureSgdSamples(SgdScratch &s, size_t need) {
  if (need <= s.bytes) return true;
  cudaFree(s.d_samples);
  s.d_samples = nullptr;
  s.bytes = 0;
  if (cudaMalloc(&s.d_samples, need) != cudaSuccess) return false;
  s.bytes = need;
  return true;
}

/* Called by the inference paths before they read the weights. */
void SgdWaitIfEverFired(cudaStream_t st) {
  SgdSync &g = Sgd();
  if (g.ok && g.ever_fired.load(std::memory_order_acquire)) {
    cudaStreamWaitEvent(st, g.done, 0);
  }
}

/** Called by the HOST-side weight readback before it copies. */
void SgdHostWaitIfEverFired() {
  SgdSync &g = Sgd();
  if (g.ok && g.ever_fired.load(std::memory_order_acquire)) {
    cudaEventSynchronize(g.done);
  }
}

float *EmaBuffer() {
  static thread_local float *buf = [] {
    float *p = nullptr;
    if (cudaMalloc(&p, kParamCount * sizeof(float)) != cudaSuccess) return
        static_cast<float *>(nullptr);
    cudaMemset(p, 0, kParamCount * sizeof(float));
    EmaRegistry &reg = Registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    reg.buffers.push_back(p);
    return p;
  }();
  return buf;
}

void ResetAllEmaBuffers() {
  EmaRegistry &reg = Registry();
  std::lock_guard<std::mutex> lock(reg.mutex);
  for (float *p : reg.buffers) {
    if (p) cudaMemset(p, 0, kParamCount * sizeof(float));
  }
}

}  // namespace

float NeuroPressPredTimeFloorMs() {
  static const float v = [] {
    const char *e = std::getenv("CLIO_NEUROPRESS_PRED_TIME_FLOOR_MS");
    const float f = (e != nullptr && *e != '\0') ? std::strtof(e, nullptr) : 1.0f;
    return (f > 0.0f) ? f : 1.0f;
  }();
  return v;
}

NeuroPressGpuWeights *NeuroPressGpuLoad(const float *weights, size_t weights_len,
                                        const float *biases, size_t biases_len,
                                        const float *x_means, const float *x_stds,
                                        const float *y_means, const float *y_stds,
                                        const float *x_mins, const float *x_maxs) {
  // Layout sanity: caller's flattened weights_ (13312 = 512+3*4096+512) plus biases_ (264 = 4*64...
  if (weights_len + biases_len != static_cast<size_t>(kParamCount)) {
    return nullptr;
  }

  NeuroPressGpuWeights *device_w = nullptr;
  if (cudaMalloc(&device_w, sizeof(NeuroPressGpuWeights)) != cudaSuccess) {
    return nullptr;
  }
  if (cudaMemset(device_w, 0, sizeof(NeuroPressGpuWeights)) != cudaSuccess) {
    cudaFree(device_w);
    return nullptr;
  }

  // Host-side: interleave weights_[]/biases_[] (Clio's own layout, 5 separate weight matrices + ...
  constexpr size_t kOffXMeans = kParamCount;
  constexpr size_t kOffXStds = kOffXMeans + kInputDim;
  constexpr size_t kOffYMeans = kOffXStds + kInputDim;
  constexpr size_t kOffYStds = kOffYMeans + kOutputDim;
  constexpr size_t kOffXLos = kOffYStds + kOutputDim;
  constexpr size_t kOffXHis = kOffXLos + kInputDim;
  constexpr size_t kPrefixFloats = kOffXHis + kInputDim;
  // The single copy is only correct if the device struct really is laid out the way this staging...
  static_assert(offsetof(NeuroPressGpuWeights, x_means) ==
                    sizeof(float) * kOffXMeans,
                "x_means must directly follow params");
  static_assert(offsetof(NeuroPressGpuWeights, x_stds) ==
                    sizeof(float) * kOffXStds,
                "x_stds must directly follow x_means");
  static_assert(offsetof(NeuroPressGpuWeights, y_means) ==
                    sizeof(float) * kOffYMeans,
                "y_means must directly follow x_stds");
  static_assert(offsetof(NeuroPressGpuWeights, y_stds) ==
                    sizeof(float) * kOffYStds,
                "y_stds must directly follow y_means");
  static_assert(offsetof(NeuroPressGpuWeights, x_los) ==
                    sizeof(float) * kOffXLos,
                "x_los must directly follow y_stds");
  static_assert(offsetof(NeuroPressGpuWeights, x_his) ==
                    sizeof(float) * kOffXHis,
                "x_his must directly follow x_los");

  float host_params[kPrefixFloats];
  size_t w_off[5] = {0, kW1, kW1 + kW234, kW1 + 2 * kW234, kW1 + 3 * kW234};
  size_t b_off[5] = {0, kHiddenDim, 2 * kHiddenDim, 3 * kHiddenDim,
                     4 * kHiddenDim};
  int dst_w_off[5] = {kOffW1, kOffW2, kOffW3, kOffW4, kOffW5};
  int dst_b_off[5] = {kOffB1, kOffB2, kOffB3, kOffB4, kOffB5};
  int w_sizes[5] = {kW1, kW234, kW234, kW234, kW5};
  int b_sizes[5] = {kHiddenDim, kHiddenDim, kHiddenDim, kHiddenDim, kOutputDim};
  for (int layer = 0; layer < 5; ++layer) {
    std::memcpy(host_params + dst_w_off[layer], weights + w_off[layer],
                sizeof(float) * static_cast<size_t>(w_sizes[layer]));
    std::memcpy(host_params + dst_b_off[layer], biases + b_off[layer],
                sizeof(float) * static_cast<size_t>(b_sizes[layer]));
  }

  // A reload restarts learning: zero every flow's gradient history, as upstream's resetAllSGDEMA...
  ResetAllEmaBuffers();

  std::memcpy(host_params + kOffXMeans, x_means, sizeof(float) * kInputDim);
  std::memcpy(host_params + kOffXStds, x_stds, sizeof(float) * kInputDim);
  std::memcpy(host_params + kOffYMeans, y_means, sizeof(float) * kOutputDim);
  std::memcpy(host_params + kOffYStds, y_stds, sizeof(float) * kOutputDim);
  // Bounds to sigma units once, here, so the kernels do one compare each.
  static const bool kInputBound = [] {
    const char *v = std::getenv("CLIO_NEUROPRESS_INPUT_BOUND");
    return !(v != nullptr && v[0] == '0');
  }();
  static const float kInputMargin = [] {
    const char *v = std::getenv("CLIO_NEUROPRESS_INPUT_BOUND_MARGIN");
    return (v != nullptr && *v != '\0') ? std::strtof(v, nullptr) : 0.0f;
  }();
  for (int i = 0; i < kInputDim; ++i) {
    float sd = x_stds[i];
    if (sd < 1e-8f) sd = 1e-8f;
    const bool have = kInputBound && x_mins != nullptr && x_maxs != nullptr;
    host_params[kOffXLos + i] =
        have ? (x_mins[i] - x_means[i]) / sd - kInputMargin : -1e30f;
    host_params[kOffXHis + i] =
        have ? (x_maxs[i] - x_means[i]) / sd + kInputMargin : 1e30f;
  }

  bool ok = cudaMemcpy(&device_w->params, host_params, sizeof(host_params),
                       cudaMemcpyHostToDevice) == cudaSuccess;
  if (!ok) {
    cudaFree(device_w);
    return nullptr;
  }
  return device_w;
}

void NeuroPressGpuFree(NeuroPressGpuWeights *w) {
  if (w) cudaFree(w);
}

void NeuroPressGpuDownloadWeights(NeuroPressGpuWeights *w, float *weights_out,
                                  float *biases_out) {
  if (!w) return;
  /* The weights this reads may still be in flight on the SGD stream. */
  SgdHostWaitIfEverFired();
  float host_params[kParamCount];
  cudaMemcpy(host_params, &w->params, sizeof(host_params),
            cudaMemcpyDeviceToHost);
  int src_w_off[5] = {kOffW1, kOffW2, kOffW3, kOffW4, kOffW5};
  int src_b_off[5] = {kOffB1, kOffB2, kOffB3, kOffB4, kOffB5};
  size_t dst_w_off[5] = {0, kW1, kW1 + kW234, kW1 + 2 * kW234,
                         kW1 + 3 * kW234};
  size_t dst_b_off[5] = {0, kHiddenDim, 2 * kHiddenDim, 3 * kHiddenDim,
                         4 * kHiddenDim};
  int w_sizes[5] = {kW1, kW234, kW234, kW234, kW5};
  int b_sizes[5] = {kHiddenDim, kHiddenDim, kHiddenDim, kHiddenDim, kOutputDim};
  for (int layer = 0; layer < 5; ++layer) {
    std::memcpy(weights_out + dst_w_off[layer], host_params + src_w_off[layer],
               sizeof(float) * static_cast<size_t>(w_sizes[layer]));
    std::memcpy(biases_out + dst_b_off[layer], host_params + src_b_off[layer],
               sizeof(float) * static_cast<size_t>(b_sizes[layer]));
  }
}

void NeuroPressGpuUploadWeights(NeuroPressGpuWeights *w, const float *weights,
                                const float *biases) {
  if (!w || !weights || !biases) return;
  // Read-modify-write: params[] also holds nothing else, but the device copy is the live one, so...
  float host_params[kParamCount];
  cudaMemcpy(host_params, &w->params, sizeof(host_params),
            cudaMemcpyDeviceToHost);
  int dst_w_off[5] = {kOffW1, kOffW2, kOffW3, kOffW4, kOffW5};
  int dst_b_off[5] = {kOffB1, kOffB2, kOffB3, kOffB4, kOffB5};
  size_t src_w_off[5] = {0, kW1, kW1 + kW234, kW1 + 2 * kW234,
                         kW1 + 3 * kW234};
  size_t src_b_off[5] = {0, kHiddenDim, 2 * kHiddenDim, 3 * kHiddenDim,
                         4 * kHiddenDim};
  int w_sizes[5] = {kW1, kW234, kW234, kW234, kW5};
  int b_sizes[5] = {kHiddenDim, kHiddenDim, kHiddenDim, kHiddenDim, kOutputDim};
  for (int layer = 0; layer < 5; ++layer) {
    std::memcpy(host_params + dst_w_off[layer], weights + src_w_off[layer],
               sizeof(float) * static_cast<size_t>(w_sizes[layer]));
    std::memcpy(host_params + dst_b_off[layer], biases + src_b_off[layer],
               sizeof(float) * static_cast<size_t>(b_sizes[layer]));
  }
  cudaMemcpy(&w->params, host_params, sizeof(host_params),
            cudaMemcpyHostToDevice);
}

// ============================================================================ Inference: one bl...
constexpr int kMaxCandidates = 32;

/** ct, dt, ratio, psnr -- the four per-candidate outputs, packed in one buffer. */
/** Eight, not four: upstream's device-resident entry point (runNNFusedInferenceCtx) hands back.... */
constexpr int kPredOutputs = 8;

/** decodeAction, in the kernel. */
__device__ __forceinline__ void DecodeAction(int action, int *algo, int *quant,
                                             int *shuffle) {
  *algo = action % 8;
  *quant = (action / 8) % 2;
  *shuffle = (action / 16) % 2;
}

/** Layers, inverse transform and clamps, shared by both inference entry points. */
__device__ __forceinline__ void NeuroPressForwardShared(
    const NeuroPressGpuWeights *__restrict__ w, const float *__restrict__ s_x,
    float *s_h1, float *s_h2, float *s_h3, float *s_h4, float *s_y, int t,
    int cand, float *__restrict__ out_comp_time,
    float *__restrict__ out_decomp_time, float *__restrict__ out_ratio,
    /** Policy ratio ceiling. */
    float ratio_cap,
    float pred_time_floor,
    float *__restrict__ out_psnr,
    /** Outputs 4-7. */
    float *__restrict__ out_rmse = nullptr,
    float *__restrict__ out_max_error = nullptr,
    float *__restrict__ out_mae = nullptr,
    float *__restrict__ out_ssim = nullptr) {
  float sum = w->params[kOffB1 + t];
  for (int i = 0; i < kInputDim; ++i) sum += w->params[kOffW1 + t * kInputDim + i] * s_x[i];
  s_h1[t] = fmaxf(0.0f, sum);
  __syncthreads();

  sum = w->params[kOffB2 + t];
  for (int i = 0; i < kHiddenDim; ++i) sum += w->params[kOffW2 + t * kHiddenDim + i] * s_h1[i];
  s_h2[t] = fmaxf(0.0f, sum);
  __syncthreads();

  sum = w->params[kOffB3 + t];
  for (int i = 0; i < kHiddenDim; ++i) sum += w->params[kOffW3 + t * kHiddenDim + i] * s_h2[i];
  s_h3[t] = fmaxf(0.0f, sum);
  __syncthreads();

  sum = w->params[kOffB4 + t];
  for (int i = 0; i < kHiddenDim; ++i) sum += w->params[kOffW4 + t * kHiddenDim + i] * s_h3[i];
  s_h4[t] = fmaxf(0.0f, sum);
  __syncthreads();

  if (t < kOutputDim) {
    float o = w->params[kOffB5 + t];
    for (int i = 0; i < kHiddenDim; ++i) o += w->params[kOffW5 + t * kHiddenDim + i] * s_h4[i];
    s_y[t] = o;
  }
  __syncthreads();

  if (t == 0) {
    float comp_time = expm1f(s_y[0] * w->y_stds[0] + w->y_means[0]);
    float decomp_time = expm1f(s_y[1] * w->y_stds[1] + w->y_means[1]);
    float ratio = expm1f(s_y[2] * w->y_stds[2] + w->y_means[2]);
    float psnr = s_y[3] * w->y_stds[3] + w->y_means[3];
    // Sanity clamps BEFORE the policy clamps, exactly as nn_gpu.cu orders them.
    comp_time = fmaxf(1e-6f, fminf(comp_time, 1e6f));
    decomp_time = fmaxf(1e-6f, fminf(decomp_time, 1e6f));
    ratio = fmaxf(0.1f, fminf(ratio, 1e5f));
    psnr = fmaxf(0.0f, fminf(psnr, 120.0f));

    out_comp_time[cand] = fmaxf(pred_time_floor, comp_time);
    out_decomp_time[cand] = fmaxf(pred_time_floor, decomp_time);
    out_ratio[cand] = fminf(ratio_cap, ratio);
    out_psnr[cand] = psnr;

    // Outputs 4-7, upstream nn_gpu.cu:211-216 for the transforms and :223-226 for the clamps.
    if (out_rmse != nullptr) {
      const float v = expm1f(s_y[4] * w->y_stds[4] + w->y_means[4]);
      out_rmse[cand] = fmaxf(0.0f, fminf(v, 1e6f));
    }
    if (out_max_error != nullptr) {
      const float v = expm1f(s_y[5] * w->y_stds[5] + w->y_means[5]);
      out_max_error[cand] = fmaxf(0.0f, fminf(v, 1e6f));
    }
    if (out_mae != nullptr) {
      const float v = expm1f(s_y[6] * w->y_stds[6] + w->y_means[6]);
      out_mae[cand] = fmaxf(0.0f, fminf(v, 1e6f));
    }
    if (out_ssim != nullptr) {
      const float ssim_nlog = s_y[7] * w->y_stds[7] + w->y_means[7];
      const float v = 1.0f - expf(-fmaxf(0.0f, ssim_nlog));
      out_ssim[cand] = fmaxf(0.0f, fminf(v, 1.0f));
    }
  }
}

/** Host-matrix entry point: unchanged behaviour, now sharing the forward pass. */
__global__ void InferKernel(const NeuroPressGpuWeights *__restrict__ w,
                            const float *__restrict__ raw_inputs,
                            float *__restrict__ out_comp_time,
                            float *__restrict__ out_decomp_time,
                            float *__restrict__ out_ratio,
                            float *__restrict__ out_psnr,
                            float ratio_cap = 100.0f,
                            float pred_time_floor = 1.0f) {
  int cand = blockIdx.x;
  int t = threadIdx.x;

  __shared__ float s_x[kInputDim];
  __shared__ float s_h1[kHiddenDim], s_h2[kHiddenDim], s_h3[kHiddenDim],
      s_h4[kHiddenDim];
  __shared__ float s_y[kOutputDim];

  if (t < kInputDim) {
    s_x[t] = NeuroPressStandardize(w, t, raw_inputs[cand * kInputDim + t]);
  }
  __syncthreads();

  NeuroPressForwardShared(w, s_x, s_h1, s_h2, s_h3, s_h4, s_y, t, cand,
                          out_comp_time, out_decomp_time, out_ratio, ratio_cap,
                          pred_time_floor,
                          out_psnr);
}

/** All eight outputs, for reporting and for differential testing against upstream. */
__global__ void InferKernelFull(const NeuroPressGpuWeights *__restrict__ w,
                                const float *__restrict__ raw_inputs,
                                float *__restrict__ out_comp_time,
                                float *__restrict__ out_decomp_time,
                                float *__restrict__ out_ratio,
                                float *__restrict__ out_psnr,
                                float *__restrict__ out_rmse,
                                float *__restrict__ out_max_error,
                                float *__restrict__ out_mae,
                                float *__restrict__ out_ssim,
                                float ratio_cap = 100.0f,
                                float pred_time_floor = 1.0f) {
  int cand = blockIdx.x;
  int t = threadIdx.x;

  __shared__ float s_x[kInputDim];
  __shared__ float s_h1[kHiddenDim], s_h2[kHiddenDim], s_h3[kHiddenDim],
      s_h4[kHiddenDim];
  __shared__ float s_y[kOutputDim];

  if (t < kInputDim) {
    s_x[t] = NeuroPressStandardize(w, t, raw_inputs[cand * kInputDim + t]);
  }
  __syncthreads();

  NeuroPressForwardShared(w, s_x, s_h1, s_h2, s_h3, s_h4, s_y, t, cand,
                          out_comp_time, out_decomp_time, out_ratio, ratio_cap,
                          pred_time_floor,
                          out_psnr, out_rmse, out_max_error, out_mae, out_ssim);
}

/** Device-stats entry point: reads the three data features straight out of device memory instea... */
__global__ void InferKernelDeviceStats(
    const NeuroPressGpuWeights *__restrict__ w,
    const int *__restrict__ action_ids,
    const ctp::DeviceFeatureStats *__restrict__ stats, float chunk_size_bytes,
    float error_bound, float *__restrict__ out_comp_time,
    float *__restrict__ out_decomp_time, float *__restrict__ out_ratio,
    float *__restrict__ out_psnr,
    /** Outputs 4-7. */
    float *__restrict__ out_rmse = nullptr,
    float *__restrict__ out_max_error = nullptr,
    float *__restrict__ out_mae = nullptr,
    float *__restrict__ out_ssim = nullptr, float ratio_cap = 100.0f,
    float pred_time_floor = 1.0f,
    /** Prediction reuse. */
    const ctp::compress::preprocess::DevicePredictionReuseState
        *__restrict__ reuse_states = nullptr,
    uint32_t reuse_slot = ctp::compress::preprocess::kNoLineageSlot) {
  int cand = blockIdx.x;
  int t = threadIdx.x;

  // The verdict was written to device memory by ReuseDecisionKernel earlier on this stream.
  if (reuse_states != nullptr &&
      reuse_slot != ctp::compress::preprocess::kNoLineageSlot) {
    const ctp::compress::preprocess::DevicePredictionReuseState &ts =
        reuse_states[reuse_slot];
    // ALL of the candidates or none: a cache that covers fewer than this call asks for would lea...
    if (!ctp::compress::preprocess::MustRunModel(ts.decision_flags) &&
        ts.has_prediction != 0 &&
        ts.cached_count >= static_cast<int>(gridDim.x)) {
      if (t == 0) {
        out_comp_time[cand] = ts.comp_time_ms[cand];
        out_decomp_time[cand] = ts.decomp_time_ms[cand];
        out_ratio[cand] = ts.ratio[cand];
        out_psnr[cand] = ts.psnr_db[cand];
        // Outputs 4-7 are not cached: the ranking path never requests them, so a caller that wan...
        if (out_rmse != nullptr) out_rmse[cand] = 0.0f;
        if (out_max_error != nullptr) out_max_error[cand] = 0.0f;
        if (out_mae != nullptr) out_mae[cand] = 0.0f;
        if (out_ssim != nullptr) out_ssim[cand] = 0.0f;
      }
      return;
    }
  }

  __shared__ float s_x[kInputDim];
  __shared__ float s_h1[kHiddenDim], s_h2[kHiddenDim], s_h3[kHiddenDim],
      s_h4[kHiddenDim];
  __shared__ float s_y[kOutputDim];

  if (t < kInputDim) {
    // The same eight inputs, in the same order, built the same way upstream builds them (nn_gpu....
    int algo, quant, shuffle;
    DecodeAction(action_ids[cand], &algo, &quant, &shuffle);
    float raw;
    if (t == 0) {
      raw = static_cast<float>(algo);
    } else if (t == 1) {
      raw = static_cast<float>(quant);
    } else if (t == 2) {
      raw = static_cast<float>(shuffle);
    } else if (t == 3) {
      raw = error_bound;
      if (quant == 0) {
        // Lossless configs were TRAINED against a 1e-7 sentinel, not a raw 0 (neural_net/core/co...
        raw = 1e-7f;
      }
    } else if (t == 4) {
      raw = chunk_size_bytes;
    } else if (t == 5) {
      raw = static_cast<float>(stats->entropy);
    } else if (t == 6) {
      raw = static_cast<float>(stats->mad);
    } else {
      raw = static_cast<float>(stats->second_derivative);
    }
    s_x[t] = NeuroPressStandardize(w, t, raw);
  }
  __syncthreads();

  NeuroPressForwardShared(w, s_x, s_h1, s_h2, s_h3, s_h4, s_y, t, cand,
                          out_comp_time, out_decomp_time, out_ratio, ratio_cap,
                          pred_time_floor,
                          out_psnr, out_rmse, out_max_error, out_mae, out_ssim);
}

/** Widest candidate set the ranking warp can hold, and the reason it is 32: that is upstream's ... */


/**
 * Analytical PSNR of quantizing a chunk under an absolute bound: the error is
 * uniform on [-eb, eb], so MSE = eb^2 / 3 against the chunk's own value range.
 * A chunk with no range (constant) quantizes exactly and scores the 120 dB
 * cap -- the same cap the network's PSNR output and AnalyticalPsnr use.
 *
 * @param range        the chunk's max - min
 * @param error_bound  the absolute bound
 * @return PSNR in dB, at most 120
 */
__device__ __forceinline__ double AnalyticQuantPsnr(double range,
                                                    double error_bound) {
  if (!(range > 0.0) || !(error_bound > 0.0)) return 120.0;
  const double mse = error_bound * error_bound / 3.0;
  return fmin(120.0, 10.0 * log10(range * range / mse));
}

/** Cost model + ranking, on the GPU. */
__global__ void RankKernel(const float *__restrict__ ct_in,
                           const float *__restrict__ dt_in,
                           const float *__restrict__ ratio_in,
                           const float *__restrict__ psnr_in,
                           const int *__restrict__ action_ids, int n,
                           double data_size_bytes, double w_ct, double w_dt,
                           double w_io, double bw, double error_bound,
                           double min_psnr, double ratio_cap,
                           double min_time_ms,
                           int *__restrict__ out_order,
                           double *__restrict__ out_scores,
                           /** See InferKernelDeviceStats: null means unchanged behaviour. */
                           const ctp::compress::preprocess::
                               DevicePredictionReuseState
                                   *__restrict__ reuse_states = nullptr,
                           uint32_t reuse_slot =
                               ctp::compress::preprocess::kNoLineageSlot,
                           /** The chunk's device stats; its value range
                               drives the quality floor. Null: the floor falls
                               back to the network's PSNR. */
                           const ctp::DeviceFeatureStats *__restrict__ stats =
                               nullptr) {
  const int tid = static_cast<int>(threadIdx.x);

  // Replay the cached ORDER as well as the cached predictions.
  if (reuse_states != nullptr &&
      reuse_slot != ctp::compress::preprocess::kNoLineageSlot) {
    const ctp::compress::preprocess::DevicePredictionReuseState &ts =
        reuse_states[reuse_slot];
    // The replay condition must be UNIFORM across the block, so `tid < n` guards the write and n...
    if (!ctp::compress::preprocess::MustRunModel(ts.decision_flags) &&
        ts.has_prediction != 0 && ts.cached_count >= n) {
      if (tid < n) {
        out_order[tid] = ts.order[tid];
        out_scores[tid] = ts.score[tid];
      }
      return;
    }
  }

  double score = -CUDART_INF;
  int idx = tid;
  // The tie key is the ACTION index, which is what upstream's network orders by -- its lanes ARE...
  int key = (tid < n) ? (action_ids[tid] * kMaxCandidates + tid)
                      : ((kMaxCandidates + tid) * kMaxCandidates + tid);

  if (tid < n) {
    // predictor.h:213-229, clamp for clamp.
    const double ct = fmax(min_time_ms, static_cast<double>(ct_in[tid]));
    const double dt_raw = static_cast<double>(dt_in[tid]);
    const double dt = (dt_raw > 0.0) ? fmax(min_time_ms, dt_raw) : ct;
    const double ratio =
        fmax(0.1, fmin(ratio_cap, static_cast<double>(ratio_in[tid])));
    const double io = (ratio > 0.0) ? (data_size_bytes / (ratio * bw)) : 1e30;
    score = -(w_ct * ct + w_dt * dt + w_io * io);

    // nn_gpu.cu, in that order.
    int algo, quant, shuffle;
    DecodeAction(action_ids[tid], &algo, &quant, &shuffle);
    const bool is_quant = (quant != 0);
    if (is_quant && error_bound <= 0.0) score = -CUDART_INF;
    // The quality floor. Two departures from nn_gpu.cu, which masks ANY action
    // on the network's PSNR: a lossless action is never masked on quality (it
    // has none to lose, and masking it can leave nothing but lossy actions to
    // fall back on), and a quantized action is judged by the analytical PSNR
    // of its chunk's value range. The network's PSNR head cannot see that
    // range -- its features carry no absolute scale -- so under an absolute
    // bound it misses the chunks quantizing erases (measured: VPIC fields
    // spanning 1e-7..1e-6 at eb 1e-4, 0 of 186 flagged; the range catches 186).
    if (min_psnr > 0.0 && is_quant) {
      const double psnr =
          (stats != nullptr)
              ? AnalyticQuantPsnr(stats->value_max - stats->value_min, error_bound)
              : static_cast<double>(psnr_in[tid]);
      if (psnr < min_psnr) score = -CUDART_INF;
    }
  }

  // Total order: score descending, then slot ascending.
  auto better = [](double a_s, int a_key, double b_s, int b_key) {
    return (a_s > b_s) || (a_s == b_s && a_key < b_key);
  };

  // Same bitonic network as nn_gpu.cu.
  for (int k = 2; k <= kMaxCandidates; k <<= 1) {
    for (int j = k >> 1; j >= 1; j >>= 1) {
      const double other_score = __shfl_xor_sync(0xFFFFFFFFu, score, j);
      const int other_idx = __shfl_xor_sync(0xFFFFFFFFu, idx, j);
      const int other_key = __shfl_xor_sync(0xFFFFFFFFu, key, j);
      const bool is_lower = ((tid ^ j) > tid);
      const bool ascending = ((tid & k) == 0);
      const bool mine_better = better(score, key, other_score, other_key);
      bool swap;
      if (is_lower) {
        swap = ascending ? !mine_better : mine_better;
      } else {
        swap = ascending ? mine_better : !mine_better;
      }
      if (swap) {
        score = other_score;
        idx = other_idx;
        key = other_key;
      }
    }
  }

  if (tid < n) {
    out_order[tid] = idx;
    // The sorted score travels with the slot.
    out_scores[tid] = score;
  }
}

/** Deferred, head-only SGD for the DEcOMPRESSION-time output, on the GPU. */
__global__ void DecompHeadSGDKernel(
    NeuroPressGpuWeights *__restrict__ w,
    const NeuroPressGpuDecompSample *__restrict__ samples, int num_samples) {
  const int t = static_cast<int>(threadIdx.x);  // 0..kHiddenDim-1

  __shared__ float s_x[kInputDim];
  __shared__ float s_h1[kHiddenDim], s_h2[kHiddenDim], s_h3[kHiddenDim],
      s_h4[kHiddenDim];
  __shared__ float s_reduce[kHiddenDim];
  __shared__ float s_err;
  __shared__ float s_mean_abs_err;

  float acc_gw = 0.0f;       // this lane's dw5[1][t], summed over samples
  float acc_gb = 0.0f;       // db5[1], lane 0 only
  float acc_abs_err = 0.0f;  // for the trust region, lane 0 only
  int valid = 0;

  for (int si = 0; si < num_samples; ++si) {
    if (t == 0) {
      int algo, quant, shuffle;
      DecodeAction(samples[si].action, &algo, &quant, &shuffle);
      float raw[kInputDim];
      raw[0] = static_cast<float>(algo);
      raw[1] = static_cast<float>(quant);
      raw[2] = static_cast<float>(shuffle);
      raw[3] = samples[si].error_bound_enc;  // RAW, no sentinel
      raw[4] = samples[si].data_size_enc;
      raw[5] = samples[si].entropy;
      raw[6] = samples[si].mad;
      raw[7] = samples[si].second_derivative;
      for (int i = 0; i < kInputDim; ++i)
        s_x[i] = NeuroPressStandardize(w, i, raw[i]);
    }
    __syncthreads();

    float sum = w->params[kOffB1 + t];
    for (int i = 0; i < kInputDim; ++i)
      sum += w->params[kOffW1 + t * kInputDim + i] * s_x[i];
    s_h1[t] = fmaxf(0.0f, sum);
    __syncthreads();

    sum = w->params[kOffB2 + t];
    for (int i = 0; i < kHiddenDim; ++i)
      sum += w->params[kOffW2 + t * kHiddenDim + i] * s_h1[i];
    s_h2[t] = fmaxf(0.0f, sum);
    __syncthreads();

    sum = w->params[kOffB3 + t];
    for (int i = 0; i < kHiddenDim; ++i)
      sum += w->params[kOffW3 + t * kHiddenDim + i] * s_h2[i];
    s_h3[t] = fmaxf(0.0f, sum);
    __syncthreads();

    sum = w->params[kOffB4 + t];
    for (int i = 0; i < kHiddenDim; ++i)
      sum += w->params[kOffW4 + t * kHiddenDim + i] * s_h3[i];
    s_h4[t] = fmaxf(0.0f, sum);
    __syncthreads();

    // Output 1's head only, error taken in log space (:2470-2488).
    if (t == 0) {
      float y_norm = w->params[kOffB5 + 1];
      for (int i = 0; i < kHiddenDim; ++i)
        y_norm += w->params[kOffW5 + 1 * kHiddenDim + i] * s_h4[i];
      float y_std1 = w->y_stds[1];
      if (y_std1 < 1e-8f) y_std1 = 1e-8f;
      const float pred_log = y_norm * y_std1 + w->y_means[1];
      const float clamped =
          fmaxf(0.01f, fminf(samples[si].actual_decomp_ms, 5000.0f));
      float err_log = pred_log - log1pf(clamped);
      err_log = fmaxf(-2.0f, fminf(2.0f, err_log));
      s_err = err_log / y_std1;
    }
    __syncthreads();

    if (fabsf(s_err) < 0.05f) {  // noise gate (:2492)
      __syncthreads();
      continue;
    }
    acc_gw += s_err * s_h4[t];
    if (t == 0) {
      acc_gb += s_err;
      acc_abs_err += fabsf(s_err);
    }
    ++valid;
    __syncthreads();
  }

  if (valid == 0) return;

  const float inv_n = 1.0f / static_cast<float>(valid);
  acc_gw *= inv_n;
  if (t == 0) acc_gb *= inv_n;

  s_reduce[t] = acc_gw * acc_gw;
  __syncthreads();
  for (int s = kHiddenDim / 2; s > 0; s >>= 1) {
    if (t < s) s_reduce[t] += s_reduce[t + s];
    __syncthreads();
  }

  // PER-THREAD g_norm, and deliberately so: upstream writes sqrtf(s_reduce[0] + (t == 0 ?
  const float g_norm =
      sqrtf(s_reduce[0] + ((t == 0) ? acc_gb * acc_gb : 0.0f)) + 1e-8f;

  if (t == 0) s_mean_abs_err = acc_abs_err * inv_n;
  __syncthreads();

  // Trust region, deliberately not Train()'s (:2538-2543).
  constexpr float kTrustK = 0.15f;
  constexpr float kMaxStep = 0.05f;
  constexpr float kMinStep = 1e-4f;
  constexpr float kWClamp = 5.0f;
  const float step =
      fmaxf(kMinStep, fminf(kMaxStep, kTrustK * s_mean_abs_err));

  const float new_w =
      w->params[kOffW5 + 1 * kHiddenDim + t] - step * (acc_gw / g_norm);
  w->params[kOffW5 + 1 * kHiddenDim + t] =
      fmaxf(-kWClamp, fminf(kWClamp, new_w));
  if (t == 0) {
    const float new_b = w->params[kOffB5 + 1] - step * (acc_gb / g_norm);
    w->params[kOffB5 + 1] = fmaxf(-kWClamp, fminf(kWClamp, new_b));
  }
}

namespace {

/** Per-thread inference scratch, allocated once. */
struct InferScratch {
  // Sized for the ACTION SPACE.
  int *d_actions = nullptr;  // [kMaxCandidates] upstream action indices

  // The candidate action list is uploaded only when it CHANGES, not per chunk.
  bool actions_installed = false;
  std::vector<int> installed_actions;

  // Sized for the BATCH, grown on demand.
  float *d_raw = nullptr;  // [cap][8]

  // EVERY value this path reads back lives in ONE allocation, so the whole ranking result return...
  void *d_out = nullptr;
  double *d_scores = nullptr;  // [cap] ranked scores, best first
  int *d_order = nullptr;      // [cap] ranked slots, same order
  float *d_pred = nullptr;     // [8][cap]
  float *d_ct = nullptr;       // = d_pred + 0*cap
  float *d_dt = nullptr;       // = d_pred + 1*cap
  float *d_r = nullptr;        // = d_pred + 2*cap
  float *d_p = nullptr;        // = d_pred + 3*cap
  float *d_rmse = nullptr;     // = d_pred + 4*cap
  float *d_maxe = nullptr;     // = d_pred + 5*cap
  float *d_mae = nullptr;      // = d_pred + 6*cap
  float *d_ssim = nullptr;     // = d_pred + 7*cap

  // Host landing buffer for that single copy, scattered to the caller's separate arrays after th...
  std::vector<unsigned char> host_out;

  int cap = 0;
  bool ok = false;
};

/** Byte size of the packed readback block for n candidates. */
constexpr size_t PackedOutBytes(size_t n) {
  return n * (sizeof(double) + sizeof(int) + sizeof(float) * kPredOutputs);
}

/** Grow the per-batch buffers when a call needs more than the last one did. */
bool EnsureInferCapacity(InferScratch &s, int n) {
  if (n <= s.cap) return true;
  cudaFree(s.d_raw);
  cudaFree(s.d_out);
  s.d_raw = nullptr;
  s.d_out = nullptr;
  s.d_scores = nullptr;
  s.d_order = nullptr;
  s.d_pred = s.d_ct = s.d_dt = s.d_r = s.d_p = nullptr;
  s.d_rmse = s.d_maxe = s.d_mae = s.d_ssim = nullptr;
  s.cap = 0;
  const size_t nn = static_cast<size_t>(n);
  if (cudaMalloc(&s.d_raw, sizeof(float) * nn * kInputDim) != cudaSuccess ||
      cudaMalloc(&s.d_out, PackedOutBytes(nn)) != cudaSuccess) {
    return false;
  }
  // Views into d_out, in the layout documented on InferScratch.
  auto *base = static_cast<unsigned char *>(s.d_out);
  s.d_scores = reinterpret_cast<double *>(base);
  s.d_order = reinterpret_cast<int *>(base + sizeof(double) * nn);
  s.d_pred = reinterpret_cast<float *>(
      base + (sizeof(double) + sizeof(int)) * nn);
  s.d_ct = s.d_pred;
  s.d_dt = s.d_pred + n;
  s.d_r = s.d_pred + 2 * n;
  s.d_p = s.d_pred + 3 * n;
  s.d_rmse = s.d_pred + 4 * n;
  s.d_maxe = s.d_pred + 5 * n;
  s.d_mae = s.d_pred + 6 * n;
  s.d_ssim = s.d_pred + 7 * n;
  // Zero ONCE, here, not per call.
  if (cudaMemset(s.d_out, 0, PackedOutBytes(nn)) != cudaSuccess) return false;
  s.host_out.resize(PackedOutBytes(nn));
  s.cap = n;
  return true;
}

/** Pull the four prediction arrays back in a single copy, wait once, then scatter to the caller... */
bool FetchPredictionsSync(InferScratch &s, int n, cudaStream_t st,
                          float *out_ct, float *out_dt, float *out_r,
                          float *out_p, int *out_order = nullptr,
                          double *out_scores = nullptr,
                          float *out_rmse = nullptr,
                          float *out_maxe = nullptr, float *out_mae = nullptr,
                          float *out_ssim = nullptr) {
  // STRIDES ARE THE ALLOCATION'S, NOT THE CALL'S.
  const size_t cap = static_cast<size_t>(s.cap);
  const size_t nn = static_cast<size_t>(n);
  const size_t stride = sizeof(float) * cap;   // between prediction arrays
  const size_t want = sizeof(float) * nn;      // what the caller asked for
  const size_t pred_off = (sizeof(double) + sizeof(int)) * cap;

  // Whether the ranking came back decides only WHERE the single copy starts.
  const bool want_rank = (out_order != nullptr);
  // Outputs 4-7 are the tail of the predictions region, so a caller that does not want them simp...
  const bool want_quality = (out_rmse != nullptr || out_maxe != nullptr ||
                             out_mae != nullptr || out_ssim != nullptr);
  const size_t off = want_rank ? 0 : pred_off;
  const size_t end =
      want_quality ? PackedOutBytes(cap)
                   : pred_off + sizeof(float) * cap * 4;
  const size_t len = end - off;
  auto *dst = s.host_out.data() + off;
  auto *src = static_cast<const unsigned char *>(s.d_out) + off;

  if (cudaMemcpyAsync(dst, src, len, cudaMemcpyDeviceToHost, st) !=
      cudaSuccess) {
    return false;
  }
  if (cudaStreamSynchronize(st) != cudaSuccess) return false;

  const unsigned char *base = s.host_out.data();
  if (want_rank) {
    std::memcpy(out_order, base + sizeof(double) * cap, sizeof(int) * nn);
    if (out_scores != nullptr) {
      std::memcpy(out_scores, base, sizeof(double) * nn);
    }
  }
  // The first four are null-checked like the last four: the all-outputs entry point documents th...
  const unsigned char *p = base + pred_off;
  if (out_ct != nullptr) std::memcpy(out_ct, p, want);
  if (out_dt != nullptr) std::memcpy(out_dt, p + stride, want);
  if (out_r != nullptr) std::memcpy(out_r, p + 2 * stride, want);
  if (out_p != nullptr) std::memcpy(out_p, p + 3 * stride, want);
  if (out_rmse != nullptr) std::memcpy(out_rmse, p + 4 * stride, want);
  if (out_maxe != nullptr) std::memcpy(out_maxe, p + 5 * stride, want);
  if (out_mae != nullptr) std::memcpy(out_mae, p + 6 * stride, want);
  if (out_ssim != nullptr) std::memcpy(out_ssim, p + 7 * stride, want);
  return true;
}

InferScratch &Infer() {
  static thread_local InferScratch *s = [] {
    auto *p = new InferScratch();
    // d_order and d_scores are no longer allocated here: they are views into the one packed d_ou...
    p->ok = cudaMalloc(&p->d_actions, sizeof(int) * kMaxCandidates) ==
                cudaSuccess &&
            EnsureInferCapacity(*p, kMaxCandidates);
    return p;
  }();
  return *s;
}

}  // namespace

bool NeuroPressGpuTrainDecompHead(NeuroPressGpuWeights *w,
                                  const NeuroPressGpuDecompSample *samples,
                                  int num_samples) {
  if (!w || !samples || num_samples <= 0) return false;

  /** Upstream's runBatchedDecompSGD, step for step. */
  SgdSync &g = Sgd();
  if (!g.ok) return false;

  const size_t bytes =
      sizeof(NeuroPressGpuDecompSample) * static_cast<size_t>(num_samples);
  SgdScratch &sc = SgdSamples();
  if (!EnsureSgdSamples(sc, bytes)) return false;

  bool ok = cudaMemcpyAsync(sc.d_samples, samples, bytes,
                            cudaMemcpyHostToDevice, g.stream) == cudaSuccess;
  if (ok) {
    DecompHeadSGDKernel<<<1, kHiddenDim, 0, g.stream>>>(
        w, static_cast<const NeuroPressGpuDecompSample *>(sc.d_samples),
        num_samples);
    ok = cudaGetLastError() == cudaSuccess;
    if (ok && cudaEventRecord(g.done, g.stream) == cudaSuccess) {
      g.ever_fired.store(true, std::memory_order_release);
    }
    /* The wait upstream keeps. */
    if (ok) ok = cudaStreamSynchronize(g.stream) == cudaSuccess;
  }
  return ok;
}

namespace {
// This thread's phase events, created once and reused.
struct PhaseEvents {
  cudaEvent_t stats_start = nullptr, stats_stop = nullptr;
  cudaEvent_t nn_start = nullptr, nn_stop = nullptr, rank_stop = nullptr;
  bool ok = false;
  bool stats = false;
  bool nn = false;
};

PhaseEvents &Phase() {
  static thread_local PhaseEvents e = [] {
    PhaseEvents p;
    p.ok = cudaEventCreate(&p.stats_start) == cudaSuccess &&
           cudaEventCreate(&p.stats_stop) == cudaSuccess &&
           cudaEventCreate(&p.nn_start) == cudaSuccess &&
           cudaEventCreate(&p.nn_stop) == cudaSuccess &&
           cudaEventCreate(&p.rank_stop) == cudaSuccess;
    return p;
  }();
  return e;
}

bool Elapsed(cudaEvent_t a, cudaEvent_t b, double *ms) {
  float f = 0.0f;
  if (cudaEventSynchronize(b) != cudaSuccess) return false;
  if (cudaEventElapsedTime(&f, a, b) != cudaSuccess) return false;
  *ms = static_cast<double>(f);
  return true;
}
}  // namespace

bool NeuroPressPhaseTimingEnabled() {
  static const bool on = [] {
    const char *p = std::getenv("CLIO_NEUROPRESS_PHASE_LOG");
    return p && *p;
  }();
  return on;
}

void MarkNeuroPressStatsPhase(void *stream, bool start) {
  if (!NeuroPressPhaseTimingEnabled()) return;
  PhaseEvents &e = Phase();
  if (!e.ok) return;
  const cudaStream_t st = static_cast<cudaStream_t>(stream);
  if (start) {
    e.stats = cudaEventRecord(e.stats_start, st) == cudaSuccess;
    e.nn = false;
  } else if (e.stats) {
    e.stats = cudaEventRecord(e.stats_stop, st) == cudaSuccess;
  }
}

bool TakeNeuroPressPhaseTimes(double *stats_ms, double *infer_ms,
                              double *rank_ms) {
  PhaseEvents &e = Phase();
  const bool stats = e.stats, nn = e.nn;
  e.stats = e.nn = false;
  if (!e.ok || !stats) return false;
  if (!Elapsed(e.stats_start, e.stats_stop, stats_ms)) return false;
  // A cached ranking launches neither the network nor the ranking.
  *infer_ms = 0.0;
  *rank_ms = 0.0;
  if (nn && !(Elapsed(e.nn_start, e.nn_stop, infer_ms) &&
              Elapsed(e.nn_stop, e.rank_stop, rank_ms))) {
    return false;
  }
  return true;
}

bool NeuroPressGpuInferBatchDeviceStats(
    NeuroPressGpuWeights *w, const void *device_stats,
    const int *action_ids, int num_candidates, float chunk_size_bytes,
    float error_bound, void *stream, float *out_comp_time_ms,
    float *out_decomp_time_ms, float *out_ratio, float *out_psnr_db,
    const GpuRankParams *rank, int *out_order, double *out_scores,
    float *out_rmse, float *out_max_error, float *out_mae, float *out_ssim,
    const ctp::compress::preprocess::PredictionReuseContext *reuse,
    ctp::compress::preprocess::PredictionReuseOutcome *out_outcome) {
  if (!w || !device_stats || !action_ids || num_candidates <= 0 ||
      num_candidates > kMaxCandidates) {
    return false;
  }
  InferScratch &s = Infer();
  if (!s.ok) return false;

  cudaStream_t st = static_cast<cudaStream_t>(stream);
  // Phase log events; read after the fetch's synchronize, so no added wait.
  PhaseEvents *phase = nullptr;
  if (NeuroPressPhaseTimingEnabled() && Phase().ok && Phase().stats) {
    phase = &Phase();
    if (cudaEventRecord(phase->nn_start, st) != cudaSuccess) phase = nullptr;
  }
  const size_t act_bytes = sizeof(int) * static_cast<size_t>(num_candidates);

  // Everything below is enqueued on the SAME stream the statistics were computed on, so the kern...
  bool ok = true;
  const bool actions_match =
      s.actions_installed &&
      s.installed_actions.size() == static_cast<size_t>(num_candidates) &&
      std::equal(s.installed_actions.begin(), s.installed_actions.end(),
                 action_ids);
  if (!actions_match) {
    ok = cudaMemcpyAsync(s.d_actions, action_ids, act_bytes,
                         cudaMemcpyHostToDevice, st) == cudaSuccess;
    if (ok) {
      s.actions_installed = true;
      s.installed_actions.assign(action_ids, action_ids + num_candidates);
    } else {
      // Leave no half-installed state: the device buffer is now indeterminate.
      s.actions_installed = false;
      s.installed_actions.clear();
    }
  }
  bool ranked = false;
  if (ok) {
      /** GPU-level barrier before reading the weights: wait for the last SGD on its stream. */
    SgdWaitIfEverFired(st);
    /** The decision goes here, AFTER the SGD barrier and BEFORE the forward pass, so its verdic... */
    if (reuse != nullptr) {
      ctp::compress::preprocess::LaunchReuseDecision(*reuse, device_stats,
                                                       st, &w->sgd_call_count);
    }
  InferKernelDeviceStats<<<num_candidates, kHiddenDim, 0, st>>>(
        w, s.d_actions,
        static_cast<const ctp::DeviceFeatureStats *>(device_stats),
        chunk_size_bytes, error_bound, s.d_ct, s.d_dt, s.d_r, s.d_p,
        /** Only ask the kernel for outputs 4-7 when the caller wants them; the transforms are s... */
        out_rmse ? s.d_rmse : nullptr, out_max_error ? s.d_maxe : nullptr,
        out_mae ? s.d_mae : nullptr, out_ssim ? s.d_ssim : nullptr,
        /** One cap for BOTH halves. */
        rank != nullptr ? static_cast<float>(rank->ratio_cap) : 100.0f,
        NeuroPressPredTimeFloorMs(),
        reuse != nullptr
            ? static_cast<const ctp::compress::preprocess::
                              DevicePredictionReuseState *>(reuse->states)
            : nullptr,
        reuse != nullptr ? reuse->slot
                         : ctp::compress::preprocess::kNoLineageSlot);
    ok = cudaGetLastError() == cudaSuccess;
  }
  if (ok && phase != nullptr &&
      cudaEventRecord(phase->nn_stop, st) != cudaSuccess) {
    phase = nullptr;
  }
  // Cost model and ordering, still on the device and still on this stream -- upstream does both ...
  if (ok && rank != nullptr && out_order != nullptr) {
    RankKernel<<<1, kMaxCandidates, 0, st>>>(
        s.d_ct, s.d_dt, s.d_r, s.d_p, s.d_actions, num_candidates,
        rank->data_size_bytes, rank->w_compress_time, rank->w_decompress_time,
        rank->w_io, rank->bandwidth_bytes_per_ms, rank->error_bound,
        rank->min_psnr, rank->ratio_cap, rank->min_time_ms, s.d_order, s.d_scores,
        reuse != nullptr
            ? static_cast<const ctp::compress::preprocess::
                              DevicePredictionReuseState *>(reuse->states)
            : nullptr,
        reuse != nullptr ? reuse->slot
                         : ctp::compress::preprocess::kNoLineageSlot,
        static_cast<const ctp::DeviceFeatureStats *>(device_stats));
    ok = cudaGetLastError() == cudaSuccess;
    // The ranking is NOT fetched here: it shares one allocation with the predictions, so FetchPr...
    ranked = ok;
  }

  /** Cache the result and advance the state -- AFTER the ranking has produced it and after both... */
  if (ok && reuse != nullptr && ranked) {
    ok = ctp::compress::preprocess::LaunchReuseCommit(
        *reuse, device_stats, s.d_scores, s.d_order, s.d_ct, s.d_dt, s.d_r,
        s.d_p, num_candidates, st, &w->sgd_call_count);
    /** The outcome rides the transfer the fetch below already performs: one extra async copy on... */
    if (ok && out_outcome != nullptr) {
      ctp::compress::preprocess::EnqueueReuseOutcome(*reuse, out_outcome,
                                                        st);
    }
  }

  if (ok) {
    ok = FetchPredictionsSync(s, num_candidates, st, out_comp_time_ms,
                              out_decomp_time_ms, out_ratio, out_psnr_db,
                              ranked ? out_order : nullptr, out_scores,
                              out_rmse, out_max_error, out_mae, out_ssim);
  }
  if (ok && phase != nullptr) {
    phase->nn = cudaEventRecord(phase->rank_stop, st) == cudaSuccess;
  }
  return ok;
}

bool NeuroPressGpuInferBatchFull(NeuroPressGpuWeights *w,
                                 const float *raw_inputs, int num_candidates,
                                 float *out_comp_time_ms,
                                 float *out_decomp_time_ms, float *out_ratio,
                                 float *out_psnr_db, float *out_rmse,
                                 float *out_max_error, float *out_mae,
                                 float *out_ssim) {
  if (!w || !raw_inputs || num_candidates <= 0) return false;

  // The SAME per-thread scratch the other two entry points use.
  const size_t in_bytes = sizeof(float) * static_cast<size_t>(num_candidates) *
                          kInputDim;

  InferScratch &s = Infer();
  if (!s.ok || !EnsureInferCapacity(s, num_candidates)) return false;
  cudaStream_t st = static_cast<cudaStream_t>(ctp::DeviceStatsStream());

  bool ok = cudaMemcpyAsync(s.d_raw, raw_inputs, in_bytes,
                            cudaMemcpyHostToDevice, st) == cudaSuccess;
  if (ok) {
    /** GPU-level barrier before reading the weights: wait for the last SGD on its stream. */
    SgdWaitIfEverFired(st);
    InferKernelFull<<<num_candidates, kHiddenDim, 0, st>>>(
        w, s.d_raw, s.d_ct, s.d_dt, s.d_r, s.d_p, s.d_rmse, s.d_maxe,
        s.d_mae, s.d_ssim, 100.0f, NeuroPressPredTimeFloorMs());
    ok = cudaGetLastError() == cudaSuccess;
  }
  if (ok) {
    // out_order null keeps this a predictions-only fetch; the quality pointers being non-null ex...
    ok = FetchPredictionsSync(s, num_candidates, st, out_comp_time_ms,
                              out_decomp_time_ms, out_ratio, out_psnr_db,
                              /*out_order=*/nullptr, /*out_scores=*/nullptr,
                              out_rmse, out_max_error, out_mae, out_ssim);
  }
  return ok;
}

bool NeuroPressGpuInferBatch(NeuroPressGpuWeights *w, const float *raw_inputs,
                             int num_candidates, float *out_comp_time_ms,
                             float *out_decomp_time_ms, float *out_ratio,
                             float *out_psnr_db) {
  if (!w || num_candidates <= 0) return false;
  InferScratch &s = Infer();
  if (!s.ok || !EnsureInferCapacity(s, num_candidates)) return false;

  // Same per-thread stream and same persistent scratch as the device-stats entry point.
  cudaStream_t st = static_cast<cudaStream_t>(ctp::DeviceStatsStream());
  const size_t out_bytes = sizeof(float) * static_cast<size_t>(num_candidates);
  const size_t in_bytes = out_bytes * kInputDim;

  // Every step is checked: a silent failure here leaves the caller's output vectors zero-filled,...
  bool ok = cudaMemcpyAsync(s.d_raw, raw_inputs, in_bytes,
                            cudaMemcpyHostToDevice, st) == cudaSuccess;
  if (ok) {
    /** GPU-level barrier before reading the weights: wait for the last SGD on its stream. */
    SgdWaitIfEverFired(st);
    /** This entry point takes no ranking parameters, so the cap stays at upstream's literal 100. */
    InferKernel<<<num_candidates, kHiddenDim, 0, st>>>(
        w, s.d_raw, s.d_ct, s.d_dt, s.d_r, s.d_p, 100.0f,
        NeuroPressPredTimeFloorMs());
    ok = cudaGetLastError() == cudaSuccess;
  }
  if (ok) {
    ok = FetchPredictionsSync(s, num_candidates, st, out_comp_time_ms,
                              out_decomp_time_ms, out_ratio, out_psnr_db);
  }
  return ok;
}

// ============================================================================ SGD: single block...
__device__ __forceinline__ void ForwardOneLayer(
    const float *__restrict__ w_layer, const float *__restrict__ b_layer,
    const float *in, int fan_in, int t, float &h_out) {
  float sum = b_layer[t];
  for (int i = 0; i < fan_in; ++i) sum += w_layer[t * fan_in + i] * in[i];
  h_out = fmaxf(0.0f, sum);
}

/** Phases 1 and 1.5 of the ported rule: forward pass, per-head error in standardized log1p spac... */
__device__ __forceinline__ void SgdForwardAndErrors(
    NeuroPressGpuWeights *w,
    const NeuroPressGpuSGDSample *__restrict__ samples, int num_samples, int t,
    const ctp::DeviceFeatureStats *__restrict__ device_stats,
    float ratio_target_cap) {
  // ---- Phase 1: per-sample forward pass + target/error computation ----
  for (int si = 0; si < num_samples; ++si) {
    if (t < kInputDim) {
      // Inputs 5-7 from DEVICE memory, as nnSGDKernel reads d_stats in-kernel (nn_gpu.cu).
      float raw = samples[si].raw_input[t];
      if (device_stats != nullptr) {
        if (t == 5) {
          raw = static_cast<float>(device_stats->entropy);
        } else if (t == 6) {
          raw = static_cast<float>(device_stats->mad);
        } else if (t == 7) {
          raw = static_cast<float>(device_stats->second_derivative);
        }
      }
      w->act_x[si][t] = NeuroPressStandardize(w, t, raw);
    }
    __syncthreads();

    float h;
    ForwardOneLayer(&w->params[kOffW1], &w->params[kOffB1], w->act_x[si],
                    kInputDim, t, h);
    w->act_h1[si][t] = h;
    __syncthreads();

    ForwardOneLayer(&w->params[kOffW2], &w->params[kOffB2], w->act_h1[si],
                    kHiddenDim, t, h);
    w->act_h2[si][t] = h;
    __syncthreads();

    ForwardOneLayer(&w->params[kOffW3], &w->params[kOffB3], w->act_h2[si],
                    kHiddenDim, t, h);
    w->act_h3[si][t] = h;
    __syncthreads();

    ForwardOneLayer(&w->params[kOffW4], &w->params[kOffB4], w->act_h3[si],
                    kHiddenDim, t, h);
    w->act_h4[si][t] = h;
    __syncthreads();

    if (t < kOutputDim) {
      float o = w->params[kOffB5 + t];
      for (int i = 0; i < kHiddenDim; ++i)
        o += w->params[kOffW5 + t * kHiddenDim + i] * w->act_h4[si][i];
      w->act_y[si][t] = o;
    }
    __syncthreads();

    if (t == 0) {
      const NeuroPressGpuSGDSample &s = samples[si];
      float d5[kOutputDim];

      float clamped_ratio = fmaxf(0.5f, fminf(s.actual_ratio, ratio_target_cap));
      float y_std2 = fmaxf(w->y_stds[2], 1e-8f);
      d5[2] = w->act_y[si][2] - (log1pf(clamped_ratio) - w->y_means[2]) / y_std2;

      if (s.actual_comp_time_ms > 0.0f) {
        float clamped = fmaxf(0.01f, fminf(s.actual_comp_time_ms, 5000.0f));
        float y_std0 = fmaxf(w->y_stds[0], 1e-8f);
        d5[0] = w->act_y[si][0] - (log1pf(clamped) - w->y_means[0]) / y_std0;
      } else {
        d5[0] = 0.0f;
      }
      if (s.actual_decomp_time_ms > 0.0f) {
        float clamped = fmaxf(0.01f, fminf(s.actual_decomp_time_ms, 5000.0f));
        float y_std1 = fmaxf(w->y_stds[1], 1e-8f);
        d5[1] = w->act_y[si][1] - (log1pf(clamped) - w->y_means[1]) / y_std1;
      } else {
        d5[1] = 0.0f;
      }
      if (s.actual_psnr_db >= 0.0f) {
        float psnr_val = (s.actual_psnr_db == 0.0f) ? 120.0f : s.actual_psnr_db;
        float clamped_psnr = fminf(psnr_val, 120.0f);
        float y_std3 = fmaxf(w->y_stds[3], 1e-8f);
        d5[3] = w->act_y[si][3] - (clamped_psnr - w->y_means[3]) / y_std3;
      } else {
        d5[3] = 0.0f;
      }
      for (int o = 4; o < kOutputDim; ++o) d5[o] = 0.0f;

      constexpr float kNoiseGateThresh = 0.10f;
      if (fabsf(d5[0]) < kNoiseGateThresh) d5[0] = 0.0f;

      for (int o = 0; o < kOutputDim; ++o) w->d5_raw[si][o] = d5[o];

      constexpr float kSgdErrorDelta = 0.5f;
      for (int o = 0; o < kOutputDim; ++o)
        d5[o] = fmaxf(-kSgdErrorDelta, fminf(d5[o], kSgdErrorDelta));
      for (int o = 0; o < kOutputDim; ++o) w->d5_clamped[si][o] = d5[o];
    }
    __syncthreads();
  }

  // ---- Phase 1.5: uncertainty weighting (Kendall et al., 2018) ----
  constexpr float kUwLr = 0.01f;
  constexpr float kUwLogVarMin = -2.0f;
  constexpr float kUwLogVarMax = 4.0f;
  if (t < kOutputDim) {
    float lv = w->log_var[t];
    float precision = expf(fmaxf(-20.0f, fminf(20.0f, -lv)));
    float raw_mse = 0.0f;
    for (int si = 0; si < num_samples; ++si) {
      float e = w->d5_raw[si][t];
      raw_mse += e * e;
    }
    raw_mse /= static_cast<float>(num_samples);
    float grad_lv = 0.5f * (1.0f - precision * raw_mse);
    lv -= kUwLr * grad_lv;
    lv = fmaxf(kUwLogVarMin, fminf(lv, kUwLogVarMax));
    w->log_var[t] = lv;
    float uw = expf(fmaxf(-20.0f, fminf(20.0f, -0.5f * lv)));
    for (int si = 0; si < num_samples; ++si) w->d5_clamped[si][t] *= uw;
  }
  __syncthreads();

}

/** SGD phase A: forward pass, per-head errors, uncertainty weighting, and the normalized L4 del... */
__global__ void SgdPrepareKernel(
    NeuroPressGpuWeights *w,
    const NeuroPressGpuSGDSample *__restrict__ samples, int num_samples,
    const ctp::DeviceFeatureStats *__restrict__ device_stats,
    float ratio_target_cap) {
  int t = threadIdx.x;  // 0..63
  __shared__ float s_reduce[kHiddenDim];

  SgdForwardAndErrors(w, samples, num_samples, t, device_stats, ratio_target_cap);

  // ---- Step 1, once per sample: L4 backward delta for ALL outputs, normalized to unit vectors.
  for (int si = 0; si < num_samples; ++si) {
    for (int o = 0; o < kOutputDim; ++o) {
      float es = w->d5_clamped[si][o];
      float dh4_t = w->params[kOffW5 + o * kHiddenDim + t] * es;
      float v = (w->act_h4[si][t] > 0.0f) ? dh4_t : 0.0f;
      w->dz4_all[si][o][t] = v;
    }
    __syncthreads();
    for (int o = 0; o < kOutputDim; ++o) {
      float local = w->dz4_all[si][o][t] * w->dz4_all[si][o][t];
      s_reduce[t] = local;
      __syncthreads();
      for (int s = kHiddenDim / 2; s > 0; s >>= 1) {
        if (t < s) s_reduce[t] += s_reduce[t + s];
        __syncthreads();
      }
      float norm = sqrtf(s_reduce[0]) + 1e-6f;
      w->dz4_all[si][o][t] /= norm;
      __syncthreads();
    }
  }

}

/** SGD phase B: ONE BLOCK PER OUTPUT HEAD. */
__global__ void SgdPerOutputKernel(NeuroPressGpuWeights *w, int num_samples,
                                   float learning_rate) {
  const int target_out = static_cast<int>(blockIdx.x);
  const int t = static_cast<int>(threadIdx.x);
  __shared__ float s_reduce[kHiddenDim];
  float *__restrict__ og = w->out_grad[target_out];

  constexpr float kGradClipThreshold = 0.1f;
  constexpr float kPcgradCosThresh = -0.1f;

  for (int i = t; i < kParamCount; i += kHiddenDim) og[i] = 0.0f;
  __syncthreads();

  for (int si = 0; si < num_samples; ++si) {
    // Step 2: PCGrad projection for target_out against every other output.
    float my_dz4 = w->dz4_all[si][target_out][t];
    for (int j = 0; j < kOutputDim; ++j) {
      if (j == target_out) continue;
      float local_dot = my_dz4 * w->dz4_all[si][j][t];
      s_reduce[t] = local_dot;
      __syncthreads();
      for (int s = kHiddenDim / 2; s > 0; s >>= 1) {
        if (t < s) s_reduce[t] += s_reduce[t + s];
        __syncthreads();
      }
      float cos_ij = s_reduce[0];
      if (cos_ij < kPcgradCosThresh) my_dz4 -= cos_ij * w->dz4_all[si][j][t];
      __syncthreads();
    }
    float err_mag = fabsf(w->d5_clamped[si][target_out]);
    float dz4 = my_dz4 * err_mag;

    // Step 3: W5/b5 gradient uses the ORIGINAL (unprojected) error.
    float error_signal = w->d5_clamped[si][target_out];
    og[kOffW5 + target_out * kHiddenDim + t] +=
        error_signal * w->act_h4[si][t];
    if (t == 0) og[kOffB5 + target_out] += error_signal;

    // Step 4/4b: L4 gradient, backward L4->L3 (using PROJECTED dz4).
    for (int i = 0; i < kHiddenDim; ++i)
      og[kOffW4 + t * kHiddenDim + i] += dz4 * w->act_h3[si][i];
    og[kOffB4 + t] += dz4;

    // Broadcast dz4[0..63] through shared memory so every thread can sum weights_[*, t] * dz4[*]...
    __shared__ float s_dz4[kHiddenDim];
    s_dz4[t] = dz4;
    __syncthreads();
    float dh3_t = 0.0f;
    for (int j = 0; j < kHiddenDim; ++j)
      dh3_t += w->params[kOffW4 + j * kHiddenDim + t] * s_dz4[j];
    float dz3 = (w->act_h3[si][t] > 0.0f) ? dh3_t : 0.0f;

    // Step 5/5b: L3 gradient, backward L3->L2.
    for (int i = 0; i < kHiddenDim; ++i)
      og[kOffW3 + t * kHiddenDim + i] += dz3 * w->act_h2[si][i];
    og[kOffB3 + t] += dz3;
    __shared__ float s_dz3[kHiddenDim];
    s_dz3[t] = dz3;
    __syncthreads();
    float dh2_t = 0.0f;
    for (int j = 0; j < kHiddenDim; ++j)
      dh2_t += w->params[kOffW3 + j * kHiddenDim + t] * s_dz3[j];
    float dz2 = (w->act_h2[si][t] > 0.0f) ? dh2_t : 0.0f;

    // Step 6/6b: L2 gradient, backward L2->L1, then L1 gradient.
    for (int i = 0; i < kHiddenDim; ++i)
      og[kOffW2 + t * kHiddenDim + i] += dz2 * w->act_h1[si][i];
    og[kOffB2 + t] += dz2;
    __shared__ float s_dz2[kHiddenDim];
    s_dz2[t] = dz2;
    __syncthreads();
    float dh1_t = 0.0f;
    for (int j = 0; j < kHiddenDim; ++j)
      dh1_t += w->params[kOffW2 + j * kHiddenDim + t] * s_dz2[j];
    float dz1 = (w->act_h1[si][t] > 0.0f) ? dh1_t : 0.0f;

    for (int i = 0; i < kInputDim; ++i)
      og[kOffW1 + t * kInputDim + i] += dz1 * w->act_x[si][i];
    og[kOffB1 + t] += dz1;
    __syncthreads();
  }  // per-sample

  // Average over samples, compute this output's gradient norm (only the params it actually touch...
  float inv_n = 1.0f / static_cast<float>(num_samples);
  float local_norm_sq = 0.0f;
  for (int i = 0; i < kInputDim; ++i) {
    int idx = kOffW1 + t * kInputDim + i;
    og[idx] *= inv_n;
    local_norm_sq += og[idx] * og[idx];
  }
  { int idx = kOffB1 + t; og[idx] *= inv_n; local_norm_sq += og[idx] * og[idx]; }
  for (int i = 0; i < kHiddenDim; ++i) {
    int idx = kOffW2 + t * kHiddenDim + i;
    og[idx] *= inv_n;
    local_norm_sq += og[idx] * og[idx];
  }
  { int idx = kOffB2 + t; og[idx] *= inv_n; local_norm_sq += og[idx] * og[idx]; }
  for (int i = 0; i < kHiddenDim; ++i) {
    int idx = kOffW3 + t * kHiddenDim + i;
    og[idx] *= inv_n;
    local_norm_sq += og[idx] * og[idx];
  }
  { int idx = kOffB3 + t; og[idx] *= inv_n; local_norm_sq += og[idx] * og[idx]; }
  for (int i = 0; i < kHiddenDim; ++i) {
    int idx = kOffW4 + t * kHiddenDim + i;
    og[idx] *= inv_n;
    local_norm_sq += og[idx] * og[idx];
  }
  { int idx = kOffB4 + t; og[idx] *= inv_n; local_norm_sq += og[idx] * og[idx]; }
  {
    int idx = kOffW5 + target_out * kHiddenDim + t;
    og[idx] *= inv_n;
    local_norm_sq += og[idx] * og[idx];
  }
  if (t == target_out) {
    int idx = kOffB5 + target_out;
    og[idx] *= inv_n;
    local_norm_sq += og[idx] * og[idx];
  }

  s_reduce[t] = local_norm_sq;
  __syncthreads();
  for (int s = kHiddenDim / 2; s > 0; s >>= 1) {
    if (t < s) s_reduce[t] += s_reduce[t + s];
    __syncthreads();
  }
  float out_norm = sqrtf(s_reduce[0]) + 1e-8f;
  float clip_scale = (out_norm > kGradClipThreshold) ? (kGradClipThreshold / out_norm) : 1.0f;
  float lr_out = learning_rate * clip_scale;

  /** The gradient is left UNSCALED and lr_out is published for phase C. */
  if (t == 0) w->lr_head[target_out] = lr_out;
  __syncthreads();
}

/** The output-space trust region, ONE BLOCK PER SAMPLE. */
__global__ void SgdTrustKernel(NeuroPressGpuWeights *w, int num_samples,
                               const float *__restrict__ ema) {
  const int si = static_cast<int>(blockIdx.x);
  const int t = static_cast<int>(threadIdx.x);
  __shared__ float s_reduce[kHiddenDim];
  if (si >= num_samples) {
    // Not a sample this call carries: publish the neutral element so the fold below can read a f...
    if (t == 0) w->dy_sample[si] = 0.0f;
    return;
  }
  __shared__ float s_tan_a[kHiddenDim];
  __shared__ float s_tan_b[kHiddenDim];
  float dy_local = 0.0f;
  {
    // L1: dz1 = dW1 x + db1. The input is fixed, so there is no dx term.
    float dz = ema[kOffB1 + t];
    for (int i = 0; i < kInputDim; ++i)
      dz += ema[kOffW1 + t * kInputDim + i] * w->act_x[si][i];
    s_tan_a[t] = (w->act_h1[si][t] > 0.0f) ? dz : 0.0f;
    __syncthreads();
    // L2: dz2 = dW2 h1 + W2 dh1 + db2, and likewise below.
    dz = ema[kOffB2 + t];
    for (int i = 0; i < kHiddenDim; ++i)
      dz += ema[kOffW2 + t * kHiddenDim + i] * w->act_h1[si][i] +
            w->params[kOffW2 + t * kHiddenDim + i] * s_tan_a[i];
    s_tan_b[t] = (w->act_h2[si][t] > 0.0f) ? dz : 0.0f;
    __syncthreads();
    dz = ema[kOffB3 + t];
    for (int i = 0; i < kHiddenDim; ++i)
      dz += ema[kOffW3 + t * kHiddenDim + i] * w->act_h2[si][i] +
            w->params[kOffW3 + t * kHiddenDim + i] * s_tan_b[i];
    s_tan_a[t] = (w->act_h3[si][t] > 0.0f) ? dz : 0.0f;
    __syncthreads();
    dz = ema[kOffB4 + t];
    for (int i = 0; i < kHiddenDim; ++i)
      dz += ema[kOffW4 + t * kHiddenDim + i] * w->act_h3[si][i] +
            w->params[kOffW4 + t * kHiddenDim + i] * s_tan_a[i];
    s_tan_b[t] = (w->act_h4[si][t] > 0.0f) ? dz : 0.0f;
    __syncthreads();
    // Heads 0..3, with or without an error: an unbounded head drifts (VPIC dt).
    if (t < 4) {
      float dy = 0.0f;
      for (int i = 0; i < kHiddenDim; ++i) {
        dy += w->params[kOffW5 + t * kHiddenDim + i] * s_tan_b[i];
        if (t != 1) dy += ema[kOffW5 + t * kHiddenDim + i] * w->act_h4[si][i];
      }
      if (t != 1) dy += ema[kOffB5 + t];
      dy_local = fmaxf(dy_local, fabsf(dy));
    }
    __syncthreads();
  }
  s_reduce[t] = dy_local;
  __syncthreads();
  for (int s2 = kHiddenDim / 2; s2 > 0; s2 >>= 1) {
    if (t < s2) s_reduce[t] = fmaxf(s_reduce[t], s_reduce[t + s2]);
    __syncthreads();
  }
  if (t == 0) w->dy_sample[si] = s_reduce[0];
}

/** The ordered fold of the eight per-head gradients, on a full grid. */
__global__ void SgdFoldKernel(NeuroPressGpuWeights *w) {
  const int stride = static_cast<int>(gridDim.x * blockDim.x);
  for (int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
       i < kParamCount; i += stride) {
    float acc = 0.0f;
    for (int b = 0; b < kOutputDim; ++b) {
      acc = __fmaf_rn(w->lr_head[b], w->out_grad[b][i], acc);
    }
    w->combined[i] = acc;
  }
}

/** SGD phase C: fold the eight per-head gradients together, then the trust-region step, anti-fl... */
__global__ void SgdApplyKernel(NeuroPressGpuWeights *w, int num_samples,
                               float *__restrict__ ema, float out_delta) {
  int t = threadIdx.x;  // 0..63
  __shared__ float s_reduce[kHiddenDim];


  // ---- Trust-region step, anti-flip damping, EMA smoothing, weight update ----
  constexpr float kEmaDecay = 0.85f;
  constexpr float kTrustK = 0.08f;
  constexpr float kMaxStep = 0.02f;
  constexpr float kMinStep = 1e-4f;
  constexpr float kAntiFlipDamp = 0.5f;
  constexpr float kWClamp = 10.0f;

  // Gradient norm (sum over all params, block-wide reduction via strided per-thread partial sums...
  float local_norm_sq = 0.0f;
  for (int i = t; i < kParamCount; i += kHiddenDim) local_norm_sq += w->combined[i] * w->combined[i];
  s_reduce[t] = local_norm_sq;
  __syncthreads();
  for (int s = kHiddenDim / 2; s > 0; s >>= 1) {
    if (t < s) s_reduce[t] += s_reduce[t + s];
    __syncthreads();
  }
  float g_norm = sqrtf(s_reduce[0]) + 1e-8f;
  float inv_norm = 1.0f / g_norm;
  for (int i = t; i < kParamCount; i += kHiddenDim) w->combined[i] *= inv_norm;
  __syncthreads();

  __shared__ float s_avg_err;
  if (t == 0) {
    float sum_err = 0.0f;
    int count = 0;
    for (int si = 0; si < num_samples; ++si) {
      for (int o = 0; o < kOutputDim; ++o) {
        float e = fabsf(w->d5_raw[si][o]);
        if (e > 0.0f) { sum_err += e; ++count; }
      }
    }
    s_avg_err = (count > 0) ? sum_err / static_cast<float>(count) : 0.0f;
  }
  __syncthreads();
  float step = fmaxf(kMinStep, fminf(kMaxStep, kTrustK * s_avg_err));

  bool warmed_up = (w->sgd_call_count > 3);
  if (warmed_up) {
    // TRUNK ONLY (params before the W5 block).
    float local_dot = 0.0f;
    for (int i = t; i < kOffW5; i += kHiddenDim) local_dot += w->combined[i] * ema[i];
    s_reduce[t] = local_dot;
    __syncthreads();
    for (int s = kHiddenDim / 2; s > 0; s >>= 1) {
      if (t < s) s_reduce[t] += s_reduce[t + s];
      __syncthreads();
    }
    // s_reduce[0] holds the block-wide dot product after the reduction above; every thread reads...
    if (s_reduce[0] < 0.0f) step *= kAntiFlipDamp;
  }

  for (int i = t; i < kParamCount; i += kHiddenDim)
    if (!isfinite(w->combined[i])) w->combined[i] = 0.0f;
  __syncthreads();

  float ema_new = 1.0f - kEmaDecay;
  for (int i = t; i < kParamCount; i += kHiddenDim)
    ema[i] = kEmaDecay * ema[i] + ema_new * w->combined[i];
  __syncthreads();

  // ---- Output-space trust region.
  if (t == 0) {
    w->sgd_step = step;
    w->sgd_g_norm = g_norm;
  }
}

/** Fold the per-sample maxima, apply the output-space bound, and take the step. */
__global__ void SgdStepKernel(NeuroPressGpuWeights *w, bool *out_applied,
                              float out_delta,
                              const float *__restrict__ ema) {
  constexpr float kWClamp = 10.0f;
  const int t = static_cast<int>(threadIdx.x);
  float step = w->sgd_step;
  const float g_norm = w->sgd_g_norm;

  if (out_delta > 0.0f) {
    // The same maximum over the same set, bracketed per sample instead of per thread.
    float dy_max = 0.0f;
    for (int si = 0; si < kMaxSamples; ++si) {
      dy_max = fmaxf(dy_max, w->dy_sample[si]);
    }
    // params -= step * ema, so a head moves by about step * dy_max.
    if (isfinite(dy_max) && dy_max * step > out_delta) step = out_delta / dy_max;
  }

  if (t == 0) w->sgd_call_count += 1;

  bool finite_ok = isfinite(step) && isfinite(g_norm);
  if (finite_ok) {
    // Output 1 (decompression time) is owned by the deferred head-only pass...
    const int skip_w5_begin = kOffW5 + 1 * kHiddenDim;
    const int skip_w5_end = skip_w5_begin + kHiddenDim;
    const int skip_b5 = kOffB5 + 1;
    for (int i = t; i < kParamCount; i += kHiddenDim) {
      if ((i >= skip_w5_begin && i < skip_w5_end) || i == skip_b5) continue;
      float p = w->params[i] - step * ema[i];
      w->params[i] = fmaxf(-kWClamp, fminf(kWClamp, p));
    }
  }
  if (t == 0) *out_applied = finite_ok;
}


/** Block-wide sum of one float per thread; every thread gets the total. */
__device__ __forceinline__ float BlockSum(float *s_reduce, int t, float v) {
  s_reduce[t] = v;
  __syncthreads();
  for (int s = kHiddenDim / 2; s > 0; s >>= 1) {
    if (t < s) s_reduce[t] += s_reduce[t + s];
    __syncthreads();
  }
  const float total = s_reduce[0];
  __syncthreads();
  return total;
}

/** The adaptive update law (CLIO_NEUROPRESS_SGD_RULE=adaptive). */
__global__ void AdaptiveSGDKernel(
    NeuroPressGpuWeights *w,
    const NeuroPressGpuSGDSample *__restrict__ samples, int num_samples,
    float learning_rate, float *__restrict__ ema, bool *out_applied,
    const ctp::DeviceFeatureStats *__restrict__ device_stats, AdaptOptions o) {
  const int t = threadIdx.x;  // 0..63
  __shared__ float s_reduce[kHiddenDim];
  __shared__ float s_dz4[kHiddenDim], s_dz3[kHiddenDim], s_dz2[kHiddenDim];

  SgdForwardAndErrors(w, samples, num_samples, t, device_stats, o.ratio_target_cap);

  for (int i = t; i < kParamCount; i += kHiddenDim) w->combined[i] = 0.0f;
  __syncthreads();

  // ---- One backward pass over all eight heads, summed over samples ----
  for (int si = 0; si < num_samples; ++si) {
    float dh4 = 0.0f;
    for (int oi = 0; oi < kOutputDim; ++oi) {
      const float e = w->d5_clamped[si][oi];
      w->combined[kOffW5 + oi * kHiddenDim + t] += e * w->act_h4[si][t];
      dh4 += w->params[kOffW5 + oi * kHiddenDim + t] * e;
    }
    if (t < kOutputDim) w->combined[kOffB5 + t] += w->d5_clamped[si][t];
    const float dz4 = (w->act_h4[si][t] > 0.0f) ? dh4 : 0.0f;

    for (int i = 0; i < kHiddenDim; ++i)
      w->combined[kOffW4 + t * kHiddenDim + i] += dz4 * w->act_h3[si][i];
    w->combined[kOffB4 + t] += dz4;
    s_dz4[t] = dz4;
    __syncthreads();

    float dh3 = 0.0f;
    for (int j = 0; j < kHiddenDim; ++j)
      dh3 += w->params[kOffW4 + j * kHiddenDim + t] * s_dz4[j];
    const float dz3 = (w->act_h3[si][t] > 0.0f) ? dh3 : 0.0f;
    for (int i = 0; i < kHiddenDim; ++i)
      w->combined[kOffW3 + t * kHiddenDim + i] += dz3 * w->act_h2[si][i];
    w->combined[kOffB3 + t] += dz3;
    s_dz3[t] = dz3;
    __syncthreads();

    float dh2 = 0.0f;
    for (int j = 0; j < kHiddenDim; ++j)
      dh2 += w->params[kOffW3 + j * kHiddenDim + t] * s_dz3[j];
    const float dz2 = (w->act_h2[si][t] > 0.0f) ? dh2 : 0.0f;
    for (int i = 0; i < kHiddenDim; ++i)
      w->combined[kOffW2 + t * kHiddenDim + i] += dz2 * w->act_h1[si][i];
    w->combined[kOffB2 + t] += dz2;
    s_dz2[t] = dz2;
    __syncthreads();

    float dh1 = 0.0f;
    for (int j = 0; j < kHiddenDim; ++j)
      dh1 += w->params[kOffW2 + j * kHiddenDim + t] * s_dz2[j];
    const float dz1 = (w->act_h1[si][t] > 0.0f) ? dh1 : 0.0f;
    for (int i = 0; i < kInputDim; ++i)
      w->combined[kOffW1 + t * kInputDim + i] += dz1 * w->act_x[si][i];
    w->combined[kOffB1 + t] += dz1;
    __syncthreads();
  }

  // ---- Mean over samples, global-norm clip (scale DOWN only) ----
  const float inv_n = 1.0f / static_cast<float>(num_samples);
  float local = 0.0f;
  for (int i = t; i < kParamCount; i += kHiddenDim) {
    float g = w->combined[i] * inv_n;
    if (!isfinite(g)) g = 0.0f;
    w->combined[i] = g;
    local += g * g;
  }
  const float g_norm = sqrtf(BlockSum(s_reduce, t, local));
  float clip = 1.0f;
  if (o.grad_clip > 0.0f && g_norm > o.grad_clip) clip = o.grad_clip / g_norm;

  // Momentum on the clipped, UNNORMALISED gradient: 0 is plain SGD.
  for (int i = t; i < kParamCount; i += kHiddenDim)
    ema[i] = o.momentum * ema[i] + (1.0f - o.momentum) * w->combined[i] * clip;
  __syncthreads();

  // ---- Per-block learning rates: decayed base, trunk frozen for HEAD_STEPS ----
  float lr = learning_rate;
  if (o.decay_tau > 0.0f) {
    lr /= sqrtf(1.0f + static_cast<float>(w->sgd_call_count) / o.decay_tau);
  }
  const bool trunk_on =
      (o.head_steps >= 0) && (w->sgd_call_count >= o.head_steps);

  // Output 1's head row/bias stays owned by the deferred decomp pass, exactly as the ported kern...
  const int skip_w5_begin = kOffW5 + 1 * kHiddenDim;
  const int skip_w5_end = skip_w5_begin + kHiddenDim;
  const int skip_b5 = kOffB5 + 1;
  float step_sq = 0.0f;
  for (int i = t; i < kParamCount; i += kHiddenDim) {
    float d;
    if (i >= kOffW5) {
      d = ((i >= skip_w5_begin && i < skip_w5_end) || i == skip_b5)
              ? 0.0f
              : lr * ema[i];
    } else {
      d = trunk_on ? lr * o.trunk_scale * ema[i] : 0.0f;
    }
    w->out_grad[0][i] = d;
    step_sq += d * d;
  }
  const float step_norm = sqrtf(BlockSum(s_reduce, t, step_sq));

  // Trust region: shrink an oversized firing, never grow a small one.
  float scale = 1.0f;
  if (o.max_step > 0.0f && step_norm > o.max_step) scale = o.max_step / step_norm;

  if (t == 0) w->sgd_call_count += 1;
  const bool finite_ok = isfinite(step_norm) && isfinite(scale);
  if (finite_ok) {
    constexpr float kWClamp = 10.0f;
    for (int i = t; i < kParamCount; i += kHiddenDim) {
      const float pv = w->params[i] - scale * w->out_grad[0][i];
      w->params[i] = fmaxf(-kWClamp, fminf(kWClamp, pv));
    }
  }
  if (t == 0) *out_applied = finite_ok;
}

bool NeuroPressGpuTrain(NeuroPressGpuWeights *w,
                        const NeuroPressGpuSGDSample *samples,
                        int num_samples, float learning_rate,
                        const void *device_stats) {
  if (!w || num_samples <= 0) return false;
  if (num_samples > kMaxSamples) num_samples = kMaxSamples;

  /** Choreography matched to upstream's runNNSGDCtx (nn_gpu.cu) step for step: async H->D of th... */
  SgdSync &g = Sgd();
  if (!g.ok) return false;

  const size_t bytes =
      sizeof(NeuroPressGpuSGDSample) * static_cast<size_t>(num_samples);
  SgdScratch &sc = SgdSamples();
  if (!EnsureSgdSamples(sc, bytes)) return false;

  float *ema = EmaBuffer();
  if (!ema) return false;

  /** The flag the kernel writes. */
  static thread_local bool *d_applied = [] {
    bool *p = nullptr;
    if (cudaMalloc(&p, sizeof(bool)) != cudaSuccess) return static_cast<bool *>(nullptr);
    return p;
  }();
  if (!d_applied) return false;

  if (cudaMemcpyAsync(sc.d_samples, samples, bytes, cudaMemcpyHostToDevice,
                      g.stream) != cudaSuccess) {
    return false;
  }

  /* On the SGD stream, so it is ordered before the kernel that reads it. */
  const ctp::DeviceFeatureStats *d_stats = nullptr;
  if (device_stats != nullptr) {
    ctp::DeviceFeatureStats *snap = SgdStatsSnapshot();
    if (snap != nullptr &&
        cudaMemcpyAsync(snap, device_stats, sizeof(ctp::DeviceFeatureStats),
                        cudaMemcpyDeviceToDevice, g.stream) == cudaSuccess) {
      d_stats = snap;
    }
    /* Fall back to the samples' host rows, not another chunk's stats. */
  }

  /** One `if` on the parsed options: `upstream` (the default) reaches exactly the kernel it alw... */
  const AdaptOptions &opt = Opts();
  const float lr = (opt.lr > 0.0f) ? opt.lr : learning_rate;
  if (opt.adaptive) {
    AdaptiveSGDKernel<<<1, kHiddenDim, 0, g.stream>>>(
        w, static_cast<const NeuroPressGpuSGDSample *>(sc.d_samples),
        num_samples, lr, ema, d_applied, d_stats, opt);
  } else {
    /** Three launches where there was one, all on the SGD stream, so they chain on the device e... */
    SgdPrepareKernel<<<1, kHiddenDim, 0, g.stream>>>(
        w, static_cast<const NeuroPressGpuSGDSample *>(sc.d_samples),
        num_samples, d_stats, opt.ratio_target_cap);
    SgdPerOutputKernel<<<kOutputDim, kHiddenDim, 0, g.stream>>>(
        w, num_samples, lr);
    constexpr int kFoldBlock = 256;
    SgdFoldKernel<<<(kParamCount + kFoldBlock - 1) / kFoldBlock, kFoldBlock, 0,
                    g.stream>>>(w);
    SgdApplyKernel<<<1, kHiddenDim, 0, g.stream>>>(
        w, num_samples, ema, opt.out_delta);
    if (opt.out_delta > 0.0f) {
      SgdTrustKernel<<<kMaxSamples, kHiddenDim, 0, g.stream>>>(w, num_samples,
                                                               ema);
    }
    SgdStepKernel<<<1, kHiddenDim, 0, g.stream>>>(w, d_applied, opt.out_delta,
                                                  ema);
  }
  if (cudaGetLastError() != cudaSuccess) return false;

  /** Only set the flag if the record succeeded, so inference never waits on an unrecorded event... */
  if (cudaEventRecord(g.done, g.stream) == cudaSuccess) {
    g.ever_fired.store(true, std::memory_order_release);
  }
  return true;
}

int NeuroPressGpuSgdCallCount(NeuroPressGpuWeights *w) {
  if (!w) return 0;
  SgdHostWaitIfEverFired();
  int n = 0;
  cudaMemcpy(&n, &w->sgd_call_count, sizeof(int), cudaMemcpyDeviceToHost);
  return n;
}

}  // namespace ctp::compress::model::gpu
