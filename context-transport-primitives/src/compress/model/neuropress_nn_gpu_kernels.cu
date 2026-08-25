/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

/**
 * @file neuropress_nn_gpu_kernels.cu
 * @brief CUDA kernels for NeuroPress inference and online SGD training.
 *
 * Ports NeuroPress's nn_gpu.cu (nnFusedInferenceKernel, nnSGDKernel) to
 * Clio's own weight layout. Weights, EMA gradient state, and uncertainty
 * log-variance all live in device memory for the lifetime of the handle --
 * the host never touches the decision data, matching the original design.
 * Per-sample activations and gradient accumulators are cached in a
 * persistent device scratch buffer (allocated once, not per-call) rather
 * than shared memory, trading a little device memory for a simpler,
 * easier-to-verify kernel; the numerical algorithm itself is unchanged from
 * upstream (same clamps, same constants, same order of operations).
 */

#include "clio_ctp/compress/model/neuropress_nn_gpu_kernels.h"
// For ctp::DeviceFeatureStats -- the device-resident feature triple the
// device-stats inference kernel reads instead of a host-built matrix.
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

/**
 * Env-gated deviations from the ported rule. Every field's default reproduces
 * upstream exactly, so a process that sets nothing runs the shipped kernel on
 * the shipped path. Read once (see Opts()) and passed to the kernels by value.
 */
struct AdaptOptions {
  bool adaptive = false;      // CLIO_NEUROPRESS_SGD_RULE=adaptive
  float lr = 0.0f;            // CLIO_NEUROPRESS_SGD_LR (<=0: caller's rate)
  int head_steps = 32;        // CLIO_NEUROPRESS_SGD_HEAD_STEPS
  float trunk_scale = 0.1f;   // CLIO_NEUROPRESS_SGD_TRUNK_SCALE
  float decay_tau = 100.0f;   // CLIO_NEUROPRESS_SGD_DECAY_TAU (0: no decay)
  float max_step = 0.02f;     // CLIO_NEUROPRESS_SGD_MAX_STEP
  float grad_clip = 1.0f;     // CLIO_NEUROPRESS_SGD_GRAD_CLIP (<=0: off)
  float momentum = 0.85f;     // CLIO_NEUROPRESS_SGD_MOMENTUM
};
}  // namespace

/** @brief Device-resident state. Persists for the handle's lifetime. */
struct NeuroPressGpuWeights {
  float params[kParamCount];        // w1,b1,w2,b2,w3,b3,w4,b4,w5,b5
  float x_means[kInputDim];
  float x_stds[kInputDim];
  float y_means[kOutputDim];
  float y_stds[kOutputDim];

  // Online-learning state (never serialized to .nnwt, matches upstream).
  float log_var[kOutputDim];
  // NOTE: the EMA gradient does NOT live here. Upstream keeps it per
  // CompContext (ctx->d_sgd_grad_buffer's EMA_REGION) while the weights and
  // sgd_call_count are global (nn_gpu.cu, :1413) -- so concurrent flows
  // smooth their gradients independently but share the model. See EmaBuffer().
  int sgd_call_count;

  // Persistent scratch for Train() -- avoids per-call cudaMalloc.
  float act_x[kMaxSamples][kInputDim];
  float act_z1[kMaxSamples][kHiddenDim], act_h1[kMaxSamples][kHiddenDim];
  float act_z2[kMaxSamples][kHiddenDim], act_h2[kMaxSamples][kHiddenDim];
  float act_z3[kMaxSamples][kHiddenDim], act_h3[kMaxSamples][kHiddenDim];
  float act_z4[kMaxSamples][kHiddenDim], act_h4[kMaxSamples][kHiddenDim];
  float act_y[kMaxSamples][kOutputDim];
  float d5_clamped[kMaxSamples][kOutputDim];
  float d5_raw[kMaxSamples][kOutputDim];
  float combined[kParamCount];
  float out_grad[kParamCount];
  float dz4_all[kOutputDim][kHiddenDim];
};

namespace {

/**
 * The options, parsed ONCE per process on first use.
 *
 * A function-local static, so the runtime and the direct-predictor harnesses
 * see the same values with no per-call getenv on the training path.
 */
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
    return a;
  }();
  return o;
}

/**
 * Per-flow EMA gradient buffer.
 *
 * Upstream splits its online-learning state: weights and sgd_call_count are
 * global, but the EMA gradient lives in a per-CompContext buffer. That split is
 * load-bearing -- the weight step is `w -= step * EMA[...]`, not the raw
 * gradient -- so whatever the buffer is scoped to is what learning is smoothed
 * over. Clio reproduces the split: shared weights and call count, separate EMA.
 *
 * The SCOPE is an approximation. Upstream's buffer belongs to a pool slot taken
 * per compression call, so gradient history follows the slot and one slot mixes
 * unrelated flows; here it belongs to the thread for its lifetime, so history
 * is continuous per worker. The two agree on the math and the split but bucket
 * samples differently, and can reach different weights on the same concurrent
 * workload. Neither is more correct -- upstream's bucketing depends on which
 * slot happens to be free, so it does not define a deterministic per-flow EMA
 * either. Thread scope is the closest stable analogue at this layer, where
 * there is no CompContext to attach to.
 *
 * Two consequences of the wider scope, neither load-bearing today: the buffer
 * count is unbounded where upstream's is capped at 9, and each is cudaMalloc'd
 * on first use and never freed, so thread churn would leak 54 KB a time.
 * Train() is reached only from the runtime's fixed worker pool.
 *
 * Registered globally so a model reload can zero every live buffer, matching
 * upstream's resetAllSGDEMABuffers(). See Registry() for why it is never torn
 * down.
 */
struct EmaRegistry {
  std::mutex mutex;
  std::vector<float *> buffers;
};

/**
 * The registry is allocated on first use and DELIBERATELY NEVER DESTROYED.
 *
 * The mutex serializes push_back against ResetAllEmaBuffers, but a destructor
 * takes no lock: at static-destruction time ~vector() would free the element
 * storage while a runtime worker can still be inside EmaBuffer()'s push_back --
 * and a growing push_back frees the old block itself, so both free the same
 * pointer. Leaking removes the destructor entirely, which is the only way to be
 * sure; there is no point in teardown at which this is safe to destroy while
 * any thread might still reach it.
 *
 * Costs nothing beyond what is already leaked on purpose -- the EMA buffers it
 * tracks are themselves never freed -- and the function-local static gives
 * thread-safe initialization with no ordering dependency on another global.
 */
EmaRegistry &Registry() {
  static EmaRegistry *r = new EmaRegistry();
  return *r;
}

/* SGD stream, completion event, and "has SGD ever run" flag.
 *
 * These mirror upstream's g_sgd_stream / g_sgd_done / g_sgd_ever_fired
 * (nn_gpu.cu). The ordering between "update the weights" and "read the
 * weights" is a GPU-side dependency there, not a host one: SGD runs on its own
 * stream and records an event, and inference issues cudaStreamWaitEvent on
 * that event before its kernel. The host never blocks for it.
 *
 * Process-wide, not per-thread, and that is required rather than convenient:
 * every worker shares ONE device weight buffer, so an update issued by one
 * worker must be visible to inference issued by another. A per-thread event
 * would only order a thread against itself. */
struct SgdSync {
  cudaStream_t stream = nullptr;
  cudaEvent_t done = nullptr;
  std::atomic<bool> ever_fired{false};
  bool ok = false;
};

SgdSync &Sgd() {
  /* Initialized in place: SgdSync holds an atomic and so is not copyable, and
     returning one from a lambda would need a copy. */
  static SgdSync s;
  static const bool once = [] {
    /* Non-blocking: this stream must not implicitly synchronize with the
       legacy default stream, or recording an event on it would serialize
       against every other worker -- the exact cost this exists to avoid. */
    s.ok = cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking) ==
               cudaSuccess &&
           cudaEventCreateWithFlags(&s.done, cudaEventDisableTiming) ==
               cudaSuccess;
    return true;
  }();
  (void)once;
  return s;
}

/* Persistent per-thread SGD sample buffer, grown rather than reallocated.
 * Upstream allocates ctx->d_sgd_samples once per CompContext
 * (gpucompress_pool.cpp) and never frees it per call; this is the same
 * property without a context pool. */
struct SgdScratch {
  void *d_samples = nullptr;
  size_t bytes = 0;
};

SgdScratch &SgdSamples() {
  static thread_local SgdScratch s;
  return s;
}

/* SGD-owned copy of the chunk's statistics: the resident buffer is reused by
 * the next chunk while a fire-and-forget launch may still be queued. Upstream's
 * D2D into ctx->d_stats (gpucompress_compress.cpp). Leaked like EmaBuffer(). */
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

/* Called by the HOST-side weight readback before it copies.
 *
 * The stream variant above expresses the dependency GPU-side, which is what
 * inference needs and what upstream does. A host readback needs the host
 * itself to wait, and needs it for a reason specific to this design: the SGD
 * stream is cudaStreamNonBlocking precisely so it does NOT serialize against
 * the legacy default stream -- which means a blocking cudaMemcpy on that
 * stream is not ordered against SGD either, and copies the weights as they
 * were BEFORE the update. That is a one-step lag, not a corruption, so it
 * reads as a plausible parity result rather than an obvious bug.
 *
 * Waiting on the event rather than the stream keeps this correct under
 * --default-stream per-thread, where "the default stream" is not one stream. */
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

NeuroPressGpuWeights *NeuroPressGpuLoad(const float *weights, size_t weights_len,
                                        const float *biases, size_t biases_len,
                                        const float *x_means, const float *x_stds,
                                        const float *y_means, const float *y_stds) {
  // Layout sanity: caller's flattened weights_ (13312 = 512+3*4096+512) plus
  // biases_ (264 = 4*64+8) must add up to exactly kParamCount (13576) when
  // interleaved below. If the .nnwt format ever changes shape this catches
  // it instead of silently corrupting device memory.
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

  // Host-side: interleave weights_[]/biases_[] (Clio's own layout, 5
  // separate weight matrices + 5 separate bias vectors) into this kernel's
  // single flat params[] buffer (w1,b1,w2,b2,...), matching the offsets
  // above. weights_len/biases_len callers pass the FULL flattened arrays;
  // per-layer sizes are fixed by the architecture (8->64->64->64->64->8).
  // Staging for a SINGLE upload. params, x_means, x_stds, y_means and y_stds
  // are contiguous at the front of NeuroPressGpuWeights, so the whole prefix
  // ships in one cudaMemcpy instead of five -- which is what upstream does
  // (nn_gpu.cu copies the entire NNWeightsGPU in one call).
  //
  // Clio cannot copy the WHOLE struct the way upstream does: ours also carries
  // the persistent Train() scratch (act_x, act_z1..., d5_clamped), which is
  // device-only and must not be uploaded. Upstream keeps its SGD scratch per
  // CompContext instead, which is why its struct is copyable wholesale. The
  // prefix is exactly the part that is model data.
  constexpr size_t kOffXMeans = kParamCount;
  constexpr size_t kOffXStds = kOffXMeans + kInputDim;
  constexpr size_t kOffYMeans = kOffXStds + kInputDim;
  constexpr size_t kOffYStds = kOffYMeans + kOutputDim;
  constexpr size_t kPrefixFloats = kOffYStds + kOutputDim;
  // The single copy is only correct if the device struct really is laid out
  // the way this staging buffer assumes. Every member is a float array so no
  // padding can appear between them, but assert it rather than trust it:
  // inserting a member into the struct above would otherwise silently corrupt
  // the normalization constants instead of failing to build.
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

  // A reload restarts learning: zero every flow's gradient history, as
  // upstream's resetAllSGDEMABuffers() does.
  ResetAllEmaBuffers();

  std::memcpy(host_params + kOffXMeans, x_means, sizeof(float) * kInputDim);
  std::memcpy(host_params + kOffXStds, x_stds, sizeof(float) * kInputDim);
  std::memcpy(host_params + kOffYMeans, y_means, sizeof(float) * kOutputDim);
  std::memcpy(host_params + kOffYStds, y_stds, sizeof(float) * kOutputDim);

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
  // Read-modify-write: params[] also holds nothing else, but the device copy
  // is the live one, so start from it rather than zero-filling anything this
  // mapping does not cover.
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

// ============================================================================
// Inference: one block per candidate, thread t owns hidden neuron t.
// ============================================================================
constexpr int kMaxCandidates = 32;

/** ct, dt, ratio, psnr -- the four per-candidate outputs, packed in one buffer. */
/* Eight, not four: upstream's device-resident entry point
   (runNNFusedInferenceCtx) hands back rmse/max_error/mae/ssim alongside the
   four the ranking uses, and Clio's could not. The extra 4*cap floats are the
   cost of that parity -- 512 bytes at the 32-wide device path. The per-chunk
   D2H is NOT doubled: the fetch copies only as far as the caller asked (see
   FetchPredictionsSync), so a ranking-only call moves exactly what it did. */
constexpr int kPredOutputs = 8;

/**
 * decodeAction, in the kernel.
 *
 * `action = algo + 8*quant + 16*shuffle` (internal.hpp). Upstream's
 * inference kernel inverts it straight off the thread index
 * (`algo_idx = tid % 8; quant = (tid/8) % 2; shuffle = (tid/16) % 2`,
 * nn_gpu.cu) and builds each config's inputs from that -- no host
 * array of per-candidate settings exists there at all. Clio passes the action
 * ids because its candidate set can be a subset (an algorithm this build
 * cannot construct is never enumerated), but the DECODE now happens here,
 * where upstream does it.
 *
 * shuffle is fed to the network as 0/1, not as the 4-byte element size:
 * `input_raw[2] = static_cast<float>(shuffle)` with shuffle = (tid/16)%2
 * (nn_gpu.cu, :142).
 */
__device__ __forceinline__ void DecodeAction(int action, int *algo, int *quant,
                                             int *shuffle) {
  *algo = action % 8;
  *quant = (action / 8) % 2;
  *shuffle = (action / 16) % 2;
}

/**
 * Layers, inverse transform and clamps, shared by both inference entry points.
 *
 * Factored out rather than duplicated: the two kernels differ ONLY in where
 * the eight raw inputs come from (a host-built matrix, or the device stats
 * struct plus a candidate descriptor). Everything after standardization is
 * the model, and two copies of it would be free to drift -- which is exactly
 * the class of divergence this whole line of work exists to catch.
 *
 * `s_x` must already hold the STANDARDIZED inputs and the block must have
 * synchronized on them before calling.
 */
__device__ __forceinline__ void NeuroPressForwardShared(
    const NeuroPressGpuWeights *__restrict__ w, const float *__restrict__ s_x,
    float *s_h1, float *s_h2, float *s_h3, float *s_h4, float *s_y, int t,
    int cand, float *__restrict__ out_comp_time,
    float *__restrict__ out_decomp_time, float *__restrict__ out_ratio,
    /* Policy ratio ceiling. Upstream's RATIO_CAP is a literal 100
       (nn_gpu.cu); it is a parameter here only so an experiment can raise it,
       and every caller that does not opt in passes 100.0f. */
    float ratio_cap,
    float *__restrict__ out_psnr,
    /* Outputs 4-7. Optional: selection reads none of them (upstream's own
       NN_INFER_OUTPUTS is 4, nn_weights.h:15), so a caller that only ranks
       passes nullptr and the inverse transforms are never evaluated. */
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
    // Sanity clamps BEFORE the policy clamps, exactly as nn_gpu.cu
    // orders them. The 1e6 ceiling is what makes a non-finite prediction
    // safe: fminf(NaN, 1e6) returns 1e6, so a head that SGD has drifted
    // into NaN ranks WORST. Without it, fmaxf(1.0f, NaN) yields 1.0 and the
    // broken candidate becomes the cheapest one in the cost model, winning
    // outright. Same reasoning for the 0.1 ratio floor.
    comp_time = fmaxf(1e-6f, fminf(comp_time, 1e6f));
    decomp_time = fmaxf(1e-6f, fminf(decomp_time, 1e6f));
    ratio = fmaxf(0.1f, fminf(ratio, 1e5f));
    psnr = fmaxf(0.0f, fminf(psnr, 120.0f));

    out_comp_time[cand] = fmaxf(1.0f, comp_time);
    out_decomp_time[cand] = fmaxf(1.0f, decomp_time);
    out_ratio[cand] = fminf(ratio_cap, ratio);
    out_psnr[cand] = psnr;

    // Outputs 4-7, upstream nn_gpu.cu:211-216 for the transforms and :223-226
    // for the clamps. 4-6 are log1p-transformed like ratio/times; 7 is stored
    // as -log(1-ssim), so it inverts with 1-exp(-x) and NOT with expm1f.
    //
    // The clamps are NOT optional and were missed on the first pass here: an
    // untrained head readily produces a small negative expm1f result, and
    // upstream floors all three error metrics at 0 (and ssim into [0,1]).
    // Without them Clio reported values like rmse = -1.4e-4, which is not a
    // rounding difference from upstream's 0 but a physically meaningless
    // number. The differential test against upstream's own per-config output
    // is what caught it.
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
                            float ratio_cap = 100.0f) {
  int cand = blockIdx.x;
  int t = threadIdx.x;

  __shared__ float s_x[kInputDim];
  __shared__ float s_h1[kHiddenDim], s_h2[kHiddenDim], s_h3[kHiddenDim],
      s_h4[kHiddenDim];
  __shared__ float s_y[kOutputDim];

  if (t < kInputDim) {
    float std_val = w->x_stds[t];
    if (std_val < 1e-8f) std_val = 1e-8f;
    const float raw =
        raw_inputs[cand * kInputDim + t];
    s_x[t] = (raw - w->x_means[t]) / std_val;
  }
  __syncthreads();

  NeuroPressForwardShared(w, s_x, s_h1, s_h2, s_h3, s_h4, s_y, t, cand,
                          out_comp_time, out_decomp_time, out_ratio, ratio_cap,
                          out_psnr);
}

/**
 * All eight outputs, for reporting and for differential testing against
 * upstream. Selection never calls this -- it needs only the first four, and
 * upstream says so itself (NN_INFER_OUTPUTS = 4, nn_weights.h:15). Kept as a
 * separate kernel rather than as optional arguments on InferKernel so the
 * ranking path's register footprint and launch signature are untouched.
 */
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
                                float ratio_cap = 100.0f) {
  int cand = blockIdx.x;
  int t = threadIdx.x;

  __shared__ float s_x[kInputDim];
  __shared__ float s_h1[kHiddenDim], s_h2[kHiddenDim], s_h3[kHiddenDim],
      s_h4[kHiddenDim];
  __shared__ float s_y[kOutputDim];

  if (t < kInputDim) {
    float std_val = w->x_stds[t];
    if (std_val < 1e-8f) std_val = 1e-8f;
    const float raw =
        raw_inputs[cand * kInputDim + t];
    s_x[t] = (raw - w->x_means[t]) / std_val;
  }
  __syncthreads();

  NeuroPressForwardShared(w, s_x, s_h1, s_h2, s_h3, s_h4, s_y, t, cand,
                          out_comp_time, out_decomp_time, out_ratio, ratio_cap,
                          out_psnr, out_rmse, out_max_error, out_mae, out_ssim);
}

/**
 * Device-stats entry point: reads the three data features straight out of
 * device memory instead of receiving them in a host-built matrix.
 *
 * This is the shape of upstream's nnFusedInferenceKernel, which takes an
 * `AutoStatsGPU*` and constructs each config's input vector in-kernel
 * (nn_gpu.cu) -- nothing about the chunk's statistics ever reaches
 * the host to get there. The five inputs that ARE host knowledge stay host
 * knowledge: four per-candidate values arrive in `cand_desc`
 * (algo_id, quant, shuffle, error-bound encoding, matching
 * FeaturesTo8Input's order) and the chunk size rides in as a scalar argument,
 * exactly as upstream passes `input_size` to its kernel.
 */
__global__ void InferKernelDeviceStats(
    const NeuroPressGpuWeights *__restrict__ w,
    const int *__restrict__ action_ids,
    const ctp::DeviceFeatureStats *__restrict__ stats, float chunk_size_bytes,
    float error_bound, float *__restrict__ out_comp_time,
    float *__restrict__ out_decomp_time, float *__restrict__ out_ratio,
    float *__restrict__ out_psnr,
    /* Outputs 4-7. Null on the ranking path, which reads none of them, so the
       transforms are not evaluated there. Present so that a caller wanting the
       data-quality predictions can have them WITHOUT leaving the device: this
       is the path whose inputs are built in-kernel from AutoStatsGPU, and
       routing such a caller through the host-matrix entry point instead would
       stage the chunk's statistics through host memory to get numbers the GPU
       already had. */
    float *__restrict__ out_rmse = nullptr,
    float *__restrict__ out_max_error = nullptr,
    float *__restrict__ out_mae = nullptr,
    float *__restrict__ out_ssim = nullptr, float ratio_cap = 100.0f) {
  int cand = blockIdx.x;
  int t = threadIdx.x;

  __shared__ float s_x[kInputDim];
  __shared__ float s_h1[kHiddenDim], s_h2[kHiddenDim], s_h3[kHiddenDim],
      s_h4[kHiddenDim];
  __shared__ float s_y[kOutputDim];

  if (t < kInputDim) {
    // The same eight inputs, in the same order, built the same way upstream
    // builds them (nn_gpu.cu): the first three come from decoding the
    // action, not from anything the host assembled.
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
        // Lossless configs were TRAINED against a 1e-7 sentinel, not a raw 0
        // (neural_net/core/configs.py: `eb_val = eb if quant else 1e-7`), and
        // upstream re-applies it here rather than upstream of here:
        // `input_raw[3] = (quant == 0) ? 1e-7f : eb_enc` with the comment
        // "Inference must match -- do not pass raw 0.0 for lossless configs"
        // (nn_gpu.cu). Slot 1 is the quantize bit, so the kernel decides
        // this from the same descriptor the rest of the vector comes from.
        //
        // Applies to INFERENCE only. Both SGD paths feed the raw bound
        // upstream, which is why FeaturesTo8Input still takes a flag and why
        // the substitution lives here and not in it.
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
    float std_val = w->x_stds[t];
    if (std_val < 1e-8f) std_val = 1e-8f;
    s_x[t] = (raw - w->x_means[t]) / std_val;
  }
  __syncthreads();

  NeuroPressForwardShared(w, s_x, s_h1, s_h2, s_h3, s_h4, s_y, t, cand,
                          out_comp_time, out_decomp_time, out_ratio, ratio_cap,
                          out_psnr, out_rmse, out_max_error, out_mae, out_ssim);
}

/**
 * Widest candidate set the ranking warp can hold, and the reason it is 32:
 * that is upstream's NN_NUM_CONFIGS (8 algorithms x quantize x byte-shuffle)
 * and therefore the most the bridge can ever enumerate.
 */


/**
 * Cost model + ranking, on the GPU.
 *
 * Upstream never ranks on the host: its fused inference kernel computes the
 * cost, applies its masks and runs a 32-lane bitonic sort in the same kernel,
 * with thread 0 writing the winner.
 *
 * One warp, lane == candidate slot, which is why kMaxCandidates is 32: both
 * upstream's NN_NUM_CONFIGS and the widest set the bridge can enumerate. Lanes
 * past `n` take -infinity and sort to the end.
 *
 * The arithmetic is RankingWeights::Score transcribed, in double, deliberately:
 * this changes WHERE the ranking happens, not what it decides, so it must
 * reproduce the host result exactly rather than closely. Upstream computes the
 * same cost in float; that difference is unchanged from before.
 *
 * The two -INFINITY masks are applied here, where upstream applies them, rather
 * than by eliding candidates before ranking and filtering after. A masked
 * action still PARTICIPATES, so when every action is masked upstream still
 * returns one -- the lowest-indexed -- instead of returning nothing.
 */
__global__ void RankKernel(const float *__restrict__ ct_in,
                           const float *__restrict__ dt_in,
                           const float *__restrict__ ratio_in,
                           const float *__restrict__ psnr_in,
                           const int *__restrict__ action_ids, int n,
                           double data_size_bytes, double w_ct, double w_dt,
                           double w_io, double bw, double error_bound,
                           double min_psnr, double ratio_cap,
                           int *__restrict__ out_order,
                           double *__restrict__ out_scores) {
  const int tid = static_cast<int>(threadIdx.x);
  double score = -CUDART_INF;
  int idx = tid;
  // The tie key is the ACTION index, which is what upstream's network orders
  // by -- its lanes ARE actions, so its strict comparators resolve a tie to
  // the lowest action. Clio's lanes are candidate slots, and slot order only
  // equals action order when the caller enumerated it that way. Keying on the
  // action directly makes the rule hold regardless, instead of depending on
  // an enumeration convention two files away.
  // Composite so the key is ALWAYS unique: action index first (upstream's
  // rule), slot second. Action ids are unique across a well-formed candidate
  // set, but nothing in this kernel can enforce that -- a caller passing a set
  // that includes algorithms outside the trained eight gets colliding ids from
  // the fallback mapping, and a bitonic sort on a non-unique key stops being a
  // permutation, silently dropping and duplicating candidates. The slot
  // tiebreak costs nothing and makes that impossible.
  int key = (tid < n) ? (action_ids[tid] * kMaxCandidates + tid)
                      : ((kMaxCandidates + tid) * kMaxCandidates + tid);

  if (tid < n) {
    // predictor.h:213-229, clamp for clamp.
    const double ct = fmax(1.0, static_cast<double>(ct_in[tid]));
    const double dt_raw = static_cast<double>(dt_in[tid]);
    const double dt = (dt_raw > 0.0) ? fmax(1.0, dt_raw) : ct;
    const double ratio =
        fmax(0.1, fmin(ratio_cap, static_cast<double>(ratio_in[tid])));
    const double io = (ratio > 0.0) ? (data_size_bytes / (ratio * bw)) : 1e30;
    score = -(w_ct * ct + w_dt * dt + w_io * io);

    // nn_gpu.cu, in that order. cand_desc slot 1 is the quantize bit
    // (FeaturesTo8Input's input 1), so the kernel reads the same flag the
    // network was fed rather than being told separately.
    int algo, quant, shuffle;
    DecodeAction(action_ids[tid], &algo, &quant, &shuffle);
    const bool is_quant = (quant != 0);
    if (is_quant && error_bound <= 0.0) score = -CUDART_INF;
    if (min_psnr > 0.0 && static_cast<double>(psnr_in[tid]) < min_psnr) {
      score = -CUDART_INF;
    }
  }

  // Total order: score descending, then slot ascending. The second key is not
  // decoration -- it is what reproduces std::stable_sort's "first enumerated
  // wins" on the ties this cost model genuinely produces (ratio saturates at
  // the 100x cap, times floor at 1 ms). Upstream relies on its network never
  // swapping equal keys to the same end; making the tie an explicit part of
  // the comparator gets there without depending on that property.
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
    // The sorted score travels with the slot. Recomputing it on the host to
    // fill in RankedPrediction::score would put the cost model back on the CPU
    // for no reason -- the kernel has just computed it.
    out_scores[tid] = score;
  }
}

/**
 * Deferred, head-only SGD for the DEcOMPRESSION-time output, on the GPU.
 *
 * Port of nnBatchedDecompSGDKernel, launched the way upstream launches it: ONE
 * block of NN_HIDDEN_DIM threads, lane t owning w5 row 1's column t.
 *
 * Inputs 0-2 are rebuilt from the action here, as upstream rebuilds them, and
 * input 3 is the RAW bound -- the 1e-7 lossless sentinel is an inference-only
 * substitution and both SGD paths feed the raw value.
 *
 * The trunk W1-W4 is read-only: a decompression-time miss must not perturb the
 * representation the other three heads share.
 */
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
      for (int i = 0; i < kInputDim; ++i) {
        float sd = w->x_stds[i];
        if (sd < 1e-8f) sd = 1e-8f;
        s_x[i] = (raw[i] - w->x_means[i]) / sd;
      }
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

  // PER-THREAD g_norm, and deliberately so: upstream writes
  //   sqrtf(s_reduce[0] + (t == 0 ? acc_gb*acc_gb : 0.0f)) + 1e-8f
  // (:2519), so lane 0 normalizes by a norm that includes the bias term and
  // every other lane does not. That is upstream's behaviour, quirk and all --
  // the host port had to reproduce it with a pair of separately computed
  // norms, and here it simply falls out.
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

/**
 * Per-thread inference scratch, allocated once.
 *
 * NeuroPressGpuInferBatch below does five cudaMalloc and five cudaFree on
 * every call, i.e. per chunk. Upstream has none on that path: the equivalent
 * buffers are fields of the CompContext it acquired
 * (ctx->d_fused_infer_output, ctx->d_fused_top_actions, ctx->d_fused_costs).
 * Sized for the full 32-action space so the candidate set can never outgrow
 * it -- that is upstream's NN_NUM_CONFIGS and the hard ceiling on what the
 * bridge can enumerate.
 *
 * Leaked deliberately, for the reason given at Registry(): releasing device
 * memory from a static destructor races the CUDA runtime's teardown.
 */
struct InferScratch {
  // Sized for the ACTION SPACE. The device-stats path ranks in a single warp,
  // so its candidate set can never exceed NN_NUM_CONFIGS.
  int *d_actions = nullptr;  // [kMaxCandidates] upstream action indices

  // The candidate action list is uploaded only when it CHANGES, not per chunk.
  //
  // For the production caller it changes once: neuropress_bridge.cc resolves
  // algorithm availability into a function-local `static const std::set`, the
  // eight trained base_ids are a static set, and the quantize/shuffle expansion
  // is fixed order. Not even the error bound moves it -- that sets each
  // candidate's `error_bound` field, while the action index is
  // algo + 8*quantize + 16*shuffle and carries no bound. So the steady state is
  // zero host-to-device transfers on the decision path.
  //
  // Other callers may legitimately vary the list (the parity harnesses do), so
  // the installed copy is compared rather than assumed. At 32 ints the compare
  // is far cheaper than the ~2.8 us latency-bound copy it avoids, and a stale
  // upload would silently score the WRONG configurations while still returning
  // a plausible ranking.
  bool actions_installed = false;
  std::vector<int> installed_actions;

  // Sized for the BATCH, grown on demand. The host-matrix entry point has no
  // warp-wide step, and Rank() calls it with whatever candidate set the caller
  // built -- the model's own action space is 32, but a caller ranking the full
  // compressor registry passes far more. Capping these at 32 made that case
  // return an empty prediction set, which Rank() then turned into an empty
  // ranking: caught by ctp_compress_model, not by anything on the NeuroPress
  // path, because only the wider callers reach it.
  float *d_raw = nullptr;  // [cap][8]

  // EVERY value this path reads back lives in ONE allocation, so the whole
  // ranking result returns in ONE cudaMemcpyAsync. These copies are
  // latency-bound -- ~2.8 us apiece at 128-512 bytes -- so the COUNT is the
  // cost, not the volume. Upstream packs its own output the same way and for
  // the same reason.
  //
  // Layout, by BYTE offset, sized by cap:
  //     [        0, 8*cap )  scores  double[cap]   -- 8-aligned, so first
  //     [    8*cap, 12*cap)  order   int[cap]
  //     [   12*cap, 28*cap)  pred    float[4][cap] -- ct | dt | ratio | psnr
  // Predictions sit LAST so a caller that wants no ranking can fetch them as a
  // contiguous suffix, still in one copy.
  //
  // The pointers below are non-owning views, which keeps every kernel call
  // site unchanged.
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

  // Host landing buffer for that single copy, scattered to the caller's
  // separate arrays after the stream wait. Per-thread and grown with cap, so a
  // steady state does no allocation.
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
  // Views into d_out, in the layout documented on InferScratch. cudaMalloc
  // returns 256-byte-aligned memory and every offset below is a multiple of
  // its element size, so each view is naturally aligned.
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
  s.host_out.resize(PackedOutBytes(nn));
  s.cap = n;
  return true;
}

/**
 * Pull the four prediction arrays back in a single copy, wait once, then
 * scatter to the caller's separate arrays.
 *
 * The scatter is host-side memcpy of a few hundred bytes, which is free next to
 * the ~2.8 us floor of a device copy -- so trading four device copies for one
 * plus four memcpys is a straight win. Both inference entry points end in the
 * same "copy, sync, return" shape, so both use this.
 *
 * The stream wait matches runNNFusedInferenceCtx's single
 * cudaStreamSynchronize(stream) (nn_gpu.cu). Calling cudaDeviceSynchronize
 * here instead would stall every other worker's compression, not just this one.
 */
bool FetchPredictionsSync(InferScratch &s, int n, cudaStream_t st,
                          float *out_ct, float *out_dt, float *out_r,
                          float *out_p, int *out_order = nullptr,
                          double *out_scores = nullptr,
                          float *out_rmse = nullptr,
                          float *out_maxe = nullptr, float *out_mae = nullptr,
                          float *out_ssim = nullptr) {
  // STRIDES ARE THE ALLOCATION'S, NOT THE CALL'S.
  //
  // EnsureInferCapacity lays d_out out by `cap`, and cap can exceed this
  // call's n (it only ever grows, and the host-matrix path ranks far more
  // candidates than the 32-wide device path). Computing these offsets from n
  // instead put every region in the wrong place whenever n < cap: ct and dt
  // then read from overlapping addresses, which is precisely the aliasing the
  // cost model must not see -- it would rank on a duplicated term. Caught by
  // test_neuropress_bridge's saw_distinct_decomp_time assertion.
  const size_t cap = static_cast<size_t>(s.cap);
  const size_t nn = static_cast<size_t>(n);
  const size_t stride = sizeof(float) * cap;   // between prediction arrays
  const size_t want = sizeof(float) * nn;      // what the caller asked for
  const size_t pred_off = (sizeof(double) + sizeof(int)) * cap;

  // Whether the ranking came back decides only WHERE the single copy starts.
  // Predictions are the last region, so "predictions only" is a contiguous
  // suffix and stays one transfer either way. The copy spans the whole
  // capacity because the regions are cap-strided; it is still ONE transfer,
  // which is the property being bought here.
  const bool want_rank = (out_order != nullptr);
  // Outputs 4-7 are the tail of the predictions region, so a caller that does
  // not want them simply stops the copy earlier. This is what keeps widening
  // the layout free for the ranking path: same bytes on the wire as before.
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
  const unsigned char *p = base + pred_off;
  std::memcpy(out_ct, p, want);
  std::memcpy(out_dt, p + stride, want);
  std::memcpy(out_r, p + 2 * stride, want);
  std::memcpy(out_p, p + 3 * stride, want);
  if (out_rmse != nullptr) std::memcpy(out_rmse, p + 4 * stride, want);
  if (out_maxe != nullptr) std::memcpy(out_maxe, p + 5 * stride, want);
  if (out_mae != nullptr) std::memcpy(out_mae, p + 6 * stride, want);
  if (out_ssim != nullptr) std::memcpy(out_ssim, p + 7 * stride, want);
  return true;
}

InferScratch &Infer() {
  static thread_local InferScratch *s = [] {
    auto *p = new InferScratch();
    // d_order and d_scores are no longer allocated here: they are views into
    // the one packed d_out block that EnsureInferCapacity owns.
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

  /* Upstream's runBatchedDecompSGD, step for step. Note it does NOT drop its
     synchronize the way runNNSGDCtx does -- it copies its SGDOutput back and
     waits -- so this keeps the wait too. Matching upstream means matching the
     asymmetry, not tidying it away. What changes is the stream (the shared SGD
     stream, so the event orders this against inference as well) and the buffer
     (persistent, not allocated and freed per call). */
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

bool NeuroPressGpuInferBatchDeviceStats(
    NeuroPressGpuWeights *w, const void *device_stats,
    const int *action_ids, int num_candidates, float chunk_size_bytes,
    float error_bound, void *stream, float *out_comp_time_ms,
    float *out_decomp_time_ms, float *out_ratio, float *out_psnr_db,
    const GpuRankParams *rank, int *out_order, double *out_scores,
    float *out_rmse, float *out_max_error, float *out_mae, float *out_ssim) {
  if (!w || !device_stats || !action_ids || num_candidates <= 0 ||
      num_candidates > kMaxCandidates) {
    return false;
  }
  InferScratch &s = Infer();
  if (!s.ok) return false;

  cudaStream_t st = static_cast<cudaStream_t>(stream);
  const size_t out_bytes = sizeof(float) * static_cast<size_t>(num_candidates);
  const size_t act_bytes = sizeof(int) * static_cast<size_t>(num_candidates);

  // Everything below is enqueued on the SAME stream the statistics were
  // computed on, so the kernels chain on the GPU with no host involvement --
  // this is the property that was missing. The descriptor upload is host
  // knowledge that upstream does not need at all: its lanes ARE the actions,
  // so it decodes them from the thread index. Clio's candidate set can be a
  // subset -- an algorithm this build cannot construct is never enumerated --
  // so the action indices have to come from somewhere. At 128 bytes, async,
  // and never waited on, this does not reintroduce a synchronization point;
  // what mattered was never the byte count but that the old path had to STOP
  // and wait twice.
  // Upload only when the list actually differs from what is on the device --
  // see InferScratch::actions_installed. For the production caller that is once
  // per thread, leaving no host-to-device transfer on the decision path.
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
      /* GPU-level barrier before reading the weights: wait for the last SGD on
       its stream. Upstream does exactly this
       (cudaStreamWaitEvent(stream, g_sgd_done, 0), nn_gpu.cu) and it is what
       lets the SGD path be fire-and-forget -- the ordering is enforced on the
       device, so neither host thread blocks for it. */
    SgdWaitIfEverFired(st);
  InferKernelDeviceStats<<<num_candidates, kHiddenDim, 0, st>>>(
        w, s.d_actions,
        static_cast<const ctp::DeviceFeatureStats *>(device_stats),
        chunk_size_bytes, error_bound, s.d_ct, s.d_dt, s.d_r, s.d_p,
        /* Only ask the kernel for outputs 4-7 when the caller wants them; the
           transforms are skipped on a null pointer, so ranking pays nothing. */
        out_rmse ? s.d_rmse : nullptr, out_max_error ? s.d_maxe : nullptr,
        out_mae ? s.d_mae : nullptr, out_ssim ? s.d_ssim : nullptr,
        /* One cap for BOTH halves. Clamping the prediction at 100 while the
           cost model scores the measurement at something higher puts the two
           on different scales and inflates the MAPE that gates SGD and
           exploration, so the ranking parameters carry the value the caller
           chose and the forward pass uses the same one. */
        rank != nullptr ? static_cast<float>(rank->ratio_cap) : 100.0f);
    ok = cudaGetLastError() == cudaSuccess;
  }
  // Cost model and ordering, still on the device and still on this stream --
  // upstream does both inside its inference kernel (nn_gpu.cu,
  // :499-532), so leaving them to the host was the last place the decision
  // came back across. One warp is enough: the candidate set can never exceed
  // NN_NUM_CONFIGS.
  if (ok && rank != nullptr && out_order != nullptr) {
    RankKernel<<<1, kMaxCandidates, 0, st>>>(
        s.d_ct, s.d_dt, s.d_r, s.d_p, s.d_actions, num_candidates,
        rank->data_size_bytes, rank->w_compress_time, rank->w_decompress_time,
        rank->w_io, rank->bandwidth_bytes_per_ms, rank->error_bound,
        rank->min_psnr, rank->ratio_cap, s.d_order, s.d_scores);
    ok = cudaGetLastError() == cudaSuccess;
    // The ranking is NOT fetched here: it shares one allocation with the
    // predictions, so FetchPredictionsSync below brings back scores, order and
    // all four prediction arrays in a single transfer.
    ranked = ok;
  }

  if (ok) {
    ok = FetchPredictionsSync(s, num_candidates, st, out_comp_time_ms,
                              out_decomp_time_ms, out_ratio, out_psnr_db,
                              ranked ? out_order : nullptr, out_scores,
                              out_rmse, out_max_error, out_mae, out_ssim);
  }
  return ok;
}

/**
 * Persistent scratch for the all-outputs path, mirroring InferScratch. Held
 * per thread and grown rather than reallocated, so a repeated caller does no
 * allocation in the steady state.
 */
struct FullScratch {
  float *d_raw = nullptr;
  float *d_out[8] = {nullptr};
  int cap = 0;
};

FullScratch &Full() {
  static thread_local FullScratch s;
  return s;
}

bool EnsureFullCapacity(FullScratch &s, int n) {
  if (n <= s.cap) return true;
  cudaFree(s.d_raw);
  for (int i = 0; i < 8; ++i) cudaFree(s.d_out[i]);
  s.d_raw = nullptr;
  for (int i = 0; i < 8; ++i) s.d_out[i] = nullptr;
  s.cap = 0;

  const size_t nn = static_cast<size_t>(n);
  if (cudaMalloc(&s.d_raw, sizeof(float) * nn * kInputDim) != cudaSuccess) {
    return false;
  }
  for (int i = 0; i < 8; ++i) {
    if (cudaMalloc(&s.d_out[i], sizeof(float) * nn) != cudaSuccess) return false;
  }
  s.cap = n;
  return true;
}

bool NeuroPressGpuInferBatchFull(NeuroPressGpuWeights *w,
                                 const float *raw_inputs, int num_candidates,
                                 float *out_comp_time_ms,
                                 float *out_decomp_time_ms, float *out_ratio,
                                 float *out_psnr_db, float *out_rmse,
                                 float *out_max_error, float *out_mae,
                                 float *out_ssim) {
  if (!w || !raw_inputs || num_candidates <= 0) return false;

  // Same discipline as the ranking path: the per-thread persistent stream and
  // a scratch buffer that is grown, not reallocated. An earlier version of
  // this function used nine cudaMalloc/cudaFree per call and a
  // cudaDeviceSynchronize -- which stalls every other worker's kernels, not
  // just this call's. That pattern was removed from the ranking path for
  // exactly that reason (see NeuroPressGpuInferBatch) and must not come back
  // in through a side entrance.
  const size_t n = static_cast<size_t>(num_candidates);
  const size_t out_bytes = sizeof(float) * n;
  const size_t in_bytes = out_bytes * kInputDim;

  FullScratch &s = Full();
  if (!EnsureFullCapacity(s, num_candidates)) return false;
  cudaStream_t st = static_cast<cudaStream_t>(ctp::DeviceStatsStream());

  bool ok = cudaMemcpyAsync(s.d_raw, raw_inputs, in_bytes,
                            cudaMemcpyHostToDevice, st) == cudaSuccess;
  if (ok) {
    /* GPU-level barrier before reading the weights: wait for the last SGD on
       its stream. Upstream does exactly this
       (cudaStreamWaitEvent(stream, g_sgd_done, 0), nn_gpu.cu) and it is what
       lets the SGD path be fire-and-forget -- the ordering is enforced on the
       device, so neither host thread blocks for it. */
    SgdWaitIfEverFired(st);
    InferKernelFull<<<num_candidates, kHiddenDim, 0, st>>>(
        w, s.d_raw, s.d_out[0], s.d_out[1], s.d_out[2], s.d_out[3],
        s.d_out[4], s.d_out[5], s.d_out[6], s.d_out[7], 100.0f);
    ok = cudaGetLastError() == cudaSuccess;
  }

  float *host[8] = {out_comp_time_ms, out_decomp_time_ms, out_ratio,
                    out_psnr_db,      out_rmse,           out_max_error,
                    out_mae,          out_ssim};
  for (int i = 0; ok && i < 8; ++i) {
    if (host[i] == nullptr) continue;
    ok = cudaMemcpyAsync(host[i], s.d_out[i], out_bytes,
                         cudaMemcpyDeviceToHost, st) == cudaSuccess;
  }
  // One wait on OUR stream, not the device.
  if (ok) ok = cudaStreamSynchronize(st) == cudaSuccess;
  return ok;
}

bool NeuroPressGpuInferBatch(NeuroPressGpuWeights *w, const float *raw_inputs,
                             int num_candidates, float *out_comp_time_ms,
                             float *out_decomp_time_ms, float *out_ratio,
                             float *out_psnr_db) {
  if (!w || num_candidates <= 0) return false;
  InferScratch &s = Infer();
  if (!s.ok || !EnsureInferCapacity(s, num_candidates)) return false;

  // Same per-thread stream and same persistent scratch as the device-stats
  // entry point. This path takes a host-built input matrix, so it is not the
  // one upstream corresponds to -- but it used five cudaMalloc/cudaFree per
  // call and a cudaDeviceSynchronize, and the latter stalls every other
  // worker's kernels, not just this call's. Nothing about being the
  // host-buffer path makes that desirable.
  cudaStream_t st = static_cast<cudaStream_t>(ctp::DeviceStatsStream());
  const size_t out_bytes = sizeof(float) * static_cast<size_t>(num_candidates);
  const size_t in_bytes = out_bytes * kInputDim;

  // Every step is checked: a silent failure here leaves the caller's output
  // vectors zero-filled, which survives the policy clamps as a complete and
  // plausible-looking ranking (ratio 0.1, 1 ms, everywhere) rather than an
  // error. NeuroPress checks cudaGetLastError and returns -1 (nn_gpu.cu:
  // 1944-1971); this is the equivalent.
  bool ok = cudaMemcpyAsync(s.d_raw, raw_inputs, in_bytes,
                            cudaMemcpyHostToDevice, st) == cudaSuccess;
  if (ok) {
    /* GPU-level barrier before reading the weights: wait for the last SGD on
       its stream. Upstream does exactly this
       (cudaStreamWaitEvent(stream, g_sgd_done, 0), nn_gpu.cu) and it is what
       lets the SGD path be fire-and-forget -- the ordering is enforced on the
       device, so neither host thread blocks for it. */
    SgdWaitIfEverFired(st);
    /* This entry point takes no ranking parameters, so the cap stays at
       upstream's literal 100. Callers that want a different ceiling go
       through the device-stats path, which carries GpuRankParams. */
    InferKernel<<<num_candidates, kHiddenDim, 0, st>>>(
        w, s.d_raw, s.d_ct, s.d_dt, s.d_r, s.d_p, 100.0f);
    ok = cudaGetLastError() == cudaSuccess;
  }
  if (ok) {
    ok = FetchPredictionsSync(s, num_candidates, st, out_comp_time_ms,
                              out_decomp_time_ms, out_ratio, out_psnr_db);
  }
  return ok;
}

// ============================================================================
// SGD: single block <<<1, kHiddenDim>>>, mirrors nnSGDKernel's algorithm.
// ============================================================================
__device__ __forceinline__ void ForwardOneLayer(
    const float *__restrict__ w_layer, const float *__restrict__ b_layer,
    const float *in, int fan_in, int t, float &z_out, float &h_out) {
  float sum = b_layer[t];
  for (int i = 0; i < fan_in; ++i) sum += w_layer[t * fan_in + i] * in[i];
  z_out = sum;
  h_out = fmaxf(0.0f, sum);
}

/**
 * Phases 1 and 1.5 of the ported rule: forward pass, per-head error in
 * standardized log1p space, and Kendall uncertainty weighting.
 *
 * Shared verbatim by both update laws below -- the adaptive rule changes only
 * what happens to the gradient afterwards, and two copies of the error math
 * would be free to drift apart.
 *
 * __forceinline__ matches NeuroPressForwardShared above and keeps this out of
 * an ABI call inside a hot single-block kernel. Measured: it does not change
 * any result -- both forms pass the parity suite identically.
 */
__device__ __forceinline__ void SgdForwardAndErrors(
    NeuroPressGpuWeights *w,
    const NeuroPressGpuSGDSample *__restrict__ samples, int num_samples, int t,
    const ctp::DeviceFeatureStats *__restrict__ device_stats) {
  // ---- Phase 1: per-sample forward pass + target/error computation ----
  for (int si = 0; si < num_samples; ++si) {
    if (t < kInputDim) {
      // Inputs 5-7 from DEVICE memory, as nnSGDKernel reads d_stats
      // in-kernel (nn_gpu.cu). Null keeps the host-matrix behaviour.
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
      float std_val = w->x_stds[t];
      if (std_val < 1e-8f) std_val = 1e-8f;
      w->act_x[si][t] =
          (raw - w->x_means[t]) / std_val;
    }
    __syncthreads();

    float z, h;
    ForwardOneLayer(&w->params[kOffW1], &w->params[kOffB1], w->act_x[si],
                    kInputDim, t, z, h);
    w->act_z1[si][t] = z;
    w->act_h1[si][t] = h;
    __syncthreads();

    ForwardOneLayer(&w->params[kOffW2], &w->params[kOffB2], w->act_h1[si],
                    kHiddenDim, t, z, h);
    w->act_z2[si][t] = z;
    w->act_h2[si][t] = h;
    __syncthreads();

    ForwardOneLayer(&w->params[kOffW3], &w->params[kOffB3], w->act_h2[si],
                    kHiddenDim, t, z, h);
    w->act_z3[si][t] = z;
    w->act_h3[si][t] = h;
    __syncthreads();

    ForwardOneLayer(&w->params[kOffW4], &w->params[kOffB4], w->act_h3[si],
                    kHiddenDim, t, z, h);
    w->act_z4[si][t] = z;
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

      float clamped_ratio = fmaxf(0.5f, fminf(s.actual_ratio, 10000.0f));
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

__global__ void SGDKernel(NeuroPressGpuWeights *w,
                          const NeuroPressGpuSGDSample *__restrict__ samples,
                          int num_samples, float learning_rate,
                          float *__restrict__ ema, bool *out_applied,
                          const ctp::DeviceFeatureStats *__restrict__
                              device_stats) {
  int t = threadIdx.x;  // 0..63
  __shared__ float s_reduce[kHiddenDim];

  SgdForwardAndErrors(w, samples, num_samples, t, device_stats);

  // ---- Phase 2: per-output backward passes with PCGrad-lite ----
  constexpr float kGradClipThreshold = 0.1f;
  constexpr float kPcgradCosThresh = -0.1f;

  for (int i = t; i < kParamCount; i += kHiddenDim) w->combined[i] = 0.0f;
  __syncthreads();

  for (int target_out = 0; target_out < kOutputDim; ++target_out) {
    for (int i = t; i < kParamCount; i += kHiddenDim) w->out_grad[i] = 0.0f;
    __syncthreads();

    for (int si = 0; si < num_samples; ++si) {
      // Step 1: L4 backward delta for ALL outputs, normalized to unit vectors.
      for (int o = 0; o < kOutputDim; ++o) {
        float es = w->d5_clamped[si][o];
        float dh4_t = w->params[kOffW5 + o * kHiddenDim + t] * es;
        float v = (w->act_z4[si][t] > 0.0f) ? dh4_t : 0.0f;
        w->dz4_all[o][t] = v;
      }
      __syncthreads();
      for (int o = 0; o < kOutputDim; ++o) {
        float local = w->dz4_all[o][t] * w->dz4_all[o][t];
        s_reduce[t] = local;
        __syncthreads();
        for (int s = kHiddenDim / 2; s > 0; s >>= 1) {
          if (t < s) s_reduce[t] += s_reduce[t + s];
          __syncthreads();
        }
        float norm = sqrtf(s_reduce[0]) + 1e-6f;
        w->dz4_all[o][t] /= norm;
        __syncthreads();
      }

      // Step 2: PCGrad projection for target_out against every other output.
      float my_dz4 = w->dz4_all[target_out][t];
      for (int j = 0; j < kOutputDim; ++j) {
        if (j == target_out) continue;
        float local_dot = my_dz4 * w->dz4_all[j][t];
        s_reduce[t] = local_dot;
        __syncthreads();
        for (int s = kHiddenDim / 2; s > 0; s >>= 1) {
          if (t < s) s_reduce[t] += s_reduce[t + s];
          __syncthreads();
        }
        float cos_ij = s_reduce[0];
        if (cos_ij < kPcgradCosThresh) my_dz4 -= cos_ij * w->dz4_all[j][t];
        __syncthreads();
      }
      float err_mag = fabsf(w->d5_clamped[si][target_out]);
      float dz4 = my_dz4 * err_mag;

      // Step 3: W5/b5 gradient uses the ORIGINAL (unprojected) error.
      float error_signal = w->d5_clamped[si][target_out];
      w->out_grad[kOffW5 + target_out * kHiddenDim + t] +=
          error_signal * w->act_h4[si][t];
      if (t == 0) w->out_grad[kOffB5 + target_out] += error_signal;

      // Step 4/4b: L4 gradient, backward L4->L3 (using PROJECTED dz4).
      for (int i = 0; i < kHiddenDim; ++i)
        w->out_grad[kOffW4 + t * kHiddenDim + i] += dz4 * w->act_h3[si][i];
      w->out_grad[kOffB4 + t] += dz4;

      // Broadcast dz4[0..63] through shared memory so every thread can sum
      // weights_[*, t] * dz4[*] for the L4->L3 backward step.
      __shared__ float s_dz4[kHiddenDim];
      s_dz4[t] = dz4;
      __syncthreads();
      float dh3_t = 0.0f;
      for (int j = 0; j < kHiddenDim; ++j)
        dh3_t += w->params[kOffW4 + j * kHiddenDim + t] * s_dz4[j];
      float dz3 = (w->act_z3[si][t] > 0.0f) ? dh3_t : 0.0f;

      // Step 5/5b: L3 gradient, backward L3->L2.
      for (int i = 0; i < kHiddenDim; ++i)
        w->out_grad[kOffW3 + t * kHiddenDim + i] += dz3 * w->act_h2[si][i];
      w->out_grad[kOffB3 + t] += dz3;
      __shared__ float s_dz3[kHiddenDim];
      s_dz3[t] = dz3;
      __syncthreads();
      float dh2_t = 0.0f;
      for (int j = 0; j < kHiddenDim; ++j)
        dh2_t += w->params[kOffW3 + j * kHiddenDim + t] * s_dz3[j];
      float dz2 = (w->act_z2[si][t] > 0.0f) ? dh2_t : 0.0f;

      // Step 6/6b: L2 gradient, backward L2->L1, then L1 gradient.
      for (int i = 0; i < kHiddenDim; ++i)
        w->out_grad[kOffW2 + t * kHiddenDim + i] += dz2 * w->act_h1[si][i];
      w->out_grad[kOffB2 + t] += dz2;
      __shared__ float s_dz2[kHiddenDim];
      s_dz2[t] = dz2;
      __syncthreads();
      float dh1_t = 0.0f;
      for (int j = 0; j < kHiddenDim; ++j)
        dh1_t += w->params[kOffW2 + j * kHiddenDim + t] * s_dz2[j];
      float dz1 = (w->act_z1[si][t] > 0.0f) ? dh1_t : 0.0f;

      for (int i = 0; i < kInputDim; ++i)
        w->out_grad[kOffW1 + t * kInputDim + i] += dz1 * w->act_x[si][i];
      w->out_grad[kOffB1 + t] += dz1;
      __syncthreads();
    }  // per-sample

    // Average over samples, compute this output's gradient norm (only the
    // params it actually touches: shared L1-L4 + its own W5/b5 row).
    float inv_n = 1.0f / static_cast<float>(num_samples);
    float local_norm_sq = 0.0f;
    for (int i = 0; i < kInputDim; ++i) {
      int idx = kOffW1 + t * kInputDim + i;
      w->out_grad[idx] *= inv_n;
      local_norm_sq += w->out_grad[idx] * w->out_grad[idx];
    }
    { int idx = kOffB1 + t; w->out_grad[idx] *= inv_n; local_norm_sq += w->out_grad[idx] * w->out_grad[idx]; }
    for (int i = 0; i < kHiddenDim; ++i) {
      int idx = kOffW2 + t * kHiddenDim + i;
      w->out_grad[idx] *= inv_n;
      local_norm_sq += w->out_grad[idx] * w->out_grad[idx];
    }
    { int idx = kOffB2 + t; w->out_grad[idx] *= inv_n; local_norm_sq += w->out_grad[idx] * w->out_grad[idx]; }
    for (int i = 0; i < kHiddenDim; ++i) {
      int idx = kOffW3 + t * kHiddenDim + i;
      w->out_grad[idx] *= inv_n;
      local_norm_sq += w->out_grad[idx] * w->out_grad[idx];
    }
    { int idx = kOffB3 + t; w->out_grad[idx] *= inv_n; local_norm_sq += w->out_grad[idx] * w->out_grad[idx]; }
    for (int i = 0; i < kHiddenDim; ++i) {
      int idx = kOffW4 + t * kHiddenDim + i;
      w->out_grad[idx] *= inv_n;
      local_norm_sq += w->out_grad[idx] * w->out_grad[idx];
    }
    { int idx = kOffB4 + t; w->out_grad[idx] *= inv_n; local_norm_sq += w->out_grad[idx] * w->out_grad[idx]; }
    {
      int idx = kOffW5 + target_out * kHiddenDim + t;
      w->out_grad[idx] *= inv_n;
      local_norm_sq += w->out_grad[idx] * w->out_grad[idx];
    }
    if (t == target_out) {
      int idx = kOffB5 + target_out;
      w->out_grad[idx] *= inv_n;
      local_norm_sq += w->out_grad[idx] * w->out_grad[idx];
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

    for (int i = 0; i < kInputDim; ++i)
      w->combined[kOffW1 + t * kInputDim + i] += lr_out * w->out_grad[kOffW1 + t * kInputDim + i];
    w->combined[kOffB1 + t] += lr_out * w->out_grad[kOffB1 + t];
    for (int i = 0; i < kHiddenDim; ++i)
      w->combined[kOffW2 + t * kHiddenDim + i] += lr_out * w->out_grad[kOffW2 + t * kHiddenDim + i];
    w->combined[kOffB2 + t] += lr_out * w->out_grad[kOffB2 + t];
    for (int i = 0; i < kHiddenDim; ++i)
      w->combined[kOffW3 + t * kHiddenDim + i] += lr_out * w->out_grad[kOffW3 + t * kHiddenDim + i];
    w->combined[kOffB3 + t] += lr_out * w->out_grad[kOffB3 + t];
    for (int i = 0; i < kHiddenDim; ++i)
      w->combined[kOffW4 + t * kHiddenDim + i] += lr_out * w->out_grad[kOffW4 + t * kHiddenDim + i];
    w->combined[kOffB4 + t] += lr_out * w->out_grad[kOffB4 + t];
    {
      int idx = kOffW5 + target_out * kHiddenDim + t;
      w->combined[idx] += lr_out * w->out_grad[idx];
    }
    if (t == target_out) {
      int idx = kOffB5 + target_out;
      w->combined[idx] += lr_out * w->out_grad[idx];
    }
    __syncthreads();
  }  // per target_out

  // ---- Trust-region step, anti-flip damping, EMA smoothing, weight update ----
  constexpr float kEmaDecay = 0.85f;
  constexpr float kTrustK = 0.08f;
  constexpr float kMaxStep = 0.02f;
  constexpr float kMinStep = 1e-4f;
  constexpr float kAntiFlipDamp = 0.5f;
  constexpr float kWClamp = 10.0f;

  // Gradient norm (sum over all params, block-wide reduction via strided
  // per-thread partial sums since kParamCount > kHiddenDim).
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
    // TRUNK ONLY (params before the W5 block). NeuroPress accumulates this
    // dot over DW1..DB4 and stops -- nn_gpu.cu has no DW5/DB5
    // term, and the omission is deliberate: the trust-region norm 60 lines
    // earlier (:1319-1326) explicitly DOES include both. Including them
    // biases the dot positive, because the W5 gradient (error * h4, with h4
    // a non-negative ReLU output) is the largest and most sign-stable block
    // and is not projected by PCGrad -- so the step got damped less often
    // than upstream, precisely in the oscillation regime the damping exists
    // to suppress.
    float local_dot = 0.0f;
    for (int i = t; i < kOffW5; i += kHiddenDim) local_dot += w->combined[i] * ema[i];
    s_reduce[t] = local_dot;
    __syncthreads();
    for (int s = kHiddenDim / 2; s > 0; s >>= 1) {
      if (t < s) s_reduce[t] += s_reduce[t + s];
      __syncthreads();
    }
    // s_reduce[0] holds the block-wide dot product after the reduction
    // above; every thread reads the same value, so `step` (a per-thread
    // local) stays consistent across the whole block.
    if (s_reduce[0] < 0.0f) step *= kAntiFlipDamp;
  }

  for (int i = t; i < kParamCount; i += kHiddenDim)
    if (!isfinite(w->combined[i])) w->combined[i] = 0.0f;
  __syncthreads();

  float ema_new = 1.0f - kEmaDecay;
  for (int i = t; i < kParamCount; i += kHiddenDim)
    ema[i] = kEmaDecay * ema[i] + ema_new * w->combined[i];
  __syncthreads();

  if (t == 0) w->sgd_call_count += 1;

  bool finite_ok = isfinite(step) && isfinite(g_norm);
  if (finite_ok) {
    // Output 1 (decompression time) is owned by the deferred head-only pass
    // (NeuroPressNNPredictor::TrainDecompHead), fed by real measured times a
    // later read supplies -- so this pass must not write its head weights.
    // Mirrors nnSGDKernel's `if (out == 1) continue;` / `t != 1`
    // (nn_gpu.cu). Output 1's error still reaches the shared trunk, exactly
    // as upstream; only w5 row 1 and b5[1] are withheld.
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

/**
 * The adaptive update law (CLIO_NEUROPRESS_SGD_RULE=adaptive).
 *
 * Same errors as the kernel above, then TRUE backprop of their sum: no
 * per-output unit-normalisation, no per-output clip-renormalise, no PCGrad and
 * no global unit-normalisation, so the learning rate survives to the weights.
 */
__global__ void AdaptiveSGDKernel(
    NeuroPressGpuWeights *w,
    const NeuroPressGpuSGDSample *__restrict__ samples, int num_samples,
    float learning_rate, float *__restrict__ ema, bool *out_applied,
    const ctp::DeviceFeatureStats *__restrict__ device_stats, AdaptOptions o) {
  const int t = threadIdx.x;  // 0..63
  __shared__ float s_reduce[kHiddenDim];
  __shared__ float s_dz4[kHiddenDim], s_dz3[kHiddenDim], s_dz2[kHiddenDim];

  SgdForwardAndErrors(w, samples, num_samples, t, device_stats);

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
    const float dz4 = (w->act_z4[si][t] > 0.0f) ? dh4 : 0.0f;

    for (int i = 0; i < kHiddenDim; ++i)
      w->combined[kOffW4 + t * kHiddenDim + i] += dz4 * w->act_h3[si][i];
    w->combined[kOffB4 + t] += dz4;
    s_dz4[t] = dz4;
    __syncthreads();

    float dh3 = 0.0f;
    for (int j = 0; j < kHiddenDim; ++j)
      dh3 += w->params[kOffW4 + j * kHiddenDim + t] * s_dz4[j];
    const float dz3 = (w->act_z3[si][t] > 0.0f) ? dh3 : 0.0f;
    for (int i = 0; i < kHiddenDim; ++i)
      w->combined[kOffW3 + t * kHiddenDim + i] += dz3 * w->act_h2[si][i];
    w->combined[kOffB3 + t] += dz3;
    s_dz3[t] = dz3;
    __syncthreads();

    float dh2 = 0.0f;
    for (int j = 0; j < kHiddenDim; ++j)
      dh2 += w->params[kOffW3 + j * kHiddenDim + t] * s_dz3[j];
    const float dz2 = (w->act_z2[si][t] > 0.0f) ? dh2 : 0.0f;
    for (int i = 0; i < kHiddenDim; ++i)
      w->combined[kOffW2 + t * kHiddenDim + i] += dz2 * w->act_h1[si][i];
    w->combined[kOffB2 + t] += dz2;
    s_dz2[t] = dz2;
    __syncthreads();

    float dh1 = 0.0f;
    for (int j = 0; j < kHiddenDim; ++j)
      dh1 += w->params[kOffW2 + j * kHiddenDim + t] * s_dz2[j];
    const float dz1 = (w->act_z1[si][t] > 0.0f) ? dh1 : 0.0f;
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

  // Output 1's head row/bias stays owned by the deferred decomp pass, exactly
  // as the ported kernel withholds it.
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
    w->out_grad[i] = d;
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
      const float pv = w->params[i] - scale * w->out_grad[i];
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

  /* Choreography matched to upstream's runNNSGDCtx (nn_gpu.cu) step for step:
   *
   *   async H->D of the samples on the SGD stream, into a buffer that already
   *   exists; launch on that stream; record the completion event; return.
   *
   * FIRE AND FORGET. There is deliberately no read-back and no synchronize.
   * Upstream removed exactly that and left the measurement in the file:
   *
   *   "P6: fire-and-forget -- no D->H readback or stream sync needed.
   *    SGDOutput ... is dead code: both callers pass &gn/&gc/&gs but never
   *    read them after the call. Removing the sync drops g_sgd_mutex hold
   *    time from 0.1-0.5ms to ~10us, eliminating serialization across
   *    concurrent workers."
   *
   * Clio's readback had the same shape and the same uselessness: a one-byte
   * `applied` flag whose only consumer was an HLOG(kDebug) field, paid for
   * with a full host synchronize on every chunk the model learned from.
   *
   * The consequence is that "did the update apply" is no longer known when
   * this returns, which is precisely upstream's contract -- it returns 0 for
   * "launched", not for "applied". The kernel still computes the flag into
   * the device buffer; nothing reads it, and nothing waits for it. */
  SgdSync &g = Sgd();
  if (!g.ok) return false;

  const size_t bytes =
      sizeof(NeuroPressGpuSGDSample) * static_cast<size_t>(num_samples);
  SgdScratch &sc = SgdSamples();
  if (!EnsureSgdSamples(sc, bytes)) return false;

  float *ema = EmaBuffer();
  if (!ema) return false;

  /* The flag the kernel writes. Persistent and never read: kept because the
     kernel's signature takes it, and dropping the parameter would diverge
     from upstream's nnSGDKernel, which also writes an output nobody reads. */
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

  /* One `if` on the parsed options: `upstream` (the default) reaches exactly
     the kernel it always did, with the same arguments. */
  const AdaptOptions &opt = Opts();
  const float lr = (opt.lr > 0.0f) ? opt.lr : learning_rate;
  if (opt.adaptive) {
    AdaptiveSGDKernel<<<1, kHiddenDim, 0, g.stream>>>(
        w, static_cast<const NeuroPressGpuSGDSample *>(sc.d_samples),
        num_samples, lr, ema, d_applied, d_stats, opt);
  } else {
    SGDKernel<<<1, kHiddenDim, 0, g.stream>>>(
        w, static_cast<const NeuroPressGpuSGDSample *>(sc.d_samples),
        num_samples, lr, ema, d_applied, d_stats);
  }
  if (cudaGetLastError() != cudaSuccess) return false;

  /* Only set the flag if the record succeeded, so inference never waits on an
     unrecorded event -- upstream's "W2" note makes the same point. */
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
