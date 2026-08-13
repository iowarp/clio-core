/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file neuropress_flow_parity.cu
 * @brief Differential test: NeuroPress's per-flow SGD state vs Clio's.
 *
 * NeuroPress splits its online-learning state in two. The weights and
 * sgd_call_count are GLOBAL (nn_gpu.cu:64, :1413), but the EMA gradient the
 * weight update actually consumes -- `w -= step * EMA[...]`
 * (nn_gpu.cu:1523-1528) -- lives PER CompContext
 * (internal.hpp:50, gpucompress_pool.cpp:121-123). Every production SGD call
 * goes through runNNSGDCtx and therefore through its own context's buffer
 * (gpucompress_compress.cpp:721, :1023; H5VLgpucompress.cu:1719, :1753); the
 * context-free runNNSGD, which the other two parity harnesses drive, uses a
 * single file-scope buffer instead (nn_gpu.cu:1590) and so cannot see this
 * split at all.
 *
 * Clio has no CompContext at this layer, so the equivalent flow scope is the
 * worker thread, and the port keeps its EMA in a thread_local buffer
 * (neuropress_nn_gpu_kernels.cu's EmaBuffer()). That correspondence was the
 * one piece of the port argued structurally rather than measured -- which is
 * the same footing the 1e-7-sentinel bug sat on before a differential run
 * found it. So this harness measures it: two upstream CompContexts against
 * two Clio worker threads, driven step-for-step through the same interleaved
 * sequence, weights diffed after every step.
 *
 * Three phases, each checking a distinct property:
 *
 *  1. TWO FLOWS. Flow 0 reports far-over outcomes, flow 1 far-under, and the
 *     two alternate. Each flow's own gradient keeps a stable sign while the
 *     interleaved sequence flips every step, so a per-flow EMA and a shared
 *     one take visibly different trajectories -- and the anti-flip damping
 *     (which reads the EMA) is live throughout.
 *
 *  2. RELOAD. Reloading the model must zero EVERY live flow's gradient
 *     history, not just the reloading thread's (upstream:
 *     resetAllSGDEMABuffers, nn_gpu.cu:1808; Clio: ResetAllEmaBuffers via the
 *     registry). Both reloads are issued from the main thread, which is
 *     neither flow, so a reset that only reached the caller shows up here.
 *
 *  3. ONE FLOW. The same step sequence run through a single flow. This is
 *     the control: it must match native's single-context run, and it must
 *     NOT match phase 1, since otherwise phases 1 and 2 would pass just as
 *     happily against a shared EMA and would be proving nothing.
 *
 * Built only when the NeuroPress source tree is present (see CMakeLists.txt);
 * this is a cross-project check, not part of the default build.
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "clio_ctp/compress/model/neuropress_nn_predictor.h"

// --- Native NeuroPress (compiled from its own source, namespace gpucompress).
// internal.hpp brings the real CompContext layout that runNNSGDCtx reads;
// declaring a stand-in here would only test the stand-in.
#include "api/internal.hpp"

extern "C" void *gpucompress_nn_get_device_ptr_impl(void);

extern cudaStream_t g_sgd_stream;
extern cudaEvent_t g_sgd_done;

// Defined in neuropress_globals_stub.cu; stands in for pool.cpp's g_comp_pool
// so resetAllSGDEMABuffers() reaches the contexts this harness creates.
extern std::vector<float *> g_harness_ctx_grad_buffers;

using ctp::compress::model::CompressionFeatures;
using ctp::compress::model::NeuroPressNNPredictor;
using ctp::compress::model::TrainingLabels;

namespace {

int g_failures = 0;
int g_checks = 0;
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

struct Layer { const char *name; const float *a; const float *b; size_t n; };

/** The ten weight blocks of two snapshots, paired for element-wise diffing. */
std::vector<Layer> PairLayers(const NNWeightsGPU &x, const NNWeightsGPU &y) {
  return {
      {"w1", x.w1, y.w1, NN_HIDDEN_DIM * NN_INPUT_DIM},
      {"b1", x.b1, y.b1, NN_HIDDEN_DIM},
      {"w2", x.w2, y.w2, NN_HIDDEN_DIM * NN_HIDDEN_DIM},
      {"b2", x.b2, y.b2, NN_HIDDEN_DIM},
      {"w3", x.w3, y.w3, NN_HIDDEN_DIM * NN_HIDDEN_DIM},
      {"b3", x.b3, y.b3, NN_HIDDEN_DIM},
      {"w4", x.w4, y.w4, NN_HIDDEN_DIM * NN_HIDDEN_DIM},
      {"b4", x.b4, y.b4, NN_HIDDEN_DIM},
      {"w5", x.w5, y.w5, NN_OUTPUT_DIM * NN_HIDDEN_DIM},
      {"b5", x.b5, y.b5, NN_OUTPUT_DIM},
  };
}

/** Element-wise weight diff, reported per layer so a divergence is localized
 *  rather than just "something moved". */
void CompareWeights(const NNWeightsGPU &nat, const NNWeightsGPU &clio,
                    const std::string &phase, double tol) {
  for (const auto &L : PairLayers(nat, clio)) {
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

/** Largest element-wise gap between two snapshots, over all ten blocks. */
double MaxAbsDiff(const NNWeightsGPU &x, const NNWeightsGPU &y) {
  double max_abs = 0.0;
  for (const auto &L : PairLayers(x, y)) {
    for (size_t i = 0; i < L.n; ++i) {
      max_abs = std::max(max_abs,
                         std::fabs(static_cast<double>(L.a[i]) - L.b[i]));
    }
  }
  return max_abs;
}

struct Chunk {
  const char *label;
  double entropy;
  double mad;
  double deriv;
  size_t size_bytes;
};

/** One training step, consumed identically by the native and Clio sides so
 *  the two cannot drift in what they were asked to learn. */
struct Step {
  Chunk chunk;
  int flow;       // which flow issues it
  int algo_idx;   // action (quant=0, shuffle=0)
  float ratio;
  float comp_ms;
  float decomp_ms;
};

// ---------------------------------------------------------------------------
// Native side: a CompContext carrying only the SGD buffers runNNSGDCtx reads.
// ---------------------------------------------------------------------------
struct NativeFlow {
  CompContext ctx{};
  AutoStatsGPU *d_stats = nullptr;

  bool Init() {
    // Same three allocations gpucompress_pool.cpp:118-130 makes per slot,
    // and the same zero-fill -- region 2 must start clean or the first
    // update consumes uninitialized momentum.
    if (cudaMalloc(&ctx.d_sgd_grad_buffer,
                   NN_SGD_GRAD_SIZE * sizeof(float)) != cudaSuccess) {
      return false;
    }
    cudaMemset(ctx.d_sgd_grad_buffer, 0, NN_SGD_GRAD_SIZE * sizeof(float));
    if (cudaMalloc(&ctx.d_sgd_output, sizeof(SGDOutput)) != cudaSuccess) {
      return false;
    }
    if (cudaMalloc(&ctx.d_sgd_samples,
                   NN_MAX_SGD_SAMPLES * sizeof(SGDSample)) != cudaSuccess) {
      return false;
    }
    if (cudaMalloc(&d_stats, sizeof(AutoStatsGPU)) != cudaSuccess) {
      return false;
    }
    g_harness_ctx_grad_buffers.push_back(ctx.d_sgd_grad_buffer);
    return true;
  }

  void Free() {
    cudaFree(ctx.d_sgd_grad_buffer);
    cudaFree(ctx.d_sgd_output);
    cudaFree(ctx.d_sgd_samples);
    cudaFree(d_stats);
  }

  /** One upstream SGD step through THIS context's gradient buffer. */
  bool Train(const Step &st) {
    AutoStatsGPU h_stats{};
    h_stats.entropy = st.chunk.entropy;
    h_stats.mad_normalized = st.chunk.mad;
    h_stats.deriv_normalized = st.chunk.deriv;
    h_stats.num_elements = st.chunk.size_bytes / sizeof(float);
    cudaMemcpy(d_stats, &h_stats, sizeof(h_stats), cudaMemcpyHostToDevice);

    SGDSample s{};
    s.action = st.algo_idx;  // quant=0, shuffle=0
    s.actual_ratio = st.ratio;
    s.actual_comp_time = st.comp_ms;
    s.actual_decomp_time = st.decomp_ms;
    s.actual_psnr = 0.0f;  // lossless

    float gn = 0; int clipped = 0, cnt = 0;
    int rc = gpucompress::runNNSGDCtx(d_stats, &s, 1, st.chunk.size_bytes,
                                      /*error_bound=*/0.0, /*lr=*/0.01f, &ctx,
                                      &gn, &clipped, &cnt);
    // runNNSGDCtx is deliberately fire-and-forget (nn_gpu.cu:2329-2333): it
    // records g_sgd_done and returns without syncing. The weight snapshot
    // below reads the result, so the sync upstream leaves to the next
    // inference has to happen here instead.
    cudaStreamSynchronize(g_sgd_stream);
    return rc == 0;
  }
};

// ---------------------------------------------------------------------------
// Clio side: a worker thread, which is the port's flow scope.
// ---------------------------------------------------------------------------
class ClioFlow {
 public:
  void Start() { thread_ = std::thread([this] { Loop(); }); }

  /** Run `job` on this flow's thread and block until it finishes. The block
   *  is the point: it makes the interleaving deterministic and identical to
   *  the native side's, so any difference in the weights is the EMA split
   *  and not a race. Every Train() from this flow lands on the same thread
   *  and therefore on the same thread_local EMA buffer. */
  void Run(std::function<void()> job) {
    {
      std::lock_guard<std::mutex> lk(m_);
      job_ = std::move(job);
      pending_ = true;
    }
    cv_.notify_one();
    std::unique_lock<std::mutex> lk(m_);
    done_cv_.wait(lk, [this] { return !pending_; });
  }

  void Stop() {
    {
      std::lock_guard<std::mutex> lk(m_);
      stop_ = true;
    }
    cv_.notify_one();
    thread_.join();
  }

 private:
  void Loop() {
    for (;;) {
      std::function<void()> job;
      {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [this] { return pending_ || stop_; });
        if (!pending_) return;  // stop_ with nothing queued
        job = std::move(job_);
      }
      job();
      {
        std::lock_guard<std::mutex> lk(m_);
        pending_ = false;
      }
      done_cv_.notify_one();
    }
  }

  std::thread thread_;
  std::mutex m_;
  std::condition_variable cv_, done_cv_;
  std::function<void()> job_;
  bool pending_ = false;
  bool stop_ = false;
};

/** The Clio-side counterpart of NativeFlow::Train -- same features, same
 *  labels, built from the same Step. */
void ClioTrain(NeuroPressNNPredictor *clio, const Step &st, bool *ok) {
  CompressionFeatures f;
  f.chunk_size_bytes = static_cast<double>(st.chunk.size_bytes);
  f.shannon_entropy = st.chunk.entropy;
  f.mad = st.chunk.mad;
  f.second_derivative_mean = st.chunk.deriv;
  f.data_type_float = 1.0;
  f.library_config_id = BaseIdForAlgoIdx(st.algo_idx) * 10 + 2;
  f.config_balanced = 1.0;
  // Identical label on both sides, decomp time included: native feeds output
  // 1's error into the shared trunk (only its HEAD weights are withheld), so
  // withholding it from Clio alone would diverge the trunks for a reason
  // that has nothing to do with the flow split under test.
  std::vector<TrainingLabels> labels = {
      TrainingLabels(st.ratio, /*psnr=*/0.0f, st.comp_ms, st.decomp_ms)};
  *ok = clio->Train({f}, labels);
}

/**
 * Drive one full pass of `steps` through both sides, diffing the weights
 * after every step.
 *
 * @param native_flows One CompContext per flow; @param clio_flows one worker
 *   thread per flow. Passing a single entry in each runs the whole sequence
 *   through one flow, which is how phase 3 builds its control.
 * @param flow_of Maps a step's flow id onto an index into those vectors.
 */
void RunSequence(const std::vector<Step> &steps, NeuroPressNNPredictor *clio,
                 const std::vector<NativeFlow *> &native_flows,
                 const std::vector<ClioFlow *> &clio_flows,
                 const std::function<size_t(int)> &flow_of,
                 const std::string &phase, NNWeightsGPU *out_native_final,
                 NNWeightsGPU *out_clio_final) {
  for (size_t i = 0; i < steps.size(); ++i) {
    const Step &st = steps[i];
    const size_t fi = flow_of(st.flow);

    if (!native_flows[fi]->Train(st)) {
      std::printf("  [FAIL] %s: native SGD failed at step %zu (cuda=%s)\n",
                  phase.c_str(), i, cudaGetErrorString(cudaGetLastError()));
      ++g_failures;
    }
    ++g_checks;

    bool clio_ok = false;
    clio_flows[fi]->Run([&] { ClioTrain(clio, st, &clio_ok); });
    Check(clio_ok, phase + ": Clio Train() succeeded at step " +
                       std::to_string(i));

    NNWeightsGPU nat{}, cl{};
    SnapshotNative(&nat);
    FlattenClio(clio->DebugWeights(), clio->DebugBiases(), &cl);
    // Tolerance matches the other harnesses: both sides accumulate float32
    // rounding in a different order (warp reductions vs sequential loops).
    CompareWeights(nat, cl, phase + " step " + std::to_string(i), 2e-5);
    if (out_native_final) *out_native_final = nat;
    if (out_clio_final) *out_clio_final = cl;
  }
  std::printf("  %s: %zu steps compared\n", phase.c_str(), steps.size());
}

/**
 * Alternating two-flow workload.
 *
 * Flow 0 always reports far-OVER outcomes and flow 1 far-UNDER, past the
 * warmup threshold (sgd_call_count > 3) where the anti-flip damping starts
 * consulting the EMA. Each flow's own gradient therefore holds a stable sign
 * while the interleaved sequence flips every step -- so a per-flow EMA damps
 * rarely and a shared one damps constantly, and the two cannot agree.
 */
std::vector<Step> BuildSteps() {
  const Chunk chunks[] = {
      {"near-zero entropy", 0.20, 0.010, 0.005, 2u << 20},
      {"low entropy",       1.50, 0.050, 0.020, 2u << 20},
      {"mid entropy",       4.00, 0.200, 0.100, 1u << 20},
      {"high entropy",      6.50, 0.600, 0.400, 4u << 20},
      {"max entropy",       7.90, 0.950, 0.900, 512u << 10},
      {"tiny chunk",        3.20, 0.150, 0.080, 64u << 10},
      {"large chunk",       2.10, 0.090, 0.030, 16u << 20},
  };
  const size_t num_chunks = sizeof(chunks) / sizeof(chunks[0]);

  std::vector<Step> steps;
  for (int i = 0; i < 16; ++i) {
    const int flow = i % 2;
    Step st;
    st.chunk = chunks[static_cast<size_t>(i) % num_chunks];
    st.flow = flow;
    st.algo_idx = i % 8;
    st.ratio = (flow == 0) ? 9.0f : 1.1f;
    st.comp_ms = (flow == 0) ? 40.0f : 2.0f;
    st.decomp_ms = (flow == 0) ? 25.0f : 1.5f;
    steps.push_back(st);
  }
  return steps;
}

/** Reload the model on both sides. Issued from the main thread, which is
 *  neither flow -- so it exercises the registry walk, not a self-reset. */
bool ReloadBoth(const std::string &weights_dir, const std::string &nnwt,
                NeuroPressNNPredictor *clio) {
  if (!gpucompress::loadNNFromBinary(nnwt.c_str())) return false;
  return clio->Load(weights_dir) && clio->IsReady();
}

}  // namespace

int main(int argc, char **argv) {
  const std::string weights_dir =
      (argc > 1) ? argv[1] : std::string(CLIO_CTP_NEUROPRESS_WEIGHTS_DIR);
  const std::string nnwt = weights_dir + "/model.nnwt";

  int dev_count = 0;
  if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) {
    std::printf("No CUDA device -- skipping flow parity check.\n");
    return 77;  // ctest SKIP_RETURN_CODE
  }

  // gpucompress_init() normally creates these; the harness skips it (that
  // path drags in nvcomp/cuSZ/ndzip), so stand them up here. Leaving them
  // null makes runNNSGDCtx's event record fail and the update silently
  // no-op. Created BEFORE the load, which resets EMA buffers.
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
  std::printf("Both sides loaded %s\n\n", nnwt.c_str());

  NativeFlow native_a, native_b;
  if (!native_a.Init() || !native_b.Init()) {
    std::printf("FATAL: could not allocate native flow contexts\n");
    return 1;
  }
  std::vector<NativeFlow *> native_flows = {&native_a, &native_b};

  ClioFlow flow_a, flow_b;
  flow_a.Start();
  flow_b.Start();
  std::vector<ClioFlow *> clio_flows = {&flow_a, &flow_b};

  const std::vector<Step> steps = BuildSteps();
  auto identity_flow = [](int f) { return static_cast<size_t>(f); };

  // ---- Phase 0: identical weights before anything runs. ----
  {
    NNWeightsGPU nat{}, cl{};
    Check(SnapshotNative(&nat), "snapshot native weights");
    FlattenClio(clio.DebugWeights(), clio.DebugBiases(), &cl);
    CompareWeights(nat, cl, "load", 0.0);  // must be bit-identical
    std::printf("[phase 0] post-load weight parity checked\n");
  }

  // ---- Phase 1: two flows, interleaved. ----
  std::printf("\n[phase 1] two flows interleaved "
              "(native: 2 CompContexts, Clio: 2 threads)\n");
  NNWeightsGPU two_flow_native{}, two_flow_clio{};
  RunSequence(steps, &clio, native_flows, clio_flows, identity_flow,
              "2-flow", &two_flow_native, &two_flow_clio);

  // ---- Phase 2: reload must clear EVERY flow's gradient history. ----
  std::printf("\n[phase 2] reload from the main thread, then replay\n");
  if (!ReloadBoth(weights_dir, nnwt, &clio)) {
    std::printf("FATAL: reload failed\n");
    return 1;
  }
  NNWeightsGPU replay_native{}, replay_clio{};
  RunSequence(steps, &clio, native_flows, clio_flows, identity_flow,
              "2-flow-replay", &replay_native, &replay_clio);
  // Both flows were warm and neither is the thread that reloaded. If the
  // reset had reached only the caller, this replay would start from stale
  // momentum and land somewhere else than phase 1 did.
  //
  // Checked on each side against ITS OWN phase-1 result, not across the two:
  // a cross-side check here would only ever exercise native's reset, since
  // the per-step CompareWeights above already pins Clio to native. Clio's
  // registry walk is the thing this phase exists to test, so it needs a
  // Clio-only baseline.
  struct ResetCase { const char *side; const NNWeightsGPU &before, &after; };
  const ResetCase reset_cases[] = {
      {"native", two_flow_native, replay_native},
      {"clio", two_flow_clio, replay_clio},
  };
  for (const auto &rc : reset_cases) {
    double drift = MaxAbsDiff(rc.before, rc.after);
    ++g_checks;
    if (drift > 0.0) {
      ++g_failures;
      std::printf("  [FAIL] %s replay after reload diverged from phase 1: "
                  "max|diff|=%.6g (stale EMA survived the reload)\n",
                  rc.side, drift);
    } else {
      std::printf("  %s replay reproduced phase 1 exactly (EMA reset reached "
                  "both flows)\n", rc.side);
    }
  }

  // ---- Phase 3: the same sequence through ONE flow. ----
  std::printf("\n[phase 3] same sequence through a single flow (control)\n");
  if (!ReloadBoth(weights_dir, nnwt, &clio)) {
    std::printf("FATAL: reload failed\n");
    return 1;
  }
  std::vector<NativeFlow *> single_native = {&native_a};
  std::vector<ClioFlow *> single_clio = {&flow_a};
  auto collapse_flow = [](int) { return static_cast<size_t>(0); };
  NNWeightsGPU one_flow_native{}, one_flow_clio{};
  RunSequence(steps, &clio, single_native, single_clio, collapse_flow,
              "1-flow", &one_flow_native, &one_flow_clio);

  // Discrimination: if routing all 16 steps through one EMA buffer landed on
  // the same weights as splitting them across two, then phases 1 and 2 would
  // pass against a shared EMA too and would be measuring nothing.
  //
  // Asserted on both sides for different reasons. Native separating its two
  // runs is what makes per-flow state a real upstream property rather than a
  // Clio invention; Clio separating its own is what proves the port did not
  // quietly collapse both flows onto one buffer.
  struct SplitCase { const char *side; const NNWeightsGPU &two, &one; };
  const SplitCase split_cases[] = {
      {"native", two_flow_native, one_flow_native},
      {"clio", two_flow_clio, one_flow_clio},
  };
  for (const auto &sc : split_cases) {
    double gap = MaxAbsDiff(sc.two, sc.one);
    ++g_checks;
    if (!(gap > 1e-6)) {
      ++g_failures;
      std::printf("  [FAIL] %s: one flow and two flows agree to %.6g -- the "
                  "EMA split is not being exercised\n", sc.side, gap);
    } else {
      std::printf("  %s one-flow vs two-flow: max|diff|=%.6g "
                  "(flows are distinct)\n", sc.side, gap);
    }
  }

  flow_a.Stop();
  flow_b.Stop();
  native_a.Free();
  native_b.Free();
  cudaStreamDestroy(g_sgd_stream);
  cudaEventDestroy(g_sgd_done);

  std::printf("\n----- observed worst-case deviation -----\n");
  std::printf("  weights : max abs err = %.6g  (%s %s)\n", g_max_abs_weight,
              g_max_abs_phase.c_str(), g_max_abs_layer);
  std::printf("\n===== %d checks, %d failures =====\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
