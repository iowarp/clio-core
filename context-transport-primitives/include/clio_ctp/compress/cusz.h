/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_CUSZ_H_
#define CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_CUSZ_H_

#if CTP_ENABLE_COMPRESS && CTP_ENABLE_CUSZ

#include <cuda_runtime.h>
#include <cusz.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

#include "compress.h"

namespace ctp {

namespace cusz_detail {

/**
 * Per-call breakdown of everything Compress() does around the codec kernel.
 *
 * `compress_ms` brackets `psz_compress_float` alone, so the stream creation,
 * the resource manager, the copy-out and the release all land in the phase
 * log's `other_ms` -- 17.45 ms a chunk on full Nyx against 0.09-0.40 ms for
 * every other codec. This says which of them it is. Off unless
 * CLIO_CUSZ_PHASE_LOG names a file; one fprintf per chunk.
 */
class SetupLog {
 public:
  static SetupLog *Get() {
    static SetupLog log;
    return &log;
  }
  bool enabled() const { return fp_ != nullptr; }
  static double MsSince(const std::chrono::steady_clock::time_point &t) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t).count();
  }
  /** Steady-clock nanoseconds, the stamp Record() expects. */
  static long long Now() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }

  /**
   * Append ONE timed span of resource work, with when it began.
   *
   * ONE ROW PER SPAN, STAMPED. This used to write one row of four durations
   * per chunk, and the harness summed them -- which over-counts, because
   * chunks compress on several workers at once: on a 1024-chunk smoke the sum
   * came to 14.307 s inside a 13.539 s arm, a "cost" larger than the bar
   * containing it. Stamped spans let the caller take the UNION instead, the
   * same treatment io_log.h gets and for the same reason.
   *
   * The spans are disjoint and deliberately kept apart: the build runs before
   * the codec kernel and the release after it, so unioning them as one
   * interval would swallow the compression in between.
   *
   * @param phase which span: "stream", "mgr" or "release".
   * @param n elements in the chunk.
   * @param ms how long this span took.
   * @param start_ns steady-clock nanoseconds when it began, from Now().
   */
  void Record(const char *phase, size_t n, double ms, long long start_ns) {
    if (fp_ == nullptr) return;
    std::lock_guard<std::mutex> lock(mutex_);
    std::fprintf(fp_, "%s,%zu,%.4f,%lld\n", phase, n, ms, start_ns);
  }

 private:
  SetupLog() {
    const char *path = std::getenv("CLIO_CUSZ_PHASE_LOG");
    if (path == nullptr || *path == '\0') return;
    fp_ = std::fopen(path, "w");
    if (fp_ != nullptr) std::fprintf(fp_, "phase,elems,ms,start_ns\n");
  }
  ~SetupLog() { if (fp_ != nullptr) std::fclose(fp_); }
  std::FILE *fp_ = nullptr;
  std::mutex mutex_;
};

/**
 * ONE CONTEXT PER THREAD, REUSED ACROSS CHUNKS -- opt-in, see ReuseManagers.
 *
 * cuSZ documents TWO manager lifecycles, and both are official:
 *   demo_v2.cuda.cc   create -> ONE psz_compress_float -> release.
 *   batch_run.cc      create -> a whole file list through it -> release once.
 * This wrapper built one per chunk, which is demo_v2's single-shot pattern
 * applied in a loop. That is sanctioned usage, not a misuse of the API -- but
 * it is the wrong one of the two for a streaming replay, and it cost 20.8 ms a
 * chunk at 8 MiB, 81% of the cuSZ arm's bar in figure 9.
 *
 * batch_run.cc's pattern is the right one here, and against upstream e1c0135
 * IT CORRUPTS ITS OUTPUT. Proven outside this tree on real Nyx data (job
 * 22322980, then 22323800 against both libraries,
 * /u/imuradli/np-fix/cusz-reuse/reuse_probe.cu):
 *
 *   blow-ups (decoded error > 100x eb):  fresh 0/10,  reused 8/10
 *   max|err| 199.579 against eb 1e-3 -- the field's own magnitude, i.e.
 *   nothing decoded -- and psz_compress_float still returns success.
 *
 * A FRESH-vs-FRESH control is what makes that readable, and it is why an
 * earlier reading of the same probe was wrong: cuSZ is NONDETERMINISTIC, so
 * chunks differ in compressed BYTES between two identical fresh runs.
 * Byte and size diffs prove nothing on their own; only the blow-up count does.
 *
 * ROOT CAUSE, and it is upstream's: compressor.inl zeroes the outlier counter
 * only in the Spline branch, while every Lorenzo kernel claims its outlier
 * slots with atomicAdd on that same counter. A fresh manager is correct only
 * because malloc_device memsets at allocation; nothing zeroes it again. So a
 * reused manager counts on from the previous chunk's total -- stale entries
 * stay at the front of the outlier array with header->splen covering them, and
 * once the inflated index passes cn_max_allowed genuine outliers are dropped.
 * Hoisting that one memset above the predictor branch fixes it: blow-ups 0/10
 * and 0/64, max|err| identical to fresh, setup 294.0 ms -> 16.1 ms over ten
 * 8 MiB chunks. A patched build lives in np-env/cusz-fixed.
 *
 * Reuse is therefore ON BY DEFAULT but PROVEN BEFORE USE: because the failure
 * is silent and cannot be read off a version string, ReuseSelfTest compresses
 * two buffers through one manager once per process and only then takes the
 * fast path. CLIO_CUSZ_REUSE_MANAGER=0 forces the per-chunk path.
 *
 * The original text follows.
 *
 * cuSZ's resource manager owns the GPU buffers the pipeline needs -- histogram,
 * Huffman book, quantisation codes -- so building one is CUDA allocation work,
 * not bookkeeping. Measured on 1024 chunks of 4 MiB (job 22317842):
 * psz_create_resource_manager 7.74 ms and psz_release_resource 3.17 ms a chunk,
 * against a 0.44 ms codec kernel; at figure 9's 8 MiB chunks that was 68 s of
 * the cuSZ arm's 81 s. Every other codec here reuses or avoids such state --
 * nvcomp caches its manager per thread, cuSZp's API is stateless -- so cuSZ
 * alone paid a full build and teardown per chunk.
 *
 * Keyed by element count: a manager is sized for one length, and a replay
 * hands out one length until a file's short tail. Thread-local, so no lock and
 * no sharing of a manager between concurrent compressions. NEVER RELEASED at
 * thread exit, deliberately and like the nvcomp slots (nvcomp.h:854): the CUDA
 * context may already be gone when a thread_local destructor runs at shutdown,
 * and freeing a manager into a dead context is a crash, not a leak worth
 * chasing -- the process is exiting.
 */
struct Context {
  cudaStream_t stream = nullptr;
  psz_resource *mgr = nullptr;
  size_t elems = 0;
};

/**
 * The calling thread's context, sized for `n` elements.
 *
 * @param n elements in this chunk.
 * @param pipeline cuSZ pipeline the manager is built for.
 * @return the context, or nullptr when the stream or the manager failed.
 */
inline Context *AcquireStream() {
  thread_local Context ctx;
  if (ctx.stream == nullptr && cudaStreamCreate(&ctx.stream) != cudaSuccess) {
    return nullptr;
  }
  return &ctx;
}

/**
 * The calling thread's manager, sized for `n` elements, built only when the
 * length changed.
 *
 * @param ctx the calling thread's context, from AcquireStream.
 * @param n elements in this chunk.
 * @param pipeline cuSZ pipeline the manager is built for.
 * @return the manager, or nullptr when it could not be built.
 */
inline psz_resource *AcquireManager(Context *ctx, size_t n,
                                    psz_pipeline pipeline) {
  if (ctx->mgr != nullptr && ctx->elems == n) return ctx->mgr;
  if (ctx->mgr != nullptr) {
    psz_release_resource(ctx->mgr);
    ctx->mgr = nullptr;
    ctx->elems = 0;
  }
  psz_len len = {n, 1, 1};
  ctx->mgr = psz_create_resource_manager(F4, len, pipeline, ctx->stream);
  if (ctx->mgr == nullptr) return nullptr;
  ctx->elems = n;
  return ctx->mgr;
}

/**
 * Compress `d_in` through `mgr` and report the OUTLIER COUNT its frame carries.
 *
 * psz_header::splen is the number of values the predictor could not quantize,
 * read straight out of the device counter that the reuse defect fails to
 * reset -- so it is the defect's own variable, not a proxy for it. Reading it
 * needs no decompression, which is what an earlier version of this test tried
 * and could not get to work (the decode half failed outright, job 22324667).
 *
 * @param mgr manager to compress through (fresh or already used).
 * @param d_in device buffer of `kN` floats.
 * @param splen set to the frame's outlier count on success.
 * @param why set to a reason when this returns false.
 * @return true when the compression produced a frame.
 */
inline bool ReuseProbeSplen(psz_resource *mgr, float *d_in, size_t *splen,
                            const char **why) {
  psz_header header;
  uint8_t *d_comp = nullptr;
  size_t comp_bytes = 0;
  psz_rc2 rc = {Abs, 1e-3, 512};
  const int st = psz_compress_float(mgr, rc, d_in, &header, &d_comp, &comp_bytes);
  if (st != 0 || d_comp == nullptr) {
    *why = (st != 0) ? "psz_compress_float returned non-zero"
                     : "psz_compress_float produced no frame";
    return false;
  }
  *splen = static_cast<size_t>(header.splen);
  return true;
}

/**
 * Does the cuSZ we actually linked survive manager reuse? A one-shot GPU test.
 *
 * WHY THIS EXISTS. Reuse is only correct against a cuSZ that zeroes its
 * outlier counter on every compression. Upstream e1c0135 resets it ONLY in the
 * Spline branch of compressor.inl, while every Lorenzo kernel claims outlier
 * slots with atomicAdd on that same counter -- so a reused manager counts on
 * from the previous chunk's total. Stale entries stay at the front of the
 * outlier array with header->splen covering them, and once the inflated index
 * passes cn_max_allowed genuine outliers are dropped instead. Measured on Nyx
 * at eb 1e-3: blow-ups FRESH 0/10, REUSE 8/10, max|err| 199.579 -- the field's
 * own magnitude, i.e. nothing decoded (job 22323800).
 *
 * THE FAILURE IS SILENT: psz_compress_float returns success and the frame
 * decodes to garbage. It cannot be inferred from a version string either, so
 * the only honest gate is to try it. This compresses TWO DIFFERENT buffers
 * through ONE manager and checks the second still round-trips; an unpatched
 * cuSZ carries the first buffer's outliers into the second and blows up by
 * five orders of magnitude, which no threshold choice can miss.
 *
 * FAIL-SAFE: anything unexpected -- allocation failure, a codec error, even a
 * FRESH round trip that misses its own bound -- returns false, and the wrapper
 * builds a manager per chunk. Slow and correct beats fast and wrong.
 *
 * Costs one 256 KiB round trip pair, once per process.
 *
 * @return true when reuse reproduced the fresh result.
 */
inline bool ReuseSelfTest() {
  constexpr size_t kN = 1u << 20;   // 1 Mi floats = 4 MiB, a real chunk size
  psz_pipeline pipeline = {Lorenzo, HistGeneric, HF, CodecNull};
  const char *why = "did not run";
  double err_fresh = -1.0, diff = -1.0, tol = 0.0;

  // Two DIFFERENT signals, each varying fast enough that the Lorenzo predictor
  // cannot quantize every sample: the defect only shows when the FIRST
  // compression leaves a non-zero outlier count for the second to count on
  // from. Adjacent samples move by ~5, far outside the +-radius*2*eb window,
  // so outliers are plentiful. Neither is flat -- a flat chunk drives cuSZ
  // into a separate, pre-existing zero-bit-Huffman failure that has nothing to
  // do with reuse and would make this test ask the wrong question.
  std::vector<float> a(kN), b(kN);
  for (size_t i = 0; i < kN; ++i) {
    a[i] = static_cast<float>(std::sin(0.050 * double(i)) * 100.0);
    b[i] = static_cast<float>(std::cos(0.037 * double(i)) * 100.0 + 7.0);
  }
  // THE CRITERION IS THE OUTLIER COUNT ITSELF. Compressing b through a manager
  // that already compressed a must report the SAME splen as compressing b
  // through a fresh one -- splen is the device counter the defect fails to
  // reset, so a broken library reports a's outliers plus b's. That is a direct
  // read of the broken variable and needs no decompression, unlike two earlier
  // versions of this test: one gated on the error bound, which cuSZ misses
  // routinely even when fresh (job 22324550), and one decoded the frame, whose
  // decompress half failed outright (job 22324667). Both refused every time
  // and reuse never engaged.
  cudaStream_t stream = nullptr;
  if (cudaStreamCreate(&stream) != cudaSuccess) {
    std::fprintf(stderr, "[clio_ctp::cusz] manager reuse DISABLED: no stream\n");
    return false;
  }
  float *d_a = nullptr, *d_b = nullptr;
  psz_resource *fresh = nullptr, *reused = nullptr;
  size_t sp_fresh = 0, sp_a = 0, sp_reuse = 0;
  bool ok = false;
  do {
    if (cudaMalloc(&d_a, kN * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_b, kN * sizeof(float)) != cudaSuccess ||
        cudaMemcpy(d_a, a.data(), kN * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_b, b.data(), kN * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      why = "device allocation or copy failed";
      break;
    }
    psz_len len = {kN, 1, 1};

    // Reference: b through a manager that has compressed nothing else.
    fresh = psz_create_resource_manager(F4, len, pipeline, stream);
    if (fresh == nullptr) { why = "could not build the reference manager"; break; }
    if (!ReuseProbeSplen(fresh, d_b, &sp_fresh, &why)) break;

    // Under test: a first, then the SAME b, through one manager.
    reused = psz_create_resource_manager(F4, len, pipeline, stream);
    if (reused == nullptr) { why = "could not build the reuse manager"; break; }
    if (!ReuseProbeSplen(reused, d_a, &sp_a, &why)) break;
    if (!ReuseProbeSplen(reused, d_b, &sp_reuse, &why)) break;

    // Without outliers from the priming pass there is nothing for a broken
    // library to carry over, so the comparison below would pass vacuously.
    if (sp_a == 0) { why = "the probe data produced no outliers to carry over"; break; }

    // A correct library reports b's own count both times. Allow 1% of slack
    // so atomics ordering cannot flip the verdict; a broken one is off by
    // sp_a, which is orders of magnitude larger.
    const size_t slack = 1 + sp_fresh / 100;
    ok = sp_reuse <= sp_fresh + slack;
    why = ok ? "reuse reported the reference outlier count"
             : "reuse carried the previous buffer's outliers";
    tol = static_cast<double>(sp_fresh + slack);
    err_fresh = static_cast<double>(sp_fresh);
    diff = static_cast<double>(sp_reuse);
  } while (false);

  if (fresh != nullptr) psz_release_resource(fresh);
  if (reused != nullptr) psz_release_resource(reused);
  if (d_a != nullptr) cudaFree(d_a);
  if (d_b != nullptr) cudaFree(d_b);
  cudaStreamDestroy(stream);

  // EVERY path says what it decided and why. The first version of this guard
  // refused on a bare `break` and printed nothing, so the fast path silently
  // never engaged -- the same class of silence it exists to prevent.
  if (!ok) {
    std::fprintf(stderr,
                 "[clio_ctp::cusz] manager reuse DISABLED (%s): reference "
                 "outliers %.0f, reused %.0f, allowed %.0f. "
                 "Building a manager per chunk -- correct, but ~20 ms a chunk. "
                 "Link a cuSZ that zeroes its outlier counter per compression "
                 "for the fast path, or set CLIO_CUSZ_REUSE_MANAGER=0 to "
                 "silence this.\n",
                 why, err_fresh, diff, tol);
  } else if (std::getenv("CLIO_CUSZ_REUSE_VERBOSE") != nullptr) {
    std::fprintf(stderr,
                 "[clio_ctp::cusz] manager reuse ENABLED (%s): reference "
                 "outliers %.0f, reused %.0f, allowed %.0f\n",
                 why, err_fresh, diff, tol);
  }
  return ok;
}

/**
 * Should this process reuse a resource manager across chunks?
 *
 * ON BY DEFAULT, but only once ReuseSelfTest has proved the linked cuSZ
 * survives it -- so the fast path is automatic where it is safe and disables
 * itself, loudly, where it is not. `CLIO_CUSZ_REUSE_MANAGER=0` forces the
 * per-chunk path without running the test.
 *
 * Resolved once: the answer cannot change inside a run, and Compress is hot.
 *
 * @return true when a manager may be kept across chunks.
 */
inline bool ReuseManagers() {
  static const bool on = [] {
    const char *v = std::getenv("CLIO_CUSZ_REUSE_MANAGER");
    if (v != nullptr && v[0] == '0' && v[1] == '\0') return false;
    return ReuseSelfTest();
  }();
  return on;
}

}  // namespace cusz_detail

/**
 * cuSZ GPU error-bounded LOSSY compressor for floating-point scientific data --
 * the GPU lossy analog of the LibPressio sz3/zfp entries, the way NvComp is the
 * GPU lossless analog of the CPU codecs.
 *
 * https://github.com/szcompressor/cuSZ
 *
 * cuSZ operates directly on GPU device memory. Like NvComp, this wrapper is
 * adaptive: a pointer that already refers to GPU-accessible memory is used in
 * place (zero-copy); otherwise the data is staged through a temporary device
 * buffer (H2D for inputs, D2H for outputs), so host callers (and the unit test
 * harness) work too while device callers get the zero-copy path automatically.
 *
 * Like the existing zfp/sz3/fpzip/zfp-sycl entries, and constrained by the
 * byte-stream Compressor interface (which carries neither type nor shape), the
 * input buffer is interpreted as a 1D array of 32-bit floats. Compress requires
 * input_size to be a multiple of sizeof(float).
 *
 * Self-describing framing: the output blob is laid out as
 *   [ Prefix (magic + elem count + psz_header) ][ cuSZ compressed byte-stream ]
 * so Decompress rebuilds the cuSZ resource manager from the embedded header
 * without any out-of-band metadata (cuSZ's compressed stream alone does not
 * carry its decode metadata).
 *
 * IMPORTANT -- LOSSY. Each value is reconstructed within the configured error
 * bound, which is ABSOLUTE by default, matching NeuroPress's unconditional
 * `rc.mode = Abs` (external_compressors.cu:118) and the cuSZp sibling in this
 * directory. Not bit-exact. Presets map to error bounds: FAST=1e-2 (loose),
 * BALANCED=1e-3, BEST=1e-4 (tight).
 *
 * Tested against the cuSZ master / 0.14 C API (psz_create_resource_manager,
 * psz_compress_float, psz_decompress_float, psz_release_resource). The NVIDIA
 * devcontainer pins a matching cuSZ revision.
 */
class Cusz : public Compressor {
 public:
  /**
   * @param eb error bound (default 1e-3).
   * @param mode error-bound mode: Abs (absolute, the default) or Rel.
   *
   * The default is ABSOLUTE because that is what NeuroPress uses:
   * `rc.mode = Abs` (external_compressors.cu:118), unconditionally, on every
   * cuSZ compression. It also matches what this class already advertises --
   * the cuSZp sibling documents "the configured ABSOLUTE error bound" and
   * passes CUSZP_MODE_OUTLIER to match upstream exactly.
   *
   * This defaulted to Rel, which silently reinterpreted every caller's bound
   * as a FRACTION OF THE DATA RANGE rather than an absolute tolerance. On a
   * [-100, 100] signal an eb of 1e-3 then permits ~0.2 of error instead of
   * 0.001 -- two orders of magnitude looser than the number the caller
   * passed. Nothing overrode it: MakeCusz (compress_factory.h:439) is the
   * only construction site in the tree and it passes eb alone.
   *
   * Existing blobs are unaffected: each one carries its own psz_header in the
   * frame prefix and Decompress rebuilds the manager from that, exactly as
   * upstream's cuszDecompress does, so a stream written under Rel still
   * decodes as Rel.
   */
  explicit Cusz(double eb = 1e-3, psz_mode mode = Abs)
      : eb_(eb), mode_(mode) {}

  bool Compress(void *output, size_t &output_size, void *input,
                size_t input_size) override {
    if (input == nullptr || (input_size % sizeof(float)) != 0) {
      return false;  // float codec; need whole 32-bit floats
    }
    if (output_size < sizeof(Prefix)) {
      return false;
    }
    const size_t n = input_size / sizeof(float);

    auto *setup_log = cusz_detail::SetupLog::Get();
    const bool log_setup = setup_log->enabled();
    double t_stream = 0.0, t_mgr = 0.0, t_copy = 0.0, t_rel = 0.0;
    long long ns_stream = 0, ns_mgr = 0, ns_rel = 0;
    // With reuse on, the stream and the manager are the calling THREAD's and
    // outlive the chunk; both are then left alone at the end of this call.
    const bool reuse = cusz_detail::ReuseManagers();
    cusz_detail::Context *ctx = nullptr;

    auto t0 = std::chrono::steady_clock::now();
    if (log_setup) ns_stream = cusz_detail::SetupLog::Now();
    cudaStream_t stream = nullptr;
    if (reuse) {
      ctx = cusz_detail::AcquireStream();
      if (ctx == nullptr) return false;
      stream = ctx->stream;
    } else if (cudaStreamCreate(&stream) != cudaSuccess) {
      return false;
    }
    t_stream = cusz_detail::SetupLog::MsSince(t0);
    float *d_in = nullptr;
    bool free_in = false;
    psz_resource *mgr = nullptr;
    bool ok = false;
    do {
      d_in = static_cast<float *>(
          ToDeviceInput(input, input_size, stream, &free_in));
      if (d_in == nullptr) break;

      psz_len len = {n, 1, 1};  // x, y, z (1D)
      psz_pipeline pipeline = {Lorenzo, HistGeneric, HF, CodecNull};
      t0 = std::chrono::steady_clock::now();
      if (log_setup) ns_mgr = cusz_detail::SetupLog::Now();
      // Reusing, this is free on every chunk but the first of a new length,
      // which is what takes the 20.8 ms/chunk build out of the cuSZ arm.
      mgr = reuse ? cusz_detail::AcquireManager(ctx, n, pipeline)
                  : psz_create_resource_manager(F4, len, pipeline, stream);
      t_mgr = cusz_detail::SetupLog::MsSince(t0);
      if (mgr == nullptr) break;

      Prefix prefix;
      std::memset(&prefix, 0, sizeof(prefix));
      prefix.magic = kMagic;
      prefix.elems = static_cast<uint64_t>(n);

      uint8_t *d_comp = nullptr;  // device buffer owned by the manager
      size_t comp_bytes = 0;
      psz_rc2 rc = {mode_, eb_, kRadius};
      // Timed with the same CUDA-event bracket nvcomp uses, so this codec is
      // comparable with the ones it is benchmarked against. Without it the
      // runtime falls back to host wall clock around all of Compress().
      // STOPPED IMMEDIATELY AFTER THE CODEC CALL. If the timer were left to
      // its destructor at end of block it would also cover the output copies
      // and the final stream sync -- exactly the over-broad window this
      // bracket exists to replace. Stop() is idempotent, so the destructor
      // firing on the `break` path is harmless.
      CodecKernelTimer _kt(stream);
      if (psz_compress_float(mgr, rc, d_in, &prefix.header, &d_comp,
                             &comp_bytes) != 0 ||
          d_comp == nullptr) {
        break;
      }
      _kt.Stop();

      t0 = std::chrono::steady_clock::now();
      const size_t total = sizeof(Prefix) + comp_bytes;
      if (total > output_size) break;  // caller buffer too small

      // Frame: [prefix][compressed stream]. d_comp lives in the manager's
      // internal device buffer, so it must be copied out before release.
      const bool out_is_device = IsDeviceAccessible(output);
      uint8_t *out = static_cast<uint8_t *>(output);
      const cudaMemcpyKind kind =
          out_is_device ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;
      if (out_is_device) {
        if (cudaMemcpyAsync(out, &prefix, sizeof(Prefix),
                            cudaMemcpyHostToDevice, stream) != cudaSuccess) {
          break;
        }
      } else {
        std::memcpy(out, &prefix, sizeof(Prefix));
      }
      if (cudaMemcpyAsync(out + sizeof(Prefix), d_comp, comp_bytes, kind,
                          stream) != cudaSuccess) {
        break;
      }
      if (cudaStreamSynchronize(stream) != cudaSuccess) break;
      t_copy = cusz_detail::SetupLog::MsSince(t0);
      output_size = total;
      ok = true;
    } while (false);

    // Reusing, neither the manager nor the stream is torn down here: they are
    // the thread's and the next chunk wants them. The staged input buffer is
    // this call's either way. See Context for why a reused manager is never
    // released at all.
    t0 = std::chrono::steady_clock::now();
    if (log_setup) ns_rel = cusz_detail::SetupLog::Now();
    if (!reuse && mgr != nullptr) psz_release_resource(mgr);
    if (free_in) cudaFree(d_in);
    if (!reuse) cudaStreamDestroy(stream);
    t_rel = cusz_detail::SetupLog::MsSince(t0);
    if (log_setup) {
      // Three stamped spans, not one summed row: see SetupLog::Record. t_copy
      // is deliberately NOT logged -- moving the compressed frame out of the
      // manager's buffer is output work every codec does, so it is never part
      // of what a "construction excluded" figure takes out.
      setup_log->Record("stream", n, t_stream, ns_stream);
      if (ns_mgr != 0) setup_log->Record("mgr", n, t_mgr, ns_mgr);
      setup_log->Record("release", n, t_rel, ns_rel);
    }
    (void)t_copy;
    return ok;
  }

  bool Decompress(void *output, size_t &output_size, void *input,
                  size_t input_size) override {
    if (output == nullptr || input == nullptr || input_size < sizeof(Prefix)) {
      return false;
    }

    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
      return false;
    }
    float *d_out = nullptr;
    bool free_out = false;
    psz_resource *mgr = nullptr;
    bool ok = false;
    do {
      // Pull the prefix (magic + header) back to the host.
      Prefix prefix;
      if (IsDeviceAccessible(input)) {
        if (cudaMemcpy(&prefix, input, sizeof(Prefix),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
          break;
        }
      } else {
        std::memcpy(&prefix, input, sizeof(Prefix));
      }
      if (prefix.magic != kMagic) break;  // not one of our blobs

      const size_t n = static_cast<size_t>(prefix.elems);
      if (n * sizeof(float) > output_size) break;  // caller buffer too small

      // cuSZ writes into device memory; use the caller's buffer if it is a GPU
      // buffer, else decode into a temp and copy D2H.
      const bool out_is_device = IsDeviceAccessible(output);
      if (out_is_device) {
        d_out = static_cast<float *>(output);
      } else {
        if (cudaMalloc(&d_out, n * sizeof(float)) != cudaSuccess) break;
        free_out = true;
      }
      // KCU_x_lorenzo_1d reads this buffer before writing it (~8000 uninit
      // reads, per initcheck); zeroed pages hid it until nvcomp dirtied them.
      if (cudaMemsetAsync(d_out, 0, n * sizeof(float), stream) != cudaSuccess) {
        break;
      }

      // The compressed stream must be device-resident for cuSZ.
      uint8_t *d_stream = nullptr;
      bool free_stream = false;
      const size_t comp_bytes = pszheader_compressed_bytes(&prefix.header);
      uint8_t *stream_src = static_cast<uint8_t *>(input) + sizeof(Prefix);
      if (IsDeviceAccessible(input)) {
        d_stream = stream_src;
      } else {
        if (cudaMalloc(&d_stream, comp_bytes) != cudaSuccess) break;
        free_stream = true;
        if (cudaMemcpyAsync(d_stream, stream_src, comp_bytes,
                            cudaMemcpyHostToDevice, stream) != cudaSuccess) {
          cudaFree(d_stream);
          break;
        }
      }

      mgr = psz_create_resource_manager_from_header(&prefix.header, stream);
      bool decoded = mgr != nullptr &&
                     psz_decompress_float(mgr, d_stream, comp_bytes, d_out) == 0;
      if (decoded && cudaStreamSynchronize(stream) == cudaSuccess) {
        if (!out_is_device) {
          decoded = cudaMemcpy(output, d_out, n * sizeof(float),
                               cudaMemcpyDeviceToHost) == cudaSuccess;
        }
        if (decoded) {
          output_size = n * sizeof(float);
          ok = true;
        }
      }
      if (free_stream) cudaFree(d_stream);
    } while (false);

    if (mgr != nullptr) psz_release_resource(mgr);
    if (free_out) cudaFree(d_out);
    cudaStreamDestroy(stream);
    return ok;
  }

  /**
   * Point this codec at the bound the caller asked for (Compressor override).
   *
   * Used to exist as a plain setter that nothing ever called, so the bound
   * came only from MakeCusz's preset menu; the runtime now calls this with
   * the request's own error bound. The MODE is untouched -- Abs, as NeuroPress
   * uses -- so this changes the tolerance, never its interpretation.
   *
   * @param eb absolute error bound; ignored when <= 0.
   * @return true when the bound was taken.
   */
  bool SetErrorBound(double eb) override {
    if (!(eb > 0.0)) return false;
    eb_ = eb;
    return true;
  }
  /** Get the error bound. */
  double GetErrorBound() const { return eb_; }

 private:
  /** cuSZ quantization radius (cuSZ's historical default). */
  static constexpr uint16_t kRadius = 512;
  static constexpr uint32_t kMagic = 0x5A535543u;  // "CUSZ"

  // Self-describing prefix carried ahead of the cuSZ codestream. Embeds the
  // psz_header so Decompress can rebuild the resource manager from the bytes.
  struct Prefix {
    uint32_t magic;
    uint32_t pad;       // keep `elems` 8-aligned
    uint64_t elems;     // float element count
    psz_header header;  // cuSZ metadata needed to decode
  };

  /**
   * True if `ptr` is dereferenceable by the GPU directly (device or UVM/managed
   * memory). Plain/pinned host pointers return false so the caller stages a
   * copy. A failed query is treated as "not device".
   */
  static bool IsDeviceAccessible(const void *ptr) {
    cudaPointerAttributes attr;
    if (cudaPointerGetAttributes(&attr, ptr) != cudaSuccess) {
      cudaGetLastError();  // reset the sticky error from the failed query
      return false;
    }
    return attr.type == cudaMemoryTypeDevice ||
           attr.type == cudaMemoryTypeManaged;
  }

  /**
   * Return a device pointer holding `size` bytes of `input`: used in place if
   * already GPU-accessible (*owned=false), else copied H2D on `stream`
   * (*owned=true). Returns nullptr on failure.
   */
  static void *ToDeviceInput(void *input, size_t size, cudaStream_t stream,
                             bool *owned) {
    *owned = false;
    if (IsDeviceAccessible(input)) {
      return input;
    }
    void *d = nullptr;
    if (cudaMalloc(&d, size) != cudaSuccess) {
      return nullptr;
    }
    if (cudaMemcpyAsync(d, input, size, cudaMemcpyHostToDevice, stream) !=
        cudaSuccess) {
      cudaFree(d);
      return nullptr;
    }
    *owned = true;
    return d;
  }

  double eb_;       // error bound
  psz_mode mode_;   // error-bound mode (Rel / Abs)
};

}  // namespace ctp

#endif  // CTP_ENABLE_COMPRESS && CTP_ENABLE_CUSZ

#endif  // CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_CUSZ_H_
