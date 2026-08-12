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

#ifndef CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_CUSZP_H_
#define CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_CUSZP_H_

#if CTP_ENABLE_COMPRESS && CTP_ENABLE_CUSZP

#include <cuda_runtime.h>
#include <cuSZp.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "compress.h"
#include <cstdio>
#include <cstdlib>

#define CUSZP_TRACE(...)                                                      \
  do {                                                                        \
    if (std::getenv("CLIO_CUSZP_TRACE")) {                                    \
      std::fprintf(stderr, "[CUSZP] " __VA_ARGS__);                           \
      std::fprintf(stderr, "\n");                                            \
      std::fflush(stderr);                                                    \
    }                                                                         \
  } while (0)

namespace ctp {

/**
 * cuSZp GPU error-bounded LOSSY compressor for floating-point data -- the
 * ultra-fast single-kernel sibling of cuSZ from the same szcompressor family.
 * cuSZp fuses the whole compress/decompress pipeline into one CUDA kernel for
 * very high end-to-end throughput.
 *
 * https://github.com/szcompressor/cuSZp  (cuSZp v3 API)
 *
 * cuSZp operates directly on GPU device memory. Like NvComp/Cusz, this wrapper
 * is adaptive: a GPU-accessible pointer is used in place (zero-copy); otherwise
 * data is staged through a temporary device buffer (H2D for inputs, D2H for
 * outputs), so host callers (and the unit test harness) work too.
 *
 * Constrained by the byte-stream Compressor interface (no type/shape), the input
 * is interpreted as a 1D array of 32-bit floats; Compress requires input_size to
 * be a multiple of sizeof(float).
 *
 * Self-describing framing: the output blob is laid out as
 *   [ Prefix (magic + mode + error bound + elem count + cmp_size) ][ stream ]
 * so Decompress reconstructs with the exact error bound and mode used at
 * compression (cuSZp's decompress requires both), independent of the preset this
 * object was built with.
 *
 * cuSZp quirk: its reported cmp_size only covers the full compressed payload for
 * MULTI-block inputs (> 32768 elements); for a single-block input it omits the
 * payload, so this wrapper retains the whole compress buffer in that case (still
 * a correct round-trip, but uncompressed). Real GPU-resident blobs are larger
 * than 128 KB and hit the multi-block, fully-compressed path. See Compress.
 *
 * IMPORTANT -- LOSSY. Each value is reconstructed within the configured ABSOLUTE
 * error bound, not bit-exact. Presets map to error bounds: FAST=1e-2 (loose),
 * BALANCED=1e-3, BEST=1e-4 (tight).
 *
 * NOTE on GPU arch: cuSZp (like cuSZ) currently builds for compute capabilities
 * up to ~sm_86. On a newer GPU only reachable via forward-compat PTX-JIT (e.g.
 * Blackwell sm_120), its JIT'd kernels can be clobbered by other CUDA modules in
 * the same process and return wrong results -- use a natively-supported GPU until
 * upstream adds the newer arch.
 *
 * Tested against the cuSZp v3 C API (cuSZp_compress / cuSZp_decompress). The
 * NVIDIA devcontainer pins cuSZp-V3.0.0.
 */
class Cuszp : public Compressor {
 public:
  /**
   * @param eb absolute error bound (default 1e-3).
   * @param mode cuSZp encoding mode (default CUSZP_MODE_OUTLIER).
   */
  explicit Cuszp(float eb = 1e-3f, cuszp_mode_t mode = CUSZP_MODE_OUTLIER)
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

    cudaStream_t stream = nullptr;
    // Use the caller's pooled stream when one is set. Creating a stream per
    // call is not just overhead: a cudaStreamCreate against a resident
    // spinning kernel has deadlocked this runtime before, and on the
    // device-to-device fault path the compressor runtime has already handed
    // us a slot stream via SetGpuStreamForThread.
    stream = CodecStream();
    float *d_in = nullptr;
    bool free_in = false;
    unsigned char *d_cmp = nullptr;  // temp worst-case device output buffer
    bool ok = false;
    do {
      d_in = static_cast<float *>(
          ToDeviceInput(input, input_size, stream, &free_in));
      if (d_in == nullptr) break;

      // cuSZp writes into a caller-provided device buffer; size it for the
      // worst case (no larger than the original data).
      if (cudaMalloc(&d_cmp, input_size) != cudaSuccess) break;

      size_t cmp_size = 0;
      uint3 dims = {0, 0, 0};  // ignored for 1D
      cuSZp_compress(d_in, d_cmp, n, &cmp_size, eb_, CUSZP_DIM_1D, dims,
                     CUSZP_TYPE_FLOAT, mode_, stream);
      if (cudaStreamSynchronize(stream) != cudaSuccess) break;
      if (cudaGetLastError() != cudaSuccess || cmp_size == 0) break;

      // How many bytes of the compressed buffer must be retained so that
      // Decompress can read everything cuSZp's decompressor touches. cuSZp's
      // reported cmp_size is the meaningful compressed size for MULTI-block
      // inputs, but for a SINGLE-block input (nbEle <= kBlockElems) cuSZp's
      // cross-block offset sync degenerates and cmp_size omits the payload
      // (it reports only the fixed per-group "rate" region). The decompressor
      // still reads the payload that physically follows, which cuSZp does not
      // expose a size for -- so for the single-block case we conservatively
      // retain the whole compress buffer (<= the original size). Net effect:
      // full compression for blobs > kBlockElems*4 bytes (the common case for
      // GPU-resident data); small blobs round-trip correctly but uncompressed.
      const size_t retain = (n > kBlockElems) ? cmp_size : input_size;

      const size_t total = sizeof(Prefix) + retain;
      if (total > output_size) break;  // caller buffer too small

      Prefix prefix;
      std::memset(&prefix, 0, sizeof(prefix));
      prefix.magic = kMagic;
      prefix.mode = static_cast<uint32_t>(mode_);
      prefix.eb = eb_;
      prefix.elems = static_cast<uint64_t>(n);
      prefix.cmp_size = static_cast<uint64_t>(cmp_size);  // decompress arg

      // Frame: [prefix][cuSZp stream].
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
      if (cudaMemcpyAsync(out + sizeof(Prefix), d_cmp, retain, kind, stream) !=
          cudaSuccess) {
        break;
      }
      if (cudaStreamSynchronize(stream) != cudaSuccess) break;
      output_size = total;
      ok = true;
    } while (false);

    if (d_cmp != nullptr) cudaFree(d_cmp);
    if (free_in) cudaFree(d_in);
    return ok;
  }

  bool Decompress(void *output, size_t &output_size, void *input,
                  size_t input_size) override {
    if (output == nullptr || input == nullptr || input_size < sizeof(Prefix)) {
      return false;
    }

    cudaStream_t stream = nullptr;
    // Use the caller's pooled stream when one is set. Creating a stream per
    // call is not just overhead: a cudaStreamCreate against a resident
    // spinning kernel has deadlocked this runtime before, and on the
    // device-to-device fault path the compressor runtime has already handed
    // us a slot stream via SetGpuStreamForThread.
    stream = CodecStream();
    float *d_out = nullptr;
    bool free_out = false;
    unsigned char *d_stream = nullptr;
    bool free_stream = false;
    bool ok = false;
    do {
      Prefix prefix;
      if (IsDeviceAccessible(input)) {
        // STREAM-ORDERED, and it must be. The caller stages the compressed
        // bytes into device memory with an ASYNC H2D on its own slot stream
        // and then calls us; a plain cudaMemcpy here is ordered against the
        // default stream, not that one, so it raced the staging copy and read
        // zeros -- prefix.magic came back 0 and every page failed to decode.
        // nvcomp never tripped this because it is handed its sizes as
        // arguments and never reads its own input from the host.
        cudaError_t pe = cudaMemcpyAsync(&prefix, input, sizeof(Prefix),
                                         cudaMemcpyDeviceToHost, stream);
        if (pe == cudaSuccess) {
          pe = cudaStreamSynchronize(stream);
        }
        if (pe != cudaSuccess) {
          CUSZP_TRACE("prefix D2H failed: %s", cudaGetErrorString(pe));
          break;
        }
      } else {
        std::memcpy(&prefix, input, sizeof(Prefix));
      }
      if (prefix.magic != kMagic) {
        CUSZP_TRACE("magic mismatch: got %x want %x (input_size=%zu dev=%d)",
                    prefix.magic, kMagic, input_size,
                    (int)IsDeviceAccessible(input));
        break;  // not one of our blobs
      }

      const size_t n = static_cast<size_t>(prefix.elems);
      if (n * sizeof(float) > output_size) {
        CUSZP_TRACE("output too small: need %zu have %zu", n * sizeof(float),
                    output_size);
        break;
      }
      // `avail` = compressed bytes physically present in the blob (what Compress
      // retained: cuSZp's cmp_size for multi-block, or the full buffer for the
      // single-block case). `cmp_arg` = the cmp_size value cuSZp itself reported,
      // which is what its decompressor expects as the size argument.
      const size_t avail = input_size - sizeof(Prefix);
      const size_t cmp_arg = static_cast<size_t>(prefix.cmp_size);

      const bool out_is_device = IsDeviceAccessible(output);
      if (out_is_device) {
        d_out = static_cast<float *>(output);
      } else {
        if (cudaMalloc(&d_out, n * sizeof(float)) != cudaSuccess) break;
        free_out = true;
      }

      // The compressed stream must be device-resident for cuSZp.
      uint8_t *stream_src = static_cast<uint8_t *>(input) + sizeof(Prefix);
      if (IsDeviceAccessible(input)) {
        d_stream = stream_src;
      } else {
        if (cudaMalloc(&d_stream, avail) != cudaSuccess) break;
        free_stream = true;
        if (cudaMemcpyAsync(d_stream, stream_src, avail,
                            cudaMemcpyHostToDevice, stream) != cudaSuccess) {
          break;
        }
      }

      uint3 dims = {0, 0, 0};
      cuSZp_decompress(d_out, d_stream, n, cmp_arg, prefix.eb, CUSZP_DIM_1D,
                       dims, CUSZP_TYPE_FLOAT,
                       static_cast<cuszp_mode_t>(prefix.mode), stream);
      const cudaError_t se = cudaStreamSynchronize(stream);
      const cudaError_t le = cudaGetLastError();
      bool decoded = se == cudaSuccess && le == cudaSuccess;
      if (!decoded) {
        CUSZP_TRACE("decode failed: sync=%s last=%s n=%zu cmp_arg=%zu avail=%zu",
                    cudaGetErrorString(se), cudaGetErrorString(le), n, cmp_arg,
                    avail);
      }
      if (decoded) {
        if (!out_is_device) {
          decoded = cudaMemcpy(output, d_out, n * sizeof(float),
                               cudaMemcpyDeviceToHost) == cudaSuccess;
        }
        if (decoded) {
          output_size = n * sizeof(float);
          ok = true;
        }
      }
    } while (false);

    if (free_stream) cudaFree(d_stream);
    if (free_out) cudaFree(d_out);
    return ok;
  }

  /** Set the absolute error bound. */
  /**
   * The stream this codec runs on. NEVER creates one.
   *
   * Creating a stream per operation was both a correctness and a performance
   * hazard: cudaStreamCreate has deadlocked this runtime against a resident
   * consumer kernel, and it is pure overhead on a path that already has a
   * stream to use. The caller supplies one through SetGpuStreamForThread; with
   * no override we run on the default stream, which is always valid in the
   * caller's context. Adopting the PROCESS-WIDE override is deliberately not
   * done here -- it belongs to another CUDA context and using it silently
   * compresses zeros.
   */
  static cudaStream_t CodecStream() {
    return static_cast<cudaStream_t>(GetGpuCodecStreamThreadOnly());
  }

  /**
   * Batched compress: issue every job, then synchronize ONCE.
   *
   * cuSZp has no batched kernel, so this cannot fuse the launches, but the
   * per-job synchronize is what actually hurt: each sync had to win a slot
   * against the consumer kernel's relaunch loop, which measured 22 decodes in
   * 280 seconds. Issuing the whole batch back-to-back and waiting once turns
   * N of those contests into one.
   */
  bool CompressBatch(CompressJob *jobs, size_t n) override {
    return RunBatch(jobs, n, /*compress=*/true);
  }

  /** Batched decompress. See CompressBatch. */
  bool DecompressBatch(CompressJob *jobs, size_t n) override {
    return RunBatch(jobs, n, /*compress=*/false);
  }

  /**
   * Phased batch runner.
   *
   * DECOMPRESS is native: all prefixes are read, ONE wait, all decodes are
   * launched, ONE wait. Two synchronizes for the whole batch instead of two
   * per job.
   *
   * STAGING IS BY RESIDENCY, NOT BY HABIT. A device `src` is decoded from
   * where it lies and a device `dst` is decoded straight into, so the common
   * case -- a compressed page on a device tier faulting into a device page
   * cache -- allocates nothing and copies nothing. A scratch slab is created
   * only if some job actually has a host pointer, and only large enough for
   * those jobs.
   *
   * COMPRESS still uses the serial fallback. cuSZp reports its compressed size
   * through a host out-parameter that is only valid after the kernel has run,
   * so batching it needs a second issue phase for the prefix writes; the
   * interface is in place for that and decompress is where the measured cost
   * was.
   */
  bool RunBatch(CompressJob *jobs, size_t n, bool compress) {
    if (compress || n == 0) {
      return RunBatchSerial(jobs, n, compress);
    }
    cudaStream_t stream = CodecStream();
    std::vector<Prefix> pre(n);
    std::vector<char> host_pre(n * sizeof(Prefix));

    // Phase 0 -- every prefix, then ONE wait.
    for (size_t i = 0; i < n; ++i) {
      jobs[i].ok = false;
      jobs[i].out_size = 0;
      if (jobs[i].src == nullptr || jobs[i].dst == nullptr ||
          jobs[i].src_size < sizeof(Prefix)) {
        continue;
      }
      if (IsDeviceAccessible(const_cast<void *>(jobs[i].src))) {
        cudaMemcpyAsync(&pre[i], jobs[i].src, sizeof(Prefix),
                        cudaMemcpyDeviceToHost, stream);
      } else {
        std::memcpy(&pre[i], jobs[i].src, sizeof(Prefix));
      }
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) return false;

    // Size a scratch slab for the HOST-side jobs only; all-device batches get
    // no slab at all.
    size_t slab_need = 0;
    for (size_t i = 0; i < n; ++i) {
      if (pre[i].magic != kMagic) continue;
      const size_t nb = static_cast<size_t>(pre[i].elems) * sizeof(float);
      if (nb == 0 || nb > jobs[i].dst_capacity) continue;
      if (!IsDeviceAccessible(const_cast<void *>(jobs[i].src))) {
        slab_need += jobs[i].src_size;
      }
      if (!IsDeviceAccessible(jobs[i].dst)) slab_need += nb;
    }
    char *slab = nullptr;
    if (slab_need > 0 && cudaMalloc(&slab, slab_need) != cudaSuccess) {
      return false;
    }
    size_t slab_off = 0;
    std::vector<void *> host_dst(n, nullptr);
    std::vector<size_t> host_dst_bytes(n, 0);

    // Phase 1 -- launch every decode, then ONE wait.
    for (size_t i = 0; i < n; ++i) {
      if (pre[i].magic != kMagic) continue;
      const size_t elems = static_cast<size_t>(pre[i].elems);
      const size_t nb = elems * sizeof(float);
      if (elems == 0 || nb > jobs[i].dst_capacity) continue;

      const unsigned char *d_stream = nullptr;
      if (IsDeviceAccessible(const_cast<void *>(jobs[i].src))) {
        d_stream = static_cast<const unsigned char *>(jobs[i].src) +
                   sizeof(Prefix);   // decode in place: no staging
      } else {
        char *st = slab + slab_off;
        slab_off += jobs[i].src_size;
        cudaMemcpyAsync(st, jobs[i].src, jobs[i].src_size,
                        cudaMemcpyHostToDevice, stream);
        d_stream =
            reinterpret_cast<const unsigned char *>(st) + sizeof(Prefix);
      }

      float *d_out = nullptr;
      if (IsDeviceAccessible(jobs[i].dst)) {
        d_out = static_cast<float *>(jobs[i].dst);  // decode straight in
      } else {
        d_out = reinterpret_cast<float *>(slab + slab_off);
        slab_off += nb;
        host_dst[i] = jobs[i].dst;
        host_dst_bytes[i] = nb;
      }

      uint3 dims = {0, 0, 0};
      cuSZp_decompress(d_out, const_cast<unsigned char *>(d_stream), elems,
                       static_cast<size_t>(pre[i].cmp_size), pre[i].eb,
                       CUSZP_DIM_1D, dims, CUSZP_TYPE_FLOAT,
                       static_cast<cuszp_mode_t>(pre[i].mode), stream);
      jobs[i].ok = true;
      jobs[i].out_size = nb;
      if (host_dst[i] != nullptr) {
        cudaMemcpyAsync(host_dst[i], d_out, nb, cudaMemcpyDeviceToHost, stream);
      }
    }
    const bool synced = cudaStreamSynchronize(stream) == cudaSuccess &&
                        cudaGetLastError() == cudaSuccess;
    if (slab != nullptr) cudaFree(slab);
    bool all = true;
    for (size_t i = 0; i < n; ++i) {
      if (!synced) {
        jobs[i].ok = false;
        jobs[i].out_size = 0;
      }
      all = all && jobs[i].ok;
    }
    return all;
  }

  void SetErrorBound(float eb) { eb_ = eb; }
  /** Get the absolute error bound. */
  float GetErrorBound() const { return eb_; }

 private:
  static constexpr uint32_t kMagic = 0x435A5370u;  // "pSZC"
  // Elements processed per cuSZp thread block (tblock_size 32 * thread_chunk
  // 1024). Inputs with nbEle <= this are single-block; see the retain logic in
  // Compress for why that case is handled specially.
  static constexpr size_t kBlockElems = 32 * 1024;

  // Self-describing prefix carried ahead of the cuSZp codestream. 32 bytes,
  // 8-aligned so the codestream that follows stays aligned for cuSZp.
  struct Prefix {
    uint32_t magic;
    uint32_t mode;      // cuszp_mode_t used at compression
    float eb;           // absolute error bound used at compression
    uint32_t pad;       // keep the 64-bit fields 8-aligned
    uint64_t elems;     // float element count
    uint64_t cmp_size;  // cuSZp's reported cmp_size (decompress size argument)
  };

  /** See ctp::Cusz::IsDeviceAccessible. */
  static bool IsDeviceAccessible(const void *ptr) {
    cudaPointerAttributes attr;
    if (cudaPointerGetAttributes(&attr, ptr) != cudaSuccess) {
      cudaGetLastError();  // reset the sticky error from the failed query
      return false;
    }
    return attr.type == cudaMemoryTypeDevice ||
           attr.type == cudaMemoryTypeManaged;
  }

  /** See ctp::Cusz::ToDeviceInput. */
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

  float eb_;          // absolute error bound
  cuszp_mode_t mode_;  // cuSZp encoding mode
};

}  // namespace ctp

#endif  // CTP_ENABLE_COMPRESS && CTP_ENABLE_CUSZP

#endif  // CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_CUSZP_H_
