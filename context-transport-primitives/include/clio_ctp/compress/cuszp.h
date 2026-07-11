/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#ifndef CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_CUSZP_H_
#define CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_CUSZP_H_

#if CTP_ENABLE_COMPRESS && CTP_ENABLE_CUSZP

#include <cuda_runtime.h>
#include <cuSZp.h>

#include <cstdint>
#include <cstring>
#include <mutex>
#include <vector>

#include "compress.h"

namespace ctp {
namespace cuszp_detail {

// Process-global pinned-host staging pool + pre-warmed stream-ordered mempool.
//
// The compressed GPU-vector fault path decompresses while a caller kernel
// spin-waits ON-DEVICE for that decompress. To not deadlock, the decompress
// must issue ZERO device-synchronizing CUDA calls. Two would otherwise remain:
//   (1) the first cudaMallocAsync (lazy mempool creation) device-syncs, and
//   (2) staging the compressed bytes host->device from PAGEABLE memory makes
//       cudaMemcpyAsync behave synchronously (a device sync).
// This pool fixes both: its constructor warms the mempool and preallocates
// pinned staging buffers, and it is constructed lazily on the FIRST compress --
// which always happens during page eviction (GPU idle), strictly before any
// read fault can occur (a page must be written before it can be faulted back).
// Pageable inputs are then bounced through a pinned buffer so the H2D is a true
// async copy.
struct PinnedPool {
  std::mutex m;
  std::vector<uint8_t *> free_;
  std::vector<uint8_t *> all_;
  size_t buf_ = 0;

  PinnedPool() {
    cudaFree(0);  // ensure a context exists
    // Warm the default stream-ordered mempool so later cudaMallocAsync calls
    // (during an on-device fault) are pure stream ops, not a syncing pool init.
    void *w = nullptr;
    if (cudaMallocAsync(&w, size_t(8) << 20, 0) == cudaSuccess) {
      cudaFreeAsync(w, 0);
      cudaStreamSynchronize(0);
    }
    buf_ = size_t(8) << 20;  // 8 MiB covers typical gpu_vector page sizes
    for (int i = 0; i < 16; ++i) {
      uint8_t *p = nullptr;
      if (cudaMallocHost((void **)&p, buf_) == cudaSuccess) {
        all_.push_back(p);
        free_.push_back(p);
      }
    }
  }

  uint8_t *Acquire(size_t need) {
    if (need > buf_) return nullptr;
    std::lock_guard<std::mutex> g(m);
    if (free_.empty()) return nullptr;
    uint8_t *p = free_.back();
    free_.pop_back();
    return p;
  }
  void Release(uint8_t *p) {
    if (!p) return;
    std::lock_guard<std::mutex> g(m);
    free_.push_back(p);
  }
};

inline PinnedPool &Pool() {
  static PinnedPool pool;  // thread-safe lazy init; first touch = first compress
  return pool;
}

}  // namespace cuszp_detail
}  // namespace ctp

namespace ctp {

/**
 * GPU error-bounded LOSSY float compressor backed by cuSZp.
 *
 * Unlike the byte-oriented lossless codecs (lz4/zstd/nvcomp-*), cuSZp is an
 * error-bounded lossy compressor for 32-bit float fields: it trades a bounded
 * amount of precision (the absolute error bound `eb`) for large ratios on smooth
 * / scientific data, and — critically for the compressed GPU vector — it runs
 * ENTIRELY ON THE GPU (the host API launches its kernels on a stream), so data
 * never has to leave the device to be compressed.
 *
 * Implements the synchronous ctp::Compressor interface. cuSZp requires device
 * pointers, so buffers are handled adaptively like the nvcomp wrapper: a
 * GPU-resident pointer is used in place (zero-copy); a host pointer is staged
 * through a temporary device buffer (H2D in, D2H out). The compressed stream is
 * NOT self-describing about `eb` or element count, so Decompress reuses the same
 * `eb` (from the preset) and derives nbEle from the caller's output size — the
 * runtime already carries the original size in its CompressionHeader.
 *
 * The input byte length must be a whole number of floats.
 */
class Cuszp : public Compressor {
 public:
  explicit Cuszp(float error_bound = 1e-3f) : eb_(error_bound) {}

  bool Compress(void *output, size_t &output_size, void *input,
                size_t input_size) override {
    if (input_size == 0 || (input_size % sizeof(float)) != 0) return false;
    const size_t nbEle = input_size / sizeof(float);
    // Force the pinned pool + mempool warm on the FIRST compress (page eviction,
    // GPU idle) so it exists before any on-device read fault.
    (void)cuszp_detail::Pool();
    cudaStream_t stream = Stream();  // persistent, non-blocking, per-thread
    if (!stream) return false;
    uint8_t *d_in = nullptr, *d_cmp = nullptr, *pin = nullptr;
    bool free_in = false, ok = false;
    do {
      d_in = ToDeviceInput(input, input_size, stream, &free_in, &pin);
      if (!d_in) break;
      // cuSZp worst case is ~the input size; add slack and always use a temp so
      // we can bounds-check the caller's buffer against the real cmp size.
      const size_t cap = input_size + input_size / 8 + 1024;
      if (cudaMallocAsync(&d_cmp, cap, stream) != cudaSuccess) break;
      size_t cmp_size = 0;
      uint3 dims = {0, 0, 0};
      // cuSZp (patched) is fully stream-ordered and syncs only `stream`
      // internally to read back cmp_size, so it never device-synchronizes.
      cuSZp_compress(d_in, d_cmp, nbEle, &cmp_size, eb_, CUSZP_DIM_1D, dims,
                     CUSZP_TYPE_FLOAT, CUSZP_MODE_OUTLIER, stream);
      if (cmp_size == 0 || cmp_size > output_size) break;  // caller buffer small
      cudaMemcpyKind kind = IsDeviceAccessible(output) ? cudaMemcpyDeviceToDevice
                                                       : cudaMemcpyDeviceToHost;
      if (cudaMemcpyAsync(output, d_cmp, cmp_size, kind, stream) != cudaSuccess)
        break;
      if (cudaStreamSynchronize(stream) != cudaSuccess) break;  // this stream only
      output_size = cmp_size;
      ok = true;
    } while (false);
    if (d_cmp) cudaFreeAsync(d_cmp, stream);
    if (free_in) cudaFreeAsync(d_in, stream);
    if (pin) cuszp_detail::Pool().Release(pin);  // safe: stream synced above
    return ok;
  }

  bool Decompress(void *output, size_t &output_size, void *input,
                  size_t input_size) override {
    if (output_size == 0 || (output_size % sizeof(float)) != 0) return false;
    const size_t nbEle = output_size / sizeof(float);
    (void)cuszp_detail::Pool();  // normally already warm from the first compress
    cudaStream_t stream = Stream();
    if (!stream) return false;
    uint8_t *d_cmp = nullptr, *d_dec = nullptr, *pin = nullptr;
    bool free_in = false, ok = false;
    do {
      d_cmp = ToDeviceInput(input, input_size, stream, &free_in, &pin);
      if (!d_cmp) break;
      if (cudaMallocAsync(&d_dec, output_size, stream) != cudaSuccess) break;
      uint3 dims = {0, 0, 0};
      cuSZp_decompress(d_dec, d_cmp, nbEle, input_size, eb_, CUSZP_DIM_1D, dims,
                       CUSZP_TYPE_FLOAT, CUSZP_MODE_OUTLIER, stream);
      cudaMemcpyKind kind = IsDeviceAccessible(output) ? cudaMemcpyDeviceToDevice
                                                       : cudaMemcpyDeviceToHost;
      if (cudaMemcpyAsync(output, d_dec, output_size, kind, stream) !=
          cudaSuccess)
        break;
      // Sync ONLY this stream (not the whole device), so a caller kernel
      // spin-waiting on-device (the gpu_vector page fault) does not deadlock us:
      // the decompress + copy run on `stream`, concurrently with that kernel.
      if (cudaStreamSynchronize(stream) != cudaSuccess) break;
      ok = true;
    } while (false);
    if (d_dec) cudaFreeAsync(d_dec, stream);
    if (free_in) cudaFreeAsync(d_cmp, stream);
    if (pin) cuszp_detail::Pool().Release(pin);  // safe: stream synced above
    return ok;
  }

 private:
  // One persistent non-blocking stream per worker thread. Not the default
  // stream (which serializes with everything) and never destroyed per call.
  // The first use also warms the stream-ordered memory pool so later
  // cudaMallocAsync calls (e.g. during an on-device page fault) are pure
  // stream-ordered ops with no device synchronization.
  static cudaStream_t Stream() {
    static thread_local cudaStream_t s = nullptr;
    if (!s) {
      if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess)
        return nullptr;
      void *warm = nullptr;
      if (cudaMallocAsync(&warm, 1 << 20, s) == cudaSuccess) {
        cudaFreeAsync(warm, s);
        cudaStreamSynchronize(s);
      }
    }
    return s;
  }

  static bool IsDeviceAccessible(const void *ptr) {
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
    if (err != cudaSuccess) {
      cudaGetLastError();  // clear sticky error from the failed query
      return false;
    }
    return attr.type == cudaMemoryTypeDevice ||
           attr.type == cudaMemoryTypeManaged;
  }

  // Stage `input` onto the device. Device/managed pointers are used in place.
  // A PAGEABLE host pointer is bounced through a preallocated PINNED buffer so
  // the H2D is a TRUE async copy (a direct async copy from pageable memory is
  // silently synchronous, which device-syncs and would deadlock an on-device
  // caller). `*pinned_out` receives the borrowed pinned buffer to release AFTER
  // the stream is synchronized.
  static uint8_t *ToDeviceInput(void *input, size_t size, cudaStream_t stream,
                                bool *owned, uint8_t **pinned_out) {
    *owned = false;
    *pinned_out = nullptr;
    if (IsDeviceAccessible(input)) return static_cast<uint8_t *>(input);
    uint8_t *d = nullptr;
    if (cudaMallocAsync(&d, size, stream) != cudaSuccess) return nullptr;
    uint8_t *pin = cuszp_detail::Pool().Acquire(size);
    if (pin) {
      std::memcpy(pin, input, size);  // pageable->pinned on CPU (no device sync)
      if (cudaMemcpyAsync(d, pin, size, cudaMemcpyHostToDevice, stream) !=
          cudaSuccess) {
        cudaFreeAsync(d, stream);
        cuszp_detail::Pool().Release(pin);
        return nullptr;
      }
      *pinned_out = pin;
    } else {
      // Pool exhausted / oversized: fall back to a direct (possibly syncing)
      // copy. Fine for host-orchestrated paths; only a risk if this happens on
      // an on-device fault with the pool full.
      if (cudaMemcpyAsync(d, input, size, cudaMemcpyHostToDevice, stream) !=
          cudaSuccess) {
        cudaFreeAsync(d, stream);
        return nullptr;
      }
    }
    *owned = true;
    return d;
  }

  float eb_;  // absolute error bound
};

}  // namespace ctp

#endif  // CTP_ENABLE_COMPRESS && CTP_ENABLE_CUSZP

#endif  // CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_CUSZP_H_
