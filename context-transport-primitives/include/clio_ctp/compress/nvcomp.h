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

#ifndef CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_NvComp_H_
#define CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_NvComp_H_

#if CTP_ENABLE_COMPRESS && CTP_ENABLE_NVCOMP

#include <cuda_runtime.h>
#include <mutex>
#include <nvcomp/ans.hpp>
#include <nvcomp/deflate.hpp>
#include <nvcomp/gdeflate.hpp>
#include <nvcomp/lz4.hpp>
#include <nvcomp/nvcompManagerFactory.hpp>
#include <nvcomp/snappy.hpp>
#include <nvcomp/zstd.hpp>

#include <cstdint>
#include <memory>

#include "compress.h"

namespace ctp {

/**
 * Supported nvcomp GPU compression algorithms. All are general-purpose,
 * byte-oriented lossless codecs that fit the byte-stream Compressor interface;
 * each is run with nvcomp's default options (no per-algorithm tuning exposed).
 * (nvcomp's GZIP manager is decompression-only, so it is intentionally absent;
 * use DEFLATE for GPU-side deflate-stream compression.)
 */
enum class NvCompAlgo {
  LZ4,
  SNAPPY,
  ZSTD,
  GDEFLATE,
  DEFLATE,
  ANS,
};

/**
 * GPU compressor backed by NVIDIA nvcomp's high-level Manager API.
 *
 * Implements the synchronous ctp::Compressor interface and runs the nvcomp
 * manager on a private CUDA stream, synchronizing before returning. Buffers are
 * handled adaptively: if a pointer already refers to GPU-accessible memory
 * (device or UVM/managed), it is used in place (zero-copy); otherwise the data
 * is staged through a temporary device buffer (H2D for inputs, D2H for outputs).
 * This keeps it correct for host callers while delivering the zero-copy path
 * automatically whenever the data is already resident on the GPU.
 *
 * The compressed bitstream uses NVCOMP_NATIVE (self-describing) format, so
 * Decompress reconstructs the manager directly from the compressed bytes.
 */
class NvComp : public Compressor {
 public:
  explicit NvComp(NvCompAlgo algo = NvCompAlgo::LZ4) : algo_(algo) {}

 private:
  /**
   * SINGLE-CHUNK BATCHED engine — no nvcomp Manager anywhere on this path.
   *
   * The Manager API cannot be used to service an on-GPU page fault: it
   * allocates and frees PINNED HOST scratch internally (create_manager /
   * configure_* / dtor), and cudaHostAlloc + cudaFreeHost implicitly
   * synchronize the device. A fault-service thread doing that blocks until
   * the GPU drains, but the GPU cannot drain because the faulting kernel is
   * spin-waiting for exactly this decompression. Observed directly: the
   * service thread parked in cudaFreeHost inside libnvcomp.
   *
   * The batched C API takes preallocated device scratch and never touches
   * host memory, so the whole fault path becomes: D2D fetch -> batched
   * decompress on a non-blocking stream -> HBM slot. Everything below is
   * allocated once per algorithm at first use (compression time, GPU idle)
   * and reused under a mutex.
   *
   * Layout: one chunk per call. The device-side argument arrays (pointers,
   * sizes, status) live in a single device allocation.
   */
  struct Engine {
    std::mutex mtx;
    cudaStream_t stream = nullptr;
    void *temp = nullptr;          // device scratch for (de)compression
    size_t temp_bytes = 0;
    void **d_ptrs = nullptr;       // [0]=in ptr, [1]=out ptr
    size_t *d_sizes = nullptr;     // [0]=in size, [1]=out cap, [2]=out actual
    nvcompStatus_t *d_status = nullptr;
    size_t chunk_cap = 0;          // uncompressed chunk size these were sized for
    bool ready = false;
  };

  /** Per-algorithm engine, sized for `chunk` uncompressed bytes. */
  Engine &GetEngine(size_t chunk) {
    static Engine eng[6];
    Engine &e = eng[static_cast<int>(algo_)];
    std::lock_guard<std::mutex> lk(e.mtx);
    if (e.stream == nullptr) {
      // Non-blocking: a default-flag stream synchronizes with the legacy
      // stream, i.e. with the very kernel waiting on this decompress.
      if (cudaStreamCreateWithFlags(&e.stream, cudaStreamNonBlocking) !=
          cudaSuccess) {
        return e;
      }
    }
    if (e.chunk_cap >= chunk && e.ready) {
      return e;
    }
    // (Re)size scratch. Happens at warmup, never per fault.
    if (e.temp) { cudaFree(e.temp); e.temp = nullptr; }
    if (e.d_ptrs) { cudaFree(e.d_ptrs); e.d_ptrs = nullptr; }
    if (e.d_sizes) { cudaFree(e.d_sizes); e.d_sizes = nullptr; }
    if (e.d_status) { cudaFree(e.d_status); e.d_status = nullptr; }
    e.ready = false;
    size_t ctmp = 0, dtmp = 0;
    if (!BatchedTempSizes(chunk, &ctmp, &dtmp)) return e;
    e.temp_bytes = ctmp > dtmp ? ctmp : dtmp;
    if (cudaMalloc(&e.temp, e.temp_bytes) != cudaSuccess) return e;
    if (cudaMalloc(&e.d_ptrs, 2 * sizeof(void *)) != cudaSuccess) return e;
    if (cudaMalloc(&e.d_sizes, 3 * sizeof(size_t)) != cudaSuccess) return e;
    if (cudaMalloc(&e.d_status, sizeof(nvcompStatus_t)) != cudaSuccess) return e;
    e.chunk_cap = chunk;
    e.ready = true;
    return e;
  }

  /** True if `ptr` is GPU-accessible (device or managed). */
  static bool IsDeviceAccessible(const void *ptr) {
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
    if (err != cudaSuccess) {
      cudaGetLastError();  // reset the sticky error from the failed query
      return false;
    }
    return attr.type == cudaMemoryTypeDevice ||
           attr.type == cudaMemoryTypeManaged;
  }

  /**
   * Return a device pointer holding `size` bytes of `input`. If `input` is
   * already GPU-accessible it is used in place (*owned = false); otherwise a
   * device buffer is allocated and the data copied H2D on `stream`
   * (*owned = true). Returns nullptr on failure (*owned = false).
   */
  static uint8_t *ToDeviceInput(void *input, size_t size, cudaStream_t stream,
                                bool *owned) {
    *owned = false;
    if (IsDeviceAccessible(input)) {
      return static_cast<uint8_t *>(input);
    }
    uint8_t *d = nullptr;
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

  /** Temp-scratch sizes for a single chunk of `chunk` uncompressed bytes. */
  bool BatchedTempSizes(size_t chunk, size_t *ctmp, size_t *dtmp) {
    nvcompStatus_t sc = nvcompErrorInternal, sd = nvcompErrorInternal;
    switch (algo_) {
      case NvCompAlgo::LZ4:
        sc = nvcompBatchedLZ4CompressGetTempSize(1, chunk, nvcompBatchedLZ4DefaultOpts, ctmp);
        sd = nvcompBatchedLZ4DecompressGetTempSize(1, chunk, dtmp);
        break;
      case NvCompAlgo::SNAPPY:
        sc = nvcompBatchedSnappyCompressGetTempSize(1, chunk, nvcompBatchedSnappyDefaultOpts, ctmp);
        sd = nvcompBatchedSnappyDecompressGetTempSize(1, chunk, dtmp);
        break;
      case NvCompAlgo::ZSTD:
        sc = nvcompBatchedZstdCompressGetTempSize(1, chunk, nvcompBatchedZstdDefaultOpts, ctmp);
        sd = nvcompBatchedZstdDecompressGetTempSize(1, chunk, dtmp);
        break;
      case NvCompAlgo::GDEFLATE:
        sc = nvcompBatchedGdeflateCompressGetTempSize(1, chunk, nvcompBatchedGdeflateDefaultOpts, ctmp);
        sd = nvcompBatchedGdeflateDecompressGetTempSize(1, chunk, dtmp);
        break;
      case NvCompAlgo::DEFLATE:
        sc = nvcompBatchedDeflateCompressGetTempSize(1, chunk, nvcompBatchedDeflateDefaultOpts, ctmp);
        sd = nvcompBatchedDeflateDecompressGetTempSize(1, chunk, dtmp);
        break;
      case NvCompAlgo::ANS:
        sc = nvcompBatchedANSCompressGetTempSize(1, chunk, nvcompBatchedANSDefaultOpts, ctmp);
        sd = nvcompBatchedANSDecompressGetTempSize(1, chunk, dtmp);
        break;
    }
    return sc == nvcompSuccess && sd == nvcompSuccess;
  }

  /** Worst-case compressed bytes for one `chunk`-byte chunk. */
  bool BatchedMaxOut(size_t chunk, size_t *maxout) {
    nvcompStatus_t st = nvcompErrorInternal;
    switch (algo_) {
      case NvCompAlgo::LZ4:
        st = nvcompBatchedLZ4CompressGetMaxOutputChunkSize(chunk, nvcompBatchedLZ4DefaultOpts, maxout); break;
      case NvCompAlgo::SNAPPY:
        st = nvcompBatchedSnappyCompressGetMaxOutputChunkSize(chunk, nvcompBatchedSnappyDefaultOpts, maxout); break;
      case NvCompAlgo::ZSTD:
        st = nvcompBatchedZstdCompressGetMaxOutputChunkSize(chunk, nvcompBatchedZstdDefaultOpts, maxout); break;
      case NvCompAlgo::GDEFLATE:
        st = nvcompBatchedGdeflateCompressGetMaxOutputChunkSize(chunk, nvcompBatchedGdeflateDefaultOpts, maxout); break;
      case NvCompAlgo::DEFLATE:
        st = nvcompBatchedDeflateCompressGetMaxOutputChunkSize(chunk, nvcompBatchedDeflateDefaultOpts, maxout); break;
      case NvCompAlgo::ANS:
        st = nvcompBatchedANSCompressGetMaxOutputChunkSize(chunk, nvcompBatchedANSDefaultOpts, maxout); break;
    }
    return st == nvcompSuccess;
  }

  nvcompStatus_t BatchedCompressAsync(Engine &e, size_t in_bytes) {
    switch (algo_) {
      case NvCompAlgo::LZ4:
        return nvcompBatchedLZ4CompressAsync((const void * const *) e.d_ptrs, e.d_sizes,
            e.chunk_cap, 1, e.temp, e.temp_bytes, (void * const *) (e.d_ptrs + 1),
            e.d_sizes + 2, nvcompBatchedLZ4DefaultOpts, e.stream);
      case NvCompAlgo::SNAPPY:
        return nvcompBatchedSnappyCompressAsync((const void * const *) e.d_ptrs, e.d_sizes,
            e.chunk_cap, 1, e.temp, e.temp_bytes, (void * const *) (e.d_ptrs + 1),
            e.d_sizes + 2, nvcompBatchedSnappyDefaultOpts, e.stream);
      case NvCompAlgo::ZSTD:
        return nvcompBatchedZstdCompressAsync((const void * const *) e.d_ptrs, e.d_sizes,
            e.chunk_cap, 1, e.temp, e.temp_bytes, (void * const *) (e.d_ptrs + 1),
            e.d_sizes + 2, nvcompBatchedZstdDefaultOpts, e.stream);
      case NvCompAlgo::GDEFLATE:
        return nvcompBatchedGdeflateCompressAsync((const void * const *) e.d_ptrs, e.d_sizes,
            e.chunk_cap, 1, e.temp, e.temp_bytes, (void * const *) (e.d_ptrs + 1),
            e.d_sizes + 2, nvcompBatchedGdeflateDefaultOpts, e.stream);
      case NvCompAlgo::DEFLATE:
        return nvcompBatchedDeflateCompressAsync((const void * const *) e.d_ptrs, e.d_sizes,
            e.chunk_cap, 1, e.temp, e.temp_bytes, (void * const *) (e.d_ptrs + 1),
            e.d_sizes + 2, nvcompBatchedDeflateDefaultOpts, e.stream);
      case NvCompAlgo::ANS:
        return nvcompBatchedANSCompressAsync((const void * const *) e.d_ptrs, e.d_sizes,
            e.chunk_cap, 1, e.temp, e.temp_bytes, (void * const *) (e.d_ptrs + 1),
            e.d_sizes + 2, nvcompBatchedANSDefaultOpts, e.stream);
    }
    return nvcompErrorInternal;
  }

  nvcompStatus_t BatchedDecompressAsync(Engine &e) {
    switch (algo_) {
      case NvCompAlgo::LZ4:
        return nvcompBatchedLZ4DecompressAsync((const void * const *) e.d_ptrs, e.d_sizes,
            e.d_sizes + 1, e.d_sizes + 2, 1, e.temp, e.temp_bytes,
            (void * const *) (e.d_ptrs + 1), e.d_status, e.stream);
      case NvCompAlgo::SNAPPY:
        return nvcompBatchedSnappyDecompressAsync((const void * const *) e.d_ptrs, e.d_sizes,
            e.d_sizes + 1, e.d_sizes + 2, 1, e.temp, e.temp_bytes,
            (void * const *) (e.d_ptrs + 1), e.d_status, e.stream);
      case NvCompAlgo::ZSTD:
        return nvcompBatchedZstdDecompressAsync((const void * const *) e.d_ptrs, e.d_sizes,
            e.d_sizes + 1, e.d_sizes + 2, 1, e.temp, e.temp_bytes,
            (void * const *) (e.d_ptrs + 1), e.d_status, e.stream);
      case NvCompAlgo::GDEFLATE:
        return nvcompBatchedGdeflateDecompressAsync((const void * const *) e.d_ptrs, e.d_sizes,
            e.d_sizes + 1, e.d_sizes + 2, 1, e.temp, e.temp_bytes,
            (void * const *) (e.d_ptrs + 1), e.d_status, e.stream);
      case NvCompAlgo::DEFLATE:
        return nvcompBatchedDeflateDecompressAsync((const void * const *) e.d_ptrs, e.d_sizes,
            e.d_sizes + 1, e.d_sizes + 2, 1, e.temp, e.temp_bytes,
            (void * const *) (e.d_ptrs + 1), e.d_status, e.stream);
      case NvCompAlgo::ANS:
        return nvcompBatchedANSDecompressAsync((const void * const *) e.d_ptrs, e.d_sizes,
            e.d_sizes + 1, e.d_sizes + 2, 1, e.temp, e.temp_bytes,
            (void * const *) (e.d_ptrs + 1), e.d_status, e.stream);
    }
    return nvcompErrorInternal;
  }

  /**
   * Wire format: [u64 uncompressed_size][u64 compressed_size][payload].
   * Self-describing so Decompress needs no manager to parse a header.
   */
  struct BatchHdr { uint64_t raw_size; uint64_t comp_size; };

 public:
  bool Compress(void *output, size_t &output_size, void *input,
                size_t input_size) override {
    if (input_size == 0) return false;
    Engine &e = GetEngine(input_size);
    std::lock_guard<std::mutex> lk(e.mtx);
    if (!e.ready) return false;

    size_t maxout = 0;
    if (!BatchedMaxOut(input_size, &maxout)) return false;
    if (output_size < sizeof(BatchHdr) + maxout) {
      // Caller's buffer may still fit the ACTUAL output; compress to scratch.
    }
    bool free_in = false, free_out = false, ok = false;
    uint8_t *d_in = nullptr, *d_out = nullptr;
    do {
      d_in = ToDeviceInput(input, input_size, e.stream, &free_in);
      if (!d_in) break;
      if (cudaMalloc(&d_out, maxout) != cudaSuccess) break;
      free_out = true;
      void *hp[2] = {d_in, d_out};
      size_t hs[3] = {input_size, maxout, 0};
      if (cudaMemcpyAsync(e.d_ptrs, hp, sizeof(hp), cudaMemcpyHostToDevice, e.stream) != cudaSuccess) break;
      if (cudaMemcpyAsync(e.d_sizes, hs, sizeof(hs), cudaMemcpyHostToDevice, e.stream) != cudaSuccess) break;
      if (BatchedCompressAsync(e, input_size) != nvcompSuccess) break;
      size_t comp = 0;
      if (cudaMemcpyAsync(&comp, e.d_sizes + 2, sizeof(size_t), cudaMemcpyDeviceToHost, e.stream) != cudaSuccess) break;
      if (cudaStreamSynchronize(e.stream) != cudaSuccess) break;
      if (comp == 0 || sizeof(BatchHdr) + comp > output_size) break;
      BatchHdr h{(uint64_t) input_size, (uint64_t) comp};
      const bool out_dev = IsDeviceAccessible(output);
      cudaMemcpyKind k = out_dev ? cudaMemcpyHostToDevice : cudaMemcpyHostToHost;
      if (cudaMemcpyAsync(output, &h, sizeof(h), k, e.stream) != cudaSuccess) break;
      if (cudaMemcpyAsync((char *) output + sizeof(h), d_out, comp,
                          out_dev ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost,
                          e.stream) != cudaSuccess) break;
      if (cudaStreamSynchronize(e.stream) != cudaSuccess) break;
      output_size = sizeof(BatchHdr) + comp;
      ok = true;
    } while (false);
    if (free_in) cudaFree(d_in);
    if (free_out) cudaFree(d_out);
    // Leave a CLEAN context for the next caller: an early `break` above can
    // abandon queued async work on our private stream, and a failed
    // cudaPointerGetAttributes / nvcomp call leaves a sticky error that would
    // otherwise surface inside an unrelated codec later in the process.
    cudaStreamSynchronize(e.stream);
    cudaGetLastError();
    return ok;
  }

  bool Decompress(void *output, size_t &output_size, void *input,
                  size_t input_size) override {
    if (input_size < sizeof(BatchHdr)) return false;
    // Header first — cheap, and tells us the chunk size to size the engine.
    BatchHdr h{};
    const bool in_dev = IsDeviceAccessible(input);
    if (in_dev) {
      if (cudaMemcpy(&h, input, sizeof(h), cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    } else {
      memcpy(&h, input, sizeof(h));
    }
    if (h.raw_size == 0 || h.raw_size > output_size) return false;

    Engine &e = GetEngine(h.raw_size);
    std::lock_guard<std::mutex> lk(e.mtx);
    if (!e.ready) return false;

    bool free_in = false, free_out = false, ok = false;
    uint8_t *d_in = nullptr, *d_out = nullptr;
    do {
      // Compressed payload must be device-resident for the batched API.
      if (in_dev) {
        d_in = (uint8_t *) input + sizeof(BatchHdr);
      } else {
        if (cudaMalloc(&d_in, h.comp_size) != cudaSuccess) break;
        free_in = true;
        if (cudaMemcpyAsync(d_in, (const char *) input + sizeof(BatchHdr), h.comp_size,
                            cudaMemcpyHostToDevice, e.stream) != cudaSuccess) break;
      }
      const bool out_dev = IsDeviceAccessible(output);
      if (out_dev) {
        d_out = (uint8_t *) output;   // ZERO-COPY into the vector's HBM slot
      } else {
        if (cudaMalloc(&d_out, h.raw_size) != cudaSuccess) break;
        free_out = true;
      }
      void *hp[2] = {d_in, d_out};
      size_t hs[3] = {(size_t) h.comp_size, (size_t) h.raw_size, 0};
      if (cudaMemcpyAsync(e.d_ptrs, hp, sizeof(hp), cudaMemcpyHostToDevice, e.stream) != cudaSuccess) break;
      if (cudaMemcpyAsync(e.d_sizes, hs, sizeof(hs), cudaMemcpyHostToDevice, e.stream) != cudaSuccess) break;
      if (BatchedDecompressAsync(e) != nvcompSuccess) break;
      if (cudaStreamSynchronize(e.stream) != cudaSuccess) break;
      if (!out_dev &&
          cudaMemcpy(output, d_out, h.raw_size, cudaMemcpyDeviceToHost) != cudaSuccess) break;
      output_size = (size_t) h.raw_size;
      ok = true;
    } while (false);
    if (free_in) cudaFree(d_in);
    if (free_out) cudaFree(d_out);
    cudaStreamSynchronize(e.stream);   // see Compress: clean-context contract
    cudaGetLastError();
    return ok;
  }

 private:
  // (The nvcomp Manager path was removed: its create/configure/destroy
  // cycle allocates and frees PINNED HOST memory, which implicitly
  // synchronizes the device and deadlocks the GPU page-fault path. The
  // batched C API above uses only preallocated device scratch.)

  NvCompAlgo algo_;
};

}  // namespace ctp

#endif  // CTP_ENABLE_COMPRESS && CTP_ENABLE_NVCOMP

#endif  // CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_NvComp_H_
