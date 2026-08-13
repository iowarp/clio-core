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
#include <nvcomp/ans.hpp>
#include <nvcomp/bitcomp.hpp>
#include <nvcomp/cascaded.hpp>
#include <nvcomp/deflate.hpp>
#include <nvcomp/gdeflate.hpp>
#include <nvcomp/lz4.hpp>
#include <nvcomp/nvcompManagerFactory.hpp>

#include <cstdlib>
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
  CASCADED,
  BITCOMP,
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

  /** @brief Manager-cache counters, mirroring upstream's hit/miss tallies. */
  struct ManagerCacheStats {
    uint64_t hits = 0;
    uint64_t misses = 0;
  };

  /** @brief Snapshot this thread's manager-cache counters. */
  static ManagerCacheStats GetManagerCacheStats() {
    const auto &c = Cache();
    return ManagerCacheStats{c.hits, c.misses};
  }

  /** @brief Drop every cached manager and zero the counters (tests). */
  static void ResetManagerCache() {
    auto &c = Cache();
    for (auto &slot : c.slots) {
      slot.mgr.reset();
      slot.algo = -1;
      slot.tick = 0;
    }
    c.clock = 0;
    c.hits = 0;
    c.misses = 0;
  }


  /**
   * CUDA-event bracket around a codec launch, and nothing else.
   *
   * Enabled only when CLIO_CODEC_KERNEL_TIMING is set: creating and
   * recording two events per call is real overhead, and the number is only
   * wanted when something is being compared. Records on the SAME stream the
   * codec runs on, so the interval covers device work rather than host
   * launch latency. NeuroPress times its equivalent call the same way
   * (gpucompress_compress.cpp:530-551).
   */
  struct KernelTimer {
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;
    bool on = false;

    static bool Enabled() {
      static const bool e = [] {
        const char *v = std::getenv("CLIO_CODEC_KERNEL_TIMING");
        return v && *v;
      }();
      return e;
    }

    explicit KernelTimer(cudaStream_t s) : stream(s) {
      LastCodecKernelMs() = -1.0;
      if (!Enabled() || !s) return;
      if (cudaEventCreate(&start) != cudaSuccess) return;
      if (cudaEventCreate(&stop) != cudaSuccess) {
        cudaEventDestroy(start);
        start = nullptr;
        return;
      }
      on = (cudaEventRecord(start, stream) == cudaSuccess);
    }

    /** Call after the launch; the elapsed time lands in LastCodecKernelMs(). */
    void Stop() {
      if (!on) return;
      if (cudaEventRecord(stop, stream) != cudaSuccess) return;
      if (cudaEventSynchronize(stop) != cudaSuccess) return;
      float ms = 0.0f;
      if (cudaEventElapsedTime(&ms, start, stop) == cudaSuccess) {
        LastCodecKernelMs() = static_cast<double>(ms);
      }
    }

    ~KernelTimer() {
      if (start) cudaEventDestroy(start);
      if (stop) cudaEventDestroy(stop);
    }
  };

  bool Compress(void *output, size_t &output_size, void *input,
                size_t input_size) override {
    // Persistent per-thread stream and cached manager. Both used to be built
    // and torn down on EVERY call, inside the window Runtime::Compress times
    // to produce the model's comp_time label -- so the label carried a cost
    // that is large relative to the compress kernels and roughly independent
    // of chunk size, flattening the time-vs-size relationship the network
    // learns. NeuroPress pays neither per chunk: a persistent context stream
    // and an LRU of managers (gpucompress_pool.cpp:236-271), with its
    // CUDA-event bracket around compress() alone.
    cudaStream_t stream = CachedStream();
    if (!stream) {
      return false;
    }
    uint8_t *d_in = nullptr;
    uint8_t *d_out = nullptr;
    bool free_in = false;
    bool free_out = false;
    bool ok = false;
    do {
      // The nvcomp Manager API reports errors by throwing (and compress() is
      // void, so a throw is its only failure signal); make_shared can also
      // throw std::bad_alloc. Catch everything so no exception escapes (honoring
      // the bool contract) and the cleanup block below always runs.
      try {
        // Input: use directly if already on the GPU, else stage a H2D copy.
        d_in = ToDeviceInput(input, input_size, stream, &free_in);
        if (!d_in) break;

        std::shared_ptr<nvcomp::nvcompManagerBase> mgr =
            GetOrCreateManager(stream);
        if (!mgr) break;
        nvcomp::CompressionConfig cfg = mgr->configure_compression(input_size);

        // Output: write straight into the caller's buffer only if it is a GPU
        // buffer large enough for nvcomp's worst case; otherwise use a temp.
        // (Device-direct requires >= max_compressed_buffer_size, while the temp
        // path only needs to fit the actual compressed size -- this asymmetry
        // is intentional; an in-between device buffer falls back to temp + D2D.)
        const bool out_is_device = IsDeviceAccessible(output);
        if (out_is_device && output_size >= cfg.max_compressed_buffer_size) {
          d_out = static_cast<uint8_t *>(output);
        } else {
          if (cudaMalloc(&d_out, cfg.max_compressed_buffer_size) !=
              cudaSuccess) {
            break;
          }
          free_out = true;
        }

        KernelTimer timer(stream);
        mgr->compress(d_in, d_out, cfg);
        timer.Stop();
        if (cudaStreamSynchronize(stream) != cudaSuccess) break;
        size_t comp_size = mgr->get_compressed_output_size(d_out);
        if (comp_size > output_size) break;  // caller buffer too small

        // Deliver to the caller's buffer if we compressed into a temp.
        if (d_out != static_cast<uint8_t *>(output)) {
          cudaMemcpyKind kind =
              out_is_device ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;
          if (cudaMemcpy(output, d_out, comp_size, kind) != cudaSuccess) break;
        }
        output_size = comp_size;
        ok = true;
      } catch (...) {
        // ok stays false; free_in/free_out (set before any throwing call) make
        // the cleanup below release exactly what was allocated.
      }
    } while (false);
    if (free_in) cudaFree(d_in);
    if (free_out) cudaFree(d_out);
    // The stream is owned by the thread's cache, not by this call.
    return ok;
  }

  bool Decompress(void *output, size_t &output_size, void *input,
                  size_t input_size) override {
    // Same persistent stream as Compress. The MANAGER cannot be cached here
    // and upstream does not cache it either: the NVCOMP_NATIVE bitstream is
    // self-describing, so create_manager derives it from the blob's own bytes
    // (compression_factory.cpp:157). Only the stream is reusable, and it is
    // inside the window Runtime::Decompress times for the decomp head.
    cudaStream_t stream = CachedStream();
    if (!stream) {
      return false;
    }
    uint8_t *d_in = nullptr;
    uint8_t *d_out = nullptr;
    bool free_in = false;
    bool free_out = false;
    bool ok = false;
    do {
      // The nvcomp Manager API reports errors by throwing (create_manager,
      // configure_decompression, and the void decompress() included); catch
      // everything so no exception escapes and the cleanup below always runs.
      try {
        // Input: use directly if already on the GPU, else stage a H2D copy.
        d_in = ToDeviceInput(input, input_size, stream, &free_in);
        if (!d_in) break;

        // create_manager parses the NVCOMP_NATIVE header and syncs the stream.
        std::shared_ptr<nvcomp::nvcompManagerBase> mgr =
            nvcomp::create_manager(d_in, stream);
        if (!mgr) break;
        nvcomp::DecompressionConfig cfg = mgr->configure_decompression(d_in);
        if (cfg.decomp_data_size > output_size) break;  // caller buffer too small

        const bool out_is_device = IsDeviceAccessible(output);
        if (out_is_device) {
          d_out = static_cast<uint8_t *>(output);
        } else {
          if (cudaMalloc(&d_out, cfg.decomp_data_size) != cudaSuccess) break;
          free_out = true;
        }

        KernelTimer timer(stream);
        mgr->decompress(d_out, d_in, cfg);
        timer.Stop();
        if (cudaStreamSynchronize(stream) != cudaSuccess) break;

        if (d_out != static_cast<uint8_t *>(output)) {
          if (cudaMemcpy(output, d_out, cfg.decomp_data_size,
                         cudaMemcpyDeviceToHost) != cudaSuccess) {
            break;
          }
        }
        output_size = cfg.decomp_data_size;
        ok = true;
      } catch (...) {
        // ok stays false; free_in/free_out (set before any throwing call) make
        // the cleanup below release exactly what was allocated.
      }
    } while (false);
    if (free_in) cudaFree(d_in);
    if (free_out) cudaFree(d_out);
    // Stream owned by the thread's cache; do not destroy it here.
    return ok;
  }

 private:
  /** Default per-chunk uncompressed size for nvcomp managers (64 KiB). */
  static constexpr size_t kChunkSize = 1 << 16;

  /**
   * True if `ptr` can be dereferenced by the GPU directly (device or UVM/
   * managed memory). Pinned-host and plain-host pointers return false so the
   * caller stages a copy. A failed query is treated as "not device".
   */
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

  /** Construct the nvcomp manager for the configured algorithm. */
  /**
   * Per-thread manager cache and its persistent stream.
   *
   * NeuroPress keeps this per CompContext -- an LRU of LRU_DEPTH = 3
   * managers keyed by algorithm, on the context's own stream, with hit/miss
   * counters (gpucompress_pool.cpp:236-271, internal.hpp:83-86). Clio has no
   * per-context object at this layer, so the equivalent scope is the worker
   * THREAD: it gives each concurrent flow its own managers and stream, which
   * is the isolation upstream gets from per-context state, without a lock on
   * the compression path. Sharing one manager across threads would not be
   * safe -- an nvcomp manager is bound to the stream it was built on.
   */
  static constexpr int kLruDepth = 3;  // CompContext::LRU_DEPTH

  struct ManagerCache {
    struct Slot {
      std::shared_ptr<nvcomp::nvcompManagerBase> mgr;
      int algo = -1;
      uint64_t tick = 0;
    };
    Slot slots[kLruDepth];
    uint64_t clock = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    cudaStream_t stream = nullptr;

    ~ManagerCache() {
      // Managers must die before the stream they were built on.
      for (auto &s : slots) s.mgr.reset();
      if (stream) cudaStreamDestroy(stream);
    }
  };

  static ManagerCache &Cache() {
    static thread_local ManagerCache cache;
    return cache;
  }

  /** @brief This thread's persistent stream, created once. */
  static cudaStream_t CachedStream() {
    auto &c = Cache();
    if (!c.stream && cudaStreamCreate(&c.stream) != cudaSuccess) {
      c.stream = nullptr;
    }
    return c.stream;
  }

  /**
   * @brief Cached manager for algo_, built on the persistent stream.
   *
   * Same walk as getOrCreateCompManager: scan for a live slot with this
   * algorithm and bump its tick on a hit; otherwise take an empty slot, or
   * evict the lowest tick, and build there.
   */
  std::shared_ptr<nvcomp::nvcompManagerBase> GetOrCreateManager(
      cudaStream_t stream) {
    auto &c = Cache();
    const int idx = static_cast<int>(algo_);

    for (auto &slot : c.slots) {
      if (slot.mgr && slot.algo == idx) {
        slot.tick = ++c.clock;
        ++c.hits;
        return slot.mgr;
      }
    }

    int victim = -1;
    for (int i = 0; i < kLruDepth; ++i) {
      if (!c.slots[i].mgr) { victim = i; break; }
    }
    if (victim < 0) {
      uint64_t min_tick = c.slots[0].tick;
      victim = 0;
      for (int i = 1; i < kLruDepth; ++i) {
        if (c.slots[i].tick < min_tick) {
          min_tick = c.slots[i].tick;
          victim = i;
        }
      }
      c.slots[victim].mgr.reset();
    }

    auto mgr = MakeManager(stream);
    if (!mgr) return nullptr;
    c.slots[victim].mgr = mgr;
    c.slots[victim].algo = idx;
    c.slots[victim].tick = ++c.clock;
    ++c.misses;
    return mgr;
  }

  std::shared_ptr<nvcomp::nvcompManagerBase> MakeManager(cudaStream_t stream) {
    switch (algo_) {
      // NOTE: nvcomp >= 5.x split each algorithm's single "DefaultOpts" into
      // separate Compress/Decompress opts (discovered while adding Cascaded/
      // Bitcomp below -- the single-opts symbols no longer exist and these
      // 6 pre-existing cases failed to compile against the installed
      // nvcomp 5.3.0.16; fixed here using the same two-opts pattern nvcomp
      // itself now uses everywhere).
      case NvCompAlgo::LZ4:
        return std::make_shared<nvcomp::LZ4Manager>(
            kChunkSize, nvcompBatchedLZ4CompressDefaultOpts,
            nvcompBatchedLZ4DecompressDefaultOpts, stream);
      case NvCompAlgo::SNAPPY:
        return std::make_shared<nvcomp::SnappyManager>(
            kChunkSize, nvcompBatchedSnappyCompressDefaultOpts,
            nvcompBatchedSnappyDecompressDefaultOpts, stream);
      case NvCompAlgo::ZSTD:
        return std::make_shared<nvcomp::ZstdManager>(
            kChunkSize, nvcompBatchedZstdCompressDefaultOpts,
            nvcompBatchedZstdDecompressDefaultOpts, stream);
      case NvCompAlgo::GDEFLATE:
        return std::make_shared<nvcomp::GdeflateManager>(
            kChunkSize, nvcompBatchedGdeflateCompressDefaultOpts,
            nvcompBatchedGdeflateDecompressDefaultOpts, stream);
      case NvCompAlgo::DEFLATE:
        return std::make_shared<nvcomp::DeflateManager>(
            kChunkSize, nvcompBatchedDeflateCompressDefaultOpts,
            nvcompBatchedDeflateDecompressDefaultOpts, stream);
      case NvCompAlgo::ANS:
        return std::make_shared<nvcomp::ANSManager>(
            kChunkSize, nvcompBatchedANSCompressDefaultOpts,
            nvcompBatchedANSDecompressDefaultOpts, stream);
      case NvCompAlgo::CASCADED: {
        // NeuroPress overrides the nvcomp default here
        // (compression_factory.cpp:121-126): "Byte-level: correct for any
        // quantized precision (INT8/16/32)". The default is NOT char --
        // nvcompBatchedCascadedCompressDefaultOpts is {4096,
        // NVCOMP_TYPE_INT, 2, 1, 1, {0}} -- so leaving it alone ran
        // Cascaded's RLE+delta+bitpack pipeline over 32-bit words instead
        // of bytes, producing a different bitstream (and so a different
        // ratio label) than the model was trained against. It also made
        // correctness depend on the input size: nvcomp documents that each
        // chunk must be a multiple of the element type's size "else this
        // may crash or produce invalid output", and the manager's final
        // chunk is input_size % 64 KiB.
        nvcompBatchedCascadedCompressOpts_t opts =
            nvcompBatchedCascadedCompressDefaultOpts;
        opts.type = NVCOMP_TYPE_CHAR;
        return std::make_shared<nvcomp::CascadedManager>(
            kChunkSize, opts, nvcompBatchedCascadedDecompressDefaultOpts,
            stream);
      }
      case NvCompAlgo::BITCOMP: {
        // Likewise (compression_factory.cpp:128-134): "Good for scientific
        // data". Bitcomp models the buffer as an array of its declared
        // type, so the default NVCOMP_TYPE_UCHAR compresses 8-byte-wide
        // scientific data byte-wise and reaches a materially different
        // ratio. algorithm 0 matches the default but is set explicitly,
        // as upstream does.
        nvcompBatchedBitcompCompressOpts_t opts =
            nvcompBatchedBitcompCompressDefaultOpts;
        opts.data_type = NVCOMP_TYPE_LONGLONG;
        opts.algorithm = 0;
        return std::make_shared<nvcomp::BitcompManager>(
            kChunkSize, opts, nvcompBatchedBitcompDecompressDefaultOpts,
            stream);
      }
    }
    return nullptr;
  }

  NvCompAlgo algo_;
};

}  // namespace ctp

#endif  // CTP_ENABLE_COMPRESS && CTP_ENABLE_NVCOMP

#endif  // CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_NvComp_H_
