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
   * (gpucompress_compress.cpp).
   */
  struct KernelTimer {
    cudaEvent_t start = nullptr, stop = nullptr;
    cudaStream_t stream = nullptr;
    bool on = false;

    /**
     * Always on now, and no longer gated by CLIO_CODEC_KERNEL_TIMING.
     *
     * The gate existed because each timer created and destroyed its own event
     * pair, which is real per-call overhead. The events are now persistent
     * per thread (ManagerCache::ev_start/ev_stop, created once), so the cost
     * of a bracket is two cudaEventRecords -- the same thing NeuroPress does
     * unconditionally with its per-CompContext events.
     *
     * It has to be unconditional because the number is no longer just a
     * report: the exploration cost model consumes it. Leaving it opt-in meant
     * cost was computed from a host wall-clock that also covered staging, a
     * possible cudaMalloc, configure_compression, the stream sync and the
     * output copy -- so CLIO's costs were systematically larger than the
     * kernel times NeuroPress ranks on, and cost decides the exploration
     * winner, the SGD compression-time target, and the error_pct gate that
     * fires either of them.
     *
     * CLIO_CODEC_KERNEL_TIMING is retained as a no-op for compatibility with
     * existing scripts; the measurement it used to switch on is now default.
     */
    explicit KernelTimer(cudaStream_t s) : stream(s) {
      LastCodecKernelMs() = -1.0;
      if (!s) return;
      auto &c = Cache();
      if (!c.ev_start && cudaEventCreate(&c.ev_start) != cudaSuccess) return;
      if (!c.ev_stop && cudaEventCreate(&c.ev_stop) != cudaSuccess) return;
      start = c.ev_start;
      stop = c.ev_stop;
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

    // No destructor work: the events belong to the thread's ManagerCache and
    // outlive every individual timer.
  };

  /**
   * @brief nvcomp's own worst case for this input, from the cached manager.
   *
   * configure_compression() is a host-side calculation on the manager this
   * thread already holds, so asking costs no device work. Returning it lets
   * Runtime::Compress allocate a buffer nvcomp can write into directly --
   * without it, ANS (whose worst case exceeds the caller's input+5% guess)
   * compresses into a temporary and copies the payload back D2D on every
   * chunk.
   * */
  size_t MaxCompressedSize(size_t input_size) override {
    if (input_size == 0) return 0;
    cudaStream_t stream = CachedStream();
    if (!stream) return 0;
    try {
      std::shared_ptr<nvcomp::nvcompManagerBase> mgr = GetOrCreateManager(stream);
      if (!mgr) return 0;
      return mgr->configure_compression(input_size).max_compressed_buffer_size;
    } catch (...) {
      return 0;  // no opinion; the caller's worst case stands
    }
  }

  bool Compress(void *output, size_t &output_size, void *input,
                size_t input_size) override {
    // Persistent per-thread stream and cached manager. Both used to be built
    // and torn down on EVERY call, inside the window Runtime::Compress times
    // to produce the model's comp_time label -- so the label carried a cost
    // that is large relative to the compress kernels and roughly independent
    // of chunk size, flattening the time-vs-size relationship the network
    // learns. NeuroPress pays neither per chunk: a persistent context stream
    // and an LRU of managers (gpucompress_pool.cpp), with its
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

        // Clear any error left by earlier work on this thread, so the
        // check after the launch can only see this compression's own.
        cudaGetLastError();
        ScratchOom();  // clear before the launch
        KernelTimer timer(stream);
        mgr->compress(d_in, d_out, cfg);
        // A launch that never ran (device out of memory is the case seen)
        // reports here and NOT through the sync below: with nothing enqueued
        // the sync succeeds, d_out keeps whatever the allocator recycled, and
        // get_compressed_output_size() then reads a plausible size out of it.
        // The result is a unique, well-sized, undecompressable blob stored as
        // a success. Checking the launch is what turns that into a failure
        // the caller can fall back from.
        if (cudaPeekAtLastError() != cudaSuccess) break;
        timer.Stop();
        if (cudaStreamSynchronize(stream) != cudaSuccess) break;
        if (cudaGetLastError() != cudaSuccess) break;
        size_t comp_size = mgr->get_compressed_output_size(d_out);
        // Zero is not a compressed stream: nvcomp writes a header even for
        // incompressible input, so it means the size came from memory the
        // kernels never wrote.
        if (comp_size == 0 || comp_size > output_size) break;

        // The scratch allocator flags a device-memory failure that nvcomp
        // itself does not report: with cudaMallocAsync (its default) the
        // failure lands on the stream, compress() returns normally and the
        // output is a plausible size that no decompressor accepts. See
        // ScratchOom().
        if (ScratchOom()) break;

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

  /**
   * @brief One slot of a stream-parallel compression sweep.
   *
   * Mirrors upstream's ExploreSlot (gpucompress_compress.cpp): its own stream,
   * its own timing events, its own manager, and its own device buffers, so K
   * of these can be in flight at once. Everything the slot owns is released by
   * ReleaseSlot().
   */
  struct AsyncSlot {
    cudaStream_t stream = nullptr;  ///< owned by the slot
    cudaEvent_t ev_start = nullptr, ev_stop = nullptr;  ///< owned by the slot
    std::shared_ptr<nvcomp::nvcompManagerBase> mgr;
    uint8_t *d_in = nullptr;
    uint8_t *d_out = nullptr;
    bool free_in = false;
    bool free_out = false;
    void *out = nullptr;  ///< caller's destination, not owned
    size_t out_capacity = 0;
    bool out_is_device = false;
    bool launched = false;
    size_t compressed_size = 0;  ///< valid after CompressFinish
    double kernel_ms = -1.0;     ///< valid after CompressFinish

    // DecompressMeasure() only. Its destination buffer is freed as soon as
    // the timing has been read, so the sweep's peak device memory grows by one
    // chunk rather than by one per slot -- slots are not released until the
    // whole sweep ends, so a per-slot buffer would be held K times over.
    cudaEvent_t ev_dstart = nullptr, ev_dstop = nullptr;  ///< owned by the slot
    double decomp_ms = -1.0;     ///< valid after DecompressMeasure; <0 = none

    AsyncSlot() = default;
    // The slot owns a stream, two events and possibly two device buffers, so
    // releasing is not optional and must not depend on the caller reaching a
    // cleanup path. Destroying here makes an early `continue` between
    // OpenSlot() and CompressLaunch() safe, which is exactly the shape the
    // preprocessing steps in between need.
    ~AsyncSlot() { ReleaseSlot(this); }
    AsyncSlot(const AsyncSlot &) = delete;
    AsyncSlot &operator=(const AsyncSlot &) = delete;
  };

  /**
   * @brief Create the slot's stream and timing events, without launching.
   *
   * Split out of CompressLaunch so a caller can run its OWN preprocessing on
   * the slot's stream first. Upstream needs exactly this: it quantizes and
   * byte-shuffles each explored candidate on s.stream and then compresses on
   * s.stream, so the whole per-slot pipeline is queued as one dependent chain
   * and nothing waits until every slot is collected
   * (gpucompress_compress.cpp). Idempotent, so CompressLaunch can call it for
   * a caller that has no preprocessing to do.
   */
  static bool OpenSlot(AsyncSlot *slot) {
    if (!slot) return false;
    if (slot->stream) return true;  // already open
    if (cudaStreamCreate(&slot->stream) != cudaSuccess) {
      slot->stream = nullptr;
      return false;
    }
    if (cudaEventCreate(&slot->ev_start) != cudaSuccess ||
        cudaEventCreate(&slot->ev_stop) != cudaSuccess) {
      ReleaseSlot(slot);
      return false;
    }
    return true;
  }

  /**
   * @brief Launch a compression on the slot's own stream and RETURN, without
   * synchronizing.
   *
   * This is the half of Compress() that can overlap. Compress() takes the
   * thread's one cached stream and blocks on it, so two codec calls on a
   * thread can never be in flight together; a caller that wants K running at
   * once pairs this with CompressFinish() instead, which is exactly how
   * upstream drives its exploration sweep -- launch all K, then sync all K.
   *
   * The manager is built fresh on this slot's stream rather than taken from
   * the thread's LRU: a manager is bound to the stream it was constructed on,
   * so a cache keyed by algorithm alone cannot serve K slots that share an
   * algorithm but not a stream. Upstream constructs its per-slot comp_mgr the
   * same way and for the same reason.
   *
   * @return true if the launch was issued; false leaves the slot released.
   */
  bool CompressLaunch(void *output, size_t output_capacity, void *input,
                      size_t input_size, AsyncSlot *slot) {
    if (!OpenSlot(slot)) return false;
    slot->out = output;
    slot->out_capacity = output_capacity;
    try {
      slot->d_in = ToDeviceInput(input, input_size, slot->stream,
                                 &slot->free_in);
      if (!slot->d_in) {
        ReleaseSlot(slot);
        return false;
      }
      slot->mgr = MakeManager(slot->stream);
      if (!slot->mgr) {
        ReleaseSlot(slot);
        return false;
      }
      InstallScratchAllocators(slot->mgr);
      nvcomp::CompressionConfig cfg = slot->mgr->configure_compression(input_size);
      slot->out_is_device = IsDeviceAccessible(output);
      if (slot->out_is_device &&
          output_capacity >= cfg.max_compressed_buffer_size) {
        slot->d_out = static_cast<uint8_t *>(output);
      } else {
        if (cudaMalloc(&slot->d_out, cfg.max_compressed_buffer_size) !=
            cudaSuccess) {
          ReleaseSlot(slot);
          return false;
        }
        slot->free_out = true;
      }
      // Events bracket the codec launch alone, same as Compress()'s
      // KernelTimer and same as upstream's per-slot ev_start/ev_stop.
      if (cudaEventRecord(slot->ev_start, slot->stream) != cudaSuccess) {
        ReleaseSlot(slot);
        return false;
      }
      cudaGetLastError();
      ScratchOom();  // clear
      slot->mgr->compress(slot->d_in, slot->d_out, cfg);
      // Same launch check as Compress(): a launch that never ran leaves the
      // slot's output buffer holding recycled memory, which CompressWait()
      // would otherwise size and deliver as a successful candidate.
      if (cudaPeekAtLastError() != cudaSuccess || ScratchOom()) {
        ReleaseSlot(slot);
        return false;
      }
      if (cudaEventRecord(slot->ev_stop, slot->stream) != cudaSuccess) {
        ReleaseSlot(slot);
        return false;
      }
      slot->launched = true;
      return true;
    } catch (...) {
      // The nvcomp Manager API reports errors by throwing, and compress() is
      // void, so a throw is its only failure signal -- same contract Compress()
      // handles.
      ReleaseSlot(slot);
      return false;
    }
  }

  /**
   * @brief Wait for a launched slot, then read its size and kernel time.
   *
   * Delivers into the caller's buffer if the codec wrote to a temporary.
   * Leaves the slot's resources intact; call ReleaseSlot() when the output has
   * been consumed.
   */
  bool CompressFinish(AsyncSlot *slot) {
    if (!slot || !slot->launched) return false;
    if (cudaStreamSynchronize(slot->stream) != cudaSuccess) return false;
    float ms = 0.0f;
    if (cudaEventElapsedTime(&ms, slot->ev_start, slot->ev_stop) ==
        cudaSuccess) {
      slot->kernel_ms = static_cast<double>(ms);
    }
    try {
      const size_t comp_size = slot->mgr->get_compressed_output_size(slot->d_out);
      // Zero means the size came from memory the kernels never wrote.
      if (comp_size == 0 || comp_size > slot->out_capacity) return false;
      if (slot->d_out != static_cast<uint8_t *>(slot->out)) {
        const cudaMemcpyKind kind = slot->out_is_device
                                        ? cudaMemcpyDeviceToDevice
                                        : cudaMemcpyDeviceToHost;
        if (cudaMemcpy(slot->out, slot->d_out, comp_size, kind) != cudaSuccess) {
          return false;
        }
      }
      slot->compressed_size = comp_size;
      return true;
    } catch (...) {
      return false;
    }
  }

  /**
   * @brief Decompress this slot's OWN compressed output back, timed.
   *
   * Exploration measures ratio and compression time per candidate but has
   * never had a decompression time: upstream substitutes the NN's prediction
   * for every alternative (gpucompress_compress.cpp, "decomp_time uses NN
   * prediction (no decomp at write)"), so the cost model's dt term is a
   * constant that cancels out of the ranking. This produces the real number.
   *
   * Call AFTER CompressFinish, BEFORE ReleaseSlot. The compressed bytes are
   * still in slot->d_out and the slot still owns its stream.
   *
   * TWO THINGS ARE DELIBERATELY OUTSIDE THE TIMED WINDOW, and they are what
   * make the result comparable to the compression time beside it:
   *
   *   create_manager parses the NVCOMP_NATIVE header and SYNCS the stream. The
   *   bitstream is self-describing so the compressing manager cannot be
   *   reused, and building one is construction, not decompression -- on a cold
   *   codec it costs more than the decompression it precedes. CompressLaunch
   *   brackets only mgr->compress() for the same reason, so bracketing only
   *   mgr->decompress() keeps ct and dt the same KIND of quantity.
   *
   *   The destination allocation, likewise.
   *
   * @param expect_size Original chunk size; a header claiming anything else
   *   means the compressed bytes are not this candidate's, and the slot is
   *   refused rather than timed.
   * @return true if decomp_ms holds a measurement.
   */
  bool DecompressMeasure(AsyncSlot *slot, size_t expect_size) {
    if (!slot || !slot->stream || !slot->d_out || slot->compressed_size == 0) {
      return false;
    }
    slot->decomp_ms = -1.0;
    uint8_t *d_decomp = nullptr;
    bool ok = false;
    try {
      std::shared_ptr<nvcomp::nvcompManagerBase> dmgr =
          nvcomp::create_manager(slot->d_out, slot->stream);
      if (dmgr) {
        InstallScratchAllocators(dmgr);
        nvcomp::DecompressionConfig dcfg =
            dmgr->configure_decompression(slot->d_out);
        if (dcfg.decomp_data_size == expect_size &&
            cudaMalloc(&d_decomp, dcfg.decomp_data_size) == cudaSuccess &&
            (slot->ev_dstart != nullptr ||
             cudaEventCreate(&slot->ev_dstart) == cudaSuccess) &&
            (slot->ev_dstop != nullptr ||
             cudaEventCreate(&slot->ev_dstop) == cudaSuccess)) {
          cudaGetLastError();
          ScratchOom();  // clear
          if (cudaEventRecord(slot->ev_dstart, slot->stream) == cudaSuccess) {
            dmgr->decompress(d_decomp, slot->d_out, dcfg);
            // Same launch check Compress()/CompressLaunch() make: kernels that
            // never ran would otherwise be timed as an extremely fast codec.
            if (cudaPeekAtLastError() == cudaSuccess && !ScratchOom() &&
                cudaEventRecord(slot->ev_dstop, slot->stream) == cudaSuccess &&
                cudaStreamSynchronize(slot->stream) == cudaSuccess) {
              float ms = 0.0f;
              if (cudaEventElapsedTime(&ms, slot->ev_dstart, slot->ev_dstop) ==
                  cudaSuccess) {
                slot->decomp_ms = static_cast<double>(ms);
                ok = true;
              }
            }
          }
        }
      }
    } catch (...) {
      // nvcomp reports errors by throwing, and decompress() is void, so a
      // throw is its only failure signal. decomp_ms stays -1.
    }
    // Freed here, not in ReleaseSlot: slots live until the whole sweep ends,
    // so holding this would multiply the sweep's device memory by K.
    if (d_decomp) cudaFree(d_decomp);
    return ok;
  }

  /** @brief Release everything the slot owns. Safe to call more than once. */
  static void ReleaseSlot(AsyncSlot *slot) {
    if (!slot) return;
    // The manager must die before the stream it was built on.
    slot->mgr.reset();
    if (slot->free_in && slot->d_in) cudaFree(slot->d_in);
    if (slot->free_out && slot->d_out) cudaFree(slot->d_out);
    slot->d_in = slot->d_out = nullptr;
    slot->free_in = slot->free_out = false;
    if (slot->ev_start) cudaEventDestroy(slot->ev_start);
    if (slot->ev_stop) cudaEventDestroy(slot->ev_stop);
    slot->ev_start = slot->ev_stop = nullptr;
    if (slot->ev_dstart) cudaEventDestroy(slot->ev_dstart);
    if (slot->ev_dstop) cudaEventDestroy(slot->ev_dstop);
    slot->ev_dstart = slot->ev_dstop = nullptr;
    if (slot->stream) cudaStreamDestroy(slot->stream);
    slot->stream = nullptr;
    slot->launched = false;
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

        cudaGetLastError();
        KernelTimer timer(stream);
        mgr->decompress(d_out, d_in, cfg);
        // A decompression whose kernels never launched would otherwise leave
        // the caller's buffer untouched and still report success.
        if (cudaPeekAtLastError() != cudaSuccess) break;
        timer.Stop();
        if (cudaStreamSynchronize(stream) != cudaSuccess) break;
        if (cudaGetLastError() != cudaSuccess) break;

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
   * counters (gpucompress_pool.cpp, internal.hpp). Clio has no
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

    // Codec-timing events, created once and reused for every launch on this
    // thread. NeuroPress does exactly this: t_start/t_stop are allocated per
    // CompContext when the pool is built (gpucompress_pool.cpp) and reused for
    // every chunk, which is what makes always-on event timing free. Creating
    // and destroying a pair per call -- as this used to -- was the whole
    // reason the measurement had to be opt-in.
    cudaEvent_t ev_start = nullptr, ev_stop = nullptr;

    ~ManagerCache() {
      // Leak the managers, deliberately. This is a thread_local destructor, so
      // it runs at THREAD exit, and on a thread that exits during process
      // shutdown nvcomp's manager destructors reproducibly segfault inside
      // libnvcomp (seen in both GPU round-trip tests: every assertion passes,
      // then the process dies in __call_tls_dtors).
      //
      // It is NOT a dead CUDA context -- cudaFree(nullptr) still returns
      // cudaSuccess right here, and destroying a manager while the process is
      // running is safe and heavily exercised, since the LRU evicts and
      // rebuilds managers mid-run constantly. What has already gone is some
      // state inside libnvcomp itself, which tears down independently of the
      // CUDA runtime and cannot be probed for from the outside. So there is no
      // condition to test: the only reliable move is not to call into nvcomp
      // from here.
      //
      // The cost is bounded and does not accumulate -- at most kLruDepth
      // managers, once, on a thread that is exiting anyway, and the driver
      // reclaims all of it at process exit. Threads that want the memory back
      // sooner can call ResetManagerCache() while they are still running,
      // which takes the normal destruction path.
      for (auto &s : slots) {
        auto *leaked =
            new std::shared_ptr<nvcomp::nvcompManagerBase>(std::move(s.mgr));
        (void)leaked;
      }
      // Events and streams are plain CUDA objects, not nvcomp state, and the
      // runtime is demonstrably still alive here -- these are safe to release.
      if (ev_start) cudaEventDestroy(ev_start);
      if (ev_stop) cudaEventDestroy(ev_stop);
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
    InstallScratchAllocators(mgr);
    if (!mgr) return nullptr;
    c.slots[victim].mgr = mgr;
    c.slots[victim].algo = idx;
    c.slots[victim].tick = ++c.clock;
    ++c.misses;
    return mgr;
  }

  /** Set when our scratch allocator could not satisfy nvcomp, cleared by the
   *  caller before each compression. Thread-local: managers are per thread. */
  static bool &ScratchOomFlag() {
    static thread_local bool oom = false;
    return oom;
  }
  static bool ScratchOom() {
    bool v = ScratchOomFlag();
    ScratchOomFlag() = false;
    return v;
  }

  /** nvcomp's own scratch allocator is cudaMallocAsync, whose failure is
   *  asynchronous -- it does not throw, so compress() reports success and
   *  writes an output no decompressor accepts. Synchronous cudaMalloc turns
   *  that into something this call can see. */
  static void InstallScratchAllocators(
      const std::shared_ptr<nvcomp::nvcompManagerBase> &mgr) {
    if (!mgr) return;
    try {
      mgr->set_scratch_allocators(
          [](size_t n) -> void * {
            void *p = nullptr;
            if (cudaMalloc(&p, n) != cudaSuccess) {
              ScratchOomFlag() = true;
              return nullptr;
            }
            return p;
          },
          [](void *p, size_t) { if (p) cudaFree(p); });
    } catch (...) {
      // CPU managers throw here; they do not use device scratch at all.
    }
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
