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
#include <nvcomp/deflate.hpp>
#include <nvcomp/gdeflate.hpp>
#include <nvcomp/lz4.hpp>
#include <nvcomp/nvcompManagerFactory.hpp>
#include <nvcomp/lz4.h>
#include <lz4.h>
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
/**
 * Raw batched-LZ4 frame codec (nvcomp LOW-LEVEL API).
 *
 * The Manager API costs a device-scratch cudaMalloc per manager and hides a
 * stream sync inside configure_decompression PER FRAME -- which caps any
 * batched decompress at one-frame-at-a-time speed (~1 GB/s measured). The
 * low-level nvcompBatchedLZ4* API decompresses EVERY chunk of EVERY frame in
 * a batch with one launch.
 *
 * Frame layout (written after the caller's own headers):
 *   RawLz4Table  { magic, nchunks, chunk payload sizes }   (host-readable)
 *   packed compressed chunk payloads
 * Chunks are 64 KiB of uncompressed data (last one short). The table is
 * fixed-size so a caller holding only the first ~300 bytes of a frame on the
 * host can drive a fully device-side batched decompress.
 */
namespace raw_lz4 {

constexpr uint32_t kMagic = 0x43344252u;   // 'RB4C'
constexpr size_t kChunk = 64 * 1024;
constexpr size_t kMaxChunks = 64;          // frames up to 4 MiB

struct Table {
  uint32_t magic;
  uint32_t nchunks;
  uint32_t comp_bytes[kMaxChunks];
};

/** Growable per-thread device scratch (never freed; sized to high water). */
inline void *DevScratch(size_t bytes) {
  static thread_local void *buf = nullptr;
  static thread_local size_t cap = 0;
  if (bytes > cap) {
    if (buf) cudaFree(buf);
    if (cudaMalloc(&buf, bytes) != cudaSuccess) {
      buf = nullptr;
      cap = 0;
      return nullptr;
    }
    cap = bytes;
  }
  return buf;
}
inline cudaStream_t RawStream() {
  static thread_local cudaStream_t s = nullptr;
  if (s == nullptr && cudaStreamCreate(&s) != cudaSuccess) s = nullptr;
  return s;
}

/** Compress `in_size` HOST bytes into `out` (host): Table + packed chunks.
 *  Returns compressed frame size (Table included), or 0 on failure. */
inline size_t CompressFrame(const void *in, size_t in_size, void *out,
                            size_t out_cap) {
  if (in_size == 0 || in_size > kChunk * kMaxChunks) return 0;
  cudaStream_t stream = RawStream();
  if (!stream) return 0;
  const size_t n = (in_size + kChunk - 1) / kChunk;
  size_t max_out = 0;
  nvcompBatchedLZ4CompressGetMaxOutputChunkSize(kChunk,
                                                nvcompBatchedLZ4DefaultOpts,
                                                &max_out);
  size_t temp = 0;
  nvcompBatchedLZ4CompressGetTempSize(n, kChunk, nvcompBatchedLZ4DefaultOpts,
                                      &temp);
  // Scratch layout: [in n*kChunk][out n*max_out][temp]
  //                 [in_ptrs n][in_sizes n][out_ptrs n][out_sizes n]
  const size_t arr = n * (sizeof(void *) * 2 + sizeof(size_t) * 2);
  uint8_t *d = (uint8_t *) DevScratch(n * kChunk + n * max_out + temp + arr +
                                      256);
  if (!d) return 0;
  uint8_t *d_in = d;
  uint8_t *d_out = d_in + n * kChunk;
  uint8_t *d_temp = d_out + n * max_out;
  uint8_t *d_arr = d_temp + temp;
  if (cudaMemcpyAsync(d_in, in, in_size, cudaMemcpyHostToDevice, stream) !=
      cudaSuccess) {
    return 0;
  }
  std::vector<const void *> h_inp(n);
  std::vector<size_t> h_ins(n);
  std::vector<void *> h_outp(n);
  for (size_t c = 0; c < n; ++c) {
    h_inp[c] = d_in + c * kChunk;
    h_ins[c] = (c + 1 < n) ? kChunk : (in_size - c * kChunk);
    h_outp[c] = d_out + c * max_out;
  }
  auto *d_inp = (const void **) d_arr;
  auto *d_ins = (size_t *) (d_arr + n * sizeof(void *));
  auto *d_outp = (void **) (d_arr + n * (sizeof(void *) + sizeof(size_t)));
  auto *d_outs =
      (size_t *) (d_arr + n * (sizeof(void *) * 2 + sizeof(size_t)));
  cudaMemcpyAsync(d_inp, h_inp.data(), n * sizeof(void *),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_ins, h_ins.data(), n * sizeof(size_t),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_outp, h_outp.data(), n * sizeof(void *),
                  cudaMemcpyHostToDevice, stream);
  if (nvcompBatchedLZ4CompressAsync(d_inp, d_ins, kChunk, n, d_temp, temp,
                                    d_outp, d_outs,
                                    nvcompBatchedLZ4DefaultOpts,
                                    stream) != nvcompSuccess) {
    return 0;
  }
  std::vector<size_t> h_outs(n);
  cudaMemcpyAsync(h_outs.data(), d_outs, n * sizeof(size_t),
                  cudaMemcpyDeviceToHost, stream);
  if (cudaStreamSynchronize(stream) != cudaSuccess) return 0;
  Table tbl{};
  tbl.magic = kMagic;
  tbl.nchunks = (uint32_t) n;
  size_t total = sizeof(Table);
  for (size_t c = 0; c < n; ++c) {
    tbl.comp_bytes[c] = (uint32_t) h_outs[c];
    total += h_outs[c];
  }
  if (total > out_cap) return 0;
  std::memcpy(out, &tbl, sizeof(Table));
  uint8_t *w = (uint8_t *) out + sizeof(Table);
  for (size_t c = 0; c < n; ++c) {
    if (cudaMemcpyAsync(w, h_outp[c], h_outs[c], cudaMemcpyDeviceToHost,
                        stream) != cudaSuccess) {
      return 0;
    }
    w += h_outs[c];
  }
  if (cudaStreamSynchronize(stream) != cudaSuccess) return 0;
  return total;
}

/** One frame of a device-side batched decompress. `frame_dev` points at the
 *  Table INSIDE device memory; `table` is the same bytes host-side (the
 *  caller already pulled its headers). */
struct FrameRef {
  const uint8_t *frame_dev;
  const Table *table;
  uint8_t *out_dev;
  size_t out_size;
};

/** Decompress every chunk of every frame with ONE nvcomp launch on `stream`.
 *  No sync: the caller owns completion. Returns false on setup failure. */
inline bool DecompressBatch(const FrameRef *frames, size_t nf,
                            cudaStream_t stream) {
  if (nf == 0) return true;
  size_t chunks = 0;
  for (size_t f = 0; f < nf; ++f) chunks += frames[f].table->nchunks;
  if (chunks == 0) return true;
  size_t temp = 0;
  nvcompBatchedLZ4DecompressGetTempSize(chunks, kChunk, &temp);
  const size_t arr =
      chunks * (sizeof(void *) * 2 + sizeof(size_t) * 3) +
      chunks * sizeof(nvcompStatus_t);
  uint8_t *d = (uint8_t *) DevScratch(temp + arr + 256);
  if (!d) return false;
  uint8_t *d_temp = d;
  uint8_t *d_arr = d + temp;
  std::vector<const void *> h_cptr(chunks);
  std::vector<size_t> h_csz(chunks), h_usz(chunks);
  std::vector<void *> h_uptr(chunks);
  size_t k = 0;
  for (size_t f = 0; f < nf; ++f) {
    const Table *t = frames[f].table;
    const uint8_t *payload = frames[f].frame_dev + sizeof(Table);
    size_t out_off = 0;
    for (uint32_t c = 0; c < t->nchunks; ++c, ++k) {
      h_cptr[k] = payload;
      h_csz[k] = t->comp_bytes[c];
      const size_t usz = (c + 1 < t->nchunks)
                             ? kChunk
                             : (frames[f].out_size - out_off);
      h_usz[k] = usz;
      h_uptr[k] = frames[f].out_dev + out_off;
      payload += t->comp_bytes[c];
      out_off += usz;
    }
  }
  auto *d_cptr = (const void **) d_arr;
  auto *d_csz = (size_t *) (d_arr + chunks * sizeof(void *));
  auto *d_usz = (size_t *) (d_arr + chunks * (sizeof(void *) + sizeof(size_t)));
  auto *d_act =
      (size_t *) (d_arr + chunks * (sizeof(void *) + 2 * sizeof(size_t)));
  auto *d_uptr =
      (void **) (d_arr + chunks * (sizeof(void *) + 3 * sizeof(size_t)));
  auto *d_status = (nvcompStatus_t *) (d_arr + chunks * (2 * sizeof(void *) +
                                                         3 * sizeof(size_t)));
  cudaMemcpyAsync(d_cptr, h_cptr.data(), chunks * sizeof(void *),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_csz, h_csz.data(), chunks * sizeof(size_t),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_usz, h_usz.data(), chunks * sizeof(size_t),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_uptr, h_uptr.data(), chunks * sizeof(void *),
                  cudaMemcpyHostToDevice, stream);
  return nvcompBatchedLZ4DecompressAsync(d_cptr, d_csz, d_usz, d_act, chunks,
                                         d_temp, temp, d_uptr, d_status,
                                         stream) == nvcompSuccess;
}

/** HOST-side decode of a raw frame (table + payload on the host): nvcomp's
 *  batched-LZ4 chunks are standard LZ4 blocks, so plain LZ4_decompress_safe
 *  handles them. Keeps host consumers (DRAM-tier fills, host GetBlobs)
 *  working on frames written by the GPU compressor. */
inline bool DecompressFrameHost(const void *frame, size_t frame_size,
                                void *out, size_t out_size) {
  const auto *t = reinterpret_cast<const Table *>(frame);
  if (frame_size < sizeof(Table) || t->magic != kMagic ||
      t->nchunks == 0 || t->nchunks > kMaxChunks) {
    return false;
  }
  const char *payload = reinterpret_cast<const char *>(frame) + sizeof(Table);
  size_t out_off = 0;
  for (uint32_t c = 0; c < t->nchunks; ++c) {
    const size_t usz =
        (c + 1 < t->nchunks) ? kChunk : (out_size - out_off);
    const int got = LZ4_decompress_safe(
        payload, reinterpret_cast<char *>(out) + out_off,
        (int) t->comp_bytes[c], (int) usz);
    if (got < 0 || (size_t) got != usz) return false;
    payload += t->comp_bytes[c];
    out_off += usz;
  }
  return out_off == out_size;
}

}  // namespace raw_lz4

class NvComp : public Compressor {
 public:
  explicit NvComp(NvCompAlgo algo = NvCompAlgo::LZ4) : algo_(algo) {}

  bool Compress(void *output, size_t &output_size, void *input,
                size_t input_size) override {
    cudaStream_t stream = CachedStream();
    if (stream == nullptr) {
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

        nvcomp::nvcompManagerBase *mgr = CachedManager(stream);
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

        mgr->compress(d_in, d_out, cfg);
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
    return ok;
  }

  bool Decompress(void *output, size_t &output_size, void *input,
                  size_t input_size) override {
    cudaStream_t stream = CachedStream();
    if (stream == nullptr) {
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

        // Typed managers decompress their own NVCOMP_NATIVE bitstreams;
        // create_manager (header parse + device scratch + stream sync PER
        // CALL) is unnecessary for bytes this codec wrote itself.
        nvcomp::nvcompManagerBase *mgr = CachedManager(stream);
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

        mgr->decompress(d_out, d_in, cfg);
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

  /**
   * Per-thread cached stream + typed manager. Building an nvcomp Manager
   * allocates device scratch (cudaMalloc) and create_manager additionally
   * parses the header AND syncs the stream -- measured ~3 ms of setup per
   * 1 MiB page, which serialized the whole decompress pipeline on the
   * allocator. Each runtime worker gets one stream and one manager per
   * algorithm, reused forever; typed managers decompress their own
   * NVCOMP_NATIVE bitstreams, so create_manager is unnecessary.
   */
  cudaStream_t CachedStream() {
    static thread_local cudaStream_t s_stream = nullptr;
    if (s_stream == nullptr) {
      if (cudaStreamCreate(&s_stream) != cudaSuccess) s_stream = nullptr;
    }
    return s_stream;
  }
  nvcomp::nvcompManagerBase *CachedManager(cudaStream_t stream) {
    static thread_local std::shared_ptr<nvcomp::nvcompManagerBase>
        s_mgr[6] = {};
    const int idx = (int) algo_;
    if (!s_mgr[idx]) {
      try {
        s_mgr[idx] = MakeManager(stream);
      } catch (...) {
        return nullptr;
      }
    }
    return s_mgr[idx].get();
  }

  /** Construct the nvcomp manager for the configured algorithm. */
  std::shared_ptr<nvcomp::nvcompManagerBase> MakeManager(cudaStream_t stream) {
    switch (algo_) {
      case NvCompAlgo::LZ4:
        return std::make_shared<nvcomp::LZ4Manager>(
            kChunkSize, nvcompBatchedLZ4DefaultOpts, stream);
      case NvCompAlgo::SNAPPY:
        return std::make_shared<nvcomp::SnappyManager>(
            kChunkSize, nvcompBatchedSnappyDefaultOpts, stream);
      case NvCompAlgo::ZSTD:
        return std::make_shared<nvcomp::ZstdManager>(
            kChunkSize, nvcompBatchedZstdDefaultOpts, stream);
      case NvCompAlgo::GDEFLATE:
        return std::make_shared<nvcomp::GdeflateManager>(
            kChunkSize, nvcompBatchedGdeflateDefaultOpts, stream);
      case NvCompAlgo::DEFLATE:
        return std::make_shared<nvcomp::DeflateManager>(
            kChunkSize, nvcompBatchedDeflateDefaultOpts, stream);
      case NvCompAlgo::ANS:
        return std::make_shared<nvcomp::ANSManager>(
            kChunkSize, nvcompBatchedANSDefaultOpts, stream);
    }
    return nullptr;
  }

  NvCompAlgo algo_;
};

}  // namespace ctp

#endif  // CTP_ENABLE_COMPRESS && CTP_ENABLE_NVCOMP

#endif  // CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_NvComp_H_
