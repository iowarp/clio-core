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

#ifndef CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_COMPRESS_H_
#define CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_COMPRESS_H_

#include <cstddef>

// #include "clio_ctp/data_structures/all.h"  // Deleted during hard refactoring

namespace ctp {

/**
 * Stream override storage for GPU codecs.
 *
 * It lives HERE, in the base header, and not in CompressionFactory, because a
 * GPU codec header cannot include the factory -- the factory includes every
 * codec header, so that direction is a cycle. CompressionFactory delegates to
 * these so there is exactly one copy of the storage.
 *
 * A codec must prefer this stream over creating its own. Per-call
 * cudaStreamCreate is not merely slow: it has deadlocked this runtime against
 * a resident spinning kernel, and on the device-to-device page-fault path the
 * compressor runtime has already provided a pooled slot stream.
 */
inline void *&GpuCodecStreamProcess() {
  static void *stream = nullptr;
  return stream;
}
inline void *&GpuCodecStreamThread() {
  static thread_local void *stream = nullptr;
  return stream;
}
/** @return the thread's stream override, else the process-wide one, else null
 *  (meaning the default stream). */
inline void *GetGpuCodecStream() {
  void *tls = GpuCodecStreamThread();
  return tls != nullptr ? tls : GpuCodecStreamProcess();
}

/**
 * The thread override ONLY -- never the process-wide default.
 *
 * A codec may adopt the thread override safely: the runtime sets it
 * immediately before the call and inside the same CUDA context the stream was
 * created in. The PROCESS-WIDE stream is not safe to adopt, because it is
 * codec_slots_[0].stream, created inside the compressor's dedicated codec
 * CUcontext. Using it from a caller running in the primary context (which is
 * where CompressIntoShm runs) silently breaks the transfers issued on it: the
 * H2D of the input never lands, the codec compresses ZEROS, and the result is
 * a compression ratio that is IDENTICAL for every error bound -- 120.471x at
 * both eb 1e-2 and 1e-3, which is how this was caught, since no lossy codec's
 * ratio can be invariant to a 10x change in bound.
 *
 * A codec with no thread override must create its own stream in whatever
 * context it was called from.
 */
inline void *GetGpuCodecStreamThreadOnly() { return GpuCodecStreamThread(); }

/**
 * One buffer pair in a batch: where the bytes come from, where they go.
 *
 * `src` and `dst` may each be host or device memory, INDEPENDENTLY, and a
 * codec is required to work out which and move the minimum. When both are on
 * the device there must be NO staging at all -- no scratch allocation, no
 * bounce through the host, no copy the codec could have avoided by decoding
 * straight into `dst`.
 */
struct CompressJob {
  const void *src = nullptr;  /**< input bytes; host or device */
  size_t src_size = 0;        /**< bytes at src */
  void *dst = nullptr;        /**< output buffer; host or device */
  size_t dst_capacity = 0;    /**< bytes available at dst */
  size_t out_size = 0;        /**< [out] bytes actually written to dst */
  bool ok = false;            /**< [out] did this job succeed */
};

class Compressor {
 public:
  Compressor() = default;
  virtual ~Compressor() = default;

  /**
   * Compress the input buffer into the output buffer
   * */
  virtual bool Compress(void *output, size_t &output_size, void *input,
                        size_t input_size) = 0;

  /**
   * Decompress the input buffer into the output buffer.
   * */
  virtual bool Decompress(void *output, size_t &output_size, void *input,
                          size_t input_size) = 0;

  /**
   * Compress a whole batch.
   *
   * The batch is the point: a GPU codec gets to issue ONE launch for every
   * job instead of one launch per page. Per-page launches were measured
   * starving against a resident consumer kernel (22 decodes in 280 seconds),
   * which is the problem this interface exists to remove.
   *
   * Each job reports its own `ok` and `out_size`; a failed job does not fail
   * its neighbours. The return value is true when EVERY job succeeded, so a
   * caller can check one bool in the common case and inspect per-job status
   * only when it is false.
   *
   * The default implementation loops over the single-buffer calls, so a CPU
   * codec needs no changes and gains nothing -- which is correct, since a
   * host codec has no launch to amortize.
   */
  virtual bool CompressBatch(CompressJob *jobs, size_t n) {
    return RunBatchSerial(jobs, n, /*compress=*/true);
  }

  /** Decompress a whole batch. See CompressBatch. */
  virtual bool DecompressBatch(CompressJob *jobs, size_t n) {
    return RunBatchSerial(jobs, n, /*compress=*/false);
  }

 protected:
  /** Shared fallback: run each job through the single-buffer entry point. */
  bool RunBatchSerial(CompressJob *jobs, size_t n, bool compress) {
    bool all = true;
    for (size_t i = 0; i < n; ++i) {
      size_t out = jobs[i].dst_capacity;
      void *in = const_cast<void *>(jobs[i].src);
      jobs[i].ok = compress ? Compress(jobs[i].dst, out, in, jobs[i].src_size)
                            : Decompress(jobs[i].dst, out, in, jobs[i].src_size);
      jobs[i].out_size = jobs[i].ok ? out : 0;
      all = all && jobs[i].ok;
    }
    return all;
  }
};

}  // namespace ctp

#endif  // CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_COMPRESS_H_
