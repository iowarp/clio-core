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
 * Stream override for GPU codecs, per thread.
 *
 * A codec must NEVER create its own stream: cudaStreamCreate has deadlocked
 * this runtime against a resident consumer kernel, and it is pure overhead on
 * a path that already has a stream. The caller installs one here immediately
 * before the call, inside the guard for the CUDA context that stream belongs
 * to; a codec with no override runs on the current context's default stream,
 * which is always valid.
 *
 * There was also a PROCESS-WIDE override. It is deleted, not deprecated. It
 * held codec_slots_[0].stream -- created inside the compressor's dedicated
 * codec CUcontext -- so any codec that picked it up while running in the
 * primary context got a stream from a context it was not in. That produced two
 * separate bugs before it was understood: cuszp silently compressed ZEROS (its
 * H2D never landed, giving a compression ratio identical at every error
 * bound), and nvcomp's internal cudaEventRecord failed 1016 times in one short
 * run and escalated to an illegal memory access that aborted the process. By
 * the end nothing read it and two places still wrote it, so it was a pure
 * footgun.
 */
inline void *&GpuCodecStreamThread() {
  static thread_local void *stream = nullptr;
  return stream;
}

/** @return the thread's stream override, or null for the default stream. */
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
