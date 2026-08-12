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
};

}  // namespace ctp

#endif  // CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_COMPRESS_H_
