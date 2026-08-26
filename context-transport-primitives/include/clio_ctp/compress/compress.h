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
   * @brief Output capacity this codec wants for `input_size` bytes, or 0 when
   * it has no opinion and the caller's own worst case should stand.
   *
   * The caller allocates the output buffer before it knows which codec will
   * run, so it uses a heuristic (input + 5%). A codec whose own worst case is
   * LARGER than that still works -- it compresses into a temporary of its own
   * and copies the result back -- but the copy is pure waste, and on the GPU
   * it is a device-to-device copy of the whole compressed payload plus a
   * cudaMalloc/cudaFree pair per call. Measured on VPIC in situ: 64 of 256
   * chunks (every nvcomp-ans one) paid 167.5 MiB of avoidable D2D per GiB
   * staged. Reporting the number here lets the caller size the buffer once
   * and let every codec write straight into it.
   * */
  virtual size_t MaxCompressedSize(size_t input_size) {
    (void)input_size;
    return 0;
  }
};

/**
 * @brief Time of the last codec kernel, in ms, or < 0 if not measured.
 *
 * Set by the GPU-backed compressors (see nvcomp.h) using CUDA events placed
 * immediately around the codec launch and nothing else. A caller that times
 * Compress()/Decompress() from the host is measuring staging copies, a
 * possible cudaMalloc, configure_compression, the stream sync and the output
 * copy along with the kernel -- fine for accounting wall time, useless for
 * comparing against another implementation's kernel time. NeuroPress
 * brackets exactly its `compressor->compress(...)` call
 * (gpucompress_compress.cpp), so this brackets exactly the
 * equivalent one and the two numbers mean the same thing.
 *
 * Thread-local: each worker times its own call, and the value is only valid
 * immediately after that call returns on the same thread.
 */
inline double &LastCodecKernelMs() {
  static thread_local double ms = -1.0;
  return ms;
}

}  // namespace ctp

#endif  // CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_COMPRESS_H_
