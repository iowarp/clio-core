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

#include <cstddef>
#include <utility>
#if CTP_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

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

#if CTP_ENABLE_CUDA
/**
 * CUDA-event bracket around a codec launch, for the wrappers that do not have
 * nvcomp's manager-owned event pair.
 *
 * WHY IT EXISTS. Only nvcomp.h used to time its launch, so cusz, cuszp and
 * ndzip fell through to compressor_runtime.cc's fallback -- host wall clock
 * around the whole Compress(), which also covers staging, a cudaMalloc,
 * configure_compression, the stream sync and the output copy. That made the
 * external codecs measured with a systematically LARGER instrument than the
 * nvcomp ones they are compared against. It showed up as an impossibility:
 * cuSZ's summed compress_ms (28.4 s) exceeded the phase containing it
 * (23.4 s), a negative residual.
 *
 * Events are created once per thread and reused, so a bracket costs two
 * cudaEventRecords -- the same trade nvcomp's KernelTimer makes.
 *
 * A codec that leaves LastCodecKernelMs() at -1 still gets the wall-clock
 * fallback, so a wrapper that cannot time itself degrades rather than lying.
 */
class CodecKernelTimer {
 public:
  /**
   * @param s   stream to bracket on
   * @param out where to store the elapsed ms; nullptr means LastCodecKernelMs().
   *
   * The out-parameter form exists so PREPROCESSING (quantize, byte shuffle) can
   * be timed with the same instrument as the codec without overwriting the
   * codec's slot. Mixing instruments inside one metric is exactly the defect
   * this class was added to remove -- and preprocessing is the LARGER component
   * on a quantized arm, so measuring it with a host clock while the codec used
   * CUDA events made the two halves incomparable. It also made the same logical
   * work instrument-dependent: cusz/cuszp quantize INSIDE their codec and were
   * event-timed, while an nvcomp arm quantized separately and was host-timed.
   *
   * The event pair is thread-local and shared, which is safe because these
   * brackets are sequential -- preprocessing completes before the codec starts.
   */
  explicit CodecKernelTimer(cudaStream_t s = nullptr, double *out = nullptr)
      : stream_(s), out_(out) {
    if (!out_) LastCodecKernelMs() = -1.0;
    auto &e = Events();
    if (!e.first && cudaEventCreate(&e.first) != cudaSuccess) return;
    if (!e.second && cudaEventCreate(&e.second) != cudaSuccess) return;
    on_ = (cudaEventRecord(e.first, stream_) == cudaSuccess);
  }
  /** Call after the launch; the elapsed time lands in LastCodecKernelMs(). */
  void Stop() {
    if (!on_) return;
    on_ = false;
    auto &e = Events();
    if (cudaEventRecord(e.second, stream_) != cudaSuccess) return;
    if (cudaEventSynchronize(e.second) != cudaSuccess) return;
    float ms = 0.0f;
    if (cudaEventElapsedTime(&ms, e.first, e.second) == cudaSuccess) {
      if (out_) *out_ += static_cast<double>(ms);   // accumulates: quantize + shuffle
      else      LastCodecKernelMs() = static_cast<double>(ms);
    }
  }
  ~CodecKernelTimer() { Stop(); }

 private:
  static std::pair<cudaEvent_t, cudaEvent_t> &Events() {
    static thread_local std::pair<cudaEvent_t, cudaEvent_t> ev{nullptr, nullptr};
    return ev;
  }
  cudaStream_t stream_ = nullptr;
  double *out_ = nullptr;
  bool on_ = false;
};
#endif  // CTP_ENABLE_CUDA

}  // namespace ctp

#endif  // CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_COMPRESS_H_
