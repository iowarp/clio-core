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

/**
 * @file lossy_codec.h
 * @brief The interface every GPU lossy codec wrapper implements for
 * lossy_sweep.cu, one codec per binary (several codecs export clashing
 * symbols, e.g. cuSZp2 and cuSZp3, cuSZ and cuSZ-Hi).
 *
 * The harness owns all timing. A wrapper splits its work into untimed set-up
 * (Prepare*, After*) and the timed Compress / Decompress calls, which take
 * and produce device memory. The error bound is always ABSOLUTE: the harness
 * converts the requested value-range-relative bound itself, so every codec
 * is held to the same number.
 */
#ifndef LOSSY_CODEC_H_
#define LOSSY_CODEC_H_

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

/** @brief Shape of one field; x varies fastest. */
struct Field {
  size_t x = 1, y = 1, z = 1;
  /** @return number of elements */
  size_t n() const { return x * y * z; }
  /** @return 1, 2 or 3: the highest dimension longer than 1 */
  int dims() const { return z > 1 ? 3 : (y > 1 ? 2 : 1); }
};

/** @brief One GPU lossy codec behind a device-to-device interface. */
class LossyCodec {
 public:
  virtual ~LossyCodec() = default;

  /** @return codec name written to the CSV */
  virtual std::string Name() const = 0;
  /** @return the configurations this codec is run under */
  virtual std::vector<std::string> Variants() const { return {"default"}; }
  /** Select one of Variants(); @return false for an unknown one */
  virtual bool SetVariant(const std::string &v) { return v == "default"; }
  /** @return capacity the compressed device buffer must have for f */
  virtual size_t MaxCompressedBytes(const Field &f) = 0;

  /**
   * Untimed set-up before Compress: contexts, workspaces, tuning.
   * @param d_in the field on the device (for codecs that tune on the data)
   * @param f    its shape
   * @param eb   absolute error bound
   * @return false when the codec cannot take this field or bound
   */
  virtual bool PrepareCompress(const float *d_in, const Field &f, double eb,
                               cudaStream_t s) {
    (void)d_in; (void)f; (void)eb; (void)s;
    return true;
  }
  /**
   * Timed: compress d_in into d_out (cap bytes).
   * @return compressed bytes, 0 on failure
   */
  virtual size_t Compress(const float *d_in, const Field &f, double eb,
                          uint8_t *d_out, size_t cap, cudaStream_t s) = 0;
  /** Untimed hook after Compress (e.g. moving a host-side stream). */
  virtual bool AfterCompress(uint8_t *d_out, size_t bytes, cudaStream_t s) {
    (void)d_out; (void)bytes; (void)s;
    return true;
  }
  /** Untimed set-up before Decompress. */
  virtual bool PrepareDecompress(const uint8_t *d_cmp, size_t bytes,
                                 const Field &f, double eb, cudaStream_t s) {
    (void)d_cmp; (void)bytes; (void)f; (void)eb; (void)s;
    return true;
  }
  /** Timed: decompress into d_out (f.n() floats). @return success */
  virtual bool Decompress(const uint8_t *d_cmp, size_t bytes, const Field &f,
                          double eb, float *d_out, cudaStream_t s) = 0;
  /** Untimed hook after Decompress, before verification. */
  virtual bool AfterDecompress(float *d_out, const Field &f, cudaStream_t s) {
    (void)d_out; (void)f; (void)s;
    return true;
  }
  /** @return a one-line note for the CSV (e.g. the tuned zfp rate), or "" */
  virtual std::string Note() const { return ""; }
};

/** Defined by exactly one codec_*.cu linked into each binary. */
std::unique_ptr<LossyCodec> MakeLossyCodec();

#endif  // LOSSY_CODEC_H_
