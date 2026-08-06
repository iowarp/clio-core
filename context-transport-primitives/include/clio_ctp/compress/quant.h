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

#ifndef CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_QUANT_H_
#define CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_QUANT_H_

#include <cstdint>
#include <cstring>

#include "clio_ctp/constants/macros.h"
#include "compress.h"

namespace ctp {

/**
 * Block-quantized codec (GGUF k-quant / legacy layouts).
 *
 * This is a DECODE-side transform, not a compressor in the usual sense: the
 * data is BORN quantized (a GGUF file is already blocks of 4/5/6/8-bit weights
 * plus per-block scales), so nothing is ever quantized by this codec and no
 * precision is ever lost by it. Compress() therefore stores the quantized
 * bytes verbatim; Decompress() expands them to F32.
 *
 * The point is that the compact form is what sits in the kHbm tier, and the
 * expansion happens on the way out -- during prefetch or eviction -- so device
 * memory holds ~8x more model per byte (Q4_K is 144 B per 256 weights vs 1 KiB
 * as F32) and the expansion cost is paid on the transfer path rather than
 * inside every consumer kernel.
 *
 * Decoding is BLOCK-wise, which is the whole reason this belongs here rather
 * than in each consumer. A per-element decoder re-derives the block's `d`,
 * `dmin`, and 6-bit-packed sub-block scales for every one of the 256 elements
 * in the block; when a warp's 32 lanes sit inside one block that is 32x
 * redundant work per pass, and it measured as an order-of-magnitude loss in a
 * GEMV that decoded weights inline. DecodeBlock() hoists all of it out.
 */
class QuantCodec : public Compressor {
 public:
  /** Quantization layouts. Values match GGUF type ids so a GGUF header can be
   *  used directly without a translation table. */
  enum class Type : int {
    kF32  = 0,
    kF16  = 1,
    kQ4_0 = 2,
    kQ8_0 = 8,
    kQ4_K = 12,
    kQ5_K = 13,
    kQ6_K = 14,
    kBF16 = 30,
  };

  static constexpr uint32_t kQK = 256;       /**< elements per k-quant block */
  static constexpr uint32_t kScaleSize = 12; /**< packed 6-bit scale bytes */

  explicit QuantCodec(Type type) : type_(type) {}

  /** Elements encoded by one block of this type. */
  static CTP_CROSS_FUN uint32_t BlockElems(Type t) {
    switch (t) {
      case Type::kF32: case Type::kF16: case Type::kBF16: return 1;
      case Type::kQ4_0: return 32;
      case Type::kQ8_0: return 32;
      default: return kQK;
    }
  }

  /** Bytes occupied by one block of this type. */
  static CTP_CROSS_FUN uint32_t BlockBytes(Type t) {
    switch (t) {
      case Type::kF32:  return 4;
      case Type::kF16:  case Type::kBF16: return 2;
      case Type::kQ4_0: return 2 + 16;
      case Type::kQ8_0: return 2 + 32;
      case Type::kQ4_K: return 4 + kScaleSize + kQK / 2;
      case Type::kQ5_K: return 4 + kScaleSize + kQK / 2 + kQK / 8;
      case Type::kQ6_K: return kQK / 2 + kQK / 4 + kQK / 16 + 2;
      default: return 0;
    }
  }

  /**
   * Decode ONE block to floats. Per-block constants are read once here, which
   * is the difference between this and decoding element-by-element.
   *
   * @param t     layout of the block
   * @param blk   first byte of the block
   * @param out   receives BlockElems(t) floats
   * @param lane  first element this caller decodes (0 for serial callers)
   * @param step  stride between elements this caller decodes (1 for serial)
   */
  static CTP_CROSS_FUN void DecodeBlock(Type t, const uint8_t *blk, float *out,
                                        uint32_t lane = 0, uint32_t step = 1) {
    const uint32_t n = BlockElems(t);
    switch (t) {
      case Type::kF32:
        if (lane == 0) out[0] = LoadF32(blk);
        return;
      case Type::kF16:
        if (lane == 0) out[0] = HalfToFloat(blk);
        return;
      case Type::kBF16:
        if (lane == 0) out[0] = BFloatToFloat(blk);
        return;
      case Type::kQ8_0: {
        const float d = HalfToFloat(blk);           // once, not per element
        for (uint32_t e = lane; e < n; e += step) {
          out[e] = d * static_cast<float>(
                           static_cast<int8_t>(blk[2 + e]));
        }
        return;
      }
      case Type::kQ4_0: {
        const float d = HalfToFloat(blk);           // once, not per element
        for (uint32_t e = lane; e < n; e += step) {
          const uint8_t v = blk[2 + (e & 15)];
          const int q = (e < 16) ? (v & 0xF) : (v >> 4);
          out[e] = d * static_cast<float>(q - 8);
        }
        return;
      }
      case Type::kQ4_K: case Type::kQ5_K: {
        // d, dmin and the 8 packed sub-block scales are block constants; a
        // per-element decoder recomputed all of them 256 times per block.
        const float d = HalfToFloat(blk);
        const float dmin = HalfToFloat(blk + 2);
        const uint8_t *sc = blk + 4;
        const bool q5 = (t == Type::kQ5_K);
        const uint8_t *qh = q5 ? blk + 4 + kScaleSize : nullptr;
        const uint8_t *qs = blk + 4 + kScaleSize + (q5 ? kQK / 8 : 0);
        float ds[8], dm[8];
        for (int j = 0; j < 8; ++j) {
          uint8_t sd, sm;
          UnpackScale(sc, j, &sd, &sm);
          ds[j] = d * static_cast<float>(sd);
          dm[j] = dmin * static_cast<float>(sm);
        }
        for (uint32_t e = lane; e < n; e += step) {
          const int j = static_cast<int>(e / 32);
          const uint32_t within = e & 31;
          const uint8_t byte = qs[(j / 2) * 32 + within];
          int q = (j & 1) ? (byte >> 4) : (byte & 0xF);
          if (q5) q |= ((qh[within] >> j) & 1) << 4;
          out[e] = ds[j] * static_cast<float>(q) - dm[j];
        }
        return;
      }
      case Type::kQ6_K: {
        const uint8_t *ql = blk;
        const uint8_t *qh = blk + kQK / 2;
        const int8_t *sc = reinterpret_cast<const int8_t *>(blk + kQK / 2 + kQK / 4);
        const float d = HalfToFloat(blk + kQK / 2 + kQK / 4 + kQK / 16);
        for (uint32_t e = lane; e < n; e += step) {
          const uint32_t half = e / 128;          // 0..1
          const uint32_t r = e % 128;
          const uint32_t lo_i = half * 64 + (r % 64);
          const uint8_t lo = (r < 64) ? (ql[lo_i] & 0xF) : (ql[lo_i] >> 4);
          const uint32_t hi_i = half * 32 + (r % 32);
          const uint32_t sh = ((r / 32) % 4) * 2;
          const uint8_t hi = (qh[hi_i] >> sh) & 3;
          const int q = static_cast<int>(lo | (hi << 4)) - 32;
          out[e] = d * static_cast<float>(sc[e / 16]) * static_cast<float>(q);
        }
        return;
      }
      default:
        return;
    }
  }

  /**
   * Store the quantized bytes verbatim.
   *
   * The input is ALREADY the compact form; re-encoding it would be the only
   * lossy step in the pipeline, so this codec never does it.
   */
  bool Compress(void *output, size_t &output_size, void *input,
                size_t input_size) override {
    if (output_size < input_size) return false;
    std::memcpy(output, input, input_size);
    output_size = input_size;
    return true;
  }

  /** Expand quantized blocks to F32. output_size is in BYTES. */
  bool Decompress(void *output, size_t &output_size, void *input,
                  size_t input_size) override {
    const uint32_t bb = BlockBytes(type_);
    const uint32_t be = BlockElems(type_);
    if (bb == 0) return false;
    const size_t nblk = input_size / bb;
    const size_t need = nblk * be * sizeof(float);
    if (output_size < need) return false;
    const uint8_t *in = static_cast<const uint8_t *>(input);
    float *out = static_cast<float *>(output);
    for (size_t b = 0; b < nblk; ++b) {
      DecodeBlock(type_, in + b * bb, out + b * be);
    }
    output_size = need;
    return true;
  }

  Type type() const { return type_; }

 private:
  /** k-quant sub-block scales are packed 6 bits at a time across 12 bytes. */
  static CTP_CROSS_FUN void UnpackScale(const uint8_t *sc, int j, uint8_t *d,
                                        uint8_t *m) {
    if (j < 4) {
      *d = sc[j] & 63;
      *m = sc[j + 4] & 63;
    } else {
      *d = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4);
      *m = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4);
    }
  }

  static CTP_CROSS_FUN float LoadF32(const uint8_t *b) {
    float f;
    std::memcpy(&f, b, 4);
    return f;
  }

  /** IEEE half -> float without intrinsics, so the same code runs on host
   *  and device (device callers may of course use __half2float). */
  static CTP_CROSS_FUN float HalfToFloat(const uint8_t *b) {
    uint16_t h;
    std::memcpy(&h, b, 2);
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16;
    const uint32_t exp = (h >> 10) & 0x1F;
    const uint32_t man = h & 0x3FF;
    uint32_t u;
    if (exp == 0) {
      if (man == 0) {
        u = sign;
      } else {
        int e = -1;
        uint32_t m = man;
        do { m <<= 1; ++e; } while ((m & 0x400) == 0);
        u = sign | ((127 - 15 - e) << 23) | ((m & 0x3FF) << 13);
      }
    } else if (exp == 31) {
      u = sign | 0x7F800000 | (man << 13);
    } else {
      u = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &u, 4);
    return f;
  }

  /** bf16 is the high 16 bits of an f32. */
  static CTP_CROSS_FUN float BFloatToFloat(const uint8_t *b) {
    uint16_t h;
    std::memcpy(&h, b, 2);
    const uint32_t u = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
  }

  Type type_;
};

}  // namespace ctp

#endif  // CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_QUANT_H_
