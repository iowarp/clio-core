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

#ifndef SAFE_BDEV_EC_GF256_H_
#define SAFE_BDEV_EC_GF256_H_

#include <cstddef>
#include <cstdint>

/**
 * Arithmetic over the Galois field GF(2^8).
 *
 * This is the algebra underneath Reed-Solomon erasure coding. Field elements
 * are bytes. Addition (and subtraction) is XOR; multiplication is carry-less
 * polynomial multiplication modulo the primitive polynomial 0x11d
 * (x^8 + x^4 + x^3 + x^2 + 1), implemented with log/exp lookup tables built
 * once at first use. The field is daemon-independent and header-only so it can
 * be linked into both the safe_bdev runtime and standalone unit tests.
 */

namespace clio::run::safe_bdev::ec {

/** Primitive polynomial for GF(2^8): x^8 + x^4 + x^3 + x^2 + 1. */
constexpr int kGfPrimitivePoly = 0x11d;

/** Log/exp tables for GF(2^8) multiplication. */
struct GfTables {
  uint8_t exp[512];  // exp[i] = generator^i (doubled to avoid a modulo)
  uint8_t log[256];  // log[exp[i]] = i; log[0] is unused

  GfTables() {
    int x = 1;
    for (int i = 0; i < 255; ++i) {
      exp[i] = static_cast<uint8_t>(x);
      log[x] = static_cast<uint8_t>(i);
      x <<= 1;
      if (x & 0x100) {
        x ^= kGfPrimitivePoly;
      }
    }
    // Duplicate the cycle so exp[a + b] needs no modulo for a,b in [0,255).
    for (int i = 255; i < 512; ++i) {
      exp[i] = exp[i - 255];
    }
    log[0] = 0;  // sentinel; never read for a valid multiply
  }
};

/** Single shared table instance (built on first call). */
inline const GfTables &GfTablesInstance() {
  static const GfTables tables;
  return tables;
}

/** Field addition (== subtraction): XOR. */
inline uint8_t GfAdd(uint8_t a, uint8_t b) { return a ^ b; }

/** Field multiplication. */
inline uint8_t GfMul(uint8_t a, uint8_t b) {
  if (a == 0 || b == 0) {
    return 0;
  }
  const GfTables &t = GfTablesInstance();
  return t.exp[static_cast<int>(t.log[a]) + static_cast<int>(t.log[b])];
}

/** Multiplicative inverse of a non-zero element. */
inline uint8_t GfInv(uint8_t a) {
  const GfTables &t = GfTablesInstance();
  // a^{-1} = generator^(255 - log[a]).
  return t.exp[255 - static_cast<int>(t.log[a])];
}

/** Field division a / b (b non-zero). */
inline uint8_t GfDiv(uint8_t a, uint8_t b) {
  if (a == 0) {
    return 0;
  }
  const GfTables &t = GfTablesInstance();
  int l = static_cast<int>(t.log[a]) - static_cast<int>(t.log[b]) + 255;
  return t.exp[l];
}

/**
 * Region multiply-accumulate: dst[i] ^= coeff * src[i] for i in [0, len).
 * This is the inner loop of both encoding (parity = sum of scaled data) and
 * decoding (data = sum of scaled survivors).
 */
inline void GfMulAddRegion(uint8_t *dst, const uint8_t *src, uint8_t coeff,
                           size_t len) {
  if (coeff == 0) {
    return;
  }
  const GfTables &t = GfTablesInstance();
  const int log_c = static_cast<int>(t.log[coeff]);
  for (size_t i = 0; i < len; ++i) {
    uint8_t s = src[i];
    if (s != 0) {
      dst[i] ^= t.exp[log_c + static_cast<int>(t.log[s])];
    }
  }
}

}  // namespace clio::run::safe_bdev::ec

#endif  // SAFE_BDEV_EC_GF256_H_
