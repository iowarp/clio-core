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
#include "clio_ctp/compress/quant.h"
#include <cstdio>
#include <cmath>
int fails = 0;
void chk(const char* n, float got, float want) {
  if (std::fabs(got - want) > 1e-4f) { printf("FAIL %s got %f want %f\n", n, got, want); ++fails; }
}
int main() {
  using Q = ctp::QuantCodec;
  // geometry must match the GGUF layouts
  chk("q4k bytes", Q::BlockBytes(Q::Type::kQ4_K), 144);
  chk("q5k bytes", Q::BlockBytes(Q::Type::kQ5_K), 176);
  chk("q6k bytes", Q::BlockBytes(Q::Type::kQ6_K), 210);
  chk("q8_0 bytes", Q::BlockBytes(Q::Type::kQ8_0), 34);

  // half decode against known bit patterns
  uint8_t h[2];
  uint16_t one = 0x3C00; memcpy(h, &one, 2);
  float o[256];
  Q::DecodeBlock(Q::Type::kF16, h, o); chk("f16 1.0", o[0], 1.0f);
  uint16_t neg2 = 0xC000; memcpy(h, &neg2, 2);
  Q::DecodeBlock(Q::Type::kF16, h, o); chk("f16 -2.0", o[0], -2.0f);

  // Q8_0: d=1.0 (half), q[i]=i-16  -> exact
  uint8_t b[64] = {0};
  memcpy(b, &one, 2);
  for (int i = 0; i < 32; ++i) b[2+i] = (uint8_t)(int8_t)(i - 16);
  Q::DecodeBlock(Q::Type::kQ8_0, b, o);
  for (int i = 0; i < 32; ++i) chk("q8_0", o[i], (float)(i-16));

  // Q4_0: d=1.0, nibbles -> (q-8)
  uint8_t c[32] = {0};
  memcpy(c, &one, 2);
  for (int i = 0; i < 16; ++i) c[2+i] = (uint8_t)((i & 0xF) | ((15 - i) << 4));
  Q::DecodeBlock(Q::Type::kQ4_0, c, o);
  for (int i = 0; i < 16; ++i) chk("q4_0 lo", o[i], (float)(i - 8));
  for (int i = 0; i < 16; ++i) chk("q4_0 hi", o[16+i], (float)((15 - i) - 8));

  // lane/step partitioning must equal the serial decode (warp-cooperative use)
  float ser[256], par[256];
  uint8_t k[144]; for (int i = 0; i < 144; ++i) k[i] = (uint8_t)(i * 7 + 3);
  Q::DecodeBlock(Q::Type::kQ4_K, k, ser);
  for (int l = 0; l < 32; ++l) Q::DecodeBlock(Q::Type::kQ4_K, k, par, l, 32);
  for (int i = 0; i < 256; ++i) chk("q4k lane==serial", par[i], ser[i]);

  printf(fails ? "FAILURES: %d\n" : "ALL PASS\n", fails);
  return fails != 0;
}
