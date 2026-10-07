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
 * dtschedule candidate presets (issue #1205). ctp::CompressionPreset is
 * 0-based while the wire format, the traces and the Q-table are 1-based;
 * casting one onto the other never tried FAST and ran BEST as "balanced".
 * No runtime needed: candidates and codecs come straight from the factory.
 */

#include <cmath>
#include <set>
#include <string>
#include <vector>

#include "../../../context-runtime/test/simple_test.h"
#include "clio_cte/compressor/compression_header.h"
#include "clio_cte/dtschedule/ccm/candidates.h"

using clio::cte::compressor::CompressPreset;
using clio::cte::compressor::FromWirePreset;
using clio::cte::compressor::ToWirePreset;

namespace {

/** Compressed size of `in` with lib at preset (0 if the codec failed). */
size_t CompressedSize(const std::string &lib, ctp::CompressionPreset preset,
                      std::vector<char> &in) {
  auto codec = ctp::CompressionFactory::GetPreset(lib, preset);
  REQUIRE(codec != nullptr);
  std::vector<char> out(in.size() * 2 + 1024);
  size_t out_size = out.size();
  if (!codec->Compress(out.data(), out_size, in.data(), in.size())) return 0;
  return out_size;
}

}  // namespace

TEST_CASE("dtschedule presets: wire conversion round-trips", "[dtschedule][1205]") {
  REQUIRE(ToWirePreset(ctp::CompressionPreset::FAST) ==
          static_cast<uint32_t>(CompressPreset::kFast));
  REQUIRE(ToWirePreset(ctp::CompressionPreset::BALANCED) ==
          static_cast<uint32_t>(CompressPreset::kBalanced));
  REQUIRE(ToWirePreset(ctp::CompressionPreset::BEST) ==
          static_cast<uint32_t>(CompressPreset::kBest));
  REQUIRE(ToWirePreset(ctp::CompressionPreset::DEFAULT) ==
          static_cast<uint32_t>(CompressPreset::kBalanced));
  for (auto p : {ctp::CompressionPreset::FAST, ctp::CompressionPreset::BALANCED,
                 ctp::CompressionPreset::BEST}) {
    REQUIRE(FromWirePreset(ToWirePreset(p)) == p);
  }
}

TEST_CASE("dtschedule presets: every library is tried fast, balanced, best",
          "[dtschedule][1205]") {
  clio::cte::dtschedule::ccm::CandidateSet set;
  REQUIRE(set.BuildAll() > 0);
  std::set<ctp::CompressionPreset> zstd;
  for (const auto &c : set.GetAll()) {
    if (c.lib_ == "zstd") zstd.insert(c.preset_);
  }
  // Before the fix this was {BALANCED, BEST, DEFAULT}: FAST never ran.
  const std::set<ctp::CompressionPreset> want = {
      ctp::CompressionPreset::FAST, ctp::CompressionPreset::BALANCED,
      ctp::CompressionPreset::BEST};
  REQUIRE(zstd == want);
}

TEST_CASE("dtschedule presets: fast and best are different levels",
          "[dtschedule][1205]") {
  // Smooth float data: zstd's levels compress it to clearly different sizes.
  std::vector<char> in(1 << 20);
  auto *f = reinterpret_cast<float *>(in.data());
  for (size_t i = 0; i < in.size() / sizeof(float); ++i) {
    f[i] = static_cast<float>(std::sin(i * 0.001) * 1000.0);
  }
  const size_t fast = CompressedSize("zstd", ctp::CompressionPreset::FAST, in);
  const size_t best = CompressedSize("zstd", ctp::CompressionPreset::BEST, in);
  INFO("zstd fast=" << fast << " best=" << best);
  REQUIRE(fast > 0);
  REQUIRE(best > 0);
  REQUIRE(best < fast);
}

SIMPLE_TEST_MAIN()
