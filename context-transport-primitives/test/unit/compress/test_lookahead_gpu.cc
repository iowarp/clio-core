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

// ctp::LookaheadCodec: groups of timesteps round-trip within the bound, the
// decoder rebuilds the encoder's reconstruction bit for bit, and on smooth
// time series the blob is smaller than coding every frame with cuSZp alone.
#include "basic_test.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "clio_ctp/compress/cuszp.h"
#include "clio_ctp/compress/lookahead.h"
#include "clio_ctp/compress/nvcomp.h"

namespace {

/** nf device frames of n floats, freed on scope exit. */
struct DeviceFrames {
  std::vector<float *> p;
  DeviceFrames(int nf, size_t n) : p(nf, nullptr) {
    for (auto &f : p) REQUIRE(cudaMalloc(&f, n * sizeof(float)) == cudaSuccess);
  }
  ~DeviceFrames() {
    for (auto *f : p) cudaFree(f);
  }
};

/** A smooth field drifting over time, or white noise when noise is set. */
std::vector<std::vector<float>> MakeSeries(int nf, size_t n, bool noise) {
  std::vector<std::vector<float>> s(nf, std::vector<float>(n));
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> u(-1.0f, 1.0f);
  for (int t = 0; t < nf; ++t) {
    for (size_t i = 0; i < n; ++i) {
      s[t][i] = noise ? u(rng)
                      : std::sin(0.001f * i + 0.05f * t) +
                            0.3f * std::cos(0.0003f * i - 0.02f * t);
    }
  }
  return s;
}

/** Bytes of coding every frame on its own with the anchor codec. */
size_t SpatialBytes(ctp::Compressor *c, const DeviceFrames &d, size_t n) {
  size_t total = 0;
  std::vector<uint8_t> buf(2 * n * sizeof(float) + 65536);
  for (float *f : d.p) {
    size_t size = buf.size();
    REQUIRE(c->Compress(buf.data(), size, f, n * sizeof(float)));
    total += size;
  }
  return total;
}

/**
 * Encodes and decodes one series; checks the bound in double precision and
 * that Decode reproduces Encode's reconstruction exactly.
 * @return blob size in bytes.
 */
size_t RoundTrip(const std::vector<std::vector<float>> &s, double eb,
                 ctp::Compressor *anchor, ctp::Compressor *packer,
                 size_t *spatial_bytes = nullptr) {
  const int nf = static_cast<int>(s.size());
  const size_t n = s[0].size(), bytes = n * sizeof(float);
  DeviceFrames in(nf, n), recon(nf, n), out(nf, n);
  for (int t = 0; t < nf; ++t) {
    REQUIRE(cudaMemcpy(in.p[t], s[t].data(), bytes, cudaMemcpyHostToDevice) ==
            cudaSuccess);
    REQUIRE(cudaMemset(out.p[t], 0xFF, bytes) == cudaSuccess);  // garbage
  }
  ctp::LookaheadCodec codec(anchor, packer);
  std::vector<uint8_t> blob;
  REQUIRE(codec.Encode(in.p.data(), nf, n, eb, &blob, recon.p.data()));
  REQUIRE(codec.Decode(blob.data(), blob.size(), out.p.data()));
  std::vector<float> a(n), b(n);
  for (int t = 0; t < nf; ++t) {
    REQUIRE(cudaMemcpy(a.data(), out.p[t], bytes, cudaMemcpyDeviceToHost) ==
            cudaSuccess);
    REQUIRE(cudaMemcpy(b.data(), recon.p[t], bytes, cudaMemcpyDeviceToHost) ==
            cudaSuccess);
    REQUIRE(std::memcmp(a.data(), b.data(), bytes) == 0);
    double worst = 0.0;
    for (size_t i = 0; i < n; ++i) {
      worst = std::fmax(worst, std::fabs(static_cast<double>(a[i]) - s[t][i]));
    }
    REQUIRE(worst <= eb);
  }
  if (spatial_bytes != nullptr) *spatial_bytes = SpatialBytes(anchor, in, n);
  return blob.size();
}

bool HaveGpu() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

}  // namespace

TEST_CASE("LookaheadCodec round-trips groups within the bound", "[gpu][lookahead]") {
  if (!HaveGpu()) {
    WARN("No CUDA device; skipping LookaheadCodec test");
    return;
  }
  const size_t n = 1 << 20;
  const double eb = 1e-3;
  ctp::Cuszp anchor;
  REQUIRE(anchor.SetErrorBound(eb));
  ctp::NvComp packer(ctp::NvCompAlgo::BITCOMP);

  PAGE_DIVIDE("smooth series: exact decode, within bound, smaller than cuSZp alone") {
    size_t spatial = 0;
    const size_t la = RoundTrip(MakeSeries(9, n, false), eb, &anchor, &packer,
                                &spatial);
    INFO("look-ahead " << la << " bytes vs cuSZp per frame " << spatial);
    REQUIRE(la < spatial);
  }
  PAGE_DIVIDE("group sizes 1, 2 and 5") {
    for (int nf : {1, 2, 5}) RoundTrip(MakeSeries(nf, n, false), eb, &anchor, &packer);
  }
  PAGE_DIVIDE("white noise still decodes within the bound") {
    RoundTrip(MakeSeries(5, n, true), eb, &anchor, &packer);
  }
}

TEST_CASE("TemporalProbe picks look-ahead only when time is smoother than space",
          "[gpu][lookahead]") {
  if (!HaveGpu()) {
    WARN("No CUDA device; skipping TemporalProbe test");
    return;
  }
  namespace pp = ctp::compress::preprocess;
  const size_t n = 1 << 20;
  const double eb = 1e-3;
  // Rough in space (a fixed random texture) but slowly drifting in time: the
  // texture cancels between frames, so look-ahead leftovers round to zero
  // while spatial neighbours predict badly.
  std::vector<float> texture(n);
  std::mt19937 rng(11);
  std::uniform_real_distribution<float> u(-1.0f, 1.0f);
  for (auto &v : texture) v = u(rng);
  std::vector<std::vector<float>> textured(3, std::vector<float>(n));
  // Smooth in space, but each frame has an unrelated phase: time is useless.
  std::vector<std::vector<float>> jumpy(3, std::vector<float>(n));
  const float phase[3] = {0.0f, 2.1f, 4.4f};
  for (int t = 0; t < 3; ++t) {
    for (size_t i = 0; i < n; ++i) {
      textured[t][i] = texture[i] + 0.1f * std::sin(0.001f * i + 0.05f * t);
      jumpy[t][i] = std::sin(0.001f * i + phase[t]);
    }
  }
  auto probe = [&](const std::vector<std::vector<float>> &s) {
    DeviceFrames d(3, n);
    for (int t = 0; t < 3; ++t) {
      REQUIRE(cudaMemcpy(d.p[t], s[t].data(), n * sizeof(float),
                         cudaMemcpyHostToDevice) == cudaSuccess);
    }
    pp::TemporalProbeResult r;
    REQUIRE(pp::TemporalProbeDevice(d.p[0], d.p[1], d.p[2], n, 1, 1, eb, &r,
                                    nullptr));
    return r;
  };
  const pp::TemporalProbeResult a = probe(textured), b = probe(jumpy);
  INFO("textured: temporal " << a.temporal_bits << " bits vs spatial "
       << a.spatial_bits << "; jumpy: temporal " << b.temporal_bits
       << " vs spatial " << b.spatial_bits);
  REQUIRE(pp::LookaheadPays(a));
  REQUIRE_FALSE(pp::LookaheadPays(b));
  REQUIRE(a.temporal_zero > 0.9);

  // The trial encode reaches the same verdicts with the real codecs.
  ctp::Cuszp anchor;
  REQUIRE(anchor.SetErrorBound(eb));
  ctp::NvComp packer(ctp::NvCompAlgo::BITCOMP);
  ctp::LookaheadCodec codec(&anchor, &packer);
  auto trial = [&](const std::vector<std::vector<float>> &s) {
    DeviceFrames d(3, n);
    for (int t = 0; t < 3; ++t) {
      REQUIRE(cudaMemcpy(d.p[t], s[t].data(), n * sizeof(float),
                         cudaMemcpyHostToDevice) == cudaSuccess);
    }
    ctp::LookaheadTrial r;
    REQUIRE(codec.Trial(d.p[0], d.p[1], d.p[2], n, eb, &r));
    return r;
  };
  const ctp::LookaheadTrial ta = trial(textured), tb = trial(jumpy);
  INFO("trial textured: spatial " << ta.spatial_bytes << " B vs look-ahead "
       << ta.lookahead_bytes << " B; jumpy: spatial " << tb.spatial_bytes
       << " B vs look-ahead " << tb.lookahead_bytes << " B");
  REQUIRE(ta.LookaheadPays());
  REQUIRE_FALSE(tb.LookaheadPays());
}

TEST_CASE("LookaheadCodec plan codes anchors first, then midpoints",
          "[lookahead]") {
  const auto steps = ctp::LookaheadCodec::Plan(9);
  REQUIRE(steps.size() == 9);
  REQUIRE(steps[0].t == 0);
  REQUIRE(steps[1].t == 8);
  REQUIRE(steps[1].kind == ctp::LookaheadKind::kForward);
  REQUIRE(steps[2].t == 4);
  REQUIRE(steps[2].lo == 0);
  REQUIRE(steps[2].hi == 8);
  std::vector<int> seen(9, 0);
  for (const auto &s : steps) seen[s.t]++;
  for (int c : seen) REQUIRE(c == 1);
}
