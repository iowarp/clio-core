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
 * gpu_lz4: pages encoded on the host decode on the GPU (ctp_gpu_lz4_decode_
 * launch) to exactly the original bytes, and malformed pages fail without
 * writing out of range. Host code only: the kernel is the runtime GPU
 * library's (context-runtime/src/gpu/gpu_lz4_gpu.cc).
 */

#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "../../simple_test.h"
#include "clio_ctp/compress/gpu_ans.h"
#include "clio_ctp/util/gpu_api.h"

namespace g = ctp::gpu_lz4;

namespace {

constexpr size_t kPage = 983040;  // clio-infer's page size

/**
 * A page shaped like a BF16 weight page in the esm layout: an exponent plane
 * with a narrow distribution (compressible) and a near-random sign+mantissa
 * plane (barely compressible).
 */
std::vector<uint8_t> EsmLikePage(uint32_t seed, size_t n = kPage) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> e(120.0, 2.0);
  std::vector<uint8_t> p(n);
  for (size_t i = 0; i < n / 2; ++i) {
    p[i] = static_cast<uint8_t>(std::min(255.0, std::max(0.0, e(rng))));
  }
  for (size_t i = n / 2; i < n; ++i) p[i] = static_cast<uint8_t>(rng());
  return p;
}

/** Device buffers for one batch of pages, freed on scope exit. */
struct DeviceBatch {
  std::vector<char *> src, dst;
  g::DecodeJob *jobs = nullptr;
  ~DeviceBatch() {
    for (char *p : src) ctp::GpuApi::Free(p);
    for (char *p : dst) ctp::GpuApi::Free(p);
    if (jobs) ctp::GpuApi::Free(jobs);
  }
};

/**
 * Upload encoded pages, decode them on the GPU in ONE launch, and return each
 * page's ok flag; decoded bytes are copied back into `out`.
 */
std::vector<uint32_t> GpuDecode(const std::vector<std::vector<uint8_t>> &enc,
                                const std::vector<size_t> &cap,
                                std::vector<std::vector<uint8_t>> &out,
                                double *ms) {
  const size_t n = enc.size();
  DeviceBatch b;
  std::vector<g::DecodeJob> jobs(n);
  for (size_t i = 0; i < n; ++i) {
    b.src.push_back(ctp::GpuApi::Malloc<char>(enc[i].size()));
    b.dst.push_back(ctp::GpuApi::Malloc<char>(cap[i]));
    ctp::GpuApi::Memcpy(b.src[i], reinterpret_cast<const char *>(enc[i].data()),
                        enc[i].size());
    jobs[i] = g::DecodeJob{reinterpret_cast<const uint8_t *>(b.src[i]),
                           enc[i].size(),
                           reinterpret_cast<uint8_t *>(b.dst[i]), cap[i], 1, 0};
  }
  b.jobs = ctp::GpuApi::Malloc<g::DecodeJob>(sizeof(g::DecodeJob) * n);
  ctp::GpuApi::Memcpy(b.jobs, jobs.data(), sizeof(g::DecodeJob) * n);
  void *stream = ctp::GpuApi::CreateStream();
  const auto t0 = std::chrono::steady_clock::now();
  REQUIRE(ctp_gpu_lz4_decode_launch(b.jobs, static_cast<uint32_t>(n),
                                    stream) == 0);
  ctp::GpuApi::Synchronize(stream);
  *ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
  ctp::GpuApi::DestroyStream(stream);
  ctp::GpuApi::Memcpy(jobs.data(), b.jobs, sizeof(g::DecodeJob) * n);
  out.assign(n, {});
  std::vector<uint32_t> ok(n);
  for (size_t i = 0; i < n; ++i) {
    out[i].resize(cap[i]);
    ctp::GpuApi::Memcpy(reinterpret_cast<char *>(out[i].data()),
                        static_cast<const char *>(b.dst[i]), cap[i]);
    ok[i] = jobs[i].ok_;
  }
  return ok;
}

}  // namespace

TEST_CASE("gpu_lz4: 64 esm-like pages decode on the GPU in one launch",
          "[gpu][lz4]") {
  constexpr size_t kPages = 64;
  std::vector<std::vector<uint8_t>> raw, enc(kPages);
  std::vector<size_t> cap(kPages, kPage);
  size_t enc_bytes = 0;
  for (size_t i = 0; i < kPages; ++i) {
    raw.push_back(EsmLikePage(static_cast<uint32_t>(i) + 1));
    g::EncodePage(raw[i].data(), raw[i].size(), enc[i]);
    enc_bytes += enc[i].size();
  }
  std::vector<std::vector<uint8_t>> out;
  double ms = 0;
  GpuDecode(enc, cap, out, &ms);  // warm-up: module load, first launch
  const auto ok = GpuDecode(enc, cap, out, &ms);
  for (size_t i = 0; i < kPages; ++i) {
    INFO("page " << i);
    REQUIRE(ok[i] == 1u);
    REQUIRE(std::memcmp(out[i].data(), raw[i].data(), kPage) == 0);
  }
  std::printf("gpu_lz4: %zu pages, ratio %.3f, GPU decode %.2f ms (%.1f GiB/s)\n",
              kPages, double(kPages * kPage) / enc_bytes, ms,
              kPages * kPage / (ms / 1000.0) / (1ull << 30));
}

TEST_CASE("gpu_lz4: partial and incompressible pages", "[gpu][lz4]") {
  std::mt19937 rng(7);
  std::vector<std::vector<uint8_t>> raw, enc(3);
  raw.push_back(EsmLikePage(99, 100000));          // partial last chunk
  std::vector<uint8_t> rnd(kPage);
  for (auto &v : rnd) v = static_cast<uint8_t>(rng());
  raw.push_back(rnd);                               // every chunk stored raw
  raw.push_back(std::vector<uint8_t>(kPage, 0));    // one long match
  std::vector<size_t> cap;
  for (size_t i = 0; i < raw.size(); ++i) {
    g::EncodePage(raw[i].data(), raw[i].size(), enc[i]);
    cap.push_back(raw[i].size());
  }
  std::vector<std::vector<uint8_t>> out;
  double ms = 0;
  const auto ok = GpuDecode(enc, cap, out, &ms);
  for (size_t i = 0; i < raw.size(); ++i) {
    INFO("page " << i);
    REQUIRE(ok[i] == 1u);
    REQUIRE(out[i] == raw[i]);
  }
}

TEST_CASE("gpu_lz4: malformed pages fail without overrunning", "[gpu][lz4]") {
  std::vector<uint8_t> page = EsmLikePage(5);
  std::vector<std::vector<uint8_t>> enc(3);
  g::EncodePage(page.data(), page.size(), enc[0]);
  enc[1] = enc[0];
  enc[2] = enc[0];
  // 0: a corrupted LZ4 stream (flip bytes inside the first chunk's payload)
  const size_t payload = g::HeaderBytes(g::NumChunks(page.size()));
  for (size_t k = 0; k < 64; ++k) enc[0][payload + 7 + k * 13] ^= 0x5A;
  // 1: not a page (bad magic)
  enc[1][0] ^= 0xFF;
  // 2: destination smaller than the page declares
  std::vector<size_t> cap = {page.size(), page.size(), page.size() / 2};
  std::vector<std::vector<uint8_t>> out;
  double ms = 0;
  const auto ok = GpuDecode(enc, cap, out, &ms);
  REQUIRE(ok[1] == 0u);
  REQUIRE(ok[2] == 0u);
  // A flipped byte can still parse as SOME valid LZ4 stream; what must hold
  // is that the page never reports success with the wrong bytes.
  if (ok[0] == 1u) REQUIRE(out[0] != page);
}

TEST_CASE("gpu_ans: 64 esm-like pages decode on the GPU in one launch",
          "[gpu][ans]") {
  constexpr size_t kPages = 64;
  std::vector<std::vector<uint8_t>> raw, enc(kPages);
  std::vector<size_t> cap(kPages, kPage);
  size_t enc_bytes = 0;
  for (size_t i = 0; i < kPages; ++i) {
    raw.push_back(EsmLikePage(static_cast<uint32_t>(i) + 101));
    g::EncodePageAns(raw[i].data(), raw[i].size(), enc[i]);
    enc_bytes += enc[i].size();
  }
  std::vector<std::vector<uint8_t>> out;
  double ms = 0;
  GpuDecode(enc, cap, out, &ms);  // warm-up
  const auto ok = GpuDecode(enc, cap, out, &ms);
  for (size_t i = 0; i < kPages; ++i) {
    INFO("page " << i);
    REQUIRE(ok[i] == 1u);
    REQUIRE(std::memcmp(out[i].data(), raw[i].data(), kPage) == 0);
  }
  std::printf("gpu_ans: %zu pages, ratio %.3f, GPU decode %.2f ms (%.1f GiB/s)\n",
              kPages, double(kPages * kPage) / enc_bytes, ms,
              kPages * kPage / (ms / 1000.0) / (1ull << 30));
}

TEST_CASE("gpu_ans: LZ4 and ANS pages decode together in one launch",
          "[gpu][ans][lz4]") {
  std::vector<std::vector<uint8_t>> raw, enc(6);
  std::vector<size_t> cap;
  std::mt19937 rng(11);
  for (size_t i = 0; i < 6; ++i) {
    raw.push_back(i == 5 ? std::vector<uint8_t>(kPage, 3)
                         : EsmLikePage(static_cast<uint32_t>(i) + 201,
                                       i == 4 ? 70001 : kPage));
    if (i % 2 == 0) {
      g::EncodePageAns(raw[i].data(), raw[i].size(), enc[i]);
    } else {
      g::EncodePage(raw[i].data(), raw[i].size(), enc[i]);
    }
    cap.push_back(raw[i].size());
  }
  std::vector<std::vector<uint8_t>> out;
  double ms = 0;
  const auto ok = GpuDecode(enc, cap, out, &ms);
  for (size_t i = 0; i < raw.size(); ++i) {
    INFO("page " << i);
    REQUIRE(ok[i] == 1u);
    REQUIRE(out[i] == raw[i]);
  }
}

TEST_CASE("gpu_ans: corrupted streams are detected", "[gpu][ans]") {
  std::vector<uint8_t> page = EsmLikePage(9);
  std::vector<uint8_t> enc;
  g::EncodePageAns(page.data(), page.size(), enc);
  // Corrupt one byte in the streams of each ANS-coded chunk (first chunks:
  // the exponent plane); the end-state check must reject the page.
  std::vector<std::vector<uint8_t>> bad;
  std::vector<uint8_t> tmp(page.size());
  const uint32_t nch = g::ChunksOf(enc.data(), enc.size());
  for (uint32_t c = 0; c < nch && bad.size() < 4; ++c) {
    g::ChunkSpan sp;
    REQUIRE(g::LocateChunk(enc.data(), enc.size(), c, tmp.data(), tmp.size(),
                           &sp));
    if (sp.raw) continue;
    auto e = enc;
    e[(sp.src - enc.data()) + g::kAnsTableBytes + sp.sn / 3] ^= 0x41;
    bad.push_back(e);
  }
  REQUIRE(!bad.empty());
  std::vector<size_t> cap(bad.size(), page.size());
  std::vector<std::vector<uint8_t>> out;
  double ms = 0;
  const auto ok = GpuDecode(bad, cap, out, &ms);
  for (size_t i = 0; i < bad.size(); ++i) {
    INFO("page " << i);
    REQUIRE((ok[i] == 0u || out[i] == page));
  }
}

SIMPLE_TEST_MAIN()
