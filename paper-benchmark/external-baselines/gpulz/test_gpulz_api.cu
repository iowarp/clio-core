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
 * @file test_gpulz_api.cu
 * @brief Round-trip checks of gpulz_api: edge cases, then real dump files.
 *
 *   test_gpulz_api [file ...]
 *
 * Edge cases (no files needed): whole blocks, a ragged tail, one byte, a
 * misaligned input, a misaligned stream, an undersized output buffer, and a
 * size mismatch at decompression. Each file is then compressed and
 * decompressed as one buffer, compared byte for byte, and its ratio and
 * throughput (CUDA events around each call) printed. Exit status is the
 * number of failures.
 */
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <vector>

#include "gpulz_api.h"

namespace {

int g_failures = 0;

/** Record and print one check. */
void Check(bool ok, const char *what) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", what);
  if (!ok) ++g_failures;
}

/**
 * Compress and decompress host bytes through device buffers at the given
 * byte offsets (to exercise misaligned pointers).
 *
 * @param host      input bytes.
 * @param in_off    offset of the device input inside its allocation.
 * @param stream_off offset of the stream copy handed to decompression.
 * @param ratio     receives n / stream bytes.
 * @param ms_c,ms_d receive compress / decompress milliseconds.
 * @return true when the output matches the input bit for bit.
 */
bool RoundTrip(const std::vector<unsigned char> &host, size_t in_off,
               size_t stream_off, double *ratio, float *ms_c, float *ms_d) {
  const size_t n = host.size();
  cudaStream_t s;
  cudaStreamCreate(&s);
  unsigned char *d_in = nullptr, *d_cmp = nullptr, *d_cmp2 = nullptr, *d_out = nullptr;
  const size_t cap = gpulz_compress_bound(n);
  cudaMalloc(&d_in, n + in_off);
  cudaMalloc(&d_cmp, cap);
  cudaMalloc(&d_cmp2, cap + stream_off);
  cudaMalloc(&d_out, n);
  cudaMemcpy(d_in + in_off, host.data(), n, cudaMemcpyHostToDevice);
  cudaEvent_t a, b, c;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  cudaEventCreate(&c);
  size_t cbytes = 0;
  cudaEventRecord(a, s);
  int rc = gpulz_compress(d_in + in_off, n, d_cmp, cap, &cbytes, s);
  cudaEventRecord(b, s);
  bool ok = rc == 0;
  if (ok) {
    cudaMemcpy(d_cmp2 + stream_off, d_cmp, cbytes, cudaMemcpyDeviceToDevice);
    rc = gpulz_decompress(d_cmp2 + stream_off, cbytes, d_out, n, s);
    cudaEventRecord(c, s);
    cudaEventSynchronize(c);
    ok = rc == 0;
  }
  if (ok) {
    std::vector<unsigned char> back(n);
    cudaMemcpy(back.data(), d_out, n, cudaMemcpyDeviceToHost);
    ok = std::memcmp(back.data(), host.data(), n) == 0;
    *ratio = static_cast<double>(n) / static_cast<double>(cbytes);
    cudaEventElapsedTime(ms_c, a, b);
    cudaEventElapsedTime(ms_d, b, c);
  }
  cudaFree(d_in);
  cudaFree(d_cmp);
  cudaFree(d_cmp2);
  cudaFree(d_out);
  cudaStreamDestroy(s);
  return ok;
}

/** Bytes that compress: slowly varying float32 values. */
std::vector<unsigned char> Smooth(size_t n) {
  std::vector<unsigned char> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = static_cast<unsigned char>((i / 64) & 0xff);
  return v;
}

/** Bytes that do not: uniform random. */
std::vector<unsigned char> Random(size_t n) {
  std::mt19937 g(7);
  std::vector<unsigned char> v(n);
  for (auto &x : v) x = static_cast<unsigned char>(g());
  return v;
}

void EdgeCases() {
  double r;
  float mc, md;
  Check(RoundTrip(Smooth(8 << 20), 0, 0, &r, &mc, &md), "8 MiB, whole blocks");
  Check(RoundTrip(Random(8 << 20), 0, 0, &r, &mc, &md), "8 MiB random (incompressible)");
  Check(RoundTrip(Smooth(1000003), 0, 0, &r, &mc, &md), "ragged tail (1000003 B)");
  Check(RoundTrip(Smooth(1), 0, 0, &r, &mc, &md), "one byte");
  Check(RoundTrip(Smooth(65536), 1, 0, &r, &mc, &md), "misaligned input");
  Check(RoundTrip(Smooth(65536), 0, 3, &r, &mc, &md), "misaligned stream");

  // An output buffer smaller than the stream must be refused, not overrun.
  const size_t n = 1 << 20;
  auto host = Random(n);
  unsigned char *d_in = nullptr, *d_cmp = nullptr, *d_out = nullptr;
  cudaMalloc(&d_in, n);
  cudaMalloc(&d_cmp, n / 2);
  cudaMalloc(&d_out, n);
  cudaMemcpy(d_in, host.data(), n, cudaMemcpyHostToDevice);
  size_t cbytes = 0;
  Check(gpulz_compress(d_in, n, d_cmp, n / 2, &cbytes, nullptr) == -2,
        "undersized output refused (-2)");
  // A stream decompressed with the wrong size must be refused.
  cudaFree(d_cmp);
  cudaMalloc(&d_cmp, gpulz_compress_bound(n));
  const int rc = gpulz_compress(d_in, n, d_cmp, gpulz_compress_bound(n), &cbytes, nullptr);
  Check(rc == 0 && gpulz_decompress(d_cmp, cbytes, d_out, n - 4, nullptr) == -3,
        "size mismatch refused (-3)");
  cudaFree(d_in);
  cudaFree(d_cmp);
  cudaFree(d_out);
}

}  // namespace

int main(int argc, char **argv) {
  EdgeCases();
  for (int i = 1; i < argc; ++i) {
    std::ifstream f(argv[i], std::ios::binary);
    std::vector<unsigned char> host((std::istreambuf_iterator<char>(f)),
                                    std::istreambuf_iterator<char>());
    if (host.empty()) {
      std::printf("SKIP  %s (empty or unreadable)\n", argv[i]);
      continue;
    }
    double ratio = 0;
    float mc = 0, md = 0;
    // Twice: the first call grows the thread's buffers; the second is timed.
    RoundTrip(host, 0, 0, &ratio, &mc, &md);
    const bool ok = RoundTrip(host, 0, 0, &ratio, &mc, &md);
    std::printf("%s  %-60s %9zu B  ratio %.3f  comp %.2f ms (%.1f GB/s)  "
                "decomp %.2f ms (%.1f GB/s)\n",
                ok ? "PASS" : "FAIL", argv[i], host.size(), ratio, mc,
                host.size() / (mc * 1e6), md, host.size() / (md * 1e6));
    if (!ok) ++g_failures;
  }
  std::printf("%d failure(s)\n", g_failures);
  return g_failures;
}
