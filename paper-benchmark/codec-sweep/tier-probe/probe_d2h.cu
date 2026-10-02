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

/**
 * @file probe_d2h.cu
 * @brief Measure the DRAM tier's bandwidth as the codec pipeline sees it: the
 * compressed chunk leaves the GPU into host DRAM. Copies TOTAL bytes in CHUNK
 * sized cudaMemcpy calls, device to pinned host and device to pageable host,
 * and a host memcpy for reference. Prints one CSV row per repetition:
 *   tier,method,transfer_bytes,rep,bytes,ms,GBps
 *
 *   probe_d2h [--chunk BYTES] [--total BYTES] [--reps N]
 */
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x)                                                            \
  do {                                                                   \
    cudaError_t e_ = (x);                                                \
    if (e_ != cudaSuccess) {                                             \
      std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_));      \
      std::exit(2);                                                      \
    }                                                                    \
  } while (0)

/**
 * Time TOTAL bytes copied device -> host in CHUNK-sized synchronous copies.
 * @param dst   host destination, TOTAL bytes
 * @param src   device source, TOTAL bytes
 * @param total bytes moved per repetition
 * @param chunk bytes per cudaMemcpy call
 * @return elapsed milliseconds (CUDA events on the legacy stream)
 */
static float TimeD2H(char *dst, const char *src, size_t total, size_t chunk) {
  cudaEvent_t a, b;
  CK(cudaEventCreate(&a));
  CK(cudaEventCreate(&b));
  CK(cudaEventRecord(a));
  for (size_t off = 0; off < total; off += chunk) {
    CK(cudaMemcpy(dst + off, src + off, chunk, cudaMemcpyDeviceToHost));
  }
  CK(cudaEventRecord(b));
  CK(cudaEventSynchronize(b));
  float ms = 0;
  CK(cudaEventElapsedTime(&ms, a, b));
  CK(cudaEventDestroy(a));
  CK(cudaEventDestroy(b));
  return ms;
}

/**
 * Time TOTAL bytes copied host -> host in CHUNK-sized memcpy calls.
 * @param dst   host destination
 * @param src   host source
 * @param total bytes per repetition
 * @param chunk bytes per memcpy call
 * @return elapsed milliseconds (steady clock)
 */
static double TimeMemcpy(char *dst, const char *src, size_t total,
                         size_t chunk) {
  auto t0 = std::chrono::steady_clock::now();
  for (size_t off = 0; off < total; off += chunk) {
    std::memcpy(dst + off, src + off, chunk);
  }
  auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

/** Print one CSV row. */
static void Row(const char *method, size_t chunk, int rep, size_t total,
                double ms) {
  std::printf("dram,%s,%zu,%d,%zu,%.4f,%.3f\n", method, chunk, rep, total, ms,
              total / (ms * 1e6));
}

int main(int argc, char **argv) {
  size_t chunk = 4u << 20, total = 2ull << 30;
  int reps = 5;
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string k = argv[i];
    if (k == "--chunk") chunk = std::strtoull(argv[i + 1], nullptr, 10);
    if (k == "--total") total = std::strtoull(argv[i + 1], nullptr, 10);
    if (k == "--reps") reps = std::atoi(argv[i + 1]);
  }
  total -= total % chunk;
  char *dev = nullptr, *pinned = nullptr;
  CK(cudaMalloc(&dev, total));
  CK(cudaMemset(dev, 0x5a, total));
  CK(cudaMallocHost(&pinned, total));
  std::vector<char> pageable(total, 1), pageable2(total, 2);  // touched
  std::printf("tier,method,transfer_bytes,rep,bytes,ms,GBps\n");
  TimeD2H(pinned, dev, total, chunk);  // warm-up, untimed
  for (int r = 1; r <= reps; ++r)
    Row("d2h_pinned", chunk, r, total, TimeD2H(pinned, dev, total, chunk));
  TimeD2H(pageable.data(), dev, total, chunk);
  for (int r = 1; r <= reps; ++r)
    Row("d2h_pageable", chunk, r, total,
        TimeD2H(pageable.data(), dev, total, chunk));
  TimeMemcpy(pageable2.data(), pageable.data(), total, chunk);
  for (int r = 1; r <= reps; ++r)
    Row("host_memcpy", chunk, r, total,
        TimeMemcpy(pageable2.data(), pageable.data(), total, chunk));
  CK(cudaFreeHost(pinned));
  CK(cudaFree(dev));
  return 0;
}
