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
 * cudaStreamCreate Cost Benchmark
 *
 * Measures the host-side cost of CUDA stream creation/destruction as
 * precisely as possible. This is the primitive that ctp::NvComp
 * (clio_ctp/compress/nvcomp.h) creates fresh on every single Compress/
 * Decompress call; this benchmark exists to quantify whether caching a
 * stream across calls would be a worthwhile optimization.
 *
 * Three patterns are measured, each as a per-call distribution (not just an
 * average), because the CUDA driver pools freed stream objects -- bulk
 * creation and create/destroy churn are NOT the same cost:
 *
 *   1. create-only (bulk, blocking)        -- cudaStreamCreate in a batch
 *   2. create+destroy churn (blocking)     -- the exact pattern NvComp uses
 *   3. create+destroy churn (non-blocking) -- cudaStreamNonBlocking variant
 *
 * Methodology for an accurate number:
 *   - The CUDA context is warmed up (cudaFree(0) + throwaway create/destroy +
 *     device sync) before any timing, since first-use context init costs
 *     hundreds of ms and would otherwise swamp the measurement.
 *   - Several hundred untimed iterations of every pattern run first so any
 *     one-time stream-pool growth is already paid before timing starts.
 *   - Each individual call is timed with std::chrono::steady_clock
 *     (a monotonic host clock); cudaStreamCreate is a synchronous host-side
 *     driver call, so CUDA events -- which time the device timeline -- are
 *     not the right tool here.
 *   - Many rounds x iterations are collected into one sample set per pattern,
 *     from which min/median/mean/stddev/p99/max are reported. A single call
 *     is a few microseconds and dominated by clock-read noise, so the
 *     reported numbers are derived from tens of thousands of samples.
 *   - Each created stream handle is folded into a `volatile` sink so the
 *     compiler cannot prove the calls are dead and elide them.
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using DurUs = std::chrono::duration<double, std::micro>;

constexpr int kWarmupIters = 500;
constexpr size_t kRounds = 10;
constexpr size_t kItersPerRound = 2000;

/** volatile sink: defeats dead-code elimination of the timed CUDA calls. */
volatile std::uintptr_t g_sink = 0;

void CheckCuda(cudaError_t err, const char *what) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(err));
    std::exit(1);
  }
}

struct Stats {
  double min;
  double median;
  double mean;
  double stddev;
  double p99;
  double max;
};

Stats Summarize(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  const size_t n = samples.size();
  double sum = 0.0;
  for (double v : samples) sum += v;
  const double mean = sum / static_cast<double>(n);
  double sq_diff = 0.0;
  for (double v : samples) sq_diff += (v - mean) * (v - mean);
  const double stddev = std::sqrt(sq_diff / static_cast<double>(n));
  auto percentile = [&](double p) {
    const size_t idx = static_cast<size_t>(p * static_cast<double>(n - 1));
    return samples[idx];
  };
  return Stats{samples.front(), percentile(0.5), mean, stddev, percentile(0.99),
               samples.back()};
}

void PrintStats(const std::string &label, size_t n, const Stats &s) {
  std::printf(
      "%-36s  n=%-7zu  min=%8.3f  median=%8.3f  mean=%8.3f  stddev=%7.3f  "
      "p99=%8.3f  max=%9.3f  (us)\n",
      label.c_str(), n, s.min, s.median, s.mean, s.stddev, s.p99, s.max);
}

/** Pay the one-time CUDA context initialization cost (100s of ms) up front. */
void WarmUpContext() {
  CheckCuda(cudaFree(nullptr), "cudaFree(0) context warmup");
  cudaStream_t s = nullptr;
  CheckCuda(cudaStreamCreate(&s), "warmup stream create");
  CheckCuda(cudaStreamDestroy(s), "warmup stream destroy");
  CheckCuda(cudaDeviceSynchronize(), "warmup device sync");
}

/**
 * Run `iters` untimed create+destroy cycles so the driver's stream pool is
 * already at steady state before any pattern is timed.
 */
void WarmUpPattern(unsigned int flags) {
  for (int i = 0; i < kWarmupIters; ++i) {
    cudaStream_t s = nullptr;
    CheckCuda(cudaStreamCreateWithFlags(&s, flags), "pattern warmup create");
    CheckCuda(cudaStreamDestroy(s), "pattern warmup destroy");
  }
}

/**
 * Pattern 1: create N streams into an array (timing each individual
 * cudaStreamCreate call), then destroy them all untimed before the next
 * round. This is the literal "cost of cudaStreamCreate" under bulk
 * allocation pressure -- the driver must grow its stream pool rather than
 * reuse just-freed slots.
 */
std::vector<double> BenchCreateOnlyBulk() {
  WarmUpPattern(cudaStreamDefault);

  std::vector<double> samples;
  samples.reserve(kRounds * kItersPerRound);
  std::vector<cudaStream_t> streams(kItersPerRound, nullptr);

  for (size_t r = 0; r < kRounds; ++r) {
    for (size_t i = 0; i < kItersPerRound; ++i) {
      const auto t0 = Clock::now();
      CheckCuda(cudaStreamCreate(&streams[i]), "cudaStreamCreate");
      const auto t1 = Clock::now();
      samples.push_back(DurUs(t1 - t0).count());
      g_sink ^= reinterpret_cast<std::uintptr_t>(streams[i]);
    }
    for (cudaStream_t s : streams) {
      CheckCuda(cudaStreamDestroy(s), "cudaStreamDestroy (bulk cleanup)");
    }
  }
  return samples;
}

/**
 * Pattern 2/3: create immediately followed by destroy, in a loop -- the
 * exact sequence ctp::NvComp::Compress/Decompress run on every call. The
 * driver pools just-freed stream objects, so this typically reports a
 * *lower* per-call cost than bulk creation above; that gap is precisely
 * what answers "would caching a stream across calls actually help?"
 */
std::vector<double> BenchCreateDestroyChurn(unsigned int flags, const char *what) {
  WarmUpPattern(flags);

  std::vector<double> samples;
  samples.reserve(kRounds * kItersPerRound);

  for (size_t r = 0; r < kRounds; ++r) {
    for (size_t i = 0; i < kItersPerRound; ++i) {
      cudaStream_t s = nullptr;
      const auto t0 = Clock::now();
      CheckCuda(cudaStreamCreateWithFlags(&s, flags), what);
      CheckCuda(cudaStreamDestroy(s), "cudaStreamDestroy (churn)");
      const auto t1 = Clock::now();
      samples.push_back(DurUs(t1 - t0).count());
      g_sink ^= reinterpret_cast<std::uintptr_t>(s);
    }
  }
  return samples;
}

}  // namespace

int main() {
  CheckCuda(cudaSetDevice(0), "cudaSetDevice");
  WarmUpContext();

  std::printf(
      "cudaStreamCreate cost benchmark -- %zu rounds x %zu timed iters per "
      "pattern (host-timed, std::chrono::steady_clock, %d untimed warmup "
      "iters per pattern)\n\n",
      kRounds, kItersPerRound, kWarmupIters);
  std::printf("%-36s  %-9s  %8s  %8s  %8s  %7s  %8s  %9s\n", "pattern",
              "samples", "min", "median", "mean", "stddev", "p99", "max");

  const auto bulk = BenchCreateOnlyBulk();
  PrintStats("create-only (bulk, blocking)", bulk.size(), Summarize(bulk));

  const auto churn_blocking =
      BenchCreateDestroyChurn(cudaStreamDefault, "cudaStreamCreateWithFlags(default)");
  PrintStats("create+destroy churn (blocking)", churn_blocking.size(),
             Summarize(churn_blocking));

  const auto churn_nonblocking = BenchCreateDestroyChurn(
      cudaStreamNonBlocking, "cudaStreamCreateWithFlags(non-blocking)");
  PrintStats("create+destroy churn (non-blocking)", churn_nonblocking.size(),
             Summarize(churn_nonblocking));

  std::printf(
      "\nNote: NvComp::Compress/Decompress create+destroy exactly one stream "
      "per call (the \"churn\" rows). Compare those numbers against the cost "
      "of the compression work itself to judge whether caching a stream "
      "across calls is worth the added lifetime/thread-safety complexity.\n");

  return 0;
}
