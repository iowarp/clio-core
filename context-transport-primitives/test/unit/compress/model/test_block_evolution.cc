/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

/**
 * @file test_block_evolution.cc
 * @brief Tests for the per-block temporal evolution metric.
 *
 * Three things are checked, in order of what would actually break:
 *   1. the closed-form value, on cases whose E can be worked out by hand;
 *   2. GPU/host agreement, on data that genuinely lives in device memory;
 *   3. the tracker's state machine -- first timestep, sampling interval,
 *      resize, and the delta_t it reports.
 *
 * MainPretest()/MainPosttest() are defined once per binary in test_models.cc.
 */
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <limits>
#include <string>
#include <vector>

#include "basic_test.h"
#include "clio_ctp/compress/preprocess/block_evolution_gpu.h"

namespace {

constexpr double kEps = ctp::kBlockEvolutionEpsilon;

/** Relative closeness, matching test_data_stats_gpu.cc's plain-std::abs
    idiom rather than Catch2's Approx (which moved namespace in v3). */
bool Close(double a, double b, double rel) {
  const double scale = std::max({1.0, std::abs(a), std::abs(b)});
  return std::abs(a - b) <= rel * scale;
}

/** A block whose elements are all `v`. */
std::vector<float> Constant(size_t n, float v) {
  return std::vector<float>(n, v);
}

}  // namespace


/** Upload two host blocks and run the DEVICE metric on them.
 *
 * ComputeBlockEvolutionHost() was removed -- the metric is CUDA-only now -- but
 * the closed-form and non-finite cases below check MATH, not which side ran it,
 * so they are kept and pointed at the device kernel through this.
 */
static bool EvoOnDevice(const float *b1, const float *b2, size_t n,
                        double epsilon, ctp::BlockEvolution *out) {
  float *d1 = ctp::GpuApi::Malloc<float>(n * sizeof(float));
  float *d2 = ctp::GpuApi::Malloc<float>(n * sizeof(float));
  if (d1 == nullptr || d2 == nullptr) return false;
  ctp::GpuApi::Memcpy(d1, b1, n * sizeof(float));
  ctp::GpuApi::Memcpy(d2, b2, n * sizeof(float));
  const bool ok = ctp::ComputeBlockEvolutionDevice(
      d1, d2, n, ctp::DataType::FLOAT32, epsilon, nullptr, out);
  ctp::GpuApi::Free(d1);
  ctp::GpuApi::Free(d2);
  return ok;
}

TEST_CASE("BlockEvolutionHostClosedForm") {
  const size_t n = 4096;

  // B1 = 1 everywhere, B2 = 2 everywhere.
  //   ||B2-B1|| = sqrt(n),  ||B1|| = sqrt(n),  ||B2|| = 2*sqrt(n)
  //   E = sqrt(n) / (sqrt(n) + 2*sqrt(n)) = 1/3
  {
    auto b1 = Constant(n, 1.0f);
    auto b2 = Constant(n, 2.0f);
    ctp::BlockEvolution e;
    REQUIRE(EvoOnDevice(b1.data(), b2.data(), n, kEps, &e));
    REQUIRE(Close(e.absolute_change, std::sqrt((double)n), 1e-12));
    REQUIRE(Close(e.normalized_change, 1.0 / 3.0, 1e-12));
    REQUIRE(e.elements_compared == n);
    REQUIRE(e.nonfinite_skipped == 0);
  }

  // Identical blocks: no evolution at all.
  {
    auto b = Constant(n, 7.5f);
    ctp::BlockEvolution e;
    REQUIRE(EvoOnDevice(b.data(), b.data(), n, kEps, &e));
    REQUIRE(e.absolute_change == 0.0);
    REQUIRE(e.normalized_change == 0.0);
  }

  // All-zero blocks: 0/(0+0+eps) = 0, and crucially NOT NaN. This is the case
  // epsilon exists for.
  {
    auto z = Constant(n, 0.0f);
    ctp::BlockEvolution e;
    REQUIRE(EvoOnDevice(z.data(), z.data(), n, kEps, &e));
    REQUIRE(std::isfinite(e.normalized_change));
    REQUIRE(e.normalized_change == 0.0);
    REQUIRE(e.b1_norm == 0.0);
  }

  // E is scale-invariant: the same relative change on values 1e18 apart in
  // magnitude gives the same normalized score. This is the property that lets
  // scores from different fields be compared, so it is worth pinning.
  {
    ctp::BlockEvolution big, small;
    auto a1 = Constant(n, 1e9f), a2 = Constant(n, 2e9f);
    auto c1 = Constant(n, 1e-9f), c2 = Constant(n, 2e-9f);
    REQUIRE(EvoOnDevice(a1.data(), a2.data(), n, kEps, &big));
    REQUIRE(EvoOnDevice(c1.data(), c2.data(), n, kEps,
                                           &small));
    REQUIRE(Close(big.normalized_change, small.normalized_change, 1e-9));
    REQUIRE(Close(small.normalized_change, 1.0 / 3.0, 1e-6));
    // ...while D does NOT collapse: it keeps the absolute scale E discards.
    REQUIRE(big.absolute_change > small.absolute_change * 1e17);
  }

  // Very small values, the edge case epsilon can silently destroy. A field at
  // 1e-15 that doubled must still report E = 1/3, not ~0.
  {
    auto s1 = Constant(n, 1e-15f), s2 = Constant(n, 2e-15f);
    ctp::BlockEvolution e;
    REQUIRE(EvoOnDevice(s1.data(), s2.data(), n, kEps, &e));
    REQUIRE(Close(e.normalized_change, 1.0 / 3.0, 1e-6));
  }

  // Monotonicity: a block that moved further scores higher.
  {
    auto b1 = Constant(n, 1.0f);
    ctp::BlockEvolution quiet, loud;
    auto near = Constant(n, 1.01f);
    auto far = Constant(n, 5.0f);
    REQUIRE(EvoOnDevice(b1.data(), near.data(), n, kEps,
                                           &quiet));
    REQUIRE(EvoOnDevice(b1.data(), far.data(), n, kEps,
                                           &loud));
    REQUIRE(quiet.normalized_change < loud.normalized_change);
    REQUIRE(quiet.normalized_change < 0.01);
  }
}

TEST_CASE("BlockEvolutionNonFinite") {
  const size_t n = 1024;
  auto b1 = Constant(n, 1.0f);
  auto b2 = Constant(n, 2.0f);
  b2[10] = std::numeric_limits<float>::quiet_NaN();
  b2[20] = std::numeric_limits<float>::infinity();
  b1[30] = -std::numeric_limits<float>::infinity();

  ctp::BlockEvolution e;
  REQUIRE(EvoOnDevice(b1.data(), b2.data(), n, kEps, &e));
  // The three poisoned pairs are excluded and counted; everything else still
  // reduces to the clean 1/3, rather than the whole block going NaN.
  REQUIRE(e.nonfinite_skipped == 3);
  REQUIRE(e.elements_compared == n - 3);
  REQUIRE(std::isfinite(e.normalized_change));
  REQUIRE(Close(e.normalized_change, 1.0 / 3.0, 1e-9));

  // Every pair poisoned: nothing left to reduce over, and it says so.
  std::vector<float> all_nan(n, std::numeric_limits<float>::quiet_NaN());
  ctp::BlockEvolution dead;
  REQUIRE_FALSE(EvoOnDevice(
      all_nan.data(), all_nan.data(), n, kEps, &dead));
  REQUIRE(dead.status == ctp::BlockEvolutionStatus::kAllNonFinite);
}

TEST_CASE("BlockEvolutionTrackerStateMachine") {
  // Host chunks, so this case runs on every build.
  const size_t n = 512;
  ctp::BlockEvolutionTracker tracker(/*sample_interval=*/10);

  auto t0 = Constant(n, 1.0f);
  auto t10 = Constant(n, 2.0f);
  const size_t bytes = n * sizeof(float);

  ctp::BlockEvolution e;

  // First sample: retained, no value yet.
  REQUIRE_FALSE(tracker.Observe("position/chunk_0", 0, t0.data(), bytes,
                                ctp::DataType::FLOAT32, nullptr, &e));
  REQUIRE(e.status == ctp::BlockEvolutionStatus::kFirstTimestep);
  REQUIRE(tracker.tracked_blocks() == 1);
  REQUIRE(tracker.retained_bytes() == bytes);

  // Off-grid timesteps are skipped and, importantly, do NOT refresh the
  // retained block -- otherwise the next comparison would span 1 step
  // instead of the configured 10.
  REQUIRE_FALSE(tracker.Observe("position/chunk_0", 3, t10.data(), bytes,
                                ctp::DataType::FLOAT32, nullptr, &e));
  REQUIRE(e.status == ctp::BlockEvolutionStatus::kNotSampled);

  // On-grid: compares against t=0, not against t=3.
  REQUIRE(tracker.Observe("position/chunk_0", 10, t10.data(), bytes,
                          ctp::DataType::FLOAT32, nullptr, &e));
  REQUIRE(e.status == ctp::BlockEvolutionStatus::kOk);
  REQUIRE(e.delta_t == 10);
  REQUIRE(Close(e.normalized_change, 1.0 / 3.0, 1e-9));
  REQUIRE(Close(e.evolution_rate, e.normalized_change / 10.0, 1e-12));

  // The retained block advanced to t=10, so t=20 against an unchanged field
  // is zero evolution -- proving the comparison is against t=10 and not
  // still against t=0.
  REQUIRE(tracker.Observe("position/chunk_0", 20, t10.data(), bytes,
                          ctp::DataType::FLOAT32, nullptr, &e));
  REQUIRE(e.delta_t == 10);
  REQUIRE(e.normalized_change == 0.0);

  // A resized block is not differenced against the old one.
  auto grown = Constant(n * 2, 2.0f);
  REQUIRE_FALSE(tracker.Observe("position/chunk_0", 30, grown.data(),
                                bytes * 2, ctp::DataType::FLOAT32, nullptr,
                                &e));
  REQUIRE(e.status == ctp::BlockEvolutionStatus::kSizeMismatch);
  // ...but it is retained at the new size, so the series resumes.
  REQUIRE(tracker.retained_bytes() == bytes * 2);
  REQUIRE(tracker.Observe("position/chunk_0", 40, grown.data(), bytes * 2,
                          ctp::DataType::FLOAT32, nullptr, &e));
  REQUIRE(e.status == ctp::BlockEvolutionStatus::kOk);

  // Distinct blocks are tracked independently.
  REQUIRE_FALSE(tracker.Observe("position/chunk_1", 40, t0.data(), bytes,
                                ctp::DataType::FLOAT32, nullptr, &e));
  REQUIRE(e.status == ctp::BlockEvolutionStatus::kFirstTimestep);
  REQUIRE(tracker.tracked_blocks() == 2);

  tracker.Clear();
  REQUIRE(tracker.tracked_blocks() == 0);
  REQUIRE(tracker.retained_bytes() == 0);
}

TEST_CASE("BlockEvolutionTrackerCapacity") {
  const size_t n = 512;
  const size_t bytes = n * sizeof(float);
  // Room for exactly one block.
  ctp::BlockEvolutionTracker tracker(1, kEps, bytes);
  auto b = Constant(n, 1.0f);
  ctp::BlockEvolution e;

  REQUIRE_FALSE(tracker.Observe("a", 0, b.data(), bytes,
                                ctp::DataType::FLOAT32, nullptr, &e));
  REQUIRE(e.status == ctp::BlockEvolutionStatus::kFirstTimestep);
  // The second block does not fit; it is refused rather than evicting the
  // first, which would silently drop a sample out of that block's series.
  REQUIRE_FALSE(tracker.Observe("b", 0, b.data(), bytes,
                                ctp::DataType::FLOAT32, nullptr, &e));
  REQUIRE(e.status == ctp::BlockEvolutionStatus::kFailed);
  REQUIRE(tracker.tracked_blocks() == 1);
}

#if CTP_ENABLE_CUDA

/**
 * The device kernel against the host reference, on data that really is in
 * device memory. Tolerance rather than equality: the kernel's tree reduction
 * and the host's serial loop sum the same terms in different orders.
 */
/* Was "DeviceMatchesHost". The host implementation is gone, so the comparison
   arm it was named for no longer exists; what remains is worth keeping on its
   own terms -- the device kernel's non-finite handling and repeatability, with
   the absolute expectations (2 skipped) that never depended on the host. */
TEST_CASE("BlockEvolutionDeviceNonFiniteAndRepeatable") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    return;
  }

  // Not constant blocks: a varying field exercises the reduction properly,
  // and 61440 B / 4 is the LAMMPS chunk this was built for.
  const size_t n = 15360;
  std::vector<float> b1(n), b2(n);
  for (size_t i = 0; i < n; ++i) {
    b1[i] = static_cast<float>(std::sin(i * 0.01) * 100.0);
    b2[i] = static_cast<float>(std::sin(i * 0.01 + 0.003) * 100.0 + 0.5);
  }
  // A NaN and an Inf, so the skip path is covered on device too.
  b2[77] = std::numeric_limits<float>::quiet_NaN();
  b1[999] = std::numeric_limits<float>::infinity();

  ctp::BlockEvolution host;
  REQUIRE(EvoOnDevice(b1.data(), b2.data(), n, kEps, &host));

  // GpuApi rather than raw cudaMalloc: this TU compiles as C++, not CUDA,
  // so the void** out-parameter would need a cast here. Same idiom as
  // test_data_stats_gpu.cc.
  float *d1 = ctp::GpuApi::Malloc<float>(n * sizeof(float));
  float *d2 = ctp::GpuApi::Malloc<float>(n * sizeof(float));
  REQUIRE(d1 != nullptr);
  REQUIRE(d2 != nullptr);
  ctp::GpuApi::Memcpy(d1, b1.data(), n * sizeof(float));
  ctp::GpuApi::Memcpy(d2, b2.data(), n * sizeof(float));

  ctp::BlockEvolution dev;
  REQUIRE(ctp::ComputeBlockEvolutionDevice(d1, d2, n, ctp::DataType::FLOAT32,
                                           kEps, nullptr, &dev));

  REQUIRE(dev.nonfinite_skipped == host.nonfinite_skipped);
  REQUIRE(dev.nonfinite_skipped == 2);
  REQUIRE(dev.elements_compared == host.elements_compared);
  REQUIRE(Close(dev.absolute_change, host.absolute_change, 1e-10));
  REQUIRE(Close(dev.b1_norm, host.b1_norm, 1e-10));
  REQUIRE(Close(dev.b2_norm, host.b2_norm, 1e-10));
  REQUIRE(Close(dev.normalized_change, host.normalized_change, 1e-10));

  ctp::GpuApi::Free(d1);
  ctp::GpuApi::Free(d2);
}

/** The tracker on device chunks: retention, comparison and the D2D advance
    all stay on the GPU, and the answer matches the host tracker's. */
TEST_CASE("BlockEvolutionTrackerDevice") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    return;
  }
  const size_t n = 4096;
  const size_t bytes = n * sizeof(float);
  std::vector<float> h1(n, 1.0f), h2(n, 2.0f);

  float *d = ctp::GpuApi::Malloc<float>(bytes);
  REQUIRE(d != nullptr);
  ctp::GpuApi::Memcpy(d, h1.data(), bytes);

  ctp::BlockEvolutionTracker tracker(/*sample_interval=*/5);
  ctp::BlockEvolution e;
  REQUIRE_FALSE(tracker.Observe("f/chunk_0", 0, d, bytes,
                                ctp::DataType::FLOAT32, nullptr, &e));
  REQUIRE(e.status == ctp::BlockEvolutionStatus::kFirstTimestep);

  // Overwrite the SAME device buffer in place -- exactly what an in-situ
  // producer does between steps. The retained copy must have been taken, or
  // this comparison would come back as zero evolution.
  ctp::GpuApi::Memcpy(d, h2.data(), bytes);
  REQUIRE(tracker.Observe("f/chunk_0", 5, d, bytes, ctp::DataType::FLOAT32,
                          nullptr, &e));
  REQUIRE(e.status == ctp::BlockEvolutionStatus::kOk);
  REQUIRE(e.delta_t == 5);
  REQUIRE(Close(e.normalized_change, 1.0 / 3.0, 1e-9));

  ctp::GpuApi::Free(d);
}

#endif  // CTP_ENABLE_CUDA
