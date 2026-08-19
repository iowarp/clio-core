/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

/**
 * @file test_data_stats_gpu.cc
 * @brief Regression test for ctp::ComputeDeviceStats (issue #693 follow-up):
 * verifies the device-native entropy/MAD/second-derivative computation
 * matches DataStatisticsFactory's host reference exactly, for data that
 * genuinely lives in GPU device memory (not just a host buffer cast through
 * the same code path).
 */
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "basic_test.h"
#include "clio_ctp/compress/preprocess/data_stats.h"

#if CTP_ENABLE_CUDA || CTP_ENABLE_ROCM
#include "clio_ctp/compress/preprocess/data_stats_gpu.h"
#include "clio_ctp/util/gpu_api.h"
#endif

// MainPretest()/MainPosttest() are defined once for the whole binary in
// test_models.cc (same target).

#if CTP_ENABLE_CUDA || CTP_ENABLE_ROCM

namespace {
// Shared scaffolding for every parity case below: compute the host
// reference, copy `data` to a real device allocation, run `gpu_call` on it,
// and compare. `gpu_call` is what varies per test -- ComputeDeviceStats
// directly, or ComputeCompressionFeatures's auto-dispatch (the same call
// EstCompressionStats makes).
template <typename T, typename Fn>
void RunParityCase(const std::vector<T> &data, ctp::DataType type,
                    Fn gpu_call) {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    return;
  }

  double host_entropy = ctp::DataStatisticsFactory::CalculateShannonEntropy(
      data.data(), data.size(), type);
  double host_mad = ctp::DataStatisticsFactory::CalculateMAD(
      data.data(), data.size(), type);
  double host_d2 = ctp::DataStatisticsFactory::CalculateSecondDerivative(
      data.data(), data.size(), type);

  T *device_data = ctp::GpuApi::Malloc<T>(data.size() * sizeof(T));
  REQUIRE(device_data != nullptr);
  ctp::GpuApi::Memcpy(device_data, data.data(), data.size() * sizeof(T));

  double gpu_entropy = -1.0, gpu_mad = -1.0, gpu_d2 = -1.0;
  gpu_call(device_data, data.size(), type, &gpu_entropy, &gpu_mad, &gpu_d2);

  REQUIRE(std::abs(gpu_entropy - host_entropy) < 1e-6);
  REQUIRE(std::abs(gpu_mad - host_mad) < 1e-6);
  REQUIRE(std::abs(gpu_d2 - host_d2) < 1e-6);

  ctp::GpuApi::Free(device_data);
}

void CallComputeDeviceStats(const void *d, size_t n, ctp::DataType t,
                             double *e, double *m, double *d2) {
  REQUIRE(ctp::ComputeDeviceStats(d, n, t, e, m, d2));
}
}  // namespace

TEST_CASE("ComputeDeviceStats matches the host reference for float data") {
  std::vector<float> data(4096);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<float>(std::sin(static_cast<double>(i) * 0.013) * 100.0);
  }
  RunParityCase(data, ctp::DataType::FLOAT32, CallComputeDeviceStats);
}

TEST_CASE("ComputeDeviceStats matches the host reference for uint8 data") {
  std::vector<uint8_t> data(8192);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<uint8_t>((i * 37 + 11) % 256);
  }
  RunParityCase(data, ctp::DataType::UINT8, CallComputeDeviceStats);
}

TEST_CASE("ComputeDeviceStats matches the host reference for constant data") {
  // Zero entropy, zero MAD, zero second derivative -- an edge case the
  // atomics-based reduction must not perturb.
  std::vector<int32_t> data(1024, 42);
  RunParityCase(data, ctp::DataType::INT32, CallComputeDeviceStats);
}

TEST_CASE("ComputeCompressionFeatures auto-dispatches device vs host, "
          "matching the exact call EstCompressionStats makes") {
  std::vector<float> data(4096);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<float>(std::cos(static_cast<double>(i) * 0.021) * 50.0);
  }
  // Same dispatch, given a device pointer instead of a host one, must reach
  // the on-device kernels and produce the same numbers -- not silently fall
  // through to the host path (which would read device memory and crash) and
  // not silently return zeros.
  RunParityCase(data, ctp::DataType::FLOAT32,
                 [](const void *d, size_t n, ctp::DataType t, double *e,
                    double *m, double *d2) {
                   ctp::ComputeCompressionFeatures(d, n, t, e, m, d2);
                 });
}


/* Selection features on float64 input (issue #693 follow-up).
 *
 * The defect was not that DataStatisticsFactory cannot do float64 -- it can,
 * and always could. It was that the NeuroPress path never asked it to: the
 * element type was pinned to FLOAT32 to match the model's normalisation, which
 * turned a float64 buffer into a REINTERPRET rather than a CONVERT. Each
 * double then reads as two float32 words, and the low word -- pure mantissa --
 * lands on IEEE-754's reserved exponent==255 about one time in 256. Those
 * words ARE NaN, NaN propagates through the mean, and MAD and the second
 * derivative came back NaN for every chunk of every float64 dataset.
 *
 * So this asserts on ComputeNeuroPressFeatures, the call the runtime actually
 * makes, and pins BOTH sides of it: declared float64 must produce finite
 * statistics, and the same bytes declared float32 must still reproduce the old
 * NaN -- otherwise the test would pass on data that never had the problem.
 *
 * The values are built so the low half of each double is exactly such a word.
 * Computed on the doubles themselves the statistics are entirely ordinary. */
TEST_CASE("Float64 selection features are finite, not NaN") {
  /* Real, varied coordinates -- then the LOW 32 bits of each are forced to a
     float32 NaN pattern. The high word is untouched, so every value stays a
     perfectly ordinary double near 100, and they differ enough to survive a
     downcast (differing only in the low mantissa would downcast to one float
     and give a legitimately zero MAD, which tests nothing). */
  std::vector<double> data;
  data.reserve(4096);
  for (int i = 0; i < 4096; ++i) {
    double v = 100.0 + i * 0.01;
    uint64_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    bits = (bits & 0xFFFFFFFF00000000ull) | 0x7FFE0000ull | (uint64_t)(i & 0xFFFF);
    double d;
    std::memcpy(&d, &bits, sizeof(d));
    data.push_back(d);
  }
  const size_t bytes = data.size() * sizeof(double);
  for (double d : data) REQUIRE(std::isfinite(d));

  /* Declared float64: converted, so the statistics describe real values. */
  double entropy = 0.0, mad = 0.0, second = 0.0;
  REQUIRE(ctp::ComputeNeuroPressFeatures(data.data(), bytes, /*data_type=*/2,
                                         &entropy, &mad, &second));
  REQUIRE(std::isfinite(entropy));
  REQUIRE(std::isfinite(mad));
  REQUIRE(std::isfinite(second));
  REQUIRE(mad > 0.0);

  /* The same bytes declared float32: reinterpreted, and the NaN is back. This
     is the behaviour every float64 write used to get, kept here so the test
     cannot pass by accident on data that was never affected. */
  double e2 = 0.0, m2 = 0.0, s2 = 0.0;
  REQUIRE(ctp::ComputeNeuroPressFeatures(data.data(), bytes, /*data_type=*/1,
                                         &e2, &m2, &s2));
  REQUIRE(std::isnan(m2));
  REQUIRE(std::isnan(s2));
}

#endif  // CTP_ENABLE_CUDA || CTP_ENABLE_ROCM
