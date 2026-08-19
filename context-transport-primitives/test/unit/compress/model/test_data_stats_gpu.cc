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

#endif  // CTP_ENABLE_CUDA || CTP_ENABLE_ROCM
