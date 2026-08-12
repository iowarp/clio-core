/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

/**
 * @file test_compress_factory_gpu.cc
 * @brief Regression test for CompressionFactory::StageInputIfNeeded (issue
 * #693 follow-up): the safety net Runtime::Compress falls back to if a CPU
 * library is ever selected for a device-resident buffer (the normal route
 * avoids this entirely -- see DefaultCandidates' include_cpu).
 */
#include <cstdint>
#include <cstring>
#include <vector>

#include "basic_test.h"
#include "clio_ctp/compress/compress_factory.h"

// MainPretest()/MainPosttest() are defined once for the whole binary in
// test_models.cc (same target).

#if CTP_ENABLE_CUDA || CTP_ENABLE_ROCM

TEST_CASE("StageInputIfNeeded copies for a CPU library, "
          "leaves a GPU library's input untouched") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    return;
  }

  constexpr size_t kSize = 4096;
  std::vector<char> pattern(kSize);
  for (size_t i = 0; i < kSize; ++i) {
    pattern[i] = static_cast<char>((i * 7 + 3) & 0xFF);
  }

  char *device_data = ctp::GpuApi::Malloc<char>(kSize);
  REQUIRE(device_data != nullptr);
  ctp::GpuApi::Memcpy(device_data, pattern.data(), kSize);

  int zstd_wire_id = ctp::CompressionFactory::WireIdForName("zstd");
  REQUIRE_FALSE(ctp::CompressionFactory::IsGpuLibraryWireId(zstd_wire_id));
  int nvcomp_wire_id = ctp::CompressionFactory::WireIdForName("nvcomp-lz4");
  REQUIRE(ctp::CompressionFactory::IsGpuLibraryWireId(nvcomp_wire_id));

  // CPU library: must stage a host copy with the right bytes.
  {
    std::vector<char> staging;
    char *result = ctp::CompressionFactory::StageInputIfNeeded(
        device_data, kSize, zstd_wire_id, staging);
    REQUIRE(result == staging.data());
    REQUIRE(result != device_data);
    REQUIRE(staging.size() == kSize);
    REQUIRE(std::memcmp(staging.data(), pattern.data(), kSize) == 0);
  }

  // GPU library: must return the device pointer unchanged -- no copy.
  {
    std::vector<char> staging;
    char *result = ctp::CompressionFactory::StageInputIfNeeded(
        device_data, kSize, nvcomp_wire_id, staging);
    REQUIRE(result == device_data);
    REQUIRE(staging.empty());
  }

  // Host input: never staged, for either library kind.
  {
    std::vector<char> staging;
    char *result = ctp::CompressionFactory::StageInputIfNeeded(
        pattern.data(), kSize, zstd_wire_id, staging);
    REQUIRE(result == pattern.data());
    REQUIRE(staging.empty());
  }

  ctp::GpuApi::Free(device_data);
}

#endif  // CTP_ENABLE_CUDA || CTP_ENABLE_ROCM
