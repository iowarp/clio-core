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
#include <chrono>
#include <cstdint>
#include <cstring>
#include <vector>

#include "basic_test.h"
#include "clio_ctp/compress/compress_factory.h"
#include "clio_ctp/compress/nvcomp.h"

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

// ---------------------------------------------------------------------------
// nvcomp manager caching (issue #693).
//
// NeuroPress does NOT build an nvcomp manager per compression. It keeps a
// small LRU of them per context, keyed by algorithm
// (gpucompress_pool.cpp, CompContext::LRU_DEPTH = 3), on a persistent
// stream, and counts hits and misses. That matters for two reasons: manager
// construction is typically hundreds of microseconds against compress kernels
// of tens, and -- because Runtime::Compress times the codec call to produce
// the model's comp_time label -- paying it on every chunk inflates that label
// by an amount roughly independent of chunk size, flattening the very
// time-vs-size relationship the network is supposed to learn.
//
// These assert the cache's SEMANTICS, not merely that something is cached.
// ---------------------------------------------------------------------------
#if CTP_ENABLE_COMPRESS && CTP_ENABLE_NVCOMP

TEST_CASE("nvcomp manager cache reuses a manager across compressions") {
  std::vector<char> src(64 * 1024);
  for (size_t i = 0; i < src.size(); ++i) {
    src[i] = static_cast<char>((i / 64) & 0x3F);  // compressible
  }
  std::vector<char> dst(src.size() * 2);

  ctp::NvComp::ResetManagerCache();
  const auto before = ctp::NvComp::GetManagerCacheStats();
  REQUIRE(before.misses == 0);
  REQUIRE(before.hits == 0);

  // Ten compressions with ONE algorithm must build ONE manager.
  for (int i = 0; i < 10; ++i) {
    ctp::NvComp comp(ctp::NvCompAlgo::LZ4);
    size_t out = dst.size();
    REQUIRE(comp.Compress(dst.data(), out, src.data(), src.size()));
    REQUIRE(out > 0);
  }
  const auto after = ctp::NvComp::GetManagerCacheStats();
  REQUIRE(after.misses == 1);   // built exactly once
  REQUIRE(after.hits == 9);     // reused every other time
}

TEST_CASE("nvcomp manager cache holds LRU_DEPTH algorithms, then evicts") {
  std::vector<char> src(32 * 1024, 0x5A);
  std::vector<char> dst(src.size() * 2);
  auto compress_with = [&](ctp::NvCompAlgo algo) {
    ctp::NvComp comp(algo);
    size_t out = dst.size();
    REQUIRE(comp.Compress(dst.data(), out, src.data(), src.size()));
  };

  ctp::NvComp::ResetManagerCache();

  // Fill exactly LRU_DEPTH slots, then revisit them: all hits, no rebuild.
  compress_with(ctp::NvCompAlgo::LZ4);
  compress_with(ctp::NvCompAlgo::SNAPPY);
  compress_with(ctp::NvCompAlgo::ZSTD);
  REQUIRE(ctp::NvComp::GetManagerCacheStats().misses == 3);

  compress_with(ctp::NvCompAlgo::LZ4);
  compress_with(ctp::NvCompAlgo::SNAPPY);
  compress_with(ctp::NvCompAlgo::ZSTD);
  REQUIRE(ctp::NvComp::GetManagerCacheStats().misses == 3);  // still 3
  REQUIRE(ctp::NvComp::GetManagerCacheStats().hits == 3);

  // A fourth algorithm must evict the least recently used (LZ4), so coming
  // back to LZ4 is a miss while the two newer ones stay hits.
  compress_with(ctp::NvCompAlgo::ANS);
  REQUIRE(ctp::NvComp::GetManagerCacheStats().misses == 4);

  const auto before_lz4 = ctp::NvComp::GetManagerCacheStats();
  compress_with(ctp::NvCompAlgo::LZ4);
  REQUIRE(ctp::NvComp::GetManagerCacheStats().misses == before_lz4.misses + 1);

  const auto before_zstd = ctp::NvComp::GetManagerCacheStats();
  compress_with(ctp::NvCompAlgo::ZSTD);
  REQUIRE(ctp::NvComp::GetManagerCacheStats().hits == before_zstd.hits + 1);
}

TEST_CASE("cached compression is materially faster than cold") {
  std::vector<char> src(256 * 1024);
  for (size_t i = 0; i < src.size(); ++i) {
    src[i] = static_cast<char>((i / 128) & 0x1F);
  }
  std::vector<char> dst(src.size() * 2);

  auto time_one = [&]() {
    ctp::NvComp comp(ctp::NvCompAlgo::LZ4);
    size_t out = dst.size();
    auto t0 = std::chrono::high_resolution_clock::now();
    REQUIRE(comp.Compress(dst.data(), out, src.data(), src.size()));
    return std::chrono::duration<double, std::milli>(
               std::chrono::high_resolution_clock::now() - t0).count();
  };

  ctp::NvComp::ResetManagerCache();
  const double cold = time_one();
  double warm = 0.0;
  for (int i = 0; i < 5; ++i) warm += time_one();
  warm /= 5.0;

  INFO("cold=" << cold << "ms warm=" << warm << "ms");
  // The point of the cache: the per-chunk cost the model is trained on no
  // longer carries manager construction. Deliberately loose -- this asserts
  // a real effect, not a specific speedup.
  REQUIRE(warm < cold);
}

#endif  // CTP_ENABLE_COMPRESS && CTP_ENABLE_NVCOMP
