/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

/**
 * @file test_preprocess.cc
 * @brief Unit tests for the compression preprocessors (issue #693):
 *        feature extraction, error-bounded quantization, and byte-plane shuffle.
 */
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "basic_test.h"
#include "clio_ctp/compress/preprocess/byte_shuffle.h"
#include "clio_ctp/compress/preprocess/feature_extractor.h"
#include "clio_ctp/compress/preprocess/quantization.h"

using namespace ctp::compress::preprocess;  // NOLINT(build/namespaces)

// Test-harness hooks required by test/unit/main.cc (no global setup needed).
void MainPretest() {}
void MainPosttest() {}

TEST_CASE("FeatureExtractor populates statistics for float data") {
  std::vector<float> data(2048);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<float>(std::sin(static_cast<double>(i) * 0.01));
  }

  auto f = FeatureExtractor::ExtractFeatures(
      data.data(), data.size() * sizeof(float), DataKind::kFloat);

  REQUIRE(f.chunk_size_bytes == static_cast<double>(data.size() * sizeof(float)));
  REQUIRE(f.data_type_float == 1.0);
  REQUIRE(f.shannon_entropy >= 0.0);
  REQUIRE(f.shannon_entropy <= 8.0);

  FeatureExtractor::SetLibraryConfig(f, 1, 2);  // BZIP2_BALANCED
  REQUIRE(f.library_config_id == 12.0);
  REQUIRE(f.config_balanced == 1.0);
  REQUIRE(f.config_fast == 0.0);
  REQUIRE(f.config_best == 0.0);
}

TEST_CASE("Quantize/Dequantize round-trips within the error bound") {
  std::vector<double> vals(512);
  for (size_t i = 0; i < vals.size(); ++i) {
    vals[i] = static_cast<double>(i) * 0.1 - 25.6;
  }
  const double error_bound = 0.05;

  auto qr = Quantize<double>(vals.data(), vals.size(), error_bound);
  auto rec = Dequantize<double>(qr);

  REQUIRE(rec.size() == vals.size());
  for (size_t i = 0; i < vals.size(); ++i) {
    REQUIRE(std::abs(rec[i] - vals[i]) <= error_bound + 1e-9);
  }
}

TEST_CASE("ByteShuffle/ByteUnshuffle is a lossless round-trip") {
  for (size_t elem_size : {2u, 4u, 8u}) {
    std::vector<uint8_t> bytes(elem_size * 300);
    for (size_t i = 0; i < bytes.size(); ++i) {
      bytes[i] = static_cast<uint8_t>((i * 37 + 11) & 0xFF);
    }
    auto shuffled = ByteShuffleVector(bytes.data(), bytes.size(), elem_size);
    REQUIRE(shuffled.size() == bytes.size());
    auto restored = ByteUnshuffleVector(shuffled.data(), shuffled.size(),
                                        elem_size);
    REQUIRE(restored == bytes);
  }
}

/**
 * The case above uses elem_size * 300 -- always an exact multiple of the
 * element, and always 2,400 bytes, which is one kShuffleChunkBytes block. So
 * it exercises neither of the two paths whose comments in byte_shuffle.h
 * record past bugs: the trailing partial element ("silent corruption on round
 * trip, since the length was still right") and the per-256-KiB-block plane
 * walk.
 *
 * neuropress_preprocess_parity.cu does cover both, across the same widths and
 * against upstream's own kernels -- but it is SKIPPED without CUDA and a
 * NeuroPress checkout, which is most builds. This pins the host round-trip
 * unconditionally so a plain build cannot regress it silently.
 */
TEST_CASE("ByteShuffle round-trips across block edges and partial elements") {
  constexpr size_t kBlock = 256 * 1024;  // kShuffleChunkBytes
  const std::vector<size_t> sizes = {
      1, 2, 3, 5, 7, 9, 15, 17, 31, 33, 127, 129,   // sub-element and ragged
      1000, 1001, 1023, 1024, 1025,
      kBlock - 1, kBlock, kBlock + 1, kBlock + 7,   // the block edge
      2 * kBlock, 2 * kBlock + 3, 4 * kBlock,       // multi-block
      318208,                                       // a real LAMMPS tail chunk
      1048576 - 1, 1048576, 1048576 + 1};           // the VOL's chunk size

  size_t with_partial_element = 0;
  for (size_t elem_size : {size_t(2), size_t(4), size_t(8)}) {
    for (size_t n : sizes) {
      std::vector<uint8_t> bytes(n);
      for (size_t i = 0; i < n; ++i) {
        bytes[i] = static_cast<uint8_t>((i * 131u + elem_size * 7u + 3u) & 0xFF);
      }
      auto shuffled = ByteShuffleVector(bytes.data(), n, elem_size);
      REQUIRE(shuffled.size() == n);
      auto restored = ByteUnshuffleVector(shuffled.data(), n, elem_size);
      REQUIRE(restored == bytes);

      for (size_t base = 0; base < n; base += kBlock) {
        const size_t chunk = std::min(kBlock, n - base);
        if (chunk % elem_size != 0) { ++with_partial_element; break; }
      }
    }
  }
  // Guards the guard: if a future change to the size list stopped producing
  // ragged blocks, this case would still pass while testing nothing new.
  REQUIRE(with_partial_element > 0);
}
