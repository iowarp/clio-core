/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

/**
 * @file test_preprocess.cc
 * @brief Unit tests for the compression preprocessors (issue #693):
 *        feature extraction, error-bounded quantization, and byte-plane shuffle.
 */
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
