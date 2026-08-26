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

/* The CPU Quantize/Dequantize and ByteShuffle/ByteUnshuffle round-trip cases
 * were removed with the implementations they covered. NeuroPress preprocessing
 * is CUDA-only, matching upstream; the device kernels are covered by
 * test_data_stats_gpu.cc and, against upstream itself, by
 * model/parity/neuropress_preprocess_parity.cu. */
