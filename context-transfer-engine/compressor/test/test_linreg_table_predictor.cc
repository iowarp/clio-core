/**
 * @file test_linreg_table_predictor.cc
 * @brief Unit tests for LinearRegressionTable predictor
 */

#include "wrp_cte/compressor/models/linreg_table_predictor.h"
#include <catch2/catch_all.hpp>
#include <iostream>
#include <cmath>
#include <fstream>
#include <sstream>

using namespace wrp_cte::compressor;

/**
 * @brief Create a temporary model file for testing
 */
std::string CreateTestModelJson() {
  std::string json = R"({
  "model_type": "linreg_table",
  "version": "1.1",
  "num_models": 3,
  "models": [
    {
      "library": "ZSTD",
      "config": "fast",
      "data_type": "char",
      "distribution": "binned_normal_w0_p0",
      "slope_compress_time": 1.5e-06,
      "intercept_compress_time": 0.01,
      "slope_decompress_time": 6.0e-07,
      "intercept_decompress_time": 0.002,
      "slope_compress_ratio": 7.0e-04,
      "intercept_compress_ratio": 5.0,
      "sample_count": 1000,
      "r2_compress_time": 0.85,
      "r2_decompress_time": 0.82,
      "r2_compress_ratio": 0.05
    },
    {
      "library": "LZ4",
      "config": "fast",
      "data_type": "char",
      "distribution": "binned_uniform_w0_p0",
      "slope_compress_time": 8.0e-07,
      "intercept_compress_time": 0.005,
      "slope_decompress_time": 2.0e-07,
      "intercept_decompress_time": 0.001,
      "slope_compress_ratio": 2.0e-04,
      "intercept_compress_ratio": 4.0,
      "sample_count": 1000,
      "r2_compress_time": 0.80,
      "r2_decompress_time": 0.75,
      "r2_compress_ratio": 0.04
    },
    {
      "library": "ZSTD",
      "config": "balanced",
      "data_type": "float",
      "distribution": "binned_gamma_w0_p0",
      "slope_compress_time": 1.0e-06,
      "intercept_compress_time": 0.015,
      "slope_decompress_time": 5.0e-07,
      "intercept_decompress_time": 0.003,
      "slope_compress_ratio": 4.5e-04,
      "intercept_compress_ratio": 4.5,
      "sample_count": 800,
      "r2_compress_time": 0.78,
      "r2_decompress_time": 0.80,
      "r2_compress_ratio": 0.03
    }
  ]
})";
  return json;
}

TEST_CASE("LinRegTablePredictor - Basic Functionality") {
  LinRegTablePredictor predictor;

  SECTION("Initial state is not ready") {
    REQUIRE_FALSE(predictor.IsReady());
    REQUIRE(predictor.GetNumModels() == 0);
  }

  SECTION("Save and Load model") {
    // Create test directory
    std::string test_dir = "/tmp/linreg_test_model";
    system(("mkdir -p " + test_dir).c_str());

    // Write test model
    std::string json = CreateTestModelJson();
    std::ofstream file(test_dir + "/linreg_table.json");
    file << json;
    file.close();

    // Load model
    REQUIRE(predictor.Load(test_dir));
    REQUIRE(predictor.IsReady());
    REQUIRE(predictor.GetNumModels() == 3);

    // Check libraries loaded
    auto libs = predictor.GetLibraries();
    REQUIRE(libs.size() == 2);  // ZSTD and LZ4

    auto configs = predictor.GetConfigurations();
    REQUIRE(configs.size() == 2);  // fast and balanced

    auto dtypes = predictor.GetDataTypes();
    REQUIRE(dtypes.size() == 2);  // char and float

    auto dists = predictor.GetDistributions();
    REQUIRE(dists.size() == 3);  // binned_normal_w0_p0, binned_uniform_w0_p0, binned_gamma_w0_p0

    // Clean up
    system(("rm -rf " + test_dir).c_str());
  }
}

TEST_CASE("LinRegTablePredictor - Prediction") {
  LinRegTablePredictor predictor;

  // Create test model
  std::string test_dir = "/tmp/linreg_test_model2";
  system(("mkdir -p " + test_dir).c_str());
  std::ofstream file(test_dir + "/linreg_table.json");
  file << CreateTestModelJson();
  file.close();

  REQUIRE(predictor.Load(test_dir));

  SECTION("PredictByKey - Known key with distribution") {
    // Test ZSTD/fast/char/binned_normal_w0_p0: y = slope * x + intercept
    // compress_time: 1.5e-6 * 10000 + 0.01 = 0.025
    auto result = predictor.PredictByKey("ZSTD", "fast", "char", "binned_normal_w0_p0", 10000);

    double expected_compress = 1.5e-6 * 10000 + 0.01;
    double expected_decompress = 6.0e-7 * 10000 + 0.002;
    double expected_ratio = 7.0e-4 * 10000 + 5.0;
    (void)expected_decompress;  // Suppress unused warning

    REQUIRE(std::abs(result.compression_time_ms - expected_compress) < 1e-6);
    REQUIRE(result.compression_ratio > 1.0);
    INFO("Compress time: " << result.compression_time_ms << " (expected: " << expected_compress << ")");
    INFO("Compress ratio: " << result.compression_ratio << " (expected: " << expected_ratio << ")");
  }

  SECTION("PredictByKey - Unknown key returns fallback") {
    auto result = predictor.PredictByKey("UNKNOWN", "fast", "char", "binned_normal_w0_p0", 10000);

    // Should return default fallback values
    REQUIRE(result.compression_time_ms > 0);
    REQUIRE(result.compression_ratio >= 1.0);
  }

  SECTION("GetCoeffs - returns coefficients for known key") {
    const auto* coeffs = predictor.GetCoeffs("LZ4", "fast", "char", "binned_uniform_w0_p0");
    REQUIRE(coeffs != nullptr);
    REQUIRE(std::abs(coeffs->slope_compress_time - 8.0e-7) < 1e-10);
    REQUIRE(coeffs->sample_count == 1000);
    REQUIRE(coeffs->r2_compress_time > 0.5);
  }

  SECTION("GetCoeffs - returns nullptr for unknown key") {
    const auto* coeffs = predictor.GetCoeffs("UNKNOWN", "fast", "char", "binned_normal_w0_p0");
    REQUIRE(coeffs == nullptr);
  }

  SECTION("Large data size predictions") {
    // Test with 1MB data
    auto result = predictor.PredictByKey("ZSTD", "fast", "char", "binned_normal_w0_p0", 1048576);

    // compress_time: 1.5e-6 * 1048576 + 0.01 ≈ 1.58 ms
    REQUIRE(result.compression_time_ms > 1.0);
    REQUIRE(result.compression_time_ms < 3.0);
    REQUIRE(result.compression_ratio > 1.0);
  }

  // Clean up
  system(("rm -rf " + test_dir).c_str());
}

TEST_CASE("LinRegTablePredictor - Training from raw data") {
  LinRegTableConfig config;
  config.min_samples = 5;  // Lower threshold for testing
  LinRegTablePredictor predictor(config);

  SECTION("TrainFromRaw with simple data") {
    // Create simple training data with enough samples per group
    std::vector<std::string> libraries;
    std::vector<std::string> configs;
    std::vector<std::string> dtypes;
    std::vector<std::string> dists;
    std::vector<double> sizes;
    std::vector<double> compress_times, decompress_times, ratios;

    // Generate 15 samples for ZSTD and 15 for LZ4
    for (int i = 0; i < 15; ++i) {
      double size = 1000 + i * 500;  // 1000 to 8000

      // ZSTD samples
      libraries.push_back("ZSTD");
      configs.push_back("fast");
      dtypes.push_back("char");
      dists.push_back("binned_normal_w0_p0");
      sizes.push_back(size);
      compress_times.push_back(1e-6 * size + 0.01);
      decompress_times.push_back(5e-7 * size + 0.005);
      ratios.push_back(5.0 + size * 1e-4);

      // LZ4 samples
      libraries.push_back("LZ4");
      configs.push_back("fast");
      dtypes.push_back("char");
      dists.push_back("binned_uniform_w0_p0");
      sizes.push_back(size);
      compress_times.push_back(8e-7 * size + 0.005);
      decompress_times.push_back(2e-7 * size + 0.002);
      ratios.push_back(4.0 + size * 5e-5);
    }

    // Train
    REQUIRE(predictor.TrainFromRaw(libraries, configs, dtypes, dists, sizes,
                                   compress_times, decompress_times, ratios));
    REQUIRE(predictor.IsReady());
    REQUIRE(predictor.GetNumModels() == 2);  // ZSTD and LZ4

    // Test predictions
    auto result = predictor.PredictByKey("ZSTD", "fast", "char", "binned_normal_w0_p0", 3000);
    REQUIRE(result.compression_time_ms > 0);

    // Save and reload to test serialization
    std::string test_dir = "/tmp/linreg_test_train";
    system(("mkdir -p " + test_dir).c_str());
    REQUIRE(predictor.Save(test_dir));

    LinRegTablePredictor predictor2;
    REQUIRE(predictor2.Load(test_dir));
    REQUIRE(predictor2.GetNumModels() == 2);

    // Clean up
    system(("rm -rf " + test_dir).c_str());
  }
}

TEST_CASE("LinRegTablePredictor - Statistics") {
  LinRegTablePredictor predictor;

  std::string test_dir = "/tmp/linreg_test_stats";
  system(("mkdir -p " + test_dir).c_str());
  std::ofstream file(test_dir + "/linreg_table.json");
  file << CreateTestModelJson();
  file.close();

  REQUIRE(predictor.Load(test_dir));

  SECTION("GetStatistics returns valid string") {
    std::string stats = predictor.GetStatistics();
    REQUIRE_FALSE(stats.empty());
    REQUIRE(stats.find("Number of models: 3") != std::string::npos);
    INFO("Statistics:\n" << stats);
  }

  // Clean up
  system(("rm -rf " + test_dir).c_str());
}

TEST_CASE("LinRegTablePredictor - Edge cases") {
  SECTION("Load from non-existent directory fails") {
    LinRegTablePredictor predictor;
    REQUIRE_FALSE(predictor.Load("/non/existent/path"));
    REQUIRE_FALSE(predictor.IsReady());
  }

  SECTION("Zero size prediction") {
    LinRegTablePredictor predictor;
    std::string test_dir = "/tmp/linreg_test_edge";
    system(("mkdir -p " + test_dir).c_str());
    std::ofstream file(test_dir + "/linreg_table.json");
    file << CreateTestModelJson();
    file.close();

    REQUIRE(predictor.Load(test_dir));

    // Zero size should still give valid predictions (intercept only)
    auto result = predictor.PredictByKey("ZSTD", "fast", "char", "binned_normal_w0_p0", 0);
    REQUIRE(result.compression_time_ms >= 0);  // Could be negative intercept
    REQUIRE(result.compression_ratio >= 1.0);   // Minimum is 1.0

    system(("rm -rf " + test_dir).c_str());
  }
}
