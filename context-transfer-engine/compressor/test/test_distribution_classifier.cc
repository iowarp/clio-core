/**
 * @file test_distribution_classifier.cc
 * @brief Tests for mathematical distribution classifier
 */

#include "wrp_cte/compressor/models/distribution_classifier.h"
#include "wrp_cte/compressor/models/data_stats.h"
#include <catch2/catch_all.hpp>
#include <iostream>
#include <random>
#include <cmath>

using namespace wrp_cte::compressor;

/**
 * Generate data from specific distributions for testing
 */
class TestDataGenerator {
 public:
  static std::vector<float> GenerateUniform(size_t n, float lo = 0.0f, float hi = 100.0f) {
    std::vector<float> data(n);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (size_t i = 0; i < n; ++i) {
      data[i] = dist(gen);
    }
    return data;
  }

  static std::vector<float> GenerateNormal(size_t n, float mean = 50.0f, float stddev = 10.0f) {
    std::vector<float> data(n);
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(mean, stddev);
    for (size_t i = 0; i < n; ++i) {
      data[i] = dist(gen);
    }
    return data;
  }

  static std::vector<float> GenerateExponential(size_t n, float lambda = 0.1f) {
    std::vector<float> data(n);
    std::mt19937 gen(42);
    std::exponential_distribution<float> dist(lambda);
    for (size_t i = 0; i < n; ++i) {
      data[i] = dist(gen);
    }
    return data;
  }

  static std::vector<float> GenerateGamma(size_t n, float shape = 2.0f, float scale = 5.0f) {
    std::vector<float> data(n);
    std::mt19937 gen(42);
    std::gamma_distribution<float> dist(shape, scale);
    for (size_t i = 0; i < n; ++i) {
      data[i] = dist(gen);
    }
    return data;
  }
};

TEST_CASE("DistributionClassifier - Uniform Detection") {
  auto data = TestDataGenerator::GenerateUniform(10000);

  auto result = DistributionClassifier<float>::Classify(data.data(), data.size());

  INFO("Detected: " << result.ToString());
  INFO("Confidence: " << result.confidence);
  INFO("Skewness: " << result.skewness << " (expected ~0)");
  INFO("Kurtosis: " << result.kurtosis << " (expected ~-1.2)");
  INFO("Scores - U:" << result.uniform_score << " N:" << result.normal_score
       << " G:" << result.gamma_score << " E:" << result.exponential_score);

  REQUIRE(result.type == DistributionType::UNIFORM);
  REQUIRE(std::abs(result.skewness) < 0.3);  // Near zero
  REQUIRE(result.kurtosis < 0);  // Negative for uniform
}

TEST_CASE("DistributionClassifier - Normal Detection") {
  auto data = TestDataGenerator::GenerateNormal(10000);

  auto result = DistributionClassifier<float>::Classify(data.data(), data.size());

  INFO("Detected: " << result.ToString());
  INFO("Confidence: " << result.confidence);
  INFO("Skewness: " << result.skewness << " (expected ~0)");
  INFO("Kurtosis: " << result.kurtosis << " (expected ~0)");
  INFO("Scores - U:" << result.uniform_score << " N:" << result.normal_score
       << " G:" << result.gamma_score << " E:" << result.exponential_score);

  REQUIRE(result.type == DistributionType::NORMAL);
  REQUIRE(std::abs(result.skewness) < 0.3);
  REQUIRE(std::abs(result.kurtosis) < 0.5);
}

TEST_CASE("DistributionClassifier - Exponential Detection") {
  auto data = TestDataGenerator::GenerateExponential(10000);

  auto result = DistributionClassifier<float>::Classify(data.data(), data.size());

  INFO("Detected: " << result.ToString());
  INFO("Confidence: " << result.confidence);
  INFO("Skewness: " << result.skewness << " (expected ~2)");
  INFO("Kurtosis: " << result.kurtosis << " (expected ~6)");
  INFO("Scores - U:" << result.uniform_score << " N:" << result.normal_score
       << " G:" << result.gamma_score << " E:" << result.exponential_score);

  REQUIRE(result.type == DistributionType::EXPONENTIAL);
  REQUIRE(result.skewness > 1.5);
  REQUIRE(result.kurtosis > 4.0);
}

TEST_CASE("DistributionClassifier - Gamma Detection") {
  // Gamma with shape=2 has skewness = 2/sqrt(2) ≈ 1.41
  auto data = TestDataGenerator::GenerateGamma(10000, 2.0f, 5.0f);

  auto result = DistributionClassifier<float>::Classify(data.data(), data.size());

  INFO("Detected: " << result.ToString());
  INFO("Confidence: " << result.confidence);
  INFO("Skewness: " << result.skewness << " (expected ~1.41)");
  INFO("Kurtosis: " << result.kurtosis << " (expected ~3)");
  INFO("Scores - U:" << result.uniform_score << " N:" << result.normal_score
       << " G:" << result.gamma_score << " E:" << result.exponential_score);

  // Gamma with shape=2 should be detected as gamma (or exponential since similar)
  REQUIRE((result.type == DistributionType::GAMMA ||
           result.type == DistributionType::EXPONENTIAL));
  REQUIRE(result.skewness > 0.5);
}

TEST_CASE("DistributionClassifier - Integer Data") {
  // Test with integer data
  std::vector<int32_t> data(10000);
  std::mt19937 gen(42);
  std::normal_distribution<float> dist(1000.0f, 100.0f);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<int32_t>(dist(gen));
  }

  auto result = DistributionClassifier<int32_t>::Classify(data.data(), data.size());

  INFO("Detected: " << result.ToString());
  INFO("Skewness: " << result.skewness);
  INFO("Kurtosis: " << result.kurtosis);

  REQUIRE(result.type == DistributionType::NORMAL);
}

TEST_CASE("DistributionClassifier - Byte Data") {
  // Test with uint8 uniform data
  std::vector<uint8_t> data(10000);
  std::mt19937 gen(42);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = gen() % 256;
  }

  auto result = DistributionClassifier<uint8_t>::Classify(data.data(), data.size());

  INFO("Detected: " << result.ToString());
  INFO("Skewness: " << result.skewness);
  INFO("Kurtosis: " << result.kurtosis);

  REQUIRE(result.type == DistributionType::UNIFORM);
}

TEST_CASE("DistributionClassifier - Constant Data") {
  std::vector<float> data(1000, 42.0f);

  auto result = DistributionClassifier<float>::Classify(data.data(), data.size());

  INFO("Detected: " << result.ToString());

  REQUIRE(result.type == DistributionType::CONSTANT);
  REQUIRE(result.confidence == 1.0);
}

TEST_CASE("DistributionClassifier - Factory Interface") {
  auto data = TestDataGenerator::GenerateNormal(5000);

  auto result = DistributionClassifierFactory::Classify(
      data.data(), data.size(), DataType::FLOAT32);

  REQUIRE(result.type == DistributionType::NORMAL);
}

TEST_CASE("DistributionClassifier - Small Data") {
  // Very small data should return unknown
  std::vector<float> data = {1.0f, 2.0f, 3.0f};

  auto result = DistributionClassifier<float>::Classify(data.data(), data.size());

  REQUIRE(result.type == DistributionType::UNKNOWN);
}
