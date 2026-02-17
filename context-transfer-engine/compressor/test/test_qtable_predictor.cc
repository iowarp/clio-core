/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * @file test_qtable_predictor.cc
 * @brief Unit tests for Q-Table compression predictor
 */

#include "wrp_cte/compressor/models/qtable_predictor.h"
#include "../../../context-runtime/test/simple_test.h"
#include "hermes_shm/util/logging.h"
#include <iostream>
#include <chrono>
#include <vector>

using namespace wrp_cte::compressor;

TEST_CASE("QTablePredictor - Train and Predict", "[compression][qtable][basic]") {
  HLOG(kInfo, "");
  HLOG(kInfo, "=== Testing Q-Table Training and Inference ===");

  // Create synthetic training data
  std::vector<CompressionFeatures> train_features;
  std::vector<TrainingLabels> train_labels;

  // Generate 100 training samples with known patterns
  for (int i = 0; i < 100; ++i) {
    CompressionFeatures f;
    f.chunk_size_bytes = 32768 + (i % 10) * 8192;
    f.target_cpu_util = 30.0 + (i % 5) * 10.0;
    f.shannon_entropy = 2.0 + (i % 8) * 0.5;
    f.mad = 0.5 + (i % 4) * 0.3;
    f.second_derivative_mean = 1.0 + (i % 6) * 0.5;
    f.library_config_id = 11 + (i % 3);  // BZIP2 variants
    f.config_fast = (i % 3 == 0) ? 1 : 0;
    f.config_balanced = (i % 3 == 1) ? 1 : 0;
    f.config_best = (i % 3 == 2) ? 1 : 0;
    f.data_type_char = (i % 2 == 0) ? 1 : 0;
    f.data_type_float = (i % 2 == 1) ? 1 : 0;

    train_features.push_back(f);

    // Synthetic labels: higher entropy -> lower compression ratio
    TrainingLabels label;
    label.compression_ratio = 5.0f - static_cast<float>(f.shannon_entropy) * 0.5f;
    label.psnr_db = 0.0f;  // Lossless
    label.compression_time_ms = 10.0f + static_cast<float>(f.shannon_entropy) * 2.0f;
    train_labels.push_back(label);
  }

  HLOG(kInfo, "Generated {} training samples", train_features.size());

  // Create and train Q-table predictor
  QTableConfig config;
  config.n_bins = 15;  // Optimal from Python experiments
  config.use_nearest_neighbor = false;

  QTablePredictor predictor(config);

  auto train_start = std::chrono::high_resolution_clock::now();
  bool train_success = predictor.Train(train_features, train_labels);
  auto train_end = std::chrono::high_resolution_clock::now();
  double train_time_ms = std::chrono::duration<double, std::milli>(train_end - train_start).count();

  REQUIRE(train_success);
  REQUIRE(predictor.IsReady());
  HLOG(kInfo, "Training completed in {} ms", train_time_ms);
  HIPRINT("{}", predictor.GetStatistics());

  // Test predictions
  HLOG(kInfo, "");
  HLOG(kInfo, "Testing predictions:");

  // Low entropy test
  CompressionFeatures low_entropy;
  low_entropy.chunk_size_bytes = 65536;
  low_entropy.target_cpu_util = 50.0;
  low_entropy.shannon_entropy = 2.0;
  low_entropy.mad = 0.5;
  low_entropy.second_derivative_mean = 1.0;
  low_entropy.library_config_id = 11;
  low_entropy.config_fast = 1;
  low_entropy.data_type_float = 1;

  auto result_low = predictor.Predict(low_entropy);
  HLOG(kInfo, "  Low entropy (2.0):  ratio = {}, inference = {} ms",
       result_low.compression_ratio, result_low.inference_time_ms);
  REQUIRE(result_low.compression_ratio > 0.0);

  // High entropy test
  CompressionFeatures high_entropy;
  high_entropy.chunk_size_bytes = 65536;
  high_entropy.target_cpu_util = 50.0;
  high_entropy.shannon_entropy = 5.5;
  high_entropy.mad = 2.0;
  high_entropy.second_derivative_mean = 3.0;
  high_entropy.library_config_id = 11;
  high_entropy.config_fast = 1;
  high_entropy.data_type_float = 1;

  auto result_high = predictor.Predict(high_entropy);
  HLOG(kInfo, "  High entropy (5.5): ratio = {}, inference = {} ms",
       result_high.compression_ratio, result_high.inference_time_ms);
  REQUIRE(result_high.compression_ratio > 0.0);

  // Check pattern
  HIPRINT("\nPattern check: ");
  if (result_low.compression_ratio > result_high.compression_ratio) {
    HIPRINT("CORRECT (low entropy -> higher ratio)\n");
  } else {
    HIPRINT("May need more training samples\n");
  }

  HLOG(kInfo, "=== Q-Table Training Test Complete ===");
}

TEST_CASE("QTablePredictor - Save and Load", "[compression][qtable][persistence]") {
  HLOG(kInfo, "");
  HLOG(kInfo, "=== Testing Q-Table Save/Load ===");

  // Create and train a simple Q-table
  std::vector<CompressionFeatures> train_features;
  std::vector<TrainingLabels> train_labels;

  for (int i = 0; i < 50; ++i) {
    CompressionFeatures f;
    f.chunk_size_bytes = 32768;
    f.target_cpu_util = 50.0;
    f.shannon_entropy = 2.0 + (i % 4) * 1.0;
    f.mad = 1.0;
    f.second_derivative_mean = 1.0;
    f.library_config_id = 11;
    f.config_fast = 1;
    f.data_type_float = 1;
    train_features.push_back(f);

    TrainingLabels label;
    label.compression_ratio = 4.0f - static_cast<float>(f.shannon_entropy) * 0.3f;
    label.psnr_db = 0.0f;
    label.compression_time_ms = 10.0f;
    train_labels.push_back(label);
  }

  QTableConfig config;
  config.n_bins = 10;
  QTablePredictor predictor1(config);

  bool trained = predictor1.Train(train_features, train_labels);
  REQUIRE(trained);

  // Get prediction from original model
  CompressionFeatures test_features;
  test_features.chunk_size_bytes = 32768;
  test_features.target_cpu_util = 50.0;
  test_features.shannon_entropy = 3.0;
  test_features.mad = 1.0;
  test_features.second_derivative_mean = 1.0;
  test_features.library_config_id = 11;
  test_features.config_fast = 1;
  test_features.data_type_float = 1;

  auto result1 = predictor1.Predict(test_features);
  HLOG(kInfo, "Original model prediction: {}", result1.compression_ratio);

  // Save the model
  std::string model_dir = "/tmp/test_qtable_model";
  bool saved = predictor1.Save(model_dir);
  REQUIRE(saved);
  HLOG(kInfo, "Model saved to: {}", model_dir);

  // Load into a new predictor
  QTablePredictor predictor2;
  bool loaded = predictor2.Load(model_dir);
  REQUIRE(loaded);
  HLOG(kInfo, "Model loaded from: {}", model_dir);

  // Compare predictions
  auto result2 = predictor2.Predict(test_features);
  HLOG(kInfo, "Loaded model prediction: {}", result2.compression_ratio);

  // Predictions should be identical
  double diff = std::abs(result1.compression_ratio - result2.compression_ratio);
  HLOG(kInfo, "Prediction difference: {}", diff);
  REQUIRE(diff < 0.0001);

  HLOG(kInfo, "=== Save/Load Test Complete ===");
}

TEST_CASE("QTablePredictor - Batch Prediction Performance", "[compression][qtable][benchmark]") {
  HLOG(kInfo, "");
  HLOG(kInfo, "=== Testing Q-Table Batch Prediction Performance ===");

  // Create and train a Q-table
  std::vector<CompressionFeatures> train_features;
  std::vector<TrainingLabels> train_labels;

  for (int i = 0; i < 200; ++i) {
    CompressionFeatures f;
    f.chunk_size_bytes = 32768 + (i % 10) * 8192;
    f.target_cpu_util = 30.0 + (i % 5) * 10.0;
    f.shannon_entropy = 2.0 + (i % 8) * 0.5;
    f.mad = 0.5 + (i % 4) * 0.3;
    f.second_derivative_mean = 1.0 + (i % 6) * 0.5;
    f.library_config_id = 11 + (i % 3);
    f.config_fast = (i % 3 == 0) ? 1 : 0;
    f.config_balanced = (i % 3 == 1) ? 1 : 0;
    f.config_best = (i % 3 == 2) ? 1 : 0;
    f.data_type_char = (i % 2 == 0) ? 1 : 0;
    f.data_type_float = (i % 2 == 1) ? 1 : 0;
    train_features.push_back(f);

    TrainingLabels label;
    label.compression_ratio = 5.0f - static_cast<float>(f.shannon_entropy) * 0.5f;
    label.psnr_db = 0.0f;
    label.compression_time_ms = 10.0f + static_cast<float>(f.shannon_entropy) * 2.0f;
    train_labels.push_back(label);
  }

  QTableConfig config;
  config.n_bins = 15;
  QTablePredictor predictor(config);

  bool trained = predictor.Train(train_features, train_labels);
  REQUIRE(trained);

  // Create test batch
  std::vector<CompressionFeatures> test_batch;
  for (int i = 0; i < 1000; ++i) {
    CompressionFeatures f;
    f.chunk_size_bytes = 32768;
    f.target_cpu_util = 50.0;
    f.shannon_entropy = 2.0 + (i % 8) * 0.5;
    f.mad = 1.0;
    f.second_derivative_mean = 1.0;
    f.library_config_id = 11;
    f.config_fast = 1;
    f.data_type_float = 1;
    test_batch.push_back(f);
  }

  // Benchmark batch prediction
  auto start = std::chrono::high_resolution_clock::now();
  auto results = predictor.PredictBatch(test_batch);
  auto end = std::chrono::high_resolution_clock::now();
  double total_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

  REQUIRE(results.size() == test_batch.size());

  double throughput = (test_batch.size() / total_time_ms) * 1000.0;
  HLOG(kInfo, "Batch size: {} samples", test_batch.size());
  HLOG(kInfo, "Total time: {} ms", total_time_ms);
  HLOG(kInfo, "Time per sample: {} ms", total_time_ms / test_batch.size());
  HLOG(kInfo, "Throughput: {} predictions/sec", throughput);
  HLOG(kInfo, "Unknown states: {} ({}%)", predictor.GetUnknownCount(),
       100.0 * predictor.GetUnknownCount() / test_batch.size());

  // Should be very fast (target: > 100k predictions/sec)
  REQUIRE(throughput > 10000.0);

  HLOG(kInfo, "=== Batch Performance Test Complete ===");
}

TEST_CASE("QTablePredictor - Nearest Neighbor Fallback", "[compression][qtable][nn]") {
  HLOG(kInfo, "");
  HLOG(kInfo, "=== Testing Q-Table with Nearest Neighbor Fallback ===");

  // Create sparse training data
  std::vector<CompressionFeatures> train_features;
  std::vector<TrainingLabels> train_labels;

  for (int i = 0; i < 30; ++i) {
    CompressionFeatures f;
    f.chunk_size_bytes = 32768;
    f.target_cpu_util = 50.0;
    f.shannon_entropy = 2.0 + (i % 3) * 2.0;  // Only 3 distinct values
    f.mad = 1.0;
    f.second_derivative_mean = 1.0;
    f.library_config_id = 11;
    f.config_fast = 1;
    f.data_type_float = 1;
    train_features.push_back(f);

    TrainingLabels label;
    label.compression_ratio = 4.0f - static_cast<float>(f.shannon_entropy) * 0.3f;
    label.psnr_db = 0.0f;
    label.compression_time_ms = 10.0f;
    train_labels.push_back(label);
  }

  // Train two predictors: one with NN, one without
  QTableConfig config_no_nn;
  config_no_nn.n_bins = 10;
  config_no_nn.use_nearest_neighbor = false;

  QTableConfig config_with_nn;
  config_with_nn.n_bins = 10;
  config_with_nn.use_nearest_neighbor = true;
  config_with_nn.nn_k = 5;

  QTablePredictor predictor_no_nn(config_no_nn);
  QTablePredictor predictor_with_nn(config_with_nn);

  REQUIRE(predictor_no_nn.Train(train_features, train_labels));
  REQUIRE(predictor_with_nn.Train(train_features, train_labels));

  HIPRINT("Without NN:\n{}", predictor_no_nn.GetStatistics());
  HIPRINT("With NN:\n{}", predictor_with_nn.GetStatistics());

  // Test on unknown state (entropy = 3.5, between 2.0 and 4.0)
  CompressionFeatures unknown_state;
  unknown_state.chunk_size_bytes = 32768;
  unknown_state.target_cpu_util = 50.0;
  unknown_state.shannon_entropy = 3.5;  // Not in training data
  unknown_state.mad = 1.0;
  unknown_state.second_derivative_mean = 1.0;
  unknown_state.library_config_id = 11;
  unknown_state.config_fast = 1;
  unknown_state.data_type_float = 1;

  auto result_no_nn = predictor_no_nn.Predict(unknown_state);
  auto result_with_nn = predictor_with_nn.Predict(unknown_state);

  HLOG(kInfo, "");
  HLOG(kInfo, "Unknown state prediction (entropy=3.5):");
  HLOG(kInfo, "  Without NN: ratio = {}", result_no_nn.compression_ratio);
  HLOG(kInfo, "  With NN:    ratio = {}", result_with_nn.compression_ratio);

  REQUIRE(result_no_nn.compression_ratio > 0.0);
  REQUIRE(result_with_nn.compression_ratio > 0.0);

  // NN should ideally give interpolated value between neighbors
  // Without NN uses global average

  HLOG(kInfo, "=== Nearest Neighbor Test Complete ===");
}

TEST_CASE("QTablePredictor - Inference Performance Benchmark", "[compression][qtable][benchmark]") {
  HLOG(kInfo, "");
  HLOG(kInfo, "=== Q-Table Inference Performance Benchmark ===");

  // Train a realistic Q-table
  std::vector<CompressionFeatures> train_features;
  std::vector<TrainingLabels> train_labels;

  for (int i = 0; i < 500; ++i) {
    CompressionFeatures f;
    f.chunk_size_bytes = 32768 + (i % 20) * 4096;
    f.target_cpu_util = 30.0 + (i % 10) * 7.0;
    f.shannon_entropy = 1.0 + (i % 15) * 0.4;
    f.mad = 0.5 + (i % 8) * 0.2;
    f.second_derivative_mean = 1.0 + (i % 10) * 0.3;
    f.library_config_id = 11 + (i % 3);
    f.config_fast = (i % 3 == 0) ? 1 : 0;
    f.config_balanced = (i % 3 == 1) ? 1 : 0;
    f.config_best = (i % 3 == 2) ? 1 : 0;
    f.data_type_char = (i % 2 == 0) ? 1 : 0;
    f.data_type_float = (i % 2 == 1) ? 1 : 0;
    train_features.push_back(f);

    TrainingLabels label;
    label.compression_ratio = 5.0f - static_cast<float>(f.shannon_entropy) * 0.5f;
    label.psnr_db = 0.0f;
    label.compression_time_ms = 10.0f + static_cast<float>(f.shannon_entropy) * 2.0f;
    train_labels.push_back(label);
  }

  QTableConfig config;
  config.n_bins = 15;
  QTablePredictor predictor(config);

  REQUIRE(predictor.Train(train_features, train_labels));

  // Benchmark different batch sizes
  std::vector<size_t> batch_sizes = {1, 16, 64, 256, 1024};

  for (size_t batch_size : batch_sizes) {
    std::vector<CompressionFeatures> batch;
    for (size_t i = 0; i < batch_size; ++i) {
      CompressionFeatures f;
      f.chunk_size_bytes = 65536;
      f.target_cpu_util = 50.0;
      f.shannon_entropy = 3.0;
      f.mad = 1.0;
      f.second_derivative_mean = 2.0;
      f.library_config_id = 11;
      f.config_fast = 1;
      f.data_type_float = 1;
      batch.push_back(f);
    }

    // Warmup
    predictor.PredictBatch(batch);

    // Benchmark
    const int num_iterations = 100;
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < num_iterations; ++iter) {
      auto results = predictor.PredictBatch(batch);
      (void)results;
    }
    auto end = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_time_ms = total_time_ms / num_iterations;
    double throughput = (batch_size / avg_time_ms) * 1000.0;

    HLOG(kInfo, "Batch size {}:", batch_size);
    HLOG(kInfo, "  Avg batch time: {} ms", avg_time_ms);
    HLOG(kInfo, "  Throughput: {} predictions/sec", throughput);
  }

  HLOG(kInfo, "=== Benchmark Complete ===");
}

SIMPLE_TEST_MAIN()
