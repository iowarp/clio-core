/**
 * @file test_qtable_predictor.cc
 * @brief Unit tests for Q-Table compression predictor
 */

#include "wrp_cte/compressor/models/qtable_predictor.h"
#include "../../../context-runtime/test/simple_test.h"
#include <iostream>
#include <chrono>
#include <vector>

using namespace wrp_cte::compressor;

TEST_CASE("QTablePredictor - Train and Predict", "[compression][qtable][basic]") {
  std::cout << "\n=== Testing Q-Table Training and Inference ===\n";

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

  std::cout << "Generated " << train_features.size() << " training samples\n";

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
  std::cout << "Training completed in " << train_time_ms << " ms\n";
  std::cout << predictor.GetStatistics();

  // Test predictions
  std::cout << "\nTesting predictions:\n";

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
  std::cout << "  Low entropy (2.0):  ratio = " << result_low.compression_ratio
            << ", inference = " << result_low.inference_time_ms << " ms\n";
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
  std::cout << "  High entropy (5.5): ratio = " << result_high.compression_ratio
            << ", inference = " << result_high.inference_time_ms << " ms\n";
  REQUIRE(result_high.compression_ratio > 0.0);

  // Check pattern
  std::cout << "\nPattern check: ";
  if (result_low.compression_ratio > result_high.compression_ratio) {
    std::cout << "CORRECT (low entropy -> higher ratio)\n";
  } else {
    std::cout << "May need more training samples\n";
  }

  std::cout << "=== Q-Table Training Test Complete ===\n";
}

TEST_CASE("QTablePredictor - Save and Load", "[compression][qtable][persistence]") {
  std::cout << "\n=== Testing Q-Table Save/Load ===\n";

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
  std::cout << "Original model prediction: " << result1.compression_ratio << "\n";

  // Save the model
  std::string model_dir = "/tmp/test_qtable_model";
  bool saved = predictor1.Save(model_dir);
  REQUIRE(saved);
  std::cout << "Model saved to: " << model_dir << "\n";

  // Load into a new predictor
  QTablePredictor predictor2;
  bool loaded = predictor2.Load(model_dir);
  REQUIRE(loaded);
  std::cout << "Model loaded from: " << model_dir << "\n";

  // Compare predictions
  auto result2 = predictor2.Predict(test_features);
  std::cout << "Loaded model prediction: " << result2.compression_ratio << "\n";

  // Predictions should be identical
  double diff = std::abs(result1.compression_ratio - result2.compression_ratio);
  std::cout << "Prediction difference: " << diff << "\n";
  REQUIRE(diff < 0.0001);

  std::cout << "=== Save/Load Test Complete ===\n";
}

TEST_CASE("QTablePredictor - Batch Prediction Performance", "[compression][qtable][benchmark]") {
  std::cout << "\n=== Testing Q-Table Batch Prediction Performance ===\n";

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
  std::cout << "Batch size: " << test_batch.size() << " samples\n";
  std::cout << "Total time: " << total_time_ms << " ms\n";
  std::cout << "Time per sample: " << (total_time_ms / test_batch.size()) << " ms\n";
  std::cout << "Throughput: " << throughput << " predictions/sec\n";
  std::cout << "Unknown states: " << predictor.GetUnknownCount()
            << " (" << (100.0 * predictor.GetUnknownCount() / test_batch.size()) << "%)\n";

  // Should be very fast (target: > 100k predictions/sec)
  REQUIRE(throughput > 10000.0);

  std::cout << "=== Batch Performance Test Complete ===\n";
}

TEST_CASE("QTablePredictor - Nearest Neighbor Fallback", "[compression][qtable][nn]") {
  std::cout << "\n=== Testing Q-Table with Nearest Neighbor Fallback ===\n";

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

  std::cout << "Without NN:\n" << predictor_no_nn.GetStatistics();
  std::cout << "With NN:\n" << predictor_with_nn.GetStatistics();

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

  std::cout << "\nUnknown state prediction (entropy=3.5):\n";
  std::cout << "  Without NN: ratio = " << result_no_nn.compression_ratio << "\n";
  std::cout << "  With NN:    ratio = " << result_with_nn.compression_ratio << "\n";

  REQUIRE(result_no_nn.compression_ratio > 0.0);
  REQUIRE(result_with_nn.compression_ratio > 0.0);

  // NN should ideally give interpolated value between neighbors
  // Without NN uses global average

  std::cout << "=== Nearest Neighbor Test Complete ===\n";
}

TEST_CASE("QTablePredictor - Inference Performance Benchmark", "[compression][qtable][benchmark]") {
  std::cout << "\n=== Q-Table Inference Performance Benchmark ===\n";

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

    std::cout << "Batch size " << batch_size << ":\n";
    std::cout << "  Avg batch time: " << avg_time_ms << " ms\n";
    std::cout << "  Throughput: " << throughput << " predictions/sec\n";
  }

  std::cout << "=== Benchmark Complete ===\n";
}

SIMPLE_TEST_MAIN()
