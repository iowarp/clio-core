/**
 * @file test_xgboost_predictor.cc
 * @brief Unit tests for XGBoost compression predictor
 */

#include "context-transport-primitives/compression/xgboost_predictor.h"
#include "../../../context-runtime/test/simple_test.h"
#include <iostream>
#include <chrono>
#include <vector>

using namespace hshm::compress;

TEST_CASE("XGBoostPredictor - Single Prediction", "[compression][xgboost]") {
  // Create predictor
  std::string model_path = "context-transport-primitives/scripts/model_output_cpp";
  XGBoostPredictor predictor;

  if (!predictor.Load(model_path)) {
    INFO("Skipping test: Models not found at " << model_path);
    INFO("Run: cd context-transport-primitives/scripts && python export_models_to_cpp.py model_output_cpp");
    return;
  }

  // Prepare test features for BZIP2 compression
  CompressionFeatures features;
  features.chunk_size_bytes = 65536;      // 64KB
  features.target_cpu_util = 50.0;
  features.shannon_entropy = 3.5;
  features.mad = 1.2;
  features.second_derivative_mean = 2.5;
  features.library_bzip2 = 1;
  features.library_zfp_tol_0_01 = 0;
  features.library_zfp_tol_0_1 = 0;
  features.data_type_char = 0;
  features.data_type_float = 1;

  // Predict
  auto result = predictor.Predict(features);

  // Verify results
  REQUIRE(result.compression_ratio > 0.0);
  REQUIRE(result.compression_ratio < 100.0);  // Reasonable range
  REQUIRE(result.psnr_db == 0.0);  // Lossless compression
  REQUIRE(result.inference_time_ms > 0.0);
  REQUIRE(result.inference_time_ms < 100.0);  // Should be fast

  std::cout << "XGBoost Single Prediction:\n";
  std::cout << "  Compression Ratio: " << result.compression_ratio << "\n";
  std::cout << "  PSNR: " << result.psnr_db << " dB\n";
  std::cout << "  Compression Time: " << result.compression_time_ms << " ms\n";
  std::cout << "  Inference Time: " << result.inference_time_ms << " ms\n";
}

TEST_CASE("XGBoostPredictor - Batch Prediction", "[compression][xgboost]") {
  std::string model_path = "context-transport-primitives/scripts/model_output_cpp";
  XGBoostPredictor predictor;

  if (!predictor.Load(model_path)) {
    INFO("Skipping test: Models not found");
    return;
  }

  // Create batch of 16 features
  std::vector<CompressionFeatures> batch;
  for (int i = 0; i < 16; ++i) {
    CompressionFeatures features;
    features.chunk_size_bytes = 65536;
    features.target_cpu_util = 50.0;
    features.shannon_entropy = 3.5;
    features.mad = 1.2;
    features.second_derivative_mean = 2.5;
    features.library_bzip2 = 1;
    features.library_zfp_tol_0_01 = 0;
    features.library_zfp_tol_0_1 = 0;
    features.data_type_char = 0;
    features.data_type_float = 1;
    batch.push_back(features);
  }

  // Batch predict
  auto start = std::chrono::high_resolution_clock::now();
  auto results = predictor.PredictBatch(batch);
  auto end = std::chrono::high_resolution_clock::now();
  double total_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

  // Verify results
  REQUIRE(results.size() == batch.size());
  for (const auto& result : results) {
    REQUIRE(result.compression_ratio > 0.0);
    REQUIRE(result.psnr_db == 0.0);  // Lossless
  }

  std::cout << "\nXGBoost Batch Prediction (16 samples):\n";
  std::cout << "  Total Time: " << total_time_ms << " ms\n";
  std::cout << "  Time per Sample: " << (total_time_ms / batch.size()) << " ms\n";
  std::cout << "  Throughput: " << ((batch.size() / total_time_ms) * 1000.0) << " predictions/sec\n";
}

TEST_CASE("XGBoostPredictor - Lossy Compression PSNR", "[compression][xgboost]") {
  std::string model_path = "context-transport-primitives/scripts/model_output_cpp";
  XGBoostPredictor predictor;

  if (!predictor.Load(model_path)) {
    INFO("Skipping test: Models not found");
    return;
  }

  // Test ZFP lossy compression
  CompressionFeatures features;
  features.chunk_size_bytes = 65536;
  features.target_cpu_util = 50.0;
  features.shannon_entropy = 3.5;
  features.mad = 1.2;
  features.second_derivative_mean = 2.5;
  features.library_bzip2 = 0;
  features.library_zfp_tol_0_01 = 1;  // Lossy
  features.library_zfp_tol_0_1 = 0;
  features.data_type_char = 0;
  features.data_type_float = 1;

  auto result = predictor.Predict(features);

  // Verify PSNR is predicted for lossy compression
  REQUIRE(result.compression_ratio > 0.0);
  REQUIRE(result.psnr_db > 0.0);  // Should have PSNR value
  REQUIRE(result.psnr_db < 200.0);  // Reasonable PSNR range

  std::cout << "\nLossy Compression (ZFP tol=0.01):\n";
  std::cout << "  Compression Ratio: " << result.compression_ratio << "\n";
  std::cout << "  PSNR: " << result.psnr_db << " dB\n";
}

TEST_CASE("XGBoostPredictor - Train and Predict In-Memory", "[compression][xgboost][training]") {
  std::cout << "\n=== Testing XGBoost Training and Inference ===\n";

  XGBoostPredictor predictor;

  // Generate synthetic training data
  std::vector<CompressionFeatures> train_features;
  std::vector<float> train_labels;

  // Create 100 training samples with varying entropy
  for (int i = 0; i < 100; ++i) {
    CompressionFeatures f;
    f.chunk_size_bytes = 65536;
    f.target_cpu_util = 50.0;
    f.shannon_entropy = 1.0 + (i % 10) * 0.5;  // 1.0 to 5.5
    f.mad = 1.0 + (i % 5) * 0.2;
    f.second_derivative_mean = 2.0;
    f.library_bzip2 = 1;
    f.library_zfp_tol_0_01 = 0;
    f.library_zfp_tol_0_1 = 0;
    f.data_type_char = 0;
    f.data_type_float = 1;
    train_features.push_back(f);

    // Synthetic label: higher entropy -> lower compression ratio
    float ratio = 6.0f - f.shannon_entropy * 0.5f;
    train_labels.push_back(ratio);
  }

  std::cout << "Generated " << train_features.size() << " training samples\n";

  // Train with default config
  XGBoostConfig config;
  config.n_estimators = 50;  // Fewer trees for faster training
  config.verbose = false;

  std::cout << "Training compression ratio model...\n";
  auto start = std::chrono::high_resolution_clock::now();
  bool train_success = predictor.TrainRatioModel(train_features, train_labels, config);
  auto end = std::chrono::high_resolution_clock::now();
  double train_time = std::chrono::duration<double, std::milli>(end - start).count();

  REQUIRE(train_success);
  REQUIRE(predictor.IsReady());
  std::cout << "Training completed in " << train_time << " ms\n";

  // Test predictions
  CompressionFeatures low_entropy;
  low_entropy.chunk_size_bytes = 65536;
  low_entropy.target_cpu_util = 50.0;
  low_entropy.shannon_entropy = 2.0;  // Low entropy
  low_entropy.mad = 1.0;
  low_entropy.second_derivative_mean = 2.0;
  low_entropy.library_bzip2 = 1;
  low_entropy.data_type_float = 1;

  CompressionFeatures high_entropy;
  high_entropy.chunk_size_bytes = 65536;
  high_entropy.target_cpu_util = 50.0;
  high_entropy.shannon_entropy = 5.5;  // High entropy
  high_entropy.mad = 1.0;
  high_entropy.second_derivative_mean = 2.0;
  high_entropy.library_bzip2 = 1;
  high_entropy.data_type_float = 1;

  auto result_low = predictor.Predict(low_entropy);
  auto result_high = predictor.Predict(high_entropy);

  std::cout << "\nTesting predictions:\n";
  std::cout << "  Low entropy (2.0):  ratio = " << result_low.compression_ratio
            << ", inference = " << result_low.inference_time_ms << " ms\n";
  std::cout << "  High entropy (5.5): ratio = " << result_high.compression_ratio
            << ", inference = " << result_high.inference_time_ms << " ms\n";

  // Verify the model learned the pattern: low entropy -> higher ratio
  bool correct_pattern = result_low.compression_ratio > result_high.compression_ratio;
  std::cout << "\nModel learned pattern: " << (correct_pattern ? "CORRECT" : "INCORRECT")
            << " (low entropy -> higher ratio)\n";
  REQUIRE(correct_pattern);

  std::cout << "=== XGBoost Training Test Complete ===\n";
}

TEST_CASE("XGBoostPredictor - Unified Multi-Output Training", "[compression][xgboost][unified]") {
  std::cout << "\n=== Testing XGBoost Unified Multi-Output Training ===\n";

  XGBoostPredictor predictor;

  // Generate synthetic training data with all outputs
  std::vector<CompressionFeatures> train_features;
  std::vector<TrainingLabels> train_labels;

  for (int i = 0; i < 100; ++i) {
    CompressionFeatures f;
    f.chunk_size_bytes = 65536;
    f.target_cpu_util = 50.0;
    f.shannon_entropy = 1.0 + (i % 10) * 0.5;
    f.mad = 1.0 + (i % 5) * 0.2;
    f.second_derivative_mean = 2.0;
    f.library_bzip2 = 1;
    f.library_zfp_tol_0_01 = 0;
    f.library_zfp_tol_0_1 = 0;
    f.data_type_char = 0;
    f.data_type_float = 1;
    train_features.push_back(f);

    // Synthetic labels
    TrainingLabels label;
    label.compression_ratio = 6.0F - static_cast<float>(f.shannon_entropy) * 0.5F;
    label.psnr_db = 0.0F;  // Lossless
    label.compression_time_ms = 10.0F + static_cast<float>(f.shannon_entropy) * 2.0F;
    train_labels.push_back(label);
  }

  std::cout << "Generated " << train_features.size() << " training samples\n";

  // Train unified model
  auto start = std::chrono::high_resolution_clock::now();
  bool train_success = predictor.Train(train_features, train_labels);
  auto end = std::chrono::high_resolution_clock::now();
  double train_time = std::chrono::duration<double, std::milli>(end - start).count();

  REQUIRE(train_success);
  REQUIRE(predictor.IsReady());
  std::cout << "Unified training completed in " << train_time << " ms\n";

  // Test predictions
  CompressionFeatures test_features;
  test_features.chunk_size_bytes = 65536;
  test_features.target_cpu_util = 50.0;
  test_features.shannon_entropy = 3.0;
  test_features.mad = 1.0;
  test_features.second_derivative_mean = 2.0;
  test_features.library_bzip2 = 1;
  test_features.data_type_float = 1;

  auto result = predictor.Predict(test_features);

  std::cout << "\nPrediction results:\n";
  std::cout << "  Compression Ratio: " << result.compression_ratio << "\n";
  std::cout << "  PSNR: " << result.psnr_db << " dB\n";
  std::cout << "  Compression Time: " << result.compression_time_ms << " ms\n";
  std::cout << "  Inference Time: " << result.inference_time_ms << " ms\n";

  REQUIRE(result.compression_ratio > 0.0);
  REQUIRE(result.compression_time_ms >= 0.0);

  std::cout << "=== XGBoost Unified Training Test Complete ===\n";
}

TEST_CASE("XGBoostPredictor - Reinforcement Learning", "[compression][xgboost][rl]") {
  std::cout << "\n=== Testing XGBoost Reinforcement Learning ===\n";

  // First, train a base model
  XGBoostPredictor predictor;

  std::vector<CompressionFeatures> train_features;
  std::vector<float> train_labels;

  for (int i = 0; i < 50; ++i) {
    CompressionFeatures f;
    f.chunk_size_bytes = 65536;
    f.target_cpu_util = 50.0;
    f.shannon_entropy = 3.0 + (i % 5) * 0.5;
    f.mad = 1.0;
    f.second_derivative_mean = 2.0;
    f.library_bzip2 = 1;
    f.data_type_float = 1;
    train_features.push_back(f);
    train_labels.push_back(4.0F - static_cast<float>(f.shannon_entropy) * 0.3F);
  }

  XGBoostConfig config;
  config.n_estimators = 30;
  config.verbose = false;
  REQUIRE(predictor.TrainRatioModel(train_features, train_labels, config));

  // Test RL experience recording
  REQUIRE(predictor.GetExperienceCount() == 0);

  // Record some experiences
  for (int i = 0; i < 50; ++i) {
    RLExperience exp;
    exp.features.chunk_size_bytes = 65536;
    exp.features.shannon_entropy = 3.0 + (i % 5) * 0.3;
    exp.features.library_bzip2 = 1;
    exp.features.data_type_float = 1;
    exp.predicted_ratio = 3.5;
    exp.actual_ratio = 3.5 + (i % 10) * 0.1;
    exp.predicted_psnr = 0;
    exp.actual_psnr = 0;
    exp.predicted_compress_time = 10.0;
    exp.actual_compress_time = 10.0 + (i % 5) * 0.5;
    predictor.RecordExperience(exp);
  }

  REQUIRE(predictor.GetExperienceCount() == 50);
  std::cout << "Recorded 50 experiences\n";

  // Test exploration mode
  double initial_epsilon = predictor.GetExplorationRate();
  std::cout << "Initial epsilon: " << initial_epsilon << "\n";
  REQUIRE(initial_epsilon > 0);

  predictor.SetExplorationMode(true);

  // Perform RL update
  RLConfig rl_config;
  rl_config.batch_size = 16;
  rl_config.epsilon_decay = 0.9;

  bool updated = predictor.UpdateFromExperiences(rl_config);
  REQUIRE(updated);
  std::cout << "RL update successful\n";

  // Check epsilon decayed
  double new_epsilon = predictor.GetExplorationRate();
  std::cout << "Epsilon after update: " << new_epsilon << "\n";
  REQUIRE(new_epsilon < initial_epsilon);

  // Clear experiences
  predictor.ClearExperiences();
  REQUIRE(predictor.GetExperienceCount() == 0);
  std::cout << "Experiences cleared\n";

  std::cout << "=== XGBoost RL Test Complete ===\n";
}

TEST_CASE("XGBoostPredictor - Inference Performance Benchmark", "[compression][xgboost][benchmark]") {
  std::cout << "\n=== XGBoost Inference Performance Benchmark ===\n";

  XGBoostPredictor predictor;

  // Train a model first
  std::vector<CompressionFeatures> train_features;
  std::vector<TrainingLabels> train_labels;

  for (int i = 0; i < 100; ++i) {
    CompressionFeatures f;
    f.chunk_size_bytes = 65536;
    f.target_cpu_util = 50.0;
    f.shannon_entropy = 1.0 + (i % 10) * 0.5;
    f.mad = 1.0;
    f.second_derivative_mean = 2.0;
    f.library_bzip2 = 1;
    f.data_type_float = 1;
    train_features.push_back(f);

    TrainingLabels label;
    label.compression_ratio = 5.0F;
    label.psnr_db = 0.0F;
    label.compression_time_ms = 10.0F;
    train_labels.push_back(label);
  }

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
      f.library_bzip2 = 1;
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
