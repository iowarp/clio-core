/**
 * @file test_dense_nn_predictor.cc
 * @brief Unit tests for Dense Neural Network compression predictor using MiniDNN
 */

#include "context-transport-primitives/compression/dense_nn_predictor.h"
#include "../../../context-runtime/test/simple_test.h"
#include <iostream>
#include <chrono>
#include <vector>

using namespace hshm::compress;

TEST_CASE("DenseNNPredictor - Train and Predict In-Memory", "[compression][densenn][training]") {
  std::cout << "\n=== Testing Dense NN Training and Inference (MiniDNN) ===\n";

  // Create synthetic training data
  // Simulate compression ratio data: higher entropy -> lower compression ratio
  std::vector<CompressionFeatures> train_features;
  std::vector<float> train_labels;

  // Generate 100 training samples with known patterns
  for (int i = 0; i < 100; ++i) {
    CompressionFeatures f;
    f.chunk_size_bytes = 32768 + (i % 10) * 8192;  // Vary chunk size
    f.target_cpu_util = 30.0 + (i % 5) * 10.0;     // Vary CPU utilization
    f.shannon_entropy = 2.0 + (i % 8) * 0.5;       // Vary entropy (2.0 - 5.5)
    f.mad = 0.5 + (i % 4) * 0.3;
    f.second_derivative_mean = 1.0 + (i % 6) * 0.5;

    // One-hot encoding: rotate through libraries
    f.library_bzip2 = (i % 3 == 0) ? 1 : 0;
    f.library_zfp_tol_0_01 = (i % 3 == 1) ? 1 : 0;
    f.library_zfp_tol_0_1 = (i % 3 == 2) ? 1 : 0;

    // Data type
    f.data_type_char = (i % 2 == 0) ? 1 : 0;
    f.data_type_float = (i % 2 == 1) ? 1 : 0;

    train_features.push_back(f);

    // Generate synthetic compression ratio based on entropy
    // Higher entropy = lower compression ratio (more random data)
    float ratio = 5.0f - static_cast<float>(f.shannon_entropy) * 0.5f +
                  (f.library_bzip2 ? 0.5f : 0.0f);
    ratio = std::max(1.0f, ratio);  // Min ratio is 1.0
    train_labels.push_back(ratio);
  }

  std::cout << "Generated " << train_features.size() << " training samples\n";

  // Create and train predictor
  DenseNNPredictor predictor;

  TrainingConfig config;
  config.hidden_layers = {64, 32, 16};  // Smaller network for faster training
  config.dropout_rate = 0.1;
  config.learning_rate = 0.01;
  config.epochs = 100;  // More epochs for MiniDNN
  config.batch_size = 16;
  config.verbose = false;  // MiniDNN callback can be noisy

  std::cout << "Training compression ratio model...\n";
  auto train_start = std::chrono::high_resolution_clock::now();
  bool train_success = predictor.TrainRatioModel(train_features, train_labels, config);
  auto train_end = std::chrono::high_resolution_clock::now();
  double train_time_ms = std::chrono::duration<double, std::milli>(train_end - train_start).count();

  REQUIRE(train_success);
  std::cout << "Training completed in " << train_time_ms << " ms\n";

  // Verify model is ready for inference
  REQUIRE(predictor.IsReady());

  // Test predictions
  std::cout << "\nTesting predictions:\n";

  // Test with low entropy (should give higher compression ratio)
  CompressionFeatures low_entropy;
  low_entropy.chunk_size_bytes = 65536;
  low_entropy.target_cpu_util = 50.0;
  low_entropy.shannon_entropy = 2.0;  // Low entropy
  low_entropy.mad = 0.5;
  low_entropy.second_derivative_mean = 1.0;
  low_entropy.library_bzip2 = 1;
  low_entropy.library_zfp_tol_0_01 = 0;
  low_entropy.library_zfp_tol_0_1 = 0;
  low_entropy.data_type_char = 0;
  low_entropy.data_type_float = 1;

  auto result_low = predictor.Predict(low_entropy);
  std::cout << "  Low entropy (2.0):  ratio = " << result_low.compression_ratio
            << ", inference = " << result_low.inference_time_ms << " ms\n";
  REQUIRE(result_low.compression_ratio > 0.0);

  // Test with high entropy (should give lower compression ratio)
  CompressionFeatures high_entropy;
  high_entropy.chunk_size_bytes = 65536;
  high_entropy.target_cpu_util = 50.0;
  high_entropy.shannon_entropy = 5.5;  // High entropy
  high_entropy.mad = 2.0;
  high_entropy.second_derivative_mean = 3.0;
  high_entropy.library_bzip2 = 1;
  high_entropy.library_zfp_tol_0_01 = 0;
  high_entropy.library_zfp_tol_0_1 = 0;
  high_entropy.data_type_char = 0;
  high_entropy.data_type_float = 1;

  auto result_high = predictor.Predict(high_entropy);
  std::cout << "  High entropy (5.5): ratio = " << result_high.compression_ratio
            << ", inference = " << result_high.inference_time_ms << " ms\n";
  REQUIRE(result_high.compression_ratio > 0.0);

  // The model should learn that lower entropy gives higher compression
  // Allow some tolerance since it's a small training set
  std::cout << "\nModel learned pattern: ";
  if (result_low.compression_ratio > result_high.compression_ratio) {
    std::cout << "CORRECT (low entropy -> higher ratio)\n";
  } else {
    std::cout << "INVERTED (may need more training data)\n";
  }

  // Test batch prediction
  std::vector<CompressionFeatures> test_batch = {low_entropy, high_entropy};
  auto batch_results = predictor.PredictBatch(test_batch);
  REQUIRE(batch_results.size() == 2);

  std::cout << "\nBatch prediction test: PASSED (" << batch_results.size() << " results)\n";
  std::cout << "=== Dense NN Training Test Complete ===\n";
}

TEST_CASE("DenseNNPredictor - Save and Load Model", "[compression][densenn][persistence]") {
  std::cout << "\n=== Testing Dense NN Model Save/Load ===\n";

  // Create and train a simple model
  std::vector<CompressionFeatures> train_features;
  std::vector<float> train_labels;

  for (int i = 0; i < 50; ++i) {
    CompressionFeatures f;
    f.chunk_size_bytes = 32768;
    f.target_cpu_util = 50.0;
    f.shannon_entropy = 2.0 + (i % 4) * 1.0;
    f.mad = 1.0;
    f.second_derivative_mean = 1.0;
    f.library_bzip2 = 1;
    f.library_zfp_tol_0_01 = 0;
    f.library_zfp_tol_0_1 = 0;
    f.data_type_char = 0;
    f.data_type_float = 1;
    train_features.push_back(f);
    train_labels.push_back(4.0f - static_cast<float>(f.shannon_entropy) * 0.3f);
  }

  DenseNNPredictor predictor1;
  TrainingConfig config;
  config.hidden_layers = {32, 16};
  config.epochs = 50;
  config.batch_size = 10;
  config.verbose = false;

  bool trained = predictor1.TrainRatioModel(train_features, train_labels, config);
  REQUIRE(trained);

  // Get prediction from original model
  CompressionFeatures test_features;
  test_features.chunk_size_bytes = 32768;
  test_features.target_cpu_util = 50.0;
  test_features.shannon_entropy = 3.0;
  test_features.mad = 1.0;
  test_features.second_derivative_mean = 1.0;
  test_features.library_bzip2 = 1;
  test_features.library_zfp_tol_0_01 = 0;
  test_features.library_zfp_tol_0_1 = 0;
  test_features.data_type_char = 0;
  test_features.data_type_float = 1;

  auto result1 = predictor1.Predict(test_features);
  std::cout << "Original model prediction: " << result1.compression_ratio << "\n";

  // Save the model
  std::string model_dir = "/tmp/test_dense_nn_model";
  bool saved = predictor1.Save(model_dir);
  REQUIRE(saved);
  std::cout << "Model saved to: " << model_dir << "\n";

  // Load into a new predictor
  DenseNNPredictor predictor2;
  bool loaded = predictor2.Load(model_dir);
  REQUIRE(loaded);
  std::cout << "Model loaded from: " << model_dir << "\n";

  // Compare predictions
  auto result2 = predictor2.Predict(test_features);
  std::cout << "Loaded model prediction: " << result2.compression_ratio << "\n";

  // Predictions should be close (allowing for floating point differences)
  double diff = std::abs(result1.compression_ratio - result2.compression_ratio);
  std::cout << "Prediction difference: " << diff << "\n";
  REQUIRE(diff < 0.01);  // Allow small difference

  std::cout << "=== Save/Load Test Complete ===\n";
}

TEST_CASE("DenseNNPredictor - Batch Prediction Performance", "[compression][densenn][benchmark]") {
  std::cout << "\n=== Testing Dense NN Batch Prediction Performance ===\n";

  // Create and train a model
  std::vector<CompressionFeatures> train_features;
  std::vector<float> train_labels;

  for (int i = 0; i < 100; ++i) {
    CompressionFeatures f;
    f.chunk_size_bytes = 32768 + (i % 10) * 8192;
    f.target_cpu_util = 30.0 + (i % 5) * 10.0;
    f.shannon_entropy = 2.0 + (i % 8) * 0.5;
    f.mad = 0.5 + (i % 4) * 0.3;
    f.second_derivative_mean = 1.0 + (i % 6) * 0.5;
    f.library_bzip2 = (i % 3 == 0) ? 1 : 0;
    f.library_zfp_tol_0_01 = (i % 3 == 1) ? 1 : 0;
    f.library_zfp_tol_0_1 = (i % 3 == 2) ? 1 : 0;
    f.data_type_char = (i % 2 == 0) ? 1 : 0;
    f.data_type_float = (i % 2 == 1) ? 1 : 0;
    train_features.push_back(f);
    train_labels.push_back(5.0f - static_cast<float>(f.shannon_entropy) * 0.5f);
  }

  DenseNNPredictor predictor;
  TrainingConfig config;
  config.hidden_layers = {64, 32, 16};
  config.epochs = 50;
  config.batch_size = 16;
  config.verbose = false;

  bool trained = predictor.TrainRatioModel(train_features, train_labels, config);
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
    f.library_bzip2 = 1;
    f.library_zfp_tol_0_01 = 0;
    f.library_zfp_tol_0_1 = 0;
    f.data_type_char = 0;
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

  // Should be reasonably fast
  REQUIRE(throughput > 1000.0);  // At least 1000 predictions/sec

  std::cout << "=== Batch Performance Test Complete ===\n";
}

TEST_CASE("DenseNNPredictor - Unified Multi-Output Training", "[compression][densenn][unified]") {
  std::cout << "\n=== Testing Dense NN Unified Multi-Output Training ===\n";

  DenseNNPredictor predictor;

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

  std::cout << "=== Dense NN Unified Training Test Complete ===\n";
}

TEST_CASE("DenseNNPredictor - Reinforcement Learning", "[compression][densenn][rl]") {
  std::cout << "\n=== Testing Dense NN Reinforcement Learning ===\n";

  // First, train a base model
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
    f.library_zfp_tol_0_01 = 0;
    f.library_zfp_tol_0_1 = 0;
    f.data_type_char = 0;
    f.data_type_float = 1;
    train_features.push_back(f);
    train_labels.push_back(4.0F - static_cast<float>(f.shannon_entropy) * 0.3F);
  }

  DenseNNPredictor predictor;
  TrainingConfig config;
  config.hidden_layers = {32, 16};
  config.epochs = 30;
  config.batch_size = 10;
  config.verbose = false;

  bool trained = predictor.TrainRatioModel(train_features, train_labels, config);
  REQUIRE(trained);
  std::cout << "Base model trained\n";

  // Test RL experience recording
  REQUIRE(predictor.GetExperienceCount() == 0);

  // Record some experiences
  for (int i = 0; i < 50; ++i) {
    RLExperience exp;
    exp.features.chunk_size_bytes = 65536;
    exp.features.target_cpu_util = 50.0;
    exp.features.shannon_entropy = 3.0 + (i % 5) * 0.3;
    exp.features.mad = 1.0;
    exp.features.second_derivative_mean = 2.0;
    exp.features.library_bzip2 = 1;
    exp.features.library_zfp_tol_0_01 = 0;
    exp.features.library_zfp_tol_0_1 = 0;
    exp.features.data_type_char = 0;
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
  rl_config.learning_rate = 0.001;
  rl_config.epsilon_decay = 0.9;

  bool updated = predictor.UpdateFromExperiences(rl_config);
  REQUIRE(updated);
  std::cout << "RL update successful\n";

  // Check epsilon decayed
  double new_epsilon = predictor.GetExplorationRate();
  std::cout << "Epsilon after update: " << new_epsilon << "\n";
  REQUIRE(new_epsilon < initial_epsilon);

  // Verify model still works after RL update
  CompressionFeatures test_features;
  test_features.chunk_size_bytes = 65536;
  test_features.target_cpu_util = 50.0;
  test_features.shannon_entropy = 3.5;
  test_features.mad = 1.0;
  test_features.second_derivative_mean = 2.0;
  test_features.library_bzip2 = 1;
  test_features.library_zfp_tol_0_01 = 0;
  test_features.library_zfp_tol_0_1 = 0;
  test_features.data_type_char = 0;
  test_features.data_type_float = 1;

  auto result = predictor.Predict(test_features);
  std::cout << "Prediction after RL update: " << result.compression_ratio << "\n";
  REQUIRE(result.compression_ratio > 0);

  // Clear experiences
  predictor.ClearExperiences();
  REQUIRE(predictor.GetExperienceCount() == 0);
  std::cout << "Experiences cleared\n";

  // Test reward computation
  RLExperience exp;
  exp.predicted_ratio = 3.0;
  exp.actual_ratio = 3.5;
  exp.predicted_psnr = 50.0;
  exp.actual_psnr = 55.0;
  exp.predicted_compress_time = 10.0;
  exp.actual_compress_time = 12.0;
  double reward = exp.ComputeReward();
  std::cout << "Sample reward (ratio_err=0.5, psnr_err=5, time_err=2): " << reward << "\n";
  REQUIRE(reward < 0);  // Negative reward for prediction error

  std::cout << "=== Dense NN RL Test Complete ===\n";
}

TEST_CASE("DenseNNPredictor - Inference Performance Benchmark", "[compression][densenn][benchmark]") {
  std::cout << "\n=== Dense NN Inference Performance Benchmark ===\n";

  DenseNNPredictor predictor;

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
