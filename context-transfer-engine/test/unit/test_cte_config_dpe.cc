#include "simple_test.h"
#include <wrp_cte/core/core_config.h>
#include <wrp_cte/core/core_dpe.h>
#include <fstream>
#include <cstdlib>
#include <filesystem>

using namespace wrp_cte::core;

// Helper function to create a temporary file path
std::string GetTempFile(const std::string& name) {
  return "/tmp/" + name + "_" + std::to_string(std::time(nullptr)) + "_" +
         std::to_string(rand());
}

// Helper function to clean up temp files
void CleanupTempFile(const std::string& path) {
  if (std::filesystem::exists(path)) {
    std::filesystem::remove(path);
  }
}

// ============================================================================
// Config Tests
// ============================================================================

TEST_CASE("Config LoadFromFile - Complete Config", "[cte][config]") {
  std::string temp_file = GetTempFile("config_complete");

  // Create a complete YAML config file with all sections
  std::string yaml_content = R"(
performance:
  target_stat_interval_ms: 5000
  max_concurrent_operations: 64
  score_threshold: 0.7
  score_difference_threshold: 0.05

targets:
  neighborhood: 4
  default_target_timeout_ms: 30000
  poll_period_ms: 5000

storage:
  - path: /tmp/test_dev1
    bdev_type: ram
    capacity_limit: 1GB
  - path: /tmp/test_dev2
    bdev_type: file
    capacity_limit: 512MB
    score: 0.8

dpe:
  dpe_type: max_bw
)";

  std::ofstream file(temp_file);
  REQUIRE(file.is_open());
  file << yaml_content;
  file.close();

  Config config;
  REQUIRE(config.LoadFromFile(temp_file));

  // Verify all fields are populated
  REQUIRE(config.performance_.target_stat_interval_ms_ == 5000);
  REQUIRE(config.performance_.max_concurrent_operations_ == 64);
  REQUIRE(config.performance_.score_threshold_ > 0.69f &&
          config.performance_.score_threshold_ < 0.71f);
  REQUIRE(config.performance_.score_difference_threshold_ > 0.04f &&
          config.performance_.score_difference_threshold_ < 0.06f);

  REQUIRE(config.targets_.neighborhood_ == 4);
  REQUIRE(config.targets_.default_target_timeout_ms_ == 30000);
  REQUIRE(config.targets_.poll_period_ms_ == 5000);

  REQUIRE(config.storage_.devices_.size() == 2);
  REQUIRE(config.storage_.devices_[0].path_ == "/tmp/test_dev1");
  REQUIRE(config.storage_.devices_[0].bdev_type_ == "ram");
  REQUIRE(config.storage_.devices_[0].capacity_limit_ == 1024ULL * 1024 * 1024);
  REQUIRE(config.storage_.devices_[1].path_ == "/tmp/test_dev2");
  REQUIRE(config.storage_.devices_[1].bdev_type_ == "file");
  REQUIRE(config.storage_.devices_[1].capacity_limit_ == 512ULL * 1024 * 1024);
  REQUIRE(config.storage_.devices_[1].score_ > 0.79f &&
          config.storage_.devices_[1].score_ < 0.81f);

  REQUIRE(config.dpe_.dpe_type_ == "max_bw");

  CleanupTempFile(temp_file);
}

TEST_CASE("Config LoadFromFile - Empty Path", "[cte][config]") {
  Config config;
  REQUIRE_FALSE(config.LoadFromFile(""));
}

TEST_CASE("Config LoadFromFile - Nonexistent File", "[cte][config]") {
  Config config;
  REQUIRE_FALSE(config.LoadFromFile("/nonexistent/path/config.yaml"));
}

TEST_CASE("Config LoadFromFile - Invalid YAML", "[cte][config]") {
  std::string temp_file = GetTempFile("config_invalid");

  std::string invalid_yaml = "{ invalid yaml [[[";
  std::ofstream file(temp_file);
  REQUIRE(file.is_open());
  file << invalid_yaml;
  file.close();

  Config config;
  REQUIRE_FALSE(config.LoadFromFile(temp_file));

  CleanupTempFile(temp_file);
}

TEST_CASE("Config LoadFromString - Valid YAML", "[cte][config]") {
  std::string yaml_string = R"(
performance:
  target_stat_interval_ms: 3000
  max_concurrent_operations: 32
  score_threshold: 0.8
  score_difference_threshold: 0.1

targets:
  neighborhood: 8
  default_target_timeout_ms: 60000
  poll_period_ms: 3000

storage:
  - path: /tmp/storage1
    bdev_type: file
    capacity_limit: 2GB

dpe:
  dpe_type: round_robin
)";

  Config config;
  REQUIRE(config.LoadFromString(yaml_string));

  REQUIRE(config.performance_.target_stat_interval_ms_ == 3000);
  REQUIRE(config.performance_.max_concurrent_operations_ == 32);
  REQUIRE(config.targets_.neighborhood_ == 8);
  REQUIRE(config.dpe_.dpe_type_ == "round_robin");
}

TEST_CASE("Config LoadFromString - Empty String", "[cte][config]") {
  Config config;
  REQUIRE_FALSE(config.LoadFromString(""));
}

TEST_CASE("Config LoadFromString - Invalid YAML", "[cte][config]") {
  std::string invalid_yaml = "} } } invalid";
  Config config;
  REQUIRE_FALSE(config.LoadFromString(invalid_yaml));
}

TEST_CASE("Config SaveToFile - Round Trip", "[cte][config]") {
  std::string temp_file = GetTempFile("config_roundtrip");

  Config config;
  config.performance_.target_stat_interval_ms_ = 4000;
  config.performance_.max_concurrent_operations_ = 128;
  config.performance_.score_threshold_ = 0.6f;
  config.performance_.score_difference_threshold_ = 0.15f;

  config.targets_.neighborhood_ = 10;
  config.targets_.default_target_timeout_ms_ = 45000;
  config.targets_.poll_period_ms_ = 4000;

  config.storage_.devices_.push_back(
      StorageDeviceConfig("/tmp/dev1", "ram", 2048ULL * 1024 * 1024));
  config.storage_.devices_.push_back(
      StorageDeviceConfig("/tmp/dev2", "file", 4096ULL * 1024 * 1024, 0.5f));

  config.dpe_.dpe_type_ = "random";

  // Save to file
  REQUIRE(config.SaveToFile(temp_file));

  // Load from file
  Config loaded_config;
  REQUIRE(loaded_config.LoadFromFile(temp_file));

  // Verify values match
  REQUIRE(loaded_config.performance_.target_stat_interval_ms_ == 4000);
  REQUIRE(loaded_config.performance_.max_concurrent_operations_ == 128);
  REQUIRE(loaded_config.targets_.neighborhood_ == 10);
  REQUIRE(loaded_config.storage_.devices_.size() == 2);
  REQUIRE(loaded_config.dpe_.dpe_type_ == "random");

  CleanupTempFile(temp_file);
}

TEST_CASE("Config Validate - Default Config", "[cte][config]") {
  Config config;
  REQUIRE(config.Validate());
}

TEST_CASE("Config Validate - target_stat_interval_ms Zero", "[cte][config]") {
  Config config;
  config.performance_.target_stat_interval_ms_ = 0;
  REQUIRE_FALSE(config.Validate());
}

TEST_CASE("Config Validate - max_concurrent_operations Zero", "[cte][config]") {
  Config config;
  config.performance_.max_concurrent_operations_ = 0;
  REQUIRE_FALSE(config.Validate());
}

TEST_CASE("Config Validate - score_threshold Negative", "[cte][config]") {
  Config config;
  config.performance_.score_threshold_ = -0.1f;
  REQUIRE_FALSE(config.Validate());
}

TEST_CASE("Config Validate - score_threshold Greater Than 1", "[cte][config]") {
  Config config;
  config.performance_.score_threshold_ = 1.5f;
  REQUIRE_FALSE(config.Validate());
}

TEST_CASE("Config Validate - score_difference_threshold Negative", "[cte][config]") {
  Config config;
  config.performance_.score_difference_threshold_ = -0.05f;
  REQUIRE_FALSE(config.Validate());
}

TEST_CASE("Config Validate - neighborhood Zero", "[cte][config]") {
  Config config;
  config.targets_.neighborhood_ = 0;
  REQUIRE_FALSE(config.Validate());
}

TEST_CASE("Config Validate - default_target_timeout_ms Zero", "[cte][config]") {
  Config config;
  config.targets_.default_target_timeout_ms_ = 0;
  REQUIRE_FALSE(config.Validate());
}

TEST_CASE("Config GetParameterString - target_stat_interval_ms", "[cte][config]") {
  Config config;
  config.performance_.target_stat_interval_ms_ = 12345;
  REQUIRE(config.GetParameterString("target_stat_interval_ms") == "12345");
}

TEST_CASE("Config GetParameterString - max_concurrent_operations", "[cte][config]") {
  Config config;
  config.performance_.max_concurrent_operations_ = 256;
  REQUIRE(config.GetParameterString("max_concurrent_operations") == "256");
}

TEST_CASE("Config GetParameterString - score_threshold", "[cte][config]") {
  Config config;
  config.performance_.score_threshold_ = 0.9f;
  std::string result = config.GetParameterString("score_threshold");
  REQUIRE(!result.empty());
}

TEST_CASE("Config GetParameterString - score_difference_threshold", "[cte][config]") {
  Config config;
  config.performance_.score_difference_threshold_ = 0.2f;
  std::string result = config.GetParameterString("score_difference_threshold");
  REQUIRE(!result.empty());
}

TEST_CASE("Config GetParameterString - neighborhood", "[cte][config]") {
  Config config;
  config.targets_.neighborhood_ = 16;
  REQUIRE(config.GetParameterString("neighborhood") == "16");
}

TEST_CASE("Config GetParameterString - default_target_timeout_ms", "[cte][config]") {
  Config config;
  config.targets_.default_target_timeout_ms_ = 50000;
  REQUIRE(config.GetParameterString("default_target_timeout_ms") == "50000");
}

TEST_CASE("Config GetParameterString - poll_period_ms", "[cte][config]") {
  Config config;
  config.targets_.poll_period_ms_ = 3000;
  REQUIRE(config.GetParameterString("poll_period_ms") == "3000");
}

TEST_CASE("Config GetParameterString - Unknown Parameter", "[cte][config]") {
  Config config;
  REQUIRE(config.GetParameterString("unknown_parameter") == "");
}

TEST_CASE("Config SetParameterFromString - target_stat_interval_ms", "[cte][config]") {
  Config config;
  REQUIRE(config.SetParameterFromString("target_stat_interval_ms", "7000"));
  REQUIRE(config.performance_.target_stat_interval_ms_ == 7000);
}

TEST_CASE("Config SetParameterFromString - max_concurrent_operations", "[cte][config]") {
  Config config;
  REQUIRE(config.SetParameterFromString("max_concurrent_operations", "512"));
  REQUIRE(config.performance_.max_concurrent_operations_ == 512);
}

TEST_CASE("Config SetParameterFromString - score_threshold", "[cte][config]") {
  Config config;
  REQUIRE(config.SetParameterFromString("score_threshold", "0.75"));
  REQUIRE(config.performance_.score_threshold_ > 0.74f &&
          config.performance_.score_threshold_ < 0.76f);
}

TEST_CASE("Config SetParameterFromString - neighborhood", "[cte][config]") {
  Config config;
  REQUIRE(config.SetParameterFromString("neighborhood", "20"));
  REQUIRE(config.targets_.neighborhood_ == 20);
}

TEST_CASE("Config SetParameterFromString - Invalid Value", "[cte][config]") {
  Config config;
  REQUIRE_FALSE(config.SetParameterFromString("target_stat_interval_ms", "not_a_number"));
}

TEST_CASE("Config SetParameterFromString - Unknown Parameter", "[cte][config]") {
  Config config;
  REQUIRE_FALSE(config.SetParameterFromString("nonexistent", "123"));
}

TEST_CASE("Config EmitYaml - Valid Roundtrip", "[cte][config]") {
  std::string temp_file = GetTempFile("config_yaml_roundtrip");

  Config config;
  config.performance_.target_stat_interval_ms_ = 9000;
  config.targets_.neighborhood_ = 12;
  config.dpe_.dpe_type_ = "random";

  REQUIRE(config.SaveToFile(temp_file));

  // Verify the file was created and contains valid YAML
  std::ifstream file(temp_file);
  REQUIRE(file.is_open());
  std::string content((std::istreambuf_iterator<char>(file)),
                      std::istreambuf_iterator<char>());
  file.close();

  REQUIRE(!content.empty());
  REQUIRE(content.find("performance") != std::string::npos);
  REQUIRE(content.find("targets") != std::string::npos);

  CleanupTempFile(temp_file);
}

TEST_CASE("Config FormatSizeBytes - Zero Bytes", "[cte][config]") {
  Config config;
  // FormatSizeBytes is private, test it indirectly through SaveToFile
  // Add a device with 0 capacity to trigger FormatSizeBytes(0)
  config.storage_.devices_.push_back(StorageDeviceConfig("/tmp/test_dev", "ram", 0));
  std::string temp_file = GetTempFile("test_format");
  config.SaveToFile(temp_file);
  CleanupTempFile(temp_file);
}

TEST_CASE("Config FormatSizeBytes - Kilobytes", "[cte][config]") {
  std::string yaml_string = R"(
storage:
  - path: /tmp/dev_kb
    bdev_type: ram
    capacity_limit: 1024
)";

  Config config;
  REQUIRE(config.LoadFromString(yaml_string));
  REQUIRE(config.storage_.devices_[0].capacity_limit_ == 1024);
}

TEST_CASE("Config FormatSizeBytes - Megabytes", "[cte][config]") {
  std::string yaml_string = R"(
storage:
  - path: /tmp/dev_mb
    bdev_type: ram
    capacity_limit: 256MB
)";

  Config config;
  REQUIRE(config.LoadFromString(yaml_string));
  REQUIRE(config.storage_.devices_[0].capacity_limit_ == 256ULL * 1024 * 1024);
}

TEST_CASE("Config ParseSizeString - Plain Bytes", "[cte][config]") {
  std::string yaml_string = R"(
storage:
  - path: /tmp/dev_bytes
    bdev_type: ram
    capacity_limit: 1000
)";

  Config config;
  REQUIRE(config.LoadFromString(yaml_string));
  REQUIRE(config.storage_.devices_[0].capacity_limit_ == 1000);
}

TEST_CASE("Config ParseSizeString - Kilobytes", "[cte][config]") {
  std::string yaml_string = R"(
storage:
  - path: /tmp/dev_kb
    bdev_type: ram
    capacity_limit: 1KB
)";

  Config config;
  REQUIRE(config.LoadFromString(yaml_string));
  REQUIRE(config.storage_.devices_[0].capacity_limit_ == 1024);
}

TEST_CASE("Config ParseSizeString - Megabytes", "[cte][config]") {
  std::string yaml_string = R"(
storage:
  - path: /tmp/dev_mb
    bdev_type: ram
    capacity_limit: 512MB
)";

  Config config;
  REQUIRE(config.LoadFromString(yaml_string));
  REQUIRE(config.storage_.devices_[0].capacity_limit_ == 512ULL * 1024 * 1024);
}

TEST_CASE("Config ParseSizeString - Gigabytes", "[cte][config]") {
  std::string yaml_string = R"(
storage:
  - path: /tmp/dev_gb
    bdev_type: ram
    capacity_limit: 4GB
)";

  Config config;
  REQUIRE(config.LoadFromString(yaml_string));
  REQUIRE(config.storage_.devices_[0].capacity_limit_ == 4ULL * 1024 * 1024 * 1024);
}

TEST_CASE("Config ParseSizeString - Terabytes", "[cte][config]") {
  std::string yaml_string = R"(
storage:
  - path: /tmp/dev_tb
    bdev_type: ram
    capacity_limit: 1TB
)";

  Config config;
  REQUIRE(config.LoadFromString(yaml_string));
  REQUIRE(config.storage_.devices_[0].capacity_limit_ == 1ULL * 1024 * 1024 * 1024 * 1024);
}

TEST_CASE("Config ParseSizeString - Decimal Values", "[cte][config]") {
  std::string yaml_string = R"(
storage:
  - path: /tmp/dev_decimal
    bdev_type: ram
    capacity_limit: 1.5GB
)";

  Config config;
  REQUIRE(config.LoadFromString(yaml_string));
  chi::u64 expected = static_cast<chi::u64>(1.5 * 1024 * 1024 * 1024);
  REQUIRE(config.storage_.devices_[0].capacity_limit_ == expected);
}

// ============================================================================
// DPE Tests
// ============================================================================

TEST_CASE("DpeTypeToString - kRandom", "[cte][dpe]") {
  std::string result = DpeTypeToString(DpeType::kRandom);
  REQUIRE(result == "random");
}

TEST_CASE("DpeTypeToString - kRoundRobin", "[cte][dpe]") {
  std::string result = DpeTypeToString(DpeType::kRoundRobin);
  REQUIRE(result == "round_robin");
}

TEST_CASE("DpeTypeToString - kMaxBW", "[cte][dpe]") {
  std::string result = DpeTypeToString(DpeType::kMaxBW);
  REQUIRE(result == "max_bw");
}

TEST_CASE("StringToDpeType - random", "[cte][dpe]") {
  DpeType type = StringToDpeType("random");
  REQUIRE(type == DpeType::kRandom);
}

TEST_CASE("StringToDpeType - round_robin", "[cte][dpe]") {
  DpeType type = StringToDpeType("round_robin");
  REQUIRE(type == DpeType::kRoundRobin);
}

TEST_CASE("StringToDpeType - roundrobin (no underscore)", "[cte][dpe]") {
  DpeType type = StringToDpeType("roundrobin");
  REQUIRE(type == DpeType::kRoundRobin);
}

TEST_CASE("StringToDpeType - max_bw", "[cte][dpe]") {
  DpeType type = StringToDpeType("max_bw");
  REQUIRE(type == DpeType::kMaxBW);
}

TEST_CASE("StringToDpeType - maxbw (no underscore)", "[cte][dpe]") {
  DpeType type = StringToDpeType("maxbw");
  REQUIRE(type == DpeType::kMaxBW);
}

TEST_CASE("StringToDpeType - Unknown String", "[cte][dpe]") {
  DpeType type = StringToDpeType("unknown_dpe");
  // Should default to kRandom
  REQUIRE(type == DpeType::kRandom);
}

TEST_CASE("RandomDpe SelectTargets - Empty Targets", "[cte][dpe]") {
  RandomDpe dpe;
  std::vector<TargetInfo> targets;
  std::vector<TargetInfo> result = dpe.SelectTargets(targets, 0.5f, 1024);
  REQUIRE(result.empty());
}

TEST_CASE("RandomDpe SelectTargets - No Available Space", "[cte][dpe]") {
  RandomDpe dpe;
  std::vector<TargetInfo> targets;

  TargetInfo target1;
  target1.remaining_space_ = 512;  // Less than data_size
  targets.push_back(target1);

  std::vector<TargetInfo> result = dpe.SelectTargets(targets, 0.5f, 1024);
  REQUIRE(result.empty());
}

TEST_CASE("RandomDpe SelectTargets - With Available Targets", "[cte][dpe]") {
  RandomDpe dpe;
  std::vector<TargetInfo> targets;

  for (int i = 0; i < 5; i++) {
    TargetInfo target;
    target.remaining_space_ = 2048;
    target.target_score_ = 0.5f;
    targets.push_back(target);
  }

  std::vector<TargetInfo> result = dpe.SelectTargets(targets, 0.5f, 1024);
  REQUIRE(result.size() == 5);
  REQUIRE(dpe.GetType() == DpeType::kRandom);
}

TEST_CASE("RoundRobinDpe SelectTargets - Empty Targets", "[cte][dpe]") {
  RoundRobinDpe dpe;
  std::vector<TargetInfo> targets;
  std::vector<TargetInfo> result = dpe.SelectTargets(targets, 0.5f, 1024);
  REQUIRE(result.empty());
}

TEST_CASE("RoundRobinDpe SelectTargets - No Available Space", "[cte][dpe]") {
  RoundRobinDpe dpe;
  std::vector<TargetInfo> targets;

  TargetInfo target1;
  target1.remaining_space_ = 256;
  targets.push_back(target1);

  std::vector<TargetInfo> result = dpe.SelectTargets(targets, 0.5f, 1024);
  REQUIRE(result.empty());
}

TEST_CASE("RoundRobinDpe SelectTargets - Rotation Behavior", "[cte][dpe]") {
  RoundRobinDpe dpe;
  std::vector<TargetInfo> targets;

  for (int i = 0; i < 3; i++) {
    TargetInfo target;
    target.remaining_space_ = 2048;
    target.target_score_ = static_cast<float>(i) * 0.1f;
    targets.push_back(target);
  }

  // Call SelectTargets multiple times to test rotation
  auto result1 = dpe.SelectTargets(targets, 0.5f, 1024);
  REQUIRE(result1.size() == 3);

  auto result2 = dpe.SelectTargets(targets, 0.5f, 1024);
  REQUIRE(result2.size() == 3);

  REQUIRE(dpe.GetType() == DpeType::kRoundRobin);
}

TEST_CASE("MaxBwDpe SelectTargets - Empty Targets", "[cte][dpe]") {
  MaxBwDpe dpe;
  std::vector<TargetInfo> targets;
  std::vector<TargetInfo> result = dpe.SelectTargets(targets, 0.5f, 1024);
  REQUIRE(result.empty());
}

TEST_CASE("MaxBwDpe SelectTargets - No Available Space", "[cte][dpe]") {
  MaxBwDpe dpe;
  std::vector<TargetInfo> targets;

  TargetInfo target;
  target.remaining_space_ = 100;
  targets.push_back(target);

  std::vector<TargetInfo> result = dpe.SelectTargets(targets, 0.5f, 1024);
  REQUIRE(result.empty());
}

TEST_CASE("MaxBwDpe SelectTargets - Large Data Size", "[cte][dpe]") {
  MaxBwDpe dpe;
  std::vector<TargetInfo> targets;

  for (int i = 0; i < 3; i++) {
    TargetInfo target;
    target.remaining_space_ = 1ULL * 1024 * 1024 * 1024;  // 1GB
    target.target_score_ = 0.3f + i * 0.1f;
    target.perf_metrics_.write_bandwidth_mbps_ = 100.0 + i * 10.0;
    target.perf_metrics_.read_bandwidth_mbps_ = 80.0 + i * 10.0;
    targets.push_back(target);
  }

  // Large data size should sort by bandwidth
  std::vector<TargetInfo> result = dpe.SelectTargets(targets, 0.5f, 64 * 1024);
  REQUIRE(result.size() > 0);
  REQUIRE(dpe.GetType() == DpeType::kMaxBW);
}

TEST_CASE("MaxBwDpe SelectTargets - Small Data Size", "[cte][dpe]") {
  MaxBwDpe dpe;
  std::vector<TargetInfo> targets;

  for (int i = 0; i < 3; i++) {
    TargetInfo target;
    target.remaining_space_ = 1ULL * 1024 * 1024 * 1024;
    target.target_score_ = 0.3f + i * 0.1f;
    target.perf_metrics_.read_latency_us_ = 100.0 - i * 10.0;
    target.perf_metrics_.write_latency_us_ = 120.0 - i * 10.0;
    targets.push_back(target);
  }

  // Small data size should sort by latency
  std::vector<TargetInfo> result = dpe.SelectTargets(targets, 0.5f, 16 * 1024);
  REQUIRE(result.size() > 0);
}

TEST_CASE("MaxBwDpe SelectTargets - Score Filtering", "[cte][dpe]") {
  MaxBwDpe dpe;
  std::vector<TargetInfo> targets;

  // Create targets with different scores
  for (int i = 0; i < 3; i++) {
    TargetInfo target;
    target.remaining_space_ = 1ULL * 1024 * 1024 * 1024;
    target.target_score_ = 0.2f + i * 0.15f;  // 0.2, 0.35, 0.5
    target.perf_metrics_.write_bandwidth_mbps_ = 100.0 + i * 10.0;
    targets.push_back(target);
  }

  // Blob score is 0.3, should filter targets with score > 0.3
  std::vector<TargetInfo> result = dpe.SelectTargets(targets, 0.3f, 64 * 1024);
  REQUIRE(result.size() > 0);
}

TEST_CASE("DpeFactory CreateDpe - By Enum kRandom", "[cte][dpe]") {
  auto dpe = DpeFactory::CreateDpe(DpeType::kRandom);
  REQUIRE(dpe != nullptr);
  REQUIRE(dpe->GetType() == DpeType::kRandom);
}

TEST_CASE("DpeFactory CreateDpe - By Enum kRoundRobin", "[cte][dpe]") {
  auto dpe = DpeFactory::CreateDpe(DpeType::kRoundRobin);
  REQUIRE(dpe != nullptr);
  REQUIRE(dpe->GetType() == DpeType::kRoundRobin);
}

TEST_CASE("DpeFactory CreateDpe - By Enum kMaxBW", "[cte][dpe]") {
  auto dpe = DpeFactory::CreateDpe(DpeType::kMaxBW);
  REQUIRE(dpe != nullptr);
  REQUIRE(dpe->GetType() == DpeType::kMaxBW);
}

TEST_CASE("DpeFactory CreateDpe - By String random", "[cte][dpe]") {
  auto dpe = DpeFactory::CreateDpe("random");
  REQUIRE(dpe != nullptr);
  REQUIRE(dpe->GetType() == DpeType::kRandom);
}

TEST_CASE("DpeFactory CreateDpe - By String round_robin", "[cte][dpe]") {
  auto dpe = DpeFactory::CreateDpe("round_robin");
  REQUIRE(dpe != nullptr);
  REQUIRE(dpe->GetType() == DpeType::kRoundRobin);
}

TEST_CASE("DpeFactory CreateDpe - By String max_bw", "[cte][dpe]") {
  auto dpe = DpeFactory::CreateDpe("max_bw");
  REQUIRE(dpe != nullptr);
  REQUIRE(dpe->GetType() == DpeType::kMaxBW);
}

TEST_CASE("DpeFactory CreateDpe - By String roundrobin (no underscore)", "[cte][dpe]") {
  auto dpe = DpeFactory::CreateDpe("roundrobin");
  REQUIRE(dpe != nullptr);
  REQUIRE(dpe->GetType() == DpeType::kRoundRobin);
}

TEST_CASE("DpeFactory CreateDpe - By String maxbw (no underscore)", "[cte][dpe]") {
  auto dpe = DpeFactory::CreateDpe("maxbw");
  REQUIRE(dpe != nullptr);
  REQUIRE(dpe->GetType() == DpeType::kMaxBW);
}

TEST_CASE("DpeFactory CreateDpe - Invalid String", "[cte][dpe]") {
  auto dpe = DpeFactory::CreateDpe("invalid_type");
  // Should default to RandomDpe
  REQUIRE(dpe != nullptr);
  REQUIRE(dpe->GetType() == DpeType::kRandom);
}

SIMPLE_TEST_MAIN()
