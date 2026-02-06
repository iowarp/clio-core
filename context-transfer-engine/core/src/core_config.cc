#include <wrp_cte/core/core_config.h>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <cctype>
#include <cstdio>
#include "hermes_shm/util/logging.h"

namespace wrp_cte::core {

// Config class implementation
bool Config::LoadFromFile(const std::string &config_file_path) {
  try {
    if (config_file_path.empty()) {
      HLOG(kError, "Config error: Empty config file path provided");
      return false;
    }
    
    // Check if file exists
    std::ifstream file_test(config_file_path);
    if (!file_test.good()) {
      HLOG(kError, "Config error: Cannot open config file: {}", config_file_path);
      return false;
    }
    file_test.close();
    
    // Load and parse YAML
    YAML::Node root = YAML::LoadFile(config_file_path);
    
    // Parse configuration using base class method
    if (!ParseYamlNode(root)) {
      HLOG(kError, "Config error: Failed to parse YAML configuration");
      return false;
    }
    
    // Validate configuration
    if (!Validate()) {
      HLOG(kError, "Config error: Configuration validation failed");
      return false;
    }
    
    HLOG(kInfo, "Configuration loaded successfully from: {}", config_file_path);
    return true;
    
  } catch (const YAML::Exception &e) {
    HLOG(kError, "YAML parsing error: {}", e.what());
    return false;
  } catch (const std::exception &e) {
    HLOG(kError, "Config loading error: {}", e.what());
    return false;
  }
}

bool Config::LoadFromString(const std::string &yaml_string) {
  try {
    if (yaml_string.empty()) {
      HLOG(kError, "Config error: Empty YAML string provided");
      return false;
    }

    // Load and parse YAML from string
    YAML::Node root = YAML::Load(yaml_string);

    // Parse configuration using base class method
    if (!ParseYamlNode(root)) {
      HLOG(kError, "Config error: Failed to parse YAML configuration from string");
      return false;
    }

    // Validate configuration
    if (!Validate()) {
      HLOG(kError, "Config error: Configuration validation failed");
      return false;
    }

    HLOG(kInfo, "Configuration loaded successfully from YAML string");
    return true;

  } catch (const YAML::Exception &e) {
    HLOG(kError, "YAML parsing error: {}", e.what());
    return false;
  } catch (const std::exception &e) {
    HLOG(kError, "Config loading error: {}", e.what());
    return false;
  }
}

bool Config::LoadFromEnvironment() {
  std::string env_path = hshm::SystemInfo::Getenv(config_env_var_);
  if (env_path.empty()) {
    HLOG(kInfo, "Config info: Environment variable {} not set, using default configuration", config_env_var_);
    return true; // Not an error, use defaults
  }

  return LoadFromFile(env_path);
}

bool Config::SaveToFile(const std::string &config_file_path) const {
  try {
    std::ofstream file(config_file_path);
    if (!file.is_open()) {
      HLOG(kError, "Config error: Cannot create config file: {}", config_file_path);
      return false;
    }
    
    YAML::Emitter emitter;
    EmitYaml(emitter);
    file << emitter.c_str();
    file.close();
    
    HLOG(kInfo, "Configuration saved to: {}", config_file_path);
    return true;
    
  } catch (const std::exception &e) {
    HLOG(kError, "Config save error: {}", e.what());
    return false;
  }
}

bool Config::Validate() const {
  // Validate performance configuration
  if (performance_.target_stat_interval_ms_ == 0 || performance_.target_stat_interval_ms_ > 60000) {
    HLOG(kError, "Config validation error: Invalid target_stat_interval_ms {} (must be 1-60000)", performance_.target_stat_interval_ms_);
    return false;
  }
  
  if (performance_.max_concurrent_operations_ == 0 || performance_.max_concurrent_operations_ > 1024) {
    HLOG(kError, "Config validation error: Invalid max_concurrent_operations {} (must be 1-1024)", performance_.max_concurrent_operations_);
    return false;
  }
  
  if (performance_.score_threshold_ < 0.0f || performance_.score_threshold_ > 1.0f) {
    HLOG(kError, "Config validation error: Invalid score_threshold {} (must be 0.0-1.0)", performance_.score_threshold_);
    return false;
  }
  
  if (performance_.score_difference_threshold_ < 0.0f || performance_.score_difference_threshold_ > 1.0f) {
    HLOG(kError, "Config validation error: Invalid score_difference_threshold {} (must be 0.0-1.0)", performance_.score_difference_threshold_);
    return false;
  }

  // Validate target configuration
  if (targets_.neighborhood_ == 0 || targets_.neighborhood_ > 1024) {
    HLOG(kError, "Config validation error: Invalid neighborhood {} (must be 1-1024)", targets_.neighborhood_);
    return false;
  }
  
  if (targets_.default_target_timeout_ms_ == 0 || targets_.default_target_timeout_ms_ > 300000) {
    HLOG(kError, "Config validation error: Invalid default_target_timeout_ms {} (must be 1-300000)", targets_.default_target_timeout_ms_);
    return false;
  }
  
  return true;
}

std::string Config::GetParameterString(const std::string &param_name) const {
  if (param_name == "target_stat_interval_ms") {
    return std::to_string(performance_.target_stat_interval_ms_);
  }
  if (param_name == "max_concurrent_operations") {
    return std::to_string(performance_.max_concurrent_operations_);
  }
  if (param_name == "score_threshold") {
    return std::to_string(performance_.score_threshold_);
  }
  if (param_name == "score_difference_threshold") {
    return std::to_string(performance_.score_difference_threshold_);
  }
  if (param_name == "neighborhood") {
    return std::to_string(targets_.neighborhood_);
  }
  if (param_name == "default_target_timeout_ms") {
    return std::to_string(targets_.default_target_timeout_ms_);
  }
  if (param_name == "poll_period_ms") {
    return std::to_string(targets_.poll_period_ms_);
  }
  if (param_name == "monitor_interval_ms") {
    return std::to_string(compression_.monitor_interval_ms_);
  }
  if (param_name == "dnn_model_weights_path") {
    return compression_.dnn_model_weights_path_;
  }
  if (param_name == "dnn_samples_before_reinforce") {
    return std::to_string(compression_.dnn_samples_before_reinforce_);
  }
  if (param_name == "trace_folder_path") {
    return compression_.trace_folder_path_;
  }

  return ""; // Parameter not found
}

bool Config::SetParameterFromString(const std::string &param_name,
                                    const std::string &value) {
  try {
    if (param_name == "target_stat_interval_ms") {
      performance_.target_stat_interval_ms_ = static_cast<chi::u32>(std::stoul(value));
      return true;
    }
    if (param_name == "max_concurrent_operations") {
      performance_.max_concurrent_operations_ = static_cast<chi::u32>(std::stoul(value));
      return true;
    }
    if (param_name == "score_threshold") {
      performance_.score_threshold_ = std::stof(value);
      return true;
    }
    if (param_name == "score_difference_threshold") {
      performance_.score_difference_threshold_ = std::stof(value);
      return true;
    }
    if (param_name == "neighborhood") {
      targets_.neighborhood_ = static_cast<chi::u32>(std::stoul(value));
      return true;
    }
    if (param_name == "default_target_timeout_ms") {
      targets_.default_target_timeout_ms_ = static_cast<chi::u32>(std::stoul(value));
      return true;
    }
    if (param_name == "poll_period_ms") {
      targets_.poll_period_ms_ = static_cast<chi::u32>(std::stoul(value));
      return true;
    }
    if (param_name == "monitor_interval_ms") {
      compression_.monitor_interval_ms_ = static_cast<chi::u32>(std::stoul(value));
      return true;
    }
    if (param_name == "dnn_model_weights_path") {
      compression_.dnn_model_weights_path_ = value;
      return true;
    }
    if (param_name == "dnn_samples_before_reinforce") {
      compression_.dnn_samples_before_reinforce_ = static_cast<chi::u32>(std::stoul(value));
      return true;
    }
    if (param_name == "trace_folder_path") {
      compression_.trace_folder_path_ = value;
      return true;
    }

    return false; // Parameter not found
    
  } catch (const std::exception &e) {
    HLOG(kError, "Config error: Invalid value '{}' for parameter '{}': {}", value, param_name, e.what());
    return false;
  }
}

bool Config::ParseYamlNode(const YAML::Node &node) {
  // Parse performance configuration
  if (node["performance"]) {
    if (!ParsePerformanceConfig(node["performance"])) {
      return false;
    }
  }
  
  // Parse target configuration
  if (node["targets"]) {
    if (!ParseTargetConfig(node["targets"])) {
      return false;
    }
  }
  
  // Parse storage configuration
  if (node["storage"]) {
    if (!ParseStorageConfig(node["storage"])) {
      return false;
    }
  }
  
  // Parse DPE configuration
  if (node["dpe"]) {
    if (!ParseDpeConfig(node["dpe"])) {
      return false;
    }
  }

  // Parse compression configuration
  if (node["compression"]) {
    if (!ParseCompressionConfig(node["compression"])) {
      return false;
    }
  }

  // Parse environment variable configuration
  if (node["config_env_var"]) {
    config_env_var_ = node["config_env_var"].as<std::string>();
  }

  return true;
}

void Config::EmitYaml(YAML::Emitter &emitter) const {
  emitter << YAML::BeginMap;

  // Emit general configuration
  emitter << YAML::Key << "config_env_var" << YAML::Value << config_env_var_.c_str();

  // Emit performance configuration
  emitter << YAML::Key << "performance" << YAML::Value << YAML::BeginMap;
  emitter << YAML::Key << "target_stat_interval_ms" << YAML::Value << performance_.target_stat_interval_ms_;
  emitter << YAML::Key << "max_concurrent_operations" << YAML::Value << performance_.max_concurrent_operations_;
  emitter << YAML::Key << "score_threshold" << YAML::Value << performance_.score_threshold_;
  emitter << YAML::Key << "score_difference_threshold" << YAML::Value << performance_.score_difference_threshold_;
  emitter << YAML::EndMap;

  // Emit target configuration
  emitter << YAML::Key << "targets" << YAML::Value << YAML::BeginMap;
  emitter << YAML::Key << "neighborhood" << YAML::Value << targets_.neighborhood_;
  emitter << YAML::Key << "default_target_timeout_ms" << YAML::Value << targets_.default_target_timeout_ms_;
  emitter << YAML::Key << "poll_period_ms" << YAML::Value << targets_.poll_period_ms_;
  emitter << YAML::EndMap;
  
  // Emit storage configuration
  if (!storage_.devices_.empty()) {
    emitter << YAML::Key << "storage" << YAML::Value << YAML::BeginSeq;
    for (const auto& device : storage_.devices_) {
      emitter << YAML::BeginMap;
      emitter << YAML::Key << "path" << YAML::Value << device.path_;
      emitter << YAML::Key << "bdev_type" << YAML::Value << device.bdev_type_;
      emitter << YAML::Key << "capacity_limit" << YAML::Value << FormatSizeBytes(device.capacity_limit_);
      
      // Emit score only if it's manually set (not using automatic scoring)
      if (device.score_ >= 0.0f) {
        emitter << YAML::Key << "score" << YAML::Value << device.score_;
      }
      
      emitter << YAML::EndMap;
    }
    emitter << YAML::EndSeq;
  }
  
  // Emit DPE configuration
  emitter << YAML::Key << "dpe" << YAML::Value << YAML::BeginMap;
  emitter << YAML::Key << "dpe_type" << YAML::Value << dpe_.dpe_type_;
  emitter << YAML::EndMap;

  // Emit compression configuration
  emitter << YAML::Key << "compression" << YAML::Value << YAML::BeginMap;
  emitter << YAML::Key << "monitor_interval_ms" << YAML::Value << compression_.monitor_interval_ms_;
  emitter << YAML::Key << "dnn_model_weights_path" << YAML::Value << compression_.dnn_model_weights_path_;
  emitter << YAML::Key << "dnn_samples_before_reinforce" << YAML::Value << compression_.dnn_samples_before_reinforce_;
  emitter << YAML::Key << "trace_folder_path" << YAML::Value << compression_.trace_folder_path_;
  emitter << YAML::EndMap;

  emitter << YAML::EndMap;
}

bool Config::ParsePerformanceConfig(const YAML::Node &node) {
  if (node["target_stat_interval_ms"]) {
    performance_.target_stat_interval_ms_ = node["target_stat_interval_ms"].as<chi::u32>();
  }

  if (node["max_concurrent_operations"]) {
    performance_.max_concurrent_operations_ = node["max_concurrent_operations"].as<chi::u32>();
  }

  if (node["score_threshold"]) {
    performance_.score_threshold_ = node["score_threshold"].as<float>();
  }

  if (node["score_difference_threshold"]) {
    performance_.score_difference_threshold_ = node["score_difference_threshold"].as<float>();
  }

  return true;
}

bool Config::ParseTargetConfig(const YAML::Node &node) {
  if (node["neighborhood"]) {
    targets_.neighborhood_ = node["neighborhood"].as<chi::u32>();
  }

  if (node["default_target_timeout_ms"]) {
    targets_.default_target_timeout_ms_ = node["default_target_timeout_ms"].as<chi::u32>();
  }

  if (node["poll_period_ms"]) {
    targets_.poll_period_ms_ = node["poll_period_ms"].as<chi::u32>();
  }

  return true;
}

bool Config::ParseStorageConfig(const YAML::Node &node) {
  if (!node.IsSequence()) {
    HLOG(kError, "Config error: Storage configuration must be a sequence");
    return false;
  }
  
  // Clear existing storage configuration
  storage_.devices_.clear();
  
  for (const auto& device_node : node) {
    StorageDeviceConfig device_config;
    
    // Parse path (required)
    if (!device_node["path"]) {
      HLOG(kError, "Config error: Storage device missing required 'path' field");
      return false;
    }
    std::string path = device_node["path"].as<std::string>();
    device_config.path_ = hshm::ConfigParse::ExpandPath(path);
    
    // Parse bdev_type (required)
    if (!device_node["bdev_type"]) {
      HLOG(kError, "Config error: Storage device missing required 'bdev_type' field");
      return false;
    }
    device_config.bdev_type_ = device_node["bdev_type"].as<std::string>();
    
    // Validate bdev_type
    if (device_config.bdev_type_ != "file" && device_config.bdev_type_ != "ram") {
      HLOG(kError, "Config error: Invalid bdev_type '{}' (must be 'file' or 'ram')", device_config.bdev_type_);
      return false;
    }
    
    // Parse capacity_limit (required)
    if (!device_node["capacity_limit"]) {
      HLOG(kError, "Config error: Storage device missing required 'capacity_limit' field");
      return false;
    }
    std::string capacity_str = device_node["capacity_limit"].as<std::string>();
    
    // Parse size string to bytes
    if (!ParseSizeString(capacity_str, device_config.capacity_limit_)) {
      HLOG(kError, "Config error: Invalid capacity_limit format '{}' for device {}", capacity_str, device_config.path_);
      return false;
    }
    
    // Parse score (optional)
    if (device_node["score"]) {
      device_config.score_ = device_node["score"].as<float>();
      
      // Validate score range
      if (device_config.score_ < 0.0f || device_config.score_ > 1.0f) {
        HLOG(kError, "Config error: Storage device score {} must be between 0.0 and 1.0 for device {}", 
              device_config.score_, device_config.path_);
        return false;
      }
    }
    // score_ defaults to -1.0f (use automatic scoring) if not specified
    
    // Validate parsed values
    if (device_config.path_.empty()) {
      HLOG(kError, "Config error: Storage device path cannot be empty");
      return false;
    }
    
    if (device_config.capacity_limit_ == 0) {
      HLOG(kError, "Config error: Storage device capacity_limit must be greater than 0");
      return false;
    }
    
    storage_.devices_.push_back(std::move(device_config));
  }
  
  HLOG(kInfo, "Parsed {} storage devices", storage_.devices_.size());
  return true;
}

bool Config::ParseDpeConfig(const YAML::Node &node) {
  if (node["dpe_type"]) {
    std::string dpe_type = node["dpe_type"].as<std::string>();

    // Validate DPE type
    if (dpe_type != "random" && dpe_type != "round_robin" &&
        dpe_type != "roundrobin" && dpe_type != "max_bw" && dpe_type != "maxbw") {
      HLOG(kError, "Config error: Invalid dpe_type '{}' (must be 'random', 'round_robin', or 'max_bw')", dpe_type);
      return false;
    }

    dpe_.dpe_type_ = dpe_type;
  }

  HLOG(kInfo, "Parsed DPE configuration: type={}", dpe_.dpe_type_);
  return true;
}

bool Config::ParseCompressionConfig(const YAML::Node &node) {
  if (node["monitor_interval_ms"]) {
    compression_.monitor_interval_ms_ = node["monitor_interval_ms"].as<chi::u32>();
  }

  if (node["qtable_model_path"]) {
    std::string path = node["qtable_model_path"].as<std::string>();
    compression_.qtable_model_path_ = hshm::ConfigParse::ExpandPath(path);
  }

  if (node["qtable_learning_rate"]) {
    compression_.qtable_learning_rate_ = node["qtable_learning_rate"].as<float>();
  }

  if (node["dnn_model_weights_path"]) {
    std::string path = node["dnn_model_weights_path"].as<std::string>();
    compression_.dnn_model_weights_path_ = hshm::ConfigParse::ExpandPath(path);
  }

  if (node["dnn_samples_before_reinforce"]) {
    compression_.dnn_samples_before_reinforce_ = node["dnn_samples_before_reinforce"].as<chi::u32>();
  }

  if (node["trace_folder_path"]) {
    std::string path = node["trace_folder_path"].as<std::string>();
    compression_.trace_folder_path_ = hshm::ConfigParse::ExpandPath(path);
  }

  HLOG(kInfo, "Parsed compression configuration: monitor_interval={}ms, qtable_model='{}', "
       "dnn_weights='{}', samples={}, trace_folder='{}'",
       compression_.monitor_interval_ms_,
       compression_.qtable_model_path_,
       compression_.dnn_model_weights_path_,
       compression_.dnn_samples_before_reinforce_,
       compression_.trace_folder_path_);
  return true;
}

bool Config::ParseSizeString(const std::string &size_str, chi::u64 &size_bytes) const {
  if (size_str.empty()) {
    return false;
  }
  
  // Extract numeric part and suffix
  std::string number_part;
  std::string suffix_part;
  
  size_t i = 0;
  // Extract digits and decimal point
  while (i < size_str.length() && 
         (std::isdigit(size_str[i]) || size_str[i] == '.')) {
    number_part += size_str[i];
    ++i;
  }
  
  // Extract suffix (skip whitespace)
  while (i < size_str.length() && std::isspace(size_str[i])) {
    ++i;
  }
  while (i < size_str.length()) {
    suffix_part += std::tolower(size_str[i]);
    ++i;
  }
  
  if (number_part.empty()) {
    return false;
  }
  
  // Parse the numeric value
  double value;
  try {
    value = std::stod(number_part);
  } catch (const std::exception &) {
    return false;
  }
  
  if (value < 0) {
    return false;
  }
  
  // Parse the suffix and convert to bytes
  chi::u64 multiplier = 1;
  if (suffix_part.empty() || suffix_part == "b" || suffix_part == "bytes") {
    multiplier = 1;
  } else if (suffix_part == "k" || suffix_part == "kb" || suffix_part == "kilobytes") {
    multiplier = 1024ULL;
  } else if (suffix_part == "m" || suffix_part == "mb" || suffix_part == "megabytes") {
    multiplier = 1024ULL * 1024ULL;
  } else if (suffix_part == "g" || suffix_part == "gb" || suffix_part == "gigabytes") {
    multiplier = 1024ULL * 1024ULL * 1024ULL;
  } else if (suffix_part == "t" || suffix_part == "tb" || suffix_part == "terabytes") {
    multiplier = 1024ULL * 1024ULL * 1024ULL * 1024ULL;
  } else if (suffix_part == "p" || suffix_part == "pb" || suffix_part == "petabytes") {
    multiplier = 1024ULL * 1024ULL * 1024ULL * 1024ULL * 1024ULL;
  } else {
    // Unknown suffix
    return false;
  }
  
  // Calculate final size in bytes
  size_bytes = static_cast<chi::u64>(value * multiplier);
  return true;
}

std::string Config::FormatSizeBytes(chi::u64 size_bytes) const {
  const char* units[] = {"B", "KB", "MB", "GB", "TB", "PB"};
  const chi::u64 base = 1024;
  
  if (size_bytes == 0) {
    return "0B";
  }
  
  // Find the appropriate unit
  size_t unit_index = 0;
  double size = static_cast<double>(size_bytes);
  
  while (size >= base && unit_index < 5) {
    size /= base;
    unit_index++;
  }
  
  // Format the size
  if (size == static_cast<chi::u64>(size)) {
    // No decimal places needed
    return std::to_string(static_cast<chi::u64>(size)) + units[unit_index];
  } else {
    // Use one decimal place
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.1f%s", size, units[unit_index]);
    return std::string(buffer);
  }
}

}  // namespace wrp_cte::core