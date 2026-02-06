/**
 * @file linreg_table_predictor.cc
 * @brief Implementation of linear regression table predictor
 */

#include "wrp_cte/compressor/models/linreg_table_predictor.h"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <ctime>
#include <set>

// JSON parsing using simple manual parsing (no external dependency)
namespace {

/**
 * @brief Skip whitespace in string
 * @param s Input string
 * @param pos Current position (updated)
 */
void SkipWhitespace(const std::string& s, size_t& pos) {
  while (pos < s.length() && std::isspace(s[pos])) {
    ++pos;
  }
}

/**
 * @brief Parse a string value from JSON
 * @param s Input string
 * @param pos Current position (updated)
 * @return Parsed string value
 */
std::string ParseJsonString(const std::string& s, size_t& pos) {
  SkipWhitespace(s, pos);
  if (s[pos] != '"') return "";
  ++pos;
  std::string result;
  while (pos < s.length() && s[pos] != '"') {
    if (s[pos] == '\\' && pos + 1 < s.length()) {
      ++pos;
    }
    result += s[pos++];
  }
  if (pos < s.length()) ++pos;  // Skip closing quote
  return result;
}

/**
 * @brief Parse a number from JSON
 * @param s Input string
 * @param pos Current position (updated)
 * @return Parsed double value
 */
double ParseJsonNumber(const std::string& s, size_t& pos) {
  SkipWhitespace(s, pos);
  std::string num_str;
  while (pos < s.length() &&
         (std::isdigit(s[pos]) || s[pos] == '.' || s[pos] == '-' ||
          s[pos] == '+' || s[pos] == 'e' || s[pos] == 'E')) {
    num_str += s[pos++];
  }
  return num_str.empty() ? 0.0 : std::stod(num_str);
}

/**
 * @brief Find key in JSON object
 * @param s Input string
 * @param key Key to find
 * @param start Start position
 * @return Position after the key's colon, or std::string::npos
 */
size_t FindJsonKey(const std::string& s, const std::string& key, size_t start = 0) {
  std::string search = "\"" + key + "\"";
  size_t pos = s.find(search, start);
  if (pos == std::string::npos) return std::string::npos;
  pos += search.length();
  SkipWhitespace(s, pos);
  if (pos < s.length() && s[pos] == ':') ++pos;
  SkipWhitespace(s, pos);
  return pos;
}

}  // anonymous namespace

namespace wrp_cte::compressor {

LinRegTablePredictor::LinRegTablePredictor()
    : config_(), ready_(false) {
  // Initialize library config ID decoder for common libraries
  // Format: base_id * 10 + preset_id (from CompressionFactory)
  // Lossless: BZIP2=1, ZSTD=2, LZ4=3, ZLIB=4, LZMA=5, BROTLI=6, SNAPPY=7, BLOSC2=8
  std::vector<std::string> libs = {"BZIP2", "ZSTD", "LZ4", "ZLIB", "LZMA", "BROTLI", "SNAPPY", "Blosc2"};
  std::vector<std::string> presets = {"fast", "balanced", "best", "default"};

  for (size_t i = 0; i < libs.size(); ++i) {
    int base_id = static_cast<int>(i + 1);
    for (size_t j = 0; j < presets.size(); ++j) {
      int preset_id = static_cast<int>(j);
      int lib_config_id = base_id * 10 + preset_id;
      id_to_lib_config_[lib_config_id] = {libs[i], presets[j]};
    }
  }
}

LinRegTablePredictor::LinRegTablePredictor(const LinRegTableConfig& config)
    : config_(config), ready_(false) {
  // Same initialization as default constructor
  std::vector<std::string> libs = {"BZIP2", "ZSTD", "LZ4", "ZLIB", "LZMA", "BROTLI", "SNAPPY", "Blosc2"};
  std::vector<std::string> presets = {"fast", "balanced", "best", "default"};

  for (size_t i = 0; i < libs.size(); ++i) {
    int base_id = static_cast<int>(i + 1);
    for (size_t j = 0; j < presets.size(); ++j) {
      int preset_id = static_cast<int>(j);
      int lib_config_id = base_id * 10 + preset_id;
      id_to_lib_config_[lib_config_id] = {libs[i], presets[j]};
    }
  }
}

LinRegTablePredictor::~LinRegTablePredictor() = default;

LinRegTablePredictor::LinRegTablePredictor(LinRegTablePredictor&& other) noexcept
    : config_(std::move(other.config_)),
      table_(std::move(other.table_)),
      ready_(other.ready_),
      id_to_lib_config_(std::move(other.id_to_lib_config_)) {
  other.ready_ = false;
}

LinRegTablePredictor& LinRegTablePredictor::operator=(LinRegTablePredictor&& other) noexcept {
  if (this != &other) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_ = std::move(other.config_);
    table_ = std::move(other.table_);
    ready_ = other.ready_;
    id_to_lib_config_ = std::move(other.id_to_lib_config_);
    other.ready_ = false;
  }
  return *this;
}

bool LinRegTablePredictor::Load(const std::string& model_dir) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Read linreg_table.json
  std::string table_path = model_dir + "/linreg_table.json";
  std::ifstream file(table_path);
  if (!file.is_open()) {
    return false;
  }

  std::stringstream buffer;
  buffer << file.rdbuf();
  std::string json_str = buffer.str();
  file.close();

  // Clear existing table
  table_.clear();

  // Parse models array
  size_t models_pos = FindJsonKey(json_str, "models");
  if (models_pos == std::string::npos) {
    return false;
  }

  // Find array start
  while (models_pos < json_str.length() && json_str[models_pos] != '[') {
    ++models_pos;
  }
  if (models_pos >= json_str.length()) return false;
  ++models_pos;  // Skip '['

  // Parse each model entry
  while (models_pos < json_str.length()) {
    SkipWhitespace(json_str, models_pos);
    if (json_str[models_pos] == ']') break;
    if (json_str[models_pos] == ',') {
      ++models_pos;
      continue;
    }
    if (json_str[models_pos] != '{') {
      ++models_pos;
      continue;
    }

    // Find the end of this object
    size_t obj_start = models_pos;
    int brace_count = 1;
    ++models_pos;
    while (models_pos < json_str.length() && brace_count > 0) {
      if (json_str[models_pos] == '{') ++brace_count;
      else if (json_str[models_pos] == '}') --brace_count;
      ++models_pos;
    }
    std::string obj_str = json_str.substr(obj_start, models_pos - obj_start);

    // Parse this model entry
    LinRegTableKey key;
    LinearRegressionCoeffs coeffs;

    size_t pos = FindJsonKey(obj_str, "library");
    if (pos != std::string::npos) key.library = ParseJsonString(obj_str, pos);

    pos = FindJsonKey(obj_str, "config");
    if (pos != std::string::npos) key.config = ParseJsonString(obj_str, pos);

    pos = FindJsonKey(obj_str, "data_type");
    if (pos != std::string::npos) key.data_type = ParseJsonString(obj_str, pos);

    pos = FindJsonKey(obj_str, "distribution");
    if (pos != std::string::npos) key.distribution = ParseJsonString(obj_str, pos);

    pos = FindJsonKey(obj_str, "slope_compress_time");
    if (pos != std::string::npos) coeffs.slope_compress_time = ParseJsonNumber(obj_str, pos);

    pos = FindJsonKey(obj_str, "intercept_compress_time");
    if (pos != std::string::npos) coeffs.intercept_compress_time = ParseJsonNumber(obj_str, pos);

    pos = FindJsonKey(obj_str, "slope_decompress_time");
    if (pos != std::string::npos) coeffs.slope_decompress_time = ParseJsonNumber(obj_str, pos);

    pos = FindJsonKey(obj_str, "intercept_decompress_time");
    if (pos != std::string::npos) coeffs.intercept_decompress_time = ParseJsonNumber(obj_str, pos);

    pos = FindJsonKey(obj_str, "slope_compress_ratio");
    if (pos != std::string::npos) coeffs.slope_compress_ratio = ParseJsonNumber(obj_str, pos);

    pos = FindJsonKey(obj_str, "intercept_compress_ratio");
    if (pos != std::string::npos) coeffs.intercept_compress_ratio = ParseJsonNumber(obj_str, pos);

    pos = FindJsonKey(obj_str, "sample_count");
    if (pos != std::string::npos) coeffs.sample_count = static_cast<size_t>(ParseJsonNumber(obj_str, pos));

    pos = FindJsonKey(obj_str, "r2_compress_time");
    if (pos != std::string::npos) coeffs.r2_compress_time = ParseJsonNumber(obj_str, pos);

    pos = FindJsonKey(obj_str, "r2_decompress_time");
    if (pos != std::string::npos) coeffs.r2_decompress_time = ParseJsonNumber(obj_str, pos);

    pos = FindJsonKey(obj_str, "r2_compress_ratio");
    if (pos != std::string::npos) coeffs.r2_compress_ratio = ParseJsonNumber(obj_str, pos);

    if (!key.library.empty() && !key.config.empty() && !key.data_type.empty()) {
      table_[key] = coeffs;
    }
  }

  ready_ = !table_.empty();
  return ready_;
}

bool LinRegTablePredictor::Save(const std::string& model_dir) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Create output file
  std::string table_path = model_dir + "/linreg_table.json";
  std::ofstream file(table_path);
  if (!file.is_open()) {
    return false;
  }

  // Write JSON
  file << "{\n";
  file << "  \"model_type\": \"linreg_table\",\n";
  file << "  \"version\": \"1.0\",\n";

  // Get current time
  std::time_t now = std::time(nullptr);
  char time_buf[64];
  std::strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
  file << "  \"created\": \"" << time_buf << "\",\n";

  file << "  \"num_models\": " << table_.size() << ",\n";
  file << "  \"models\": [\n";

  bool first = true;
  for (const auto& entry : table_) {
    if (!first) file << ",\n";
    first = false;

    const auto& key = entry.first;
    const auto& coeffs = entry.second;

    file << "    {\n";
    file << "      \"library\": \"" << key.library << "\",\n";
    file << "      \"config\": \"" << key.config << "\",\n";
    file << "      \"data_type\": \"" << key.data_type << "\",\n";
    file << "      \"distribution\": \"" << key.distribution << "\",\n";
    file << "      \"slope_compress_time\": " << coeffs.slope_compress_time << ",\n";
    file << "      \"intercept_compress_time\": " << coeffs.intercept_compress_time << ",\n";
    file << "      \"slope_decompress_time\": " << coeffs.slope_decompress_time << ",\n";
    file << "      \"intercept_decompress_time\": " << coeffs.intercept_decompress_time << ",\n";
    file << "      \"slope_compress_ratio\": " << coeffs.slope_compress_ratio << ",\n";
    file << "      \"intercept_compress_ratio\": " << coeffs.intercept_compress_ratio << ",\n";
    file << "      \"sample_count\": " << coeffs.sample_count << ",\n";
    file << "      \"r2_compress_time\": " << coeffs.r2_compress_time << ",\n";
    file << "      \"r2_decompress_time\": " << coeffs.r2_decompress_time << ",\n";
    file << "      \"r2_compress_ratio\": " << coeffs.r2_compress_ratio << "\n";
    file << "    }";
  }

  file << "\n  ]\n";
  file << "}\n";

  file.close();

  // Also write metadata.json
  std::string meta_path = model_dir + "/metadata.json";
  std::ofstream meta_file(meta_path);
  if (meta_file.is_open()) {
    meta_file << "{\n";
    meta_file << "  \"model_type\": \"linreg_table\",\n";
    meta_file << "  \"version\": \"1.0\",\n";
    meta_file << "  \"created\": \"" << time_buf << "\",\n";
    meta_file << "  \"num_models\": " << table_.size() << ",\n";
    meta_file << "  \"features\": [\"data_size\"],\n";
    meta_file << "  \"targets\": [\"compress_time_ms\", \"decompress_time_ms\", \"compression_ratio\"]\n";
    meta_file << "}\n";
    meta_file.close();
  }

  return true;
}

bool LinRegTablePredictor::IsReady() const {
  return ready_;
}

CompressionPrediction LinRegTablePredictor::Predict(const CompressionFeatures& features) {
  std::string library, config;
  DecodeLibraryConfigId(static_cast<int>(features.library_config_id), library, config);
  std::string data_type = GetDataTypeFromFeatures(features);
  return PredictByKey(library, config, data_type, features.chunk_size_bytes);
}

std::vector<CompressionPrediction> LinRegTablePredictor::PredictBatch(
    const std::vector<CompressionFeatures>& batch) {
  std::vector<CompressionPrediction> results;
  results.reserve(batch.size());
  for (const auto& features : batch) {
    results.push_back(Predict(features));
  }
  return results;
}

CompressionPrediction LinRegTablePredictor::PredictByKey(
    const std::string& library, const std::string& config,
    const std::string& data_type, const std::string& distribution,
    double data_size) {
  std::lock_guard<std::mutex> lock(mutex_);

  CompressionPrediction result;

  LinRegTableKey key(library, config, data_type, distribution);
  auto it = table_.find(key);

  if (it != table_.end()) {
    const auto& coeffs = it->second;
    result.compression_time_ms = coeffs.PredictCompressTime(data_size);
    result.compression_ratio = coeffs.PredictCompressRatio(data_size);
    // Store decompress time in psnr_db field (or add new field)
    // For now, calculate and store both
    double decompress_time = coeffs.PredictDecompressTime(data_size);
    (void)decompress_time;  // Suppress unused warning
    // Note: CompressionPrediction only has compression_time_ms, not decompress
    // We can extend it later or use psnr_db temporarily
    result.psnr_db = 0.0;  // Lossless, no PSNR
    result.inference_time_ms = 0.001;  // Linear regression is instant
  } else {
    // Fallback to default values
    result.compression_time_ms = config_.fallback_compress_time;
    result.compression_ratio = config_.fallback_compress_ratio;
    result.psnr_db = 0.0;
    result.inference_time_ms = 0.001;
  }

  return result;
}

CompressionPrediction LinRegTablePredictor::PredictByKey(
    const std::string& library, const std::string& config,
    const std::string& data_type, double data_size) {
  // Try with empty distribution first, then use fallback
  return PredictByKey(library, config, data_type, "", data_size);
}

const LinearRegressionCoeffs* LinRegTablePredictor::GetCoeffs(
    const std::string& library, const std::string& config,
    const std::string& data_type, const std::string& distribution) const {
  LinRegTableKey key(library, config, data_type, distribution);
  auto it = table_.find(key);
  if (it != table_.end()) {
    return &(it->second);
  }
  return nullptr;
}

void LinRegTablePredictor::FitLinearRegression(
    const std::vector<double>& x, const std::vector<double>& y,
    double& slope, double& intercept, double& r2) {
  size_t n = x.size();
  if (n < 2) {
    slope = 0.0;
    intercept = n > 0 ? y[0] : 0.0;
    r2 = 0.0;
    return;
  }

  // Calculate means
  double sum_x = std::accumulate(x.begin(), x.end(), 0.0);
  double sum_y = std::accumulate(y.begin(), y.end(), 0.0);
  double mean_x = sum_x / n;
  double mean_y = sum_y / n;

  // Calculate slope and intercept using OLS
  double numerator = 0.0;
  double denominator = 0.0;
  for (size_t i = 0; i < n; ++i) {
    double dx = x[i] - mean_x;
    numerator += dx * (y[i] - mean_y);
    denominator += dx * dx;
  }

  if (std::abs(denominator) < 1e-10) {
    slope = 0.0;
    intercept = mean_y;
    r2 = 0.0;
    return;
  }

  slope = numerator / denominator;
  intercept = mean_y - slope * mean_x;

  // Calculate R² score
  double ss_res = 0.0;
  double ss_tot = 0.0;
  for (size_t i = 0; i < n; ++i) {
    double y_pred = slope * x[i] + intercept;
    ss_res += (y[i] - y_pred) * (y[i] - y_pred);
    ss_tot += (y[i] - mean_y) * (y[i] - mean_y);
  }

  r2 = (ss_tot > 1e-10) ? (1.0 - ss_res / ss_tot) : 0.0;
}

bool LinRegTablePredictor::Train(const std::vector<CompressionFeatures>& features,
                                 const std::vector<TrainingLabels>& labels) {
  (void)features;
  (void)labels;
  // Not implemented - use TrainFromRaw instead
  return false;
}

bool LinRegTablePredictor::TrainFromRaw(
    const std::vector<std::string>& libraries,
    const std::vector<std::string>& configs,
    const std::vector<std::string>& data_types,
    const std::vector<std::string>& distributions,
    const std::vector<double>& data_sizes,
    const std::vector<double>& compress_times,
    const std::vector<double>& decompress_times,
    const std::vector<double>& compress_ratios) {

  size_t n = libraries.size();
  if (n != configs.size() || n != data_types.size() || n != distributions.size() ||
      n != data_sizes.size() || n != compress_times.size() ||
      n != decompress_times.size() || n != compress_ratios.size()) {
    return false;
  }

  // Group data by (library, config, data_type, distribution)
  std::map<LinRegTableKey, std::vector<size_t>> groups;
  for (size_t i = 0; i < n; ++i) {
    LinRegTableKey key(libraries[i], configs[i], data_types[i], distributions[i]);
    groups[key].push_back(i);
  }

  // Fit linear regression for each group
  table_.clear();
  for (const auto& group : groups) {
    const auto& key = group.first;
    const auto& indices = group.second;

    if (indices.size() < static_cast<size_t>(config_.min_samples)) {
      continue;
    }

    // Extract data for this group
    std::vector<double> x_data, y_compress, y_decompress, y_ratio;
    x_data.reserve(indices.size());
    y_compress.reserve(indices.size());
    y_decompress.reserve(indices.size());
    y_ratio.reserve(indices.size());

    for (size_t idx : indices) {
      x_data.push_back(data_sizes[idx]);
      y_compress.push_back(compress_times[idx]);
      y_decompress.push_back(decompress_times[idx]);
      y_ratio.push_back(compress_ratios[idx]);
    }

    // Fit linear regression for each target
    LinearRegressionCoeffs coeffs;
    coeffs.sample_count = indices.size();

    FitLinearRegression(x_data, y_compress,
                        coeffs.slope_compress_time,
                        coeffs.intercept_compress_time,
                        coeffs.r2_compress_time);

    FitLinearRegression(x_data, y_decompress,
                        coeffs.slope_decompress_time,
                        coeffs.intercept_decompress_time,
                        coeffs.r2_decompress_time);

    FitLinearRegression(x_data, y_ratio,
                        coeffs.slope_compress_ratio,
                        coeffs.intercept_compress_ratio,
                        coeffs.r2_compress_ratio);

    table_[key] = coeffs;
  }

  ready_ = !table_.empty();
  return ready_;
}

void LinRegTablePredictor::DecodeLibraryConfigId(
    int library_config_id, std::string& library, std::string& config) const {
  auto it = id_to_lib_config_.find(library_config_id);
  if (it != id_to_lib_config_.end()) {
    library = it->second.first;
    config = it->second.second;
  } else {
    library = "unknown";
    config = "unknown";
  }
}

std::string LinRegTablePredictor::GetDataTypeFromFeatures(
    const CompressionFeatures& features) const {
  if (features.data_type_char > 0.5) {
    return "char";
  } else if (features.data_type_float > 0.5) {
    return "float";
  } else {
    return "int";  // Default
  }
}

std::string LinRegTablePredictor::GetStatistics() const {
  std::stringstream ss;
  ss << "LinRegTablePredictor Statistics:\n";
  ss << "  Ready: " << (ready_ ? "yes" : "no") << "\n";
  ss << "  Number of models: " << table_.size() << "\n";

  if (!table_.empty()) {
    double avg_r2_compress = 0.0;
    double avg_r2_decompress = 0.0;
    double avg_r2_ratio = 0.0;
    size_t total_samples = 0;

    for (const auto& entry : table_) {
      const auto& coeffs = entry.second;
      avg_r2_compress += coeffs.r2_compress_time;
      avg_r2_decompress += coeffs.r2_decompress_time;
      avg_r2_ratio += coeffs.r2_compress_ratio;
      total_samples += coeffs.sample_count;
    }

    avg_r2_compress /= table_.size();
    avg_r2_decompress /= table_.size();
    avg_r2_ratio /= table_.size();

    ss << "  Total training samples: " << total_samples << "\n";
    ss << "  Average R² (compress_time): " << avg_r2_compress << "\n";
    ss << "  Average R² (decompress_time): " << avg_r2_decompress << "\n";
    ss << "  Average R² (compress_ratio): " << avg_r2_ratio << "\n";
  }

  return ss.str();
}

std::vector<std::string> LinRegTablePredictor::GetLibraries() const {
  std::set<std::string> libs;
  for (const auto& entry : table_) {
    libs.insert(entry.first.library);
  }
  return std::vector<std::string>(libs.begin(), libs.end());
}

std::vector<std::string> LinRegTablePredictor::GetConfigurations() const {
  std::set<std::string> configs;
  for (const auto& entry : table_) {
    configs.insert(entry.first.config);
  }
  return std::vector<std::string>(configs.begin(), configs.end());
}

std::vector<std::string> LinRegTablePredictor::GetDataTypes() const {
  std::set<std::string> types;
  for (const auto& entry : table_) {
    types.insert(entry.first.data_type);
  }
  return std::vector<std::string>(types.begin(), types.end());
}

std::vector<std::string> LinRegTablePredictor::GetDistributions() const {
  std::set<std::string> dists;
  for (const auto& entry : table_) {
    if (!entry.first.distribution.empty()) {
      dists.insert(entry.first.distribution);
    }
  }
  return std::vector<std::string>(dists.begin(), dists.end());
}

}  // namespace wrp_cte::compressor
