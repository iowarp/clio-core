/**
 * @file xgboost_predictor.cc
 * @brief Implementation of XGBoost-based compression predictor
 */

#include "context-transport-primitives/compression/xgboost_predictor.h"
#include <fstream>
#include <sstream>
#include <chrono>
#include <cstring>
#include <filesystem>

namespace hshm {
namespace compress {

namespace fs = std::filesystem;

/**
 * @brief Macro to check XGBoost API return code
 */
#define XGBOOST_CHECK(call)                                    \
  do {                                                         \
    int ret = (call);                                          \
    if (ret != 0) {                                            \
      return false;                                            \
    }                                                          \
  } while (0)

XGBoostPredictor::XGBoostPredictor()
    : ratio_booster_(nullptr),
      psnr_booster_(nullptr),
      time_booster_(nullptr),
      is_ready_(false),
      epsilon_(0.1),
      exploration_enabled_(false),
      rng_(std::random_device{}()),
      experience_count_(0) {}

XGBoostPredictor::~XGBoostPredictor() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (ratio_booster_ != nullptr) {
    XGBoosterFree(ratio_booster_);
    ratio_booster_ = nullptr;
  }
  if (psnr_booster_ != nullptr) {
    XGBoosterFree(psnr_booster_);
    psnr_booster_ = nullptr;
  }
  if (time_booster_ != nullptr) {
    XGBoosterFree(time_booster_);
    time_booster_ = nullptr;
  }
}

XGBoostPredictor::XGBoostPredictor(XGBoostPredictor&& other) noexcept
    : ratio_booster_(other.ratio_booster_),
      psnr_booster_(other.psnr_booster_),
      time_booster_(other.time_booster_),
      is_ready_(other.is_ready_),
      experience_buffer_(std::move(other.experience_buffer_)),
      epsilon_(other.epsilon_),
      exploration_enabled_(other.exploration_enabled_),
      rng_(std::move(other.rng_)),
      experience_count_(other.experience_count_) {
  other.ratio_booster_ = nullptr;
  other.psnr_booster_ = nullptr;
  other.time_booster_ = nullptr;
  other.is_ready_ = false;
  other.experience_count_ = 0;
}

XGBoostPredictor& XGBoostPredictor::operator=(XGBoostPredictor&& other) noexcept {
  if (this != &other) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (ratio_booster_ != nullptr) {
      XGBoosterFree(ratio_booster_);
    }
    if (psnr_booster_ != nullptr) {
      XGBoosterFree(psnr_booster_);
    }
    if (time_booster_ != nullptr) {
      XGBoosterFree(time_booster_);
    }
    ratio_booster_ = other.ratio_booster_;
    psnr_booster_ = other.psnr_booster_;
    time_booster_ = other.time_booster_;
    is_ready_ = other.is_ready_;
    experience_buffer_ = std::move(other.experience_buffer_);
    epsilon_ = other.epsilon_;
    exploration_enabled_ = other.exploration_enabled_;
    rng_ = std::move(other.rng_);
    experience_count_ = other.experience_count_;
    other.ratio_booster_ = nullptr;
    other.psnr_booster_ = nullptr;
    other.time_booster_ = nullptr;
    other.is_ready_ = false;
    other.experience_count_ = 0;
  }
  return *this;
}

bool XGBoostPredictor::Load(const std::string& model_dir) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Free existing boosters
  if (ratio_booster_ != nullptr) {
    XGBoosterFree(ratio_booster_);
    ratio_booster_ = nullptr;
  }
  if (psnr_booster_ != nullptr) {
    XGBoosterFree(psnr_booster_);
    psnr_booster_ = nullptr;
  }
  if (time_booster_ != nullptr) {
    XGBoosterFree(time_booster_);
    time_booster_ = nullptr;
  }
  is_ready_ = false;

  // Check model files exist
  fs::path dir(model_dir);
  fs::path ratio_model_path = dir / "compression_ratio_model.json";
  fs::path psnr_model_path = dir / "psnr_model.json";
  fs::path time_model_path = dir / "compression_time_model.json";

  if (!fs::exists(ratio_model_path)) {
    return false;
  }

  // Create and load compression ratio booster
  XGBOOST_CHECK(XGBoosterCreate(nullptr, 0, &ratio_booster_));
  XGBOOST_CHECK(XGBoosterLoadModel(ratio_booster_, ratio_model_path.string().c_str()));

  // Load PSNR booster if it exists (optional for lossless-only use)
  if (fs::exists(psnr_model_path)) {
    XGBOOST_CHECK(XGBoosterCreate(nullptr, 0, &psnr_booster_));
    XGBOOST_CHECK(XGBoosterLoadModel(psnr_booster_, psnr_model_path.string().c_str()));
  }

  // Load compression time booster if it exists (optional)
  if (fs::exists(time_model_path)) {
    XGBOOST_CHECK(XGBoosterCreate(nullptr, 0, &time_booster_));
    XGBOOST_CHECK(XGBoosterLoadModel(time_booster_, time_model_path.string().c_str()));
  }

  is_ready_ = true;
  return true;
}

bool XGBoostPredictor::Save(const std::string& model_dir) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (!is_ready_ || ratio_booster_ == nullptr) {
    return false;
  }

  // Create directory if it doesn't exist
  fs::path dir(model_dir);
  if (!fs::exists(dir)) {
    fs::create_directories(dir);
  }

  // Save compression ratio model
  fs::path ratio_model_path = dir / "compression_ratio_model.json";
  XGBOOST_CHECK(XGBoosterSaveModel(ratio_booster_, ratio_model_path.string().c_str()));

  // Save PSNR model if it exists
  if (psnr_booster_ != nullptr) {
    fs::path psnr_model_path = dir / "psnr_model.json";
    XGBOOST_CHECK(XGBoosterSaveModel(psnr_booster_, psnr_model_path.string().c_str()));
  }

  // Save compression time model if it exists
  if (time_booster_ != nullptr) {
    fs::path time_model_path = dir / "compression_time_model.json";
    XGBOOST_CHECK(XGBoosterSaveModel(time_booster_, time_model_path.string().c_str()));
  }

  return true;
}

bool XGBoostPredictor::IsReady() const {
  return is_ready_;
}

CompressionPrediction XGBoostPredictor::Predict(const CompressionFeatures& features) {
  std::vector<CompressionFeatures> batch = {features};
  auto results = PredictBatch(batch);
  if (!results.empty()) {
    return results[0];
  }
  return CompressionPrediction();
}

std::vector<CompressionPrediction> XGBoostPredictor::PredictBatch(
    const std::vector<CompressionFeatures>& batch) {
  std::vector<CompressionPrediction> results;
  results.reserve(batch.size());

  if (!is_ready_ || batch.empty()) {
    return results;
  }

  auto start_time = std::chrono::high_resolution_clock::now();

  std::lock_guard<std::mutex> lock(mutex_);

  // Create DMatrix from features
  DMatrixHandle dmatrix = CreateDMatrix(batch, nullptr);
  if (dmatrix == nullptr) {
    return results;
  }

  // Predict compression ratios
  std::vector<float> ratio_predictions;
  if (!PredictWithBooster(ratio_booster_, dmatrix, ratio_predictions)) {
    FreeDMatrix(dmatrix);
    return results;
  }

  // Predict PSNR if booster exists
  std::vector<float> psnr_predictions(batch.size(), 0.0F);
  if (psnr_booster_ != nullptr) {
    PredictWithBooster(psnr_booster_, dmatrix, psnr_predictions);
  }

  // Predict compression time if booster exists
  std::vector<float> time_predictions(batch.size(), 0.0F);
  if (time_booster_ != nullptr) {
    PredictWithBooster(time_booster_, dmatrix, time_predictions);
  }

  FreeDMatrix(dmatrix);

  auto end_time = std::chrono::high_resolution_clock::now();
  double total_time_ms = std::chrono::duration<double, std::milli>(
      end_time - start_time).count();
  double per_sample_time_ms = total_time_ms / static_cast<double>(batch.size());

  // Build results
  for (size_t i = 0; i < batch.size(); ++i) {
    double psnr = 0.0;
    // Only report PSNR for lossy compression
    if (IsLossyCompression(batch[i])) {
      psnr = static_cast<double>(psnr_predictions[i]);
    }
    results.emplace_back(
        static_cast<double>(ratio_predictions[i]),
        psnr,
        static_cast<double>(time_predictions[i]),
        per_sample_time_ms);
  }

  return results;
}

bool XGBoostPredictor::TrainRatioModel(
    const std::vector<CompressionFeatures>& features,
    const std::vector<float>& labels,
    const XGBoostConfig& config) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (features.size() != labels.size() || features.empty()) {
    return false;
  }

  // Free existing ratio booster
  if (ratio_booster_ != nullptr) {
    XGBoosterFree(ratio_booster_);
    ratio_booster_ = nullptr;
  }

  // Create DMatrix with labels
  DMatrixHandle dmatrix = CreateDMatrix(features, &labels);
  if (dmatrix == nullptr) {
    return false;
  }

  // Train the booster
  bool success = TrainBooster(dmatrix, config, ratio_booster_);
  FreeDMatrix(dmatrix);

  if (success && ratio_booster_ != nullptr) {
    is_ready_ = true;
  }

  return success;
}

bool XGBoostPredictor::TrainPSNRModel(
    const std::vector<CompressionFeatures>& features,
    const std::vector<float>& labels,
    const XGBoostConfig& config) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (features.size() != labels.size() || features.empty()) {
    return false;
  }

  // Free existing PSNR booster
  if (psnr_booster_ != nullptr) {
    XGBoosterFree(psnr_booster_);
    psnr_booster_ = nullptr;
  }

  // Create DMatrix with labels
  DMatrixHandle dmatrix = CreateDMatrix(features, &labels);
  if (dmatrix == nullptr) {
    return false;
  }

  // Train the booster
  bool success = TrainBooster(dmatrix, config, psnr_booster_);
  FreeDMatrix(dmatrix);

  return success;
}

bool XGBoostPredictor::TrainTimeModel(
    const std::vector<CompressionFeatures>& features,
    const std::vector<float>& labels,
    const XGBoostConfig& config) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (features.size() != labels.size() || features.empty()) {
    return false;
  }

  // Free existing time booster
  if (time_booster_ != nullptr) {
    XGBoosterFree(time_booster_);
    time_booster_ = nullptr;
  }

  // Create DMatrix with labels
  DMatrixHandle dmatrix = CreateDMatrix(features, &labels);
  if (dmatrix == nullptr) {
    return false;
  }

  // Train the booster
  bool success = TrainBooster(dmatrix, config, time_booster_);
  FreeDMatrix(dmatrix);

  return success;
}

bool XGBoostPredictor::Train(
    const std::vector<CompressionFeatures>& features,
    const std::vector<TrainingLabels>& labels) {
  return TrainWithConfig(features, labels, XGBoostConfig());
}

bool XGBoostPredictor::TrainWithConfig(
    const std::vector<CompressionFeatures>& features,
    const std::vector<TrainingLabels>& labels,
    const XGBoostConfig& config) {
  if (features.size() != labels.size() || features.empty()) {
    return false;
  }

  // Extract individual label vectors
  std::vector<float> ratio_labels;
  std::vector<float> psnr_labels;
  std::vector<float> time_labels;
  ratio_labels.reserve(labels.size());
  psnr_labels.reserve(labels.size());
  time_labels.reserve(labels.size());

  for (const auto& label : labels) {
    ratio_labels.push_back(label.compression_ratio);
    psnr_labels.push_back(label.psnr_db);
    time_labels.push_back(label.compression_time_ms);
  }

  // Train all three models
  // Note: We don't hold the lock across all three to allow parallelism
  bool ratio_ok = TrainRatioModel(features, ratio_labels, config);
  if (!ratio_ok) {
    return false;
  }

  bool psnr_ok = TrainPSNRModel(features, psnr_labels, config);
  bool time_ok = TrainTimeModel(features, time_labels, config);

  // Return true if at least ratio model trained (PSNR and time are optional)
  return ratio_ok && (psnr_ok || time_ok || true);
}

std::string XGBoostPredictor::GetLastError() const {
  const char* err = XGBGetLastError();
  return err ? std::string(err) : "";
}

DMatrixHandle XGBoostPredictor::CreateDMatrix(
    const std::vector<CompressionFeatures>& features,
    const std::vector<float>* labels) {
  if (features.empty()) {
    return nullptr;
  }

  bst_ulong num_rows = static_cast<bst_ulong>(features.size());
  bst_ulong num_cols = static_cast<bst_ulong>(CompressionFeatures::NumFeatures());

  // Flatten features into a contiguous array (row-major)
  std::vector<float> data;
  data.reserve(num_rows * num_cols);
  for (const auto& f : features) {
    auto vec = f.ToVector();
    data.insert(data.end(), vec.begin(), vec.end());
  }

  // Use XGDMatrixCreateFromMat which is simpler and more reliable
  DMatrixHandle dmatrix = nullptr;
  int ret = XGDMatrixCreateFromMat(
      data.data(),
      num_rows,
      num_cols,
      std::numeric_limits<float>::quiet_NaN(),  // missing value
      &dmatrix);

  if (ret != 0 || dmatrix == nullptr) {
    return nullptr;
  }

  // Set labels if provided
  if (labels != nullptr && !labels->empty()) {
    ret = XGDMatrixSetFloatInfo(dmatrix, "label", labels->data(), labels->size());
    if (ret != 0) {
      XGDMatrixFree(dmatrix);
      return nullptr;
    }
  }

  return dmatrix;
}

void XGBoostPredictor::FreeDMatrix(DMatrixHandle dmatrix) {
  if (dmatrix != nullptr) {
    XGDMatrixFree(dmatrix);
  }
}

bool XGBoostPredictor::PredictWithBooster(
    BoosterHandle booster,
    DMatrixHandle dmatrix,
    std::vector<float>& out_predictions) {
  if (booster == nullptr || dmatrix == nullptr) {
    return false;
  }

  // Use simpler prediction API
  bst_ulong out_len = 0;
  const float* out_result = nullptr;

  int ret = XGBoosterPredict(
      booster, dmatrix,
      0,       // option_mask: 0 = normal prediction
      0,       // ntree_limit: 0 = use all trees
      0,       // training: false
      &out_len,
      &out_result);

  if (ret != 0 || out_result == nullptr || out_len == 0) {
    return false;
  }

  // Copy predictions
  out_predictions.assign(out_result, out_result + out_len);
  return true;
}

bool XGBoostPredictor::TrainBooster(
    DMatrixHandle dmatrix,
    const XGBoostConfig& config,
    BoosterHandle& out_booster) {
  // Create booster
  DMatrixHandle dmats[] = {dmatrix};
  XGBOOST_CHECK(XGBoosterCreate(dmats, 1, &out_booster));

  // Set parameters
  std::ostringstream oss;

  oss.str("");
  oss << config.max_depth;
  XGBOOST_CHECK(XGBoosterSetParam(out_booster, "max_depth", oss.str().c_str()));

  oss.str("");
  oss << config.learning_rate;
  XGBOOST_CHECK(XGBoosterSetParam(out_booster, "eta", oss.str().c_str()));

  oss.str("");
  oss << config.subsample;
  XGBOOST_CHECK(XGBoosterSetParam(out_booster, "subsample", oss.str().c_str()));

  oss.str("");
  oss << config.colsample_bytree;
  XGBOOST_CHECK(XGBoosterSetParam(out_booster, "colsample_bytree", oss.str().c_str()));

  oss.str("");
  oss << config.reg_lambda;
  XGBOOST_CHECK(XGBoosterSetParam(out_booster, "lambda", oss.str().c_str()));

  oss.str("");
  oss << config.min_child_weight;
  XGBOOST_CHECK(XGBoosterSetParam(out_booster, "min_child_weight", oss.str().c_str()));

  oss.str("");
  oss << config.seed;
  XGBOOST_CHECK(XGBoosterSetParam(out_booster, "seed", oss.str().c_str()));

  XGBOOST_CHECK(XGBoosterSetParam(out_booster, "objective", "reg:squarederror"));

  // Training iterations
  for (int iter = 0; iter < config.n_estimators; ++iter) {
    XGBOOST_CHECK(XGBoosterUpdateOneIter(out_booster, iter, dmatrix));

    if (config.verbose && (iter + 1) % 50 == 0) {
      // Could add evaluation metric printing here
    }
  }

  return true;
}

bool XGBoostPredictor::IsLossyCompression(const CompressionFeatures& features) const {
  // Check if any lossy compression library is selected
  return features.library_zfp_tol_0_01 > 0.5 ||
         features.library_zfp_tol_0_1 > 0.5;
}

// ============================================================================
// Reinforcement Learning Implementation
// ============================================================================

void XGBoostPredictor::RecordExperience(const RLExperience& experience) {
  std::lock_guard<std::mutex> lock(mutex_);

  experience_buffer_.push_back(experience);
  experience_count_++;

  // Limit buffer size (default 10000)
  while (experience_buffer_.size() > 10000) {
    experience_buffer_.pop_front();
  }
}

bool XGBoostPredictor::UpdateFromExperiences(const RLConfig& config) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Need enough experiences for a batch
  if (experience_buffer_.size() < config.batch_size) {
    return false;
  }

  // Need a trained model to update
  if (!is_ready_ || ratio_booster_ == nullptr) {
    return false;
  }

  // Sample a random batch from experience buffer
  std::vector<CompressionFeatures> batch_features;
  std::vector<float> batch_ratio_labels;
  std::vector<float> batch_psnr_labels;
  std::vector<float> batch_time_labels;
  batch_features.reserve(config.batch_size);
  batch_ratio_labels.reserve(config.batch_size);
  batch_psnr_labels.reserve(config.batch_size);
  batch_time_labels.reserve(config.batch_size);

  std::uniform_int_distribution<size_t> dist(0, experience_buffer_.size() - 1);
  for (size_t i = 0; i < config.batch_size; ++i) {
    size_t idx = dist(rng_);
    const auto& exp = experience_buffer_[idx];
    batch_features.push_back(exp.features);
    batch_ratio_labels.push_back(static_cast<float>(exp.actual_ratio));
    batch_psnr_labels.push_back(static_cast<float>(exp.actual_psnr));
    batch_time_labels.push_back(static_cast<float>(exp.actual_compress_time));
  }

  // Create DMatrix with new labels
  DMatrixHandle dmatrix = CreateDMatrix(batch_features, &batch_ratio_labels);
  if (dmatrix == nullptr) {
    return false;
  }

  // Perform incremental update on ratio booster
  // XGBoost supports incremental training via UpdateOneIter
  int ret = XGBoosterUpdateOneIter(ratio_booster_,
                                   static_cast<int>(experience_count_),
                                   dmatrix);
  FreeDMatrix(dmatrix);

  if (ret != 0) {
    return false;
  }

  // Update PSNR booster if available
  if (psnr_booster_ != nullptr) {
    DMatrixHandle psnr_dmatrix = CreateDMatrix(batch_features, &batch_psnr_labels);
    if (psnr_dmatrix != nullptr) {
      XGBoosterUpdateOneIter(psnr_booster_,
                             static_cast<int>(experience_count_),
                             psnr_dmatrix);
      FreeDMatrix(psnr_dmatrix);
    }
  }

  // Update time booster if available
  if (time_booster_ != nullptr) {
    DMatrixHandle time_dmatrix = CreateDMatrix(batch_features, &batch_time_labels);
    if (time_dmatrix != nullptr) {
      XGBoosterUpdateOneIter(time_booster_,
                             static_cast<int>(experience_count_),
                             time_dmatrix);
      FreeDMatrix(time_dmatrix);
    }
  }

  // Decay epsilon
  epsilon_ = std::max(config.min_epsilon, epsilon_ * config.epsilon_decay);

  return true;
}

void XGBoostPredictor::SetExplorationMode(bool enable) {
  std::lock_guard<std::mutex> lock(mutex_);
  exploration_enabled_ = enable;
}

double XGBoostPredictor::GetExplorationRate() const {
  return epsilon_;
}

size_t XGBoostPredictor::GetExperienceCount() const {
  return experience_count_;
}

void XGBoostPredictor::ClearExperiences() {
  std::lock_guard<std::mutex> lock(mutex_);
  experience_buffer_.clear();
  experience_count_ = 0;
}

}  // namespace compress
}  // namespace hshm
