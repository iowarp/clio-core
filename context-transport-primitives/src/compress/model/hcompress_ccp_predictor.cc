/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#include "clio_ctp/compress/model/hcompress_ccp_predictor.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace ctp::compress::model {

namespace {
/** The ratio training target, clamped when configured. Shared by the seed and
 *  the feedback path: a clamp applied to one and not the other would make the
 *  two phases fit different quantities. */
double RatioTarget(double measured, double cap) {
  return (cap > 0.0) ? std::min(measured, cap) : measured;
}
}  // namespace

// ---------------------------------------------------------------------------
// Recursive least squares
// ---------------------------------------------------------------------------
void CcpRls::Reset(size_t dim, double regularization) {
  dim_ = dim;
  samples_ = 0;
  w_.assign(dim, 0.0);
  // P = (1/C) I. C is dlib rls's `C`: it is the inverse prior covariance, so a
  // SMALL C means a broad prior and a fit that follows the data.
  const double c = (regularization > 0.0) ? regularization : 1e-3;
  p_.assign(dim * dim, 0.0);
  for (size_t i = 0; i < dim; ++i) p_[i * dim + i] = 1.0 / c;
}

namespace {
/** In-place Gauss-Jordan inverse with partial pivoting. Returns false on a
 *  singular matrix, which for these normal equations means a category that
 *  never occurred -- the caller keeps the prior rather than producing NaNs. */
bool InvertInPlace(std::vector<double>* m, size_t n) {
  std::vector<double> inv(n * n, 0.0);
  for (size_t i = 0; i < n; ++i) inv[i * n + i] = 1.0;
  std::vector<double>& a = *m;
  for (size_t col = 0; col < n; ++col) {
    size_t piv = col;
    double best = std::fabs(a[col * n + col]);
    for (size_t r = col + 1; r < n; ++r) {
      const double v = std::fabs(a[r * n + col]);
      if (v > best) { best = v; piv = r; }
    }
    if (!(best > 1e-300)) return false;
    if (piv != col) {
      for (size_t k = 0; k < n; ++k) {
        std::swap(a[col * n + k], a[piv * n + k]);
        std::swap(inv[col * n + k], inv[piv * n + k]);
      }
    }
    const double d = a[col * n + col];
    for (size_t k = 0; k < n; ++k) {
      a[col * n + k] /= d;
      inv[col * n + k] /= d;
    }
    for (size_t r = 0; r < n; ++r) {
      if (r == col) continue;
      const double f = a[r * n + col];
      if (f == 0.0) continue;
      for (size_t k = 0; k < n; ++k) {
        a[r * n + k] -= f * a[col * n + k];
        inv[r * n + k] -= f * inv[col * n + k];
      }
    }
  }
  *m = inv;
  return true;
}
}  // namespace

void CcpRls::InitFromRidge(const std::vector<double>& a,
                           const std::vector<double>& b, size_t samples,
                           double regularization) {
  if (dim_ == 0 || a.size() != dim_ * dim_ || b.size() != dim_) return;
  const double c = (regularization > 0.0) ? regularization : 1e-3;
  std::vector<double> m = a;
  for (size_t i = 0; i < dim_; ++i) m[i * dim_ + i] += c;
  if (!InvertInPlace(&m, dim_)) return;  // keep the prior
  for (size_t i = 0; i < dim_; ++i) {
    double s = 0.0;
    for (size_t j = 0; j < dim_; ++j) s += m[i * dim_ + j] * b[j];
    w_[i] = s;
  }
  p_ = m;
  samples_ = samples;
}

void CcpRls::Update(const std::vector<double>& x, double y, double forget) {
  if (dim_ == 0 || x.size() != dim_) return;
  const double lambda = (forget > 0.0 && forget <= 1.0) ? forget : 1.0;

  // Px = P x, and den = lambda + x'Px.
  std::vector<double> px(dim_, 0.0);
  double den = lambda;
  for (size_t i = 0; i < dim_; ++i) {
    double s = 0.0;
    const double* row = &p_[i * dim_];
    for (size_t j = 0; j < dim_; ++j) s += row[j] * x[j];
    px[i] = s;
  }
  for (size_t i = 0; i < dim_; ++i) den += x[i] * px[i];
  if (!(den > 0.0) || !std::isfinite(den)) return;

  // w += (Px/den) * (y - w'x)
  double pred = 0.0;
  for (size_t i = 0; i < dim_; ++i) pred += w_[i] * x[i];
  const double err = y - pred;
  for (size_t i = 0; i < dim_; ++i) w_[i] += (px[i] / den) * err;

  // P = (P - Px Px'/den) / lambda. P stays symmetric, so x'P is (Px)'.
  for (size_t i = 0; i < dim_; ++i) {
    double* row = &p_[i * dim_];
    for (size_t j = 0; j < dim_; ++j) {
      row[j] = (row[j] - px[i] * px[j] / den) / lambda;
    }
  }
  ++samples_;
}

double CcpRls::Predict(const std::vector<double>& x) const {
  double s = 0.0;
  const size_t n = std::min(x.size(), w_.size());
  for (size_t i = 0; i < n; ++i) s += w_[i] * x[i];
  return s;
}

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------
size_t HCompressCcpPredictor::Index(std::vector<std::string>* vocab,
                                    const std::string& value,
                                    bool grow) const {
  if (value.empty()) return std::string::npos;  // input unavailable here
  for (size_t i = 0; i < vocab->size(); ++i) {
    if ((*vocab)[i] == value) return i;
  }
  if (!grow) return std::string::npos;
  vocab->push_back(value);
  return vocab->size() - 1;
}

size_t HCompressCcpPredictor::Dimension() const {
  return 1 + types_.size() + formats_.size() + libraries_.size() +
         distributions_.size();
}

std::vector<double> HCompressCcpPredictor::Encode(
    const CcpInputs& inputs) const {
  std::vector<double> x(Dimension(), 0.0);
  x[0] = 1.0;  // intercept
  size_t off = 1;
  auto put = [&x, &off](const std::vector<std::string>& vocab,
                        const std::string& value) {
    if (!value.empty()) {
      for (size_t i = 0; i < vocab.size(); ++i) {
        if (vocab[i] == value) {
          x[off + i] = 1.0;
          break;
        }
      }
    }
    // A value the seed never saw, or an input that is unavailable here, leaves
    // this whole block at zero: the prediction falls back to the intercept plus
    // whichever blocks ARE known. That is the honest behaviour for a model
    // whose cells come from a profiler -- there is no entry to interpolate
    // from -- and the harness reports how often it happened rather than
    // letting it pass as a normal prediction.
    off += vocab.size();
  };
  put(types_, inputs.data_type);
  put(formats_, inputs.data_format);
  put(libraries_, inputs.library);
  put(distributions_, inputs.distribution);
  return x;
}

void HCompressCcpPredictor::Rebuild(size_t dim) {
  // Keep what has been learned and give the new columns the prior. Resetting
  // outright would throw away the seed the moment a workload shows a class the
  // profiler never covered -- which is exactly when the feedback variant is
  // supposed to be able to learn one.
  auto grow = [dim, this](CcpRls* rls) {
    std::vector<double> w = rls->Weights();
    const size_t samples = rls->Samples();
    w.resize(dim, 0.0);
    CcpRls fresh;
    fresh.Reset(dim, config_.regularization);
    fresh.SetWeights(std::move(w), samples);
    *rls = fresh;
  };
  grow(&comp_speed_);
  grow(&decomp_speed_);
  grow(&ratio_);
}

std::string HCompressCcpPredictor::LibraryKey(const std::string& library_name,
                                              bool quantize,
                                              bool byte_shuffle) {
  return library_name + "|q" + (quantize ? "1" : "0") + "|s" +
         (byte_shuffle ? "1" : "0");
}

std::string HCompressCcpPredictor::LibraryKey(
    const CompressionFeatures& features) {
  // No name in the features, so the encoded id stands in for it. The id is
  // stable (base_id*10 + preset), which is all a category key needs.
  std::ostringstream os;
  os << "libcfg" << static_cast<long long>(features.library_config_id);
  return LibraryKey(os.str(), features.quantize != 0.0,
                    features.byte_shuffle != 0.0);
}

// ---------------------------------------------------------------------------
// Seeding and feedback
// ---------------------------------------------------------------------------
void HCompressCcpPredictor::Seed(const std::vector<CcpObservation>& rows) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Pass 1: the vocabularies, so the encoded width is final before any fit.
  for (const auto& r : rows) {
    Index(&types_, r.inputs.data_type, true);
    Index(&formats_, r.inputs.data_format, true);
    Index(&libraries_, r.inputs.library, true);
    Index(&distributions_, r.inputs.distribution, true);
  }
  const size_t dim = Dimension();
  comp_speed_.Reset(dim, config_.regularization);
  decomp_speed_.Reset(dim, config_.regularization);
  ratio_.Reset(dim, config_.regularization);

  // Pass 2: the fit, as normal equations -- one accumulation pass, then one
  // solve per target. See CcpRls::InitFromRidge for why not the streaming form.
  struct Acc {
    std::vector<double> a, b;
    size_t n = 0;
    void Init(size_t d) { a.assign(d * d, 0.0); b.assign(d, 0.0); n = 0; }
    void Add(const std::vector<double>& x, double y) {
      const size_t d = b.size();
      for (size_t i = 0; i < d; ++i) {
        if (x[i] == 0.0) continue;         // one-hot: most entries are zero
        b[i] += x[i] * y;
        for (size_t j = 0; j < d; ++j) {
          if (x[j] != 0.0) a[i * d + j] += x[i] * x[j];
        }
      }
      ++n;
    }
  } acc_ct, acc_dt, acc_ratio;
  acc_ct.Init(dim);
  acc_dt.Init(dim);
  acc_ratio.Init(dim);

  for (const auto& r : rows) {
    const std::vector<double> x = Encode(r.inputs);
    if (r.bytes > 0.0 && r.compress_time_ms > 0.0) {
      acc_ct.Add(x, r.bytes / (r.compress_time_ms * 1000.0));
    }
    if (r.bytes > 0.0 && r.decompress_time_ms > 0.0) {
      acc_dt.Add(x, r.bytes / (r.decompress_time_ms * 1000.0));
    }
    if (r.compression_ratio > 0.0) {
      acc_ratio.Add(x, RatioTarget(r.compression_ratio, config_.ratio_target_cap));
    }
  }
  comp_speed_.InitFromRidge(acc_ct.a, acc_ct.b, acc_ct.n, config_.regularization);
  decomp_speed_.InitFromRidge(acc_dt.a, acc_dt.b, acc_dt.n, config_.regularization);
  ratio_.InitFromRidge(acc_ratio.a, acc_ratio.b, acc_ratio.n,
                       config_.regularization);
  seed_rows_ += rows.size();
}

void HCompressCcpPredictor::SetInputs(const CcpInputs& inputs) {
  std::lock_guard<std::mutex> lock(mutex_);
  inputs_ = inputs;
}

bool HCompressCcpPredictor::Observe(const CcpObservation& row) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_.push_back(row);
    if (pending_.size() < std::max<size_t>(1, config_.feedback_interval)) {
      return false;
    }
  }
  return ApplyFeedback() > 0;
}

size_t HCompressCcpPredictor::ApplyFeedback() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (pending_.empty()) return 0;

  // A feedback row may carry a category the profiler never covered (a
  // distribution class absent from the seed, say). Widen first, then fit, so
  // the new cell can actually be learned instead of being folded into the
  // intercept forever.
  bool widened = false;
  for (const auto& r : pending_) {
    if (Index(&types_, r.inputs.data_type, true) != std::string::npos) {}
    if (Index(&formats_, r.inputs.data_format, true) != std::string::npos) {}
    if (Index(&libraries_, r.inputs.library, true) != std::string::npos) {}
    if (Index(&distributions_, r.inputs.distribution, true) !=
        std::string::npos) {}
  }
  if (Dimension() != comp_speed_.Weights().size()) {
    Rebuild(Dimension());
    widened = true;
  }
  (void)widened;

  const double forget = config_.forget_factor;
  for (const auto& r : pending_) {
    const std::vector<double> x = Encode(r.inputs);
    if (r.bytes > 0.0 && r.compress_time_ms > 0.0) {
      comp_speed_.Update(x, r.bytes / (r.compress_time_ms * 1000.0), forget);
    }
    if (r.bytes > 0.0 && r.decompress_time_ms > 0.0) {
      decomp_speed_.Update(x, r.bytes / (r.decompress_time_ms * 1000.0),
                           forget);
    }
    if (r.compression_ratio > 0.0) {
      ratio_.Update(x, RatioTarget(r.compression_ratio, config_.ratio_target_cap),
                    forget);
    }
  }
  const size_t n = pending_.size();
  pending_.clear();
  ++feedback_updates_;
  return n;
}

// ---------------------------------------------------------------------------
// Prediction
// ---------------------------------------------------------------------------
CompressionPrediction HCompressCcpPredictor::PredictWith(
    const CcpInputs& inputs, double bytes) const {
  const auto t0 = std::chrono::high_resolution_clock::now();
  CompressionPrediction out;
  std::vector<double> x;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    x = Encode(inputs);
    const double cs = std::max(config_.min_speed_mbps, comp_speed_.Predict(x));
    const double ds =
        std::max(config_.min_speed_mbps, decomp_speed_.Predict(x));
    // bytes / (MB/s * 1000) = ms. The regression sees no size; the size enters
    // only here, which is the whole reason the paper can regress a speed.
    out.compression_time_ms = (bytes > 0.0) ? bytes / (cs * 1000.0) : 0.0;
    out.decompression_time_ms = (bytes > 0.0) ? bytes / (ds * 1000.0) : 0.0;
    out.compression_ratio = std::max(
        config_.min_ratio, std::min(config_.max_ratio, ratio_.Predict(x)));
  }
  // NOT PREDICTED, and not 0: this codebase reads psnr_db == 0 as "measured and
  // lossless", i.e. maximal quality. The HCompress model has no quality output
  // at all, so it must report absence.
  out.psnr_db = -1.0;
  out.inference_time_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::high_resolution_clock::now() - t0)
          .count();
  return out;
}

CompressionPrediction HCompressCcpPredictor::PredictFor(
    const std::string& library, double bytes) const {
  CcpInputs in = inputs_;
  in.library = library;
  return PredictWith(in, bytes);
}

CompressionPrediction HCompressCcpPredictor::Predict(
    const CompressionFeatures& features) {
  CcpInputs in = inputs_;
  in.library = LibraryKey(features);
  return PredictWith(in, features.chunk_size_bytes);
}

std::vector<CompressionPrediction> HCompressCcpPredictor::PredictBatch(
    const std::vector<CompressionFeatures>& batch) {
  std::vector<CompressionPrediction> out;
  out.reserve(batch.size());
  for (const auto& f : batch) out.push_back(Predict(f));
  return out;
}

bool HCompressCcpPredictor::IsReady() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return comp_speed_.Ready() || ratio_.Ready();
}

std::vector<std::string> HCompressCcpPredictor::Vocabulary(
    const std::string& field) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (field == "data_type") return types_;
  if (field == "data_format") return formats_;
  if (field == "library") return libraries_;
  if (field == "distribution") return distributions_;
  return {};
}

// ---------------------------------------------------------------------------
// The JSON seed
// ---------------------------------------------------------------------------
namespace {
void WriteStrings(std::ostream& os, const char* key,
                  const std::vector<std::string>& v) {
  os << "  \"" << key << "\": [";
  for (size_t i = 0; i < v.size(); ++i) {
    os << (i ? ", " : "") << '"' << v[i] << '"';
  }
  os << "],\n";
}

void WriteDoubles(std::ostream& os, const char* key,
                  const std::vector<double>& v, bool last) {
  os << "  \"" << key << "\": [";
  os.precision(17);
  for (size_t i = 0; i < v.size(); ++i) os << (i ? ", " : "") << v[i];
  os << "]" << (last ? "\n" : ",\n");
}

/** The value text for "key", or empty. Flat object only, which is all this
 *  file is -- deliberately, so the seed stays readable and parseable without
 *  pulling a JSON library into the compress layer. */
std::string Field(const std::string& s, const std::string& key) {
  const size_t k = s.find("\"" + key + "\"");
  if (k == std::string::npos) return "";
  size_t c = s.find(':', k);
  if (c == std::string::npos) return "";
  ++c;
  while (c < s.size() && isspace(static_cast<unsigned char>(s[c]))) ++c;
  if (c >= s.size()) return "";
  if (s[c] == '[') {
    const size_t e = s.find(']', c);
    return e == std::string::npos ? "" : s.substr(c + 1, e - c - 1);
  }
  const size_t e = s.find_first_of(",\n}", c);
  return s.substr(c, (e == std::string::npos ? s.size() : e) - c);
}

std::vector<std::string> SplitStrings(const std::string& body) {
  std::vector<std::string> out;
  size_t i = 0;
  while (true) {
    const size_t a = body.find('"', i);
    if (a == std::string::npos) break;
    const size_t b = body.find('"', a + 1);
    if (b == std::string::npos) break;
    out.push_back(body.substr(a + 1, b - a - 1));
    i = b + 1;
  }
  return out;
}

std::vector<double> SplitDoubles(const std::string& body) {
  std::vector<double> out;
  std::istringstream is(body);
  std::string tok;
  while (std::getline(is, tok, ',')) {
    try {
      out.push_back(std::stod(tok));
    } catch (...) {
    }
  }
  return out;
}
}  // namespace

bool HCompressCcpPredictor::Save(const std::string& model_dir) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::ofstream f(model_dir + "/hcompress_ccp_seed.json");
  if (!f.is_open()) return false;
  f << "{\n";
  f << "  \"model_type\": \"hcompress_ccp\",\n";
  f << "  \"version\": \"1.0\",\n";
  f << "  \"note\": \"Expected Compression Cost, HCompress (IPDPS 2020) Sec."
       " IV-D: linear regression on data type, data format, compression"
       " library and data distribution; outputs compression speed (MB/s),"
       " decompression speed (MB/s) and compression ratio; no quality"
       " output.\",\n";
  f << "  \"seed_rows\": " << seed_rows_ << ",\n";
  f << "  \"regularization\": " << config_.regularization << ",\n";
  f << "  \"forget_factor\": " << config_.forget_factor << ",\n";
  f << "  \"feedback_interval\": " << config_.feedback_interval << ",\n";
  f << "  \"feedback_updates\": " << feedback_updates_ << ",\n";
  f << "  \"ratio_target_cap\": " << config_.ratio_target_cap << ",\n";
  WriteStrings(f, "data_types", types_);
  WriteStrings(f, "data_formats", formats_);
  WriteStrings(f, "libraries", libraries_);
  WriteStrings(f, "distributions", distributions_);
  f << "  \"samples\": [" << comp_speed_.Samples() << ", "
    << decomp_speed_.Samples() << ", " << ratio_.Samples() << "],\n";
  WriteDoubles(f, "w_compress_speed_mbps", comp_speed_.Weights(), false);
  WriteDoubles(f, "w_decompress_speed_mbps", decomp_speed_.Weights(), false);
  WriteDoubles(f, "w_compression_ratio", ratio_.Weights(), true);
  f << "}\n";
  return static_cast<bool>(f);
}

bool HCompressCcpPredictor::Load(const std::string& model_dir) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::ifstream f(model_dir + "/hcompress_ccp_seed.json");
  if (!f.is_open()) return false;
  std::stringstream ss;
  ss << f.rdbuf();
  const std::string s = ss.str();

  types_ = SplitStrings(Field(s, "data_types"));
  formats_ = SplitStrings(Field(s, "data_formats"));
  libraries_ = SplitStrings(Field(s, "libraries"));
  distributions_ = SplitStrings(Field(s, "distributions"));
  const std::vector<double> samples = SplitDoubles(Field(s, "samples"));
  const std::vector<double> wc = SplitDoubles(Field(s, "w_compress_speed_mbps"));
  const std::vector<double> wd =
      SplitDoubles(Field(s, "w_decompress_speed_mbps"));
  const std::vector<double> wr = SplitDoubles(Field(s, "w_compression_ratio"));
  const std::string rows = Field(s, "seed_rows");
  if (!rows.empty()) {
    try {
      seed_rows_ = static_cast<size_t>(std::stoull(rows));
    } catch (...) {
    }
  }

  const size_t dim = Dimension();
  if (wc.size() != dim || wd.size() != dim || wr.size() != dim) return false;
  comp_speed_.Reset(dim, config_.regularization);
  decomp_speed_.Reset(dim, config_.regularization);
  ratio_.Reset(dim, config_.regularization);
  comp_speed_.SetWeights(wc, samples.size() > 0 ? size_t(samples[0]) : 1);
  decomp_speed_.SetWeights(wd, samples.size() > 1 ? size_t(samples[1]) : 1);
  ratio_.SetWeights(wr, samples.size() > 2 ? size_t(samples[2]) : 1);
  return true;
}

}  // namespace ctp::compress::model
