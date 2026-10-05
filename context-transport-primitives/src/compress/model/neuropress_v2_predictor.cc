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
 * @file neuropress_v2_predictor.cc
 * @brief NeuroPress v2 host side: loading, the host network, ranking and
 * online learning. The GPU ranking is in neuropress_v2_kernels.cu.
 */

#include "clio_ctp/compress/model/neuropress_v2_predictor.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <numeric>
#include <sstream>
#include <sys/stat.h>

#include "clio_ctp/compress/gpu_setting_codec.h"
#include "clio_ctp/compress/preprocess/data_stats.h"
#include "clio_ctp/compress/preprocess/data_stats_gpu.h"
#include "neuropress_v2_kernels.h"

namespace ctp::compress::model {

namespace {

constexpr uint32_t kMagic = 0x4E4E5754;  // 'NNWT'
constexpr uint32_t kVersion = 3;

/** @brief Little-endian reader over a whole file in memory. */
class Reader {
 public:
  explicit Reader(std::vector<char> buf) : buf_(std::move(buf)) {}
  /** @return false when fewer than n bytes are left. */
  bool Take(void *dst, size_t n) {
    if (pos_ + n > buf_.size()) return false;
    std::memcpy(dst, buf_.data() + pos_, n);
    pos_ += n;
    return true;
  }
  /** Read one u32. */
  bool U32(uint32_t *v) { return Take(v, sizeof(*v)); }
  /** Append n floats to out. */
  bool Floats(size_t n, std::vector<float> *out) {
    const size_t at = out->size();
    out->resize(at + n);
    return Take(out->data() + at, n * sizeof(float));
  }
  /** @return true when every byte was consumed. */
  bool AtEnd() const { return pos_ == buf_.size(); }

 private:
  std::vector<char> buf_;
  size_t pos_ = 0;
};

/** @return true when path names a directory. */
bool IsDirectory(const std::string &path) {
  struct stat st;
  return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

/** @return the whole file, or an empty vector on failure. */
std::vector<char> ReadFile(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return {};
  return std::vector<char>(std::istreambuf_iterator<char>(f),
                           std::istreambuf_iterator<char>());
}

/** @return the '\n'-separated settings of a table. */
std::vector<std::string> SplitLines(const std::string &table) {
  std::vector<std::string> out;
  std::stringstream ss(table);
  for (std::string line; std::getline(ss, line, '\n');) out.push_back(line);
  return out;
}

}  // namespace

NeuroPressV2Predictor::NeuroPressV2Predictor() = default;

NeuroPressV2Predictor::~NeuroPressV2Predictor() {
  v2::FreeDevice(d_params_);
  v2::FreeDevice(d_available_);
  v2::FreeDevice(d_abs_err_);
  v2::DestroyStreamAndEvent(train_stream_, train_event_);
  for (int i = 0; i < kTimerSlots; ++i) {
    v2::DestroyEvent(t_start_[i]);
    v2::DestroyEvent(t_stop_[i]);
  }
}

std::string NeuroPressV2Predictor::ResolvePath(const std::string &path_or_dir) {
  if (!IsDirectory(path_or_dir)) return path_or_dir;
  return path_or_dir + "/model_v2.nnwt";
}

bool NeuroPressV2Predictor::IsV2File(const std::string &path_or_dir) {
  std::ifstream f(ResolvePath(path_or_dir), std::ios::binary);
  uint32_t head[2] = {0, 0};
  if (!f.read(reinterpret_cast<char *>(head), sizeof(head))) return false;
  return head[0] == kMagic && head[1] == kVersion;
}

bool NeuroPressV2Predictor::ParseFile(const std::string &path) {
  std::vector<char> bytes = ReadFile(path);
  if (bytes.empty()) return (error_ = "cannot read " + path), false;
  Reader r(std::move(bytes));
  uint32_t magic = 0, version = 0, n_layers = 0;
  if (!r.U32(&magic) || !r.U32(&version) || !r.U32(&n_layers) ||
      magic != kMagic || version != kVersion || n_layers == 0 ||
      n_layers > static_cast<uint32_t>(v2::kMaxLayers)) {
    return (error_ = path + " is not an NNWT version-3 model"), false;
  }
  dims_.assign(n_layers + 1, 0);
  for (auto &d : dims_) {
    uint32_t v = 0;
    if (!r.U32(&v) || v == 0 || v > static_cast<uint32_t>(v2::kMaxWidth)) {
      return (error_ = "bad layer width in " + path), false;
    }
    d = static_cast<int>(v);
  }
  uint32_t fset = 0, per = 0, n_set = 0, tbytes = 0;
  if (!r.U32(&fset) || !r.U32(&per) || !r.U32(&n_set) || !r.U32(&tbytes) ||
      fset != kFeatureSet || dims_[0] != 4 ||
      per != static_cast<uint32_t>(kOutputsPerSetting) ||
      dims_.back() != static_cast<int>(per * n_set)) {
    return (error_ = "feature set or output layout not supported"), false;
  }
  std::string table(tbytes, '\0');
  if (!r.Take(table.data(), tbytes)) return (error_ = "truncated table"), false;
  const auto settings = SplitLines(table);
  if (settings.size() != n_set ||
      n_set != static_cast<uint32_t>(ctp::kGpuSettingCount)) {
    return (error_ = "the model's settings are not the codec's 45"), false;
  }
  for (uint32_t i = 0; i < n_set; ++i) {
    if (settings[i] != ctp::GpuSettingSpec(static_cast<int>(i))) {
      return (error_ = "setting " + std::to_string(i) + " is '" + settings[i] +
                       "', the codec's is '" + ctp::GpuSettingSpec(i) + "'"),
             false;
    }
  }
  n_layers_ = static_cast<int>(n_layers);
  n_settings_ = static_cast<int>(n_set);
  params_.clear();
  const size_t in = dims_[0], out = dims_.back();
  xm_off_ = 0, xs_off_ = in, ym_off_ = 2 * in, ys_off_ = 2 * in + out;
  if (!r.Floats(2 * in + 2 * out, &params_)) return (error_ = "truncated"), false;
  w_off_.assign(n_layers_, 0);
  b_off_.assign(n_layers_, 0);
  for (int l = 0; l < n_layers_; ++l) {
    w_off_[l] = params_.size();
    if (!r.Floats(static_cast<size_t>(dims_[l + 1]) * dims_[l], &params_)) {
      return (error_ = "truncated weights"), false;
    }
    b_off_[l] = params_.size();
    if (!r.Floats(dims_[l + 1], &params_)) return (error_ = "truncated"), false;
  }
  if (!r.AtEnd()) return (error_ = "trailing bytes in " + path), false;
  return true;
}

bool NeuroPressV2Predictor::Load(const std::string &path_or_dir) {
  std::lock_guard<std::mutex> lock(train_mutex_);
  ready_ = false;
  error_.clear();
  updates_ = 0;
  if (!ParseFile(ResolvePath(path_or_dir))) return false;
  available_.assign(n_settings_, 0);
  for (int s = 0; s < n_settings_; ++s) {
    available_[s] = ctp::GpuSettingAvailable(s) ? 1 : 0;
  }
  v2::FreeDevice(d_params_);
  v2::FreeDevice(d_available_);
  d_params_ = nullptr;
  d_available_ = nullptr;
  void *p = nullptr, *a = nullptr;
  if (!v2::Upload(params_.data(), params_.size() * sizeof(float), &p) ||
      !v2::Upload(available_.data(), available_.size(), &a)) {
    v2::FreeDevice(p);
    return (error_ = "cannot upload the model to the GPU"), false;
  }
  d_params_ = static_cast<float *>(p);
  d_available_ = static_cast<unsigned char *>(a);
  if (train_stream_ == nullptr &&
      !v2::CreateStreamAndEvent(&train_stream_, &train_event_)) {
    return (error_ = "cannot create the GPU update stream"), false;
  }
  for (int k = 0; k < kTimerSlots; ++k) {
    if (t_start_[k] == nullptr) t_start_[k] = v2::CreateTimingEvent();
    if (t_stop_[k] == nullptr) t_stop_[k] = v2::CreateTimingEvent();
  }
  if (d_abs_err_ == nullptr) {
    const double zero = 0.0;
    void *e = nullptr;
    if (!v2::Upload(&zero, sizeof(zero), &e)) {
      return (error_ = "cannot allocate the update's error slot"), false;
    }
    d_abs_err_ = static_cast<double *>(e);
  }
  host_dirty_ = false;
  ready_ = true;
  return true;
}

NeuroPressV2Stats NeuroPressV2Predictor::ComputeStats(const float *data,
                                                      size_t n) {
  NeuroPressV2Stats s;
  if (data == nullptr || n == 0) return s;
  size_t hist[256] = {};
  const auto *bytes = reinterpret_cast<const unsigned char *>(data);
  for (size_t i = 0; i < n * sizeof(float); ++i) ++hist[bytes[i]];
  const double total = static_cast<double>(n * sizeof(float));
  for (size_t h : hist) {
    if (h == 0) continue;
    const double p = static_cast<double>(h) / total;
    s.entropy -= p * std::log2(p);
  }
  double sum = 0.0;
  s.vmin = s.vmax = data[0];
  for (size_t i = 0; i < n; ++i) {
    const double v = data[i];
    sum += v;
    s.vmin = std::min(s.vmin, v);
    s.vmax = std::max(s.vmax, v);
  }
  const double mean = sum / static_cast<double>(n);
  double dev = 0.0, d2 = 0.0;
  for (size_t i = 0; i < n; ++i) dev += std::fabs(data[i] - mean);
  for (size_t i = 2; i < n; ++i) {
    d2 += std::fabs(static_cast<double>(data[i]) - 2.0 * data[i - 1] +
                    static_cast<double>(data[i - 2]));
  }
  s.mad = dev / static_cast<double>(n);
  s.d2 = n > 2 ? d2 / static_cast<double>(n - 2) : 0.0;
  return s;
}

namespace {
/** This thread's last conversion time, ms. */
double &LastConvertMsSlot() {
  static thread_local double ms = 0.0;
  return ms;
}
}  // namespace

double NeuroPressV2Predictor::LastConvertMs() { return LastConvertMsSlot(); }

size_t NeuroPressV2Predictor::DtypeBytes(int dtype) {
  switch (dtype) {
    case 2: case 11: case 12: return 8;
    case 3: case 4: case 7: case 8: return 2;
    case 5: case 6: return 1;
    default: return 4;
  }
}

namespace {

/** IEEE half to float (normals, subnormals, inf and NaN). */
float HalfToFloat(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t mant = h & 0x3FFu;
  uint32_t bits;
  if (exp == 0x1Fu) {
    bits = sign | 0x7F800000u | (mant << 13);
  } else if (exp != 0) {
    bits = sign | ((exp + 112u) << 23) | (mant << 13);
  } else if (mant == 0) {
    bits = sign;
  } else {
    exp = 113;  // normalise the subnormal
    while ((mant & 0x400u) == 0) {
      mant <<= 1;
      --exp;
    }
    bits = sign | (exp << 23) | ((mant & 0x3FFu) << 13);
  }
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

/** One host element of a dtype code as float. */
float HostElement(const unsigned char *p, size_t i, int dtype) {
  switch (dtype) {
    case 2: { double v; std::memcpy(&v, p + 8 * i, 8); return static_cast<float>(v); }
    case 3: { uint16_t v; std::memcpy(&v, p + 2 * i, 2); return HalfToFloat(v); }
    case 4: {
      uint16_t v;
      std::memcpy(&v, p + 2 * i, 2);
      const uint32_t bits = static_cast<uint32_t>(v) << 16;
      float f;
      std::memcpy(&f, &bits, 4);
      return f;
    }
    case 5: return static_cast<float>(static_cast<int8_t>(p[i]));
    case 6: return static_cast<float>(p[i]);
    case 7: { int16_t v; std::memcpy(&v, p + 2 * i, 2); return static_cast<float>(v); }
    case 8: { uint16_t v; std::memcpy(&v, p + 2 * i, 2); return static_cast<float>(v); }
    case 9: { int32_t v; std::memcpy(&v, p + 4 * i, 4); return static_cast<float>(v); }
    case 10: { uint32_t v; std::memcpy(&v, p + 4 * i, 4); return static_cast<float>(v); }
    case 11: { int64_t v; std::memcpy(&v, p + 8 * i, 8); return static_cast<float>(v); }
    case 12: { uint64_t v; std::memcpy(&v, p + 8 * i, 8); return static_cast<float>(v); }
    default: { float v; std::memcpy(&v, p + 4 * i, 4); return v; }
  }
}

}  // namespace

std::vector<float> NeuroPressV2Predictor::ToFloat32Host(const void *data,
                                                        size_t bytes,
                                                        int dtype) {
  const size_t n = bytes / DtypeBytes(dtype);
  std::vector<float> out(n);
  const auto *p = static_cast<const unsigned char *>(data);
  for (size_t i = 0; i < n; ++i) out[i] = HostElement(p, i, dtype);
  return out;
}

const void *NeuroPressV2Predictor::DeviceStatsAsFloat32(const void *chunk,
                                                        size_t bytes,
                                                        int dtype,
                                                        void *stream) {
  const size_t n = bytes / DtypeBytes(dtype);
  if (n == 0) return nullptr;
  // float32 (1) and unknown codes are read as float32 bytes as they are.
  const float *values = static_cast<const float *>(chunk);
  LastConvertMsSlot() = 0.0;
  if (dtype >= 2 && dtype <= 12) {
    values = v2::ConvertToFloat32(chunk, n, dtype, stream, &LastConvertMsSlot());
    if (values == nullptr) return nullptr;
  }
  return ctp::ComputeDeviceStatsResident(values, n, ctp::DataType::FLOAT32,
                                         stream);
}

NeuroPressV2Features NeuroPressV2Predictor::MakeFeatures(
    size_t chunk_bytes, const NeuroPressV2Stats &s) {
  double range = s.vmax - s.vmin;
  if (!(range > 0.0)) range = 1.0;
  NeuroPressV2Features f;
  f.x[0] = static_cast<float>(std::log2(static_cast<double>(chunk_bytes)));
  f.x[1] = static_cast<float>(s.entropy);
  f.x[2] = static_cast<float>(std::log10(s.mad / range + 1e-12));
  f.x[3] = static_cast<float>(std::log10(s.d2 / range + 1e-12));
  return f;
}

void NeuroPressV2Predictor::SyncHost() const {
  std::lock_guard<std::mutex> lock(host_mutex_);
  if (!host_dirty_) return;
  if (v2::CopyToHost(params_.data(), d_params_, params_.size() * sizeof(float),
                     train_stream_)) {
    host_dirty_ = false;
  }
}

void NeuroPressV2Predictor::FillDesc(v2::NetDesc *d) const {
  d->n_layers = n_layers_;
  for (int l = 0; l <= n_layers_; ++l) d->dims[l] = dims_[l];
  for (int l = 0; l < n_layers_; ++l) {
    d->w_off[l] = w_off_[l];
    d->b_off[l] = b_off_[l];
  }
  d->xm_off = xm_off_, d->xs_off = xs_off_, d->ym_off = ym_off_;
  d->ys_off = ys_off_;
  d->n_settings = n_settings_;
}

void NeuroPressV2Predictor::ForwardHostFull(const NeuroPressV2Features &f,
                                            std::vector<float> *out) const {
  SyncHost();
  std::vector<float> h(dims_[0]);
  for (int i = 0; i < dims_[0]; ++i) {
    h[i] = (f.x[i] - params_[xm_off_ + i]) / params_[xs_off_ + i];
  }
  for (int l = 0; l < n_layers_; ++l) {
    const int n_in = dims_[l], n_out = dims_[l + 1];
    const float *w = params_.data() + w_off_[l];
    const float *b = params_.data() + b_off_[l];
    std::vector<float> next(n_out);
    for (int o = 0; o < n_out; ++o) {
      float s = b[o];
      for (int i = 0; i < n_in; ++i) s += w[o * n_in + i] * h[i];
      next[o] = (l + 1 < n_layers_) ? std::max(0.0f, s) : s;
    }
    h.swap(next);
  }
  out->resize(h.size());
  for (size_t o = 0; o < h.size(); ++o) {
    (*out)[o] = h[o] * params_[ys_off_ + o] + params_[ym_off_ + o];
  }
}

void NeuroPressV2Predictor::ForwardHost(const NeuroPressV2Features &f,
                                        std::vector<float> *out) const {
  ForwardHostFull(f, out);
}

double NeuroPressV2Predictor::Cost(int s, double ct, double dt, double ratio,
                                   size_t chunk_bytes,
                                   const NeuroPressV2CostWeights &w) const {
  const double c = w.w_ct * ct + w.w_dt * dt +
                   w.w_io * static_cast<double>(chunk_bytes) /
                       (ratio * w.bw_bytes_per_ms);
  if (!available_[s] || !std::isfinite(c) || !(ratio > 0.0)) {
    return std::numeric_limits<double>::infinity();
  }
  return c;
}

std::vector<NeuroPressV2Prediction> NeuroPressV2Predictor::RankHost(
    const NeuroPressV2Features &f, size_t chunk_bytes,
    const NeuroPressV2CostWeights &w) const {
  if (!ready_) return {};
  std::vector<float> y;
  ForwardHost(f, &y);
  std::vector<NeuroPressV2Prediction> out(n_settings_);
  for (int s = 0; s < n_settings_; ++s) {
    auto &p = out[s];
    p.setting = s;
    p.comp_ms = std::exp(static_cast<double>(y[3 * s]));
    p.decomp_ms = std::exp(static_cast<double>(y[3 * s + 1]));
    p.ratio = std::exp(static_cast<double>(y[3 * s + 2]));
    p.cost = Cost(s, p.comp_ms, p.decomp_ms, p.ratio, chunk_bytes, w);
  }
  std::stable_sort(out.begin(), out.end(),
                   [](const NeuroPressV2Prediction &a,
                      const NeuroPressV2Prediction &b) { return a.cost < b.cost; });
  return out;
}

std::vector<NeuroPressV2Prediction> NeuroPressV2Predictor::RankDevice(
    const void *device_stats, size_t chunk_bytes,
    const NeuroPressV2CostWeights &w, void *stream,
    NeuroPressV2Features *features_out) {
  if (!ready_ || device_stats == nullptr) return {};
  v2::NetDesc d;
  FillDesc(&d);
  // The ranking stream waits for the last GPU update (no host sync).
  if (!v2::StreamWaitEvent(stream, train_event_)) return {};
  v2::RankArgs a;
  a.w_ct = w.w_ct, a.w_dt = w.w_dt, a.w_io = w.w_io;
  a.bw_bytes_per_ms = w.bw_bytes_per_ms;
  a.chunk_bytes = static_cast<double>(chunk_bytes);
  v2::RankOut r;
  if (!v2::RankOnDevice(d_params_, d, d_available_, device_stats, a, stream,
                        &r)) {
    return {};
  }
  if (features_out != nullptr) std::copy(r.x, r.x + 4, features_out->x);
  std::vector<NeuroPressV2Prediction> out(n_settings_);
  for (int k = 0; k < n_settings_; ++k) {
    const int s = r.order[k];
    out[k] = {s, r.ct[s], r.dt[s], r.ratio[s], r.cost[s]};
  }
  return out;
}

bool NeuroPressV2Predictor::TrainSetting(const NeuroPressV2Features &f,
                                         int setting, double comp_ms,
                                         double decomp_ms, double ratio,
                                         double lr, double *abs_err) {
  if (!ready_ || setting < 0 || setting >= n_settings_ || !(lr > 0.0)) {
    return false;
  }
  v2::TrainArgs a;
  std::copy(f.x, f.x + 4, a.x);
  a.setting = setting;
  a.label[0] = comp_ms;
  a.label[1] = decomp_ms;
  a.label[2] = ratio;
  a.lr = lr;
  int labelled = 0;
  for (double v : a.label) labelled += (v > 0.0 && std::isfinite(v)) ? 1 : 0;
  if (labelled == 0) return false;
  v2::NetDesc d;
  FillDesc(&d);
  std::lock_guard<std::mutex> lock(train_mutex_);
  // A slot comes back after kTimerSlots updates; its update is long done.
  const int slot = t_next_;
  t_next_ = (t_next_ + 1) % kTimerSlots;
  HarvestTimer(slot, /*wait=*/true);
  if (!v2::TrainOnDevice(d_params_, d, a, train_stream_, train_event_,
                         d_abs_err_, t_start_[slot], t_stop_[slot])) {
    return false;
  }
  t_pending_[slot] = t_start_[slot] != nullptr && t_stop_[slot] != nullptr;
  {
    std::lock_guard<std::mutex> host_lock(host_mutex_);
    host_dirty_ = true;
  }
  ++updates_;
  if (abs_err != nullptr &&
      !v2::CopyToHost(abs_err, d_abs_err_, sizeof(double), train_stream_)) {
    return false;
  }
  return true;
}

void NeuroPressV2Predictor::HarvestTimer(int slot, bool wait) {
  if (!t_pending_[slot]) return;
  if (!(wait ? v2::EventSync(t_stop_[slot]) : v2::EventDone(t_stop_[slot]))) {
    return;
  }
  const double ms = v2::EventElapsedMs(t_start_[slot], t_stop_[slot]);
  if (ms >= 0.0) train_gpu_ms_ += ms;
  t_pending_[slot] = false;
}

double NeuroPressV2Predictor::TakeTrainGpuMs() {
  std::lock_guard<std::mutex> lock(train_mutex_);
  for (int k = 0; k < kTimerSlots; ++k) HarvestTimer(k, /*wait=*/false);
  const double ms = train_gpu_ms_;
  train_gpu_ms_ = 0.0;
  return ms;
}

namespace {
/** The per-thread event pair behind GpuTimerStart/GpuTimerStopMs. */
struct ThreadGpuTimer {
  void *start = nullptr;
  void *stop = nullptr;
  bool on = false;
};
ThreadGpuTimer &TimerOfThisThread() {
  static thread_local ThreadGpuTimer t;
  return t;
}
}  // namespace

void NeuroPressV2Predictor::GpuTimerStart(void *stream) {
  ThreadGpuTimer &t = TimerOfThisThread();
  if (t.start == nullptr) t.start = v2::CreateTimingEvent();
  if (t.stop == nullptr) t.stop = v2::CreateTimingEvent();
  t.on = v2::RecordEvent(t.start, stream);
}

double NeuroPressV2Predictor::GpuTimerStopMs(void *stream) {
  ThreadGpuTimer &t = TimerOfThisThread();
  if (!t.on) return -1.0;
  t.on = false;
  if (!v2::RecordEvent(t.stop, stream) || !v2::EventSync(t.stop)) return -1.0;
  return v2::EventElapsedMs(t.start, t.stop);
}

}  // namespace ctp::compress::model
