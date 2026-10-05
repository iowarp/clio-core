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
 * @file neuropress_v2_selection.cc
 * @brief NeuroPress v2 in the compressor: ranking a chunk over the 45 GPU
 * settings, online learning from measured results (write and read),
 * exploration of the next-ranked settings, per-chunk storage tiers and the
 * v2 logs. v1 is untouched; exactly one of the two is loaded (see Create).
 */

#include "clio_cte/compressor/async_log.h"

#include <clio_ctp/compress/compress_factory.h>
#include <clio_ctp/compress/gpu_setting_codec.h>
#include <clio_ctp/compress/preprocess/data_stats.h>
#include <clio_ctp/compress/preprocess/data_stats_gpu.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <unordered_map>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <vector>

#include "clio_cte/compressor/compressor_runtime.h"
#include "clio_cte/compressor/neuropress_path_trace.h"
#include "clio_cte/compressor/neuropress_telemetry.h"

namespace clio::cte::compressor {

namespace {

constexpr int kNpSettingWire = 24;  ///< "np-setting", compress_factory.h

/** @return the store setting's index (stored raw, wire 0). */
int StoreSetting() {
  static const int idx = ctp::GpuSettingIndex("store");
  return idx;
}

/**
 * @return the cost of a measured or predicted outcome under w; +inf when the
 *         ratio is not positive
 */
double V2Cost(const ctp::compress::model::NeuroPressV2CostWeights &w,
              double comp_ms, double decomp_ms, double ratio, double bytes) {
  if (!(ratio > 0.0)) return std::numeric_limits<double>::infinity();
  return w.w_ct * comp_ms + w.w_dt * decomp_ms +
         w.w_io * bytes / (ratio * w.bw_bytes_per_ms);
}

/** @brief This thread's last v2 selection timing. */
struct SelectTiming {
  double select_ms = 0.0;   ///< wall ms, conversion excluded
  double convert_ms = 0.0;  ///< float32 conversion ms (excluded everywhere)
};
thread_local SelectTiming g_select_timing;

/** @return CLIO_NEUROPRESS_TIERS as one bandwidth per round-robin slot. */
const std::vector<double> &TierPattern() {
  static const std::vector<double> pattern = [] {
    std::vector<double> p;
    const char *e = std::getenv("CLIO_NEUROPRESS_TIERS");
    if (e == nullptr) return p;
    std::stringstream ss(e);
    for (std::string item; std::getline(ss, item, ',');) {
      const size_t colon = item.find(':');
      const double bw = std::strtod(item.c_str(), nullptr);
      const long n = colon == std::string::npos
                         ? 1 : std::strtol(item.c_str() + colon + 1, nullptr, 10);
      for (long k = 0; bw > 0.0 && k < n; ++k) p.push_back(bw);
    }
    return p;
  }();
  return pattern;
}

/** @return a CSV log opened for appending, or nullptr when env is unset. */
std::ofstream *OpenLog(const char *env) {
  const char *path = std::getenv(env);
  if (path == nullptr || *path == '\0') return nullptr;
  auto *out = new std::ofstream(path, std::ios::app);
  *out << std::setprecision(9);
  return out;
}

}  // namespace

ctp::compress::model::NeuroPressV2CostWeights Runtime::V2CostWeights(
    double bw) const {
  const auto cw = NeuroPressResolvedCostWeights();
  ctp::compress::model::NeuroPressV2CostWeights w;
  w.w_ct = config_.neuropress_best_mode_ ? 0.0 : cw.ct;
  w.w_dt = config_.neuropress_best_mode_ ? 0.0 : cw.dt;
  w.w_io = cw.io;
  w.bw_bytes_per_ms = bw > 0.0 ? bw : cw.bw;
  return w;
}

double Runtime::NextV2TierBw() {
  static std::atomic<unsigned long long> seq{0};
  const auto &p = TierPattern();
  if (p.empty()) return 0.0;
  return p[seq.fetch_add(1, std::memory_order_relaxed) % p.size()];
}

void Runtime::TakeV2SelectTiming(double *select_ms, double *convert_ms) {
  *select_ms = g_select_timing.select_ms;
  *convert_ms = g_select_timing.convert_ms;
  g_select_timing = SelectTiming{};
}

void Runtime::LogV2Predictions(
    const std::string &blob, clio::run::u64 chunk_size,
    const ctp::compress::model::NeuroPressV2Features &f,
    const std::vector<CompressionStats> &stats, double bw, double select_ms,
    double convert_ms) {
  static std::ofstream *out = OpenLog("CLIO_NEUROPRESS_V2_PRED_LOG");
  if (out == nullptr || !neuropress_v2_) return;
  // Copies for the log thread (async_log.h): the row is formatted and written
  // there, off the compress path.
  constexpr int n = ctp::kGpuSettingCount;
  std::vector<std::array<double, 3>> vals(n);
  std::vector<char> have(n, 0);
  for (const auto &st : stats) {
    if (st.compress_preset_ >= 0 && st.compress_preset_ < n) {
      vals[st.compress_preset_] = {st.compress_time_ms_, st.decompress_time_ms_,
                                   static_cast<double>(st.compression_ratio_)};
      have[st.compress_preset_] = 1;
    }
  }
  const std::array<float, 4> x = {f.x[0], f.x[1], f.x[2], f.x[3]};
  PostLogWork([blob, chunk_size, bw_v = V2CostWeights(bw).bw_bytes_per_ms,
               select_ms, convert_ms, x, updates = neuropress_v2_->UpdateCount(),
               vals = std::move(vals), have = std::move(have)]() {
    static bool header = false;  // touched by the log thread only
    if (!header) {
      *out << "blob,bytes,tier_bw,select_ms,convert_ms,x0,x1,x2,x3,updates";
      for (int s = 0; s < n; ++s) *out << ",ct" << s << ",dt" << s << ",r" << s;
      *out << "\n";
      header = true;
    }
    *out << blob << "," << chunk_size << "," << bw_v << "," << select_ms << ","
         << convert_ms;
    for (float v : x) *out << "," << v;
    *out << "," << updates;
    for (int s = 0; s < n; ++s) {
      if (!have[s]) {
        *out << ",,,";
      } else {
        *out << "," << vals[s][0] << "," << vals[s][1] << "," << vals[s][2];
      }
    }
    *out << "\n";
    out->flush();
  });
}

void Runtime::LogV2Measured(const std::string &blob, double bw, int setting,
                            const char *role, double comp_ms, double decomp_ms,
                            double ratio, double cost, bool adopted) {
  static std::ofstream *out = OpenLog("CLIO_NEUROPRESS_V2_EXPLORE_LOG");
  if (out == nullptr) return;
  // Formatted and written on the log thread (async_log.h).
  PostLogWork([blob, bw_v = V2CostWeights(bw).bw_bytes_per_ms, setting,
               role = std::string(role), comp_ms, decomp_ms, ratio, cost,
               adopted]() {
    static bool header = false;  // touched by the log thread only
    if (!header) {
      *out << "blob,tier_bw,setting,spec,role,comp_ms,decomp_ms,ratio,cost,"
              "adopted\n";
      header = true;
    }
    *out << blob << "," << bw_v << "," << setting << ","
         << ctp::GpuSettingSpec(setting) << "," << role << "," << comp_ms << ","
         << decomp_ms << "," << ratio << "," << cost << "," << (adopted ? 1 : 0)
         << "\n";
    out->flush();
  });
}

#if CTP_ENABLE_NEUROPRESS_GPU

namespace {

/**
 * @brief The fixed-single-codec baseline's setting, from
 * CLIO_NEUROPRESS_FIXED_SETTING (a setting index or its specification).
 *
 * When set, every chunk is stored with that one setting and NeuroPress v2
 * does no work (no features, no network); unset, v2 chooses normally.
 *
 * @return the setting index, or -1 when unset or unknown
 */
int FixedV2Setting() {
  static const int fixed = [] {
    const char *e = std::getenv("CLIO_NEUROPRESS_FIXED_SETTING");
    if (e == nullptr || *e == '\0') return -1;
    char *end = nullptr;
    const long v = std::strtol(e, &end, 10);
    const int idx = (end != e && *end == '\0') ? static_cast<int>(v)
                                               : ctp::GpuSettingIndex(e);
    if (idx < 0 || idx >= ctp::kGpuSettingCount) {
      HLOG(kError, "CLIO_NEUROPRESS_FIXED_SETTING='{}' is not a setting; "
           "NeuroPress v2 chooses normally", e);
      return -1;
    }
    HLOG(kInfo, "NeuroPress v2 fixed-codec baseline: every chunk uses {}",
         ctp::GpuSettingSpec(idx));
    return idx;
  }();
  return fixed;
}

/** The blob whose chunk is being selected, set by the write path. */
thread_local std::string g_selection_blob;

/**
 * @brief The per-chunk setting map of the oracle baseline, from
 * CLIO_NEUROPRESS_SETTING_MAP: a CSV of "blob,setting" lines (setting index).
 *
 * When set, each chunk is stored with its mapped setting (the cheapest one
 * for that chunk, from an exhaustive search) and NeuroPress v2 does no work.
 *
 * @return the map, empty when unset or unreadable
 */
const std::unordered_map<std::string, int> &OracleV2Map() {
  static const std::unordered_map<std::string, int> map = [] {
    std::unordered_map<std::string, int> m;
    const char *path = std::getenv("CLIO_NEUROPRESS_SETTING_MAP");
    if (path == nullptr || *path == '\0') return m;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
      const auto comma = line.rfind(',');
      if (comma == std::string::npos) continue;
      char *end = nullptr;
      const long v = std::strtol(line.c_str() + comma + 1, &end, 10);
      if (end == line.c_str() + comma + 1) continue;  // header or bad line
      m[line.substr(0, comma)] = static_cast<int>(v);
    }
    HLOG(kInfo, "NeuroPress v2 oracle baseline: {} chunk settings from {}",
         m.size(), path);
    return m;
  }();
  return map;
}

}  // namespace

std::vector<CompressionStats> Runtime::NeuroPressV2RankChunk(
    const void *chunk, clio::run::u64 chunk_size, const Context &context,
    double *out_entropy, double *out_mad, double *out_second_deriv,
    bool *out_gpu_failed,
    ctp::compress::model::NeuroPressV2Features *out_features, double bw) {
  using ctp::compress::model::NeuroPressV2Predictor;
  // Oracle baseline: this chunk's own best setting, no prediction work.
  if (const auto &oracle = OracleV2Map(); !oracle.empty()) {
    const auto it = oracle.find(g_selection_blob);
    if (it != oracle.end() && it->second >= 0 && it->second < ctp::kGpuSettingCount) {
      RecordSelectionPhases(-1.0, 0.0, -1.0, /*reused=*/false, 0.0, 0.0);
      return {CompressionStats(it->second == StoreSetting() ? 0 : kNpSettingWire,
                               it->second, 0.0, 0.0, 0.0, 0.0)};
    }
    HLOG(kError, "NeuroPress v2 oracle baseline: no setting for '{}'",
         g_selection_blob);
  }
  // Fixed-single-codec baseline: one setting, no prediction work at all.
  if (const int fixed = FixedV2Setting(); fixed >= 0) {
    RecordSelectionPhases(-1.0, 0.0, -1.0, /*reused=*/false, 0.0, 0.0);
    return {CompressionStats(fixed == StoreSetting() ? 0 : kNpSettingWire,
                             fixed, 0.0, 0.0, 0.0, 0.0)};
  }
  const auto t0 = std::chrono::steady_clock::now();
  const auto w = V2CostWeights(bw);
  ctp::compress::model::NeuroPressV2Features f;
  std::vector<ctp::compress::model::NeuroPressV2Prediction> ranked;
  double entropy = 0.0, mad = 0.0, d2 = 0.0, convert_ms = 0.0;
  double nn_gpu_ms = -1.0;
  if (ctp::IsDevicePointer(chunk)) {
    // Converted to float32 by value on the GPU (timed, then excluded), then
    // the statistics and the ranking chain on the same stream.
    void *stream = ctp::DeviceStatsStream();
    // GPU time of features + network + ranking (CUDA events), conversion
    // (timed on its own) taken out.
    NeuroPressV2Predictor::GpuTimerStart(stream);
    const void *st = NeuroPressV2Predictor::DeviceStatsAsFloat32(
        chunk, chunk_size, context.data_type_, stream);
    convert_ms = NeuroPressV2Predictor::LastConvertMs();
    if (st != nullptr) {
      ranked = neuropress_v2_->RankDevice(st, chunk_size, w, stream, &f);
      ctp::ReadDeviceFeatureStats(st, &entropy, &mad, &d2, stream);
    }
    const double gpu_ms = NeuroPressV2Predictor::GpuTimerStopMs(stream);
    if (gpu_ms >= 0.0) nn_gpu_ms = std::max(0.0, gpu_ms - convert_ms);
    if (ranked.empty()) {
      HLOG(kError, "NeuroPress v2: device ranking failed for a {}-byte chunk",
           chunk_size);
      if (out_gpu_failed) *out_gpu_failed = true;
      return {};
    }
  } else {
    const auto c0 = std::chrono::steady_clock::now();
    const std::vector<float> values = NeuroPressV2Predictor::ToFloat32Host(
        chunk, chunk_size, context.data_type_);
    convert_ms = std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - c0).count();
    if (values.empty()) return {};
    const auto s = NeuroPressV2Predictor::ComputeStats(values.data(),
                                                       values.size());
    f = NeuroPressV2Predictor::MakeFeatures(chunk_size, s);
    ranked = neuropress_v2_->RankHost(f, chunk_size, w);
    entropy = s.entropy, mad = s.mad, d2 = s.d2;
  }
  const double wall = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0).count();
  g_select_timing = SelectTiming{std::max(0.0, wall - convert_ms), convert_ms};
  RecordSelectionPhases(-1.0, std::max(0.0, wall - convert_ms), -1.0,
                        /*reused=*/false, convert_ms, nn_gpu_ms);
  if (out_entropy) *out_entropy = entropy;
  if (out_mad) *out_mad = mad;
  if (out_second_deriv) *out_second_deriv = d2;
  if (out_features) *out_features = f;
  std::vector<CompressionStats> out;
  out.reserve(ranked.size());
  for (const auto &p : ranked) {
    if (!std::isfinite(p.cost)) continue;  // unavailable in this build
    out.emplace_back(p.setting == StoreSetting() ? 0 : kNpSettingWire,
                     p.setting, p.ratio, p.comp_ms, p.decomp_ms, 0.0);
  }
  CLIO_PATH_TRACE("2 infer    v2 ONE forward pass (4->64x4->135), %zu settings "
                  "ranked at %.3g B/ms; primary=%s", out.size(),
                  w.bw_bytes_per_ms,
                  out.empty() ? "-" : ctp::GpuSettingSpec(
                                          out.front().compress_preset_));
  return out;
}

double Runtime::NeuroPressV2LearnPrimary(
    const ctp::compress::model::NeuroPressV2Features &features,
    const CompressionStats &predicted, const Context &context,
    clio::run::u64 chunk_size, double bw, bool *trained) {
  if (trained) *trained = false;
  const auto w = V2CostWeights(bw);
  const double bytes = static_cast<double>(chunk_size);
  const bool dt_measured = context.actual_decompress_time_ms_ > 0.0;
  const double act_dt = dt_measured ? context.actual_decompress_time_ms_
                                    : predicted.decompress_time_ms_;
  const double pred_cost =
      V2Cost(w, predicted.compress_time_ms_, predicted.decompress_time_ms_,
             predicted.compression_ratio_, bytes);
  const double act_cost =
      V2Cost(w, context.actual_compress_time_ms_, act_dt,
             context.actual_compression_ratio_, bytes);
  const double err = (act_cost > 0.0 && std::isfinite(act_cost))
                         ? std::fabs(act_cost - pred_cost) / act_cost
                         : 0.0;
  if (config_.neuropress_online_learning_enabled_ &&
      !config_.neuropress_best_mode_ &&
      err > static_cast<double>(config_.neuropress_mape_threshold_)) {
    const bool ok = neuropress_v2_->TrainSetting(
        features, predicted.compress_preset_, context.actual_compress_time_ms_,
        dt_measured ? context.actual_decompress_time_ms_ : -1.0,
        context.actual_compression_ratio_, config_.neuropress_learning_rate_);
    if (trained) *trained = ok;
    HLOG(kDebug, "NeuroPress v2 SGD: setting={} cost_err={} trained={}",
         ctp::GpuSettingSpec(predicted.compress_preset_), err, ok);
  }
  return err;
}

namespace {

/**
 * @brief Compress one alternative into a device scratch buffer and, when
 * asked, decode it into another, timing both with the codec's CUDA events.
 *
 * Nothing goes through host memory: the compressed payload stays in `slot`.
 *
 * @param codec      the alternative's setting codec
 * @param chunk      the chunk (device or host)
 * @param chunk_size its size
 * @param slot       GpuSettingScratch slot that receives the payload
 * @param measure_dt also decode it (into kExploreDecodeSlot) and time that
 * @param comp       out: compressed bytes
 * @param ct         out: compress GPU ms
 * @param dt         out: decompress GPU ms, -1 when not measured
 * @return false when compression fails or the payload does not decode
 */
bool MeasureAlternative(ctp::Compressor *codec, const void *chunk,
                        size_t chunk_size, int slot, bool measure_dt,
                        size_t *comp, double *ct, double *dt) {
  const size_t cap = codec->MaxCompressedSize(chunk_size);
  void *d_comp = ctp::GpuSettingScratch(slot, cap);
  *comp = cap;
  if (cap == 0 || d_comp == nullptr ||
      !codec->Compress(d_comp, *comp, const_cast<void *>(chunk), chunk_size)) {
    return false;
  }
  *ct = ctp::LastCodecKernelMs();
  *dt = -1.0;
  if (!measure_dt) return true;
  void *d_back = ctp::GpuSettingScratch(kExploreDecodeSlot, chunk_size);
  size_t back_size = chunk_size;
  if (d_back == nullptr || !codec->Decompress(d_back, back_size, d_comp, *comp)) {
    return false;
  }
  *dt = ctp::LastCodecKernelMs();
  return true;
}

}  // namespace

void SetV2SelectionBlob(const std::string &blob) { g_selection_blob = blob; }

Runtime::V2ExploreWinner Runtime::NeuroPressV2Explore(
    const std::string &blob, double bw, const void *chunk,
    clio::run::u64 chunk_size, const std::vector<CompressionStats> &stats,
    int primary_setting, bool primary_raw, double primary_cost,
    bool measure_dt,
    const ctp::compress::model::NeuroPressV2Features &features) {
  V2ExploreWinner win;
  const auto w = V2CostWeights(bw);
  const double bytes = static_cast<double>(chunk_size);
  const bool learn = config_.neuropress_online_learning_enabled_ &&
                     !config_.neuropress_best_mode_;
  double best = primary_cost;
  int examined = 0;
  struct Measured { int setting; double ct, dt, ratio, cost; };
  std::vector<Measured> rows;
  // Payloads stay on the GPU: each alternative compresses into cur_slot and
  // decodes into kExploreDecodeSlot; a new winner swaps cur_slot and
  // best_slot, so its payload is kept without a copy.
  int cur_slot = kExploreCompSlotA, best_slot = kExploreCompSlotB;
  size_t best_comp = 0;
  for (const auto &alt : stats) {
    // A primary stored raw left no measurement of its setting: measure it
    // too, outside the K alternatives.
    const bool is_primary = alt.compress_preset_ == primary_setting;
    if (alt.compress_lib_ != kNpSettingWire || (is_primary && !primary_raw)) {
      continue;
    }
    if (!is_primary && examined++ >= config_.neuropress_exploration_k_) break;
    auto codec = ctp::CompressionFactory::GetGpuSetting(alt.compress_preset_);
    if (!codec) continue;
    size_t comp = 0;
    double ct = -1.0, dt = -1.0;
    if (!MeasureAlternative(codec.get(), chunk, chunk_size, cur_slot,
                            measure_dt, &comp, &ct, &dt)) {
      // Compression failed, or the setting could not decode its own output:
      // not a real measurement, so neither trained on nor adoptable.
      HLOG(kWarning, "NeuroPress v2 explore: {} failed its round trip on "
           "'{}'; skipped", ctp::GpuSettingSpec(alt.compress_preset_), blob);
      continue;
    }
    ++win.measured;
    win.gpu_ms += ct + (dt >= 0.0 ? dt : 0.0);
    const double ratio = bytes / static_cast<double>(comp);
    if (learn && ct >= 0.0) {
      const auto train_t0 = std::chrono::steady_clock::now();
      const bool ok = neuropress_v2_->TrainSetting(
          features, alt.compress_preset_, ct, dt, ratio,
          config_.neuropress_learning_rate_);
      win.train_ms += std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - train_t0)
                          .count();
      if (ok) ++win.trained;
    }
    const double cost =
        V2Cost(w, ct, dt >= 0.0 ? dt : alt.decompress_time_ms_, ratio, bytes);
    rows.push_back({alt.compress_preset_, ct, dt, ratio, cost});
    CLIO_PATH_TRACE("6 explore  v2 %s ratio=%.3f ct=%.3f ms cost=%.4f vs %.4f",
                    ctp::GpuSettingSpec(alt.compress_preset_), ratio, ct, cost,
                    best);
    if (cost < best && comp < chunk_size) {
      best = cost;
      win.have = true;
      win.setting = alt.compress_preset_;
      best_comp = comp;
      std::swap(cur_slot, best_slot);
      win.ratio = ratio;
      win.comp_ms = ct;
      win.decomp_ms = dt;
      win.cost = cost;
    }
  }
  // The winner's payload to the host, once: Clio's put takes host memory.
  if (win.have) {
    win.payload.resize(best_comp);
    if (!ctp::GpuSettingCopyToHost(win.payload.data(),
                                   ctp::GpuSettingScratch(best_slot, best_comp),
                                   best_comp)) {
      win.have = false;  // keep the primary
      win.payload.clear();
    }
  }
  for (const auto &r : rows) {
    LogV2Measured(blob, bw, r.setting, "alt", r.ct, r.dt, r.ratio, r.cost,
                  win.have && r.setting == win.setting);
  }
  return win;
}

#else  // NeuroPress v2 needs the CUDA build; Create refuses a v2 model here.

void SetV2SelectionBlob(const std::string &) {}

std::vector<CompressionStats> Runtime::NeuroPressV2RankChunk(
    const void *, clio::run::u64, const Context &, double *, double *,
    double *, bool *out_gpu_failed,
    ctp::compress::model::NeuroPressV2Features *, double) {
  if (out_gpu_failed) *out_gpu_failed = true;
  return {};
}

double Runtime::NeuroPressV2LearnPrimary(
    const ctp::compress::model::NeuroPressV2Features &,
    const CompressionStats &, const Context &, clio::run::u64, double,
    bool *trained) {
  if (trained) *trained = false;
  return 0.0;
}

Runtime::V2ExploreWinner Runtime::NeuroPressV2Explore(
    const std::string &, double, const void *, clio::run::u64,
    const std::vector<CompressionStats> &, int, bool, double, bool,
    const ctp::compress::model::NeuroPressV2Features &) {
  return V2ExploreWinner{};
}

#endif  // CTP_ENABLE_NEUROPRESS_GPU

}  // namespace clio::cte::compressor
