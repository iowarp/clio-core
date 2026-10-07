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

#include <clio_cte/dtschedule/ccm/ccm_manager.h>
#include <clio_cte/dtschedule/ccm/qtable_predictor.h>
#include <clio_cte/dtschedule/ccm/ema_predictor.h>
#include <clio_cte/dtschedule/ccm/fixed_predictor.h>
#include <clio_cte/dtschedule/ccm/oracle_predictor.h>
#include <clio_cte/dtschedule/dtschedule_tasks.h>
#include <clio_ctp/util/logging.h>
#include <random>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

namespace clio::cte::dtschedule::ccm {

bool CcmManager::Init(const clio::cte::dtschedule::DtscheduleConfig &config,
                      const std::string &model_dir) {
  std::lock_guard<std::mutex> lock(config_lock_);
  config_ = std::make_unique<DtscheduleConfig>(config);

  // Build candidate set from factory
  size_t n_candidates = candidates_.BuildAll();
  HLOG(kInfo, "CCM: Built {} candidates from factory", n_candidates);

  // Create predictor from config
  predictor_ = CreatePredictor(config.ccm_, model_dir);
  if (!predictor_) {
    HLOG(kWarning, "CCM: Failed to create predictor for '{}'", config.ccm_);
    return false;
  }

  HLOG(kInfo, "CCM: Init complete (ccm={}, {} candidates)",
       config.ccm_, n_candidates);
  StartResampleThread();
  return true;
}

Features CcmManager::ComputeFeatures(const void *blob_data,
                                     size_t size) const {
  auto *data_stats = DataStats::GetInstance();
  auto r = data_stats->Calculate(reinterpret_cast<const uint8_t *>(blob_data),
                                 size);
  return Features{r.entropy, r.mad, r.d2, size, static_cast<int>(r.dtype)};
}

std::vector<Candidate> CcmManager::RestrictToFixed(
    const std::vector<Candidate> &filtered,
    std::vector<CandidateRecord> *rejected) {
  auto *fixed = dynamic_cast<FixedPredictor *>(predictor_.get());
  if (fixed == nullptr) {
    return filtered;
  }
  std::vector<Candidate> kept;
  for (const auto &c : filtered) {
    if (c.lib_ == fixed->GetLib() && c.preset_ == fixed->GetPreset()) {
      kept.push_back(c);
    } else {
      rejected->push_back(
          CandidateRecord{c.lib_, c.preset_, 0, 0, 0, 0, kReasonFixed});
    }
  }
  return kept;
}

std::map<CcmManager::ProbeKey, Prediction> CcmManager::ProbeCandidates(
    const void *blob_data, size_t size,
    const std::vector<Candidate> &candidates, size_t sample_max) const {
  std::map<ProbeKey, Prediction> out;
  const size_t sample = std::min(size, std::max<size_t>(sample_max, 1));
  const double scale = static_cast<double>(size) / static_cast<double>(sample);
  std::vector<char> comp(sample + (sample >> 3) + (64 << 10));
  std::vector<char> back(sample);
  auto *in = const_cast<char *>(reinterpret_cast<const char *>(blob_data));
  for (const auto &c : candidates) {
    auto codec = ctp::CompressionFactory::GetPreset(c.lib_, c.preset_);
    if (!codec) {
      continue;
    }
    size_t comp_size = comp.size();
    auto t0 = std::chrono::steady_clock::now();
    bool ok = codec->Compress(comp.data(), comp_size, in, sample);
    auto t1 = std::chrono::steady_clock::now();
    if (!ok || comp_size == 0) {
      continue;
    }
    size_t back_size = sample;
    bool dok = codec->Decompress(back.data(), back_size, comp.data(), comp_size);
    auto t2 = std::chrono::steady_clock::now();
    Prediction p;
    p.ctime_ms_ = std::chrono::duration<double, std::milli>(t1 - t0).count() * scale;
    p.dtime_ms_ = dok ? std::chrono::duration<double, std::milli>(t2 - t1).count() * scale
                      : p.ctime_ms_;
    p.ratio_ = static_cast<double>(sample) / static_cast<double>(comp_size);
    out[ProbeKey{c.lib_, c.preset_}] = p;
  }
  return out;
}

void CcmManager::PrimePredictor(const void *blob_data, size_t size,
                                const std::vector<Candidate> &candidates) {
  auto *ema = dynamic_cast<EmaPredictor *>(predictor_.get());
  auto *oracle = dynamic_cast<OraclePredictor *>(predictor_.get());
  if ((ema == nullptr || ema->IsSeeded()) && oracle == nullptr) {
    return;
  }
  auto measured = ProbeCandidates(blob_data, size, candidates);
  if (oracle != nullptr) {
    oracle->SetMeasurements(std::move(measured));
    return;
  }
  Features f = ComputeFeatures(blob_data, size);
  for (const auto &[key, p] : measured) {
    ema->Observe(f, key.first, key.second, p.ctime_ms_, p.dtime_ms_, p.ratio_);
  }
  ema->MarkSeeded();
}

ccm::Decision CcmManager::SelectCodec(const void *blob_data,
                                      size_t size,
                                      const std::string &blob_name,
                                      double ratio_noise_sigma,
                                      double load_mult,
                                      double tier_bw_mb_ms,
                                      const std::function<double(uint64_t)>
                                          *store_bw) {
  Features features = ComputeFeatures(blob_data, size);
  std::lock_guard<std::mutex> lock(config_lock_);
  if (!config_) {
    return ccm::Decision{};
  }
  std::string obj_override;
  double max_err_override = -1.0;
  std::vector<std::string> comp_pref_override;
  auto [stage_idx, was_overridden] = config_->ApplyQosStage(
      blob_name, obj_override, max_err_override, comp_pref_override);
  const double max_error =
      (max_err_override >= 0.0) ? max_err_override : config_->max_error_;
  const auto &comp_pref =
      was_overridden ? comp_pref_override : config_->compression_preference_;
  const auto &objective =
      !obj_override.empty() ? obj_override : config_->objective_;
  std::vector<CandidateRecord> rejected;
  auto filtered = candidates_.Filter(blob_name, comp_pref, max_error,
                                     config_->lossy_allowlist_,
                                     features.dtype_, &rejected);
  std::lock_guard<std::mutex> pred_lock(predictor_lock_);
  if (!predictor_) {
    HLOG(kWarning, "CCM: No predictor; skipping compression");
    return ccm::Decision{};
  }
  auto ranked = RestrictToFixed(filtered, &rejected);
  PrimePredictor(blob_data, size, ranked);
  // Gbps -> MB/ms: x Gbit/s = x/8 GB/s = x/8 MB/ms.
  // Pass load_mult to scale compress+store cost terms (phase 3+)
  // A fixed CCM is the "always this codec" baseline: it never falls back to
  // raw on cost (only when the codec is not useful at all).
  const bool compare_raw =
      dynamic_cast<FixedPredictor *>(predictor_.get()) == nullptr;
  auto decision = Ranker::Rank(ranked, features, size, predictor_.get(),
                               objective,
                               tier_bw_mb_ms > 0.0 ? tier_bw_mb_ms
                                                   : config_->net_bw_gbps_ / 8.0,
                               load_mult,
                               ratio_noise_sigma, &rng_, compare_raw,
                               store_bw);
  decision.qos_stage_index_ = stage_idx;
  decision.candidates_.insert(decision.candidates_.end(), rejected.begin(),
                              rejected.end());
  return decision;
}

void CcmManager::Observe(const void *blob_data,
                         size_t size,
                         const std::string &blob_name,
                         const std::string &lib,
                         ctp::CompressionPreset preset,
                         double pred_ctime_ms,
                         double pred_ratio,
                         double obs_ctime_ms,
                         double obs_dtime_ms,
                         double obs_ratio) {
  Features features = ComputeFeatures(blob_data, size);
  double resample_error = 0.0, resample_chance = 0.0, max_error = 0.0;
  std::vector<std::string> allowlist, preference;
  {
    std::lock_guard<std::mutex> lock(config_lock_);
    if (config_) {
      resample_error = config_->resample_error_;
      resample_chance = config_->resample_chance_;
      max_error = config_->max_error_;
      allowlist = config_->lossy_allowlist_;
      preference = config_->compression_preference_;
    }
  }
  std::lock_guard<std::mutex> lock(predictor_lock_);
  if (!predictor_) {
    return;
  }
  predictor_->Observe(features, lib, preset, obs_ctime_ms, obs_dtime_ms,
                      obs_ratio);
  const double ratio_err =
      std::fabs(pred_ratio - obs_ratio) / std::max(obs_ratio, 1e-9);
  const double time_err =
      std::fabs(pred_ctime_ms - obs_ctime_ms) / std::max(obs_ctime_ms, 1e-3);
  if (resample_error <= 0.0 || std::max(ratio_err, time_err) <= resample_error) {
    return;
  }
  if (dynamic_cast<FixedPredictor *>(predictor_.get()) != nullptr ||
      dynamic_cast<OraclePredictor *>(predictor_.get()) != nullptr) {
    return;
  }
  std::uniform_real_distribution<double> coin(0.0, 1.0);
  if (coin(rng_) > resample_chance) {
    return;
  }
  // Probing every candidate on the chunk costs seconds (lzma, brotli-best);
  // doing it here would hold predictor_lock_ and stall every worker's
  // SelectCodec. Hand a bounded sample to the resample thread instead.
  auto cands = candidates_.Filter(blob_name, preference, max_error, allowlist,
                                  features.dtype_, nullptr);
  QueueResample(blob_data, size, features, std::move(cands), blob_name);
  HLOG(kDebug, "CCM: resample queued for {} (err ratio={:.2f} time={:.2f})",
       blob_name, ratio_err, time_err);
}

CcmManager::~CcmManager() {
  {
    std::lock_guard<std::mutex> lock(resample_lock_);
    resample_stop_ = true;
  }
  resample_cv_.notify_all();
  if (resample_thread_.joinable()) {
    resample_thread_.join();
  }
}

void CcmManager::StartResampleThread() {
  std::lock_guard<std::mutex> lock(resample_lock_);
  if (resample_thread_.joinable() || resample_stop_) {
    return;
  }
  resample_thread_ = std::thread([this]() { ResampleLoop(); });
}

void CcmManager::QueueResample(const void *blob_data, size_t size,
                               const Features &features,
                               std::vector<Candidate> cands,
                               const std::string &blob_name) {
  if (cands.empty() || size == 0) {
    return;
  }
  const size_t sample = std::min(size, kResampleSampleBytes);
  auto job = std::make_unique<ResampleJob>();
  job->sample_.assign(reinterpret_cast<const char *>(blob_data),
                      reinterpret_cast<const char *>(blob_data) + sample);
  job->full_size_ = size;
  job->features_ = features;
  job->cands_ = std::move(cands);
  job->blob_name_ = blob_name;
  {
    std::lock_guard<std::mutex> lock(resample_lock_);
    if (pending_resample_) {
      resamples_dropped_.fetch_add(1);
      return;
    }
    pending_resample_ = std::move(job);
  }
  resample_cv_.notify_one();
}

void CcmManager::ResampleLoop() {
  while (true) {
    std::unique_ptr<ResampleJob> job;
    {
      std::unique_lock<std::mutex> lock(resample_lock_);
      resample_cv_.wait(lock, [this]() {
        return resample_stop_ || pending_resample_ != nullptr;
      });
      if (resample_stop_) {
        return;
      }
      job = std::move(pending_resample_);
    }
    // ProbeCandidates scales timings by full_size / sample.
    auto measured = ProbeCandidates(job->sample_.data(), job->full_size_,
                                    job->cands_, job->sample_.size());
    {
      std::lock_guard<std::mutex> lock(predictor_lock_);
      if (predictor_) {
        for (const auto &[key, p] : measured) {
          predictor_->Observe(job->features_, key.first, key.second,
                              p.ctime_ms_, p.dtime_ms_, p.ratio_);
        }
      }
    }
    resamples_.fetch_add(1);
    HLOG(kDebug, "CCM: resampled {} candidates for {} ({} bytes of {})",
         measured.size(), job->blob_name_, job->sample_.size(),
         job->full_size_);
  }
}

bool CcmManager::SetPredictor(const std::string &ccm_spec,
                              const std::string &model_dir) {
  auto new_predictor = CreatePredictor(ccm_spec, model_dir);
  if (!new_predictor) {
    HLOG(kWarning, "CCM: Failed to create predictor for '{}'", ccm_spec);
    return false;
  }

  std::lock_guard<std::mutex> lock(predictor_lock_);
  predictor_ = std::move(new_predictor);

  std::lock_guard<std::mutex> config_lock(config_lock_);
  if (config_) {
    config_->ccm_ = ccm_spec;
  }

  HLOG(kInfo, "CCM: Switched to predictor '{}'", ccm_spec);
  return true;
}

void CcmManager::UpdateConfig(const clio::cte::dtschedule::DtscheduleConfig &config) {
  std::lock_guard<std::mutex> lock(config_lock_);
  config_ = std::make_unique<DtscheduleConfig>(config);
}

std::unique_ptr<CcmPredictor> CcmManager::CreatePredictor(
    const std::string &ccm_spec,
    const std::string &model_dir) {
  // Parse spec: "qtable", "ema", "fixed:<lib>[:<preset>]", "oracle"

  if (ccm_spec == "qtable") {
    auto predictor = std::make_unique<QtablePredictor>();
    if (predictor->Load(model_dir)) {
      return predictor;
    }
    return nullptr;
  } else if (ccm_spec == "ema") {
    return std::make_unique<EmaPredictor>();
  } else if (ccm_spec == "oracle") {
    return std::make_unique<OraclePredictor>();
  } else if (ccm_spec.substr(0, 6) == "fixed:") {
    // Parse "fixed:<lib>[:<preset>]"
    size_t first_colon = ccm_spec.find(':');
    size_t second_colon = ccm_spec.find(':', first_colon + 1);

    std::string lib = ccm_spec.substr(first_colon + 1,
                                      second_colon == std::string::npos
                                          ? std::string::npos
                                          : second_colon - first_colon - 1);

    ctp::CompressionPreset preset = ctp::CompressionPreset::BALANCED;
    if (second_colon != std::string::npos) {
      std::string preset_str = ccm_spec.substr(second_colon + 1);
      if (preset_str == "fast") {
        preset = ctp::CompressionPreset::FAST;
      } else if (preset_str == "balanced") {
        preset = ctp::CompressionPreset::BALANCED;
      } else if (preset_str == "best") {
        preset = ctp::CompressionPreset::BEST;
      }
    }

    return std::make_unique<FixedPredictor>(lib, preset);
  }

  HLOG(kWarning, "CCM: Unknown CCM spec '{}'", ccm_spec);
  return nullptr;
}

}  // namespace clio::cte::dtschedule::ccm
