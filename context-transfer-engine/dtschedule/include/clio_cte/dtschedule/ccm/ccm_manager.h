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

#ifndef CLIO_CTE_DTSCHEDULE_CCM_CCM_MANAGER_H_
#define CLIO_CTE_DTSCHEDULE_CCM_CCM_MANAGER_H_

#include <clio_cte/dtschedule/ccm/predictor.h>
#include <clio_cte/dtschedule/ccm/candidates.h>
#include <clio_cte/dtschedule/ccm/ranker.h>
#include <clio_cte/dtschedule/ccm/data_stats.h>

#include <atomic>
#include <functional>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <map>
#include <random>
#include <thread>
#include <utility>

// Forward declaration is enough for pointers/references, but we'll do full inclusion in.cc

namespace clio::cte::dtschedule {

// Forward declaration for the header
struct DtscheduleConfig;

namespace ccm {

/**
 * Codec Characteristic Model (CCM) manager.
 *
 * Central interface for all codec selection. Owns the predictor, candidates,
 * ranker, and data stats calculator. Called from PutBlob to decide which
 * codec to use.
 *
 * Main entry point: SelectCodec(blob_data, size, blob_name) → Decision.
 */
class CcmManager {
 public:
  CcmManager() = default;
  /** Stops the background resample thread. */
  ~CcmManager();

  /**
   * Initialize the CCM from configuration.
   *
   * Builds candidate set, loads predictor model (if Q-table), and caches
   * factory presets. Called once at module Create().
   *
   * @param config Parsed DtscheduleConfig
   * @param model_dir Model directory for Q-table (used if config.ccm_ = qtable)
   * @return True if init succeeded
   */
  bool Init(const DtscheduleConfig &config, const std::string &model_dir);

  /**
   * Select a codec for a blob and predict its characteristics.
   *
   * High-level interface: sample the blob, compute stats, filter candidates,
   * rank, and return the chosen codec. Used by PutBlob.
   *
   * @param blob_data Raw blob bytes
   * @param size Blob size in bytes
   * @param blob_name Blob name (for regex matching)
   * @param ratio_noise_sigma Multiplicative noise on predicted ratio (0=off)
   * @return Decision with chosen codec and predictions, or skip compression
   */
  /**
   * Select codec for a blob based on data characteristics and load state.
   *
   * @param blob_data Raw blob bytes
   * @param size Blob size
   * @param blob_name Blob name (for stage matching and resampling)
   * @param ratio_noise_sigma Multiplicative noise on predicted ratio
   * @param load_mult Load multiplier for compress+store cost terms (phase 3+)
   * @return Decision with chosen codec and predictions
   */
  ccm::Decision SelectCodec(const void *blob_data,
                            size_t size,
                            const std::string &blob_name,
                            double ratio_noise_sigma = 0.0,
                            double load_mult = 1.0,
                            double tier_bw_mb_ms = 0.0,
                            const std::function<double(uint64_t)> *store_bw =
                                nullptr);

  /**
   * Record observed compression performance.
   *
   * Called after compression completes. Updates predictor state and may
   * trigger resampling (for Q-table).
   *
   * @param blob_data Raw blob bytes (for resampling)
   * @param size Blob size
   * @param blob_name Blob name (for stage overrides)
   * @param lib Compression library used
   * @param preset Compression preset used
   * @param obs_ctime_ms Observed compression time
   * @param obs_dtime_ms Observed decompression time
   * @param obs_ratio Observed compression ratio
   */
  /**
   * Feed an observation back to the predictor and apply the resample rule:
   * when the prediction missed by more than resample_error, with probability
   * resample_chance every candidate is run on the chunk and observed.
   */
  void Observe(const void *blob_data,
               size_t size,
               const std::string &blob_name,
               const std::string &lib,
               ctp::CompressionPreset preset,
               double pred_ctime_ms,
               double pred_ratio,
               double obs_ctime_ms,
               double obs_dtime_ms,
               double obs_ratio);

  using ProbeKey = std::pair<std::string, ctp::CompressionPreset>;
  /**
   * Compress and decompress a sample of the chunk with every candidate and
   * return measured time and ratio per candidate scaled to the full size.
   * Used to seed the EMA, by the oracle, and for resampling.
   */
  std::map<ProbeKey, Prediction> ProbeCandidates(
      const void *blob_data, size_t size,
      const std::vector<Candidate> &candidates,
      size_t sample_max = kProbeSampleBytes) const;

  /** Largest sample a probe compresses (timings are scaled to the chunk). */
  static constexpr size_t kProbeSampleBytes = 1 << 20;
  /** Sample used by background resampling: bounded so every candidate,
   *  including lzma/brotli-best, finishes in well under a second. */
  static constexpr size_t kResampleSampleBytes = 256 << 10;

  /** Number of resample probes completed by the background thread. */
  uint64_t Resamples() const { return resamples_.load(); }
  /** Number of resample requests dropped because one was already pending. */
  uint64_t ResamplesDropped() const { return resamples_dropped_.load(); }

  /**
   * Switch the CCM predictor at runtime (for ablations and E10).
   *
   * @param ccm_spec New CCM spec string (e.g., "qtable", "fixed:zstd:balanced")
   * @param model_dir Model directory for Q-table (if switching to qtable)
   * @return True if switch succeeded
   */
  bool SetPredictor(const std::string &ccm_spec,
                    const std::string &model_dir = "");

  /**
   * Update QoS configuration at runtime.
   *
   * Rebuilds the filtered candidate set for per-stage overrides.
   * Called by SetKnobs if workflow_aware or stage-matched config changes.
   *
   * @param config Updated DtscheduleConfig
   */
  void UpdateConfig(const DtscheduleConfig &config);
  /** Data statistics of a chunk, for trace rows. */
  Features ComputeFeaturesPublic(const void *blob_data, size_t size) const {
    return ComputeFeatures(blob_data, size);
  }

 private:
  /**
   * Create predictor from CCM spec string.
   *
   * Parses "qtable", "ema", "fixed:<lib>[:<preset>]", "oracle".
   *
   * @param ccm_spec CCM spec
   * @param model_dir Model directory (for qtable)
   * @return Predictor object or nullptr on error
   */
  std::unique_ptr<CcmPredictor> CreatePredictor(const std::string &ccm_spec,
                                                const std::string &model_dir);

  std::unique_ptr<DtscheduleConfig> config_;   ///< Cached config
  std::unique_ptr<CcmPredictor> predictor_;   ///< Current predictor
  CandidateSet candidates_;                    ///< Built once at Init
  std::mutex config_lock_;                     ///< Protect config_ updates
  std::mutex predictor_lock_;                  ///< Protect predictor_ swaps
  std::mt19937_64 rng_{std::random_device{}()};  ///< Noise and resample draws

  /** Build features for a chunk (shared by SelectCodec and Observe). */
  Features ComputeFeatures(const void *blob_data, size_t size) const;
  /** Restrict to the fixed codec, recording the rest with reason "fixed". */
  std::vector<Candidate> RestrictToFixed(const std::vector<Candidate> &filtered,
                                         std::vector<CandidateRecord> *rejected);
  /** Seed EMA / oracle predictors from a probe; predictor_lock_ must be held. */
  void PrimePredictor(const void *blob_data, size_t size,
                      const std::vector<Candidate> &candidates);

  /** A chunk sample waiting for the background resample thread. */
  struct ResampleJob {
    std::vector<char> sample_;         ///< Copy of the chunk prefix
    size_t full_size_ = 0;             ///< Size of the chunk it came from
    Features features_;                ///< Features of the full chunk
    std::vector<Candidate> cands_;     ///< Candidates to probe
    std::string blob_name_;            ///< For the debug log
  };
  std::unique_ptr<ResampleJob> pending_resample_;  ///< Single-slot queue
  std::mutex resample_lock_;                       ///< Guards the slot
  std::condition_variable resample_cv_;            ///< Wakes the thread
  std::thread resample_thread_;                    ///< Probes off the put path
  bool resample_stop_ = false;                     ///< Set by the destructor
  std::atomic<uint64_t> resamples_{0};             ///< Probes completed
  std::atomic<uint64_t> resamples_dropped_{0};     ///< Requests dropped (busy)

  /**
   * Hand a chunk to the resample thread. Copies at most kResampleSampleBytes
   * so the put path never waits on a codec. Drops the request when a probe
   * is already pending (single flight).
   */
  void QueueResample(const void *blob_data, size_t size,
                     const Features &features, std::vector<Candidate> cands,
                     const std::string &blob_name);
  /** Body of resample_thread_: probe pending jobs and feed the predictor. */
  void ResampleLoop();
  /** Start resample_thread_ once (idempotent). */
  void StartResampleThread();
};

}  // namespace ccm
}  // namespace clio::cte::dtschedule

#endif  // CLIO_CTE_DTSCHEDULE_CCM_CCM_MANAGER_H_
