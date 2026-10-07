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

#ifndef CLIO_CTE_DTSCHEDULE_CCM_CANDIDATES_H_
#define CLIO_CTE_DTSCHEDULE_CCM_CANDIDATES_H_

#include <vector>
#include <string>
#include <regex>
#include <clio_ctp/compress/compress_factory.h>

namespace clio::cte::dtschedule::ccm {

/**
 * One candidate codec (library, preset) pair.
 */
struct Candidate {
  std::string lib_;                  ///< Library name (e.g., "zstd")
  ctp::CompressionPreset preset_;   ///< Compression preset
  bool is_lossy_;                    ///< True if lossy (sz3, zfp, fpzip)
};

/**
 * Candidate set builder: filters candidates based on QoS constraints.
 *
 * Builds the initial set of (lib, preset) pairs from the compression factory,
 * then applies QoS filters: compression_preference, max_error, lossy_allowlist,
 * and per-stage overrides via regex matching.
 */
/**
 * One evaluated candidate of a decision (written to the .cand trace).
 * reason is "ok" for ranked candidates, otherwise why it was excluded.
 */
struct CandidateRecord {
  std::string lib_;                      ///< Library name
  ctp::CompressionPreset preset_;        ///< Compression preset
  double pred_ctime_ms_ = 0.0;           ///< Predicted compress time (ms)
  double pred_dtime_ms_ = 0.0;           ///< Predicted decompress time (ms)
  double pred_ratio_ = 0.0;              ///< Predicted ratio (original/compressed)
  double cost_ = 0.0;                    ///< Total cost (ms); 0 unless ranked
  std::string reason_;                   ///< ok | qos_* | unavailable | fixed | skip_ratio
};

/** Reason strings used in CandidateRecord::reason_. */
constexpr const char *kReasonOk = "ok";
constexpr const char *kReasonLossyNotAllowed = "qos_lossy_not_allowed";
constexpr const char *kReasonErrorBound = "qos_error_bound";
constexpr const char *kReasonPreference = "qos_preference";
constexpr const char *kReasonFixed = "fixed";
constexpr const char *kReasonSkipRatio = "skip_ratio";
/** Raw candidate record: storing raw lost to the chosen codec. */
constexpr const char *kReasonRawCostlier = "raw_costlier";
/** Pseudo-library name of the raw (no compression) alternative. */
constexpr const char *kRawLib = "raw";

class CandidateSet {
 public:
  /**
   * Construct an empty candidate set.
   *
   * Call BuildAll to populate from factory, then ApplyQos to filter.
   */
  CandidateSet() = default;

  /**
   * Build all available candidates from the compression factory.
   *
   * Probes CompressionFactory::GetPreset for every (lib, preset) pair,
   * caches available candidates, and notes which are lossy.
   *
   * Lossy libraries: sz3, zfp, fpzip.
   * For each lib, try presets: kFast (1), kBalanced (2), kBest (3).
   * Some libs (snappy, blosc2) only support kBalanced (2).
   *
   * @return Number of available candidates
   */
  size_t BuildAll();

  /**
   * Filter candidates based on QoS constraints and blob name.
   *
   * Applies:
   * - compression_preference: if non-empty, restrict to those libs
   * - max_error: if 0, no lossy candidates; if >0, filter lossy by error bound
   * - lossy_allowlist: regex patterns for blob names (lossy only if matched)
   * - stage overrides: per-stage config changes (via regex on blob name)
   *
   * @param blob_name The blob being compressed (for regex matching)
   * @param preference_libs If non-empty, restrict to these libraries
   * @param max_error Error bound (0 = lossless only)
   * @param allowlist Regex patterns for lossy (empty = no lossy)
   * @param dtype Data type for lossy filter (lossy only for float dtype)
   * @return Filtered candidate list
   */
  std::vector<Candidate> Filter(const std::string &blob_name,
                                const std::vector<std::string> &preference_libs,
                                double max_error,
                                const std::vector<std::string> &allowlist,
                                int dtype,
                                std::vector<CandidateRecord> *rejected) const;

  /**
   * Get all currently available candidates (before filtering).
   */
  const std::vector<Candidate> &GetAll() const { return all_candidates_; }

 private:
  std::vector<Candidate> all_candidates_;  ///< All available (lib, preset) pairs

  /**
   * Check if lossy candidate error bound <= max_error.
   *
   * Maps libpressio preset levels to relative error bounds:
   * - sz3: fast=0.05, balanced=0.01, best=0.005
   * - zfp: fast=rate8, balanced=rate16, best=accuracy1e-3
   * - fpzip: fast=12bits, balanced=18bits, best=21bits
   *
   * @param lib Library name
   * @param preset Compression preset
   * @param max_error Maximum allowed relative error
   * @return True if this lossy preset's bound fits within max_error
   */
  bool LossyBoundOk(const std::string &lib,
                    ctp::CompressionPreset preset,
                    double max_error) const;
};

}  // namespace clio::cte::dtschedule::ccm

#endif  // CLIO_CTE_DTSCHEDULE_CCM_CANDIDATES_H_
