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

#include <clio_cte/dtschedule/ccm/candidates.h>

namespace clio::cte::dtschedule::ccm {

size_t CandidateSet::BuildAll() {
  all_candidates_.clear();

  // Lossless libraries and their presets
  const std::vector<std::string> lossless_libs = {
    "zstd", "lz4", "zlib", "bzip2", "lzma", "brotli", "snappy", "blosc2"
  };

  // Lossy libraries
  const std::vector<std::string> lossy_libs = { "sz3", "zfp", "fpzip" };

  // Try each lossless library at all three presets
  for (const auto &lib : lossless_libs) {
    for (int preset_val = 1; preset_val <= 3; ++preset_val) {
      auto preset = static_cast<ctp::CompressionPreset>(preset_val);
      auto compressor = ctp::CompressionFactory::GetPreset(lib, preset);
      if (compressor) {
        all_candidates_.push_back(Candidate{lib, preset, false});
      }
    }
  }

  // Try each lossy library at all three presets
  for (const auto &lib : lossy_libs) {
    for (int preset_val = 1; preset_val <= 3; ++preset_val) {
      auto preset = static_cast<ctp::CompressionPreset>(preset_val);
      auto compressor = ctp::CompressionFactory::GetPreset(lib, preset);
      if (compressor) {
        all_candidates_.push_back(Candidate{lib, preset, true});
      }
    }
  }

  return all_candidates_.size();
}

std::vector<Candidate> CandidateSet::Filter(
    const std::string &blob_name,
    const std::vector<std::string> &preference_libs,
    double max_error,
    const std::vector<std::string> &allowlist,
    int dtype,
    std::vector<CandidateRecord> *rejected) const {
  std::vector<Candidate> filtered;
  auto reject = [&](const Candidate &c, const char *reason) {
    if (rejected != nullptr) {
      rejected->push_back(CandidateRecord{c.lib_, c.preset_, 0.0, 0.0, 0.0,
                                          0.0, reason});
    }
  };

  for (const auto &candidate : all_candidates_) {
    // Filter 1: compression_preference (restrict to preferred libs)
    if (!preference_libs.empty()) {
      bool found = false;
      for (const auto &pref : preference_libs) {
        if (pref == candidate.lib_) {
          found = true;
          break;
        }
      }
      if (!found) {
        reject(candidate, kReasonPreference);
        continue;
      }
    }

    // Filter 2: lossy candidates only when max_error > 0 and dtype is float
    if (candidate.is_lossy_) {
      if (max_error <= 0.0 || dtype != 1) {  // dtype 1 = float
        reject(candidate, kReasonLossyNotAllowed);
        continue;
      }

      // Check allowlist regex match
      bool matches_allowlist = false;
      for (const auto &pattern_str : allowlist) {
        try {
          std::regex pattern(pattern_str);
          if (std::regex_match(blob_name, pattern)) {
            matches_allowlist = true;
            break;
          }
        } catch (const std::regex_error &) {
          // Skip invalid regex patterns
        }
      }
      if (!matches_allowlist) {
        reject(candidate, kReasonLossyNotAllowed);
        continue;
      }

      // Check lossy error bound
      if (!LossyBoundOk(candidate.lib_, candidate.preset_, max_error)) {
        reject(candidate, kReasonErrorBound);
        continue;
      }
    }

    filtered.push_back(candidate);
  }

  return filtered;
}

bool CandidateSet::LossyBoundOk(const std::string &lib,
                                ctp::CompressionPreset preset,
                                double max_error) const {
  // Map lossy library/preset to estimated relative error bound
  if (lib == "sz3") {
    // sz3: fast=0.05, balanced=0.01, best=0.005
    if (preset == ctp::CompressionPreset::FAST) {
      return 0.05 <= max_error;
    } else if (preset == ctp::CompressionPreset::BALANCED) {
      return 0.01 <= max_error;
    } else {  // BEST
      return 0.005 <= max_error;
    }
  } else if (lib == "zfp") {
    // zfp: fast=rate8, balanced=rate16, best=accuracy1e-3
    // Map to rough relative error estimates
    if (preset == ctp::CompressionPreset::FAST) {
      return 0.1 <= max_error;  // rate 8 is aggressive
    } else if (preset == ctp::CompressionPreset::BALANCED) {
      return 0.01 <= max_error;  // rate 16 is moderate
    } else {  // BEST
      return 0.001 <= max_error;  // accuracy 1e-3 is precise
    }
  } else if (lib == "fpzip") {
    // fpzip: fast=12bits, balanced=18bits, best=21bits
    // Map to rough relative error estimates
    if (preset == ctp::CompressionPreset::FAST) {
      return 0.01 <= max_error;  // 12 bits is coarse
    } else if (preset == ctp::CompressionPreset::BALANCED) {
      return 0.0001 <= max_error;  // 18 bits is good
    } else {  // BEST
      return 0.00001 <= max_error;  // 21 bits is very good
    }
  }

  // Unknown lossy lib: assume not OK
  return false;
}

}  // namespace clio::cte::dtschedule::ccm
