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

#ifndef BDEV_PREDICTION_PARSE_H_
#define BDEV_PREDICTION_PARSE_H_

#include <clio_runtime/types.h>

#include <cctype>
#include <optional>
#include <string>

namespace clio::run::bdev {

/** TTL (days) reported for a drive the prediction server considers healthy. */
constexpr clio::run::u32 kHealthyTtlDays = 999999;

/**
 * Extract "days_remaining" from a prediction-server JSON response.
 *
 * Lightweight hand-rolled search to avoid pulling in a full JSON library.
 *
 * @param json Response body returned by SystemInfo::PredictDriveFailure.
 * @return The parsed day count; kHealthyTtlDays when the value is null or
 *         unparseable; std::nullopt when the key is absent (e.g. an
 *         {"error": ...} reply), meaning the caller should keep its current
 *         estimate.
 */
inline std::optional<clio::run::u32> ParseDaysRemaining(
    const std::string &json) {
  const std::string key = "\"days_remaining\"";
  auto pos = json.find(key);
  if (pos == std::string::npos) return std::nullopt;
  pos += key.size();
  // Skip whitespace and colon
  while (pos < json.size() && (json[pos] == ' ' || json[pos] == ':')) ++pos;
  // "null" means the server says the drive is healthy
  if (json.compare(pos, 4, "null") == 0) return kHealthyTtlDays;
  if (pos >= json.size() ||
      !std::isdigit(static_cast<unsigned char>(json[pos]))) {
    return kHealthyTtlDays;  // Unparseable -> treat as healthy
  }
  clio::run::u32 days = 0;
  while (pos < json.size() &&
         std::isdigit(static_cast<unsigned char>(json[pos]))) {
    days = days * 10 + static_cast<clio::run::u32>(json[pos] - '0');
    ++pos;
  }
  return days;
}

}  // namespace clio::run::bdev

#endif  // BDEV_PREDICTION_PARSE_H_
