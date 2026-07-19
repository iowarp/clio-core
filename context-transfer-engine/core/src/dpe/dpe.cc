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

// DPE type conversions + factory. Each placement POLICY implementation lives
// in its own source file in this directory; register new policies here.

#include <clio_cte/core/dpe/dpe.h>
#include <clio_cte/core/dpe/random_dpe.h>
#include <clio_cte/core/dpe/round_robin_dpe.h>
#include <clio_cte/core/dpe/max_bw_dpe.h>

#include "clio_ctp/util/logging.h"

namespace clio::cte::core {

// DPE Type conversion functions
DpeType StringToDpeType(const std::string& dpe_str) {
  if (dpe_str == "random") {
    return DpeType::kRandom;
  } else if (dpe_str == "round_robin" || dpe_str == "roundrobin") {
    return DpeType::kRoundRobin;
  } else if (dpe_str == "max_bw" || dpe_str == "maxbw") {
    return DpeType::kMaxBW;
  } else {
    HLOG(kError, "Unknown DPE type: {}, defaulting to random", dpe_str);
    return DpeType::kRandom;
  }
}

std::string DpeTypeToString(DpeType dpe_type) {
  switch (dpe_type) {
    case DpeType::kRandom:
      return "random";
    case DpeType::kRoundRobin:
      return "round_robin";
    case DpeType::kMaxBW:
      return "max_bw";
    default:
      return "random";
  }
}

// DpeFactory Implementation
std::unique_ptr<DataPlacementEngine> DpeFactory::CreateDpe(DpeType dpe_type) {
  switch (dpe_type) {
    case DpeType::kRandom:
      return std::make_unique<RandomDpe>();
    case DpeType::kRoundRobin:
      return std::make_unique<RoundRobinDpe>();
    case DpeType::kMaxBW:
      return std::make_unique<MaxBwDpe>();
    default:
      HLOG(kError, "Unknown DPE type, defaulting to Random");
      return std::make_unique<RandomDpe>();
  }
}

std::unique_ptr<DataPlacementEngine> DpeFactory::CreateDpe(const std::string& dpe_str) {
  return CreateDpe(StringToDpeType(dpe_str));
}

} // namespace clio::cte::core
