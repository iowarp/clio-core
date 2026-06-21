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

#ifndef WRPCTE_CORE_DPE_H_
#define WRPCTE_CORE_DPE_H_

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_tasks.h>
#include <vector>
#include <string>
#include <memory>
#include <random>

namespace clio::cte::core {

/**
 * Data Placement Engine types
 */
enum class DpeType : clio::run::u32 {
  kRandom = 0,    // Random placement
  kRoundRobin = 1, // Round-robin placement
  kMaxBW = 2      // Max bandwidth placement
};

/**
 * Convert DPE type string to enum
 */
DpeType StringToDpeType(const std::string& dpe_str);

/**
 * Convert DPE type enum to string
 */
std::string DpeTypeToString(DpeType dpe_type);

/**
 * Abstract Data Placement Engine interface
 */
class DataPlacementEngine {
public:
  virtual ~DataPlacementEngine() = default;
  
  /**
   * Select targets for data placement
   * @param targets Available targets for placement
   * @param blob_score Score of the blob (0-1)
   * @param data_size Size of data to be placed
   * @return Vector of ordered targets, empty if no suitable targets
   */
  virtual std::vector<TargetInfo> SelectTargets(const std::vector<TargetInfo>& targets, 
                                               float blob_score, 
                                               clio::run::u64 data_size) = 0;
  
  /**
   * Get the DPE type
   */
  virtual DpeType GetType() const = 0;
};

/**
 * Random Data Placement Engine
 */
class RandomDpe : public DataPlacementEngine {
public:
  RandomDpe();
  
  std::vector<TargetInfo> SelectTargets(const std::vector<TargetInfo>& targets, 
                                       float blob_score, 
                                       clio::run::u64 data_size) override;
  
  DpeType GetType() const override { return DpeType::kRandom; }

private:
  std::mt19937 rng_;
};

/**
 * Round-Robin Data Placement Engine
 */
class RoundRobinDpe : public DataPlacementEngine {
public:
  RoundRobinDpe();
  
  std::vector<TargetInfo> SelectTargets(const std::vector<TargetInfo>& targets, 
                                       float blob_score, 
                                       clio::run::u64 data_size) override;
  
  DpeType GetType() const override { return DpeType::kRoundRobin; }

private:
  static std::atomic<clio::run::u32> round_robin_counter_;
};

/**
 * Max Bandwidth Data Placement Engine
 */
class MaxBwDpe : public DataPlacementEngine {
public:
  MaxBwDpe();
  
  std::vector<TargetInfo> SelectTargets(const std::vector<TargetInfo>& targets, 
                                       float blob_score, 
                                       clio::run::u64 data_size) override;
  
  DpeType GetType() const override { return DpeType::kMaxBW; }

private:
  static constexpr clio::run::u64 kLatencyThreshold = 32 * 1024; // 32KB threshold
};

/**
 * Data Placement Engine Factory
 */
class DpeFactory {
public:
  /**
   * Create a DPE instance
   * @param dpe_type Type of DPE to create
   * @return Unique pointer to DPE instance
   */
  static std::unique_ptr<DataPlacementEngine> CreateDpe(DpeType dpe_type);
  
  /**
   * Create a DPE instance from string
   * @param dpe_str DPE type as string
   * @return Unique pointer to DPE instance
   */
  static std::unique_ptr<DataPlacementEngine> CreateDpe(const std::string& dpe_str);
};

} // namespace clio::cte::core

#endif // WRPCTE_CORE_DPE_H_