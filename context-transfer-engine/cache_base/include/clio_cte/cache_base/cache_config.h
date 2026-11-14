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

#ifndef CLIO_CTE_CACHE_BASE_CACHE_CONFIG_H_
#define CLIO_CTE_CACHE_BASE_CACHE_CONFIG_H_

#include <clio_runtime/clio_runtime.h>
#include <clio_ctp/util/msan.h>
#include <clio_runtime/admin/admin_tasks.h>

#include <string>

namespace clio::cte::cache_base {

/** Which write policy a cache module implements. */
enum class CachePolicy : clio::run::u32 {
  kWriteThru = 0,  ///< ack follows the persistent write
  kWriteBack = 1,  ///< ack follows the volatile write; flushed later
};

/** When a read miss copies the blob into the volatile core. */
enum class FillMode : clio::run::u32 {
  kOnReadMiss = 0,  ///< every miss fills
  kOnHitCount = 1,  ///< the Nth miss of a blob fills (fill_hits_)
  kNone = 2,        ///< never fill (volatile holds only written data)
};

/** Which cached copy is dropped first when the volatile core is full. */
enum class EvictMode : clio::run::u32 {
  kLru = 0,  ///< least recently used
  kLfu = 1,  ///< least frequently used
};

/** Parse "major.minor" into a PoolId; returns false on a malformed value. */
inline bool ParsePoolIdString(const std::string &s, clio::run::PoolId *out) {
  auto dot = s.find('.');
  if (dot == std::string::npos) return false;
  try {
    *out = clio::run::PoolId(
        static_cast<clio::run::u32>(std::stoul(s.substr(0, dot))),
        static_cast<clio::run::u32>(std::stoul(s.substr(dot + 1))));
  } catch (...) {
    return false;
  }
  return true;
}

/**
 * Configuration shared by the policy cache modules. Compose keys:
 *
 *   next_volatile: "540.0"         # volatile core (required)
 *   next_persistent: "561.0"       # persistent chain (omit: volatile only)
 *   fill: on_read_miss             # | on_hit_count:N | none
 *   evict: lru                     # | lfu
 *   flush_period_ms: 10000         # write-back only; 0 = no periodic flush
 *   fallback_to_persistent: true   # write-back: write around when full
 */
struct PolicyCacheParams {
  clio::run::PoolId volatile_pool_;    ///< next_volatile
  clio::run::PoolId persistent_pool_;  ///< next_persistent (null = none)
  clio::run::u32 fill_mode_ = static_cast<clio::run::u32>(FillMode::kOnReadMiss);
  clio::run::u32 fill_hits_ = 2;  ///< N of on_hit_count:N
  clio::run::u32 evict_mode_ = static_cast<clio::run::u32>(EvictMode::kLru);
  clio::run::u32 flush_period_ms_ = 10000;
  bool fallback_to_persistent_ = true;

  PolicyCacheParams()
      : volatile_pool_(clio::run::PoolId::GetNull()),
        persistent_pool_(clio::run::PoolId::GetNull()) {}

  /** Fill mode as an enum. */
  FillMode Fill() const { return static_cast<FillMode>(fill_mode_); }
  /** Eviction mode as an enum. */
  EvictMode Evict() const { return static_cast<EvictMode>(evict_mode_); }

  template <class Archive>
  void serialize(Archive &ar) {
    // Every field round-trips (the rule three silently-dropped-field bugs
    // in the older interposers bought).
    ar(volatile_pool_, persistent_pool_, fill_mode_, fill_hits_,
       evict_mode_, flush_period_ms_, fallback_to_persistent_);
  }

  /**
   * Parse the `fill` key: "on_read_miss", "none" or "on_hit_count:N".
   * @param value the YAML scalar
   * @return false when the value is not one of those
   */
  bool ParseFill(const std::string &value) {
    if (value == "on_read_miss") {
      fill_mode_ = static_cast<clio::run::u32>(FillMode::kOnReadMiss);
      return true;
    }
    if (value == "none") {
      fill_mode_ = static_cast<clio::run::u32>(FillMode::kNone);
      return true;
    }
    const std::string prefix = "on_hit_count:";
    if (value.compare(0, prefix.size(), prefix) == 0) {
      try {
        fill_hits_ = static_cast<clio::run::u32>(
            std::stoul(value.substr(prefix.size())));
      } catch (...) {
        return false;
      }
      if (fill_hits_ == 0) fill_hits_ = 1;
      fill_mode_ = static_cast<clio::run::u32>(FillMode::kOnHitCount);
      return true;
    }
    return false;
  }

  /**
   * Load from the compose entry's YAML. Malformed values are logged and
   * leave the default in place.
   * @param pool_config the compose pool entry
   */
  void LoadConfig(const clio::run::PoolConfig &pool_config) {
    if (pool_config.config_.empty()) return;
    try {
      YAML::Node node = YAML::Load(pool_config.config_);
      // yaml-cpp is prebuilt: its scalars carry no MSan shadow.
      ctp::MsanUnpoisonYaml(node);
      if (node["next_volatile"] &&
          !ParsePoolIdString(node["next_volatile"].as<std::string>(),
                             &volatile_pool_)) {
        HLOG(kError, "policy cache: malformed next_volatile");
      }
      if (node["next_persistent"] &&
          !ParsePoolIdString(node["next_persistent"].as<std::string>(),
                             &persistent_pool_)) {
        HLOG(kError, "policy cache: malformed next_persistent");
      }
      if (node["fill"] && !ParseFill(node["fill"].as<std::string>())) {
        HLOG(kError,
             "policy cache: fill must be on_read_miss, none or "
             "on_hit_count:N");
      }
      if (node["evict"]) {
        const std::string e = node["evict"].as<std::string>();
        if (e == "lru") {
          evict_mode_ = static_cast<clio::run::u32>(EvictMode::kLru);
        } else if (e == "lfu") {
          evict_mode_ = static_cast<clio::run::u32>(EvictMode::kLfu);
        } else {
          HLOG(kError, "policy cache: evict must be lru or lfu");
        }
      }
      if (node["flush_period_ms"]) {
        flush_period_ms_ = node["flush_period_ms"].as<clio::run::u32>();
      }
      if (node["fallback_to_persistent"]) {
        fallback_to_persistent_ = node["fallback_to_persistent"].as<bool>();
      }
    } catch (const std::exception &e) {
      HLOG(kError, "policy cache: config parse error: {}", e.what());
    }
  }
};

/**
 * Per-module config type: the shared params plus the module's library name
 * (GetOrCreatePoolTask and CLIO_TASK_CC read chimod_lib_name).
 * @tparam Traits supplies `static constexpr const char *kLibName`
 */
template <typename Traits>
struct PolicyCacheConfig : public PolicyCacheParams {
  static constexpr const char *chimod_lib_name = Traits::kLibName;

  PolicyCacheConfig() = default;
  PolicyCacheConfig(const clio::run::PoolId &pool_id,
                    const PolicyCacheConfig &other)
      : PolicyCacheParams(other) {
    (void)pool_id;
  }

  template <class Archive>
  void serialize(Archive &ar) {
    PolicyCacheParams::serialize(ar);
  }
};

}  // namespace clio::cte::cache_base

#endif  // CLIO_CTE_CACHE_BASE_CACHE_CONFIG_H_
