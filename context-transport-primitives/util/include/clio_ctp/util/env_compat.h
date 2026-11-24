/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core (BSD 3-Clause license, see COPYING).
 */
#ifndef CLIO_CTP_UTIL_ENV_COMPAT_H_
#define CLIO_CTP_UTIL_ENV_COMPAT_H_

#include <cstdlib>
#include <string>

namespace ctp::env {

/**
 * Env-var reader. Pass just the suffix, e.g. GetCompat("WITH_RUNTIME") reads
 * CLIO_WITH_RUNTIME. Returns nullptr if not set.
 *
 * Defined in clio_ctp (lower-level than clio_runtime) so that ctp-internal
 * code (e.g. lightbeam transports) can read CLIO_* env vars without an upward
 * dependency on clio_runtime. clio_runtime.h exposes the same helper under
 * clio::run::env::GetCompat via a `using` alias.
 */
inline const char* GetCompat(const char* suffix) {
  std::string clio_name = std::string("CLIO_") + suffix;
  return std::getenv(clio_name.c_str());
}

}  // namespace ctp::env

#endif  // CLIO_CTP_UTIL_ENV_COMPAT_H_
