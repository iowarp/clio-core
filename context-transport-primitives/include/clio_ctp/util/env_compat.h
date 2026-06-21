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
 * Backward-compat env-var reader.
 *
 * Prefers the new CLIO_<suffix> name and falls back to the legacy CHI_<suffix>
 * if the new name is not set. Returns nullptr if neither is set.
 *
 * Pass just the suffix, e.g. GetCompat("WITH_RUNTIME") looks at
 * CLIO_WITH_RUNTIME, then CHI_WITH_RUNTIME.
 *
 * Defined in clio_ctp (lower-level than clio_runtime) so that ctp-internal
 * code that reads legacy CHI_* env vars (e.g. lightbeam transports) can use
 * it without an upward dependency on clio_runtime. clio_runtime.h exposes
 * the same helper under clio::run::env::GetCompat via a `using` alias.
 */
inline const char* GetCompat(const char* suffix) {
  std::string clio_name = std::string("CLIO_") + suffix;
  if (const char* v = std::getenv(clio_name.c_str())) return v;
  std::string chi_name = std::string("CHI_") + suffix;
  return std::getenv(chi_name.c_str());
}

}  // namespace ctp::env

#endif  // CLIO_CTP_UTIL_ENV_COMPAT_H_
