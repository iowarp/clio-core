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

#ifndef CTP_SHM_INCLUDE_HSHM_SHM_UTIL_LOGGING_H_
#define CTP_SHM_INCLUDE_HSHM_SHM_UTIL_LOGGING_H_

#include <climits>
#include <fstream>
#include <iomanip>
#include <iostream>
#include "formatter.h"
#include "clio_ctp/introspect/system_info.h"

/**
 * Log level codes for filtering messages at compile-time and runtime
 * Lower values = more verbose, higher values = less verbose
 * Defined as macros for global accessibility without namespace qualification
 */
#ifndef kDebug
#define kDebug 0    /**< Low-priority debugging information */
#endif
#ifndef kInfo
#define kInfo 1     /**< Useful information the user should know */
#endif
#ifndef kSuccess
#define kSuccess 2  /**< Operation completed successfully */
#endif
#ifndef kWarning
#define kWarning 3  /**< Something might be wrong */
#endif
#ifndef kError
#define kError 4    /**< A non-fatal error has occurred */
#endif
#ifndef kFatal
#define kFatal 5    /**< A fatal error has occurred */
#endif

/**
 * Compile-time log level threshold
 * Messages below this level will be compiled out entirely
 * Default: kInfo (1) - debug messages excluded in release builds
 */
#ifndef CTP_LOG_LEVEL
#define CTP_LOG_LEVEL kInfo
#endif

namespace ctp {

/** Simplify access to Logger singleton */
#define CTP_LOG ctp::CrossSingleton<ctp::Logger>::GetInstance()

/**
 * Clio Print. Like printf, except types are inferred
 */
#define HIPRINT(...) CTP_LOG->Print(__VA_ARGS__)

/**
 * Clio CTP Log - Unified logging macro
 *
 * Messages with LOG_CODE < CTP_LOG_LEVEL are compiled out entirely.
 * Messages with LOG_CODE >= CTP_LOG_LEVEL are subject to runtime filtering
 * via the CTP_LOG_LEVEL environment variable.
 *
 * @param LOG_CODE The log level (kDebug, kInfo, kWarning, kError, kFatal)
 * @param ... Format string and arguments
 */
// On the host pass, dispatch through the Logger singleton.
//
// On any device pass (CUDA/ROCm/SYCL), HLOG must be a no-op:
//   - CTP_LOG resolves through CrossSingleton<Logger>::GetInstance(),
//     which holds a function-local static. DPC++ rejects non-const
//     statics in SYCL kernels.
//   - The arguments (cast to (void)) need to still be parsed so callers
//     don't accidentally rely on lazy evaluation.
//   - Defensive HLOG calls in transitively-reachable code (e.g. inside
//     ctp::ipc::vector growth, deserialization paths) get traced by DPC++
//     during kernel JIT even when not executed; making this a no-op
//     stops that trace at the HLOG seam.
#if !CTP_IS_DEVICE_PASS
#define HLOG(LOG_CODE, ...)                                               \
  do {                                                                    \
    if constexpr (LOG_CODE >= CTP_LOG_LEVEL) {                           \
      CTP_LOG->Log<LOG_CODE>(__FILE__, __func__, __LINE__, __VA_ARGS__); \
    }                                                                     \
  } while (false)
#else
#define HLOG(LOG_CODE, ...) ((void)0)
#endif

/**
 * Logger class for handling log output
 *
 * Supports:
 * - Runtime log level filtering via CTP_LOG_LEVEL environment variable
 * - File output via CTP_LOG_OUT environment variable
 * - Routing to stdout (debug/info) or stderr (warning/error/fatal)
 */
class Logger {
 public:
  FILE *fout_;
  int runtime_log_level_;  /**< Runtime log level threshold */

  CTP_CROSS_FUN
  Logger() {
#if CTP_IS_HOST && !CTP_IS_DEVICE_PASS
    fout_ = nullptr;
    runtime_log_level_ = CTP_LOG_LEVEL;  // Default to compile-time level

    // Check for runtime log level override
    std::string level_env = ctp::SystemInfo::Getenv(
        "CTP_LOG_LEVEL", ctp::Unit<size_t>::Megabytes(1));
    if (!level_env.empty()) {
      // Parse log level - accept both numeric and string values
      if (level_env == "debug" || level_env == "DEBUG" || level_env == "0") {
        runtime_log_level_ = kDebug;
      } else if (level_env == "info" || level_env == "INFO" || level_env == "1") {
        runtime_log_level_ = kInfo;
      } else if (level_env == "success" || level_env == "SUCCESS" || level_env == "2") {
        runtime_log_level_ = kSuccess;
      } else if (level_env == "warning" || level_env == "WARNING" || level_env == "3") {
        runtime_log_level_ = kWarning;
      } else if (level_env == "error" || level_env == "ERROR" || level_env == "4") {
        runtime_log_level_ = kError;
      } else if (level_env == "fatal" || level_env == "FATAL" || level_env == "5") {
        runtime_log_level_ = kFatal;
      } else {
        // Try to parse as integer
        try {
          runtime_log_level_ = std::stoi(level_env);
        } catch (...) {
          // Keep default on parse failure
        }
      }
    }

    // Check for file output
    std::string env = ctp::SystemInfo::Getenv(
        "CTP_LOG_OUT", ctp::Unit<size_t>::Megabytes(1));
    if (!env.empty()) {
      fout_ = fopen(env.c_str(), "w");
    }
#endif
  }

  /**
   * Get the string representation of a log level
   * @param level The log level
   * @return String name of the log level
   */
  CTP_CROSS_FUN
  static const char* GetLevelString(int level) {
    switch (level) {
      case kDebug: return "DEBUG";
      case kInfo: return "INFO";
      case kSuccess: return "SUCCESS";
      case kWarning: return "WARNING";
      case kError: return "ERROR";
      case kFatal: return "FATAL";
      default: return "UNKNOWN";
    }
  }

  /**
   * Get the ANSI color code for a log level
   * @param level The log level
   * @return ANSI escape sequence for the log level color
   */
  CTP_CROSS_FUN
  static const char* GetLevelColor(int level) {
    switch (level) {
      case kDebug: return "\033[90m";    // Dark Grey
      case kInfo: return "\033[97m";     // White
      case kSuccess: return "\033[32m";  // Green
      case kWarning: return "\033[33m";  // Yellow
      case kError: return "\033[31m";    // Red
      case kFatal: return "\033[31m";    // Red
      default: return "\033[0m";         // Reset
    }
  }

  /**
   * Check if a log level should be output based on runtime level
   * @param level The log level to check
   * @return true if the message should be logged
   */
  CTP_CROSS_FUN
  bool ShouldLog(int level) const {
    return level >= runtime_log_level_;
  }

  template <typename... Args>
  CTP_CROSS_FUN void Print(const char *fmt, Args &&...args) {
#if CTP_IS_HOST && !CTP_IS_DEVICE_PASS
    std::string msg = ctp::Formatter::format(fmt, std::forward<Args>(args)...);
    std::string out = ctp::Formatter::format("{}\n", msg);
    // Logs go to stderr, never stdout: stdout is reserved for a program's actual
    // data output (e.g. h5dump through the CTE VOL connector), which log lines
    // would otherwise corrupt.
    std::cerr << out;
    if (fout_) {
      fwrite(out.data(), 1, out.size(), fout_);
    }
#endif
  }

  template <int LOG_CODE, typename... Args>
  CTP_CROSS_FUN void Log(const char *path, const char *func, int line,
                          const char *fmt, Args &&...args) {
#if CTP_IS_HOST && !CTP_IS_DEVICE_PASS
    // Runtime log level check
    if (!ShouldLog(LOG_CODE)) {
      return;
    }

    const char* level = GetLevelString(LOG_CODE);
    const char* color = GetLevelColor(LOG_CODE);
    const char* reset = "\033[0m";
    std::string msg = ctp::Formatter::format(fmt, std::forward<Args>(args)...);
    int tid = SystemInfo::GetTid();
    std::string out = ctp::Formatter::format(
        "{}{}:{} {} {} {} {}{}\n",
        color, path, line, level, tid, func, msg, reset);

    // All log levels go to stderr. stdout is reserved for a program's actual
    // data output — e.g. h5dump reading through the CTE HDF5 VOL connector emits
    // the dumped dataset on stdout, and any log line written there corrupts it.
    std::cerr << out;
    fflush(stderr);

    // Write to file without color codes
    if (fout_) {
      std::string file_out = ctp::Formatter::format(
          "{}:{} {} {} {} {}\n", path, line, level, tid, func, msg);
      fwrite(file_out.data(), 1, file_out.size(), fout_);
      fflush(fout_);
    }

    // Fatal errors terminate the program
    if (LOG_CODE == kFatal) {
      exit(1);
    }
#endif
  }
};

}  // namespace ctp

#endif  // CTP_SHM_INCLUDE_HSHM_SHM_UTIL_LOGGING_H_
