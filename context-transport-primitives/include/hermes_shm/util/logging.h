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

#ifndef HSHM_SHM_INCLUDE_HSHM_SHM_UTIL_LOGGING_H_
#define HSHM_SHM_INCLUDE_HSHM_SHM_UTIL_LOGGING_H_

#include <climits>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include "formatter.h"
#include "hermes_shm/introspect/system_info.h"

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
#ifndef HSHM_LOG_LEVEL
#define HSHM_LOG_LEVEL kInfo
#endif

namespace hshm {

/** Simplify access to Logger singleton */
#define HSHM_LOG hshm::CrossSingleton<hshm::Logger>::GetInstance()

/**
 * Hermes Print. Like printf, except types are inferred
 */
#define HIPRINT(...) HSHM_LOG->Print(__VA_ARGS__)

/**
 * Hermes SHM Log - Unified logging macro
 *
 * Messages with LOG_CODE < HSHM_LOG_LEVEL are compiled out entirely.
 * Messages with LOG_CODE >= HSHM_LOG_LEVEL are subject to runtime filtering
 * via the HSHM_LOG_LEVEL environment variable.
 *
 * @param LOG_CODE The log level (kDebug, kInfo, kWarning, kError, kFatal)
 * @param ... Format string and arguments
 */
#define HLOG(LOG_CODE, ...)                                               \
  do {                                                                    \
    if constexpr (LOG_CODE >= HSHM_LOG_LEVEL) {                           \
      HSHM_LOG->Log<LOG_CODE>(__FILE__, __func__, __LINE__, __VA_ARGS__); \
    }                                                                     \
  } while (false)

/**
 * Logger class for handling log output
 *
 * Supports:
 * - Runtime log level filtering via HSHM_LOG_LEVEL environment variable
 * - File output via HSHM_LOG_OUT environment variable
 * - Routing to stdout (debug/info) or stderr (warning/error/fatal)
 */
class Logger {
 public:
  FILE *fout_;
  int runtime_log_level_;  /**< Runtime log level threshold */

  HSHM_CROSS_FUN
  Logger() {
#if HSHM_IS_HOST
    fout_ = nullptr;
    runtime_log_level_ = HSHM_LOG_LEVEL;  // Default to compile-time level

    // Check for runtime log level override
    std::string level_env = hshm::SystemInfo::Getenv(
        "HSHM_LOG_LEVEL", hshm::Unit<size_t>::Megabytes(1));
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
    std::string env = hshm::SystemInfo::Getenv(
        "HSHM_LOG_OUT", hshm::Unit<size_t>::Megabytes(1));
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
  HSHM_CROSS_FUN
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
  HSHM_CROSS_FUN
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
  HSHM_CROSS_FUN
  bool ShouldLog(int level) const {
    return level >= runtime_log_level_;
  }

  template <typename... Args>
  HSHM_CROSS_FUN void Print(const char *fmt, Args &&...args) {
#if HSHM_IS_HOST
    std::string msg = hshm::Formatter::format(fmt, std::forward<Args>(args)...);
    std::string out = hshm::Formatter::format("{}\n", msg);
    std::cout << out;
    if (fout_) {
      fwrite(out.data(), 1, out.size(), fout_);
    }
#endif
  }

  template <int LOG_CODE, typename... Args>
  HSHM_CROSS_FUN void Log(const char *path, const char *func, int line,
                          const char *fmt, Args &&...args) {
#if HSHM_IS_HOST
    // Runtime log level check
    if (!ShouldLog(LOG_CODE)) {
      return;
    }

    const char* level = GetLevelString(LOG_CODE);
    const char* color = GetLevelColor(LOG_CODE);
    const char* reset = "\033[0m";
    std::string msg = hshm::Formatter::format(fmt, std::forward<Args>(args)...);
    int tid = SystemInfo::GetTid();
    std::string out = hshm::Formatter::format(
        "{}{}:{} {} {} {} {}{}\n",
        color, path, line, level, tid, func, msg, reset);

    // Route to appropriate output stream
    // Debug, Info, and Success go to stdout; Warning/Error/Fatal go to stderr
    if (LOG_CODE <= kSuccess) {
      std::cout << out;
      fflush(stdout);
    } else {
      std::cerr << out;
      fflush(stderr);
    }

    // Write to file without color codes
    if (fout_) {
      std::string file_out = hshm::Formatter::format(
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

}  // namespace hshm

#endif  // HSHM_SHM_INCLUDE_HSHM_SHM_UTIL_LOGGING_H_
