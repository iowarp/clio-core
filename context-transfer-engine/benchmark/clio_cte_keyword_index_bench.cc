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

#include <clio_cte/core/keyword_index.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>

#if defined(__APPLE__)
#include <mach/mach.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

struct Args {
  std::size_t blobs_ = 1000;
  std::size_t matches_ = 10;
  std::size_t query_iters_ = 20;
  std::size_t payload_bytes_ = 128;
  std::size_t vocabulary_ = 100;
  std::size_t max_rss_mib_ = 0;
  bool valid_ = true;
};

/**
 * Parse a nonnegative integer command-line value.
 *
 * @param value Text value to parse.
 * @param flag Flag name used in error messages.
 * @param valid Set false when parsing fails.
 * @return Parsed value, or zero on failure.
 */
std::size_t ParseSize(const char *value, const char *flag, bool &valid) {
  char *end = nullptr;
  const unsigned long long parsed = std::strtoull(value, &end, 10);
  if (end == value || *end != '\0') {
    std::cerr << "Invalid value for " << flag << ": " << value << '\n';
    valid = false;
    return 0;
  }
  return static_cast<std::size_t>(parsed);
}

/**
 * Parse benchmark command-line arguments.
 *
 * @param argc Number of command-line arguments.
 * @param argv Command-line argument array.
 * @return Parsed benchmark arguments.
 */
Args ParseArgs(int argc, char **argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    if (i + 1 >= argc) {
      std::cerr << "Missing value for " << flag << '\n';
      args.valid_ = false;
      break;
    }
    const char *value = argv[++i];
    if (flag == "--blobs") {
      args.blobs_ = ParseSize(value, "--blobs", args.valid_);
    } else if (flag == "--matches") {
      args.matches_ = ParseSize(value, "--matches", args.valid_);
    } else if (flag == "--query-iters") {
      args.query_iters_ = ParseSize(value, "--query-iters", args.valid_);
    } else if (flag == "--payload-bytes") {
      args.payload_bytes_ =
          ParseSize(value, "--payload-bytes", args.valid_);
    } else if (flag == "--vocabulary") {
      args.vocabulary_ = ParseSize(value, "--vocabulary", args.valid_);
    } else if (flag == "--max-rss-mib") {
      args.max_rss_mib_ = ParseSize(value, "--max-rss-mib", args.valid_);
    } else {
      std::cerr << "Unknown flag: " << flag << '\n';
      args.valid_ = false;
    }
  }
  constexpr std::size_t kTargetKeywordBytes = 13;
  if (args.blobs_ == 0 || args.query_iters_ == 0 ||
      args.payload_bytes_ < kTargetKeywordBytes || args.vocabulary_ == 0 ||
      args.matches_ > args.blobs_) {
    std::cerr << "Require blobs, query-iters, and vocabulary > 0, "
                 "payload-bytes >= 13, and matches <= blobs\n";
    args.valid_ = false;
  }
  return args;
}

/**
 * Read the process's current resident set size.
 *
 * @return Resident bytes, or zero when unsupported.
 */
std::size_t CurrentResidentBytes() {
#if defined(__APPLE__)
  mach_task_basic_info_data_t info;
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  const kern_return_t result =
      task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                reinterpret_cast<task_info_t>(&info), &count);
  return result == KERN_SUCCESS ? static_cast<std::size_t>(info.resident_size)
                                : 0;
#elif defined(__linux__)
  std::ifstream statm("/proc/self/statm");
  std::size_t total_pages = 0;
  std::size_t resident_pages = 0;
  statm >> total_pages >> resident_pages;
  (void)total_pages;
  return resident_pages * static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
#else
  return 0;
#endif
}

/**
 * Convert a byte count to mebibytes.
 *
 * @param bytes Byte count.
 * @return Mebibyte value.
 */
double ToMebibytes(std::size_t bytes) {
  constexpr double kBytesPerMebibyte = 1024.0 * 1024.0;
  return static_cast<double>(bytes) / kBytesPerMebibyte;
}

/**
 * Convert a clock duration to milliseconds.
 *
 * @param duration Duration to convert.
 * @return Duration in milliseconds.
 */
double ToMilliseconds(Clock::duration duration) {
  return std::chrono::duration<double, std::milli>(duration).count();
}

/**
 * Build a deterministic fixed-size payload and an optional target term.
 *
 * @param index Blob sequence number.
 * @param matches Whether this blob should match the target keyword.
 * @param payload_bytes Desired payload size in bytes.
 * @param vocabulary Number of distinct non-target terms.
 * @return Blob text to index.
 */
std::string MakePayload(std::size_t index, bool matches,
                        std::size_t payload_bytes, std::size_t vocabulary) {
  const std::string filler = " term" + std::to_string(index % vocabulary);
  std::string payload;
  payload.reserve(payload_bytes);
  if (matches) {
    payload = "targetkeyword";
  }
  while (payload.size() < payload_bytes) {
    payload.append(filler, 0,
                   std::min(filler.size(), payload_bytes - payload.size()));
  }
  return payload;
}

/**
 * Determine whether the configured resident-memory ceiling was exceeded.
 *
 * @param args Benchmark configuration.
 * @param resident_bytes Current process resident memory in bytes.
 * @return True when a nonzero ceiling was exceeded.
 */
bool ExceedsMemoryLimit(const Args &args, std::size_t resident_bytes) {
  constexpr std::size_t kBytesPerMebibyte = 1024 * 1024;
  return args.max_rss_mib_ > 0 &&
         resident_bytes > args.max_rss_mib_ * kBytesPerMebibyte;
}

}  // namespace

/**
 * Run an isolated inverted-index scaling benchmark.
 *
 * @param argc Number of command-line arguments.
 * @param argv Command-line argument array.
 * @return Zero on success and nonzero for invalid arguments.
 */
int main(int argc, char **argv) {
  const Args args = ParseArgs(argc, argv);
  if (!args.valid_) {
    return 1;
  }

  const std::size_t rss_before = CurrentResidentBytes();
  clio::cte::core::KeywordIndex index;
  const auto build_start = Clock::now();
  std::size_t indexed_blobs = 0;
  bool memory_limit_hit = false;
  for (; indexed_blobs < args.blobs_; ++indexed_blobs) {
    const std::string payload =
        MakePayload(indexed_blobs, indexed_blobs < args.matches_,
                    args.payload_bytes_, args.vocabulary_);
    index.Update(1, 0, "blob_" + std::to_string(indexed_blobs), payload.data(),
                 payload.size());
    if ((indexed_blobs + 1) % 10000 == 0 &&
        ExceedsMemoryLimit(args, CurrentResidentBytes())) {
      memory_limit_hit = true;
      ++indexed_blobs;
      break;
    }
  }
  const double build_ms = ToMilliseconds(Clock::now() - build_start);
  const std::size_t rss_after = CurrentResidentBytes();

  const std::unordered_set<std::string> query = {"targetkeyword"};
  double query_total_ms = 0.0;
  double query_min_ms = 1e30;
  double query_max_ms = 0.0;
  std::size_t candidates = 0;
  for (std::size_t iteration = 0; iteration < args.query_iters_; ++iteration) {
    const auto query_start = Clock::now();
    const auto snapshot = index.Find(query);
    const double query_ms = ToMilliseconds(Clock::now() - query_start);
    query_total_ms += query_ms;
    query_min_ms = std::min(query_min_ms, query_ms);
    query_max_ms = std::max(query_max_ms, query_ms);
    candidates = snapshot.documents_.size();
  }

  const std::size_t rss_delta =
      rss_after >= rss_before ? rss_after - rss_before : 0;
  std::cout << "[KEYWORD_INDEX_BENCH]"
            << " requested_blobs=" << args.blobs_
            << " indexed_blobs=" << indexed_blobs
            << " matches=" << args.matches_
            << " candidates=" << candidates << " build_ms=" << build_ms
            << " query_avg_ms=" << query_total_ms / args.query_iters_
            << " query_min_ms=" << query_min_ms
            << " query_max_ms=" << query_max_ms
            << " rss_before_mib=" << ToMebibytes(rss_before)
            << " rss_after_mib=" << ToMebibytes(rss_after)
            << " rss_delta_mib=" << ToMebibytes(rss_delta)
            << " payload_bytes=" << args.payload_bytes_
            << " vocabulary=" << args.vocabulary_
            << " memory_limit_hit=" << memory_limit_hit << '\n';
  const std::size_t expected_matches =
      std::min(args.matches_, indexed_blobs);
  if (memory_limit_hit) {
    return 3;
  }
  return candidates == expected_matches ? 0 : 2;
}
