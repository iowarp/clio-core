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

#ifndef HSHM_CONFIG_PARSE_PARSER_H
#define HSHM_CONFIG_PARSE_PARSER_H

#include <float.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

#include <cstdlib>
#include <iomanip>
#include <list>
// <regex> removed: ExpandPath now uses std::string::find to avoid std::locale
// false positives under MemorySanitizer (uninstrumented conda libstdc++).
#include <string>
// MSan: mark yaml Node scalar bytes as initialized after YAML::Load(File).
// Precompiled yaml-cpp.so propagates "uninitialized" bytes (from uninstrumented
// libstdc++) through our instrumented _M_assign into Node scalar strings;
// __msan_unpoison fixes the false-positive use-of-uninitialized-value reports.
#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
#include <sanitizer/msan_interface.h>
#endif
#endif

#include "formatter.h"
#include "hermes_shm/constants/macros.h"
#include "logging.h"
#include "yaml-cpp/yaml.h"

namespace hshm {

class ConfigParse {
 public:
  static void rm_char(std::string &str, char ch) {
    str.erase(std::remove(str.begin(), str.end(), ch), str.end());
  }

  /**
   * parse a hostfile string
   * [] represents a range to generate
   * ; represents a new host name completely
   *
   * Example: hello[00-09,10]-40g;hello2[11-13]-40g
   * */
  static void ParseHostNameString(std::string hostname_set_str,
                                  std::vector<std::string> &list) {
    // Remove all whitespace characters from host name
    rm_char(hostname_set_str, ' ');
    rm_char(hostname_set_str, '\n');
    rm_char(hostname_set_str, '\r');
    rm_char(hostname_set_str, '\t');
    if (hostname_set_str.size() == 0) {
      return;
    }
    // Expand hostnames
    std::stringstream ss(hostname_set_str);
    while (ss.good()) {
      // Get the current host
      std::string hostname;
      std::getline(ss, hostname, ';');

      // Divide the hostname string into prefix, ranges, and suffix
      auto lbracket = hostname.find_first_of('[');
      auto rbracket = hostname.find_last_of(']');
      std::string prefix, ranges_str, suffix;
      if (lbracket != std::string::npos && rbracket != std::string::npos) {
        /*
         * For example, hello[00-09]-40g
         * lbracket = 5
         * rbracket = 11
         * prefix = hello (length: 5)
         * range = 00-09 (length: 5)
         * suffix = -40g (length: 4)
         * */
        prefix = hostname.substr(0, lbracket);
        ranges_str = hostname.substr(lbracket + 1, rbracket - lbracket - 1);
        suffix = hostname.substr(rbracket + 1);
      } else {
        list.emplace_back(hostname);
        continue;
      }

      // Parse the range list into a tuple of (min, max, num_width)
      std::stringstream ss_ranges(ranges_str);
      std::vector<std::tuple<int, int, int>> ranges;
      while (ss_ranges.good()) {
        // Parse ',' and remove spaces
        std::string range_str;
        std::getline(ss_ranges, range_str, ',');
        rm_char(range_str, ' ');

        // Divide the range by '-'
        auto dash = range_str.find_first_of('-');
        if (dash != std::string::npos) {
          int min = 0, max = 0;
          // Get the minimum and maximum value
          std::string min_str = range_str.substr(0, dash);
          std::string max_str = range_str.substr(dash + 1);
          std::stringstream(min_str) >> min;
          std::stringstream(max_str) >> max;

          // Check for leading 0s
          int num_width = 0;
          if (min_str.size() == max_str.size()) {
            num_width = min_str.size();
          }

          // Place the range with width
          ranges.emplace_back(min, max, num_width);
        } else if (range_str.size()) {
          int val = 0;
          std::stringstream(range_str) >> val;
          ranges.emplace_back(val, val, range_str.size());
        }
      }

      // Expand the host names by each range
      for (auto &range : ranges) {
        int min = std::get<0>(range);
        int max = std::get<1>(range);
        int num_width = std::get<2>(range);

        for (int i = min; i <= max; ++i) {
          std::stringstream host_ss;
          host_ss << prefix;
          host_ss << std::setw(num_width) << std::setfill('0') << i;
          host_ss << suffix;
          list.emplace_back(host_ss.str());
        }
      }
    }
  }

  /** parse the suffix of \a num_text NUMBER text */
  static std::string ParseNumberSuffix(const std::string &num_text) {
    size_t i;
    for (i = 0; i < num_text.size(); ++i) {
      char c = num_text[i];
      // Skip numbers
      if ('0' <= c && c <= '9') {
        continue;
      }
      // Skip whitespace
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
        continue;
      }
      // Skip period (for floats)
      if (c == '.') {
        continue;
      }
      break;
    }
    return std::string(num_text.begin() + i, num_text.end());
    ;
  }

  /** parse the number of \a num_text NUMBER text */
  template <typename T>
  static T ParseNumber(const std::string &num_text) {
    T size = {};
    if (num_text == "inf") {
      return std::numeric_limits<T>::max();
    }
    std::stringstream(num_text) >> size;
    return size;
  }

  /** Converts \a size_text SIZE text into a size_t */
  static hshm::u64 ParseSize(const std::string &size_text) {
    auto size = ParseNumber<double>(size_text);
    if (size_text == "inf") {
      return std::numeric_limits<hshm::u64>::max();
    }
    std::string suffix = ParseNumberSuffix(size_text);
    if (suffix.empty()) {
      return Unit<hshm::u64>::Bytes(size);
    } else if (suffix[0] == 'k' || suffix[0] == 'K') {
      return hshm::Unit<hshm::u64>::Kilobytes(size);
    } else if (suffix[0] == 'm' || suffix[0] == 'M') {
      return hshm::Unit<hshm::u64>::Megabytes(size);
    } else if (suffix[0] == 'g' || suffix[0] == 'G') {
      return hshm::Unit<hshm::u64>::Gigabytes(size);
    } else if (suffix[0] == 't' || suffix[0] == 'T') {
      return hshm::Unit<hshm::u64>::Terabytes(size);
    } else if (suffix[0] == 'p' || suffix[0] == 'P') {
      return hshm::Unit<hshm::u64>::Petabytes(size);
    } else {
      HLOG(kFatal, "Could not parse the size: {}", size_text);
      exit(1);
    }
  }

  /** Returns bandwidth (bytes / second) */
  static hshm::u64 ParseBandwidth(const std::string &size_text) {
    return ParseSize(size_text);
  }

  /** Returns latency (nanoseconds) */
  static hshm::u64 ParseLatency(const std::string &latency_text) {
    auto size = ParseNumber<double>(latency_text);
    std::string suffix = ParseNumberSuffix(latency_text);
    if (suffix.empty()) {
      return Unit<hshm::u64>::Bytes(size);
    } else if (suffix[0] == 'n' || suffix[0] == 'N') {
      return Unit<hshm::u64>::Bytes(size);
    } else if (suffix[0] == 'u' || suffix[0] == 'U') {
      return hshm::Unit<hshm::u64>::Kilobytes(size);
    } else if (suffix[0] == 'm' || suffix[0] == 'M') {
      return hshm::Unit<hshm::u64>::Megabytes(size);
    } else if (suffix[0] == 's' || suffix[0] == 'S') {
      return hshm::Unit<hshm::u64>::Terabytes(size);
    }
    HLOG(kFatal, "Could not parse the latency: {}", latency_text);
    return 0;
  }

  /** Expands all environment variables in a path string */
  static std::string ExpandPath(std::string path) {
    size_t pos = 0;
    while ((pos = path.find("${", pos)) != std::string::npos) {
      size_t end = path.find("}", pos + 2);
      if (end == std::string::npos) break;
      std::string env_name = path.substr(pos + 2, end - pos - 2);
      std::string env_val = hshm::SystemInfo::Getenv(
          env_name.c_str(), hshm::Unit<size_t>::Megabytes(1));
      path.replace(pos, end - pos + 1, env_val);
      pos += env_val.size();
    }
    return path;
  }

  /** Parse hostfile */
  static std::vector<std::string> ParseHostfile(const std::string &path) {
    std::vector<std::string> hosts;
    std::ifstream file(path);
    if (file.is_open()) {
      std::string line;
      while (std::getline(file, line)) {
        hshm::ConfigParse::ParseHostNameString(line, hosts);
      }
      file.close();
    } else {
      HLOG(kError, "Could not open the hostfile: {}", path);
    }
    return hosts;
  }

#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
  /**
   * Traverse a fully-loaded YAML tree and mark all scalar string bytes as
   * initialized in MSan's shadow memory.
   *
   * Background: precompiled yaml-cpp.so calls back into our MSan-instrumented
   * binary via PLT (e.g. std::string::_M_assign).  The uninstrumented scanner
   * bytes are propagated into yaml Node scalar strings and appear
   * "uninitialized" to MSan.  After YAML::Load(File) returns, the tree is
   * structurally valid; we just need to tell MSan the character bytes are OK.
   * Called by BaseConfig::LoadText and BaseConfig::LoadFromFile.
   */
  static void MsanUnpoisonYamlNode(const YAML::Node &node) {
    if (!node.IsDefined() || node.IsNull()) return;
    if (node.IsScalar()) {
      const std::string &s = node.Scalar();
      if (!s.empty()) {
        __msan_unpoison(s.data(), s.size());
      }
    } else if (node.IsSequence()) {
      for (auto it = node.begin(); it != node.end(); ++it) {
        MsanUnpoisonYamlNode(*it);
      }
    } else if (node.IsMap()) {
      for (auto it = node.begin(); it != node.end(); ++it) {
        MsanUnpoisonYamlNode(it->first);
        MsanUnpoisonYamlNode(it->second);
      }
    }
  }
#endif
#endif
};

/**
 * Base class for configuration files
 * */
class BaseConfig {
 public:
  /** load configuration from a string */
  void LoadText(const std::string &config_string, bool with_default = true) {
    if (with_default) {
      LoadDefault();
    }
    if (config_string.size() == 0) {
      return;
    }
    YAML::Node yaml_conf = YAML::Load(config_string);
#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
    hshm::ConfigParse::MsanUnpoisonYamlNode(yaml_conf);
#endif
#endif
    ParseYAML(yaml_conf);
  }

  /** load configuration from file */
  void LoadFromFile(const std::string &path, bool with_default = true) {
    if (with_default) {
      LoadDefault();
    }
    if (path.size() == 0) {
      return;
    }
    auto real_path = hshm::ConfigParse::ExpandPath(path);
    try {
      YAML::Node yaml_conf = YAML::LoadFile(real_path);
#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
      hshm::ConfigParse::MsanUnpoisonYamlNode(yaml_conf);
#endif
#endif
      ParseYAML(yaml_conf);
    } catch (std::exception &e) {
      HLOG(kFatal, e.what());
    }
  }

  /** load the default configuration */
  virtual void LoadDefault() = 0;

 public:
  /** parse \a list_node vector from configuration file in YAML */
  template <typename T, typename VEC_TYPE = std::vector<T>>
  static void ParseVector(YAML::Node list_node, VEC_TYPE &list) {
    for (auto val_node : list_node) {
      list.emplace_back(val_node.as<T>());
    }
  }

  /** clear + parse \a list_node vector from configuration file in YAML */
  template <typename T, typename VEC_TYPE = std::vector<T>>
  static void ClearParseVector(YAML::Node list_node, VEC_TYPE &list) {
    list.clear();
    for (auto val_node : list_node) {
      list.emplace_back(val_node.as<T>());
    }
  }

 private:
  virtual void ParseYAML(YAML::Node &yaml_conf) = 0;
};

}  // namespace hshm

#endif  // HSHM_CONFIG_PARSE_PARSER_H
