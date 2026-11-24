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

/**
 * OMNI YAML ingest for CAE. See include/clio_cae/core/omni_loader.h.
 */

#include <clio_cae/core/omni_loader.h>

#include <clio_ctp/util/config_parse.h>
#include <clio_ctp/util/logging.h>
#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace clio::cae::core {

namespace {

// Every per-transfer key LoadOmni reads. Anything else is warned about.
const std::set<std::string>& KnownTransferKeys() {
  static const std::set<std::string> keys = {
      "src",       "dst",       "format",         "depends_on",
      "range_off", "range_size", "src_token",     "dst_token",
      "dataset_filter", "run",  "run_out"};
  return keys;
}

// Env-expand `raw` and make it absolute, resolving a relative path against
// the directory that holds the OMNI file.
std::string ResolvePath(const std::string& raw,
                        const std::filesystem::path& base) {
  std::filesystem::path p(ctp::ConfigParse::ExpandPath(raw));
  if (p.is_relative()) p = base / p;
  return p.lexically_normal().string();
}

}  // namespace

std::vector<AssimilationCtx> LoadOmni(const std::string& omni_path,
                                      std::vector<RunStep>* runs) {
  HLOG(kInfo, "Loading OMNI file: {}", omni_path);

  YAML::Node config;
  try {
    config = YAML::LoadFile(omni_path);
  } catch (const YAML::Exception& e) {
    throw std::runtime_error("Failed to load OMNI file: " +
                             std::string(e.what()));
  }

  // Check for required 'transfers' key
  if (!config["transfers"]) {
    throw std::runtime_error("OMNI file missing required 'transfers' key");
  }

  const YAML::Node& transfers = config["transfers"];
  if (!transfers.IsSequence()) {
    throw std::runtime_error("OMNI 'transfers' must be a sequence/array");
  }

  std::vector<clio::cae::core::AssimilationCtx> contexts;
  contexts.reserve(transfers.size());
  const std::filesystem::path omni_dir =
      std::filesystem::absolute(omni_path).parent_path();

  // Parse each transfer entry
  for (size_t i = 0; i < transfers.size(); ++i) {
    const YAML::Node& transfer = transfers[i];

    // Validate required fields
    if (!transfer["src"]) {
      throw std::runtime_error("Transfer " + std::to_string(i + 1) +
                               " missing required 'src' field");
    }
    if (!transfer["dst"]) {
      throw std::runtime_error("Transfer " + std::to_string(i + 1) +
                               " missing required 'dst' field");
    }
    if (!transfer["format"]) {
      throw std::runtime_error("Transfer " + std::to_string(i + 1) +
                               " missing required 'format' field");
    }

    if (transfer.IsMap()) {
      for (const auto& kv : transfer) {
        const std::string key = kv.first.as<std::string>();
        if (KnownTransferKeys().count(key) == 0) {
          HLOG(kWarning, "Transfer {}: unknown key '{}' is ignored", (i + 1),
               key);
        }
      }
    }

    AssimilationCtx ctx;
    ctx.src = transfer["src"].as<std::string>();
    ctx.dst = transfer["dst"].as<std::string>();
    ctx.format = transfer["format"].as<std::string>();
    ctx.depends_on =
        transfer["depends_on"] ? transfer["depends_on"].as<std::string>() : "";
    ctx.range_off =
        transfer["range_off"] ? transfer["range_off"].as<size_t>() : 0;
    ctx.range_size =
        transfer["range_size"] ? transfer["range_size"].as<size_t>() : 0;

    // Parse tokens and expand environment variables
    if (transfer["src_token"]) {
      std::string raw_token = transfer["src_token"].as<std::string>();
      ctx.src_token = ctp::ConfigParse::ExpandPath(raw_token);
    }
    if (transfer["dst_token"]) {
      std::string raw_token = transfer["dst_token"].as<std::string>();
      ctx.dst_token = ctp::ConfigParse::ExpandPath(raw_token);
    }

    // Parse dataset_filter for HDF5 and other hierarchical formats
    if (transfer["dataset_filter"]) {
      const YAML::Node& filter = transfer["dataset_filter"];

      // Parse include_patterns
      if (filter["include_patterns"]) {
        const YAML::Node& include_node = filter["include_patterns"];
        if (include_node.IsSequence()) {
          for (size_t j = 0; j < include_node.size(); ++j) {
            ctx.include_patterns.push_back(include_node[j].as<std::string>());
          }
        }
      }

      // Parse exclude_patterns
      if (filter["exclude_patterns"]) {
        const YAML::Node& exclude_node = filter["exclude_patterns"];
        if (exclude_node.IsSequence()) {
          for (size_t j = 0; j < exclude_node.size(); ++j) {
            ctx.exclude_patterns.push_back(exclude_node[j].as<std::string>());
          }
        }
      }
    }

    // Optional run step: a script that turns src into run_out before ingest.
    if (transfer["run"] || transfer["run_out"]) {
      const std::string where = "Transfer " + std::to_string(i + 1);
      if (!transfer["run"]) {
        throw std::runtime_error(where + " has 'run_out' but no 'run' script");
      }
      if (!transfer["run_out"]) {
        throw std::runtime_error(
            where + " has 'run' but no 'run_out': name the file the script "
            "writes, which is what gets ingested");
      }
      if (runs == nullptr) {
        throw std::runtime_error(
            where + " has a 'run' step, which this caller does not execute");
      }
      RunStep step;
      step.transfer = i;
      step.script = ResolvePath(transfer["run"].as<std::string>(), omni_dir);
      step.out = ResolvePath(transfer["run_out"].as<std::string>(), omni_dir);
      runs->push_back(step);
    }

    contexts.push_back(ctx);

    HLOG(kInfo, "  Loaded transfer {}/{}: ", (i + 1), transfers.size());
    HLOG(kInfo, "    src: {}", ctx.src);
    HLOG(kInfo, "    dst: {}", ctx.dst);
    HLOG(kInfo, "    format: {}", ctx.format);
    if (runs != nullptr && !runs->empty() && runs->back().transfer == i) {
      HLOG(kInfo, "    run: {} -> {}", runs->back().script, runs->back().out);
    }
    if (!ctx.src_token.empty()) {
      HLOG(kInfo, "    src_token: <set>");
    }
    if (!ctx.dst_token.empty()) {
      HLOG(kInfo, "    dst_token: <set>");
    }
    if (!ctx.include_patterns.empty()) {
      std::string patterns_str = "    dataset_filter.include_patterns: [";
      for (size_t j = 0; j < ctx.include_patterns.size(); ++j) {
        if (j > 0) patterns_str += ", ";
        patterns_str += "\"" + ctx.include_patterns[j] + "\"";
      }
      patterns_str += "]";
      HIPRINT("{}", patterns_str);
    }
    if (!ctx.exclude_patterns.empty()) {
      std::string patterns_str = "    dataset_filter.exclude_patterns: [";
      for (size_t j = 0; j < ctx.exclude_patterns.size(); ++j) {
        if (j > 0) patterns_str += ", ";
        patterns_str += "\"" + ctx.exclude_patterns[j] + "\"";
      }
      patterns_str += "]";
      HIPRINT("{}", patterns_str);
    }
  }

  HLOG(kSuccess, "Successfully loaded {} transfer(s) from OMNI file", contexts.size());
  return contexts;
}

}  // namespace clio::cae::core
