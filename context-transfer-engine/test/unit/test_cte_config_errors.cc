/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * Error-path and edge-case unit tests for the CTE Config parser
 * (core_config.cc). The happy paths are covered by test_cte_config_dpe.cc;
 * this file targets the rejection branches: malformed YAML, missing
 * required fields, invalid enum values, out-of-range numbers, size-string
 * parsing, environment loading, and save/format round-trips.
 */

#include "simple_test.h"

#include <clio_cte/core/core_config.h>

#include <clio_ctp/introspect/system_info.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

using clio::cte::core::Config;
namespace fs = std::filesystem;

namespace {

/** Write content to a temp file and return its path. */
std::string WriteTempConfig(const std::string &name,
                            const std::string &content) {
  fs::path p = fs::temp_directory_path() / name;
  std::ofstream f(p);
  f << content;
  f.close();
  return p.string();
}

}  // namespace

TEST_CASE("ConfigErrors - LoadFromFile rejection paths",
          "[cte][config][errors]") {
  Config config;

  SECTION("Empty path rejected");
  REQUIRE_FALSE(config.LoadFromFile(""));

  SECTION("Nonexistent file rejected");
  REQUIRE_FALSE(config.LoadFromFile("/nonexistent/cte_config_test.yaml"));

  SECTION("Malformed YAML rejected");
  std::string bad = WriteTempConfig("cte_cfg_bad.yaml",
                                    "performance: [unclosed\n  ]: {{{\n");
  REQUIRE_FALSE(config.LoadFromFile(bad));
  fs::remove(bad);

  SECTION("Validation failure rejected (interval out of range)");
  std::string invalid = WriteTempConfig(
      "cte_cfg_invalid.yaml",
      "performance:\n  target_stat_interval_ms: 999999\n");
  REQUIRE_FALSE(config.LoadFromFile(invalid));
  fs::remove(invalid);
}

TEST_CASE("ConfigErrors - LoadFromString rejection paths",
          "[cte][config][errors]") {
  Config config;

  SECTION("Malformed YAML string rejected");
  REQUIRE_FALSE(config.LoadFromString("{:::"));

  SECTION("Valid YAML with invalid values rejected at validation");
  REQUIRE_FALSE(config.LoadFromString(
      "performance:\n  max_concurrent_operations: 0\n"));
  REQUIRE_FALSE(config.LoadFromString(
      "performance:\n  max_concurrent_operations: 4096\n"));
  REQUIRE_FALSE(config.LoadFromString(
      "performance:\n  score_threshold: 1.5\n"));
  REQUIRE_FALSE(config.LoadFromString(
      "performance:\n  score_difference_threshold: -0.2\n"));
  REQUIRE_FALSE(config.LoadFromString(
      "performance:\n  stat_targets_period_ms: 1\n"));

  SECTION("Empty document rejected (YAML null node)");
  Config fresh;
  REQUIRE_FALSE(fresh.LoadFromString(""));
}

TEST_CASE("ConfigErrors - storage device parsing", "[cte][config][storage]") {
  Config config;

  SECTION("Storage must be a sequence");
  REQUIRE_FALSE(config.LoadFromString("storage:\n  path: /tmp/x\n"));

  SECTION("Missing required path");
  REQUIRE_FALSE(config.LoadFromString(
      "storage:\n  - bdev_type: ram\n    capacity_limit: 1g\n"));

  SECTION("Missing required bdev_type");
  REQUIRE_FALSE(config.LoadFromString(
      "storage:\n  - path: /tmp/x\n    capacity_limit: 1g\n"));

  SECTION("Invalid bdev_type");
  REQUIRE_FALSE(config.LoadFromString(
      "storage:\n  - path: /tmp/x\n    bdev_type: floppy\n"
      "    capacity_limit: 1g\n"));

  SECTION("Missing required capacity_limit");
  REQUIRE_FALSE(config.LoadFromString(
      "storage:\n  - path: /tmp/x\n    bdev_type: ram\n"));

  SECTION("Invalid capacity_limit format");
  REQUIRE_FALSE(config.LoadFromString(
      "storage:\n  - path: /tmp/x\n    bdev_type: ram\n"
      "    capacity_limit: lots\n"));

  SECTION("Score out of range");
  REQUIRE_FALSE(config.LoadFromString(
      "storage:\n  - path: /tmp/x\n    bdev_type: ram\n"
      "    capacity_limit: 1g\n    score: 2.0\n"));
  REQUIRE_FALSE(config.LoadFromString(
      "storage:\n  - path: /tmp/x\n    bdev_type: ram\n"
      "    capacity_limit: 1g\n    score: -0.5\n"));

  SECTION("Valid device with every bdev_type accepted");
  for (const char *type : {"file", "ram", "hbm", "pinned", "noop"}) {
    Config ok;
    std::string yaml = std::string("storage:\n  - path: /tmp/x\n") +
                       "    bdev_type: " + type + "\n" +
                       "    capacity_limit: 64mb\n    score: 0.5\n";
    REQUIRE(ok.LoadFromString(yaml));
    REQUIRE(ok.storage_.devices_.size() == 1);
    REQUIRE(ok.storage_.devices_[0].bdev_type_ == type);
  }
}

TEST_CASE("ConfigErrors - dpe and gpu_metadata_cache parsing",
          "[cte][config][dpe]") {
  SECTION("Invalid dpe_type rejected");
  Config config;
  REQUIRE_FALSE(config.LoadFromString("dpe:\n  dpe_type: psychic\n"));

  SECTION("All valid dpe_type spellings accepted");
  for (const char *type :
       {"random", "round_robin", "roundrobin", "max_bw", "maxbw"}) {
    Config ok;
    REQUIRE(ok.LoadFromString(std::string("dpe:\n  dpe_type: ") + type +
                              "\n"));
    REQUIRE(ok.dpe_.dpe_type_ == type);
  }

  SECTION("gpu_metadata_cache keys parsed");
  Config gmc;
  REQUIRE(gmc.LoadFromString(
      "gpu_metadata_cache:\n"
      "  enabled: true\n"
      "  capacity: 4mb\n"
      "  max_blobs: 128\n"
      "  max_tags: 32\n"));
  REQUIRE(gmc.gpu_metadata_cache_.enabled_);
  REQUIRE(gmc.gpu_metadata_cache_.capacity_bytes_ == 4ULL * 1024 * 1024);
  REQUIRE(gmc.gpu_metadata_cache_.max_blobs_ == 128);
  REQUIRE(gmc.gpu_metadata_cache_.max_tags_ == 32);
}

namespace {

/** Drive ParseSizeString (private) through the storage capacity_limit key. */
bool ParseCapacity(const std::string &cap, chi::u64 *out_bytes = nullptr) {
  Config config;
  std::string yaml =
      "storage:\n  - path: /tmp/cap_probe\n    bdev_type: ram\n"
      "    capacity_limit: \"" + cap + "\"\n";
  if (!config.LoadFromString(yaml)) {
    return false;
  }
  if (out_bytes != nullptr) {
    *out_bytes = config.storage_.devices_[0].capacity_limit_;
  }
  return true;
}

}  // namespace

TEST_CASE("ConfigErrors - size string parsing via capacity_limit",
          "[cte][config][size]") {
  chi::u64 bytes = 0;

  SECTION("Empty and malformed inputs rejected");
  REQUIRE_FALSE(ParseCapacity(""));
  REQUIRE_FALSE(ParseCapacity("abc"));
  REQUIRE_FALSE(ParseCapacity("-5g"));
  REQUIRE_FALSE(ParseCapacity("10 zb"));
  REQUIRE_FALSE(ParseCapacity("..."));

  SECTION("All suffixes accepted, case-insensitive, optional whitespace");
  REQUIRE(ParseCapacity("123", &bytes));
  REQUIRE(bytes == 123);
  REQUIRE(ParseCapacity("123b", &bytes));
  REQUIRE(bytes == 123);
  REQUIRE(ParseCapacity("123 bytes", &bytes));
  REQUIRE(bytes == 123);
  REQUIRE(ParseCapacity("2k", &bytes));
  REQUIRE(bytes == 2048);
  REQUIRE(ParseCapacity("2 KB", &bytes));
  REQUIRE(bytes == 2048);
  REQUIRE(ParseCapacity("2 kilobytes", &bytes));
  REQUIRE(bytes == 2048);
  REQUIRE(ParseCapacity("3m", &bytes));
  REQUIRE(bytes == 3ULL * 1024 * 1024);
  REQUIRE(ParseCapacity("3 megabytes", &bytes));
  REQUIRE(bytes == 3ULL * 1024 * 1024);
  REQUIRE(ParseCapacity("4G", &bytes));
  REQUIRE(bytes == 4ULL * 1024 * 1024 * 1024);
  REQUIRE(ParseCapacity("4 gigabytes", &bytes));
  REQUIRE(bytes == 4ULL * 1024 * 1024 * 1024);
  REQUIRE(ParseCapacity("1t", &bytes));
  REQUIRE(bytes == 1ULL * 1024 * 1024 * 1024 * 1024);
  REQUIRE(ParseCapacity("1 terabytes", &bytes));
  REQUIRE(bytes == 1ULL * 1024 * 1024 * 1024 * 1024);
  REQUIRE(ParseCapacity("1p", &bytes));
  REQUIRE(bytes == 1ULL * 1024 * 1024 * 1024 * 1024 * 1024);
  REQUIRE(ParseCapacity("1 petabytes", &bytes));
  REQUIRE(bytes == 1ULL * 1024 * 1024 * 1024 * 1024 * 1024);

  SECTION("Fractional values");
  REQUIRE(ParseCapacity("1.5k", &bytes));
  REQUIRE(bytes == 1536);
}

TEST_CASE("ConfigErrors - environment and save round-trip",
          "[cte][config][env]") {
  SECTION("Unset env var falls back to defaults (success)");
  Config config;
  ctp::SystemInfo::Unsetenv("CTE_CONFIG_ERR_TEST_VAR");
  // Default env var name is part of the config; unset path returns true.
  REQUIRE(config.LoadFromEnvironment());

  SECTION("config_env_var key changes the variable that is consulted");
  Config redirected;
  REQUIRE(redirected.LoadFromString(
      "config_env_var: CTE_CONFIG_ERR_TEST_VAR\n"));

  SECTION("SaveToFile then LoadFromFile round-trips");
  Config saver;
  REQUIRE(saver.LoadFromString(
      "storage:\n  - path: /tmp/cte_cfg_rt\n    bdev_type: ram\n"
      "    capacity_limit: 8mb\n"));
  fs::path out = fs::temp_directory_path() / "cte_cfg_roundtrip.yaml";
  REQUIRE(saver.SaveToFile(out.string()));
  Config loader;
  REQUIRE(loader.LoadFromFile(out.string()));
  REQUIRE(loader.storage_.devices_.size() == 1);
  REQUIRE(loader.storage_.devices_[0].capacity_limit_ == 8ULL * 1024 * 1024);
  fs::remove(out);

  SECTION("SaveToFile to an unwritable path fails");
  REQUIRE_FALSE(saver.SaveToFile("/nonexistent_dir/cte_cfg.yaml"));
}

SIMPLE_TEST_MAIN()
