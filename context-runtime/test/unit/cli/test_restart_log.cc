/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * Unit tests for the restart write-ahead log (clio::run::RestartLog).
 *
 * Pure file-level tests — no daemon. Exercises append, live-set computation
 * (add/rm with last-op-wins), persistence across reopen, and compaction
 * (including the "already minimal → no-op" path).
 */

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

#include <clio_runtime/restart_log.h>

#include "simple_test.h"

namespace fs = std::filesystem;

namespace {
bool Contains(const std::vector<std::string>& v, const std::string& s) {
  return std::find(v.begin(), v.end(), s) != v.end();
}
}  // namespace

TEST_CASE("RestartLog - add/rm live set, persistence, compaction",
          "[cli][restart_log]") {
  const fs::path dir = fs::temp_directory_path() / "clio_restart_log_test";
  fs::remove_all(dir);
  fs::create_directories(dir);
  const std::string log_path = (dir / "restart_log.bin").string();

  const std::string a = "/abs/path/a.yaml";
  const std::string b = "/abs/path/b.yaml";
  const std::string c = "/abs/path/c.yaml";

  // Empty log.
  {
    clio::run::RestartLog log(log_path);
    REQUIRE(log.ReadAll().empty());
    REQUIRE(log.LiveSet().empty());
    REQUIRE_FALSE(log.IsRestartable(a));
  }

  // Adds accumulate in order; duplicate add does not duplicate the live entry.
  {
    clio::run::RestartLog log(log_path);
    REQUIRE(log.AppendAdd(a));
    REQUIRE(log.AppendAdd(b));
    REQUIRE(log.AppendAdd(a));  // duplicate
    auto live = log.LiveSet();
    REQUIRE(live.size() == 2);
    REQUIRE(Contains(live, a));
    REQUIRE(Contains(live, b));
    REQUIRE(log.IsRestartable(a));
    REQUIRE(log.IsRestartable(b));
  }

  // rm removes from the live set; survives reopen (persistence).
  {
    clio::run::RestartLog log(log_path);
    REQUIRE(log.AppendRm(a));
    clio::run::RestartLog reopened(log_path);
    auto live = reopened.LiveSet();
    REQUIRE(live.size() == 1);
    REQUIRE(Contains(live, b));
    REQUIRE_FALSE(reopened.IsRestartable(a));
  }

  // Compaction collapses the add(a)/add(b)/add(a)/rm(a) history to one add(b).
  {
    clio::run::RestartLog log(log_path);
    REQUIRE(log.ReadAll().size() > 1);  // still has the full history
    REQUIRE(log.Compact());
    auto entries = log.ReadAll();
    REQUIRE(entries.size() == 1);
    REQUIRE(entries[0].op == clio::run::RestartLog::Op::kAdd);
    REQUIRE(entries[0].path == b);
    REQUIRE(log.LiveSet() == std::vector<std::string>{b});
  }

  // Compact is a no-op (and does not corrupt) when already minimal.
  {
    clio::run::RestartLog log(log_path);
    REQUIRE(log.Compact());
    auto entries = log.ReadAll();
    REQUIRE(entries.size() == 1);
    REQUIRE(entries[0].path == b);
  }

  // A dangling rm (no matching add) is dropped by compaction.
  {
    clio::run::RestartLog log(log_path);
    REQUIRE(log.AppendAdd(c));
    REQUIRE(log.AppendRm(c));
    REQUIRE(log.Compact());
    auto live = log.LiveSet();
    REQUIRE(live.size() == 1);
    REQUIRE(Contains(live, b));
    REQUIRE_FALSE(Contains(live, c));
  }

  fs::remove_all(dir);
}

SIMPLE_TEST_MAIN()
