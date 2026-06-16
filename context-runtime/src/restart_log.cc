/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 */

/**
 * RestartLog implementation — append-only binary WAL of compose files to
 * re-launch on `clio_run start`. See restart_log.h for the format.
 */

#include "clio_runtime/restart_log.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <unordered_set>

#include "clio_ctp/util/logging.h"

namespace clio::run {

namespace fs = std::filesystem;

std::string RestartLog::DefaultPath() {
  // An explicit override wins: lets a test (or an admin) point the WAL at an
  // isolated location instead of the shared ~/.clio/restart_log.bin. Child
  // clio_run processes inherit the env var, so daemon + CLI agree on the path.
  const char* override_path = std::getenv("CLIO_RESTART_LOG");
  if (override_path != nullptr && override_path[0] != '\0') {
    return override_path;
  }
  const char* home = std::getenv("HOME");
  std::string base = (home != nullptr && home[0] != '\0') ? home : ".";
  return base + "/.clio/restart_log.bin";
}

bool RestartLog::EnsureParentDir() const {
  fs::path parent = fs::path(log_path_).parent_path();
  if (parent.empty()) {
    return true;
  }
  std::error_code ec;
  fs::create_directories(parent, ec);
  if (ec) {
    HLOG(kError, "RestartLog: failed to create {}: {}", parent.string(),
         ec.message());
    return false;
  }
  return true;
}

bool RestartLog::AppendEntry(Op op, const std::string& container_path) {
  if (!EnsureParentDir()) {
    return false;
  }
  std::ofstream ofs(log_path_, std::ios::binary | std::ios::app);
  if (!ofs.is_open()) {
    HLOG(kError, "RestartLog: cannot open {} for append", log_path_);
    return false;
  }
  uint8_t op_byte = static_cast<uint8_t>(op);
  uint32_t len = static_cast<uint32_t>(container_path.size());
  ofs.write(reinterpret_cast<const char*>(&op_byte), sizeof(op_byte));
  ofs.write(reinterpret_cast<const char*>(&len), sizeof(len));
  ofs.write(container_path.data(), static_cast<std::streamsize>(len));
  if (!ofs.good()) {
    HLOG(kError, "RestartLog: write to {} failed", log_path_);
    return false;
  }
  return true;
}

bool RestartLog::AppendAdd(const std::string& container_path) {
  return AppendEntry(Op::kAdd, container_path);
}

bool RestartLog::AppendRm(const std::string& container_path) {
  return AppendEntry(Op::kRm, container_path);
}

std::vector<RestartLog::Entry> RestartLog::ReadAll() const {
  std::vector<Entry> entries;
  std::ifstream ifs(log_path_, std::ios::binary);
  if (!ifs.is_open()) {
    return entries;  // No log yet → empty.
  }
  while (true) {
    uint8_t op_byte = 0;
    if (!ifs.read(reinterpret_cast<char*>(&op_byte), sizeof(op_byte))) {
      break;  // Clean EOF.
    }
    uint32_t len = 0;
    if (!ifs.read(reinterpret_cast<char*>(&len), sizeof(len))) {
      HLOG(kWarning, "RestartLog: truncated record header in {}", log_path_);
      break;
    }
    std::string path(len, '\0');
    if (len > 0 &&
        !ifs.read(path.data(), static_cast<std::streamsize>(len))) {
      HLOG(kWarning, "RestartLog: truncated record body in {}", log_path_);
      break;
    }
    Op op = (op_byte == static_cast<uint8_t>(Op::kRm)) ? Op::kRm : Op::kAdd;
    entries.push_back(Entry{op, std::move(path)});
  }
  return entries;
}

std::vector<std::string> RestartLog::LiveSet() const {
  // Replay in order: track membership and the order each path was first added.
  std::vector<std::string> order;
  std::unordered_set<std::string> live;
  for (const auto& e : ReadAll()) {
    if (e.op == Op::kAdd) {
      if (live.insert(e.path).second) {
        order.push_back(e.path);  // First time seen-live → record order.
      }
    } else {  // kRm
      live.erase(e.path);
    }
  }
  // Emit paths still live, preserving first-add order.
  std::vector<std::string> result;
  result.reserve(live.size());
  for (const auto& p : order) {
    if (live.contains(p)) {
      result.push_back(p);
    }
  }
  return result;
}

bool RestartLog::IsRestartable(const std::string& container_path) const {
  std::vector<std::string> live = LiveSet();
  return std::find(live.begin(), live.end(), container_path) != live.end();
}

// True iff `entries` is already exactly one add per live path, in live-set
// order, with no rm entries (i.e. the log is in compacted/minimal form).
static bool IsMinimal(const std::vector<RestartLog::Entry>& entries,
                      const std::vector<std::string>& live) {
  if (entries.size() != live.size()) {
    return false;
  }
  for (size_t i = 0; i < entries.size(); ++i) {
    if (entries[i].op != RestartLog::Op::kAdd || entries[i].path != live[i]) {
      return false;
    }
  }
  return true;
}

bool RestartLog::Compact() {
  std::vector<Entry> entries = ReadAll();
  std::vector<std::string> live = LiveSet();

  if (IsMinimal(entries, live)) {
    return true;  // Nothing to do.
  }

  if (!EnsureParentDir()) {
    return false;
  }
  // Write the minimal log to a temp file, then atomically rename over it.
  std::string tmp_path = log_path_ + ".tmp";
  {
    std::ofstream ofs(tmp_path, std::ios::binary | std::ios::trunc);
    if (!ofs.is_open()) {
      HLOG(kError, "RestartLog: cannot open {} for compaction", tmp_path);
      return false;
    }
    for (const auto& p : live) {
      uint8_t op_byte = static_cast<uint8_t>(Op::kAdd);
      uint32_t len = static_cast<uint32_t>(p.size());
      ofs.write(reinterpret_cast<const char*>(&op_byte), sizeof(op_byte));
      ofs.write(reinterpret_cast<const char*>(&len), sizeof(len));
      ofs.write(p.data(), static_cast<std::streamsize>(len));
    }
    if (!ofs.good()) {
      HLOG(kError, "RestartLog: failed writing compacted log {}", tmp_path);
      return false;
    }
  }
  std::error_code ec;
  fs::rename(tmp_path, log_path_, ec);
  if (ec) {
    HLOG(kError, "RestartLog: failed to rename {} -> {}: {}", tmp_path,
         log_path_, ec.message());
    fs::remove(tmp_path, ec);
    return false;
  }
  HLOG(kInfo, "RestartLog: compacted {} entries -> {} live containers",
       entries.size(), live.size());
  return true;
}

}  // namespace clio::run
