/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 */

#ifndef CLIO_RUNTIME_INCLUDE_CLIO_RUNTIME_RESTART_LOG_H_
#define CLIO_RUNTIME_INCLUDE_CLIO_RUNTIME_RESTART_LOG_H_

#include <cstdint>
#include <string>
#include <vector>

namespace clio::run {

/**
 * RestartLog — append-only binary write-ahead log of the "containers"
 * (compose files) that should be re-composed when `clio_run start` runs.
 *
 * Two operations are recorded, each referencing the absolute path of a
 * compose file:
 *   - add_container <path>   (registers the file for restart)
 *   - rm_container  <path>   (unregisters it)
 *
 * The live set is the set of paths whose most recent operation is `add`.
 * `clio_run start` replays the log to obtain this set and re-composes each
 * file. `Compact()` rewrites the log down to one `add` per live path (no
 * `rm` entries, no duplicates); it is a no-op when the log is already
 * minimal.
 *
 * On-disk format (little-endian, host order — single-host log for now):
 *   repeated records of
 *     u8  op        (0 = add_container, 1 = rm_container)
 *     u32 path_len
 *     u8  path[path_len]
 *
 * The default location is ~/.clio/restart_log.bin.
 */
class RestartLog {
 public:
  enum class Op : uint8_t { kAdd = 0, kRm = 1 };

  struct Entry {
    Op op;
    std::string path;
  };

  /** Construct over an explicit log path (mainly for tests). */
  explicit RestartLog(std::string log_path) : log_path_(std::move(log_path)) {}

  /** Construct over the default ~/.clio/restart_log.bin location. */
  RestartLog() : log_path_(DefaultPath()) {}

  /**
   * Resolve the default log path. Honors the CLIO_RESTART_LOG env var as an
   * explicit override (used to isolate the WAL in tests); otherwise
   * $HOME/.clio/restart_log.bin.
   */
  static std::string DefaultPath();

  const std::string& path() const { return log_path_; }

  /** Append an add_container record. Returns false on I/O error. */
  bool AppendAdd(const std::string& container_path);

  /** Append an rm_container record. Returns false on I/O error. */
  bool AppendRm(const std::string& container_path);

  /** Read every record in write order. Empty if the log does not exist. */
  std::vector<Entry> ReadAll() const;

  /**
   * Compute the live set: every path whose most recent op is `add`, in the
   * order it was first added (later duplicate adds do not reorder it).
   */
  std::vector<std::string> LiveSet() const;

  /** True if `path` is currently in the live set. */
  bool IsRestartable(const std::string& container_path) const;

  /**
   * Rewrite the log to its minimal form: one add_container per live path,
   * in live-set order, with no rm entries. No-op (and no rewrite) if the
   * on-disk log already equals that minimal form. Returns false on I/O error.
   */
  bool Compact();

 private:
  bool AppendEntry(Op op, const std::string& container_path);
  bool EnsureParentDir() const;

  std::string log_path_;
};

}  // namespace clio::run

#endif  // CLIO_RUNTIME_INCLUDE_CLIO_RUNTIME_RESTART_LOG_H_
