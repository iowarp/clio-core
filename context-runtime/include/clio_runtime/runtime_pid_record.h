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

#ifndef CLIO_RUNTIME_RUNTIME_PID_RECORD_H_
#define CLIO_RUNTIME_RUNTIME_PID_RECORD_H_

/**
 * The runtime's pid record: a one-line file in the per-user memfd dir naming
 * the process that owns a given runtime port.
 *
 * Only Linux backs shared-memory segments with /proc/<pid>/fd symlinks, which
 * is how `clio_run stop`/`status` normally learn the local runtime's pid.
 * macOS and BSD have no memfd and no /proc, so their segments are plain files
 * that name no owner (see SystemInfo::CreateNewSharedMemory) — without this
 * record, stop cannot escalate SIGTERM/SIGKILL against a wedged runtime and
 * status cannot report one as UNRESPONSIVE.
 *
 * The record is written by the runtime as its segments come up and removed as
 * they go down, so its lifetime brackets the segments' exactly like the Linux
 * symlink does. Its name follows the chi_<what>_<user>_<port> convention the
 * port-scoped sweeps already recognise, so a runtime that dies without its
 * teardown leaves the record behind as one more stale artifact that
 * `clio_run stop` removes.
 */

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>

#include "clio_ctp/introspect/system_info.h"
#include "clio_ctp/util/config_parse.h"
#include "clio_runtime/types.h"

namespace clio::run {

/**
 * Path of the pid record for a runtime port.
 * @param port the runtime port the record is keyed on
 * @return absolute path of the pid record
 */
inline std::string RuntimePidRecordPath(u32 port) {
  const std::string name =
      ctp::ConfigParse::ExpandPath("chi_runtime_pid_${USER}") + "_" +
      std::to_string(port);
  return ctp::SystemInfo::GetMemfdPath(name);
}

/**
 * Read the pid recorded for this port. Rejects a partially written record
 * (one with no terminating newline) and any non-numeric content: a truncated
 * pid names an unrelated process, and this value drives kill escalation.
 * @param port the runtime port the record is keyed on
 * @return the recorded pid (which may be dead), or -1 if absent/unreadable
 */
inline int ReadRuntimePidRecord(u32 port) {
  std::ifstream in(RuntimePidRecordPath(port));
  std::string line;
  if (!in.is_open() || !std::getline(in, line) || in.eof() || line.empty()) {
    return -1;
  }
  for (char c : line) {
    if (c < '0' || c > '9') {
      return -1;
    }
  }
  const int pid = std::atoi(line.c_str());
  return pid > 0 ? pid : -1;
}

/**
 * Record a pid as the owner of this runtime port. Written as a single line
 * so a concurrent reader either sees the whole pid or rejects the record.
 * @param port the runtime port to key the record on
 * @param pid the owning runtime's pid
 */
inline void WriteRuntimePidRecord(u32 port, int pid) {
  ctp::SystemInfo::EnsureMemfdDir();
  std::ofstream out(RuntimePidRecordPath(port), std::ios::trunc);
  if (out.is_open()) {
    out << pid << "\n";
  }
}

/**
 * Delete this port's pid record. Idempotent: the artifact sweeps remove it
 * too, for runtimes that died without running their teardown.
 * @param port the runtime port the record is keyed on
 */
inline void RemoveRuntimePidRecord(u32 port) {
  std::error_code ec;
  std::filesystem::remove(RuntimePidRecordPath(port), ec);
}

/** Filename prefix shared by every pid record (used by the memfd-dir sweeps
 *  to tell a pid record apart from a segment entry). */
inline constexpr const char *kRuntimePidRecordPrefix = "chi_runtime_pid_";

}  // namespace clio::run

#endif  // CLIO_RUNTIME_RUNTIME_PID_RECORD_H_
