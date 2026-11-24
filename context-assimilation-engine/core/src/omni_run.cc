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
 * OMNI `run:` steps. See include/clio_cae/core/omni_run.h.
 */

#include <clio_cae/core/omni_run.h>

#include <clio_ctp/util/logging.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

#ifdef _WIN32
#include <process.h>
#else
#include <spawn.h>
#include <sys/wait.h>

#include <cerrno>

extern char** environ;
#endif

namespace clio::cae::core {

namespace {

bool StartsWith(const std::string& s, const char* prefix) {
  return s.rfind(prefix, 0) == 0;
}

#ifdef _WIN32
// _spawnv joins argv with spaces, so an argument with spaces has to be quoted.
std::string Quote(const std::string& a) {
  if (!a.empty() && a.find_first_of(" \t\"") == std::string::npos) return a;
  std::string q = "\"";
  for (char c : a) {
    if (c == '"') q += '\\';
    q += c;
  }
  return q + "\"";
}
#endif

}  // namespace

std::string RunSourceArg(const std::string& src) {
  for (const char* scheme : {"file::", "hdf5::"}) {
    if (StartsWith(src, scheme)) return src.substr(std::strlen(scheme));
  }
  return src;
}

std::string RunOutputSrc(const std::string& format, const std::string& out) {
  return (StartsWith(format, "hdf5") ? "hdf5::" : "file::") + out;
}

int RunScript(const std::string& script, const std::vector<std::string>& args,
              std::string* err) {
#ifdef _WIN32
  // .bat/.cmd scripts need the command interpreter; cmd /C runs any of them.
  std::vector<std::string> q = {"cmd.exe", "/C", Quote(script)};
  for (const auto& a : args) q.push_back(Quote(a));
  std::vector<const char*> argv;
  for (const auto& a : q) argv.push_back(a.c_str());
  argv.push_back(nullptr);
  intptr_t rc = _spawnvp(_P_WAIT, "cmd.exe", argv.data());
  if (rc == -1) {
    *err = "could not start cmd.exe for '" + script + "'";
    return -1;
  }
  return static_cast<int>(rc);
#else
  std::vector<std::string> owned = {script};
  owned.insert(owned.end(), args.begin(), args.end());
  std::vector<char*> argv;
  for (auto& a : owned) argv.push_back(a.data());
  argv.push_back(nullptr);
  pid_t pid = 0;
  int rc = posix_spawn(&pid, script.c_str(), nullptr, nullptr, argv.data(),
                       environ);
  if (rc != 0) {
    *err = "could not start '" + script + "': " + std::strerror(rc) +
           (rc == EACCES ? " (is it executable? chmod +x)" : "");
    return -1;
  }
  int status = 0;
  while (waitpid(pid, &status, 0) < 0) {
    if (errno != EINTR) {
      *err = std::string("waitpid failed: ") + std::strerror(errno);
      return -1;
    }
  }
  if (WIFSIGNALED(status)) {
    *err = "'" + script + "' was killed by signal " +
           std::to_string(WTERMSIG(status));
    return -1;
  }
  // Some C libraries report a failed exec as exit 127 instead of an error.
  int code = WEXITSTATUS(status);
  if (code == 127) {
    *err = "'" + script + "' exited 127 (command not found, or it could not "
           "be executed)";
  }
  return code;
#endif
}

std::string ExecuteRunSteps(const std::vector<RunStep>& steps,
                            std::vector<AssimilationCtx>* contexts) {
  for (const RunStep& step : steps) {
    const std::string where = "transfer " + std::to_string(step.transfer + 1);
    if (step.transfer >= contexts->size()) {
      return where + ": run step has no matching transfer";
    }
    AssimilationCtx& ctx = (*contexts)[step.transfer];
    std::error_code ec;
    if (!std::filesystem::is_regular_file(step.script, ec)) {
      return where + ": run script '" + step.script + "' does not exist";
    }
    // A stale file from an earlier run must not pass for this run's output.
    std::filesystem::remove(step.out, ec);
    if (ec) {
      return where + ": cannot remove old run_out '" + step.out +
             "': " + ec.message();
    }

    const std::string arg = RunSourceArg(ctx.src);
    HLOG(kInfo, "Running {}: {} {} {}", where, step.script, arg, step.out);
    std::string err;
    int code = RunScript(step.script, {arg, step.out}, &err);
    if (code != 0) {
      return where + ": run script failed" +
             (err.empty() ? " with exit code " + std::to_string(code)
                          : ": " + err);
    }
    if (!std::filesystem::is_regular_file(step.out, ec)) {
      return where + ": run script exited 0 but did not write run_out '" +
             step.out + "'";
    }

    const std::string old_src = ctx.src;
    ctx.src = RunOutputSrc(ctx.format, step.out);
    for (size_t j = step.transfer + 1; j < contexts->size(); ++j) {
      if ((*contexts)[j].depends_on == old_src) {
        (*contexts)[j].depends_on = ctx.src;
      }
    }
    HLOG(kInfo, "  {} now ingests {} ({} bytes)", where, ctx.src,
         std::filesystem::file_size(step.out, ec));
  }
  return "";
}

}  // namespace clio::cae::core
