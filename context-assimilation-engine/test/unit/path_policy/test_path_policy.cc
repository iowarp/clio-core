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
 * test_path_policy.cc - CAE's local path policy (path_policy.h): which paths
 * file::/hdf5:: sources may read and ExportData may write. No runtime: each
 * case builds a PathPolicy directly, in a scratch directory under the system
 * temp directory.
 */

#include <clio_cae/core/factory/path_policy.h>
#include <clio_ctp/util/logging.h>

#include <exception>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>

#ifdef _WIN32
#include <crtdbg.h>
#include <cstdlib>
#include <process.h>  // _getpid: Windows has no unistd.h
#define getpid _getpid
#else
#include <unistd.h>
#endif

namespace fs = std::filesystem;
using clio::cae::core::PathPolicy;

namespace {

int failures = 0;

void Expect(const std::string &label, bool ok) {
  if (ok) {
    HLOG(kSuccess, "PASS: {}", label);
  } else {
    HLOG(kError, "FAIL: {}", label);
    ++failures;
  }
}

bool Reads(const PathPolicy &p, const fs::path &path) {
  std::string reason;
  return p.Allows(path.string(), false, reason);
}

bool Writes(const PathPolicy &p, const fs::path &path) {
  std::string reason;
  return p.Allows(path.string(), true, reason);
}

/**
 * Create a directory symlink; false when the platform refuses. On Windows an
 * account without SeCreateSymbolicLinkPrivilege (any non-elevated account with
 * Developer Mode off, e.g. the win-10 CDash runner) cannot create symlinks;
 * the checks that need one are then skipped rather than failed.
 */
bool Symlink(const fs::path &target, const fs::path &link) {
  std::error_code ec;
  fs::create_directory_symlink(target, link, ec);
  if (ec) {
    HLOG(kWarning, "SKIP: cannot create symlink {} -> {}: {}", link.string(),
         target.string(), ec.message());
    return false;
  }
  return true;
}

/**
 * Route Debug-CRT assertion and error reports to stderr and make abort()
 * silent. A Debug build reports them through a MODAL dialog; under ctest
 * nobody can dismiss it, so the test would hang until its timeout with no
 * message (seen on the win-10 CDash runner when create_directory_symlink
 * threw out of main). Same as SIMPLE_TEST_DISABLE_CRT_DIALOGS().
 */
void DisableCrtDialogs() {
#ifdef _WIN32
  _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
  _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
  _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
  _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
  _set_error_mode(_OUT_TO_STDERR);
  _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
#endif
}

void RunChecks(const fs::path &base) {
  fs::remove_all(base);
  fs::create_directories(base / "data" / "sub");
  fs::create_directories(base / "data2");
  fs::create_directories(base / "home" / ".ssh");
  std::ofstream(base / "data" / "sub" / "a.bin") << "a";
  std::ofstream(base / "home" / ".ssh" / "id_ed25519") << "key";
  const bool keys_link =
      Symlink(base / "home" / ".ssh", base / "data" / "keys");
  const bool out_link = Symlink(base / "data2", base / "data" / "out");

  // Default policy: no allowed_roots.
  const PathPolicy open_policy({});
  Expect("default: ordinary file is readable",
         Reads(open_policy, base / "data" / "sub" / "a.bin"));
  Expect("default: file under .ssh is refused",
         !Reads(open_policy, base / "home" / ".ssh" / "id_ed25519"));
  if (keys_link) {
    Expect("default: symlink into .ssh is refused",
           !Reads(open_policy, base / "data" / "keys" / "id_ed25519"));
  }
  Expect("default: .. into .ssh is refused",
         !Reads(open_policy, base / "data" / ".." / "home" / ".ssh" / "x"));
  Expect("default: /proc is refused", !Reads(open_policy, "/proc/self/environ"));
  Expect("default: a .netrc file is refused",
         !Reads(open_policy, base / "home" / ".netrc"));
  Expect("default: ordinary output is writable",
         Writes(open_policy, base / "data" / "out.h5"));
  Expect("default: hidden output file is refused",
         !Writes(open_policy, base / "home" / ".bashrc"));
  Expect("default: output under /etc is refused",
         !Writes(open_policy, "/etc/clio_test_output"));
  Expect("default: empty path is refused", !Reads(open_policy, ""));

  // Configured roots.
  const PathPolicy rooted({(base / "data").string()});
  Expect("roots: file under the root is readable",
         Reads(rooted, base / "data" / "sub" / "a.bin"));
  Expect("roots: new output under the root is writable",
         Writes(rooted, base / "data" / "sub" / "new.h5"));
  Expect("roots: sibling with the root as a name prefix is refused",
         !Reads(rooted, base / "data2" / "x"));
  Expect("roots: .. out of the root is refused",
         !Reads(rooted, base / "data" / ".." / "data2" / "x"));
  if (out_link) {
    Expect("roots: symlink out of the root is refused",
           !Writes(rooted, base / "data" / "out" / "x.h5"));
  }
  Expect("roots: .ssh stays refused inside a root",
         !Reads(PathPolicy({(base / "home").string()}),
                base / "home" / ".ssh" / "id_ed25519"));
  Expect("roots: a path outside every root is refused",
         !Reads(rooted, "/etc/hostname"));
}

}  // namespace

int main() {
  DisableCrtDialogs();
  const fs::path base =
      fs::temp_directory_path() /
      ("cae_path_policy_" + std::to_string(static_cast<long>(::getpid())));
  // An exception escaping main would std::terminate -> abort(): on Windows a
  // Debug-CRT dialog instead of a failed test, and the scratch tree left
  // behind. Report it as a failure and still clean up.
  try {
    RunChecks(base);
  } catch (const std::exception &e) {
    HLOG(kError, "FAIL: unexpected exception: {}", e.what());
    ++failures;
  }
  std::error_code ec;
  fs::remove_all(base, ec);

  if (failures) {
    HLOG(kError, "{} path policy check(s) failed", failures);
    return 1;
  }
  HLOG(kSuccess, "All path policy checks passed");
  return 0;
}
