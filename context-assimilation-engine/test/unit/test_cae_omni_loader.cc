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
 * Unit tests for clio::cae::core::LoadOmni (core/src/omni_loader.cc).
 *
 * LoadOmni is everything the clio_cae binary does before it needs a live
 * runtime. It was unreachable from any test because it shared a translation
 * unit with main(), and the file read 0% on Codecov. Every throw site and
 * every optional field is exercised here; none of it needs a daemon.
 */

#include "simple_test.h"

#include <clio_cae/core/omni_loader.h>
#include <clio_cae/core/omni_run.h>

#include <chrono>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

using clio::cae::core::AssimilationCtx;
using clio::cae::core::ExecuteRunSteps;
using clio::cae::core::LoadOmni;
using clio::cae::core::RunOutputSrc;
using clio::cae::core::RunScript;
using clio::cae::core::RunSourceArg;
using clio::cae::core::RunStep;

namespace {

// Writes `body` to a uniquely named temp file and removes it when the guard
// dies, so a failing REQUIRE cannot leak the fixture.
class TempYaml {
 public:
  explicit TempYaml(const std::string &body) {
    static int counter = 0;
    path_ = std::filesystem::temp_directory_path() /
            ("cae_omni_test_" + std::to_string(++counter) + ".yaml");
    std::ofstream out(path_);
    out << body;
  }

  ~TempYaml() {
    std::error_code ec;
    std::filesystem::remove(path_, ec);
  }

  TempYaml(const TempYaml &) = delete;
  TempYaml &operator=(const TempYaml &) = delete;

  std::string Path() const { return path_.string(); }

 private:
  std::filesystem::path path_;
};

// LoadOmni reports every failure as a std::runtime_error. Asserting on the
// message keeps these from passing for the wrong reason: a YAML typo in the
// fixture would otherwise throw and satisfy a bare "it threw" check.
bool ThrowsWith(const std::string &yaml_body, const std::string &needle) {
  TempYaml f(yaml_body);
  try {
    (void)LoadOmni(f.Path());
  } catch (const std::exception &e) {
    return std::string(e.what()).find(needle) != std::string::npos;
  }
  return false;
}

}  // namespace

TEST_CASE("LoadOmni - missing file throws", "[cae][omni][error]") {
  bool threw = false;
  try {
    (void)LoadOmni("/nonexistent/path/no_such_omni_file.yaml");
  } catch (const std::exception &) {
    threw = true;
  }
  REQUIRE(threw);
}

TEST_CASE("LoadOmni - malformed YAML throws", "[cae][omni][error]") {
  REQUIRE(ThrowsWith("transfers: [unclosed\n", "Failed to load OMNI file"));
}

TEST_CASE("LoadOmni - missing transfers key throws", "[cae][omni][error]") {
  REQUIRE(ThrowsWith("other_key: 1\n", "missing required 'transfers' key"));
}

TEST_CASE("LoadOmni - non-sequence transfers throws", "[cae][omni][error]") {
  REQUIRE(ThrowsWith("transfers:\n  a: 1\n",
                     "'transfers' must be a sequence/array"));
}

TEST_CASE("LoadOmni - transfer missing src throws", "[cae][omni][error]") {
  REQUIRE(ThrowsWith("transfers:\n  - dst: d\n    format: binary\n",
                     "Transfer 1 missing required 'src' field"));
}

TEST_CASE("LoadOmni - transfer missing dst throws", "[cae][omni][error]") {
  REQUIRE(ThrowsWith("transfers:\n  - src: s\n    format: binary\n",
                     "Transfer 1 missing required 'dst' field"));
}

TEST_CASE("LoadOmni - transfer missing format throws", "[cae][omni][error]") {
  REQUIRE(ThrowsWith("transfers:\n  - src: s\n    dst: d\n",
                     "Transfer 1 missing required 'format' field"));
}

TEST_CASE("LoadOmni - error names the offending transfer by position",
          "[cae][omni][error]") {
  // The first entry is complete, so the 1-based index has to point at the
  // second one rather than at whichever entry failed.
  REQUIRE(ThrowsWith(
      "transfers:\n"
      "  - src: s1\n    dst: d1\n    format: binary\n"
      "  - src: s2\n    dst: d2\n",
      "Transfer 2 missing required 'format' field"));
}

TEST_CASE("LoadOmni - empty transfers list yields no contexts", "[cae][omni]") {
  TempYaml f("transfers: []\n");
  std::vector<AssimilationCtx> ctxs = LoadOmni(f.Path());
  REQUIRE(ctxs.empty());
}

TEST_CASE("LoadOmni - minimal transfer defaults the optional fields",
          "[cae][omni]") {
  TempYaml f(
      "transfers:\n  - src: /in/a.bin\n    dst: /out/a\n    format: binary\n");
  std::vector<AssimilationCtx> ctxs = LoadOmni(f.Path());
  REQUIRE(ctxs.size() == 1);
  REQUIRE(ctxs[0].src == "/in/a.bin");
  REQUIRE(ctxs[0].dst == "/out/a");
  REQUIRE(ctxs[0].format == "binary");
  REQUIRE(ctxs[0].depends_on.empty());
  REQUIRE(ctxs[0].range_off == 0);
  REQUIRE(ctxs[0].range_size == 0);
  REQUIRE(ctxs[0].src_token.empty());
  REQUIRE(ctxs[0].dst_token.empty());
  REQUIRE(ctxs[0].include_patterns.empty());
  REQUIRE(ctxs[0].exclude_patterns.empty());
}

TEST_CASE("LoadOmni - optional scalar fields are read", "[cae][omni]") {
  TempYaml f(
      "transfers:\n"
      "  - src: /in/a.bin\n    dst: /out/a\n    format: binary\n"
      "    depends_on: earlier_step\n"
      "    range_off: 4096\n"
      "    range_size: 8192\n");
  std::vector<AssimilationCtx> ctxs = LoadOmni(f.Path());
  REQUIRE(ctxs.size() == 1);
  REQUIRE(ctxs[0].depends_on == "earlier_step");
  REQUIRE(ctxs[0].range_off == 4096);
  REQUIRE(ctxs[0].range_size == 8192);
}

TEST_CASE("LoadOmni - tokens pass through ExpandPath", "[cae][omni][token]") {
  SIMPLE_TEST_SETENV("CAE_OMNI_TEST_TOKEN", "expanded_secret");
  TempYaml f(
      "transfers:\n"
      "  - src: /in/a.bin\n    dst: /out/a\n    format: binary\n"
      "    src_token: ${CAE_OMNI_TEST_TOKEN}\n"
      "    dst_token: literal_token\n");
  std::vector<AssimilationCtx> ctxs = LoadOmni(f.Path());
  REQUIRE(ctxs.size() == 1);
  // ${VAR} is substituted; a token with no ${} is passed through untouched.
  REQUIRE(ctxs[0].src_token == "expanded_secret");
  REQUIRE(ctxs[0].dst_token == "literal_token");
}

TEST_CASE("LoadOmni - dataset_filter patterns are collected",
          "[cae][omni][filter]") {
  TempYaml f(
      "transfers:\n"
      "  - src: /in/a.h5\n    dst: /out/a\n    format: hdf5\n"
      "    dataset_filter:\n"
      "      include_patterns:\n"
      "        - /temperature\n"
      "        - /pressure\n"
      "      exclude_patterns:\n"
      "        - /debug\n");
  std::vector<AssimilationCtx> ctxs = LoadOmni(f.Path());
  REQUIRE(ctxs.size() == 1);
  REQUIRE(ctxs[0].include_patterns.size() == 2);
  REQUIRE(ctxs[0].include_patterns[0] == "/temperature");
  REQUIRE(ctxs[0].include_patterns[1] == "/pressure");
  REQUIRE(ctxs[0].exclude_patterns.size() == 1);
  REQUIRE(ctxs[0].exclude_patterns[0] == "/debug");
}

TEST_CASE("LoadOmni - dataset_filter with only includes",
          "[cae][omni][filter]") {
  TempYaml f(
      "transfers:\n"
      "  - src: /in/a.h5\n    dst: /out/a\n    format: hdf5\n"
      "    dataset_filter:\n"
      "      include_patterns:\n        - /only\n");
  std::vector<AssimilationCtx> ctxs = LoadOmni(f.Path());
  REQUIRE(ctxs.size() == 1);
  REQUIRE(ctxs[0].include_patterns.size() == 1);
  REQUIRE(ctxs[0].exclude_patterns.empty());
}

TEST_CASE("LoadOmni - non-sequence pattern lists are ignored, not fatal",
          "[cae][omni][filter]") {
  // The IsSequence() guards mean a scalar here is skipped rather than thrown
  // on, and the transfer still loads.
  TempYaml f(
      "transfers:\n"
      "  - src: /in/a.h5\n    dst: /out/a\n    format: hdf5\n"
      "    dataset_filter:\n"
      "      include_patterns: not_a_list\n"
      "      exclude_patterns: also_not_a_list\n");
  std::vector<AssimilationCtx> ctxs = LoadOmni(f.Path());
  REQUIRE(ctxs.size() == 1);
  REQUIRE(ctxs[0].include_patterns.empty());
  REQUIRE(ctxs[0].exclude_patterns.empty());
}

TEST_CASE("LoadOmni - an empty dataset_filter is harmless",
          "[cae][omni][filter]") {
  TempYaml f(
      "transfers:\n"
      "  - src: /in/a.h5\n    dst: /out/a\n    format: hdf5\n"
      "    dataset_filter: {}\n");
  std::vector<AssimilationCtx> ctxs = LoadOmni(f.Path());
  REQUIRE(ctxs.size() == 1);
  REQUIRE(ctxs[0].include_patterns.empty());
}

TEST_CASE("LoadOmni - multiple transfers preserve order", "[cae][omni]") {
  TempYaml f(
      "transfers:\n"
      "  - src: /in/1\n    dst: /out/1\n    format: binary\n"
      "  - src: /in/2\n    dst: /out/2\n    format: hdf5\n"
      "  - src: /in/3\n    dst: /out/3\n    format: binary\n");
  std::vector<AssimilationCtx> ctxs = LoadOmni(f.Path());
  REQUIRE(ctxs.size() == 3);
  REQUIRE(ctxs[0].src == "/in/1");
  REQUIRE(ctxs[1].src == "/in/2");
  REQUIRE(ctxs[1].format == "hdf5");
  REQUIRE(ctxs[2].src == "/in/3");
}

//------------------------------------------------------------------------------
// run: steps
//------------------------------------------------------------------------------

namespace {

namespace fs = std::filesystem;

// A fresh directory removed with everything in it when the guard dies.
class TempDir {
 public:
  TempDir() {
    static int counter = 0;
    path_ = fs::temp_directory_path() /
            ("cae_omni_run_" +
             std::to_string(std::chrono::steady_clock::now()
                                .time_since_epoch()
                                .count()) +
             "_" + std::to_string(++counter));
    fs::create_directories(path_);
  }
  ~TempDir() {
    std::error_code ec;
    fs::remove_all(path_, ec);
  }
  TempDir(const TempDir &) = delete;
  TempDir &operator=(const TempDir &) = delete;

  fs::path Path() const { return path_; }

  // Write `body` to `name` inside the directory and return its path.
  std::string Write(const std::string &name, const std::string &body,
                    bool executable = false) const {
    fs::path p = path_ / name;
    {
      std::ofstream out(p, std::ios::binary);
      out << body;
    }
    if (executable) {
      fs::permissions(p, fs::perms::owner_all, fs::perm_options::add);
    }
    return p.string();
  }

 private:
  fs::path path_;
};

std::string ReadAll(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(in),
                     std::istreambuf_iterator<char>());
}

}  // namespace

TEST_CASE("RunSourceArg - local schemes give the path, others the URL",
          "[cae][omni][run]") {
  REQUIRE(RunSourceArg("file::/data/a.csv") == "/data/a.csv");
  REQUIRE(RunSourceArg("hdf5::/data/a.h5") == "/data/a.h5");
  REQUIRE(RunSourceArg("s3://bucket/key") == "s3://bucket/key");
  REQUIRE(RunSourceArg("https://host/x.zip") == "https://host/x.zip");
  REQUIRE(RunSourceArg("/bare/path") == "/bare/path");
}

TEST_CASE("RunOutputSrc - hdf5 formats get hdf5::, the rest file::",
          "[cae][omni][run]") {
  REQUIRE(RunOutputSrc("binary", "/o/x.parquet") == "file::/o/x.parquet");
  REQUIRE(RunOutputSrc("hdf5", "/o/x.h5") == "hdf5::/o/x.h5");
  REQUIRE(RunOutputSrc("hdf5_vol", "/o/x.h5") == "hdf5::/o/x.h5");
}

#ifndef _WIN32  // POSIX paths and setenv
TEST_CASE("LoadOmni - run and run_out resolve against the OMNI file",
          "[cae][omni][run]") {
  TempDir d;
  ::setenv("CAE_RUN_TEST_OUT", "/scratch/run_test", 1);
  std::string omni = d.Write(
      "a.yaml",
      "transfers:\n"
      "  - src: file::/in/a.csv\n    dst: iowarp::a\n    format: binary\n"
      "  - src: file::/in/b.csv\n    dst: iowarp::b\n    format: binary\n"
      "    run: scripts/../conv.sh\n"
      "    run_out: ${CAE_RUN_TEST_OUT}/b.parquet\n");
  std::vector<RunStep> runs;
  std::vector<AssimilationCtx> ctxs = LoadOmni(omni, &runs);
  REQUIRE(ctxs.size() == 2);
  REQUIRE(runs.size() == 1);
  REQUIRE(runs[0].transfer == 1);
  REQUIRE(runs[0].script == (d.Path() / "conv.sh").string());
  REQUIRE(runs[0].out == "/scratch/run_test/b.parquet");
  // Loading alone changes nothing: the source is rewritten only after the
  // script has run.
  REQUIRE(ctxs[1].src == "file::/in/b.csv");
}
#endif  // _WIN32

TEST_CASE("LoadOmni - run without run_out, and the reverse, throw",
          "[cae][omni][run][error]") {
  REQUIRE(ThrowsWith("transfers:\n  - src: s\n    dst: d\n    format: binary\n"
                     "    run: x.sh\n",
                     "Transfer 1 has 'run' but no 'run_out'"));
  REQUIRE(ThrowsWith("transfers:\n  - src: s\n    dst: d\n    format: binary\n"
                     "    run_out: o\n",
                     "Transfer 1 has 'run_out' but no 'run' script"));
}

TEST_CASE("LoadOmni - a caller that cannot run steps rejects run",
          "[cae][omni][run][error]") {
  // The one-argument form must not quietly ingest the untransformed source.
  REQUIRE(ThrowsWith("transfers:\n  - src: s\n    dst: d\n    format: binary\n"
                     "    run: x.sh\n    run_out: o\n",
                     "this caller does not execute"));
}

TEST_CASE("LoadOmni - unknown keys load with a warning",
          "[cae][omni]") {
  TempYaml f(
      "transfers:\n"
      "  - src: /in/a\n    dst: /out/a\n    format: binary\n"
      "    lambda: not_a_cae_key\n");
  std::vector<AssimilationCtx> ctxs = LoadOmni(f.Path());
  REQUIRE(ctxs.size() == 1);
}

TEST_CASE("ExecuteRunSteps - a step index past the transfers fails",
          "[cae][omni][run][error]") {
  std::vector<AssimilationCtx> ctxs(1);
  RunStep step;
  step.transfer = 3;
  step.script = "/bin/true";
  step.out = "/tmp/never";
  std::string err = ExecuteRunSteps({step}, &ctxs);
  REQUIRE(err.find("no matching transfer") != std::string::npos);
}

TEST_CASE("ExecuteRunSteps - a missing script fails before anything runs",
          "[cae][omni][run][error]") {
  TempDir d;
  std::vector<AssimilationCtx> ctxs(1);
  ctxs[0].src = "file::/in/a";
  RunStep step{0, (d.Path() / "nope.sh").string(), (d.Path() / "o").string()};
  std::string err = ExecuteRunSteps({step}, &ctxs);
  REQUIRE(err.find("transfer 1: run script") != std::string::npos);
  REQUIRE(err.find("does not exist") != std::string::npos);
  REQUIRE(ctxs[0].src == "file::/in/a");
}

#ifndef _WIN32
TEST_CASE("ExecuteRunSteps - the output replaces the source",
          "[cae][omni][run]") {
  TempDir d;
  std::string in = d.Write("in.csv", "a,b\n1,2\n");
  // The script records the arguments it got, then transforms the input.
  std::string script = d.Write(
      "conv.sh",
      "#!/bin/sh\nprintf '%s\\n%s\\n' \"$1\" \"$2\" > \"$2.args\"\n"
      "tr ',' ';' < \"$1\" > \"$2\"\n",
      true);
  std::string out = (d.Path() / "out.txt").string();

  std::vector<AssimilationCtx> ctxs(3);
  ctxs[0].src = "file::" + in;
  ctxs[0].dst = "iowarp::a";
  ctxs[0].format = "binary";
  ctxs[1].src = "file::/other";
  ctxs[1].depends_on = "file::" + in;  // names the source that gets replaced
  ctxs[2].src = "file::/third";
  ctxs[2].depends_on = "iowarp::a";    // names the dst, which is unchanged
  std::string err = ExecuteRunSteps({RunStep{0, script, out}}, &ctxs);
  REQUIRE(err.empty());
  REQUIRE(ReadAll(out) == "a;b\n1;2\n");
  REQUIRE(ReadAll(out + ".args") == in + "\n" + out + "\n");
  REQUIRE(ctxs[0].src == "file::" + out);
  REQUIRE(ctxs[1].depends_on == "file::" + out);
  REQUIRE(ctxs[2].depends_on == "iowarp::a");
}

TEST_CASE("ExecuteRunSteps - hdf5 transfers ingest the output as hdf5::",
          "[cae][omni][run]") {
  TempDir d;
  std::string script =
      d.Write("mk.sh", "#!/bin/sh\nprintf 'h5' > \"$2\"\n", true);
  std::string out = (d.Path() / "x.h5").string();
  std::vector<AssimilationCtx> ctxs(1);
  ctxs[0].src = "s3://bucket/raw.sgy";
  ctxs[0].format = "hdf5";
  REQUIRE(ExecuteRunSteps({RunStep{0, script, out}}, &ctxs).empty());
  REQUIRE(ctxs[0].src == "hdf5::" + out);
}

TEST_CASE("ExecuteRunSteps - a failing script stops the ingest",
          "[cae][omni][run][error]") {
  TempDir d;
  std::string fail = d.Write("fail.sh", "#!/bin/sh\nexit 3\n", true);
  std::string ok = d.Write("ok.sh", "#!/bin/sh\ntouch \"$2\"\n", true);
  std::vector<AssimilationCtx> ctxs(2);
  ctxs[0].src = "file::/a";
  ctxs[1].src = "file::/b";
  std::string out2 = (d.Path() / "o2").string();
  std::string err = ExecuteRunSteps(
      {RunStep{0, fail, (d.Path() / "o1").string()}, RunStep{1, ok, out2}},
      &ctxs);
  REQUIRE(err.find("transfer 1: run script failed with exit code 3") !=
          std::string::npos);
  REQUIRE(ctxs[0].src == "file::/a");
  REQUIRE_FALSE(fs::exists(out2));  // later steps did not run
}

TEST_CASE("ExecuteRunSteps - exit 0 without run_out fails, even if stale",
          "[cae][omni][run][error]") {
  TempDir d;
  std::string script = d.Write("noop.sh", "#!/bin/sh\nexit 0\n", true);
  // A leftover from an earlier run must not count as this run's output.
  std::string out = d.Write("stale.out", "old");
  std::vector<AssimilationCtx> ctxs(1);
  ctxs[0].src = "file::/a";
  std::string err = ExecuteRunSteps({RunStep{0, script, out}}, &ctxs);
  REQUIRE(err.find("did not write run_out") != std::string::npos);
  REQUIRE_FALSE(fs::exists(out));
}

TEST_CASE("ExecuteRunSteps - an old run_out that cannot be removed fails",
          "[cae][omni][run][error]") {
  TempDir d;
  std::string script = d.Write("noop.sh", "#!/bin/sh\nexit 0\n", true);
  // A non-empty directory where the output should go cannot be removed.
  fs::create_directories(d.Path() / "busy" / "child");
  std::vector<AssimilationCtx> ctxs(1);
  ctxs[0].src = "file::/a";
  std::string err = ExecuteRunSteps(
      {RunStep{0, script, (d.Path() / "busy").string()}}, &ctxs);
  REQUIRE(err.find("cannot remove old run_out") != std::string::npos);
}

TEST_CASE("RunScript - start failures and signals are reported",
          "[cae][omni][run][error]") {
  TempDir d;
  std::string err;
  // Not executable.
  std::string plain = d.Write("plain.sh", "#!/bin/sh\nexit 0\n");
  REQUIRE(RunScript(plain, {}, &err) != 0);
  REQUIRE_FALSE(err.empty());

  // Killed by a signal.
  err.clear();
  std::string killed =
      d.Write("killed.sh", "#!/bin/sh\nkill -9 $$\n", true);
  REQUIRE(RunScript(killed, {}, &err) == -1);
  REQUIRE(err.find("signal 9") != std::string::npos);

  // Exit 127: the shell's "command not found".
  err.clear();
  std::string missing = d.Write(
      "missing.sh", "#!/bin/sh\n/nonexistent/cae_run_command\n", true);
  REQUIRE(RunScript(missing, {}, &err) == 127);
  REQUIRE(err.find("exited 127") != std::string::npos);

  // Success passes arguments through and reports 0.
  err.clear();
  std::string echo =
      d.Write("echo.sh", "#!/bin/sh\n[ \"$1\" = 'a b' ] && [ \"$2\" = c ]\n",
              true);
  REQUIRE(RunScript(echo, {"a b", "c"}, &err) == 0);
  REQUIRE(err.empty());
}

TEST_CASE("RunScript - an exit status that cannot be collected is a failure",
          "[cae][omni][run][error]") {
  // A process that ignores SIGCHLD has its children reaped by the kernel, so
  // waitpid fails with ECHILD. The script may have failed; it must not count
  // as a success.
  struct IgnoreSigchld {
    struct sigaction old {};
    IgnoreSigchld() {
      struct sigaction ign {};
      ign.sa_handler = SIG_IGN;
      sigemptyset(&ign.sa_mask);
      sigaction(SIGCHLD, &ign, &old);
    }
    ~IgnoreSigchld() { sigaction(SIGCHLD, &old, nullptr); }
  };
  TempDir d;
  std::string script = d.Write("ok.sh", "#!/bin/sh\nexit 0\n", true);
  std::string err;
  int code = 0;
  {
    IgnoreSigchld guard;
    code = RunScript(script, {}, &err);
  }
  REQUIRE(code == -1);
  REQUIRE(err.find("waitpid failed") != std::string::npos);
}
#endif  // _WIN32

SIMPLE_TEST_MAIN()
