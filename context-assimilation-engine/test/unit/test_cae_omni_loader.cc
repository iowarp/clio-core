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

#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

using clio::cae::core::AssimilationCtx;
using clio::cae::core::LoadOmni;

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

SIMPLE_TEST_MAIN()
