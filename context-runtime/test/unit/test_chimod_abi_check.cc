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
 * ModuleManager ABI-fingerprint check: a ChiMod whose get_chimod_abi()
 * differs from the runtime's (a stale .so left behind by an incremental
 * rebuild), or that predates the fingerprint, must be skipped rather than
 * instantiated. The four libraries are minimal stand-ins built by
 * test_chimod_abi_fake.cc with different FAKE_ABI_MODE values; their paths
 * come in as compile definitions.
 */

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>

#include "clio_runtime/container.h"
#include "clio_runtime/module_manager.h"
#include "clio_ctp/introspect/system_info.h"
#include "simple_test.h"

TEST_CASE("ChiMod with the runtime's ABI fingerprint loads",
          "[module_manager][abi]") {
  clio::run::ModuleManager mm;
  REQUIRE(mm.LoadChiMod(ABI_FAKE_MATCH_LIB));
  REQUIRE(mm.GetChiMod("abi_fake_match") != nullptr);
}

TEST_CASE("ChiMod with a different ABI fingerprint is skipped",
          "[module_manager][abi]") {
  clio::run::ModuleManager mm;
  REQUIRE_FALSE(mm.LoadChiMod(ABI_FAKE_STALE_LIB));
  REQUIRE(mm.GetChiMod("abi_fake_stale") == nullptr);
}

TEST_CASE("ChiMod without an ABI fingerprint is skipped",
          "[module_manager][abi]") {
  clio::run::ModuleManager mm;
  REQUIRE_FALSE(mm.LoadChiMod(ABI_FAKE_NONE_LIB));
  REQUIRE(mm.GetChiMod("abi_fake_none") == nullptr);
}

TEST_CASE("ChiMod with no get_chimod_abi reachable at all is skipped",
          "[module_manager][abi]") {
  clio::run::ModuleManager mm;
  REQUIRE_FALSE(mm.LoadChiMod(ABI_FAKE_BARE_LIB));
  REQUIRE(mm.GetChiMod("abi_fake_bare") == nullptr);
}

TEST_CASE("GetModulePathFor names the file that defines a symbol",
          "[system_info][abi]") {
  REQUIRE(ctp::SystemInfo::GetModulePathFor(nullptr).empty());
  // A heap address belongs to no loaded module.
  auto heap = std::make_unique<int>(0);
  REQUIRE(ctp::SystemInfo::GetModulePathFor(heap.get()).empty());
  // A function in libclio_ctp resolves to that library's file.
  void *fn = reinterpret_cast<void *>(&ctp::SystemInfo::GetModulePathFor);
  std::string path = ctp::SystemInfo::GetModulePathFor(fn);
  REQUIRE_FALSE(path.empty());
  REQUIRE(ctp::SystemInfo::GetModuleDirectoryFor(fn) ==
          std::filesystem::path(path).parent_path().string());
}

TEST_CASE("CLIO_CHIMOD_ABI_CHECK=0 loads a mismatched ChiMod",
          "[module_manager][abi]") {
  SIMPLE_TEST_SETENV("CLIO_CHIMOD_ABI_CHECK", "0");
  clio::run::ModuleManager mm;
  bool loaded = mm.LoadChiMod(ABI_FAKE_STALE_LIB);
  SIMPLE_TEST_SETENV("CLIO_CHIMOD_ABI_CHECK", "");  // any value but "0" checks
  REQUIRE(loaded);
  REQUIRE(mm.GetChiMod("abi_fake_stale") != nullptr);
}

SIMPLE_TEST_MAIN()
