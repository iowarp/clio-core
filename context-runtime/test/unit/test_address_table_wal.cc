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
 * Address-table WAL replay order (issue #1193): when several
 * domain_table.* files remap the same container, the mapping after a
 * restart is the latest one in time -- not whichever file the directory
 * listed last -- and within one file append order wins even if the wall
 * clock stepped back.
 */

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "clio_runtime/clio_runtime.h"
#include "clio_runtime/config_manager.h"
#include "clio_runtime/pool_manager.h"
#include "simple_test.h"

namespace fs = std::filesystem;
using clio::run::ContainerId;
using clio::run::PoolId;
using clio::run::u32;
using clio::run::u64;

namespace {

/** A container id no pool uses, so remapping it disturbs no routing. */
constexpr ContainerId kContainer = 777;

/** One record: [ts][pool][container][old][new], as WriteAddressTableWAL. */
struct Rec {
  u64 ts;
  u32 new_node;
};

/**
 * Write a domain_table file.
 * @param path the file
 * @param recs its records, in append order
 */
void WriteWal(const fs::path &path, const std::vector<Rec> &recs) {
  std::ofstream o(path, std::ios::binary | std::ios::trunc);
  const PoolId pool = clio::run::kAdminPoolId;
  const u32 container = kContainer, old_node = 0;
  for (const Rec &r : recs) {
    o.write(reinterpret_cast<const char *>(&r.ts), sizeof(r.ts));
    o.write(reinterpret_cast<const char *>(&pool), sizeof(pool));
    o.write(reinterpret_cast<const char *>(&container), sizeof(container));
    o.write(reinterpret_cast<const char *>(&old_node), sizeof(old_node));
    o.write(reinterpret_cast<const char *>(&r.new_node), sizeof(r.new_node));
  }
}

/** The WAL directory of this runtime. */
fs::path WalDir() {
  auto *config = CLIO_CONFIG_MANAGER;
  return fs::path(config->GetConfDir()) / "wal";
}

}  // namespace

TEST_CASE("Address-table WAL replays in time order", "[wal][1193]") {
  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true));
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  auto *pm = CLIO_POOL_MANAGER;
  REQUIRE(pm != nullptr);
  const fs::path dir = WalDir();
  fs::create_directories(dir);
  // Names chosen so name order and time order disagree, in both directions.
  const fs::path a = dir / "domain_table.1193.0.3.bin";
  const fs::path b = dir / "domain_table.1193.0.7.bin";

  SECTION("the later stamp wins whichever file is listed last") {
    WriteWal(a, {{200, 5}});  // later, listed first
    WriteWal(b, {{100, 9}});  // earlier, listed last
    pm->ReplayAddressTableWAL();
    REQUIRE(pm->GetContainerNodeId(clio::run::kAdminPoolId, kContainer) == 5);

    WriteWal(a, {{100, 9}});
    WriteWal(b, {{200, 5}});
    pm->ReplayAddressTableWAL();
    REQUIRE(pm->GetContainerNodeId(clio::run::kAdminPoolId, kContainer) == 5);
  }

  SECTION("append order wins inside a file when the clock stepped back") {
    // Two remaps from one node; the clock stepped back between them. Another
    // node's remap falls between the two stamps.
    WriteWal(a, {{300, 11}, {250, 12}});
    WriteWal(b, {{280, 13}});
    pm->ReplayAddressTableWAL();
    REQUIRE(pm->GetContainerNodeId(clio::run::kAdminPoolId, kContainer) == 12);
  }

  fs::remove(a);
  fs::remove(b);
}

SIMPLE_TEST_MAIN()
