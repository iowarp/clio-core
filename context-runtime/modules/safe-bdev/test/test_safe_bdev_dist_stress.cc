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
 * Distributed safe_bdev stress driver. Runs as a CLIENT on node 0 of a live
 * runtime cluster (CLIO_WITH_RUNTIME=0) and builds one array whose disks
 * are file bdevs on different nodes: disk k lives on node k, parity on
 * nodes 1 and 2, a spare on node 0. It grows the array across the nodes,
 * then asks the harness to kill a node (a whole node unplugged), checks that
 * every byte still reads and that in-place overwrites land while the node
 * is gone, and rebuilds the lost disk onto the spare.
 *
 * Handshake with the harness: the driver prints "SBD_WAIT <name>" and waits
 * for $CLIO_SBD_SYNC/<name>.go. Env: CLIO_SBD_NODES (>= 4),
 * CLIO_SAFE_STRESS_DIR (node-local, same path on every node), CLIO_SBD_SYNC
 * (shared).
 */

#include <fstream>

#include "safe_bdev_stress_util.h"

using namespace sbstress;
using namespace std::chrono_literals;

namespace {

/**
 * Pause until the harness has done `name` (e.g. killed a node).
 * @param name step name
 */
void WaitFor(const std::string &name) {
  const char *dir = std::getenv("CLIO_SBD_SYNC");
  REQUIRE(dir != nullptr);
  const std::string go = std::string(dir) + "/" + name + ".go";
  std::printf("SBD_WAIT %s\n", name.c_str());
  std::fflush(stdout);
  for (int i = 0; i < 600 && !std::filesystem::exists(go); ++i) {
    std::this_thread::sleep_for(500ms);
  }
  REQUIRE(std::filesystem::exists(go));
}

/**
 * A disk on `node`: a file bdev pool with a container on every node, used
 * on one.
 * @param name pool/file name stem
 * @param id pool id
 * @param node the node it belongs to
 * @param d receives the disk
 * @return true on success
 */
bool NodeDisk(const std::string &name, clio::run::PoolId id,
              clio::run::u32 node, Disk *d) {
  d->path = (StressDir() / ("sbd_" + name + ".bin")).string();
  d->id = id;
  d->node = node;
  clio::run::bdev::Client c(id);
  auto t = c.AsyncCreate(clio::run::PoolQuery::Dynamic(), d->path, id,
                         clio::run::bdev::BdevType::kFile, kMemberSize);
  t.Wait();
  return t->GetReturnCode() == 0;
}

}  // namespace

TEST_CASE("safe_bdev_distributed_node_loss", "[safe_bdev][distributed]") {
  EnsureInit();
  REQUIRE(g_initialized);
  const char *ne = std::getenv("CLIO_SBD_NODES");
  REQUIRE(ne != nullptr);
  REQUIRE(std::atoi(ne) >= 4);
  const clio::run::u32 base = 98000;
  Array a("sbd_dist", base);
  clio::run::u32 seed = 1;
  Disk d[4], p1, p2, spare;
  for (clio::run::u32 k = 0; k < 4; ++k) {
    REQUIRE(NodeDisk("d" + std::to_string(k), clio::run::PoolId(base + 1 + k, 0),
                     k, &d[k]));
  }
  REQUIRE(NodeDisk("p1", clio::run::PoolId(base + 10, 0), 1, &p1));
  REQUIRE(NodeDisk("p2", clio::run::PoolId(base + 11, 0), 2, &p2));
  REQUIRE(NodeDisk("spare", clio::run::PoolId(base + 12, 0), 0, &spare));

  // One disk on node 0, mirrored on node 1; then grown across four nodes
  // with a second parity disk on node 2.
  REQUIRE(a.Create(d[0], /*max_failures=*/2, /*distributed=*/true) == 0);
  WriteMany(a, &seed, 20);
  REQUIRE(a.Add(p1, true) == 0);
  REQUIRE(a.BuildParity() == 0);
  REQUIRE(a.VerifyAll("node 0 + parity on node 1") == 0);
  for (int k = 1; k < 4; ++k) REQUIRE(a.Add(d[k], false) == 0);
  REQUIRE(a.Add(p2, true) == 0);
  WriteMany(a, &seed, 40);
  REQUIRE(a.BuildParity() == 0);
  REQUIRE(a.VerifyAll("4 nodes") == 0);

  // Node 3 dies: its disk is gone with it.
  WaitFor("kill3");
  REQUIRE(a.VerifyAll("node 3 down") == 0);
  REQUIRE(a.OverwriteAll(7000) == 0);
  REQUIRE(a.VerifyAll("overwritten while node 3 down") == 0);

  // Rebuild node 3's disk onto the spare on node 0.
  REQUIRE(a.Recover(d[3], spare) == 0);
  REQUIRE(a.VerifyAll("node 3 disk rebuilt on node 0") == 0);
  WriteMany(a, &seed, 20);
  REQUIRE(a.BuildParity() == 0);
  REQUIRE(a.VerifyAll("final") == 0);
}

SIMPLE_TEST_MAIN()
