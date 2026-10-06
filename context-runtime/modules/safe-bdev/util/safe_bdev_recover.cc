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
 * clio_safe_bdev_recover: replace a failed member of a running safe_bdev
 * array -- the operator's "swap the dead disk" step.
 *
 * Creates the replacement file bdev on THIS node (persistent, so a restart
 * re-creates it) and asks the array to rebuild the failed member onto it
 * (RecoverBdev): data members get their live chunks reconstructed from the
 * survivors + parity, parity members their stripes re-encoded.
 *
 *   clio_safe_bdev_recover --array 7100.0 --failed 7000.0 \
 *       --member /mnt/nvme/new_m0.dat --capacity 2147483648 \
 *       --new-id 7200.0
 *
 * Connects as a client to the runtime named by the usual CLIO_* environment
 * (CLIO_SERVER_CONF, CLIO_IPC_MODE). Exit 0 on success; prints one line.
 */

#include <cstdio>
#include <cstdlib>
#include <string>

#include <clio_runtime/bdev/bdev_client.h>
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/pool_query.h>
#include <clio_runtime/safe_bdev/safe_bdev_client.h>

namespace {

/** Command-line options. */
struct Options {
  std::string array;     ///< safe_bdev pool id ("major.minor")
  std::string failed;    ///< failed member's pool id
  std::string member;    ///< replacement member's backing file
  std::string new_id;    ///< pool id for the replacement member
  unsigned long long capacity = 0;  ///< replacement capacity (bytes)
};

/**
 * Parse argv into options.
 * @param argc argument count
 * @param argv arguments
 * @param o receives the options
 * @return true when every required option is present
 */
bool ParseArgs(int argc, char **argv, Options *o) {
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string k = argv[i];
    const std::string v = argv[i + 1];
    if (k == "--array") o->array = v;
    else if (k == "--failed") o->failed = v;
    else if (k == "--member") o->member = v;
    else if (k == "--new-id") o->new_id = v;
    else if (k == "--capacity") o->capacity = std::strtoull(v.c_str(), nullptr, 10);
    else return false;
  }
  return !o->array.empty() && !o->failed.empty() && !o->member.empty() &&
         !o->new_id.empty() && o->capacity > 0;
}

/**
 * Create the replacement member bdev on this node.
 * @param o options (member path, capacity, pool id)
 * @param id receives the created pool id
 * @return 0 or the create's return code
 */
clio::run::u32 CreateMember(const Options &o, clio::run::PoolId *id) {
  const clio::run::PoolId want = clio::run::PoolId::FromString(o.new_id);
  clio::run::bdev::Client disk(want);
  disk.SetPersistent(true);  // a restart re-creates it, as compose's members
  auto t = disk.AsyncCreate(clio::run::PoolQuery::Local(), o.member, want,
                            clio::run::bdev::BdevType::kFile, o.capacity);
  t.Wait();
  *id = t->new_pool_id_;
  return t->GetReturnCode();
}

}  // namespace

/**
 * Entry point: create the replacement member and rebuild onto it.
 * @param argc argument count
 * @param argv arguments (see the file comment)
 * @return 0 on success, 1 on bad usage, 2 on a failed create, 3 on a failed
 *         rebuild
 */
int main(int argc, char **argv) {
  Options o;
  if (!ParseArgs(argc, argv, &o)) {
    std::fprintf(stderr,
                 "usage: %s --array ID --failed ID --member PATH "
                 "--capacity BYTES --new-id ID\n", argv[0]);
    return 1;
  }
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false)) {
    std::fprintf(stderr, "cannot connect to the runtime\n");
    return 2;
  }
  clio::run::PoolId member_id;
  const clio::run::u32 crc = CreateMember(o, &member_id);
  if (crc != 0) {
    std::printf("{\"ok\": false, \"step\": \"create\", \"rc\": %u}\n", crc);
    return 2;
  }
  auto *ipc = CLIO_IPC;
  clio::run::safe_bdev::Client array(clio::run::PoolId::FromString(o.array));
  auto rec = array.AsyncRecoverBdev(
      clio::run::PoolQuery::Local(), clio::run::PoolId::FromString(o.failed),
      o.member, static_cast<clio::run::u32>(ipc->GetNodeId()), member_id);
  rec.Wait();
  const clio::run::u32 rrc = rec->GetReturnCode();
  std::printf("{\"ok\": %s, \"step\": \"recover\", \"rc\": %u, "
              "\"member_pool\": \"%s\"}\n", rrc == 0 ? "true" : "false", rrc,
              member_id.ToString().c_str());
  return rrc == 0 ? 0 : 3;
}
