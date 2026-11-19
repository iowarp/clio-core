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
 * Restart / crash checker for the shm write-through cache. One phase per
 * invocation (the harness restarts and kills runtimes between phases and
 * runs every node's phase to completion before the next, so the harness is
 * the barrier):
 *
 *   RESTART_PHASE=write  : put RESTART_BLOBS blobs of version RESTART_VER
 *                          into this node's tag, then SyncTag (fsync).
 *   RESTART_PHASE=churn  : overwrite this node's blobs with versions
 *                          RESTART_VER, RESTART_VER+1, ... and never sync,
 *                          until killed or RESTART_CHURN_S seconds pass.
 *   RESTART_PHASE=verify : read every node's blobs through the cache. Each
 *                          must equal exactly one version's pattern in
 *                          [RESTART_VMIN, RESTART_VMAX] or the optional
 *                          [RESTART_XMIN, RESTART_XMAX] (no torn, stale or
 *                          missing blob). RESTART_ONLY_NODE / _SKIP_NODE
 *                          restrict which nodes' tags have a range.
 *
 * Exit 0 = every check passed; prints one line per bad blob.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

using clio::cte::core::TagId;
using clio::run::u64;

/** Integer env knob with a default. */
long Knob(const char *name, long def) {
  const char *e = std::getenv(name);
  return (e != nullptr && *e) ? std::atol(e) : def;
}

/** The process client, bound to the cache pool. */
clio::cte::core::Client *Io() { return CLIO_CTE_CLIENT; }

/** Deterministic bytes for (key, version). */
std::string Pattern(u64 size, u64 key, u64 version) {
  std::string v(size, '\0');
  u64 x = key * 0x9E3779B97F4A7C15ull + version * 0xBF58476D1CE4E5B9ull + 1;
  for (u64 i = 0; i < size; ++i) {
    x ^= x >> 31;
    x *= 0x94D049BB133111EBull;
    v[i] = static_cast<char>(x >> 56);
  }
  return v;
}

/** Blob size of blob i: a mix of small and large blobs. */
u64 BlobSize(long i) {
  static const u64 kSizes[] = {4096, 65536, 1 << 20, 12288, 262144};
  return kSizes[i % 5];
}

/** Key of blob i of node n (distinct across nodes). */
u64 Key(long n, long i) { return static_cast<u64>(n) * 1000003ull + i; }

/** This node's tag. */
TagId NodeTag(long n) {
  auto t = Io()->AsyncGetOrCreateTag("wtc_restart_n" + std::to_string(n));
  t.Wait();
  if (t->GetReturnCode() != 0) {
    fprintf(stderr, "[TAG-ERR] node=%ld rc=%u\n", n, t->GetReturnCode());
    std::exit(3);
  }
  return t->tag_id_;
}

/** Put version `ver` of every blob of node `n`. @return failed puts. */
long PutAll(long n, long blobs, u64 ver) {
  TagId tag = NodeTag(n);
  long bad = 0;
  for (long i = 0; i < blobs; ++i) {
    std::string v = Pattern(BlobSize(i), Key(n, i), ver);
    const clio::run::u32 rc =
        Io()->PutBlob(tag, "b" + std::to_string(i), 0, v.size(), v.data());
    if (rc != 0) {
      fprintf(stderr, "[PUT-ERR] node=%ld blob=%ld ver=%llu rc=%u\n", n, i,
              (unsigned long long)ver, rc);
      ++bad;
    }
  }
  return bad;
}

/** Phase write: put version RESTART_VER and fsync. */
int PhaseWrite(long node, long blobs) {
  const u64 ver = static_cast<u64>(Knob("RESTART_VER", 1));
  long bad = PutAll(node, blobs, ver);
  auto s = Io()->AsyncSyncTag(NodeTag(node));
  s.Wait();
  printf("WRITE node=%ld ver=%llu blobs=%ld put_err=%ld sync_rc=%u\n", node,
         (unsigned long long)ver, blobs, bad, s->GetReturnCode());
  return (bad == 0 && s->GetReturnCode() == 0) ? 0 : 1;
}

/** Phase churn: unsynced overwrites until killed or the time limit. */
int PhaseChurn(long node, long blobs) {
  u64 ver = static_cast<u64>(Knob("RESTART_VER", 2));
  const long secs = Knob("RESTART_CHURN_S", 60);
  auto t0 = std::chrono::steady_clock::now();
  while (std::chrono::steady_clock::now() - t0 < std::chrono::seconds(secs)) {
    PutAll(node, blobs, ver);
    printf("CHURN node=%ld ver=%llu done\n", node, (unsigned long long)ver);
    fflush(stdout);
    ++ver;
  }
  return 0;
}

/** Phase verify: every blob equals exactly one allowed version. */
int PhaseVerify(long nodes, long blobs) {
  const u64 vmin = static_cast<u64>(Knob("RESTART_VMIN", 1));
  const u64 vmax = static_cast<u64>(Knob("RESTART_VMAX", vmin));
  // Optional second range (e.g. unsynced churn versions after a crash).
  const u64 xmin = static_cast<u64>(Knob("RESTART_XMIN", 0));
  const u64 xmax = static_cast<u64>(Knob("RESTART_XMAX", 0));
  const long only = Knob("RESTART_ONLY_NODE", -1);
  const long skip = Knob("RESTART_SKIP_NODE", -1);
  long ok = 0;
  long bad = 0;
  for (long n = 0; n < nodes; ++n) {
    if ((only >= 0 && n != only) || n == skip) continue;
    TagId tag = NodeTag(n);
    for (long i = 0; i < blobs; ++i) {
      std::string got(BlobSize(i), '\0');
      const clio::run::u32 rc = Io()->GetBlob(
          tag, "b" + std::to_string(i), 0, got.size(), got.data());
      if (rc != 0) {
        printf("BAD node=%ld blob=%ld read_rc=%u\n", n, i, rc);
        ++bad;
        continue;
      }
      u64 match = 0;
      for (u64 v = vmin; v <= vmax && match == 0; ++v) {
        if (got == Pattern(got.size(), Key(n, i), v)) match = v;
      }
      for (u64 v = xmin; xmin != 0 && v <= xmax && match == 0; ++v) {
        if (got == Pattern(got.size(), Key(n, i), v)) match = v;
      }
      if (match == 0) {
        printf("BAD node=%ld blob=%ld no version in [%llu,%llu] matches\n", n,
               i, (unsigned long long)vmin, (unsigned long long)vmax);
        ++bad;
      } else {
        ++ok;
      }
    }
  }
  printf("VERIFY ok=%ld bad=%ld range=[%llu,%llu]\n", ok, bad,
         (unsigned long long)vmin, (unsigned long long)vmax);
  return bad == 0 ? 0 : 1;
}

}  // namespace

int main() {
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false) ||
      !clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    fprintf(stderr, "client init failed\n");
    return 2;
  }
  const long node = Knob("NODE_ID", 1) - 1;
  const long nodes = Knob("CTE_NUM_NODES", 1);
  const long blobs = Knob("RESTART_BLOBS", 100);
  const char *ph = std::getenv("RESTART_PHASE");
  const std::string phase = ph != nullptr ? ph : "verify";
  fprintf(stderr, "wtc_restart node=%ld/%ld phase=%s cache=%s\n", node, nodes,
          phase.c_str(), Io()->HasCache() ? "attached" : "NOT attached");
  if (std::getenv("STRESS_EXPECT_WTC") != nullptr &&
      std::string(std::getenv("STRESS_EXPECT_WTC")) == "1" && !Io()->HasCache()) {
    printf("FAIL cache not attached\n");
    return 1;
  }
  if (phase == "write") return PhaseWrite(node, blobs);
  if (phase == "churn") return PhaseChurn(node, blobs);
  return PhaseVerify(nodes, blobs);
}
