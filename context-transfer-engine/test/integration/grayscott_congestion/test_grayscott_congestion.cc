/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * Gray-Scott L=512 congestion-collapse reproducer (issue #774).
 *
 * On Delta (2 nodes x 128 ranks, clio-core 2.2) gray-scott works at L=256 but
 * the writer stalls at output 1 at L=512: each daemon ingests ~1 GB (one
 * output) and then goes flat, with 91/128 ranks sleep-polling in
 * Future::Wait(). Diagnosis (#774): a live queueing collapse, not a deadlock —
 * the 8x payload multiplies (a) the 64 KB physical-block fragmentation (one
 * co-awaited AllocateFromTarget round-trip per block, core_runtime.cc:3825)
 * and (b) HashBlobToContainer's ~50% cross-node scatter, until the receiving
 * daemon falls behind and the 1 s ZMQ_SNDTIMEO + EAGAIN retry re-queue
 * (zmq_transport.h:359-366) turns back-pressure into retry churn on both
 * daemons' (only 4) worker threads.
 *
 * This binary is one SYMMETRIC producer per node of a 2-node docker cluster
 * (docker-compose.yaml here). Each node mimics gray-scott's per-output I/O:
 *
 *   per output r:
 *     - PUT  GS_BLOBS blobs of GS_BLOB_KB each into this node's tag, with
 *       distinct names => HashBlobToContainer scatters ~50% to the OTHER
 *       node's container (full payload rides the mesh task). All puts are
 *       issued async first, then waited — the same many-outstanding fan-out
 *       the 128 gray-scott ranks per node generate.
 *     - GET  every blob back twice (the hermes_derived engine's
 *       ComputeDerivedVariables reads U and V twice per output).
 *     - log wall time + goodput for the output ("output line").
 *
 * Analog scaling (per node, per output):
 *   L=256-analog: GS_BLOB_KB=512  -> blobs of 0.5 MB,  8 64KB-blocks each
 *   L=512-analog: GS_BLOB_KB=4096 -> blobs of 4 MB,   64 64KB-blocks each
 * i.e. the same 8x bytes / 8x block-transaction ratio as L=256 -> L=512.
 *
 * Success criterion for the REPRO: the small analog completes all outputs on
 * both nodes; the large analog exhibits the cliff — output 1 completes, then
 * throughput collapses (output time explodes / container timeout) while the
 * daemons' worker threads spin. The run script (run_tests.sh) measures both.
 *
 * Env knobs (all optional):
 *   GS_ROLE       "producer" (only role; both nodes run it)
 *   GS_NODE_ID    1 or 2 (blob-name namespace; from compose)
 *   GS_BLOB_KB    payload per blob in KiB           (default 512)
 *   GS_BLOBS      blobs per output                  (default 64)
 *   GS_OUTPUTS    number of outputs                 (default 3)
 *   GS_GET_PASSES read-back passes per output       (default 2)
 *   GS_OUTPUT_TIMEOUT_SEC per-output stall cutoff   (default 300)
 */
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>  // _exit

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>

namespace {

constexpr clio::run::u32 kCtePoolMajor = 512;  // cte_main in clio_config.yaml

long EnvLong(const char *name, long def) {
  const char *e = std::getenv(name);
  if (!e || !*e) return def;
  long v = std::atol(e);
  return v > 0 ? v : def;
}

void Log(int node, const std::string &msg) {
  std::fprintf(stderr, "[gs-congest n%d] %s\n", node, msg.c_str());
  std::fflush(stderr);
}

double SecsSince(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
      .count();
}

}  // namespace

int main() {
  const int node = static_cast<int>(EnvLong("GS_NODE_ID", 1));
  const clio::run::u64 blob_kb = EnvLong("GS_BLOB_KB", 512);
  const int blobs = static_cast<int>(EnvLong("GS_BLOBS", 64));
  const int outputs = static_cast<int>(EnvLong("GS_OUTPUTS", 3));
  const int get_passes = static_cast<int>(EnvLong("GS_GET_PASSES", 2));
  const double out_timeout = static_cast<double>(EnvLong("GS_OUTPUT_TIMEOUT_SEC", 300));
  const clio::run::u64 blob_bytes = blob_kb * 1024ull;
  const double out_mb =
      static_cast<double>(blob_bytes) * blobs / (1024.0 * 1024.0);

  Log(node, "config: blob_kb=" + std::to_string(blob_kb) +
                " blobs=" + std::to_string(blobs) +
                " outputs=" + std::to_string(outputs) +
                " get_passes=" + std::to_string(get_passes) +
                " (put payload/output = " + std::to_string(out_mb) + " MB)");

  // Hard watchdog: Future::Wait() blocks with NO timeout, so a fully-stalled
  // first future would otherwise never reach the in-loop stall checks (this is
  // exactly the parked-forever state the Delta ranks were found in). If the
  // whole run overshoots its budget, report the stall and _exit(42).
  static std::atomic<bool> g_done{false};
  const double budget =
      out_timeout * outputs + 120.0;  // all outputs + startup slack
  std::thread([node, budget] {
    auto t0 = std::chrono::steady_clock::now();
    while (!g_done.load()) {
      std::this_thread::sleep_for(std::chrono::seconds(2));
      if (SecsSince(t0) > budget) {
        Log(node, "WATCHDOG: run exceeded " + std::to_string(budget) +
                      "s with futures still pending — STALL (collapse)");
        std::fflush(nullptr);
        _exit(42);
      }
    }
  }).detach();

  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false)) {
    Log(node, "FAIL: CLIO_INIT(kClient) failed");
    return 2;
  }
  auto *ipc = CLIO_IPC;

  clio::cte::core::Client core;
  core.Init(clio::run::PoolId(kCtePoolMajor, 0));

  // Per-node tag: the cross-node traffic comes from HashBlobToContainer's
  // per-blob-name scatter, not from where the tag lives. Dynamic() is the
  // intended cluster-visible tag creation (as in the #714 cross-node blob
  // test). NOTE (robustness finding, #774): with PoolQuery::Local() here — a
  // node-local tag — cross-node hash-routed Puts neither complete nor error:
  // the client parks forever in Future::Wait() (the very signature the Delta
  // ranks showed). Reproduce that mode with GS_TAG_LOCAL=1.
  const std::string tag_name = "gs_congest_n" + std::to_string(node);
  const bool tag_local = std::getenv("GS_TAG_LOCAL") &&
                         std::atoi(std::getenv("GS_TAG_LOCAL")) != 0;
  auto mk = core.AsyncGetOrCreateTag(tag_name, clio::cte::core::TagId::GetNull(),
                                     tag_local ? clio::run::PoolQuery::Local()
                                               : clio::run::PoolQuery::Dynamic());
  mk.Wait();
  if (mk->GetReturnCode() != 0 || mk->tag_id_.IsNull()) {
    Log(node, "FAIL: GetOrCreateTag rc=" + std::to_string(mk->GetReturnCode()));
    return 3;
  }
  clio::cte::core::TagId tag = mk->tag_id_;

  // One reusable buffer per in-flight blob (client_data_segment holds them).
  std::vector<ctp::ipc::FullPtr<char>> bufs;
  bufs.reserve(blobs);
  for (int i = 0; i < blobs; ++i) {
    ctp::ipc::FullPtr<char> b = ipc->AllocateBuffer(blob_bytes);
    if (b.IsNull()) {
      Log(node, "FAIL: AllocateBuffer(" + std::to_string(blob_bytes) + ") @" +
                    std::to_string(i));
      return 4;
    }
    std::memset(b.ptr_, 'a' + (i % 26), blob_bytes);
    bufs.push_back(b);
  }

  bool stalled = false;
  for (int r = 0; r < outputs && !stalled; ++r) {
    auto t0 = std::chrono::steady_clock::now();

    // ---- PUT phase: async fan-out, then wait (gray-scott per-output write) --
    {
      std::vector<clio::run::Future<clio::cte::core::PutBlobTask>> futs;
      futs.reserve(blobs);
      for (int i = 0; i < blobs; ++i) {
        std::string name = "n" + std::to_string(node) + "_o" +
                           std::to_string(r) + "_b" + std::to_string(i);
        futs.push_back(core.AsyncPutBlob(tag, name, 0, blob_bytes,
                                         bufs[i].shm_.template Cast<void>()));
      }
      const bool trace = std::getenv("GS_TRACE") != nullptr;
      for (int i = 0; i < blobs; ++i) {
        if (trace) Log(node, "waiting put o" + std::to_string(r) + " b" + std::to_string(i));
        futs[i].Wait();
        if (trace) Log(node, "done    put o" + std::to_string(r) + " b" + std::to_string(i) +
                                 " rc=" + std::to_string(futs[i]->GetReturnCode()));
        if (futs[i]->GetReturnCode() != 0) {
          Log(node, "PUT rc=" + std::to_string(futs[i]->GetReturnCode()) +
                        " output=" + std::to_string(r) +
                        " blob=" + std::to_string(i));
        }
        if (SecsSince(t0) > out_timeout) {
          Log(node, "STALLED in PUT wait: output=" + std::to_string(r) +
                        " blob=" + std::to_string(i) + " t=" +
                        std::to_string(SecsSince(t0)) + "s");
          stalled = true;
          break;
        }
      }
    }
    double t_put = SecsSince(t0);

    // ---- GET phase: read every blob back GS_GET_PASSES times (derived vars) -
    double t_get = 0;
    if (!stalled) {
      auto g0 = std::chrono::steady_clock::now();
      for (int pass = 0; pass < get_passes && !stalled; ++pass) {
        std::vector<clio::run::Future<clio::cte::core::GetBlobTask>> futs;
        futs.reserve(blobs);
        for (int i = 0; i < blobs; ++i) {
          std::string name = "n" + std::to_string(node) + "_o" +
                             std::to_string(r) + "_b" + std::to_string(i);
          futs.push_back(core.AsyncGetBlob(tag, name, 0, blob_bytes, 0u,
                                           bufs[i].shm_.template Cast<void>()));
        }
        for (int i = 0; i < blobs; ++i) {
          futs[i].Wait();
          if (futs[i]->GetReturnCode() != 0) {
            Log(node, "GET rc=" + std::to_string(futs[i]->GetReturnCode()) +
                          " output=" + std::to_string(r) + " pass=" +
                          std::to_string(pass) + " blob=" + std::to_string(i));
          }
          if (SecsSince(g0) > out_timeout) {
            Log(node, "STALLED in GET wait: output=" + std::to_string(r) +
                          " pass=" + std::to_string(pass) + " blob=" +
                          std::to_string(i) + " t=" +
                          std::to_string(SecsSince(g0)) + "s");
            stalled = true;
            break;
          }
        }
      }
      t_get = SecsSince(g0);
    }

    // The "output line" run_tests.sh parses: completion + rate per output.
    Log(node, "OUTPUT " + std::to_string(r) + (stalled ? " STALLED" : " done") +
                  ": put=" + std::to_string(t_put) + "s (" +
                  std::to_string(out_mb / (t_put > 0 ? t_put : 1e-9)) +
                  " MB/s), get" + std::to_string(get_passes) + "x=" +
                  std::to_string(t_get) + "s");
  }

  g_done.store(true);
  for (auto &b : bufs) ipc->FreeBuffer(b);

  if (stalled) {
    Log(node, "RESULT: STALL (congestion collapse reproduced at blob_kb=" +
                  std::to_string(blob_kb) + ")");
    return 42;  // distinct exit for "reproduced the stall"
  }
  Log(node, "RESULT: COMPLETED all " + std::to_string(outputs) + " outputs");
  return 0;
}
