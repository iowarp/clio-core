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

// ---------------------------------------------------------------------------
// thallium_bench: raw libthallium async-RPC microbenchmark.
//
// Measures how well thallium itself drives async RPCs, independent of the
// lightbeam (lbm) transport wrapper. Uses thallium directly over the
// shared-memory transport (na+sm) so we capture the RPC-engine overhead of an
// intra-node round trip without involving a real NIC.
//
// Both the server (RPC handler) and the client (RPC submitter) live in the
// same process and the same tl::engine: the client looks up the engine's own
// self address and fires RPCs at it over na+sm shared-memory loopback. The
// handler runs on one of the engine's `rpc_thread_count` Argobots execution
// streams; a dedicated progress thread drives Mercury.
//
// Tunables (see --help):
//   --rpc-threads N   engine rpc_thread_count (handler ES pool size)
//   --batch B         async RPCs kept in flight at once before draining
//   --iters M         total RPCs to issue (timed)
//   --payload S       request payload bytes (0 = bare int request)
//   --warmup W        untimed warmup RPCs
//   --protocol P      mercury protocol (default na+sm)
//
// Reports overall IOPS (RPCs/sec) and average latency per RPC.
// ---------------------------------------------------------------------------

#include <thallium.hpp>
#include <thallium/serialization/stl/string.hpp>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace tl = thallium;

namespace {

constexpr const char *kRpcName = "thallium_bench_rpc";

struct Config {
  std::string protocol = "na+sm";
  int rpc_threads = 4;
  int batch = 16;
  long iters = 100000;
  long payload = 0;
  long warmup = 1000;
};

void PrintUsage(const char *prog) {
  std::cout
      << "Usage: " << prog << " [options]\n"
      << "  --protocol P     mercury protocol (default na+sm)\n"
      << "  --rpc-threads N  engine RPC handler threads (default 4)\n"
      << "  --batch B        async RPCs in flight at once (default 16)\n"
      << "  --iters M        total timed RPCs (default 100000)\n"
      << "  --payload S      request payload bytes, 0 = bare int (default 0)\n"
      << "  --warmup W       untimed warmup RPCs (default 1000)\n"
      << "  --help           show this message\n";
}

bool ParseArgs(int argc, char **argv, Config &cfg) {
  auto need = [&](int &i) -> const char * {
    if (i + 1 >= argc) {
      std::cerr << "missing value for " << argv[i] << "\n";
      return nullptr;
    }
    return argv[++i];
  };
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--help" || a == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    } else if (a == "--protocol") {
      const char *v = need(i);
      if (!v) return false;
      cfg.protocol = v;
    } else if (a == "--rpc-threads") {
      const char *v = need(i);
      if (!v) return false;
      cfg.rpc_threads = std::atoi(v);
    } else if (a == "--batch") {
      const char *v = need(i);
      if (!v) return false;
      cfg.batch = std::atoi(v);
    } else if (a == "--iters") {
      const char *v = need(i);
      if (!v) return false;
      cfg.iters = std::atol(v);
    } else if (a == "--payload") {
      const char *v = need(i);
      if (!v) return false;
      cfg.payload = std::atol(v);
    } else if (a == "--warmup") {
      const char *v = need(i);
      if (!v) return false;
      cfg.warmup = std::atol(v);
    } else {
      std::cerr << "unknown argument: " << a << "\n";
      PrintUsage(argv[0]);
      return false;
    }
  }
  if (cfg.rpc_threads < 1) cfg.rpc_threads = 1;
  if (cfg.batch < 1) cfg.batch = 1;
  if (cfg.iters < 1) cfg.iters = 1;
  if (cfg.payload < 0) cfg.payload = 0;
  if (cfg.warmup < 0) cfg.warmup = 0;
  return true;
}

// Issue `count` async RPCs keeping at most `batch` in flight. Returns the
// summed responses (the handler echoes back the request payload size) so the
// optimizer cannot elide the round trips.
uint64_t RunPhase(tl::remote_procedure &rpc, const tl::endpoint &ep,
                  const std::string &payload, long count, int batch) {
  uint64_t checksum = 0;
  long submitted = 0;
  std::vector<tl::async_response> inflight;
  inflight.reserve(static_cast<size_t>(batch));
  while (submitted < count) {
    long this_batch = std::min<long>(batch, count - submitted);
    inflight.clear();
    for (long i = 0; i < this_batch; ++i) {
      inflight.push_back(rpc.on(ep).async(payload));
    }
    for (auto &resp : inflight) {
      int ret = resp.wait();
      checksum += static_cast<uint64_t>(ret);
    }
    submitted += this_batch;
  }
  return checksum;
}

}  // namespace

int main(int argc, char **argv) {
  Config cfg;
  if (!ParseArgs(argc, argv, cfg)) return 1;

  // Server-mode engine with a dedicated progress thread and an RPC handler
  // pool of cfg.rpc_threads Argobots execution streams.
  tl::engine engine(cfg.protocol, THALLIUM_SERVER_MODE,
                    true /*use_progress_thread*/,
                    cfg.rpc_threads /*rpc_thread_count*/);

  // Scope the RPC handle, the self endpoint, and the payload so they are all
  // destroyed *before* engine.finalize() below. The endpoint destructor calls
  // margo_addr_free(), which segfaults if margo has already been torn down --
  // so the endpoint must die while the engine is still alive.
  {
    // Handler: deserialize the payload string and respond with its size. Kept
    // trivial so the measurement reflects RPC-engine overhead, not handler
    // work.
    tl::remote_procedure rpc = engine.define(
        kRpcName, [](const tl::request &req, const std::string &payload) {
          req.respond(static_cast<int>(payload.size()));
        });

    // Client and server share the engine: look up our own self address and
    // fire RPCs at it over the na+sm shared-memory loopback.
    tl::endpoint self = engine.lookup(std::string(engine.self()));

    std::string payload(static_cast<size_t>(cfg.payload), 'x');

    std::cout << "thallium_bench\n"
              << "  protocol     : " << cfg.protocol << "\n"
              << "  self addr    : " << std::string(engine.self()) << "\n"
              << "  rpc threads  : " << cfg.rpc_threads << "\n"
              << "  batch size   : " << cfg.batch << "\n"
              << "  iters        : " << cfg.iters << "\n"
              << "  payload bytes: " << cfg.payload << "\n"
              << "  warmup       : " << cfg.warmup << "\n"
              << std::flush;

    // Warmup (untimed) to amortize connection/registration setup.
    if (cfg.warmup > 0) {
      RunPhase(rpc, self, payload, cfg.warmup, cfg.batch);
    }

    auto t0 = std::chrono::steady_clock::now();
    uint64_t checksum = RunPhase(rpc, self, payload, cfg.iters, cfg.batch);
    auto t1 = std::chrono::steady_clock::now();

    double elapsed_s = std::chrono::duration<double>(t1 - t0).count();
    double iops = elapsed_s > 0 ? cfg.iters / elapsed_s : 0.0;
    // Amortized per-RPC latency: with `batch` requests in flight this is the
    // throughput-derived latency, not the latency of a single isolated call.
    double avg_lat_us = cfg.iters > 0 ? elapsed_s * 1e6 / cfg.iters : 0.0;

    std::cout << "results\n"
              << "  elapsed (s)        : " << elapsed_s << "\n"
              << "  total RPCs         : " << cfg.iters << "\n"
              << "  overall IOPS       : " << iops << "\n"
              << "  avg latency (us)   : " << avg_lat_us << "\n"
              << "  checksum           : " << checksum << "\n"
              << std::flush;
  }  // rpc / self / payload destroyed here, while margo is still alive.

  // Now finalize the server-mode engine explicitly. Without this the engine
  // destructor blocks forever (a server engine parks until it is finalized);
  // calling it here stops the progress thread and lets the engine destructor
  // run as a clean no-op. It must come *after* the endpoint above is gone.
  engine.finalize();
  return 0;
}
