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
 * clio_cte_shm_wtc_bench: per-op cost of the CTE task path vs the shm
 * write-through cache path, from a separate client process.
 *
 * Start a runtime with cte_config_shm_wtc.yaml (core 512.0 + cache 565.0),
 * then run this. For each size it measures, per path:
 *   put    - AsyncPutBlob + Wait, distinct blobs
 *   get    - AsyncGetBlob + Wait of a blob it just wrote
 *   size   - AsyncGetBlobSize + Wait
 *   burst  - N puts then one SyncTag (end-to-end durable throughput)
 * All times are milliseconds.
 *
 * Usage: clio_cte_shm_wtc_bench [ops_per_size=2000]
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

using clio::cte::core::Client;
using clio::cte::core::TagId;
using Clock = std::chrono::steady_clock;

/** Milliseconds since `t0`. */
double Ms(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

/** Results of one path at one size. */
struct Row {
  double put_ms = 0;
  double get_ms = 0;
  double size_ms = 0;
  double burst_ms = 0;
};

/** Run every measurement against `io` with `n` ops of `bytes`. */
Row Measure(Client &io, const std::string &label, size_t bytes, int n) {
  Row row;
  auto tag_task = io.AsyncGetOrCreateTag("wtc_bench_" + label + "_" +
                                         std::to_string(bytes));
  tag_task.Wait();
  const TagId tag = tag_task->tag_id_;
  std::string buf(bytes, 'x');
  std::string back(bytes, '\0');

  auto t0 = Clock::now();
  for (int i = 0; i < n; ++i) {
    auto p = io.AsyncPutBlob(tag, "b" + std::to_string(i), 0, bytes, buf.data());
    p.Wait();
  }
  row.put_ms = Ms(t0) / n;

  t0 = Clock::now();
  for (int i = 0; i < n; ++i) {
    auto g = io.AsyncGetBlob(tag, "b" + std::to_string(i), 0, bytes, 0,
                             back.data());
    g.Wait();
  }
  row.get_ms = Ms(t0) / n;

  t0 = Clock::now();
  for (int i = 0; i < n; ++i) {
    auto s = io.AsyncGetBlobSize(tag, "b" + std::to_string(i));
    s.Wait();
  }
  row.size_ms = Ms(t0) / n;

  t0 = Clock::now();
  for (int i = 0; i < n; ++i) {
    auto p = io.AsyncPutBlob(tag, "burst" + std::to_string(i), 0, bytes,
                             buf.data());
    p.Wait();
  }
  auto sync = io.AsyncSyncTag(tag, clio::run::PoolQuery::Local());
  sync.Wait();
  row.burst_ms = Ms(t0);
  return row;
}

}  // namespace

int main(int argc, char **argv) {
  const int n = argc > 1 ? std::atoi(argv[1]) : 2000;
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false)) {
    std::fprintf(stderr, "CLIO_INIT failed (is the runtime up?)\n");
    return 2;
  }
  Client task_path(clio::cte::core::kCtePoolId);
  Client cache_path(clio::cte::core::wtc::kDefaultPoolId);
  if (!cache_path.HasWtc()) {
    std::fprintf(stderr, "cache pool 565.0 not mapped; compose it first\n");
    return 2;
  }
  std::printf("%-8s %-6s %12s %12s %12s %14s %12s\n", "size", "path",
              "put_ms", "get_ms", "size_ms", "burst_ms", "burst_MBps");
  for (size_t bytes : {4096ul, 65536ul, 1048576ul}) {
    const int ops = bytes >= 1048576 ? n / 8 : n;
    for (int which = 0; which < 2; ++which) {
      Client &io = which == 0 ? task_path : cache_path;
      const char *name = which == 0 ? "task" : "shm";
      Row r = Measure(io, name, bytes, ops);
      const double mbps = (static_cast<double>(bytes) * ops / 1048576.0) /
                          (r.burst_ms / 1000.0);
      std::printf("%-8zu %-6s %12.4f %12.4f %12.4f %14.2f %12.1f\n", bytes,
                  name, r.put_ms, r.get_ms, r.size_ms, r.burst_ms, mbps);
    }
  }
  return 0;
}
