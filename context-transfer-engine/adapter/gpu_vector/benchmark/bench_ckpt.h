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
 * The end-of-run checkpoint every paged benchmark that carries persistent
 * state takes: one vector.Copy of the final state.
 *
 * vector.Copy IS the checkpoint. It publishes the vector's resident state and
 * registers a copy-on-write tag over it; with a persistent tier configured
 * the copy is durable, and nothing else is needed.
 *
 * ONE HAZARD, AND IT IS DISTRIBUTED-ONLY. Copy publishes EVERY resident
 * frame -- there is no dirty bit -- and in a multi-node run this node's cache
 * can still hold a PEER's page at an older generation (a halo plane, a peer's
 * weight band read before the peer updated it). Publishing that would put
 * stale bytes over the peer's current page and into the checkpoint. So with
 * more than one node the cache is dropped first: every paged kernel publishes
 * its own writes as it makes them, so the CTE is already current for this
 * node's pages, and the Copy's writeback then has nothing stale to send.
 * Single-node the frames ARE the truth and the Copy's writeback is kept.
 *
 * SYNC MODE (--ckpt-sync in every benchmark). By default the Copy is lazy
 * copy-on-write: pages materialise when first touched. With `sync` the Copy
 * materialises every page before returning, so the checkpoint's full data
 * movement lands inside the timed CHECKPOINT line.
 *
 * Host-only: include inside the !CTP_IS_DEVICE_PASS guard.
 */
#ifndef CLIO_GV_BENCH_CKPT_H_
#define CLIO_GV_BENCH_CKPT_H_

#include <clio_cte/gpu_vector/gpu_vector.h>

#include <chrono>
#include <cstdio>
#include <memory>
#include <string>

namespace clio_bench_ckpt {

/**
 * Checkpoint `vec`'s final state as tag `name` with vector.Copy.
 *
 * Call once, after the run and every gate that reads the live cache: the
 * multi-node path drops the cache.
 *
 * @param vec   the live vector
 * @param name  checkpoint tag; identical on every node (the vector's tag is
 *              shared, so all nodes register the same copy)
 * @param nodes node count; > 1 drops the cache before copying (see above)
 * @param sync  true: fully synchronous Copy (every page materialised before
 *              return); false: lazy copy-on-write
 * @return the checkpoint handle. Keep it alive until the run ends: dropping
 *         it lets the copy tag go away with it.
 */
template <typename T>
std::unique_ptr<clio::cte::gpu_vector::Vector<T>> FinalCheckpoint(
    clio::cte::gpu_vector::Vector<T> &vec, const std::string &name,
    unsigned nodes, bool sync = false) {
  const auto t0 = std::chrono::steady_clock::now();
  if (nodes > 1) vec.ClearCache(0);
  auto ck = vec.Copy(name, sync);
  const double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count();
  std::printf("  CHECKPOINT: vector.Copy -> %s (%s, %.2f ms)\n", name.c_str(),
              sync ? "sync" : "lazy", ms);
  return ck;
}

}  // namespace clio_bench_ckpt

#endif  // CLIO_GV_BENCH_CKPT_H_
