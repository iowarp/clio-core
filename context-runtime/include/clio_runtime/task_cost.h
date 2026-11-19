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

#ifndef CLIO_RUNTIME_INCLUDE_TASK_COST_H_
#define CLIO_RUNTIME_INCLUDE_TASK_COST_H_

/**
 * @file task_cost.h
 * Shared building blocks for Container::GetTaskStats overrides.
 *
 * The scheduler predicts a task's CPU time as a * (compute_ + 1) and its wall
 * (hold) time as b * (wall_time_ + 1), with a and b learned per method. The
 * features a module returns therefore only need the right SHAPE and order of
 * magnitude: a data task must scale with its payload, a control task must be
 * a deliberate constant rather than a silent zero. The wall model is trained
 * on the time a task OCCUPIES its worker (co_await suspensions excluded), so a
 * constant should seed that hold, and PredictHold caps any single booking at
 * kMaxBookedHoldUs. Every module estimates
 * with the same rates below so that one method's features are comparable to
 * another's before the coefficients have converged.
 *
 * Rates (all per microsecond, so features come out in microseconds):
 *  - kCopyBytesPerCpuUs (10 KB/us = ~10 GB/s): one core copying warm data.
 *    The CPU half of any put/get/read/write is a staging copy.
 *  - kStorageBytesPerWallUs (500 B/us = ~500 MB/s): the house wall-time
 *    convention the CTE core, bdev and filesystem already used. Disk-class
 *    bdevs persist a learned coefficient against THIS scale, so changing it
 *    would mis-scale every restored profile.
 *  - kMemoryBytesPerWallUs (5 KB/us = ~5 GB/s): a single-thread memcpy into
 *    a RAM / pinned / HBM-staged tier. Memory-class bdevs never restore a
 *    persisted profile, so they can use their real speed.
 *  - kNetworkBytesPerWallUs (1 KB/us = ~1 GB/s) plus kNetworkRttUs: one
 *    ZeroMQ/TCP transfer to a peer node (replication pushes, handoffs).
 *  - kCompressBytesPerCpuUs / kDecompressBytesPerCpuUs (300 / 900 B/us):
 *    zstd/lz4-class codecs at their balanced presets, ~30x / ~10x the CPU of a
 *    plain copy per byte.
 */

#include <cstddef>

#include "clio_runtime/task.h"
#include "clio_runtime/types.h"

namespace clio::run::cost {

/** CPU bytes copied per microsecond (~10 GB/s single-core memcpy). */
constexpr float kCopyBytesPerCpuUs = 10000.0f;
/** Wall bytes per microsecond on the storage path (~500 MB/s). */
constexpr float kStorageBytesPerWallUs = 500.0f;
/** Wall bytes per microsecond into a memory-class tier (~5 GB/s). */
constexpr float kMemoryBytesPerWallUs = 5000.0f;
/** Wall bytes per microsecond over the inter-node transport (~1 GB/s). */
constexpr float kNetworkBytesPerWallUs = 1000.0f;
/** One inter-node round trip, microseconds (ZeroMQ over TCP on a LAN). */
constexpr float kNetworkRttUs = 50.0f;
/** CPU bytes per microsecond for compression (~300 MB/s). */
constexpr float kCompressBytesPerCpuUs = 300.0f;
/** CPU bytes per microsecond for decompression (~900 MB/s). */
constexpr float kDecompressBytesPerCpuUs = 900.0f;
/** Fixed CPU charge of any data task: dispatch, lookup, completion. */
constexpr size_t kDataBaseComputeUs = 2;

/**
 * Build a stat for a task whose cost is driven by its payload.
 * @param bytes payload bytes the task moves (becomes io_size_)
 * @param cpu_bytes_per_us CPU throughput; compute_ = bytes / this + base
 * @param wall_bytes_per_us wall throughput; wall_time_ = bytes / this + base
 * @param base_compute_us fixed CPU microseconds added to compute_
 * @param base_wall_us fixed wall microseconds added to wall_time_
 * @return the populated TaskStat
 */
inline TaskStat Data(u64 bytes, float cpu_bytes_per_us = kCopyBytesPerCpuUs,
                     float wall_bytes_per_us = kStorageBytesPerWallUs,
                     size_t base_compute_us = kDataBaseComputeUs,
                     float base_wall_us = 0.0f) {
  TaskStat stat;
  stat.io_size_ = static_cast<size_t>(bytes);
  stat.compute_ =
      static_cast<size_t>(static_cast<float>(bytes) / cpu_bytes_per_us) +
      base_compute_us;
  stat.wall_time_ =
      static_cast<float>(bytes) / wall_bytes_per_us + base_wall_us;
  return stat;
}

/**
 * Build a stat for a task whose cost does not depend on a payload.
 * @param compute_us estimated CPU microseconds (becomes compute_)
 * @param wall_us estimated wall microseconds (becomes wall_time_)
 * @param io_bytes bytes the task moves, if any (becomes io_size_)
 * @return the populated TaskStat
 */
inline TaskStat Fixed(size_t compute_us, float wall_us, u64 io_bytes = 0) {
  TaskStat stat;
  stat.io_size_ = static_cast<size_t>(io_bytes);
  stat.compute_ = compute_us;
  stat.wall_time_ = wall_us;
  return stat;
}

/**
 * Scale a per-item fixed cost by an item count (batched control verbs).
 * @param items number of items the task processes
 * @param base_compute_us fixed CPU microseconds for the task itself
 * @param per_item_compute_us CPU microseconds per item
 * @param base_wall_us fixed wall microseconds for the task itself
 * @param per_item_wall_us wall microseconds per item
 * @return the populated TaskStat (io_size_ = 0)
 */
inline TaskStat PerItem(u64 items, size_t base_compute_us,
                        size_t per_item_compute_us, float base_wall_us,
                        float per_item_wall_us) {
  return Fixed(base_compute_us +
                   static_cast<size_t>(items) * per_item_compute_us,
               base_wall_us + static_cast<float>(items) * per_item_wall_us);
}

/** Reading a counter or a small in-memory struct: ~1 us CPU. */
inline TaskStat Trivial() { return Fixed(1, 2.0f); }
/** One metadata map lookup or mutation: ~10 us CPU. */
inline TaskStat Metadata() { return Fixed(10, 15.0f); }
/** A periodic poll that is idle in the common case. */
inline TaskStat IdlePeriodic() { return Fixed(5, 10.0f); }
/** A regex / index scan over a metadata table (~200 us measured, TagQuery). */
inline TaskStat Scan() { return Fixed(200, 300.0f); }
/** Container Create: config parse, allocator and sub-pool setup. */
inline TaskStat ContainerCreate() { return Fixed(100, 2000.0f); }
/** Container Destroy: tear down state, release sub-pools. */
inline TaskStat ContainerDestroy() { return Fixed(50, 500.0f); }
/**
 * A device durability barrier (fsync / flush of a bdev). The fsync itself is
 * offloaded (BlockingOffload) and the task only polls it, so the seed is the
 * worker hold of that poll loop, not the device's milliseconds.
 */
inline TaskStat DeviceSync() { return Fixed(10, 100.0f); }

}  // namespace clio::run::cost

#endif  // CLIO_RUNTIME_INCLUDE_TASK_COST_H_
