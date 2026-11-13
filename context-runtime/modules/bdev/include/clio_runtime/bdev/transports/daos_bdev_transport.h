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

#ifndef CLIO_BDEV_DAOS_TRANSPORT_H_
#define CLIO_BDEV_DAOS_TRANSPORT_H_

#include <clio_runtime/bdev/bdev_alloc_log.h>
#include <clio_runtime/bdev/transports/bdev_transport.h>
#include <clio_runtime/bdev/transports/block_allocator.h>
#include <clio_runtime/bdev/transports/daos_bdev_path.h>

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace clio::run::bdev {

namespace daosio {
class DfsFile;
}  // namespace daosio

/**
 * Native DAOS block device (BdevType::kDaos, bdev_type "daos").
 *
 * The device is ONE file of a DAOS POSIX container, reached through the DAOS
 * client library (libdfs) instead of a dfuse mount: the block allocator hands
 * out offsets inside the file and every block read/write is a dfs_read /
 * dfs_write at that offset. The pool name is the device's location:
 *
 *   daos://<pool>/<container>/<file path>[?chunk_size=..&oclass=..&...]
 *
 * (see DaosBdevPath). A DFS file rather than a bare DAOS array object because
 * the containers these tiers live in are of type POSIX: a file created through
 * DFS is a regular member of the container's namespace (visible to dfuse, `ls`,
 * `du`, `rm`, and removed with its directory), while an array object created
 * outside DFS would be an unreferenced object in a container whose layout DFS
 * owns. Underneath, a DFS file IS a DAOS array with 1-byte cells and the
 * configured chunk size, so the data path is the same.
 *
 * I/O never blocks a runtime worker, and a worker never calls DAOS: the task
 * coroutine hands each request to one of the process's dedicated DAOS I/O
 * threads (daosio::IoThread; io_threads option / CLIO_DAOS_IO_THREADS) and
 * yields until the request's done flag is set. Each I/O thread owns one DAOS
 * event queue, issues the dfs_write / dfs_read on it and reaps it. When every
 * I/O thread is at its bound the coroutine yields until there is room.
 *
 * Capacity is the configured size (capacity_limit); DAOS allocates space as
 * data is written, so nothing is preallocated. A DAOS update is durable when
 * it completes, so Sync only has the allocator log to flush.
 *
 * Not supported: the synchronous DirectRead/DirectWrite fast paths (memory
 * tiers only), striping over several files, the O_DIRECT bounce path, lazy
 * growth units and the `<file>.fail` fault-injection marker of the file
 * transport -- none of them applies to a DAOS file.
 */
class DaosBdevTransport : public BdevTransport {
 public:
  DaosBdevTransport();
  ~DaosBdevTransport() override;

  /**
   * Open (creating if needed) the DAOS file named by `pool_name` and set up
   * the allocator. Blocking, like FsBdevTransport::Init: it runs once, at
   * pool creation. On any failure everything acquired is released and the
   * device is refused.
   * @param params creation parameters (total_size_ is the capacity)
   * @param pool_name the daos:// URI of the device
   * @param runtime the parent bdev runtime (unused)
   * @return true if the device is ready
   */
  bool Init(const CreateParams &params, const std::string &pool_name,
            Runtime *runtime) override;

  /** Stop the log thread and close the file, container, pool and library
   *  reference, in that order (idempotent). */
  void Destroy() override;

  /**
   * Allocate blocks and record them in the allocator log.
   * @param size total bytes wanted
   * @param worker_id requesting worker
   * @param blocks out: the allocated blocks
   * @return false if the device has no room
   */
  bool AllocateBlocks(size_t size, int worker_id,
                      std::vector<Block> &blocks) override;

  /**
   * Record the frees in the allocator log, then return the blocks. The
   * bytes stay in DAOS until they are overwritten.
   * @param worker_id requesting worker
   * @param blocks blocks to free
   */
  void FreeBlocks(int worker_id, const std::vector<Block> &blocks) override;

  /**
   * Write the task's data to its blocks (one asynchronous dfs_write per
   * block, issued by a DAOS I/O thread and awaited without blocking the
   * worker).
   * @param task the write task
   * @return coroutine handle
   */
  clio::run::TaskResume WriteBlocks(
      ctp::ipc::FullPtr<WriteTask> task) override;

  /**
   * Read the task's blocks (one asynchronous dfs_read per block, issued by
   * a DAOS I/O thread and awaited without blocking the worker). Blocks that
   * were allocated but never
   * written read as zeros: the destination is zeroed before the read unless
   * the path says prezero=0, and bytes past the end of the file are zeroed
   * after it.
   * @param task the read task
   * @return coroutine handle
   */
  clio::run::TaskResume ReadBlocks(ctp::ipc::FullPtr<ReadTask> task) override;

  /** @return the device capacity in bytes. */
  clio::run::u64 GetCapacity() const override {
    return allocator_.GetCapacity();
  }
  /** @return bytes the allocator can still hand out. */
  clio::run::u64 GetRemainingSize() const override {
    return allocator_.GetRemainingSize();
  }

  /** Make the allocator log durable and compact it when history dominates. */
  void FlushAllocLog() override;

  /**
   * Make completed writes durable. A DAOS write is durable when its event
   * completes, so only the allocator log needs flushing.
   * @return true (the log flush has no failure channel)
   */
  bool Sync() override;

 private:
  /** Allocator-log flush period of the sync thread, in milliseconds. */
  static constexpr int kAllocLogSyncPeriodMs = 50;
  /** How long a waiting task sleeps between checks of its request's done
   *  flag (and between attempts to queue it when the I/O threads are full),
   *  in microseconds (the value FsBdevTransport uses). */
  static constexpr double kPollYieldUs = 10.0;

  /**
   * Work out the device capacity: the configured size, else the size of an
   * existing file.
   * @param params creation parameters
   * @param capacity out: capacity in bytes
   * @return false when no capacity is configured and the file is empty
   */
  bool ResolveCapacity(const CreateParams &params, clio::run::u64 *capacity);

  /**
   * Open the allocator-state log on its LOCAL path and, when recovering,
   * rebuild the allocator from it. The log is the file transport's
   * AllocatorLog, unchanged; it is used only when a path is given
   * (CreateParams::alloc_log_path_ or the URI's alloc_log option).
   * @param params creation parameters
   * @return false if a requested log cannot be opened
   */
  bool OpenAllocLog(const CreateParams &params);

  /**
   * Append one record per block to the allocator log.
   * @param blocks blocks just allocated or freed
   * @param is_free true for frees
   */
  void LogBlocks(const std::vector<Block> &blocks, bool is_free);

  /** Sync-thread body: FlushAllocLog every kAllocLogSyncPeriodMs. */
  void AllocLogSyncLoop();

  /** Stop and join the sync thread (idempotent). */
  void StopAllocLogSync();

  /**
   * Log a failed block I/O (rate-limited: the first 20, then powers of two).
   * @param op "write" or "read"
   * @param raw_rc the DAOS result code (DER code or errno)
   * @param offset device offset of the block
   * @param size bytes requested
   */
  void LogIoFailure(const char *op, int raw_rc, clio::run::u64 offset,
                    clio::run::u64 size);

  StandardBlockAllocator allocator_;        /**< Offsets inside the file. */
  std::unique_ptr<daosio::DfsFile> file_;   /**< The open DAOS file. */
  DaosBdevPath path_;                       /**< Parsed pool name. */
  std::string name_;                        /**< Pool name, for messages. */
  std::atomic<clio::run::u64> io_failures_{0}; /**< Failed block I/Os. */

  AllocatorLog alloc_log_;       /**< Allocator write-ahead log (local file). */
  bool has_alloc_log_ = false;   /**< alloc_log_ is open. */
  std::thread sync_thread_;      /**< Periodic log flush. */
  std::mutex sync_mu_;           /**< Guards sync_stop_. */
  std::condition_variable sync_cv_; /**< Wakes the sync thread to stop. */
  bool sync_stop_ = false;       /**< Sync thread must exit. */
};

}  // namespace clio::run::bdev

#endif  // CLIO_BDEV_DAOS_TRANSPORT_H_
