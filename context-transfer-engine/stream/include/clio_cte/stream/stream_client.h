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

#ifndef CLIO_CTE_STREAM_STREAM_CLIENT_H_
#define CLIO_CTE_STREAM_STREAM_CLIENT_H_

#include <cstring>
#include <string>
#include <vector>

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/stream/stream_tasks.h>

namespace clio::cte::stream {

/**
 * Client of the stream pool.
 *
 * A stream is a CTE tag whose bytes live in kStreamPageSize page blobs.
 * Every stream has a HOME container that owns its logical size and merges
 * its deferred appends. The caller chooses the home and must use the same
 * one for the stream's whole life (clio-fs uses the inode's home; other
 * callers can use DefaultHome).
 */
class Client : public clio::run::ContainerClient {
 public:
  Client() { pool_id_ = kStreamPoolId; }
  explicit Client(const clio::run::PoolId &pool_id) { pool_id_ = pool_id; }

  /** @return the bound pool id. */
  const clio::run::PoolId &GetPoolId() const { return pool_id_; }

  /**
   * Home for callers without their own placement: hash of the tag.
   * @param tag stream tag
   * @param num_containers containers in the stream pool
   * @return container id
   */
  static clio::run::u32 DefaultHome(const clio::cte::core::TagId &tag,
                                    clio::run::u32 num_containers) {
    if (num_containers == 0) return 0;
    clio::run::u64 x = TagKey(tag) + 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return static_cast<clio::run::u32>(x % num_containers);
  }

  /**
   * Pack a tag id into one 64-bit key.
   * @param tag tag id
   * @return (major << 32) | minor
   */
  static clio::run::u64 TagKey(const clio::cte::core::TagId &tag) {
    return (static_cast<clio::run::u64>(tag.major_) << 32) |
           static_cast<clio::run::u64>(tag.minor_);
  }

#if CTP_IS_HOST
  /**
   * Create the stream pool.
   * @param pool_query where to create (use Dynamic)
   * @param pool_name pool name
   * @param custom_pool_id pool id (kStreamPoolId)
   * @param params configuration
   * @return creation future
   */
  clio::run::Future<CreateTask> AsyncCreate(
      const clio::run::PoolQuery &pool_query, const std::string &pool_name,
      const clio::run::PoolId &custom_pool_id, const StreamConfig &params) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<CreateTask>(
        clio::run::CreateTaskId(), clio::run::kAdminPoolId, pool_query,
        StreamConfig::chimod_lib_name, pool_name, custom_pool_id, this,
        params);
    pool_id_ = custom_pool_id;
    return ipc->Send(task);
  }

  /**
   * Read or change a stream's logical size at its home.
   * @param tag stream tag
   * @param home the stream's home container
   * @param op size operation
   * @param value operand
   * @return future; old_size_ / new_size_ are valid after Wait()
   */
  clio::run::Future<SizeOpTask> AsyncSizeOp(const clio::cte::core::TagId &tag,
                                            clio::run::u32 home,
                                            StreamSizeOp op,
                                            clio::run::u64 value = 0) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<SizeOpTask>(clio::run::CreateTaskId(), pool_id_,
                                         clio::run::PoolQuery::DirectId(home),
                                         tag, op, value);
    return ipc->Send(task);
  }

  /**
   * Deferred append of bytes already in a task-visible buffer.
   * @param tag stream tag
   * @param home the stream's home container
   * @param data buffer (AllocateBuffer)
   * @param size bytes
   * @return future; completes once the bytes are durably staged
   */
  clio::run::Future<AppendTask> AsyncAppend(const clio::cte::core::TagId &tag,
                                            clio::run::u32 home,
                                            ctp::ipc::ShmPtr<> data,
                                            clio::run::u64 size) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<AppendTask>(clio::run::CreateTaskId(), pool_id_,
                                         clio::run::PoolQuery::Local(), tag,
                                         home, size, data);
    return ipc->Send(task);
  }

  /**
   * Synchronous deferred append from private memory.
   * @param tag stream tag
   * @param home the stream's home container
   * @param data bytes to append
   * @param size byte count
   * @return 0 on success, else an error code
   */
  clio::run::u32 Append(const clio::cte::core::TagId &tag, clio::run::u32 home,
                        const char *data, clio::run::u64 size) {
    auto *ipc = CLIO_CPU_IPC;
    ctp::ipc::FullPtr<char> buf = ipc->AllocateBuffer(size);
    if (buf.IsNull()) return 1;
    std::memcpy(buf.ptr_, data, size);
    auto f = AsyncAppend(tag, home, buf.shm_.template Cast<void>(), size);
    f.Wait();
    clio::run::u32 rc = f->GetReturnCode();
    ipc->FreeBuffer(buf);
    return rc;
  }

  /**
   * Wait until every append this node accepted for the stream is merged.
   * @param tag stream tag
   * @param home the stream's home container
   * @return future; size_ is the stream size afterwards
   */
  clio::run::Future<FlushTask> AsyncFlush(const clio::cte::core::TagId &tag,
                                          clio::run::u32 home) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<FlushTask>(clio::run::CreateTaskId(), pool_id_,
                                        clio::run::PoolQuery::Local(), tag,
                                        home);
    return ipc->Send(task);
  }

  /**
   * Start this node's periodic append drain (runtime-internal).
   * @param period_us drain period in microseconds
   * @return future of the periodic task (never completes)
   */
  clio::run::Future<SequenceTask> AsyncSequence(double period_us) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<SequenceTask>(clio::run::CreateTaskId(), pool_id_,
                                           clio::run::PoolQuery::Local());
    task->SetPeriod(period_us, clio::run::kMicro);
    task->SetFlags(TASK_PERIODIC);
    return ipc->Send(task);
  }

  /**
   * Send this node's batch of appends to a stream's home, which merges it
   * (runtime-internal).
   * @param tag stream tag
   * @param home the stream's home container
   * @param entries this node's pending appends for the stream, in order
   * @param payload their bytes, concatenated (entries' payload_off_)
   * @return future; completes once the bytes are in place
   */
  clio::run::Future<PlanTask> AsyncPlan(const clio::cte::core::TagId &tag,
                                        clio::run::u32 home,
                                        const std::vector<AppendEntry> &entries,
                                        const std::string &payload) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<PlanTask>(clio::run::CreateTaskId(), pool_id_,
                                       clio::run::PoolQuery::DirectId(home),
                                       tag, entries, payload);
    return ipc->Send(task);
  }
#endif  // CTP_IS_HOST
};

}  // namespace clio::cte::stream

#endif  // CLIO_CTE_STREAM_STREAM_CLIENT_H_
