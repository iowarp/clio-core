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

#ifndef WRPCTE_CORE_CLIENT_H_
#define WRPCTE_CORE_CLIENT_H_

#include <tuple>
#include <clio_runtime/clio_runtime.h>
#include <clio_ctp/util/singleton.h>
#include <clio_cte/api.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/core/blob_batch.h>
#include <clio_cte/core/shm_wtc.h>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace clio::cte::core {

#if !CTP_IS_GPU_COMPILER
static_assert(wtc::kPerBlobPutFlags == kCtePutDroppable,
              "the shm cache carries exactly the per-blob put flags");
#endif

class Client : public clio::run::ContainerClient {
 public:
  CTP_CROSS_FUN Client() = default;
  CTP_CROSS_FUN explicit Client(const clio::run::PoolId &pool_id) { Init(pool_id); }

#if CTP_IS_HOST
  /**
   * Asynchronous container creation - returns immediately
   * @param pool_query Pool query for task routing
   * @param pool_name Name of the pool
   * @param custom_pool_id Explicit pool ID
   * @param params Create parameters
   */
  clio::run::Future<CreateTask> AsyncCreate(
      const clio::run::PoolQuery &pool_query, const std::string &pool_name,
      const clio::run::PoolId &custom_pool_id,
      const CreateParams &params = CreateParams()) {
    auto *ipc_manager = CLIO_CPU_IPC;

    // CRITICAL: CreateTask MUST use admin pool for GetOrCreatePool processing
    // Pass 'this' as client pointer for PostWait callback
    auto task = ipc_manager->NewTask<CreateTask>(
        clio::run::CreateTaskId(),
        clio::run::kAdminPoolId,  // Always use admin pool for CreateTask
        pool_query,
        CreateParams::chimod_lib_name,  // ChiMod name from CreateParams
        pool_name,                      // Pool name from parameter
        custom_pool_id,                 // Explicit pool ID from parameter
        this,                           // Client pointer for PostWait
        params);                        // CreateParams with configuration

    // Submit to runtime
    return ipc_manager->Send(task);
  }

  /**
   * GPU-callable AsyncCreate: takes const char* names for GPU kernel use.
   * Routes via the caller's pool_query (PoolQuery::Dynamic()).
   * @param pool_query Pool query for task routing
   * @param pool_name Name of the pool (const char*, GPU-safe)
   * @param custom_pool_id Explicit pool ID
   */
  clio::run::Future<CreateTask> AsyncCreate(
      const clio::run::PoolQuery &pool_query, const char *pool_name,
      const clio::run::PoolId &custom_pool_id) {
    auto *ipc_manager = CLIO_CPU_IPC;
    auto task = ipc_manager->NewTask<CreateTask>(
        clio::run::CreateTaskId(),
        clio::run::kAdminPoolId,
        pool_query,
        CreateParams::chimod_lib_name,
        pool_name,
        custom_pool_id,
        static_cast<clio::run::ContainerClient *>(nullptr));
    return ipc_manager->Send(task);
  }

  /**
   * Monitor container state - asynchronous
   */
  clio::run::Future<MonitorTask> AsyncMonitor(const clio::run::PoolQuery &pool_query,
                                        const std::string &query) {
    auto *ipc_manager = CLIO_CPU_IPC;
    auto task = ipc_manager->NewTask<MonitorTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, query);
    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous target registration - returns immediately
   * @param target_name Name of the target to register
   * @param bdev_type Block device type
   * @param total_size Total size of the target
   * @param target_query Pool query for target routing
   * @param bdev_id Block device ID
   * @param pool_query Pool query for task routing (default: Dynamic)
   * @param attach_existing 1 = bind to an existing pool at bdev_id
   * @param growth_unit file bdev lazy-growth step
   * @param persistence_level the bdev's durability label
   *        (clio::run::bdev::PersistenceLevel as int); -1 = from the core's
   *        storage config, or volatile for an unlisted target. A volatile
   *        bdev is refused by a persistent core (kCteVolatileTargetRc); a
   *        persistent bdev in a volatile core only logs a warning.
   */
  clio::run::Future<RegisterTargetTask> AsyncRegisterTarget(
      const std::string &target_name, clio::run::bdev::BdevType bdev_type,
      clio::run::u64 total_size,
      const clio::run::PoolQuery &target_query = clio::run::PoolQuery::Local(),
      const clio::run::PoolId &bdev_id = clio::run::PoolId::GetNull(),
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic(),
      clio::run::u32 attach_existing = 0,
      clio::run::u64 growth_unit = clio::run::u64(1) << 30,
      clio::run::i32 persistence_level = -1) {
    auto *ipc_manager = CLIO_CPU_IPC;

    auto task = ipc_manager->NewTask<RegisterTargetTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, target_name,
        bdev_type, total_size, target_query, bdev_id, attach_existing,
        growth_unit, persistence_level);

    return ipc_manager->Send(task);
  }

  /**
   * Switch this core between volatile and persistent at runtime.
   * To persistent it is refused (kCteVolatileTargetRc) while a volatile bdev
   * is registered; it then opens the WAL (if metadata_log_path is set) and
   * writes a full snapshot. To volatile the WAL stops. Lasts until the core
   * restarts, which goes back to the config.
   * @param durability kCteDurabilityVolatile or kCteDurabilityPersistent
   * @param pool_query containers to switch (default: this node's)
   * @return future for the SetDurabilityTask (previous_ = the old class)
   */
  clio::run::Future<SetDurabilityTask> AsyncSetDurability(
      clio::run::u32 durability,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Local()) {
    auto *ipc_manager = CLIO_CPU_IPC;
    auto task = ipc_manager->NewTask<SetDurabilityTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, durability);
    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous target unregistration - returns immediately
   * @param target_name Name of the target to unregister
   * @param pool_query Pool query for task routing (default: Dynamic)
   */
  clio::run::Future<UnregisterTargetTask> AsyncUnregisterTarget(
      const std::string &target_name,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic()) {
    auto *ipc_manager = CLIO_CPU_IPC;

    auto task = ipc_manager->NewTask<UnregisterTargetTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, target_name);

    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous target listing - returns immediately
   * @param pool_query Pool query for task routing (default: Dynamic)
   */
  clio::run::Future<ListTargetsTask> AsyncListTargets(
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic()) {
    auto *ipc_manager = CLIO_CPU_IPC;

    auto task = ipc_manager->NewTask<ListTargetsTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query);

    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous target stats update - returns immediately
   * @param pool_query Pool query for task routing (default: Dynamic)
   * @param period_ms Period for periodic execution in milliseconds (0 = one-shot)
   */
  clio::run::Future<StatTargetsTask> AsyncStatTargets(
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic(),
      clio::run::u32 period_ms = 0) {
    auto *ipc_manager = CLIO_CPU_IPC;

    auto task = ipc_manager->NewTask<StatTargetsTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query);

    // Set task as periodic if period is specified
    if (period_ms > 0) {
      task->SetPeriod(static_cast<double>(period_ms), clio::run::kMilli);
      task->SetFlags(TASK_PERIODIC);
    }

    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous get target info - returns target score, capacity, and stats
   * @param target_name Name of the target
   * @param pool_query Pool query for task routing (default: Dynamic)
   */
  clio::run::Future<GetTargetInfoTask> AsyncGetTargetInfo(
      const std::string &target_name,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic()) {
    auto *ipc_manager = CLIO_CPU_IPC;

    auto task = ipc_manager->NewTask<GetTargetInfoTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, target_name);

    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous get or create tag - returns immediately
   * @param tag_name Name of the tag
   * @param tag_id Optional tag ID
   * @param pool_query Pool query for task routing (default: Dynamic)
   */
  clio::run::Future<GetOrCreateTagTask<CreateParams>> AsyncGetOrCreateTag(
      const std::string &tag_name,
      const TagId &tag_id = TagId::GetNull(),
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic()) {
    auto *ipc_manager = CLIO_CPU_IPC;

    auto task = ipc_manager->NewTask<GetOrCreateTagTask<CreateParams>>(
        clio::run::CreateTaskId(), pool_id_, pool_query, tag_name,
        tag_id);

    // Inline-eligible (CLIO_ENABLE_INLINE_RUN=1): nested metadata
    // lookups from co-located chimods (clio-fs Open does three in a
    // row) each paid a queue+schedule+wake hop; inline they run on
    // the calling fiber. Falls back to Send everywhere else.
    return CLIO_RUN_INLINE(task);
  }

  /**
   * GetOrCreateTag WITH a fault handler (checkpointing / lazy copy): blobs
   * this tag does not hold are resolved by `fault_pool_id` (e.g. the
   * checkpoint chimod), which receives `fault_params` -- for checkpoint,
   * the SOURCE tag name -- in Context::fault_params_ on every fault.
   */
  clio::run::Future<GetOrCreateTagTask<CreateParams>> AsyncGetOrCreateTag(
      const std::string &tag_name, const clio::run::PoolId &fault_pool_id,
      const std::string &fault_pool_name, const std::string &fault_params,
      const TagId &tag_id = TagId::GetNull(),
      const clio::run::PoolQuery &pool_query =
          clio::run::PoolQuery::Dynamic()) {
    auto *ipc_manager = CLIO_CPU_IPC;
    auto task = ipc_manager->NewTask<GetOrCreateTagTask<CreateParams>>(
        clio::run::CreateTaskId(), pool_id_, pool_query, tag_name, tag_id);
    task->fault_pool_id_ = fault_pool_id;
    task->fault_pool_name_ = fault_pool_name.c_str();
    task->fault_params_ = fault_params.c_str();
    return CLIO_RUN_INLINE(task);
  }

  /**
   * Asynchronous put blob with optional compression context - returns immediately
   * @param tag_id Tag ID
   * @param blob_name Name of the blob
   * @param offset Offset within blob
   * @param size Size of data
   * @param blob_data Shared memory pointer to data
   * @param score Blob score for placement: -1.0=unknown (auto), 0.0-1.0=explicit tier
   * @param context Compression context
   * @param flags Operation flags
   * @param pool_query Pool query for task routing (default: Dynamic)
   */
  clio::run::Future<PutBlobTask> AsyncPutBlob(
      const TagId &tag_id,
      const char *blob_name,
      clio::run::u64 offset, clio::run::u64 size,
      ctp::ipc::ShmPtr<> blob_data, float score = -1.0f,
      const Context &context = Context(),
      clio::run::u32 flags = 0,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic()) {
    auto *ipc_manager = CLIO_CPU_IPC;
    auto task = ipc_manager->NewTask<PutBlobTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, tag_id,
        blob_name, offset, size, blob_data, score, context, flags);

    // Stamp submit time so the receiver can compute end-to-end
    // submit→recv latency. steady_clock is monotonic on each node;
    // cross-node comparisons assume NTP-synced wall clocks (ares is
    // ~ms-synced via the cluster's chrony). Set after NewTask so it
    // overwrites the ctor's 0.
    task.get()->submit_ts_ns_ =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count();

    return ipc_manager->Send(task);
  }

  /** std::string overload */
  clio::run::Future<PutBlobTask> AsyncPutBlob(
      const TagId &tag_id,
      const std::string &blob_name,
      clio::run::u64 offset, clio::run::u64 size,
      ctp::ipc::ShmPtr<> blob_data, float score = -1.0f,
      const Context &context = Context(),
      clio::run::u32 flags = 0,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic()) {
    return AsyncPutBlob(tag_id, blob_name.c_str(), offset, size,
                        blob_data, score, context, flags, pool_query);
  }

  /**
   * Vectored put (issue #820): write N regions of one blob in a SINGLE task.
   *
   * The runtime acquires the blob's write token once, sizes the blob to cover
   * the union of the regions, and applies each region in list order — so N
   * disjoint writes to one blob cost one token acquire and one metadata
   * mutation instead of N of each, and two regions covering the same bytes
   * resolve last-writer-wins (which N racing single-region tasks do not).
   *
   * Each segment keeps its own buffer, so callers never have to gather their
   * payloads into one contiguous allocation. Node-local: the segment buffers
   * are shared-memory pointers, so submit this to a local pool.
   */
  clio::run::Future<PutBlobTask> AsyncPutBlobVectored(
      const TagId &tag_id,
      const char *blob_name,
      const std::vector<BlobSegment> &segments, float score = -1.0f,
      const Context &context = Context(),
      clio::run::u32 flags = 0,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic()) {
    auto *ipc_manager = CLIO_CPU_IPC;
    // offset/size/data stay zero-valued: segments_ is authoritative when set,
    // and the runtime derives the union range from it.
    auto task = ipc_manager->NewTask<PutBlobTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, tag_id,
        blob_name, static_cast<clio::run::u64>(0), static_cast<clio::run::u64>(0),
        ctp::ipc::ShmPtr<>::GetNull(), score, context, flags);
    auto *t = task.get();
    for (const auto &seg : segments) {
      t->segments_.push_back(BlobSegment(seg.blob_off_, seg.size_, seg.data_));
    }
    t->submit_ts_ns_ =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count();
    return ipc_manager->Send(task);
  }

  /** std::string overload */
  clio::run::Future<PutBlobTask> AsyncPutBlobVectored(
      const TagId &tag_id,
      const std::string &blob_name,
      const std::vector<BlobSegment> &segments, float score = -1.0f,
      const Context &context = Context(),
      clio::run::u32 flags = 0,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic()) {
    return AsyncPutBlobVectored(tag_id, blob_name.c_str(), segments, score,
                                context, flags, pool_query);
  }

  /**
   * A vectored segment whose buffer is caller-owned PRIVATE memory (the
   * private-path analog of BlobSegment). data_ is the source for a put and
   * the destination for a get.
   */
  struct PrivBlobSegment {
    clio::run::u64 blob_off_;
    clio::run::u64 size_;
    char *data_;
    PrivBlobSegment(clio::run::u64 off, clio::run::u64 size, char *data)
        : blob_off_(off), size_(size), data_(data) {}
  };

  /**
   * PRIVATE-MEMORY vectored put: write N regions of one blob in a single task,
   * each region sourced from a caller-owned private buffer. Completes the
   * shared/private matrix for the vectored APIs (scalar Put/Get already have
   * both, issues #823/#830).
   *
   * Runtime (co-located) mode: each segment's pointer is wrapped as a
   * null-allocator ShmPtr and the bdev writes read straight from the caller's
   * buffers — no staging, no copy. Client mode: all segments are staged
   * through ONE SHM buffer (a single allocation + one memcpy per segment);
   * ~PutBlobTask frees it via TASK_DATA_OWNER. The caller's buffers are free
   * to reuse as soon as this returns in client mode, and after Wait() in
   * runtime mode (same contract as the scalar private put).
   */
  clio::run::Future<PutBlobTask> AsyncPutBlobVectored(
      const TagId &tag_id, const std::string &blob_name,
      const std::vector<PrivBlobSegment> &segments, float score = -1.0f,
      const Context &context = Context(),
      clio::run::u32 flags = 0,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic()) {
    auto *ipc_manager = CLIO_CPU_IPC;

    if (CLIO_RUNTIME_MANAGER->IsRuntime()) {
      auto task = ipc_manager->NewTask<PutBlobTask>(
          clio::run::CreateTaskId(), pool_id_, pool_query, tag_id,
          blob_name.c_str(), static_cast<clio::run::u64>(0),
          static_cast<clio::run::u64>(0), ctp::ipc::ShmPtr<>::GetNull(), score,
          context, flags);
      auto *t = task.get();
      for (const auto &seg : segments) {
        t->segments_.push_back(BlobSegment(
            seg.blob_off_, seg.size_, ctp::ipc::ShmPtr<>::FromRaw(seg.data_)));
      }
      return ipc_manager->Send(task);
    }

    // Client mode: one staging allocation for all segments.
    clio::run::u64 total = 0;
    for (const auto &seg : segments) total += seg.size_;
    ctp::ipc::FullPtr<char> staging = ipc_manager->AllocateBuffer(total);
    if (staging.IsNull()) {
      return clio::run::Future<PutBlobTask>();
    }
    auto task = ipc_manager->NewTask<PutBlobTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, tag_id,
        blob_name.c_str(), static_cast<clio::run::u64>(0),
        static_cast<clio::run::u64>(0), ctp::ipc::ShmPtr<>(staging.shm_),
        score, context, flags);
    auto *t = task.get();
    clio::run::u64 off = 0;
    for (const auto &seg : segments) {
      std::memcpy(staging.ptr_ + off, seg.data_, seg.size_);
      t->segments_.push_back(BlobSegment(
          seg.blob_off_, seg.size_, ctp::ipc::ShmPtr<>(staging.shm_) + off));
      off += seg.size_;
    }
    // blob_data_ carries the staging buffer purely for ownership: the PutBlob
    // handler ignores blob_data_ whenever segments_ is non-empty, and setting
    // TASK_DATA_OWNER after Send keeps the flag off the daemon's copy (same
    // ordering rationale as the scalar private put).
    auto fut = ipc_manager->Send(task);
    t->SetFlags(TASK_DATA_OWNER);
    return fut;
  }

  /**
   * Private-memory AsyncPutBlob (issue #830): write a blob region straight from
   * a caller-owned PRIVATE buffer (const char*), instead of making the caller
   * hand-manage a shared-memory buffer (allocate → copy in → pass ShmPtr →
   * free) as the ShmPtr overload above requires. This is the write-side analog
   * of the private-memory AsyncGetBlob (issue #823).
   *
   * Two paths, fastest first:
   *  - Runtime (co-located) mode: the daemon shares this address space, so the
   *    private pointer is wrapped as a null-allocator ShmPtr — IpcManager::
   *    ToFullPtr resolves such a pointer's offset AS the absolute address — and
   *    the bdev write reads DIRECTLY from the caller's buffer. No staging
   *    buffer, no copy, and NOT TASK_DATA_OWNER (the buffer is the caller's).
   *  - Client mode: the daemon cannot reach private memory, so the write is
   *    staged through a freshly allocated SHM buffer — the private bytes are
   *    copied in ONCE, then the task carries that buffer. The task is marked
   *    TASK_DATA_OWNER so ~PutBlobTask frees the staging buffer once the write
   *    completes (i.e. after the caller's Wait() returns).
   *
   * Task path only: it never consults the shm cache (a Future needs a task).
   * Use PutBlob() for the cache-first synchronous write.
   *
   * @return A Future over the put; readable after Wait() (GetReturnCode()==0 on
   *         success). An empty Future (Wait() returns immediately, get() is
   *         null) is returned for a degenerate request (size==0 or null
   *         source) or when SHM staging could not be allocated in client mode.
   */
  clio::run::Future<PutBlobTask> AsyncPutBlob(
      const TagId &tag_id, const std::string &blob_name,
      clio::run::u64 offset, clio::run::u64 size, const char *priv_data,
      float score = -1.0f, const Context &context = Context(),
      clio::run::u32 flags = 0,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic()) {
    auto *ipc_manager = CLIO_CPU_IPC;

    // Degenerate requests (no payload / no source) are rejected CLIENT-SIDE
    // with an empty Future — same contract as the staging-allocation failure
    // below: Wait() succeeds immediately, get() is null, and nothing reaches
    // the runtime. The earlier version routed these through the ShmPtr
    // overload with a null buffer, which the co-located handler rejects
    // cleanly — but in CLIENT mode Send's serialization bulk-exposes `size`
    // bytes from the null pointer and segfaults (cte_putblob_priv_separate,
    // "rejects degenerate requests").
    if (size == 0 || priv_data == nullptr) {
      return clio::run::Future<PutBlobTask>();
    }

    if (CLIO_RUNTIME_MANAGER->IsRuntime()) {
      // Co-located daemon: read directly from the private buffer. The null
      // AllocatorId marks the offset as an absolute process address, so the
      // bdev write pulls the source bytes straight out of `priv_data`.
      ctp::ipc::ShmPtr<> raw =
          ctp::ipc::ShmPtr<>::FromRaw(const_cast<char *>(priv_data));
      auto task = ipc_manager->NewTask<PutBlobTask>(
          clio::run::CreateTaskId(), pool_id_, pool_query, tag_id, blob_name,
          offset, size, raw, score, context, flags);
      task.get()->submit_ts_ns_ =
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now().time_since_epoch())
              .count();
      return ipc_manager->Send(task);
    }

    // Client: stage through an SHM buffer, copying the private bytes in ONCE.
    ctp::ipc::FullPtr<char> staging = ipc_manager->AllocateBuffer(size);
    if (staging.IsNull()) {
      return clio::run::Future<PutBlobTask>();
    }
    std::memcpy(staging.ptr_, priv_data, size);
    auto task = ipc_manager->NewTask<PutBlobTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, tag_id, blob_name,
        offset, size, ctp::ipc::ShmPtr<>(staging.shm_), score, context, flags);
    auto *t = task.get();
    t->submit_ts_ns_ =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count();
    auto fut = ipc_manager->Send(task);
    // Mark ownership only AFTER Send has serialized the task: Task::SerializeIn
    // ships task_flags_, so setting TASK_DATA_OWNER earlier would hand the flag
    // to the daemon, whose task shares the very same physical SHM buffer on the
    // local path — and it would free the client's buffer out from under us.
    // Set client-side, this instance's ~PutBlobTask frees the staging buffer
    // when the Future (task) is destroyed, after the write has completed.
    t->SetFlags(TASK_DATA_OWNER);
    return fut;
  }

  /**
   * AsyncMultiPutVectored (issue #862): ship a batch of whole-value puts to
   * DIFFERENT blobs as ONE task. All payloads must already live in a single
   * staged SHM buffer (`data`); `descs` names each put. Executes inline when
   * CLIO_RUN_INLINE is eligible, otherwise Sends one task — either way the
   * per-put scheduling/completion cost is amortized across the batch.
   */
  clio::run::Future<MultiPutBlobTask> AsyncMultiPutVectored(
      ctp::ipc::ShmPtr<> data, clio::run::u64 data_len,
      const std::vector<MultiPutDesc> &descs,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Local(),
      const Context &context = Context()) {
    auto *ipc_manager = CLIO_CPU_IPC;
    std::string packed = EncodeMultiPutDescs(descs);
    TagId rt = descs.empty() ? TagId::GetNull() : descs.front().tag_id_;
    const std::string rb = descs.empty() ? std::string() : descs.front().blob_name_;
    auto task = ipc_manager->NewTask<MultiPutBlobTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, rt, rb, data,
        data_len, packed, context);
    if (CLIO_RUNTIME_MANAGER->IsRuntime()) {
      // One shared object; flag must be set before Send (see PutBlob staging).
      task.get()->SetFlags(TASK_DATA_OWNER);
      return CLIO_RUN_INLINE(task);
    }
    auto fut = CLIO_RUN_INLINE(task);
    task.get()->SetFlags(TASK_DATA_OWNER);
    return fut;
  }

  // ===========================================================================
  // The two paths.
  //
  // SHM CACHE PATH: when this client is connected to the shared-memory cache
  // module (core_shm_cache_writethru) of its stack, PutBlob / GetBlob /
  // GetBlobSize run against the cache's segment in this process. A put the
  // segment accepts creates no task, allocates nothing in the runtime and
  // waits for nothing; the cache's flusher carries it down the chain.
  //
  // TASK PATH: everything else -- no cache connected, or a request the
  // segment cannot serve -- is a task to pool_id_. A connected client is
  // bound to the cache pool or the cache's entry pool, so its tasks pass the
  // cache, which orders them against the writes buffered in the segment.
  //
  // The keys to connect come from the CTE Create (CreateParams::cache_keys_).
  // The Async* methods are the task path only: a Future needs a task, so they
  // never consult the cache.
  // ===========================================================================

  /**
   * Connect to the shm cache module named by @p keys.
   *
   * The client must already be bound to the cache pool or to the pool the
   * cache names as its entry (ShmCacheKeys::FrontPool): only then does every
   * task-path request pass through the cache. The segment must belong to the
   * runtime this process is attached to through shared memory.
   *
   * @param keys segment keys returned by the CTE Create
   * @return true when connected; false leaves the client on the task path
   */
  bool ConnectCache(const ShmCacheKeys &keys) {
    DisconnectCache();
    if (!keys.Valid()) {
      return false;
    }
    const clio::run::PoolId cache_pool =
        clio::run::PoolId::FromU64(keys.cache_pool_);
    if (pool_id_ != cache_pool && pool_id_.ToU64() != keys.entry_pool_) {
      return false;  // this client's tasks would bypass the cache
    }
    auto *ipc = CLIO_CPU_IPC;
    if (ipc == nullptr || ipc->GetMetadataAllocator() == nullptr ||
        static_cast<clio::run::u32>(ipc->GetRuntimePid()) !=
            keys.runtime_pid_) {
      return false;  // not attached to that runtime's shared memory
    }
    wtc::View *view = AttachCacheView(keys);
    if (view == nullptr) {
      return false;
    }
    cache_ = view;
    cache_keys_ = keys;
    return true;
  }

  /**
   * Connect a client that was bound with Init() rather than created. Gets
   * the keys from the same Create CLIO_CTE_CLIENT_INIT issues -- a
   * get-or-create of the canonical CTE pool (kCtePoolId) -- then connects
   * with them. Like CLIO_CTE_CLIENT_INIT, it creates that pool if it does
   * not exist yet.
   * @return true when connected
   */
  bool ConnectCache() {
    ShmCacheKeys keys;
    if (!FetchCacheKeys(&keys)) {
      return false;
    }
    return ConnectCache(keys);
  }

  /**
   * Ask the canonical CTE pool for the cache keys of this node.
   * @param out receives the keys (zero when no cache fronts the stack)
   * @return true when the lookup answered
   */
  bool FetchCacheKeys(ShmCacheKeys *out) {
    Client probe;
    auto task = probe.AsyncCreate(clio::run::PoolQuery::Dynamic(),
                                  std::string(kCtePoolName), kCtePoolId,
                                  CreateParams());
    task.Wait();
    if (task->GetReturnCode() != 0 || task->new_pool_id_.IsNull()) {
      return false;
    }
    // Deserialized directly (not GetParams()): GetParams' compose branch
    // would pull the runtime-only config parser into the client library.
    *out = clio::run::Task::Deserialize<CreateParams>(task->chimod_params_)
               .cache_keys_;
    return true;
  }

  /** Drop the cache connection; the client stays on the task path. */
  void DisconnectCache() {
    cache_ = nullptr;
    cache_keys_ = ShmCacheKeys();
  }

  /** @return true when connected to a ready cache segment */
  bool HasCache() const { return cache_ != nullptr && cache_->Ready(); }

  /** @return the keys of the connected cache (zero when not connected) */
  const ShmCacheKeys &CacheKeys() const { return cache_keys_; }

  /** @return the connected cache's counters, or nullptr */
  const wtc::Stats *CacheStats() const {
    return HasCache() ? &cache_->GetRoot()->stats_ : nullptr;
  }

  /**
   * Write a blob region from a caller-owned buffer and wait for it.
   *
   * SHM cache path first, always when a cache is connected: the bytes are
   * copied into the cache segment and the call returns -- no task, no
   * runtime allocation, no wait. The score and context are not carried by
   * the cache. Only when no cache is connected, or the segment cannot take
   * the write, is it the task path: the bytes are staged into a runtime
   * buffer, a PutBlob task is sent, and the call waits.
   *
   * @param tag_id tag of the blob
   * @param blob_name blob name
   * @param offset offset within the blob
   * @param size bytes to write
   * @param src source bytes (private memory; reusable when this returns)
   * @param score placement score (task path only)
   * @param ctx operation context, or nullptr for none (task path only)
   * @param flags put flags (kCtePut*)
   * @param pool_query task-path routing
   * @return 0 on success; EINVAL for a null source, ENOMEM when the task
   *         path could not stage the bytes, else the PutBlob return code
   */
  clio::run::u32 PutBlob(
      const TagId &tag_id, const std::string &blob_name, clio::run::u64 offset,
      clio::run::u64 size, const char *src, float score = -1.0f,
      const Context *ctx = nullptr, clio::run::u32 flags = 0,
      const clio::run::PoolQuery &pool_query =
          clio::run::PoolQuery::Dynamic()) {
    if (size == 0) {
      return 0;
    }
    if (src == nullptr) {
      return EINVAL;
    }
    if (TryPutCached(tag_id, blob_name, offset, size, src, flags)) {
      return 0;
    }
    const Context none;
    auto fut = AsyncPutBlob(tag_id, blob_name, offset, size, src, score,
                            ctx != nullptr ? *ctx : none, flags, pool_query);
    if (fut.get() == nullptr) {
      return ENOMEM;
    }
    fut.Wait();
    return fut->GetReturnCode();
  }

  /** Client retries of a put refused with kBusy before the task path. */
  static constexpr int kCacheBusyRetries = 8;
  /** First back-off between kBusy retries (us); doubles per retry. */
  static constexpr clio::run::u64 kCacheBusyBackoffStartUs = 4;
  /** Back-off cap between kBusy retries (us). With the start above the
   *  whole retry budget is under 1 ms before the task path takes over. */
  static constexpr clio::run::u64 kCacheBusyBackoffMaxUs = 256;

  /**
   * The SHM cache half of PutBlob alone: copy a write into the segment or
   * report that it cannot. A put refused because the blob's entry is
   * pinned (kBusy: a flush in flight, its buffer must grow) is retried
   * with a bounded exponential back-off, holding no lock while it sleeps,
   * so the flush that holds the pin (and the home's invalidation it
   * waits on) can finish. Any other refusal returns at once.
   * @param tag_id tag of the blob
   * @param blob_name blob name
   * @param offset offset within the blob
   * @param size bytes to write
   * @param src source bytes
   * @param flags put flags (kCtePut*)
   * @return true if the segment took the write; false: use the task path
   */
  bool TryPutCached(const TagId &tag_id, const std::string &blob_name,
                    clio::run::u64 offset, clio::run::u64 size,
                    const char *src, clio::run::u32 flags) {
    if (!HasCache()) {
      return false;
    }
    clio::run::u64 backoff_us = kCacheBusyBackoffStartUs;
    for (int attempt = 0;; ++attempt) {
      const wtc::Status st =
          cache_->TryPut(tag_id.major_, tag_id.minor_, blob_name.data(),
                         blob_name.size(), offset, size, src, flags,
                         /*may_block=*/false);
      if (st == wtc::Status::kOk) {
        return true;
      }
      if (st != wtc::Status::kBusy || attempt >= kCacheBusyRetries) {
        return false;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(backoff_us));
      backoff_us = std::min(backoff_us * 2, kCacheBusyBackoffMaxUs);
    }
  }

  /**
   * Read a blob region into a caller-owned buffer.
   *
   * SHM cache path first: with a cache connected, no flags and no context,
   * a range the segment holds completely is copied out with no task.
   * Otherwise it is the task path (a GetBlob task, waited on).
   *
   * @param tag_id tag of the blob
   * @param blob_name blob name
   * @param offset offset within the blob
   * @param size bytes to read
   * @param dst destination (private memory)
   * @param flags get flags; any flag takes the task path
   * @param ctx operation context, or nullptr for none
   * @param pool_query task-path routing
   * @return 0 on success; EINVAL for a null destination, ENOMEM when the
   *         task path could not stage the read, else the GetBlob return code
   */
  clio::run::u32 GetBlob(
      const TagId &tag_id, const std::string &blob_name, clio::run::u64 offset,
      clio::run::u64 size, char *dst, clio::run::u32 flags = 0,
      const Context *ctx = nullptr,
      const clio::run::PoolQuery &pool_query =
          clio::run::PoolQuery::Dynamic()) {
    if (size == 0) {
      return 0;
    }
    if (dst == nullptr) {
      return EINVAL;
    }
    if (TryGetCached(tag_id, blob_name, offset, size, dst, flags, ctx)) {
      return 0;
    }
    const Context none;
    auto fut = AsyncGetBlob(tag_id, blob_name, offset, size, flags, dst,
                            pool_query, ctx != nullptr ? *ctx : none);
    if (fut.get() == nullptr) {
      return ENOMEM;
    }
    fut.Wait();
    return fut->GetReturnCode();
  }

  /**
   * The SHM cache half of GetBlob alone: serve a read from the segment or
   * report that it cannot. For callers with their own task-path fallback.
   * @param tag_id tag of the blob
   * @param blob_name blob name
   * @param offset offset within the blob
   * @param size bytes to read
   * @param dst destination (host memory)
   * @param flags get flags; any flag declines
   * @param ctx operation context; any context declines
   * @return true when dst holds the bytes
   */
  bool TryGetCached(const TagId &tag_id, const std::string &blob_name,
                    clio::run::u64 offset, clio::run::u64 size, char *dst,
                    clio::run::u32 flags = 0,
                    const Context *ctx = nullptr) {
    return HasCache() && flags == 0 && ctx == nullptr && size != 0 &&
           dst != nullptr &&
           cache_->TryGet(tag_id.major_, tag_id.minor_, blob_name.data(),
                          blob_name.size(), offset, size,
                          dst) == wtc::Status::kOk;
  }

  /**
   * Size of a blob: from the cache segment when it knows the size
   * completely, else a GetBlobSize task.
   * @param tag_id tag of the blob
   * @param blob_name blob name
   * @param size receives the size
   * @param pool_query task-path routing
   * @return 0 on success, else the GetBlobSize return code
   */
  clio::run::u32 GetBlobSize(
      const TagId &tag_id, const std::string &blob_name, clio::run::u64 *size,
      const clio::run::PoolQuery &pool_query =
          clio::run::PoolQuery::Dynamic()) {
    clio::run::u64 cached = 0;
    if (HasCache() &&
        cache_->TryGetSize(tag_id.major_, tag_id.minor_, blob_name.data(),
                           blob_name.size(), &cached) == wtc::Status::kOk) {
      *size = cached;
      return 0;
    }
    auto fut = AsyncGetBlobSize(tag_id, blob_name, pool_query);
    fut.Wait();
    *size = fut->size_;
    return fut->GetReturnCode();
  }

  /**
   * Get-or-create a tag by name. SHM cache path first: a tag the connected
   * cache already holds (name binding and attributes) resolves with no task.
   * Otherwise a GetOrCreateTag task.
   * @param tag_name tag name
   * @param tag_id receives the tag id
   * @param tag_size if non-null, receives the tag's size
   * @param pool_query task-path routing
   * @return 0 on success, else the GetOrCreateTag return code
   */
  clio::run::u32 GetOrCreateTag(
      const std::string &tag_name, TagId *tag_id,
      clio::run::u64 *tag_size = nullptr,
      const clio::run::PoolQuery &pool_query =
          clio::run::PoolQuery::Dynamic()) {
    clio::run::u32 maj = 0;
    clio::run::u32 min = 0;
    wtc::TagAttrs attrs;
    if (HasCache() && !tag_name.empty() &&
        cache_->LookupTagByName(tag_name.data(), tag_name.size(), &maj,
                                &min) &&
        cache_->LookupTagAttrs(maj, min, &attrs)) {
      *tag_id = TagId(maj, min);
      if (tag_size != nullptr) {
        *tag_size = attrs.size_;
      }
      return 0;
    }
    auto fut = AsyncGetOrCreateTag(tag_name, TagId::GetNull(), pool_query);
    fut.Wait();
    *tag_id = fut->tag_id_;
    if (tag_size != nullptr) {
      *tag_size = fut->tag_size_;
    }
    return fut->GetReturnCode();
  }

  /**
   * Asynchronous get blob into a shared-memory buffer (task path only).
   * @param tag_id Tag ID
   * @param blob_name Name of the blob
   * @param offset Offset within blob
   * @param size Size of data
   * @param flags Operation flags
   * @param blob_data Shared memory pointer for output
   * @param pool_query Pool query for task routing (default: Dynamic)
   * @param context Context for I/O emulation control (issue #747)
   * @return Future over the GetBlob task
   */
  clio::run::Future<GetBlobTask> AsyncGetBlob(
      const TagId &tag_id,
      const char *blob_name,
      clio::run::u64 offset, clio::run::u64 size,
      clio::run::u32 flags,
      ctp::ipc::ShmPtr<> blob_data,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic(),
      const Context &context = Context()) {
    auto *ipc_manager = CLIO_CPU_IPC;

    auto task = ipc_manager->NewTask<GetBlobTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, tag_id,
        blob_name, offset, size, flags, blob_data, context);

    return ipc_manager->Send(task);
  }

  /** std::string overload */
  clio::run::Future<GetBlobTask> AsyncGetBlob(
      const TagId &tag_id,
      const std::string &blob_name,
      clio::run::u64 offset, clio::run::u64 size,
      clio::run::u32 flags,
      ctp::ipc::ShmPtr<> blob_data,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic(),
      const Context &context = Context()) {
    return AsyncGetBlob(tag_id, blob_name.c_str(), offset, size,
                        flags, blob_data, pool_query, context);
  }

  /**
   * Private-memory AsyncGetBlob (issue #823): read a blob region straight into
   * a caller-owned PRIVATE buffer (char*), instead of making the caller
   * hand-manage a shared-memory buffer (allocate → pass ShmPtr → copy out →
   * free) as the ShmPtr overload above requires.
   *
   * Task path only (use GetBlob() for the cache-first synchronous read).
   * Two paths:
   *  - Runtime (co-located) mode: the daemon shares this address space, so the
   *    private pointer is wrapped as a null-allocator ShmPtr — IpcManager::
   *    ToFullPtr resolves such a pointer's offset AS the absolute address — and
   *    the bdev read lands DIRECTLY in the caller's buffer. No staging buffer,
   *    no copy, and NOT TASK_DATA_OWNER (the buffer is the caller's, not ours).
   *  - Client mode: the daemon cannot reach private memory, so the read is
   *    staged through a freshly allocated SHM buffer. The task is marked
   *    TASK_DATA_OWNER (its destructor frees the staging buffer on DelTask) and
   *    GetBlobTask::PostWait() copies the staged bytes into the caller's buffer
   *    when the read completes.
   *
   * @return A Future over the read; after Wait() the task is dereferenceable
   *         (GetReturnCode()==0 on success). The ONLY empty Future is the
   *         client-mode staging-allocation failure.
   */
  clio::run::Future<GetBlobTask> AsyncGetBlob(
      const TagId &tag_id, const std::string &blob_name,
      clio::run::u64 offset, clio::run::u64 size, clio::run::u32 flags,
      char *priv_data,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic(),
      const Context &context = Context()) {
    auto *ipc_manager = CLIO_CPU_IPC;

    if (CLIO_RUNTIME_MANAGER->IsRuntime()) {
      // Co-located daemon: write directly into the private buffer. The null
      // AllocatorId marks the offset as an absolute process address.
      ctp::ipc::ShmPtr<> raw = ctp::ipc::ShmPtr<>::FromRaw(priv_data);
      auto task = ipc_manager->NewTask<GetBlobTask>(
          clio::run::CreateTaskId(), pool_id_, pool_query, tag_id, blob_name,
          offset, size, flags, raw, context);
      return ipc_manager->Send(task);
    }

    // Client: stage through an SHM buffer; PostWait() copies it into priv_data
    // and ~GetBlobTask (TASK_DATA_OWNER) frees the staging buffer.
    ctp::ipc::FullPtr<char> staging = ipc_manager->AllocateBuffer(size);
    if (staging.IsNull()) {
      return clio::run::Future<GetBlobTask>();
    }
    auto task = ipc_manager->NewTask<GetBlobTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, tag_id, blob_name,
        offset, size, flags, ctp::ipc::ShmPtr<>(staging.shm_), context);
    auto *t = task.get();
    // priv_dest_/priv_src_ are NOT serialized, so they stay on this client
    // instance (the daemon's copy default-constructs them to null).
    t->priv_dest_ = priv_data;
    t->priv_src_ = staging.ptr_;
    auto fut = ipc_manager->Send(task);
    // Mark ownership only AFTER Send has serialized the task: Task::SerializeIn
    // ships task_flags_, so setting TASK_DATA_OWNER earlier would hand the flag
    // to the daemon, whose task shares the very same physical SHM buffer on the
    // local path — and it would free the client's buffer out from under us.
    // Set client-side, this instance's ~GetBlobTask frees the staging buffer
    // when the Future (task) is destroyed. Same object the Future retains.
    t->SetFlags(TASK_DATA_OWNER);
    return fut;
  }

  /**
   * Vectored get (issue #820): read N regions of one blob in a SINGLE task,
   * each region landing in its OWN buffer.
   *
   * All regions are served from one block-layout snapshot, so the whole read is
   * a single consistent view of the blob rather than N independent ones. Because
   * each segment names its own destination, a caller merging several readers'
   * requests needs no scatter copy afterwards — the runtime fills each reader's
   * buffer directly. Node-local (segment buffers are SHM pointers).
   */
  clio::run::Future<GetBlobTask> AsyncGetBlobVectored(
      const TagId &tag_id,
      const char *blob_name,
      const std::vector<BlobSegment> &segments,
      clio::run::u32 flags = 0,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic(),
      const Context &context = Context()) {
    auto *ipc_manager = CLIO_CPU_IPC;
    auto task = ipc_manager->NewTask<GetBlobTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, tag_id, blob_name,
        static_cast<clio::run::u64>(0), static_cast<clio::run::u64>(0), flags,
        ctp::ipc::ShmPtr<>::GetNull(), context);
    auto *t = task.get();
    for (const auto &seg : segments) {
      t->segments_.push_back(BlobSegment(seg.blob_off_, seg.size_, seg.data_));
    }
    return ipc_manager->Send(task);
  }

  /** std::string overload */
  clio::run::Future<GetBlobTask> AsyncGetBlobVectored(
      const TagId &tag_id,
      const std::string &blob_name,
      const std::vector<BlobSegment> &segments,
      clio::run::u32 flags = 0,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic(),
      const Context &context = Context()) {
    return AsyncGetBlobVectored(tag_id, blob_name.c_str(), segments, flags,
                                pool_query, context);
  }

  /**
   * PRIVATE-MEMORY vectored get: read N regions of one blob in a single task,
   * each region landing in a caller-owned private buffer. Completes the
   * shared/private matrix for the vectored APIs.
   *
   * Two paths (mirrors the scalar private get):
   *  - Runtime (co-located) mode: each segment's pointer is wrapped as a
   *    null-allocator ShmPtr; the bdev reads land directly in the caller's
   *    buffers.
   *  - Client mode: staged through ONE SHM buffer; PostWait() scatters each
   *    slice to its private destination (GetBlobTask::priv_scatter_) and
   *    ~GetBlobTask frees the staging buffer via TASK_DATA_OWNER.
   */
  clio::run::Future<GetBlobTask> AsyncGetBlobVectored(
      const TagId &tag_id, const std::string &blob_name,
      const std::vector<PrivBlobSegment> &segments,
      clio::run::u32 flags = 0,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic(),
      const Context &context = Context()) {
    auto *ipc_manager = CLIO_CPU_IPC;

    if (CLIO_RUNTIME_MANAGER->IsRuntime()) {
      auto task = ipc_manager->NewTask<GetBlobTask>(
          clio::run::CreateTaskId(), pool_id_, pool_query, tag_id,
          blob_name.c_str(), static_cast<clio::run::u64>(0),
          static_cast<clio::run::u64>(0), flags, ctp::ipc::ShmPtr<>::GetNull(),
          context);
      auto *t = task.get();
      for (const auto &seg : segments) {
        t->segments_.push_back(BlobSegment(
            seg.blob_off_, seg.size_, ctp::ipc::ShmPtr<>::FromRaw(seg.data_)));
      }
      return ipc_manager->Send(task);
    }

    // Client mode: stage all segments through ONE SHM buffer and scatter on
    // PostWait.
    clio::run::u64 total = 0;
    for (const auto &seg : segments) total += seg.size_;
    ctp::ipc::FullPtr<char> staging = ipc_manager->AllocateBuffer(total);
    if (staging.IsNull()) {
      return clio::run::Future<GetBlobTask>();
    }
    auto task = ipc_manager->NewTask<GetBlobTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, tag_id,
        blob_name.c_str(), static_cast<clio::run::u64>(0),
        static_cast<clio::run::u64>(0), flags,
        ctp::ipc::ShmPtr<>(staging.shm_), context);
    auto *t = task.get();
    auto *scatter = new std::vector<GetBlobTask::PrivScatter>();
    scatter->reserve(segments.size());
    clio::run::u64 off = 0;
    for (const auto &seg : segments) {
      t->segments_.push_back(BlobSegment(
          seg.blob_off_, seg.size_, ctp::ipc::ShmPtr<>(staging.shm_) + off));
      scatter->push_back(GetBlobTask::PrivScatter{
          seg.data_, staging.ptr_ + off, static_cast<size_t>(seg.size_)});
      off += seg.size_;
    }
    // priv_scatter_ is NOT serialized: it stays on this client instance for
    // PostWait; the daemon's copy default-constructs to null. blob_data_
    // carries the staging buffer for ownership only (the vectored handler
    // reads segments_, not blob_data_); TASK_DATA_OWNER is set after Send so
    // the flag never reaches the daemon's copy.
    t->priv_scatter_ = scatter;
    auto fut = ipc_manager->Send(task);
    t->SetFlags(TASK_DATA_OWNER);
    return fut;
  }

  /**
   * Asynchronous reorganize blob - returns immediately
   * @param tag_id Tag ID
   * @param blob_name Name of the blob
   * @param new_score New placement score
   * @param pool_query Pool query for task routing (default: Dynamic)
   */
  /**
   * @param replica 0 (default) reorganizes the primary; N > 0 migrates
   *        replica N by ITS score (issue #886; REPLICA_FIXED = no-op).
   */
  clio::run::Future<ReorganizeBlobTask> AsyncReorganizeBlob(
      const TagId &tag_id, const std::string &blob_name, float new_score,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic(),
      int replica = 0) {
    auto *ipc_manager = CLIO_CPU_IPC;

    auto task = ipc_manager->NewTask<ReorganizeBlobTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, tag_id,
        blob_name, new_score, replica);

    return ipc_manager->Send(task);
  }

  // ===========================================================================
  // Fully-POD, GPU-compatible blob ops (issue #556). Same parameters as the
  // non-POD versions; the task carries the blob name in an inline
  // fixed_string<32> (capped at 31 chars), so no SSO/SVO fixup is ever needed.
  // ===========================================================================

  clio::run::Future<PodPutBlobTask> AsyncPodPutBlob(
      const TagId &tag_id, const char *blob_name, clio::run::u64 offset,
      clio::run::u64 size, ctp::ipc::ShmPtr<> blob_data, float score = -1.0f,
      const Context &context = Context(), clio::run::u32 flags = 0,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic()) {
    auto *ipc_manager = CLIO_CPU_IPC;
    auto task = ipc_manager->NewTask<PodPutBlobTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, tag_id, blob_name,
        offset, size, blob_data, score, context, flags);
    task.get()->submit_ts_ns_ =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count();
    return ipc_manager->Send(task);
  }

  /** std::string overload */
  clio::run::Future<PodPutBlobTask> AsyncPodPutBlob(
      const TagId &tag_id, const std::string &blob_name, clio::run::u64 offset,
      clio::run::u64 size, ctp::ipc::ShmPtr<> blob_data, float score = -1.0f,
      const Context &context = Context(), clio::run::u32 flags = 0,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic()) {
    return AsyncPodPutBlob(tag_id, blob_name.c_str(), offset, size, blob_data,
                           score, context, flags, pool_query);
  }

  clio::run::Future<PodGetBlobTask> AsyncPodGetBlob(
      const TagId &tag_id, const char *blob_name, clio::run::u64 offset,
      clio::run::u64 size, clio::run::u32 flags, ctp::ipc::ShmPtr<> blob_data,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic()) {
    auto *ipc_manager = CLIO_CPU_IPC;
    auto task = ipc_manager->NewTask<PodGetBlobTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, tag_id, blob_name,
        offset, size, flags, blob_data);
    return ipc_manager->Send(task);
  }

  /** std::string overload */
  clio::run::Future<PodGetBlobTask> AsyncPodGetBlob(
      const TagId &tag_id, const std::string &blob_name, clio::run::u64 offset,
      clio::run::u64 size, clio::run::u32 flags, ctp::ipc::ShmPtr<> blob_data,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic()) {
    return AsyncPodGetBlob(tag_id, blob_name.c_str(), offset, size, flags,
                           blob_data, pool_query);
  }

  // ===========================================================================
  // Batched POD paging. One task carries up to kPodMultiMax page requests, so
  // a full page-cache flush costs a handful of submissions instead of one per
  // page. Build the task, Add() records until it returns false, then Send.
  // ===========================================================================

  /** Allocate an empty batch of `TaskT` (PodMulti{Put,Get}BlobTask /
   *  PodMultiScoreTask) for the caller to fill with Add(). */
  template <typename TaskT>
  clio::run::shared_ptr<TaskT> NewPodBatch(
      const TagId &tag_id, const Context &context = Context(),
      clio::run::u32 flags = 0,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic()) {
    auto *ipc_manager = CLIO_CPU_IPC;
    auto task = ipc_manager->NewTask<TaskT>(clio::run::CreateTaskId(), pool_id_,
                                            pool_query, tag_id);
    task.get()->context_ = context;
    task.get()->flags_ = flags;
    return task;
  }

  /** Submit a batch built with NewPodBatch(). */
  template <typename TaskT>
  clio::run::Future<TaskT> AsyncPodBatch(clio::run::shared_ptr<TaskT> &task) {
    return CLIO_CPU_IPC->Send(task);
  }

  clio::run::Future<PodReorganizeBlobTask> AsyncPodReorganizeBlob(
      const TagId &tag_id, const std::string &blob_name, float new_score,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic()) {
    auto *ipc_manager = CLIO_CPU_IPC;
    auto task = ipc_manager->NewTask<PodReorganizeBlobTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, tag_id,
        blob_name.c_str(), new_score);
    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous delete blob - returns immediately
   * @param tag_id Tag ID
   * @param blob_name Name of the blob
   * @param pool_query Pool query for task routing (default: Dynamic)
   */
  /**
   * Ask whether a byte range of a blob is physically present in the tier.
   *
   * Metadata only — it moves no payload, so it is far cheaper than the read
   * it lets a caller avoid. The intended shape is: try the shared-memory
   * path, and on a miss ask this rather than abandoning the whole request,
   * because a miss in the SHM mirror does not distinguish a hole (zeros are
   * correct) from bytes that exist but are only reachable by RPC.
   *
   * @param tag_id Tag owning the blob
   * @param blob_name Blob name
   * @param offset Start of the range of interest
   * @param size Length of the range; 0 means "to the end of the blob"
   */
  clio::run::Future<GetResidencyTask> AsyncGetResidency(
      const TagId &tag_id, const std::string &blob_name,
      clio::run::u64 offset = 0, clio::run::u64 size = 0,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic()) {
    auto *ipc_manager = CLIO_CPU_IPC;
    auto task = ipc_manager->NewTask<GetResidencyTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, tag_id, blob_name,
        offset, size);
    return ipc_manager->Send(task);
  }

  /**
   * Asynchronously delete a blob.
   * @param tag_id the blob's tag
   * @param blob_name the blob
   * @param pool_query routing (Dynamic: the blob's owner)
   * @param del_flags kDelCacheCopyOnly: drop only the target node's cache
   *        copy (a coherence invalidation); 0: delete the blob
   * @return the task future
   */
  clio::run::Future<DelBlobTask> AsyncDelBlob(
      const TagId &tag_id,
      const std::string &blob_name,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic(),
      clio::run::u32 del_flags = 0) {
    auto *ipc_manager = CLIO_CPU_IPC;

    auto task = ipc_manager->NewTask<DelBlobTask>(clio::run::CreateTaskId(), pool_id_,
                                                  pool_query,
                                                  tag_id, blob_name, del_flags);

    return ipc_manager->Send(task);
  }

  /**
   * Asynchronously evict data off a tier by score until a byte budget is met.
   * Frees the lowest-score blobs residing on any target whose score is at least
   * min_tier_score, cheapest-first, until at least bytes of physical capacity
   * has been reclaimed (or no candidates remain). Broadcast across all
   * containers; the returned task's bytes_evicted_/blobs_evicted_ are the
   * tier-wide totals.
   * @param min_tier_score Only evict blobs on targets with score >= this
   *                       (0.0 = any tier)
   * @param bytes Reclaim at least this many bytes from the tier
   */
  clio::run::Future<EvictTask> AsyncEvict(
      float min_tier_score, clio::run::u64 bytes,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Broadcast(),
      clio::run::u32 droppable_only = 0) {
    auto *ipc_manager = CLIO_CPU_IPC;
    auto task = ipc_manager->NewTask<EvictTask>(clio::run::CreateTaskId(),
                                                pool_id_, pool_query,
                                                min_tier_score, bytes,
                                                droppable_only);
    return ipc_manager->Send(task);
  }

  /**
   * Asynchronously set the data organizer's phase hint (Method::kReorganizeHint).
   * The value is opaque to the core: it is stored on every container and read
   * by the configured DataOrganizer on its next Reorganize round, which decides
   * what a given value means for its policy.
   * @param hint Opaque phase indicator (e.g. algorithm phase or iteration)
   * @param pool_query Routing; Broadcast by default so all containers agree
   */
  clio::run::Future<ReorganizeHintTask> AsyncReorganizeHint(
      clio::run::i32 hint,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Broadcast()) {
    auto *ipc_manager = CLIO_CPU_IPC;
    auto task = ipc_manager->NewTask<ReorganizeHintTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, hint);
    return ipc_manager->Send(task);
  }

  /**
   * Set the data organizer's phase hint and wait until every container has it.
   * @param hint Opaque phase indicator; see AsyncReorganizeHint
   * @return the task's return code (0 on success)
   */
  clio::run::u32 ReorganizeHint(clio::run::i32 hint) {
    auto task = AsyncReorganizeHint(hint);
    task.Wait();
    return task->GetReturnCode();
  }

  /**
   * Asynchronously truncate a blob to an exact logical size (grow/shrink).
   * @param tag_id Tag the blob belongs to
   * @param blob_name Blob to resize
   * @param new_size Target size in bytes (0 frees all blocks)
   * @param pool_query Pool query for task routing (default: Dynamic)
   */
  clio::run::Future<TruncateBlobTask> AsyncTruncateBlob(
      const TagId &tag_id,
      const std::string &blob_name,
      clio::run::u64 new_size,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic()) {
    auto *ipc_manager = CLIO_CPU_IPC;

    auto task = ipc_manager->NewTask<TruncateBlobTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, tag_id, blob_name, new_size);

    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous delete tag by tag ID - returns immediately
   * @param tag_id Tag ID to delete
   * @param pool_query Pool query for task routing (default: Dynamic)
   */
  clio::run::Future<DelTagTask> AsyncDelTag(
      const TagId &tag_id,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic()) {
    auto *ipc_manager = CLIO_CPU_IPC;

    auto task = ipc_manager->NewTask<DelTagTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, tag_id);

    return ipc_manager->Send(task);
  }

  /**
   * Delete a set of blobs in ONE task. The core splits the set by owner:
   * each owner deletes its subset as one plan (one FreeBlocks per pool for
   * the whole subset), all owners concurrently.
   * @param keys       (tag, blob) pairs to delete
   * @param pool_query routing (default Dynamic: the first blob's owner,
   *                   which forwards the other owners' subsets)
   * @param del_flags  kDelCacheCopyOnly to drop only cache copies, else 0
   * @return future; num_ok_ = blobs deleted, first_rc_ = first failure
   *         (1 = a blob was not found)
   */
  clio::run::Future<MultiDelBlobTask> AsyncMultiDelBlob(
      const std::vector<MultiDelKey> &keys,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic(),
      clio::run::u32 del_flags = 0) {
    auto *ipc_manager = CLIO_CPU_IPC;
    const TagId route_tag = keys.empty() ? TagId() : keys.front().tag_id_;
    const std::string route_blob =
        keys.empty() ? std::string() : keys.front().blob_name_;
    auto task = ipc_manager->NewTask<MultiDelBlobTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, route_tag, route_blob,
        EncodeMultiDelKeys(keys), del_flags);
    return ipc_manager->Send(task);
  }

  /**
   * Delete a set of tags (by id) in ONE task per container. Like DelTag by
   * id, a container removes its own share of each tag, so the default
   * Broadcast deletes the set cluster-wide.
   * @param tags       tag ids to delete
   * @param pool_query routing (default Broadcast)
   * @return future; num_ok_ = tags deleted (summed over containers)
   */
  clio::run::Future<MultiDelTagTask> AsyncMultiDelTag(
      const std::vector<TagId> &tags,
      const clio::run::PoolQuery &pool_query =
          clio::run::PoolQuery::Broadcast()) {
    auto *ipc_manager = CLIO_CPU_IPC;
    auto task = ipc_manager->NewTask<MultiDelTagTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query,
        EncodeMultiDelTags(tags));
    return ipc_manager->Send(task);
  }

  /**
   * Broadcast a copy-on-write link (kCowLink) to every container of this pool.
   * Each container records that `copy_tag` must hold a blob's pre-write bytes
   * before a write lands on the same blob of `source_tag`, and registers the
   * copy's fault handler locally.
   * @param source_tag     the tag a lazy copy was taken from
   * @param copy_tag       the lazy copy
   * @param fault_pool_id  pool that materialises the copy (checkpoint)
   * @param fault_pool_name name of that pool
   * @param fault_params   the source tag NAME (the fault handler's parameter)
   * @param pool_query     routing (default: Broadcast)
   * @return future for the broadcast link task
   */
  clio::run::Future<CowLinkTask> AsyncCowLink(
      const TagId &source_tag, const TagId &copy_tag,
      const clio::run::PoolId &fault_pool_id,
      const std::string &fault_pool_name, const std::string &fault_params,
      const clio::run::PoolQuery &pool_query =
          clio::run::PoolQuery::Broadcast()) {
    auto *ipc_manager = CLIO_CPU_IPC;
    auto task = ipc_manager->NewTask<CowLinkTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, source_tag, copy_tag,
        fault_pool_id, fault_pool_name, fault_params);
    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous delete tag by tag name - returns immediately
   * @param tag_name Tag name to delete
   * @param pool_query Pool query for task routing (default: Dynamic)
   */
  clio::run::Future<DelTagTask> AsyncDelTag(
      const std::string &tag_name,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic(),
      bool posix_unlink = false) {
    auto *ipc_manager = CLIO_CPU_IPC;

    auto task = ipc_manager->NewTask<DelTagTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, tag_name);
    // POSIX unlink (#680): promote a surviving alias instead of cascade-deleting.
    task->posix_unlink_ = posix_unlink ? 1u : 0u;

    return ipc_manager->Send(task);
  }

  /**
   * Rename one of a tag's names, keeping its TagId (and all blobs). A tag
   * that already has new_name loses it. tag_id may be null if the caller
   * only knows the name. Batched form: AsyncMultiRenameTag.
   * @param old_name Current tag name
   * @param new_name Desired tag name
   * @param tag_id   Tag id (optional; TagId::GetNull() to resolve by name)
   * @param pool_query Routing (default Dynamic: the old name's owner)
   */
  clio::run::Future<RenameTagTask> AsyncRenameTag(
      const std::string &old_name,
      const std::string &new_name,
      const TagId &tag_id = TagId::GetNull(),
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic()) {
    auto *ipc_manager = CLIO_CPU_IPC;

    auto task = ipc_manager->NewTask<RenameTagTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, tag_id, old_name, new_name);

    return ipc_manager->Send(task);
  }

  /**
   * Bind an additional name to an EXISTING tag's id (tag-level hard link).
   * The alias shares the target's TagId and therefore all its blobs. If the
   * target does not exist, the returned task's found_ is 0 (error). If the
   * alias already names a tag, that tag is returned. Overload: target
   * identified by TagId. Batched form: AsyncMultiGetOrCreateTagAlias.
   * @param existing_id Target tag id (must exist)
   * @param alias_name  New name to bind to it
   */
  clio::run::Future<GetOrCreateTagAliasTask> AsyncGetOrCreateTagAlias(
      const TagId &existing_id,
      const std::string &alias_name,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic()) {
    auto *ipc_manager = CLIO_CPU_IPC;
    auto task = ipc_manager->NewTask<GetOrCreateTagAliasTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, existing_id,
        std::string(), alias_name);
    return ipc_manager->Send(task);
  }

  /**
   * Overload: target identified by name (resolved + verified server-side).
   * @param existing_name Target tag name (must exist)
   * @param alias_name    New name to bind to it
   */
  clio::run::Future<GetOrCreateTagAliasTask> AsyncGetOrCreateTagAlias(
      const std::string &existing_name,
      const std::string &alias_name,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic()) {
    auto *ipc_manager = CLIO_CPU_IPC;
    auto task = ipc_manager->NewTask<GetOrCreateTagAliasTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, TagId::GetNull(),
        existing_name, alias_name);
    return ipc_manager->Send(task);
  }

  /**
   * A tag's canonical name, from the tag's home; the result is in
   * found_/tag_name_.
   * @param tag_id Tag ID to resolve
   * @param pool_query Pool query for task routing (default: the tag's home)
   */
  clio::run::Future<GetTagNameTask> AsyncGetTagName(
      const TagId &tag_id,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic()) {
    auto *ipc_manager = CLIO_CPU_IPC;

    auto task = ipc_manager->NewTask<GetTagNameTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, tag_id);

    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous get tag size - returns immediately
   * @param tag_id Tag ID
   * @param pool_query Pool query for task routing (default: Dynamic)
   */
  clio::run::Future<GetTagSizeTask> AsyncGetTagSize(
      const TagId &tag_id,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic()) {
    auto *ipc_manager = CLIO_CPU_IPC;

    auto task = ipc_manager->NewTask<GetTagSizeTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, tag_id);

    // Cluster-wide size/times (Broadcast) through a connected shm cache:
    // the task goes to the cache pool's TagAttrs verb, which answers from its
    // segment when cached and otherwise fetches the total once and caches it.
    if (pool_query.IsBroadcastMode() && !tag_id.IsNull() && HasCache()) {
      task->pool_id_ = clio::run::PoolId::FromU64(cache_keys_.cache_pool_);
      task->pool_query_ = clio::run::PoolQuery::Local();
      task->method_ = wtc::kTagAttrsMethod;
      return ipc_manager->Send(task);
    }

    // Inline-eligible (CLIO_ENABLE_INLINE_RUN=1): nested metadata
    // lookups from co-located chimods (clio-fs Open does three in a
    // row) each paid a queue+schedule+wake hop; inline they run on
    // the calling fiber. Falls back to Send everywhere else.
    return CLIO_RUN_INLINE(task);
  }

  /**
   * Get max (total) storage capacity in bytes.
   * @param pool_query Local() sums this node's targets; Broadcast() sums the
   *        whole cluster (AggregateOut adds per-node results). Default Local.
   */
  clio::run::Future<GetCapacityTask> AsyncGetCapacity(
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Local()) {
    auto *ipc_manager = CLIO_CPU_IPC;
    auto task = ipc_manager->NewTask<GetCapacityTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query);
    return ipc_manager->Send(task);
  }

  /**
   * Get the number of extra names (tag-level hard links) bound to a tag, by
   * path/name. Excludes the canonical name, so the POSIX link count is
   * num_aliases_ + 1. found_ is 1 if the tag exists.
   * @param tag_name Tag name / absolute path.
   * @param pool_query Broadcast() finds the tag wherever it lives; Local() if
   *        the caller knows the tag is on this node. Default Local.
   */
  clio::run::Future<GetNumAliasesTask> AsyncGetNumAliases(
      const std::string &tag_name,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Local()) {
    auto *ipc_manager = CLIO_CPU_IPC;
    auto task = ipc_manager->NewTask<GetNumAliasesTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, tag_name, TagId::GetNull());
    return ipc_manager->Send(task);
  }

  /**
   * Get the number of extra names bound to a tag, by id. See the by-name
   * overload above for the link-count semantics.
   */
  clio::run::Future<GetNumAliasesTask> AsyncGetNumAliases(
      const TagId &tag_id,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Local()) {
    auto *ipc_manager = CLIO_CPU_IPC;
    auto task = ipc_manager->NewTask<GetNumAliasesTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, std::string(), tag_id);
    return ipc_manager->Send(task);
  }

  /**
   * The batched tag-name verb: kAddName / kRemoveName / kRename records (see
   * EncodeTagNameOp). The receiving container sends each step to the
   * container holding its state; results_ holds one result per record.
   * @param ops encoded batch
   * @param pool_query default Local()
   */
  clio::run::Future<UpdateTagNamesTask> AsyncUpdateTagNames(
      const std::string &ops,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Local()) {
    auto *ipc_manager = CLIO_CPU_IPC;
    auto task = ipc_manager->NewTask<UpdateTagNamesTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, ops);
    return ipc_manager->Send(task);
  }

  /**
   * Batched RenameTag: one UpdateTagNames carrying a kRename per entry.
   * Results (one per entry, op_ = TagNameResult) are in the task's results_.
   * @param renames (tag id or null, old name, new name) per rename
   */
  clio::run::Future<UpdateTagNamesTask> AsyncMultiRenameTag(
      const std::vector<std::tuple<TagId, std::string, std::string>> &renames) {
    std::string ops;
    for (const auto &r : renames) {
      EncodeTagNameOp(&ops, TagNameOp::kRename, std::get<0>(r), 0,
                      std::get<1>(r), std::get<2>(r));
    }
    return AsyncUpdateTagNames(ops);
  }

  /**
   * Batched GetOrCreateTagAlias: one UpdateTagNames carrying a kAddName per
   * entry (the target must exist).
   * @param aliases (target tag id, alias name) per alias
   */
  clio::run::Future<UpdateTagNamesTask> AsyncMultiGetOrCreateTagAlias(
      const std::vector<std::pair<TagId, std::string>> &aliases) {
    std::string ops;
    for (const auto &a : aliases) {
      EncodeTagNameOp(&ops, TagNameOp::kAddName, a.first, kTnMustExist,
                      a.second);
    }
    return AsyncUpdateTagNames(ops);
  }

  /**
   * Asynchronous poll telemetry log - returns immediately
   * @param minimum_logical_time Minimum logical time filter
   * @param pool_query Pool query for task routing (default: Dynamic)
   */
  clio::run::Future<PollTelemetryLogTask> AsyncPollTelemetryLog(
      std::uint64_t minimum_logical_time,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic()) {
    auto *ipc_manager = CLIO_CPU_IPC;

    auto task = ipc_manager->NewTask<PollTelemetryLogTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query,
        minimum_logical_time);

    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous get blob score - returns immediately
   * @param tag_id Tag ID
   * @param blob_name Name of the blob
   * @param pool_query Pool query for task routing (default: Dynamic)
   */
  clio::run::Future<GetBlobScoreTask> AsyncGetBlobScore(
      const TagId &tag_id, const std::string &blob_name,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic()) {
    auto *ipc_manager = CLIO_CPU_IPC;

    auto task = ipc_manager->NewTask<GetBlobScoreTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, tag_id,
        blob_name);

    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous get blob size - returns immediately
   * @param tag_id Tag ID
   * @param blob_name Name of the blob
   * @param pool_query Pool query for task routing (default: Dynamic)
   */
  clio::run::Future<GetBlobSizeTask> AsyncGetBlobSize(
      const TagId &tag_id,
      const std::string &blob_name,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic(),
      int replica = 0) {
    auto *ipc_manager = CLIO_CPU_IPC;

    auto task = ipc_manager->NewTask<GetBlobSizeTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, tag_id,
        blob_name, replica);

    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous get blob info - returns immediately
   * Gets comprehensive blob metadata including score and block placement
   * @param tag_id Tag ID for blob lookup
   * @param blob_name Name of the blob
   * @param pool_query Pool query for task routing (default: Dynamic)
   * @return Future containing score, size, and block placement info
   */
  clio::run::Future<GetBlobInfoTask> AsyncGetBlobInfo(
      const TagId &tag_id,
      const std::string &blob_name,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic()) {
    auto *ipc_manager = CLIO_CPU_IPC;

    auto task = ipc_manager->NewTask<GetBlobInfoTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, tag_id,
        blob_name);

    return ipc_manager->Send(task);
  }

  /**
   * Register `node_id` as holding a cached/replicated copy of the blob
   * (issue #886 coherence). Routed to the blob's owner container; the next
   * primary write there invalidates the copy before completing.
   */
  clio::run::Future<RegisterReplicaContainerTask> AsyncRegisterReplicaContainer(
      const TagId &tag_id, const std::string &blob_name,
      clio::run::u64 node_id,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic(),
      clio::run::u64 expected_version = 0) {
    auto *ipc_manager = CLIO_CPU_IPC;
    auto task = ipc_manager->NewTask<RegisterReplicaContainerTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, tag_id, blob_name,
        node_id, expected_version);
    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous get contained blobs - returns immediately
   * @param tag_id Tag ID
   * @param pool_query Pool query for task routing (default: Dynamic)
   */
  clio::run::Future<GetContainedBlobsTask> AsyncGetContainedBlobs(
      const TagId &tag_id,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic()) {
    auto *ipc_manager = CLIO_CPU_IPC;

    auto task = ipc_manager->NewTask<GetContainedBlobsTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, tag_id);

    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous tag query - returns immediately
   * @param tag_regex Tag regex pattern to match
   * @param max_tags Maximum number of tags to return (0 = no limit)
   * @param pool_query Pool query for routing (default: Broadcast)
   * @return Future for async operation
   */
  clio::run::Future<TagQueryTask> AsyncTagQuery(
      const std::string &tag_regex, clio::run::u32 max_tags = 0,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Broadcast()) {
    auto *ipc_manager = CLIO_CPU_IPC;

    auto task = ipc_manager->NewTask<TagQueryTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, tag_regex, max_tags);

    // Inline-eligible (CLIO_ENABLE_INLINE_RUN=1): nested metadata
    // lookups from co-located chimods (clio-fs Open does three in a
    // row) each paid a queue+schedule+wake hop; inline they run on
    // the calling fiber. Falls back to Send everywhere else.
    return CLIO_RUN_INLINE(task);
  }

  /**
   * Asynchronous blob query - returns immediately
   * @param tag_regex Tag regex pattern to match
   * @param blob_regex Blob regex pattern to match
   * @param max_blobs Maximum number of blobs to return (0 = no limit)
   * @param pool_query Pool query for routing (default: Broadcast)
   * @return Future for async operation
   */
  clio::run::Future<BlobQueryTask> AsyncBlobQuery(
      const std::string &tag_regex, const std::string &blob_regex,
      clio::run::u32 max_blobs = 0,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Broadcast()) {
    auto *ipc_manager = CLIO_CPU_IPC;

    auto task = ipc_manager->NewTask<BlobQueryTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, tag_regex, blob_regex,
        max_blobs);

    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous semantic search — BM25 keyword scoring over blob bytes.
   * @param tag_regex   Full-string match against tag names (std::regex_match)
   * @param blob_regex  Full-string match against blob names within matching tags
   * @param query_text  Natural-language query; tokenized and scored vs blobs
   * @param k           Maximum number of results returned, ordered by
   *                    descending BM25 score. 0 means "no cap".
   * @param pool_query  Default Broadcast — same as BlobQuery — so the
   *                    search runs across every tag-owning container.
   */
  clio::run::Future<SemanticSearchTask> AsyncSemanticSearch(
      const std::string &tag_regex, const std::string &blob_regex,
      const std::string &query_text, clio::run::u32 k = 10,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Broadcast()) {
    auto *ipc_manager = CLIO_CPU_IPC;
    auto task = ipc_manager->NewTask<SemanticSearchTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, tag_regex, blob_regex,
        query_text, k);
    return ipc_manager->Send(task);
  }
  /**
   * Asynchronous temporal search — filter blobs by last-modified timestamp.
   * @param tag_regex    Full-string match against tag names (std::regex_match)
   * @param blob_regex   Full-string match against blob names within matching tags
   * @param time_begin   Inclusive lower bound, epoch nanoseconds (0 = no lower bound)
   * @param time_end     Inclusive upper bound, epoch nanoseconds (0 = no upper bound)
   * @param max_entries  Cap on returned results, sorted ascending by timestamp (0 = unlimited)
   * @param pool_query   Default Broadcast — search across every tag-owning container.
   */
  clio::run::Future<TemporalSearchTask> AsyncTemporalSearch(
      const std::string &tag_regex, const std::string &blob_regex,
      Timestamp time_begin = 0, Timestamp time_end = 0,
      clio::run::u32 max_entries = 0,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Broadcast()) {
    auto *ipc_manager = CLIO_CPU_IPC;
    auto task = ipc_manager->NewTask<TemporalSearchTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, tag_regex, blob_regex,
        time_begin, time_end, max_entries);
    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous flush metadata - returns immediately
   * @param pool_query Pool query for task routing (default: Local)
   * @param period_us Period in microseconds (0 = one-shot)
   */
  clio::run::Future<FlushMetadataTask> AsyncFlushMetadata(
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Local(),
      double period_us = 0) {
    auto *ipc_manager = CLIO_CPU_IPC;

    auto task = ipc_manager->NewTask<FlushMetadataTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query);

    if (period_us > 0) {
      task->SetPeriod(period_us, clio::run::kMicro);
      task->SetFlags(TASK_PERIODIC);
    }

    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous flush data - returns immediately
   * @param pool_query Pool query for task routing (default: Local)
   * @param target_persistence_level Minimum persistence level for flush target
   * @param period_us Period in microseconds (0 = one-shot)
   */
  clio::run::Future<FlushDataTask> AsyncFlushData(
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Local(),
      int target_persistence_level = 1,
      double period_us = 0) {
    auto *ipc_manager = CLIO_CPU_IPC;

    auto task = ipc_manager->NewTask<FlushDataTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, target_persistence_level);

    if (period_us > 0) {
      task->SetPeriod(period_us, clio::run::kMicro);
      task->SetFlags(TASK_PERIODIC);
    }

    return ipc_manager->Send(task);
  }

  /**
   * List the blobs (tag id + name) whose names fully match `blob_regex` on
   * the containers `pool_query` reaches - asynchronous. Local() lists this
   * node's blobs; Broadcast() the whole cluster's.
   * @param blob_regex std::regex over blob names
   * @param pool_query routing (default: Local)
   * @return future for the ListLocalBlobsTask
   */
  clio::run::Future<ListLocalBlobsTask> AsyncListLocalBlobs(
      const std::string &blob_regex,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Local()) {
    auto *ipc_manager = CLIO_CPU_IPC;
    auto task = ipc_manager->NewTask<ListLocalBlobsTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, blob_regex);
    return ipc_manager->Send(task);
  }

  /**
   * One operation on a tag's xattr section - asynchronous. See TagXattrTask
   * for the ops, flags and return codes.
   * @param tag_id tag whose section is addressed
   * @param op kTagXattrSet / kTagXattrGet / kTagXattrDel / kTagXattrList
   * @param flags kTagXattrCreate / kTagXattrReplace / kTagXattrMustExist
   * @param key the key (List: the cursor; keys after it are returned)
   * @param value the value to store (Set only)
   * @param max_keys List page size (0 = all)
   * @param pool_query routing (default: Dynamic, the tag id's hashed owner)
   * @return future for the TagXattrTask
   */
  clio::run::Future<TagXattrTask> AsyncTagXattr(
      const TagId &tag_id, clio::run::u32 op, clio::run::u32 flags,
      const std::string &key, const std::string &value = std::string(),
      clio::run::u32 max_keys = 0,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic()) {
    auto *ipc_manager = CLIO_CPU_IPC;
    auto task = ipc_manager->NewTask<TagXattrTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, tag_id, op, flags,
        key, value, max_keys);
    return ipc_manager->Send(task);
  }

  /**
   * Insert or replace one key of a tag's xattr section - asynchronous.
   * @param tag_id the tag
   * @param key the key
   * @param value the value
   * @param flags kTagXattrCreate / kTagXattrReplace (0 = upsert)
   * @param pool_query routing (default: Dynamic)
   * @return future for the TagXattrTask (rc 0 on success)
   */
  clio::run::Future<TagXattrTask> AsyncTagSetXattr(
      const TagId &tag_id, const std::string &key, const std::string &value,
      clio::run::u32 flags = 0,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic()) {
    return AsyncTagXattr(tag_id, kTagXattrSet, flags, key, value, 0,
                         pool_query);
  }

  /**
   * Read one key of a tag's xattr section - asynchronous.
   * @param tag_id the tag
   * @param key the key
   * @param pool_query routing (default: Dynamic)
   * @return future for the TagXattrTask (found_, value_)
   */
  clio::run::Future<TagXattrTask> AsyncTagGetXattr(
      const TagId &tag_id, const std::string &key,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic()) {
    return AsyncTagXattr(tag_id, kTagXattrGet, 0, key, std::string(), 0,
                         pool_query);
  }

  /**
   * Remove one key of a tag's xattr section, or every key when `key` is
   * empty - asynchronous.
   * @param tag_id the tag
   * @param key the key (empty = all)
   * @param flags kTagXattrMustExist (0 = absent is fine)
   * @param pool_query routing (default: Dynamic)
   * @return future for the TagXattrTask
   */
  clio::run::Future<TagXattrTask> AsyncTagDelXattr(
      const TagId &tag_id, const std::string &key, clio::run::u32 flags = 0,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic()) {
    return AsyncTagXattr(tag_id, kTagXattrDel, flags, key, std::string(), 0,
                         pool_query);
  }

  /**
   * List the keys of a tag's xattr section in order - asynchronous.
   * @param tag_id the tag
   * @param cursor return keys strictly after this one (empty = from start)
   * @param max_keys page size (0 = all)
   * @param pool_query routing (default: Dynamic)
   * @return future for the TagXattrTask (keys_; found_ = 1 if more remain)
   */
  clio::run::Future<TagXattrTask> AsyncTagListXattr(
      const TagId &tag_id, const std::string &cursor = std::string(),
      clio::run::u32 max_keys = 0,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic()) {
    return AsyncTagXattr(tag_id, kTagXattrList, 0, cursor, std::string(),
                         max_keys, pool_query);
  }

  /**
   * fsync(2) for one tag - asynchronous. Broadcast by default so every
   * container makes the blobs it holds durable (see SyncTagTask).
   * @param tag_id tag whose blobs must become durable
   * @param pool_query routing (default: Broadcast)
   * @param min_persistence minimum tier level; < 0 uses the configured
   *        flush_data_min_persistence
   * @return future for the SyncTagTask (rc kSyncNoSpaceRc / kSyncIoRc on
   *         failure; deferred_ = 1 when fsync_mode is "deferred")
   */
  clio::run::Future<SyncTagTask> AsyncSyncTag(
      const TagId &tag_id,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Broadcast(),
      int min_persistence = -1) {
    auto *ipc_manager = CLIO_CPU_IPC;
    auto task = ipc_manager->NewTask<SyncTagTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, tag_id,
        min_persistence);
    if (pool_query.IsBroadcastMode()) {
      task->sync_flags_ |= kSyncFannedOut;  // every container gets this sync
    }
    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous dynamic reorganize - periodic internal data-organizer driver
   * (issue #738). Spawned from the CTE server's Create() once per configured
   * organizer replica; each firing delegates to the configured DataOrganizer.
   * @param pool_query Pool query for task routing (default: Local)
   * @param replica_id 0-based organizer replica index (partitions blob space)
   * @param period_us Period in microseconds (0 = one-shot)
   */
  clio::run::Future<DynamicReorganizeTask> AsyncDynamicReorganize(
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Local(),
      clio::run::u32 replica_id = 0,
      double period_us = 0) {
    auto *ipc_manager = CLIO_CPU_IPC;

    auto task = ipc_manager->NewTask<DynamicReorganizeTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, replica_id);

    if (period_us > 0) {
      task->SetPeriod(period_us, clio::run::kMicro);
      task->SetFlags(TASK_PERIODIC);
    }

    return ipc_manager->Send(task);
  }
#endif  // CTP_IS_HOST

#if CTP_IS_HOST
 private:
  /** Connected cache segment (process-wide view, never freed), or null. */
  wtc::View *cache_ = nullptr;
  /** Keys of the connected cache (zero when not connected). */
  ShmCacheKeys cache_keys_;

  /** Guards CacheViews(). A function-local static, not a member, so Client
   *  stays copyable (callers copy it; a mutex member would delete that). */
  static std::mutex &CacheViewMutex() {
    static std::mutex mu;
    return mu;
  }

  /** Process-wide segment views, one per (runtime pid, cache pool). Clients
   *  hold raw pointers to them, so they live until process exit. */
  static std::unordered_map<std::string, std::unique_ptr<wtc::View>> &
  CacheViews() {
    static std::unordered_map<std::string, std::unique_ptr<wtc::View>> views;
    return views;
  }

  /**
   * Map the segment the keys name, or reuse this process's mapping of it.
   * @param keys valid cache keys
   * @return the ready view, or nullptr when the segment cannot be mapped
   */
  static wtc::View *AttachCacheView(const ShmCacheKeys &keys) {
    const std::string name = wtc::SegmentName(
        keys.runtime_pid_, clio::run::PoolId::FromU64(keys.cache_pool_));
    std::lock_guard<std::mutex> lk(CacheViewMutex());
    auto &views = CacheViews();
    auto it = views.find(name);
    if (it != views.end()) {
      return it->second->Ready() ? it->second.get() : nullptr;
    }
    auto view = std::make_unique<wtc::View>();
    if (!view->Attach(name, keys.root_off_)) {
      return nullptr;
    }
    wtc::View *raw = view.get();
    views[name] = std::move(view);
    return raw;
  }
#endif
};

// Global pointer-based singleton for CTE client with lazy initialization
CLIO_CTE_DEFINE_GLOBAL_PTR_VAR_H(clio::cte::core::Client, g_cte_client);

/**
 * Initialize CTE client and configuration subsystem
 * @param config_path Optional path to configuration file
 * @param pool_query Pool query type for CTE container creation (default:
 * Dynamic)
 * @return true if initialization succeeded, false otherwise
 */
bool CLIO_CTE_CLIENT_INIT(
    const std::string &config_path = "",
    const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic());

/**
 * Tag wrapper class - provides convenient API for tag operations
 */
class Tag {
 private:
  TagId tag_id_;
  std::string tag_name_;

 public:
  /**
   * Constructor - Call the CLIO_CTE client GetOrCreateTag function
   * @param tag_name Tag name to get or create
   */
  explicit Tag(const std::string &tag_name);

  /**
   * Constructor - Does not call CLIO_CTE client function, just sets the TagId
   * variable
   * @param tag_id Tag ID to use directly
   */
  explicit Tag(const TagId &tag_id);

  /**
   * PutBlob - Allocates a SHM pointer and then calls PutBlob (SHM)
   * @param blob_name Name of the blob
   * @param data Raw data pointer
   * @param data_size Size of data
   * @param off Offset within blob (default 0)
   * @param score Blob score for placement decisions (default 1.0)
   * @param context Compression context for workflow-aware decisions (default empty)
   */
  void PutBlob(const std::string &blob_name, const char *data, size_t data_size,
               size_t off = 0, float score = 1.0f, const Context &context = Context());

  /**
   * PutBlob (SHM) - Direct shared memory version
   * @param blob_name Name of the blob
   * @param data Shared memory pointer to data
   * @param data_size Size of data
   * @param off Offset within blob (default 0)
   * @param score Blob score for placement: -1.0=unknown (auto), 0.0-1.0=explicit tier
   * @param context Compression context for workflow-aware decisions (default empty)
   */
  void PutBlob(const std::string &blob_name, const ctp::ipc::ShmPtr<> &data,
               size_t data_size, size_t off = 0, float score = -1.0f,
               const Context &context = Context());

  /**
   * Asynchronous PutBlob (SHM) - Caller must manage shared memory lifecycle
   * @param blob_name Name of the blob
   * @param data Shared memory pointer to data (must remain valid until task
   * completes)
   * @param data_size Size of data
   * @param off Offset within blob (default 0)
   * @param score Blob score for placement: -1.0=unknown (auto), 0.0-1.0=explicit tier
   * @param context Compression context for workflow-aware decisions (default empty)
   * @return Task pointer for async operation
   * @note For raw data, caller must allocate shared memory using
   * CLIO_IPC->AllocateBuffer<void>() and keep the FullPtr alive until the async
   * task completes
   */
  clio::run::Future<PutBlobTask> AsyncPutBlob(const std::string &blob_name,
                                        const ctp::ipc::ShmPtr<> &data,
                                        size_t data_size, size_t off = 0,
                                        float score = -1.0f,
                                        const Context &context = Context());

  /**
   * Asynchronous private-memory PutBlob (issue #830): write a blob region
   * straight from a caller-owned PRIVATE buffer (const char*, e.g. heap/stack),
   * with no manual shared-memory management. Write-side analog of the #823
   * private GetBlob; delegates to CoreClient::AsyncPutBlob(const char*) — see
   * there for the two paths (runtime-direct no-copy / client-staging with
   * TASK_DATA_OWNER).
   *
   * @param blob_name Name of the blob to write
   * @param data Source PRIVATE buffer (must stay valid until Wait() returns)
   * @param data_size Number of bytes to write
   * @param off Offset within blob (default 0)
   * @param score Blob score for placement (default -1.0 = auto)
   * @param context Compression context (default empty)
   * @return Future over the write; readable after Wait() (GetReturnCode()==0 on
   *         success). An empty Future is returned only if client-mode SHM
   *         staging could not be allocated.
   */
  clio::run::Future<PutBlobTask> AsyncPutBlob(const std::string &blob_name,
                                              const char *data,
                                              size_t data_size, size_t off = 0,
                                              float score = -1.0f,
                                              const Context &context = Context());

  /**
   * GetBlob - Allocates shared memory, retrieves blob data, copies to output
   * buffer
   * @param blob_name Name of the blob to retrieve
   * @param data Output buffer to copy blob data into (must be pre-allocated by
   * caller)
   * @param data_size Size of data to retrieve (must be > 0)
   * @param off Offset within blob (default 0)
   * @note Automatically handles shared memory allocation/deallocation
   */
  void GetBlob(const std::string &blob_name, char *data, size_t data_size,
               size_t off = 0);

  /**
   * GetBlob (SHM) - Retrieves blob data into pre-allocated shared memory buffer
   * @param blob_name Name of the blob to retrieve
   * @param data Pre-allocated shared memory pointer for output data (must not
   * be null)
   * @param data_size Size of data to retrieve (must be > 0)
   * @param off Offset within blob (default 0)
   * @note Caller must pre-allocate shared memory using
   * CLIO_IPC->AllocateBuffer<void>(data_size)
   */
  void GetBlob(const std::string &blob_name, ctp::ipc::ShmPtr<> data,
               size_t data_size, size_t off = 0);

  /**
   * Asynchronous private-memory GetBlob (issue #823): read a blob region
   * straight into a caller-owned PRIVATE buffer (char*, e.g. heap/stack), with
   * no manual shared-memory management. Delegates to
   * CoreClient::AsyncGetBlob(char*) — the task path (runtime-direct /
   * client-staging).
   *
   * @param blob_name Name of the blob to retrieve
   * @param data Output PRIVATE buffer (pre-allocated by caller, >= data_size)
   * @param data_size Size of data to retrieve (must be > 0)
   * @param off Offset within blob (default 0)
   * @return Future over the read; the task is readable after Wait()
   *         (GetReturnCode()==0 on success) and the buffer holds the bytes.
   */
  clio::run::Future<GetBlobTask> AsyncGetBlob(const std::string &blob_name,
                                              char *data, size_t data_size,
                                              size_t off = 0);

  /**
   * Get blob score
   * @param blob_name Name of the blob
   * @return Blob score (0.0-1.0)
   */
  float GetBlobScore(const std::string &blob_name);

  /**
   * Get blob size
   * @param blob_name Name of the blob
   * @return Blob size in bytes
   */
  clio::run::u64 GetBlobSize(const std::string &blob_name);

  /**
   * Get all blob names contained in this tag
   * @return Vector of blob names in this tag
   */
  std::vector<std::string> GetContainedBlobs();

  /**
   * Reorganize blob with new score for data placement optimization
   * @param blob_name Name of the blob to reorganize
   * @param new_score New placement score (0.0-1.0, higher = faster tier)
   */
  void ReorganizeBlob(const std::string &blob_name, float new_score);

  /**
   * Get the TagId for this tag
   * @return TagId of this tag
   */
  const TagId &GetTagId() const { return tag_id_; }
};

// Flush + reset the global Tag::PutBlob(const char*) timing accumulator
// and print a per-call breakdown of alloc / memcpy / RPC / free.
// Intended to be called by the POSIX (and other) adapters at the end of
// a `Filesystem::Write` so a single line summarises one user-visible
// write operation. Pass a short label that identifies the caller
// (e.g. "Write off=0 size=4G").
void FlushPutBlobTiming(const char *label);

}  // namespace clio::cte::core

// Global singleton macro for CTE client access (returns pointer, not reference)
#define CLIO_CTE_CLIENT                               \
  (&(*CTP_GET_GLOBAL_PTR_VAR(clio::cte::core::Client, \
                              clio::cte::core::g_cte_client)))

// Intermediate `clio_cte` namespace-spelling alias. In-tree code uses
// `clio::cte::core::*`; downstream that migrated to the `clio_cte::*` waypoint
// keeps compiling via this alias. Safe to use the simple `namespace X = Y;`
// form because no external chimod opens `namespace clio_cte::xxx {}`.
// (The wrp_cte/ forwarder shim tree and the wrp_cte::/WRP_CTE_* compat aliases
// were removed; downstream must use the clio_cte / clio::cte names.)
namespace clio_cte = clio::cte;

#endif  // WRPCTE_CORE_CLIENT_H_
