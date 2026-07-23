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

#include <clio_runtime/clio_runtime.h>
#include <clio_ctp/util/singleton.h>
#include <clio_cte/api.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/core/shm_metadata_cache.h>
#include <clio_runtime/bdev/transports/mem_bdev_transport.h>
#include <cstring>
#include <unordered_map>

namespace clio::cte::core {

class Client : public clio::run::ContainerClient {
 public:
  CTP_CROSS_FUN Client() = default;
  CTP_CROSS_FUN explicit Client(const clio::run::PoolId &pool_id) { Init(pool_id); }

#if CTP_IS_HOST
  // =========================================================================
  // issue #783: shared-memory metadata cache (client side).
  //
  // Strictly an OPTIMIZATION. Every accessor reports failure rather than
  // guessing, and a caller that gets `false` must fall back to the normal RPC
  // path. The cache can be absent (no metadata segment), stale (writes are
  // asynchronous), or full -- none of which are errors.
  // =========================================================================

  /**
   * Attach the cache using the root offset returned by Create.
   *
   * @param root_off offset from CreateParams::shm_cache_root_off_. 0 means the
   *        runtime is not caching, and this simply leaves the cache disabled.
   * @return true if the cache is now usable.
   */
  bool AttachShmCache(clio::run::u64 root_off) {
    shm_root_ = nullptr;
    if (root_off == 0) {
      HLOG(kDebug, "[#783] AttachShmCache: root_off=0 (runtime not caching)");
      return false;  // runtime is not caching -- not an error
    }
    auto *ipc = CLIO_CPU_IPC;
    if (ipc == nullptr) {
      return false;
    }
    auto *alloc = ipc->GetMetadataAllocator();
    if (alloc == nullptr) {
      HLOG(kDebug, "[#783] AttachShmCache: no metadata allocator attached");
      return false;  // segment not attached (e.g. TCP client, or remote node)
    }
    auto *root = reinterpret_cast<ShmMetadataCacheRoot *>(
        reinterpret_cast<char *>(alloc) + root_off);
    // Refuse anything we do not recognize. The cache is derived state, so
    // declining to use it is always safe; guessing at a layout is not.
    if (root->ready_ != 1 ||
        root->version_ != ShmMetadataCacheRoot::kLayoutVersion) {
      HLOG(kDebug,
           "[#783] AttachShmCache: rejected root (ready={}, version={}, "
           "expected version={})",
           root->ready_, root->version_,
           ShmMetadataCacheRoot::kLayoutVersion);
      return false;
    }
    shm_root_ = root;
    return true;
  }

  /**
   * Attach by discovering the cache through the metadata-segment directory.
   *
   * This is the RELIABLE path. The CreateTask overload below only carries a
   * non-zero offset for the process that actually caused the pool to be
   * created; in practice compose creates the CTE pool at startup, so every
   * later client sees zero there. Discovery must not depend on being first.
   */
  bool AttachShmCache() {
    auto *ipc = CLIO_CPU_IPC;
    if (ipc == nullptr) {
      return false;
    }
    auto *dir = ipc->GetMetadataDirectory();
    if (dir == nullptr) {
      return false;
    }
    // Look up THIS client's pool. Using a shared slot would attach whichever
    // CTE pool cached last, silently reading another pool's metadata.
    return AttachShmCache(dir->FindRoot(pool_id_.ToU64()));
  }

  /** Convenience: attach from a completed CreateTask, falling back to the
   *  directory when the task carries no offset (i.e. the pool already
   *  existed, which is the common case). */
  bool AttachShmCache(CreateTask &task) {
    clio::run::u64 off = task.GetParams().shm_cache_root_off_;
    if (off != 0 && AttachShmCache(off)) {
      return true;
    }
    return AttachShmCache();
  }

  bool HasShmCache() const { return shm_root_ != nullptr; }

  /**
   * Zero-IPC metadata read.
   *
   * @return true if a consistent record was found in shared memory. false
   *         means "not cached / could not read consistently" -- ALWAYS fall
   *         back to the RPC path, never treat it as "blob does not exist".
   */
  bool TryGetBlobRecordShm(const TagId &tag_id, const std::string &blob_name,
                           ShmBlobRecord *out) const {
    if (shm_root_ == nullptr || out == nullptr) {
      return false;
    }
    // Key is built into a stack buffer: a client must never allocate inside
    // the runtime-owned segment, and this is the hot path.
    char key[512];
    size_t n = MakeShmBlobKey(tag_id, blob_name.data(), blob_name.size(), key,
                              sizeof(key));
    if (n == 0) {
      return false;  // name too long for the fast path
    }
    return shm_root_->blob_key_to_info_.TryGetBytes(key, n, out);
  }

  /**
   * Zero-IPC PAYLOAD read (issue #783 phase 6).
   *
   * Copies blob bytes straight out of the RAM bdev's shared-memory segment,
   * with no round-trip at all.
   *
   * COHERENCE (design §5.3): the metadata seqlock protects the record copy,
   * but the hazard is the DataOrganizer relocating blocks WHILE we memcpy.
   * So `placement_gen_` is validated before and AFTER the copy, and a change
   * discards the bytes and reports failure. Without the post-check this
   * function would silently return another blob's data.
   *
   * @return true only if `size` bytes were copied AND placement did not move.
   *         Any false -- not cached, not direct-readable, segment missing,
   *         placement changed -- means "fall back to RPC", never "no data".
   */
  bool TryReadBlobShm(const TagId &tag_id, const std::string &blob_name,
                      char *out, size_t size, size_t offset = 0) {
    if (shm_root_ == nullptr || out == nullptr || size == 0) {
      return false;
    }
    ShmBlobRecord rec;
    if (!TryGetBlobRecordShm(tag_id, blob_name, &rec)) {
      return false;
    }
    if (!rec.IsDirectReadable()) {
      return false;  // file/remote/GPU-tier blob, or truncated block list
    }
    if (offset + size > rec.total_size_) {
      return false;
    }
    const clio::run::u64 gen_before = rec.placement_gen_;

    size_t copied = 0;
    clio::run::u64 want_from = offset;
    for (clio::run::u32 i = 0; i < rec.num_blocks_ && copied < size; ++i) {
      const ShmBlockDesc &b = rec.blocks_[i];
      if (b.size_ <= want_from) {
        want_from -= b.size_;  // this block is entirely before `offset`
        continue;
      }
      char *base = MapRamBdev(b.target_pool_);
      if (base == nullptr) {
        return false;  // cannot reach this device -> RPC
      }
      clio::run::u64 intra = want_from;
      size_t avail = static_cast<size_t>(b.size_ - intra);
      size_t chunk = std::min(avail, size - copied);
      std::memcpy(out + copied, base + b.target_offset_ + intra, chunk);
      copied += chunk;
      want_from = 0;
    }
    if (copied != size) {
      return false;
    }

    // Re-validate placement. If the blob moved while we were copying, the
    // bytes may be a mix of two blobs -- discard and let the caller use RPC.
    ShmBlobRecord after;
    if (!TryGetBlobRecordShm(tag_id, blob_name, &after)) {
      return false;
    }
    if (after.placement_gen_ != gen_before) {
      return false;
    }
    return true;
  }

  /** Zero-IPC tag-name lookup. */
  bool TryGetTagIdShm(const std::string &tag_name, TagId *out) const {
    if (shm_root_ == nullptr || out == nullptr) {
      return false;
    }
    return shm_root_->tag_name_to_id_.TryGetBytes(tag_name.data(),
                                                  tag_name.size(), out);
  }
#endif

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
   * Routes to CPU admin worker via PoolQuery::ToLocalCpu().
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
   */
  clio::run::Future<RegisterTargetTask> AsyncRegisterTarget(
      const std::string &target_name, clio::run::bdev::BdevType bdev_type,
      clio::run::u64 total_size,
      const clio::run::PoolQuery &target_query = clio::run::PoolQuery::Local(),
      const clio::run::PoolId &bdev_id = clio::run::PoolId::GetNull(),
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic(),
      clio::run::u32 attach_existing = 0) {
    auto *ipc_manager = CLIO_CPU_IPC;

    auto task = ipc_manager->NewTask<RegisterTargetTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, target_name,
        bdev_type, total_size, target_query, bdev_id, attach_existing);

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

    return ipc_manager->Send(task);
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
   * Asynchronous get blob - returns immediately
   * @param tag_id Tag ID
   * @param blob_name Name of the blob
   * @param offset Offset within blob
   * @param size Size of data
   * @param flags Operation flags
   * @param blob_data Shared memory pointer for output
   * @param pool_query Pool query for task routing (default: Dynamic)
   * @param context Context for I/O emulation control (issue #747)
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
   * Asynchronous reorganize blob - returns immediately
   * @param tag_id Tag ID
   * @param blob_name Name of the blob
   * @param new_score New placement score
   * @param pool_query Pool query for task routing (default: Dynamic)
   */
  clio::run::Future<ReorganizeBlobTask> AsyncReorganizeBlob(
      const TagId &tag_id, const std::string &blob_name, float new_score,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic()) {
    auto *ipc_manager = CLIO_CPU_IPC;

    auto task = ipc_manager->NewTask<ReorganizeBlobTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, tag_id,
        blob_name, new_score);

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
  clio::run::Future<DelBlobTask> AsyncDelBlob(
      const TagId &tag_id,
      const std::string &blob_name,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic()) {
    auto *ipc_manager = CLIO_CPU_IPC;

    auto task = ipc_manager->NewTask<DelBlobTask>(clio::run::CreateTaskId(), pool_id_,
                                                  pool_query,
                                                  tag_id, blob_name);

    return ipc_manager->Send(task);
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
   * Rename a tag, keeping its TagId (and all blobs). Broadcast so every
   * container moves the name binding it holds. tag_id may be null if the
   * caller only knows the name; pass it when known (e.g. from a prior open).
   * @param old_name Current tag name
   * @param new_name Desired tag name
   * @param tag_id   Tag id (optional; TagId::GetNull() to resolve by name)
   * @param pool_query Routing (default Broadcast)
   */
  clio::run::Future<RenameTagTask> AsyncRenameTag(
      const std::string &old_name,
      const std::string &new_name,
      const TagId &tag_id = TagId::GetNull(),
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Broadcast()) {
    auto *ipc_manager = CLIO_CPU_IPC;

    auto task = ipc_manager->NewTask<RenameTagTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, tag_id, old_name, new_name);

    return ipc_manager->Send(task);
  }

  /**
   * Bind an additional name to an EXISTING tag's id (tag-level hard link).
   * The alias shares the target's TagId and therefore all its blobs. If the
   * target does not exist, the returned task's found_ is 0 (error). Broadcast.
   * Overload: target identified by TagId.
   * @param existing_id Target tag id (must exist)
   * @param alias_name  New name to bind to it
   */
  clio::run::Future<GetOrCreateTagAliasTask> AsyncGetOrCreateTagAlias(
      const TagId &existing_id,
      const std::string &alias_name,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Broadcast()) {
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
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Broadcast()) {
    auto *ipc_manager = CLIO_CPU_IPC;
    auto task = ipc_manager->NewTask<GetOrCreateTagAliasTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, TagId::GetNull(),
        existing_name, alias_name);
    return ipc_manager->Send(task);
  }

  /**
   * Resolve a TagId to its full, absolute tag name. Tag names are stored
   * relatively ("$tagid{parent}/leaf"); this returns the fully-resolved path
   * (or the verbatim name for a flat tag). Broadcast so the container that
   * owns the tag's metadata answers; the result is in found_/tag_name_.
   * @param tag_id Tag ID to resolve
   * @param pool_query Pool query for task routing (default: Broadcast)
   */
  clio::run::Future<GetTagNameTask> AsyncGetTagName(
      const TagId &tag_id,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Broadcast()) {
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

    return ipc_manager->Send(task);
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
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic()) {
    auto *ipc_manager = CLIO_CPU_IPC;

    auto task = ipc_manager->NewTask<GetBlobSizeTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, tag_id,
        blob_name);

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

    return ipc_manager->Send(task);
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
  // issue #783: resolved address of the SHM metadata cache root in THIS
  // process, or nullptr when caching is unavailable. Not owned -- the runtime
  // owns the segment and may drop the cache at any time, which is why every
  // read is validated rather than trusted.
  ShmMetadataCacheRoot *shm_root_ = nullptr;

  /** One attached RAM-bdev segment, cached so a hot read does not re-attach. */
  struct RamBdevMap {
    ctp::ipc::PosixShmMmap backend;
    char *base = nullptr;  /**< byte 0 of device space */
  };
  // Keyed by target pool. Attaching is not free, and the fast path is meant to
  // be a few hundred nanoseconds, so a miss here would dominate the cost.
  std::unordered_map<clio::run::u64, RamBdevMap> ram_bdevs_;

  /**
   * Resolve (attaching on first use) the base address of a RAM bdev's shared
   * memory in THIS process, or nullptr when unavailable.
   *
   * The segment name is reconstructed from the runtime pid plus the target
   * pool id, both of which the client already holds -- no name is published
   * anywhere. A negative result is cached as a null base so a device that
   * cannot be mapped is not retried on every single read.
   */
  char *MapRamBdev(const clio::run::PoolId &pool_id) {
    clio::run::u64 key = pool_id.ToU64();
    auto it = ram_bdevs_.find(key);
    if (it != ram_bdevs_.end()) {
      return it->second.base;
    }
    auto *ipc = CLIO_CPU_IPC;
    RamBdevMap &slot = ram_bdevs_[key];  // caches the negative result too
    if (ipc == nullptr) {
      return nullptr;
    }
    clio::run::u32 server_pid = static_cast<clio::run::u32>(ipc->GetRuntimePid());
    if (server_pid == 0) {
      return nullptr;
    }
    std::string name =
        clio::run::bdev::MemBdevTransport::ShmSegmentName(server_pid, pool_id);
    if (!slot.backend.shm_attach(name)) {
      return nullptr;
    }
    char *raw = reinterpret_cast<char *>(slot.backend.data_);
    if (raw == nullptr) {
      return nullptr;
    }
    auto *hdr =
        reinterpret_cast<clio::run::bdev::MemBdevTransport::ShmRamHeader *>(raw);
    // Refuse a segment we do not recognize or that is being torn down --
    // Destroy() clears ready_ before unmapping precisely so this check works.
    if (hdr->version_ !=
            clio::run::bdev::MemBdevTransport::ShmRamHeader::kVersion ||
        hdr->ready_ != 1) {
      return nullptr;
    }
    slot.base = raw + hdr->data_off_;
    return slot.base;
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
