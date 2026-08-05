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
#include <clio_cte/core/blob_batch.h>
#include <clio_cte/core/shm_metadata_cache.h>
#include <clio_runtime/bdev/transports/mem_bdev_transport.h>
#include <atomic>
#include <cstring>
#include <deque>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

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
    shm_replica_serving_ = false;
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

  /**
   * Attach a DIFFERENT pool's mirror — the interposition case (issue #886).
   * A client bound to an interposer pool (e.g. replication at 561.0) reads
   * primaries that the CORE pool mirrors, so the zero-IPC fast path must
   * attach the core's directory slot. Safe under the replication interposer
   * because writes update the primary synchronously in the put path (the
   * mirror is never stale) and a dropped primary is re-mirrored EMPTY, so
   * the fast path misses and falls back to the task path — which is the
   * interposer's replica-serving ladder.
   *
   * This binding ALSO enables serving-replica reads (the untransformed
   * cache copy): for a stack-bound client the task path would run the
   * whole interposer chain and hand back PRODUCER bytes, so raw replica
   * bytes are the correct answer. A client bound directly to the mirrored
   * pool keeps the core's stored-bytes contract (#818: GetBlob returns
   * STORED bytes) and must never be short-cut onto a raw copy — the
   * replication sweep reads through exactly such a client, and copying raw
   * bytes as if they were the stored form corrupts every replica.
   */
  bool AttachShmCacheOf(const clio::run::PoolId &mirror_pool) {
    auto *ipc = CLIO_CPU_IPC;
    if (ipc == nullptr) {
      return false;
    }
    auto *dir = ipc->GetMetadataDirectory();
    if (dir == nullptr) {
      return false;
    }
    if (!AttachShmCache(dir->FindRoot(mirror_pool.ToU64()))) {
      return false;
    }
    shm_replica_serving_ = (mirror_pool != pool_id_);
    return true;
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
    // Source selection (issue #886 cache/replication split): the primary when
    // it is direct-readable and covers the range; otherwise the published
    // serving replica — the UNTRANSFORMED node-local copy the cache chimod
    // maintains while the authoritative bytes live transformed below. Both
    // are guarded by the same placement generation.
    //
    // Bound by the CACHED PREFIX, not by the blob's total size: a truncated
    // primary record describes only its first kMaxInlineBlocks blocks, and a
    // read past them has no block to resolve against.
    const bool primary_ok =
        rec.IsDirectReadable() && offset + size <= rec.CoveredBytes();
    // Serving-replica reads only for STACK-bound clients (AttachShmCacheOf):
    // they alias the whole interposer chain, whose task path returns
    // producer bytes. A direct core client keeps stored-bytes semantics.
    const bool replica_ok = shm_replica_serving_ && !primary_ok &&
                            rec.HasServableReplica() &&
                            offset + size <= rec.RepCoveredBytes();
    if (!primary_ok && !replica_ok) {
      return false;  // transformed/file/remote/GPU-tier and no serving replica
    }
    const ShmBlockDesc *src_blocks = primary_ok ? rec.blocks_ : rec.rep_blocks_;
    const clio::run::u32 src_nblocks =
        primary_ok ? rec.num_blocks_ : rec.rep_num_blocks_;
    const clio::run::u64 gen_before = rec.placement_gen_;

    size_t copied = 0;
    clio::run::u64 want_from = offset;
    for (clio::run::u32 i = 0; i < src_nblocks && copied < size; ++i) {
      const ShmBlockDesc &b = src_blocks[i];
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

  /**
   * Zero-copy VIEW of a blob's payload in the shared RAM-bdev segment
   * (issues #859/#862). Returns a pointer INTO the mapped segment for a
   * single-extent, direct-readable blob, plus the placement generation the
   * caller must re-validate with CheckBlobGenShm AFTER consuming the bytes
   * (same optimistic discipline as TryReadBlobShm, with the consume replacing
   * the copy). Refuses multi-extent, truncated, transformed (#818), or
   * non-RAM blobs -- the caller falls back to a copying read.
   *
   * LIFETIME/SAFETY: the pointer is valid only while the blob's placement is
   * unchanged; a failed CheckBlobGenShm means the bytes consumed may be torn
   * and the operation must be retried via a copying path. An in-place
   * overwrite (same placement) does NOT bump the generation -- concurrent
   * same-blob overwrite vs view is torn-content-visible, exactly as it is
   * for the copying fast path.
   */
  bool TryGetBlobViewShm(const TagId &tag_id, const std::string &blob_name,
                         const char **ptr, clio::run::u64 *size,
                         clio::run::u64 *gen) {
    if (shm_root_ == nullptr || ptr == nullptr || size == nullptr ||
        gen == nullptr) {
      return false;
    }
    ShmBlobRecord rec;
    if (!TryGetBlobRecordShm(tag_id, blob_name, &rec)) {
      return false;
    }
    // Source selection (issue #886): the primary when viewable, else the
    // published serving replica (the untransformed cache copy) when it is a
    // single extent. Same zero-size distrust and generation contract.
    const bool primary_view = rec.IsDirectReadable() && rec.num_blocks_ == 1 &&
                              (rec.flags_ & kShmBlobTruncated) == 0 &&
                              rec.total_size_ != 0;
    // Stack-bound clients only, same rule as TryReadBlobShm above.
    const bool replica_view = shm_replica_serving_ && !primary_view &&
                              rec.HasServableReplica() &&
                              rec.rep_num_blocks_ == 1 &&
                              rec.rep_total_size_ != 0;
    if (!primary_view && !replica_view) {
      // A zero-size record is untrustworthy, not "empty" (issue #862): a
      // fresh blob's record can be mirrored before its completing put
      // republishes the size. Refuse so the caller falls back to RPC.
      return false;
    }
    const ShmBlockDesc &src =
        primary_view ? rec.blocks_[0] : rec.rep_blocks_[0];
    char *base = MapRamBdev(src.target_pool_);
    if (base == nullptr) {
      return false;
    }
    *ptr = base + src.target_offset_;
    *size = primary_view ? rec.total_size_ : rec.rep_total_size_;
    *gen = rec.placement_gen_;
    return true;
  }

  /** Re-validate a view taken with TryGetBlobViewShm: true iff the blob's
   *  placement generation is unchanged (bytes consumed from the view were
   *  stable). */
  bool CheckBlobGenShm(const TagId &tag_id, const std::string &blob_name,
                       clio::run::u64 gen) {
    ShmBlobRecord rec;
    if (!TryGetBlobRecordShm(tag_id, blob_name, &rec)) {
      return false;
    }
    return rec.placement_gen_ == gen;
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
   * The issue #830 client-mode zero-IPC fast path (write straight into the
   * cached blob, skipping the task entirely) is deliberately NOT taken here: a
   * token-less direct write to the RAM bdev segment can race the DataOrganizer
   * relocating the blob — it frees/reuses those blocks under the per-blob write
   * token a pure client cannot hold, so the write would corrupt whichever blob
   * next owns them. Unlike the read fast path, a placement_gen recheck can
   * DETECT that race but not UNDO the write. Making it safe needs a
   * client-visible pin/lease against reorganization; left as a follow-up.
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

    if (CLIO_RUNTIME_MANAGER->IsRuntime() && !NoPrivPutEnv()) {
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

    // Client (or CLIO_CTE_NO_PRIV_PUT): stage through an SHM buffer, copying
    // the private bytes in ONCE.
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
    if (CLIO_RUNTIME_MANAGER->IsRuntime()) {
      // Co-located (CLIO_CTE_NO_PRIV_PUT staging): client and daemon share this
      // ONE task object — nothing is serialized, so the flag never "ships", and
      // setting it after Send would race the worker's task_flags_ reads. Set it
      // BEFORE Send; the single ~PutBlobTask frees the staging buffer exactly
      // once, after the future completes.
      t->SetFlags(TASK_DATA_OWNER);
      return ipc_manager->Send(task);
    }
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

  // ==== Deferred-put pipeline (issue #862) ==================================
  //
  // Promoted from the YCSB binding: a process-wide registry of in-flight
  // deferred puts giving callers (a) unbounded async writes with flow control
  // tied to REAL shared-memory capacity rather than a fixed depth, and (b)
  // read-after-write consistency without waiting on unrelated writes at read
  // time unless needed.
  //
  //  - AsyncPutBlobDefer: submit a private-memory put and register its future
  //    in the registry (FIFO + per-key table). If client-mode SHM staging is
  //    exhausted (AsyncPutBlob returns an empty future), the oldest deferred
  //    puts are awaited one at a time — each releasing its staging buffer —
  //    until the submit succeeds.
  //  - AsyncGetBlobDefer: await any pending put(s) for the SAME blob first
  //    (Wait on their futures), then read — so a get after an acked put never
  //    misses or returns pre-put bytes.
  //  - AwaitPutsUntilSpace: await oldest deferred puts until at most
  //    `max_inflight_bytes` of payload remain in flight (0 = drain).
  //
  // The registry is process-wide (Client instances may be copied; a static
  // registry keeps the API on the client without breaking copyability).
  // Completion failures are counted, not thrown: poll DeferErrorCount().
  struct DeferredPut {
    clio::run::Future<clio::run::Task> fut_;  // one future may cover a BATCH
    struct Ent {
      clio::run::u64 key_;   // DeferKeyHash of (tag, name)
      clio::run::u64 seq_;   // submission order; identifies latest put per key
      clio::run::u64 size_;
    };
    std::vector<Ent> ents_;
    // Pool-managed staging buffer (issue #892): returned to the registry's
    // staging pool at reap instead of freed. Null for non-pooled paths.
    ctp::ipc::FullPtr<char> staging_;
    clio::run::u64 staging_size_ = 0;
  };
  // Keys are 64-bit FNV-1a hashes (no per-op allocation) and the per-key
  // pending table is sharded 16 ways: a client-mode mixed workload keeps puts
  // in flight nearly always, so EVERY read consults this table — a single
  // global mutex + a heap-allocated string key per read collapsed the
  // separated-runtime read throughput (~3x on YCSB B/D). A hash collision
  // only causes a spurious same-shard await — harmless for correctness.
  struct DeferRegistry {
    static constexpr size_t kShards = 16;
    // Per-key pending state: EVERY in-flight put's extent for the key, not
    // just the latest. A put task carries its own bytes (SHM staging in both
    // modes — see AsyncPutBlobDefer), so a read can be composed from the
    // newest-wins union of the pending extents regardless of runtime mode.
    // data_ stays valid exactly as long as its extent is listed (the reaper
    // removes the extent under the shard lock BEFORE the task is freed).
    struct PendingExtent {
      clio::run::u64 seq_ = 0;
      const char *data_ = nullptr;
      clio::run::u64 offset_ = 0;
      clio::run::u64 size_ = 0;
    };
    struct KeyPending {
      clio::run::u32 count_ = 0;
      std::vector<PendingExtent> extents_;  // submission order (seq ascending)
    };
    struct Shard {
      std::mutex mtx_;
      std::unordered_map<clio::run::u64, KeyPending> per_key_;
    };
    Shard shards_[kShards];
    std::mutex mtx_;  // guards fifo_ + inflight_bytes_
    std::deque<DeferredPut> fifo_;
    clio::run::u64 inflight_bytes_ = 0;
    // Lock-free emptiness signal: readers on the hot path check this before
    // touching any lock — a pure-read phase pays one relaxed load per get.
    std::atomic<clio::run::u64> pending_count_{0};
    std::atomic<clio::run::u64> errors_{0};
    static DeferRegistry &Get() {
      static DeferRegistry r;
      return r;
    }
    std::atomic<clio::run::u64> seq_gen_{0};
    // Accumulating batch (issue #862, AsyncMultiPutVectored as the deferred
    // pipeline's output): puts bump-copy into a staging chunk and register
    // their extents immediately (reads serve from them pre-flush); the chunk
    // ships as ONE MultiPutBlobTask when kBatchMax puts accumulate, the chunk
    // fills, or an await needs it flushed. kBatchChunk caps how many payload
    // bytes a batch accumulates AND is the large-value bypass threshold: a
    // single value this size or larger skips batching entirely (one direct
    // put), so batches only ever aggregate values smaller than it.
    static constexpr clio::run::u64 kBatchMax = 64;
    static constexpr clio::run::u64 kBatchChunk = 128 * 1024;
    // Staging buffer pool for LARGE deferred puts (issue #892): allocating a
    // fresh SHM buffer per 1 MiB put costs first-touch page faults plus an
    // allocator walk that degrades to milliseconds under churn — measured as
    // THE client-side submit bottleneck (p50 2.7-4.4 ms/submit vs ~0.1 ms
    // with a recycled buffer). Completed puts return their staging here;
    // submits pop an exact-size match (pre-faulted, no allocator).
    static constexpr size_t kPoolMaxBufs = 64;
    std::mutex pool_mtx_;
    std::vector<std::pair<ctp::ipc::FullPtr<char>, clio::run::u64>> pool_;
    std::mutex batch_mtx_;
    std::atomic<Client *> flush_client_{nullptr};  // for static await entries
    ctp::ipc::FullPtr<char> batch_chunk_{};
    clio::run::u64 batch_cap_ = 0;
    clio::run::u64 batch_used_ = 0;
    std::vector<MultiPutDesc> batch_descs_;
    std::vector<DeferredPut::Ent> batch_ents_;
    Shard &ShardFor(clio::run::u64 key) { return shards_[key % kShards]; }
    bool IsKeyPending(clio::run::u64 key) {
      Shard &sh = ShardFor(key);
      std::lock_guard<std::mutex> lk(sh.mtx_);
      return sh.per_key_.find(key) != sh.per_key_.end();
    }
    void KeyAdd(clio::run::u64 key, clio::run::u64 seq, const char *data,
                clio::run::u64 offset, clio::run::u64 size) {
      Shard &sh = ShardFor(key);
      std::lock_guard<std::mutex> lk(sh.mtx_);
      KeyPending &kp = sh.per_key_[key];
      kp.count_++;
      if (data != nullptr) {
        kp.extents_.push_back(PendingExtent{seq, data, offset, size});
      }
    }
    void KeyRelease(clio::run::u64 key, clio::run::u64 seq) {
      Shard &sh = ShardFor(key);
      std::lock_guard<std::mutex> lk(sh.mtx_);
      auto it = sh.per_key_.find(key);
      if (it == sh.per_key_.end()) return;
      // Remove the retiring put's extent BEFORE its buffer is freed; later
      // reads compose from the remaining pending extents (or fall back).
      auto &ex = it->second.extents_;
      for (size_t i = 0; i < ex.size(); ++i) {
        if (ex[i].seq_ == seq) {
          ex.erase(ex.begin() + static_cast<long>(i));
          break;
        }
      }
      if (--(it->second.count_) == 0) {
        sh.per_key_.erase(it);
      }
    }
    /** Compose [offset, offset+size) from the SET of pending puts for the
     *  key, newest submission winning per byte. 1 = fully served; 0 = no
     *  pending put for the key; -1 = pending but the union does not cover
     *  the whole range — the caller must fall back to awaiting. The copy
     *  runs under the shard lock, which is what keeps every source buffer
     *  alive for its duration. */
    int TryServe(clio::run::u64 key, clio::run::u64 offset, char *dst,
                 clio::run::u64 size, clio::run::u64 *served_size) {
      Shard &sh = ShardFor(key);
      std::lock_guard<std::mutex> lk(sh.mtx_);
      auto it = sh.per_key_.find(key);
      if (it == sh.per_key_.end()) return 0;
      const auto &ex = it->second.extents_;
      if (ex.empty()) return -1;
      // Newest-first overlay: fill remaining gaps of the request from each
      // extent until nothing is uncovered. Extent counts are tiny (usually
      // 1), so a simple gap list is enough.
      struct Gap { clio::run::u64 lo, hi; };
      std::vector<Gap> gaps{{offset, offset + size}};
      for (size_t i = ex.size(); i-- > 0 && !gaps.empty();) {
        const PendingExtent &e = ex[i];
        clio::run::u64 elo = e.offset_, ehi = e.offset_ + e.size_;
        std::vector<Gap> next;
        for (const Gap &g : gaps) {
          clio::run::u64 lo = g.lo > elo ? g.lo : elo;
          clio::run::u64 hi = g.hi < ehi ? g.hi : ehi;
          if (lo >= hi) {  // no overlap
            next.push_back(g);
            continue;
          }
          std::memcpy(dst + (lo - offset), e.data_ + (lo - elo), hi - lo);
          if (g.lo < lo) next.push_back(Gap{g.lo, lo});
          if (hi < g.hi) next.push_back(Gap{hi, g.hi});
        }
        gaps.swap(next);
      }
      if (!gaps.empty()) return -1;
      if (served_size != nullptr) *served_size = size;
      return 1;
    }
  };

  /** Allocation-free 64-bit key for (tag, blob name): FNV-1a. */
  static clio::run::u64 DeferKeyHash(const TagId &tag_id,
                                     const std::string &name) {
    clio::run::u64 h = 1469598103934665603ull;
    const auto *t = reinterpret_cast<const unsigned char *>(&tag_id);
    for (size_t i = 0; i < sizeof(tag_id); ++i) {
      h = (h ^ t[i]) * 1099511628211ull;
    }
    for (unsigned char c : name) {
      h = (h ^ c) * 1099511628211ull;
    }
    return h;
  }

  /** Await the single oldest deferred put. @return false if none was
   *  available to claim (the fifo may be empty while other threads are still
   *  mid-Wait on claimed entries — per_key_/pending_count_/inflight_bytes_
   *  stay accounted until THEIR waits finish).
   *
   *  Ordering matters (read-after-write): the per-key/pending/bytes
   *  bookkeeping is released only AFTER Wait() returns. Releasing it at pop
   *  time opened a race where a reader's AwaitPendingPuts saw "not pending"
   *  while the reaping thread was still waiting on that key's put — the read
   *  then missed the not-yet-published blob (the residual YCSB-D NOT_FOUNDs
   *  that survived the first RAW implementation). */
  /** Pop a recycled staging buffer of EXACTLY `size` bytes, or allocate a
   *  fresh one. Pool hits skip both the allocator walk and first-touch
   *  faults (issue #892). */
  static ctp::ipc::FullPtr<char> PoolAllocStaging(clio::run::u64 size) {
    DeferRegistry &reg = DeferRegistry::Get();
    {
      std::lock_guard<std::mutex> lk(reg.pool_mtx_);
      for (size_t i = 0; i < reg.pool_.size(); ++i) {
        if (reg.pool_[i].second == size) {
          ctp::ipc::FullPtr<char> buf = reg.pool_[i].first;
          reg.pool_[i] = reg.pool_.back();
          reg.pool_.pop_back();
          return buf;
        }
      }
    }
    return CLIO_IPC->AllocateBuffer(size);
  }

  /** Return a staging buffer to the pool (or free it when the pool is
   *  full). */
  static void PoolFreeStaging(ctp::ipc::FullPtr<char> buf,
                              clio::run::u64 size) {
    if (buf.IsNull()) {
      return;
    }
    DeferRegistry &reg = DeferRegistry::Get();
    {
      std::lock_guard<std::mutex> lk(reg.pool_mtx_);
      if (reg.pool_.size() < DeferRegistry::kPoolMaxBufs) {
        reg.pool_.emplace_back(buf, size);
        return;
      }
    }
    CLIO_IPC->FreeBuffer(buf);
  }

  static bool DeferAwaitOldest() {
    DeferRegistry &reg = DeferRegistry::Get();
    DeferredPut entry;
    bool claimed = false;
    while (!claimed) {
      {
        std::lock_guard<std::mutex> lk(reg.mtx_);
        if (!reg.fifo_.empty()) {
          entry = std::move(reg.fifo_.front());
          reg.fifo_.pop_front();
          claimed = true;
          break;
        }
      }
      // Fifo empty: the puts the caller waits on may still sit in the
      // ACCUMULATING batch. Flush it OUTSIDE every registry lock (the flush
      // itself takes batch_mtx_ and then reg.mtx_ to publish) and retry; if
      // the batch is also empty, there is genuinely nothing to await.
      Client *fc = reg.flush_client_.load(std::memory_order_acquire);
      if (fc == nullptr) {
        return false;
      }
      bool have_batch;
      {
        std::lock_guard<std::mutex> blk(reg.batch_mtx_);
        have_batch = !reg.batch_descs_.empty();
      }
      if (!have_batch) {
        return false;
      }
      fc->FlushDeferBatch();
    }
    entry.fut_.Wait();
    auto *t = entry.fut_.get();
    if (t == nullptr || t->GetReturnCode() != 0) {
      reg.errors_.fetch_add(1);
    }
    // Recycle the pool-managed staging buffer (issue #892).
    PoolFreeStaging(entry.staging_, entry.staging_size_);
    clio::run::u64 bytes = 0;
    for (const auto &e : entry.ents_) bytes += e.size_;
    {
      std::lock_guard<std::mutex> lk(reg.mtx_);
      reg.pending_count_.fetch_sub(entry.ents_.size(),
                                   std::memory_order_relaxed);
      reg.inflight_bytes_ -= bytes;
    }
    for (const auto &e : entry.ents_) {
      reg.KeyRelease(e.key_, e.seq_);
    }
    return true;
  }

  /**
   * Non-blocking reap of ALREADY-COMPLETED oldest deferred puts (issue
   * #892): pops FIFO fronts whose futures are complete, releasing their
   * bookkeeping and returning pooled staging buffers. Called opportunistically
   * at submit time so a long burst continuously recycles its staging instead
   * of allocating fresh (fault-cold) buffers for every put.
   */
  static void DeferReapCompleted() {
    DeferRegistry &reg = DeferRegistry::Get();
    while (true) {
      DeferredPut entry;
      {
        std::lock_guard<std::mutex> lk(reg.mtx_);
        if (reg.fifo_.empty()) {
          return;
        }
        auto *t = reg.fifo_.front().fut_.get();
        if (t == nullptr || !t->IsComplete()) {
          return;
        }
        entry = std::move(reg.fifo_.front());
        reg.fifo_.pop_front();
      }
      entry.fut_.Wait();  // already complete — returns immediately
      auto *t = entry.fut_.get();
      if (t == nullptr || t->GetReturnCode() != 0) {
        reg.errors_.fetch_add(1);
      }
      PoolFreeStaging(entry.staging_, entry.staging_size_);
      clio::run::u64 bytes = 0;
      for (const auto &e : entry.ents_) bytes += e.size_;
      {
        std::lock_guard<std::mutex> lk(reg.mtx_);
        reg.pending_count_.fetch_sub(entry.ents_.size(),
                                     std::memory_order_relaxed);
        reg.inflight_bytes_ -= bytes;
      }
      for (const auto &e : entry.ents_) {
        reg.KeyRelease(e.key_, e.seq_);
      }
    }
  }

  /**
   * Deferred put: submit AND register; the registry owns the future and THIS
   * CALL OWNS A COPY of the bytes — `priv_data` may be reused or freed the
   * moment it returns, in every mode. In runtime (co-located) mode the copy
   * is staged in SHARED MEMORY and the put reads it directly (one copy, no
   * caller-lifetime coupling); client mode stages identically inside
   * AsyncPutBlob. Puts therefore grow until shared memory is genuinely
   * exhausted, at which point this call awaits the oldest deferred puts —
   * each releasing its staging — until the allocation succeeds.
   *
   * @param max_inflight_bytes optional pacing wall owned by this method:
   *        before submitting, await oldest puts until at most this much
   *        payload remains in flight. 0 (default) = no wall — bounded only
   *        by shared memory itself.
   * @return 0 submitted; -1 degenerate request (size 0 / null source); -2
   *         shared memory exhausted with nothing left to await.
   */
  int AsyncPutBlobDefer(
      const TagId &tag_id, const std::string &blob_name, clio::run::u64 offset,
      clio::run::u64 size, const char *priv_data, float score = -1.0f,
      const Context &context = Context(), clio::run::u32 flags = 0,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic(),
      clio::run::u64 max_inflight_bytes = 0) {
    if (size == 0 || priv_data == nullptr) {
      return -1;
    }
    if (max_inflight_bytes != 0) {
      AwaitPutsUntilSpace(max_inflight_bytes);
    }
    if (size >= DeferRegistry::kBatchChunk) {
      // Recycle completed puts' staging FIRST (non-blocking), so a burst
      // feeds its own pool instead of allocating fault-cold buffers.
      DeferReapCompleted();
      // Stage through the RECYCLED pool (issue #892): an exact-size pool hit
      // is a pre-faulted buffer and no allocator walk — the fresh-allocation
      // path was the measured submit bottleneck. The SHM-pointer put overload
      // leaves ownership with us; the buffer returns to the pool at reap.
      ctp::ipc::FullPtr<char> staging = PoolAllocStaging(size);
      if (!staging.IsNull()) {
        std::memcpy(staging.ptr_, priv_data, size);
        int rc = DeferPutLarge(
            tag_id, blob_name, offset, size, staging.ptr_,
            [&] {
              return AsyncPutBlob(tag_id, blob_name, offset, size,
                                  staging.shm_.template Cast<void>(), score,
                                  context, flags, pool_query);
            },
            staging, size);
        if (rc != 0) {
          PoolFreeStaging(staging, size);
        }
        return rc;
      }
      return DeferPutLarge(tag_id, blob_name, offset, size, priv_data, [&] {
        return AsyncPutBlob(tag_id, blob_name, offset, size, priv_data, score,
                            context, flags, pool_query);
      });
    }
    (void)score;
    (void)context;
    (void)flags;
    DeferRegistry &reg = DeferRegistry::Get();
    reg.flush_client_.store(this, std::memory_order_release);
    clio::run::u64 key = DeferKeyHash(tag_id, blob_name);
    clio::run::u64 seq = reg.seq_gen_.fetch_add(1) + 1;
    const char *copied = nullptr;
    {
      std::unique_lock<std::mutex> lk(reg.batch_mtx_);
      char *dst = ReserveDeferBatchLocked(reg, lk, size);
      if (dst == nullptr) {
        return -2;
      }
      std::memcpy(dst, priv_data, size);
      copied = dst;
      MultiPutDesc d;
      d.tag_id_ = tag_id;
      d.blob_name_ = blob_name;
      d.offset_ = offset;
      d.size_ = size;
      d.payload_off_ = reg.batch_used_;
      reg.batch_descs_.push_back(std::move(d));
      reg.batch_ents_.push_back(DeferredPut::Ent{key, seq, size});
      reg.batch_used_ += size;
    }
    // Extent registered as soon as the bytes are copied: reads serve
    // read-your-writes from the ACCUMULATING batch, before it even ships.
    reg.KeyAdd(key, seq, copied, offset, size);
    {
      std::lock_guard<std::mutex> lk(reg.mtx_);
      reg.pending_count_.fetch_add(1, std::memory_order_relaxed);
      reg.inflight_bytes_ += size;
    }
    return 0;
  }

  /**
   * Deferred VECTORED put of ONE blob (issue #862): the segments are
   * bump-copied CONTIGUOUSLY into the accumulating deferred batch — for
   * callers that assemble a value from parts (e.g. the lmcache CLIOKV1
   * header + metadata + payload), the assembly IS the staging copy; no
   * extra gather pass. The segments must tile a contiguous range ascending
   * from segments[0].blob_off_ (the pipeline registers ONE extent per put).
   * Same ownership contract as AsyncPutBlobDefer: every source buffer may
   * be reused or freed the moment this returns, in every mode.
   *
   * @return 0 submitted; -1 degenerate request (no segments, a null/empty
   *         segment, or a gap between segments); -2 shared memory exhausted
   *         with nothing left to await.
   */
  int AsyncPutBlobVectoredDefer(
      const TagId &tag_id, const std::string &blob_name,
      const std::vector<PrivBlobSegment> &segments,
      clio::run::u64 max_inflight_bytes = 0) {
    if (segments.empty()) {
      return -1;
    }
    clio::run::u64 total = 0;
    clio::run::u64 next_off = segments.front().blob_off_;
    for (const auto &seg : segments) {
      if (seg.data_ == nullptr || seg.size_ == 0 ||
          seg.blob_off_ != next_off) {
        return -1;
      }
      next_off += seg.size_;
      total += seg.size_;
    }
    if (max_inflight_bytes != 0) {
      AwaitPutsUntilSpace(max_inflight_bytes);
    }
    const clio::run::u64 front_off = segments.front().blob_off_;
    if (total >= DeferRegistry::kBatchChunk) {
      // Multi-extent source: register count-only (no served extent) — a read
      // of a still-pending large record awaits it instead of composing.
      return DeferPutLarge(tag_id, blob_name, front_off, total, nullptr, [&] {
        return AsyncPutBlobVectored(tag_id, blob_name, segments);
      });
    }
    DeferRegistry &reg = DeferRegistry::Get();
    reg.flush_client_.store(this, std::memory_order_release);
    clio::run::u64 key = DeferKeyHash(tag_id, blob_name);
    clio::run::u64 seq = reg.seq_gen_.fetch_add(1) + 1;
    const clio::run::u64 offset = front_off;
    const char *copied = nullptr;
    {
      std::unique_lock<std::mutex> lk(reg.batch_mtx_);
      char *dst = ReserveDeferBatchLocked(reg, lk, total);
      if (dst == nullptr) {
        return -2;
      }
      copied = dst;
      for (const auto &seg : segments) {
        std::memcpy(dst, seg.data_, seg.size_);
        dst += seg.size_;
      }
      MultiPutDesc d;
      d.tag_id_ = tag_id;
      d.blob_name_ = blob_name;
      d.offset_ = offset;
      d.size_ = total;
      d.payload_off_ = reg.batch_used_;
      reg.batch_descs_.push_back(std::move(d));
      reg.batch_ents_.push_back(DeferredPut::Ent{key, seq, total});
      reg.batch_used_ += total;
    }
    reg.KeyAdd(key, seq, copied, offset, total);
    {
      std::lock_guard<std::mutex> lk(reg.mtx_);
      reg.pending_count_.fetch_add(1, std::memory_order_relaxed);
      reg.inflight_bytes_ += total;
    }
    return 0;
  }

  /**
   * LARGE-value deferred put: values at or above kBatchChunk bypass the
   * accumulating batch entirely. Batching a multi-MB value degenerates into a
   * one-put batch that re-arms the chunk and copies UNDER the global
   * batch_mtx_ (serializing every submitting thread), the copy first-touch
   * faults a fresh SHM chunk per put, and MultiPutBlob-written blobs skip the
   * zero-IPC read mirror — measured 20x slower puts and 3x slower reads on
   * 3.7MB LMCache records. Instead the value ships as ONE private put and is
   * registered in the registry, so FIFO awaits, flow control, and per-key
   * pending checks all still hold.
   *
   * OWNERSHIP CAVEAT (differs from the batched small-value path): the private
   * put is zero-copy with a co-located runtime, so THERE the caller must keep
   * the source buffer(s) stable until this put is awaited (AwaitPendingPuts /
   * AwaitPutsUntilSpace / a same-key AsyncGetBlobDefer). Client mode stages
   * at submit as always, and buffers are free on return.
   *
   * @param runtime_extent When non-null and co-located, registered as the
   *        put's served extent for read-your-writes; null registers the key
   *        count-only (pending reads await instead of composing).
   * @param submit Submits the put; retried after awaiting the oldest deferred
   *        put whenever it returns an empty future (SHM exhaustion).
   */
  template <typename SubmitFn>
  int DeferPutLarge(const TagId &tag_id, const std::string &blob_name,
                    clio::run::u64 offset, clio::run::u64 size,
                    const char *runtime_extent, SubmitFn &&submit,
                    ctp::ipc::FullPtr<char> staging =
                        ctp::ipc::FullPtr<char>::GetNull(),
                    clio::run::u64 staging_size = 0) {
    DeferRegistry &reg = DeferRegistry::Get();
    reg.flush_client_.store(this, std::memory_order_release);
    auto fut = submit();
    while (fut.IsNull()) {
      if (!DeferAwaitOldest()) {
        return -2;
      }
      fut = submit();
    }
    clio::run::u64 key = DeferKeyHash(tag_id, blob_name);
    clio::run::u64 seq = reg.seq_gen_.fetch_add(1) + 1;
    const char *ext =
        CLIO_RUNTIME_MANAGER->IsRuntime() ? runtime_extent : nullptr;
    reg.KeyAdd(key, seq, ext, offset, size);
    DeferredPut rec;
    rec.fut_ = fut.template Cast<clio::run::Task>();
    rec.ents_.push_back(DeferredPut::Ent{key, seq, size});
    rec.staging_ = staging;
    rec.staging_size_ = staging_size;
    {
      std::lock_guard<std::mutex> lk(reg.mtx_);
      reg.fifo_.push_back(std::move(rec));
      reg.pending_count_.fetch_add(1, std::memory_order_relaxed);
      reg.inflight_bytes_ += size;
    }
    return 0;
  }

  /** Ensure the accumulating batch chunk has room for `size` more bytes,
   *  shipping the current batch and re-arming a fresh chunk as needed
   *  (awaiting oldest puts when shared memory is exhausted — those waits
   *  drop `lk`, which must hold batch_mtx_ on entry and holds it again on
   *  return). Does NOT bump batch_used_; the caller copies to the returned
   *  pointer and records its desc/ent first.
   *  @return destination inside the chunk, or nullptr when shared memory is
   *  exhausted with nothing left to await. */
  char *ReserveDeferBatchLocked(DeferRegistry &reg,
                                std::unique_lock<std::mutex> &lk,
                                clio::run::u64 size) {
    auto *ipc_manager = CLIO_CPU_IPC;
    if (reg.batch_chunk_.IsNull() || reg.batch_used_ + size > reg.batch_cap_ ||
        reg.batch_descs_.size() >= DeferRegistry::kBatchMax) {
      FlushDeferBatchLocked(reg);
      clio::run::u64 cap = size > DeferRegistry::kBatchChunk
                               ? size
                               : DeferRegistry::kBatchChunk;
      ctp::ipc::FullPtr<char> chunk = ipc_manager->AllocateBuffer(cap);
      while (chunk.IsNull()) {
        // Shared memory exhausted: waits must happen OUTSIDE batch_mtx_.
        lk.unlock();
        if (!DeferAwaitOldest()) {
          return nullptr;
        }
        lk.lock();
        if (!reg.batch_chunk_.IsNull() &&
            reg.batch_used_ + size <= reg.batch_cap_) {
          chunk = reg.batch_chunk_;  // another thread already re-armed
          break;
        }
        chunk = ipc_manager->AllocateBuffer(cap);
      }
      if (reg.batch_chunk_.ptr_ != chunk.ptr_) {
        reg.batch_chunk_ = chunk;
        reg.batch_cap_ = cap;
        reg.batch_used_ = 0;
      }
    }
    return reg.batch_chunk_.ptr_ + reg.batch_used_;
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

  /** Ship the accumulating deferred batch (if any). Caller holds batch_mtx_. */
  void FlushDeferBatchLocked(DeferRegistry &reg) {
    if (reg.batch_descs_.empty()) {
      if (!reg.batch_chunk_.IsNull()) {
        return;  // armed but empty chunk stays for the next put
      }
      return;
    }
    auto fut = AsyncMultiPutVectored(ctp::ipc::ShmPtr<>(reg.batch_chunk_.shm_),
                                     reg.batch_used_, reg.batch_descs_);
    DeferredPut rec;
    rec.fut_ = fut.template Cast<clio::run::Task>();
    rec.ents_ = std::move(reg.batch_ents_);
    {
      std::lock_guard<std::mutex> lk(reg.mtx_);
      reg.fifo_.push_back(std::move(rec));
    }
    reg.batch_descs_.clear();
    reg.batch_ents_.clear();
    reg.batch_chunk_ = ctp::ipc::FullPtr<char>();
    reg.batch_cap_ = 0;
    reg.batch_used_ = 0;
  }

  /** Flush the accumulating batch if it has entries (public entry). */
  void FlushDeferBatch() {
    DeferRegistry &reg = DeferRegistry::Get();
    std::lock_guard<std::mutex> lk(reg.batch_mtx_);
    FlushDeferBatchLocked(reg);
  }

  /** Await every deferred put targeting (tag_id, blob_name). FIFO order, so
   *  older unrelated puts ahead of them are completed too (harmless). */
  static void AwaitPendingPuts(const TagId &tag_id,
                               const std::string &blob_name) {
    DeferRegistry &reg = DeferRegistry::Get();
    if (reg.pending_count_.load(std::memory_order_relaxed) == 0) {
      return;  // nothing deferred anywhere -> no key can be pending
    }
    clio::run::u64 key = DeferKeyHash(tag_id, blob_name);
    while (true) {
      if (!reg.IsKeyPending(key)) {
        return;
      }
      if (!DeferAwaitOldest()) {
        // Nothing claimable, but the key is still accounted: another thread
        // is mid-Wait on its put. Spin-yield until that wait retires it.
        std::this_thread::yield();
      }
    }
  }

  /**
   * Read-after-write-consistent private-memory get: if a deferred put for this
   * blob is still in flight, Wait for it (them) FIRST, then read — the read
   * takes the same SHM fast path / RPC fallback as AsyncGetBlob.
   */
  clio::run::Future<GetBlobTask> AsyncGetBlobDefer(
      const TagId &tag_id, const std::string &blob_name, clio::run::u64 offset,
      clio::run::u64 size, char *priv_data, clio::run::u32 flags = 0,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic(),
      const Context &context = Context()) {
    // Read-your-writes WITHOUT blocking: if the latest pending deferred put
    // fully covers the requested range, copy straight from the in-flight
    // put's bytes and return an already-COMPLETE future (same synthesized-
    // task contract as the TryShmGet fast path). Only when a pending put
    // exists but cannot serve the range (partial overlap, data pointer
    // retired mid-reap) do we fall back to awaiting it.
    if (priv_data != nullptr && flags == 0) {
      DeferRegistry &reg = DeferRegistry::Get();
      if (reg.pending_count_.load(std::memory_order_relaxed) != 0) {
        clio::run::u64 key = DeferKeyHash(tag_id, blob_name);
        int served = reg.TryServe(key, offset, priv_data, size, nullptr);
        if (served > 0) {
          auto *ipc_manager = CLIO_CPU_IPC;
          auto task = ipc_manager->NewTask<GetBlobTask>(
              clio::run::CreateTaskId(), pool_id_, pool_query, tag_id,
              blob_name, offset, size, flags, ctp::ipc::ShmPtr<>(), context);
          auto fut = clio::run::Future<GetBlobTask>(task->pool_id_,
                                                    task->method_, task);
          fut.GetFutureShm()->origin_ = clio::run::ClientOrigin::kClientShm;
          task->return_code_ = 0;
          task->SetComplete();
          return fut;
        }
        if (served < 0) {
          AwaitPendingPuts(tag_id, blob_name);
        }
      }
    } else {
      AwaitPendingPuts(tag_id, blob_name);
    }
    return AsyncGetBlob(tag_id, blob_name, offset, size, flags, priv_data,
                        pool_query, context);
  }

  /**
   * Serve a whole-value read of (tag, blob) straight from the LATEST pending
   * deferred put, without waiting for it. Intended for callers whose puts are
   * whole-blob overwrites (the deferred pipeline's primary use): when the
   * pending put covers `offset`, its bytes from `offset` to its end ARE the
   * current value tail.
   * @return >0: bytes copied into dst; 0: no pending put for this blob;
   *         -1: pending put cannot serve (range/retired) — caller should
   *             AwaitPendingPuts then read normally; -2: dst too small.
   */
  long long TryGetPendingPut(const TagId &tag_id, const std::string &blob_name,
                             clio::run::u64 offset, char *dst,
                             clio::run::u64 dst_cap) {
    DeferRegistry &reg = DeferRegistry::Get();
    if (reg.pending_count_.load(std::memory_order_relaxed) == 0) {
      return 0;
    }
    clio::run::u64 key = DeferKeyHash(tag_id, blob_name);
    DeferRegistry::Shard &sh = reg.ShardFor(key);
    std::lock_guard<std::mutex> lk(sh.mtx_);
    auto it = sh.per_key_.find(key);
    if (it == sh.per_key_.end()) return 0;
    const auto &ex = it->second.extents_;
    // Whole-value semantics need the NEWEST put to define the value; serve
    // its tail from `offset`. (Composing a whole value under newer partial
    // overwrites would need the base blob too — that corner falls back.)
    if (ex.empty()) return -1;
    const auto &e = ex.back();
    if (e.data_ == nullptr || offset < e.offset_ ||
        offset >= e.offset_ + e.size_) {
      return -1;
    }
    clio::run::u64 tail = e.offset_ + e.size_ - offset;
    if (tail > dst_cap) return -2;
    std::memcpy(dst, e.data_ + (offset - e.offset_), tail);
    return static_cast<long long>(tail);
  }

  /** True iff a deferred put for (tag, blob) is still pending. */
  static bool HasPendingPut(const TagId &tag_id,
                            const std::string &blob_name) {
    DeferRegistry &reg = DeferRegistry::Get();
    if (reg.pending_count_.load(std::memory_order_relaxed) == 0) {
      return false;
    }
    return reg.IsKeyPending(DeferKeyHash(tag_id, blob_name));
  }

  /**
   * Await oldest deferred puts until at most `max_inflight_bytes` of payload
   * remain in flight. 0 = full drain. @return in-flight bytes on return.
   */
  static clio::run::u64 AwaitPutsUntilSpace(clio::run::u64 max_inflight_bytes) {
    DeferRegistry &reg = DeferRegistry::Get();
    while (true) {
      {
        std::lock_guard<std::mutex> lk(reg.mtx_);
        if (reg.inflight_bytes_ <= max_inflight_bytes) {
          return reg.inflight_bytes_;
        }
      }
      if (!DeferAwaitOldest()) {
        // In-flight bytes are retired only when their Wait completes; if
        // other threads hold the remaining claims, yield until they finish.
        std::this_thread::yield();
      }
    }
  }

  /** Deferred puts that completed with a nonzero return code (sticky). */
  static clio::run::u64 DeferErrorCount() {
    return DeferRegistry::Get().errors_.load();
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
  /**
   * The zero-IPC read fast path, NATIVE to AsyncGetBlob (issues #783/#817).
   *
   * If the whole get can be served from the shared metadata cache + RAM-bdev
   * segment, copy it into `dst` and set *fut to an already-COMPLETE future:
   * a real GetBlobTask (never Sent) with return_code_==0 and IsComplete()
   * set, so Wait() returns instantly and the task is safe to dereference —
   * every caller gets the optimization with no special-casing, exactly like
   * the PutBlob path shapes.
   *
   * TryReadBlobShm carries its own guards (cache attached and ready, blob
   * RAM-resident and direct-readable, placement generation unchanged across
   * the copy); any miss returns false and the caller Sends the RPC task, so
   * this is only ever faster, never wrong for a settled blob. Semantics note:
   * the mirror is republished AFTER the authoritative update, so a reader
   * racing its OWN just-completed rewrite of the same bytes can briefly see
   * the previous value (clio-fs drains overlapping writes first for exactly
   * this reason). Gated to flags==0 so flagged gets keep full RPC semantics.
   * Attaches lazily: a client can come up before its pool is composed, and a
   * one-shot attach at init would pin that process to the RPC path forever.
   *
   * @param dst            where the bytes land (shared OR private memory)
   * @param task_blob_data blob_data_ recorded on the synthesized task (the
   *                       destination ShmPtr for the shared overload; null
   *                       for the private overload — PostWait is a no-op)
   */
  /** True when CLIO_FORCE_NET is set (force every op through the net path —
   * the client-side read fast paths must stand down so the force_net test
   * suites keep testing what they claim to). Read once. */
  static bool ForceNetEnv() {
    static const bool v = [] {
      const char *e = std::getenv("CLIO_FORCE_NET");
      return e != nullptr && e[0] != '\0' && !(e[0] == '0' && e[1] == '\0');
    }();
    return v;
  }

  /** True when CLIO_CTE_NO_PRIV_PUT is set: the private-memory AsyncPutBlob
   * stands down its runtime-mode zero-copy branch (null-allocator ShmPtr over
   * the caller's buffer) and every put goes through the plain staged path —
   * allocate SHM, copy once, send — exactly like a pure client. Benchmarking
   * knob (issue #862): isolates what the #830 zero-copy path buys co-located
   * writers. Read once. */
  static bool NoPrivPutEnv() {
    static const bool v = [] {
      const char *e = std::getenv("CLIO_CTE_NO_PRIV_PUT");
      return e != nullptr && e[0] != '\0' && !(e[0] == '0' && e[1] == '\0');
    }();
    return v;
  }

  bool TryShmGet(const TagId &tag_id, const char *blob_name,
                 clio::run::u64 offset, clio::run::u64 size,
                 clio::run::u32 flags, char *dst,
                 ctp::ipc::ShmPtr<> task_blob_data,
                 const clio::run::PoolQuery &pool_query,
                 const Context &context,
                 clio::run::Future<GetBlobTask> *fut) {
    // Emulated gets must reach the runtime (they model I/O, not perform it),
    // and flagged gets keep full RPC semantics. CLIO_FORCE_NET exists to push
    // every op through the network path (the force_net test suites); serving
    // reads client-side would silently turn those suites into no-ops.
    // Replica-targeted reads (issue #886, context.replica_ != 0) must also
    // reach the runtime: the SHM mirror publishes the PRIMARY's block layout
    // only, so serving them here would silently return primary bytes.
    if (dst == nullptr || size == 0 || flags != 0 || context.emulate_ ||
        context.replica_ != 0 || ForceNetEnv()) {
      return false;
    }
    if (!HasShmCache() && !AttachShmCache()) {
      return false;
    }
    if (!TryReadBlobShm(tag_id, blob_name, dst, size, offset)) {
      return false;
    }
    auto *ipc_manager = CLIO_CPU_IPC;
    auto task = ipc_manager->NewTask<GetBlobTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, tag_id, blob_name,
        offset, size, flags, task_blob_data, context);
    *fut = clio::run::Future<GetBlobTask>(task->pool_id_, task->method_, task);
    fut->GetFutureShm()->origin_ = clio::run::ClientOrigin::kClientShm;
    task->return_code_ = 0;
    task->SetComplete();
    return true;
  }

  clio::run::Future<GetBlobTask> AsyncGetBlob(
      const TagId &tag_id,
      const char *blob_name,
      clio::run::u64 offset, clio::run::u64 size,
      clio::run::u32 flags,
      ctp::ipc::ShmPtr<> blob_data,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic(),
      const Context &context = Context()) {
    auto *ipc_manager = CLIO_CPU_IPC;

    // DEFAULT-ON zero-IPC read — see TryShmGet above.
    {
      char *dst = blob_data.IsNull()
                      ? nullptr
                      : ipc_manager->ToFullPtr<char>(
                            blob_data.template Cast<char>()).ptr_;
      clio::run::Future<GetBlobTask> fut;
      if (TryShmGet(tag_id, blob_name, offset, size, flags, dst, blob_data,
                    pool_query, context, &fut)) {
        return fut;
      }
    }

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
   * Three paths, fastest first:
   *  - Shared-cache hit (TryShmGet): the bytes are copied out of the RAM
   *    bdev's shared segment with ZERO IPC and an already-COMPLETE Future is
   *    returned (real task, rc==0) — Wait() returns immediately and the task
   *    is safe to dereference, same contract as every other path.
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
   *         on every path (GetReturnCode()==0 on success), including the
   *         cache-hit path. The ONLY empty Future is the client-mode
   *         staging-allocation failure.
   */
  clio::run::Future<GetBlobTask> AsyncGetBlob(
      const TagId &tag_id, const std::string &blob_name,
      clio::run::u64 offset, clio::run::u64 size, clio::run::u32 flags,
      char *priv_data,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic(),
      const Context &context = Context()) {
    auto *ipc_manager = CLIO_CPU_IPC;

    // Fastest path: node-local RAM-resident blob → copy straight out of shared
    // memory (see TryShmGet). The synthesized task's blob_data_ stays null and
    // priv_dest_/priv_src_ default null, so PostWait and ~GetBlobTask are
    // no-ops — the bytes are already in the caller's buffer.
    {
      clio::run::Future<GetBlobTask> fut;
      if (TryShmGet(tag_id, blob_name.c_str(), offset, size, flags, priv_data,
                    ctp::ipc::ShmPtr<>::GetNull(), pool_query, context,
                    &fut)) {
        return fut;
      }
    }

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

    // Zero-IPC fast path, ALL-OR-NOTHING: if every segment can be served from
    // the shared cache, return a synthesized complete future (same contract as
    // the scalar TryShmGet). A partial hit falls through to the RPC, which
    // simply overwrites any segments already copied — correct either way.
    if (!segments.empty() && flags == 0 && !context.emulate_ &&
        !ForceNetEnv() && (HasShmCache() || AttachShmCache())) {
      bool all = true;
      for (const auto &seg : segments) {
        char *dst = seg.data_.IsNull()
                        ? nullptr
                        : ipc_manager->ToFullPtr<char>(
                              seg.data_.template Cast<char>()).ptr_;
        if (dst == nullptr || seg.size_ == 0 ||
            !TryReadBlobShm(tag_id, blob_name, dst, seg.size_,
                            seg.blob_off_)) {
          all = false;
          break;
        }
      }
      if (all) {
        auto task = ipc_manager->NewTask<GetBlobTask>(
            clio::run::CreateTaskId(), pool_id_, pool_query, tag_id, blob_name,
            static_cast<clio::run::u64>(0), static_cast<clio::run::u64>(0),
            flags, ctp::ipc::ShmPtr<>::GetNull(), context);
        clio::run::Future<GetBlobTask> fut(task->pool_id_, task->method_,
                                           task);
        fut.GetFutureShm()->origin_ = clio::run::ClientOrigin::kClientShm;
        task->return_code_ = 0;
        task->SetComplete();
        return fut;
      }
    }

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
   * Three paths, fastest first (mirrors the scalar private get):
   *  - Shared-cache hit, ALL-OR-NOTHING: every segment is copied out of the
   *    RAM bdev's shared segment with zero IPC and an already-COMPLETE future
   *    is returned (real task, rc==0, dereferenceable).
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

    // Zero-IPC fast path (all-or-nothing across segments).
    if (!segments.empty() && flags == 0 && !context.emulate_ &&
        !ForceNetEnv() && (HasShmCache() || AttachShmCache())) {
      bool all = true;
      for (const auto &seg : segments) {
        if (seg.data_ == nullptr || seg.size_ == 0 ||
            !TryReadBlobShm(tag_id, blob_name, seg.data_, seg.size_,
                            seg.blob_off_)) {
          all = false;
          break;
        }
      }
      if (all) {
        auto task = ipc_manager->NewTask<GetBlobTask>(
            clio::run::CreateTaskId(), pool_id_, pool_query, tag_id,
            blob_name.c_str(), static_cast<clio::run::u64>(0),
            static_cast<clio::run::u64>(0), flags,
            ctp::ipc::ShmPtr<>::GetNull(), context);
        clio::run::Future<GetBlobTask> fut(task->pool_id_, task->method_,
                                           task);
        fut.GetFutureShm()->origin_ = clio::run::ClientOrigin::kClientShm;
        task->return_code_ = 0;
        task->SetComplete();
        return fut;
      }
    }

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
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Broadcast()) {
    auto *ipc_manager = CLIO_CPU_IPC;
    auto task = ipc_manager->NewTask<EvictTask>(clio::run::CreateTaskId(),
                                                pool_id_, pool_query,
                                                min_tier_score, bytes);
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
  // True only when the mirror was attached via AttachShmCacheOf for a pool
  // OTHER than this client's own (interposition binding): gates the
  // serving-replica fast path (see AttachShmCacheOf).
  bool shm_replica_serving_ = false;

  /** One attached RAM-bdev segment, cached so a hot read does not re-attach. */
  struct RamBdevMap {
    ctp::ipc::PosixShmMmap backend;
    char *base = nullptr;  /**< byte 0 of device space */
  };
  // Keyed by target pool. Attaching is not free, and the fast path is meant to
  // be a few hundred nanoseconds, so a miss here would dominate the cost.
  std::unordered_map<clio::run::u64, RamBdevMap> ram_bdevs_;
  /**
   * Guards ram_bdevs_ (issue #817).
   *
   * The process-wide CTE client is shared by every thread of an application,
   * and the POSIX/STDIO interceptors hand it arbitrary multi-threaded callers,
   * so two threads racing a first read of different targets would otherwise
   * rehash the map concurrently -- a use-after-free, i.e. a segfault, on the
   * hot path. Shared for the steady state (every read after the first),
   * exclusive only to attach.
   *
   * A free function rather than a member because Client must stay
   * move-assignable (`cte_ = Client(pool_id)` in the filesystem chimod, and the
   * generated lib_exec), which a std::shared_mutex member would delete. One
   * lock covering every instance costs nothing: it is contended only on the
   * first read of a target in a process.
   */
  static std::shared_mutex &RamBdevMutex() {
    static std::shared_mutex mu;
    return mu;
  }

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
    {
      // Steady state: the segment is already attached, so this is a shared
      // lock and a hash lookup.
      std::shared_lock<std::shared_mutex> rd(RamBdevMutex());
      auto it = ram_bdevs_.find(key);
      if (it != ram_bdevs_.end()) {
        return it->second.base;
      }
    }
    std::unique_lock<std::shared_mutex> wr(RamBdevMutex());
    // Re-check: another thread may have attached between the two locks.
    auto found = ram_bdevs_.find(key);
    if (found != ram_bdevs_.end()) {
      return found->second.base;
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
   * CoreClient::AsyncGetBlob(char*) — see there for the three paths (shared
   * cache / runtime-direct / client-staging).
   *
   * @param blob_name Name of the blob to retrieve
   * @param data Output PRIVATE buffer (pre-allocated by caller, >= data_size)
   * @param data_size Size of data to retrieve (must be > 0)
   * @param off Offset within blob (default 0)
   * @return Future over the read. On the shared-cache path the Future is EMPTY
   *         (Wait() returns immediately; do not dereference it); otherwise the
   *         task is readable after Wait() (GetReturnCode()==0 on success). The
   *         buffer holds the bytes once Wait() returns.
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
