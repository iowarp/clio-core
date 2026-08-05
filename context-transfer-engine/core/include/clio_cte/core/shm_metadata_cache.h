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

#ifndef CLIO_CTE_CORE_SHM_METADATA_CACHE_H_
#define CLIO_CTE_CORE_SHM_METADATA_CACHE_H_

#include <clio_ctp/data_structures/ipc/string.h>
#include <clio_ctp/data_structures/ipc/unordered_map.h>

#include "clio_cte/core/blob_transform.h"
#include "clio_runtime/clio_runtime.h"
#include "clio_runtime/types.h"

namespace clio::cte::core {

/** Allocator backing the metadata segment (same type as the main segment). */
using ShmCacheAlloc = CLIO_TASK_ALLOC_T;
using ShmCacheString = ctp::ipc::string<ShmCacheAlloc>;

/**
 * Maximum blocks stored inline in a cached blob record.
 *
 * FIXED, because capping the block list keeps the record a plain POD with no
 * nested allocation and no second pointer chase for a lock-free reader.
 *
 * SIZED AGAINST THE ALLOCATOR, NOT AGAINST "small blobs" (issue #817). The
 * CTE caps every physical block at kMaxBlockChunk = 64 KB (core_runtime.cc,
 * ExtendBlob) so that freed extents can be reused under fragmentation, which
 * means block count scales with blob SIZE: a 1 MiB blob is 16 blocks, not one.
 * At the original value of 8 every blob above 512 KB was marked truncated and
 * refused -- including every clio-fs page, since kFsPageSize is exactly 1 MiB.
 * 16 covers a full page.
 *
 * The cost is resident: each block descriptor is 32 B, so this is +256 B per
 * cached blob (the table is preallocated, see ShmMetadataCacheRoot). Raising it
 * further to chase larger blobs is the wrong trade -- the round-trip it saves
 * is already noise next to the memcpy at that size.
 */
static constexpr clio::run::u32 kMaxInlineBlocks = 16;

/**
 * One block of a cached blob.
 *
 * POD ONLY. Note what is absent: the runtime's BlobBlock embeds a
 * clio::run::bdev::Client, which derives from ContainerClient and therefore
 * carries a VTABLE POINTER -- a process-local address that is meaningless in a
 * segment mapped by another process. The client is reconstructed runtime-side
 * from target_pool_ when needed. General rule: no virtual functions in any
 * SHM-resident type.
 */
struct ShmBlockDesc {
  clio::run::PoolId target_pool_;  /**< bdev pool holding this block */
  clio::run::u64 target_offset_;   /**< offset within the target */
  clio::run::u64 size_;            /**< logical bytes used */
  clio::run::u32 bdev_type_;       /**< clio::run::bdev::BdevType */
  clio::run::u32 node_id_;         /**< owning node; only local is cacheable */
};

/** Flags on a cached blob record. */
enum ShmBlobFlags : clio::run::u32 {
  /** Every block is node-local AND RAM-backed, so the payload is directly
   *  readable from shared memory. Without this the client must use RPC. */
  kShmBlobDirectReadable = 1u << 0,
  /** The blob had more blocks than kMaxInlineBlocks, so blocks_ describes only
   *  a PREFIX of it. Reads bounded by CoveredBytes() are still served; reads
   *  past the prefix must fall back to RPC. */
  kShmBlobTruncated = 1u << 1,
  /** The stored bytes are transformed (compressed/encrypted/...), so copying
   *  them out verbatim would hand the caller codec bytes. Mirrors
   *  BlobInfo::IsTransformed(); see transform_flags_ below. */
  kShmBlobTransformed = 1u << 2,
};

/**
 * Cached blob metadata.
 *
 * Trivially copyable by construction: the map's reader copies the whole record
 * out between two reads of the slot generation, so it must contain no
 * allocator-owned members (an ipc::string here would make the record
 * non-copyable and defeat the seqlock). The blob NAME is the map key, so it is
 * not duplicated here.
 */
struct ShmBlobRecord {
  clio::run::u64 total_size_;
  clio::run::u64 last_modified_;
  clio::run::u64 last_read_;
  /**
   * Bumped by the runtime on every change to this blob's PLACEMENT. A client
   * must re-read this after copying the payload and discard the result if it
   * moved: the seqlock only protects the metadata copy, while the hazard
   * (design §5.3) is the DataOrganizer relocating blocks during the payload
   * read itself.
   */
  clio::run::u64 placement_gen_;
  float score_;
  clio::run::u32 flags_;
  clio::run::u32 num_blocks_;
  /**
   * Verbatim mirror of BlobInfo::transform_flags_ (issue #818) -- the
   * authoritative "these are not the caller's bytes" word, carried so a client
   * can tell WHY the fast path was refused and, later, undo the transform
   * itself. Occupies the slot formerly named `reserved_`, so the record's size
   * and layout are unchanged.
   *
   * Note total_size_ is the STORED size. For a transformed blob that is the
   * post-transform byte count, which is not the blob's logical length -- one
   * more reason a transformed record must never serve a direct read.
   */
  clio::run::u32 transform_flags_;
  ShmBlockDesc blocks_[kMaxInlineBlocks];

  /**
   * Serving-replica extension (issue #886 cache/replication split). When the
   * PRIMARY is refused (transformed, absent, or not direct-readable), a
   * client may serve the read from an UNTRANSFORMED node-local RAM-backed
   * replica instead — in practice the REPLICA_CACHE copy the cache chimod
   * maintains while the authoritative bytes live compressed further down
   * the chain. rep_direct_ follows the same default-refuse discipline as
   * kShmBlobDirectReadable: it is set only when EVERY published replica
   * block qualifies, and replicas whose layouts exceed the inline array are
   * not published at all (no truncated-prefix mode for replicas).
   * placement_gen_ covers replica layout changes too — replica writers and
   * the cache-reclaim path bump it before republishing.
   */
  clio::run::u32 rep_direct_;      // 1 = replica descriptor serves reads
  clio::run::u32 rep_num_blocks_;
  clio::run::u64 rep_total_size_;  // the replica copy's logical size
  ShmBlockDesc rep_blocks_[kMaxInlineBlocks];

  ShmBlobRecord()
      : total_size_(0),
        last_modified_(0),
        last_read_(0),
        placement_gen_(0),
        score_(0.0f),
        flags_(0),
        num_blocks_(0),
        transform_flags_(0),
        rep_direct_(0),
        rep_num_blocks_(0),
        rep_total_size_(0) {}

  /** Bytes readable from the published replica descriptor. */
  clio::run::u64 RepCoveredBytes() const {
    clio::run::u64 n = 0;
    for (clio::run::u32 i = 0; i < rep_num_blocks_ && i < kMaxInlineBlocks;
         ++i) {
      n += rep_blocks_[i].size_;
    }
    return n;
  }

  /** True when the replica descriptor may serve direct reads. */
  bool HasServableReplica() const {
    return rep_direct_ != 0 && rep_num_blocks_ > 0;
  }

  /** True if this blob's stored bytes have been rewritten by some transform. */
  bool IsTransformed() const {
    return transform_flags_ != 0 || (flags_ & kShmBlobTransformed) != 0;
  }

  /**
   * True if the payload may be read directly out of shared memory.
   *
   * A TRUNCATED record still qualifies (issue #817). The cached blocks are the
   * blob's first blocks in logical order, so they describe a known prefix
   * exactly; refusing the whole blob threw away complete information about the
   * part it did have. Callers must bound the read by CoveredBytes(), not by
   * total_size_ -- that is what keeps a truncated record safe.
   *
   * A TRANSFORMED record never qualifies (issue #818): its stored bytes are a
   * codec's output, so copying them out would succeed and return the wrong
   * thing. The transform check is deliberately redundant with the runtime
   * clearing kShmBlobDirectReadable when it builds the record: this is the
   * "default must be refuse" discipline the cache is built on, so a future
   * writer that forgets to clear the flag still cannot cause a client to hand
   * codec bytes back as data.
   */
  bool IsDirectReadable() const {
    return (flags_ & kShmBlobDirectReadable) != 0 && !IsTransformed() &&
           num_blocks_ > 0;
  }

  /**
   * Logical bytes described by the cached blocks: the length of the prefix of
   * this blob that can be read from shared memory. Equals total_size_ unless
   * the record is truncated.
   */
  clio::run::u64 CoveredBytes() const {
    clio::run::u64 n = 0;
    for (clio::run::u32 i = 0; i < num_blocks_ && i < kMaxInlineBlocks; ++i) {
      n += blocks_[i].size_;
    }
    return n;
  }
};

/** Cached tag metadata. POD, for the same reason as ShmBlobRecord. */
struct ShmTagRecord {
  clio::run::u64 total_size_;
  clio::run::u64 last_modified_;
  clio::run::u64 last_read_;
  clio::run::u64 last_changed_;

  ShmTagRecord()
      : total_size_(0), last_modified_(0), last_read_(0), last_changed_(0) {}
};

using ShmTagIdMap = ctp::ipc::unordered_map<ShmCacheString, clio::run::UniqueId,
                                            ShmCacheAlloc>;
using ShmTagInfoMap =
    ctp::ipc::unordered_map<clio::run::UniqueId, ShmTagRecord, ShmCacheAlloc>;
using ShmBlobInfoMap =
    ctp::ipc::unordered_map<ShmCacheString, ShmBlobRecord, ShmCacheAlloc>;

/**
 * Root of the CTE shared-memory metadata cache (issue #783).
 *
 * OWNERSHIP: created and written ONLY by the runtime. Clients attach it
 * read-mostly -- they map the segment read-write because taking a lease is a
 * store, but by convention they write nothing except lock words.
 *
 * AUTHORITY: this is a CACHE, never the source of truth. The runtime keeps its
 * own structures and may refuse to populate, defer updating, or drop this
 * wholesale at any moment. Two consequences the design leans on: a client that
 * corrupts it can only degrade other clients, and a wedged cache is recovered
 * by dropping it rather than being a data-loss event.
 *
 * SIZING IS PERMANENT. The maps never rehash (see ipc::unordered_map), because
 * a rehash would free a table out from under untracked cross-process readers.
 * Capacity is therefore chosen once, here, and a full table degrades to the RPC
 * path rather than growing.
 */
struct ShmMetadataCacheRoot {
  /** Layout version. A client that does not recognize it must not attach --
   *  the cache is derived state, so refusing is always safe. */
  // v2 (issue #817): kMaxInlineBlocks 8 -> 16, and a truncated record is now
  // readable up to CoveredBytes(). Both change the record layout/semantics, so
  // a v1 client must refuse rather than misread it.
  static constexpr clio::run::u32 kLayoutVersion = 3;

  clio::run::u32 version_;
  clio::run::u32 ready_;  /**< 0 until fully constructed; clients must check */
  ShmTagIdMap tag_name_to_id_;
  ShmTagInfoMap tag_id_to_info_;
  ShmBlobInfoMap blob_key_to_info_;

  ShmMetadataCacheRoot() : version_(0), ready_(0) {}
};

/**
 * Build the blob-map key.
 *
 * Format is "<tag_major>.<tag_minor>.<blob_name>", matching the composite key
 * the CTE runtime already builds for tag_blob_name_to_info_ EXACTLY. Using one
 * convention rather than two is what makes the dual-write trivially
 * consistent -- a second format would let the cache and the authoritative map
 * disagree about which blob a key names.
 *
 * Written into a caller-provided buffer so the CLIENT never allocates: clients
 * are readers, and allocating inside the runtime-owned segment is exactly what
 * they must not do. Returns the length, or 0 if the buffer is too small.
 */
inline size_t MakeShmBlobKey(const clio::run::UniqueId &tag_id,
                             const char *blob_name, size_t blob_name_len,
                             char *out, size_t out_cap) {
  auto write_u32 = [&](clio::run::u32 v, size_t &pos) -> bool {
    char tmp[12];
    int n = 0;
    if (v == 0) {
      tmp[n++] = '0';
    }
    while (v > 0) {
      tmp[n++] = static_cast<char>('0' + (v % 10));
      v /= 10;
    }
    if (pos + static_cast<size_t>(n) >= out_cap) {
      return false;
    }
    for (int i = n - 1; i >= 0; --i) {
      out[pos++] = tmp[i];
    }
    return true;
  };

  size_t pos = 0;
  if (!write_u32(tag_id.major_, pos)) {
    return 0;
  }
  if (pos + 1 >= out_cap) {
    return 0;
  }
  out[pos++] = '.';
  if (!write_u32(tag_id.minor_, pos)) {
    return 0;
  }
  if (pos + 1 >= out_cap) {
    return 0;
  }
  out[pos++] = '.';
  if (pos + blob_name_len >= out_cap) {
    return 0;
  }
  for (size_t i = 0; i < blob_name_len; ++i) {
    out[pos++] = blob_name[i];
  }
  out[pos] = '\0';
  return pos;
}

/**
 * Runtime-side owner of the shared-memory metadata cache (issue #783).
 *
 * ENTIRELY BEST-EFFORT. Every method is a no-op when the cache is disabled, and
 * every failure is swallowed. That is deliberate: the cache is derived state,
 * so the correct response to any problem is to stop caching, not to fail the
 * operation the caller was actually performing. A metadata write must never be
 * rejected because a cache mirror could not be updated.
 *
 * Not thread-safe on its own -- callers mirror under whatever synchronization
 * already guards the authoritative structure.
 */
class ShmMetadataCache {
 public:
  ShmMetadataCache() = default;

  /**
   * Construct the cache in the runtime's metadata segment.
   *
   * @return true if the cache is live. false simply means caching is off:
   *         no metadata segment (small host, attach failure), or the
   *         allocation did not fit. Callers must not treat this as an error.
   */
  bool Create(size_t tag_capacity, size_t blob_capacity,
              const clio::run::PoolId &owner_pool) {
#if !CTP_IS_HOST
    // The metadata cache lives in the runtime's host-side segment and is
    // constructed by the runtime. GetMetadataAllocator()/GetMetadataDirectory()
    // are host-only IpcManager members, absent in the device (nvcc) pass — so
    // this body does not compile there. Device code never owns or builds the
    // cache; report "caching off", which callers already treat as a valid,
    // non-error state (see the docstring above). Without this guard the whole
    // header failed the CUDA build (GPU Tests workflow).
    (void)tag_capacity;
    (void)blob_capacity;
    (void)owner_pool;
    return false;
#else
    auto *ipc = CLIO_IPC;
    if (ipc == nullptr) {
      return false;
    }
    alloc_ = ipc->GetMetadataAllocator();
    if (alloc_ == nullptr) {
      return false;  // segment unavailable -> caching disabled, not an error
    }
    try {
      auto fp = alloc_->template Allocate<ShmMetadataCacheRoot>(
          sizeof(ShmMetadataCacheRoot));
      if (fp.IsNull()) {
        alloc_ = nullptr;
        return false;
      }
      root_ = fp.ptr_;
      new (root_) ShmMetadataCacheRoot();
      new (&root_->tag_name_to_id_) ShmTagIdMap(alloc_, tag_capacity);
      new (&root_->tag_id_to_info_) ShmTagInfoMap(alloc_, tag_capacity);
      new (&root_->blob_key_to_info_) ShmBlobInfoMap(alloc_, blob_capacity);
      if (!root_->tag_name_to_id_.valid() || !root_->tag_id_to_info_.valid() ||
          !root_->blob_key_to_info_.valid()) {
        root_ = nullptr;
        alloc_ = nullptr;
        return false;
      }
      root_->version_ = ShmMetadataCacheRoot::kLayoutVersion;
      // ready_ LAST: a client that observes ready_ == 1 must be guaranteed the
      // maps behind it are fully constructed.
      root_->ready_ = 1;
      // Publish in the segment directory so ANY client can find the cache,
      // not just one that happened to trigger pool creation. Registered after
      // ready_ for the same ordering reason.
      // Register under THIS pool's id. CTE is multi-pool, so a single shared
      // slot would let the most recently created pool hijack every client.
      if (auto *dir = ipc->GetMetadataDirectory()) {
        dir->RegisterRoot(owner_pool.ToU64(),
                          static_cast<clio::run::u64>(RootOffset()));
      }
      return true;
    } catch (...) {
      root_ = nullptr;
      alloc_ = nullptr;
      return false;
    }
#endif  // CTP_IS_HOST
  }

  bool IsEnabled() const { return root_ != nullptr && alloc_ != nullptr; }

  /** Byte offset of the root within the metadata allocator, which is what a
   *  client needs in order to find the cache. Propagated via CreateTask in
   *  Phase 4. */
  size_t RootOffset() const {
    if (!IsEnabled()) {
      return 0;
    }
    return static_cast<size_t>(reinterpret_cast<const char *>(root_) -
                               reinterpret_cast<const char *>(alloc_));
  }

  /** Mirror one blob. `composite_key` MUST be the same key the authoritative
   *  map uses, so the two cannot disagree about identity. */
  void PutBlob(const std::string &composite_key, const ShmBlobRecord &rec) {
    if (!IsEnabled()) {
      return;
    }
    try {
      ShmBlobInfoMap::BytesProbe p{composite_key.data(), composite_key.size()};
      root_->blob_key_to_info_.InsertOrAssign(
          p,
          ShmCacheString::HashBytes(composite_key.data(), composite_key.size()),
          rec);
      // A false return means the table is full. Nothing to do: the blob is
      // simply not cached and clients take the RPC path for it.
    } catch (...) {
    }
  }

  void EraseBlob(const std::string &composite_key) {
    if (!IsEnabled()) {
      return;
    }
    try {
      ShmBlobInfoMap::BytesProbe p{composite_key.data(), composite_key.size()};
      root_->blob_key_to_info_.Erase(
          p, ShmCacheString::HashBytes(composite_key.data(),
                                       composite_key.size()));
    } catch (...) {
    }
  }

  void PutTagId(const std::string &tag_name, const clio::run::UniqueId &id) {
    if (!IsEnabled()) {
      return;
    }
    try {
      ShmTagIdMap::BytesProbe p{tag_name.data(), tag_name.size()};
      root_->tag_name_to_id_.InsertOrAssign(
          p, ShmCacheString::HashBytes(tag_name.data(), tag_name.size()), id);
    } catch (...) {
    }
  }

  void EraseTagId(const std::string &tag_name) {
    if (!IsEnabled()) {
      return;
    }
    try {
      ShmTagIdMap::BytesProbe p{tag_name.data(), tag_name.size()};
      root_->tag_name_to_id_.Erase(
          p, ShmCacheString::HashBytes(tag_name.data(), tag_name.size()));
    } catch (...) {
    }
  }

  void PutTagInfo(const clio::run::UniqueId &id, const ShmTagRecord &rec) {
    if (!IsEnabled()) {
      return;
    }
    try {
      root_->tag_id_to_info_.InsertOrAssign(id, ShmTagInfoMap::Hash(id), rec);
    } catch (...) {
    }
  }

  void EraseTagInfo(const clio::run::UniqueId &id) {
    if (!IsEnabled()) {
      return;
    }
    try {
      root_->tag_id_to_info_.Erase(id, ShmTagInfoMap::Hash(id));
    } catch (...) {
    }
  }

  ShmMetadataCacheRoot *root() { return root_; }

 private:
  ShmCacheAlloc *alloc_ = nullptr;
  ShmMetadataCacheRoot *root_ = nullptr;
};

}  // namespace clio::cte::core

#endif  // CLIO_CTE_CORE_SHM_METADATA_CACHE_H_
