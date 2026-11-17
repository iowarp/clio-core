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
 * cached blob record. Raising it
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
  clio::run::u64 content_seq_;  /**< in-place write seqlock; odd = writing */
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
        content_seq_(0),
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

}  // namespace clio::cte::core

#endif  // CLIO_CTE_CORE_SHM_METADATA_CACHE_H_
