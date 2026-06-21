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

#ifndef WRPCTE_CORE_GPU_METADATA_CACHE_H_
#define WRPCTE_CORE_GPU_METADATA_CACHE_H_

/**
 * GPU-accessible projection of the CTE Core metadata.
 *
 * Why this header exists:
 *   The CPU-side CTE metadata uses clio::run::priv::string and clio::run::priv::vector
 *   templated on CLIO_PRIV_ALLOC_T, whose layout differs between SYCL host
 *   and device passes (see context-transfer-engine/test/unit/gpu/
 *   test_sycl_cte_putblob.cc). That makes those types unusable from a
 *   single-source SYCL kernel. The GPU metadata cache here uses ONLY POD
 *   types so the same struct definitions are layout-identical in any
 *   compilation pass and can sit in managed/shared USM accessed by both
 *   the CPU runtime (via GPU kernels launched from PutBlob/etc.) and
 *   downstream GPU kernels.
 *
 * Concurrency model:
 *   The CTE Core server is the SOLE writer (PutBlob/DelBlob/etc. launch
 *   one-WI kernels that mutate the cache). Other GPU kernels are readers.
 *   Slot state transitions are: kEmpty -> kOccupied -> kTombstone
 *   (delete tombstones are kept to preserve open-addressing probe chains).
 *
 * Sizing:
 *   max_blobs_ / max_tags_ are fixed at Create time and embedded in the
 *   header. The blob / tag slot arrays follow contiguously in the same
 *   USM allocation; the layout is computed by Layout() so callers can
 *   carve a single GpuApi::MallocManaged region.
 */

#include <clio_runtime/types.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace clio::cte::core {

namespace gpu_cache {

/** Maximum length (including the null terminator) of a name in the cache. */
static constexpr clio::run::u32 kMaxName = 64;

/** Slot states (POD, replicated to host and device atomics-of-int32 below). */
enum SlotState : clio::run::u32 {
  kEmpty = 0,
  kOccupied = 1,
  kTombstone = 2,
};

/**
 * Tag slot entry.
 * tag_id_.major_/.minor_ form the lookup key. tag_name_ is for debugging
 * and to let GPU readers identify the tag by name.
 */
struct GpuTagEntry {
  clio::run::u32 state_;        /**< SlotState */
  clio::run::u32 tag_major_;    /**< TagId.major_ */
  clio::run::u32 tag_minor_;    /**< TagId.minor_ */
  clio::run::u32 _pad0_;
  char tag_name_[kMaxName];

  CTP_CROSS_FUN void Reset() {
    state_ = kEmpty;
    tag_major_ = 0;
    tag_minor_ = 0;
    _pad0_ = 0;
    tag_name_[0] = '\0';
  }
};

/**
 * Blob slot entry.
 * Lookup key is (tag_major_, tag_minor_, blob_name_). size_ records the
 * blob's logical size; storage_class_ encodes which CTE tier the blob is
 * resident on (matches the bdev-type strings — see core_config.h).
 */
struct GpuBlobEntry {
  clio::run::u32 state_;          /**< SlotState */
  clio::run::u32 tag_major_;      /**< Owning TagId.major_ */
  clio::run::u32 tag_minor_;      /**< Owning TagId.minor_ */
  clio::run::u32 storage_class_;  /**< StorageClass below */
  clio::run::u64 size_;           /**< Blob logical size in bytes */
  float score_;             /**< Placement score 0-1 */
  clio::run::u32 _pad0_;
  char blob_name_[kMaxName];

  CTP_CROSS_FUN void Reset() {
    state_ = kEmpty;
    tag_major_ = 0;
    tag_minor_ = 0;
    storage_class_ = 0;
    size_ = 0;
    score_ = 0.0f;
    _pad0_ = 0;
    blob_name_[0] = '\0';
  }
};

/**
 * Storage class for a cached blob — derived from the bdev_type the
 * server placed it on. Only "GPU-reachable" classes (Ram / Hbm /
 * Pinned) are added to the cache; File / Noop blobs are not eligible
 * for projection (they aren't directly accessible from device code
 * via UVA / managed memory).
 */
enum StorageClass : clio::run::u32 {
  kStorageUnknown = 0,
  kStorageRam = 1,      /**< host RAM bdev (DRAM)            */
  kStorageHbm = 2,      /**< on-device HBM bdev              */
  kStoragePinned = 3,   /**< pinned / mapped host memory     */
};

/** Whether a storage class is eligible for the GPU cache. */
CTP_CROSS_FUN inline bool IsGpuVisible(clio::run::u32 sc) {
  return sc == kStorageRam || sc == kStorageHbm || sc == kStoragePinned;
}

/** Map a bdev-type string to a StorageClass. Used on the host. */
inline clio::run::u32 BdevTypeToStorageClass(const char *bdev_type) {
  if (!bdev_type) return kStorageUnknown;
  if (std::strcmp(bdev_type, "ram") == 0) return kStorageRam;
  if (std::strcmp(bdev_type, "hbm") == 0) return kStorageHbm;
  if (std::strcmp(bdev_type, "pinned") == 0) return kStoragePinned;
  return kStorageUnknown;
}

/** Copy at most kMaxName-1 bytes of src into dst, NUL-terminated. */
CTP_CROSS_FUN inline void CopyName(char *dst, const char *src) {
  clio::run::u32 i = 0;
  if (src) {
    for (; i + 1 < kMaxName && src[i] != '\0'; ++i) dst[i] = src[i];
  }
  dst[i] = '\0';
}

/** GPU-safe length (no libc dependency). */
CTP_CROSS_FUN inline clio::run::u32 NameLen(const char *s) {
  clio::run::u32 n = 0;
  while (n < kMaxName && s[n] != '\0') ++n;
  return n;
}

/** GPU-safe equality (bounded). */
CTP_CROSS_FUN inline bool NameEq(const char *a, const char *b) {
  for (clio::run::u32 i = 0; i < kMaxName; ++i) {
    if (a[i] != b[i]) return false;
    if (a[i] == '\0') return true;
  }
  return true;
}

/**
 * FNV-1a 64-bit string hash, GPU-safe.
 * Used for both blob and tag name keys. Combined with the tag id for
 * blob entries.
 */
CTP_CROSS_FUN inline clio::run::u64 FnvHashString(const char *s) {
  clio::run::u64 h = 0xcbf29ce484222325ULL;
  for (clio::run::u32 i = 0; i < kMaxName && s[i] != '\0'; ++i) {
    h ^= static_cast<clio::run::u64>(static_cast<unsigned char>(s[i]));
    h *= 0x100000001b3ULL;
  }
  return h;
}

/** Hash a (tag, blob_name) tuple. */
CTP_CROSS_FUN inline clio::run::u64 HashBlobKey(clio::run::u32 tag_major, clio::run::u32 tag_minor,
                                             const char *blob_name) {
  clio::run::u64 h = FnvHashString(blob_name);
  h ^= (static_cast<clio::run::u64>(tag_major) << 32) | tag_minor;
  h *= 0x100000001b3ULL;
  return h;
}

/** Hash a tag (major, minor) key. */
CTP_CROSS_FUN inline clio::run::u64 HashTagKey(clio::run::u32 tag_major, clio::run::u32 tag_minor) {
  clio::run::u64 h = (static_cast<clio::run::u64>(tag_major) << 32) | tag_minor;
  h ^= 0xcbf29ce484222325ULL;
  h *= 0x100000001b3ULL;
  return h;
}

}  // namespace gpu_cache

/**
 * Header for the GPU metadata cache region.
 *
 * The full memory layout is:
 *
 *   [GpuMetadataCacheHeader]                       (sizeof header)
 *   [GpuTagEntry  tags_[max_tags_]]                (max_tags_ * sizeof tag)
 *   [GpuBlobEntry blobs_[max_blobs_]]              (max_blobs_ * sizeof blob)
 *
 * The arrays are reachable via TagSlots() / BlobSlots(); pointers are
 * computed relative to `this` so the same handle works on both host and
 * device when the region lives in shared USM.
 */
struct GpuMetadataCacheHeader {
  clio::run::u32 magic_;           /**< 0xCAFEC7E0 — quick sanity check */
  clio::run::u32 version_;         /**< Layout version */
  clio::run::u32 max_tags_;        /**< Tag-slot capacity (open-addressing) */
  clio::run::u32 max_blobs_;       /**< Blob-slot capacity (open-addressing) */
  clio::run::u64 region_bytes_;    /**< Total bytes of the USM region */
  clio::run::u32 num_tags_;        /**< Currently populated tag entries */
  clio::run::u32 num_blobs_;       /**< Currently populated blob entries */
  clio::run::u32 _pad0_;
  clio::run::u32 _pad1_;

  static constexpr clio::run::u32 kMagic = 0xCAFEC7E0u;
  static constexpr clio::run::u32 kVersion = 1u;

  /**
   * Compute total region size for a given map capacity.
   * @param max_tags_in tag-slot count
   * @param max_blobs_in blob-slot count
   */
  static size_t Layout(clio::run::u32 max_tags_in, clio::run::u32 max_blobs_in) {
    return sizeof(GpuMetadataCacheHeader) +
           static_cast<size_t>(max_tags_in) * sizeof(gpu_cache::GpuTagEntry) +
           static_cast<size_t>(max_blobs_in) * sizeof(gpu_cache::GpuBlobEntry);
  }

  /** Pointer to the tag slot array. */
  CTP_CROSS_FUN gpu_cache::GpuTagEntry *TagSlots() {
    return reinterpret_cast<gpu_cache::GpuTagEntry *>(
        reinterpret_cast<char *>(this) + sizeof(GpuMetadataCacheHeader));
  }
  CTP_CROSS_FUN const gpu_cache::GpuTagEntry *TagSlots() const {
    return reinterpret_cast<const gpu_cache::GpuTagEntry *>(
        reinterpret_cast<const char *>(this) + sizeof(GpuMetadataCacheHeader));
  }

  /** Pointer to the blob slot array. */
  CTP_CROSS_FUN gpu_cache::GpuBlobEntry *BlobSlots() {
    return reinterpret_cast<gpu_cache::GpuBlobEntry *>(
        reinterpret_cast<char *>(TagSlots()) +
        static_cast<size_t>(max_tags_) * sizeof(gpu_cache::GpuTagEntry));
  }
  CTP_CROSS_FUN const gpu_cache::GpuBlobEntry *BlobSlots() const {
    return reinterpret_cast<const gpu_cache::GpuBlobEntry *>(
        reinterpret_cast<const char *>(TagSlots()) +
        static_cast<size_t>(max_tags_) * sizeof(gpu_cache::GpuTagEntry));
  }

  /**
   * Initialize an empty cache header in-place. Caller is responsible
   * for having allocated at least Layout(max_tags_in, max_blobs_in)
   * bytes at `this`.
   */
  void Init(clio::run::u32 max_tags_in, clio::run::u32 max_blobs_in,
            clio::run::u64 region_bytes) {
    magic_ = kMagic;
    version_ = kVersion;
    max_tags_ = max_tags_in;
    max_blobs_ = max_blobs_in;
    region_bytes_ = region_bytes;
    num_tags_ = 0;
    num_blobs_ = 0;
    _pad0_ = 0;
    _pad1_ = 0;
    auto *t = TagSlots();
    for (clio::run::u32 i = 0; i < max_tags_; ++i) t[i].Reset();
    auto *b = BlobSlots();
    for (clio::run::u32 i = 0; i < max_blobs_; ++i) b[i].Reset();
  }

  CTP_CROSS_FUN bool ValidMagic() const {
    return magic_ == kMagic && version_ == kVersion;
  }
};

/**
 * Find or insert a tag entry. Returns a pointer to the slot used, or
 * nullptr if the table is full (the caller should treat that as a
 * cache miss, NOT a hard error — the cache is best-effort).
 *
 * Open-addressing with linear probing. Tombstones are reused for
 * inserts but probe walks continue past them.
 */
CTP_CROSS_FUN inline gpu_cache::GpuTagEntry *GpuCacheUpsertTag(
    GpuMetadataCacheHeader *hdr, clio::run::u32 tag_major, clio::run::u32 tag_minor,
    const char *tag_name) {
  if (!hdr || hdr->max_tags_ == 0) return nullptr;
  clio::run::u64 h = gpu_cache::HashTagKey(tag_major, tag_minor);
  clio::run::u32 cap = hdr->max_tags_;
  auto *slots = hdr->TagSlots();
  gpu_cache::GpuTagEntry *first_tomb = nullptr;
  for (clio::run::u32 i = 0; i < cap; ++i) {
    clio::run::u32 idx = static_cast<clio::run::u32>((h + i) % cap);
    auto &s = slots[idx];
    if (s.state_ == gpu_cache::kEmpty) {
      auto *target = first_tomb ? first_tomb : &s;
      target->state_ = gpu_cache::kOccupied;
      target->tag_major_ = tag_major;
      target->tag_minor_ = tag_minor;
      gpu_cache::CopyName(target->tag_name_, tag_name);
      hdr->num_tags_ += 1;
      return target;
    }
    if (s.state_ == gpu_cache::kTombstone) {
      if (!first_tomb) first_tomb = &s;
      continue;
    }
    if (s.tag_major_ == tag_major && s.tag_minor_ == tag_minor) {
      gpu_cache::CopyName(s.tag_name_, tag_name);
      return &s;
    }
  }
  if (first_tomb) {
    first_tomb->state_ = gpu_cache::kOccupied;
    first_tomb->tag_major_ = tag_major;
    first_tomb->tag_minor_ = tag_minor;
    gpu_cache::CopyName(first_tomb->tag_name_, tag_name);
    hdr->num_tags_ += 1;
    return first_tomb;
  }
  return nullptr;
}

/**
 * Find an existing blob entry, or claim a free slot for insertion.
 * Returns nullptr if the table is full.
 */
CTP_CROSS_FUN inline gpu_cache::GpuBlobEntry *GpuCacheUpsertBlob(
    GpuMetadataCacheHeader *hdr, clio::run::u32 tag_major, clio::run::u32 tag_minor,
    const char *blob_name, clio::run::u64 size, float score, clio::run::u32 storage_class) {
  if (!hdr || hdr->max_blobs_ == 0) return nullptr;
  clio::run::u64 h = gpu_cache::HashBlobKey(tag_major, tag_minor, blob_name);
  clio::run::u32 cap = hdr->max_blobs_;
  auto *slots = hdr->BlobSlots();
  gpu_cache::GpuBlobEntry *first_tomb = nullptr;
  for (clio::run::u32 i = 0; i < cap; ++i) {
    clio::run::u32 idx = static_cast<clio::run::u32>((h + i) % cap);
    auto &s = slots[idx];
    if (s.state_ == gpu_cache::kEmpty) {
      auto *target = first_tomb ? first_tomb : &s;
      target->state_ = gpu_cache::kOccupied;
      target->tag_major_ = tag_major;
      target->tag_minor_ = tag_minor;
      target->size_ = size;
      target->score_ = score;
      target->storage_class_ = storage_class;
      gpu_cache::CopyName(target->blob_name_, blob_name);
      hdr->num_blobs_ += 1;
      return target;
    }
    if (s.state_ == gpu_cache::kTombstone) {
      if (!first_tomb) first_tomb = &s;
      continue;
    }
    if (s.tag_major_ == tag_major && s.tag_minor_ == tag_minor &&
        gpu_cache::NameEq(s.blob_name_, blob_name)) {
      s.size_ = size;
      s.score_ = score;
      s.storage_class_ = storage_class;
      return &s;
    }
  }
  if (first_tomb) {
    first_tomb->state_ = gpu_cache::kOccupied;
    first_tomb->tag_major_ = tag_major;
    first_tomb->tag_minor_ = tag_minor;
    first_tomb->size_ = size;
    first_tomb->score_ = score;
    first_tomb->storage_class_ = storage_class;
    gpu_cache::CopyName(first_tomb->blob_name_, blob_name);
    hdr->num_blobs_ += 1;
    return first_tomb;
  }
  return nullptr;
}

/** Mark a blob as a tombstone if found; no-op otherwise. */
CTP_CROSS_FUN inline bool GpuCacheRemoveBlob(GpuMetadataCacheHeader *hdr,
                                               clio::run::u32 tag_major,
                                               clio::run::u32 tag_minor,
                                               const char *blob_name) {
  if (!hdr || hdr->max_blobs_ == 0) return false;
  clio::run::u64 h = gpu_cache::HashBlobKey(tag_major, tag_minor, blob_name);
  clio::run::u32 cap = hdr->max_blobs_;
  auto *slots = hdr->BlobSlots();
  for (clio::run::u32 i = 0; i < cap; ++i) {
    clio::run::u32 idx = static_cast<clio::run::u32>((h + i) % cap);
    auto &s = slots[idx];
    if (s.state_ == gpu_cache::kEmpty) return false;
    if (s.state_ == gpu_cache::kTombstone) continue;
    if (s.tag_major_ == tag_major && s.tag_minor_ == tag_minor &&
        gpu_cache::NameEq(s.blob_name_, blob_name)) {
      s.state_ = gpu_cache::kTombstone;
      if (hdr->num_blobs_ > 0) hdr->num_blobs_ -= 1;
      return true;
    }
  }
  return false;
}

/**
 * Mark a tag as tombstone and remove all blobs owned by that tag.
 * Returns the number of blob entries removed.
 */
CTP_CROSS_FUN inline clio::run::u32 GpuCacheRemoveTag(GpuMetadataCacheHeader *hdr,
                                                   clio::run::u32 tag_major,
                                                   clio::run::u32 tag_minor) {
  if (!hdr) return 0;
  // Tombstone the tag.
  if (hdr->max_tags_ > 0) {
    clio::run::u64 h = gpu_cache::HashTagKey(tag_major, tag_minor);
    clio::run::u32 cap = hdr->max_tags_;
    auto *slots = hdr->TagSlots();
    for (clio::run::u32 i = 0; i < cap; ++i) {
      clio::run::u32 idx = static_cast<clio::run::u32>((h + i) % cap);
      auto &s = slots[idx];
      if (s.state_ == gpu_cache::kEmpty) break;
      if (s.state_ == gpu_cache::kTombstone) continue;
      if (s.tag_major_ == tag_major && s.tag_minor_ == tag_minor) {
        s.state_ = gpu_cache::kTombstone;
        if (hdr->num_tags_ > 0) hdr->num_tags_ -= 1;
        break;
      }
    }
  }
  // Cascade-tombstone all blob entries owned by the tag.
  clio::run::u32 removed = 0;
  if (hdr->max_blobs_ > 0) {
    clio::run::u32 cap = hdr->max_blobs_;
    auto *slots = hdr->BlobSlots();
    for (clio::run::u32 i = 0; i < cap; ++i) {
      auto &s = slots[i];
      if (s.state_ == gpu_cache::kOccupied &&
          s.tag_major_ == tag_major && s.tag_minor_ == tag_minor) {
        s.state_ = gpu_cache::kTombstone;
        removed += 1;
      }
    }
    if (hdr->num_blobs_ >= removed) hdr->num_blobs_ -= removed;
    else hdr->num_blobs_ = 0;
  }
  return removed;
}

/**
 * Read-only lookup of a blob entry. Used by GPU readers.
 * Returns nullptr if the blob is not currently in the cache.
 */
CTP_CROSS_FUN inline const gpu_cache::GpuBlobEntry *GpuCacheFindBlob(
    const GpuMetadataCacheHeader *hdr, clio::run::u32 tag_major, clio::run::u32 tag_minor,
    const char *blob_name) {
  if (!hdr || hdr->max_blobs_ == 0) return nullptr;
  clio::run::u64 h = gpu_cache::HashBlobKey(tag_major, tag_minor, blob_name);
  clio::run::u32 cap = hdr->max_blobs_;
  const auto *slots = hdr->BlobSlots();
  for (clio::run::u32 i = 0; i < cap; ++i) {
    clio::run::u32 idx = static_cast<clio::run::u32>((h + i) % cap);
    const auto &s = slots[idx];
    if (s.state_ == gpu_cache::kEmpty) return nullptr;
    if (s.state_ == gpu_cache::kTombstone) continue;
    if (s.tag_major_ == tag_major && s.tag_minor_ == tag_minor &&
        gpu_cache::NameEq(s.blob_name_, blob_name)) {
      return &s;
    }
  }
  return nullptr;
}

/** Read-only lookup of a tag entry. */
CTP_CROSS_FUN inline const gpu_cache::GpuTagEntry *GpuCacheFindTag(
    const GpuMetadataCacheHeader *hdr, clio::run::u32 tag_major,
    clio::run::u32 tag_minor) {
  if (!hdr || hdr->max_tags_ == 0) return nullptr;
  clio::run::u64 h = gpu_cache::HashTagKey(tag_major, tag_minor);
  clio::run::u32 cap = hdr->max_tags_;
  const auto *slots = hdr->TagSlots();
  for (clio::run::u32 i = 0; i < cap; ++i) {
    clio::run::u32 idx = static_cast<clio::run::u32>((h + i) % cap);
    const auto &s = slots[idx];
    if (s.state_ == gpu_cache::kEmpty) return nullptr;
    if (s.state_ == gpu_cache::kTombstone) continue;
    if (s.tag_major_ == tag_major && s.tag_minor_ == tag_minor) {
      return &s;
    }
  }
  return nullptr;
}

}  // namespace clio::cte::core

#endif  // WRPCTE_CORE_GPU_METADATA_CACHE_H_
