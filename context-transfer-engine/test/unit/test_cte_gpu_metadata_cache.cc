/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * Unit tests for the GPU metadata cache (gpu_metadata_cache.h).
 *
 * The cache is a POD open-addressing hash table designed to live in
 * managed/shared USM, but every operation is plain CPU code — so the whole
 * header is testable host-side with a malloc'd region.
 */

#include "simple_test.h"

#include <clio_cte/core/gpu_metadata_cache.h>

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using clio::cte::core::GpuCacheFindBlob;
using clio::cte::core::GpuCacheFindTag;
using clio::cte::core::GpuCacheRemoveBlob;
using clio::cte::core::GpuCacheRemoveTag;
using clio::cte::core::GpuCacheUpsertBlob;
using clio::cte::core::GpuCacheUpsertTag;
using clio::cte::core::GpuMetadataCacheHeader;
namespace gpu_cache = clio::cte::core::gpu_cache;

namespace {

/** RAII region holding an initialized cache of the given capacities. */
struct CacheRegion {
  void *mem_ = nullptr;
  GpuMetadataCacheHeader *hdr_ = nullptr;

  CacheRegion(chi::u32 max_tags, chi::u32 max_blobs) {
    size_t bytes = GpuMetadataCacheHeader::Layout(max_tags, max_blobs);
    mem_ = std::calloc(1, bytes);
    hdr_ = static_cast<GpuMetadataCacheHeader *>(mem_);
    hdr_->Init(max_tags, max_blobs, bytes);
  }
  ~CacheRegion() { std::free(mem_); }
  CacheRegion(const CacheRegion &) = delete;
  CacheRegion &operator=(const CacheRegion &) = delete;
};

}  // namespace

TEST_CASE("GpuMetadataCache - layout, init, and magic",
          "[cte][gpu_cache][layout]") {
  SECTION("Layout grows with capacities");
  size_t base = GpuMetadataCacheHeader::Layout(0, 0);
  REQUIRE(base >= sizeof(GpuMetadataCacheHeader));
  REQUIRE(GpuMetadataCacheHeader::Layout(4, 0) ==
          base + 4 * sizeof(gpu_cache::GpuTagEntry));
  REQUIRE(GpuMetadataCacheHeader::Layout(0, 8) ==
          base + 8 * sizeof(gpu_cache::GpuBlobEntry));

  SECTION("Init fills header and resets slots");
  CacheRegion region(8, 16);
  REQUIRE(region.hdr_->ValidMagic());
  REQUIRE(region.hdr_->max_tags_ == 8);
  REQUIRE(region.hdr_->max_blobs_ == 16);
  REQUIRE(region.hdr_->num_tags_ == 0);
  REQUIRE(region.hdr_->num_blobs_ == 0);
  for (chi::u32 i = 0; i < 8; ++i) {
    REQUIRE(region.hdr_->TagSlots()[i].state_ == gpu_cache::kEmpty);
  }
  for (chi::u32 i = 0; i < 16; ++i) {
    REQUIRE(region.hdr_->BlobSlots()[i].state_ == gpu_cache::kEmpty);
  }

  SECTION("Corrupted magic detected");
  region.hdr_->magic_ = 0xDEADBEEF;
  REQUIRE_FALSE(region.hdr_->ValidMagic());
  region.hdr_->magic_ = GpuMetadataCacheHeader::kMagic;

  SECTION("Const slot accessors agree with mutable ones");
  const GpuMetadataCacheHeader *chdr = region.hdr_;
  REQUIRE(chdr->TagSlots() == region.hdr_->TagSlots());
  REQUIRE(chdr->BlobSlots() == region.hdr_->BlobSlots());
}

TEST_CASE("GpuMetadataCache - name and hash helpers",
          "[cte][gpu_cache][helpers]") {
  SECTION("CopyName handles null, short, and overlong sources");
  char dst[gpu_cache::kMaxName];
  gpu_cache::CopyName(dst, nullptr);
  REQUIRE(dst[0] == '\0');
  gpu_cache::CopyName(dst, "hello");
  REQUIRE(std::strcmp(dst, "hello") == 0);
  std::string overlong(2 * gpu_cache::kMaxName, 'x');
  gpu_cache::CopyName(dst, overlong.c_str());
  REQUIRE(gpu_cache::NameLen(dst) <= gpu_cache::kMaxName);

  SECTION("NameLen counts up to kMaxName");
  REQUIRE(gpu_cache::NameLen("") == 0);
  REQUIRE(gpu_cache::NameLen("abc") == 3);

  SECTION("NameEq compares within kMaxName");
  REQUIRE(gpu_cache::NameEq("same", "same"));
  REQUIRE_FALSE(gpu_cache::NameEq("same", "diff"));
  REQUIRE_FALSE(gpu_cache::NameEq("same", "sameer"));

  SECTION("Hashes are deterministic and key-sensitive");
  REQUIRE(gpu_cache::FnvHashString("blob_a") ==
          gpu_cache::FnvHashString("blob_a"));
  REQUIRE(gpu_cache::FnvHashString("blob_a") !=
          gpu_cache::FnvHashString("blob_b"));
  REQUIRE(gpu_cache::HashTagKey(1, 2) != gpu_cache::HashTagKey(2, 1));
  REQUIRE(gpu_cache::HashBlobKey(1, 2, "n") ==
          gpu_cache::HashBlobKey(1, 2, "n"));
  REQUIRE(gpu_cache::HashBlobKey(1, 2, "n") !=
          gpu_cache::HashBlobKey(1, 3, "n"));
}

TEST_CASE("GpuMetadataCache - storage class helpers",
          "[cte][gpu_cache][storage]") {
  SECTION("BdevTypeToStorageClass maps known types");
  REQUIRE(gpu_cache::BdevTypeToStorageClass("ram") == gpu_cache::kStorageRam);
  REQUIRE(gpu_cache::BdevTypeToStorageClass("hbm") == gpu_cache::kStorageHbm);
  REQUIRE(gpu_cache::BdevTypeToStorageClass("pinned") ==
          gpu_cache::kStoragePinned);
  REQUIRE(gpu_cache::BdevTypeToStorageClass("file") ==
          gpu_cache::kStorageUnknown);
  REQUIRE(gpu_cache::BdevTypeToStorageClass(nullptr) ==
          gpu_cache::kStorageUnknown);

  SECTION("IsGpuVisible accepts only GPU-reachable classes");
  REQUIRE(gpu_cache::IsGpuVisible(gpu_cache::kStorageRam));
  REQUIRE(gpu_cache::IsGpuVisible(gpu_cache::kStorageHbm));
  REQUIRE(gpu_cache::IsGpuVisible(gpu_cache::kStoragePinned));
  REQUIRE_FALSE(gpu_cache::IsGpuVisible(gpu_cache::kStorageUnknown));
}

TEST_CASE("GpuMetadataCache - tag upsert, find, remove",
          "[cte][gpu_cache][tags]") {
  CacheRegion region(8, 8);
  auto *hdr = region.hdr_;

  SECTION("Insert and find a tag");
  auto *t1 = GpuCacheUpsertTag(hdr, 1, 0, "tag_one");
  REQUIRE(t1 != nullptr);
  REQUIRE(t1->state_ == gpu_cache::kOccupied);
  REQUIRE(hdr->num_tags_ == 1);
  const auto *found = GpuCacheFindTag(hdr, 1, 0);
  REQUIRE(found != nullptr);
  REQUIRE(gpu_cache::NameEq(found->tag_name_, "tag_one"));

  SECTION("Upsert of an existing tag updates in place");
  auto *t1b = GpuCacheUpsertTag(hdr, 1, 0, "tag_one_renamed");
  REQUIRE(t1b == t1);
  REQUIRE(hdr->num_tags_ == 1);
  REQUIRE(gpu_cache::NameEq(t1->tag_name_, "tag_one_renamed"));

  SECTION("Find of a missing tag returns null");
  REQUIRE(GpuCacheFindTag(hdr, 99, 99) == nullptr);

  SECTION("Remove tombstones the tag");
  GpuCacheUpsertTag(hdr, 2, 0, "tag_two");
  REQUIRE(hdr->num_tags_ == 2);
  GpuCacheRemoveTag(hdr, 1, 0);
  REQUIRE(hdr->num_tags_ == 1);
  REQUIRE(GpuCacheFindTag(hdr, 1, 0) == nullptr);
  REQUIRE(GpuCacheFindTag(hdr, 2, 0) != nullptr);

  SECTION("Tombstoned slot is reused by a new insert");
  auto *t3 = GpuCacheUpsertTag(hdr, 3, 0, "tag_three");
  REQUIRE(t3 != nullptr);
  REQUIRE(hdr->num_tags_ == 2);

  SECTION("Table full returns null");
  CacheRegion tiny(2, 2);
  REQUIRE(GpuCacheUpsertTag(tiny.hdr_, 1, 1, "a") != nullptr);
  REQUIRE(GpuCacheUpsertTag(tiny.hdr_, 2, 2, "b") != nullptr);
  REQUIRE(GpuCacheUpsertTag(tiny.hdr_, 3, 3, "c") == nullptr);

  SECTION("Zero-capacity and null header are rejected");
  CacheRegion empty(0, 0);
  REQUIRE(GpuCacheUpsertTag(empty.hdr_, 1, 1, "a") == nullptr);
  REQUIRE(GpuCacheFindTag(empty.hdr_, 1, 1) == nullptr);
  REQUIRE(GpuCacheUpsertTag(nullptr, 1, 1, "a") == nullptr);
  REQUIRE(GpuCacheFindTag(nullptr, 1, 1) == nullptr);
  REQUIRE(GpuCacheRemoveTag(nullptr, 1, 1) == 0);
}

TEST_CASE("GpuMetadataCache - blob upsert, find, remove",
          "[cte][gpu_cache][blobs]") {
  CacheRegion region(8, 8);
  auto *hdr = region.hdr_;

  SECTION("Insert and find a blob");
  auto *b1 = GpuCacheUpsertBlob(hdr, 1, 0, "blob_one", 4096, 0.5f,
                                gpu_cache::kStorageRam);
  REQUIRE(b1 != nullptr);
  REQUIRE(b1->size_ == 4096);
  REQUIRE(hdr->num_blobs_ == 1);
  const auto *found = GpuCacheFindBlob(hdr, 1, 0, "blob_one");
  REQUIRE(found != nullptr);
  REQUIRE(found->storage_class_ == gpu_cache::kStorageRam);

  SECTION("Upsert of an existing blob updates size/score/class in place");
  auto *b1b = GpuCacheUpsertBlob(hdr, 1, 0, "blob_one", 8192, 0.9f,
                                 gpu_cache::kStorageHbm);
  REQUIRE(b1b == b1);
  REQUIRE(hdr->num_blobs_ == 1);
  REQUIRE(b1->size_ == 8192);
  REQUIRE(b1->storage_class_ == gpu_cache::kStorageHbm);

  SECTION("Blobs with the same name under different tags are distinct");
  auto *b2 = GpuCacheUpsertBlob(hdr, 2, 0, "blob_one", 1, 0.1f,
                                gpu_cache::kStorageRam);
  REQUIRE(b2 != nullptr);
  REQUIRE(b2 != b1);
  REQUIRE(hdr->num_blobs_ == 2);

  SECTION("Find misses: wrong name, wrong tag");
  REQUIRE(GpuCacheFindBlob(hdr, 1, 0, "nope") == nullptr);
  REQUIRE(GpuCacheFindBlob(hdr, 9, 9, "blob_one") == nullptr);

  SECTION("RemoveBlob tombstones exactly the requested blob");
  REQUIRE(GpuCacheRemoveBlob(hdr, 1, 0, "blob_one"));
  REQUIRE(hdr->num_blobs_ == 1);
  REQUIRE(GpuCacheFindBlob(hdr, 1, 0, "blob_one") == nullptr);
  REQUIRE(GpuCacheFindBlob(hdr, 2, 0, "blob_one") != nullptr);
  REQUIRE_FALSE(GpuCacheRemoveBlob(hdr, 1, 0, "blob_one"));

  SECTION("Null header rejected");
  REQUIRE(GpuCacheUpsertBlob(nullptr, 1, 0, "x", 0, 0.f,
                             gpu_cache::kStorageRam) == nullptr);
  REQUIRE(GpuCacheFindBlob(nullptr, 1, 0, "x") == nullptr);
  REQUIRE_FALSE(GpuCacheRemoveBlob(nullptr, 1, 0, "x"));
}

TEST_CASE("GpuMetadataCache - RemoveTag cascades to owned blobs",
          "[cte][gpu_cache][cascade]") {
  CacheRegion region(8, 16);
  auto *hdr = region.hdr_;

  GpuCacheUpsertTag(hdr, 5, 0, "victim");
  GpuCacheUpsertTag(hdr, 6, 0, "survivor");
  GpuCacheUpsertBlob(hdr, 5, 0, "v1", 1, 0.f, gpu_cache::kStorageRam);
  GpuCacheUpsertBlob(hdr, 5, 0, "v2", 2, 0.f, gpu_cache::kStorageRam);
  GpuCacheUpsertBlob(hdr, 6, 0, "s1", 3, 0.f, gpu_cache::kStorageRam);
  REQUIRE(hdr->num_tags_ == 2);
  REQUIRE(hdr->num_blobs_ == 3);

  SECTION("Removing the victim removes both of its blobs");
  chi::u32 removed = GpuCacheRemoveTag(hdr, 5, 0);
  REQUIRE(removed == 2);
  REQUIRE(hdr->num_tags_ == 1);
  REQUIRE(hdr->num_blobs_ == 1);
  REQUIRE(GpuCacheFindTag(hdr, 5, 0) == nullptr);
  REQUIRE(GpuCacheFindBlob(hdr, 5, 0, "v1") == nullptr);
  REQUIRE(GpuCacheFindBlob(hdr, 5, 0, "v2") == nullptr);
  REQUIRE(GpuCacheFindBlob(hdr, 6, 0, "s1") != nullptr);

  SECTION("Removing a missing tag is a harmless no-op");
  REQUIRE(GpuCacheRemoveTag(hdr, 42, 42) == 0);
}

TEST_CASE("GpuMetadataCache - probe chains survive tombstones",
          "[cte][gpu_cache][probing]") {
  // Small table forces collisions so the linear-probe + tombstone-skip
  // paths in Upsert/Find/Remove are exercised.
  CacheRegion region(4, 4);
  auto *hdr = region.hdr_;

  std::vector<std::string> names = {"a", "b", "c", "d"};
  for (const auto &n : names) {
    REQUIRE(GpuCacheUpsertBlob(hdr, 7, 7, n.c_str(), 1, 0.f,
                               gpu_cache::kStorageRam) != nullptr);
  }
  REQUIRE(hdr->num_blobs_ == 4);
  // Full table: a fifth insert must fail.
  REQUIRE(GpuCacheUpsertBlob(hdr, 7, 7, "e", 1, 0.f,
                             gpu_cache::kStorageRam) == nullptr);

  // Remove an entry in the middle of the probe chain; the rest must
  // remain findable through the tombstone.
  REQUIRE(GpuCacheRemoveBlob(hdr, 7, 7, "b"));
  for (const auto &n : {"a", "c", "d"}) {
    REQUIRE(GpuCacheFindBlob(hdr, 7, 7, n) != nullptr);
  }
  REQUIRE(GpuCacheFindBlob(hdr, 7, 7, "b") == nullptr);

  // The tombstone is reused on re-insert.
  REQUIRE(GpuCacheUpsertBlob(hdr, 7, 7, "e", 1, 0.f,
                             gpu_cache::kStorageRam) != nullptr);
  REQUIRE(hdr->num_blobs_ == 4);
}

SIMPLE_TEST_MAIN()
