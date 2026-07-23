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

/**
 * Phase 3 data-model tests for the CTE shared-memory metadata cache (#783).
 *
 * Pure data-structure level: builds the cache in a real shared-memory segment
 * and exercises it from a FORKED CHILD, which is the property that matters --
 * every offset must resolve at a different base address in another process.
 */

#include "clio_cte/core/shm_metadata_cache.h"

#include <clio_ctp/memory/backend/posix_shm_mmap.h>
#include <clio_runtime/bdev/bdev_tasks.h>

// fork()/waitpid() are POSIX-only; the cross-process case is compiled out on
// Windows rather than replaced with a same-process imitation, which would not
// exercise the property it exists to prove (offsets resolving at a different
// base address).
#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "simple_test.h"

using namespace clio::cte::core;

namespace {

/** Unique-ish segment name so parallel test runs do not collide. */
std::string SegName() {
  return "cte_shm_cache_test_" + std::to_string(ctp::SystemInfo::GetPid());
}

struct Fixture {
  ctp::ipc::PosixShmMmap backend;
  ShmCacheAlloc *alloc = nullptr;
  ShmMetadataCacheRoot *root = nullptr;
  std::string name;

  bool Create() {
    name = SegName();
    ctp::ipc::AllocatorId id =
        ctp::ipc::AllocatorId::Get(ctp::SystemInfo::GetPid(), 77);
    if (!backend.shm_init(id, ctp::Unit<size_t>::Megabytes(64), name)) {
      return false;
    }
    alloc = backend.MakeAlloc<ShmCacheAlloc>();
    if (alloc == nullptr) {
      return false;
    }
    auto fp = alloc->template Allocate<ShmMetadataCacheRoot>(
        sizeof(ShmMetadataCacheRoot));
    if (fp.IsNull()) {
      return false;
    }
    root = fp.ptr_;
    new (root) ShmMetadataCacheRoot();
    new (&root->tag_name_to_id_) ShmTagIdMap(alloc, 256);
    new (&root->tag_id_to_info_) ShmTagInfoMap(alloc, 256);
    new (&root->blob_key_to_info_) ShmBlobInfoMap(alloc, 1024);
    root->version_ = ShmMetadataCacheRoot::kLayoutVersion;
    root->ready_ = 1;
    return true;
  }

  /** Offset of the root within the segment, which is what a client receives. */
  size_t RootOffset() const {
    return reinterpret_cast<const char *>(root) -
           reinterpret_cast<const char *>(alloc);
  }

  void Destroy() { backend.shm_destroy(); }
};

bool PutBlob(Fixture &f, const std::string &key, const ShmBlobRecord &rec) {
  ShmBlobInfoMap::BytesProbe p{key.data(), key.size()};
  return f.root->blob_key_to_info_.InsertOrAssign(
      p, ShmCacheString::HashBytes(key.data(), key.size()), rec);
}

}  // namespace

TEST_CASE("ShmCache: records are trivially copyable", "[shm_cache]") {
  // The map copies values out between two generation reads, so a non-trivially
  // copyable record would silently defeat the seqlock.
  REQUIRE(std::is_trivially_copyable<ShmBlockDesc>::value);
  REQUIRE(std::is_trivially_copyable<ShmBlobRecord>::value);
  REQUIRE(std::is_trivially_copyable<ShmTagRecord>::value);
}

TEST_CASE("ShmCache: no vtables in SHM-resident types", "[shm_cache]") {
  // A vtable pointer is a process-local address; storing one in a segment
  // another process maps is undefined behaviour. This is why the block
  // descriptor is POD instead of embedding a bdev::Client.
  REQUIRE_FALSE(std::is_polymorphic<ShmBlockDesc>::value);
  REQUIRE_FALSE(std::is_polymorphic<ShmBlobRecord>::value);
  REQUIRE_FALSE(std::is_polymorphic<ShmTagRecord>::value);
}

TEST_CASE("ShmCache: blob key construction", "[shm_cache]") {
  char buf[128];
  clio::run::UniqueId tag(12, 34);
  size_t n = MakeShmBlobKey(tag, "myblob", 6, buf, sizeof(buf));
  REQUIRE(n > 0);
  REQUIRE(std::string(buf, n) == "12.34.myblob");

  // Too-small buffer must refuse rather than truncate: a truncated key would
  // alias a different blob.
  char small[4];
  REQUIRE(MakeShmBlobKey(tag, "myblob", 6, small, sizeof(small)) == 0);
}

TEST_CASE("ShmCache: insert and read back in-process", "[shm_cache]") {
  Fixture f;
  REQUIRE(f.Create());

  ShmBlobRecord rec;
  rec.total_size_ = 4096;
  rec.num_blocks_ = 1;
  rec.flags_ = kShmBlobDirectReadable;
  rec.placement_gen_ = 7;
  rec.blocks_[0].target_offset_ = 512;
  rec.blocks_[0].size_ = 4096;
  rec.blocks_[0].bdev_type_ =
      static_cast<clio::run::u32>(clio::run::bdev::BdevType::kRam);
  REQUIRE(PutBlob(f, "1.0.blob_a", rec));

  ShmBlobRecord out;
  REQUIRE(f.root->blob_key_to_info_.TryGetBytes("1.0.blob_a", 10, &out));
  REQUIRE(out.total_size_ == 4096);
  REQUIRE(out.num_blocks_ == 1);
  REQUIRE(out.blocks_[0].target_offset_ == 512);
  REQUIRE(out.IsDirectReadable());

  f.Destroy();
}

TEST_CASE("ShmCache: truncated blob is not direct-readable", "[shm_cache]") {
  ShmBlobRecord rec;
  rec.num_blocks_ = kMaxInlineBlocks;
  rec.flags_ = kShmBlobDirectReadable | kShmBlobTruncated;
  // A blob with more blocks than fit must never be payload-read from the
  // cache -- the block list it carries is incomplete.
  REQUIRE_FALSE(rec.IsDirectReadable());
}

TEST_CASE("ShmCache: tag maps round-trip", "[shm_cache]") {
  Fixture f;
  REQUIRE(f.Create());

  clio::run::UniqueId tag(3, 9);
  std::string name = "my_tag";
  ShmTagIdMap::BytesProbe p{name.data(), name.size()};
  REQUIRE(f.root->tag_name_to_id_.InsertOrAssign(
      p, ShmCacheString::HashBytes(name.data(), name.size()), tag));

  ShmTagRecord trec;
  trec.total_size_ = 123;
  trec.last_modified_ = 456;
  REQUIRE(f.root->tag_id_to_info_.InsertOrAssign(
      tag, ShmTagInfoMap::Hash(tag), trec));

  clio::run::UniqueId got_id;
  REQUIRE(f.root->tag_name_to_id_.TryGetBytes(name.data(), name.size(),
                                              &got_id));
  REQUIRE(got_id.major_ == 3);
  REQUIRE(got_id.minor_ == 9);

  ShmTagRecord got_rec;
  REQUIRE(f.root->tag_id_to_info_.TryGet(tag, ShmTagInfoMap::Hash(tag),
                                         &got_rec));
  REQUIRE(got_rec.total_size_ == 123);
  REQUIRE(got_rec.last_modified_ == 456);

  f.Destroy();
}

#ifndef _WIN32
TEST_CASE("ShmCache: readable from another process at a different address", "[shm_cache]") {
  // THE test that matters: every internal reference is a segment-relative
  // offset, so a child that maps the same segment at a different base address
  // must resolve identical data. A raw pointer anywhere would fail here.
  Fixture f;
  REQUIRE(f.Create());

  const int kEntries = 200;
  for (int i = 0; i < kEntries; ++i) {
    ShmBlobRecord rec;
    rec.total_size_ = static_cast<clio::run::u64>(i) * 100;
    rec.num_blocks_ = 1;
    rec.placement_gen_ = static_cast<clio::run::u64>(i);
    rec.flags_ = kShmBlobDirectReadable;
    rec.blocks_[0].target_offset_ = static_cast<clio::run::u64>(i) * 4096;
    rec.blocks_[0].size_ = 100;
    REQUIRE(PutBlob(f, "1.0.blob_" + std::to_string(i), rec));
  }
  // Long key crossing the SSO boundary, to prove spilled keys resolve too.
  std::string longkey = "1.0." + std::string(120, 'L');
  ShmBlobRecord big;
  big.total_size_ = 999999;
  REQUIRE(PutBlob(f, longkey, big));

  size_t root_off = f.RootOffset();
  std::string seg = f.name;

  pid_t pid = fork();
  if (pid == 0) {
    // Child: attach the segment fresh. mmap will very likely place it at a
    // different base address than the parent's.
    ctp::ipc::PosixShmMmap child_backend;
    int rc = 0;
    if (!child_backend.shm_attach(seg)) {
      _exit(21);
    }
    auto *child_alloc = child_backend.AttachAlloc<ShmCacheAlloc>();
    if (child_alloc == nullptr) {
      _exit(22);
    }
    auto *child_root = reinterpret_cast<ShmMetadataCacheRoot *>(
        reinterpret_cast<char *>(child_alloc) + root_off);
    if (child_root->ready_ != 1 ||
        child_root->version_ != ShmMetadataCacheRoot::kLayoutVersion) {
      _exit(23);
    }
    for (int i = 0; i < kEntries; ++i) {
      std::string k = "1.0.blob_" + std::to_string(i);
      ShmBlobRecord out;
      if (!child_root->blob_key_to_info_.TryGetBytes(k.data(), k.size(),
                                                     &out)) {
        rc = 24;
        break;
      }
      if (out.total_size_ != static_cast<clio::run::u64>(i) * 100 ||
          out.blocks_[0].target_offset_ !=
              static_cast<clio::run::u64>(i) * 4096) {
        rc = 25;
        break;
      }
    }
    if (rc == 0) {
      ShmBlobRecord out;
      if (!child_root->blob_key_to_info_.TryGetBytes(longkey.data(),
                                                     longkey.size(), &out)) {
        rc = 26;
      } else if (out.total_size_ != 999999) {
        rc = 27;
      }
    }
    // A key that was never inserted must miss, not alias something else.
    if (rc == 0) {
      ShmBlobRecord out;
      if (child_root->blob_key_to_info_.TryGetBytes("1.0.nope", 8, &out)) {
        rc = 28;
      }
    }
    _exit(rc);
  }

  int status = 0;
  waitpid(pid, &status, 0);
  REQUIRE(WIFEXITED(status));
  REQUIRE(WEXITSTATUS(status) == 0);

  f.Destroy();
}
#endif  // !_WIN32

SIMPLE_TEST_MAIN()
