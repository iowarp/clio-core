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
 * COPY-ON-WRITE TESTS (lazy checkpoints, vector.Copy with sync=false)
 *
 * A lazy copy holds no bytes until it is touched. Its contract is a
 * point-in-time snapshot: whatever the source held when the copy was taken,
 * even if the source is overwritten or deleted later. Verifies, on one node:
 *   - an overwrite of part of a source blob leaves the copy with the OLD bytes
 *     while the source gets the new ones;
 *   - a second overwrite does not disturb a copy that already holds the blob;
 *   - a source blob that is never written is served lazily to the copy with
 *     the source's bytes;
 *   - two copies taken at different times each keep their own version;
 *   - deleting a source blob that a copy has not materialised keeps the copy's
 *     bytes;
 *   - destroying a copy removes its link, and writes to the source then work;
 *   - a tag with no copies of its own is unaffected by the hook.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/checkpoint/checkpoint_client.h>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>

#include "simple_test.h"

namespace fs = std::filesystem;

namespace {

constexpr clio::run::u64 kBlobSize = 64 * 1024;

std::string TestDataDir() {
  const char *d = clio::run::env::GetCompat("TEST_DATA_DIR");
  return (d && *d) ? d : ".";
}

/** Blob content of one version: every byte carries the version letter. */
std::string Version(char v) { return std::string(kBlobSize, v); }

/**
 * Owns the test's runtime, config file and checkpoint pool. Construction
 * brings up a single-node CTE with the checkpoint chimod registered, and
 * destruction removes the temporary files.
 */
class CowFixture {
 public:
  CowFixture() {
    config_path_ = TestDataDir() + "/checkpoint_cow_config.yaml";
    restart_log_path_ = TestDataDir() + "/checkpoint_cow_restart.bin";
    Cleanup();
    CreateConfigFile();
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", config_path_.c_str(), 1);
    // Hermetic pool set: a restart log from an earlier run would resurrect
    // pools with other create-params.
    ctp::SystemInfo::Setenv("CLIO_RESTART_LOG", restart_log_path_.c_str(), 1);

    bool success = clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
    REQUIRE(success);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    success = clio::cte::core::CLIO_CTE_CLIENT_INIT();
    REQUIRE(success);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // The checkpoint pool, created the way gpu_vector::Vector::Copy creates it:
    // Dynamic, so it exists on every node a fault can resolve on.
    clio::cte::checkpoint::Client ckpt(
        clio::cte::checkpoint::kCheckpointPoolId,
        clio::cte::core::kCtePoolId);
    clio::cte::checkpoint::CheckpointConfig params;
    params.next_pool_id_ = clio::cte::core::kCtePoolId;
    auto create = ckpt.AsyncCreateCheckpoint(
        clio::run::PoolQuery::Dynamic(),
        clio::cte::checkpoint::kCheckpointPoolName,
        clio::cte::checkpoint::kCheckpointPoolId, params);
    create.Wait();
    REQUIRE(create->GetReturnCode() == 0);
  }

  ~CowFixture() { Cleanup(); }

 private:
  void Cleanup() {
    if (fs::exists(config_path_)) fs::remove(config_path_);
    if (fs::exists(restart_log_path_)) fs::remove(restart_log_path_);
  }

  void CreateConfigFile() {
    std::ofstream config_file(config_path_);
    REQUIRE(config_file.is_open());
    config_file << R"(
# Copy-on-write test configuration
runtime:
  num_threads: 4
  queue_depth: 1024

compose:
  - mod_name: clio_cte_core
    pool_name: clio_cte
    pool_query: local
    pool_id: 512.0

    targets:
      neighborhood: 1
      default_target_timeout_ms: 30000
      poll_period_ms: 5000

    storage:
      - path: "ram::checkpoint_cow_dram"
        bdev_type: "ram"
        capacity_limit: "128MB"
        score: 1.0

    dpe:
      dpe_type: "max_bw"
)";
    config_file.close();
  }

  std::string config_path_;
  std::string restart_log_path_;
};

}  // namespace

TEST_CASE("CopyOnWrite - lazy copies are point-in-time snapshots",
          "[cte][checkpoint][cow]") {
  CowFixture fixture;
  auto *core = CLIO_CTE_CLIENT;
  REQUIRE(core != nullptr);

  // Put a whole blob. Returns the core's rc.
  auto put_blob = [core](clio::cte::core::TagId tag, const std::string &name,
                         clio::run::u64 off, const std::string &data) {
    auto put = core->AsyncPutBlob(tag, name.c_str(), off, data.size(),
                                  data.data());
    put.Wait();
    return put->GetReturnCode();
  };
  // Read a blob into a string. Returns the core's rc.
  auto get_blob = [core](clio::cte::core::TagId tag, const std::string &name,
                         clio::run::u64 off, clio::run::u64 size,
                         std::string *out) {
    out->assign(size, '\0');
    auto get = core->AsyncGetBlob(tag, name.c_str(), off, size, /*flags=*/0,
                                  out->data());
    get.Wait();
    return get->GetReturnCode();
  };
  auto tag_of = [core](const std::string &name) {
    auto tf = core->AsyncGetOrCreateTag(name);
    tf.Wait();
    REQUIRE(tf->GetReturnCode() == 0);
    return tf->tag_id_;
  };
  // A lazy copy of `source_name`, taken now (the vector.Copy registration).
  auto lazy_copy = [core](const std::string &copy_name,
                          const std::string &source_name) {
    auto tf = core->AsyncGetOrCreateTag(
        copy_name, clio::cte::checkpoint::kCheckpointPoolId,
        clio::cte::checkpoint::kCheckpointPoolName, source_name);
    tf.Wait();
    REQUIRE(tf->GetReturnCode() == 0);
    return tf->tag_id_;
  };

  // ----------------------------------------------------------------------
  // Source with four blobs, all version 'a'.
  // ----------------------------------------------------------------------
  const std::string kSrc = "cow_src";
  const auto src = tag_of(kSrc);
  for (const char *name : {"b0", "b1", "b2", "b3"}) {
    REQUIRE(put_blob(src, name, 0, Version('a')) == 0);
  }

  // ----------------------------------------------------------------------
  // 1. Overwrite PART of a source blob after a lazy copy was taken.
  //    The copy keeps the old bytes; the source gets the new ones.
  // ----------------------------------------------------------------------
  const auto c1 = lazy_copy("cow_c1", kSrc);
  const clio::run::u64 kOff = 1000, kLen = 4096;
  const std::string patch(kLen, 'P');
  REQUIRE(put_blob(src, "b0", kOff, patch) == 0);

  std::string got;
  REQUIRE(get_blob(c1, "b0", 0, kBlobSize, &got) == 0);
  REQUIRE(got == Version('a'));

  std::string src_want = Version('a');
  src_want.replace(kOff, kLen, patch);
  REQUIRE(get_blob(src, "b0", 0, kBlobSize, &got) == 0);
  REQUIRE(got == src_want);

  // A second overwrite must not disturb the copy, which already holds b0.
  REQUIRE(put_blob(src, "b0", 0, Version('z')) == 0);
  REQUIRE(get_blob(c1, "b0", 0, kBlobSize, &got) == 0);
  REQUIRE(got == Version('a'));

  // ----------------------------------------------------------------------
  // 2. A blob the source never overwrote: the copy faults it lazily and
  //    serves the source's bytes.
  // ----------------------------------------------------------------------
  REQUIRE(get_blob(c1, "b1", 0, kBlobSize, &got) == 0);
  REQUIRE(got == Version('a'));
  REQUIRE(get_blob(c1, "b2", 0, kBlobSize, &got) == 0);
  REQUIRE(got == Version('a'));

  // ----------------------------------------------------------------------
  // 3. Two copies taken at different times each keep their own version.
  //    Source b0 is 'z' now. c2 snapshots 'z'; then b0 becomes 'y'; c3
  //    snapshots 'y'; then b0 becomes 'x'.
  // ----------------------------------------------------------------------
  const auto c2 = lazy_copy("cow_c2", kSrc);
  REQUIRE(put_blob(src, "b0", 0, Version('y')) == 0);
  const auto c3 = lazy_copy("cow_c3", kSrc);
  REQUIRE(put_blob(src, "b0", 0, Version('x')) == 0);

  REQUIRE(get_blob(c2, "b0", 0, kBlobSize, &got) == 0);
  REQUIRE(got == Version('z'));
  REQUIRE(get_blob(c3, "b0", 0, kBlobSize, &got) == 0);
  REQUIRE(got == Version('y'));
  REQUIRE(get_blob(src, "b0", 0, kBlobSize, &got) == 0);
  REQUIRE(got == Version('x'));

  // ----------------------------------------------------------------------
  // 4. Deleting a source blob that a copy has not materialised keeps the
  //    copy's bytes. b3 was never overwritten, so the copy's snapshot is 'a'.
  // ----------------------------------------------------------------------
  const auto c4 = lazy_copy("cow_c4", kSrc);
  {
    auto del = core->AsyncDelBlob(src, "b3");
    del.Wait();
    REQUIRE(del->GetReturnCode() == 0);
  }
  REQUIRE(get_blob(c4, "b3", 0, kBlobSize, &got) == 0);
  REQUIRE(got == Version('a'));
  REQUIRE(get_blob(src, "b3", 0, kBlobSize, &got) != 0);

  // ----------------------------------------------------------------------
  // 5. Destroying a copy removes its link: a later write to the source must
  //    still succeed, and the surviving copy keeps its snapshot.
  // ----------------------------------------------------------------------
  {
    auto del = core->AsyncDelTag(std::string("cow_c2"));
    del.Wait();
    REQUIRE(del->GetReturnCode() == 0);
  }
  REQUIRE(put_blob(src, "b0", 0, Version('w')) == 0);
  REQUIRE(get_blob(c3, "b0", 0, kBlobSize, &got) == 0);
  REQUIRE(got == Version('y'));
  REQUIRE(get_blob(src, "b0", 0, kBlobSize, &got) == 0);
  REQUIRE(got == Version('w'));

  // ----------------------------------------------------------------------
  // 6. A tag with no copies of its own is unaffected by the hook while other
  //    tags have live copies.
  // ----------------------------------------------------------------------
  const auto plain = tag_of("cow_plain");
  REQUIRE(put_blob(plain, "p0", 0, Version('q')) == 0);
  REQUIRE(put_blob(plain, "p0", 0, Version('r')) == 0);
  REQUIRE(get_blob(plain, "p0", 0, kBlobSize, &got) == 0);
  REQUIRE(got == Version('r'));

  // ----------------------------------------------------------------------
  // 7. ABSENT AT SNAPSHOT: a blob the source did not have when the copy was
  //    taken must stay absent in the copy, even after the source creates it.
  //    The copy gets an empty tombstone, so it never faults the blob again.
  // ----------------------------------------------------------------------
  const auto c5 = lazy_copy("cow_c5", kSrc);
  REQUIRE(put_blob(src, "nb", 0, Version('n')) == 0);
  REQUIRE(get_blob(src, "nb", 0, kBlobSize, &got) == 0);
  REQUIRE(got == Version('n'));
  REQUIRE(get_blob(c5, "nb", 0, kBlobSize, &got) != 0);  // absent in the copy
  REQUIRE(put_blob(src, "nb", 0, Version('m')) == 0);
  REQUIRE(get_blob(c5, "nb", 0, kBlobSize, &got) != 0);  // still absent
  REQUIRE(get_blob(src, "nb", 0, kBlobSize, &got) == 0);
  REQUIRE(got == Version('m'));

  // ----------------------------------------------------------------------
  // 8. TAG DELETE keeps a blob whose lazy copy could not take its snapshot.
  //    The copy's fault pool does not exist, so the materialise fails. The
  //    delete must keep the blobs (still sized and readable), and the copy,
  //    re-registered with the real pool by the source's id (its name is now
  //    unbound), reads the snapshot bytes from the kept blobs.
  // ----------------------------------------------------------------------
  const std::string kDelSrc = "cow_del_src";
  const auto del_src = tag_of(kDelSrc);
  REQUIRE(put_blob(del_src, "d0", 0, Version('a')) == 0);
  REQUIRE(put_blob(del_src, "d1", 0, Version('a')) == 0);
  auto tag_bytes = [core](const clio::cte::core::TagId &tag) {
    auto sz = core->AsyncGetTagSize(tag);
    sz.Wait();
    REQUIRE(sz->GetReturnCode() == 0);
    return static_cast<clio::run::u64>(sz->tag_size_);
  };
  const clio::run::u64 del_bytes = tag_bytes(del_src);
  REQUIRE(del_bytes == 2 * kBlobSize);
  {
    auto tf = core->AsyncGetOrCreateTag(
        "cow_del_copy", clio::run::PoolId(999, 0),
        clio::cte::checkpoint::kCheckpointPoolName, kDelSrc);
    tf.Wait();
    REQUIRE(tf->GetReturnCode() == 0);
  }
  {
    auto del = core->AsyncDelTag(kDelSrc);
    del.Wait();
  }
  // The snapshot could not be taken, so the blobs were kept: the tag still
  // accounts for them and they still read.
  REQUIRE(tag_bytes(del_src) == del_bytes);
  REQUIRE(get_blob(del_src, "d0", 0, kBlobSize, &got) == 0);
  REQUIRE(got == Version('a'));
  REQUIRE(get_blob(del_src, "d1", 0, kBlobSize, &got) == 0);
  REQUIRE(got == Version('a'));
  // Re-register the copy with the real pool, naming the source by id.
  const auto del_copy = lazy_copy(
      "cow_del_copy", clio::cte::core::CowSourceParam(del_src));
  REQUIRE(get_blob(del_copy, "d0", 0, kBlobSize, &got) == 0);
  REQUIRE(got == Version('a'));
  REQUIRE(get_blob(del_copy, "d1", 0, kBlobSize, &got) == 0);
  REQUIRE(got == Version('a'));

  // ----------------------------------------------------------------------
  // 9. ID-FORM BINDING: "#major.minor" round-trips, and malformed forms are
  //    rejected so a plain tag name is never mistaken for an id.
  // ----------------------------------------------------------------------
  {
    clio::cte::core::TagId id;
    id.major_ = 12;
    id.minor_ = 34;
    REQUIRE(clio::cte::core::CowSourceParam(id) == "#12.34");
    clio::cte::core::TagId parsed = clio::cte::core::TagId::GetNull();
    REQUIRE(clio::cte::core::ParseCowSourceParam("#12.34", &parsed));
    REQUIRE(parsed.major_ == 12);
    REQUIRE(parsed.minor_ == 34);
    REQUIRE_FALSE(clio::cte::core::ParseCowSourceParam("cow_src", &parsed));
    REQUIRE_FALSE(clio::cte::core::ParseCowSourceParam("#12", &parsed));
    REQUIRE_FALSE(clio::cte::core::ParseCowSourceParam("#.4", &parsed));
    REQUIRE_FALSE(clio::cte::core::ParseCowSourceParam("#1.x", &parsed));
    REQUIRE_FALSE(clio::cte::core::ParseCowSourceParam("#4294967296.1", &parsed));
  }

  std::printf("copy-on-write tests passed\n");
}

SIMPLE_TEST_MAIN()
