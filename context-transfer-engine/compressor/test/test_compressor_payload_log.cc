/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file test_compressor_payload_log.cc
 * @brief The payload log must describe the bytes that were STORED.
 *
 * CLIO_NEUROPRESS_SELECTION_LOG writes a companion `.payload` file with one
 * row per blob: the compressed length, a hash of the codec output, and
 * whether compression was kept. Every consumer reads it as "what this chunk
 * cost on the tier".
 *
 * It was written from inside Compress(), which runs before exploration can
 * replace the result. When exploration adopted an alternative -- which is its
 * entire purpose -- the log kept describing the PRIMARY: a chunk stored at
 * 3,549 bytes was reported at 10,466, and a chunk stored compressed was
 * reported `beneficial=0` because the primary had expanded it.
 *
 * The error is not random. It only ever misreports the modes that explore,
 * and always in the direction that makes them look worse, so a comparison
 * built on this log ranks exploration below configurations it actually beats.
 * A nine-way Gray-Scott comparison drew exactly that wrong conclusion; it was
 * caught only by checking the tier file size on disk, which disagreed with
 * the log by 4.5 MB on best mode.
 *
 * Own executable, not a case in test_compressor_functional: both log handles
 * are function-local statics resolved on first use, so whichever test runs
 * first fixes the path for the whole process. A second test case here would
 * inherit this one's file.
 */

#include "simple_test.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/compressor/compressor_client.h>
#include <clio_cte/compressor/compressor_runtime.h>
#include <clio_cte/compressor/compressor_tasks.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_runtime/bdev/bdev_tasks.h>

using clio::cte::compressor::Client;
using clio::cte::compressor::CompressorConfig;
using clio::cte::core::Context;

namespace {

/** Clio's on-disk header, excluded from the payload log's length by design. */
constexpr size_t kCompressionHeaderBytes = 24;

std::string g_sel_log;   // set before anything touches the log statics

std::vector<char> GenerateTestData(size_t size, const std::string &pattern) {
  std::vector<char> d(size);
  if (pattern == "repeating") {
    const char *unit = "ABCDEFGH";
    for (size_t i = 0; i < size; ++i) d[i] = unit[i % 8];
  } else if (pattern == "zeros") {
    std::memset(d.data(), 0, size);
  } else {  // "text": compressible but not trivially so
    const char *w[] = {"the ", "quick ", "brown ", "fox ", "jumps ", "over "};
    size_t o = 0;
    while (o < size) {
      const char *s = w[(o / 7) % 6];
      size_t n = std::min(std::strlen(s), size - o);
      std::memcpy(d.data() + o, s, n);
      o += n;
    }
  }
  return d;
}

struct PayloadRow {
  size_t compressed_size = 0;
  int beneficial = 0;
};

std::map<std::string, PayloadRow> ReadPayloadLog(const std::string &path) {
  std::map<std::string, PayloadRow> out;
  std::ifstream f(path);
  std::string line;
  if (!std::getline(f, line)) return out;  // header
  while (std::getline(f, line)) {
    std::stringstream ss(line);
    std::string blob, size, hash, ben;
    if (!std::getline(ss, blob, ',')) continue;
    if (!std::getline(ss, size, ',')) continue;
    if (!std::getline(ss, hash, ',')) continue;
    if (!std::getline(ss, ben, ',')) continue;
    PayloadRow r;
    r.compressed_size = std::stoull(size);
    r.beneficial = std::stoi(ben);
    out[blob] = r;  // last row wins, which is what a consumer sees
  }
  return out;
}

/** Blobs where exploration replaced the primary, from the explore log. */
std::set<std::string> AdoptedBlobs(const std::string &path) {
  std::set<std::string> out;
  std::ifstream f(path);
  std::string line;
  if (!std::getline(f, line)) return out;
  const std::vector<std::string> cols = [&] {
    std::vector<std::string> c;
    std::stringstream ss(line);
    std::string t;
    while (std::getline(ss, t, ',')) c.push_back(t);
    return c;
  }();
  size_t blob_i = 0, adopt_i = 0;
  for (size_t i = 0; i < cols.size(); ++i) {
    if (cols[i] == "blob") blob_i = i;
    if (cols[i] == "adopted") adopt_i = i;
  }
  while (std::getline(f, line)) {
    std::vector<std::string> v;
    std::stringstream ss(line);
    std::string t;
    while (std::getline(ss, t, ',')) v.push_back(t);
    if (v.size() > std::max(blob_i, adopt_i) && v[adopt_i] == "1")
      out.insert(v[blob_i]);
  }
  return out;
}

}  // namespace

TEST_CASE("Payload log reports the stored bytes, not the primary's",
          "[compressor][payload_log][exploration][693]") {
#ifndef CLIO_CTP_NEUROPRESS_WEIGHTS_DIR
#error "CLIO_CTP_NEUROPRESS_WEIGHTS_DIR must be set by CMake"
#endif
  REQUIRE(!g_sel_log.empty());
  const std::string payload_path = g_sel_log + ".payload";
  const std::string explore_path = g_sel_log + ".explore";

  // Same setup the functional suite uses: client mode with runtime, a core
  // pool with a RAM target (without one every PutBlob fails in DPE placement),
  // then a compressor pool wired to it.
  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true));

  const auto core_pool = clio::run::PoolId(1, 1);
  clio::cte::core::Client core_client;
  clio::cte::core::CreateParams core_params;
  auto core_create = core_client.AsyncCreate(clio::run::PoolQuery::Local(),
                                             "payloadlog_core", core_pool,
                                             core_params);
  core_create.Wait();
  core_client.Init(core_pool);
  auto reg = core_client.AsyncRegisterTarget(
      "payloadlog_ram_target", clio::run::bdev::BdevType::kRam,
      static_cast<clio::run::u64>(4) * 1024 * 1024 * 1024,
      clio::run::PoolQuery::Local(), clio::run::PoolId(700, 0));
  reg.Wait();
  REQUIRE(reg->GetReturnCode() == 0);

  CompressorConfig cfg;
  cfg.neuropress_model_path_ = CLIO_CTP_NEUROPRESS_WEIGHTS_DIR;
  cfg.next_pool_id_ = core_pool;
  cfg.neuropress_online_learning_enabled_ = true;  // computes error_pct
  cfg.neuropress_exploration_enabled_ = true;      // the path under test
  cfg.neuropress_exploration_threshold_ = 0.0f;    // fire on every chunk
  cfg.neuropress_exploration_k_ = 31;              // full action space

  const auto comp_pool = clio::run::PoolId(2, 92);
  Client client;
  auto comp_create = client.AsyncCreateCompressor(
      clio::run::PoolQuery::Local(), "payloadlog_comp", comp_pool, cfg);
  comp_create.Wait();
  client.Init(comp_pool);

  auto tag_task = core_client.AsyncGetOrCreateTag("payloadlog_tag");
  tag_task.Wait();
  const auto tag_id = tag_task->tag_id_;

  struct Trial { const char *pattern; size_t size; };
  const std::vector<Trial> trials = {
      {"repeating", 1024 * 1024}, {"zeros", 1024 * 1024},
      {"text", 512 * 1024},       {"repeating", 256 * 1024},
  };

  // Final stored size per blob, straight off the task. An adopted winner is
  // committed by assigning task->context_ wholesale, so this is the adopted
  // codec's total when exploration replaced the primary.
  std::map<std::string, size_t> stored;

  for (size_t i = 0; i < trials.size(); ++i) {
    const auto data = GenerateTestData(trials[i].size, trials[i].pattern);
    auto buf = CLIO_IPC->AllocateBuffer(data.size());
    REQUIRE(!buf.IsNull());
    std::memcpy(buf.ptr_, data.data(), data.size());

    const std::string blob = std::string("payload_blob_") + std::to_string(i);
    Context ctx;
    ctx.dynamic_compress_ = 0;
    ctx.max_performance_ = false;

    auto task = client.AsyncDynamicSchedule(
        clio::run::PoolQuery::Local(), tag_id, blob, 0, data.size(),
        buf.shm_.template Cast<void>(), 0.5f, ctx, 0, core_pool);
    task.Wait();
    REQUIRE(task->return_code_ == 0);
    if (task->context_.actual_compressed_size_ > 0)
      stored[blob] = task->context_.actual_compressed_size_;
    CLIO_IPC->FreeBuffer(buf);
  }

  const auto rows = ReadPayloadLog(payload_path);
  REQUIRE(!rows.empty());

  // Non-vacuity: this only tests anything if exploration actually replaced a
  // primary somewhere. Without this the whole case passes on a run where
  // nothing was adopted and the bug would be invisible.
  const auto adopted = AdoptedBlobs(explore_path);
  INFO("blobs where exploration adopted an alternative: " << adopted.size());
  REQUIRE(!adopted.empty());

  size_t checked = 0;
  for (const auto &[blob, total] : stored) {
    auto it = rows.find(blob);
    if (it == rows.end()) continue;
    // The log excludes Clio's header by design, so the row plus one header is
    // the total the chunk occupies -- the number the task reports.
    const size_t from_log = it->second.compressed_size + kCompressionHeaderBytes;
    INFO("blob " << blob << ": payload log says " << from_log
                 << ", actually stored " << total);
    REQUIRE(from_log == total);
    REQUIRE(it->second.beneficial == 1);
    ++checked;
  }
  REQUIRE(checked > 0);
}

// Hand-rolled rather than SIMPLE_TEST_MAIN(): the log paths must be in the
// environment BEFORE any Clio call, because both log handles are
// function-local statics that capture their path the first time they are
// touched. The macro offers no hook ahead of run_all_tests().
int main(int argc, char *argv[]) {
  setenv("CLIO_UNIT_TEST", "1", 1);
  char tmpl[] = "/tmp/clio_payload_log_testXXXXXX";
  const char *dir = mkdtemp(tmpl);
  if (dir == nullptr) {
    std::fprintf(stderr, "mkdtemp failed\n");
    return 1;
  }
  g_sel_log = std::string(dir) + "/sel.csv";
  setenv("CLIO_NEUROPRESS_SELECTION_LOG", g_sel_log.c_str(), 1);
  setenv("CLIO_NEUROPRESS_EXPLORE_LOG", (g_sel_log + ".explore").c_str(), 1);

  std::string filter = (argc > 1) ? argv[1] : "";
  int result = SimpleTest::run_all_tests(filter);
  SIMPLE_TEST_PROCESS_EXIT(result);
  if (SimpleTest::g_test_finalize) SimpleTest::g_test_finalize();
  return result;
}
