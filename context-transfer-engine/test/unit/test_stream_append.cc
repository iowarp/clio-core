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
 * Stream chimod: logical sizes and the deferred append pipeline on one node
 * (sizes, concurrent writers, page-boundary appends, append after truncate,
 * appends to a dropped stream).
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/stream/stream_client.h>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "simple_test.h"

namespace fs = std::filesystem;
using clio::cte::core::TagId;
using clio::cte::stream::StreamSizeOp;

namespace {

constexpr clio::run::u32 kHome = 0;  ///< single-node test: container 0
constexpr int kWriters = 8;
constexpr int kPerWriter = 125;
constexpr size_t kRecLen = 100;

/** @return directory for test scratch files. */
std::string DataDir() {
  const char *d = clio::run::env::GetCompat("TEST_DATA_DIR");
  return (d && *d) ? d : ".";
}

/** Server config: CTE core on a RAM tier plus the stream pool over it. */
void WriteConfig(const std::string &path, const std::string &log_path) {
  std::ofstream f(path);
  REQUIRE(f.is_open());
  f << R"(
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
      - path: "ram::stream_test_dram"
        bdev_type: "ram"
        capacity_limit: "256MB"
        score: 1.0
    dpe:
      dpe_type: "max_bw"
  - mod_name: clio_cte_stream
    pool_name: clio_cte_stream
    pool_query: local
    pool_id: "565.0"
    next_pool_id: "512.0"
    log_path: ")" << log_path << R"("
)";
}

/** Fixture: runtime + CTE client with the stream pool composed. */
class StreamFixture {
 public:
  StreamFixture() {
    conf_ = DataDir() + "/stream_append_config.yaml";
    restart_log_ = DataDir() + "/stream_append_restart.bin";
    WriteConfig(conf_, DataDir() + "/stream_append_log");
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", conf_.c_str(), 1);
    ctp::SystemInfo::Setenv("CLIO_RESTART_LOG", restart_log_.c_str(), 1);
    REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true));
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    REQUIRE(clio::cte::core::CLIO_CTE_CLIENT_INIT());
  }
  ~StreamFixture() {
    std::error_code ec;
    fs::remove(conf_, ec);
    fs::remove(restart_log_, ec);
  }

 private:
  std::string conf_;
  std::string restart_log_;
};

/** Create (or find) a tag for a test stream. */
TagId MakeTag(const std::string &name) {
  auto *cte = CLIO_CTE_CLIENT;
  auto t = cte->AsyncGetOrCreateTag(name);
  t.Wait();
  REQUIRE(t->GetReturnCode() == 0);
  return t->tag_id_;
}

/** Read [0, size) of a stream from its page blobs. */
std::string ReadStream(const TagId &tag, clio::run::u64 size) {
  auto *cte = CLIO_CTE_CLIENT;
  std::string out(size, '\0');
  clio::run::u64 off = 0;
  while (off < size) {
    const clio::run::u64 n =
        std::min(clio::cte::stream::kStreamPageSize, size - off);
    auto g = cte->AsyncGetBlob(tag, clio::cte::stream::StreamPageName(off), 0,
                               n, 0u, out.data() + off);
    g.Wait();
    REQUIRE(g->GetReturnCode() == 0);
    off += n;
  }
  return out;
}

/** Stream size at the home. */
clio::run::u64 SizeOf(clio::cte::stream::Client &s, const TagId &tag) {
  auto f = s.AsyncSizeOp(tag, kHome, StreamSizeOp::kGet);
  f.Wait();
  REQUIRE(f->GetReturnCode() == 0);
  return f->new_size_;
}

/** Flush and require success; returns the size. */
clio::run::u64 FlushOf(clio::cte::stream::Client &s, const TagId &tag) {
  auto f = s.AsyncFlush(tag, kHome);
  f.Wait();
  REQUIRE(f->GetReturnCode() == 0);
  return f->size_;
}

/** A self-identifying fixed-length record. */
std::string Record(int writer, int seq) {
  char buf[kRecLen + 1];
  std::snprintf(buf, sizeof(buf), "W%02d S%05d ", writer, seq);
  std::string r(buf);
  r.resize(kRecLen - 1, static_cast<char>('a' + (writer + seq) % 26));
  r.push_back('\n');
  return r;
}

}  // namespace

TEST_CASE("Stream sizes", "[stream]") {
  StreamFixture fx;
  clio::cte::stream::Client s;
  TagId tag = MakeTag("stream_sizes");
  auto op = [&](StreamSizeOp o, clio::run::u64 v, clio::run::u64 *old_sz) {
    auto f = s.AsyncSizeOp(tag, kHome, o, v);
    f.Wait();
    REQUIRE(f->GetReturnCode() == 0);
    if (old_sz != nullptr) *old_sz = f->old_size_;
    return f->new_size_;
  };
  clio::run::u64 old_sz = 0;
  REQUIRE(op(StreamSizeOp::kSet, 100, &old_sz) == 100);
  REQUIRE(old_sz == 0);
  REQUIRE(op(StreamSizeOp::kMax, 50, nullptr) == 100);
  REQUIRE(op(StreamSizeOp::kMax, 150, nullptr) == 150);
  REQUIRE(op(StreamSizeOp::kReserve, 10, &old_sz) == 160);
  REQUIRE(old_sz == 150);
  REQUIRE(op(StreamSizeOp::kGet, 0, nullptr) == 160);
  op(StreamSizeOp::kDrop, 0, nullptr);
  REQUIRE(op(StreamSizeOp::kGet, 0, nullptr) == 0);
}

TEST_CASE("Stream concurrent appends merge exactly once, in writer order",
          "[stream]") {
  StreamFixture fx;
  clio::cte::stream::Client s;
  TagId tag = MakeTag("stream_writers");
  std::vector<std::thread> ts;
  std::vector<int> failures(kWriters, 0);
  for (int w = 0; w < kWriters; ++w) {
    ts.emplace_back([&, w] {
      for (int i = 0; i < kPerWriter; ++i) {
        const std::string r = Record(w, i);
        if (s.Append(tag, kHome, r.data(), r.size()) != 0) failures[w]++;
      }
    });
  }
  for (auto &t : ts) t.join();
  for (int w = 0; w < kWriters; ++w) REQUIRE(failures[w] == 0);
  const clio::run::u64 want = kWriters * kPerWriter * kRecLen;
  REQUIRE(FlushOf(s, tag) == want);
  const std::string data = ReadStream(tag, want);
  std::map<int, int> next;  // writer -> next expected seq
  for (size_t off = 0; off < data.size(); off += kRecLen) {
    int w = -1, q = -1;
    REQUIRE(std::sscanf(data.c_str() + off, "W%02d S%05d", &w, &q) == 2);
    REQUIRE(data.compare(off, kRecLen, Record(w, q)) == 0);
    REQUIRE(q == next[w]);  // each writer's records in its own order, once
    next[w] = q + 1;
  }
  for (int w = 0; w < kWriters; ++w) REQUIRE(next[w] == kPerWriter);
}

TEST_CASE("Stream appends spanning page boundaries", "[stream]") {
  StreamFixture fx;
  clio::cte::stream::Client s;
  TagId tag = MakeTag("stream_pages");
  constexpr int kChunks = 5;
  constexpr size_t kChunk = 700 * 1024;
  std::string want;
  for (int c = 0; c < kChunks; ++c) {
    std::string chunk(kChunk, '\0');
    for (size_t i = 0; i < kChunk; ++i) {
      chunk[i] = static_cast<char>((c * 131 + i) % 251);
    }
    REQUIRE(s.Append(tag, kHome, chunk.data(), chunk.size()) == 0);
    want += chunk;
  }
  REQUIRE(FlushOf(s, tag) == want.size());
  REQUIRE(ReadStream(tag, want.size()) == want);
}

TEST_CASE("Stream append after truncate lands at the new end", "[stream]") {
  StreamFixture fx;
  clio::cte::stream::Client s;
  TagId tag = MakeTag("stream_trunc");
  const std::string a(1000, 'A');
  const std::string b(20, 'B');
  REQUIRE(s.Append(tag, kHome, a.data(), a.size()) == 0);
  REQUIRE(FlushOf(s, tag) == 1000);
  auto t = s.AsyncSizeOp(tag, kHome, StreamSizeOp::kSet, 10);
  t.Wait();
  REQUIRE(t->GetReturnCode() == 0);
  REQUIRE(s.Append(tag, kHome, b.data(), b.size()) == 0);
  REQUIRE(FlushOf(s, tag) == 30);
  REQUIRE(ReadStream(tag, 30) == std::string(10, 'A') + b);
}

TEST_CASE("Stream appends to a dropped stream are discarded", "[stream]") {
  StreamFixture fx;
  clio::cte::stream::Client s;
  auto *cte = CLIO_CTE_CLIENT;
  TagId tag = MakeTag("stream_dropped");
  auto d = s.AsyncSizeOp(tag, kHome, StreamSizeOp::kDrop);
  d.Wait();
  REQUIRE(d->GetReturnCode() == 0);
  const std::string x(4096, 'x');
  for (int i = 0; i < 10; ++i) {
    REQUIRE(s.Append(tag, kHome, x.data(), x.size()) == 0);
  }
  REQUIRE(FlushOf(s, tag) == 0);
  REQUIRE(SizeOf(s, tag) == 0);
  auto sz = cte->AsyncGetBlobSize(tag, "0");
  sz.Wait();
  REQUIRE((sz->GetReturnCode() != 0 || sz->size_ == 0));
  auto st = cte->AsyncGetOrCreateTag("_clio_stream_staging");
  st.Wait();
  char prefix[64];
  std::snprintf(prefix, sizeof(prefix), "sa.%x.%x.", tag.major_, tag.minor_);
  // Discarded staged bytes are reaped in the background: wait for it.
  bool clean = false;
  for (int attempt = 0; attempt < 100 && !clean; ++attempt) {
    auto blobs = cte->AsyncGetContainedBlobs(st->tag_id_);
    blobs.Wait();
    clean = true;
    for (const auto &n : blobs->blob_names_) {
      if (n.rfind(prefix, 0) == 0) clean = false;
    }
    if (!clean) std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  REQUIRE(clean);  // no staged bytes left behind
}

SIMPLE_TEST_MAIN()
