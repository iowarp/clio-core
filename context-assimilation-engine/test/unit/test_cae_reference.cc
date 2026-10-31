/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * CAE ingest-on-fault assimilation (AssimilationCtx::on_fault).
 *
 * format "reference" maps a file into a CTE tag WITHOUT copying it: the CAE
 * records the extent and becomes the tag's fault handler, so a GetBlob on
 * page N that the CTE does not hold is read from the file by the CAE. These
 * tests drive that path end to end through the CTE pool, the way the GPU
 * vector does, and check:
 *   - full pages, sub-page offsets and the partial last page (zero tail);
 *   - range_off / range_size select a window of the file;
 *   - no blob is ever stored in a CTE tier by assimilation itself;
 *   - a PutBlob makes a page resident and later reads see it (overlay);
 *   - names that are not a page index fail instead of reading garbage;
 *   - a missing source file fails the ParseOmni.
 */

#include "simple_test.h"


#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include <clio_cae/core/constants.h>
#include <clio_cae/core/core_client.h>
#include <clio_cae/core/factory/assimilation_ctx.h>
#include <clio_cte/core/core_client.h>
#include <clio_ctp/introspect/system_info.h>

namespace fs = std::filesystem;
using namespace std::chrono_literals;
using clio::cae::core::AssimilationCtx;

namespace {
constexpr clio::run::u64 kPage = 4096;
// 5 full pages plus a 1000-byte tail: the last page is partial.
constexpr clio::run::u64 kFileSize = 5 * kPage + 1000;

/** Byte value at absolute file offset `i` (never constant across a page). */
unsigned char Pattern(clio::run::u64 i) {
  return static_cast<unsigned char>((i * 131u + (i >> 12) * 7u + 3u) & 0xFF);
}

fs::path g_file;

void EnsureRuntime() {
  static bool up = false;
  if (up) return;
  fs::path config_path =
      fs::path(__FILE__).parent_path() / "test_cae_reference_config.yaml";
  ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", config_path.string(), 1);
  // The test tier is 32 MB: keep a reserve that fits it (default 512 MB).
  ctp::SystemInfo::Setenv("CLIO_CAE_FAULT_RESERVE_MB", "8", 1);
  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kServer));
  SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;
  std::this_thread::sleep_for(1s);
  CLIO_CTE_CLIENT->Init(clio::cte::core::kCtePoolId);

  g_file = fs::temp_directory_path() /
           ("cae_ref_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".bin");
  std::ofstream out(g_file, std::ios::binary);
  std::vector<char> buf(kFileSize);
  for (clio::run::u64 i = 0; i < kFileSize; ++i) buf[i] = (char)Pattern(i);
  out.write(buf.data(), (std::streamsize)buf.size());
  out.close();
  REQUIRE(fs::file_size(g_file) == kFileSize);
  up = true;
}

/** ParseOmni one by-reference context; returns result_code_. */
clio::run::u32 AssimilateRef(const std::string &src, const std::string &tag,
                             clio::run::u64 off = 0, clio::run::u64 size = 0) {
  clio::cae::core::Client cae(clio::cae::core::kCaePoolId);
  std::vector<AssimilationCtx> ctxs(1);
  ctxs[0].src = "file::" + src;
  ctxs[0].dst = "iowarp::" + tag;
  ctxs[0].format = "binary";
  ctxs[0].range_off = off;
  ctxs[0].range_size = size;
  ctxs[0].on_fault = true;
  ctxs[0].chunk_size = kPage;
  ctxs[0].blob_prefix = "";
  auto f = cae.AsyncParseOmni(ctxs);
  f.Wait();
  return f->GetReturnCode() != 0 ? f->GetReturnCode() : f->result_code_;
}

clio::cte::core::TagId TagOf(const std::string &tag) {
  auto t = CLIO_CTE_CLIENT->AsyncGetOrCreateTag(tag);
  t.Wait();
  REQUIRE(t->GetReturnCode() == 0);
  return t->tag_id_;
}

struct Read {
  clio::run::u32 rc;
  std::vector<unsigned char> data;
};

/** GetBlob through the CTE pool, exactly like a GPU-vector page miss. */
Read GetPage(const clio::cte::core::TagId &tag, const std::string &name,
             clio::run::u64 off, clio::run::u64 size) {
  auto buf = CLIO_IPC->AllocateBuffer(size);
  REQUIRE(!buf.IsNull());
  std::memset(buf.ptr_, 0xEE, size);  // poison
  auto g = CLIO_CTE_CLIENT->AsyncGetBlob(tag, name, off, size, 0,
                                         buf.shm_.template Cast<void>());
  g.Wait();
  Read r{g->GetReturnCode(), {}};
  const auto *p = reinterpret_cast<const unsigned char *>(buf.ptr_);
  r.data.assign(p, p + size);
  return r;
}

bool MatchesFile(const std::vector<unsigned char> &d, clio::run::u64 file_off) {
  for (size_t i = 0; i < d.size(); ++i) {
    const clio::run::u64 fo = file_off + i;
    const unsigned char want = fo < kFileSize ? Pattern(fo) : 0;
    if (d[i] != want) return false;
  }
  return true;
}
}  // namespace

TEST_CASE("reference: pages are read from the source file on a CTE miss",
          "[cae][reference]") {
  EnsureRuntime();
  REQUIRE(AssimilateRef(g_file.string(), "ref_full") == 0);
  const auto tag = TagOf("ref_full");

  // Every full page, read twice (a fault does not make it resident).
  for (int pass = 0; pass < 2; ++pass) {
    for (clio::run::u64 pg = 0; pg < 5; ++pg) {
      Read r = GetPage(tag, std::to_string(pg), 0, kPage);
      REQUIRE(r.rc == 0);
      REQUIRE(MatchesFile(r.data, pg * kPage));
    }
  }

  // Sub-page offset and size inside page 2.
  {
    Read r = GetPage(tag, "2", 100, 333);
    REQUIRE(r.rc == 0);
    REQUIRE(MatchesFile(r.data, 2 * kPage + 100));
  }

  // Partial last page: 1000 file bytes, then zeros (MatchesFile expects 0
  // past EOF).
  {
    Read r = GetPage(tag, "5", 0, kPage);
    REQUIRE(r.rc == 0);
    REQUIRE(MatchesFile(r.data, 5 * kPage));
    REQUIRE(r.data[999] == Pattern(5 * kPage + 999));
    REQUIRE(r.data[1000] == 0);
  }

  // Wholly past the extent: zeros, not an error (the pager may round up).
  {
    Read r = GetPage(tag, "9", 0, kPage);
    REQUIRE(r.rc == 0);
    for (unsigned char c : r.data) REQUIRE(c == 0);
  }
}

TEST_CASE("on_fault: nothing is ingested until a blob faults",
          "[cae][reference][onfault]") {
  EnsureRuntime();
  REQUIRE(AssimilateRef(g_file.string(), "ref_nocopy") == 0);
  const auto tag = TagOf("ref_nocopy");
  auto size_of = [&]() {
    auto sz = CLIO_CTE_CLIENT->AsyncGetTagSize(tag);
    sz.Wait();
    REQUIRE(sz->GetReturnCode() == 0);
    return sz->tag_size_;
  };
  REQUIRE(size_of() == 0);  // assimilation itself ingests nothing
  REQUIRE(GetPage(tag, "1", 0, kPage).rc == 0);
  REQUIRE(size_of() == kPage);  // the faulted blob was ingested ...
  Read again = GetPage(tag, "1", 0, kPage);
  REQUIRE(again.rc == 0);  // ... and serves the next read
  REQUIRE(MatchesFile(again.data, kPage));
  REQUIRE(size_of() == kPage);  // only the blob that faulted
}

TEST_CASE("reference: range_off/range_size select a window of the file",
          "[cae][reference][range]") {
  EnsureRuntime();
  const clio::run::u64 base = kPage + 512;  // deliberately not page aligned
  REQUIRE(AssimilateRef(g_file.string(), "ref_range", base, 2 * kPage) == 0);
  const auto tag = TagOf("ref_range");

  Read p0 = GetPage(tag, "0", 0, kPage);
  REQUIRE(p0.rc == 0);
  REQUIRE(MatchesFile(p0.data, base));

  Read p1 = GetPage(tag, "1", 0, kPage);
  REQUIRE(p1.rc == 0);
  REQUIRE(MatchesFile(p1.data, base + kPage));

  // Page 2 is past range_size: zeros even though the FILE has bytes there.
  Read p2 = GetPage(tag, "2", 0, kPage);
  REQUIRE(p2.rc == 0);
  for (unsigned char c : p2.data) REQUIRE(c == 0);
}

TEST_CASE("reference: a written page overlays the source",
          "[cae][reference][overlay]") {
  EnsureRuntime();
  REQUIRE(AssimilateRef(g_file.string(), "ref_overlay") == 0);
  const auto tag = TagOf("ref_overlay");

  auto buf = CLIO_IPC->AllocateBuffer(kPage);
  REQUIRE(!buf.IsNull());
  std::memset(buf.ptr_, 0x5A, kPage);
  auto put = CLIO_CTE_CLIENT->AsyncPutBlob(tag, "3", 0, kPage,
                                           buf.shm_.template Cast<void>(),
                                           1.0f);
  put.Wait();
  REQUIRE(put->GetReturnCode() == 0);

  Read w = GetPage(tag, "3", 0, kPage);
  REQUIRE(w.rc == 0);
  for (unsigned char c : w.data) REQUIRE(c == 0x5A);

  // Neighbours still come from the file, and the file is untouched.
  Read n = GetPage(tag, "2", 0, kPage);
  REQUIRE(n.rc == 0);
  REQUIRE(MatchesFile(n.data, 2 * kPage));
  std::ifstream in(g_file, std::ios::binary);
  in.seekg(3 * kPage);
  char c0 = 0;
  in.read(&c0, 1);
  REQUIRE((unsigned char)c0 == Pattern(3 * kPage));
}

TEST_CASE("reference: a partial write keeps the source bytes around it",
          "[cae][reference][overlay][partial]") {
  EnsureRuntime();
  REQUIRE(AssimilateRef(g_file.string(), "ref_partial") == 0);
  const auto tag = TagOf("ref_partial");

  // 16 bytes into the middle of page 4. CTE first asks CAE to materialise
  // the page from the file, then applies the write on top.
  constexpr clio::run::u64 kOff = 1000, kLen = 16;
  auto buf = CLIO_IPC->AllocateBuffer(kLen);
  REQUIRE(!buf.IsNull());
  std::memset(buf.ptr_, 0x77, kLen);
  auto put = CLIO_CTE_CLIENT->AsyncPutBlob(tag, "4", kOff, kLen,
                                           buf.shm_.template Cast<void>(),
                                           1.0f);
  put.Wait();
  REQUIRE(put->GetReturnCode() == 0);

  Read r = GetPage(tag, "4", 0, kPage);
  REQUIRE(r.rc == 0);
  for (clio::run::u64 i = 0; i < kPage; ++i) {
    const unsigned char want =
        (i >= kOff && i < kOff + kLen) ? 0x77 : Pattern(4 * kPage + i);
    REQUIRE(r.data[i] == want);
  }
  // Exactly the written page became resident -- one page, not the file.
  auto sz = CLIO_CTE_CLIENT->AsyncGetTagSize(tag);
  sz.Wait();
  REQUIRE(sz->tag_size_ == kPage);
}

TEST_CASE("reference: non-page blob names fail cleanly",
          "[cae][reference][error]") {
  EnsureRuntime();
  REQUIRE(AssimilateRef(g_file.string(), "ref_names") == 0);
  const auto tag = TagOf("ref_names");
  REQUIRE(GetPage(tag, "not_a_page", 0, 64).rc != 0);
  REQUIRE(GetPage(tag, "1_pi3", 0, 64).rc != 0);
}

TEST_CASE("on_fault: a missing source fails the read that faults",
          "[cae][reference][error]") {
  EnsureRuntime();
  // Nothing is read at assimilation time, so the setup succeeds ...
  REQUIRE(AssimilateRef("/nonexistent/cae_ref_missing.bin", "ref_missing") == 0);
  // ... and the first fault reports the missing source.
  REQUIRE(GetPage(TagOf("ref_missing"), "0", 0, 64).rc != 0);
}

TEST_CASE("on_fault: ingestion never takes the tier's last space",
          "[cae][reference][onfault][headroom]") {
  // A 40 MB source through a 32 MB tier with an 8 MB reserve: reads must
  // keep working, the ingested blobs must stop short of the reserve, and an
  // unrelated (non-re-ingestable) write must still find space. Before the
  // headroom rule, ingestion filled the tier and a 16-node job deadlocked on
  // its own collective's put.
  EnsureRuntime();
  constexpr clio::run::u64 kBig = 40ull << 20, kChunk = 64ull << 10;
  const fs::path big = fs::temp_directory_path() /
      ("cae_ref_big_" + std::to_string(
          std::chrono::steady_clock::now().time_since_epoch().count()) + ".bin");
  {
    std::ofstream out(big, std::ios::binary);
    std::vector<char> buf(kChunk);
    for (clio::run::u64 o = 0; o < kBig; o += kChunk) {
      for (clio::run::u64 i = 0; i < kChunk; ++i) buf[i] = (char)Pattern(o + i);
      out.write(buf.data(), (std::streamsize)kChunk);
    }
  }
  clio::cae::core::Client cae(clio::cae::core::kCaePoolId);
  std::vector<AssimilationCtx> ctxs(1);
  ctxs[0].src = "file::" + big.string();
  ctxs[0].dst = "iowarp::ref_headroom";
  ctxs[0].format = "binary";
  ctxs[0].on_fault = true;
  ctxs[0].chunk_size = kChunk;
  ctxs[0].blob_prefix = "";
  auto f = cae.AsyncParseOmni(ctxs);
  f.Wait();
  REQUIRE(f->GetReturnCode() == 0);
  REQUIRE(f->result_code_ == 0);
  const auto tag = TagOf("ref_headroom");

  for (clio::run::u64 pg = 0; pg < kBig / kChunk; ++pg) {
    Read r = GetPage(tag, std::to_string(pg), 0, kChunk);
    REQUIRE(r.rc == 0);
    REQUIRE(r.data[0] == Pattern(pg * kChunk));
    REQUIRE(r.data[kChunk - 1] == Pattern(pg * kChunk + kChunk - 1));
  }
  auto sz = CLIO_CTE_CLIENT->AsyncGetTagSize(tag);
  sz.Wait();
  REQUIRE(sz->tag_size_ > 0);              // it did ingest ...
  REQUIRE(sz->tag_size_ <= (24ull << 20));  // ... but kept the 8 MB reserve

  // The space it left is usable by everyone else.
  const auto other = TagOf("ref_headroom_other");
  auto buf = CLIO_IPC->AllocateBuffer(1ull << 20);
  REQUIRE(!buf.IsNull());
  std::memset(buf.ptr_, 0x33, 1ull << 20);
  auto put = CLIO_CTE_CLIENT->AsyncPutBlob(other, "collective", 0, 1ull << 20,
                                           buf.shm_.template Cast<void>(),
                                           1.0f);
  put.Wait();
  REQUIRE(put->GetReturnCode() == 0);
  // Best effort: the assimilator keeps its source open between faults, and
  // Windows refuses to delete an open file.
  std::error_code ec;
  fs::remove(big, ec);
}

SIMPLE_TEST_MAIN()
