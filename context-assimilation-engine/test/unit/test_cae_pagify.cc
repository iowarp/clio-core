/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * pagify through the CAE's CTE interceptor, end to end.
 *
 * test_pagify.cc pins the translation and assembly against a stub. This
 * pins the part that only a running system can show: a client asking a CAE
 * pool for "page:<id>" gets a constructed page back, built from blobs it
 * never names, with the pagemap travelling as a blob alongside the data.
 *
 * The shapes under test are the ones that make paging non-trivial: a page
 * assembled from SEVERAL blobs, and a page only PARTIALLY covered, whose
 * uncovered bytes must read as zero.
 */

#include "simple_test.h"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#include <clio_cae/core/constants.h>
#include <clio_cae/core/core_client.h>
#include <clio_cae/core/factory/page_map.h>
#include <clio_cte/core/core_client.h>
#include <clio_ctp/introspect/system_info.h>

namespace fs = std::filesystem;
using namespace std::chrono_literals;
using clio::cae::core::PageExtent;
using clio::cae::core::PageMap;

namespace {
constexpr clio::run::u64 kPage = 4096;

/** PutBlob a byte pattern under `name`. */
void PutFilled(clio::cte::core::Client *cte,
               const clio::cte::core::TagId &tag, const std::string &name,
               unsigned char fill, size_t size) {
  auto buf = CLIO_IPC->AllocateBuffer(size);
  REQUIRE(!buf.IsNull());
  std::memset(buf.ptr_, fill, size);
  auto put = cte->AsyncPutBlob(tag, name, 0, size,
                               buf.shm_.template Cast<void>(), 1.0f);
  put.Wait();
  REQUIRE(put->GetReturnCode() == 0);
}
}  // namespace

TEST_CASE("pagify: CAE serves page:<id> built from blobs the caller never "
          "names", "[cae][pagify][interpose]") {
  fs::path config_path =
      fs::path(__FILE__).parent_path() / "test_cae_pagify_config.yaml";
  ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", config_path.string(), 1);

  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kServer));
  SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;
  std::this_thread::sleep_for(1s);

  auto *cte = CLIO_CTE_CLIENT;
  cte->Init(clio::cte::core::kCtePoolId);

  auto tag_task = cte->AsyncGetOrCreateTag("pagify_data");
  tag_task.Wait();
  REQUIRE(tag_task->GetReturnCode() == 0);
  const auto tag_id = tag_task->tag_id_;
  REQUIRE(!tag_id.IsNull());

  // ---- data: page 0 is MIXED (three blobs), page 1 is PARTIAL ----------
  PutFilled(cte, tag_id, "a", 0xA1, 1024);
  PutFilled(cte, tag_id, "b", 0xB2, 2048);
  PutFilled(cte, tag_id, "c", 0xC3, 1024);
  PutFilled(cte, tag_id, "d", 0xD4, 1000);

  std::vector<PageExtent> ex = {
      PageExtent{"a", 0, 0, 1024},
      PageExtent{"b", 0, 1024, 2048},
      PageExtent{"c", 0, 3072, 1024},
      // Page 1: only 1000 bytes at offset 500 within the page.
      PageExtent{"d", 0, kPage + 500, 1000},
  };
  const std::string ser = PageMap::Build(kPage, ex).Serialize();

  // The map travels WITH the data, as a plain blob on the same tag.
  {
    auto buf = CLIO_IPC->AllocateBuffer(ser.size());
    REQUIRE(!buf.IsNull());
    std::memcpy(buf.ptr_, ser.data(), ser.size());
    auto put = cte->AsyncPutBlob(tag_id, "pagemap", 0, ser.size(),
                                 buf.shm_.template Cast<void>(), 1.0f);
    put.Wait();
    REQUIRE(put->GetReturnCode() == 0);
  }

  // ---- read pages BY NUMBER through the CAE pool -----------------------
  clio::cte::core::Client via_cae(clio::cae::core::kCaePoolId);

  // Page 0: three blobs, fully covered.
  {
    auto buf = CLIO_IPC->AllocateBuffer(kPage);
    REQUIRE(!buf.IsNull());
    std::memset(buf.ptr_, 0xEE, kPage);       // poison
    auto get = via_cae.AsyncGetBlob(tag_id, "page:0", 0, kPage, 0,
                                    buf.shm_.template Cast<void>());
    get.Wait();
    REQUIRE(get->GetReturnCode() == 0);
    const auto *p = reinterpret_cast<const unsigned char *>(buf.ptr_);
    REQUIRE(p[0] == 0xA1);
    REQUIRE(p[1023] == 0xA1);
    REQUIRE(p[1024] == 0xB2);
    REQUIRE(p[3071] == 0xB2);
    REQUIRE(p[3072] == 0xC3);
    REQUIRE(p[4095] == 0xC3);
  }

  // Page 1: partial coverage; everything outside the extent reads zero.
  {
    auto buf = CLIO_IPC->AllocateBuffer(kPage);
    REQUIRE(!buf.IsNull());
    std::memset(buf.ptr_, 0xEE, kPage);       // poison
    auto get = via_cae.AsyncGetBlob(tag_id, "page:1", 0, kPage, 0,
                                    buf.shm_.template Cast<void>());
    get.Wait();
    REQUIRE(get->GetReturnCode() == 0);
    const auto *p = reinterpret_cast<const unsigned char *>(buf.ptr_);
    REQUIRE(p[499] == 0x00);
    REQUIRE(p[500] == 0xD4);
    REQUIRE(p[1499] == 0xD4);
    REQUIRE(p[1500] == 0x00);
    REQUIRE(p[4095] == 0x00);
  }

  // A normally-named blob still forwards untouched: pagify is opt-in per
  // request, so an unpagified caller sees no behaviour change.
  {
    auto buf = CLIO_IPC->AllocateBuffer(1024);
    REQUIRE(!buf.IsNull());
    std::memset(buf.ptr_, 0, 1024);
    auto get = via_cae.AsyncGetBlob(tag_id, "a", 0, 1024, 0,
                                    buf.shm_.template Cast<void>());
    get.Wait();
    REQUIRE(get->GetReturnCode() == 0);
    REQUIRE(reinterpret_cast<const unsigned char *>(buf.ptr_)[0] == 0xA1);
  }
}

SIMPLE_TEST_MAIN()
