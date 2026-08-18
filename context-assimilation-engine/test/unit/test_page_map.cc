/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * PageMap: unified page numbering, including the PARTIAL and MIXED cases.
 *
 * The point of the extent form is that a consumer asks for "page N" and
 * never learns whether that page came from one blob, several, or part of
 * one -- so these tests pin exactly those shapes.
 */

#include "simple_test.h"

#include <clio_cae/core/factory/page_map.h>

#include <string>
#include <vector>

using clio::cae::core::PageExtent;
using clio::cae::core::PageMap;

namespace {
constexpr clio::run::u64 kPage = 1024;
}

TEST_CASE("PageMap: one blob per page (the aligned case)",
          "[cae][pagemap]") {
  std::vector<PageExtent> ex;
  for (int p = 0; p < 4; ++p) {
    ex.push_back(PageExtent{"b0_pi" + std::to_string(p), 0,
                            (clio::run::u64) p * kPage, kPage});
  }
  PageMap m = PageMap::Build(kPage, ex);
  REQUIRE(m.NumPages() == 4);
  auto s = m.ToPage(2);
  REQUIRE(s.size() == 1);
  REQUIRE(s[0].blob_name == "b0_pi2");
  REQUIRE(s[0].blob_off == 0);
  REQUIRE(s[0].dst_off == 0);
  REQUIRE(s[0].size == kPage);
}

TEST_CASE("PageMap: MIXED page -- several blobs share one page",
          "[cae][pagemap][mixed]") {
  // Three small blobs packed into page 0, a fourth spilling into page 1.
  PageMap m = PageMap::Build(
      kPage, {PageExtent{"a", 0, 0, 400},
              PageExtent{"b", 0, 400, 400},
              PageExtent{"c", 0, 800, 224},
              PageExtent{"d", 0, 1024, 512}});
  REQUIRE(m.NumPages() == 2);

  auto p0 = m.ToPage(0);
  REQUIRE(p0.size() == 3);                 // MIXED: three blobs in one page
  REQUIRE(p0[0].blob_name == "a");
  REQUIRE(p0[0].dst_off == 0);
  REQUIRE(p0[1].blob_name == "b");
  REQUIRE(p0[1].dst_off == 400);
  REQUIRE(p0[2].blob_name == "c");
  REQUIRE(p0[2].dst_off == 800);
  REQUIRE(p0[2].size == 224);

  auto p1 = m.ToPage(1);
  REQUIRE(p1.size() == 1);
  REQUIRE(p1[0].blob_name == "d");
  REQUIRE(p1[0].size == 512);              // PARTIAL: page half covered
}

TEST_CASE("PageMap: PARTIAL -- a blob spanning pages is split",
          "[cae][pagemap][partial]") {
  // One blob covering 2.5 pages starting mid-page.
  PageMap m = PageMap::Build(kPage, {PageExtent{"big", 0, 512, 2560}});
  REQUIRE(m.NumPages() == 3);              // bytes 512..3072 = 3 pages

  auto p0 = m.ToPage(0);
  REQUIRE(p0.size() == 1);
  REQUIRE(p0[0].dst_off == 512);           // second half of page 0
  REQUIRE(p0[0].blob_off == 0);
  REQUIRE(p0[0].size == 512);

  auto p1 = m.ToPage(1);
  REQUIRE(p1.size() == 1);
  REQUIRE(p1[0].dst_off == 0);             // full page from the middle
  REQUIRE(p1[0].blob_off == 512);
  REQUIRE(p1[0].size == kPage);

  auto p3 = m.ToPage(3);
  REQUIRE(p3.size() == 0);                 // past the last extent: empty
}

TEST_CASE("PageMap: FromPage inverts ToPage", "[cae][pagemap]") {
  PageMap m = PageMap::Build(kPage, {PageExtent{"big", 0, 512, 2560}});
  // Bytes 100..1200 of "big" land in pages 0, 1 and 2.
  auto refs = m.FromPage("big", 100, 1100);
  REQUIRE(refs.size() == 2);
  REQUIRE(refs[0].page_id == 0);
  REQUIRE(refs[0].dst_off == 612);         // 512 + 100
  REQUIRE(refs[0].size == 412);            // to the end of page 0
  REQUIRE(refs[1].page_id == 1);
  REQUIRE(refs[1].dst_off == 0);
  REQUIRE(refs[1].size == 688);
  // Round trip: every ref's bytes appear in that page's slices.
  for (const auto &r : refs) {
    bool found = false;
    for (const auto &s : m.ToPage(r.page_id)) {
      if (s.blob_name == "big" && r.dst_off >= s.dst_off &&
          r.dst_off + r.size <= s.dst_off + s.size) {
        found = true;
      }
    }
    REQUIRE(found);
  }
}

TEST_CASE("PageMap: serialize/parse round trip", "[cae][pagemap]") {
  PageMap m = PageMap::Build(
      kPage, {PageExtent{"a", 0, 0, 400}, PageExtent{"d", 16, 1024, 512}});
  PageMap back = PageMap::Parse(m.Serialize());
  REQUIRE(back.PageSize() == m.PageSize());
  REQUIRE(back.NumPages() == m.NumPages());
  REQUIRE(back.Extents().size() == m.Extents().size());
  REQUIRE(back.Extents()[1].blob_name == "d");
  REQUIRE(back.Extents()[1].blob_off == 16);
  auto s = back.ToPage(1);
  REQUIRE(s.size() == 1);
  REQUIRE(s[0].blob_off == 16);
}

SIMPLE_TEST_MAIN()
