/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * Pagify: constructing a page from a page NUMBER alone.
 *
 * PageMap already pins the arithmetic (test_page_map.cc). What matters
 * here is the construction: the caller passes a page id and gets a fully
 * defined page back, whether that page came from one blob, several, part
 * of one, or a region nothing maps to. The stub reader stands in for the
 * CTE so the assembly is tested without a runtime.
 */

#include "simple_test.h"

#include <clio_cae/core/factory/pagify.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

using clio::cae::core::PageExtent;
using clio::cae::core::PageIoResult;
using clio::cae::core::PageMap;
using clio::cae::core::Pagify;

namespace {

constexpr clio::run::u64 kPage = 1024;

/** In-memory stand-in for the blob store. */
class FakeBlobs {
 public:
  void Put(const std::string &name, unsigned char fill, size_t size) {
    blobs_[name] = std::vector<unsigned char>(size, fill);
  }
  std::vector<unsigned char> &Get(const std::string &name) {
    return blobs_[name];
  }

  clio::cae::core::BlobReadFn Reader() {
    return [this](const std::string &n, clio::run::u64 off,
                  clio::run::u64 size, void *dst) -> clio::run::u64 {
      auto it = blobs_.find(n);
      if (it == blobs_.end()) return 0;
      if (off + size > it->second.size()) return 0;
      std::memcpy(dst, it->second.data() + off, (size_t) size);
      ++reads_;
      return size;
    };
  }

  clio::cae::core::BlobWriteFn Writer() {
    return [this](const std::string &n, clio::run::u64 off,
                  clio::run::u64 size, const void *src) -> clio::run::u64 {
      auto it = blobs_.find(n);
      if (it == blobs_.end()) return 0;
      if (off + size > it->second.size()) return 0;
      std::memcpy(it->second.data() + off, src, (size_t) size);
      return size;
    };
  }

  int reads() const { return reads_; }

 private:
  std::map<std::string, std::vector<unsigned char>> blobs_;
  int reads_ = 0;
};

}  // namespace

TEST_CASE("Pagify: one blob per page", "[cae][pagify]") {
  FakeBlobs fb;
  std::vector<PageExtent> ex;
  for (int p = 0; p < 3; ++p) {
    const std::string n = "b0_pi" + std::to_string(p);
    fb.Put(n, (unsigned char) (0x10 + p), kPage);
    ex.push_back(PageExtent{n, 0, (clio::run::u64) p * kPage, kPage});
  }
  Pagify pg(PageMap::Build(kPage, ex));

  std::vector<unsigned char> page(kPage, 0xAA);
  PageIoResult r = pg.ReadPage(1, page.data(), fb.Reader());
  REQUIRE(r.ok);
  REQUIRE(r.slices == 1);
  REQUIRE(r.bytes == kPage);
  REQUIRE(r.zero_filled == 0);
  for (clio::run::u64 i = 0; i < kPage; ++i) REQUIRE(page[i] == 0x11);
}

TEST_CASE("Pagify: MIXED page is assembled from several blobs",
          "[cae][pagify]") {
  // Three blobs share one page: 256 + 512 + 256.
  FakeBlobs fb;
  fb.Put("a", 0xA1, 256);
  fb.Put("b", 0xB2, 512);
  fb.Put("c", 0xC3, 256);
  std::vector<PageExtent> ex = {
      PageExtent{"a", 0, 0, 256},
      PageExtent{"b", 0, 256, 512},
      PageExtent{"c", 0, 768, 256},
  };
  Pagify pg(PageMap::Build(kPage, ex));

  std::vector<unsigned char> page(kPage, 0);
  PageIoResult r = pg.ReadPage(0, page.data(), fb.Reader());
  REQUIRE(r.ok);
  REQUIRE(r.slices == 3);
  REQUIRE(r.bytes == kPage);
  REQUIRE(r.zero_filled == 0);
  REQUIRE(page[0] == 0xA1);
  REQUIRE(page[255] == 0xA1);
  REQUIRE(page[256] == 0xB2);
  REQUIRE(page[767] == 0xB2);
  REQUIRE(page[768] == 0xC3);
  REQUIRE(page[1023] == 0xC3);
}

TEST_CASE("Pagify: PARTIAL coverage zero-fills the rest", "[cae][pagify]") {
  // A blob covering only the middle 300 bytes of the page. Everything
  // outside it must read as zero, not as leftover buffer contents.
  FakeBlobs fb;
  fb.Put("mid", 0x77, 300);
  std::vector<PageExtent> ex = {PageExtent{"mid", 0, 400, 300}};
  Pagify pg(PageMap::Build(kPage, ex));

  std::vector<unsigned char> page(kPage, 0xEE);   // poison
  PageIoResult r = pg.ReadPage(0, page.data(), fb.Reader());
  REQUIRE(r.ok);
  REQUIRE(r.slices == 1);
  REQUIRE(r.bytes == 300);
  REQUIRE(r.zero_filled == kPage - 300);
  REQUIRE(page[399] == 0x00);
  REQUIRE(page[400] == 0x77);
  REQUIRE(page[699] == 0x77);
  REQUIRE(page[700] == 0x00);
}

TEST_CASE("Pagify: a blob spanning pages is split across them",
          "[cae][pagify]") {
  // One 2560-byte blob starting mid-page: touches pages 0, 1 and 2.
  FakeBlobs fb;
  fb.Put("big", 0x5A, 2560);
  std::vector<PageExtent> ex = {PageExtent{"big", 0, 512, 2560}};
  Pagify pg(PageMap::Build(kPage, ex));

  std::vector<unsigned char> p0(kPage, 0), p1(kPage, 0), p2(kPage, 0);
  REQUIRE(pg.ReadPage(0, p0.data(), fb.Reader()).ok);
  REQUIRE(pg.ReadPage(1, p1.data(), fb.Reader()).ok);
  REQUIRE(pg.ReadPage(2, p2.data(), fb.Reader()).ok);

  REQUIRE(p0[511] == 0x00);      // before the blob starts
  REQUIRE(p0[512] == 0x5A);
  for (clio::run::u64 i = 0; i < kPage; ++i) REQUIRE(p1[i] == 0x5A);
  REQUIRE(p2[1023] == 0x5A);     // 512 + 2560 == 3072, exactly 3 pages

  // And the inverse direction agrees on which pages hold those bytes.
  auto pages = pg.PagesFor("big", 0, 2560);
  REQUIRE(pages.size() == 3);
  REQUIRE(pages[0] == 0);
  REQUIRE(pages[1] == 1);
  REQUIRE(pages[2] == 2);
}

TEST_CASE("Pagify: write-back reaches the composing blobs",
          "[cae][pagify]") {
  FakeBlobs fb;
  fb.Put("a", 0x00, 256);
  fb.Put("b", 0x00, 768);
  std::vector<PageExtent> ex = {
      PageExtent{"a", 0, 0, 256},
      PageExtent{"b", 0, 256, 768},
  };
  Pagify pg(PageMap::Build(kPage, ex));

  std::vector<unsigned char> page(kPage);
  for (clio::run::u64 i = 0; i < kPage; ++i) {
    page[i] = (unsigned char) (i & 0xFF);
  }
  PageIoResult w = pg.WritePage(0, page.data(), fb.Writer());
  REQUIRE(w.ok);
  REQUIRE(w.bytes == kPage);

  // Read it straight back through the same map.
  std::vector<unsigned char> back(kPage, 0);
  REQUIRE(pg.ReadPage(0, back.data(), fb.Reader()).ok);
  REQUIRE(std::memcmp(back.data(), page.data(), kPage) == 0);
}

TEST_CASE("Pagify: a failed slice fails the page", "[cae][pagify]") {
  // A map naming a blob the store does not have must NOT report success
  // with a half-built page.
  FakeBlobs fb;
  fb.Put("present", 0x01, 512);
  std::vector<PageExtent> ex = {
      PageExtent{"present", 0, 0, 512},
      PageExtent{"missing", 0, 512, 512},
  };
  Pagify pg(PageMap::Build(kPage, ex));

  std::vector<unsigned char> page(kPage, 0);
  PageIoResult r = pg.ReadPage(0, page.data(), fb.Reader());
  REQUIRE(!r.ok);
}

SIMPLE_TEST_MAIN()
