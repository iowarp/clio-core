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
 * The pure logic of the FUSE adapter's sequential read-ahead
 * (adapter/libfuse/fuse_readahead.h), exercised without a runtime or a
 * mount: the per-handle sequential detector, the read-ahead window, and the
 * page arithmetic at page boundaries and at end of file.
 */
#include <cstdint>

#include "../../../../../context-runtime/test/simple_test.h"
#include "../../../../adapter/libfuse/fuse_readahead.h"

using namespace clio::cae::fuse;

namespace {
constexpr uint64_t kPage = 1ull << 20;  // the clio-fs page (kFsPageSize)
constexpr uint64_t kReq = 512ull << 10;  // the kernel's read-ahead request
}  // namespace

TEST_CASE("readahead detector: needs two sequential reads", "[fuse][ra]") {
  RaSeqDetector d;
  REQUIRE_FALSE(d.Active());
  REQUIRE(d.End() == 0);
  REQUIRE(d.Observe(0, kReq, kPage) == RaSeq::kRandom);  // first read
  REQUIRE_FALSE(d.Active());
  REQUIRE(d.Observe(kReq, kReq, kPage) == RaSeq::kSequential);
  REQUIRE_FALSE(d.Active());
  REQUIRE(d.Observe(2 * kReq, kReq, kPage) == RaSeq::kSequential);
  REQUIRE(d.Active());
  REQUIRE(d.End() == 3 * kReq);
}

TEST_CASE("readahead detector: random access never activates", "[fuse][ra]") {
  // ior-hard style: 47008-byte records at a stride of many pages.
  RaSeqDetector d;
  const uint64_t rec = 47008, stride = 97 * rec;
  for (uint64_t i = 0; i < 1000; ++i) {
    REQUIRE(d.Observe(i * stride, rec, kPage) == RaSeq::kRandom);
    REQUIRE_FALSE(d.Active());
  }
  // Backward jumps and far forward jumps alike.
  REQUIRE(d.Observe(5 * kPage, rec, kPage) == RaSeq::kRandom);
  REQUIRE(d.Observe(900 * kPage, rec, kPage) == RaSeq::kRandom);
  REQUIRE_FALSE(d.Active());
}

TEST_CASE("readahead detector: a random read resets the streak", "[fuse][ra]") {
  RaSeqDetector d;
  d.Observe(0, kReq, kPage);
  d.Observe(kReq, kReq, kPage);
  d.Observe(2 * kReq, kReq, kPage);
  REQUIRE(d.Active());
  REQUIRE(d.Observe(100 * kPage, kReq, kPage) == RaSeq::kRandom);
  REQUIRE_FALSE(d.Active());
  REQUIRE(d.End() == 100 * kPage + kReq);
  // One sequential read after the jump is not enough again.
  REQUIRE(d.Observe(100 * kPage + kReq, kReq, kPage) == RaSeq::kSequential);
  REQUIRE_FALSE(d.Active());
  REQUIRE(d.Observe(101 * kPage, kReq, kPage) == RaSeq::kSequential);
  REQUIRE(d.Active());
}

TEST_CASE("readahead detector: out-of-order kernel requests", "[fuse][ra]") {
  // Requests R0..R5 of 512 KiB; arrival order R0 R2 R1 R3 R5 R4.
  RaSeqDetector d;
  REQUIRE(d.Observe(0 * kReq, kReq, kPage) == RaSeq::kRandom);
  // R2 starts in the page after the one R0 ended in: sequential.
  REQUIRE(d.Observe(2 * kReq, kReq, kPage) == RaSeq::kSequential);
  REQUIRE(d.End() == 3 * kReq);
  // R1 is one page behind the stream's last page: a straggler, ignored.
  REQUIRE(d.Observe(1 * kReq, kReq, kPage) == RaSeq::kStraggler);
  REQUIRE(d.End() == 3 * kReq);
  REQUIRE(d.Observe(3 * kReq, kReq, kPage) == RaSeq::kSequential);
  REQUIRE(d.Active());
  // R5 before R4: R5 is in the next page; R4 then starts in the page R5
  // ended in (page 2), so it is sequential too.
  REQUIRE(d.Observe(5 * kReq, kReq, kPage) == RaSeq::kSequential);
  REQUIRE(d.End() == 6 * kReq);
  REQUIRE(d.Observe(4 * kReq, kReq, kPage) == RaSeq::kSequential);
  REQUIRE(d.Active());
  REQUIRE(d.End() == 6 * kReq);  // the end never moves back
  // A late request inside the stream's last page is sequential and does not
  // pull the end back either.
  REQUIRE(d.Observe(5 * kReq + 4096, 4096, kPage) == RaSeq::kSequential);
  REQUIRE(d.End() == 6 * kReq);
  // Two pages behind is not a straggler.
  REQUIRE(d.Observe(0, kReq, kPage) == RaSeq::kRandom);
  REQUIRE_FALSE(d.Active());
}

TEST_CASE("readahead detector: page-boundary ends", "[fuse][ra]") {
  // A read ending exactly on a page boundary ended IN the page before it.
  RaSeqDetector d;
  d.Observe(0, kPage, kPage);  // ends at 1 MiB: last byte in page 0
  // Page 1 is the next page; page 2 is not.
  REQUIRE(d.Observe(2 * kPage, 4096, kPage) == RaSeq::kRandom);
  RaSeqDetector e;
  e.Observe(0, kPage, kPage);
  REQUIRE(e.Observe(2 * kPage - 1, 1, kPage) == RaSeq::kSequential);
  // Exact continuation is sequential whatever the pages.
  RaSeqDetector f;
  f.Observe(10 * kPage, 3 * kPage, kPage);
  REQUIRE(f.Observe(13 * kPage, kPage, kPage) == RaSeq::kSequential);
}

TEST_CASE("readahead window: pages kept ahead, cut at EOF", "[fuse][ra]") {
  // Stream ended at 1.5 MiB (in page 1); 4 ahead; 100 MiB file.
  RaWindow w = RaWindowFor(3 * kReq, 100 * kPage, 4, kPage);
  REQUIRE_FALSE(w.Empty());
  REQUIRE(w.first == 1);
  REQUIRE(w.last == 5);
  REQUIRE(w.Has(1));
  REQUIRE(w.Has(5));
  REQUIRE_FALSE(w.Has(0));
  REQUIRE_FALSE(w.Has(6));
  // An end on a page boundary belongs to the page before it.
  w = RaWindowFor(2 * kPage, 100 * kPage, 4, kPage);
  REQUIRE(w.first == 1);
  REQUIRE(w.last == 5);
  // Cut at the file's last page: 3 MiB + 1 byte has pages 0..3.
  w = RaWindowFor(kPage + 1, 3 * kPage + 1, 8, kPage);
  REQUIRE(w.first == 1);
  REQUIRE(w.last == 3);
  // A file of exactly 3 pages has pages 0..2.
  w = RaWindowFor(kPage + 1, 3 * kPage, 8, kPage);
  REQUIRE(w.last == 2);
  // Stream at EOF: only its own page.
  w = RaWindowFor(3 * kPage, 3 * kPage, 8, kPage);
  REQUIRE(w.first == 2);
  REQUIRE(w.last == 2);
  // Nothing to keep: no stream, empty file, or the file shrank under it.
  REQUIRE(RaWindowFor(0, 3 * kPage, 8, kPage).Empty());
  REQUIRE(RaWindowFor(4096, 0, 8, kPage).Empty());
  REQUIRE(RaWindowFor(5 * kPage, 3 * kPage, 8, kPage).Empty());
  // The window never holds more than ahead + 1 pages.
  for (uint32_t n = 1; n <= kRaMaxPages; ++n) {
    w = RaWindowFor(7 * kPage + 5, 1000 * kPage, n, kPage);
    REQUIRE(w.last - w.first + 1 == n + 1);
  }
}

TEST_CASE("readahead page length: never past the file size", "[fuse][ra]") {
  REQUIRE(RaPageLen(0, 10 * kPage, kPage) == kPage);
  REQUIRE(RaPageLen(9, 10 * kPage, kPage) == kPage);
  REQUIRE(RaPageLen(10, 10 * kPage, kPage) == 0);  // at EOF
  REQUIRE(RaPageLen(11, 10 * kPage, kPage) == 0);  // past EOF
  REQUIRE(RaPageLen(3, 3 * kPage + 1, kPage) == 1);  // 1-byte last page
  REQUIRE(RaPageLen(3, 3 * kPage + 4097, kPage) == 4097);
  REQUIRE(RaPageLen(0, 100, kPage) == 100);  // file smaller than a page
  REQUIRE(RaPageLen(0, 0, kPage) == 0);      // empty file
}

TEST_CASE("readahead cover: a part must lie in the fetched bytes",
          "[fuse][ra]") {
  REQUIRE(RaCovers(kPage, 0, kPage));
  REQUIRE(RaCovers(kPage, kReq, kReq));
  REQUIRE(RaCovers(kPage, kPage - 1, 1));
  REQUIRE_FALSE(RaCovers(kPage, kPage, 1));
  REQUIRE_FALSE(RaCovers(kPage, kReq, kReq + 1));
  // Short last page fetched with 4097 bytes: a part past them (the file
  // grew since) is not served from it.
  REQUIRE(RaCovers(4097, 0, 4097));
  REQUIRE(RaCovers(4097, 4096, 1));
  REQUIRE_FALSE(RaCovers(4097, 0, 4098));
  REQUIRE_FALSE(RaCovers(4097, 4097, 1));
  REQUIRE_FALSE(RaCovers(4097, 8192, 1));
  REQUIRE_FALSE(RaCovers(kPage, 0, 0));  // an empty part is never a hit
  REQUIRE_FALSE(RaCovers(0, 0, 1));
  // No overflow on absurd inputs.
  REQUIRE_FALSE(RaCovers(kPage, 1, ~0ull));
  REQUIRE_FALSE(RaCovers(kPage, ~0ull, 2));
}

TEST_CASE("readahead: a read spanning pages splits like the read loop",
          "[fuse][ra]") {
  // The read loop cuts [off, off+want) at page boundaries; each part must
  // be covered by a full-length page, and the last by the EOF-short page.
  const uint64_t fsize = 5 * kPage + 1000;
  uint64_t off = 4 * kPage - 100, want = kPage + 1100;  // ends at EOF
  uint64_t done = 0, parts = 0;
  while (done < want) {
    const uint64_t cur = off + done, page_off = cur % kPage;
    const uint64_t n = std::min(kPage - page_off, want - done);
    REQUIRE(RaCovers(RaPageLen(cur / kPage, fsize, kPage), page_off, n));
    done += n;
    ++parts;
  }
  REQUIRE(parts == 3);
  REQUIRE(off + done == fsize);
  // One byte more would leave the last page's fetched bytes.
  REQUIRE_FALSE(RaCovers(RaPageLen(5, fsize, kPage), 0, 1001));
}

TEST_CASE("readahead setting: CLIO_FUSE_READAHEAD_PAGES", "[fuse][ra]") {
  REQUIRE(RaParsePages(nullptr) == 0);
  REQUIRE(RaParsePages("") == 0);
  REQUIRE(RaParsePages("0") == 0);
  REQUIRE(RaParsePages("-3") == 0);
  REQUIRE(RaParsePages("abc") == 0);
  REQUIRE(RaParsePages("1") == 1);
  REQUIRE(RaParsePages("8") == 8);
  REQUIRE(RaParsePages("64") == kRaMaxPages);
  REQUIRE(RaParsePages("100000") == kRaMaxPages);
}

SIMPLE_TEST_MAIN()
