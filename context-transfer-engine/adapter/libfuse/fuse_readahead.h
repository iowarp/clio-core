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
 * Pure logic of the FUSE adapter's opt-in sequential read-ahead
 * (CLIO_FUSE_READAHEAD_PAGES): the per-handle sequential detector and the
 * page arithmetic of the read-ahead window. No runtime types, so it is
 * unit-tested standalone (test/unit/adapters/libfuse/test_fuse_readahead.cc);
 * the cache that owns buffers and fetches lives in fuse_cte.cc.
 */

#ifndef CLIO_CTE_ADAPTER_LIBFUSE_FUSE_READAHEAD_H_
#define CLIO_CTE_ADAPTER_LIBFUSE_FUSE_READAHEAD_H_

#include <algorithm>
#include <cstdint>
#include <cstdlib>

namespace clio::cae::fuse {

/** Consecutive sequential reads a handle must show before read-ahead. */
static constexpr uint32_t kRaMinStreak = 2;
/** Largest CLIO_FUSE_READAHEAD_PAGES honoured (pages kept ahead per handle). */
static constexpr uint32_t kRaMaxPages = 64;

/** How one read relates to the read before it on the same handle. */
enum class RaSeq {
  kSequential,  ///< continues the stream: counts towards the streak
  kStraggler,   ///< a late request one page behind: ignored, nothing changes
  kRandom       ///< anything else (also a handle's first read): streak reset
};

/**
 * Sequential-read detector of one open handle.
 *
 * With E the end (exclusive) of the stream so far and Pe the page holding
 * its last byte ((E - 1) / page), a read starting at `off` is
 *  - sequential when off == E, or page(off) == Pe, or page(off) == Pe + 1
 *    (the kernel's read-ahead requests for one file arrive on several
 *    threads, slightly out of order);
 *  - a straggler when page(off) == Pe - 1: a late request of the same
 *    stream that crossed a page boundary behind its successor. It neither
 *    counts nor resets;
 *  - random otherwise, and for the first read of the handle.
 * Read-ahead is active once kRaMinStreak sequential reads came in a row.
 * Not thread safe: the owner serializes calls.
 */
class RaSeqDetector {
 public:
  /**
   * Classify a read and update the stream state.
   * @param off first byte of the read
   * @param len bytes of the read (> 0)
   * @param page_size bytes per page (> 0)
   * @return how the read relates to the previous one
   */
  RaSeq Observe(uint64_t off, uint64_t len, uint64_t page_size) {
    const uint64_t end = off + len;
    if (!has_prev_) {
      has_prev_ = true;
      end_ = end;
      streak_ = 0;
      return RaSeq::kRandom;
    }
    const uint64_t end_page = (end_ - 1) / page_size;
    const uint64_t start_page = off / page_size;
    if (off == end_ || start_page == end_page || start_page == end_page + 1) {
      // The end never moves back: a late request inside the last page must
      // not pull the read-ahead window behind requests already served.
      end_ = std::max(end_, end);
      if (streak_ < kRaMinStreak) ++streak_;
      return RaSeq::kSequential;
    }
    if (start_page + 1 == end_page) return RaSeq::kStraggler;
    end_ = end;
    streak_ = 0;
    return RaSeq::kRandom;
  }

  /** @return whether enough sequential reads came in a row to read ahead */
  bool Active() const { return streak_ >= kRaMinStreak; }

  /** @return the end (exclusive) of the stream so far; 0 before any read */
  uint64_t End() const { return has_prev_ ? end_ : 0; }

 private:
  bool has_prev_ = false;  ///< a read was observed
  uint64_t end_ = 0;       ///< end (exclusive) of the stream so far
  uint32_t streak_ = 0;    ///< sequential reads in a row, capped
};

/** An inclusive range of page numbers; empty when first > last. */
struct RaWindow {
  uint64_t first = 1;  ///< first page
  uint64_t last = 0;   ///< last page
  /** @return whether the window holds no page */
  bool Empty() const { return first > last; }
  /** @return whether `page` lies in the window */
  bool Has(uint64_t page) const { return page >= first && page <= last; }
};

/**
 * The pages to keep fetched for a stream whose stream end is `end`: the page
 * holding its last byte and the `ahead` pages after it, cut at the file's
 * last page.
 * @param end end (exclusive) of the stream so far
 * @param fsize the file's size
 * @param ahead pages to keep ahead (CLIO_FUSE_READAHEAD_PAGES)
 * @param page_size bytes per page (> 0)
 * @return the window; empty when end is 0 or beyond the file
 */
inline RaWindow RaWindowFor(uint64_t end, uint64_t fsize, uint32_t ahead,
                            uint64_t page_size) {
  RaWindow w;
  if (end == 0 || fsize == 0 || end > fsize) return w;
  w.first = (end - 1) / page_size;
  w.last = std::min(w.first + ahead, (fsize - 1) / page_size);
  return w;
}

/**
 * Bytes of page `page` that lie inside a file of `fsize` bytes: what a
 * prefetch of that page asks for (never past the size a read clamps to).
 * @param page page number
 * @param fsize the file's size
 * @param page_size bytes per page (> 0)
 * @return 0 for a page at or past the end, else 1..page_size
 */
inline uint64_t RaPageLen(uint64_t page, uint64_t fsize, uint64_t page_size) {
  const uint64_t start = page * page_size;
  if (start >= fsize) return 0;
  return std::min(page_size, fsize - start);
}

/**
 * Whether a cached page fetched with `page_len` bytes can answer the part
 * [page_off, page_off + n) of it.
 * @param page_len bytes the page was fetched with (RaPageLen then)
 * @param page_off offset of the part inside the page
 * @param n bytes of the part
 * @return true when the part lies wholly inside the fetched bytes
 */
inline bool RaCovers(uint64_t page_len, uint64_t page_off, uint64_t n) {
  return n != 0 && page_off <= page_len && n <= page_len - page_off;
}

/**
 * Parse CLIO_FUSE_READAHEAD_PAGES.
 * @param env the variable's value (may be null)
 * @return pages to keep ahead: 0 (off) for null, empty, non-numeric or
 *         non-positive values, else the value capped at kRaMaxPages
 */
inline uint32_t RaParsePages(const char *env) {
  if (env == nullptr || *env == '\0') return 0;
  const long v = std::strtol(env, nullptr, 10);
  if (v <= 0) return 0;
  return v > static_cast<long>(kRaMaxPages) ? kRaMaxPages
                                            : static_cast<uint32_t>(v);
}

}  // namespace clio::cae::fuse

#endif  // CLIO_CTE_ADAPTER_LIBFUSE_FUSE_READAHEAD_H_
