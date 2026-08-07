/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * Unified page numbering for vector-backed data (pagify).
 *
 * A gpu_vector addresses its data as PAGES: page 0, page 1, page 2, ... and
 * nothing else. It does not know blob names, families, layouts, or which
 * assimilator produced the data. Turning a page number into bytes is the
 * job of whoever assimilated the data, because only that component knows
 * how the source maps onto the page space.
 *
 * That mapping is expressed here as a PageMap: an ordered list of extents,
 * each saying "these bytes of this blob occupy this range of the page
 * space". From it, both directions are mechanical:
 *
 *   ToPage(page_id)   -> the slices that compose that page
 *   FromPage(blob, ...) -> the pages those blob bytes land in
 *
 * The extent form is what makes PARTIAL and MIXED paging fall out for
 * free: a page may be composed of several blobs (mixed), and a blob may
 * span pages or occupy only part of one (partial). A one-blob-per-page
 * layout is just the special case where every extent is page-aligned and
 * page-sized.
 */
#ifndef CLIO_CAE_CORE_PAGE_MAP_H_
#define CLIO_CAE_CORE_PAGE_MAP_H_

#include <clio_runtime/types.h>

#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace clio::cae::core {

/** One contiguous run of a blob placed in the page space. */
struct PageExtent {
  std::string blob_name;      /**< source blob */
  clio::run::u64 blob_off;    /**< offset within that blob */
  clio::run::u64 page_off;    /**< offset within the GLOBAL page space */
  clio::run::u64 size;        /**< bytes */
};

/** A piece of one page, resolved to the blob bytes that fill it. */
struct PageSlice {
  std::string blob_name;
  clio::run::u64 blob_off;    /**< read from here ... */
  clio::run::u64 dst_off;     /**< ... into this offset within the page */
  clio::run::u64 size;
};

/** Which pages a blob range touches, and where inside them it lands. */
struct PageRef {
  clio::run::u64 page_id;
  clio::run::u64 dst_off;     /**< offset within that page */
  clio::run::u64 blob_off;
  clio::run::u64 size;
};

/**
 * The page-space layout of one assimilated dataset.
 *
 * Extents must be sorted by page_off and non-overlapping; Build() enforces
 * that. Gaps are legal and read as zero, which is what lets an assimilator
 * align a blob to a page boundary without inventing filler blobs.
 */
class PageMap {
 public:
  PageMap() = default;

  /** page_size must be non-zero; extents are sorted and validated. */
  static PageMap Build(clio::run::u64 page_size,
                       std::vector<PageExtent> extents) {
    PageMap m;
    m.page_size_ = page_size == 0 ? 1 : page_size;
    std::sort(extents.begin(), extents.end(),
              [](const PageExtent &a, const PageExtent &b) {
                return a.page_off < b.page_off;
              });
    clio::run::u64 end = 0;
    for (const auto &e : extents) {
      if (e.size == 0) continue;
      if (e.page_off < end) continue;   // overlap: drop, first writer wins
      end = e.page_off + e.size;
      m.extents_.push_back(e);
    }
    m.total_bytes_ = end;
    return m;
  }

  clio::run::u64 PageSize() const { return page_size_; }
  clio::run::u64 TotalBytes() const { return total_bytes_; }
  clio::run::u64 NumPages() const {
    return (total_bytes_ + page_size_ - 1) / page_size_;
  }
  const std::vector<PageExtent> &Extents() const { return extents_; }

  /**
   * ToPage: the slices that compose page `page_id`.
   *
   * A page with several slices is MIXED (more than one blob contributes);
   * a slice narrower than the page is PARTIAL. Bytes not covered by any
   * slice are zero.
   */
  std::vector<PageSlice> ToPage(clio::run::u64 page_id) const {
    std::vector<PageSlice> out;
    const clio::run::u64 lo = page_id * page_size_;
    const clio::run::u64 hi = lo + page_size_;
    // First extent whose end passes lo.
    auto it = std::lower_bound(
        extents_.begin(), extents_.end(), lo,
        [](const PageExtent &e, clio::run::u64 v) {
          return e.page_off + e.size <= v;
        });
    for (; it != extents_.end() && it->page_off < hi; ++it) {
      const clio::run::u64 s = std::max(lo, it->page_off);
      const clio::run::u64 e = std::min(hi, it->page_off + it->size);
      if (s >= e) continue;
      out.push_back(PageSlice{it->blob_name,
                              it->blob_off + (s - it->page_off),
                              s - lo,
                              e - s});
    }
    return out;
  }

  /** FromPage: the pages that hold [blob_off, blob_off+size) of `blob`. */
  std::vector<PageRef> FromPage(const std::string &blob,
                                clio::run::u64 blob_off,
                                clio::run::u64 size) const {
    std::vector<PageRef> out;
    for (const auto &ex : extents_) {
      if (ex.blob_name != blob) continue;
      const clio::run::u64 s = std::max(blob_off, ex.blob_off);
      const clio::run::u64 e =
          std::min(blob_off + size, ex.blob_off + ex.size);
      if (s >= e) continue;
      clio::run::u64 cur = ex.page_off + (s - ex.blob_off);
      clio::run::u64 left = e - s;
      clio::run::u64 boff = s;
      while (left > 0) {
        const clio::run::u64 pid = cur / page_size_;
        const clio::run::u64 dst = cur - pid * page_size_;
        const clio::run::u64 n = std::min(left, page_size_ - dst);
        out.push_back(PageRef{pid, dst, boff, n});
        cur += n;
        boff += n;
        left -= n;
      }
    }
    return out;
  }

  /** Serialize for the "pagemap" blob (one extent per line). */
  std::string Serialize() const {
    std::string s = "pagemap\npage_size=" + std::to_string(page_size_) +
                    "\ntotal=" + std::to_string(total_bytes_) + "\n";
    for (const auto &e : extents_) {
      s += e.blob_name + " " + std::to_string(e.blob_off) + " " +
           std::to_string(e.page_off) + " " + std::to_string(e.size) + "\n";
    }
    return s;
  }

  /** Inverse of Serialize; returns an empty map on malformed input. */
  static PageMap Parse(const std::string &text) {
    clio::run::u64 ps = 0;
    std::vector<PageExtent> ex;
    size_t pos = 0;
    while (pos < text.size()) {
      size_t nl = text.find('\n', pos);
      if (nl == std::string::npos) nl = text.size();
      const std::string line = text.substr(pos, nl - pos);
      pos = nl + 1;
      if (line.rfind("page_size=", 0) == 0) {
        ps = std::strtoull(line.c_str() + 10, nullptr, 10);
      } else if (line.empty() || line == "pagemap" ||
                 line.rfind("total=", 0) == 0) {
        continue;
      } else {
        PageExtent e;
        char name[256] = {0};
        unsigned long long a = 0, b = 0, c = 0;
        if (sscanf(line.c_str(), "%255s %llu %llu %llu", name, &a, &b, &c) ==
            4) {
          e.blob_name = name;
          e.blob_off = a;
          e.page_off = b;
          e.size = c;
          ex.push_back(e);
        }
      }
    }
    if (ps == 0) return PageMap();
    return Build(ps, std::move(ex));
  }

 private:
  clio::run::u64 page_size_ = 0;
  clio::run::u64 total_bytes_ = 0;
  std::vector<PageExtent> extents_;
};

}  // namespace clio::cae::core

#endif  // CLIO_CAE_CORE_PAGE_MAP_H_
