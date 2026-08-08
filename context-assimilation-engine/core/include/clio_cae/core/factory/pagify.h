/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * pagify: serve UNIFIED PAGE NUMBERS on top of arbitrarily-named blobs.
 *
 * A paged consumer (the gpu_vector) addresses its data as page 0, page 1,
 * page 2, ... and nothing else. It does not know blob names, families, or
 * layouts. Turning a page number into bytes is this layer's job, because
 * only the component that assimilated the data knows how the source maps
 * onto the page space -- that mapping is the PageMap (see page_map.h).
 *
 * The split is deliberate:
 *
 *   PageMap  -- pure arithmetic. Which blob bytes live in which page.
 *   Pagify   -- the I/O. Fetch those slices and CONSTRUCT the page.
 *
 * Construction is the part a caller must not have to repeat: a page can be
 * composed of several blobs (mixed), a blob can cover only part of a page
 * (partial), and a region covered by no extent must read as zero rather
 * than as whatever the buffer happened to hold. Pagify does all three.
 *
 * I/O is injected as a callable rather than hard-wired to the CTE client,
 * so the translation and assembly are testable without a runtime, and so
 * the same code serves a chimod interposer, a direct client, or a test
 * stub without duplication.
 */
#ifndef CLIO_CAE_CORE_PAGIFY_H_
#define CLIO_CAE_CORE_PAGIFY_H_

#include <clio_runtime/types.h>

#include <cstring>

#include <functional>
#include <string>
#include <vector>

#include "page_map.h"

namespace clio::cae::core {

/**
 * Reads a byte range of one blob into `dst`.
 * Returns the number of bytes actually read; anything short of `size` is
 * treated as a failed slice by the caller.
 */
using BlobReadFn = std::function<clio::run::u64(
    const std::string &blob_name, clio::run::u64 blob_off,
    clio::run::u64 size, void *dst)>;

/** Writes a byte range of one blob from `src`. Returns bytes written. */
using BlobWriteFn = std::function<clio::run::u64(
    const std::string &blob_name, clio::run::u64 blob_off,
    clio::run::u64 size, const void *src)>;

/** Outcome of a page operation, with enough detail to tell "nothing was
 *  mapped here" apart from "a read failed", which are very different bugs. */
struct PageIoResult {
  bool ok = false;              /**< every mapped slice transferred fully */
  clio::run::u32 slices = 0;    /**< slices the map produced for the page */
  clio::run::u64 bytes = 0;     /**< bytes transferred from/to blobs */
  clio::run::u64 zero_filled = 0;  /**< page bytes covered by no extent */
};

/**
 * Page-level I/O over a PageMap.
 *
 * Holds no buffers and no connection: the map is the state, and the I/O is
 * whatever callable the caller supplies.
 */
class Pagify {
 public:
  Pagify() = default;
  explicit Pagify(PageMap map) : map_(std::move(map)) {}

  const PageMap &Map() const { return map_; }
  void SetMap(PageMap map) { map_ = std::move(map); }
  clio::run::u64 PageSize() const { return map_.PageSize(); }
  clio::run::u64 NumPages() const { return map_.NumPages(); }

  /**
   * Construct page `page_id` into `dst` (must hold PageSize() bytes).
   *
   * Bytes not covered by any extent are zeroed, so a page at the tail of
   * the data -- or one with a hole -- is fully defined on return rather
   * than carrying stale buffer contents.
   */
  PageIoResult ReadPage(clio::run::u64 page_id, void *dst,
                        const BlobReadFn &read) const {
    PageIoResult r;
    const clio::run::u64 psz = map_.PageSize();
    if (dst == nullptr || psz == 0 || !read) return r;
    auto *out = static_cast<unsigned char *>(dst);

    const std::vector<PageSlice> slices = map_.ToPage(page_id);
    r.slices = static_cast<clio::run::u32>(slices.size());

    // Zero first, then fill: simpler and safer than tracking the gaps
    // between slices, and the gaps are exactly what must read as zero.
    std::memset(out, 0, static_cast<size_t>(psz));
    clio::run::u64 covered = 0;
    for (const PageSlice &s : slices) {
      if (s.dst_off + s.size > psz) return r;   // map disagrees with page
      const clio::run::u64 got =
          read(s.blob_name, s.blob_off, s.size, out + s.dst_off);
      if (got != s.size) return r;
      covered += s.size;
      r.bytes += got;
    }
    r.zero_filled = psz - covered;
    r.ok = true;
    return r;
  }

  /**
   * Write page `page_id` from `src` back to the blobs that compose it.
   *
   * Only mapped regions are written; a gap in the page has no home to be
   * written to and is silently skipped, which is the correct inverse of
   * ReadPage zero-filling it.
   */
  PageIoResult WritePage(clio::run::u64 page_id, const void *src,
                         const BlobWriteFn &write) const {
    PageIoResult r;
    const clio::run::u64 psz = map_.PageSize();
    if (src == nullptr || psz == 0 || !write) return r;
    const auto *in = static_cast<const unsigned char *>(src);

    const std::vector<PageSlice> slices = map_.ToPage(page_id);
    r.slices = static_cast<clio::run::u32>(slices.size());
    clio::run::u64 covered = 0;
    for (const PageSlice &s : slices) {
      if (s.dst_off + s.size > psz) return r;
      const clio::run::u64 put =
          write(s.blob_name, s.blob_off, s.size, in + s.dst_off);
      if (put != s.size) return r;
      covered += s.size;
      r.bytes += put;
    }
    r.zero_filled = psz - covered;
    r.ok = true;
    return r;
  }

  /** Pages that must be fetched to cover [blob_off, blob_off+size) of
   *  `blob`. The inverse direction, for callers that think in blob ranges
   *  but must drive a paged cache. */
  std::vector<clio::run::u64> PagesFor(const std::string &blob,
                                       clio::run::u64 blob_off,
                                       clio::run::u64 size) const {
    std::vector<clio::run::u64> out;
    for (const PageRef &ref : map_.FromPage(blob, blob_off, size)) {
      if (out.empty() || out.back() != ref.page_id) out.push_back(ref.page_id);
    }
    return out;
  }

 private:
  PageMap map_;
};

}  // namespace clio::cae::core

#endif  // CLIO_CAE_CORE_PAGIFY_H_
