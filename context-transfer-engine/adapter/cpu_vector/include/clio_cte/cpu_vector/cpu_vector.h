/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * cpu_vector: a vector whose backing store is the CTE and whose working set
 * lives in HOST memory, paged in on demand by whichever host thread asks.
 *
 * The CPU sibling of gpu_vector. Same model -- a set-associative page cache
 * in front of CTE blobs, explicit Fetch / HoldPage / UnpinRange / Flush, no
 * dirty bit (write-site publish is the caller's contract) -- and the same
 * host helpers under the same names. It is NOT a drop-in replacement:
 *
 *   - There is no device view. GetDevice(), DeviceVector and Init(block) do
 *     not exist; the verbs are members of Vector itself and are thread-safe,
 *     so any number of host threads drive one Vector directly.
 *   - The verbs BLOCK instead of being coroutines. Fetch returns when the
 *     bytes are resident; Flush returns when the put has landed.
 *   - The constructor takes no GPU ids and no block count -- nothing is
 *     dimensioned per block, because there are no per-block task sets.
 *   - A caller error (HoldPage on a page that was never fetched) throws
 *     std::runtime_error instead of trapping the kernel.
 *
 * INTEROPERABLE WITH gpu_vector ON THE SAME TAG. Page p is blob "p" in
 * decimal, written at kVectorBlobScore with the same compression context, so
 * a cpu_vector can preload what a gpu_vector kernel reads and download what
 * it wrote, as long as both use the same page_bytes and element type.
 */
#ifndef CLIO_CTE_CPU_VECTOR_CPU_VECTOR_H_
#define CLIO_CTE_CPU_VECTOR_CPU_VECTOR_H_

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/types.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/checkpoint/checkpoint_client.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace clio::cte::cpu_vector {

using clio::run::u32;
using clio::run::u64;

/** page_num of an empty frame. Same value as gpu_vector::kNoPage. */
constexpr u64 kNoPage = ~static_cast<u64>(0);

/** The CTE blob score every page is written at. MUST equal
 *  gpu_vector::kVectorBlobScore -- see that definition for why 0.5 decides
 *  which tiers a page can reach. */
constexpr float kVectorBlobScore = 0.5f;

/** A frame's eviction rank until someone rescored it. Lowest is evicted. */
constexpr float kDefaultScore = 0.5f;

/** How long a Fetch waits for a frame to come free before concluding the
 *  cache is too small for what is pinned, and throwing. */
constexpr u32 kAllocWaitMs = 30000;

/** One frame of the cache. Host memory, so no layout constraints; `pins` is
 *  atomic because a Held guard releases it without taking the set lock.
 *  Everything else is guarded by the frame's set mutex. */
struct Frame {
  u64 page_num = kNoPage;
  char *data = nullptr;          // page_bytes of storage, null when empty
  u64 last_access = 0;           // LRU tie-break
  u64 generation = 0;            // generation of the bytes held, 0 = unknown
  std::atomic<float> score{kDefaultScore};
  std::atomic<u32> pins{0};
  u32 flushing = 0;              // puts outstanding from this frame
  u32 fetching = 0;              // a get is filling this frame
  u32 valid_lo = 0;              // element interval actually present
  u32 valid_hi = 0;              //   (valid_hi <= valid_lo: empty)
};

/** Snapshot of one frame, for ReadTable(). */
struct Page {
  u64 page_num;
  void *data;
  u64 last_access;
  u64 generation;
  float score;
  u32 pins;
  u32 flushing;
  u32 fetching;
  u32 valid_lo;
  u32 valid_hi;
};

/** A rescore request for PrefetchNow. Same shape as gpu_vector's. */
struct PrefetchHint {
  u64 page_;
  float score_;
};

/** Counters for PrefetchNow/ScoreAllPages. */
struct PrefetchStats {
  u64 hints_ = 0;
  u64 deduped_ = 0;
  u64 sent_ = 0;
  u64 batches_ = 0;
  u64 promotions_ = 0;
  u64 demotions_ = 0;
  u64 not_found_ = 0;
  u64 errors_ = 0;
};

/**
 * A held page range. Elements [begin_off(), begin_off()+run()) are reachable
 * through operator[] by ABSOLUTE element offset, exactly like gpu_vector's
 * Held. Non-owning by default: Fetch took the pin and UnpinRange releases it.
 */
template <typename T>
class Held {
 public:
  Held() = default;
  Held(Frame *frame, T *data, u64 begin, u64 run, bool owns_pin)
      : frame_(frame), data_(data), begin_(begin), run_(run),
        owns_pin_(owns_pin) {}
  Held(Held &&o) noexcept { Steal(o); }
  Held &operator=(Held &&o) noexcept {
    if (this != &o) {
      Unpin();
      Steal(o);
    }
    return *this;
  }
  Held(const Held &) = delete;
  Held &operator=(const Held &) = delete;
  ~Held() { Unpin(); }

  T *ptr() const { return data_; }
  u64 run() const { return run_; }
  u64 begin_off() const { return begin_; }
  explicit operator bool() const { return frame_ != nullptr; }
  T &operator[](u64 off) const { return data_[off - begin_]; }

  /** Set this frame's eviction rank. Higher means keep. */
  void Rescore(float rank) const {
    if (frame_ != nullptr) frame_->score.store(rank, std::memory_order_relaxed);
  }

 private:
  void Unpin() {
    if (frame_ != nullptr && owns_pin_) {
      frame_->pins.fetch_sub(1, std::memory_order_acq_rel);
    }
    frame_ = nullptr;
  }
  void Steal(Held &o) {
    frame_ = o.frame_;
    data_ = o.data_;
    begin_ = o.begin_;
    run_ = o.run_;
    owns_pin_ = o.owns_pin_;
    o.frame_ = nullptr;
  }

  Frame *frame_ = nullptr;
  T *data_ = nullptr;
  u64 begin_ = 0;
  u64 run_ = 0;
  bool owns_pin_ = false;
};

template <typename T>
class Vector {
 public:
  /** Counters, populated only when EnableStats() was called. Same fields as
   *  gpu_vector's Stats minus the GPU cycle counters. */
  struct Stats {
    u64 faults = 0;         // pages (or page ranges) read in
    u64 puts = 0;           // ranges written back
    u64 evicts = 0;         // frames reclaimed
    u64 get_errors = 0;     // gets that returned non-zero
    u64 gen_ok = 0;         // generational demand: resident accepted
    u64 gen_stale = 0;      // generational demand: refetch claimed
    u64 gen_busy = 0;       // generational demand: a peer was refetching
    u64 flush_skipped = 0;  // flush ranges not resident: DROPPED
    u64 put_errors = 0;     // writebacks that returned non-zero
    u64 alloc_waits = 0;    // waits for a frame to come free
  };

  /** Ranges one Fetch or Flush may name, as gpu_vector. */
  static constexpr u32 kMaxFetchRanges = 8;

  /**
   * @param tag_name       CTE tag representing this vector
   * @param page_bytes     page granularity; a multiple of sizeof(T)
   * @param num_elems      logical length of the vector
   * @param nsets          associative sets in the cache
   * @param set_size       frames (tags) in one set
   * @param storage_pool_id CTE core pool; null means the default CTE pool
   * @param compress_lib   CTE compression library id, 0 = none
   * @param compress_preset CTE compression preset
   * @param capacity_pages page-sized regions backing the frames; 0 means
   *        nsets * set_size (one region per frame). Smaller than that makes
   *        the frames tags over a tighter memory budget, as in gpu_vector.
   */
  Vector(const std::string &tag_name, u64 page_bytes, u64 num_elems,
         u32 nsets, u32 set_size,
         clio::run::PoolId storage_pool_id = clio::run::PoolId::GetNull(),
         int compress_lib = 0, int compress_preset = 1,
         u32 capacity_pages = 0)
      : storage_pool_id_(storage_pool_id.IsNull() ? clio::cte::core::kCtePoolId
                                                  : storage_pool_id),
        compress_lib_(compress_lib),
        compress_preset_(compress_preset),
        page_bytes_(page_bytes),
        nsets_(nsets),
        set_size_(set_size),
        capacity_pages_(capacity_pages != 0 ? capacity_pages
                                            : nsets * set_size),
        num_elems_(num_elems),
        tag_name_(tag_name),
        core_(storage_pool_id_) {
    if (page_bytes_ == 0 || nsets_ == 0 || set_size_ == 0) {
      throw std::runtime_error("cpu_vector: zero page size / sets / set size");
    }
    if (page_bytes_ % sizeof(T) != 0) {
      throw std::runtime_error("cpu_vector: page_bytes is not a multiple of "
                               "the element size");
    }
    clio::cte::core::Client cte(clio::cte::core::kCtePoolId);
    auto tag = cte.AsyncGetOrCreateTag(tag_name);
    tag.Wait();
    if (tag->GetReturnCode() != 0) {
      throw std::runtime_error("cpu_vector: could not resolve tag " + tag_name);
    }
    tag_id_ = tag->tag_id_;
    BuildCache();
  }

  ~Vector() = default;
  Vector(const Vector &) = delete;
  Vector &operator=(const Vector &) = delete;

  // ======================= geometry ================================

  u64 size() const { return num_elems_; }
  clio::cte::core::TagId TagId() const { return tag_id_; }
  u64 PageBytes() const { return page_bytes_; }
  u64 ElemsPerPage() const { return page_bytes_ / sizeof(T); }
  u64 NumPages() const {
    const u64 epp = ElemsPerPage();
    return (num_elems_ + epp - 1) / epp;
  }
  u32 NumSets() const { return nsets_; }
  u32 SetSize() const { return set_size_; }
  u64 NumFrames() const { return static_cast<u64>(nsets_) * set_size_; }
  u32 Regions() const { return capacity_pages_; }
  u64 PageOf(u64 off) const { return off / ElemsPerPage(); }
  /** Start of the page containing `off`; pairs with PageSpan. */
  u64 PageLo(u64 off) const { return PageOf(off) * ElemsPerPage(); }
  /** Element count from PageLo(off) covering whole pages through
   *  [off, off+count). */
  u64 PageSpan(u64 off, u64 count) const {
    const u64 last = off + (count != 0 ? count - 1 : 0);
    return (PageOf(last) + 1) * ElemsPerPage() - PageLo(off);
  }

  /** A page's home set. MUST match gpu_vector's HomeSet/SetOf. */
  u32 HomeSet(u64 pn) const {
    const u64 mixed = (pn ^ (pn >> 29)) * 0x9E3779B97F4A7C15ull;
    return static_cast<u32>((mixed >> 32) % nsets_);
  }

  // ======================= the verbs ===============================
  //
  // Thread-safe. Any number of host threads may call these concurrently on
  // one Vector. They replace DeviceVector's coroutine verbs one for one.

  /**
   * Make every page of the given element ranges resident and PIN it.
   * Arguments are (offset, count) pairs, up to kMaxFetchRanges of them.
   *
   * @param generation 0 for an ordinary fetch; otherwise a generational get
   *        that blocks server-side until the writer has published at least
   *        this generation, and a resident frame older than it is refetched.
   * A page that has no blob yet (ordinary fetch only) reads as zeros.
   * Every page fetched here stays pinned until UnpinRange releases it.
   * @throws std::runtime_error if a get fails or no frame can be found
   */
  template <typename... Args>
  void Fetch(u64 generation, Args... args) {
    static_assert(sizeof...(Args) % 2 == 0 && sizeof...(Args) > 0,
                  "Fetch takes (offset, count) pairs");
    std::vector<std::pair<u64, u64>> r;
    Gather(r, static_cast<u64>(args)...);
    FetchRanges(r, generation, /*wait_for_frames=*/true);
  }

  /** Release the pins Fetch took on every page of [off, off+count). */
  void UnpinRange(u64 off, u64 count) {
    if (count == 0) return;
    const u64 p0 = PageOf(off), p1 = PageOf(off + count - 1);
    for (u64 pn = p0; pn <= p1; ++pn) {
      const u32 s = HomeSet(pn);
      std::lock_guard<std::mutex> lk(sets_[s].mu);
      Frame *f = FindLocked(s, pn);
      if (f != nullptr && f->pins.load() > 0) f->pins.fetch_sub(1);
      sets_[s].cv.notify_all();
    }
  }
  /** Two ranges at once, as gpu_vector's UnpinRange2. */
  void UnpinRange2(u64 off0, u64 count0, u64 off1, u64 count1) {
    UnpinRange(off0, count0);
    UnpinRange(off1, count1);
  }

  /**
   * The part of [off, off+count) that lies on off's page. Does NOT fetch: a
   * page that was not named in a Fetch is a caller error. A read hold also
   * requires the elements to have been fetched (resident is not valid); a
   * write hold marks them valid.
   *
   * The returned Held is non-owning, as on the GPU: Fetch's pin keeps the
   * frame alive and UnpinRange releases it.
   * @throws std::runtime_error on a caller error
   */
  Held<T> HoldPage(u64 off, u64 count, bool write = false) {
    const u64 pn = PageOf(off);
    const u32 s = HomeSet(pn);
    const u64 epp = ElemsPerPage();
    const u32 in = static_cast<u32>(off % epp);
    u64 run = epp - in;
    if (run > count) run = count;
    // The last page is partial: nothing past num_elems_ exists to hold.
    if (off >= num_elems_) {
      throw std::runtime_error("cpu_vector: HoldPage past the end (offset " +
                               std::to_string(off) + ")");
    }
    if (run > num_elems_ - off) run = num_elems_ - off;
    std::unique_lock<std::mutex> lk(sets_[s].mu);
    Frame *f = nullptr;
    for (;;) {
      f = FindLocked(s, pn);
      if (f == nullptr || f->fetching == 0) break;
      sets_[s].cv.wait(lk);
    }
    if (f == nullptr) {
      throw std::runtime_error(
          "cpu_vector: HoldPage: page " + std::to_string(pn) +
          " is not resident -- name it in a Fetch first");
    }
    const u32 hi = in + static_cast<u32>(run);
    if (!write) {
      if (!Covers(*f, in, hi)) {
        throw std::runtime_error(
            "cpu_vector: HoldPage: elements [" + std::to_string(in) + "," +
            std::to_string(hi) + ") of page " + std::to_string(pn) +
            " were never fetched -- the frame holds [" +
            std::to_string(f->valid_lo) + "," + std::to_string(f->valid_hi) +
            ")");
      }
    } else {
      Widen(*f, in, hi);
    }
    f->last_access = ++clock_;
    T *base = reinterpret_cast<T *>(f->data);
    return Held<T>(f, base + in, off, run, /*owns_pin=*/false);
  }

  /** Fetch the whole pages under [off, off+count) and hold off's page.
   *  The pins stay until UnpinRange(PageLo(off), PageSpan(off, count)). */
  Held<T> UpdateRange(u64 off, u64 count, bool write = false,
                      u64 generation = 0) {
    Fetch(generation, PageLo(off), PageSpan(off, count));
    return HoldPage(off, count, write);
  }

  /**
   * Write back the named element ranges, as (offset, count) pairs, and wait
   * for the puts to land. Only the elements a frame actually holds are sent;
   * a range whose page is not resident is DROPPED and counted in
   * flush_skipped, as on the GPU.
   * @param generation 0 for an ordinary put; otherwise the generation to
   *        stamp, which releases generational readers waiting on it
   * @throws std::runtime_error if any put fails
   */
  template <typename... Args>
  void Flush(u64 generation, Args... args) {
    static_assert(sizeof...(Args) % 2 == 0 && sizeof...(Args) > 0,
                  "Flush takes (offset, count) pairs");
    std::vector<std::pair<u64, u64>> r;
    Gather(r, static_cast<u64>(args)...);
    FlushRanges(r, generation);
  }

  // ======================= host data movement ======================

  /** Write pages [pg_lo, pg_hi) straight to the CTE; `fill(pg, buf)` supplies
   *  each page's bytes. Bypasses the cache: resident copies are not updated,
   *  so ClearCache first if those pages may be resident. */
  template <typename FillFn>
  void PreloadPages(u64 pg_lo, u64 pg_hi, FillFn fill) {
    std::vector<char> buf(static_cast<size_t>(page_bytes_));
    for (u64 pg = pg_lo; pg < pg_hi; ++pg) {
      std::memset(buf.data(), 0, buf.size());
      fill(pg, buf.data());
      auto f = core_.AsyncPutBlob(tag_id_, PageName(pg), 0, page_bytes_,
                                  buf.data(), kVectorBlobScore, PageContext(),
                                  0u, clio::run::PoolQuery::Dynamic());
      f.Wait();
      if (f->GetReturnCode() != 0) {
        throw std::runtime_error("cpu_vector: preload of page " +
                                 std::to_string(pg) + " failed");
      }
    }
  }

  /** Write `n` elements from host memory into the vector (CTE side). */
  void Preload(const T *src, u64 n) {
    const u64 epp = ElemsPerPage();
    PreloadPages(0, (n + epp - 1) / epp, [&](u64 pg, char *buf) {
      const u64 off = pg * epp;
      const u64 cnt = (epp < n - off) ? epp : (n - off);
      std::memcpy(buf, src + off, static_cast<size_t>(cnt * sizeof(T)));
    });
  }

  /** Read pages [pg_lo, pg_hi) from the CTE; `sink(pg, bytes)` receives each
   *  page that has a blob. Bypasses the cache: flush first.
   *  @return pages read */
  template <typename SinkFn>
  u64 DownloadPages(u64 pg_lo, u64 pg_hi, SinkFn sink) {
    std::vector<char> buf(static_cast<size_t>(page_bytes_));
    u64 got = 0;
    for (u64 pg = pg_lo; pg < pg_hi; ++pg) {
      if (!ReadPage(pg, buf.data())) continue;
      sink(pg, static_cast<const char *>(buf.data()));
      ++got;
    }
    return got;
  }

  /** Read `n` elements out of the vector into host memory.
   *  @return elements actually read */
  u64 Download(T *dst, u64 n) {
    const u64 epp = ElemsPerPage();
    u64 got = 0;
    DownloadPages(0, (n + epp - 1) / epp, [&](u64 pg, const char *bytes) {
      const u64 off = pg * epp;
      if (off >= n) return;
      const u64 cnt = (epp < n - off) ? epp : (n - off);
      std::memcpy(dst + off, bytes, static_cast<size_t>(cnt * sizeof(T)));
      got += cnt;
    });
    return got;
  }

  /** Make pages [pg_lo, pg_hi) resident, best effort: a page whose set has
   *  no free frame is skipped rather than waited for. Leaves them unpinned. */
  void Prefetch(u64 pg_lo, u64 pg_hi) {
    const u64 epp = ElemsPerPage();
    for (u64 pg = pg_lo; pg < pg_hi && pg < NumPages(); ++pg) {
      std::vector<std::pair<u64, u64>> r{{pg * epp, epp}};
      if (FetchRanges(r, 0, /*wait_for_frames=*/false)) UnpinRange(pg * epp, epp);
    }
  }

  /** Write every resident frame's valid elements back to the CTE. There is
   *  no dirty bit, so this sends ALL of them. */
  void FlushResidentToCte() {
    std::vector<std::pair<u64, u64>> r;
    const u64 epp = ElemsPerPage();
    for (u32 s = 0; s < nsets_; ++s) {
      std::lock_guard<std::mutex> lk(sets_[s].mu);
      for (u32 i = 0; i < set_size_; ++i) {
        const Frame &f = frames_[FrameIdx(s, i)];
        if (f.page_num == kNoPage || f.valid_hi <= f.valid_lo) continue;
        r.emplace_back(f.page_num * epp + f.valid_lo, f.valid_hi - f.valid_lo);
      }
    }
    for (size_t i = 0; i < r.size(); i += kMaxFetchRanges) {
      std::vector<std::pair<u64, u64>> chunk(
          r.begin() + i, r.begin() + std::min(r.size(), i + kMaxFetchRanges));
      FlushRanges(chunk, 0);
    }
  }

  /** Drop every resident page. Unflushed writes are DISCARDED -- flush first.
   *  Must not race the verbs. */
  void ClearCache() {
    for (u32 s = 0; s < nsets_; ++s) {
      std::lock_guard<std::mutex> lk(sets_[s].mu);
      for (u32 i = 0; i < set_size_; ++i) ResetFrame(frames_[FrameIdx(s, i)]);
    }
    std::lock_guard<std::mutex> lk(free_mu_);
    free_.clear();
    for (u32 i = capacity_pages_; i > 0; --i) free_.push_back(i - 1);
  }

  // ======================= checkpoints =============================

  /**
   * Lazily-initialised copy of this vector: the checkpoint primitive. Same
   * contract as gpu_vector::Vector::Copy -- flush resident state, register
   * the checkpoint chimod as the copy tag's fault handler with this tag as
   * its source, and with sync=true materialise every page before returning.
   */
  std::unique_ptr<Vector<T>> Copy(const std::string &new_name,
                                  bool sync = false) {
    using clio::cte::core::Context;
    if (tag_name_.size() >= Context::kFaultParamsSize) {
      throw std::runtime_error("cpu_vector: source tag name too long for "
                               "fault params: " + tag_name_);
    }
    FlushResidentToCte();
    {
      clio::cte::checkpoint::Client ckpt(
          clio::cte::checkpoint::kCheckpointPoolId, storage_pool_id_);
      clio::cte::checkpoint::CheckpointConfig params;
      params.next_pool_id_ = storage_pool_id_;
      auto cf = ckpt.AsyncCreateCheckpoint(
          clio::run::PoolQuery::Local(),
          clio::cte::checkpoint::kCheckpointPoolName,
          clio::cte::checkpoint::kCheckpointPoolId, params);
      cf.Wait();
      if (cf->GetReturnCode() != 0) {
        throw std::runtime_error("cpu_vector: checkpoint pool create failed");
      }
    }
    {
      clio::cte::core::Client core(storage_pool_id_);
      auto tf = core.AsyncGetOrCreateTag(
          new_name, clio::cte::checkpoint::kCheckpointPoolId,
          clio::cte::checkpoint::kCheckpointPoolName, tag_name_);
      tf.Wait();
      if (tf->GetReturnCode() != 0) {
        throw std::runtime_error("cpu_vector: could not create copy tag " +
                                 new_name);
      }
    }
    std::unique_ptr<Vector<T>> out(new Vector<T>(
        new_name, page_bytes_, num_elems_, nsets_, set_size_,
        storage_pool_id_, compress_lib_, compress_preset_, capacity_pages_));
    out->ckpt_src_ = tag_name_;
    if (sync) {
      const u64 np = out->NumPages();
      if (out->MaterializeAll() != np) {
        throw std::runtime_error("cpu_vector: synchronous copy " + new_name +
                                 " failed to materialise");
      }
    }
    return out;
  }

  /** Force a lazy Copy() to become real bytes, server-side. See
   *  gpu_vector::Vector::MaterializeAll.
   *  @return pages materialised (0 on the first failure) */
  u64 MaterializeAll(u32 inflight = 32) {
    if (ckpt_src_.empty()) {
      std::fprintf(stderr, "cpu_vector: MaterializeAll on '%s', which is not "
                   "a Copy() -- there is no source to materialise from\n",
                   tag_name_.c_str());
      return 0;
    }
    auto *ipc = CLIO_CPU_IPC;
    const u64 np = NumPages();
    std::vector<clio::run::Future<clio::cte::core::GetBlobTask>> futs;
    futs.reserve(inflight);
    u64 done = 0;
    auto reap = [&]() -> bool {
      for (auto &f : futs) {
        f.Wait();
        if (f->GetReturnCode() != 0) {
          std::fprintf(stderr,
                       "cpu_vector: MaterializeAll page %llu rc=%d (tag '%s' "
                       "src '%s')\n", (unsigned long long)done,
                       f->GetReturnCode(), tag_name_.c_str(),
                       ckpt_src_.c_str());
          return false;
        }
        ++done;
      }
      futs.clear();
      return true;
    };
    for (u64 pg = 0; pg < np; ++pg) {
      clio::cte::core::Context ctx = PageContext();
      std::snprintf(ctx.fault_params_,
                    clio::cte::core::Context::kFaultParamsSize, "%s",
                    ckpt_src_.c_str());
      ctx.op_flags_ &= ~clio::cte::core::Context::kBlobNameRawInt32;
      auto sub = ipc->NewTask<clio::cte::core::GetBlobTask>(
          clio::run::CreateTaskId(),
          clio::cte::checkpoint::kCheckpointPoolId,
          clio::run::PoolQuery::Local(), tag_id_, PageName(pg), 0, 0, 0,
          ctp::ipc::ShmPtr<>::GetNull(), ctx);
      futs.push_back(ipc->Send(sub));
      if (futs.size() >= inflight && !reap()) return 0;
    }
    if (!reap()) return 0;
    return done;
  }

  // ======================= tier hints ==============================

  /** Rescore pages in the CTE (tier placement), deduplicated against what
   *  was last sent. Non-blocking; DrainPrefetch waits. */
  void PrefetchNow(const std::vector<PrefetchHint> &hints) {
    if (hints.empty()) return;
    std::lock_guard<std::mutex> lk(prefetch_mu_);
    ReapPrefetchLocked(false);
    prefetch_stats_.hints_ += hints.size();
    SubmitHintsLocked(hints);
  }

  /** Score every page of this vector (e.g. drain a checkpoint downward). */
  void ScoreAllPages(float score) {
    std::vector<PrefetchHint> hints;
    const u64 np = NumPages();
    hints.reserve(static_cast<size_t>(np));
    for (u64 pg = 0; pg < np; ++pg) hints.push_back(PrefetchHint{pg, score});
    PrefetchNow(hints);
  }

  void DrainPrefetch() {
    std::lock_guard<std::mutex> lk(prefetch_mu_);
    ReapPrefetchLocked(true);
  }
  /** Forget what was sent; call per run, not per phase (see gpu_vector). */
  void ResetPrefetchHistory() {
    std::lock_guard<std::mutex> lk(prefetch_mu_);
    last_sent_.clear();
  }
  PrefetchStats ReadPrefetchStats() const { return prefetch_stats_; }
  void ResetPrefetchStats() { prefetch_stats_ = PrefetchStats(); }

  // ======================= diagnostics =============================

  void EnableStats() { stats_on_.store(true); }
  void ResetStats() {
    for (auto &c : stat_) c.store(0);
  }
  Stats ReadStats() const {
    Stats s;
    s.faults = stat_[kFaults].load();
    s.puts = stat_[kPuts].load();
    s.evicts = stat_[kEvicts].load();
    s.get_errors = stat_[kGetErrors].load();
    s.gen_ok = stat_[kGenOk].load();
    s.gen_stale = stat_[kGenStale].load();
    s.gen_busy = stat_[kGenBusy].load();
    s.flush_skipped = stat_[kFlushSkipped].load();
    s.put_errors = stat_[kPutErrors].load();
    s.alloc_waits = stat_[kAllocWaits].load();
    return s;
  }
  /** gpu_vector spelling (ReadStats(gpu_id)); there is one cache, so the
   *  id is ignored. Lets code written against either vector read stats. */
  Stats ReadStats(int /*gpu_id*/) const { return ReadStats(); }

  /** Every frame's metadata. */
  std::vector<Page> ReadTable() const {
    std::vector<Page> out;
    out.reserve(static_cast<size_t>(NumFrames()));
    for (u32 s = 0; s < nsets_; ++s) {
      std::lock_guard<std::mutex> lk(sets_[s].mu);
      for (u32 i = 0; i < set_size_; ++i) {
        const Frame &f = frames_[FrameIdx(s, i)];
        out.push_back(Page{f.page_num, f.data, f.last_access, f.generation,
                           f.score.load(), f.pins.load(), f.flushing,
                           f.fetching, f.valid_lo, f.valid_hi});
      }
    }
    return out;
  }

  /** Occupied frames; 0 means the cache is empty. */
  u32 Occupied() const {
    u32 n = 0;
    for (const Page &p : ReadTable()) n += (p.page_num != kNoPage);
    return n;
  }

  /** Frames holding the most-duplicated page. Always 1 for a healthy cache. */
  u32 MaxPageCopies() const {
    std::map<u64, u32> seen;
    u32 worst = 0;
    for (const Page &p : ReadTable()) {
      if (p.page_num == kNoPage) continue;
      worst = std::max(worst, ++seen[p.page_num]);
    }
    return worst;
  }

 private:
  // ======================= internals ===============================

  enum StatIdx : u32 {
    kFaults, kPuts, kEvicts, kGetErrors, kGenOk, kGenStale, kGenBusy,
    kFlushSkipped, kPutErrors, kAllocWaits, kNumStats
  };

  struct Set {
    mutable std::mutex mu;
    std::condition_variable cv;
  };

  /** A get this thread claimed: fill `frame` with [lo, hi) of page pn. Up
   *  to two pieces, because extending a valid interval can need both ends. */
  struct Transfer {
    Frame *frame;
    u32 set;
    u64 pn;
    u64 generation;
    u32 nlo[2], nhi[2];
    u32 npieces;
    bool replace;  // stale generation: drop the old valid range
    clio::run::Future<clio::cte::core::GetBlobTask> fut[2];
  };

  void Stat(StatIdx i, u64 n = 1) {
    if (stats_on_.load(std::memory_order_relaxed)) stat_[i].fetch_add(n);
  }

  void BuildCache() {
    sets_.reset(new Set[nsets_]);
    frames_.reset(new Frame[static_cast<size_t>(NumFrames())]);
    regions_.reset(new char[static_cast<size_t>(capacity_pages_) * page_bytes_]);
    free_.reserve(capacity_pages_);
    for (u32 i = capacity_pages_; i > 0; --i) free_.push_back(i - 1);
  }

  size_t FrameIdx(u32 set, u32 i) const {
    return static_cast<size_t>(set) * set_size_ + i;
  }

  Frame *FindLocked(u32 set, u64 pn) {
    for (u32 i = 0; i < set_size_; ++i) {
      Frame &f = frames_[FrameIdx(set, i)];
      if (f.page_num == pn) return &f;
    }
    return nullptr;
  }

  static bool Covers(const Frame &f, u32 lo, u32 hi) {
    return f.valid_lo <= lo && hi <= f.valid_hi && f.valid_lo < f.valid_hi;
  }

  /** Grow the valid interval to the hull of itself and [lo, hi). */
  static void Widen(Frame &f, u32 lo, u32 hi) {
    if (f.valid_hi <= f.valid_lo) {
      f.valid_lo = lo;
      f.valid_hi = hi;
    } else {
      f.valid_lo = std::min(f.valid_lo, lo);
      f.valid_hi = std::max(f.valid_hi, hi);
    }
  }

  void ResetFrame(Frame &f) {
    f.page_num = kNoPage;
    f.data = nullptr;
    f.last_access = 0;
    f.generation = 0;
    f.score.store(kDefaultScore);
    f.pins.store(0);
    f.flushing = 0;
    f.fetching = 0;
    f.valid_lo = 0;
    f.valid_hi = 0;
  }

  static bool Evictable(const Frame &f) {
    return f.page_num != kNoPage && f.pins.load() == 0 && f.fetching == 0 &&
           f.flushing == 0;
  }

  /** Lowest score, then least recently used, among this set's evictable
   *  frames. Caller holds the set lock. */
  Frame *VictimLocked(u32 set) {
    Frame *best = nullptr;
    for (u32 i = 0; i < set_size_; ++i) {
      Frame &f = frames_[FrameIdx(set, i)];
      if (!Evictable(f)) continue;
      if (best == nullptr || f.score.load() < best->score.load() ||
          (f.score.load() == best->score.load() &&
           f.last_access < best->last_access)) {
        best = &f;
      }
    }
    return best;
  }

  char *PopRegion() {
    std::lock_guard<std::mutex> lk(free_mu_);
    if (free_.empty()) return nullptr;
    const u32 r = free_.back();
    free_.pop_back();
    return regions_.get() + static_cast<size_t>(r) * page_bytes_;
  }

  /** Take a region from an evictable frame in ANOTHER set. try_lock only:
   *  the caller holds its own set's lock, and blocking on a second one in
   *  arbitrary order could deadlock with a thread doing the reverse. */
  char *StealRegion(u32 home) {
    for (u32 k = 1; k < nsets_; ++k) {
      const u32 s = (home + k) % nsets_;
      std::unique_lock<std::mutex> lk(sets_[s].mu, std::try_to_lock);
      if (!lk.owns_lock()) continue;
      Frame *v = VictimLocked(s);
      if (v == nullptr) continue;
      char *d = v->data;
      ResetFrame(*v);
      Stat(kEvicts);
      return d;
    }
    return nullptr;
  }

  /** A frame for page pn in `set`, claimed (page_num set, data assigned,
   *  valid empty), or null if none can be had right now. Caller holds the
   *  set lock. */
  Frame *ClaimFrameLocked(u32 set, u64 pn) {
    Frame *empty = nullptr;
    for (u32 i = 0; i < set_size_ && empty == nullptr; ++i) {
      Frame &f = frames_[FrameIdx(set, i)];
      if (f.page_num == kNoPage) empty = &f;
    }
    Frame *slot = nullptr;
    char *region = nullptr;
    if (empty != nullptr) region = PopRegion();
    if (region != nullptr) {
      slot = empty;
    } else if (Frame *v = VictimLocked(set)) {
      region = v->data;
      ResetFrame(*v);
      Stat(kEvicts);
      slot = v;
    } else if (empty != nullptr && (region = StealRegion(set)) != nullptr) {
      slot = empty;
    }
    if (slot == nullptr) return nullptr;
    slot->page_num = pn;
    slot->data = region;
    slot->valid_lo = slot->valid_hi = 0;
    slot->generation = 0;
    slot->score.store(kDefaultScore);
    return slot;
  }

  template <typename... Rest>
  static void Gather(std::vector<std::pair<u64, u64>> &r, u64 off, u64 count,
                     Rest... rest) {
    r.emplace_back(off, count);
    if constexpr (sizeof...(Rest) > 0) Gather(r, rest...);
  }

  /**
   * The fetch engine.
   *
   * Three phases, so no thread ever waits while holding a claim another
   * thread may be waiting on:
   *   1. claim: for each page, under its set lock, pin it and either accept
   *      the resident bytes, claim the frame for our own transfer, or note
   *      that a peer is filling it;
   *   2. transfer: issue every claimed get concurrently, wait, publish;
   *   3. peers: wait for frames a peer was filling, and re-run the whole
   *      fetch for any page that still does not cover what we asked for.
   * When no frame can be had in phase 1, the transfers claimed so far are
   * completed first (they may be what frees one), then the wait begins.
   *
   * @return false only when wait_for_frames is false and a page was skipped;
   *         the pages that WERE fetched stay pinned either way
   */
  bool FetchRanges(const std::vector<std::pair<u64, u64>> &ranges,
                   u64 generation, bool wait_for_frames) {
    if (ranges.size() > kMaxFetchRanges) {
      throw std::runtime_error("cpu_vector: Fetch names more than "
                               "kMaxFetchRanges ranges");
    }
    const u64 epp = ElemsPerPage();
    std::vector<Transfer> mine;
    std::vector<std::pair<u64, std::pair<u32, u32>>> peers;  // pn, [lo,hi)
    bool complete = true;
    std::string error;

    for (const auto &rg : ranges) {
      if (rg.second == 0) continue;
      const u64 end = std::min(rg.first + rg.second, num_elems_);
      if (rg.first >= end) continue;
      for (u64 pn = PageOf(rg.first); pn <= PageOf(end - 1); ++pn) {
        const u64 pbase = pn * epp;
        const u32 lo = static_cast<u32>(std::max(rg.first, pbase) - pbase);
        const u32 hi = static_cast<u32>(std::min(end, pbase + epp) - pbase);
        const u32 s = HomeSet(pn);
        auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(kAllocWaitMs);
        for (;;) {
          std::unique_lock<std::mutex> lk(sets_[s].mu);
          Frame *f = FindLocked(s, pn);
          if (f != nullptr) {
            f->pins.fetch_add(1);
            f->last_access = ++clock_;
            if (f->fetching != 0) {
              if (generation != 0) Stat(kGenBusy);
              peers.push_back({pn, {lo, hi}});
              break;
            }
            const bool stale = generation != 0 && f->generation < generation;
            if (!stale && Covers(*f, lo, hi)) {
              if (generation != 0) Stat(kGenOk);
              break;
            }
            if (stale) Stat(kGenStale);
            mine.push_back(PlanTransfer(f, s, pn, lo, hi, generation, stale));
            f->fetching = 1;
            break;
          }
          f = ClaimFrameLocked(s, pn);
          if (f != nullptr) {
            f->pins.fetch_add(1);
            f->last_access = ++clock_;
            f->fetching = 1;
            mine.push_back(PlanTransfer(f, s, pn, lo, hi, generation, true));
            break;
          }
          // NO FRAME. Finish our own claims first -- they hold nothing a
          // peer waits for once published, and they may be all that stands
          // between us and a free frame if we are our own obstacle.
          if (!mine.empty()) {
            lk.unlock();
            RunTransfers(mine, error);
            mine.clear();
            continue;
          }
          if (!wait_for_frames) {
            complete = false;
            break;
          }
          Stat(kAllocWaits);
          if (sets_[s].cv.wait_until(lk, deadline) ==
              std::cv_status::timeout) {
            lk.unlock();
            ReleasePins(ranges, pn, lo);
            throw std::runtime_error(
                "cpu_vector: no frame for page " + std::to_string(pn) +
                " in set " + std::to_string(s) + " after " +
                std::to_string(kAllocWaitMs) + " ms -- every frame is pinned "
                "or in flight; the cache is too small for what is held");
          }
        }
      }
    }

    RunTransfers(mine, error);
    if (!error.empty()) {
      // Failed frames were emptied (their pin went with them); every other
      // page this call pinned is released before the throw.
      ReleasePins(ranges, kNoPage, 0);
      throw std::runtime_error(error);
    }

    // Peers' frames: wait for their fill, then re-check coverage.
    std::vector<std::pair<u64, u64>> redo;
    for (const auto &pr : peers) {
      const u64 pn = pr.first;
      const u32 s = HomeSet(pn);
      std::unique_lock<std::mutex> lk(sets_[s].mu);
      Frame *f = nullptr;
      sets_[s].cv.wait(lk, [&] {
        f = FindLocked(s, pn);
        return f == nullptr || f->fetching == 0;
      });
      const bool ok = f != nullptr && Covers(*f, pr.second.first,
                                             pr.second.second) &&
                      (generation == 0 || f->generation >= generation);
      if (!ok) {
        // Our pin is on this frame only if it still holds pn; drop it and
        // fetch again from scratch, which pins afresh.
        if (f != nullptr && f->pins.load() > 0) f->pins.fetch_sub(1);
        redo.emplace_back(pn * epp + pr.second.first,
                          pr.second.second - pr.second.first);
      }
    }
    for (size_t i = 0; i < redo.size(); i += kMaxFetchRanges) {
      std::vector<std::pair<u64, u64>> chunk(
          redo.begin() + i,
          redo.begin() + std::min(redo.size(), i + kMaxFetchRanges));
      complete = FetchRanges(chunk, generation, wait_for_frames) && complete;
    }
    return complete;
  }

  /** After a fatal wait: drop the pins this call took on pages before
   *  (pn, lo) in range order. Best effort -- the call is about to throw. */
  void ReleasePins(const std::vector<std::pair<u64, u64>> &ranges, u64 stop_pn,
                   u32 stop_lo) {
    const u64 epp = ElemsPerPage();
    for (const auto &rg : ranges) {
      if (rg.second == 0) continue;
      const u64 end = std::min(rg.first + rg.second, num_elems_);
      for (u64 pn = PageOf(rg.first); rg.first < end && pn <= PageOf(end - 1);
           ++pn) {
        const u64 pbase = pn * epp;
        const u32 lo = static_cast<u32>(std::max(rg.first, pbase) - pbase);
        if (pn == stop_pn && lo == stop_lo) return;
        UnpinRange(pbase + lo, 1);
      }
    }
  }

  /** What to read into frame f so it covers [lo, hi). */
  Transfer PlanTransfer(Frame *f, u32 s, u64 pn, u32 lo, u32 hi, u64 gen,
                        bool replace) {
    Transfer t{};
    t.frame = f;
    t.set = s;
    t.pn = pn;
    t.generation = gen;
    t.replace = replace || f->valid_hi <= f->valid_lo;
    if (t.replace) {
      t.nlo[0] = lo;
      t.nhi[0] = hi;
      t.npieces = 1;
      return t;
    }
    // Extend the existing interval without re-reading (and so overwriting)
    // what it already holds -- those elements may carry unflushed writes.
    // A gap between the two intervals is read too, so the result is one
    // contiguous valid range.
    t.npieces = 0;
    if (lo < f->valid_lo) {
      t.nlo[t.npieces] = lo;
      t.nhi[t.npieces++] = f->valid_lo;
    }
    if (hi > f->valid_hi) {
      t.nlo[t.npieces] = f->valid_hi;
      t.nhi[t.npieces++] = hi;
    }
    return t;
  }

  /** Issue every planned get, wait for all, publish each frame. Errors are
   *  appended to `error`; the frame of a failed transfer is emptied. */
  void RunTransfers(std::vector<Transfer> &ts, std::string &error) {
    if (ts.empty()) return;
    for (auto &t : ts) {
      clio::cte::core::Context ctx = PageContext();
      if (t.generation != 0) {
        ctx.op_flags_ |= clio::cte::core::Context::kGenerational;
        ctx.generation_ = t.generation;
      } else {
        // A never-written page reads as zeros. The runtime reports a missing
        // blob as rc 1 -- the same code as a real failure -- so it cannot be
        // told apart afterwards; create_on_get_ makes the read succeed with
        // the buffer UNTOUCHED, hence the memset below.
        ctx.create_on_get_ = true;
      }
      for (u32 k = 0; k < t.npieces; ++k) {
        const u64 lo_b = static_cast<u64>(t.nlo[k]) * sizeof(T);
        const u64 n_b = static_cast<u64>(t.nhi[k] - t.nlo[k]) * sizeof(T);
        if (t.generation == 0) {
          std::memset(t.frame->data + lo_b, 0, static_cast<size_t>(n_b));
        }
        t.fut[k] = core_.AsyncGetBlob(tag_id_, PageName(t.pn), lo_b, n_b, 0u,
                                      t.frame->data + lo_b,
                                      clio::run::PoolQuery::Dynamic(), ctx);
      }
    }
    for (auto &t : ts) {
      bool ok = true;
      for (u32 k = 0; k < t.npieces; ++k) {
        t.fut[k].Wait();
        const int rc =
            t.fut[k].get() != nullptr ? t.fut[k]->GetReturnCode() : -1;
        if (rc == 0) continue;
        ok = false;
        Stat(kGetErrors);
        if (error.empty()) {
          error = "cpu_vector: get of page " + std::to_string(t.pn) +
                  " failed rc=" + std::to_string(rc);
        }
      }
      Stat(kFaults);
      std::lock_guard<std::mutex> lk(sets_[t.set].mu);
      Frame &f = *t.frame;
      f.fetching = 0;
      if (ok) {
        if (t.replace) {
          f.valid_lo = t.nlo[0];
          f.valid_hi = t.nhi[0];
        } else {
          for (u32 k = 0; k < t.npieces; ++k) Widen(f, t.nlo[k], t.nhi[k]);
        }
        if (t.generation != 0) f.generation = t.generation;
      } else {
        // Our pin goes with the frame; give the region back.
        char *d = f.data;
        ResetFrame(f);
        std::lock_guard<std::mutex> fl(free_mu_);
        free_.push_back(static_cast<u32>((d - regions_.get()) / page_bytes_));
      }
      sets_[t.set].cv.notify_all();
    }
  }

  /** The flush engine: send the valid part of each named range, wait. */
  void FlushRanges(const std::vector<std::pair<u64, u64>> &ranges,
                   u64 generation) {
    if (ranges.size() > kMaxFetchRanges) {
      throw std::runtime_error("cpu_vector: Flush names more than "
                               "kMaxFetchRanges ranges");
    }
    struct Put {
      Frame *frame;
      u32 set;
      u64 pn;
      clio::run::Future<clio::cte::core::PutBlobTask> fut;
    };
    const u64 epp = ElemsPerPage();
    clio::cte::core::Context ctx = PageContext();
    if (generation != 0) {
      ctx.op_flags_ |= clio::cte::core::Context::kGenerational;
      ctx.generation_ = generation;
    }
    std::vector<Put> puts;
    for (const auto &rg : ranges) {
      if (rg.second == 0) continue;
      const u64 end = std::min(rg.first + rg.second, num_elems_);
      if (rg.first >= end) continue;
      for (u64 pn = PageOf(rg.first); pn <= PageOf(end - 1); ++pn) {
        const u64 pbase = pn * epp;
        u32 lo = static_cast<u32>(std::max(rg.first, pbase) - pbase);
        u32 hi = static_cast<u32>(std::min(end, pbase + epp) - pbase);
        const u32 s = HomeSet(pn);
        std::unique_lock<std::mutex> lk(sets_[s].mu);
        Frame *f = nullptr;
        sets_[s].cv.wait(lk, [&] {
          f = FindLocked(s, pn);
          return f == nullptr || f->fetching == 0;
        });
        if (f != nullptr) {
          lo = std::max(lo, f->valid_lo);
          hi = std::min(hi, f->valid_hi);
        }
        if (f == nullptr || hi <= lo) {
          Stat(kFlushSkipped);
          continue;
        }
        ++f->flushing;
        lk.unlock();
        const u64 lo_b = static_cast<u64>(lo) * sizeof(T);
        const u64 n_b = static_cast<u64>(hi - lo) * sizeof(T);
        puts.push_back(Put{f, s, pn,
                           core_.AsyncPutBlob(tag_id_, PageName(pn), lo_b, n_b,
                                              f->data + lo_b, kVectorBlobScore,
                                              ctx, 0u,
                                              clio::run::PoolQuery::Dynamic())});
      }
    }
    std::string error;
    for (auto &p : puts) {
      p.fut.Wait();
      const int rc = p.fut.get() != nullptr ? p.fut->GetReturnCode() : -1;
      Stat(kPuts);
      if (rc != 0) {
        Stat(kPutErrors);
        if (error.empty()) {
          error = "cpu_vector: put of page " + std::to_string(p.pn) +
                  " failed rc=" + std::to_string(rc);
        }
      }
      std::lock_guard<std::mutex> lk(sets_[p.set].mu);
      --p.frame->flushing;
      sets_[p.set].cv.notify_all();
    }
    if (!error.empty()) throw std::runtime_error(error);
  }

  void SubmitHintsLocked(const std::vector<PrefetchHint> &hints) {
    std::map<u64, float> merged;
    for (const auto &h : hints) merged[h.page_] = h.score_;
    clio::run::shared_ptr<clio::cte::core::PodMultiScoreTask> batch;
    u32 in_batch = 0;
    for (const auto &kv : merged) {
      auto seen = last_sent_.find(kv.first);
      if (seen != last_sent_.end() &&
          std::fabs(seen->second - kv.second) < 1e-3f) {
        ++prefetch_stats_.deduped_;
        continue;
      }
      const float prev =
          seen == last_sent_.end() ? kVectorBlobScore : seen->second;
      if (kv.second > prev) ++prefetch_stats_.promotions_;
      else if (kv.second < prev) ++prefetch_stats_.demotions_;
      last_sent_[kv.first] = kv.second;
      if (!batch) {
        batch = core_.template NewPodBatch<clio::cte::core::PodMultiScoreTask>(
            tag_id_, PageContext(), clio::cte::core::kCtePrefetchHint,
            clio::run::PoolQuery::Dynamic());
        in_batch = 0;
      }
      batch.get()->Add(PageName(kv.first).c_str(), 0, 0, ctp::ipc::ShmPtr<>(),
                       kv.second);
      ++in_batch;
      ++prefetch_stats_.sent_;
      if (in_batch >= clio::cte::core::kPodMultiMax) {
        pending_.push_back(core_.AsyncPodBatch(batch));
        ++prefetch_stats_.batches_;
        batch = {};
      }
    }
    if (batch && in_batch > 0) {
      pending_.push_back(core_.AsyncPodBatch(batch));
      ++prefetch_stats_.batches_;
    }
  }

  void ReapPrefetchLocked(bool block) {
    size_t keep = 0;
    for (size_t i = 0; i < pending_.size(); ++i) {
      auto &f = pending_[i];
      if (f.IsNull()) continue;
      if (block) {
        f.Wait();
      } else if (!f.IsComplete()) {
        if (keep != i) pending_[keep] = std::move(f);
        ++keep;
        continue;
      }
      const u32 n = std::min(f->count_, clio::cte::core::kPodMultiMax);
      for (u32 r = 0; r < n; ++r) {
        const u32 rc = f->reqs_[r].rc_;
        if (rc == 0) continue;
        if (rc == 3) ++prefetch_stats_.not_found_;
        else ++prefetch_stats_.errors_;
      }
    }
    pending_.resize(block ? 0 : keep);
  }

  /** A page's blob name: decimal, as gpu_vector. */
  std::string PageName(u64 pg) const { return std::to_string(pg); }

  clio::cte::core::Context PageContext() const {
    clio::cte::core::Context c;
    c.compress_lib_ = compress_lib_;
    c.compress_preset_ = compress_preset_;
    return c;
  }

  bool ReadPage(u64 pg, char *out) {
    auto ctx = PageContext();
    auto f = core_.AsyncGetBlob(tag_id_, PageName(pg), 0, page_bytes_, 0u, out,
                                clio::run::PoolQuery::Dynamic(), ctx);
    f.Wait();
    return f.get() != nullptr && f->GetReturnCode() == 0;
  }

  clio::run::PoolId storage_pool_id_;
  int compress_lib_ = 0;
  int compress_preset_ = 1;
  u64 page_bytes_ = 0;
  u32 nsets_ = 0;
  u32 set_size_ = 0;
  u32 capacity_pages_ = 0;
  u64 num_elems_ = 0;
  std::string tag_name_;
  clio::cte::core::TagId tag_id_;
  clio::cte::core::Client core_;
  std::string ckpt_src_;

  std::unique_ptr<Set[]> sets_;
  std::unique_ptr<Frame[]> frames_;
  std::unique_ptr<char[]> regions_;
  std::mutex free_mu_;
  std::vector<u32> free_;
  std::atomic<u64> clock_{0};

  std::atomic<bool> stats_on_{false};
  std::atomic<u64> stat_[kNumStats] = {};

  std::mutex prefetch_mu_;
  PrefetchStats prefetch_stats_;
  std::map<u64, float> last_sent_;
  std::vector<clio::run::Future<clio::cte::core::PodMultiScoreTask>> pending_;
};

}  // namespace clio::cte::cpu_vector

#endif  // CLIO_CTE_CPU_VECTOR_CPU_VECTOR_H_
