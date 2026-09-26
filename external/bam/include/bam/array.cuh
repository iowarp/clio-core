/**
 * bam/array.cuh -- bam::Array<T> for transparent storage-backed GPU arrays
 *
 * Memory hierarchy:
 *   GPU HBM (VRAM) ← page cache (fast, limited)
 *       ↕ cache miss / writeback
 *   Host DRAM       ← backing store (pinned cudaMallocHost)
 *       ↕ (optional, future)
 *   NVMe SSD        ← persistent storage
 *
 * Usage from a GPU kernel:
 *
 *   __global__ void my_kernel(bam::ArrayDevice<float> arr) {
 *       float val = arr.read(threadIdx.x);
 *       arr.write(threadIdx.x, val * 2.0f);
 *   }
 *
 * Host-side setup:
 *
 *   bam::Array<float> arr(num_elements, cache);
 *   my_kernel<<<grid, block>>>(arr.device());
 */
#ifndef BAM_ARRAY_CUH
#define BAM_ARRAY_CUH

#include <bam/types.h>
#include <bam/page_cache.cuh>
#include <cuda_runtime.h>
#include <cstddef>

namespace bam {

/* ------------------------------------------------------------------ */
/* Device-side array accessor                                          */
/* ------------------------------------------------------------------ */


/* ------------------------------------------------------------------ */
/* Host-side array manager                                             */
/* ------------------------------------------------------------------ */

/* ------------------------------------------------------------------ */
/* Upstream-compatible device API                                       */
/*                                                                      */
/* github.com/ZaidQureshi/bam exposes THREE ways to hold a page across   */
/* many accesses, and they are the reason BaM is fast on sequential      */
/* work: acquire_page/release_page (which hand back the element range    */
/* the page covers, so the kernel streams it on a raw pointer),          */
/* bam_ptr (a smart pointer that keeps the page between accesses), and a */
/* TLB. This file previously offered only read(idx)/write(idx,val),      */
/* which re-resolves the page -- two atomics -- on EVERY element. A      */
/* benchmark written against that API measures the one access pattern    */
/* upstream provides these mechanisms to avoid.                          */
/* ------------------------------------------------------------------ */

/**
 * The page-resident window around a logical index: `data` is a raw pointer
 * into the pinned cache page, valid for elements [start, end) while the
 * pin returned in `r` is held.
 */
template <typename T>
struct PageWindow {
  T      *data  = nullptr;   // points at element `start`
  size_t  start = 0;
  size_t  end   = 0;
  uint64_t page_off = 0;     // the pin token: pass back to release_page
};

template <typename T>
struct array_d_t {
  PageCacheDeviceState cache_state;
  QueuePairDevice      qp;
  uint64_t            *buf_bus_addrs;
  const uint8_t       *host_base;
  uint64_t             num_elements;
  uint64_t             total_bytes;
  BackendType          backend;

  /** Fill a cache page from the backend. Shared by every accessor. */
  __device__ void load_page_(uint8_t *page, uint64_t page_off) const {
    if (backend == BackendType::kNvme) {
      const uint32_t slot =
          (uint32_t)((page_off >> cache_state.page_shift) %
                     cache_state.num_pages);
      nvme_read_page(const_cast<QueuePairDevice &>(qp), buf_bus_addrs[slot],
                     page_off, (uint32_t)cache_state.page_size);
    } else {
      host_read_page(page, host_base, page_off,
                     (uint32_t)cache_state.page_size);
    }
    page_cache_finish_load(const_cast<PageCacheDeviceState &>(cache_state),
                           page_off);
  }

  /**
   * ACQUIRE ONCE, STREAM THE PAGE. `start`/`end` bound the elements this
   * page covers so the caller can loop on `win.data` without paying a
   * translation per element. Release exactly once when done.
   */
  __device__ PageWindow<T> acquire_page(size_t i) const {
    PageWindow<T> win;
    const uint64_t byte_off = (uint64_t)i * sizeof(T);
    const uint64_t page_off =
        byte_off & ~((uint64_t)cache_state.page_size - 1);
    bool needs_load = false;
    uint8_t *page = page_cache_acquire_pinned(
        const_cast<PageCacheDeviceState &>(cache_state), page_off,
        &needs_load);
    if (needs_load) load_page_(page, page_off);
    const size_t per_page = cache_state.page_size / sizeof(T);
    win.start = (size_t)(page_off / sizeof(T));
    win.end = win.start + per_page;
    if (win.end > num_elements) win.end = (size_t)num_elements;
    win.data = reinterpret_cast<T *>(page);
    win.page_off = page_off;
    return win;
  }

  __device__ void release_page(const PageWindow<T> &win) const {
    page_cache_release(const_cast<PageCacheDeviceState &>(cache_state),
                       win.page_off);
  }

  __device__ void mark_page_dirty(size_t i) const {
    const uint64_t byte_off = (uint64_t)i * sizeof(T);
    page_cache_mark_dirty(const_cast<PageCacheDeviceState &>(cache_state),
                          byte_off & ~((uint64_t)cache_state.page_size - 1));
  }

  /** Single-element accessors. Each pays a full resolve -- that is what
   *  they are for; use acquire_page or bam_ptr for runs. */
  __device__ T seq_read(size_t i) const {
    PageWindow<T> w = acquire_page(i);
    T v = w.data[i - w.start];
    release_page(w);
    return v;
  }
  __device__ void seq_write(size_t i, T val) const {
    PageWindow<T> w = acquire_page(i);
    w.data[i - w.start] = val;
    mark_page_dirty(i);
    release_page(w);
  }
  __device__ T operator[](size_t i) const { return seq_read(i); }
  __device__ void operator()(size_t i, T val) const { seq_write(i, val); }

  /**
   * Read element at logical index.
   *
   * 1. Compute page-aligned offset and in-page offset.
   * 2. Acquire cache page (may be a hit or miss).
   * 3. On miss: load entire page from DRAM into HBM cache.
   * 4. Read the element from the HBM cache page.
   */
  __device__ T read(uint64_t idx) const {
    uint64_t byte_off = idx * sizeof(T);
    uint64_t page_off = byte_off & ~((uint64_t)cache_state.page_size - 1);
    uint32_t in_page  = (uint32_t)(byte_off & ((uint64_t)cache_state.page_size - 1));

    bool needs_load;
    uint8_t *page = page_cache_acquire_pinned(
        const_cast<PageCacheDeviceState &>(cache_state), page_off, &needs_load);

    if (needs_load) {
      if (backend == BackendType::kNvme) {
        uint32_t slot = (uint32_t)((page_off >> cache_state.page_shift) % cache_state.num_pages);
        nvme_read_page(const_cast<QueuePairDevice &>(qp),
                       buf_bus_addrs[slot], page_off, cache_state.page_size);
      } else {
        host_read_page(page, host_base, page_off, cache_state.page_size);
      }
      page_cache_finish_load(
          const_cast<PageCacheDeviceState &>(cache_state), page_off);
    }

    T val = *reinterpret_cast<const T *>(page + in_page);
    page_cache_release(const_cast<PageCacheDeviceState &>(cache_state),
                       page_off);
    return val;
  }

  /**
   * Write element at logical index.
   *
   * Same as read but also writes back and marks dirty.
   */
  __device__ void write(uint64_t idx, T val) {
    uint64_t byte_off = idx * sizeof(T);
    uint64_t page_off = byte_off & ~((uint64_t)cache_state.page_size - 1);
    uint32_t in_page  = (uint32_t)(byte_off & ((uint64_t)cache_state.page_size - 1));

    bool needs_load;
    uint8_t *page = page_cache_acquire_pinned(
        const_cast<PageCacheDeviceState &>(cache_state), page_off, &needs_load);

    if (needs_load) {
      // Read-before-write for partial-page updates
      if (backend == BackendType::kNvme) {
        uint32_t slot = (uint32_t)((page_off >> cache_state.page_shift) % cache_state.num_pages);
        nvme_read_page(const_cast<QueuePairDevice &>(qp),
                       buf_bus_addrs[slot], page_off, cache_state.page_size);
      } else {
        host_read_page(page, host_base, page_off, cache_state.page_size);
      }
      page_cache_finish_load(
          const_cast<PageCacheDeviceState &>(cache_state), page_off);
    }

    *reinterpret_cast<T *>(page + in_page) = val;

    // Write through to DRAM backing store for durability
    if (backend == BackendType::kHostMemory) {
      host_write_page(page, const_cast<uint8_t *>(host_base),
                      page_off, cache_state.page_size);
    }

    page_cache_mark_dirty(
        const_cast<PageCacheDeviceState &>(cache_state), page_off);
    page_cache_release(const_cast<PageCacheDeviceState &>(cache_state),
                       page_off);
  }
};

/** Backwards-compatible name for the pre-parity accessor type. */
template <typename T>
using ArrayDevice = array_d_t<T>;

/**
 * bam_ptr -- keeps the acquired page between accesses and only re-resolves
 * when the index leaves it. This is the abstraction that makes a strided or
 * clustered access pattern cost one translation per PAGE instead of one per
 * element.
 *
 * Releases before it acquires (see the one-pin-per-thread contract on
 * PageCacheDeviceState::page_refs): on a direct-mapped cache, holding the
 * old page while claiming a new one that maps to the same slot would be a
 * wait on yourself.
 */
template <typename T>
struct bam_ptr {
  const array_d_t<T> *arr;
  PageWindow<T> win;
  bool held;

  __device__ explicit bam_ptr(const array_d_t<T> *a)
      : arr(a), win(), held(false) {}
  __device__ ~bam_ptr() { reset(); }

  __device__ void reset() {
    if (held) { arr->release_page(win); held = false; }
  }
  __device__ void update_page(size_t i) {
    if (held && i >= win.start && i < win.end) return;   // still covered
    reset();
    win = arr->acquire_page(i);
    held = true;
  }
  __device__ T &operator[](size_t i) {
    update_page(i);
    return win.data[i - win.start];
  }
  __device__ T at(size_t i) {
    update_page(i);
    // BYPASS L1. When the page was filled by the emulated controller the
    // bytes arrived by host DMA, and the per-SM L1 is not coherent with an
    // external writer -- an ordinary load can return the line as it was
    // before the fill (observed as every list entry reading zero, so the
    // force pass found no pairs and PE came out 0). A cache-global load
    // reads past L1. Costs nothing on a hit that was filled by the device
    // itself, which is why it is unconditional rather than a mode.
    const T *p = win.data + (i - win.start);
    if (sizeof(T) == 4) {
      const int v = __ldcg(reinterpret_cast<const int *>(p));
      T out;
      __builtin_memcpy(&out, &v, 4);
      return out;
    }
    return *p;
  }
};

class PageCache;  // Forward declaration

template <typename T>
class Array {
 public:
  Array(uint64_t num_elements, PageCache &cache);
  ~Array();

  ArrayDevice<T> device() const { return dev_; }
  uint64_t total_bytes() const { return num_elements_ * sizeof(T); }
  uint64_t size() const { return num_elements_; }

  /**
   * Pre-populate the DRAM backing store with host data.
   */
  int load_from_host(const T *host_data, uint64_t count);

 private:
  uint64_t num_elements_;
  ArrayDevice<T> dev_;
  PageCache *cache_;
};

}  // namespace bam

#endif  // BAM_ARRAY_CUH
