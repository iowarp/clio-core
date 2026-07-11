/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#ifndef CLIO_CTE_GPU_VECTOR_TRANSACTION_H_
#define CLIO_CTE_GPU_VECTOR_TRANSACTION_H_

#include "gpu_vector.h"

#include <vector>

namespace clio::cte::gpu_vector {

#if !CTP_IS_DEVICE_PASS

/**
 * Windowed, host-orchestrated prefetch transactions for the compressed GPU
 * vector (the issue-#700 Transaction API, adapted to GPU + compression).
 *
 * WHY: an on-device page fault cannot be serviced by a GPU compressor on a
 * single device -- a spin-waiting fault kernel and the cuSZp decompress kernel
 * do not co-schedule, so they deadlock. A Transaction instead PREFETCHES a
 * window of pages into HBM from the host (GPU idle) BEFORE the device kernel
 * touches them, so the kernel only ever reads resident pages. The resident
 * footprint is one window (gpu_pages_per_block) regardless of dataset size, so
 * datasets far larger than the HBM cache are swept window-by-window.
 *
 * A concrete Transaction supplies the ORDER of window starts (the access
 * pattern): SequentialTransaction sweeps ascending; PseudoRandomTransaction
 * visits windows in a caller-provided (e.g. shuffled) order.
 *
 * Single-block, single-tier vectors (nblocks == 1, host_pages_per_block == 0);
 * the window size equals the HBM cache (gpu_pages_per_block).
 *
 * body signature: void(u64 win_lo_elem, u64 win_hi_elem, DeviceView<T> view).
 * The body launches its device kernel(s) over [win_lo_elem, win_hi_elem) and is
 * responsible for synchronizing before the next window reuses the slots.
 */
template <typename T>
class Transaction {
 public:
  explicit Transaction(Vector<T> &vec) : vec_(vec) {}
  virtual ~Transaction() = default;

  template <typename Body>
  void Iterate(Body &&body) {
    auto view = vec_.Device();
    const clio::run::u64 cap = view.page_capacity_t;
    const clio::run::u32 window = view.base.gpu_pages_per_block;
    for (clio::run::u64 fp : WindowStarts()) {
      vec_.PrefetchWindowSync(fp);  // host decompress the window into HBM (GPU idle)
      body(fp * cap, (fp + window) * cap, vec_.Device());
    }
  }

 protected:
  // The sequence of window-start page indices (each window is `window` pages).
  virtual std::vector<clio::run::u64> WindowStarts() = 0;
  Vector<T> &vec_;
};

/** Ascending sweep of [first_page, first_page + npages) in window steps. */
template <typename T>
class SequentialTransaction : public Transaction<T> {
 public:
  SequentialTransaction(Vector<T> &vec, clio::run::u64 first_page,
                        clio::run::u64 npages)
      : Transaction<T>(vec), first_(first_page), n_(npages) {}

 protected:
  std::vector<clio::run::u64> WindowStarts() override {
    const clio::run::u32 window =
        this->vec_.Device().base.gpu_pages_per_block;
    std::vector<clio::run::u64> starts;
    for (clio::run::u64 p = first_; p < first_ + n_; p += window)
      starts.push_back(p);
    return starts;
  }

 private:
  clio::run::u64 first_, n_;
};

/** Visit windows in a caller-provided order (e.g. shuffled window starts). */
template <typename T>
class PseudoRandomTransaction : public Transaction<T> {
 public:
  PseudoRandomTransaction(Vector<T> &vec, std::vector<clio::run::u64> starts)
      : Transaction<T>(vec), starts_(std::move(starts)) {}

 protected:
  std::vector<clio::run::u64> WindowStarts() override { return starts_; }

 private:
  std::vector<clio::run::u64> starts_;
};

#endif  // !CTP_IS_DEVICE_PASS

}  // namespace clio::cte::gpu_vector

#endif  // CLIO_CTE_GPU_VECTOR_TRANSACTION_H_
