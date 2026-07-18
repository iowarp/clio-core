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
 * vector (the issue-#700 MegaMmap Transaction API, adapted to GPU + compression).
 *
 * A Transaction sweeps a set of page-windows and, as #700 specifies, PREFETCHES
 * THE NEXT window while the body computes on the current one (and evicts the
 * window behind by reusing its buffer). So the fetch/decompress of window W+1
 * overlaps the compute of window W -- which is exactly what hides slow-tier
 * (IO-bound) reads: the GPU never stalls waiting for the next window to arrive.
 *
 * A concrete Transaction supplies the ORDER of window starts (the access
 * pattern): SequentialTransaction sweeps ascending; PseudoRandomTransaction
 * visits windows in a caller-provided (e.g. shuffled) order.
 *
 * The vector must be created with gpu_pages_per_block == 2*window (single-block,
 * single-tier): two buffers, so window W+1 loads into one while the body reads
 * window W from the other. `window` here is gpu_pages_per_block/2.
 *
 * body signature: void(u64 win_lo_elem, u64 win_hi_elem, DeviceView<T> view,
 *                      cudaStream_t stream).
 * The body LAUNCHES its device kernel(s) on `stream` and does NOT synchronize;
 * the transaction synchronizes after issuing the next window's prefetch, so the
 * body's compute overlaps that prefetch.
 */
template <typename T>
class Transaction {
 public:
  explicit Transaction(Vector<T> &vec) : vec_(vec) {
    cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
  }
  virtual ~Transaction() {
    if (stream_) cudaStreamDestroy(stream_);
  }

  template <typename Body>
  void Iterate(Body &&body) {
    const std::vector<clio::run::u64> starts = WindowStarts();
    if (starts.empty()) return;
    const clio::run::u32 window = Window();
    const clio::run::u64 cap = vec_.Device().page_capacity_t;
    // Prime buffer 0 with the first window.
    vec_.PrefetchPagesSync(starts[0], window, /*slot_base=*/0);
    for (size_t i = 0; i < starts.size(); ++i) {
      const clio::run::u64 lo = starts[i] * cap;
      const clio::run::u64 hi = lo + static_cast<clio::run::u64>(window) * cap;
      body(lo, hi, vec_.Device(), stream_);  // launch compute for window i (async)
      if (i + 1 < starts.size()) {
        // Prefetch window i+1 into the OTHER buffer while the body runs.
        vec_.PrefetchPagesSync(starts[i + 1], window,
                               static_cast<clio::run::u32>((i + 1) & 1u) * window);
      }
      cudaStreamSynchronize(stream_);  // wait for window i's compute
    }
  }

  clio::run::u32 Window() const {
    return vec_.Device().base.gpu_pages_per_block / 2;  // double-buffered
  }

 protected:
  virtual std::vector<clio::run::u64> WindowStarts() = 0;
  Vector<T> &vec_;
  cudaStream_t stream_ = nullptr;
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
    const clio::run::u32 window = this->Window();
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
