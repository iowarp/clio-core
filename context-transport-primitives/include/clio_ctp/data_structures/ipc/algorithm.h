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

#ifndef CTP_SHM_SHM_DATA_STRUCTURES_CONTAINERS_ALGORITHM_H
#define CTP_SHM_SHM_DATA_STRUCTURES_CONTAINERS_ALGORITHM_H

#include <clio_ctp/constants/macros.h>
#include <clio_ctp/types/numbers.h>

namespace ctp {

/** Iterator type */
template <typename IterT>
using iterator_type_v =
    typename std::remove_reference<decltype(*std::declval<IterT>())>::type;

namespace detail {

/**
 * Swap two values.
 *
 * Deliberately kept out of namespace ctp. A generic, unconstrained `swap` in
 * ctp is found by ADL for every ctp type -- and for pointers to one -- where
 * it ties exactly with `std::swap` and makes the unqualified `swap()` calls
 * inside libc++ ambiguous. `std::vector<ctp::shared_ptr<T, A>>` reallocating
 * in `__split_buffer` is one such call.
 */
template <typename T>
CTP_CROSS_FUN void swap(T &a, T &b) {
  T tmp = a;
  a = b;
  b = tmp;
}

}  // namespace detail

/** Default sorting algorithm */
#define CTP_DEFAULT_SORT_CMP ctp ::less_than<iterator_type_v<IterT>>

/** Greater than comparison */
template <typename IterT>
CTP_CROSS_FUN bool greater_than(const IterT &a, const IterT &b) {
  return a > b;
}

/** Less than comparison */
template <typename IterT>
CTP_CROSS_FUN bool less_than(const IterT &a, const IterT &b) {
  return a < b;
}

/** Comparitor type */
template <typename T>
using Comparitor = bool (*)(const T &a, const T &b);

/** Check if a set of values is sorted */
template <typename IterT, typename CmpT = Comparitor<iterator_type_v<IterT>>>
CTP_CROSS_FUN bool is_sorted(IterT start, const IterT &end,
                              CmpT &&cmp = CTP_DEFAULT_SORT_CMP) {
  if (start == end) {
    return true;
  }
  auto prev = start;
  ++start;
  for (; start != end; ++start) {
    if (!cmp(*prev, *start)) {
      return false;
    }
    ++prev;
  }
  return true;
}

/** General sort forward declaration */
template <typename IterT, typename CmpT = Comparitor<iterator_type_v<IterT>>,
          int INSERT_SORT_CUTOFF = 32>
CTP_CROSS_FUN void sort(IterT start, const IterT &end,
                         CmpT &&cmp = CTP_DEFAULT_SORT_CMP);

/** Insertion sort forward declaration */
template <typename IterT, typename CmpT = Comparitor<iterator_type_v<IterT>>>
CTP_CROSS_FUN void insertion_sort(IterT start, const IterT &end,
                                   CmpT &&cmp = CTP_DEFAULT_SORT_CMP);

/** heap_sort forward declaration */
template <typename IterT, typename CmpT = Comparitor<iterator_type_v<IterT>>>
CTP_CROSS_FUN void heap_sort(IterT start, const IterT &end,
                              CmpT &&cmp = CTP_DEFAULT_SORT_CMP);

/** Quicksort forward declaration */
template <typename IterT, typename CmpT = Comparitor<iterator_type_v<IterT>>>
CTP_CROSS_FUN void quick_sort(IterT start, const IterT &end,
                               CmpT &&cmp = CTP_DEFAULT_SORT_CMP);

/** Sort using either insertion sort of quick_sort */
template <typename IterT, typename CmpT, int INSERT_SORT_CUTOFF>
CTP_CROSS_FUN void sort(IterT start, const IterT &end, CmpT &&cmp) {
  if (end - start < INSERT_SORT_CUTOFF) {
    insertion_sort(start, end, cmp);
  } else {
    quick_sort(start, end, cmp);
  }
}

/** Sort a set of values using insertion sort */
template <typename IterT, typename CmpT>
CTP_CROSS_FUN void insertion_sort(IterT start, const IterT &end, CmpT &&cmp) {
  if (start == end) {
    return;
  }
  for (auto i = start; i != end; ++i) {
    auto j = i;
    while (j != start && cmp(*(j), *(j - 1))) {
      detail::swap(*j, *(j - 1));
      --j;
    }
  }
}

/** Build a hepa from a vector */
template <typename IterT, typename CmpT>
CTP_CROSS_FUN void heapify(IterT start, size_t n, size_t i, CmpT &&cmp) {
  size_t largest = i;
  size_t left = 2 * i + 1;
  size_t right = 2 * i + 2;

  if (left < n && cmp(*(start + largest), *(start + left))) largest = left;

  if (right < n && cmp(*(start + largest), *(start + right))) largest = right;

  if (largest != i) {
    detail::swap(*(start + i), *(start + largest));
    heapify(start, n, largest, cmp);
  }
}

/** Sort a set of values with heap sort */
template <typename IterT, typename CmpT>
CTP_CROSS_FUN void heap_sort(IterT start, const IterT &end, CmpT &&cmp) {
  size_t n = end - start;

  // Build heap
  for (int i = n / 2 - 1; i >= 0; i--) heapify(start, n, i, cmp);

  // Extract elements from heap one by one
  for (int i = n - 1; i > 0; i--) {
    detail::swap(*start, *(start + i));
    heapify(start, i, 0, cmp);
  }
}

/** Sort a set of values using quick sort */
template <typename IterT, typename CmpT>
CTP_CROSS_FUN void quick_sort(IterT start, const IterT &end, CmpT &&cmp) {
  if (start >= end - 1) {
    return;
  }
  auto pivot = start + (end - start) / 2;
  auto pivot_val = *pivot;
  detail::swap(*pivot, *(end - 1));
  auto store = start;
  for (auto i = start; i < end - 1; ++i) {
    if (cmp(*(i), *(end - 1))) {
      detail::swap(*store, *i);
      ++store;
    }
  }
  detail::swap(*store, *(end - 1));
  sort(start, store, cmp);
  sort(store + 1, end, cmp);
}

}  // namespace ctp

#endif  // CTP_SHM_SHM_DATA_STRUCTURES_CONTAINERS_ALGORITHM_H
