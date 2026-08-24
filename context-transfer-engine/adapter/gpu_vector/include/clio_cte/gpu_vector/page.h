/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * One cached page of a gpu_vector, and the per-block task set.
 *
 * A page is one CTE blob ("p<N>"). Pages are grouped per block: block b owns
 * pages[b * pages_per_block .. +pages_per_block).
 */
#ifndef CLIO_CTE_GPU_VECTOR_PAGE_H_
#define CLIO_CTE_GPU_VECTOR_PAGE_H_

#include <clio_runtime/gpu/future.h>
#include <clio_runtime/types.h>
#include <clio_cte/core/core_tasks.h>

namespace clio::cte::gpu_vector {

/** page_num of an empty slot. */
constexpr clio::run::u64 kNoPage = ~static_cast<clio::run::u64>(0);

using MultiGetSlot = clio::cte::core::PodMultiGetBlobTask;
using MultiPutSlot = clio::cte::core::PodMultiPutBlobTask;

/** One resident page. */
struct Page {
  clio::run::u64 page_num;    // kNoPage when free
  void *data;                 // this page's bytes
  float score;                // eviction rank; EvictPages takes the lowest
  clio::run::u64 last_access; // breaks score ties (LRU)
  clio::run::u32 pins;        // holders; a pinned page is never a victim
  clio::run::u32 dirty;       // written since faulted or flushed
  clio::run::u32 flushing;    // a put is outstanding
  clio::run::u32 fetching;    // a get is outstanding
  /** WHICH ELEMENTS OF THIS FRAME ARE ACTUALLY THERE, as offsets within the
   *  page. Residency is per PAGE but a transfer is per RANGE, so "this page
   *  is present" is not the same statement as "the bytes you asked for are
   *  present": a fetch of part of a page leaves the rest of the frame
   *  holding whatever it held before. valid_hi <= valid_lo means empty. */
  clio::run::u32 valid_lo;
  clio::run::u32 valid_hi;
  /** Generation of the bytes in this frame, for generational reads. 0 means
   *  "unknown" -- a frame filled by a non-generational fetch says nothing
   *  about which publication it caught. */
  clio::run::u64 generation;
};

/**
 * The tasks a block owns. One set per page table: a bulk put for flushing and
 * a bulk get for fetching.
 */
struct BlockTasks {
  MultiPutSlot *flush;
  MultiGetSlot *fetch;
  clio::run::gpu::Future<clio::cte::core::PodMultiPutBlobTask> flush_fut;
  clio::run::gpu::Future<clio::cte::core::PodMultiGetBlobTask> fetch_fut;
  clio::run::u32 flush_n;     // records in the outstanding flush
  clio::run::u32 fetch_n;     // records in the outstanding fetch
  clio::run::u32 flush_busy;  // a flush is in flight
  clio::run::u32 fetch_busy;  // a fetch is in flight
  clio::run::u32 seq;         // bumped per submission so TaskIds differ
  /** Page table index each record came from, so completion can clear the
   *  right frames' flags. */
  clio::run::u32 flush_slot[clio::cte::core::kPodMultiMax];
  clio::run::u32 fetch_slot[clio::cte::core::kPodMultiMax];
  /** The element interval each fetch record covers, so completion can widen
   *  the frame's valid range to exactly what landed. */
  clio::run::u32 fetch_vlo[clio::cte::core::kPodMultiMax];
  clio::run::u32 fetch_vhi[clio::cte::core::kPodMultiMax];
  /** Generation the in-flight fetch is waiting for (0 = ordinary fetch). */
  clio::run::u64 fetch_generation;
  /** Generation stamped on the in-flight flush (0 = ordinary flush). */
  clio::run::u64 flush_generation;
};

/**
 * A TaskId that is actually unique on the device.
 *
 * CreateTaskId()'s device implementation is a stub returning a constant, so
 * every task submitted from a kernel would share one identity. Identity here
 * is (table, kind, submission sequence).
 */
CTP_INLINE_CROSS_FUN clio::run::TaskId DeviceTaskId(clio::run::u64 table,
                                                    clio::run::u32 kind,
                                                    clio::run::u32 seq) {
  clio::run::TaskId id;
  id.pid_ = 0;
  id.tid_ = 0;
  id.major_ = static_cast<clio::run::u32>(table) * 2u + kind + 1u;
  id.replica_id_ = 0;
  id.unique_ = seq + 1u;
  id.node_id_ = 0;
  return id;
}

constexpr clio::run::u32 kKindFlush = 0;
constexpr clio::run::u32 kKindFetch = 1;

}  // namespace clio::cte::gpu_vector

#endif  // CLIO_CTE_GPU_VECTOR_PAGE_H_
