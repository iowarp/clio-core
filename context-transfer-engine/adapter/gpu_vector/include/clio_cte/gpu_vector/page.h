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

/**
 * The CTE BLOB score every page is written at, by both the flush path and the
 * create-on-get in the fetch path.
 *
 * NOT the same thing as kDefaultScore, which is a frame's eviction rank
 * inside the GPU page cache and never leaves the device. This one is the
 * number the DPE tiers on, and the two being both "a score" and both
 * per-page is exactly how they got confused.
 *
 * IT IS 0.5, AND THAT DECIDES WHICH TIERS A PAGE CAN EVER REACH. MaxBwDpe
 * splits tiers on target_score <= blob_score, so a tier scored ABOVE 0.5 is
 * excluded from the preferred group for every page this vector writes. In the
 * Gray-Scott config -- HBM 1.0, host RAM 0.2, storage 0.0 -- that means pages
 * never land in HBM at all on their own, which the benchmark's TIER SPLIT
 * line reports as "nothing landed in the fastest tier". A prefetcher hinting
 * at 1.0 is what makes the fast tier reachable.
 *
 * Named here rather than repeated as a literal at the two Add() call sites
 * because a prefetcher has to know what score a page starts at in order to
 * say whether its own hint is a promotion or a demotion.
 */
constexpr float kVectorBlobScore = 0.5f;

using MultiGetSlot = clio::cte::core::PodMultiGetBlobTask;
using MultiPutSlot = clio::cte::core::PodMultiPutBlobTask;

/** One resident page. */
struct Page {
  clio::run::u64 page_num;    // kNoPage when free
  void *data;                 // this page's bytes
  float score;                // eviction rank; EvictPages takes the lowest
  clio::run::u64 last_access; // breaks score ties (LRU)
  clio::run::u32 pins;        // holders; a pinned page is never a victim
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
  /** Generation the NEXT fetch will name (0 = ordinary fetch). Written by
   *  BeginFetch before it may have to drain a previous fetch, so it is NOT
   *  the generation the in-flight batch was fetched at -- see
   *  fetch_gen_sub. */
  clio::run::u64 fetch_generation;
  /** THE GENERATION THE IN-FLIGHT BATCH WAS ACTUALLY SUBMITTED AT.
   *
   *  PublishFetch must stamp the frames with the version they HOLD, and that
   *  is this one, not fetch_generation. BeginFetch(g) assigns
   *  fetch_generation = g and only then drains a still-in-flight previous
   *  fetch, so publishing with fetch_generation stamped the PREVIOUS batch's
   *  pages with g -- pages claiming a version they do not hold. The next
   *  demand for g then computed `p->generation < g` as false and skipped the
   *  refetch: no fault, no error, stale data. That is the md_bench halo
   *  exchange moving nothing (x faults=36, get_errors=0, identical with the
   *  exchange disabled). */
  clio::run::u64 fetch_gen_sub;
  /** Generation stamped on the in-flight flush (0 = ordinary flush). */
  clio::run::u64 flush_generation;
  /** A SubmitFetch that could not claim every page keeps what it claimed and
   *  says where to resume: fetch_partial != 0 means fetch_n records are
   *  staged and the walk continues at range fetch_resume_r, page
   *  fetch_resume_pn. fetch_stalls counts the yields taken while waiting. */
  clio::run::u32 fetch_partial;
  clio::run::u32 fetch_resume_r;
  clio::run::u64 fetch_resume_pn;
  clio::run::u32 fetch_stalls;
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
