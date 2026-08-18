/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * One cached page of a gpu_vector, and the per-block page table.
 *
 * A page is the unit of everything: it is one CTE blob ("p<N>"), one
 * PodGetBlobTask on fault, one PodPutBlobTask on flush, and one
 * PodReorganizeBlobTask on rescore. Each page owns its three task slots
 * outright rather than sharing a pool, so a fault, a flush and a rescore on
 * the same page never contend for a slot and the code never has to reason
 * about slot reuse.
 *
 * Pages are grouped per CUDA block. Block b owns pages[b * pages_per_block
 * .. +pages_per_block), so two blocks never touch the same entry and the
 * page table needs no locking at all.
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
 * A task is SELF-CONTAINED: its completion state lives in the task itself
 * (the separate FutureShm was removed when Future was simplified), so a
 * page's task slots are just the tasks. Each page owns its own three, which
 * is why a fault, a flush and a rescore on one page never contend.
 */
using PutSlot = clio::cte::core::PodPutBlobTask;
using GetSlot = clio::cte::core::PodGetBlobTask;
using RescoreSlot = clio::cte::core::PodReorganizeBlobTask;

using MultiPutSlot = clio::cte::core::PodMultiPutBlobTask;
using MultiGetSlot = clio::cte::core::PodMultiGetBlobTask;

/**
 * One batched-paging slot: a put task and a get task, their futures, and the
 * page-table index each record came from.
 *
 * The record->slot mapping has to be recorded at fill time because the batch
 * only carries blob NAMES; without it the completion pass cannot tell which
 * page a per-record return code belongs to, and a partial failure would have
 * to be treated as total.
 *
 * Put and get share the entry rather than getting separate pools because a
 * block is either flushing its cache or filling it; the two never overlap
 * within one block, which is the same reasoning that gives each page its own
 * three scalar slots.
 */
struct MultiBatch {
  MultiPutSlot *put;
  MultiGetSlot *get;
  clio::run::gpu::Future<clio::cte::core::PodMultiPutBlobTask> put_fut;
  clio::run::gpu::Future<clio::cte::core::PodMultiGetBlobTask> get_fut;
  clio::run::u32 page_slot[clio::cte::core::kPodMultiMax];
  /** Nonzero: get_fut carries an ASYNC batch that has not been settled.
   *  Its pages are marked fetching=2; AwaitFetch on any of them settles the
   *  whole batch. One outstanding async batch per table. */
  clio::run::u32 async_pending;
  /** Number of records in the un-settled async batch. */
  clio::run::u32 async_n;
};

/** One resident page. */
struct Page {
  /** Which page of the vector this slot holds; kNoPage when free. */
  clio::run::u64 page_num;
  /** Claim generation: bumped every time this slot is (re)claimed. A reader
   *  that captured gen at hold time and sees it unchanged after computing
   *  knows the slot was never recycled mid-read (seqlock validate). */
  clio::run::u32 gen;
  /** This page's bytes, in the device page backend. */
  void *data;
  /** Set by RescorePage; EvictPages takes the lowest. */
  float score;
  /**
   * The score the KERNEL set through RescorePage, kept apart from `score`.
   *
   * `score` cannot carry a user hint. It is bumped +1.0 on every touch, aged
   * -0.02 on every claim scan, and reset to 2.0/0.0 whenever the slot is
   * refilled, so a value written into it is eroded within ~50 claims and
   * clobbered by the next touch. An eviction policy reading it would rank the
   * access pattern, not the caller's intent.
   *
   * `has_user` separates "the user asked for 0.0" from "the user never said
   * anything", which no sentinel float could. Both are cleared when the slot
   * is refilled with a DIFFERENT page: a hint belongs to a page, not a slot.
   */
  float user_score;
  clio::run::u32 has_user;
  /** clock64() at last access; breaks score ties (LRU). */
  clio::run::u64 last_access;
  /**
   * Readers currently holding a RAW pointer into this page.
   *
   * A claim must not recycle a pinned slot. Without this, a kernel that
   * records page pointers and then reads through them has no protection at
   * all: an eviction mid-read leaves the pointers addressing another page's
   * bytes, every element still gets read, and the results are quietly wrong.
   * That is not hypothetical -- it is how the LJ pair kernel returned E_pair
   * -5.7229688 against -6.7733681 at 256,000 atoms with a PERFECT pair count.
   *
   * Counted rather than a flag because shared tables let several blocks hold
   * the same page at once.
   */
  clio::run::u32 pins;
  /** 1 when the page has been written since it was faulted or flushed. */
  clio::run::u32 dirty;
  /** 1 while a put issued by BeginFlush is still outstanding. */
  clio::run::u32 flushing;
  /**
   * 1 while an ASYNCHRONOUS get issued by BeginFetch is still in flight.
   *
   * A fetching page is not yet readable and must never be chosen as an
   * eviction victim -- its slot is already promised to an in-flight transfer.
   * Resolve waits on it rather than re-issuing.
   */
  clio::run::u32 fetching;
  /** 1 while a fire-and-forget rescore is still outstanding on this slot. */
  /**
   * This page's in-flight put is an EVICTION, not a plain writeback.
   *
   * ReapFlushed cannot otherwise tell them apart, and the two want opposite
   * things when the put lands: an eviction frees the slot, a BeginFlush
   * leaves the page resident and clean. Without this it did the former for
   * both, so every BeginFlush silently dropped the page from the cache and
   * counted itself as an eviction.
   */
  clio::run::u32 evicting;
  clio::run::u32 rescoring;
  /** Bumped on every task submission so no two carry the same TaskId. */
  clio::run::u32 seq;

  /** This page's own task slots, in the registered device backend. */
  PutSlot *put;
  GetSlot *get;
  RescoreSlot *rescore;

  /** Outstanding put, waited on by WaitFlush. */
  clio::run::gpu::Future<clio::cte::core::PodPutBlobTask> put_fut;
  /** Outstanding asynchronous get, waited on by Resolve/WaitFetch. */
  clio::run::gpu::Future<clio::cte::core::PodGetBlobTask> get_fut;
  /** Outstanding rescore, waited on before the slot is reused. */
  clio::run::gpu::Future<clio::cte::core::PodReorganizeBlobTask> rescore_fut;
};

/**
 * A TaskId that is actually unique on the device.
 *
 * CreateTaskId()'s device implementation is a STUB: it returns a constant
 * (major_=1, unique_=1, pid/tid=0) for every call, so every task submitted
 * from a kernel shares one identity. A single synchronous task survives
 * that; concurrent ones do not -- the runtime aborted with "null RunContext
 * -- task not executing" as soon as BeginFlush had several puts in flight.
 *
 * Identity here is (global page slot, task kind, submission sequence),
 * which is unique across blocks, across the three task kinds a page owns,
 * and across resubmissions of the same slot.
 */
CTP_INLINE_CROSS_FUN clio::run::TaskId DeviceTaskId(clio::run::u64 slot,
                                                    clio::run::u32 kind,
                                                    clio::run::u32 seq) {
  clio::run::TaskId id;
  id.pid_ = 0;
  id.tid_ = 0;
  id.major_ = static_cast<clio::run::u32>(slot) * 3u + kind + 1u;
  id.replica_id_ = 0;
  id.unique_ = seq + 1u;
  id.node_id_ = 0;
  return id;
}

/** Task kinds, used only to keep a page's three ids distinct. */
constexpr clio::run::u32 kKindPut = 0;
constexpr clio::run::u32 kKindGet = 1;
constexpr clio::run::u32 kKindRescore = 2;

/** Compose a page's blob name: "p<page_num>". Device-callable, no CRT. */
CTP_INLINE_CROSS_FUN void PageBlobName(clio::run::u64 page_num, char *out) {
  char digits[24];
  int n = 0;
  if (page_num == 0) {
    digits[n++] = '0';
  }
  while (page_num > 0) {
    digits[n++] = static_cast<char>('0' + (page_num % 10));
    page_num /= 10;
  }
  int w = 0;
  out[w++] = 'p';
  while (n > 0) {
    out[w++] = digits[--n];
  }
  out[w] = '\0';
}

}  // namespace clio::cte::gpu_vector

#endif  // CLIO_CTE_GPU_VECTOR_PAGE_H_
