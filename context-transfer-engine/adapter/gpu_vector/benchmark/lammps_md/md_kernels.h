/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * The MD workload: every device coroutine, once, for both backends.
 *
 * Included by EXACTLY ONE translation unit per build -- the CUDA launch TU or
 * the SYCL one -- because it DEFINES the device globals below. The host driver
 * reaches them through the narrow symbol enum in md_launch.h instead.
 */
#ifndef CLIO_GV_BENCH_MD_KERNELS_H_
#define CLIO_GV_BENCH_MD_KERNELS_H_

#include "md_common.h"

#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_runtime/gpu/yield_stack.h>
#include <clio_runtime/gpu/yieldable.h>

/**
 * EVERY MD DEVICE GLOBAL, IN ONE STRUCT.
 *
 * SYCL has no __device__ variables -- a mutable namespace-scope global is not
 * addressable from a kernel -- so the SYCL backend holds a device_global
 * POINTER to storage the host allocates, exactly as the runtime's yield
 * machinery does in yield_stack.h. Gathering all nineteen into one struct is
 * what makes that affordable: one pointer, one allocation, and the seam's
 * symbol accessors become a memcpy at an offset instead of nineteen
 * per-backend switch arms.
 *
 * CUDA keeps a DIRECT instance rather than a pointer, on purpose. This
 * benchmark is register-pressure sensitive (clio_coro_regcap, explicit launch
 * bounds), and putting a pointer load in front of every read of `md_flush` --
 * which the publish decision consults at write sites -- would change codegen
 * in the row that is already validated. The indirection is a SYCL cost, paid
 * only by SYCL.
 *
 * Field order is the seam's ABI: md_launch.h maps MdSym to offsetof here.
 */
/**
 * THE BLOCK'S REDUCTION SCRATCH, and the only shared memory left in this
 * benchmark.
 *
 * The block-uniform tables moved to CLIO_SHARED_PERSIST long ago -- that arena
 * is global-backed and survives a park -- so what remains is `red`, used only
 * inside the tree reduction at the end of a pass. That window contains no
 * co_await, which is what makes real local memory safe here: the yield driver
 * exits the kernel to suspend, so anything in __shared__ across a suspension
 * would be garbage.
 *
 * CUDA carves it out of the dynamic shared block, past the yield machinery's
 * own bytes. SYCL has no `extern __shared__`, so it takes a fixed-size group
 * local allocation instead -- MD_LB_THREADS is the launch-bounds maximum, so
 * it is exactly as large as the block can ever be.
 */
#if CTP_ENABLE_SYCL
#define MD_RED_SCRATCH(name)                                                  \
  double *name = *::sycl::ext::oneapi::group_local_memory_for_overwrite<      \
      double[MD_LB_THREADS]>(                                                 \
      ::sycl::ext::oneapi::this_work_item::get_nd_item<1>().get_group())
#else
#define MD_RED_SCRATCH(name)                                                  \
  extern __shared__ char smem_raw[];                                          \
  double *name = reinterpret_cast<double *>(smem_raw + CLIO_YIELD_SMEM_BYTES)
#endif

/**
 * The one device printf in this file. SYCL device code cannot call a
 * C-variadic function, and the compat header's replacement is injected into
 * clio::run::gpu -- which unqualified lookup at THIS file's global scope
 * would never reach. One call site, so it is spelled out rather than given a
 * global using-declaration that would put a variadic template in the same
 * overload set as the C library's printf for the whole TU.
 */
#if CTP_ENABLE_SYCL
#define MD_DEV_PRINTF ::clio::run::gpu::printf
#else
#define MD_DEV_PRINTF printf
#endif

struct MdGlobals {
  unsigned long long md_cyc[8];
  u32 publish;
  u32 pub_interior;
  u32 md_flush;
  unsigned long long gather_wrote;
  unsigned long long pub_flush_cyc;
  unsigned long long pub_fetch_cyc;
  unsigned long long gather_sample;
  u32 adm_used;
  u32 xmask[8192];
  u32 nlmask[8192];
  unsigned long long pages_done;
  unsigned long long third_sink;
  unsigned long long read_bad[4];
  unsigned long long read_sample[8];
  bool probe_ldcg;
  unsigned long long read_geom[12];
  unsigned long long blk_done[64];
  unsigned long long blk_last[64];
};

#if CTP_ENABLE_SYCL
/* The owning TU defines it; see the same rule in yield_stack.h. */
#if defined(CLIO_SYCL_KERNEL_TU)
inline ::sycl::ext::oneapi::experimental::device_global<
    MdGlobals *, ::clio::run::gpu::SyclImageScope> g_md_dg;
CTP_GPU_FUN inline MdGlobals &MdG() { return *g_md_dg.get(); }
#else
CTP_GPU_FUN inline MdGlobals &MdG() {
  // Parse-only: a kernel here would dereference null immediately, which is
  // the loudest failure available.
  return *static_cast<MdGlobals *>(nullptr);
}
#endif
#else
__device__ inline MdGlobals g_md_globals;
CTP_GPU_FUN inline MdGlobals &MdG() { return g_md_globals; }
#endif


/** Page iterations actually completed by the integrator, summed over
 *  blocks. Distinguishes WORK DROPPED (driver/park bookkeeping) from DATA
 *  CORRUPTED (the paging path): the expected count is exact and known. */
/** MD_NO_PUBLISH=1 drops every publish. The physics is then WRONG (blocks
 *  read each other's stale pages); it exists only so the device-side cost of
 *  publishing can be measured as a difference against a correct run. Uniform
 *  across all threads, so the co_await stays out of a divergent branch. */
/** Publish the slab INTERIOR (gen-0 track-the-store flushes). Single node
 *  needs it -- OOC eviction performs no I/O and --ckpt reads the store.
 *  Decomposed runs are resident, and a peer only ever reads the BOUNDARY
 *  planes, which exchange() publishes generationally -- so the interior
 *  flush there was per-step traffic with no reader: measured 12 puts/step
 *  on x alone against MPI's 0.53 MB/step ledger. */
/** OUT-OF-CORE MD: publish every x/v/dst write at the write site. The md
 *  path treats x and v as device-canonical -- no flushes, the frames ARE the
 *  data -- which is only sound while nothing is ever evicted. Under a
 *  --slots region cap an evicted page's next fault reads the STORE, and an
 *  unpublished kick means the store still holds older positions: the
 *  trajectory partly FREEZES, which the drift gate cannot tell from perfect
 *  conservation (KE 8270.5 -> 8270.4 over 20 steps while the resident run
 *  melted to 4945). Host sets this whenever the cache is capped below the
 *  working set; resident runs keep the flush-free fast path. */
/** MD_RESORT_DEBUG: slots the gather actually wrote, plus one sample. */
/** Publish-kernel latency split, in device cycles: the flush await vs the
 *  peer-halo fetch await. The host round counter cannot tell them apart. */

/** HOLD-SET ADMISSION for the force chunks. A chunk pins its whole x
 *  stencil before computing, chunks drift apart (no inter-chunk barrier), so
 *  under an out-of-core region pool the union of pinned pages can reach the
 *  pool size with every block still waiting for MORE pages -- a circular
 *  partial-hold wait no retry can break (measured: 20 of 20 regions pinned,
 *  0 on the free lists, claim spinning 4 s). The cure is the one the old
 *  EnterHoldSet API implemented before the shared-cache rebuild deleted it:
 *  reserve the ENTIRE hold set up front, while holding nothing. Reservations
 *  never exceed the pool minus slack, so every admitted chunk can finish its
 *  pins and progress is guaranteed. Slack covers the pinners that do not
 *  reserve: the async publish's fetch-pins and a peer's generational refetch.
 *  Costs nothing when the pool dwarfs the demand -- the CAS succeeds first
 *  try -- so the gate is always on. */

/** One admission attempt. Thread 0 owns the CAS; every other thread reports
 *  "not waiting" and the yield macro's vote broadcasts thread 0's verdict. */
__device__ bool AdmitWaits(u32 need, u32 pool, u32 slack) {
  if (threadIdx.x != 0) return false;
  // A demand near or above the pool degrades to THE WHOLE POOL, not to
  // `need`: degrading to need requires cur==0 to admit, and the persistent
  // halo reservation keeps cur at 6 forever -- a small pool then never
  // admitted anything. Whole-pool is safe because every force-phase pinner
  // reserves its ENTIRE hold set first: sum(reservations) <= pool means
  // every admitted party can hold all its pages at once and finish.
  const u32 budget = (pool > need + slack) ? (pool - slack) : pool;
  const u32 cur = *(volatile u32 *)&MdG().adm_used;
  if (cur + need <= budget &&
      atomicCAS(&MdG().adm_used, cur, cur + need) == cur) {
    return false;                    // admitted -- the reservation is taken
  }
  return true;
}

/** A PARK, NEVER A SPIN. The first cut of this gate busy-waited with
 *  __nanosleep inside the kernel -- but under the yield driver the current
 *  reservation holders are PARKED coroutines that only resume (and release)
 *  after the kernel exits its round. A spin keeps the round alive forever:
 *  the whole grid wedged in the very first neighbour build. Waiting must go
 *  through the same yield the holders use. Tag 0 means "always resume", so
 *  every round retries the CAS until it lands. */
/** Chunk-tier slack: room the consumer chunks must always leave for the
 *  PRODUCER. The publish's two x-pinning blocks (boundary flush 6, halo warm
 *  6) admit with slack 0 -- against the whole pool -- because every node's
 *  progress transitively depends on its peer's flush landing. Making the
 *  publish queue behind the chunks deadlocked BOTH nodes: chunks parked on
 *  the peer's generation held 30 reservations, the publish that would
 *  satisfy the peer sat at the gate behind them (admit stall need=6
 *  used=30), and the peer's generational get errored out (DEVICE FATAL 7)
 *  waiting for a flush that could never land.
 *
 *  8, NOT 14: slack covers only demand that never RESERVES -- the boundary
 *  flush (6) plus margin. The halo warm's 6 pages are a STANDING
 *  reservation, already inside `used`; budgeting slack for them too made
 *  pool 24 unrunnable by arithmetic alone (budget 10, standing 6, chunk
 *  need 6 -- a permanent admit stall before the first force chunk). */
__device__ constexpr u32 kAdmitSlackChunks = 8u;

__device__ gy::YCoroTask AdmitSpans(u32 need, u32 pool, u32 slack) {
  if (need == 0u) co_return;
  u32 rounds = 0;
  for (;;) {
    bool wait = false;
    if (threadIdx.x == 0) {
      wait = AdmitWaits(need, pool, slack);
      // A parked kernel exits its round, so this printf reaches the host
      // console mid-wedge -- unlike a printf before __trap, which dies with
      // the context. Purely diagnostic; the wait itself never gives up.
      if (wait && ++rounds == 100000u) {
        rounds = 0;
        MD_DEV_PRINTF(
            "[gpu_vector] admit stall: need=%u pool=%u slack=%u used=%u\n",
               need, pool, slack, *(volatile u32 *)&MdG().adm_used);
      }
    }
    __syncthreads();
    if (!__syncthreads_or(wait ? 1 : 0)) break;
    co_await ::clio::run::gpu::YCoroSuspend{0ull};
  }
  co_return;
}

__device__ void ReleaseSpans(u32 need) {
  __syncthreads();
  if (threadIdx.x == 0 && need != 0u) {
    // CLAMPED. An unmatched release must not wrap the counter: a single
    // underflow parks every later AdmitSpans forever (cur ~4e9 admits
    // nothing), and that presents as a silent whole-run wedge with every
    // cache counter clean.
    for (;;) {
      const u32 cur = *(volatile u32 *)&MdG().adm_used;
      const u32 sub = cur < need ? cur : need;
      if (sub == 0u) break;
      if (atomicCAS(&MdG().adm_used, cur, cur - sub) == cur) break;
    }
  }
}

/** SHARING PROBE: bit b set means block b held that page at least once.
 *  16 blocks fit a u32; the popcount per page is the sharing degree. */
__device__ void MarkPages(u32 *mask, u64 p0, u64 p1, u32 block) {
  if (threadIdx.x != 0 || block >= 32) return;
  for (u64 p = p0; p <= p1 && p < 8192; ++p) {
    atomicOr(&mask[p], 1u << block);
  }
}

/** Sink for the ballistic path's optional THIRD vector read, so the load
 *  cannot be optimised away while the physics stays untouched. */
/** READ-ONLY PROBE (--readprobe). Every experiment so far assumed writes
 *  were being lost; this asks the other half of the question -- does the
 *  FAULT PATH ever deliver wrong bytes? The vector is seeded with a value
 *  that is unique per element and exact in float, then only READ.
 *  [0] mismatches [1] first bad page [2] first bad index. */
/** First bad sample: {claimed, page, elem, observed-value-as-u32-bits}. */
/** Set from MD_PROBE_LDCG: read the probe's elements bypassing L1. */
/** {frame0, pages_per_block, page_bytes, elems_per_page} of the probe's table. */
/** Per LOGICAL block: page iterations completed, and the last page index
 *  reached. Says WHICH block loses work, and where it stopped. */

/** Guards a block may hold over one row of the paged list. Deliberately
 *  small: this array lives in the per-thread coroutine FRAME, so every
 *  slot costs frame footprint whether used or not, and the list page is
 *  sized to hold a whole row (one guard, two across a boundary). The host
 *  validates the bound in configuration terms before any run. */

/**
 * The block-uniform tables the list passes stage out of the coroutine frame.
 *
 * Declared as a TYPE, not carved out of `extern __shared__` with casts: the
 * arena is one struct, so its size is a compile-time fact instead of a
 * hand-summed byte count that has to be kept in step at the launch site.
 */
struct MdTables {
  const float *sp0[9];
  const float *sp1[9];
  u64 srun[9];
  u64 qoff[9];
  u32 qspan[12];
  const int *np[kMaxNlGuards];
  u64 gs[kMaxNlGuards];
  u64 gl[kMaxNlGuards];
};


/**
 * Worst-case x-span guards a force chunk holds at once: three z planes, each
 * up to two y ranges when the stencil crosses the y wrap, each range up to
 * two guards when it straddles a page boundary. Reserved up front because
 * the true count is not known until the spans are computed, and reserving
 * after the first hold is exactly the deadlock admission exists to prevent.
 */

/**
 * MINIMUM frames a table must have for these kernels to make progress.
 *
 * A cache is not a free parameter: it has to hold everything a kernel pins
 * AT ONCE, plus room to fetch the next span while those stay pinned. Below
 * that, EvictPages has nothing it may take and traps -- correctly, because
 * the request cannot be satisfied. Asking for fewer slots does not produce
 * "more paging", it produces an impossible configuration.
 *
 * x    the pair loop pins up to kSpanGuards spans, and the gather runs as
 *      (src=dx, srcx=dx, dst=dx2) so ONE table supplies both span sets;
 *      +2 for the destination row, +2 free frames for the next fetch.
 * f    two guards (a row and its straddle), +2 to fetch against.
 * nl   kMaxNlGuards, +2 to fetch against.
 */

/* GV_MD_CORO now lives in md_common.h; the driver needs it too. */


/** Pages a range touches -- the vector no longer exposes this. */
template <typename V>
CTP_GPU_FUN u32 PagesSpanned(const V &v, u64 off, u64 count) {
  if (count == 0) return 0;
  return static_cast<u32>(v.PageOf(off + count - 1) - v.PageOf(off)) + 1u;
}


/**
 * The keystone layout (eternia.md section 2): atoms live bin-major in
 * padded bins; an atom's index is bin * cap + slot; empty slots carry
 * type = -1 in x's .w lane. Pages contain WHOLE bins (cap * kStride divides
 * the page), so every kernel's working set is a set of whole pages known
 * from geometry. Stage 1 only needs the layout to exist and to survive the
 * integrator; binning DYNAMICS (resort, stencils) arrive in stage 2.
 */
__device__ gy::YCoroMain ForceCoro(gv::DeviceVector<float> x,
                                   gv::DeviceVector<float> f,
                                   u32 nb, u32 cap, float box, float cutoff,
                                   int eflag, double *acc, u32 z0, u32 z1,
                                   u32 nblocks, u32 block, u64 hgen,
                                   bool force_all) {
  MD_RED_SCRATCH(red);
  const u64 row_elems = static_cast<u64>(nb) * cap * kStride;
  // THIS NODE'S SLAB, in z-planes. The nine-row stencil still reaches one
  // plane either side, so the rows it READS run past [z0, z1) -- those are
  // the halo, and they are this node's copy of a neighbour's rows, refreshed
  // through the generational path before the pass runs.
  const u64 row_lo = static_cast<u64>(z0) * nb;
  const u64 row_hi = static_cast<u64>(z1) * nb;
  const float c2 = cutoff * cutoff;
  const float halfL = 0.5f * box;
  double pe = 0.0, w = 0.0, npairs = 0.0;

  for (u64 row = row_lo + block; row < row_hi; row += nblocks) {
    const u32 by = static_cast<u32>(row % nb);
    const u32 bz = static_cast<u32>(row / nb);

    // Hold the NINE stencil rows (this row is rows[4]); <= 2 guards each.
    gv::Held<float> hg[9][2];
    u64 rbase[9], rrun0[9];
    const float *rp0[9], *rp1[9];
    for (int q = 0; q < 9; ++q) {
      const int dz = q / 3 - 1, dy = q % 3 - 1;
      const u32 wz = (bz + nb + dz) % nb;
      const u32 wy = (by + nb + dy) % nb;
      const u64 rowbin0 = (static_cast<u64>(wz) * nb + wy) * nb;
      rbase[q] = rowbin0 * cap * kStride;
      // THE VERSION DEMAND BELONGS HERE, on the read that consumes the data.
      // A z-plane outside [z0,z1) is a neighbour's, and a resident copy of it
      // is last step's. Generation 0 means "any version", so asking for it
      // here -- which is what this line used to do -- accepts the stale copy
      // and makes the whole exchange a no-op no matter what the publisher
      // does. hgen == 0 restores that unsynchronised behaviour deliberately
      // (single node, or MD_NO_HALO).
      // MD_FORCE_GEN makes EVERY stencil row generational, so a single-node
      // run can answer "does a generational fetch refuse a resident older
      // copy?" without a second node. hgen is then a generation nobody has
      // published, which must NOT be served.
      const bool halo = force_all || (wz < z0 || wz >= z1);
      co_await x.Fetch(halo ? hgen : 0, rbase[q], row_elems);
      hg[q][0] = co_await x.HoldPage(rbase[q], row_elems);
      rrun0[q] = hg[q][0].run();
      if (rrun0[q] < row_elems) {
        hg[q][1] = co_await x.HoldPage(rbase[q] + rrun0[q],
                                       row_elems - rrun0[q]);
      }
      rp0[q] = hg[q][0].ptr();
      rp1[q] = hg[q][1] ? hg[q][1].ptr() : nullptr;
    }
    // Write-hold this row of f and zero it.
    const u64 fbase =
        (static_cast<u64>(bz) * nb + by) * static_cast<u64>(nb) * cap * kStride;
    co_await f.Fetch(0, fbase, row_elems);
    gv::Held<float> hf0 = co_await f.HoldPage(fbase, row_elems, true);
    gv::Held<float> hf1;
    const u64 frun0 = hf0.run();
    if (frun0 < row_elems) {
      hf1 = co_await f.HoldPage(fbase + frun0, row_elems - frun0, true);
    }
    float *const fp0 = hf0.ptr();
    float *const fp1 = hf1 ? hf1.ptr() : nullptr;
    for (u64 e = threadIdx.x; e < row_elems; e += blockDim.x) {
      (e < frun0 ? fp0[e] : fp1[e - frun0]) = 0.0f;
    }
    __syncthreads();

    // One thread per i-slot of the row.
    const u64 islots = static_cast<u64>(nb) * cap;
    for (u64 s = threadIdx.x; s < islots; s += blockDim.x) {
      const u64 ei = s * kStride;
      const float *const ip =
          (ei < rrun0[4]) ? rp0[4] + ei : rp1[4] + (ei - rrun0[4]);
      if (ip[3] < 0.0f) continue;   // padded slot
      const float xi = ip[0], yi = ip[1], zi = ip[2];
      const u32 bx = static_cast<u32>(s / cap);
      float fx = 0.0f, fy = 0.0f, fz = 0.0f;
      for (int q = 0; q < 9; ++q) {
        for (int dxx = -1; dxx <= 1; ++dxx) {
          const u32 jbx = (bx + nb + dxx) % nb;
          const u64 jb = static_cast<u64>(jbx) * cap * kStride;
          for (u32 sj = 0; sj < cap; ++sj) {
            const u64 ej = jb + static_cast<u64>(sj) * kStride;
            const float *const jp =
                (ej < rrun0[q]) ? rp0[q] + ej : rp1[q] + (ej - rrun0[q]);
            const float tj = jp[3];
            if (tj < 0.0f) continue;
            if (q == 4 && jbx == bx && sj == s % cap) continue;   // self
            float ddx = xi - jp[0];
            float ddy = yi - jp[1];
            float ddz = zi - jp[2];
            // Minimum image: bins wrap, so a raw delta can be ~box.
            if (ddx > halfL) ddx -= box; else if (ddx < -halfL) ddx += box;
            if (ddy > halfL) ddy -= box; else if (ddy < -halfL) ddy += box;
            if (ddz > halfL) ddz -= box; else if (ddz < -halfL) ddz += box;
            const float rsq = ddx * ddx + ddy * ddy + ddz * ddz;
            if (rsq >= c2) continue;
            const float r2i = 1.0f / rsq;
            const float r6i = r2i * r2i * r2i;
            const float fpair = r6i * (48.0f * r6i - 24.0f) * r2i;
            fx = __fmaf_rn(ddx, fpair, fx);
            fy = __fmaf_rn(ddy, fpair, fy);
            fz = __fmaf_rn(ddz, fpair, fz);
            if (eflag) {
              pe += 0.5 * static_cast<double>(4.0f * r6i * (r6i - 1.0f));
              w += 0.5 * static_cast<double>(r6i * (48.0f * r6i - 24.0f));
              npairs += 1.0;
            }
          }
        }
      }
      float *const op = (ei < frun0) ? fp0 + ei : fp1 + (ei - frun0);
      op[0] = fx;
      op[1] = fy;
      op[2] = fz;
    }
    __syncthreads();
    // GIVE THE PINS BACK. In shared mode Fetch is the pinner and UnpinRange
    // the releaser, one unpin per fetch over the same range; a range fetched
    // and never released is a frame no block can reclaim. A no-op under
    // private tables, where the guards above own their pins -- which is what
    // lets this one kernel run unmodified under both modes.
    for (int q = 0; q < 9; ++q) x.UnpinRange(rbase[q], row_elems);
    f.UnpinRange(fbase, row_elems);
  }  // guards die per row

  if (eflag) {
    const double vals[3] = {pe, w, npairs};
    for (int q = 0; q < 3; ++q) {
      red[threadIdx.x] = vals[q];
      __syncthreads();
      for (u32 wd = blockDim.x / 2; wd > 0; wd >>= 1) {
        if (threadIdx.x < wd) red[threadIdx.x] += red[threadIdx.x + wd];
        __syncthreads();
      }
      if (threadIdx.x == 0) atomicAdd(&acc[q], red[0]);
      __syncthreads();
    }
  }
}

/**
 * K2c (stage 3): build the Verlet list on the PAGED neigh vector.
 *
 * TRANSPOSED, PADDED layout -- entry k of row-slot s lives at
 *     row * islots * maxneigh  +  k * islots  +  s
 * so at every k the threads of a warp read CONSECUTIVE addresses: fully
 * coalesced. The CSR layout this replaces gave each thread its own
 * contiguous run, which is the worst pattern a GPU can be handed (32
 * threads, 32 unrelated cache lines) plus a divergent per-entry guard
 * walk; measured, the CSR list LOST to cell-direct outright -- 532 ms of
 * force time against 427 -- despite examining ~8x fewer candidates.
 *
 * Fixed padding also deletes the count/prefix-sum/fill round trip: counts
 * go straight to the resident d_cnt in one pass and nothing crosses PCIe
 * at a rebuild. Overflow past maxneigh sets d_err and the host refuses.
 *
 * Entries are packed ROW-RELATIVE: (q << 16) | row_slot, where q is the
 * stencil row (0..8); the force pass decodes with two shifts instead of a
 * global-index page lookup. 16 bits hold any row (nb * cap < 65536,
 * checked at startup).
 */
__device__ gy::YCoroMain BuildListCoro(gv::DeviceVector<float> x,
                                       gv::DeviceVector<int> nl, u32 nb,
                                       u32 cap, float box, float rlist,
                                       u32 maxneigh, u32 *d_cnt, int *d_err,
                                       u32 rowchunk, u32 z0, u32 z1,
                                       u32 nblocks, u32 block, u64 hgen) {
  // BLOCK-UNIFORM TABLES LIVE IN SHARED, NOT IN THE FRAME. Every thread
  // holds the same rows and the same list guards, but a thread-local array
  // indexed by a runtime value (rp0[q], np[gi]) cannot be a register: it
  // lands in the coroutine frame, which IS the yield-stack lane in GLOBAL
  // memory. That made ~6 dependent global loads of pure bookkeeping per
  // entry and is why the list pass first measured SLOWER than cell-direct
  // despite 17x fewer candidates. Filled once per row after every hold
  // (no co_await follows, so shared survives) and read by all threads.
  // SURVIVES A PARK. These were staged in plain __shared__, which the driver
  // destroys when it exits the kernel to suspend, so every one of them had to
  // be re-published by hand after the last hold that could suspend -- and a
  // suspend added anywhere after that fill would have read stale pointers.
  CLIO_SHARED_PERSIST(MdTables, s_tbl);
  const float **s_sp0 = s_tbl.sp0;
  const float **s_sp1 = s_tbl.sp1;
  u64 *s_srun = s_tbl.srun;
  u64 *s_qoff = s_tbl.qoff;
  u32 *s_qspan = s_tbl.qspan;
  const int **s_np = s_tbl.np;
  u64 *s_gs = s_tbl.gs;
  u64 *s_gl = s_tbl.gl;
  const u64 row_elems = static_cast<u64>(nb) * cap * kStride;
  const u64 nrows = static_cast<u64>(nb) * nb;
  const u64 islots = static_cast<u64>(nb) * cap;
  const u64 rowlist = islots * maxneigh;
  const float r2list = rlist * rlist;
  const float halfL = 0.5f * box;
  // Same amortization as the force pass: the stencil spans serve a CHUNK
  // of rows, so the three holds are paid once per chunk rather than once
  // per row.
  const u32 cpz = (nb + rowchunk - 1) / rowchunk;
  // Chunks are (bz, y-run) pairs, so a z-plane slab is a contiguous chunk
  // range: [z0*cpz, z1*cpz).
  const u64 ch_lo = static_cast<u64>(z0) * cpz;
  const u64 ch_hi = static_cast<u64>(z1) * cpz;
  for (u64 ch = ch_lo + block; ch < ch_hi; ch += nblocks) {
    const u32 bz = static_cast<u32>(ch / cpz);
    const u32 y0 = static_cast<u32>(ch % cpz) * rowchunk;
    if (y0 >= nb) continue;
    const u32 ylast = (y0 + rowchunk - 1 < nb) ? (y0 + rowchunk - 1)
                                               : (nb - 1);
    // Same admission as the force chunk, and for the same reason: this
    // kernel's span guards live for the whole chunk and its list guards are
    // WRITE holds, so it exhausts the very same tables. Admitting only the
    // force kernel just relocates the exhaustion to this one -- every hold
    // set in the grid has to be admitted, or none of them are protected.
    u32 span_guards = 0;
    {
      const int lo0 = static_cast<int>(y0) - 1;
      const int hi0 = static_cast<int>(ylast) + 1;
      for (int dz = -1; dz <= 1; ++dz) {
        const u32 wz = (bz + nb + dz) % nb;
        u32 rl[2], rn[2], nr = 0;
        if (lo0 < 0) {
          rl[nr] = 0; rn[nr] = static_cast<u32>(hi0) + 1u; ++nr;
          rl[nr] = nb - 1u; rn[nr] = 1u; ++nr;
        } else if (hi0 > static_cast<int>(nb) - 1) {
          rl[nr] = static_cast<u32>(lo0);
          rn[nr] = nb - static_cast<u32>(lo0); ++nr;
          rl[nr] = 0; rn[nr] = 1u; ++nr;
        } else {
          rl[nr] = static_cast<u32>(lo0);
          rn[nr] = static_cast<u32>(hi0 - lo0 + 1); ++nr;
        }
        for (u32 t = 0; t < nr; ++t) {
          const u64 rb =
              ((static_cast<u64>(wz) * nb + rl[t]) * nb) * cap * kStride;
          span_guards += PagesSpanned(x, rb, static_cast<u64>(rn[t]) * row_elems);
        }
      }
    }
    // LATCHED. The reservation taken here is the one given back at the
    // release below; recomputing it there would let a block give back a
    // reservation it never took.
    co_await AdmitSpans(span_guards, x.Regions(), kAdmitSlackChunks);
    {   // guards die at the close of this scope, before the reservations go back
    gv::Held<float> hg[6][2];
    u64 srun[6];
    const float *sp0[6], *sp1[6];
    u32 sbase[6], scnt[6], sdz[6];
    // WHAT THE FETCH PINNED, so the chunk can give it back. In shared mode
    // Fetch is the pinner and UnpinRange the releaser; a range fetched and
    // never released is a frame no other block can ever reclaim. A no-op
    // under private tables, where the guards above own their own pins.
    u64 sxrb[6], sxlen[6];
    u32 nspans = 0;
    for (int dz = -1; dz <= 1; ++dz) {
      const u32 wz = (bz + nb + dz) % nb;
      const int lo = static_cast<int>(y0) - 1;
      const int hi = static_cast<int>(ylast) + 1;
      u32 rl[2], rn[2], nr = 0;
      if (lo < 0) {
        rl[nr] = 0; rn[nr] = static_cast<u32>(hi) + 1u; ++nr;
        rl[nr] = nb - 1u; rn[nr] = 1u; ++nr;
      } else if (hi > static_cast<int>(nb) - 1) {
        rl[nr] = static_cast<u32>(lo); rn[nr] = nb - static_cast<u32>(lo); ++nr;
        rl[nr] = 0; rn[nr] = 1u; ++nr;
      } else {
        rl[nr] = static_cast<u32>(lo);
        rn[nr] = static_cast<u32>(hi - lo + 1); ++nr;
      }
      for (u32 t = 0; t < nr; ++t) {
        const u64 rb =
            ((static_cast<u64>(wz) * nb + rl[t]) * nb) * cap * kStride;
        const u64 len = static_cast<u64>(rn[t]) * row_elems;
        // Same rule as ForceCoro: a plane outside this node's slab belongs to
        // a neighbour, and only a generational fetch refuses last step's copy.
        co_await x.Fetch((wz < z0 || wz >= z1) ? hgen : 0, rb, len);
        hg[nspans][0] = co_await x.HoldPage(rb, len);
        srun[nspans] = hg[nspans][0].run();
        if (srun[nspans] < len) {
          hg[nspans][1] =
              co_await x.HoldPage(rb + srun[nspans], len - srun[nspans]);
        }
        MarkPages(MdG().xmask, x.PageOf(rb), x.PageOf(rb + len - 1), block);
        sxrb[nspans] = rb;
        sxlen[nspans] = len;
        sp0[nspans] = hg[nspans][0].ptr();
        sp1[nspans] = hg[nspans][1] ? hg[nspans][1].ptr() : nullptr;
        sbase[nspans] = rl[t];
        scnt[nspans] = rn[t];
        sdz[nspans] = static_cast<u32>(dz + 1);
        ++nspans;
      }
    }
    for (u32 by = y0; by <= ylast; ++by) {
    const u64 row = static_cast<u64>(bz) * nb + by;
    // Write-hold this row's whole list region (host-checked to fit the
    // guard array): the working-set lower bound for the neigh vector.
    gv::Held<int> hn[kMaxNlGuards];
    int *np[kMaxNlGuards];
    u64 gstart[kMaxNlGuards], glen[kMaxNlGuards];
    u32 nguards = 0;
    {
      const u64 nb0 = row * rowlist;
      u64 off = 0;
      while (off < rowlist && nguards < kMaxNlGuards) {
        co_await nl.Fetch(0, nl.PageLo(nb0 + off), nl.PageSpan(nb0 + off, 1));
        hn[nguards] =
            co_await nl.HoldPage(nb0 + off, rowlist - off, /*write=*/true);
        MarkPages(MdG().nlmask, nl.PageOf(nb0 + off),
                  nl.PageOf(nb0 + off + hn[nguards].run() - 1), block);
        np[nguards] = hn[nguards].ptr();
        gstart[nguards] = off;
        glen[nguards] = hn[nguards].run();
        off += hn[nguards].run();
        ++nguards;
      }
    }
    // Publish the block-uniform tables. This used to be a RE-publication
    // that had to sit after the last hold that could suspend, because the
    // arena was plain __shared__ and the driver destroys shared when it
    // exits the kernel to park. CLIO_SHARED_PERSIST carries it across the
    // suspension now, so this is an ordinary fill and a co_await added below
    // it is no longer a silent corruption.
    if (threadIdx.x == 0) {
      for (u32 t = 0; t < nspans; ++t) {
        s_sp0[t] = sp0[t];
        s_sp1[t] = sp1[t];
        s_srun[t] = srun[t];
      }
      for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
          const u32 wy = (by + nb + dy) % nb;
          const int q = (dz + 1) * 3 + (dy + 1);
          for (u32 t = 0; t < nspans; ++t) {
            if (sdz[t] != static_cast<u32>(dz + 1)) continue;
            if (wy >= sbase[t] && wy < sbase[t] + scnt[t]) {
              s_qspan[q] = t;
              s_qoff[q] = static_cast<u64>(wy - sbase[t]) * row_elems;
              break;
            }
          }
        }
      }
      for (u32 q = 0; q < nguards; ++q) {
        s_np[q] = np[q];
        s_gs[q] = gstart[q];
        s_gl[q] = glen[q];
      }
    }
    __syncthreads();
    const u64 slotbase = row * islots;
    const u32 sp4 = s_qspan[4];
    const u64 off4 = s_qoff[4], run4 = s_srun[sp4];
    const float *const ip0 = s_sp0[sp4];
    const float *const ip1 = s_sp1[sp4];
    for (u64 s = threadIdx.x; s < islots; s += blockDim.x) {
      const u64 e = off4 + s * kStride;
      const float *const ip = (e < run4) ? ip0 + e : ip1 + (e - run4);
      if (ip[3] < 0.0f) {
        d_cnt[slotbase + s] = 0;
        continue;
      }
      const float xi = ip[0], yi = ip[1], zi = ip[2];
      const u32 bx = static_cast<u32>(s / cap);
      u32 cnt = 0, gi = 0;
      for (int q = 0; q < 9; ++q) {
        for (int dxx = -1; dxx <= 1; ++dxx) {
          const u32 jbx = (bx + nb + dxx) % nb;
          const u64 jb = static_cast<u64>(jbx) * cap * kStride;
          const u32 spq = s_qspan[q];
          const u64 rq = s_srun[spq];
          const u64 qo = s_qoff[q];
          const float *const qp0 = s_sp0[spq];
          const float *const qp1 = s_sp1[spq];
          for (u32 sj = 0; sj < cap; ++sj) {
            const u64 ej = qo + jb + static_cast<u64>(sj) * kStride;
            const float *const jp = (ej < rq) ? qp0 + ej : qp1 + (ej - rq);
            if (jp[3] < 0.0f) continue;
            if (q == 4 && jbx == bx && sj == s % cap) continue;
            float ddx = xi - jp[0];
            float ddy = yi - jp[1];
            float ddz = zi - jp[2];
            if (ddx > halfL) ddx -= box; else if (ddx < -halfL) ddx += box;
            if (ddy > halfL) ddy -= box; else if (ddy < -halfL) ddy += box;
            if (ddz > halfL) ddz -= box; else if (ddz < -halfL) ddz += box;
            const float rsq = ddx * ddx + ddy * ddy + ddz * ddz;
            if (rsq >= r2list) continue;
            if (cnt >= maxneigh) {   // refuse, never overrun
              *d_err = 1;
              continue;
            }
            const u64 o = static_cast<u64>(cnt) * islots + s;
            while (gi + 1 < nguards && o >= s_gs[gi] + s_gl[gi]) ++gi;
            // Entries are STENCIL-RELATIVE, (q, slot-within-row), not
            // span-relative: the two passes then need not agree on how
            // they group rows into held spans, which is what lets the
            // force pass amortize its holds over a CHUNK of rows while
            // the build pass holds per row.
            const_cast<int *>(s_np[gi])[o - s_gs[gi]] = static_cast<int>(
                (static_cast<u32>(q) << 16) | (jbx * cap + sj));
            ++cnt;
          }
        }
      }
      d_cnt[slotbase + s] = cnt;
    }
    __syncthreads();
    // PUBLISH THIS LIST ROW AND LET THE FRAME GO.
    //
    // The row was write-held, and a dirty page is unevictable -- the vector
    // will not write it back on the caller's behalf. Building every row this
    // block owns without flushing therefore fills the table with dirty frames
    // that EvictPages cannot reclaim, and it traps. That, not any working
    // set, is what set the list cache floor: a block builds ~53 rows, so it
    // needed ~64 frames purely to keep every dirty row. The pass only ever
    // reads one row at a time, so with the flush the cache can be tiny.
    co_await nl.BeginFlush(0, row * rowlist, rowlist);
    // Safe after BeginFlush and not before the writes: a frame with a flush
    // in flight is not an eviction candidate, so the row's bytes cannot be
    // dropped between the submit and the wait.
    //
    // RECOMPUTED FROM gstart, NOT STORED. Two more u64 arrays here overflow
    // the coroutine frame (2144 > 2048) -- and they were redundant: gstart
    // already records the offset each fetch was issued at, and nothing
    // between the fetch and here suspends, so the arithmetic costs
    // registers rather than frame.
    for (u32 gq = 0; gq < nguards; ++gq) {
      const u64 fo = row * rowlist + gstart[gq];
      nl.UnpinRange(nl.PageLo(fo), nl.PageSpan(fo, 1));
    }
    }   // per-row loop
    // Inside the guard scope: sxrb/nspans live here, and the pins must go
    // back before the next chunk fetches its own spans.
    for (u32 sq = 0; sq < nspans; ++sq) x.UnpinRange(sxrb[sq], sxlen[sq]);
    }
    ReleaseSpans(span_guards);
    co_await nl.EndFlush();
  }     // per-chunk loop
}

/**
 * K3-list (stage 3): the force pass streaming the Verlet list. Same
 * nine-row x holds and f write-hold as the cell-direct pass; the candidate
 * scan is replaced by this atom's padded column of entries, decoded
 * (q, row_slot) with two shifts. Entries cover cutoff + skin; the cutoff
 * test here skips the skin shell, which is what keeps the list valid
 * between rebuilds.
 */
__device__ gy::YCoroMain ListForceCoro(gv::DeviceVector<float> x,
                                       gv::DeviceVector<float> f,
                                       gv::DeviceVector<int> nl, u32 nb,
                                       u32 cap, float box, float cutoff,
                                       u32 maxneigh, const u32 *d_cnt,
                                       int eflag, double *acc, int nocompute,
                                       u32 rowchunk, u32 z0, u32 z1,
                                       u32 nblocks, u32 block, u64 hgen,
                                       bool force_all, u32 band) {
  MD_RED_SCRATCH(red);
  // BLOCK-UNIFORM TABLES LIVE IN SHARED, NOT IN THE FRAME. Every thread
  // holds the same rows and the same list guards, but a thread-local array
  // indexed by a runtime value (rp0[q], np[gi]) cannot be a register: it
  // lands in the coroutine frame, which IS the yield-stack lane in GLOBAL
  // memory. That made ~6 dependent global loads of pure bookkeeping per
  // entry and is why the list pass first measured SLOWER than cell-direct
  // despite 17x fewer candidates. Filled once per row after every hold
  // (no co_await follows, so shared survives) and read by all threads.
  // SURVIVES A PARK. These were staged in plain __shared__, which the driver
  // destroys when it exits the kernel to suspend, so every one of them had to
  // be re-published by hand after the last hold that could suspend -- and a
  // suspend added anywhere after that fill would have read stale pointers.
  CLIO_SHARED_PERSIST(MdTables, s_tbl);
  const float **s_sp0 = s_tbl.sp0;
  const float **s_sp1 = s_tbl.sp1;
  u64 *s_srun = s_tbl.srun;
  u64 *s_qoff = s_tbl.qoff;
  u32 *s_qspan = s_tbl.qspan;
  const int **s_np = s_tbl.np;
  u64 *s_gs = s_tbl.gs;
  u64 *s_gl = s_tbl.gl;
  const u64 row_elems = static_cast<u64>(nb) * cap * kStride;
  const u64 nrows = static_cast<u64>(nb) * nb;
  const u64 islots = static_cast<u64>(nb) * cap;
  const u64 rowlist = islots * maxneigh;
  const float c2 = cutoff * cutoff;
  const float halfL = 0.5f * box;
  double pe = 0.0, w = 0.0, npairs = 0.0;
  // AMORTIZE THE STENCIL HOLDS OVER A CHUNK OF ROWS. Holds are what this
  // pass costs (with the pair loop deleted it ran 245 ms against 252 with
  // it), and a single row needed 3 span holds to serve 1 row of work. A
  // chunk of R rows in one z-plane needs the SAME 3 spans -- rows
  // [y0-1, ylast+1] -- so the cost per row falls by R. Rows are grouped
  // within a plane, never across one, because that is the axis along which
  // the index space is contiguous.
  const u32 cpz = (nb + rowchunk - 1) / rowchunk;
  const u64 nchunks = static_cast<u64>(nb) * cpz;
  const u64 ch_lo = static_cast<u64>(z0) * cpz;
  const u64 ch_hi = static_cast<u64>(z1) * cpz;
  for (u64 ch = ch_lo + block; ch < ch_hi; ch += nblocks) {
    const long long _r0 = clock64();
    const u32 bz = static_cast<u32>(ch / cpz);
    // BAND SPLIT for comm/compute overlap: interior chunks (band 1) touch no
    // halo plane -- their stencil stays inside [z0, z1) -- so they can run
    // WHILE the exchange is still in flight. Boundary chunks (band 2) are
    // the two z-planes whose stencil reaches the halo; they run after the
    // exchange lands. band 0 = everything (single node, gates).
    const bool on_boundary = (bz == z0 || bz + 1u == z1);
    if (band == 1u && on_boundary) continue;
    if (band == 2u && !on_boundary) continue;
    const u32 y0 = static_cast<u32>(ch % cpz) * rowchunk;
    if (y0 >= nb) continue;
    const u32 ylast = (y0 + rowchunk - 1 < nb) ? (y0 + rowchunk - 1)
                                               : (nb - 1);
    // The stencil rows for the whole chunk: [y0-1, ylast+1] mod nb at
    // each of the three z planes, as one contiguous run (two when the
    // range crosses the y wrap).
    //
    // ADMITTED BEFORE THE FIRST HOLD. These span guards stay alive for the
    // whole chunk, so without admission enough blocks pin the entire table
    // between them and no claim can succeed -- measured as 48 of 48 slots
    // pinned at a failing claim, and as a 32-slot cache that wedges at 64
    // blocks but runs at 32. Waiting here costs nothing held; waiting after
    // the first hold is what deadlocks.
    // How many spans this chunk will actually hold. The y range
    // [y0-1, ylast+1] splits in two only when it crosses the wrap, and the
    // split depends solely on (y0, ylast, nb) -- the same for all three z
    // planes. So the count is pure geometry, known BEFORE any hold, and
    // reserving it beats reserving the worst case: at 48 slots the blanket
    // 12 admitted only 4 of 64 blocks and cost 124 ms/step against 43.
    // EXACTLY how many x guards this chunk will hold. The span geometry is
    // pure arithmetic, and PagesSpanned turns each span into the number of
    // guards holding it takes, so the reservation is the true hold set
    // rather than a bound. The bound (3 planes x 2 ranges x 2 guards = 12)
    // over-reserved badly enough to admit only a few of 64 blocks, which is
    // what made admission cost 2.5-3x and kept it opt-in.
    u32 span_guards = 0;
    {
      const int lo0 = static_cast<int>(y0) - 1;
      const int hi0 = static_cast<int>(ylast) + 1;
      for (int dz = -1; dz <= 1; ++dz) {
        const u32 wz = (bz + nb + dz) % nb;
        u32 rl[2], rn[2], nr = 0;
        if (lo0 < 0) {
          rl[nr] = 0; rn[nr] = static_cast<u32>(hi0) + 1u; ++nr;
          rl[nr] = nb - 1u; rn[nr] = 1u; ++nr;
        } else if (hi0 > static_cast<int>(nb) - 1) {
          rl[nr] = static_cast<u32>(lo0);
          rn[nr] = nb - static_cast<u32>(lo0); ++nr;
          rl[nr] = 0; rn[nr] = 1u; ++nr;
        } else {
          rl[nr] = static_cast<u32>(lo0);
          rn[nr] = static_cast<u32>(hi0 - lo0 + 1); ++nr;
        }
        for (u32 t = 0; t < nr; ++t) {
          const u64 rb =
              ((static_cast<u64>(wz) * nb + rl[t]) * nb) * cap * kStride;
          span_guards += PagesSpanned(x, rb, static_cast<u64>(rn[t]) * row_elems);
        }
      }
    }
    // ALL THREE RESERVATIONS BEFORE ANY HOLD. Reserving x, taking the x
    // holds, and only then waiting for f/nl admission is the very
    // hold-while-waiting pattern admission exists to remove -- just moved
    // across vectors: blocks sit on x slots waiting for f, so x is
    // exhausted by blocks that are themselves blocked. Acquiring every
    // reservation up front, while holding nothing, is what makes the
    // guarantee hold across the whole working set rather than per vector.
    co_await AdmitSpans(span_guards, x.Regions(), kAdmitSlackChunks);
    {   // guards die at the close of this scope, before the release below
    gv::Held<float> hg[6][2];
    u64 srun[6];
    const float *sp0[6], *sp1[6];
    u32 sbase[6], scnt[6], sdz[6];
    // WHAT THE FETCH PINNED, so the chunk can give it back. In shared mode
    // Fetch is the pinner and UnpinRange the releaser; a range fetched and
    // never released is a frame no other block can ever reclaim. A no-op
    // under private tables, where the guards above own their own pins.
    u64 sxrb[6], sxlen[6];
    u32 nspans = 0;
    for (int dz = -1; dz <= 1; ++dz) {
      const u32 wz = (bz + nb + dz) % nb;
      const int lo = static_cast<int>(y0) - 1;
      const int hi = static_cast<int>(ylast) + 1;
      u32 rl[2], rn[2], nr = 0;
      if (lo < 0) {
        rl[nr] = 0; rn[nr] = static_cast<u32>(hi) + 1u; ++nr;
        rl[nr] = nb - 1u; rn[nr] = 1u; ++nr;
      } else if (hi > static_cast<int>(nb) - 1) {
        rl[nr] = static_cast<u32>(lo); rn[nr] = nb - static_cast<u32>(lo); ++nr;
        rl[nr] = 0; rn[nr] = 1u; ++nr;
      } else {
        rl[nr] = static_cast<u32>(lo);
        rn[nr] = static_cast<u32>(hi - lo + 1); ++nr;
      }
      for (u32 t = 0; t < nr; ++t) {
        const u64 rb =
            ((static_cast<u64>(wz) * nb + rl[t]) * nb) * cap * kStride;
        const u64 len = static_cast<u64>(rn[t]) * row_elems;
        // Same rule as ForceCoro: a plane outside this node's slab belongs to
        // a neighbour, and only a generational fetch refuses last step's copy.
        co_await x.Fetch((force_all || wz < z0 || wz >= z1) ? hgen : 0,
                         rb, len);
        hg[nspans][0] = co_await x.HoldPage(rb, len);
        srun[nspans] = hg[nspans][0].run();
        if (srun[nspans] < len) {
          hg[nspans][1] =
              co_await x.HoldPage(rb + srun[nspans], len - srun[nspans]);
        }
        MarkPages(MdG().xmask, x.PageOf(rb), x.PageOf(rb + len - 1), block);
        sxrb[nspans] = rb;
        sxlen[nspans] = len;
        sp0[nspans] = hg[nspans][0].ptr();
        sp1[nspans] = hg[nspans][1] ? hg[nspans][1].ptr() : nullptr;
        sbase[nspans] = rl[t];
        scnt[nspans] = rn[t];
        sdz[nspans] = static_cast<u32>(dz + 1);
        ++nspans;
      }
    }
    if (threadIdx.x == 0) atomicAdd(&MdG().md_cyc[0], (unsigned long long)(clock64() - _r0));
    for (u32 by = y0; by <= ylast; ++by) {
    const u64 row = static_cast<u64>(bz) * nb + by;
    // Which span holds each stencil row, and at what offset. Block-uniform
    // and only 9 entries, so thread 0 resolves it once per row.
    const long long _f0 = clock64();
    const u64 fbase = row * row_elems;
    co_await f.Fetch(0, fbase, row_elems);
    // ADMISSION, and in a FIXED ORDER: x at the chunk, then f, then nl.
    // Every paged vector needs it, not just x -- f and nl have their own
    // tables and can exhaust them exactly the same way. Tightening only x's
    // reservation re-wedged a 32-slot run precisely because the blanket
    // reservation had been throttling f/nl concurrency by accident. A fixed
    // acquisition order is what keeps the nesting (x held while waiting for
    // f) from forming a cycle.
    gv::Held<float> hf0 = co_await f.HoldPage(fbase, row_elems, true);
    gv::Held<float> hf1;
    const u64 frun0 = hf0.run();
    if (frun0 < row_elems) {
      hf1 = co_await f.HoldPage(fbase + frun0, row_elems - frun0, true);
    }
    float *const fp0 = hf0.ptr();
    float *const fp1 = hf1 ? hf1.ptr() : nullptr;
    for (u64 e = threadIdx.x; e < row_elems; e += blockDim.x) {
      (e < frun0 ? fp0[e] : fp1[e - frun0]) = 0.0f;
    }
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(&MdG().md_cyc[1], (unsigned long long)(clock64() - _f0));
    const long long _l0 = clock64();
    // Read-hold this row's whole list region.
    gv::Held<int> hn[kMaxNlGuards];
    const int *np[kMaxNlGuards];
    u64 gstart[kMaxNlGuards], glen[kMaxNlGuards];
    u32 nguards = 0;
    {
      const u64 nb0 = row * rowlist;
      u64 off = 0;
      while (off < rowlist && nguards < kMaxNlGuards) {
        co_await nl.Fetch(0, nl.PageLo(nb0 + off), nl.PageSpan(nb0 + off, 1));
        hn[nguards] = co_await nl.HoldPage(nb0 + off, rowlist - off);
        MarkPages(MdG().nlmask, nl.PageOf(nb0 + off),
                  nl.PageOf(nb0 + off + hn[nguards].run() - 1), block);
        np[nguards] = hn[nguards].ptr();
        gstart[nguards] = off;
        glen[nguards] = hn[nguards].run();
        off += hn[nguards].run();
        ++nguards;
      }
    }
    // Publish the block-uniform tables. This used to be a RE-publication
    // that had to sit after the last hold that could suspend, because the
    // arena was plain __shared__ and the driver destroys shared when it
    // exits the kernel to park. CLIO_SHARED_PERSIST carries it across the
    // suspension now, so this is an ordinary fill and a co_await added below
    // it is no longer a silent corruption.
    if (threadIdx.x == 0) {
      for (u32 t = 0; t < nspans; ++t) {
        s_sp0[t] = sp0[t];
        s_sp1[t] = sp1[t];
        s_srun[t] = srun[t];
      }
      for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
          const u32 wy = (by + nb + dy) % nb;
          const int q = (dz + 1) * 3 + (dy + 1);
          for (u32 t = 0; t < nspans; ++t) {
            if (sdz[t] != static_cast<u32>(dz + 1)) continue;
            if (wy >= sbase[t] && wy < sbase[t] + scnt[t]) {
              s_qspan[q] = t;
              s_qoff[q] = static_cast<u64>(wy - sbase[t]) * row_elems;
              break;
            }
          }
        }
      }
      for (u32 q = 0; q < nguards; ++q) {
        s_np[q] = np[q];
        s_gs[q] = gstart[q];
        s_gl[q] = glen[q];
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(&MdG().md_cyc[2], (unsigned long long)(clock64() - _l0));
    const long long _p0 = clock64();
    const u64 slotbase = row * islots;
    const u32 sp4 = s_qspan[4];
    const u64 off4 = s_qoff[4], run4 = s_srun[sp4];
    const float *const ip0 = s_sp0[sp4];
    const float *const ip1 = s_sp1[sp4];
    for (u64 s = threadIdx.x; s < islots; s += blockDim.x) {
      const u64 e = off4 + s * kStride;
      const float *const ip = (e < run4) ? ip0 + e : ip1 + (e - run4);
      if (ip[3] < 0.0f) continue;
      const float xi = ip[0], yi = ip[1], zi = ip[2];
      const u32 cnt = nocompute ? 0u : d_cnt[slotbase + s];
      float fx = 0.0f, fy = 0.0f, fz = 0.0f;
      u32 gi = 0;
      for (u32 k = 0; k < cnt; ++k) {
        const u64 o = static_cast<u64>(k) * islots + s;
        while (gi + 1 < nguards && o >= s_gs[gi] + s_gl[gi]) ++gi;
        const u32 ent = static_cast<u32>(s_np[gi][o - s_gs[gi]]);
        const u32 q = ent >> 16;
        const u32 spq = s_qspan[q];
        const u64 ej = s_qoff[q] +
                       static_cast<u64>(ent & 0xffffu) * kStride;
        const u64 rq = s_srun[spq];
        const float *const jp =
            (ej < rq) ? s_sp0[spq] + ej : s_sp1[spq] + (ej - rq);
        float ddx = xi - jp[0];
        float ddy = yi - jp[1];
        float ddz = zi - jp[2];
        if (ddx > halfL) ddx -= box; else if (ddx < -halfL) ddx += box;
        if (ddy > halfL) ddy -= box; else if (ddy < -halfL) ddy += box;
        if (ddz > halfL) ddz -= box; else if (ddz < -halfL) ddz += box;
        const float rsq = ddx * ddx + ddy * ddy + ddz * ddz;
        if (rsq >= c2) continue;
        const float r2i = 1.0f / rsq;
        const float r6i = r2i * r2i * r2i;
        const float fpair = r6i * (48.0f * r6i - 24.0f) * r2i;
        fx = __fmaf_rn(ddx, fpair, fx);
        fy = __fmaf_rn(ddy, fpair, fy);
        fz = __fmaf_rn(ddz, fpair, fz);
        if (eflag) {
          pe += 0.5 * static_cast<double>(4.0f * r6i * (r6i - 1.0f));
          w += 0.5 * static_cast<double>(r6i * (48.0f * r6i - 24.0f));
          npairs += 1.0;
        }
      }
      const u64 fe = s * kStride;
      float *const op = (fe < frun0) ? fp0 + fe : fp1 + (fe - frun0);
      op[0] = fx;
      op[1] = fy;
      op[2] = fz;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      atomicAdd(&MdG().md_cyc[3], (unsigned long long)(clock64() - _p0));
      // NOT the chunk total here. _r0 is stamped once per CHUNK, so
      // accumulating (now - _r0) once per ROW sums rowchunk overlapping
      // spans and inflates the total by ~rowchunk -- which made the named
      // phases look like a quarter of the kernel and invented a 74%
      // "unattributed" gap that does not exist. The chunk total is taken
      // once, after the row loop.
      atomicAdd(&MdG().md_cyc[5], 1ull);
    }
    // The row's own pins go back here; the chunk's x spans go back below.
    // f is written and never flushed by design -- the next kernel reads it
    // out of the cache -- so this releases the pin, not the page.
    // Recomputed from gstart; see the note in the build pass.
    for (u32 gq = 0; gq < nguards; ++gq) {
      const u64 fo = row * rowlist + gstart[gq];
      nl.UnpinRange(nl.PageLo(fo), nl.PageSpan(fo, 1));
    }
    f.UnpinRange(fbase, row_elems);
    }   // per-row loop
    // The chunk total, taken ONCE. See the note at MdG().md_cyc[5].
    if (threadIdx.x == 0) {
      atomicAdd(&MdG().md_cyc[4], (unsigned long long)(clock64() - _r0));
    }
    // Every span guard above is dead here (the chunk body scope ends with
    // this brace), so the reservation is given back exactly once per
    // the Admit. The only chunk-level `continue` is above the Admit.
    for (u32 sq = 0; sq < nspans; ++sq) x.UnpinRange(sxrb[sq], sxlen[sq]);
    }
    ReleaseSpans(span_guards);
  }     // per-chunk loop
  if (eflag) {
    const double vals[3] = {pe, w, npairs};
    for (int q = 0; q < 3; ++q) {
      red[threadIdx.x] = vals[q];
      __syncthreads();
      for (u32 wd = blockDim.x / 2; wd > 0; wd >>= 1) {
        if (threadIdx.x < wd) red[threadIdx.x] += red[threadIdx.x + wd];
        __syncthreads();
      }
      if (threadIdx.x == 0) atomicAdd(&acc[q], red[0]);
      __syncthreads();
    }
  }
}

/**
 * K2a: destination pass of the resort (stage 2). Positions have drifted
 * (and possibly left the box); wrap each atom into [0, box), compute its
 * new bin, claim a slot with an atomic per-bin counter, and record the
 * destination in d_dest (resident index class). The wrapped coordinates
 * are stored back, exactly as LAMMPS wraps at reneighbour. An atom moves
 * at most one bin per rebuild window (the skin rule), which is what makes
 * K2b's scatter row-local. Bin overflow sets d_err; the host refuses.
 */
/**
 * K2a-i: WRAP ONLY. Fold this node's own atoms back into the box.
 *
 * Split out of the old single-pass rebin because the SLOT ASSIGNMENT below
 * has to read a neighbour's positions, and it must read them ALREADY WRAPPED
 * -- otherwise the two nodes compute different bins for the same atom. So:
 * everyone wraps their own, everyone publishes, then everyone assigns.
 * This pass writes only pages this node owns, so one-writer-per-page holds.
 */
__device__ gy::YCoroMain RebinWrapCoro(gv::DeviceVector<float> x, u32 nb,
                                       u32 cap, float box, u32 z0, u32 z1,
                                       u32 nblocks, u32 block) {
  const u64 epp = x.ElemsPerPage();
  const Slab sl = SlabOf(nb, cap, z0, z1, epp);
  for (u64 pg = sl.pg_lo + block; pg < sl.pg_hi; pg += nblocks) {
    const u64 e0 = (pg * epp > sl.lo) ? pg * epp : sl.lo;
    const u64 e1 = ((pg + 1) * epp < sl.hi) ? (pg + 1) * epp : sl.hi;
    if (e1 <= e0) continue;
    co_await x.Fetch(0, pg * epp, epp);
    auto hx = co_await x.HoldPage(pg * epp, epp, /*write=*/true);
    float *const px = hx.ptr();
    const u64 nslots = epp / kStride;
    const u64 s_lo = (e0 - pg * epp) / kStride;
    const u64 s_hi = (e1 - pg * epp) / kStride;
    for (u64 s = s_lo + threadIdx.x; s < s_hi; s += blockDim.x) {
      const u64 e = s * kStride;
      if (px[e + 3] < 0.0f) continue;
      float px0 = px[e + 0], py0 = px[e + 1], pz0 = px[e + 2];
      if (px0 < 0.0f) px0 += box; else if (px0 >= box) px0 -= box;
      if (py0 < 0.0f) py0 += box; else if (py0 >= box) py0 -= box;
      if (pz0 < 0.0f) pz0 += box; else if (pz0 >= box) pz0 -= box;
      px[e + 0] = px0; px[e + 1] = py0; px[e + 2] = pz0;
    }
    __syncthreads();
    // PUBLISH AT THE WRITE SITE -- eviction performs no I/O, so a wrapped
    // page evicted before the flush refaults to its PRE-WRAP positions and
    // the resort bins atoms a box-length away. Resident runs never noticed:
    // the frame was the only copy anyone read.
    if (MdG().md_flush) {
      co_await x.BeginFlush(0, e0, e1 - e0);
    }
    x.UnpinRange(pg * epp, epp);
  }
  if (MdG().md_flush) co_await x.EndFlush();
}

/**
 * K2a-ii: ASSIGN. Claim a destination slot for every atom that lands in a bin
 * THIS NODE OWNS -- and for no other.
 *
 * THIS IS THE MIGRATION HANDOFF. The old pass had each node assign slots for
 * its OWN atoms wherever they landed, so an atom drifting out of node0's slab
 * got a d_dest pointing into node1's rows -- which node0's gather (bounded to
 * node0's rows) skipped, and which node1 had never computed. Both dropped it,
 * and every mid-flight resort shed the atoms that crossed a boundary:
 * measured as pairs 4741632 -> 4580898 at the first one.
 *
 * Inverting the ownership fixes it. A node scans its slab PLUS one plane
 * either side -- an atom moves at most one bin per rebuild, so that is every
 * atom that can reach it -- and claims a slot only when the DESTINATION bin is
 * inside [z0, z1). bincnt is then authoritative for the bins it counts,
 * because exactly one node counts them, and the arriving migrant has an owner.
 * The halo planes are read-only here, so the write-per-page invariant the
 * generational design rests on is untouched.
 */
__device__ gy::YCoroMain RebinAssignCoro(gv::DeviceVector<float> x, u32 nb,
                                         u32 cap, float box, u32 *bincnt,
                                         u32 *d_dest, int *d_err, u32 z0,
                                         u32 z1, u32 nblocks, u32 block,
                                         u64 hgen) {
  const u64 epp = x.ElemsPerPage();
  const float fnb = static_cast<float>(nb);
  const u64 row_elems = static_cast<u64>(nb) * cap * kStride;
  const u64 plane_elems = static_cast<u64>(nb) * row_elems;
  const u64 islots_r = static_cast<u64>(nb) * cap;
  const u32 mine = z1 - z0;
  const bool whole = (mine >= nb);          // single node owns everything
  // ONE PASS PER PAGE, NEVER TWO.
  //
  // A PLANE-indexed scan only works when planes are a whole number of pages,
  // and that is enforced ONLY for --nodes > 1 (the seam rule). On one node an
  // unaligned lattice shares a boundary page between consecutive planes, so a
  // plane loop visits it twice and assigns its atoms two slots -- which is a
  // corruption this pass would introduce rather than fix. When this node owns
  // everything there are no halo planes to single out, so sweep the pages
  // directly and skip the plane decomposition entirely.
  const u64 total_elems = static_cast<u64>(nb) * plane_elems;
  const u64 npages_all = (total_elems + epp - 1) / epp;
  const u32 nscan = whole ? 1u : (mine + 2u);
  for (u32 i = 0; i < nscan; ++i) {
    const u32 wz = whole ? 0u : ((z0 + nb - 1u + i) % nb);
    const bool in_mine = whole || (wz >= z0 && wz < z1);
    const u64 p_lo = whole ? 0ull : static_cast<u64>(wz) * plane_elems;
    const u64 pg_lo = whole ? 0ull : (p_lo / epp);
    const u64 pg_hi =
        whole ? npages_all : ((p_lo + plane_elems + epp - 1) / epp);
    for (u64 pg = pg_lo + block; pg < pg_hi; pg += nblocks) {
      // A HALO PLANE IS A NEIGHBOUR'S AND MUST BE DEMANDED AT THE GENERATION
      // THE WRAP WAS PUBLISHED AS. Generation 0 would accept last step's
      // positions and bin the atom from them.
      co_await x.Fetch(in_mine ? 0 : hgen, pg * epp, epp);
      auto hx = co_await x.HoldPage(pg * epp, epp);
      const float *const px = hx.ptr();
      const u64 nslots = epp / kStride;
      const u64 slot0 = pg * nslots;
      for (u64 s = threadIdx.x; s < nslots; s += blockDim.x) {
        const u64 e = s * kStride;
        if (px[e + 3] < 0.0f) continue;          // padded slot
        const float px0 = px[e + 0], py0 = px[e + 1], pz0 = px[e + 2];
        u32 bx = static_cast<u32>(px0 * fnb / box);
        u32 by = static_cast<u32>(py0 * fnb / box);
        u32 bz = static_cast<u32>(pz0 * fnb / box);
        if (bx >= nb) bx = nb - 1;
        if (by >= nb) by = nb - 1;
        if (bz >= nb) bz = nb - 1;
        if (!whole && (bz < z0 || bz >= z1)) continue;   // not my bin
        const u64 bin = (static_cast<u64>(bz) * nb + by) * nb + bx;
        const u32 slot = atomicAdd(&bincnt[bin], 1u);
        if (slot >= cap) { *d_err = 1; continue; }
        d_dest[slot0 + s] = static_cast<u32>(bin * cap + slot);
        // THE GATHER CAN ONLY REACH ONE BIN IN Y AND Z (see K2b): an atom
        // that crossed more than one is never collected and vanishes.
        const u64 srow = (slot0 + s) / islots_r;
        const u32 sby = static_cast<u32>(srow % nb);
        const u32 sbz = static_cast<u32>(srow / nb);
        const u32 dy = MdMinU32((by + nb - sby) % nb, (sby + nb - by) % nb);
        const u32 dz = MdMinU32((bz + nb - sbz) % nb, (sbz + nb - bz) % nb);
        if (dy > 1u || dz > 1u) *d_err = 2;
      }
      __syncthreads();
      x.UnpinRange(pg * epp, epp);
    }
  }
}

/**
 * K2b: apply the permutation for ONE vector, src -> dst (the ping-pong
 * buffer). Row-parallel like the force pass: a source row's destinations
 * lie within +-1 bin, i.e. inside the same NINE-row stencil, so the block
 * write-holds those rows of dst collectively and threads scatter. dst rows
 * were pre-sentineled by SentinelCoro. All slot claims were made in K2a,
 * so no two atoms share a destination. `keep_w` carries the type lane for
 * x and writes 0 for v.
 */
/**
 * K2b as a GATHER, not a scatter (the page-boundary rule).
 *
 * A scatter has block r writing destination rows r-1..r+1 while block r+1
 * writes r..r+2: their destination spans OVERLAP, so two blocks write the
 * same page. Resident that is invisible -- one frame, disjoint slots,
 * nothing evicts -- but with page-granular writeback one block's eviction
 * discards the other's writes, which showed up out of core as an
 * intermittent CUDA fatal, a 10x energy drift, or NaN, run to run on the
 * same command.
 *
 * Inverted, each block OWNS one destination row and PULLS into it: it
 * scans the nine source rows that could possibly target this row (an atom
 * moves at most one bin per rebuild window) and copies the ones whose
 * recorded destination lands here. Every destination slot then has exactly
 * ONE writer and no page is written by two blocks. The cost is reading
 * nine source rows per destination row instead of one, which is the same
 * stencil shape the force pass already pays and is safe to share because
 * it is read-only.
 */
__device__ gy::YCoroMain GatherCoro(gv::DeviceVector<float> src,
                                    gv::DeviceVector<float> srcx,
                                    gv::DeviceVector<float> dst, u32 nb,
                                    u32 cap, const u32 *d_dest, int keep_w,
                                    u32 z0, u32 z1, u32 nblocks, u32 block,
                                    u64 hgen) {
  const u64 islots = static_cast<u64>(nb) * cap;
  const u64 row_elems = islots * kStride;
  const u64 row_lo = static_cast<u64>(z0) * nb;
  const u64 row_hi = static_cast<u64>(z1) * nb;
  for (u64 row = row_lo + block; row < row_hi; row += nblocks) {
    const u32 by = static_cast<u32>(row % nb);
    const u32 bz = static_cast<u32>(row / nb);
    // THE ONLY WRITE HOLD: this block's own destination row.
    {co_await dst.Fetch(0, row * row_elems, row_elems);
       // every guard dies at this scope's close, before the reservations
    gv::Held<float> hd0 = co_await dst.HoldPage(row * row_elems, row_elems,
                                                /*write=*/true);
    gv::Held<float> hd1;
    const u64 drun = hd0.run();
    if (drun < row_elems) {
      co_await dst.Fetch(0, dst.PageLo(row * row_elems + drun),
                         dst.PageSpan(row * row_elems + drun, 1));
      hd1 = co_await dst.HoldPage(row * row_elems + drun, row_elems - drun,
                                  /*write=*/true);
    }
    float *const dp0 = hd0.ptr();
    float *const dp1 = hd1 ? hd1.ptr() : nullptr;
    // RESET THE DESTINATION ROW TO PADDED FIRST.
    //
    // The scatter below writes only the slots that RECEIVE an atom. Every
    // other slot would keep this buffer's contents from the previous resort,
    // and w >= 0 marks a slot occupied -- so those ghosts are read back as
    // real atoms sitting on top of the live ones, and the pair force between
    // a ghost and its own current copy is what blows the energy up. The
    // step-0 resort gate cannot see this: the buffer is padded at setup, so
    // only the SECOND resort onwards inherits stale slots.
    for (u64 s = threadIdx.x; s < islots; s += blockDim.x) {
      const u64 de = s * kStride;
      float *const dp = (de < drun) ? dp0 + de : dp1 + (de - drun);
      dp[3] = -1.0f;
    }
    __syncthreads();
    // The nine candidate source rows, held as SPANS (one or two contiguous
    // runs per dz), of both vectors -- they are the same vector on the
    // position pass, where the second hold is a cache hit. Row-at-a-time
    // holds needed 36 live guards and overflowed the coroutine frame.
    gv::Held<float> hs[6][2], hx[6][2];
    u64 srun[6], xrun[6];
    const float *sp0[6], *sp1[6], *xp0[6], *xp1[6];
    u32 qspan[9];
    u64 qoff[9];
    u64 srow[9];
    u32 nspans = 0;
    for (int dz = -1; dz <= 1; ++dz) {
      const u32 wz = (bz + nb + dz) % nb;
      const int lo = static_cast<int>(by) - 1;
      const int hi = static_cast<int>(by) + 1;
      u32 rl[2], rn[2], nr = 0;
      if (lo < 0) {
        rl[nr] = 0; rn[nr] = static_cast<u32>(hi) + 1u; ++nr;
        rl[nr] = nb - 1u; rn[nr] = 1u; ++nr;
      } else if (hi > static_cast<int>(nb) - 1) {
        rl[nr] = static_cast<u32>(lo); rn[nr] = nb - static_cast<u32>(lo); ++nr;
        rl[nr] = 0; rn[nr] = 1u; ++nr;
      } else {
        rl[nr] = static_cast<u32>(lo); rn[nr] = 3u; ++nr;
      }
      for (u32 t = 0; t < nr; ++t) {
        const u64 rb =
            (static_cast<u64>(wz) * nb + rl[t]) * row_elems;
        const u64 len = static_cast<u64>(rn[t]) * row_elems;
        // WHOLE PAGES, BECAUSE THIS READER RE-READS THEM. The nine stencil
        // spans of a row overlap heavily: neighbouring spans land in the same
        // pages, and a page made valid over only one span's slice forces the
        // next span to refetch it. Asking for the pages outright makes the
        // second span a cache hit. The vector fetches exactly what it is
        // told -- so a caller that wants pages has to say pages.
        // A SOURCE PLANE OUTSIDE THIS SLAB IS A NEIGHBOUR'S, so the same
        // rule the force pass follows applies here: generation 0 accepts the
        // resident copy however old, and the gather would scatter atoms from
        // LAST step's positions.
        co_await src.Fetch((wz < z0 || wz >= z1) ? hgen : 0,
                           src.PageLo(rb), src.PageSpan(rb, len));
        hs[nspans][0] = co_await src.HoldPage(rb, len);
        srun[nspans] = hs[nspans][0].run();
        if (srun[nspans] < len) {
          hs[nspans][1] =
              co_await src.HoldPage(rb + srun[nspans], len - srun[nspans]);
        }
        sp0[nspans] = hs[nspans][0].ptr();
        sp1[nspans] = hs[nspans][1] ? hs[nspans][1].ptr() : nullptr;
        co_await srcx.Fetch((wz < z0 || wz >= z1) ? hgen : 0,
                            srcx.PageLo(rb), srcx.PageSpan(rb, len));
        hx[nspans][0] = co_await srcx.HoldPage(rb, len);
        xrun[nspans] = hx[nspans][0].run();
        if (xrun[nspans] < len) {
          co_await srcx.Fetch((wz < z0 || wz >= z1) ? hgen : 0,
                              srcx.PageLo(rb + xrun[nspans]),
                              srcx.PageSpan(rb + xrun[nspans], 1));
          hx[nspans][1] =
              co_await srcx.HoldPage(rb + xrun[nspans], len - xrun[nspans]);
        }
        xp0[nspans] = hx[nspans][0].ptr();
        xp1[nspans] = hx[nspans][1] ? hx[nspans][1].ptr() : nullptr;
        // Which logical stencil rows this run covers.
        for (int dy = -1; dy <= 1; ++dy) {
          const u32 wy = (by + nb + dy) % nb;
          if (wy < rl[t] || wy >= rl[t] + rn[t]) continue;
          const int q = (dz + 1) * 3 + (dy + 1);
          qspan[q] = nspans;
          qoff[q] = static_cast<u64>(wy - rl[t]) * row_elems;
          srow[q] = static_cast<u64>(wz) * nb + wy;
        }
        ++nspans;
      }
    }
    for (u64 idx = threadIdx.x; idx < 9 * islots; idx += blockDim.x) {
      const u32 q = static_cast<u32>(idx / islots);
      const u64 sslot = idx % islots;
      const u32 sp = qspan[q];
      const u64 e = qoff[q] + sslot * kStride;
      const float *const xs =
          (e < xrun[sp]) ? xp0[sp] + e : xp1[sp] + (e - xrun[sp]);
      if (xs[3] < 0.0f) continue;              // padded source slot
      const u32 dest = d_dest[srow[q] * islots + sslot];
      if (dest == ~0u) continue;               // overflow victim
      if (static_cast<u64>(dest) / islots != row) continue;   // not ours
      atomicAdd(&MdG().gather_wrote, 1ull);
      const u64 de = (static_cast<u64>(dest) % islots) * kStride;
      float *const dp = (de < drun) ? dp0 + de : dp1 + (de - drun);
      const float *const sv =
          (e < srun[sp]) ? sp0[sp] + e : sp1[sp] + (e - srun[sp]);
      dp[0] = sv[0];
      dp[1] = sv[1];
      dp[2] = sv[2];
      dp[3] = keep_w ? sv[3] : 0.0f;
    }
    __syncthreads();
    // PUBLISH THIS ROW. Each block owns a private page table, so a row left
    // in this block's cache is invisible to every other block. Flush exactly
    // the row -- a whole-page flush would also send this block's stale copy
    // of the ~11 OTHER rows sharing the page and clobber their owners.
    if (MdG().publish && MdG().pub_interior) {
      co_await dst.Flush(0, row * row_elems, row_elems);
    }
    // One unpin per fetch, in the same shape the fetches were issued: the
    // stencil spans of src and srcx (the same vector on the position pass,
    // so it was pinned twice and is released twice), the second srcx page
    // when a span straddles, then this row of dst.
    //
    // THE SPAN GEOMETRY IS RECOMPUTED, NOT REMEMBERED. Carrying (rb, len)
    // for six spans costs 96 bytes of coroutine frame, which is exactly what
    // overflowed it (2144 > 2048). This loop suspends nowhere, so the same
    // arithmetic lives in registers instead. It must walk dz and t in the
    // ORIGINAL ORDER, because that order is what defined the span index.
    {
      u32 sq = 0;
      for (int dz = -1; dz <= 1; ++dz) {
        const u32 wz = (bz + nb + dz) % nb;
        const int lo = static_cast<int>(by) - 1;
        const int hi = static_cast<int>(by) + 1;
        u32 rl[2], rn[2], nr = 0;
        if (lo < 0) {
          rl[nr] = 0; rn[nr] = static_cast<u32>(hi) + 1u; ++nr;
          rl[nr] = nb - 1u; rn[nr] = 1u; ++nr;
        } else if (hi > static_cast<int>(nb) - 1) {
          rl[nr] = static_cast<u32>(lo); rn[nr] = nb - static_cast<u32>(lo);
          ++nr;
          rl[nr] = 0; rn[nr] = 1u; ++nr;
        } else {
          rl[nr] = static_cast<u32>(lo); rn[nr] = 3u; ++nr;
        }
        for (u32 t = 0; t < nr && sq < nspans; ++t, ++sq) {
          const u64 rb = (static_cast<u64>(wz) * nb + rl[t]) * row_elems;
          const u64 len = static_cast<u64>(rn[t]) * row_elems;
          src.UnpinRange(src.PageLo(rb), src.PageSpan(rb, len));
          srcx.UnpinRange(srcx.PageLo(rb), srcx.PageSpan(rb, len));
          if (xrun[sq] < len) {
            srcx.UnpinRange(srcx.PageLo(rb + xrun[sq]),
                            srcx.PageSpan(rb + xrun[sq], 1));
          }
        }
      }
    }
    if (MdG().md_flush) {
      co_await dst.BeginFlush(0, row * row_elems, row_elems);
    }
    dst.UnpinRange(row * row_elems, row_elems);
    if (drun < row_elems) {
      dst.UnpinRange(dst.PageLo(row * row_elems + drun),
                     dst.PageSpan(row * row_elems + drun, 1));
    }
    }   // guards dead here
    if (false) {
    }
  }     // per-row loop
  if (MdG().md_flush) co_await dst.EndFlush();
}

/** Pre-sentinel every slot of a ping-pong destination vector. */
__device__ gy::YCoroMain SentinelCoro(gv::DeviceVector<float> dst, u32 nb,
                                      u32 cap, u32 z0, u32 z1, u32 nblocks,
                                      u32 block) {
  const u64 epp = dst.ElemsPerPage();
  const Slab sl = SlabOf(nb, cap, z0, z1, epp);
  for (u64 pg = sl.pg_lo + block; pg < sl.pg_hi; pg += nblocks) {
    const u64 e0 = (pg * epp > sl.lo) ? pg * epp : sl.lo;
    const u64 e1 = ((pg + 1) * epp < sl.hi) ? (pg + 1) * epp : sl.hi;
    if (e1 <= e0) continue;
    co_await dst.Fetch(0, pg * epp, epp);
    auto h = co_await dst.HoldPage(pg * epp, epp, /*write=*/true);
    float *const p = h.ptr();
    const u64 s_lo = (e0 - pg * epp) / kStride;
    const u64 s_hi = (e1 - pg * epp) / kStride;
    for (u64 s = s_lo + threadIdx.x; s < s_hi; s += blockDim.x) {
      p[s * kStride + 3] = -1.0f;
    }
    __syncthreads();
    if (MdG().md_flush) co_await dst.BeginFlush(0, e0, e1 - e0);
    dst.UnpinRange(pg * epp, epp);
  }
  if (MdG().md_flush) co_await dst.EndFlush();
}

/** K1/K4 for real MD: the kick reads the FORCE VECTOR (mass = 1). Same
 *  page-parallel shape as the ballistic form. */
__device__ gy::YCoroMain MDIntegrateCoro(gv::DeviceVector<float> x,
                                         gv::DeviceVector<float> v,
                                         gv::DeviceVector<float> f,
                                         float dt, int drift, u32 nb, u32 cap,
                                         u32 z0, u32 z1, u32 nblocks,
                                         u32 block) {
  const u64 epp = x.ElemsPerPage();
  const Slab sl = SlabOf(nb, cap, z0, z1, epp);
  const float half = 0.5f * dt;
  for (u64 pg = sl.pg_lo + block; pg < sl.pg_hi; pg += nblocks) {
    // CLIPPED TO THE SLAB. The seam page belongs to two nodes; this one
    // touches only its own elements of it, and flushes only those.
    const u64 e0 = (pg * epp > sl.lo) ? pg * epp : sl.lo;
    const u64 e1 = ((pg + 1) * epp < sl.hi) ? (pg + 1) * epp : sl.hi;
    if (e1 <= e0) continue;
    co_await x.Fetch(0, pg * epp, epp);
    auto hx = co_await x.HoldPage(pg * epp, epp, /*write=*/true);
    co_await v.Fetch(0, pg * epp, epp);
    auto hv = co_await v.HoldPage(pg * epp, epp, /*write=*/true);
    co_await f.Fetch(0, pg * epp, epp);
    auto hf = co_await f.HoldPage(pg * epp, epp);
    float *const px = hx.ptr();
    float *const pv = hv.ptr();
    const float *const pf = hf.ptr();
    // Slots of THIS page that lie in the slab.
    const u64 s_lo = (e0 - pg * epp) / kStride;
    const u64 s_hi = (e1 - pg * epp) / kStride;
    for (u64 s = s_lo + threadIdx.x; s < s_hi; s += blockDim.x) {
      const u64 e = s * kStride;
      if (px[e + 3] < 0.0f) continue;
      const float vx0 = __fmaf_rn(half, pf[e + 0], pv[e + 0]);
      const float vy0 = __fmaf_rn(half, pf[e + 1], pv[e + 1]);
      const float vz0 = __fmaf_rn(half, pf[e + 2], pv[e + 2]);
      pv[e + 0] = vx0;
      pv[e + 1] = vy0;
      pv[e + 2] = vz0;
      if (drift) {
        px[e + 0] = __fmaf_rn(dt, vx0, px[e + 0]);
        px[e + 1] = __fmaf_rn(dt, vy0, px[e + 1]);
        px[e + 2] = __fmaf_rn(dt, vz0, px[e + 2]);
      }
    }
    __syncthreads();
    // PUBLISH AT THE WRITE SITE (out-of-core only): eviction performs no
    // I/O, so an unflushed kick is silently undone by the next fault of
    // this page. Clipped to the slab's own elements -- the seam page
    // belongs to two nodes and each may flush only its own half.
    if (MdG().md_flush) {
      if (drift) co_await x.BeginFlush(0, e0, e1 - e0);
      co_await v.BeginFlush(0, e0, e1 - e0);
    }
    x.UnpinRange(pg * epp, epp);
    v.UnpinRange(pg * epp, epp);
    f.UnpinRange(pg * epp, epp);
  }
  if (MdG().md_flush) {
    if (drift) co_await x.EndFlush();
    co_await v.EndFlush();
  }
}

/** Flush every dirty page of the SHARED x/v tables to the backing store, so
 *  host Download (which reads the store, not the frames) sees the truth.
 *  One block does it; the others exit immediately. Validation-path only --
 *  steady-state MD never flushes x/v (they are device-canonical). */

/** Seeded value of element i: unique and exactly representable (the whole
 *  vector is well under 2^24 elements). */
CTP_INLINE_CROSS_FUN float ProbeVal(u64 i) { return static_cast<float>(i); }

/** ld.global.cg / ld.global.cv, SPELLED OUT RATHER THAN NAMED.
 *
 *  nvcc gets these as __ldcg/__ldcv from sm_32_intrinsics.h. Clang's CUDA
 *  headers DELIBERATELY do not include that header (see the comment in
 *  __clang_cuda_runtime_wrapper.h: it defines __shfl and __ldg with inline
 *  asm, and clang wants its own intrinsics for those), so under clang the
 *  names do not exist for scalar types -- only the __half / __nv_bfloat16
 *  overloads cuda_fp16.hpp brings in, which is what the overload resolution
 *  error listed. Since clang IS the toolchain here (gpu_vector is device
 *  coroutines end to end, CLIO_GPU_CLANG), calling __ldcg is calling nothing.
 *
 *  The INSTRUCTIONS are available on every arch this builds for; only the
 *  declarations are missing. Emit them. The non-NVPTX arm is the host pass
 *  (and a SYCL host compile), where the probe's cache semantics are moot. */
__device__ inline float ProbeLoadCG(const float *p) {
#if defined(__CUDA_ARCH__) || defined(__NVPTX__)
  float v;
  asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(v) : "l"(p) : "memory");
  return v;
#else
  return *p;
#endif
}

/** ld.global.cv: bypasses L1 AND treats L2 as volatile -- the "is the data
 *  actually there" read below. Same missing-declaration story as CG. */
__device__ inline float ProbeLoadCV(const float *p) {
#if defined(__CUDA_ARCH__) || defined(__NVPTX__)
  float v;
  asm volatile("ld.global.cv.f32 %0, [%1];" : "=f"(v) : "l"(p) : "memory");
  return v;
#else
  return *p;
#endif
}

__device__ gy::YCoroMain ReadProbeCoro(gv::DeviceVector<float> x, u64 passes,
                                       u32 nblocks, u32 block) {
  const u64 epp = x.ElemsPerPage();
  const u64 npages = (x.size() + epp - 1) / epp;
  // Publish this block's table geometry once, so a bad guard pointer can be
  // converted to an actual slot index instead of being guessed at from raw
  // addresses (which are only comparable within one process).
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    MdG().read_geom[1] = x.SetSize();
    MdG().read_geom[2] = x.PageBytes();
    MdG().read_geom[3] = epp;
  }
  for (u64 it = 0; it < passes; ++it) {
    for (u64 pg = block; pg < npages; pg += nblocks) {
      co_await x.Fetch(0, pg * epp, epp);
      auto h = co_await x.HoldPage(pg * epp, epp);  // READ hold only
      // Is the guard born valid? Its slot must hold the page we asked for.
      // This separates "the hold returned the wrong frame" from "the frame
      // was right and the slot was recycled out from under the pin".

      const float *const p = h.ptr();
      for (u64 i = threadIdx.x; i < epp; i += blockDim.x) {
        // MD_PROBE_LDCG=1: read past the (non-coherent) per-SM L1. If the
        // mismatches vanish only under this, the bytes were always in the
        // frame and the kernel was reading a stale cached line -- host DMA
        // that lands mid-kernel is not visible to an ordinary ld.global.
        const float seen = MdG().probe_ldcg ? ProbeLoadCG(&p[i]) : p[i];
        if (seen != ProbeVal(pg * epp + i)) {
          atomicAdd(&MdG().read_bad[0], 1ull);
          // IS THE DATA ACTUALLY THERE? Re-read the SAME element fully
          // uncached (ld.global.cv bypasses L1 AND treats L2 as volatile),
          // with no delay -- this is a pure cache question, not a timing
          // one. A match here means the bytes were in the frame all along
          // and the kernel was reading a stale cached line.
          if (ProbeLoadCV(&p[i]) == ProbeVal(pg * epp + i)) {
            atomicAdd(&MdG().read_bad[1], 1ull);
          }
          // DOES IT HEAL? Wait, re-read the SAME address through the SAME
          // guard, and see whether the value arrives late. If it does, the
          // page was published as resident before its transfer landed; if
          // it stays wrong, the get truly never wrote the frame.
          // The white-box duplicate/stale-frame diagnostics that lived here
          // are gone with the accessors they used. A page cannot occupy two
          // frames of one table now: the fault publishes into the frame it
          // claimed, and the guard pins it.
        }
      }
      __syncthreads();
      x.UnpinRange(pg * epp, epp);
    }
  }
}

__device__ gy::YCoroMain IntegrateCoro(gv::DeviceVector<float> x,
                                       gv::DeviceVector<float> v,
                                       gv::DeviceVector<float> third,
                                       int use_third, float dt, float gx,
                                       float gy_, float gz, int drift,
                                       u32 nblocks, u32 block) {
  const u64 epp = x.ElemsPerPage();
  const u64 npages = (x.size() + epp - 1) / epp;
  const float half = 0.5f * dt;
  for (u64 pg = block; pg < npages; pg += nblocks) {
    co_await x.Fetch(0, pg * epp, epp);
    auto hx = co_await x.HoldPage(pg * epp, epp, /*write=*/true);
    co_await v.Fetch(0, pg * epp, epp);
    auto hv = co_await v.HoldPage(pg * epp, epp, /*write=*/true);
    // THE TEST: a THIRD vector held read-only alongside the other two, the
    // pattern the real integrator has (x write, v write, f read) and the
    // ballistic gate never had. The value is only sunk, so the physics --
    // and therefore the bitwise comparison against the host replica --
    // is untouched.
    gv::Held<float> ht;
    if (use_third) {
      co_await third.Fetch(0, pg * epp, epp);
      ht = co_await third.HoldPage(pg * epp, epp);
    }
    float *const px = hx.ptr();
    float *const pv = hv.ptr();
    if (use_third && threadIdx.x == 0) {
      atomicAdd(&MdG().third_sink,
                static_cast<unsigned long long>(__float_as_uint(ht.ptr()[0])));
    }
    const u64 nslots = epp / kStride;
    for (u64 s = threadIdx.x; s < nslots; s += blockDim.x) {
      const u64 e = s * kStride;
      if (px[e + 3] < 0.0f) continue;   // padded slot
      // EXPLICIT fma on every update, mirrored by std::fmaf in the host
      // replica: the bitwise gate must not depend on which contractions a
      // compiler happens to choose.
      const float vx0 = __fmaf_rn(half, gx, pv[e + 0]);
      const float vy0 = __fmaf_rn(half, gy_, pv[e + 1]);
      const float vz0 = __fmaf_rn(half, gz, pv[e + 2]);
      pv[e + 0] = vx0;
      pv[e + 1] = vy0;
      pv[e + 2] = vz0;
      if (drift) {
        px[e + 0] = __fmaf_rn(dt, vx0, px[e + 0]);
        px[e + 1] = __fmaf_rn(dt, vy0, px[e + 1]);
        px[e + 2] = __fmaf_rn(dt, vz0, px[e + 2]);
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      atomicAdd(&MdG().pages_done, 1ull);
      if (block < 64) {
        atomicAdd(&MdG().blk_done[block], 1ull);
        MdG().blk_last[block] = pg;
      }
    }
    // PUBLISH THIS PAGE, BY NAME. A block's table is private, so a page left
    // here is invisible to every other block -- and both the force stencil
    // and the resort gather read pages this block owns. Flushing the exact
    // page just written keeps every put disjoint: no two blocks can name the
    // same range, which is precisely what a whole-table flush could not
    // promise. Async, so the put overlaps the next page's integration.
    const u64 cnt = (x.size() - pg * epp < epp) ? x.size() - pg * epp : epp;
    if (MdG().publish && MdG().pub_interior) {
      co_await x.BeginFlush(0, pg * epp, cnt);
      co_await v.BeginFlush(0, pg * epp, cnt);
    }
    x.UnpinRange(pg * epp, epp);
    v.UnpinRange(pg * epp, epp);
    if (use_third) third.UnpinRange(pg * epp, epp);
  }
  // Land them before the kernel exits: the next kernel reads these pages
  // from other blocks, so they must be in the backing store by then.
  co_await x.EndFlush();
  co_await v.EndFlush();
}

/** K5: thermo -- KE and net momentum, block-reduced then atomically merged.
 *  Read-holds only. Reduction scratch is dynamic shared memory after the
 *  yield header. */
__device__ gy::YCoroMain ThermoCoro(gv::DeviceVector<float> x,
                                    gv::DeviceVector<float> v,
                                    double *out, u32 nb, u32 cap, u32 z0,
                                    u32 z1, u32 nblocks, u32 block) {
  MD_RED_SCRATCH(red);
  const u64 epp = x.ElemsPerPage();
  const Slab sl = SlabOf(nb, cap, z0, z1, epp);
  double ke = 0.0, mx = 0.0, my = 0.0, mz = 0.0;
  for (u64 pg = sl.pg_lo + block; pg < sl.pg_hi; pg += nblocks) {
    const u64 e0 = (pg * epp > sl.lo) ? pg * epp : sl.lo;
    const u64 e1 = ((pg + 1) * epp < sl.hi) ? (pg + 1) * epp : sl.hi;
    if (e1 <= e0) continue;
    co_await x.Fetch(0, pg * epp, epp);
    auto hx = co_await x.HoldPage(pg * epp, epp);
    co_await v.Fetch(0, pg * epp, epp);
    auto hv = co_await v.HoldPage(pg * epp, epp);
    const float *const px = hx.ptr();
    const float *const pv = hv.ptr();
    // Slots of THIS page that lie in the slab.
    const u64 s_lo = (e0 - pg * epp) / kStride;
    const u64 s_hi = (e1 - pg * epp) / kStride;
    for (u64 s = s_lo + threadIdx.x; s < s_hi; s += blockDim.x) {
      const u64 e = s * kStride;
      if (px[e + 3] < 0.0f) continue;
      const double vx = pv[e + 0], vy = pv[e + 1], vz = pv[e + 2];
      ke += 0.5 * (vx * vx + vy * vy + vz * vz);   // m = 1
      mx += vx; my += vy; mz += vz;
    }
    __syncthreads();
    x.UnpinRange(pg * epp, epp);
    v.UnpinRange(pg * epp, epp);
  }
  // Tree-reduce the four sums through shared, one at a time.
  const double vals[4] = {ke, mx, my, mz};
  for (int q = 0; q < 4; ++q) {
    red[threadIdx.x] = vals[q];
    __syncthreads();
    for (u32 w = blockDim.x / 2; w > 0; w >>= 1) {
      if (threadIdx.x < w) red[threadIdx.x] += red[threadIdx.x + w];
      __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(&out[q], red[0]);
    __syncthreads();
  }
}


/**
 * PUBLISH THIS NODE'S SLAB, as a generation.
 *
 * x and v are device-canonical inside one process -- nobody else can see the
 * frames, so nothing has to leave them. Across nodes that is exactly wrong:
 * the neighbour's force stencil reads one plane past this slab, and the
 * resort's gather reaches one plane in y AND z, so both arrays have to be in
 * the store before the neighbour looks. Flushed AS generation `gen`, which is
 * what lets the reader below demand that version rather than whatever it
 * happens to have.
 */
__device__ gy::YCoroMain PublishSlabCoro(gv::DeviceVector<float> x,
                                         gv::DeviceVector<float> v, u32 nb,
                                         u32 cap, u32 z0, u32 z1, u64 gen,
                                         u32 halo_first, u32 nblocks,
                                         u32 block) {
  const u64 epp = x.ElemsPerPage();
  // BATCHED, NOT PAGE-AT-A-TIME. This kernel is pure store latency: after
  // the force pass stopped parking, the per-page loop here -- one awaited
  // flush per page, one awaited fetch per halo page -- was 60% of the whole
  // run (15.6 ms/step against MPI's 6.3 TOTAL). GatherRanges is variadic and
  // a Multi task carries 64 records, so both boundary planes go out as ONE
  // flush per vector and both halo planes come back as ONE fetch: ~3 store
  // round trips per step instead of ~12.
  //
  // TWO BLOCKS, ONE VECTOR EACH. The wait is latency, not bandwidth; more
  // blocks would just contend for the same task slots.
  const u64 row_elems_p = static_cast<u64>(nb) * cap * kStride;
  const u64 plane_elems_p = static_cast<u64>(nb) * row_elems_p;
  const u64 lo_pl = static_cast<u64>(z0) * plane_elems_p;
  const u64 hi_pl = static_cast<u64>(z1 - 1) * plane_elems_p;
  const u64 below = static_cast<u64>((z0 + nb - 1u) % nb) * plane_elems_p;
  const u64 above = static_cast<u64>(z1 % nb) * plane_elems_p;
  // THREE INDEPENDENT LATENCIES, THREE BLOCKS. The halo wait depends on the
  // PEER's stamp, not on this node's flush -- measured 38% flush / 62% fetch
  // stacked SEQUENTIALLY in one block, so the step paid flush + fetch when it
  // only owes max(flush, fetch). Each block carries its own task slots, so
  // the three waits genuinely overlap.
  // THE PUBLISH RESERVES LIKE EVERYONE ELSE. These fetches pin up to four
  // planes of x while the force chunks hold their admitted stencils; as
  // unreserved pinners they broke the admission invariant, and at a pool two
  // pages short of the working set the first exchange deadlocked -- publish
  // waiting for regions the chunks pinned, chunks waiting for regions the
  // publish pinned. With every x-pinner reserving first, total pins never
  // exceed pool minus slack, so someone can always finish.
  if (block == 0) {
    const u32 pub_need = PagesSpanned(x, lo_pl, plane_elems_p) +
                         PagesSpanned(x, hi_pl, plane_elems_p);
    co_await AdmitSpans(pub_need, x.Regions(), 0u);
    const long long _c0 = clock64();
    co_await x.Fetch(0, lo_pl, plane_elems_p, hi_pl, plane_elems_p);
    co_await x.BeginFlush(gen, lo_pl, plane_elems_p, hi_pl, plane_elems_p);
    co_await x.EndFlush();
    if (threadIdx.x == 0) {
      atomicAdd(&MdG().pub_flush_cyc,
                (unsigned long long)(clock64() - _c0));
    }
    x.UnpinRange(lo_pl, plane_elems_p);
    x.UnpinRange(hi_pl, plane_elems_p);
    ReleaseSpans(pub_need);
  } else if (block == 1) {
    co_await v.Fetch(0, lo_pl, plane_elems_p, hi_pl, plane_elems_p);
    co_await v.BeginFlush(gen, lo_pl, plane_elems_p, hi_pl, plane_elems_p);
    co_await v.EndFlush();
    v.UnpinRange(lo_pl, plane_elems_p);
    v.UnpinRange(hi_pl, plane_elems_p);
  } else if (block == 2) {
    // The demand stays on the consumer (force still asks hgen); this fetch
    // warms the frames to `gen` -- AND THE PINS IT TAKES ARE KEPT. Under an
    // out-of-core pool the halo planes are the one place whose backing store
    // MUTATES while a pass reads them: the peer's write-site flushes land at
    // its own pace, so a halo page evicted mid-pass refaults into a NEWER
    // snapshot than its neighbours -- mixed-generation forces, 3e-04 drift
    // against 5e-07 everywhere the halo stayed resident, with every counter
    // clean. Pinning is the consumer-side fix (no barrier, no peer help):
    // the halo stays resident like an MPI ghost buffer, and each exchange
    // refreshes the SAME frames in place. Every fetch adds one pin; giving
    // back LAST exchange's keeps the count at exactly one between steps.
    // The reservation is taken once and released only when the resort swap
    // hands the halo role to the other vector (HaloUnpinKernel).
    const u32 pub_need = PagesSpanned(x, below, plane_elems_p) +
                         PagesSpanned(x, above, plane_elems_p);
    if (halo_first != 0u) co_await AdmitSpans(pub_need, x.Regions(), 0u);
    const long long _c1 = clock64();
    co_await x.Fetch(gen, below, plane_elems_p, above, plane_elems_p);
    if (threadIdx.x == 0) {
      atomicAdd(&MdG().pub_fetch_cyc,
                (unsigned long long)(clock64() - _c1));
    }
    if (halo_first == 0u) {
      x.UnpinRange(below, plane_elems_p);
      x.UnpinRange(above, plane_elems_p);
    }
  }
}

/** Take the halo pin on a vector whose halo the NEXT pass will read at
 *  `gen` -- the gather-scoped twin of the publish's persistent x pin. The v
 *  halo is read only by the resort's gather, so pinning it per-resort beats
 *  warming it every step; without the pin an evicted v page refaults
 *  MID-GATHER into whatever generation the peer has flushed since (the peer
 *  has no barrier and may already be publishing the next step), and a
 *  migrating atom crosses with one generation's position and another's
 *  velocity. */
__device__ gy::YCoroMain HaloPinCoro(gv::DeviceVector<float> x, u32 nb,
                                     u32 cap, u32 z0, u32 z1, u64 gen,
                                     u32 block) {
  if (block == 0) {
    const u64 row_elems_p = static_cast<u64>(nb) * cap * kStride;
    const u64 plane_elems_p = static_cast<u64>(nb) * row_elems_p;
    const u64 below = static_cast<u64>((z0 + nb - 1u) % nb) * plane_elems_p;
    const u64 above = static_cast<u64>(z1 % nb) * plane_elems_p;
    const u32 need = PagesSpanned(x, below, plane_elems_p) +
                     PagesSpanned(x, above, plane_elems_p);
    co_await AdmitSpans(need, x.Regions(), 0u);
    co_await x.Fetch(gen, below, plane_elems_p, above, plane_elems_p);
    // Pins deliberately kept; HaloUnpinKernel gives them back.
  }
  co_return;
}

/** REFAULT-STALENESS PROBE (MD_VERIFY_REFAULT=N): write a round-stamped
 *  pattern to every slab page, publish it at the write site, drop every
 *  frame, refault, and compare. The pattern encodes (page, round) in floats
 *  small enough to be exact, so a mismatch does not just say "wrong" -- its
 *  value says WHICH ROUND's bytes the store returned. Pages hash across the
 *  cluster, so roughly half of every node's slab exercises the REMOTE
 *  put/get path -- the one leg the single-node out-of-core gates never
 *  touch. */
__device__ float RefaultPattern(u64 pg, u64 round, u64 e) {
  return (float)(((pg * 131ull + round * 4099ull + (e & 63ull)) & 0xFFFFull));
}

__device__ gy::YCoroMain RefaultWriteCoro(gv::DeviceVector<float> x,
                                          u64 pg_lo, u64 pg_hi, u64 round,
                                          u64 ppp, u32 nblocks, u32 block) {
  const u64 epp = x.ElemsPerPage();
  for (u64 pg = pg_lo + block; pg < pg_hi; pg += nblocks) {
    co_await x.Fetch(0, pg * epp, epp);
    auto h = co_await x.HoldPage(pg * epp, epp, /*write=*/true);
    float *p = h.ptr();
    for (u64 e = threadIdx.x; e < epp; e += blockDim.x) {
      p[e] = RefaultPattern(pg, round, e);
    }
    __syncthreads();
    co_await x.BeginFlush(0, pg * epp, epp);
    x.UnpinRange(pg * epp, epp);
  }
  co_await x.EndFlush();
  // THE EXCHANGE HALF: stamp the boundary planes at round+1, exactly the
  // publish's job in the md step. ppp == 0 turns the protocol off.
  if (ppp != 0 && block == 0) {
    co_await x.BeginFlush(round + 1, pg_lo * epp, ppp * epp,
                          (pg_hi - ppp) * epp, ppp * epp);
    co_await x.EndFlush();
  }
}

/** Which round wrote this value? -1 if it matches no round's pattern. */
__device__ int RefaultDecode(u64 pg, u64 e, float v) {
  for (int r = 0; r < 64; ++r) {
    if (v == RefaultPattern(pg, (u64)r, e)) return r;
  }
  return -1;
}

__device__ gy::YCoroMain RefaultVerifyCoro(gv::DeviceVector<float> x,
                                           u64 pg_lo, u64 pg_hi, u64 round,
                                           u64 ppp, u64 below_pg, u64 above_pg,
                                           u32 nblocks, u32 block,
                                           unsigned long long *d_out) {
  const u64 epp = x.ElemsPerPage();
  for (u64 pg = pg_lo + block; pg < pg_hi; pg += nblocks) {
    co_await x.Fetch(0, pg * epp, epp);
    auto h = co_await x.HoldPage(pg * epp, epp);
    const float *p = h.ptr();
    for (u64 e = threadIdx.x; e < epp; e += blockDim.x) {
      const float want = RefaultPattern(pg, round, e);
      if (p[e] != want) {
        atomicAdd(&d_out[0], 1ull);
        // First offender: page, element, and the observed bits.
        if (atomicCAS((unsigned long long *)&d_out[1], 0ull, 1ull) == 0ull) {
          d_out[2] = pg;
          d_out[3] = e;
          d_out[4] = (unsigned long long)__float_as_uint(p[e]);
        }
      }
    }
    __syncthreads();
    x.UnpinRange(pg * epp, epp);
  }
  // THE CONSUMER HALF: read the PEER's boundary (our halo) at >= round+1 and
  // decode WHICH ROUND each element's bytes came from. `exact` is the only
  // healthy answer; `newer` is the free-running peer overwriting the store
  // mid-protocol; `older` is a broken gate; `garbage` is a torn read.
  if (ppp != 0 && block == 0) {
    co_await x.Fetch(round + 1, below_pg * epp, ppp * epp, above_pg * epp,
                     ppp * epp);
    for (u32 half = 0; half < 2; ++half) {
      const u64 base = (half == 0 ? below_pg : above_pg);
      for (u64 pg = base; pg < base + ppp; ++pg) {
        auto h = co_await x.HoldPage(pg * epp, epp);
        const float *p = h.ptr();
        for (u64 e = threadIdx.x; e < epp; e += blockDim.x) {
          const int r = RefaultDecode(pg, e, p[e]);
          if (r == (int)round) {
            atomicAdd(&d_out[5], 1ull);
          } else if (r > (int)round) {
            atomicAdd(&d_out[6], 1ull);
          } else if (r >= 0) {
            atomicAdd(&d_out[7], 1ull);
          } else {
            atomicAdd(&d_out[8], 1ull);
          }
        }
        __syncthreads();
      }
    }
    x.UnpinRange(below_pg * epp, ppp * epp);
    x.UnpinRange(above_pg * epp, ppp * epp);
  }
}

/** Give back the persistent halo pin before the resort swaps the handles:
 *  the OTHER vector carries the halo from the next exchange on, and a pin
 *  left behind would strand two planes of the scatter destination's pool. */
__device__ gy::YCoroMain HaloUnpinCoro(gv::DeviceVector<float> x, u32 nb,
                                       u32 cap, u32 z0, u32 z1, u32 block) {
  if (block == 0) {
    const u64 row_elems_p = static_cast<u64>(nb) * cap * kStride;
    const u64 plane_elems_p = static_cast<u64>(nb) * row_elems_p;
    const u64 below = static_cast<u64>((z0 + nb - 1u) % nb) * plane_elems_p;
    const u64 above = static_cast<u64>(z1 % nb) * plane_elems_p;
    const u32 need = PagesSpanned(x, below, plane_elems_p) +
                     PagesSpanned(x, above, plane_elems_p);
    x.UnpinRange(below, plane_elems_p);
    x.UnpinRange(above, plane_elems_p);
    ReleaseSpans(need);
  }
  co_return;
}



#endif  // CLIO_GV_BENCH_MD_KERNELS_H_
