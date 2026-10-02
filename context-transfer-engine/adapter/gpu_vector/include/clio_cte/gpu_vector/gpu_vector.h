/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * gpu_vector: a vector whose backing store is the CTE and whose working set
 * lives in GPU memory, paged in on demand from inside a kernel.
 *
 * Host side (this class) owns the allocations and hands out a device view;
 * device side (DeviceVector) does the paging. A kernel can allocate nothing,
 * so every buffer, task and page table is allocated and registered here.
 */
#ifndef CLIO_CTE_GPU_VECTOR_GPU_VECTOR_H_
#define CLIO_CTE_GPU_VECTOR_GPU_VECTOR_H_

#include <clio_runtime/clio_runtime.h>
#include <clio_ctp/util/gpu_api.h>
#include <clio_runtime/types.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#if !CTP_IS_DEVICE_PASS
// Vector::Copy (host-only) creates the checkpoint fault-handler pool.
#include <clio_cte/checkpoint/checkpoint_client.h>
#endif
#include <clio_cte/gpu_vector/device_vector.h>
#include <clio_cte/gpu_vector/page.h>
#include <clio_cte/gpu_vector/prefetch.h>

#include <unistd.h>

#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

namespace clio::cte::gpu_vector {

/*
 * The host side is guarded on CTP_ENABLE_GPU, not CTP_ENABLE_CUDA.
 *
 * Every one of these bodies is written against ctp::GpuApi, which already
 * dispatches to CUDA, ROCm or SYCL -- there is not a raw cuda* call in this
 * file. The narrower guard was silently compiling the host half of the
 * vector to NOTHING on a SYCL build: PublishHeader never allocated the
 * device VecHeader, so every kernel got a DeviceVector whose h_ was null and
 * the first launch died with CUDA_ERROR_ILLEGAL_ADDRESS. UploadBytes was a
 * no-op for the same reason, so the page table and task slots were never
 * uploaded either. Widening changes nothing for CUDA (CTP_ENABLE_GPU is
 * implied by it).
 */

/** A device-resident task region and its host snapshot (see
 *  ResumeWhenComplete). One per vector device table; host-only. */
struct TaskRegion {
  char *base = nullptr;
  std::vector<char> snap;
  clio::run::u32 last_block = 0;
  bool fresh = false;
};
inline std::vector<TaskRegion> &TaskRegions() {
  static std::vector<TaskRegion> regions;
  return regions;
}
inline void RegisterTaskRegion(char *base, size_t bytes) {
  TaskRegion r;
  r.base = base;
  r.snap.resize(bytes);
  TaskRegions().push_back(std::move(r));
}
inline TaskRegion *FindTaskRegion(clio::run::u64 tag) {
  for (auto &r : TaskRegions()) {
    const auto lo = reinterpret_cast<clio::run::u64>(r.base);
    if (tag >= lo && tag + sizeof(unsigned int) <= lo + r.snap.size()) return &r;
  }
  return nullptr;
}

/**
 * Resume a parked block ONLY when the completion word its wait tag names has
 * flipped.
 *
 * Pass this as RunToCompletion's resume_when. Without it the driver relaunches
 * every parked block every round, which polls a fault by kernel launch: the
 * block wakes, sees its transfer has not landed, and parks again. With it the
 * host reads the completion word and holds the block back until the task is
 * actually done -- one relaunch per fault.
 *
 * A wait tag of 0 means "no condition", so those blocks always resume.
 */
inline bool ResumeWhenComplete(clio::run::u32 block, clio::run::u64 tag) {
#if CTP_ENABLE_GPU
  // ONE COPY PER SPIN, NOT ONE PER BLOCK. The completion words live in the
  // task sets, which are device memory (the kernel does device-scope atomics
  // on them; PVC faults an atomic on host memory). Reading them one blocking
  // 4-byte memcpy per parked block per spin of the driver was 867k Level Zero
  // copies in a 4-step Gray-Scott run, ~50 us each: at 1024 parked blocks a
  // single spin cost ~50 ms before anything relaunched. Instead the whole
  // task region is snapshotted in one copy when a new spin starts (the driver
  // walks pending blocks in increasing order, so a block index that does not
  // increase marks a new spin) and every block's word is read from that.
  if (tag == 0) return true;
  TaskRegion *r = FindTaskRegion(tag);
  if (r == nullptr) {
    unsigned int done = 0;
    ctp::GpuApi::Memcpy(reinterpret_cast<char *>(&done),
                        reinterpret_cast<const char *>(tag), sizeof(done));
    return (done & 1u) != 0u;
  }
  if (!r->fresh || block <= r->last_block) {
    ctp::GpuApi::Memcpy(r->snap.data(), r->base, r->snap.size());
    r->fresh = true;
  }
  r->last_block = block;
  unsigned int done = 0;
  std::memcpy(&done, r->snap.data() + (tag - reinterpret_cast<clio::run::u64>(r->base)),
              sizeof(done));
  return (done & 1u) != 0u;
#else
  (void) tag;
  return true;
#endif
}

// The host class is HOST-ONLY. Its body calls host-only client methods
// (AsyncGetOrCreateTag), and those are non-dependent names, so the device
// pass would fail to parse them even though it never instantiates the class.
#if !CTP_IS_DEVICE_PASS

/**
 * The process-wide fatal channel: eight slots of PINNED HOST memory that the
 * device writes just before it traps.
 *
 * Host memory because nothing else survives a trap -- device printf is
 * buffered and dies with the context, and so does device memory, which is why
 * a trapping kernel reports only "CUDA Error 715: an illegal instruction" and
 * nothing about the cause. Under unified addressing the device can store
 * straight into a cudaMallocHost allocation, so this costs nothing until it
 * is used.
 *
 * One buffer for the process, not one per vector: a trap is fatal, so only
 * the first one matters, and every vector can point at the same slots.
 */
inline unsigned long long *FatalSlots() {
#if CTP_ENABLE_GPU
  static unsigned long long *slots = [] {
#if CTP_ENABLE_SYCL
    // DEVICE memory under SYCL. The latch is an atomicCAS from device code,
    // and on Aurora's Max 1550 a device atomic to malloc_host memory faults
    // outright (usm_atomic_host_allocations=0), while one to malloc_shared
    // faults whenever the page happens to be resident on the HOST at that
    // moment -- an AtomicAccessViolation at PDE level, not a migration.
    // Shared memory passed benchmark/usm_atomic_probe.cc only because that
    // kernel's launch migrated the page first; here the host memsets the
    // slots at start-up and the device first touches them seconds later,
    // mid-run, and the trap's own report became the crash that hid it
    // (weights, lbann). Device memory takes atomics unconditionally. The
    // host reads the note from FatalMirror() instead.
    auto *p = ctp::GpuApi::Malloc<unsigned long long>(
        8 * sizeof(unsigned long long));
    if (p != nullptr) ctp::GpuApi::Memset(p, 0, 8 * sizeof(unsigned long long));
#else
    auto *p = ctp::GpuApi::MallocHost<unsigned long long>(
        8 * sizeof(unsigned long long));
    if (p != nullptr) std::memset(p, 0, 8 * sizeof(unsigned long long));
#endif
    return p;
  }();
  return slots;
#else
  return nullptr;
#endif
}

/** Name of a fatal code, for the report.
 *  @param code slot 0 of the fatal channel
 *  @return a short description, "unknown" for a code this build does not know */
inline const char *FatalWhat(unsigned long long code) {
  switch (code) {
    case kFatalInitBlock:   return "Init(block) beyond the task table";
    case kFatalNotResident: return "HoldPage: page not resident";
    case kFatalNotCovered:  return "HoldPage: range never fetched";
    case kFatalUnbound:     return "vector used before Init()";
    case kFatalSetFull:     return "AllocatePage: set full";
    case kFatalFlushSplit:  return "flush range needs more records";
    case kFatalGetFailed:
      return "fetch returned an error; its pages were left EMPTY "
             "(a generational get names a generation the writer has "
             "not published)";
    default: return "unknown";
  }
}

/** Format the eight fatal slots. Plain snprintf into the caller's buffer, so
 *  the abort handler below can use it too.
 *  @param f   the slots (host-readable)
 *  @param buf output
 *  @param n   bytes in buf
 *  @return bytes written, 0 when nothing was latched */
inline int FatalFormat(const unsigned long long *f, char *buf, size_t n) {
  if (f == nullptr || f[0] == kFatalNone) return 0;
  int w = std::snprintf(
      buf, n, "[gpu_vector] DEVICE FATAL %llu (%s): a1=%llu a2=%llu a3=%llu "
      "block=%llu\n", f[0], FatalWhat(f[0]), f[1], f[2], f[3], f[4]);
  if (w > 0 && f[0] == kFatalSetFull && static_cast<size_t>(w) < n) {
    // Slots 5-7 carry ReportSetFull's tallies, packed as documented there.
    const unsigned long long t4 = f[5], t5 = f[6], t6 = f[7];
    const int w2 = std::snprintf(
        buf + w, n - static_cast<size_t>(w),
        "[gpu_vector]   home set: %llu pinned, %llu in flight | all sets: "
        "%llu resident, %llu pinned, %llu fetching, %llu flushing, %llu "
        "empty | regions: %llu of %llu on free lists\n",
        f[3], t4 & 0xFFFFull, t5 & 0xFFFFull, (t5 >> 16) & 0xFFFFull,
        (t4 >> 16) & 0xFFFFull, (t4 >> 32) & 0xFFFFull, (t5 >> 32) & 0xFFFFull,
        t6 & 0xFFFFFFFFull, t6 >> 32);
    if (w2 > 0) w += w2;
  }
  return w < 0 ? 0 : (w > static_cast<int>(n) ? static_cast<int>(n) : w);
}

/**
 * The HOST-READABLE copy of the fatal channel.
 *
 * Under SYCL the latch itself is device memory (see FatalSlots) and the
 * device mirrors each note here with plain stores; elsewhere this IS the
 * latch. On Level Zero a device trap does not return to the caller -- the
 * driver aborts the process -- so the only way the note reaches a human is
 * a SIGABRT handler that prints this buffer on the way down, installed on
 * first use. Without it a trap reads as an unexplained AtomicAccessViolation
 * or a "fault at 0x0", which is what every FATAL on Aurora looked like.
 */
inline unsigned long long *FatalMirror() {
#if CTP_ENABLE_GPU
  static unsigned long long *mirror = [] {
#if CTP_ENABLE_SYCL
    auto *p = ctp::GpuApi::MallocHost<unsigned long long>(
        8 * sizeof(unsigned long long));
    if (p != nullptr) std::memset(p, 0, 8 * sizeof(unsigned long long));
#else
    unsigned long long *p = FatalSlots();
#endif
    struct Reporter {
      static void OnAbort(int sig) {
        char buf[512];
        const int w = FatalFormat(FatalMirror(), buf, sizeof(buf));
        if (w > 0) {
          const ssize_t r = ::write(2, buf, static_cast<size_t>(w));
          (void)r;
        }
        std::signal(sig, SIG_DFL);
        std::raise(sig);
      }
    };
    // SIGABRT ONLY. The Level Zero driver owns SIGSEGV: a host access to a
    // shared-USM page the device holds is a fault the driver's handler
    // resolves by migrating the page back. Taking SIGSEGV here killed
    // lammps_md three seconds in, on its fatal-channel poller's first read
    // after a kernel had pulled the page to the device.
    std::signal(SIGABRT, &Reporter::OnAbort);
    return p;
  }();
  return mirror;
#else
  return nullptr;
#endif
}

/** Human-readable form of whatever the device latched, or "" if nothing. */
inline std::string FatalReport() {
  char buf[512];
  const int w = FatalFormat(FatalMirror(), buf, sizeof(buf));
  if (w <= 0) return std::string();
  std::string s(buf, static_cast<size_t>(w));
  while (!s.empty() && s.back() == '\n') s.pop_back();
  return s;
}

template <typename T>
class Vector {
 public:
  /** Counters, populated only when EnableStats() was called. */
  struct Stats {
    clio::run::u64 faults = 0;   // pages read in
    clio::run::u64 puts = 0;     // pages written back
    clio::run::u64 evicts = 0;   // frames reclaimed
    clio::run::u64 get_errors = 0;   // faults that returned non-zero
    clio::run::u64 gen_ok = 0;       // generational demand: resident accepted
    clio::run::u64 gen_stale = 0;    // generational demand: refetch claimed
    clio::run::u64 gen_busy = 0;     // generational demand: CAS lost to peer
    clio::run::u64 flush_skipped = 0;  // flush pages not resident: DROPPED
    clio::run::u64 put_errors = 0;   // writebacks that returned non-zero
    clio::run::u64 alloc_waits = 0;  // page-cache backoff retries (FaultPage)
    // GV_COMM_TIMING only (zero otherwise): GPU cycles in the Fetch, Hold
    // and Flush calls, a benchmark's busy segments, and the call counts.
    clio::run::u64 fetch_cyc = 0, hold_cyc = 0, flush_cyc = 0, busy_cyc = 0;
    clio::run::u64 fetch_calls = 0, hold_calls = 0, flush_calls = 0;
  };

  /**
   * @param tag_name   CTE tag representing this vector
   * @param gpu_ids    GPUs to build device views for
   * @param page_bytes page granularity
   * @param nblocks    CUDA BLOCKS the kernel will launch. This dimensions the
   *        per-block task sets -- each block owns its MultiGet and MultiPut,
   *        because two blocks staging into one slot would overwrite each
   *        other's records mid-transfer -- and, by default, the set count.
   * @param set_size   frames in one associative set. Enough of them to cover
   *        every block that can hold a page of one set at once, plus room for
   *        hash collisions.
   * @param num_elems  logical length of the vector
   * @param nsets      associative sets; 0 means "as many as there are blocks".
   *        Separate from nblocks because the cache's geometry and the grid's
   *        width are unrelated numbers -- they were one field, and reading
   *        `nblocks` as either "tables" or "sets" depending on context is
   *        exactly what made this class hard to follow.
   *
   * THERE IS ONE CACHE. `nblocks` sets, `pages_per_block` frames each, and a
   * page's home set is Hash(page_num) % nblocks -- a function of the PAGE, so
   * every CUDA block looks in the same place and a page touched by all of
   * them is stored ONCE. The per-block page table this class used to build is
   * gone: it stored a hot page once per block, and every cross-block
   * correctness rule (invalidate after a resort, borrow a peer's frame,
   * publish before another block can read) existed to paper over that.
   */
  Vector(const std::string &tag_name, const std::vector<int> &gpu_ids,
         clio::run::u64 page_bytes, clio::run::u32 nblocks,
         clio::run::u32 set_size, clio::run::u64 num_elems,
         clio::run::PoolId storage_pool_id = clio::run::PoolId::GetNull(),
         int compress_lib = 0, int compress_preset = 1,
         clio::run::u32 nsets = 0, clio::run::u32 capacity_pages = 0)
      : storage_pool_id_(storage_pool_id.IsNull() ? clio::cte::core::kCtePoolId
                                                  : storage_pool_id),
        compress_lib_(compress_lib),
        compress_preset_(compress_preset),
        page_bytes_(page_bytes),
        nblocks_(nblocks),
        nsets_(nsets != 0 ? nsets : nblocks),
        set_size_(set_size),
        // CAPACITY IS PAGES, NOT SLOTS. Slots are 64-byte tags; capacity is
        // the page-sized storage behind them, and it is what the cache
        // actually costs. 0 keeps the old meaning (one region per slot) so a
        // caller that has not thought about it is no worse off.
        capacity_pages_(capacity_pages != 0
                            ? capacity_pages
                            : (nsets != 0 ? nsets : nblocks) * set_size),
        num_elems_(num_elems),
        tag_name_(tag_name),
        gpu_ids_(gpu_ids) {
    if (page_bytes_ == 0 || nblocks_ == 0 || set_size_ == 0 || nsets_ == 0) {
      throw std::runtime_error("gpu_vector: zero page size / blocks / sets");
    }
    clio::cte::core::Client cte(clio::cte::core::kCtePoolId);
    auto tag = cte.AsyncGetOrCreateTag(tag_name);
    tag.Wait();
    if (tag->GetReturnCode() != 0) {
      throw std::runtime_error("gpu_vector: could not resolve tag " + tag_name);
    }
    tag_id_ = tag->tag_id_;
    try {
      for (int gpu : gpu_ids) BuildDevice(gpu);
    } catch (...) {
      for (auto &kv : devs_) Free(kv.second);
      devs_.clear();
      throw;
    }
  }

  ~Vector() {
    for (auto &kv : devs_) Free(kv.second);
  }

  Vector(const Vector &) = delete;
  Vector &operator=(const Vector &) = delete;

  /** The value a kernel takes by value. */
  DeviceVector<T> GetDevice(int gpu_id) const {
    auto it = devs_.find(gpu_id);
    if (it == devs_.end()) {
      throw std::runtime_error("gpu_vector: no device view for this GPU");
    }
    return it->second.view;
  }

  /**
   * Lazily-initialised copy of this vector: the checkpoint primitive.
   *
   * First publishes every GPU's resident state to the CTE (the copy must
   * snapshot what the vector holds NOW, not what happened to be flushed),
   * then creates tag `new_name` with the checkpoint chimod registered as
   * its fault handler and THIS vector's tag as the source. No page bytes
   * are duplicated here -- a page of the copy materialises from the source
   * the first time anything touches it (get OR put), so the checkpoint
   * costs I/O only for pages that are later read back or overwritten.
   *
   * The returned handle is CTE-ONLY: it builds NO device views (those cost
   * a full page-cache allocation in VRAM). Host Download/Preload work; a
   * kernel that must read the snapshot binds its own Vector to `new_name`.
   *
   * @param new_name tag of the copy
   * @param sync     false (default): lazy copy-on-write, as above. true:
   *                 FULLY SYNCHRONOUS -- every page of the copy is
   *                 materialised (MaterializeAll) before Copy returns, so the
   *                 copy holds real bytes equal to the source at this call
   *                 and later writes to the source cannot leak into it.
   * @return the copy's handle
   * @throws std::runtime_error if any step, including a sync
   *         materialisation, fails
   */
  std::unique_ptr<Vector<T>> Copy(const std::string &new_name,
                                  bool sync = false) {
    using clio::cte::core::Context;
    if (tag_name_.size() >= Context::kFaultParamsSize) {
      // The registration would be silently truncated at fault time and the
      // handler would resolve a DIFFERENT source tag.
      throw std::runtime_error("gpu_vector: source tag name too long for "
                               "fault params: " + tag_name_);
    }
    // The snapshot is of the vector's CURRENT bytes, so anything resident
    // only in GPU caches goes to the CTE first.
    FlushResidentToCte();
    // The chimod pool that services the faults (idempotent create).
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
        throw std::runtime_error("gpu_vector: checkpoint pool create failed");
      }
    }
    // Register the fault handler on the copy's tag BEFORE the Vector ctor
    // resolves it. GetOrCreateTag without fault arguments leaves an existing
    // registration alone, so the ctor's own lookup cannot clobber this.
    {
      clio::cte::core::Client core(storage_pool_id_);
      auto tf = core.AsyncGetOrCreateTag(
          new_name, clio::cte::checkpoint::kCheckpointPoolId,
          clio::cte::checkpoint::kCheckpointPoolName, tag_name_);
      tf.Wait();
      if (tf->GetReturnCode() != 0) {
        throw std::runtime_error("gpu_vector: could not create copy tag " +
                                 new_name);
      }
    }
    std::unique_ptr<Vector<T>> out(new Vector<T>(
        new_name, std::vector<int>{}, page_bytes_, nblocks_, set_size_,
        num_elems_, storage_pool_id_, compress_lib_, compress_preset_,
        nsets_, capacity_pages_));
    // MaterializeAll needs the SOURCE name: the fault it issues carries the
    // source in fault_params_, exactly as the core's own internal fault does.
    out->ckpt_src_ = tag_name_;
    if (sync) {
      const clio::run::u64 np = out->NumPages();
      if (out->MaterializeAll() != np) {
        throw std::runtime_error("gpu_vector: synchronous copy " + new_name +
                                 " failed to materialise");
      }
    }
    return out;
  }

  /**
   * Force a lazy Copy() to become REAL BYTES, server-side.
   *
   * Copy() is copy-on-write: it registers a fault handler and duplicates
   * nothing, so an untouched checkpoint occupies zero storage. That is the
   * point of it -- and it makes an untouched checkpoint useless as a
   * checkpoint, because the handler materialises from the SOURCE AT FAULT
   * TIME. A workload that keeps mutating the source (Gray-Scott overwrites
   * each region pair every second step) therefore has a window: materialise
   * inside it and the snapshot is the state at Copy(); materialise after it
   * and the snapshot silently holds LATER bytes.
   *
   * The mechanism is the put-side materialise-only fault: a put of size 0
   * with a null pointer finds no blob under the copy tag, dispatches the
   * fault, and the checkpoint chimod reads the whole source blob and writes
   * it into the copy. NO BYTES CROSS TO THE HOST -- the copy happens entirely
   * inside the runtime, which is why this is not simply "read the checkpoint
   * back".
   *
   * Submitted in flights rather than one at a time: each materialisation is a
   * whole-blob read plus a whole-blob write, and serialising a 1 GiB
   * checkpoint's worth of them at ~1 ms each is minutes of nothing.
   *
   * @return pages materialised (0 on the first failure, which is reported).
   */
  clio::run::u64 MaterializeAll(clio::run::u32 inflight = 32) {
    if (ckpt_src_.empty()) {
      std::fprintf(stderr, "gpu_vector: MaterializeAll on '%s', which is not "
                   "a Copy() -- there is no source to materialise from\n",
                   tag_name_.c_str());
      return 0;
    }
    // THE MATERIALISE-ONLY FAULT, ISSUED DIRECTLY.
    //
    // NOT a size-0 PutBlob: PutBlobImpl rejects size == 0 with rc=2 long
    // before it reaches the fault dispatch, so the "put-side materialise-only
    // fault" the chimod documents is generated INTERNALLY by the core and is
    // not reachable from a client put. And not an ordinary Get either -- that
    // materialises, but only by dragging every byte back to the host, which
    // for a 32 GiB checkpoint stream is 32 GiB of pointless PCIe traffic.
    //
    // So we send the core's own fault task: a GetBlobTask addressed to the
    // CHECKPOINT POOL with size 0 and a null destination, carrying the source
    // tag in fault_params_. The handler reads the whole source blob and puts
    // it into the copy, entirely inside the runtime.
    auto *ipc = CLIO_CPU_IPC;
    const clio::run::u64 np = NumPages();
    std::vector<clio::run::Future<clio::cte::core::GetBlobTask>> futs;
    futs.reserve(inflight);
    clio::run::u64 done = 0;
    auto reap = [&]() -> bool {
      for (auto &f : futs) {
        f.Wait();
        if (f->GetReturnCode() != 0) {
          std::fprintf(stderr,
                       "gpu_vector: MaterializeAll page %llu rc=%d (tag '%s' "
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
    for (clio::run::u64 pg = 0; pg < np; ++pg) {
      clio::cte::core::Context ctx = PageContext();
      // The handler decodes fault_params_ as the source tag NAME.
      std::snprintf(ctx.fault_params_,
                    clio::cte::core::Context::kFaultParamsSize, "%s",
                    ckpt_src_.c_str());
      // The name goes on the wire decoded; the raw-int32 flag would make the
      // handler decode a decimal name a second time.
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

  /**
   * Score every page of this vector, for a caller draining a checkpoint out
   * of the fast tier. Same batched rescore path as the prefetcher.
   *
   * A materialised checkpoint lands at blob score 1.0 (the fault handler puts
   * at -1.0, which PutBlobImpl resolves to 1.0 for a new blob), so it
   * outranks the live data on every tier and fills the fast one with bytes
   * nobody will read again. Draining it is not an optimisation -- it is the
   * difference between a fast tier holding the working set and a fast tier
   * holding cold history.
   */
  void ScoreAllPages(float score) {
    const clio::run::u64 np = NumPages();
    std::vector<PrefetchHint> hints;
    hints.reserve(static_cast<size_t>(np));
    for (clio::run::u64 pg = 0; pg < np; ++pg) {
      hints.push_back(PrefetchHint{pg, score});
    }
    PrefetchNow(hints);
  }

  /**
   * HOST-SIDE writeback of every resident frame's valid elements, on every
   * GPU this vector has views for. There is no dirty bit (write-site publish
   * is the kernel-side contract), so this publishes ALL resident valid
   * ranges: for pages the kernels already flushed the put is byte-identical,
   * and for anything they had not yet published this is what makes the CTE
   * current. No-op with no device views or an empty cache.
   */
  void FlushResidentToCte() {
#if CTP_ENABLE_GPU
    clio::cte::core::Client core(storage_pool_id_);
    std::vector<char> buf(static_cast<size_t>(page_bytes_));
    for (auto &kv : devs_) {
      const std::vector<Page> tbl = ReadTable(kv.first);
      for (const Page &p : tbl) {
        if (p.page_num == kNoPage || p.data == nullptr) continue;
        if (p.valid_hi <= p.valid_lo) continue;
        const clio::run::u64 lo_b =
            static_cast<clio::run::u64>(p.valid_lo) * sizeof(T);
        const clio::run::u64 hi_b =
            static_cast<clio::run::u64>(p.valid_hi) * sizeof(T);
        ctp::GpuApi::Memcpy(buf.data(),
                            static_cast<const char *>(p.data) + lo_b,
                            static_cast<size_t>(hi_b - lo_b));
        auto f = core.AsyncPutBlob(tag_id_, PageName(p.page_num), lo_b,
                                   hi_b - lo_b, buf.data(), 0.5f,
                                   PageContext(), 0u,
                                   clio::run::PoolQuery::Dynamic());
        f.Wait();
      }
    }
#endif
  }

  clio::cte::core::TagId TagId() const { return tag_id_; }
  clio::run::u64 PageBytes() const { return page_bytes_; }
  clio::run::u64 ElemsPerPage() const { return page_bytes_ / sizeof(T); }
  clio::run::u64 NumPages() const {
    const clio::run::u64 epp = ElemsPerPage();
    return (num_elems_ + epp - 1) / epp;
  }

  void EnableStats() {
#if CTP_ENABLE_GPU
    for (auto &kv : devs_) {
      if (kv.second.stats != nullptr) continue;
      const size_t bytes = 17 * sizeof(unsigned long long);
      auto *c = reinterpret_cast<unsigned long long *>(
          ctp::GpuApi::Malloc<char>(bytes));
      if (c == nullptr) throw std::runtime_error("gpu_vector: stats alloc failed");
      ctp::GpuApi::Memset(c, 0, bytes);
      kv.second.stats = c;
      kv.second.hdr.stat_faults_ = c;
      kv.second.hdr.stat_puts_ = c + 1;
      kv.second.hdr.stat_evicts_ = c + 2;
      kv.second.hdr.stat_get_errors_ = c + 3;
      kv.second.hdr.stat_put_errors_ = c + 4;
      // WHERE A GENERATIONAL DEMAND ACTUALLY GOES. Three outcomes, and only
      // one of them transfers anything.
      kv.second.hdr.stat_gen_ok_ = c + 5;      // resident copy accepted
      kv.second.hdr.stat_gen_stale_ = c + 6;   // stale -> refetch claimed
      kv.second.hdr.stat_gen_busy_ = c + 7;    // stale but CAS lost
      // Pages a flush could not FIND -- silently dropped writebacks.
      kv.second.hdr.stat_flush_skipped_ = c + 8;
      kv.second.hdr.stat_alloc_waits_ = c + 9;
      kv.second.hdr.stat_cyc_ = c + 10;        // 7 slots: see VecHeader
      PublishHeader(kv.second);
    }
#endif
  }

  /**
   * HOST-SIDE data movement. Not part of the device API: a kernel never calls
   * these. They exist because host data has to reach the vector somehow, and
   * because benches reset residency between measured rounds.
   */

  /**
   * A page's home set, host side. MUST mirror DeviceVector::SetOf exactly --
   * the host and the device disagreeing about where a page lives is a page
   * the device can never find and a slot nothing can reclaim.
   */
  clio::run::u32 HomeSet(clio::run::u64 pn) const {
    const clio::run::u64 mixed = (pn ^ (pn >> 29)) * 0x9E3779B97F4A7C15ull;
    return static_cast<clio::run::u32>((mixed >> 32) % nsets_);
  }

  /**
   * Make pages [pg_lo, pg_hi) resident, best effort: ONCE each, in the page's
   * home set, because the whole grid shares one cache.
   *
   * `tables` named how many per-block tables to fill and is ignored -- there
   * is one cache. Filling a table at a time put each page in as many wrong
   * sets as there were blocks: invisible to a device lookup, which searches
   * only the home set, while occupying every slot the real fetch then needed.
   */
  void Prefetch(clio::run::u64 pg_lo, clio::run::u64 pg_hi, int gpu_id = 0,
                clio::run::u32 tables = 0) {
#if CTP_ENABLE_GPU
    (void) tables;
    auto it = devs_.find(gpu_id);
    if (it == devs_.end()) return;
    clio::cte::core::Client core(storage_pool_id_);
    PrefetchShared(it->second, pg_lo, pg_hi, core);
#else
    (void) pg_lo; (void) pg_hi; (void) gpu_id; (void) tables;
#endif
  }

  /** Drop every resident page. Dirty pages are DISCARDED -- flush first. */
  void ClearCache(int gpu_id = 0) {
#if CTP_ENABLE_GPU
    auto it = devs_.find(gpu_id);
    if (it == devs_.end()) return;
    const clio::run::u64 nslots =
        static_cast<clio::run::u64>(nsets_) * set_size_;
    std::vector<Page> tbl(static_cast<size_t>(nslots));
    ctp::GpuApi::Memcpy(reinterpret_cast<char *>(tbl.data()),
                        it->second.table_base,
                        static_cast<size_t>(nslots * sizeof(Page)));
    for (auto &p : tbl) {
      p.page_num = kNoPage;
      p.data = nullptr;      // the region goes back to its owner's free list
      p.pins = 0;
      p.flushing = 0;
      p.fetching = 0;
      p.score = kDefaultScore;
      p.valid_lo = 0;
      p.valid_hi = 0;
      p.last_access = 0;
    }
    ctp::GpuApi::Memcpy(it->second.table_base,
                        reinterpret_cast<const char *>(tbl.data()),
                        static_cast<size_t>(nslots * sizeof(Page)));
    // AND EVERY REGION GOES BACK. Emptying the slots while leaving the free
    // lists as they were would strand the storage those slots held: the tags
    // say the cache is empty and the allocator says it has nothing to give.
    RebuildFreeLists(it->second, 0);
#else
    (void) gpu_id;
#endif
  }

  /** Write pages [pg_lo, pg_hi); `fill(pg, buf)` supplies each page's bytes. */
  template <typename FillFn>
  void PreloadPages(clio::run::u64 pg_lo, clio::run::u64 pg_hi, FillFn fill) {
    clio::cte::core::Client core(storage_pool_id_);
    std::vector<char> buf(static_cast<size_t>(page_bytes_));
    for (clio::run::u64 pg = pg_lo; pg < pg_hi; ++pg) {
      std::memset(buf.data(), 0, buf.size());
      fill(pg, buf.data());
      auto f = core.AsyncPutBlob(tag_id_, PageName(pg), 0, page_bytes_,
                                 buf.data(), 0.5f, PageContext(), 0u,
                                 clio::run::PoolQuery::Dynamic());
      f.Wait();
    }
  }

  /** Write `n` elements from host memory into the vector. */
  void Preload(const T *src, clio::run::u64 n) {
    const clio::run::u64 epp = ElemsPerPage();
    PreloadPages(0, (n + epp - 1) / epp, [&](clio::run::u64 pg, char *buf) {
      const clio::run::u64 off = pg * epp;
      const clio::run::u64 cnt = (epp < n - off) ? epp : (n - off);
      std::memcpy(buf, src + off, static_cast<size_t>(cnt * sizeof(T)));
    });
  }

  /** Read `n` elements out of the vector into host memory.
   *  @return elements actually read. */
  clio::run::u64 Download(T *dst, clio::run::u64 n) {
    const clio::run::u64 epp = ElemsPerPage();
    clio::run::u64 got = 0;
    DownloadPages(0, (n + epp - 1) / epp,
                  [&](clio::run::u64 pg, const char *bytes) {
                    const clio::run::u64 off = pg * epp;
                    if (off >= n) return;
                    const clio::run::u64 cnt = (epp < n - off) ? epp : (n - off);
                    std::memcpy(dst + off, bytes,
                                static_cast<size_t>(cnt * sizeof(T)));
                    got += cnt;
                  });
    return got;
  }

  /** Read pages [pg_lo, pg_hi); `sink(pg, bytes)` receives each page. */
  template <typename SinkFn>
  clio::run::u64 DownloadPages(clio::run::u64 pg_lo, clio::run::u64 pg_hi,
                               SinkFn sink) {
    clio::cte::core::Client core(storage_pool_id_);
    std::vector<char> buf(static_cast<size_t>(page_bytes_));
    clio::run::u64 got = 0;
    for (clio::run::u64 pg = pg_lo; pg < pg_hi; ++pg) {
      if (!ReadPage(core, pg, buf.data())) continue;
      sink(pg, static_cast<const char *>(buf.data()));
      ++got;
    }
    return got;
  }

  /** Zero the counters, for benches that measure per-round. */
  void ResetStats() {
#if CTP_ENABLE_GPU
    for (auto &kv : devs_) {
      if (kv.second.stats == nullptr) continue;
      ctp::GpuApi::Memset(kv.second.stats, 0, 5 * sizeof(unsigned long long));
    }
#endif
  }

  /**
   * Copy every frame's metadata to the host.
   *
   * THE DUPLICATION CHECK NEEDS TO SEE THE WHOLE ARRAY AT ONCE. "Is this
   * cache actually shared" is answered by counting how many frames hold the
   * same page_num: one in shared mode, up to one-per-CUDA-block in private
   * mode. That is not visible from any single set.
   */
  std::vector<Page> ReadTable(int gpu_id) const {
    auto it = devs_.find(gpu_id);
    if (it == devs_.end()) return {};
    const clio::run::u64 nslots =
        static_cast<clio::run::u64>(nsets_) * set_size_;
    std::vector<Page> out(static_cast<size_t>(nslots));
    ctp::GpuApi::Memcpy(reinterpret_cast<char *>(out.data()),
                        reinterpret_cast<const char *>(it->second.table_base),
                        static_cast<size_t>(nslots * sizeof(Page)));
    return out;
  }

  /** Occupied frames in the table -- 0 means the cache has been emptied. */
  clio::run::u32 Occupied(int gpu_id) const {
    const std::vector<Page> tbl = ReadTable(gpu_id);
    clio::run::u32 n = 0;
    for (const Page &p : tbl) n += (p.page_num != kNoPage);
    return n;
  }

  /** How many frames hold the most-duplicated resident page. 1 == shared. */
  clio::run::u32 MaxPageCopies(int gpu_id) const {
    const std::vector<Page> tbl = ReadTable(gpu_id);
    std::map<clio::run::u64, clio::run::u32> seen;
    clio::run::u32 worst = 0;
    for (const Page &p : tbl) {
      if (p.page_num == kNoPage) continue;
      const clio::run::u32 n = ++seen[p.page_num];
      if (n > worst) worst = n;
    }
    return worst;
  }

  Stats ReadStats(int gpu_id) const {
    Stats s;
#if CTP_ENABLE_GPU
    auto it = devs_.find(gpu_id);
    if (it == devs_.end() || it->second.stats == nullptr) return s;
    unsigned long long h[17] = {};
    ctp::GpuApi::Memcpy(h, it->second.stats, sizeof(h));
    s.fetch_cyc = h[10];
    s.hold_cyc = h[11];
    s.flush_cyc = h[12];
    s.busy_cyc = h[13];
    s.fetch_calls = h[14];
    s.hold_calls = h[15];
    s.flush_calls = h[16];
    s.faults = h[0];
    s.alloc_waits = h[9];
    s.puts = h[1];
    s.evicts = h[2];
    s.get_errors = h[3];
    s.gen_ok = h[5];
    s.gen_stale = h[6];
    s.gen_busy = h[7];
    s.flush_skipped = h[8];
    s.put_errors = h[4];
#else
    (void) gpu_id;
#endif
    return s;
  }

  // =====================================================================
  // PREFETCHING -- see prefetch.h for what this is and is not.
  //
  // A prefetcher turns "where is this block in its computation" into "which
  // pages should be hot"; this half turns those hints into CTE bulk rescores
  // and keeps them off the driver's critical path. The two are separate
  // because the first is workload knowledge and the second is plumbing that
  // no workload should have to reimplement.
  // =====================================================================

  /**
   * Attach a prefetcher. Several may be registered; their hints merge into
   * one sink for the round and LAST WRITER WINS for a repeated page -- which
   * is what lets a generic stride prefetcher run underneath a
   * workload-specific one as a floor.
   *
   * EXPLICIT, NEVER AUTOMATIC. A prefetcher that fired by default would
   * change the tier placement of every existing benchmark, including the ones
   * whose published numbers assume the frecency organizer is the only thing
   * moving blobs.
   */
  void RegisterPrefetcher(std::shared_ptr<Prefetcher> p) {
    if (p != nullptr) prefetchers_.push_back(std::move(p));
  }

  void SetPrefetchPolicy(const PrefetchPolicy &p) { prefetch_policy_ = p; }
  const PrefetchPolicy &GetPrefetchPolicy() const { return prefetch_policy_; }
  bool HasPrefetchers() const { return !prefetchers_.empty(); }

  /** Names of the registered prefetchers, for the report line. */
  std::string PrefetcherNames() const {
    std::string s;
    for (const auto &p : prefetchers_) {
      if (!s.empty()) s += "+";
      s += p->Name();
    }
    return s.empty() ? std::string("none") : s;
  }

  PrefetchStats ReadPrefetchStats() const { return prefetch_stats_; }
  void ResetPrefetchStats() { prefetch_stats_ = PrefetchStats(); }

  /**
   * Forget what has already been sent.
   *
   * CALL THIS PER RUN, NOT PER PHASE, and in particular NOT when the meaning
   * of a page changes.
   *
   * `last_sent_` is a model of ONE thing: the score the CTE currently holds
   * for a page. That model survives a change of meaning perfectly well -- a
   * page's score is whatever was last set on it, whether the workload now
   * thinks of it as an input or an output. Dedup suppresses a hint only when
   * the DESIRED score equals the last one SENT, which is exactly the case
   * where the CTE would compare the two and do nothing anyway.
   *
   * Clearing it per step (Gray-Scott's region swap) looked like the cautious
   * choice and was a real defect: it threw away accurate state, so every page
   * was re-sent its current score at the top of each step -- a rescore the
   * CTE no-ops -- and the promotion/demotion tally was computed against an
   * assumed prior instead of the known one. That is how a run reported
   * promote=896 against demote=32512 while sending 66048 records.
   *
   * It exists for the start of a RUN, where the vector may have been rebuilt
   * or the tiers reset underneath the map.
   */
  void ResetPrefetchHistory() { last_sent_.clear(); }

  /** Tell every prefetcher a new kernel run is starting. */
  void PrefetchRunBegin() {
    for (auto &p : prefetchers_) p->OnRunBegin();
  }

  /**
   * The observer to install on a yieldable driver:
   *
   *     drv.SetYieldObserver(vec.YieldObserver());
   *
   * Captures `this`; the vector must outlive the driver, which it does in
   * every sane arrangement (the vector is what the kernel is paging against).
   */
  clio::run::gpu::YieldObserverFn YieldObserver() {
    return [this](const clio::run::gpu::YieldBlockState *states,
                  clio::run::u32 nblocks, clio::run::u32 round) {
      OnYieldRound(states, nblocks, round);
    };
  }

  /**
   * Run every prefetcher over one round's block states and submit whatever
   * survives dedup. Called from the driver's post-round gap: no kernel is
   * resident, so the rescores this issues overlap the NEXT round's kernel.
   */
  void OnYieldRound(const clio::run::gpu::YieldBlockState *states,
                    clio::run::u32 nblocks, clio::run::u32 round) {
    if (prefetchers_.empty() || states == nullptr) return;
    // Reap LAST round's submissions first. Never wait on them: a rescore that
    // has not landed yet is still useful, and one that failed is counted, not
    // retried. Waiting here would put tier migration back on the critical
    // path, which is the whole thing this exists to take it off.
    ReapPrefetchFutures(/*block=*/false);

    sink_.Clear();
    for (clio::run::u32 b = 0; b < nblocks; ++b) {
      YieldEvent ev;
      ev.block_ = b;
      ev.round_ = round;
      ev.status_ = states[b].status_;
      ev.resume_point_ = states[b].resume_point_;
      ev.wait_tag_ = states[b].wait_tag_;
      ev.cursor_ = states[b].cursor_;
      ev.cursor_aux_ = states[b].cursor_aux_;
      for (auto &p : prefetchers_) p->OnYield(ev, sink_);
    }
    for (auto &p : prefetchers_) p->OnRoundEnd(round, sink_);

    prefetch_stats_.hints_ += sink_.hints().size();
    prefetch_stats_.dropped_ += sink_.dropped();
    SubmitHints(sink_.hints());
  }

  /**
   * Submit hints directly, without a Prefetcher or the yield observer.
   *
   * For an organizer that runs at a PHASE boundary rather than at a yield: a
   * batch loop that knows the next batch's page set exactly, one batch ahead,
   * has nothing to infer from the coroutine cursor and no reason to wait for a
   * round to end. It just says what it wants moved.
   *
   * Same dedup, batching and non-blocking submission as the yield path, so a
   * page already at the requested score costs a map lookup and nothing else.
   */
  void PrefetchNow(const std::vector<PrefetchHint> &hints) {
    if (hints.empty()) return;
    ReapPrefetchFutures(/*block=*/false);
    prefetch_stats_.hints_ += hints.size();
    SubmitHints(hints);
  }

  /** Wait for every outstanding rescore. For the end of a run, where the
   *  question "did the prefetcher's work actually land" has to be answerable
   *  before the tier occupancy is read. */
  void DrainPrefetch() { ReapPrefetchFutures(/*block=*/true); }

 private:
  /**
   * Dedup, batch, submit.
   *
   * DEDUP IS NOT AN OPTIMIZATION HERE, it is what makes the whole thing
   * affordable. A block parks several times inside one Gray-Scott
   * z-iteration, so without it the same eight promotions are re-sent on every
   * one of those rounds and the CTE pays a task submission to discover each
   * time that the score has not moved. The threshold mirrors the CTE's own
   * score_difference_threshold, which is what it would compare against
   * anyway.
   */
  void SubmitHints(const std::vector<PrefetchHint> &hints) {
    if (hints.empty()) return;
    // Collapse duplicates WITHIN the round first -- several blocks hint the
    // same halo plane, and last writer wins by construction.
    round_merge_.clear();
    for (const auto &h : hints) round_merge_[h.page_] = h.score_;

    auto *core = PrefetchClient();
    if (core == nullptr) return;
    clio::run::shared_ptr<clio::cte::core::PodMultiScoreTask> batch;
    clio::run::u32 in_batch = 0;
    for (const auto &kv : round_merge_) {
      const clio::run::u64 pg = kv.first;
      const float score = kv.second;
      auto seen = last_sent_.find(pg);
      if (seen != last_sent_.end() &&
          std::fabs(seen->second - score) < prefetch_policy_.epsilon_) {
        ++prefetch_stats_.deduped_;
        continue;
      }
      // A page nobody has hinted about yet sits at the score the vector WROTE
      // it at -- kVectorBlobScore -- not at kDefaultScore, which is a frame's
      // eviction rank inside the GPU cache and has nothing to do with the
      // tier the blob is in. Using the latter made every first-time hot hint
      // (1.0 against a phantom prior of 1.0) count as neither a promotion nor
      // a demotion, and the run reported 224 promotions against 3968
      // demotions while visibly filling the HBM tier from empty.
      const float prev = (seen == last_sent_.end()) ? kVectorBlobScore
                                                    : seen->second;
      if (score > prev) {
        ++prefetch_stats_.promotions_;
      } else if (score < prev) {
        ++prefetch_stats_.demotions_;
      }
      last_sent_[pg] = score;

      if (!batch) {
        // Dynamic routing: in a multi-node pool the blob's metadata lives on
        // its hashed owner, and a hint about a neighbour's halo plane is a
        // legitimate thing to emit. (PodMultiScore itself dispatches its
        // per-record PodReorganizeBlob locally, so a cross-node hint is
        // currently a no-op rather than a fault -- which is the right failure
        // mode for a hint, and is why this is not gated on node count.)
        // kCtePrefetchHint: this batch's completion latency is off everyone's
        // critical path -- nothing waits on a rescore, by construction. The
        // flag is what lets the runtime treat it as background work and keep
        // a latency-critical demand fault from queueing behind it.
        batch = core->template NewPodBatch<clio::cte::core::PodMultiScoreTask>(
            tag_id_, PageContext(), clio::cte::core::kCtePrefetchHint,
            clio::run::PoolQuery::Dynamic());
        in_batch = 0;
      }
      // Only blob_name_ and score_ carry meaning for a rescore; size 0 and a
      // null pointer are what the handler expects for the rest.
      batch.get()->Add(PageName(pg).c_str(), 0, 0, ctp::ipc::ShmPtr<>(), score);
      ++in_batch;
      ++prefetch_stats_.sent_;
      if (in_batch >= clio::cte::core::kPodMultiMax) {
        pending_.push_back(core->AsyncPodBatch(batch));
        ++prefetch_stats_.batches_;
        batch = {};
      }
    }
    if (batch && in_batch > 0) {
      pending_.push_back(core->AsyncPodBatch(batch));
      ++prefetch_stats_.batches_;
    }
  }

  /** Retire completed rescore batches. Non-blocking unless asked. */
  void ReapPrefetchFutures(bool block) {
    size_t keep = 0;
    for (size_t i = 0; i < pending_.size(); ++i) {
      auto &f = pending_[i];
      if (f.IsNull()) continue;
      if (block) {
        f.Wait();
      } else if (!f.IsComplete()) {
        // GUARDED, because keep == i on every element until the first one is
        // dropped -- which is the common case, since most rounds retire
        // nothing. `pending_[i] = std::move(pending_[i])` is a self-move, and
        // whether that is harmless depends on Future's move-assignment
        // rather than on anything visible here. Not worth relying on.
        if (keep != i) pending_[keep] = std::move(f);
        ++keep;
        continue;
      }
      // PER-RECORD, and split by reason. A batch's return code is only its
      // FIRST failure, and the failures are not all the same kind of event:
      // rc 3 is "blob not found", which for a hint is entirely expected --
      // the lookahead runs past the end of what has been written, and on the
      // first step of Gray-Scott the two output regions have no blobs at all
      // because the seed writes only u and v. Folding those into an error
      // count made a healthy run report 1008 errors and look broken. A real
      // refusal (a score out of range, a blob the CTE would not move) is a
      // different thing and stays counted as one.
      const clio::run::u32 n =
          std::min(f->count_, clio::cte::core::kPodMultiMax);
      for (clio::run::u32 r = 0; r < n; ++r) {
        const clio::run::u32 rc = f->reqs_[r].rc_;
        if (rc == 0) continue;
        if (rc == 3) {
          ++prefetch_stats_.not_found_;
        } else {
          ++prefetch_stats_.errors_;
        }
      }
    }
    pending_.resize(block ? 0 : keep);
  }

  /** One long-lived client for rescores, created on first use: the vector is
   *  often constructed before a caller decides to prefetch at all. */
  clio::cte::core::Client *PrefetchClient() {
    if (!prefetch_core_) {
      prefetch_core_ =
          std::make_unique<clio::cte::core::Client>(storage_pool_id_);
    }
    return prefetch_core_.get();
  }

  std::vector<std::shared_ptr<Prefetcher>> prefetchers_;
  PrefetchPolicy prefetch_policy_;
  PrefetchStats prefetch_stats_;
  PrefetchSink sink_;
  /** page -> score last actually submitted. The dedup filter. */
  std::map<clio::run::u64, float> last_sent_;
  /** page -> score, within one round. Reused to avoid a per-round alloc. */
  std::map<clio::run::u64, float> round_merge_;
  std::vector<clio::run::Future<clio::cte::core::PodMultiScoreTask>> pending_;
  std::unique_ptr<clio::cte::core::Client> prefetch_core_;
  /** Source tag name when this Vector is a Copy(); empty otherwise. Only
   *  MaterializeAll needs it, and only a copy can be materialised. */
  std::string ckpt_src_;


  struct DevState {
    int gpu_id = 0;
    DeviceVector<T> view;
    VecHeader hdr;
    VecHeader *d_hdr = nullptr;
    char *pages_base = nullptr;    // page bytes
    char *table_base = nullptr;    // Page[]
    char *tasks_base = nullptr;    // the four task objects per block
    char *btbl_base = nullptr;     // BlockTasks[]
    ctp::ipc::AllocatorId pages_alloc;
    ctp::ipc::AllocatorId table_alloc;
    ctp::ipc::AllocatorId tasks_alloc;
    ctp::ipc::AllocatorId btbl_alloc;
    ctp::ipc::AllocatorId locks_alloc;
    ctp::ipc::AllocatorId count_alloc;
    char *locks_base = nullptr;
    char *count_base = nullptr;
    clio::run::u32 nregions = 0;
    clio::run::u32 per_block = 0;
    unsigned long long *stats = nullptr;
  };

  /** Bytes of one block's four tasks, laid out in this order. */
  static constexpr size_t kTaskSetBytes =
      sizeof(MultiPutSlot) + sizeof(MultiGetSlot);

  void BuildDevice(int gpu_id) {
    auto *ipc = CLIO_CPU_IPC;
    DevState st;
    st.gpu_id = gpu_id;
    const clio::run::u64 nslots =
        static_cast<clio::run::u64>(nsets_) * set_size_;
    // REGIONS ARE THE COST; SLOTS ARE TAGS. Round the capacity up so every
    // block owns the same number, which is what makes ownership a divide
    // rather than a table.
    const clio::run::u32 per_block =
        (capacity_pages_ + nblocks_ - 1) / nblocks_;
    const clio::run::u32 nregions = per_block * nblocks_;

    st.pages_alloc = ipc->AllocateAndRegisterGpuBackend(
        gpu_id, clio::run::gpu::IpcManager::MemKind::kDeviceMem,
        static_cast<size_t>(nregions) * page_bytes_, &st.pages_base);
    st.table_alloc = ipc->AllocateAndRegisterGpuBackend(
        gpu_id, clio::run::gpu::IpcManager::MemKind::kDeviceMem,
        nslots * sizeof(Page), &st.table_base);
    st.tasks_alloc = ipc->AllocateAndRegisterGpuBackend(
        gpu_id, clio::run::gpu::IpcManager::MemKind::kDeviceMem,
        nblocks_ * kTaskSetBytes, &st.tasks_base);
    RegisterTaskRegion(st.tasks_base, nblocks_ * kTaskSetBytes);
    st.btbl_alloc = ipc->AllocateAndRegisterGpuBackend(
        gpu_id, clio::run::gpu::IpcManager::MemKind::kDeviceMem,
        nblocks_ * sizeof(BlockTasks), &st.btbl_base);
    if (st.pages_alloc.IsNull() || st.table_alloc.IsNull() ||
        st.tasks_alloc.IsNull() || st.btbl_alloc.IsNull()) {
      throw std::runtime_error("gpu_vector: device backend allocation failed");
    }
    {
      // One lock per set and one per block free list, strided so no two share
      // a cache line, plus the free lists themselves: nregions region indices
      // and a head/tail per block. All of it is small -- four bytes per
      // region against the page it stands for.
      const clio::run::u32 kLockStride = 32;
      const size_t lock_ints =
          static_cast<size_t>(nsets_ + nblocks_) * kLockStride;
      st.locks_alloc = ipc->AllocateAndRegisterGpuBackend(
          gpu_id, clio::run::gpu::IpcManager::MemKind::kDeviceMem,
          lock_ints * sizeof(int), &st.locks_base);
      st.count_alloc = ipc->AllocateAndRegisterGpuBackend(
          gpu_id, clio::run::gpu::IpcManager::MemKind::kDeviceMem,
          (static_cast<size_t>(nregions) + 2 * nblocks_) *
              sizeof(clio::run::u32),
          &st.count_base);
      if (st.locks_alloc.IsNull() || st.count_alloc.IsNull()) {
        throw std::runtime_error("gpu_vector: shared-cache allocation failed");
      }
      ctp::GpuApi::Memset(st.locks_base, 0, lock_ints * sizeof(int));
    }

    InitPageTable(st, nslots);
    InitFreeLists(st, nregions, per_block);
    InitTasks(st);

    st.hdr.pages_ = reinterpret_cast<Page *>(st.table_base);
    st.hdr.tasks_ = reinterpret_cast<BlockTasks *>(st.btbl_base);
    st.hdr.page_bytes_ = page_bytes_;
    st.hdr.elems_per_page_ = page_bytes_ / sizeof(T);
    st.hdr.num_elems_ = num_elems_;
    st.hdr.nblocks_ = nblocks_;
    st.hdr.nsets_ = nsets_;
    st.hdr.set_size_ = set_size_;
    st.hdr.set_locks_ = reinterpret_cast<int *>(st.locks_base);
    // The block free-list locks live after the set locks in the same block.
    st.hdr.free_lock_ =
        reinterpret_cast<int *>(st.locks_base) +
        static_cast<size_t>(nsets_) * 32;
    st.nregions = nregions;
    st.per_block = per_block;
    st.hdr.regions_ = st.pages_base;
    st.hdr.nregions_ = nregions;
    st.hdr.regions_per_block_ = per_block;
    st.hdr.free_q_ = reinterpret_cast<clio::run::u32 *>(st.count_base);
    st.hdr.free_head_ = st.hdr.free_q_ + nregions;
    st.hdr.free_tail_ = st.hdr.free_head_ + nblocks_;
    st.hdr.tag_id_ = tag_id_;
    st.hdr.pool_id_ = storage_pool_id_;
    st.hdr.task_alloc_id_ = st.tasks_alloc;
    st.hdr.compress_lib_ = compress_lib_;
    st.hdr.compress_preset_ = compress_preset_;
    st.hdr.stat_faults_ = nullptr;
    st.hdr.stat_puts_ = nullptr;
    st.hdr.stat_evicts_ = nullptr;
    st.hdr.stat_get_errors_ = nullptr;
    st.hdr.stat_gen_ok_ = nullptr;
    st.hdr.stat_gen_stale_ = nullptr;
    st.hdr.stat_gen_busy_ = nullptr;
    st.hdr.stat_flush_skipped_ = nullptr;
    st.hdr.stat_put_errors_ = nullptr;
    st.hdr.stat_alloc_waits_ = nullptr;
    st.hdr.stat_cyc_ = nullptr;
    st.hdr.fatal_ = FatalSlots();
    st.hdr.fatal_mirror_ = FatalMirror();
    PublishHeader(st);
    devs_[gpu_id] = st;
  }

  /** Build the Page[] on the host -- it holds pointers a kernel cannot
   *  compute -- then upload it once. */
  /**
   * Prefetch into a SHARED cache: one copy of each page, in its home set.
   *
   * Stops at the byte budget rather than at the end of the frame array. The
   * array is 2x over-provisioned for collisions, so filling it would hand the
   * device a cache twice the size it was asked for and leave cache_frames_
   * describing a budget already blown. A page whose home set is full is
   * simply left out -- residency here is best effort, and the device fault
   * path will bring it in.
   */
  void PrefetchShared(DevState &st, clio::run::u64 pg_lo, clio::run::u64 pg_hi,
                      clio::cte::core::Client &core) {
#if CTP_ENABLE_GPU
    const clio::run::u64 nslots =
        static_cast<clio::run::u64>(nsets_) * set_size_;
    std::vector<Page> tbl(static_cast<size_t>(nslots));
    ctp::GpuApi::Memcpy(reinterpret_cast<char *>(tbl.data()), st.table_base,
                        static_cast<size_t>(nslots * sizeof(Page)));
    std::vector<char> buf(static_cast<size_t>(page_bytes_));
    unsigned long long placed = 0;
    for (const auto &p : tbl) {
      if (p.page_num != kNoPage) ++placed;
    }
    const unsigned long long budget = nslots;
    for (clio::run::u64 pg = pg_lo; pg < pg_hi && placed < budget; ++pg) {
      const size_t base =
          static_cast<size_t>(HomeSet(pg)) * set_size_;
      size_t slot = base;
      const size_t end = base + set_size_;
      while (slot < end && tbl[slot].page_num != kNoPage) ++slot;
      if (slot >= end) continue;                 // home set full
      if (!ReadPage(core, pg, buf.data())) continue;
      // A SLOT NEEDS A REGION NOW. Hand them out in order and rebuild each
      // block's free list from what is left, which keeps the host and the
      // device agreeing about who owns what.
      if (placed >= st.nregions) break;
      const clio::run::u32 ridx = static_cast<clio::run::u32>(placed);
      char *region = st.pages_base + static_cast<size_t>(ridx) * page_bytes_;
      ctp::GpuApi::Memcpy(region, buf.data(),
                          static_cast<size_t>(page_bytes_));
      tbl[slot].data = region;
      tbl[slot].page_num = pg;
      tbl[slot].pins = 0;
      tbl[slot].flushing = 0;
      tbl[slot].fetching = 0;
      tbl[slot].score = kDefaultScore;
      tbl[slot].valid_lo = 0;
      tbl[slot].valid_hi = static_cast<clio::run::u32>(page_bytes_ / sizeof(T));
      ++placed;
    }
    ctp::GpuApi::Memcpy(st.table_base,
                        reinterpret_cast<const char *>(tbl.data()),
                        static_cast<size_t>(nslots * sizeof(Page)));
    RebuildFreeLists(st, static_cast<clio::run::u32>(placed));
#else
    (void) st; (void) pg_lo; (void) pg_hi; (void) core;
#endif
  }

  /**
   * Fill every block's free list with the regions it owns.
   *
   * Block b owns [b*per_block, (b+1)*per_block), which is what makes
   * RegionOwner a divide on the address instead of a lookup.
   */
  void InitFreeLists(DevState &st, clio::run::u32 nregions,
                     clio::run::u32 per_block) {
#if CTP_ENABLE_GPU
    std::vector<clio::run::u32> q(nregions);
    for (clio::run::u32 i = 0; i < nregions; ++i) q[i] = i;
    std::vector<clio::run::u32> head(nblocks_, 0), tail(nblocks_, per_block);
    auto *base = reinterpret_cast<clio::run::u32 *>(st.count_base);
    UploadBytes(q.data(), reinterpret_cast<char *>(base),
                q.size() * sizeof(clio::run::u32));
    UploadBytes(head.data(), reinterpret_cast<char *>(base + nregions),
                head.size() * sizeof(clio::run::u32));
    UploadBytes(tail.data(),
                reinterpret_cast<char *>(base + nregions + nblocks_),
                tail.size() * sizeof(clio::run::u32));
#else
    (void) st; (void) nregions; (void) per_block;
#endif
  }

  /**
   * Rebuild every block's free list, given that regions [0, used) are now
   * held by slots. A block keeps the ones it owns that nobody took.
   */
  void RebuildFreeLists(DevState &st, clio::run::u32 used) {
#if CTP_ENABLE_GPU
    if (st.count_base == nullptr || st.per_block == 0) return;
    std::vector<clio::run::u32> q(st.nregions, 0);
    std::vector<clio::run::u32> head(nblocks_, 0), tail(nblocks_, 0);
    for (clio::run::u32 b = 0; b < nblocks_; ++b) {
      const clio::run::u32 lo = b * st.per_block;
      clio::run::u32 n = 0;
      for (clio::run::u32 i = lo; i < lo + st.per_block; ++i) {
        if (i >= used) q[lo + n++] = i;
      }
      head[b] = 0;
      tail[b] = n;
    }
    auto *base = reinterpret_cast<clio::run::u32 *>(st.count_base);
    UploadBytes(q.data(), reinterpret_cast<char *>(base),
                q.size() * sizeof(clio::run::u32));
    UploadBytes(head.data(), reinterpret_cast<char *>(base + st.nregions),
                head.size() * sizeof(clio::run::u32));
    UploadBytes(tail.data(),
                reinterpret_cast<char *>(base + st.nregions + nblocks_),
                tail.size() * sizeof(clio::run::u32));
#else
    (void) st; (void) used;
#endif
  }

  void InitPageTable(DevState &st, clio::run::u64 nslots) {
    std::vector<Page> tbl(static_cast<size_t>(nslots));
    for (clio::run::u64 i = 0; i < nslots; ++i) {
      Page &p = tbl[static_cast<size_t>(i)];
      p.page_num = kNoPage;
      // A SLOT STARTS WITH NO STORAGE. It gets a region when it claims a
      // page and gives it back when it is evicted.
      p.data = nullptr;
      p.score = kDefaultScore;
      p.last_access = 0;
      p.pins = 0;
      p.valid_lo = 0;
      p.valid_hi = 0;
      p.flushing = 0;
      p.fetching = 0;
    }
    UploadBytes(tbl.data(), st.table_base, nslots * sizeof(Page));
  }

  /**
   * Construct each block's four tasks and its BlockTasks record.
   *
   * The tasks must be CONSTRUCTED, not zeroed: each embeds its own
   * RunContext and the runtime derefs it on arrival, so a zeroed slot
   * reaches the worker as "null RunContext" and aborts it. Each is also
   * stamped with its POD size, which RecvIn reads to know how many bytes to
   * copy back off the device without dereferencing the task.
   */
  void InitTasks(DevState &st) {
    const clio::run::PoolQuery local = clio::run::PoolQuery::Dynamic();
    std::vector<char> raw(nblocks_ * kTaskSetBytes, 0);
    std::vector<BlockTasks> btbl(nblocks_);
    for (clio::run::u32 b = 0; b < nblocks_; ++b) {
      char *slot = raw.data() + static_cast<size_t>(b) * kTaskSetBytes;
      char *dev = st.tasks_base + static_cast<size_t>(b) * kTaskSetBytes;

      auto *fl = new (slot) MultiPutSlot(clio::run::CreateTaskId(),
                                         clio::cte::core::kCtePoolId, local,
                                         tag_id_);
      fl->fut_.task_size_ = static_cast<clio::run::u32>(sizeof(MultiPutSlot));

      char *p2 = slot + sizeof(MultiPutSlot);
      auto *ft = new (p2) MultiGetSlot(clio::run::CreateTaskId(),
                                       clio::cte::core::kCtePoolId, local,
                                       tag_id_);
      ft->fut_.task_size_ = static_cast<clio::run::u32>(sizeof(MultiGetSlot));

      BlockTasks &bt = btbl[b];
      std::memset(&bt, 0, sizeof(bt));
      bt.flush = reinterpret_cast<MultiPutSlot *>(dev);
      bt.fetch = reinterpret_cast<MultiGetSlot *>(dev + sizeof(MultiPutSlot));
    }
    UploadBytes(raw.data(), st.tasks_base, nblocks_ * kTaskSetBytes);
    UploadBytes(btbl.data(), st.btbl_base, nblocks_ * sizeof(BlockTasks));
  }

  void PublishHeader(DevState &st) {
#if CTP_ENABLE_GPU
    if (st.d_hdr == nullptr) {
      st.d_hdr = ctp::GpuApi::Malloc<VecHeader>(sizeof(VecHeader));
      if (st.d_hdr == nullptr) {
        throw std::runtime_error("gpu_vector: header allocation failed");
      }
    }
    ctp::GpuApi::Memcpy(st.d_hdr, &st.hdr, sizeof(VecHeader));
    st.view = DeviceVector<T>(st.d_hdr);
#endif
  }

  /** A page's blob name: the page number in decimal. Matches what the
   *  runtime renders for a device request (Context::kBlobNameRawInt32). */
  std::string PageName(clio::run::u64 pg) const { return std::to_string(pg); }

  clio::cte::core::Context PageContext() const {
    clio::cte::core::Context c;
    c.compress_lib_ = compress_lib_;
    c.compress_preset_ = compress_preset_;
    return c;
  }

  /** One page's bytes into `out`; false when the page has no blob yet. */
  bool ReadPage(clio::cte::core::Client &core, clio::run::u64 pg, char *out) {
    auto ctx = PageContext();
    auto f = core.AsyncGetBlob(tag_id_, PageName(pg), 0, page_bytes_, 0u, out,
                               clio::run::PoolQuery::Dynamic(), ctx);
    f.Wait();
    return f.get() != nullptr && f->GetReturnCode() == 0;
  }

  static void UploadBytes(const void *src, void *dst, clio::run::u64 bytes) {
#if CTP_ENABLE_GPU
    ctp::GpuApi::Memcpy(dst, src, static_cast<size_t>(bytes));
#else
    (void) src; (void) dst; (void) bytes;
#endif
  }

  static void Free(DevState &st) {
    auto *ipc = CLIO_CPU_IPC;
    const auto gpu = static_cast<clio::run::u32>(st.gpu_id);
    if (ipc != nullptr) {
      if (!st.pages_alloc.IsNull()) ipc->FreeGpuBackend(gpu, st.pages_alloc);
      if (!st.table_alloc.IsNull()) ipc->FreeGpuBackend(gpu, st.table_alloc);
      if (!st.tasks_alloc.IsNull()) ipc->FreeGpuBackend(gpu, st.tasks_alloc);
      if (!st.btbl_alloc.IsNull()) ipc->FreeGpuBackend(gpu, st.btbl_alloc);
    }
    st.pages_alloc = ctp::ipc::AllocatorId::GetNull();
    st.table_alloc = ctp::ipc::AllocatorId::GetNull();
    st.tasks_alloc = ctp::ipc::AllocatorId::GetNull();
    st.btbl_alloc = ctp::ipc::AllocatorId::GetNull();
#if CTP_ENABLE_GPU
    if (st.stats != nullptr) {
      ctp::GpuApi::Free(st.stats);
      st.stats = nullptr;
    }
    if (st.d_hdr != nullptr) {
      ctp::GpuApi::Free(st.d_hdr);
      st.d_hdr = nullptr;
    }
#endif
  }

  clio::run::PoolId storage_pool_id_;
  int compress_lib_ = 0;
  int compress_preset_ = 1;
  clio::run::u64 page_bytes_ = 0;
  clio::run::u32 nblocks_ = 0;   // CUDA blocks == task sets
  clio::run::u32 nsets_ = 0;     // associative sets in the cache
  clio::run::u32 set_size_ = 0;  // TAGS per set
  clio::run::u32 capacity_pages_ = 0;  // page-sized regions in the allocator
  clio::run::u64 num_elems_ = 0;
  clio::cte::core::TagId tag_id_;
  std::string tag_name_;
  std::vector<int> gpu_ids_;
  std::map<int, DevState> devs_;
};

#endif  // !CTP_IS_DEVICE_PASS

}  // namespace clio::cte::gpu_vector

#endif  // CLIO_CTE_GPU_VECTOR_GPU_VECTOR_H_
