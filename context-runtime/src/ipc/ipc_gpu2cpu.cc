/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#include <atomic>
#include <exception>
#include <unordered_map>
#include "clio_runtime/ipc/ipc_gpu2cpu.h"
#include "clio_runtime/gpu/gpu_device_ring.h"

#if CTP_ENABLE_CUDA || CTP_ENABLE_ROCM || CTP_ENABLE_SYCL

#include "clio_ctp/util/gpu_api.h"
#include "clio_runtime/gpu/future.h"
#include "clio_runtime/gpu/gpu_ipc_manager.h"
#include "clio_runtime/ipc_manager.h"
#include "clio_runtime/singletons.h"
#include "clio_runtime/worker.h"

namespace clio::run {

/**
 * RecvIn (producer-only gpu2cpu pop): pop one gpu::Future<Task> off `gpu_lane`,
 * D2H-copy the gpu::FutureShm + POD task out of device memory when the kernel
 * allocated in kDeviceMem (the CPU cannot dereference device pointers), wrap the
 * host-resident task in a clio::run::Future<Task>, stash the original device
 * pointers + size on the chi FutureShm (so SendOut can H2D-copy the mutated POD
 * back and flip FUTURE_COMPLETE), then route it. Moved here from the worker so
 * the worker never deserializes tasks/futures. Runs on the worker thread.
 */

// ---------------------------------------------------------------------------
// Lifecycle event log for the per-slot scratch race (issue #961).
//
// A tiny always-on ring of {seq, tid, kind, slot, aux} records, dumped from a
// chained std::terminate handler -- so when "null RunContext" throws anywhere,
// the last events for every slot print without instrumenting task.h. Lock-free
// via a relaxed atomic index; recording costs ~2 atomics per event on a path
// that already does synchronous CUDA copies.
// ---------------------------------------------------------------------------
namespace {
struct EvRec {
  unsigned long long seq;
  unsigned long long slot;   // device task address
  unsigned long long aux;    // task_id unique / rc
  unsigned int tid;          // worker id (or 0xFFFF for unknown)
  char kind;                 // S=stage C=beginctx R=routed P=pod-writeback F=flip
};
constexpr unsigned kEvCap = 4096;
EvRec g_ev[kEvCap];
std::atomic<unsigned long long> g_ev_seq{0};

void EvPush(char kind, unsigned tid, unsigned long long slot,
            unsigned long long aux) {
  const unsigned long long n =
      g_ev_seq.fetch_add(1, std::memory_order_relaxed);
  EvRec &r = g_ev[n % kEvCap];
  r.seq = n; r.slot = slot; r.aux = aux; r.tid = tid; r.kind = kind;
}

std::terminate_handler g_prev_term = nullptr;
void EvDumpOnTerminate() {
  const unsigned long long end = g_ev_seq.load();
  const unsigned long long begin = end > 256 ? end - 256 : 0;
  fprintf(stderr, "==== gpu2cpu evlog (last %llu events) ====\n", end - begin);
  for (unsigned long long n = begin; n < end; ++n) {
    const EvRec &r = g_ev[n % kEvCap];
    if (r.seq != n) continue;   // overwritten mid-dump
    fprintf(stderr, "  #%llu w%u %c slot=%llx aux=%llu\n",
            r.seq, r.tid, r.kind, r.slot, r.aux);
  }
  fprintf(stderr, "==== end evlog ====\n");
  if (g_prev_term) g_prev_term();
  abort();
}
struct EvInit {
  EvInit() { g_prev_term = std::set_terminate(EvDumpOnTerminate); }
} g_ev_init;
}  // namespace

bool IpcGpu2Cpu::RecvIn(IpcManager *ipc, GpuTaskLane *gpu_lane, Worker *worker) {
  gpu::Future<Task> gpu_future;
  // Device-ring path: the queue lives on the GPU, so there is nothing to Pop
  // from -- the mirror hands back one submission from its batched drain. The
  // entry carries exactly what the legacy queue entry did (raw task address,
  // its backend's AllocatorId, and the POD size), so everything below this
  // point is untouched by which transport delivered it.
  auto *gpu_ipc = ipc->GetGpuIpcManager();
  bool from_ring = false;
  if (gpu_ipc != nullptr) {
    GpuRingEntry e;
    if (gpu_ipc->RingNext(0, &e)) {
      ctp::ipc::FullPtr<Task> fp;
      fp.shm_.alloc_id_ = ctp::ipc::AllocatorId{e.alloc_major, e.alloc_minor};
      fp.shm_.off_ = e.task_addr;
      fp.ptr_ = reinterpret_cast<Task *>(e.task_addr);
      gpu_future = gpu::Future<Task>(fp, e.task_size);
      from_ring = true;
    }
  }
  if (!from_ring && !gpu_lane->Pop(gpu_future)) {
    return false;
  }
  const u32 worker_id = worker->GetId();
  HLOG(kDebug, "IpcGpu2Cpu::RecvIn: worker {} popped task from gpu2cpu queue",
       worker_id);

  worker->SetCurrentTask(clio::run::shared_ptr<Task>());

  ctp::ipc::ShmPtr<Task> task_shmptr = gpu_future.GetTaskPtr().shm_;
  if (task_shmptr.IsNull()) {
    HLOG(kError, "IpcGpu2Cpu::RecvIn: worker {} null task ShmPtr in queue entry",
         worker_id);
    return true;
  }

  void *gpu_task_raw = reinterpret_cast<void *>(task_shmptr.off_.load());
  if (!gpu_task_raw) {
    HLOG(kError, "IpcGpu2Cpu::RecvIn: worker {} null task off_ in queue entry",
         worker_id);
    return true;
  }

  // Detect whether the Task struct sits in pure device memory (host cannot
  // dereference it). ctp::IsDevicePointer returns false on host builds.
  bool task_on_device = ctp::IsDevicePointer(gpu_task_raw);

  // The self-contained Task carries its POD size in fut_.task_size_; the kernel
  // cached it in the queue entry so we know the copy size without reading the
  // task first.
  u32 task_pod_size = gpu_future.GetTaskSize();
  if (task_pod_size == 0) {
    HLOG(kError,
         "IpcGpu2Cpu::RecvIn: worker {} queue task_size_=0 — kernel did not "
         "stamp fut_.task_size_ before Send",
         worker_id);
    return true;
  }

  // Host-resident copy of the device task, allocated PER TASK.
  //
  // A shared buffer cannot work here. The copy is wrapped NON-OWNING and
  // handed to a COROUTINE, which parks at its first await (a PutBlob waiting
  // on bdev I/O). The worker then services the next GPU task, so any buffer
  // shared between them is overwritten while the parked coroutine is still
  // executing out of it. A thread_local buffer limited device work to ONE
  // task in flight per worker; a thread_local RING merely moved the limit to
  // its depth (and a depth large enough for 128 blocks x 4 pages blew the
  // thread's TLS and segfaulted).
  //
  // Allocating per CALL fixes the aliasing but creates an ownership problem:
  // nothing can say when the copy is dead. Freeing it in SendOut looks right
  // and is not -- SendOut is NOT the last reference, and doing so aborted with
  // "RunFuture: null RunContext" as soon as slots were reused concurrently.
  //
  // Keying on the DEVICE task address solves both. Distinct slots get distinct
  // buffers, so nothing aliases; and a slot is only re-submitted after its
  // previous completion has been observed by the kernel, so reusing its buffer
  // is safe by the same invariant the producer already relies on. Nothing is
  // freed, and the total is bounded by the task slots the client allocated.
  //
  // The buffer is sized to the TASK, not to a fixed constant. It used to be a
  // flat 4096 bytes, which silently rejected any task bigger than that -- the
  // batched paging tasks (PodMultiPutBlob and friends, ~5.4 KB with 64 records)
  // were dropped here with only a log line, so the kernel waited forever on a
  // completion that was never going to come. Sizing per slot also stops the
  // scalar slots, which need a few hundred bytes, from pinning 4 KB each.
  //
  // kTaskScratchMax is a sanity bound, not a working limit: a task_size_ past
  // it means the stamp is garbage, and copying that many bytes out of device
  // memory would be the actual bug.
  static constexpr size_t kTaskScratchMax = 1 << 20;  // 1 MB
  if (task_pod_size > kTaskScratchMax) {
    HLOG(kError,
         "IpcGpu2Cpu::RecvIn: worker {} task_pod_size {} exceeds max {}",
         worker_id, task_pod_size, kTaskScratchMax);
    return true;
  }
  struct TaskScratch {
    char *buf = nullptr;
    size_t bytes = 0;
  };
  thread_local std::unordered_map<const void *, TaskScratch>
      task_scratch_by_slot;
  TaskScratch &scratch = task_scratch_by_slot[gpu_task_raw];
  // A slot always submits the same task type, so this grows at most once.
  if (scratch.buf != nullptr && scratch.bytes < task_pod_size) {
    // Deliberately not freed: another in-flight coroutine may still be running
    // out of the old buffer (see the ownership note above). Slots are bounded,
    // and this can only happen once per slot.
    scratch.buf = nullptr;
  }
  const size_t kTaskScratchBytes = (task_pod_size > 4096)
                                       ? static_cast<size_t>(task_pod_size)
                                       : 4096;
  char *&slot_buf = scratch.buf;
  if (slot_buf == nullptr) {
    scratch.bytes = kTaskScratchBytes;
    // PINNED, not new[]. cudaMemcpyAsync from PAGEABLE host memory is
    // effectively synchronous -- the driver stages it through an internal
    // pinned buffer -- so SendOut's "async" completion copies degenerated to
    // synchronous ones and measured no faster than the stream-synchronising
    // path they replaced (read 785 -> 744 MB/s, i.e. slightly worse for the
    // extra bookkeeping). Pinning the staging buffer is the prerequisite that
    // makes the async completion path actually asynchronous. These buffers
    // are per device-task-slot and long-lived, so the pinning cost is paid
    // once per slot, not per task.
#if CTP_ENABLE_CUDA
    // Slot scratch must be PINNED (pageable makes every completion copy an
    // internally staged synchronous one -- measured as the difference
    // between a working and a crawling MoE fault pipeline) and must NOT be
    // allocated with cudaHostAlloc here: this code runs on the fault-service
    // path while a faulting kernel is resident, and cudaHostAlloc takes the
    // CUDA context WRITE lock -- the session's recurring deadlock triangle.
    //
    // So the scratch comes from a pinned POOL allocated once, lazily, in
    // 64 MB slabs. The slab allocation itself still calls cudaHostAlloc, but
    // only when the pool is exhausted; the pool is sized so that happens at
    // startup (first slot discoveries) and effectively never mid-decode.
    // Slabs are never freed; slot scratch was already immortal by design.
    struct PinnedPool {
      char *cur = nullptr;
      size_t left = 0;
      char *take(size_t n) {
        n = (n + 255) & ~size_t(255);
        if (n > left) {
          const size_t slab = n > (64u << 20) ? n : (64u << 20);
          void *p = nullptr;
          if (cudaHostAlloc(&p, slab, cudaHostAllocDefault) != cudaSuccess) {
            return nullptr;
          }
          cur = static_cast<char *>(p);
          left = slab;
        }
        char *r = cur;
        cur += n;
        left -= n;
        return r;
      }
    };
    static thread_local PinnedPool pool;
    char *pooled = pool.take(kTaskScratchBytes);
    if (pooled != nullptr) {
      slot_buf = pooled;
    } else {
      slot_buf = new char[kTaskScratchBytes];
    }
#else
    slot_buf = new char[kTaskScratchBytes];
#endif
  }
  char *task_scratch = slot_buf;

  Task *task_raw = nullptr;
  if (task_on_device) {
    ctp::DeviceAwareMemcpy(task_scratch, gpu_task_raw, task_pod_size);
    task_raw = reinterpret_cast<Task *>(task_scratch);
  } else {
    task_raw = static_cast<Task *>(gpu_task_raw);
  }
  EvPush('S', worker_id, reinterpret_cast<unsigned long long>(gpu_task_raw),
         task_raw->task_id_.unique_);

  // Signal completion directly on the (possibly device-resident) Task's
  // embedded flag so the kernel poll-loop unblocks on the error paths below.
  // is_complete_ is fut_'s first member (atomic<u32> whose storage is `.x`),
  // so its device address is gpu_task_raw + (offset of fut_.is_complete_.x).
  auto signal_task_complete = [&]() {
    if (task_on_device) {
      size_t off = reinterpret_cast<char *>(&task_raw->fut_.is_complete_.x) -
                   reinterpret_cast<char *>(task_raw);
      u32 one = 1;
      ctp::DeviceAwareMemcpy(reinterpret_cast<char *>(gpu_task_raw) + off, &one,
                             sizeof(u32));
    } else {
      task_raw->fut_.is_complete_.store(1);
    }
  };

  PoolId pool_id = task_raw->pool_id_;
  u32 method_id = task_raw->method_;

  // task_raw points into a reused worker scratch buffer (or device memory), not
  // a make_shared block — wrap it NON-OWNING so the Future frees nothing.
  Future<Task> future(pool_id, method_id,
                      clio::run::shared_ptr<Task>::WrapNonOwning(task_raw));
  if (future.GetFutureShmPtr().IsNull()) {
    HLOG(kError,
         "IpcGpu2Cpu::RecvIn: worker {} Future construction failed (pool={}, "
         "method={})",
         worker_id, pool_id, method_id);
    signal_task_complete();

    return true;
  }

  auto chi_fshm = future.GetFutureShm();
  chi_fshm->origin_ = ClientOrigin::kClientGpu2Cpu;
  // Stash the device-side task pointer + size so SendOut can H2D-copy the
  // mutated POD back and flip the task's completion flag (cudaMemcpy when in
  // kDeviceMem). The Task is its own completion record — no gpu::FutureShm.
  chi_fshm->gpu_task_device_ptr_ =
      task_on_device ? reinterpret_cast<uintptr_t>(gpu_task_raw) : 0;
  chi_fshm->gpu_task_size_ = task_pod_size;

  auto *pool_manager = CLIO_POOL_MANAGER;
  auto container = pool_manager->GetStaticContainer(pool_id).get();
  if (!container) {
    HLOG(kError,
         "IpcGpu2Cpu::RecvIn: worker {} Container not found (pool={}, method={})",
         worker_id, pool_id, method_id);
    task_raw->SetReturnCode(1);
    signal_task_complete();
    return true;
  }

  // No post-copy fixup: tasks that cross this boundary must be bitwise
  // relocatable (fully-POD, e.g. the Pod*Blob tasks) — the D2H copy above is
  // already a correct host-side task.

  // Allocate the task's RunContext (and resolve its container) now that it is
  // deserialized, so RouteTask / the worker have an active RunContext.
  future.GetTaskPtr()->BeginRunContext();
  EvPush('C', worker_id, reinterpret_cast<unsigned long long>(gpu_task_raw),
         task_raw->task_id_.unique_);

  RouteResult route_result = ipc->RouteTask(future, /*force_enqueue=*/true);
  HLOG(kDebug,
       "IpcGpu2Cpu::RecvIn: worker {} RouteTask returned {} pool={} method={}",
       worker_id, (int)route_result, pool_id, method_id);
  return true;
}

/**
 * RecvIn (legacy copy-space overload): producer-only — the GPU never serializes
 * a task through lightbeam, and the gpu2cpu-pop RecvIn above already wrapped the
 * popped task pointer in a clio::run::Future<Task>. We just hand it back.
 */
clio::run::shared_ptr<Task> IpcGpu2Cpu::RecvIn(
    IpcManager *ipc, Future<Task> &future,
    u32 method_id, ctp::lbm::Transport *recv_transport) {
  (void)ipc; (void)method_id; (void)recv_transport;
  return future.GetTaskPtr();
}

/**
 * SendOut: writes the (mutated) POD task bytes back to the original device
 * address (when the kernel allocated in kDeviceMem) and flips the task's
 * embedded completion flag (task->fut_.is_complete_) so the kernel poll-loop
 * unblocks. There is no separate gpu::FutureShm — the Task is its own record.
 *
 * For kPinnedHost / kManagedUvm the host scratch copy IS the authoritative
 * storage (CPU and GPU share the address), so no writeback is needed and we
 * just mark complete in place. For kDeviceMem we cudaMemcpy the POD payload,
 * then a separate 4-byte cudaMemcpy of is_complete_=1 (ordered AFTER the POD,
 * so the kernel never sees completion before the outputs are written) — a
 * single aligned 32-bit write is observed whole by the device's volatile read.
 */
void IpcGpu2Cpu::SendOut(
    IpcManager *ipc, const clio::run::shared_ptr<Task> &task_ptr) {
  auto future_shm = task_ptr->RunFuture().GetFutureShm();
  HLOG(kDebug, "IpcGpu2Cpu::SendOut: pool={} method={}",
       task_ptr->pool_id_, task_ptr->method_);
  Task *host_task = task_ptr.get();
  const bool device_task =
      future_shm->gpu_task_device_ptr_ != 0 && future_shm->gpu_task_size_ != 0;

  // ASYNC COMPLETION PATH.
  //
  // DeviceAwareMemcpy ends in cudaStreamSynchronize and opens with two
  // cudaPointerGetAttributes queries, and SendOut calls it TWICE per task --
  // so every completion cost two full stream syncs and four attribute queries
  // while the worker sat blocked. Issuing both copies async on the ring's
  // dedicated stream lets the driver pipeline them and returns the worker to
  // its lane immediately.
  //
  // The ordering invariant survives BY CONSTRUCTION rather than by blocking:
  // the two copies go to the SAME stream in the required order, so the flag
  // cannot land before the payload. That flag is the kernel's release signal
  // -- the moment it flips the producer may resubmit the slot -- and stream
  // order is exactly the guarantee that makes deferring it safe.
  //
  // The host staging buffer stays valid across the async copies: it is keyed
  // by device task address and only reused when the kernel resubmits that
  // slot, which it can only do after observing the flag, which the stream
  // orders after the payload copy has completed.
  void *ring_stream = nullptr;
  {
    auto *gpu_ipc = ipc ? ipc->GetGpuIpcManager() : nullptr;
    if (gpu_ipc != nullptr) ring_stream = gpu_ipc->GetRingStream(0);
  }
  // OFF by default: MEASURED SLOWER, twice, for a structural reason.
  //
  // The completion flag is the kernel's release signal, and a demand fault is
  // a BLOCKING round trip -- the kernel spins on that flag and can do nothing
  // until it lands. Deferring the write onto a stream (which also carries
  // RingNext's drain copies and their cudaStreamSynchronize) delays precisely
  // what the GPU is waiting for:
  //     sync completions   792 MB/s     async   672 MB/s   (read, pinned)
  //     sync completions  6119 MB/s     async  5746 MB/s   (flush, 8 blocks)
  // Pinning the staging buffer first (so cudaMemcpyAsync is genuinely async
  // rather than internally staged) did not rescue it -- that removed the
  // confound, and the result held.
  //
  // Asynchrony pays when nothing is blocked on the result. Here something
  // always is. It is kept behind this flag because it WOULD pay for a
  // fire-and-forget completion class (e.g. rescore hints, which no one waits
  // on) -- that is the shape worth revisiting, not this one.
  static const bool async_complete = [] {
    const char *e = std::getenv("CLIO_GPU_ASYNC_COMPLETE");
    return e != nullptr && !(std::string(e) == "0" || std::string(e) == "false");
  }();
  const bool use_async = device_task && async_complete && ring_stream != nullptr;

  if (device_task) {
    EvPush('P', 0xFFFF, future_shm->gpu_task_device_ptr_,
           host_task->task_id_.unique_);
  }
  if (device_task && !use_async) {
    // kDeviceMem: write back the mutated POD. is_complete_ is still 0 here on
    // purpose -- the flag is flipped at the END of this function, once nothing
    // else will touch the task (see below).
    ctp::DeviceAwareMemcpy(
        reinterpret_cast<void *>(future_shm->gpu_task_device_ptr_), host_task,
        future_shm->gpu_task_size_);
  } else if (use_async) {
    ctp::GpuApi::MemcpyAsync(
        reinterpret_cast<char *>(future_shm->gpu_task_device_ptr_),
        reinterpret_cast<const char *>(host_task),
        future_shm->gpu_task_size_, ring_stream);
  }

  // Mark complete: for kPinnedHost / kManagedUvm this storage is shared with
  // the device (the kernel's volatile poll sees it); also wakes host waiters.
  host_task->SetComplete();

  // Producer-only model: the client owns the device-memory backend that holds
  // the task — the runtime does not free it.
  task_ptr->ClearFlags(TASK_DATA_OWNER);

  // The device completion flag goes LAST, once nothing here will touch this
  // task again. It is the kernel's release signal: the instant it flips, the
  // producer may re-submit that page's slot, and any work still pending in
  // this function would race the new submission.
  if (device_task) {
    size_t complete_off =
        reinterpret_cast<char *>(&host_task->fut_.is_complete_.x) -
        reinterpret_cast<char *>(host_task);
    if (use_async) {
      // The flag byte is read from the STAGING task, which already holds 1
      // after SetComplete() above -- so this needs no separate host source
      // whose lifetime we would have to manage past this return.
      ctp::GpuApi::MemcpyAsync(
          reinterpret_cast<char *>(future_shm->gpu_task_device_ptr_) +
              complete_off,
          reinterpret_cast<const char *>(host_task) + complete_off,
          sizeof(u32), ring_stream);
    } else {
      u32 one = 1;
      ctp::DeviceAwareMemcpy(
          reinterpret_cast<char *>(future_shm->gpu_task_device_ptr_) +
              complete_off,
          &one, sizeof(u32));
      EvPush('F', 0xFFFF, future_shm->gpu_task_device_ptr_,
             host_task->task_id_.unique_);
    }
  }
  (void)ipc;
}

}  // namespace clio::run

#endif  // CTP_ENABLE_CUDA || CTP_ENABLE_ROCM || CTP_ENABLE_SYCL
