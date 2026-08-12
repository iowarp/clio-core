/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#ifndef CLIO_RUNTIME_INCLUDE_IPC_GPU2CPU_IMPL_H_
#define CLIO_RUNTIME_INCLUDE_IPC_GPU2CPU_IMPL_H_

#include "clio_runtime/ipc/ipc_gpu2cpu.h"
#include "clio_runtime/gpu/gpu_device_ring.h"
#include "clio_runtime/gpu/submit_probe.h" 
#include "clio_ctp/util/gpu_intrinsics.h"

#if CTP_ENABLE_CUDA || CTP_ENABLE_ROCM || CTP_ENABLE_SYCL

namespace clio::run {

#if CTP_IS_GPU_COMPILER || CTP_IS_SYCL_COMPILER
/**
 * GPU-side SendIn.
 *
 * Producer-only design: the host pre-allocated the (self-contained) Task in a
 * registered backend and passed `task_ptr` to the kernel, which already
 * mutated the POD input fields. We stamp the task's size + clear its
 * completion flag in the embedded FutureInfo (task->fut_), build a
 * gpu::Future<TaskT>, and push a base-Task handle onto gpu2cpu_queue with a
 * system fence. There is no separate gpu::FutureShm — the Task is its own
 * completion record.
 *
 * Threading:
 *   - CUDA/ROCm: only thread 0 of the block enqueues; other threads
 *     return an empty future (caller is expected to broadcast).
 *   - SYCL: kernels are single_task by convention, so the full WI runs.
 */
template <bool Probing, typename TaskT>
CTP_GPU_FUN gpu::Future<TaskT> IpcGpu2Cpu::SendIn(
    gpu::IpcManager *ipc, const ctp::ipc::FullPtr<TaskT> &task_ptr) {
  gpu::Future<TaskT> future;

  // Any thread may enqueue. This used to be `if (threadIdx.x != 0) return
  // future;` -- a silent no-op for every thread but 0, from a "caller
  // broadcasts" design the gpu_vector never followed. The vector serialises
  // submissions under a per-block lock, and the winning lane is whichever
  // thread got the lock: with one warp the hardware happens to resolve the
  // contention to lane 0 (so every test passed), with several warps the
  // winner is usually NOT threadIdx.x==0 and its page get was silently
  // discarded -- Wait() returned on an empty future, the stale return code
  // read 0, and the slot served its PREVIOUS page's bytes as the new page.
  // Measured: 96 of 128 pages corrupt at 256 threads, zero at 32. The queue
  // push below is a multi-producer atomic claim, so concurrent senders from
  // any thread are safe; callers that want one submission per block must
  // elect a sender themselves, which every caller in the tree already does.

  if (task_ptr.IsNull() || !ipc->gpu_info_.gpu2cpu_queue) {
    return future;
  }

  // The body is split by Probing so the non-probing instantiation carries NONE
  // of the probe locals (prec + the 5 p_*_cy/ns stamps) in its IR — the runtime
  // `if (prec)` guards below cannot free those registers because both branches
  // are reachable at compile time, so we drop them structurally instead. The
  // Probing==true branch is byte-identical to the original body (every existing
  // caller defaults to it), so armed behavior is unchanged.
  if constexpr (Probing) {
    // Submit-path probe (off unless armed): the caller parked its record slot on
    // the block-shared IpcManager. Every stamp below is held in a REGISTER and the
    // record is written once, after the Push — so the probe's own global stores land
    // outside the region it is timing and cannot inflate it.
#if CTP_IS_GPU_COMPILER
    gpu::SubmitProbeRec *prec =
        gpu::ProbeRec(ipc->gpu_info_.probe_, ipc->probe_slot_);
    unsigned long long p_enter_ns = 0, p_enter_cy = 0, p_fields_cy = 0,
                       p_prefence_cy = 0, p_postfence_cy = 0;
    if (prec) {
      // globaltimer first, clock64 last: the cycle counter defines hop 1's left
      // edge, so the (slower) special-register read of globaltimer must not sit
      // between it and the field stores.
      p_enter_ns = gpu::ProbeNowNs();
      p_enter_cy = gpu::ProbeCycles();
    }
#endif

    // Self-contained Task: stamp its POD size + clear its completion flag in the
    // embedded FutureInfo (no co-located gpu::FutureShm).
    const u32 task_size = static_cast<u32>(sizeof(TaskT));
    task_ptr.ptr_->fut_.task_size_ = task_size;
    task_ptr.ptr_->fut_.is_complete_.store(0);

#if CTP_IS_GPU_COMPILER
    // Hop 1 boundary: the task-field stores are done; the fence has not run yet.
    if (prec) p_fields_cy = gpu::ProbeCycles();
#endif

    future = gpu::Future<TaskT>(task_ptr, task_size);

    // Queue entry uses the base Task type with raw addressing so the CPU worker
    // can dereference the task pointer directly; task_size rides along so it does
    // not need to read the task first.
    ctp::ipc::FullPtr<Task> task_for_queue;
    task_for_queue.shm_.alloc_id_ = task_ptr.shm_.alloc_id_;
    task_for_queue.shm_.off_ = reinterpret_cast<size_t>(task_ptr.ptr_);
    task_for_queue.ptr_ = static_cast<Task *>(task_ptr.ptr_);
    gpu::Future<Task> task_future(task_for_queue, task_size);

#if CTP_IS_GPU_COMPILER
    // Everything from here to c_pushed is the true price of GPU initiation: a
    // PCIe-visible system-scope flush plus one lock-free tail bump.
    if (prec) p_prefence_cy = gpu::ProbeCycles();
#endif

    CTP_DEVICE_FENCE_SYSTEM();

#if CTP_IS_GPU_COMPILER
    // Splits the fence from the push. The design claims the fence is what GPU
    // initiation costs; the ring lane lives in PINNED HOST memory, so the push's own
    // atomics cross PCIe too. These must be separated or the wrong one gets blamed.
    if (prec) p_postfence_cy = gpu::ProbeCycles();
#endif

    // DEVICE RING (CLIO_GPU_DEVRING=1): push with a device-scope atomic into
    // device memory. No system-scoped atomic per submission -- the CPU picks
    // these up in batches. Falls through to the legacy managed queue when the
    // ring was not allocated. (The probe still records: hop 3 then measures a
    // device-memory push instead of the pinned-host lane, which is exactly
    // the comparison the ring exists to make.)
    if (ipc->gpu_info_.gpu2cpu_ring != nullptr) {
      GpuRingEntry e;
      e.task_addr = reinterpret_cast<u64>(task_ptr.ptr_);
      e.alloc_major = task_ptr.shm_.alloc_id_.major_;
      e.alloc_minor = task_ptr.shm_.alloc_id_.minor_;
      e.task_size = task_size;
      ipc->gpu_info_.gpu2cpu_ring->Push(e);
    } else {
      auto &qlane = ipc->gpu_info_.gpu2cpu_queue->GetLane(0, 0);
      // Push RETURNS FALSE when the lane is full, and dropping the task there
      // is unrecoverable: the caller is about to wait on a completion flag
      // nothing will ever set. Measured with a demand-paged vector: 128
      // blocks x 1 page passed while 128 x 2 hung -- a burst overrunning the
      // lane, not a total-work limit. Spinning until it lands is also what
      // makes this safe under EITHER lane flag (WAIT_FOR_SPACE today, where
      // the loop never iterates, or ERROR_ON_NO_SPACE), which retires the
      // TODO(ring) that used to sit here.
      while (!qlane.Push(task_future)) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        __nanosleep(64);
#endif
      }
    }

#if CTP_IS_GPU_COMPILER
    if (prec) {
      // Close the measured region on the cycle counter FIRST, then read the wall
      // clock, then commit — so neither the globaltimer read nor these stores are
      // charged to the push.
      const unsigned long long p_pushed_cy = gpu::ProbeCycles();
      const unsigned long long p_pushed_ns = gpu::ProbeNowNs();
      prec->task_ptr = reinterpret_cast<unsigned long long>(task_ptr.ptr_);
      prec->d_enter = p_enter_ns;
      prec->d_pushed = p_pushed_ns;
      prec->c_enter = p_enter_cy;
      prec->c_fields = p_fields_cy;
      prec->c_prefence = p_prefence_cy;
      prec->c_postfence = p_postfence_cy;
      prec->c_pushed = p_pushed_cy;
    }
#endif
  } else {
    // Non-probing instantiation: the exact same submit sequence with every probe
    // line removed, so this instantiation reserves no probe registers at all.
    const u32 task_size = static_cast<u32>(sizeof(TaskT));
    task_ptr.ptr_->fut_.task_size_ = task_size;
    task_ptr.ptr_->fut_.is_complete_.store(0);

    future = gpu::Future<TaskT>(task_ptr, task_size);

    ctp::ipc::FullPtr<Task> task_for_queue;
    task_for_queue.shm_.alloc_id_ = task_ptr.shm_.alloc_id_;
    task_for_queue.shm_.off_ = reinterpret_cast<size_t>(task_ptr.ptr_);
    task_for_queue.ptr_ = static_cast<Task *>(task_ptr.ptr_);
    gpu::Future<Task> task_future(task_for_queue, task_size);

    CTP_DEVICE_FENCE_SYSTEM();

    if (ipc->gpu_info_.gpu2cpu_ring != nullptr) {
      GpuRingEntry e;
      e.task_addr = reinterpret_cast<u64>(task_ptr.ptr_);
      e.alloc_major = task_ptr.shm_.alloc_id_.major_;
      e.alloc_minor = task_ptr.shm_.alloc_id_.minor_;
      e.task_size = task_size;
      ipc->gpu_info_.gpu2cpu_ring->Push(e);
    } else {
      auto &qlane = ipc->gpu_info_.gpu2cpu_queue->GetLane(0, 0);
      // See the probing branch: a dropped Push hangs the waiter forever.
      while (!qlane.Push(task_future)) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        __nanosleep(64);
#endif
      }
    }
  }
  return future;
}
#endif  // CTP_IS_GPU_COMPILER || CTP_IS_SYCL_COMPILER

#if CTP_IS_GPU_COMPILER || CTP_IS_SYCL_COMPILER
/**
 * GPU-side Wait.
 *
 * Polls the task's embedded completion flag (task->fut_.is_complete_) via a
 * volatile read. Backend is pinned host or UVM, so the CPU GPU-worker's
 * system-scope write is visible to a device-side volatile read through PCIe
 * cache snooping. is_complete_ is a ctp::ipc::atomic<u32> whose storage is its
 * first member `.x`.
 */
template <typename TaskT, typename AllocT>
CTP_CROSS_FUN void gpu::Future<TaskT, AllocT>::Wait() {
#if CTP_IS_GPU || CTP_IS_SYCL_DEVICE
  // Any thread may wait. This was `if (threadIdx.x != 0) return;` -- the same
  // broadcast-era guard as SendIn's, and the second half of the same bug: once
  // sends were allowed from any lane, a non-zero faulting lane would enqueue
  // the get and then "wait" by returning INSTANTLY, reading the page while the
  // CPU was still filling it. Symptom moved from deterministic stale pages to
  // racy ones. The poll and the system fence below are per-thread safe.
  if (task_ptr_.IsNull()) return;
  volatile unsigned int *fp = reinterpret_cast<volatile unsigned int *>(
      &task_ptr_.ptr_->fut_.is_complete_.x);
  while (((*fp) & 1u) == 0u) {}
  CTP_DEVICE_FENCE_SYSTEM();
#endif
}
#endif  // CTP_IS_GPU_COMPILER || CTP_IS_SYCL_COMPILER

}  // namespace clio::run

#endif  // CTP_ENABLE_CUDA || CTP_ENABLE_ROCM || CTP_ENABLE_SYCL
#endif  // CLIO_RUNTIME_INCLUDE_IPC_GPU2CPU_IMPL_H_
