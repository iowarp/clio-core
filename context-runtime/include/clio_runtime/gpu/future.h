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

#ifndef CLIO_RUNTIME_INCLUDE_GPU_FUTURE_H_
#define CLIO_RUNTIME_INCLUDE_GPU_FUTURE_H_

#include "clio_runtime/types.h"
#include "clio_ctp/memory/allocator/allocator.h"

namespace clio::run {
// Forward-declared so gpu::Future::operator clio::run::Future<>() can resolve
// without dragging the full clio::run::Future header into device-pass code.
template <typename TaskT, typename AllocT>
class Future;
namespace gpu {

/**
 * gpu::FutureShm - completion record for a single gpu2cpu task.
 *
 * Lives in a client-owned, admin-registered device-memory backend
 * (pinned host, UVM, or pure device memory). Co-located with its task
 * struct: the task is at `addr - sizeof(TaskT)`. The CPU GPU worker
 * pops a gpu::Future<Task> off gpu2cpu_queue, resolves both ShmPtrs
 * via the registered backend map, copies the POD task into a CPU-side
 * scratch slot, dispatches the task, then writes back the POD output
 * bytes and sets FUTURE_COMPLETE here so the kernel poll-loop unblocks.
 *
 * Only one flag survives the producer-only redesign: FUTURE_COMPLETE.
 * Origin tags, device-scope flags, parent-coroutine pointers, and the
 * cudaMemcpy POD-copy fields all moved to the CPU runtime side or
 * disappeared entirely.
 */
struct FutureShm {
  static constexpr u32 FUTURE_COMPLETE = 1;

  /** sizeof(TaskT) — needed by the CPU worker to D2H-copy the POD task. */
  u32 task_size_;

  /** Atomic completion flag. */
  ctp::abitfield32_t flags_;

  CTP_CROSS_FUN FutureShm() : task_size_(0) { flags_.Clear(); }

  /** Reset for reuse. Called by ClientSend before pushing onto gpu2cpu. */
  CTP_CROSS_FUN void Reset(u32 task_size) {
    task_size_ = task_size;
    flags_.Clear();
  }
};

/**
 * gpu::Future - lightweight handle for a pending gpu2cpu task.
 *
 * Carries ShmPtrs to both the task and its co-located FutureShm so the
 * CPU worker can resolve them via the registered backend map. The kernel
 * just polls FUTURE_COMPLETE; there is no allocator coupling, no cleanup
 * (the host owns the backend), and no coroutine/await machinery.
 */
template <typename TaskT, typename AllocT = CLIO_QUEUE_ALLOC_T>
class Future {
 public:
  using FutureT = FutureShm;

  template <typename OtherTaskT, typename OtherAllocT>
  friend class Future;

 private:
  ctp::ipc::FullPtr<TaskT> task_ptr_;
  ctp::ipc::ShmPtr<FutureT> future_shm_;

 public:
  CTP_CROSS_FUN Future() = default;

  CTP_CROSS_FUN Future(ctp::ipc::ShmPtr<FutureT> future_shm,
                        const ctp::ipc::FullPtr<TaskT> &task_ptr)
      : future_shm_(future_shm) {
    task_ptr_.shm_ = task_ptr.shm_;
    task_ptr_.ptr_ = task_ptr.ptr_;
  }

  CTP_CROSS_FUN explicit Future(const ctp::ipc::ShmPtr<FutureT> &future_shm_ptr)
      : future_shm_(future_shm_ptr) {
    task_ptr_.SetNull();
  }

  CTP_CROSS_FUN Future(const Future &other)
      : future_shm_(other.future_shm_) {
    task_ptr_.shm_ = other.task_ptr_.shm_;
    task_ptr_.ptr_ = other.task_ptr_.ptr_;
  }

  CTP_CROSS_FUN Future &operator=(const Future &other) {
    if (this != &other) {
      task_ptr_.shm_ = other.task_ptr_.shm_;
      task_ptr_.ptr_ = other.task_ptr_.ptr_;
      future_shm_ = other.future_shm_;
    }
    return *this;
  }

  CTP_CROSS_FUN Future(Future &&other) noexcept
      : future_shm_(std::move(other.future_shm_)) {
    task_ptr_.shm_ = other.task_ptr_.shm_;
    task_ptr_.ptr_ = other.task_ptr_.ptr_;
    other.task_ptr_.SetNull();
  }

  CTP_CROSS_FUN Future &operator=(Future &&other) noexcept {
    if (this != &other) {
      task_ptr_.shm_ = other.task_ptr_.shm_;
      task_ptr_.ptr_ = other.task_ptr_.ptr_;
      future_shm_ = std::move(other.future_shm_);
      other.task_ptr_.SetNull();
      other.future_shm_.SetNull();
    }
    return *this;
  }

  CTP_CROSS_FUN ~Future() = default;

  CTP_CROSS_FUN TaskT *get() const { return task_ptr_.ptr_; }
  CTP_CROSS_FUN TaskT &operator*() const { return *task_ptr_.ptr_; }
  CTP_CROSS_FUN TaskT *operator->() const { return task_ptr_.ptr_; }
  CTP_CROSS_FUN bool IsNull() const { return task_ptr_.IsNull(); }

  ctp::ipc::FullPtr<TaskT> &GetTaskPtr() { return task_ptr_; }
  const ctp::ipc::FullPtr<TaskT> &GetTaskPtr() const { return task_ptr_; }
  CTP_CROSS_FUN ctp::ipc::ShmPtr<FutureT> GetFutureShmPtr() const {
    return future_shm_;
  }

  /**
   * Resolve the FutureShm to a raw pointer.
   *
   * Inside the kernel, ShmPtr::off_ holds the device-side address of the
   * FutureShm directly (the host pre-constructed the task+FutureShm pair
   * in a registered backend, so the kernel has the resolved address). On
   * the CPU worker side this is unused — the worker resolves via the
   * registered backend map instead.
   */
  CTP_CROSS_FUN FutureT *GetFutureShmPtrRaw() const {
    if (future_shm_.IsNull()) return nullptr;
    return reinterpret_cast<FutureT *>(future_shm_.off_.load());
  }

  /**
   * Block until the CPU runtime sets FUTURE_COMPLETE on this FutureShm.
   *
   * Single Wait path — no Cpu2Gpu/Gpu2Gpu/Gpu2Cpu variants. The FutureShm
   * lives in pinned host or UVM memory, so a volatile read on the device
   * side sees the CPU's system-scope write through PCIe cache snooping.
   */
  CTP_CROSS_FUN void Wait();

  /**
   * Conversion to clio::run::Future<TaskT> for host return-type compatibility.
   * Always produces an empty clio::run::Future on the host (host-side Send is
   * not GPU-aware in the producer-only model).
   */
  CTP_CROSS_FUN operator clio::run::Future<TaskT, AllocT>() const {
    return clio::run::Future<TaskT, AllocT>();
  }
};

}  // namespace gpu

/** Queue type for the per-device gpu2cpu_queue (stores gpu::Future<Task>). */
using GpuTaskQueue =
    ctp::ipc::multi_mpsc_ring_buffer<gpu::Future<Task>, CLIO_QUEUE_ALLOC_T>;
/** Single lane within a GpuTaskQueue. */
using GpuTaskLane = GpuTaskQueue::ring_buffer_type;

}  // namespace clio::run

#endif  // CLIO_RUNTIME_INCLUDE_GPU_FUTURE_H_
