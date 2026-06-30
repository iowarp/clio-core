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
 * gpu::Future - lightweight handle for a pending gpu2cpu task.
 *
 * The Task is now self-contained (issue "Simplify Future"): its embedded
 * FutureInfo (task->fut_) carries the completion flag (is_complete_) and the
 * POD size (task_size_), so there is no longer a co-located gpu::FutureShm.
 * This handle just carries a FullPtr to the task plus a cached task_size_ (so
 * the CPU GPU-worker knows how many POD bytes to D2H-copy without first
 * reading the task). The kernel polls task->fut_.is_complete_ directly; no
 * allocator coupling, no cleanup (the host owns the backend), no coroutine
 * machinery.
 */
template <typename TaskT, typename AllocT = CLIO_QUEUE_ALLOC_T>
class Future {
 public:
  template <typename OtherTaskT, typename OtherAllocT>
  friend class Future;

 private:
  ctp::ipc::FullPtr<TaskT> task_ptr_;
  u32 task_size_;  ///< sizeof(TaskT); CPU worker uses it to D2H-copy the POD

 public:
  CTP_CROSS_FUN Future() : task_size_(0) {}

  CTP_CROSS_FUN Future(const ctp::ipc::FullPtr<TaskT> &task_ptr, u32 task_size)
      : task_size_(task_size) {
    task_ptr_.shm_ = task_ptr.shm_;
    task_ptr_.ptr_ = task_ptr.ptr_;
  }

  CTP_CROSS_FUN Future(const Future &other) : task_size_(other.task_size_) {
    task_ptr_.shm_ = other.task_ptr_.shm_;
    task_ptr_.ptr_ = other.task_ptr_.ptr_;
  }

  CTP_CROSS_FUN Future &operator=(const Future &other) {
    if (this != &other) {
      task_ptr_.shm_ = other.task_ptr_.shm_;
      task_ptr_.ptr_ = other.task_ptr_.ptr_;
      task_size_ = other.task_size_;
    }
    return *this;
  }

  CTP_CROSS_FUN Future(Future &&other) noexcept : task_size_(other.task_size_) {
    task_ptr_.shm_ = other.task_ptr_.shm_;
    task_ptr_.ptr_ = other.task_ptr_.ptr_;
    other.task_ptr_.SetNull();
  }

  CTP_CROSS_FUN Future &operator=(Future &&other) noexcept {
    if (this != &other) {
      task_ptr_.shm_ = other.task_ptr_.shm_;
      task_ptr_.ptr_ = other.task_ptr_.ptr_;
      task_size_ = other.task_size_;
      other.task_ptr_.SetNull();
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
  CTP_CROSS_FUN u32 GetTaskSize() const { return task_size_; }

  /**
   * Block until the CPU runtime marks the task complete.
   *
   * Polls task->fut_.is_complete_ (in the task POD, which lives in pinned host
   * or UVM memory), so a volatile read on the device sees the CPU's
   * system-scope write through PCIe cache snooping.
   */
  CTP_CROSS_FUN void Wait();

  /**
   * Conversion to clio::run::Future<TaskT> for host return-type compatibility.
   * Always produces an empty clio::run::Future on the host (host-side Send is
   * not GPU-aware in the producer-only model).
   */
  CTP_HOST_FUN operator clio::run::Future<TaskT, AllocT>() const {
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
