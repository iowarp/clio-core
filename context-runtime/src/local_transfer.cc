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

/**
 * LocalTransfer implementation
 */

#include "chimaera/local_transfer.h"

#include <cstring>

#include "chimaera/container.h"

namespace chi {

LocalTransfer::LocalTransfer(std::vector<char>&& data,
                             hipc::FullPtr<FutureShm> future_shm,
                             hipc::FullPtr<Task> task_ptr, u32 method_id,
                             Container* container)
    : data_(std::move(data)),
      future_shm_(future_shm),
      task_ptr_(task_ptr),
      method_id_(method_id),
      container_(container),
      bytes_transferred_(0),
      total_size_(data_.size()),
      is_sender_(true),
      is_initialized_(true) {
  // Set the output total_size_ in FutureShm so receiver knows total size
  if (!future_shm_.IsNull()) {
    future_shm_->output_.total_written_.store(total_size_, std::memory_order_release);
  }
}

LocalTransfer::LocalTransfer(hipc::FullPtr<FutureShm> future_shm,
                             size_t total_size)
    : future_shm_(future_shm),
      bytes_transferred_(0),
      total_size_(total_size),
      is_sender_(false),
      is_initialized_(true) {
  // Pre-allocate buffer for receiving data
  data_.reserve(total_size);
}

LocalTransfer::LocalTransfer(LocalTransfer&& other) noexcept
    : data_(std::move(other.data_)),
      future_shm_(other.future_shm_),
      task_ptr_(other.task_ptr_),
      method_id_(other.method_id_),
      container_(other.container_),
      bytes_transferred_(other.bytes_transferred_),
      total_size_(other.total_size_),
      is_sender_(other.is_sender_),
      is_initialized_(other.is_initialized_) {
  other.future_shm_.SetNull();
  other.task_ptr_.SetNull();
  other.method_id_ = 0;
  other.container_ = nullptr;
  other.bytes_transferred_ = 0;
  other.total_size_ = 0;
  other.is_initialized_ = false;
}

LocalTransfer& LocalTransfer::operator=(LocalTransfer&& other) noexcept {
  if (this != &other) {
    data_ = std::move(other.data_);
    future_shm_ = other.future_shm_;
    task_ptr_ = other.task_ptr_;
    method_id_ = other.method_id_;
    container_ = other.container_;
    bytes_transferred_ = other.bytes_transferred_;
    total_size_ = other.total_size_;
    is_sender_ = other.is_sender_;
    is_initialized_ = other.is_initialized_;

    other.future_shm_.SetNull();
    other.task_ptr_.SetNull();
    other.method_id_ = 0;
    other.container_ = nullptr;
    other.bytes_transferred_ = 0;
    other.total_size_ = 0;
    other.is_initialized_ = false;
  }
  return *this;
}

bool LocalTransfer::Send(u32 max_xfer_time_us) {
  if (!is_initialized_ || !is_sender_ || future_shm_.IsNull()) {
    HLOG(kError,
         "LocalTransfer::Send: Invalid state - initialized={}, "
         "is_sender={}, future_shm_null={}",
         is_initialized_, is_sender_, future_shm_.IsNull());
    return false;
  }

  if (bytes_transferred_ >= total_size_) {
    // All data already transferred - ensure FUTURE_COMPLETE is set
    // This handles the case where total_size_ == 0 (no output data)
    SetComplete();
    return true;
  }

  // Get copy space capacity
  size_t capacity = future_shm_->output_.copy_space_size_;
  if (capacity == 0) {
    HLOG(kError, "LocalTransfer::Send: copy_space capacity is 0");
    return false;
  }

  // Start time tracking
  hshm::Timepoint start_time;
  start_time.Now();
  double max_time_us = static_cast<double>(max_xfer_time_us);

  while (bytes_transferred_ < total_size_) {
    // Check time budget
    double elapsed_us = start_time.GetUsecFromStart();
    if (elapsed_us > max_time_us) {
      HLOG(kDebug,
           "LocalTransfer::Send: Time budget exhausted ({:.1f} us), "
           "transferred {}/{} bytes",
           elapsed_us, bytes_transferred_, total_size_);
      return false;
    }

    // Check if FUTURE_NEW_DATA is still set (client hasn't consumed yet)
    if (future_shm_->flags_.Any(FutureShm::FUTURE_NEW_DATA)) {
      // Wait briefly for client to consume (up to 100us per attempt)
      hshm::Timepoint wait_start;
      wait_start.Now();
      bool consumed = false;

      while (wait_start.GetUsecFromStart() < 100.0) {
        if (!future_shm_->flags_.Any(FutureShm::FUTURE_NEW_DATA)) {
          consumed = true;
          break;
        }
        HSHM_THREAD_MODEL->Yield();
      }

      if (!consumed) {
        // Client not consuming fast enough - return to allow other work
        HLOG(kDebug,
             "LocalTransfer::Send: Client not consuming, "
             "transferred {}/{} bytes",
             bytes_transferred_, total_size_);
        return false;
      }
    }

    // Calculate chunk size
    size_t remaining = total_size_ - bytes_transferred_;
    size_t chunk_size = std::min(remaining, capacity);

    // Copy data to copy_space
    std::memcpy(future_shm_->copy_space, data_.data() + bytes_transferred_,
                chunk_size);

    // Memory fence: Ensure copy_space writes are visible before flag
    std::atomic_thread_fence(std::memory_order_release);

    // Set FUTURE_NEW_DATA flag to signal chunk is ready
    future_shm_->flags_.SetBits(FutureShm::FUTURE_NEW_DATA);

    bytes_transferred_ += chunk_size;
  }

  // All data sent - mark complete
  SetComplete();
  return true;
}

bool LocalTransfer::Recv() {
  if (!is_initialized_ || is_sender_ || future_shm_.IsNull()) {
    HLOG(kError,
         "LocalTransfer::Recv: Invalid state - initialized={}, "
         "is_sender={}, future_shm_null={}",
         is_initialized_, is_sender_, future_shm_.IsNull());
    return false;
  }

  if (bytes_transferred_ >= total_size_) {
    return true;
  }

  // Get copy space capacity
  size_t capacity = future_shm_->output_.copy_space_size_;
  if (capacity == 0) {
    HLOG(kError, "LocalTransfer::Recv: copy_space capacity is 0");
    return false;
  }

  while (bytes_transferred_ < total_size_) {
    // Wait for FUTURE_NEW_DATA to be set
    while (!future_shm_->flags_.Any(FutureShm::FUTURE_NEW_DATA)) {
      HSHM_THREAD_MODEL->Yield();
    }

    // Memory fence: Ensure we see all worker writes to copy_space
    std::atomic_thread_fence(std::memory_order_acquire);

    // Compute chunk size mathematically
    size_t remaining = total_size_ - bytes_transferred_;
    size_t chunk_size = std::min(remaining, capacity);

    // Copy data from copy_space to our buffer
    data_.insert(data_.end(), future_shm_->copy_space,
                 future_shm_->copy_space + chunk_size);

    bytes_transferred_ += chunk_size;

    // Memory fence: Ensure our reads complete before unsetting flag
    std::atomic_thread_fence(std::memory_order_release);

    // Unset FUTURE_NEW_DATA to signal we consumed the data
    future_shm_->flags_.UnsetBits(FutureShm::FUTURE_NEW_DATA);
  }

  // All data received
  return true;
}

void LocalTransfer::SetComplete() {
  if (!future_shm_.IsNull()) {
    future_shm_->flags_.SetBits(FutureShm::FUTURE_COMPLETE);
  }

  // Delete task created from serialized client data if container is set
  if (container_ != nullptr && !task_ptr_.IsNull()) {
    container_->DelTask(method_id_, task_ptr_);
  }
}

}  // namespace chi
