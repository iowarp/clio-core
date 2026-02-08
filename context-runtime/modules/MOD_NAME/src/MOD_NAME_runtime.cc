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
 * Runtime implementation for MOD_NAME
 *
 * Contains the server-side task processing logic.
 */

#include "../include/chimaera/MOD_NAME/MOD_NAME_runtime.h"

#include <chrono>

namespace chimaera::MOD_NAME {

// Method implementations for Runtime class

// Virtual method implementations (Init, Run, Del, SaveTask, LoadTask, NewCopy, Aggregate) now in autogen/MOD_NAME_lib_exec.cc

//===========================================================================
// Method implementations
//===========================================================================

chi::TaskResume Runtime::Create(hipc::FullPtr<CreateTask> task, chi::RunContext &rctx) {
  HLOG(kDebug, "MOD_NAME: Executing Create task for pool {}", task->pool_id_);

  // Container is already initialized via Init() before Create is called

  create_count_++;

  HLOG(kDebug,
        "MOD_NAME: Container created and initialized for pool: {} (ID: {}, "
        "count: {})",
        pool_name_, task->pool_id_, create_count_);
  (void)rctx;
  co_return;
}

chi::TaskResume Runtime::Custom(hipc::FullPtr<CustomTask> task, chi::RunContext &rctx) {
  HLOG(kDebug, "MOD_NAME: Executing Custom task with data: {}",
        task->data_.c_str());

  custom_count_++;

  // Process custom task here
  // In a real implementation, this would perform the custom operation

  HLOG(kDebug, "MOD_NAME: Custom completed (count: {})", custom_count_);
  (void)rctx;
  co_return;
}

chi::TaskResume Runtime::Destroy(hipc::FullPtr<DestroyTask> task, chi::RunContext &rctx) {
  HLOG(kDebug, "MOD_NAME: Executing Destroy task - Pool ID: {}",
        task->target_pool_id_);

  // Initialize output values
  task->return_code_ = 0;
  task->error_message_ = "";

  // In a real implementation, this would clean up MOD_NAME-specific resources
  // For now, just mark as successful
  HLOG(kDebug, "MOD_NAME: Container destroyed successfully");
  (void)rctx;
  co_return;
}

chi::u64 Runtime::GetWorkRemaining() const {
  // Template container implementation returns 0 (no work tracking)
  return 0;
}

//===========================================================================
// Task Serialization Method Implementations now in autogen/MOD_NAME_lib_exec.cc
//===========================================================================

chi::TaskResume Runtime::CoMutexTest(hipc::FullPtr<CoMutexTestTask> task,
                          chi::RunContext &rctx) {
  HLOG(kDebug, "MOD_NAME: Executing CoMutexTest task {} (hold: {}ms)",
        task->test_id_, task->hold_duration_ms_);

  // Use actual CoMutex synchronization primitive
  chi::ScopedCoMutex lock(test_comutex_);

  // Hold the mutex for the specified duration
  if (task->hold_duration_ms_ > 0) {
    auto start = std::chrono::high_resolution_clock::now();
    while (true) {
      auto now = std::chrono::high_resolution_clock::now();
      auto duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(now - start)
              .count();
      if (duration >= task->hold_duration_ms_) {
        break;
      }
    }
  }

  task->return_code_ = 0; // Success (0 means success in most conventions)
  HLOG(kDebug, "MOD_NAME: CoMutexTest {} completed", task->test_id_);
  (void)rctx;
  co_return;
}

chi::TaskResume Runtime::CoRwLockTest(hipc::FullPtr<CoRwLockTestTask> task,
                           chi::RunContext &rctx) {
  HLOG(kDebug, "MOD_NAME: Executing CoRwLockTest task {} ({}, hold: {}ms)",
        task->test_id_, (task->is_writer_ ? "writer" : "reader"),
        task->hold_duration_ms_);

  // Use actual CoRwLock synchronization primitive with appropriate lock type
  if (task->is_writer_) {
    chi::ScopedCoRwWriteLock lock(test_corwlock_);

    // Hold the write lock for the specified duration
    if (task->hold_duration_ms_ > 0) {
      auto start = std::chrono::high_resolution_clock::now();
      while (true) {
        auto now = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - start)
                .count();
        if (duration >= task->hold_duration_ms_) {
          break;
        }
      }
    }
  } else {
    chi::ScopedCoRwReadLock lock(test_corwlock_);

    // Hold the read lock for the specified duration
    if (task->hold_duration_ms_ > 0) {
      auto start = std::chrono::high_resolution_clock::now();
      while (true) {
        auto now = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - start)
                .count();
        if (duration >= task->hold_duration_ms_) {
          break;
        }
      }
    }
  }

  task->return_code_ = 0; // Success (0 means success in most conventions)
  HLOG(kDebug, "MOD_NAME: CoRwLockTest {} completed", task->test_id_);
  (void)rctx;
  co_return;
}

chi::TaskResume Runtime::WaitTest(hipc::FullPtr<WaitTestTask> task,
                                  chi::RunContext &rctx) {
  HLOG(kDebug,
        "MOD_NAME: Executing WaitTest task {} (depth: {}, current_depth: {})",
        task->test_id_, task->depth_, task->current_depth_);

  // Increment current depth
  task->current_depth_++;

  // If we haven't reached the target depth, create a subtask and wait for it
  if (task->current_depth_ < task->depth_) {
    HLOG(kDebug,
          "MOD_NAME: WaitTest {} creating recursive subtask at depth {}",
          task->test_id_, task->current_depth_);

    // Use the client API for recursive calls - this tests the co_await
    // functionality properly. Create a subtask with remaining depth
    chi::u32 remaining_depth = task->depth_ - task->current_depth_;
    auto subtask = client_.AsyncWaitTest(
        task->pool_query_, remaining_depth, task->test_id_);
    co_await subtask;
    chi::u32 origin_task_final_depth = subtask->current_depth_;
    (void)origin_task_final_depth;

    // The subtask returns the final depth it reached, so we set our depth to
    // that
    task->current_depth_ = task->depth_;

    HLOG(kDebug,
          "MOD_NAME: WaitTest {} subtask completed via client API, final "
          "depth: {}",
          task->test_id_, task->current_depth_);
  }

  HLOG(kDebug, "MOD_NAME: WaitTest {} completed at depth {}", task->test_id_,
        task->current_depth_);
  co_return;
}

chi::TaskResume Runtime::TestLargeOutput(hipc::FullPtr<TestLargeOutputTask> task,
                                         chi::RunContext &rctx) {
  HLOG(kDebug, "MOD_NAME: Executing TestLargeOutput task");

  // Allocate 1MB vector (1024 * 1024 bytes)
  constexpr size_t kOutputSize = 1024 * 1024;
  task->data_.resize(kOutputSize);

  // Fill with pattern: data[i] = i % 256
  for (size_t i = 0; i < kOutputSize; ++i) {
    task->data_[i] = static_cast<uint8_t>(i % 256);
  }

  HLOG(kDebug, "MOD_NAME: TestLargeOutput completed with {}KB output",
       kOutputSize / 1024);
  (void)rctx;
  co_return;
}

chi::TaskResume Runtime::GpuSubmit(hipc::FullPtr<GpuSubmitTask> task,
                                   chi::RunContext &rctx) {
  HLOG(kDebug, "MOD_NAME: Executing GpuSubmit task from GPU {}, test_value={}",
       task->gpu_id_, task->test_value_);

  // Simple computation to verify task executed correctly
  // Result = test_value * 2 + gpu_id
  task->result_value_ = (task->test_value_ * 2) + task->gpu_id_;

  HLOG(kDebug, "MOD_NAME: GpuSubmit completed, result_value={}",
       task->result_value_);
  (void)rctx;
  co_return;
}

// Static member definitions
chi::CoMutex Runtime::test_comutex_;
chi::CoRwLock Runtime::test_corwlock_;

} // namespace chimaera::MOD_NAME

// Define ChiMod entry points using CHI_TASK_CC macro
CHI_TASK_CC(chimaera::MOD_NAME::Runtime)