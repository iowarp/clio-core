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
 * Runtime implementation for Simple Mod ChiMod
 *
 * Minimal ChiMod for testing external development patterns.
 * Contains the server-side task processing logic with basic functionality.
 */

#include "chimaera/simple_mod/simple_mod_runtime.h"

#include <chimaera/chimaera_manager.h>
#include <chimaera/module_manager.h>
#include <chimaera/pool_manager.h>
#include <chimaera/task_archives.h>
#include <hermes_shm/memory/allocator/malloc_allocator.h>

#include <iostream>

namespace external_test::simple_mod {

// Method implementations for Runtime class

void Runtime::Init(const chi::PoolId &pool_id, const std::string &pool_name) {
  // Call base class Init to set pool_id_ and pool_name_
  chi::Container::Init(pool_id, pool_name))

  // Initialize the client for this ChiMod
  client_ = Client(pool_id);
}

// Virtual method implementations now in autogen/simple_mod_lib_exec.cc

//===========================================================================
// Method implementations
//===========================================================================

void Runtime::Create(hipc::FullPtr<CreateTask> task, chi::RunContext &rctx) {
  // Simple mod container creation logic
  HLOG(kInfo, "SimpleMod: Initializing simple_mod container");

  // Initialize the Simple Mod container with pool information from the task
  // Note: CreateLocalQueue has been removed in favor of unified scheduling
  // CreateLocalQueue(kMetadataQueue, 1, 1);  // Metadata operations

  create_count_++;

  // Set success result
  task->return_code_ = 0;
  task->error_message_ = "";

  HLOG(kSuccess, "SimpleMod: Container created and initialized for pool: {} (ID: {}, count: {})",
       pool_name_, task->pool_id_, create_count_);
}

void Runtime::Destroy(hipc::FullPtr<DestroyTask> task, chi::RunContext &rctx) {
  HLOG(kInfo, "SimpleMod: Executing Destroy task - Pool ID: {}", task->target_pool_id_);

  // Initialize output values
  task->return_code_ = 0;
  task->error_message_ = "";

  try {
    // Simple destruction logic - just log that we're destroyed
    HLOG(kSuccess, "SimpleMod: Container destroyed successfully - ID: {}", task->target_pool_id_);

    // Set success result
    task->return_code_ = 0;

  } catch (const std::exception &e) {
    task->return_code_ = 99;
    task->error_message_ = chi::priv::string(
        HSHM_MALLOC,
        std::string("Exception during simple_mod destruction: ") + e.what());
    HLOG(kError, "SimpleMod: Destruction failed with exception: {}", e.what());
  }
}

void Runtime::Flush(hipc::FullPtr<FlushTask> task, chi::RunContext &rctx) {
  HLOG(kInfo, "SimpleMod: Executing Flush task");

  // Simple flush implementation - just report no work remaining
  task->return_code_ = 0;
  task->total_work_done_ = GetWorkRemaining();

  HLOG(kSuccess, "SimpleMod: Flush completed - work done: {}", task->total_work_done_);
}

chi::u64 Runtime::GetWorkRemaining() const {
  // Simple mod typically has no pending work, returns 0
  return 0;
}

// Note: Task serialization methods (SaveIn, LoadIn, SaveOut, LoadOut, NewCopy)
// are automatically generated in autogen/simple_mod_lib_exec.cc and should not
// be manually implemented here.

} // namespace external_test::simple_mod