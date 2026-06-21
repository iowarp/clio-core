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

#ifndef SIMPLE_MOD_CLIENT_H_
#define SIMPLE_MOD_CLIENT_H_

#include <clio_runtime/clio_runtime.h>

#include "simple_mod_tasks.h"

/**
 * Client API for Simple Mod ChiMod
 *
 * Minimal ChiMod for testing external development patterns.
 * All methods return Future objects - call Wait() to block for completion.
 * Task cleanup is automatic when Future goes out of scope after Wait().
 */

namespace external_test::simple_mod {

class Client : public clio::run::ContainerClient {
 public:
  /** Default constructor */
  Client() = default;

  /** Constructor with pool ID */
  explicit Client(const clio::run::PoolId& pool_id) { Init(pool_id); }

  /**
   * Create the Simple Mod container (asynchronous)
   * @param mctx Memory context
   * @param pool_query Pool routing information
   * @return Future for the CreateTask
   */
  clio::run::Future<CreateTask> AsyncCreate(const ctp::ipc::MemContext& mctx,
                                       const clio::run::PoolQuery& pool_query) {
    (void)mctx;  // Memory context not needed for task creation
    auto* ipc_manager = CLIO_IPC;

    // Use admin pool for CreateTask as per CLAUDE.md requirements
    // Pass 'this' as client pointer for PostWait callback
    auto task = ipc_manager->NewTask<CreateTask>(
        clio::run::CreateTaskId(), clio::run::kAdminPoolId, pool_query,
        "external_test_simple_mod", "simple_mod_pool", pool_id_,
        this);

    return ipc_manager->Send(task);
  }

  /**
   * Destroy the Simple Mod container (asynchronous)
   * @param mctx Memory context
   * @param pool_query Pool routing information
   * @return Future for the DestroyTask
   */
  clio::run::Future<DestroyTask> AsyncDestroy(const ctp::ipc::MemContext& mctx,
                                         const clio::run::PoolQuery& pool_query) {
    (void)mctx;  // Memory context not needed for task creation
    auto* ipc_manager = CLIO_IPC;

    auto task = ipc_manager->NewTask<DestroyTask>(clio::run::CreateTaskId(), pool_id_,
                                                  pool_query, pool_id_, 0);

    return ipc_manager->Send(task);
  }

  /**
   * Flush simple mod operations (asynchronous)
   * @param mctx Memory context
   * @param pool_query Pool routing information
   * @return Future for the FlushTask
   */
  clio::run::Future<FlushTask> AsyncFlush(const ctp::ipc::MemContext& mctx,
                                     const clio::run::PoolQuery& pool_query) {
    (void)mctx;  // Memory context not needed for task creation
    auto* ipc_manager = CLIO_IPC;

    auto task = ipc_manager->NewTask<FlushTask>(clio::run::CreateTaskId(), pool_id_,
                                                pool_query);

    return ipc_manager->Send(task);
  }
};

}  // namespace external_test::simple_mod

#endif  // SIMPLE_MOD_CLIENT_H_
