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

#ifndef CLIO_CAE_SUMMARIZER_SUMMARIZER_CLIENT_H_
#define CLIO_CAE_SUMMARIZER_SUMMARIZER_CLIENT_H_

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/admin/admin_client.h>
#include <clio_cte/core/core_client.h>
#include <clio_cae/summarizer/summarizer_tasks.h>

#include <string>

namespace clio::cae::summarizer {

/**
 * Summarizer chimod client. The pool speaks the CTE core's task interface —
 * point a clio::cte::core::Client at kSummarizerPoolId (or at any pool whose
 * chain forwards into it) and every blob verb works unchanged, picking up
 * transparent summarization on the way through. This class exists to create
 * the pool with its chain configuration.
 */
class Client : public clio::cte::core::Client {
 public:
  Client() = default;

  /**
   * @param summarizer_pool_id The summarizer's own pool.
   * @param next_pool_id The pool it forwards to (normally the CTE core);
   *        also what the inherited CTE verbs on this client address.
   */
  Client(const clio::run::PoolId &summarizer_pool_id,
         const clio::run::PoolId &next_pool_id)
      : summarizer_pool_id_(summarizer_pool_id) {
    clio::cte::core::Client::Init(next_pool_id);
  }

#if CTP_IS_HOST
  /**
   * Create the summarizer container. Set params.next_pool_id_ to the pool it
   * forwards to (normally the CTE core).
   * @param pool_query Routing policy for the create task.
   * @param pool_name User-defined pool name.
   * @param custom_pool_id Pool id to create the container at.
   * @param params Chain + summarization configuration.
   */
  clio::run::Future<CreateTask> AsyncCreateSummarizer(
      const clio::run::PoolQuery &pool_query, const std::string &pool_name,
      const clio::run::PoolId &custom_pool_id,
      const SummarizerConfig &params) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<CreateTask>(
        clio::run::CreateTaskId(), clio::run::kAdminPoolId, pool_query,
        SummarizerConfig::chimod_lib_name, pool_name, custom_pool_id, this,
        params);
    auto fut = ipc->Send(task);
    summarizer_pool_id_ = custom_pool_id;
    return fut;
  }
#endif  // CTP_IS_HOST

  clio::run::PoolId summarizer_pool_id_;
};

}  // namespace clio::cae::summarizer

#endif  // CLIO_CAE_SUMMARIZER_SUMMARIZER_CLIENT_H_
