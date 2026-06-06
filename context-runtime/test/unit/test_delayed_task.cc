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
 * Verifies that the runtime still handles tasks correctly after workers have
 * been idle (sleeping on SIGUSR1 via signal_mpmc_queue) for an extended
 * period.  The test waits 3 seconds before submitting a MOD_NAME custom task
 * so that workers are guaranteed to have entered the event-wait path.
 */

#include "../simple_test.h"
#include <chrono>
#include <thread>

using namespace std::chrono_literals;

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/singletons.h>
#include <clio_runtime/types.h>

#include <clio_runtime/MOD_NAME/MOD_NAME_client.h>
#include <clio_runtime/MOD_NAME/MOD_NAME_tasks.h>

namespace {
constexpr chi::PoolId kDelayedTestPoolId = chi::PoolId(200, 0);
bool g_initialized = false;
}  // namespace

class DelayedTaskFixture {
 public:
  DelayedTaskFixture() {
    if (!g_initialized) {
      bool ok = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
      if (ok) {
        g_initialized = true;
        SimpleTest::g_test_finalize = chi::CHIMAERA_FINALIZE;
        std::this_thread::sleep_for(500ms);
      }
    }
  }
};

TEST_CASE("Delayed MOD_NAME task fires after 3s worker idle",
          "[runtime][delayed_task]") {
  DelayedTaskFixture fixture;
  REQUIRE(g_initialized);

  // Let workers go idle — they should block on SIGUSR1 via signal_mpmc_queue.
  std::this_thread::sleep_for(3s);

  // Create the MOD_NAME pool.
  clio::run::MOD_NAME::Client client;
  chi::PoolQuery pool_query = chi::PoolQuery::Dynamic();
  auto create_task =
      client.AsyncCreate(pool_query, "delayed_test_pool", kDelayedTestPoolId);
  create_task.Wait();
  client.pool_id_ = create_task->new_pool_id_;
  REQUIRE(create_task->return_code_ == 0);

  // Fire a custom task and verify it completes.
  auto task = client.AsyncCustom(pool_query, "delayed_input", 1);
  task.Wait();
  REQUIRE(task->return_code_ == 0);
}

SIMPLE_TEST_MAIN()
