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
 * CPU-side tests for Part 3: GPU Task Submission
 *
 * This test suite validates end-to-end GPU task submission:
 * - GPU queue infrastructure initialization
 * - CPU-based task submission
 * - GPU kernel task submission (GPU kernel test requires CUDA/ROCm)
 */

#include "simple_test.h"
#include <chrono>
#include <thread>

using namespace std::chrono_literals;

// Include Chimaera headers
#include <chimaera/chimaera.h>
#include <chimaera/singletons.h>
#include <chimaera/types.h>
#include <chimaera/pool_query.h>
#include <chimaera/task.h>

// Include MOD_NAME client and tasks
#include <chimaera/MOD_NAME/MOD_NAME_client.h>
#include <chimaera/MOD_NAME/MOD_NAME_tasks.h>

// Forward declare the C++ wrapper function from GPU file
// This function is always available when linking with GPU object library
extern "C" int run_gpu_kernel_task_submission_test(chi::PoolId pool_id, chi::u32 test_value);

// Global initialization state
static bool g_initialized = false;
static int g_test_counter = 0;

/**
 * Test: Verify GPU queue infrastructure is initialized
 */
TEST_CASE("gpu_queue_initialization", "[gpu][infrastructure][.skip]") {
  if (!g_initialized) {
    bool success = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
    REQUIRE(success);
    g_initialized = true;
    std::this_thread::sleep_for(500ms); // Give runtime time to initialize
  }

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
  auto* ipc = CHI_IPC;
  REQUIRE(ipc != nullptr);

  // Check GPU queue count
  size_t num_gpus = ipc->GetGpuQueueCount();
  int expected_gpus = hshm::GpuApi::GetDeviceCount();

  REQUIRE(static_cast<int>(num_gpus) == expected_gpus);

  // Verify each GPU queue
  for (size_t gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
    chi::TaskQueue* gpu_queue = ipc->GetGpuQueue(gpu_id);
    REQUIRE(gpu_queue != nullptr);

    if (gpu_queue) {
      // Verify queue has expected structure
      REQUIRE(gpu_queue->GetNumLanes() > 0);
    }
  }

  INFO("GPU queue initialization verified for " + std::to_string(num_gpus) + " GPU(s)");
#else
  INFO("GPU support not compiled in, skipping GPU queue checks");
#endif
}

/**
 * Test: CPU-side task submission and execution
 */
TEST_CASE("gpu_task_cpu_submission", "[gpu][cpu_submission]") {
  std::cout << "[TEST START] gpu_task_cpu_submission" << std::endl;

  // Initialize if not already done
  if (!g_initialized) {
    bool success = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
    REQUIRE(success);
    g_initialized = true;
    std::this_thread::sleep_for(500ms); // Give runtime time to initialize
  }

  // Create unique pool ID for this test
  g_test_counter++;
  std::cout << "[TEST] Creating pool_id" << std::endl;
  chi::PoolId pool_id(10000, g_test_counter);
  std::cout << "[TEST] pool_id created: " << pool_id.ToU64() << std::endl;

  // Create MOD_NAME container
  INFO("Creating MOD_NAME client");
  chimaera::MOD_NAME::Client client(pool_id);
  std::string pool_name = "gpu_test_pool_" + std::to_string(pool_id.ToU64());
  INFO("Calling AsyncCreate");
  auto create_task = client.AsyncCreate(chi::PoolQuery::Dynamic(), pool_name, pool_id);
  INFO("Waiting for AsyncCreate to complete");
  create_task.Wait();
  INFO("AsyncCreate completed");

  REQUIRE(create_task->return_code_ == 0);

  // Give container time to initialize
  std::this_thread::sleep_for(100ms);

  // Test simple task execution first
  INFO("Testing CustomTask before GpuSubmitTask");
  auto custom_future = client.AsyncCustom(chi::PoolQuery::Local(), "test", 1);
  custom_future.Wait();
  INFO("CustomTask completed successfully");

  // Now test GpuSubmit task execution
  const chi::u32 test_value = 123;
  const chi::u32 gpu_id = 0;

  INFO("Testing GpuSubmitTask");
  auto submit_future = client.AsyncGpuSubmit(chi::PoolQuery::Local(), gpu_id, test_value);
  INFO("AsyncGpuSubmit called, waiting...");
  submit_future.Wait();

  // Verify task executed
  REQUIRE(submit_future->GetReturnCode() == 0);

  // Verify result computation: result = test_value * 2 + gpu_id
  chi::u32 expected_result = (test_value * 2) + gpu_id;
  REQUIRE(submit_future->result_value_ == expected_result);

  INFO("GpuSubmit task executed successfully with correct result");
}

/**
 * Test: Multiple GPU task executions
 */
TEST_CASE("gpu_task_multiple_executions", "[gpu][multiple]") {
  REQUIRE(g_initialized);

  // Create unique pool ID for this test
  g_test_counter++;
  chi::PoolId pool_id(10000, g_test_counter);

  // Create MOD_NAME container
  chimaera::MOD_NAME::Client client(pool_id);
  std::string pool_name = "gpu_multi_test_" + std::to_string(pool_id.ToU64());
  auto create_task = client.AsyncCreate(chi::PoolQuery::Dynamic(), pool_name, pool_id);
  create_task.Wait();

  REQUIRE(create_task->return_code_ == 0);

  // Give container time to initialize
  std::this_thread::sleep_for(100ms);

  // Submit multiple tasks
  const int num_tasks = 5;
  for (int i = 0; i < num_tasks; ++i) {
    chi::u32 test_value = 100 + i;
    chi::u32 gpu_id = 0;

    auto submit_future = client.AsyncGpuSubmit(chi::PoolQuery::Local(), gpu_id, test_value);
    submit_future.Wait();

    // Verify task executed
    REQUIRE(submit_future->GetReturnCode() == 0);

    // Verify result computation: result = test_value * 2 + gpu_id
    chi::u32 expected_result = (test_value * 2) + gpu_id;
    REQUIRE(submit_future->result_value_ == expected_result);
  }

  INFO("Multiple GpuSubmit tasks executed successfully");
}

/**
 * Test: GPU kernel task submission
 * CRITICAL Part 3 test: GPU kernel calls NewTask and Send
 * This test is always compiled and calls into GPU code via wrapper function
 */
TEST_CASE("gpu_kernel_task_submission", "[gpu][kernel_submit]") {
  // Initialize if not already done
  if (!g_initialized) {
    bool success = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
    REQUIRE(success);
    g_initialized = true;
    std::this_thread::sleep_for(500ms); // Give runtime time to initialize
  }

  // Create unique pool ID for this test
  g_test_counter++;
  chi::PoolId pool_id(10000, g_test_counter);

  // Create MOD_NAME container
  chimaera::MOD_NAME::Client client(pool_id);
  std::string pool_name = "gpu_kernel_test_" + std::to_string(pool_id.ToU64());
  auto create_task = client.AsyncCreate(chi::PoolQuery::Dynamic(), pool_name, pool_id);
  create_task.Wait();

  REQUIRE(create_task->return_code_ == 0);

  // Give container time to initialize
  std::this_thread::sleep_for(100ms);

  // Run GPU kernel test via wrapper function (defined in GPU file)
  chi::u32 test_value = 999;
  int result = run_gpu_kernel_task_submission_test(pool_id, test_value);

  // Show result for debugging
  INFO("GPU kernel test result: " + std::to_string(result));

  // Verify success with detailed step-by-step diagnostics
  if (result == -100) {
    INFO("GPU backend initialization failed");
  } else if (result == -200) {
    INFO("CUDA synchronization failed");
  } else if (result == -201) {
    INFO("Kernel launch error");
  } else if (result == -777) {
    INFO("DIAGNOSTIC: Kernel entered but stopped before CHIMAERA_GPU_INIT");
  } else if (result == -666) {
    INFO("DIAGNOSTIC: CHIMAERA_GPU_INIT completed but stopped before IPC manager check");
  } else if (result == -10) {
    INFO("Step 0 FAILED: g_ipc_manager pointer is null after CHIMAERA_GPU_INIT");
  } else if (result == -11) {
    INFO("Step 1 FAILED: Backend allocation test (64 bytes) failed - AllocateBuffer returned null");
  } else if (result == -12) {
    INFO("Step 2 FAILED: Task-sized buffer allocation failed - AllocateBuffer(sizeof(GpuSubmitTask)) returned null");
  } else if (result == -130) {
    INFO("Step 3 FAILED: Out of memory before NewTask - AllocateBuffer(task_size) failed");
  } else if (result == -131) {
    INFO("Step 3 FAILED: AllocateBuffer inside manual NewTask path returned null");
  } else if (result == -132) {
    INFO("Step 3 FAILED: Placement new constructor returned nullptr");
  } else if (result == -133) {
    INFO("Step 3 FAILED: FullPtr construction from task pointer failed");
  } else if (result == -13) {
    INFO("Step 3 FAILED: NewTask returned null - task construction failed");
  } else if (result == 0 || result == -999) {
    INFO("GPU kernel did not set result flags (initialization issue?)");
  }

  REQUIRE(result == 1);
  INFO("SUCCESS: All steps passed - GPU kernel created and submitted task!");
}

//==============================================================================
// MAIN TEST RUNNER
//==============================================================================

SIMPLE_TEST_MAIN()
