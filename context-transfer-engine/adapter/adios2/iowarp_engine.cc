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

#include "iowarp_engine.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>

#include <clio_ctp/util/logging.h>

namespace coeus {

/**
 * Main constructor for IowarpEngine
 * @param io ADIOS2 IO object
 * @param name Engine name
 * @param mode Open mode (Read, Write, Append, etc.)
 * @param comm MPI communicator
 */
IowarpEngine::IowarpEngine(adios2::core::IO &io, const std::string &name,
                           const adios2::Mode mode, adios2::helper::Comm comm)
    : adios2::plugin::PluginEngineInterface(io, name, mode, std::move(comm)),
      current_tag_(nullptr),
      current_step_(0),
      rank_(m_Comm.Rank()),
      open_(false),
      total_io_time_ms_(0.0) {
  HLOG(kDebug, "[IowarpEngine] Constructor entered, rank={}, name={}", rank_, name);

  // At >=512 nodes (>=6144 ranks at 12 ppn) calling CLIO_CTE_CLIENT_INIT
  // simultaneously across all ranks overwhelms each daemon's local 9416
  // ROUTER — every per-rank ZMTP greeting times out and the SIM aborts
  // before step 1. Stagger init per local-rank within a node so the
  // burst spreads over ~3s instead of arriving as a single thundering
  // herd. The world-rank stagger is unnecessary because each daemon
  // only sees its node's local ranks (PPN of them).
  // PPN is read from IOWARP_PPN env (jarvis-cd's adios2_gray_scott pkg
  // exports this), default 12.
  {
    const char *ppn_env = std::getenv("IOWARP_PPN");
    int ppn = (ppn_env && *ppn_env) ? std::atoi(ppn_env) : 12;
    if (ppn < 1) ppn = 1;
    int local_rank = rank_ % ppn;
    // Per-local-rank stagger step (μs); default 250 ms so 12 ranks
    // spread over 3s. Tunable via CLIO_INIT_STAGGER_MS.
    const char *stag_env = clio::run::env::GetCompat("INIT_STAGGER_MS");
    int stagger_ms = (stag_env && *stag_env) ? std::atoi(stag_env) : 250;
    if (stagger_ms < 0) stagger_ms = 0;
    if (local_rank > 0) {
      ::usleep(static_cast<useconds_t>(local_rank) *
               static_cast<useconds_t>(stagger_ms) * 1000);
    }
  }

  // Initialize CTE client - assumes CLIO Runtime runtime is already running.
  // Retry with jittered backoff: at >=512 nodes the local daemon is
  // busy serving cross-node SWIM probes (511+ peers) and may take many
  // seconds to drain its 9416 ROUTER accept queue; a few short retries
  // are not enough. Default 60 attempts × ~3s mean = ~3 min budget.
  // Jitter desynchronizes 12 same-node ranks so they don't all retry
  // on the same second. Tunable via CLIO_INIT_ATTEMPTS and
  // CLIO_INIT_SLEEP_MS (sleep is mean; actual is uniform [0.5x, 1.5x]).
  HLOG(kDebug, "[IowarpEngine] About to call CLIO_CTE_CLIENT_INIT");
  const char *att_env = clio::run::env::GetCompat("INIT_ATTEMPTS");
  int max_attempts = (att_env && *att_env) ? std::atoi(att_env) : 60;
  if (max_attempts < 1) max_attempts = 1;
  const char *slp_env = clio::run::env::GetCompat("INIT_SLEEP_MS");
  int mean_sleep_ms = (slp_env && *slp_env) ? std::atoi(slp_env) : 3000;
  if (mean_sleep_ms < 1) mean_sleep_ms = 1;
  // Per-rank seed so each rank's jitter sequence differs.
  unsigned int rng_state =
      static_cast<unsigned int>(rank_ * 2654435761u) ^
      static_cast<unsigned int>(::getpid());
  bool ok = false;
  for (int attempt = 0; attempt < max_attempts; ++attempt) {
    if (clio::cte::core::CLIO_CTE_CLIENT_INIT("", clio::run::PoolQuery::Local())) {
      ok = true;
      break;
    }
    HLOG(kWarning,
         "[IowarpEngine] CLIO_CTE_CLIENT_INIT failed (rank={}, attempt={}/{}); retrying",
         rank_, attempt + 1, max_attempts);
    // uniform jitter in [0.5, 1.5] × mean
    int rnd = rand_r(&rng_state) % 1001;  // 0..1000
    int jitter_ms = (mean_sleep_ms * (500 + rnd)) / 1000;
    ::usleep(static_cast<useconds_t>(jitter_ms) * 1000);
  }
  if (!ok) {
    throw std::runtime_error(
        "IowarpEngine: CLIO_CTE_CLIENT_INIT failed after " +
        std::to_string(max_attempts) +
        " attempts - is Clio runtime running?");
  }
  HLOG(kDebug, "[IowarpEngine] CLIO_CTE_CLIENT_INIT completed");

  // Start wall clock timer
  wall_clock_start_ = std::chrono::high_resolution_clock::now();

  HLOG(kDebug, "[IowarpEngine] Constructor completed, starting timing measurement");

  if (rank_ == 0) {
    HLOG(kInfo, "[IowarpEngine] Starting timing measurement");
  }
}

/**
 * Destructor
 */
IowarpEngine::~IowarpEngine() {
  if (open_) {
    DoClose();
  }

  // Calculate total wall clock time
  auto wall_clock_end = std::chrono::high_resolution_clock::now();
  double total_wall_time_ms =
      std::chrono::duration<double, std::milli>(wall_clock_end - wall_clock_start_).count();
  double compute_time_ms = total_wall_time_ms - total_io_time_ms_;

  if (rank_ == 0) {
    HLOG(kInfo, "");
    HLOG(kInfo, "========================================");
    HLOG(kInfo, "[IowarpEngine] Timing Summary");
    HLOG(kInfo, "========================================");
    HLOG(kInfo, "Total wall time:  {} ms", total_wall_time_ms);
    HLOG(kInfo, "Total I/O time:   {} ms", total_io_time_ms_);
    HLOG(kInfo, "Compute time:     {} ms", compute_time_ms);
    HLOG(kInfo, "I/O percentage:   {}%", (total_io_time_ms_ / total_wall_time_ms * 100.0));
    HLOG(kInfo, "Compute percentage: {}%", (compute_time_ms / total_wall_time_ms * 100.0));
    HLOG(kInfo, "========================================");
    HLOG(kInfo, "");
  }
}

/**
 * Initialize the engine
 */
void IowarpEngine::Init_() {
  HLOG(kDebug, "[IowarpEngine] Init_() entered, open_={}", open_);

  if (open_) {
    throw std::runtime_error("IowarpEngine::Init_: Engine already initialized");
  }

  // Create or get tag for this ADIOS file/session
  // Use the engine name as the tag name
  try {
    HLOG(kDebug, "[IowarpEngine] About to create Tag with name={}", m_Name);
    current_tag_ = std::make_unique<clio::cte::core::Tag>(m_Name);
    HLOG(kDebug, "[IowarpEngine] Tag created successfully");
    open_ = true;
  } catch (const std::exception &e) {
    HLOG(kDebug, "[IowarpEngine] Tag creation failed: {}", e.what());
    throw std::runtime_error(
        std::string("IowarpEngine::Init_: Failed to create/get tag: ") +
        e.what());
  }

  HLOG(kDebug, "[IowarpEngine] Init_() completed");
}

/**
 * Begin a new step
 * @param mode Step mode (Read, Append, Update)
 * @param timeoutSeconds Timeout in seconds (-1 for no timeout)
 * @return Step status
 */
adios2::StepStatus IowarpEngine::BeginStep(adios2::StepMode mode,
                                           const float timeoutSeconds) {
  HLOG(kDebug, "[IowarpEngine] BeginStep() entered, open_={}", open_);

  (void)mode;            // Suppress unused parameter warning
  (void)timeoutSeconds;  // Suppress unused parameter warning

  // Lazy initialization if not already initialized
  if (!open_) {
    HLOG(kDebug, "[IowarpEngine] BeginStep() calling Init_()");
    Init_();
    HLOG(kDebug, "[IowarpEngine] BeginStep() Init_() returned");
  }

  // Increment step counter
  IncrementCurrentStep();
  HLOG(kDebug, "[IowarpEngine] BeginStep() completed, step={}", current_step_);

  return adios2::StepStatus::OK;
}

/**
 * End the current step
 */
void IowarpEngine::EndStep() {
  if (!open_) {
    throw std::runtime_error("IowarpEngine::EndStep: Engine not initialized");
  }

  // Timing measurement for I/O operations
  auto io_start = std::chrono::high_resolution_clock::now();

  // Process all deferred put tasks from this step
  for (auto &deferred : deferred_tasks_) {
    // Set TASK_DATA_OWNER flag so task destructor will free the buffer
    auto *task_ptr = deferred.task.get();
    if (task_ptr != nullptr) {
      task_ptr->SetFlags(TASK_DATA_OWNER);
    }

    // Wait for task to complete
    deferred.task.Wait();
  }

  // Clear the deferred tasks vector for the next step
  deferred_tasks_.clear();

  // Measure and log I/O time
  auto io_end = std::chrono::high_resolution_clock::now();
  double io_time_ms = std::chrono::duration<double, std::milli>(io_end - io_start).count();

  // Accumulate total I/O time
  total_io_time_ms_ += io_time_ms;

  // Log per-step I/O time
  if (rank_ == 0) {
    HLOG(kInfo, "[IowarpEngine] Step {} I/O time: {} ms (Total I/O: {} ms)",
         current_step_, io_time_ms, total_io_time_ms_);
  }
}

/**
 * Get current step number
 * @return Current step
 */
size_t IowarpEngine::CurrentStep() const { return current_step_; }

/**
 * Close the engine
 * @param transportIndex Transport index to close (-1 for all)
 */
void IowarpEngine::DoClose(const int transportIndex) {
  (void)transportIndex;  // Suppress unused parameter warning

  if (!open_) {
    return;
  }

  // Clean up resources
  current_tag_.reset();
  open_ = false;
}

/**
 * Put data synchronously
 * @tparam T Data type
 * @param variable ADIOS2 variable
 * @param values Data pointer
 */
template <typename T>
void IowarpEngine::DoPutSync_(const adios2::core::Variable<T> &variable,
                              const T *values) {
  // Lazy initialization if not already initialized
  if (!open_) {
    Init_();
  }

  if (!current_tag_) {
    throw std::runtime_error("IowarpEngine::DoPutSync_: No active tag");
  }

  // Calculate blob name from variable name, current step, and rank
  std::string blob_name =
      variable.m_Name + "_step_" + std::to_string(current_step_) +
      "_rank_" + std::to_string(rank_);

  // Calculate data size using m_Count (local selection size), not m_Shape (global)
  size_t element_count = 1;
  if (!variable.m_Count.empty()) {
    for (size_t dim : variable.m_Count) {
      element_count *= dim;
    }
  } else if (!variable.m_Shape.empty()) {
    for (size_t dim : variable.m_Shape) {
      element_count *= dim;
    }
  }
  size_t data_size = element_count * sizeof(T);

  // Put blob to CTE synchronously
  try {
    current_tag_->PutBlob(blob_name, reinterpret_cast<const char *>(values),
                          data_size, 0, 1.0f);
  } catch (const std::exception &e) {
    throw std::runtime_error(
        std::string("IowarpEngine::DoPutSync_: Failed to put blob: ") +
        e.what());
  }
}

/**
 * Put data asynchronously
 * @tparam T Data type
 * @param variable ADIOS2 variable
 * @param values Data pointer
 */
template <typename T>
void IowarpEngine::DoPutDeferred_(const adios2::core::Variable<T> &variable,
                                  const T *values) {
  // Lazy initialization if not already initialized
  if (!open_) {
    Init_();
  }

  if (!current_tag_) {
    throw std::runtime_error("IowarpEngine::DoPutDeferred_: No active tag");
  }

  // Calculate blob name from variable name, current step, and rank
  // Each rank writes its own portion with a unique blob name
  std::string blob_name =
      variable.m_Name + "_step_" + std::to_string(current_step_) +
      "_rank_" + std::to_string(rank_);

  // Calculate data size using m_Count (local selection size), not m_Shape (global)
  // In MPI applications, each rank only has a portion of the global array
  size_t element_count = 1;
  if (!variable.m_Count.empty()) {
    // Use local count if available (for selections/MPI decomposition)
    for (size_t dim : variable.m_Count) {
      element_count *= dim;
    }
  } else if (!variable.m_Shape.empty()) {
    // Fall back to global shape if no selection
    for (size_t dim : variable.m_Shape) {
      element_count *= dim;
    }
  }
  size_t data_size = element_count * sizeof(T);

  // Put blob asynchronously
  try {
    auto *ipc_manager = CLIO_IPC;
    if (ipc_manager == nullptr) {
      throw std::runtime_error("IowarpEngine::DoPutDeferred_: CLIO_IPC is null");
    }

    // Allocate shared memory buffer and copy data
    auto buffer = ipc_manager->AllocateBuffer(data_size);
    if (buffer.ptr_ == nullptr) {
      throw std::runtime_error(
          "IowarpEngine::DoPutDeferred_: Failed to allocate buffer");
    }

    // Check if values pointer is valid
    if (values == nullptr) {
      throw std::runtime_error(
          "IowarpEngine::DoPutDeferred_: values pointer is null");
    }

    std::memcpy(buffer.ptr_, values, data_size);

    auto task = current_tag_->AsyncPutBlob(
        blob_name, buffer.shm_.template Cast<void>(), data_size, 0, 1.0f);

    // Store task and buffer in deferred_tasks_ vector
    // Buffer will be kept alive until EndStep processes the task
    deferred_tasks_.emplace_back(DeferredTask{std::move(task), std::move(buffer)});
  } catch (const std::exception &e) {
    throw std::runtime_error(
        std::string("IowarpEngine::DoPutDeferred_: Failed to put blob: ") +
        e.what());
  }
}

/**
 * Get data synchronously
 * @tparam T Data type
 * @param variable ADIOS2 variable
 * @param values Output buffer
 */
template <typename T>
void IowarpEngine::DoGetSync_(const adios2::core::Variable<T> &variable,
                              T *values) {
  // Lazy initialization if not already initialized
  if (!open_) {
    Init_();
  }

  if (!current_tag_) {
    throw std::runtime_error("IowarpEngine::DoGetSync_: No active tag");
  }

  // Calculate blob name from variable name, current step, and rank
  std::string blob_name =
      variable.m_Name + "_step_" + std::to_string(current_step_) +
      "_rank_" + std::to_string(rank_);

  // Calculate expected data size using m_Count (local selection size)
  size_t element_count = 1;
  if (!variable.m_Count.empty()) {
    for (size_t dim : variable.m_Count) {
      element_count *= dim;
    }
  } else if (!variable.m_Shape.empty()) {
    for (size_t dim : variable.m_Shape) {
      element_count *= dim;
    }
  }
  size_t expected_size = element_count * sizeof(T);

  // Get blob from CTE synchronously
  try {
    current_tag_->GetBlob(blob_name, reinterpret_cast<char *>(values),
                          expected_size, 0);
  } catch (const std::exception &e) {
    throw std::runtime_error(
        std::string("IowarpEngine::DoGetSync_: Failed to get blob: ") +
        e.what());
  }
}

/**
 * Get data asynchronously
 * @tparam T Data type
 * @param variable ADIOS2 variable
 * @param values Output buffer
 */
template <typename T>
void IowarpEngine::DoGetDeferred_(const adios2::core::Variable<T> &variable,
                                  T *values) {
  // Lazy initialization if not already initialized
  if (!open_) {
    Init_();
  }

  if (!current_tag_) {
    throw std::runtime_error("IowarpEngine::DoGetDeferred_: No active tag");
  }

  // For now, just call the sync version
  // In production, should use async API and track the Future
  DoGetSync_(variable, values);
}

}  // namespace coeus

/**
 * C wrapper to create engine
 * @param io ADIOS2 IO object
 * @param name Engine name
 * @param mode Open mode
 * @param comm MPI communicator
 * @return Engine pointer
 */
extern "C" {
coeus::IowarpEngine *EngineCreate(adios2::core::IO &io, const std::string &name,
                                  const adios2::Mode mode,
                                  adios2::helper::Comm comm) {
  return new coeus::IowarpEngine(io, name, mode, std::move(comm));
}

/**
 * C wrapper to destroy engine
 * @param obj Engine pointer to destroy
 */
void EngineDestroy(coeus::IowarpEngine *obj) { delete obj; }
}
