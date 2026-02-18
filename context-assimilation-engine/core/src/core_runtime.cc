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

#include <wrp_cae/core/core_runtime.h>
#include <wrp_cae/core/factory/assimilation_ctx.h>
#include <wrp_cae/core/factory/assimilator_factory.h>
#ifdef WRP_CAE_ENABLE_HDF5
#include <hdf5.h>
#include <wrp_cae/core/factory/hdf5_file_assimilator.h>
#endif

#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include <sstream>
#include <vector>

// Include wrp_cte headers before opening namespace to avoid Method namespace
// collision
#include <wrp_cte/core/core_client.h>
#include <wrp_cte/core/core_tasks.h>

// Define ChiMod entry points using CHI_TASK_CC macro
CHI_TASK_CC(wrp_cae::core::Runtime)

namespace wrp_cae::core {

void Runtime::Create(hipc::FullPtr<CreateTask> task, chi::RunContext& ctx) {
  // Container is already initialized via Init() before Create is called
  // Do NOT call Init() here

  // Initialize CTE client using the CTE pool ID
  cte_client_ =
      std::make_shared<wrp_cte::core::Client>(wrp_cte::core::kCtePoolId);

  // Additional container-specific initialization logic here
  HLOG(kInfo, "Core container created and initialized for pool: {} (ID: {})",
       pool_name_, pool_id_);
}

chi::u64 Runtime::GetWorkRemaining() const {
  // CAE doesn't currently track work remaining
  // Return 0 to indicate no pending work
  return 0;
}

chi::TaskResume Runtime::ParseOmni(hipc::FullPtr<ParseOmniTask> task,
                                   chi::RunContext& ctx) {
  HLOG(kInfo, "ParseOmni called with {} bytes of serialized data",
       task->serialized_ctx_.size());

  // Deserialize the vector of AssimilationCtx
  std::vector<AssimilationCtx> assimilation_contexts;
  try {
    std::stringstream ss(task->serialized_ctx_.str());
    cereal::BinaryInputArchive ar(ss);
    ar(assimilation_contexts);
  } catch (const std::exception& e) {
    HLOG(kError, "ParseOmni: Failed to deserialize AssimilationCtx vector: {}",
         e.what());
    task->result_code_ = -1;
    task->error_message_ = e.what();
    task->num_tasks_scheduled_ = 0;
    co_return;
  }

  HLOG(kInfo, "ParseOmni: Processing {} assimilation contexts",
       assimilation_contexts.size());

  // Process each assimilation context
  chi::u32 tasks_scheduled = 0;
  AssimilatorFactory factory(cte_client_);

  for (size_t i = 0; i < assimilation_contexts.size(); ++i) {
    const auto& assimilation_ctx = assimilation_contexts[i];

    HLOG(kInfo,
         "ParseOmni: Processing context {}/{} - src: {}, dst: {}, format: {}",
         i + 1, assimilation_contexts.size(), assimilation_ctx.src,
         assimilation_ctx.dst, assimilation_ctx.format);

    // Get appropriate assimilator for this context
    auto assimilator = factory.Get(assimilation_ctx.src);

    if (!assimilator) {
      HLOG(kError, "ParseOmni: No assimilator found for source: {}",
           assimilation_ctx.src);
      task->result_code_ = -2;
      task->error_message_ =
          "No assimilator found for source: " + assimilation_ctx.src;
      task->num_tasks_scheduled_ = tasks_scheduled;
      co_return;
    }

    // Schedule the assimilation using co_await
    int result = 0;
    co_await assimilator->Schedule(assimilation_ctx, result);
    if (result != 0) {
      HLOG(
          kError,
          "ParseOmni: Assimilator failed for context {}/{} with error code: {}",
          i + 1, assimilation_contexts.size(), result);
      task->result_code_ = result;
      task->error_message_ = std::string("Assimilator failed");
      task->num_tasks_scheduled_ = tasks_scheduled;
      co_return;
    }

    tasks_scheduled++;
  }

  // Success
  task->result_code_ = 0;
  task->error_message_ = "";
  task->num_tasks_scheduled_ = tasks_scheduled;

  HLOG(kInfo, "ParseOmni: Successfully scheduled {} assimilations",
       tasks_scheduled);
  co_return;
}

chi::TaskResume Runtime::ProcessHdf5Dataset(
    hipc::FullPtr<ProcessHdf5DatasetTask> task, chi::RunContext& ctx) {
#ifdef WRP_CAE_ENABLE_HDF5
  HLOG(kInfo, "ProcessHdf5Dataset: file='{}', dataset='{}', tag_prefix='{}'",
       task->file_path_.str(), task->dataset_path_.str(),
       task->tag_prefix_.str());

  // Open the HDF5 file
  hid_t file_id =
      H5Fopen(task->file_path_.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0) {
    HLOG(kError, "ProcessHdf5Dataset: Failed to open HDF5 file: {}",
         task->file_path_.str());
    task->result_code_ = -1;
    task->error_message_ =
        chi::priv::string("Failed to open HDF5 file", HSHM_MALLOC);
    co_return;
  }

  // Create assimilator and process the dataset
  wrp_cae::core::Hdf5FileAssimilator assimilator(cte_client_);
  int result = 0;
  co_await assimilator.ProcessDataset(file_id, task->dataset_path_.str(),
                                      task->tag_prefix_.str(), result);

  // Close the HDF5 file
  H5Fclose(file_id);

  if (result != 0) {
    HLOG(kError,
         "ProcessHdf5Dataset: Failed to process dataset '{}' (error: {})",
         task->dataset_path_.str(), result);
    task->result_code_ = result;
    task->error_message_ =
        chi::priv::string("Dataset processing failed", HSHM_MALLOC);
  } else {
    HLOG(kInfo, "ProcessHdf5Dataset: Successfully processed dataset '{}'",
         task->dataset_path_.str());
    task->result_code_ = 0;
  }
#else
  task->result_code_ = -1;
  task->error_message_ =
      chi::priv::string("HDF5 support not compiled in", HSHM_MALLOC);
#endif
  co_return;
}

}  // namespace wrp_cae::core
