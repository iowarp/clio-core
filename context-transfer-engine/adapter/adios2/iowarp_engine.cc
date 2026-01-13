/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Distributed under BSD 3-Clause license.                                   *
 * Copyright by the Illinois Institute of Technology.                        *
 * All rights reserved.                                                      *
 *                                                                           *
 * This file is part of IOWarp Core. The full IOWarp Core copyright         *
 * notice, including terms governing use, modification, and redistribution,  *
 * is contained in the COPYING file, which can be found at the top directory.*
 * If you do not have access to the file, you may request a copy             *
 * from scslab@iit.edu.                                                      *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "iowarp_engine.h"

#include <stdexcept>
#include <string>

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
      open_(false) {
  // CTE client will be accessed via WRP_CTE_CLIENT singleton when needed
  wrp_cte::core::WRP_CTE_CLIENT_INIT("", chi::PoolQuery::Local());
}

/**
 * Destructor
 */
IowarpEngine::~IowarpEngine() {
  if (open_) {
    DoClose();
  }
}

/**
 * Initialize the engine
 */
void IowarpEngine::Init_() {
  if (open_) {
    throw std::runtime_error("IowarpEngine::Init_: Engine already initialized");
  }

  // Create or get tag for this ADIOS file/session
  // Use the engine name as the tag name
  try {
    current_tag_ = std::make_unique<wrp_cte::core::Tag>(m_Name);
    open_ = true;
  } catch (const std::exception &e) {
    throw std::runtime_error(
        std::string("IowarpEngine::Init_: Failed to create/get tag: ") +
        e.what());
  }
}

/**
 * Begin a new step
 * @param mode Step mode (Read, Append, Update)
 * @param timeoutSeconds Timeout in seconds (-1 for no timeout)
 * @return Step status
 */
adios2::StepStatus IowarpEngine::BeginStep(adios2::StepMode mode,
                                           const float timeoutSeconds) {
  (void)mode;            // Suppress unused parameter warning
  (void)timeoutSeconds;  // Suppress unused parameter warning

  // Lazy initialization if not already initialized
  if (!open_) {
    Init_();
  }

  // Increment step counter
  IncrementCurrentStep();

  return adios2::StepStatus::OK;
}

/**
 * End the current step
 */
void IowarpEngine::EndStep() {
  if (!open_) {
    throw std::runtime_error("IowarpEngine::EndStep: Engine not initialized");
  }

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

  // Calculate blob name from variable name and current step
  std::string blob_name =
      variable.m_Name + "_step_" + std::to_string(current_step_);

  // Calculate data size
  size_t element_count = 1;
  for (size_t dim : variable.m_Shape) {
    element_count *= dim;
  }
  size_t data_size = element_count * sizeof(T);

  // Put blob to CTE synchronously
  try {
    current_tag_->PutBlob(blob_name, reinterpret_cast<const char *>(values),
                          data_size, 0);
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

  // Calculate blob name from variable name and current step
  std::string blob_name =
      variable.m_Name + "_step_" + std::to_string(current_step_);

  // Calculate data size
  size_t element_count = 1;
  for (size_t dim : variable.m_Shape) {
    element_count *= dim;
  }
  size_t data_size = element_count * sizeof(T);

  // Allocate shared memory for the data
  auto *ipc_manager = CHI_IPC;
  auto buffer = ipc_manager->AllocateBuffer(data_size);

  // Copy data to shared memory
  std::memcpy(buffer.ptr_, values, data_size);

  // Put blob asynchronously
  try {
    auto task = current_tag_->AsyncPutBlob(
        blob_name, buffer.shm_.template Cast<void>(), data_size, 0, 1.0F);

    // Store task and buffer in deferred_tasks_ vector
    // Buffer will be kept alive until EndStep processes the task
    deferred_tasks_.push_back({std::move(task), std::move(buffer)});
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

  // Calculate blob name from variable name and current step
  std::string blob_name =
      variable.m_Name + "_step_" + std::to_string(current_step_);

  // Calculate data size
  size_t element_count = 1;
  for (size_t dim : variable.m_Shape) {
    element_count *= dim;
  }
  size_t data_size = element_count * sizeof(T);

  // Get blob from CTE synchronously
  try {
    current_tag_->GetBlob(blob_name, reinterpret_cast<char *>(values),
                          data_size, 0);
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
