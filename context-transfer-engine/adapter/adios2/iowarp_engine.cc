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

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
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
      open_(false),
      compress_mode_(0),
      compress_lib_(0),
      compress_trace_(false),
      total_io_time_ms_(0.0) {
  std::cerr << "[IowarpEngine] DEBUG: Constructor entered, rank=" << rank_
            << ", name=" << name << std::endl;
  std::cerr.flush();

  // Initialize CTE client - assumes Chimaera runtime is already running
  std::cerr << "[IowarpEngine] DEBUG: About to call WRP_CTE_CLIENT_INIT" << std::endl;
  std::cerr.flush();
  wrp_cte::core::WRP_CTE_CLIENT_INIT("", chi::PoolQuery::Local());
  std::cerr << "[IowarpEngine] DEBUG: WRP_CTE_CLIENT_INIT completed" << std::endl;
  std::cerr.flush();

  // Read compression environment variables
  std::cerr << "[IowarpEngine] DEBUG: About to read compression env vars" << std::endl;
  std::cerr.flush();
  ReadCompressionEnvVars();
  std::cerr << "[IowarpEngine] DEBUG: Compression env vars read, mode=" << compress_mode_ << std::endl;
  std::cerr.flush();

#ifdef WRP_CTE_ENABLE_COMPRESS
  // Initialize compressor client if compression is enabled
  if (compress_mode_ != 0) {
    std::cerr << "[IowarpEngine] DEBUG: Creating compressor client" << std::endl;
    std::cerr.flush();
    compressor_client_ = std::make_unique<wrp_cte::compressor::Client>();
    std::cerr << "[IowarpEngine] DEBUG: Compressor client created, calling AsyncCreate" << std::endl;
    std::cerr.flush();
    // Create the compressor pool
    auto create_task = compressor_client_->AsyncCreate(
        chi::PoolQuery::Local(),
        "wrp_cte_compressor",
        chi::PoolId(513, 0));
    std::cerr << "[IowarpEngine] DEBUG: AsyncCreate called, about to Wait()" << std::endl;
    std::cerr.flush();
    create_task.Wait();
    std::cerr << "[IowarpEngine] DEBUG: Wait() completed, return_code=" << create_task->GetReturnCode() << std::endl;
    std::cerr.flush();
    if (create_task->GetReturnCode() != 0) {
      if (rank_ == 0) {
        std::cerr << "[IowarpEngine] Warning: Failed to create compressor pool, "
                  << "compression may not work correctly" << std::endl;
      }
    } else {
      if (rank_ == 0) {
        std::cerr << "[IowarpEngine] Compressor client initialized" << std::endl;
      }
    }
  }
#endif

  // Start wall clock timer
  wall_clock_start_ = std::chrono::high_resolution_clock::now();

  std::cerr << "[IowarpEngine] DEBUG: Constructor completed, starting timing measurement" << std::endl;
  std::cerr.flush();

  if (rank_ == 0) {
    std::cerr << "[IowarpEngine] Starting timing measurement" << std::endl;
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
    std::cerr << "\n========================================" << std::endl;
    std::cerr << "[IowarpEngine] Timing Summary" << std::endl;
    std::cerr << "========================================" << std::endl;
    std::cerr << "Total wall time:  " << total_wall_time_ms << " ms" << std::endl;
    std::cerr << "Total I/O time:   " << total_io_time_ms_ << " ms" << std::endl;
    std::cerr << "Compute time:     " << compute_time_ms << " ms" << std::endl;
    std::cerr << "I/O percentage:   " << (total_io_time_ms_ / total_wall_time_ms * 100.0) << "%" << std::endl;
    std::cerr << "Compute percentage: " << (compute_time_ms / total_wall_time_ms * 100.0) << "%" << std::endl;
    std::cerr << "========================================\n" << std::endl;
  }
}

/**
 * Initialize the engine
 */
void IowarpEngine::Init_() {
  std::cerr << "[IowarpEngine] DEBUG: Init_() entered, open_=" << open_ << std::endl;
  std::cerr.flush();

  if (open_) {
    throw std::runtime_error("IowarpEngine::Init_: Engine already initialized");
  }

  // Create or get tag for this ADIOS file/session
  // Use the engine name as the tag name
  try {
    std::cerr << "[IowarpEngine] DEBUG: About to create Tag with name=" << m_Name << std::endl;
    std::cerr.flush();
    current_tag_ = std::make_unique<wrp_cte::core::Tag>(m_Name);
    std::cerr << "[IowarpEngine] DEBUG: Tag created successfully" << std::endl;
    std::cerr.flush();
    open_ = true;
  } catch (const std::exception &e) {
    std::cerr << "[IowarpEngine] DEBUG: Tag creation failed: " << e.what() << std::endl;
    std::cerr.flush();
    throw std::runtime_error(
        std::string("IowarpEngine::Init_: Failed to create/get tag: ") +
        e.what());
  }

  std::cerr << "[IowarpEngine] DEBUG: Init_() completed" << std::endl;
  std::cerr.flush();
}

/**
 * Begin a new step
 * @param mode Step mode (Read, Append, Update)
 * @param timeoutSeconds Timeout in seconds (-1 for no timeout)
 * @return Step status
 */
adios2::StepStatus IowarpEngine::BeginStep(adios2::StepMode mode,
                                           const float timeoutSeconds) {
  std::cerr << "[IowarpEngine] DEBUG: BeginStep() entered, open_=" << open_ << std::endl;
  std::cerr.flush();

  (void)mode;            // Suppress unused parameter warning
  (void)timeoutSeconds;  // Suppress unused parameter warning

  // Lazy initialization if not already initialized
  if (!open_) {
    std::cerr << "[IowarpEngine] DEBUG: BeginStep() calling Init_()" << std::endl;
    std::cerr.flush();
    Init_();
    std::cerr << "[IowarpEngine] DEBUG: BeginStep() Init_() returned" << std::endl;
    std::cerr.flush();
  }

  // Increment step counter
  IncrementCurrentStep();
  std::cerr << "[IowarpEngine] DEBUG: BeginStep() completed, step=" << current_step_ << std::endl;
  std::cerr.flush();

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
  size_t total_original_size = 0;
  size_t total_compressed_size = 0;
  double total_compress_time_ms = 0.0;
  size_t num_tasks = 0;

  for (auto &deferred : deferred_tasks_) {
    // Set TASK_DATA_OWNER flag so task destructor will free the buffer
    auto *task_ptr = deferred.task.get();
    if (task_ptr != nullptr) {
      task_ptr->SetFlags(TASK_DATA_OWNER);
    }

    // Wait for task to complete
    deferred.task.Wait();

    // Extract compression statistics from context
    if (task_ptr != nullptr) {
      const auto &ctx = task_ptr->context_;
      if (ctx.actual_original_size_ > 0) {
        total_original_size += ctx.actual_original_size_;
        total_compressed_size += ctx.actual_compressed_size_;
        total_compress_time_ms += ctx.actual_compress_time_ms_;
        num_tasks++;

        // Log individual task compression stats if tracing is enabled
        if (rank_ == 0 && compress_trace_) {
          std::cerr << "[IowarpEngine] Compression stats: "
                    << "original=" << ctx.actual_original_size_ << " bytes, "
                    << "compressed=" << ctx.actual_compressed_size_ << " bytes, "
                    << "ratio=" << ctx.actual_compression_ratio_ << ", "
                    << "time=" << ctx.actual_compress_time_ms_ << " ms, "
                    << "PSNR=" << ctx.actual_psnr_db_ << " dB" << std::endl;
        }
      }
    }
  }

  // Clear the deferred tasks vector for the next step
  deferred_tasks_.clear();

  // Print aggregated compression statistics for this step
  if (rank_ == 0 && num_tasks > 0) {
    double overall_ratio = (total_compressed_size > 0)
                               ? static_cast<double>(total_original_size) / total_compressed_size
                               : 1.0;
    std::cerr << "[IowarpEngine] Step " << current_step_ << " compression summary: "
              << "tasks=" << num_tasks << ", "
              << "total_original=" << total_original_size << " bytes, "
              << "total_compressed=" << total_compressed_size << " bytes, "
              << "overall_ratio=" << overall_ratio << ", "
              << "total_compress_time=" << total_compress_time_ms << " ms" << std::endl;
  }

  // Measure and log I/O time
  auto io_end = std::chrono::high_resolution_clock::now();
  double io_time_ms = std::chrono::duration<double, std::milli>(io_end - io_start).count();

  // Accumulate total I/O time
  total_io_time_ms_ += io_time_ms;

  // Log per-step I/O time
  if (rank_ == 0) {
    std::cerr << "[IowarpEngine] Step " << current_step_
              << " I/O time: " << io_time_ms << " ms"
              << " (Total I/O: " << total_io_time_ms_ << " ms)" << std::endl;
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
 * Parse compression library name to ID
 * @param lib_str Library name string
 * @return Library ID or -1 if unknown
 */
int IowarpEngine::ParseCompressionLib(const std::string &lib_str) {
  std::string lower = lib_str;
  std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

  // Compression library IDs (from core_runtime.cc)
  if (lower == "brotli") return 0;
  if (lower == "bzip2") return 1;
  if (lower == "blosc2") return 2;
  if (lower == "fpzip") return 3;
  if (lower == "lz4") return 4;
  if (lower == "lzma") return 5;
  if (lower == "snappy") return 6;
  if (lower == "sz3") return 7;
  if (lower == "zfp") return 8;
  if (lower == "zlib") return 9;
  if (lower == "zstd") return 10;

  // Try parsing as integer
  try {
    return std::stoi(lib_str);
  } catch (...) {
    return 10;  // Default to ZSTD
  }
}

/**
 * Read compression environment variables
 *
 * Environment variables:
 *   IOWARP_COMPRESS: Compression type
 *     - "none": No compression
 *     - "dynamic": Dynamic compression (system chooses algorithm)
 *     - "<library>": Static compression with named library
 *       (zstd, lz4, zlib, snappy, brotli, blosc2, bzip2, lzma, fpzip, sz3, zfp)
 *
 *   IOWARP_COMPRESS_TRACE: Enable compression tracing
 *     - "on", "1", "true": Enable tracing
 *     - Otherwise: Disable tracing
 *
 * Legacy environment variables (for backward compatibility):
 *   COMPRESS_MODE: none (0), static (1), dynamic (2)
 *   COMPRESS_LIB: library ID for static compression
 *   COMPRESS_TRACE: on (true) or off (false)
 */
void IowarpEngine::ReadCompressionEnvVars() {
  // First check for new unified IOWARP_COMPRESS variable
  const char* compress_env = std::getenv("IOWARP_COMPRESS");
  if (compress_env != nullptr) {
    std::string compress_str(compress_env);
    std::string lower = compress_str;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    if (lower == "none" || lower == "off" || lower == "0") {
      compress_mode_ = 0;  // No compression
      compress_lib_ = 0;
    } else if (lower == "dynamic" || lower == "auto") {
      compress_mode_ = 2;  // Dynamic compression
      compress_lib_ = 10;  // Default to ZSTD for dynamic
    } else {
      // Treat as specific compression library name
      compress_mode_ = 1;  // Static compression
      compress_lib_ = ParseCompressionLib(compress_str);
    }

    // Log the compression configuration
    if (rank_ == 0) {
      const char* lib_names[] = {"brotli", "bzip2", "blosc2", "fpzip", "lz4",
                                  "lzma", "snappy", "sz3", "zfp", "zlib", "zstd"};
      std::string mode_name = (compress_mode_ == 0) ? "none" :
                              (compress_mode_ == 2) ? "dynamic" :
                              (compress_lib_ >= 0 && compress_lib_ <= 10) ?
                                lib_names[compress_lib_] : "unknown";
      std::cerr << "[IowarpEngine] Compression: " << mode_name << std::endl;
    }
  } else {
    // Fall back to legacy environment variables
    const char* mode_env = std::getenv("COMPRESS_MODE");
    if (mode_env != nullptr) {
      std::string mode_str(mode_env);
      if (mode_str == "none") {
        compress_mode_ = 0;
      } else if (mode_str == "static") {
        compress_mode_ = 1;
      } else if (mode_str == "dynamic") {
        compress_mode_ = 2;
      }
    }

    const char* lib_env = std::getenv("COMPRESS_LIB");
    if (lib_env != nullptr) {
      compress_lib_ = ParseCompressionLib(lib_env);
    }
  }

  // Read trace setting (check both new and legacy)
  const char* trace_env = std::getenv("IOWARP_COMPRESS_TRACE");
  if (trace_env == nullptr) {
    trace_env = std::getenv("COMPRESS_TRACE");
  }
  if (trace_env != nullptr) {
    std::string trace_str(trace_env);
    std::transform(trace_str.begin(), trace_str.end(), trace_str.begin(), ::tolower);
    compress_trace_ = (trace_str == "on" || trace_str == "1" || trace_str == "true");
  }
}

/**
 * Create Context object for Put operations based on environment settings
 * @return Context object configured from environment variables
 */
wrp_cte::core::Context IowarpEngine::CreateCompressionContext() {
  wrp_cte::core::Context context;

  context.dynamic_compress_ = compress_mode_;
  context.compress_lib_ = compress_lib_;
  context.trace_ = compress_trace_;

  // Set other context fields as needed
  context.target_psnr_ = 0;          // No PSNR requirement by default
  context.psnr_chance_ = 100;        // Always validate PSNR if required
  context.max_performance_ = false;  // Optimize for ratio by default
  context.consumer_node_ = -1;       // Unknown consumer node
  context.data_type_ = 0;            // Unknown data type

  return context;
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

  // Create compression context from environment variables
  auto context = CreateCompressionContext();

#ifdef WRP_CTE_ENABLE_COMPRESS
  // Call compressor for dynamic scheduling if compression is enabled
  if (compressor_client_ && compress_mode_ == 2) {
    compressor_client_->DynamicSchedule(
        data_size, const_cast<void*>(reinterpret_cast<const void*>(values)),
        context, chi::PoolQuery::Local());
  }
#endif

  // Put blob to CTE synchronously
  try {
    current_tag_->PutBlob(blob_name, reinterpret_cast<const char *>(values),
                          data_size, 0, 1.0f, context);
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
    auto *ipc_manager = CHI_IPC;
    if (ipc_manager == nullptr) {
      throw std::runtime_error("IowarpEngine::DoPutDeferred_: CHI_IPC is null");
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

    // Create compression context from environment variables
    auto context = CreateCompressionContext();

#ifdef WRP_CTE_ENABLE_COMPRESS
    // Call compressor for dynamic scheduling if compression is enabled
    if (compressor_client_ && compress_mode_ == 2) {
      compressor_client_->DynamicSchedule(
          data_size, buffer.ptr_, context, chi::PoolQuery::Local());
    }
#endif

    auto task = current_tag_->AsyncPutBlob(
        blob_name, buffer.shm_.template Cast<void>(), data_size, 0, 1.0f, context);

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
