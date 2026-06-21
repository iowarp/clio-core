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

#pragma once

// Include ADIOS2 headers FIRST to avoid macro conflicts
#include <adios2.h>
#include <adios2/engine/plugin/PluginEngineInterface.h>

// Then include IOWarp headers
#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>

namespace coeus {

class IowarpEngine : public adios2::plugin::PluginEngineInterface {
 public:
  /** Construct the IowarpEngine */
  IowarpEngine(adios2::core::IO &io,  // NOLINT
               const std::string &name, const adios2::Mode mode,
               adios2::helper::Comm comm);

  /** Destructor */
  ~IowarpEngine() override;

  /**
   * Define the beginning of a step. A step is typically the offset from
   * the beginning of a file. It is measured as a size_t.
   *
   * Logically, a "step" represents a snapshot of the data at a specific time,
   * and can be thought of as a frame in a video or a snapshot of a simulation.
   * */
  adios2::StepStatus BeginStep(adios2::StepMode mode,
                               const float timeoutSeconds = -1.0) override;

  /** Define the end of a step */
  void EndStep() override;

  /**
   * Returns the current step
   * */
  size_t CurrentStep() const override;

 protected:
  /** Initialize parameters */
  void InitParameters() override {}

  /** Initialize transports */
  void InitTransports() override {}

  /** Initialize (wrapper around Init_)*/
  void Init() override { Init_(); }

  /** Actual engine initialization */
  void Init_();

  /** Close a particular transport */
  void DoClose(const int transportIndex = -1) override;

  /** Place data in CTE */
  template <typename T>
  void DoPutSync_(const adios2::core::Variable<T> &variable, const T *values);

  /** Place data in CTE asynchronously */
  template <typename T>
  void DoPutDeferred_(const adios2::core::Variable<T> &variable,
                      const T *values);

  /** Get data from CTE (sync) */
  template <typename T>
  void DoGetSync_(const adios2::core::Variable<T> &variable, T *values);

  /** Get data from CTE (async) */
  template <typename T>
  void DoGetDeferred_(const adios2::core::Variable<T> &variable, T *values);

#define declare_type(T)                                                     \
  void DoPutSync(adios2::core::Variable<T> &variable, const T *values)      \
      override {                                                            \
    DoPutSync_(variable, values);                                           \
  }                                                                         \
  void DoPutDeferred(adios2::core::Variable<T> &variable, const T *values)  \
      override {                                                            \
    DoPutDeferred_(variable, values);                                       \
  }                                                                         \
  void DoGetSync(adios2::core::Variable<T> &variable, T *values) override { \
    DoGetSync_(variable, values);                                           \
  }                                                                         \
  void DoGetDeferred(adios2::core::Variable<T> &variable, T *values)        \
      override {                                                            \
    DoGetDeferred_(variable, values);                                       \
  }
  ADIOS2_FOREACH_STDTYPE_1ARG(declare_type)
#undef declare_type

 private:
  /** Structure to hold deferred task and its buffer */
  struct DeferredTask {
    clio::run::Future<clio::cte::core::PutBlobTask> task;
    ctp::ipc::FullPtr<char> buffer;
  };

  /** CTE Tag for this ADIOS file/session */
  std::unique_ptr<clio::cte::core::Tag> current_tag_;

  /** Current step counter */
  size_t current_step_;

  /** Process rank */
  int rank_;

  /** Engine open status */
  bool open_;

  /** Vector of deferred put tasks for current step */
  std::vector<DeferredTask> deferred_tasks_;

  /** Total I/O time in milliseconds */
  double total_io_time_ms_;

  /** Wall clock start time */
  std::chrono::high_resolution_clock::time_point wall_clock_start_;

  /** Increment the current step */
  void IncrementCurrentStep() { current_step_++; }

};

}  // namespace coeus

/**
 * This is how ADIOS figures out where to dynamically load the engine.
 * */
extern "C" {

/** C wrapper to create engine */
coeus::IowarpEngine *EngineCreate(adios2::core::IO &io,  // NOLINT
                                  const std::string &name,
                                  const adios2::Mode mode,
                                  adios2::helper::Comm comm);

/** C wrapper to destroy engine */
void EngineDestroy(coeus::IowarpEngine *obj);
}