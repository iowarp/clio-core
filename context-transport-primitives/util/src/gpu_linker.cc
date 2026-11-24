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

#include "clio_ctp/util/gpu_linker.h"
#include "clio_ctp/util/logging.h"

#if CTP_ENABLE_CUDA
#include <cuda.h>
#endif

namespace ctp {

GpuLinker::GpuLinker() = default;

GpuLinker::~GpuLinker() {
  Unload();
}

void GpuLinker::AddModule(const std::string &name, const void *fatbin,
                          size_t size) {
  modules_.push_back({name, fatbin, size});
  HLOG(kInfo, "GpuLinker: Added module '{}' ({} bytes)", name, size);
}

bool GpuLinker::Link() {
#if CTP_ENABLE_CUDA
  if (modules_.empty()) {
    HLOG(kError, "GpuLinker::Link: No modules registered");
    return false;
  }

  Unload();

  // cuInit is idempotent
  CUresult res = cuInit(0);
  if (res != CUDA_SUCCESS) {
    const char *err_str = nullptr;
    cuGetErrorString(res, &err_str);
    HLOG(kError, "GpuLinker::Link: cuInit failed: {}", err_str ? err_str : "unknown");
    return false;
  }

  // Get current context (created by CUDA runtime)
  CUcontext ctx = nullptr;
  res = cuCtxGetCurrent(&ctx);
  if (res != CUDA_SUCCESS || ctx == nullptr) {
    // Push primary context for device 0
    CUdevice dev;
    cuDeviceGet(&dev, 0);
    cuDevicePrimaryCtxRetain(&ctx, dev);
    cuCtxSetCurrent(ctx);
  }

  // Set up JIT log buffers
  static const size_t kLogSize = 8192;
  char info_log[kLogSize] = {};
  char error_log[kLogSize] = {};

  CUjit_option options[] = {
    CU_JIT_INFO_LOG_BUFFER,
    CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
    CU_JIT_ERROR_LOG_BUFFER,
    CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
  };
  void *option_vals[] = {
    info_log,
    reinterpret_cast<void *>(kLogSize),
    error_log,
    reinterpret_cast<void *>(kLogSize),
  };

  CUlinkState link_state;
  res = cuLinkCreate(4, options, option_vals, &link_state);
  if (res != CUDA_SUCCESS) {
    const char *err_str = nullptr;
    cuGetErrorString(res, &err_str);
    HLOG(kError, "GpuLinker::Link: cuLinkCreate failed: {}", err_str ? err_str : "unknown");
    return false;
  }

  // Add each module's fatbin
  for (const auto &mod : modules_) {
    res = cuLinkAddData(link_state, CU_JIT_INPUT_FATBINARY,
                        const_cast<void *>(mod.data), mod.size,
                        mod.name.c_str(), 0, nullptr, nullptr);
    if (res != CUDA_SUCCESS) {
      const char *err_str = nullptr;
      cuGetErrorString(res, &err_str);
      HLOG(kError, "GpuLinker::Link: cuLinkAddData failed for '{}': {} | JIT error: {}",
            mod.name, err_str ? err_str : "unknown", error_log);
      cuLinkDestroy(link_state);
      return false;
    }
  }

  // Complete linking
  void *cubin = nullptr;
  size_t cubin_size = 0;
  res = cuLinkComplete(link_state, &cubin, &cubin_size);
  if (res != CUDA_SUCCESS) {
    const char *err_str = nullptr;
    cuGetErrorString(res, &err_str);
    HLOG(kError, "GpuLinker::Link: cuLinkComplete failed: {} | JIT error: {}",
          err_str ? err_str : "unknown", error_log);
    cuLinkDestroy(link_state);
    return false;
  }

  if (info_log[0] != '\0') {
    HLOG(kInfo, "GpuLinker JIT info: {}", info_log);
  }

  // Load the cubin into a CUmodule
  CUmodule module;
  res = cuModuleLoadData(&module, cubin);
  if (res != CUDA_SUCCESS) {
    const char *err_str = nullptr;
    cuGetErrorString(res, &err_str);
    HLOG(kError, "GpuLinker::Link: cuModuleLoadData failed: {}", err_str ? err_str : "unknown");
    cuLinkDestroy(link_state);
    return false;
  }

  cu_module_ = reinterpret_cast<void *>(module);
  linked_ = true;

  cuLinkDestroy(link_state);

  HLOG(kInfo, "GpuLinker::Link: Successfully linked {} modules ({} byte cubin)",
        modules_.size(), cubin_size);
  return true;
#else
  HLOG(kError, "GpuLinker::Link: CUDA not enabled");
  return false;
#endif
}

void *GpuLinker::GetFunction(const char *kernel_name) {
#if CTP_ENABLE_CUDA
  if (!linked_ || !cu_module_) {
    HLOG(kError, "GpuLinker::GetFunction: Not linked");
    return nullptr;
  }

  CUfunction func;
  CUresult res = cuModuleGetFunction(&func, reinterpret_cast<CUmodule>(cu_module_),
                                     kernel_name);
  if (res != CUDA_SUCCESS) {
    const char *err_str = nullptr;
    cuGetErrorString(res, &err_str);
    HLOG(kError, "GpuLinker::GetFunction: '{}' not found: {}",
          kernel_name, err_str ? err_str : "unknown");
    return nullptr;
  }

  return reinterpret_cast<void *>(func);
#else
  return nullptr;
#endif
}

bool GpuLinker::LaunchKernel(void *func, unsigned gridX, unsigned gridY,
                             unsigned gridZ, unsigned blockX, unsigned blockY,
                             unsigned blockZ, unsigned sharedMem,
                             void *stream, void **params) {
#if CTP_ENABLE_CUDA
  if (!func) {
    HLOG(kError, "GpuLinker::LaunchKernel: null function");
    return false;
  }

  CUresult res = cuLaunchKernel(
      reinterpret_cast<CUfunction>(func),
      gridX, gridY, gridZ,
      blockX, blockY, blockZ,
      sharedMem,
      static_cast<CUstream>(stream),
      params, nullptr);

  if (res != CUDA_SUCCESS) {
    const char *err_str = nullptr;
    cuGetErrorString(res, &err_str);
    HLOG(kError, "GpuLinker::LaunchKernel failed: {}", err_str ? err_str : "unknown");
    return false;
  }
  return true;
#else
  return false;
#endif
}

void GpuLinker::Unload() {
#if CTP_ENABLE_CUDA
  if (cu_module_) {
    cuModuleUnload(reinterpret_cast<CUmodule>(cu_module_));
    cu_module_ = nullptr;
  }
#endif
  linked_ = false;
}

bool GpuLinker::IsLinked() const {
  return linked_;
}

}  // namespace ctp
