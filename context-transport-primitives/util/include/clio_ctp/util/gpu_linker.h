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

#ifndef CTP_UTIL_GPU_LINKER_H
#define CTP_UTIL_GPU_LINKER_H

#include <string>
#include <vector>
#include "clio_ctp/constants/macros.h"

namespace ctp {

struct GpuDeviceCode {
  std::string name;
  const void *data;
  size_t size;
};

class GpuLinker {
 public:
  CTP_DLL GpuLinker();
  CTP_DLL ~GpuLinker();

  GpuLinker(const GpuLinker &) = delete;
  GpuLinker &operator=(const GpuLinker &) = delete;

  CTP_DLL void AddModule(const std::string &name, const void *fatbin,
                           size_t size);
  CTP_DLL bool Link();
  CTP_DLL void *GetFunction(const char *kernel_name);
  CTP_DLL bool LaunchKernel(void *func, unsigned gridX, unsigned gridY,
                              unsigned gridZ, unsigned blockX, unsigned blockY,
                              unsigned blockZ, unsigned sharedMem,
                              void *stream, void **params);
  CTP_DLL void Unload();
  CTP_DLL bool IsLinked() const;

 private:
  std::vector<GpuDeviceCode> modules_;
  void *cu_module_ = nullptr;
  bool linked_ = false;
};

}  // namespace ctp

#endif  // CTP_UTIL_GPU_LINKER_H
