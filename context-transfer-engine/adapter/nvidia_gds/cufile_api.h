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



/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Distributed under BSD 3-Clause license.                                   *
 * Copyright by The HDF Group.                                               *
 * Copyright by the Illinois Institute of Technology.                        *
 * All rights reserved.                                                      *
 *                                                                           *
 * This file is part of Clio. The full Clio copyright notice, including  *
 * terms governing use, modification, and redistribution, is contained in    *
 * the COPYING file, which can be found at the top directory. If you do not  *
 * have access to the file, you may request a copy from help@hdfgroup.org.   *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef CLIO_CTE_ADAPTER_NVIDIA_GDS_CUFILE_API_H_
#define CLIO_CTE_ADAPTER_NVIDIA_GDS_CUFILE_API_H_

#include <cuda_runtime.h>
#include <cufile.h>
#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

#include "clio_ctp/util/logging.h"
#include "clio_ctp/util/real_api.h"
#include "clio_ctp/util/singleton.h"

extern "C" {
typedef CUfileError_t (*cuFileHandleRegister_t)(CUfileHandle_t *,
                                                CUfileDescr_t *);
typedef void (*cuFileHandleDeregister_t)(CUfileHandle_t);
typedef CUfileError_t (*cuFileBufRegister_t)(const void *, size_t, int);
typedef CUfileError_t (*cuFileBufDeregister_t)(const void *);
typedef ssize_t (*cuFileRead_t)(CUfileHandle_t, void *, size_t, off_t, off_t);
typedef ssize_t (*cuFileWrite_t)(CUfileHandle_t, const void *, size_t, off_t,
                                 off_t);
typedef long (*cuFileUseCount_t)(void);
typedef CUfileError_t (*cuFileDriverGetProperties_t)(CUfileDrvProps_t *);
typedef CUfileError_t (*cuFileDriverSetPollMode_t)(bool, size_t);
typedef CUfileError_t (*cuFileDriverSetMaxDirectIOSize_t)(size_t);
typedef CUfileError_t (*cuFileDriverSetMaxCacheSize_t)(size_t);
typedef CUfileError_t (*cuFileDriverSetMaxPinnedMemSize_t)(size_t);
typedef CUfileError_t (*cuFileBatchIOSetUp_t)(CUfileBatchHandle_t *, unsigned);
typedef CUfileError_t (*cuFileBatchIOSubmit_t)(CUfileBatchHandle_t, unsigned,
                                               CUfileIOParams_t *,
                                               unsigned int);
typedef CUfileError_t (*cuFileBatchIOGetStatus_t)(CUfileBatchHandle_t, unsigned,
                                                  unsigned *,
                                                  CUfileIOEvents_t *,
                                                  struct timespec *);
typedef CUfileError_t (*cuFileBatchIOCancel_t)(CUfileBatchHandle_t);
typedef void (*cuFileBatchIODestroy_t)(CUfileBatchHandle_t);
}

namespace clio::cae {

using ctp::RealApi;

/** CuFileApi class */
class CuFileApi : public RealApi {
 public:
  cuFileHandleRegister_t cuFileHandleRegister = nullptr;
  cuFileHandleDeregister_t cuFileHandleDeregister = nullptr;
  cuFileBufRegister_t cuFileBufRegister = nullptr;
  cuFileBufDeregister_t cuFileBufDeregister = nullptr;
  cuFileRead_t cuFileRead = nullptr;
  cuFileWrite_t cuFileWrite = nullptr;
  cuFileUseCount_t cuFileUseCount = nullptr;
  cuFileDriverGetProperties_t cuFileDriverGetProperties = nullptr;
  cuFileDriverSetPollMode_t cuFileDriverSetPollMode = nullptr;
  cuFileDriverSetMaxDirectIOSize_t cuFileDriverSetMaxDirectIOSize = nullptr;
  cuFileDriverSetMaxCacheSize_t cuFileDriverSetMaxCacheSize = nullptr;
  cuFileDriverSetMaxPinnedMemSize_t cuFileDriverSetMaxPinnedMemSize = nullptr;
  cuFileBatchIOSetUp_t cuFileBatchIOSetUp = nullptr;
  cuFileBatchIOSubmit_t cuFileBatchIOSubmit = nullptr;
  cuFileBatchIOGetStatus_t cuFileBatchIOGetStatus = nullptr;
  cuFileBatchIOCancel_t cuFileBatchIOCancel = nullptr;
  cuFileBatchIODestroy_t cuFileBatchIODestroy = nullptr;

  CuFileApi() : RealApi("cuFileHandleRegister", "cuFile_Intercepted") {
    cuFileHandleRegister =
        (cuFileHandleRegister_t)dlsym(real_lib_, "cuFileHandleRegister");
    cuFileHandleDeregister =
        (cuFileHandleDeregister_t)dlsym(real_lib_, "cuFileHandleDeregister");
    cuFileBufRegister =
        (cuFileBufRegister_t)dlsym(real_lib_, "cuFileBufRegister");
    cuFileBufDeregister =
        (cuFileBufDeregister_t)dlsym(real_lib_, "cuFileBufDeregister");
    cuFileRead = (cuFileRead_t)dlsym(real_lib_, "cuFileRead");
    cuFileWrite = (cuFileWrite_t)dlsym(real_lib_, "cuFileWrite");
    cuFileUseCount = (cuFileUseCount_t)dlsym(real_lib_, "cuFileUseCount");
    cuFileDriverGetProperties = (cuFileDriverGetProperties_t)dlsym(
        real_lib_, "cuFileDriverGetProperties");
    cuFileDriverSetPollMode =
        (cuFileDriverSetPollMode_t)dlsym(real_lib_, "cuFileDriverSetPollMode");
    cuFileDriverSetMaxDirectIOSize = (cuFileDriverSetMaxDirectIOSize_t)dlsym(
        real_lib_, "cuFileDriverSetMaxDirectIOSize");
    cuFileDriverSetMaxCacheSize = (cuFileDriverSetMaxCacheSize_t)dlsym(
        real_lib_, "cuFileDriverSetMaxCacheSize");
    cuFileDriverSetMaxPinnedMemSize = (cuFileDriverSetMaxPinnedMemSize_t)dlsym(
        real_lib_, "cuFileDriverSetMaxPinnedMemSize");
    cuFileBatchIOSetUp =
        (cuFileBatchIOSetUp_t)dlsym(real_lib_, "cuFileBatchIOSetUp");
    cuFileBatchIOSubmit =
        (cuFileBatchIOSubmit_t)dlsym(real_lib_, "cuFileBatchIOSubmit");
    cuFileBatchIOGetStatus =
        (cuFileBatchIOGetStatus_t)dlsym(real_lib_, "cuFileBatchIOGetStatus");
    cuFileBatchIOCancel =
        (cuFileBatchIOCancel_t)dlsym(real_lib_, "cuFileBatchIOCancel");
    cuFileBatchIODestroy =
        (cuFileBatchIODestroy_t)dlsym(real_lib_, "cuFileBatchIODestroy");

    if (!cuFileHandleRegister || !cuFileHandleDeregister || !cuFileRead ||
        !cuFileWrite || !cuFileBufRegister || !cuFileBufDeregister) {
      HLOG(kError, "Failed to load cuFile symbols: {}", dlerror());
      exit(EXIT_FAILURE);
    }
  }
};

}  // namespace clio::cae

#include "clio_ctp/util/singleton.h"

// Singleton macros
#define CLIO_CTE_CUFILE_API \
  ctp::Singleton<::clio::cae::CuFileApi>::GetInstance()

#endif  // CLIO_CTE_ADAPTER_NVIDIA_GDS_CUFILE_API_H_
