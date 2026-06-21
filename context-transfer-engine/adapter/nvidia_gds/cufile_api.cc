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

bool cuFile_Intercepted = true;

#include "cufile_api.h"

#include <limits.h>
#include <sys/file.h>

#include <cstdio>

#include "adapter/posix/posix_api.h"

namespace stdfs = std::filesystem;

extern "C" {
// Interceptor functions
CUfileError_t cuFileHandleRegister(CUfileHandle_t *fh, CUfileDescr_t *descr) {
  //    printf("Intercepted the REAL API\n");
  (*fh) = descr;
  CUfileError_t ret;
  ret.err = CU_FILE_SUCCESS;
  return ret;
  // return CLIO_CTE_CUFILE_API->cuFileHandleRegister(fh, descr);
}

void cuFileHandleDeregister(CUfileHandle_t fh) {
  //    printf("Intercepted the REAL API\n");
  close(((CUfileDescr_t *)fh)->handle.fd);
  // CLIO_CTE_CUFILE_API->cuFileHandleDeregister(fh);
}

CUfileError_t cuFileBufRegister(const void *buf, size_t size, int flags) {
  //    printf("Intercepted the REAL API\n");
  return CLIO_CTE_CUFILE_API->cuFileBufRegister(buf, size, flags);
}

CUfileError_t cuFileBufDeregister(const void *buf) {
  //    printf("Intercepted the REAL API\n");
  return CLIO_CTE_CUFILE_API->cuFileBufDeregister(buf);
}

ssize_t cuFileRead(CUfileHandle_t fh, void *buf, size_t size, off_t offset,
                   off_t offset2) {
  //    printf("Intercepted the REAL API\n");
  char *host_data = (char *)malloc(size);
  CUfileDescr_t *descr = (CUfileDescr_t *)fh;
  ssize_t ret = read(descr->handle.fd, host_data, size);
  cudaMemcpy(buf, host_data, size, cudaMemcpyHostToDevice);
  free(host_data);
  return ret;
  // return CLIO_CTE_CUFILE_API->cuFileRead(fh, buf, size, offset, offset2);
}

ssize_t cuFileWrite(CUfileHandle_t fh, const void *buf, size_t size,
                    off_t offset, off_t offset2) {
  //    printf("Intercepted the REAL API\n");
  // Read data from GPU using cudaMemcpy
  // FullPtr<char> p = CLIO_CLIENT->AllocateBuffer(size);
  char *host_data = (char *)malloc(size);
  cudaMemcpy(host_data, buf, size, cudaMemcpyDeviceToHost);
  // Write data to Clio
  CUfileDescr_t *descr = (CUfileDescr_t *)fh;
  ssize_t ret = write(descr->handle.fd, host_data, size);
  free(host_data);
  return ret;
  // return CLIO_CTE_CUFILE_API->cuFileWrite(fh, buf, size, offset, offset2);
}

long cuFileUseCount() {
  //    printf("Intercepted the REAL API\n");
  return CLIO_CTE_CUFILE_API->cuFileUseCount();
}

CUfileError_t cuFileDriverGetProperties(CUfileDrvProps_t *props) {
  //    printf("Intercepted the REAL API\n");
  return CLIO_CTE_CUFILE_API->cuFileDriverGetProperties(props);
}

CUfileError_t cuFileDriverSetPollMode(bool poll_mode,
                                      size_t poll_threshold_size) {
  //    printf("Intercepted the REAL API\n");
  return CLIO_CTE_CUFILE_API->cuFileDriverSetPollMode(poll_mode,
                                                    poll_threshold_size);
}

CUfileError_t cuFileDriverSetMaxDirectIOSize(size_t size) {
  //    printf("Intercepted the REAL API\n");
  return CLIO_CTE_CUFILE_API->cuFileDriverSetMaxDirectIOSize(size);
}

CUfileError_t cuFileDriverSetMaxCacheSize(size_t size) {
  //    printf("Intercepted the REAL API\n");
  return CLIO_CTE_CUFILE_API->cuFileDriverSetMaxCacheSize(size);
}

CUfileError_t cuFileDriverSetMaxPinnedMemSize(size_t size) {
  //    printf("Intercepted the REAL API\n");
  return CLIO_CTE_CUFILE_API->cuFileDriverSetMaxPinnedMemSize(size);
}

CUfileError_t cuFileBatchIOSetUp(CUfileBatchHandle_t *handle, unsigned flags) {
  //    printf("Intercepted the REAL API\n");
  return CLIO_CTE_CUFILE_API->cuFileBatchIOSetUp(handle, flags);
}

CUfileError_t cuFileBatchIOSubmit(CUfileBatchHandle_t handle, unsigned num_ios,
                                  CUfileIOParams_t *io_params,
                                  unsigned int flags) {
  //    printf("Intercepted the REAL API\n");
  return CLIO_CTE_CUFILE_API->cuFileBatchIOSubmit(handle, num_ios, io_params,
                                                flags);
}

CUfileError_t cuFileBatchIOGetStatus(CUfileBatchHandle_t handle,
                                     unsigned num_ios, unsigned *num_completed,
                                     CUfileIOEvents_t *events,
                                     struct timespec *timeout) {
  //    printf("Intercepted the REAL API\n");
  return CLIO_CTE_CUFILE_API->cuFileBatchIOGetStatus(
      handle, num_ios, num_completed, events, timeout);
}

CUfileError_t cuFileBatchIOCancel(CUfileBatchHandle_t handle) {
  //    printf("Intercepted the REAL API\n");
  return CLIO_CTE_CUFILE_API->cuFileBatchIOCancel(handle);
}

void cuFileBatchIODestroy(CUfileBatchHandle_t handle) {
  //    printf("Intercepted the REAL API\n");
  CLIO_CTE_CUFILE_API->cuFileBatchIODestroy(handle);
}
}  // extern C
