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

#include <cerrno>
#include <cstdio>

// pread/pwrite below resolve to whatever is interposed (the POSIX adapter
// when it is preloaded), so a clio:: path reaches the CTE.
#include <unistd.h>

namespace {

/**
 * Bounce-buffer one cuFile transfer between a file range and device memory.
 * Batch I/O (cuFileBatchIO*) is NOT routed here; it goes to the real cuFile
 * library unchanged.
 * @param fh the registered handle (a heap copy of the caller's descriptor)
 * @param dev_base device buffer base the caller passed
 * @param size bytes to transfer
 * @param file_offset file offset of the transfer
 * @param dev_offset offset into dev_base
 * @param to_device true: file -> device (read); false: device -> file
 * @return bytes transferred, -1 with errno set on a file error, or
 *         -CU_FILE_CUDA_DRIVER_ERROR when the device copy failed
 */
ssize_t BounceIo(CUfileHandle_t fh, char *dev_base, size_t size,
                 off_t file_offset, off_t dev_offset, bool to_device) {
  if (fh == nullptr || dev_base == nullptr) {
    errno = EINVAL;
    return -1;
  }
  if (size == 0) return 0;
  const int fd = static_cast<CUfileDescr_t *>(fh)->handle.fd;
  char *dev = dev_base + dev_offset;
  char *host = static_cast<char *>(malloc(size));
  if (host == nullptr) {
    errno = ENOMEM;
    return -1;
  }
  ssize_t ret;
  if (to_device) {
    // At file_offset, into dev_base + devPtr_offset: both offsets used to be
    // ignored, so every transfer hit the fd's position and the buffer's
    // start (#1191).
    ret = pread(fd, host, size, file_offset);
    if (ret > 0 &&
        cudaMemcpy(dev, host, static_cast<size_t>(ret),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      ret = -static_cast<ssize_t>(CU_FILE_CUDA_DRIVER_ERROR);
    }
  } else {
    if (cudaMemcpy(host, dev, size, cudaMemcpyDeviceToHost) != cudaSuccess) {
      ret = -static_cast<ssize_t>(CU_FILE_CUDA_DRIVER_ERROR);
    } else {
      ret = pwrite(fd, host, size, file_offset);
    }
  }
  const int saved = errno;
  free(host);
  errno = saved;
  return ret;
}

}  // namespace

extern "C" {
// Interceptor functions
CUfileError_t cuFileHandleRegister(CUfileHandle_t *fh, CUfileDescr_t *descr) {
  CUfileError_t ret;
  if (fh == nullptr || descr == nullptr) {
    ret.err = CU_FILE_INVALID_VALUE;
    return ret;
  }
  // Keep a copy: the caller's descriptor is commonly a stack variable that
  // is gone by the first cuFileRead.
  *fh = new CUfileDescr_t(*descr);
  ret.err = CU_FILE_SUCCESS;
  return ret;
}

void cuFileHandleDeregister(CUfileHandle_t fh) {
  // The fd belongs to the caller, as with the real cuFile: deregistering
  // must not close it.
  delete static_cast<CUfileDescr_t *>(fh);
}

CUfileError_t cuFileBufRegister(const void *buf, size_t size, int flags) {
  return CLIO_CTE_CUFILE_API->cuFileBufRegister(buf, size, flags);
}

CUfileError_t cuFileBufDeregister(const void *buf) {
  return CLIO_CTE_CUFILE_API->cuFileBufDeregister(buf);
}

ssize_t cuFileRead(CUfileHandle_t fh, void *devPtr_base, size_t size,
                   off_t file_offset, off_t devPtr_offset) {
  return BounceIo(fh, static_cast<char *>(devPtr_base), size, file_offset,
                  devPtr_offset, /*to_device=*/true);
}

ssize_t cuFileWrite(CUfileHandle_t fh, const void *devPtr_base, size_t size,
                    off_t file_offset, off_t devPtr_offset) {
  return BounceIo(fh, const_cast<char *>(static_cast<const char *>(devPtr_base)),
                  size, file_offset, devPtr_offset, /*to_device=*/false);
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
