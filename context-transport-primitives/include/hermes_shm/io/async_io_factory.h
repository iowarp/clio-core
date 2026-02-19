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

#ifndef HSHM_SHM_INCLUDE_HSHM_SHM_IO_ASYNC_IO_FACTORY_H_
#define HSHM_SHM_INCLUDE_HSHM_SHM_IO_ASYNC_IO_FACTORY_H_

#include <memory>
#include "async_io.h"

#if HSHM_ENABLE_LIBAIO
#include "libaio_io.h"
#endif

#if HSHM_ENABLE_IO_URING
#include "iouring_io.h"
#endif

#if !defined(_WIN32)
#include "posix_aio.h"
#endif

#ifdef _WIN32
#include "iocp_io.h"
#endif

namespace hshm {

enum class AsyncIoBackend {
  kLinuxAio,   /**< libaio (Linux) */
  kIoUring,    /**< io_uring (Linux 5.1+) */
  kPosixAio,   /**< POSIX aio_read/aio_write */
  kIocp,       /**< Windows I/O Completion Ports */
  kDefault     /**< Auto-select best available for platform */
};

class AsyncIoFactory {
 public:
  static std::unique_ptr<AsyncIO> Get(
      uint32_t io_depth,
      AsyncIoBackend backend = AsyncIoBackend::kDefault) {
    if (backend == AsyncIoBackend::kDefault) {
      backend = GetDefaultBackend();
    }

    switch (backend) {
#if HSHM_ENABLE_LIBAIO
      case AsyncIoBackend::kLinuxAio:
        return std::make_unique<LinuxAioAsyncIO>(io_depth);
#endif

#if HSHM_ENABLE_IO_URING
      case AsyncIoBackend::kIoUring:
        return std::make_unique<IoUringAsyncIO>(io_depth);
#endif

#if !defined(_WIN32)
      case AsyncIoBackend::kPosixAio:
        return std::make_unique<PosixAsyncIO>(io_depth);
#endif

#ifdef _WIN32
      case AsyncIoBackend::kIocp:
        return std::make_unique<IocpAsyncIO>(io_depth);
#endif

      default:
        return nullptr;
    }
  }

 private:
  static AsyncIoBackend GetDefaultBackend() {
#if HSHM_ENABLE_IO_URING
    return AsyncIoBackend::kIoUring;
#elif HSHM_ENABLE_LIBAIO
    return AsyncIoBackend::kLinuxAio;
#elif defined(_WIN32)
    return AsyncIoBackend::kIocp;
#else
    return AsyncIoBackend::kPosixAio;
#endif
  }
};

}  // namespace hshm

#endif  // HSHM_SHM_INCLUDE_HSHM_SHM_IO_ASYNC_IO_FACTORY_H_
