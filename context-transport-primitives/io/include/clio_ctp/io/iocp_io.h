/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

#ifndef CTP_SHM_INCLUDE_HSHM_SHM_IO_IOCP_IO_H_
#define CTP_SHM_INCLUDE_HSHM_SHM_IO_IOCP_IO_H_

#ifdef _WIN32

#include "async_io.h"

#include <atomic>
#include <cstdint>
#include <memory>

#include "clio_ctp/constants/macros.h"

namespace ctp {

/**
 * Windows IOCP-based AsyncIO. Counterpart of LinuxAioAsyncIO.
 *
 * The implementation lives entirely in iocp_io.cc — this header is
 * deliberately Win32-free so consumers don't drag <windows.h> macros
 * (Yield, SendMessage, min, max, ...) into their translation units.
 */
class IocpAsyncIO : public AsyncIO {
 public:
  CTP_DLL explicit IocpAsyncIO(uint32_t io_depth);
  CTP_DLL ~IocpAsyncIO() override;

  CTP_DLL bool Open(const std::string &path, int flags, mode_t mode) override;
  CTP_DLL ssize_t GetFileSize() const override;
  CTP_DLL bool Truncate(size_t size) override;
  CTP_DLL IoToken Write(void *buffer, size_t size, int64_t offset) override;
  CTP_DLL IoToken Read(void *buffer, size_t size, int64_t offset) override;
  CTP_DLL bool IsComplete(IoToken token, IoResult &result) override;
  CTP_DLL void Close() override;
  CTP_DLL int GetEventFd() const override;

 private:
  struct Impl;                 // defined in iocp_io.cc
  std::unique_ptr<Impl> impl_;
};

}  // namespace ctp

#endif  // _WIN32

#endif  // CTP_SHM_INCLUDE_HSHM_SHM_IO_IOCP_IO_H_
