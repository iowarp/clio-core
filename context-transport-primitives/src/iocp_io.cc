/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 *
 * Windows IOCP backend for ctp::AsyncIO. All <windows.h> usage is
 * contained in this translation unit — the header (iocp_io.h) is Win32-
 * free so consumers don't pick up Yield/SendMessage/min/max macros.
 */

#ifdef _WIN32

#include "clio_ctp/io/iocp_io.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <winioctl.h>
#ifdef Yield
#undef Yield
#endif
#ifdef SendMessage
#undef SendMessage
#endif
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

#include <fcntl.h>

namespace ctp {

namespace {
constexpr ULONG_PTR kCompletionKey = 1;
}  // namespace

struct IocpAsyncIO::Impl {
  HANDLE iocp = nullptr;
  HANDLE regular_fd = INVALID_HANDLE_VALUE;
  HANDLE direct_fd = INVALID_HANDLE_VALUE;
  uint32_t io_depth = 0;
  std::atomic<IoToken> next_token{1};
  std::mutex mutex;
  std::unordered_set<IoToken> in_flight;
  std::unordered_map<IoToken, IoResult> completed;

  // Per-operation tracking. OVERLAPPED must be the first field so a
  // pointer-to-IoOp is a valid pointer-to-OVERLAPPED for Win32.
  struct IoOp {
    OVERLAPPED overlapped;
    IoToken token;
    bool is_write;
  };

  void HarvestCompletions() {
    // Called with `mutex` held.
    if (!iocp) return;
    constexpr ULONG kMax = 32;
    OVERLAPPED_ENTRY entries[kMax];
    ULONG removed = 0;
    if (!::GetQueuedCompletionStatusEx(iocp, entries, kMax, &removed,
                                        /*timeout=*/0,
                                        /*alertable=*/FALSE)) {
      return;
    }
    for (ULONG i = 0; i < removed; ++i) {
      auto *op = reinterpret_cast<IoOp *>(entries[i].lpOverlapped);
      if (!op) continue;
      IoResult res;
      res.bytes_transferred =
          static_cast<ssize_t>(entries[i].dwNumberOfBytesTransferred);
      LONG status = static_cast<LONG>(op->overlapped.Internal);
      if (status != 0) {
        res.bytes_transferred = -1;
        res.error_code = static_cast<int>(status);
      } else {
        res.error_code = 0;
      }
      completed[op->token] = res;
      delete op;
    }
  }

  void DrainAllCompletions() {
    constexpr ULONG kMax = 64;
    OVERLAPPED_ENTRY entries[kMax];
    ULONG removed = 0;
    while (::GetQueuedCompletionStatusEx(iocp, entries, kMax, &removed,
                                          /*timeout=*/0,
                                          /*alertable=*/FALSE) &&
           removed > 0) {
      for (ULONG i = 0; i < removed; ++i) {
        delete reinterpret_cast<IoOp *>(entries[i].lpOverlapped);
      }
    }
  }

  /**
   * Pick the handle for one I/O. FILE_FLAG_NO_BUFFERING requires the buffer
   * address, the transfer size AND the file offset to be sector-aligned;
   * any one of them unaligned makes ReadFile/WriteFile fail with
   * ERROR_INVALID_PARAMETER (87), so all three are checked.
   * @param buffer user buffer
   * @param size transfer size in bytes
   * @param offset file offset in bytes
   * @return the unbuffered handle when legal, else the buffered one
   */
  HANDLE SelectHandle(void *buffer, size_t size, int64_t offset) const {
    if (direct_fd != INVALID_HANDLE_VALUE &&
        (reinterpret_cast<uintptr_t>(buffer) % 4096 == 0) &&
        (size % 4096 == 0) && (offset % 4096 == 0)) {
      return direct_fd;
    }
    return regular_fd;
  }

  IoToken SubmitIO(void *buffer, size_t size, int64_t offset, bool is_write) {
    std::lock_guard<std::mutex> lock(mutex);

    // A negative offset or a transfer wider than a DWORD cannot be expressed
    // in one OVERLAPPED ReadFile/WriteFile call (#1059).
    if (offset < 0 || size > static_cast<size_t>(MAXDWORD)) {
      return kInvalidIoToken;
    }
    HANDLE h = SelectHandle(buffer, size, offset);
    if (h == INVALID_HANDLE_VALUE) return kInvalidIoToken;

    IoToken token = next_token.fetch_add(1, std::memory_order_relaxed);

    auto *op = new IoOp{};
    op->overlapped.Offset = static_cast<DWORD>(offset & 0xFFFFFFFF);
    op->overlapped.OffsetHigh =
        static_cast<DWORD>((static_cast<uint64_t>(offset) >> 32) & 0xFFFFFFFF);
    op->token = token;
    op->is_write = is_write;

    BOOL ok;
    DWORD bytes_immediate = 0;
    if (is_write) {
      ok = ::WriteFile(h, buffer, static_cast<DWORD>(size),
                        &bytes_immediate, &op->overlapped);
    } else {
      ok = ::ReadFile(h, buffer, static_cast<DWORD>(size),
                       &bytes_immediate, &op->overlapped);
    }

    if (!ok) {
      DWORD err = ::GetLastError();
      if (err != ERROR_IO_PENDING) {
        IoResult res;
        res.bytes_transferred = -1;
        res.error_code = static_cast<int>(err);
        completed[token] = res;
        delete op;
        in_flight.insert(token);
        return token;
      }
    }
    in_flight.insert(token);
    return token;
  }
};

namespace {

/**
 * Mark a file sparse (FSCTL_SET_SPARSE), best effort.
 *
 * The handle is FILE_FLAG_OVERLAPPED, so DeviceIoControl needs a real
 * OVERLAPPED; the call must happen before the handle is bound to the
 * completion port so its completion is not delivered there.
 * @param h overlapped file handle not yet associated with an IOCP
 * @return true if the filesystem accepted the sparse attribute
 */
bool MarkSparse(HANDLE h) {
  OVERLAPPED ov{};
  ov.hEvent = ::CreateEventA(nullptr, TRUE, FALSE, nullptr);
  if (!ov.hEvent) return false;
  DWORD bytes = 0;
  BOOL ok = ::DeviceIoControl(h, FSCTL_SET_SPARSE, nullptr, 0, nullptr, 0,
                              &bytes, &ov);
  if (!ok && ::GetLastError() == ERROR_IO_PENDING) {
    ok = ::GetOverlappedResult(h, &ov, &bytes, TRUE);
  }
  ::CloseHandle(ov.hEvent);
  return ok != FALSE;
}

}  // namespace

IocpAsyncIO::IocpAsyncIO(uint32_t io_depth) : impl_(std::make_unique<Impl>()) {
  impl_->io_depth = io_depth;
}

IocpAsyncIO::~IocpAsyncIO() { Close(); }

bool IocpAsyncIO::Open(const std::string &path, int flags, mode_t mode) {
  (void)mode;  // POSIX mode bits aren't directly used on Windows.
  std::lock_guard<std::mutex> lock(impl_->mutex);

  DWORD access = 0;
  if ((flags & O_RDWR) == O_RDWR) {
    access = GENERIC_READ | GENERIC_WRITE;
  } else if ((flags & O_WRONLY) == O_WRONLY) {
    access = GENERIC_WRITE;
  } else {
    access = GENERIC_READ;
  }
  DWORD creation = (flags & O_CREAT) ? OPEN_ALWAYS : OPEN_EXISTING;
  if (flags & O_TRUNC) creation = CREATE_ALWAYS;

  DWORD share = FILE_SHARE_READ | FILE_SHARE_WRITE;

  impl_->regular_fd = ::CreateFileA(path.c_str(), access, share, nullptr,
                                     creation, FILE_FLAG_OVERLAPPED, nullptr);
  if (impl_->regular_fd == INVALID_HANDLE_VALUE) return false;

  // Mark the file sparse so extending it (Truncate -> SetEndOfFile) only
  // reserves logical size; clusters are allocated as blocks are actually
  // written, matching ftruncate on Linux (#1100). Without this, NTFS
  // allocates -- and charges quota for -- the whole extent up front.
  // Best effort: FAT/exFAT have no sparse support, and then we simply get
  // the old eager behaviour.
  MarkSparse(impl_->regular_fd);

  impl_->direct_fd = ::CreateFileA(
      path.c_str(), access, share, nullptr, OPEN_EXISTING,
      FILE_FLAG_OVERLAPPED | FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH,
      nullptr);
  // direct_fd may be INVALID_HANDLE_VALUE; that's fine, we fall back to
  // the regular (buffered) handle for every op.

  impl_->iocp = ::CreateIoCompletionPort(impl_->regular_fd, nullptr,
                                          kCompletionKey, 0);
  if (!impl_->iocp) {
    ::CloseHandle(impl_->regular_fd);
    impl_->regular_fd = INVALID_HANDLE_VALUE;
    if (impl_->direct_fd != INVALID_HANDLE_VALUE) {
      ::CloseHandle(impl_->direct_fd);
      impl_->direct_fd = INVALID_HANDLE_VALUE;
    }
    return false;
  }
  if (impl_->direct_fd != INVALID_HANDLE_VALUE) {
    if (!::CreateIoCompletionPort(impl_->direct_fd, impl_->iocp,
                                   kCompletionKey, 0)) {
      ::CloseHandle(impl_->direct_fd);
      impl_->direct_fd = INVALID_HANDLE_VALUE;
    }
  }
  return true;
}

ssize_t IocpAsyncIO::GetFileSize() const {
  HANDLE h = (impl_->regular_fd != INVALID_HANDLE_VALUE) ? impl_->regular_fd
                                                          : impl_->direct_fd;
  if (h == INVALID_HANDLE_VALUE) return -1;
  LARGE_INTEGER sz;
  if (!::GetFileSizeEx(h, &sz)) return -1;
  return static_cast<ssize_t>(sz.QuadPart);
}

bool IocpAsyncIO::Truncate(size_t size) {
  HANDLE h = (impl_->regular_fd != INVALID_HANDLE_VALUE) ? impl_->regular_fd
                                                          : impl_->direct_fd;
  if (h == INVALID_HANDLE_VALUE) return false;
  LARGE_INTEGER pos;
  pos.QuadPart = static_cast<LONGLONG>(size);
  if (!::SetFilePointerEx(h, pos, nullptr, FILE_BEGIN)) return false;
  if (!::SetEndOfFile(h)) return false;
  return true;
}

IoToken IocpAsyncIO::Write(void *buffer, size_t size, int64_t offset) {
  return impl_->SubmitIO(buffer, size, offset, /*is_write=*/true);
}

IoToken IocpAsyncIO::Read(void *buffer, size_t size, int64_t offset) {
  return impl_->SubmitIO(buffer, size, offset, /*is_write=*/false);
}

bool IocpAsyncIO::IsComplete(IoToken token, IoResult &result) {
  std::lock_guard<std::mutex> lock(impl_->mutex);

  auto already = impl_->completed.find(token);
  if (already != impl_->completed.end()) {
    result = already->second;
    impl_->completed.erase(already);
    impl_->in_flight.erase(token);
    return true;
  }

  impl_->HarvestCompletions();

  auto it = impl_->completed.find(token);
  if (it != impl_->completed.end()) {
    result = it->second;
    impl_->completed.erase(it);
    impl_->in_flight.erase(token);
    return true;
  }
  return false;
}

void IocpAsyncIO::Close() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (impl_->regular_fd != INVALID_HANDLE_VALUE) {
    ::CancelIoEx(impl_->regular_fd, nullptr);
    ::CloseHandle(impl_->regular_fd);
    impl_->regular_fd = INVALID_HANDLE_VALUE;
  }
  if (impl_->direct_fd != INVALID_HANDLE_VALUE) {
    ::CancelIoEx(impl_->direct_fd, nullptr);
    ::CloseHandle(impl_->direct_fd);
    impl_->direct_fd = INVALID_HANDLE_VALUE;
  }
  if (impl_->iocp) {
    impl_->DrainAllCompletions();
    ::CloseHandle(impl_->iocp);
    impl_->iocp = nullptr;
  }
  impl_->in_flight.clear();
  impl_->completed.clear();
}

int IocpAsyncIO::GetEventFd() const { return -1; }

}  // namespace ctp

#endif  // _WIN32
