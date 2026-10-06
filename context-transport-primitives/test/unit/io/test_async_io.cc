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

#include "basic_test.h"
#include "clio_ctp/io/async_io_factory.h"

#include <fcntl.h>
#ifdef _WIN32
#include <io.h>
#include <process.h>
#else
#include <unistd.h>
#endif
#include <cstdlib>
#include <cstring>
#include <string>
#include <filesystem>

static const size_t kBlockSize = 4096;
static const size_t kIoDepth = 32;

/** Create a cross-platform temporary file and return its path */
static std::string CreateTempFile(const std::string &prefix) {
  auto tmp_dir = std::filesystem::temp_directory_path();
  auto path = tmp_dir / (prefix + "_" + std::to_string(
#ifdef _WIN32
      _getpid()
#else
      getpid()
#endif
  ) + "_" + std::to_string(rand()));
  // Create the file
  std::ofstream ofs(path.string());
  ofs.close();
  return path.string();
}

/** Cross-platform aligned allocation */
static void *AlignedMalloc(size_t alignment, size_t size) {
#ifdef _WIN32
  return _aligned_malloc(size, alignment);
#else
  void *ptr = nullptr;
  if (posix_memalign(&ptr, alignment, size) != 0) return nullptr;
  return ptr;
#endif
}

/** Cross-platform aligned free */
static void AlignedFree(void *ptr) {
#ifdef _WIN32
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}

/**
 * Helper: run a write-then-read round trip using aligned buffers (O_DIRECT path).
 * Returns true if the data read back matches what was written.
 */
static bool RunAlignedWriteReadTest(ctp::AsyncIoBackend backend) {
  auto aio = ctp::AsyncIoFactory::Get(kIoDepth, backend);
  if (!aio) {
    return false;  // Backend not available on this platform
  }

  // Create a temp file
  std::string path = CreateTempFile("test_async_io");

  REQUIRE(aio->Open(path, O_RDWR, 0644));
  REQUIRE(aio->Truncate(kBlockSize));

  // Allocate aligned buffers for O_DIRECT path testing
  void *write_buf = AlignedMalloc(4096, kBlockSize);
  void *read_buf = AlignedMalloc(4096, kBlockSize);
  REQUIRE(write_buf != nullptr);
  REQUIRE(read_buf != nullptr);

  // Fill write buffer with a known pattern
  memset(write_buf, 0xAB, kBlockSize);
  memset(read_buf, 0, kBlockSize);

  // Write
  auto token = aio->Write(write_buf, kBlockSize, 0);
  REQUIRE(token != ctp::kInvalidIoToken);

  // Poll for write completion
  ctp::IoResult result;
  int attempts = 0;
  while (!aio->IsComplete(token, result)) {
    ++attempts;
    REQUIRE(attempts < 1000000);
  }
  REQUIRE(result.error_code == 0);
  REQUIRE(result.bytes_transferred == static_cast<ssize_t>(kBlockSize));

  // Read back
  token = aio->Read(read_buf, kBlockSize, 0);
  REQUIRE(token != ctp::kInvalidIoToken);

  attempts = 0;
  while (!aio->IsComplete(token, result)) {
    ++attempts;
    REQUIRE(attempts < 1000000);
  }
  REQUIRE(result.error_code == 0);
  REQUIRE(result.bytes_transferred == static_cast<ssize_t>(kBlockSize));

  // Verify contents
  bool match = (memcmp(write_buf, read_buf, kBlockSize) == 0);

  // Cleanup
  aio->Close();
  std::filesystem::remove(path);
  AlignedFree(write_buf);
  AlignedFree(read_buf);

  return match;
}

/**
 * Helper: run a write-then-read round trip using unaligned buffers (regular fd path).
 * Returns true if the data read back matches what was written.
 */
static bool RunUnalignedWriteReadTest(ctp::AsyncIoBackend backend) {
  auto aio = ctp::AsyncIoFactory::Get(kIoDepth, backend);
  if (!aio) {
    return false;
  }

  std::string path = CreateTempFile("test_async_io_unaligned");

  REQUIRE(aio->Open(path, O_RDWR, 0644));

  // Use a non-page-aligned size to force the regular fd path
  const size_t kUnalignedSize = 1000;
  REQUIRE(aio->Truncate(kUnalignedSize));

  // Use malloc (not aligned) to get an unaligned buffer
  void *write_buf = malloc(kUnalignedSize);
  void *read_buf = malloc(kUnalignedSize);
  REQUIRE(write_buf != nullptr);
  REQUIRE(read_buf != nullptr);

  memset(write_buf, 0xCD, kUnalignedSize);
  memset(read_buf, 0, kUnalignedSize);

  // Write
  auto token = aio->Write(write_buf, kUnalignedSize, 0);
  REQUIRE(token != ctp::kInvalidIoToken);

  ctp::IoResult result;
  int attempts = 0;
  while (!aio->IsComplete(token, result)) {
    ++attempts;
    REQUIRE(attempts < 1000000);
  }
  REQUIRE(result.error_code == 0);

  // Read back
  token = aio->Read(read_buf, kUnalignedSize, 0);
  REQUIRE(token != ctp::kInvalidIoToken);

  attempts = 0;
  while (!aio->IsComplete(token, result)) {
    ++attempts;
    REQUIRE(attempts < 1000000);
  }
  REQUIRE(result.error_code == 0);

  bool match = (memcmp(write_buf, read_buf, kUnalignedSize) == 0);

  aio->Close();
  std::filesystem::remove(path);
  free(write_buf);
  free(read_buf);

  return match;
}

TEST_CASE("TestAsyncIO") {
  PAGE_DIVIDE("Default") {
    bool ok = RunAlignedWriteReadTest(ctp::AsyncIoBackend::kDefault);
    REQUIRE(ok);
  }

  PAGE_DIVIDE("DefaultUnaligned") {
    bool ok = RunUnalignedWriteReadTest(ctp::AsyncIoBackend::kDefault);
    REQUIRE(ok);
  }

#if CTP_ENABLE_LIBAIO
  PAGE_DIVIDE("LinuxAio") {
    bool ok = RunAlignedWriteReadTest(ctp::AsyncIoBackend::kLinuxAio);
    REQUIRE(ok);
  }

  PAGE_DIVIDE("LinuxAioUnaligned") {
    bool ok = RunUnalignedWriteReadTest(ctp::AsyncIoBackend::kLinuxAio);
    REQUIRE(ok);
  }
#endif

#if CTP_ENABLE_IO_URING
  // io_uring can be compiled in yet refused by the kernel or a seccomp
  // profile (Docker's default blocks io_uring_setup): the factory then returns
  // nullptr for it and the default backend falls back (#1177).
  const bool uring = ctp::AsyncIoFactory::IsBackendUsable(
      ctp::AsyncIoBackend::kIoUring);
  PAGE_DIVIDE("IoUring") {
    if (!uring) {
      WARN("io_uring is not usable here; skipping its explicit sections");
      REQUIRE(ctp::AsyncIoFactory::Get(kIoDepth,
                                       ctp::AsyncIoBackend::kIoUring) ==
              nullptr);
    } else {
      bool ok = RunAlignedWriteReadTest(ctp::AsyncIoBackend::kIoUring);
      REQUIRE(ok);
    }
  }

  PAGE_DIVIDE("IoUringUnaligned") {
    if (uring) {
      bool ok = RunUnalignedWriteReadTest(ctp::AsyncIoBackend::kIoUring);
      REQUIRE(ok);
    }
  }
#endif

#if !defined(_WIN32)
  PAGE_DIVIDE("PosixAio") {
    bool ok = RunAlignedWriteReadTest(ctp::AsyncIoBackend::kPosixAio);
    REQUIRE(ok);
  }

  PAGE_DIVIDE("PosixAioUnaligned") {
    bool ok = RunUnalignedWriteReadTest(ctp::AsyncIoBackend::kPosixAio);
    REQUIRE(ok);
  }
#endif

#ifdef _WIN32
  PAGE_DIVIDE("Iocp") {
    bool ok = RunAlignedWriteReadTest(ctp::AsyncIoBackend::kIocp);
    REQUIRE(ok);
  }

  PAGE_DIVIDE("IocpUnaligned") {
    bool ok = RunUnalignedWriteReadTest(ctp::AsyncIoBackend::kIocp);
    REQUIRE(ok);
  }
#endif

#if CTP_ENABLE_NIXL
  PAGE_DIVIDE("Nixl") {
    bool ok = RunAlignedWriteReadTest(ctp::AsyncIoBackend::kNixl);
    REQUIRE(ok);
  }

  PAGE_DIVIDE("NixlUnaligned") {
    bool ok = RunUnalignedWriteReadTest(ctp::AsyncIoBackend::kNixl);
    REQUIRE(ok);
  }
#endif
}
