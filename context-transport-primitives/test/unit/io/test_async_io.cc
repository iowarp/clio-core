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
#include "hermes_shm/io/async_io_factory.h"

#include <fcntl.h>
#include <unistd.h>
#include <cstdlib>
#include <cstring>
#include <string>

static const size_t kBlockSize = 4096;
static const size_t kIoDepth = 32;

/**
 * Helper: run a write-then-read round trip using aligned buffers (O_DIRECT path).
 * Returns true if the data read back matches what was written.
 */
static bool RunAlignedWriteReadTest(hshm::AsyncIoBackend backend) {
  auto aio = hshm::AsyncIoFactory::Get(kIoDepth, backend);
  if (!aio) {
    return false;  // Backend not available on this platform
  }

  // Create a temp file
  std::string path = "/tmp/test_async_io_XXXXXX";
  int tmpfd = mkstemp(path.data());
  REQUIRE(tmpfd >= 0);
  close(tmpfd);

  REQUIRE(aio->Open(path, O_RDWR, 0644));
  REQUIRE(aio->Truncate(kBlockSize));

  // Allocate aligned buffers for O_DIRECT path testing
  void *write_buf = nullptr;
  void *read_buf = nullptr;
  REQUIRE(posix_memalign(&write_buf, 4096, kBlockSize) == 0);
  REQUIRE(posix_memalign(&read_buf, 4096, kBlockSize) == 0);

  // Fill write buffer with a known pattern
  memset(write_buf, 0xAB, kBlockSize);
  memset(read_buf, 0, kBlockSize);

  // Write
  auto token = aio->Write(write_buf, kBlockSize, 0);
  REQUIRE(token != hshm::kInvalidIoToken);

  // Poll for write completion
  hshm::IoResult result;
  int attempts = 0;
  while (!aio->IsComplete(token, result)) {
    ++attempts;
    REQUIRE(attempts < 1000000);
  }
  REQUIRE(result.error_code == 0);
  REQUIRE(result.bytes_transferred == static_cast<ssize_t>(kBlockSize));

  // Read back
  token = aio->Read(read_buf, kBlockSize, 0);
  REQUIRE(token != hshm::kInvalidIoToken);

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
  unlink(path.c_str());
  free(write_buf);
  free(read_buf);

  return match;
}

/**
 * Helper: run a write-then-read round trip using unaligned buffers (regular fd path).
 * Returns true if the data read back matches what was written.
 */
static bool RunUnalignedWriteReadTest(hshm::AsyncIoBackend backend) {
  auto aio = hshm::AsyncIoFactory::Get(kIoDepth, backend);
  if (!aio) {
    return false;
  }

  std::string path = "/tmp/test_async_io_unaligned_XXXXXX";
  int tmpfd = mkstemp(path.data());
  REQUIRE(tmpfd >= 0);
  close(tmpfd);

  REQUIRE(aio->Open(path, O_RDWR, 0644));

  // Use a non-page-aligned size to force the regular fd path
  const size_t kUnalignedSize = 1000;
  REQUIRE(aio->Truncate(kUnalignedSize));

  // Use malloc (not posix_memalign) to get an unaligned buffer
  void *write_buf = malloc(kUnalignedSize);
  void *read_buf = malloc(kUnalignedSize);
  REQUIRE(write_buf != nullptr);
  REQUIRE(read_buf != nullptr);

  memset(write_buf, 0xCD, kUnalignedSize);
  memset(read_buf, 0, kUnalignedSize);

  // Write
  auto token = aio->Write(write_buf, kUnalignedSize, 0);
  REQUIRE(token != hshm::kInvalidIoToken);

  hshm::IoResult result;
  int attempts = 0;
  while (!aio->IsComplete(token, result)) {
    ++attempts;
    REQUIRE(attempts < 1000000);
  }
  REQUIRE(result.error_code == 0);

  // Read back
  token = aio->Read(read_buf, kUnalignedSize, 0);
  REQUIRE(token != hshm::kInvalidIoToken);

  attempts = 0;
  while (!aio->IsComplete(token, result)) {
    ++attempts;
    REQUIRE(attempts < 1000000);
  }
  REQUIRE(result.error_code == 0);

  bool match = (memcmp(write_buf, read_buf, kUnalignedSize) == 0);

  aio->Close();
  unlink(path.c_str());
  free(write_buf);
  free(read_buf);

  return match;
}

TEST_CASE("TestAsyncIO") {
  PAGE_DIVIDE("Default") {
    bool ok = RunAlignedWriteReadTest(hshm::AsyncIoBackend::kDefault);
    REQUIRE(ok);
  }

  PAGE_DIVIDE("DefaultUnaligned") {
    bool ok = RunUnalignedWriteReadTest(hshm::AsyncIoBackend::kDefault);
    REQUIRE(ok);
  }

#if HSHM_ENABLE_LIBAIO
  PAGE_DIVIDE("LinuxAio") {
    bool ok = RunAlignedWriteReadTest(hshm::AsyncIoBackend::kLinuxAio);
    REQUIRE(ok);
  }

  PAGE_DIVIDE("LinuxAioUnaligned") {
    bool ok = RunUnalignedWriteReadTest(hshm::AsyncIoBackend::kLinuxAio);
    REQUIRE(ok);
  }
#endif

#if HSHM_ENABLE_IO_URING
  PAGE_DIVIDE("IoUring") {
    bool ok = RunAlignedWriteReadTest(hshm::AsyncIoBackend::kIoUring);
    REQUIRE(ok);
  }

  PAGE_DIVIDE("IoUringUnaligned") {
    bool ok = RunUnalignedWriteReadTest(hshm::AsyncIoBackend::kIoUring);
    REQUIRE(ok);
  }
#endif

#if !defined(_WIN32)
  PAGE_DIVIDE("PosixAio") {
    bool ok = RunAlignedWriteReadTest(hshm::AsyncIoBackend::kPosixAio);
    REQUIRE(ok);
  }

  PAGE_DIVIDE("PosixAioUnaligned") {
    bool ok = RunUnalignedWriteReadTest(hshm::AsyncIoBackend::kPosixAio);
    REQUIRE(ok);
  }
#endif
}
