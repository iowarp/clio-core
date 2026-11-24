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

/**
 * Large-file regressions for the AsyncIO backends.
 *
 * #1059: file offsets were passed as off_t, a 32-bit long under MSVC, so any
 *        I/O at or past 2 GiB wrapped negative and IOCP rejected it with
 *        ERROR_INVALID_PARAMETER (87). Reads and writes past 4 GiB must land
 *        at the requested offset.
 * #1100: extending a bdev backing file on Windows allocated (and charged
 *        quota for) the whole extent up front. Extending must stay sparse.
 */

#include "basic_test.h"
#include "clio_ctp/io/async_io_factory.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <process.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

namespace {

constexpr size_t kIoDepth = 32;
constexpr size_t kBlock = 4096;
constexpr uint64_t kGiB = 1024ull * 1024ull * 1024ull;
/** Logical size of the test file: well past both the 2 and 4 GiB edges. */
constexpr uint64_t kFileSize = 8 * kGiB;
/** Allocation ceiling for the sparse check: the test writes a few KiB. */
constexpr uint64_t kMaxAllocated = 64ull * 1024ull * 1024ull;

/**
 * Create an empty temp file.
 * @param prefix file name prefix
 * @return absolute path of the created file
 */
std::string MakeTempFile(const std::string &prefix) {
#ifdef _WIN32
  int pid = _getpid();
#else
  int pid = getpid();
#endif
  auto path = std::filesystem::temp_directory_path() /
              (prefix + "_" + std::to_string(pid) + "_" +
               std::to_string(rand()));
  std::ofstream(path.string()).close();
  return path.string();
}

/**
 * Bytes the filesystem actually allocated for a file (not its logical size).
 * @param path file path
 * @return allocated bytes, or UINT64_MAX if it cannot be determined
 */
uint64_t AllocatedBytes(const std::string &path) {
#ifdef _WIN32
  DWORD high = 0;
  DWORD low = ::GetCompressedFileSizeA(path.c_str(), &high);
  if (low == INVALID_FILE_SIZE && ::GetLastError() != NO_ERROR) {
    return UINT64_MAX;
  }
  return (static_cast<uint64_t>(high) << 32) | low;
#else
  struct stat st;
  if (stat(path.c_str(), &st) != 0) return UINT64_MAX;
  return static_cast<uint64_t>(st.st_blocks) * 512ull;
#endif
}

/** Aligned allocation for the unbuffered (O_DIRECT / NO_BUFFERING) path. */
void *AlignedAlloc(size_t size) {
#ifdef _WIN32
  return _aligned_malloc(size, kBlock);
#else
  void *p = nullptr;
  return posix_memalign(&p, kBlock, size) == 0 ? p : nullptr;
#endif
}

/** Free memory from AlignedAlloc. */
void AlignedRelease(void *p) {
#ifdef _WIN32
  _aligned_free(p);
#else
  free(p);
#endif
}

/**
 * Submit one I/O and spin until it completes.
 * @param aio backend
 * @param write true for a write, false for a read
 * @param buf buffer
 * @param size transfer size
 * @param offset file offset
 * @return the completion result (error_code != 0 on failure)
 */
ctp::IoResult DoIo(ctp::AsyncIO &aio, bool write, void *buf, size_t size,
                   int64_t offset) {
  ctp::IoToken tok = write ? aio.Write(buf, size, offset)
                           : aio.Read(buf, size, offset);
  ctp::IoResult res{-1, -1};
  if (tok == ctp::kInvalidIoToken) return res;
  while (!aio.IsComplete(tok, res)) {
  }
  return res;
}

/**
 * Write a pattern at `offset`, read it back, and compare.
 * @param aio backend
 * @param buf write buffer (size bytes)
 * @param scratch read buffer (size bytes)
 * @param size transfer size
 * @param offset file offset
 * @param fill pattern byte
 */
void RoundTrip(ctp::AsyncIO &aio, char *buf, char *scratch, size_t size,
               int64_t offset, int fill) {
  memset(buf, fill, size);
  memset(scratch, 0, size);
  ctp::IoResult w = DoIo(aio, true, buf, size, offset);
  REQUIRE(w.error_code == 0);
  REQUIRE(w.bytes_transferred == static_cast<ssize_t>(size));
  ctp::IoResult r = DoIo(aio, false, scratch, size, offset);
  REQUIRE(r.error_code == 0);
  REQUIRE(r.bytes_transferred == static_cast<ssize_t>(size));
  REQUIRE(memcmp(buf, scratch, size) == 0);
}

}  // namespace

/**
 * Run every large-offset check against one backend.
 * @param backend the AsyncIO backend under test
 */
static void RunLargeOffsetChecks(ctp::AsyncIoBackend backend) {
  auto aio = ctp::AsyncIoFactory::Get(kIoDepth, backend);
  REQUIRE(aio != nullptr);
  std::string path = MakeTempFile("test_async_io_large");
  REQUIRE(aio->Open(path, O_RDWR, 0644));
  REQUIRE(aio->Truncate(kFileSize));
  REQUIRE(static_cast<uint64_t>(aio->GetFileSize()) == kFileSize);

  char *buf = static_cast<char *>(AlignedAlloc(kBlock));
  char *scratch = static_cast<char *>(AlignedAlloc(kBlock));
  REQUIRE(buf != nullptr);
  REQUIRE(scratch != nullptr);

  PAGE_DIVIDE("AlignedPast2GiB") {
    RoundTrip(*aio, buf, scratch, kBlock,
              static_cast<int64_t>(2 * kGiB + 16 * kBlock), 0x5A);
  }
  PAGE_DIVIDE("AlignedPast4GiB") {
    RoundTrip(*aio, buf, scratch, kBlock,
              static_cast<int64_t>(5 * kGiB + 7 * kBlock), 0xC3);
  }
  PAGE_DIVIDE("UnalignedOffsetPast4GiB") {
    // Aligned buffer and size but an unaligned offset: must take the
    // buffered handle rather than fail on the unbuffered one.
    RoundTrip(*aio, buf, scratch, kBlock,
              static_cast<int64_t>(6 * kGiB + 123), 0x3C);
  }
  PAGE_DIVIDE("DistinctOffsetsDoNotAlias") {
    // Under 32-bit wrap, 1 GiB + x and 5 GiB + x hit the same bytes.
    const int64_t lo = static_cast<int64_t>(1 * kGiB + 64 * kBlock);
    const int64_t hi = static_cast<int64_t>(5 * kGiB + 64 * kBlock);
    memset(buf, 0x11, kBlock);
    REQUIRE(DoIo(*aio, true, buf, kBlock, lo).error_code == 0);
    memset(buf, 0x22, kBlock);
    REQUIRE(DoIo(*aio, true, buf, kBlock, hi).error_code == 0);
    REQUIRE(DoIo(*aio, false, scratch, kBlock, lo).error_code == 0);
    REQUIRE(static_cast<unsigned char>(scratch[0]) == 0x11);
    REQUIRE(static_cast<unsigned char>(scratch[kBlock - 1]) == 0x11);
  }
  PAGE_DIVIDE("NegativeOffsetRejected") {
    ctp::IoResult r = DoIo(*aio, true, buf, kBlock, -static_cast<int64_t>(kBlock));
    REQUIRE(r.error_code != 0);
  }
  PAGE_DIVIDE("ExtendIsSparse") {
    uint64_t allocated = AllocatedBytes(path);
    REQUIRE(allocated != UINT64_MAX);
    INFO("allocated bytes = " << allocated);
    REQUIRE(allocated < kMaxAllocated);
  }

  aio->Close();
  AlignedRelease(buf);
  AlignedRelease(scratch);
  std::error_code ec;
  std::filesystem::remove(path, ec);
}

TEST_CASE("TestAsyncIOLargeOffsets") {
  RunLargeOffsetChecks(ctp::AsyncIoBackend::kDefault);
}

#if !defined(_WIN32)
// POSIX AIO is the fallback when io_uring is unavailable (containers,
// locked-down kernels); cover its off_t conversion explicitly.
TEST_CASE("TestAsyncIOLargeOffsetsPosixAio") {
  RunLargeOffsetChecks(ctp::AsyncIoBackend::kPosixAio);
}
#endif
