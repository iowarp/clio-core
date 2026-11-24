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
 * cuFile interceptor offsets (issue #1191). cuFileRead/cuFileWrite used to
 * ignore file_offset and devPtr_offset, transferring at the descriptor's
 * current position and the buffer's start; registration kept a pointer to
 * the caller's (stack) descriptor, and deregistration closed the caller's fd.
 *
 * cuFile's API takes a raw POSIX descriptor, so this test opens one.
 */

#include <cuda_runtime.h>
#include <cufile.h>
#include <fcntl.h>
#include <unistd.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "../../../../../context-runtime/test/simple_test.h"

namespace fs = std::filesystem;

namespace {

constexpr size_t kFileBytes = 64 * 1024;

/** Expected byte at a file position. */
char At(size_t pos) { return static_cast<char>((pos * 7 + 3) % 251); }

/**
 * Register fd through a descriptor that dies with this frame, as callers
 * commonly do.
 * @param fd the open file
 * @return the handle
 */
CUfileHandle_t RegisterFromStack(int fd) {
  CUfileDescr_t descr;
  std::memset(&descr, 0, sizeof(descr));
  descr.handle.fd = fd;
  descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
  CUfileHandle_t fh = nullptr;
  REQUIRE(cuFileHandleRegister(&fh, &descr).err == CU_FILE_SUCCESS);
  std::memset(&descr, 0xAB, sizeof(descr));  // the frame's bytes are reused
  return fh;
}

}  // namespace

TEST_CASE("cuFile interceptor honors file and device offsets",
          "[gds][1191]") {
  const fs::path path = fs::temp_directory_path() / "clio_cufile_offsets.bin";
  {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    for (size_t i = 0; i < kFileBytes; ++i) f.put(At(i));
  }
  int fd = open(path.c_str(), O_RDWR);
  REQUIRE(fd >= 0);
  CUfileHandle_t fh = RegisterFromStack(fd);

  constexpr size_t kDevBytes = 16 * 1024;
  char *dev = nullptr;
  REQUIRE(cudaMalloc(&dev, kDevBytes) == cudaSuccess);
  REQUIRE(cudaMemset(dev, 0, kDevBytes) == cudaSuccess);

  // Read 4 KiB from file offset 8192 into dev + 1024.
  constexpr off_t kFileOff = 8192, kDevOff = 1024;
  constexpr size_t kLen = 4096;
  REQUIRE(cuFileRead(fh, dev, kLen, kFileOff, kDevOff) ==
          static_cast<ssize_t>(kLen));
  std::vector<char> back(kDevBytes);
  REQUIRE(cudaMemcpy(back.data(), dev, kDevBytes, cudaMemcpyDeviceToHost) ==
          cudaSuccess);
  for (size_t i = 0; i < static_cast<size_t>(kDevOff); ++i) {
    REQUIRE(back[i] == 0);  // bytes before devPtr_offset untouched
  }
  for (size_t i = 0; i < kLen; ++i) {
    REQUIRE(back[kDevOff + i] == At(kFileOff + i));
  }

  // Write 2 KiB from dev + 1024 to file offset 40000.
  constexpr off_t kWriteOff = 40000;
  constexpr size_t kWriteLen = 2048;
  REQUIRE(cuFileWrite(fh, dev, kWriteLen, kWriteOff, kDevOff) ==
          static_cast<ssize_t>(kWriteLen));

  cuFileHandleDeregister(fh);
  // The caller still owns the descriptor.
  REQUIRE(lseek(fd, 0, SEEK_CUR) != -1);
  close(fd);
  cudaFree(dev);

  std::ifstream f(path, std::ios::binary);
  std::vector<char> file((std::istreambuf_iterator<char>(f)), {});
  REQUIRE(file.size() == kFileBytes);
  for (size_t i = 0; i < kWriteLen; ++i) {
    REQUIRE(file[kWriteOff + i] == At(kFileOff + i));
  }
  // Nothing else moved: in particular not the start of the file, where the
  // old position-based write landed.
  for (size_t i = 0; i < 1024; ++i) REQUIRE(file[i] == At(i));
  fs::remove(path);
}

SIMPLE_TEST_MAIN()
