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
 * Unit tests for the NIXL lightbeam transport.
 *
 * Tests cover:
 *   1. DRAM→DRAM loopback transfer
 *   2. DRAM→FILE (CPU→storage) transfer via dst_fd in LbmContext
 */

#include <clio_ctp/lightbeam/nixl_transport.h>

#include <cassert>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

using namespace ctp::lbm;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Returns a temp-file path based on the given suffix. */
static std::string MakeTmpPath(const std::string& suffix) {
  return std::string("/tmp/nixl_lbm_test_") + suffix;
}

/** Assert helper that prints the message and aborts on failure. */
static void Require(bool cond, const std::string& msg) {
  if (!cond) {
    std::cerr << "[FAIL] " << msg << "\n";
    std::abort();
  }
}

// ---------------------------------------------------------------------------
// Test 1: DRAM → DRAM loopback
// ---------------------------------------------------------------------------

/**
 * @brief Verify that NixlTransport copies DRAM data to pre-allocated
 *        destination buffers within the same process.
 */
static void TestDramToMem() {
#if CTP_ENABLE_NIXL
  std::cout << "\n==== TestDramToMem (DRAM→DRAM loopback via NIXL) ====\n";

  const std::string payload = "nixl_dram_loopback_test";
  const size_t kSize = payload.size();

  // Source buffer
  std::vector<char> src(payload.begin(), payload.end());

  // Destination buffer (pre-allocated; NixlTransport will write here)
  std::vector<char> dst(kSize, '\0');

  NixlTransport transport(TransportMode::kClient, "test_dram_agent");

  // Build send metadata
  LbmMeta<> meta;
  Bulk send_bulk =
      transport.Expose(ctp::ipc::FullPtr<char>(src.data()), kSize, BULK_XFER);
  meta.send.push_back(send_bulk);

  // Pre-populate recv with destination buffer so NixlTransport writes there
  Bulk recv_bulk;
  recv_bulk.size = kSize;
  recv_bulk.flags = ctp::bitfield32_t(BULK_XFER);
  recv_bulk.data = ctp::ipc::FullPtr<char>(dst.data());
  meta.recv.push_back(recv_bulk);

  int rc = transport.Send(meta);
  Require(rc == 0, "TestDramToMem: Send returned non-zero: " + std::to_string(rc));

  std::string received(dst.begin(), dst.end());
  std::cout << "Received: " << received << "\n";
  Require(received == payload,
          "TestDramToMem: data mismatch; expected='" + payload +
          "' got='" + received + "'");

  std::cout << "[TestDramToMem] PASS\n";
#else
  std::cout << "NIXL not enabled, skipping TestDramToMem\n";
#endif
}

// ---------------------------------------------------------------------------
// Test 2: DRAM → FILE (CPU → storage)
// ---------------------------------------------------------------------------

/**
 * @brief Verify that NixlTransport writes DRAM data to a file when
 *        LbmContext carries a valid dst_fd_.
 */
static void TestDramToFile() {
#if CTP_ENABLE_NIXL
  std::cout << "\n==== TestDramToFile (DRAM→FILE via NIXL POSIX backend) ====\n";

  const std::string payload = "nixl_file_transfer_test_payload";
  const size_t kSize = payload.size();

  std::string path = MakeTmpPath("dram_to_file.bin");

  // Open file for writing
  int fd = ::open(path.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
  Require(fd >= 0, "TestDramToFile: failed to open " + path);

  // Grow file to at least kSize (NIXL POSIX backend requires pre-allocated space)
  if (::ftruncate(fd, static_cast<off_t>(kSize)) != 0) {
    ::close(fd);
    ::unlink(path.c_str());
    Require(false, "TestDramToFile: ftruncate failed");
  }

  // Source buffer
  std::vector<char> src(payload.begin(), payload.end());

  NixlTransport transport(TransportMode::kClient, "test_file_agent");

  // Build send metadata
  LbmMeta<> meta;
  Bulk send_bulk =
      transport.Expose(ctp::ipc::FullPtr<char>(src.data()), kSize, BULK_XFER);
  meta.send.push_back(send_bulk);

  // Build context: write to fd at offset 0
  LbmContext ctx(LBM_SYNC, 0, fd, /*dst_offset=*/0);

  int rc = transport.Send(meta, ctx);
  ::close(fd);

  Require(rc == 0,
          "TestDramToFile: Send returned non-zero: " + std::to_string(rc));

  // Read back and verify
  std::vector<char> readback(kSize, '\0');
  int rfd = ::open(path.c_str(), O_RDONLY);
  Require(rfd >= 0, "TestDramToFile: failed to open for readback");
  ssize_t n = ::read(rfd, readback.data(), kSize);
  ::close(rfd);
  ::unlink(path.c_str());

  Require(static_cast<size_t>(n) == kSize,
          "TestDramToFile: short read, got " + std::to_string(n) +
          " expected " + std::to_string(kSize));

  std::string received(readback.begin(), readback.end());
  std::cout << "File contents: " << received << "\n";
  Require(received == payload,
          "TestDramToFile: data mismatch; expected='" + payload +
          "' got='" + received + "'");

  std::cout << "[TestDramToFile] PASS\n";
#else
  std::cout << "NIXL not enabled, skipping TestDramToFile\n";
#endif
}

// ---------------------------------------------------------------------------
// Test 3: Multiple bulk entries (scatter-gather to file)
// ---------------------------------------------------------------------------

/**
 * @brief Verify that multiple BULK_XFER entries are written contiguously
 *        to the destination file.
 */
static void TestMultiBulkToFile() {
#if CTP_ENABLE_NIXL
  std::cout << "\n==== TestMultiBulkToFile (scatter-gather → FILE) ====\n";

  const std::string part1 = "Hello, ";
  const std::string part2 = "NIXL!";
  const std::string expected = part1 + part2;

  std::string path = MakeTmpPath("multi_bulk_file.bin");

  int fd = ::open(path.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
  Require(fd >= 0, "TestMultiBulkToFile: open failed");

  off_t total = static_cast<off_t>(part1.size() + part2.size());
  Require(::ftruncate(fd, total) == 0, "TestMultiBulkToFile: ftruncate failed");

  std::vector<char> buf1(part1.begin(), part1.end());
  std::vector<char> buf2(part2.begin(), part2.end());

  NixlTransport transport(TransportMode::kClient, "test_multi_agent");

  LbmMeta<> meta;
  meta.send.push_back(
      transport.Expose(ctp::ipc::FullPtr<char>(buf1.data()), buf1.size(), BULK_XFER));
  meta.send.push_back(
      transport.Expose(ctp::ipc::FullPtr<char>(buf2.data()), buf2.size(), BULK_XFER));

  LbmContext ctx(LBM_SYNC, 0, fd, /*dst_offset=*/0);
  int rc = transport.Send(meta, ctx);
  ::close(fd);

  Require(rc == 0, "TestMultiBulkToFile: Send failed: " + std::to_string(rc));

  // Verify
  std::vector<char> readback(expected.size(), '\0');
  int rfd = ::open(path.c_str(), O_RDONLY);
  Require(rfd >= 0, "TestMultiBulkToFile: readback open failed");
  ssize_t n = ::read(rfd, readback.data(), expected.size());
  ::close(rfd);
  ::unlink(path.c_str());

  Require(static_cast<size_t>(n) == expected.size(),
          "TestMultiBulkToFile: short read");
  std::string received(readback.begin(), readback.end());
  std::cout << "Combined file contents: " << received << "\n";
  Require(received == expected,
          "TestMultiBulkToFile: mismatch; expected='" + expected +
          "' got='" + received + "'");

  std::cout << "[TestMultiBulkToFile] PASS\n";
#else
  std::cout << "NIXL not enabled, skipping TestMultiBulkToFile\n";
#endif
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
  TestDramToMem();
  TestDramToFile();
  TestMultiBulkToFile();

  std::cout << "\nAll NIXL transport tests passed!\n";
  return 0;
}
