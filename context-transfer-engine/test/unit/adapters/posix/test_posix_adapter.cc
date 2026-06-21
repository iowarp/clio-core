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
 * POSIX ADAPTER UNIT TESTS
 *
 * This test suite provides unit tests for the WRP CTE POSIX adapter that
 * directly link to the adapter libraries without using LD_PRELOAD.
 *
 * Test Cases:
 * 1. Open-Write-Read-Close: Basic file I/O operations with data verification
 */

#include <catch2/catch_all.hpp>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <thread>
#include <unistd.h>
#include <vector>

#include "clio_runtime/clio_runtime.h"
#include "clio_runtime/bdev/bdev_client.h"
#include "clio_cte/core/core_client.h"

using namespace std::chrono_literals;
namespace stdfs = std::filesystem;

// Test constants — paths use the clio:: prefix that opts in interception.
const size_t kTestFileSize = 16 * 1024 * 1024; // 16MB
const std::string kTestDir = "/tmp";
const std::string kTestBackendFile = "/tmp/clio_cte_posix_test.dat";
const std::string kTestFile = "clio::" + kTestBackendFile;

/**
 * Initialize CTE runtime and register test target. Must be called before
 * any POSIX adapter operation that should go through CTE.
 */
bool initializeRuntime() {
  static bool initialized = false;
  if (initialized) {
    return true;
  }

  INFO("Initializing Clio runtime...");
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true)) {
    INFO("Clio initialization failed");
    return false;
  }
  INFO("✓ Clio runtime initialized");

  INFO("Initializing CTE runtime...");
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    INFO("CTE initialization failed");
    return false;
  }
  INFO("✓ CTE runtime initialized");

  // Give the CTE core pool a file-backed storage target so the chimod has
  // somewhere to place blobs (mirrors the libfuse adapter test fixture).
  INFO("Registering test target with CTE...");
  auto *cte_client = CLIO_CTE_CLIENT;
  clio::run::PoolId bdev_pool_id(950, 0);
  clio::run::bdev::Client bdev_client(bdev_pool_id);
  auto create_task = bdev_client.AsyncCreate(
      clio::run::PoolQuery::Dynamic(), kTestBackendFile, bdev_pool_id,
      clio::run::bdev::BdevType::kFile);
  create_task.Wait();
  std::this_thread::sleep_for(100ms);
  auto reg_task = cte_client->AsyncRegisterTarget(
      kTestBackendFile, clio::run::bdev::BdevType::kFile, kTestFileSize * 10,
      clio::run::PoolQuery::Local(), bdev_pool_id);
  reg_task.Wait();
  if (reg_task->GetReturnCode() != 0) {
    INFO("Failed to register target, rc=" << reg_task->GetReturnCode());
    return false;
  }
  std::this_thread::sleep_for(100ms);
  INFO("✓ Test target registered successfully");

  initialized = true;
  return true;
}

/**
 * POSIX Adapter Test: Open-Write-Read-Close
 *
 * This test verifies basic POSIX file operations through the CTE adapter:
 * 1. Opens a file in /tmp directory
 * 2. Writes 16MB of test data to the file
 * 3. Reads 16MB from the file
 * 4. Verifies the write and read data match
 * 5. Closes the file
 * 6. Removes the file
 */
TEST_CASE("POSIX Adapter: Open-Write-Read-Close", "[posix][adapter]") {
  // Initialize CTE runtime and register test target
  REQUIRE(initializeRuntime());

  // Clean up any existing test file
  // stdfs takes plain filesystem paths; the backend file is what lands
  // on disk, so clean that, not the clio:: alias.
  if (stdfs::exists(kTestBackendFile)) {
    stdfs::remove(kTestBackendFile);
  }

  SECTION("Basic file I/O operations") {
    // Step 1: Open file for writing
    INFO("Step 1: Opening file for writing...");
    int fd = open(kTestFile.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
    REQUIRE(fd >= 0);
    INFO("✓ File opened successfully with fd=" << fd);

    // Step 2: Prepare test data
    INFO("Step 2: Preparing 16MB test data...");
    std::vector<char> write_data(kTestFileSize);

    // Fill with a pattern that's easy to verify
    for (size_t i = 0; i < kTestFileSize; ++i) {
      write_data[i] = static_cast<char>(i % 256);
    }
    INFO("✓ Test data prepared with repeating byte pattern");

    // Step 3: Write data to file
    INFO("Step 3: Writing 16MB to file...");
    ssize_t bytes_written = write(fd, write_data.data(), kTestFileSize);
    REQUIRE(bytes_written == static_cast<ssize_t>(kTestFileSize));
    INFO("✓ Successfully wrote " << bytes_written << " bytes");

    // Step 4: Close file after writing
    INFO("Step 4: Closing file after writing...");
    int close_result = close(fd);
    REQUIRE(close_result == 0);
    INFO("✓ File closed successfully");

    // Step 5: Open file for reading
    INFO("Step 5: Opening file for reading...");
    fd = open(kTestFile.c_str(), O_RDONLY);
    REQUIRE(fd >= 0);
    INFO("✓ File reopened for reading with fd=" << fd);

    // Step 6: Read data from file
    INFO("Step 6: Reading 16MB from file...");
    std::vector<char> read_data(kTestFileSize);
    ssize_t bytes_read = read(fd, read_data.data(), kTestFileSize);
    REQUIRE(bytes_read == static_cast<ssize_t>(kTestFileSize));
    INFO("✓ Successfully read " << bytes_read << " bytes");

    // Step 7: Verify data integrity
    INFO("Step 7: Verifying data integrity...");
    bool data_matches = (write_data == read_data);
    REQUIRE(data_matches);
    INFO("✓ Data verification successful - write and read data match");

    // Step 8: Close file after reading
    INFO("Step 8: Closing file after reading...");
    close_result = close(fd);
    REQUIRE(close_result == 0);
    INFO("✓ File closed successfully");

    // Step 9: Remove the file through the adapter. Data lives in CTE (not a
    // passthrough file on disk), so removal goes through the chimod, not stdfs.
    INFO("Step 9: Removing test file...");
    REQUIRE(unlink(kTestFile.c_str()) == 0);
    INFO("✓ Test file removed successfully");
  }
}

/**
 * Additional test for file size verification
 */
TEST_CASE("POSIX Adapter: File Size Verification", "[posix][adapter]") {
  // Initialize CTE runtime and register test target
  REQUIRE(initializeRuntime());

  // Clean up any existing test file
  // stdfs takes plain filesystem paths; the backend file is what lands
  // on disk, so clean that, not the clio:: alias.
  if (stdfs::exists(kTestBackendFile)) {
    stdfs::remove(kTestBackendFile);
  }

  SECTION("Verify file size after write operations") {
    // Create and write to file
    int fd = open(kTestFile.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
    REQUIRE(fd >= 0);

    const size_t test_size = 1024; // 1KB for quick test
    std::vector<char> data(test_size, 'A');

    ssize_t bytes_written = write(fd, data.data(), test_size);
    REQUIRE(bytes_written == static_cast<ssize_t>(test_size));

    close(fd);

    // Verify the logical size through the adapter. Data lives in CTE, so
    // query the chimod (via fstat) rather than stat-ing a backend file.
    int rfd = open(kTestFile.c_str(), O_RDONLY);
    REQUIRE(rfd >= 0);
    struct stat st;
    REQUIRE(fstat(rfd, &st) == 0);
    REQUIRE(st.st_size == static_cast<off_t>(test_size));
    REQUIRE(close(rfd) == 0);

    // Clean up
    REQUIRE(unlink(kTestFile.c_str()) == 0);
  }
}

/* ===========================================================================
 * clio:: prefix-gating tests
 *
 * The new design: a path is intercepted iff it starts with "clio::".
 * Verify both branches:
 *  - clio:: path is intercepted, ends up tracked, and gets a CTE-issued fd
 *  - bare path is NOT intercepted, gets a kernel fd, and is not in the MDM
 *
 * We can't peek at the MDM directly without internal headers, but
 * CTE-issued fds are >= 8192 (see PosixFs::IsFdTracked) while the
 * kernel keeps app fds < 8192 for tiny processes, so checking the fd
 * range is a reliable proxy for "did this get intercepted".
 * =========================================================================*/

TEST_CASE("POSIX Adapter: clio:: prefix opts in interception",
          "[posix][adapter][prefix]") {
  REQUIRE(initializeRuntime());

  const std::string backend = "/tmp/clio_cte_prefix_test.dat";
  const std::string clio = "clio::" + backend;

  // Start clean.
  stdfs::remove(backend);

  SECTION("clio:: path is intercepted") {
    int fd = open(clio.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
    REQUIRE(fd >= 0);
    // CTE-issued fds live above 8192; kernel fds are tiny in this test
    // process, so a high fd is a reliable signal we went through CTE.
    REQUIRE(fd >= 8192);
    REQUIRE(close(fd) == 0);
    stdfs::remove(backend);
  }

  SECTION("bare path is NOT intercepted") {
    int fd = open(backend.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
    REQUIRE(fd >= 0);
    REQUIRE(fd < 8192);  // plain kernel fd
    REQUIRE(close(fd) == 0);
    stdfs::remove(backend);
  }
}

TEST_CASE("POSIX Adapter: clio:: data round-trips through CTE",
          "[posix][adapter][prefix]") {
  REQUIRE(initializeRuntime());

  // Open via clio::, write, read back through the adapter. Data lives in CTE
  // (the chimod), so there is no passthrough file on disk — and definitely no
  // literal "clio::"-named file.
  const std::string backend = "/tmp/clio_cte_strip_test.dat";
  const std::string clio = "clio::" + backend;

  int fd = open(clio.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0644);
  REQUIRE(fd >= 0);
  const char *payload = "clio";
  REQUIRE(write(fd, payload, 4) == 4);
  REQUIRE(lseek(fd, 0, SEEK_SET) == 0);
  char rb[4] = {0};
  REQUIRE(read(fd, rb, 4) == 4);
  REQUIRE(memcmp(rb, payload, 4) == 0);
  REQUIRE(close(fd) == 0);

  REQUIRE(unlink(clio.c_str()) == 0);
}

/* ===========================================================================
 * stat-family correctness
 *
 * The historically-fragile area. Verify every field we promise to
 * populate is non-zero and reasonable.
 * =========================================================================*/

TEST_CASE("POSIX Adapter: fstat returns full struct stat",
          "[posix][adapter][stat]") {
  REQUIRE(initializeRuntime());

  const std::string backend = "/tmp/clio_cte_fstat_test.dat";
  const std::string clio = "clio::" + backend;
  stdfs::remove(backend);

  int fd = open(clio.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
  REQUIRE(fd >= 0);

  const size_t kPayload = 4096 + 7;  // odd size on purpose
  std::vector<char> data(kPayload, 'Z');
  REQUIRE(write(fd, data.data(), kPayload) == static_cast<ssize_t>(kPayload));

  struct stat st;
  memset(&st, 0xAB, sizeof(st));  // poison to catch missed fields
  REQUIRE(fstat(fd, &st) == 0);

  // Size first — the most common reason apps stat a file.
  REQUIRE(st.st_size == static_cast<off_t>(kPayload));
  // Regular file with sane permission bits.
  REQUIRE(S_ISREG(st.st_mode));
  REQUIRE((st.st_mode & 0777) == 0644);
  // Owner = current user.
  REQUIRE(st.st_uid == geteuid());
  REQUIRE(st.st_gid == getegid());
  // Single link to itself.
  REQUIRE(st.st_nlink == 1);
  // Synthetic device id is non-zero; inode is non-zero so apps can
  // use (dev, ino) as a unique key.
  REQUIRE(st.st_dev != 0);
  REQUIRE(st.st_ino != 0);
  // st_blocks counts 512-byte units, rounded up.
  REQUIRE(st.st_blocks >= static_cast<blkcnt_t>((kPayload + 511) / 512));
  // 1 MiB preferred block size from the adapter.
  REQUIRE(st.st_blksize == 1024 * 1024);

  REQUIRE(close(fd) == 0);
  stdfs::remove(backend);
}

TEST_CASE("POSIX Adapter: stat-by-path matches fstat-by-fd",
          "[posix][adapter][stat]") {
  REQUIRE(initializeRuntime());

  const std::string backend = "/tmp/clio_cte_statpath_test.dat";
  const std::string clio = "clio::" + backend;
  stdfs::remove(backend);

  int fd = open(clio.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
  REQUIRE(fd >= 0);
  std::vector<char> data(2048, 'q');
  REQUIRE(write(fd, data.data(), data.size()) ==
          static_cast<ssize_t>(data.size()));

  struct stat by_fd;
  REQUIRE(fstat(fd, &by_fd) == 0);

  struct stat by_path;
  REQUIRE(stat(clio.c_str(), &by_path) == 0);

  // Size/mode/owner/blksize must agree.
  REQUIRE(by_fd.st_size == by_path.st_size);
  REQUIRE(by_fd.st_mode == by_path.st_mode);
  REQUIRE(by_fd.st_uid == by_path.st_uid);
  REQUIRE(by_fd.st_gid == by_path.st_gid);
  REQUIRE(by_fd.st_blksize == by_path.st_blksize);
  // Inode is derived from the bare path → must be stable between
  // fstat and stat-by-clio-path.
  REQUIRE(by_fd.st_ino == by_path.st_ino);

  REQUIRE(close(fd) == 0);
  stdfs::remove(backend);
}

TEST_CASE("POSIX Adapter: stat() on missing clio:: path returns ENOENT",
          "[posix][adapter][stat][errors]") {
  REQUIRE(initializeRuntime());

  struct stat st;
  errno = 0;
  // No prior open of this path, no backend file exists.
  int result = stat("clio::/tmp/clio_cte_does_not_exist.dat", &st);
  REQUIRE(result == -1);
  REQUIRE(errno == ENOENT);
}

/* ===========================================================================
 * Misc adapter coverage: lseek, ftruncate, unlink, fsync
 * =========================================================================*/

TEST_CASE("POSIX Adapter: seek + truncate + sync + unlink",
          "[posix][adapter][ops]") {
  REQUIRE(initializeRuntime());

  const std::string backend = "/tmp/clio_cte_ops_test.dat";
  const std::string clio = "clio::" + backend;
  stdfs::remove(backend);

  int fd = open(clio.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0644);
  REQUIRE(fd >= 0);

  // Write 4 KiB, seek to start, read back, then truncate to 1 KiB.
  std::vector<char> w(4096, 'X');
  REQUIRE(write(fd, w.data(), w.size()) == static_cast<ssize_t>(w.size()));

  off_t back = lseek(fd, 0, SEEK_SET);
  REQUIRE(back == 0);

  std::vector<char> r(4096, 0);
  REQUIRE(read(fd, r.data(), r.size()) == static_cast<ssize_t>(r.size()));
  REQUIRE(r == w);

  REQUIRE(ftruncate(fd, 1024) == 0);
  struct stat st;
  REQUIRE(fstat(fd, &st) == 0);
  REQUIRE(st.st_size == 1024);

  REQUIRE(fsync(fd) == 0);
  REQUIRE(close(fd) == 0);

  REQUIRE(unlink(clio.c_str()) == 0);
  // After unlink, the file is gone from CTE: stat must report ENOENT.
  struct stat probe;
  errno = 0;
  REQUIRE(stat(clio.c_str(), &probe) == -1);
  REQUIRE(errno == ENOENT);
}
