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
 * Comprehensive unit tests for bdev ChiMod
 *
 * Tests the complete bdev functionality: container creation, block allocation,
 * write/read operations, async I/O, performance metrics, and error handling.
 * Uses simple custom test framework for testing.
 */

#ifndef _WIN32
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <chrono>
#include <fstream>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "simple_test.h"

using namespace std::chrono_literals;

// Include Chimaera headers
#include <chimaera/chimaera.h>
#include <chimaera/pool_query.h>
#include <chimaera/singletons.h>
#include <chimaera/types.h>

// Include bdev client and tasks
#include <chimaera/bdev/bdev_client.h>
#include <chimaera/bdev/bdev_tasks.h>

// Include admin client for pool management
#include <chimaera/admin/admin_client.h>
#include <chimaera/admin/admin_tasks.h>

namespace {
// Test configuration constants
constexpr chi::u32 kTestTimeoutMs = 10000;
constexpr chi::u32 kMaxRetries = 100;
constexpr chi::u32 kRetryDelayMs = 50;

// Note: Tests use default pool ID (0) instead of hardcoding specific values

// Test file configurations
const std::string kTestFilePrefix = "/tmp/test_bdev_";
const chi::u64 kDefaultFileSize = 10 * 1024 * 1024;  // 10MB
const chi::u64 kLargeFileSize = 100 * 1024 * 1024;   // 100MB

// Block size constants for testing
const chi::u64 k4KB = 4096;
const chi::u64 k64KB = 65536;
const chi::u64 k256KB = 262144;
const chi::u64 k1MB = 1048576;

// Global test state
bool g_initialized = false;
int g_test_counter = 0;

/**
 * Helper function to wrap a single block in a chi::priv::vector
 * @param block Single block to wrap
 * @return chi::priv::vector containing the single block
 */
inline chi::priv::vector<chimaera::bdev::Block> WrapBlock(
    const chimaera::bdev::Block& block) {
  chi::priv::vector<chimaera::bdev::Block> blocks(HSHM_MALLOC);
  blocks.push_back(block);
  return blocks;
}

/**
 * Helper function to convert std::vector of blocks to chi::priv::vector
 * @param blocks_vec std::vector of blocks
 * @return chi::priv::vector containing all blocks
 */
inline chi::priv::vector<chimaera::bdev::Block> ConvertBlocks(
    const std::vector<chimaera::bdev::Block>& blocks_vec) {
  chi::priv::vector<chimaera::bdev::Block> blocks(HSHM_MALLOC);
  for (const auto& block : blocks_vec) {
    blocks.push_back(block);
  }
  return blocks;
}

/**
 * Simple test fixture for bdev ChiMod tests
 * Handles setup and teardown of runtime, client, and test files
 */
class BdevChimodFixture {
 public:
  BdevChimodFixture() : current_test_file_("") {
    // Initialize Chimaera once per test suite
    if (!g_initialized) {
      HLOG(kInfo, "Initializing Chimaera...");
      bool success = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
      if (success) {
        g_initialized = true;
        std::this_thread::sleep_for(500ms);
        HLOG(kInfo, "Chimaera initialization successful");
      } else {
        HLOG(kInfo, "Failed to initialize Chimaera");
      }
    }

    // Generate unique test file name
    current_test_file_ = kTestFilePrefix + std::to_string(getpid()) + "_" +
                         std::to_string(++g_test_counter) + ".dat";
  }

  ~BdevChimodFixture() { cleanup(); }

  /**
   * Create a test file with specified size
   */
  bool createTestFile(chi::u64 size = kDefaultFileSize) {
    // Create the test file
    std::ofstream file(current_test_file_, std::ios::binary);
    if (!file.is_open()) {
      return false;
    }

    // Write zeros to create file of specified size
    std::vector<char> buffer(4096, 0);
    chi::u64 written = 0;

    while (written < size) {
      chi::u64 to_write =
          std::min(static_cast<chi::u64>(buffer.size()), size - written);
      file.write(buffer.data(), to_write);
      if (!file.good()) {
        file.close();
        return false;
      }
      written += to_write;
    }

    file.close();

    // Verify file was created with correct size
    struct stat st;
    if (stat(current_test_file_.c_str(), &st) != 0) {
      return false;
    }

    return static_cast<chi::u64>(st.st_size) == size;
  }

  /**
   * Get the current test file path
   */
  const std::string& getTestFile() const { return current_test_file_; }

  /**
   * Generate test data with specified pattern
   */
  std::vector<hshm::u8> generateTestData(size_t size, hshm::u8 pattern = 0xAB) {
    std::vector<hshm::u8> data;
    data.reserve(size);

    // Create a repeating pattern
    for (size_t i = 0; i < size; ++i) {
      data.push_back(static_cast<hshm::u8>((pattern + i) % 256));
    }

    return data;
  }

  /**
   * Get the number of containers from environment variable
   * @return Number of containers in the distributed setup
   */
  chi::u32 getNumContainers() const {
    // First check CHI_NUM_CONTAINERS environment variable
    const char* num_containers_env = std::getenv("CHI_NUM_CONTAINERS");
    if (num_containers_env) {
      chi::u32 num_containers = std::atoi(num_containers_env);
      if (num_containers > 0) {
        HLOG(kInfo, "Using CHI_NUM_CONTAINERS={} from environment",
             num_containers);
        return num_containers;
      }
    }

    // Default to 1 for local/non-distributed tests
    HLOG(kInfo,
         "CHI_NUM_CONTAINERS not set, defaulting to 1 container (local test)");
    return 1;
  }

  /**
   * Validate that allocated blocks meet the requested size requirement
   * @param blocks Vector of allocated blocks
   * @param requested_size The originally requested allocation size
   * @return true if sum of block sizes >= requested_size
   */
  bool validateBlockAllocation(const std::vector<chimaera::bdev::Block>& blocks,
                               chi::u64 requested_size) const {
    if (blocks.empty()) {
      return false;
    }

    chi::u64 total_size = 0;
    for (const auto& block : blocks) {
      total_size += block.size_;
    }

    return total_size >= requested_size;
  }

  /**
   * Create bdev container using async API
   * @param client Reference to bdev client
   * @param pool_query Pool query for routing
   * @param pool_name Pool name (file path for file bdev)
   * @param pool_id Pool ID to use
   * @param bdev_type Type of bdev (file or RAM)
   * @param total_size Total size (0 = use file size)
   * @return true if creation succeeded
   */
  static bool CreateBdevAsync(chimaera::bdev::Client& client,
                              const chi::PoolQuery& pool_query,
                              const std::string& pool_name,
                              const chi::PoolId& pool_id,
                              chimaera::bdev::BdevType bdev_type,
                              chi::u64 total_size = 0) {
    auto create_task = client.AsyncCreate(pool_query, pool_name, pool_id,
                                          bdev_type, total_size);
    create_task.Wait();

    client.pool_id_ = create_task->new_pool_id_;
    client.return_code_ = create_task->return_code_;

    return create_task->GetReturnCode() == 0;
  }

  /**
   * Allocate blocks using async API
   * @param client Reference to bdev client
   * @param pool_query Pool query for routing
   * @param size Size to allocate
   * @return Vector of allocated blocks
   */
  static std::vector<chimaera::bdev::Block> AllocateBlocksAsync(
      chimaera::bdev::Client& client, const chi::PoolQuery& pool_query,
      chi::u64 size) {
    auto alloc_task = client.AsyncAllocateBlocks(pool_query, size);
    alloc_task.Wait();

    std::vector<chimaera::bdev::Block> blocks;
    for (size_t i = 0; i < alloc_task->blocks_.size(); ++i) {
      blocks.push_back(alloc_task->blocks_[i]);
    }
    return blocks;
  }

  /**
   * Get stats using async API
   * @param client Reference to bdev client
   * @param remaining_size Output parameter for remaining size
   * @return Performance metrics
   */
  static chimaera::bdev::PerfMetrics GetStatsAsync(
      chimaera::bdev::Client& client, chi::u64& remaining_size) {
    auto stats_task = client.AsyncGetStats();
    stats_task.Wait();

    chimaera::bdev::PerfMetrics metrics = stats_task->metrics_;
    remaining_size = stats_task->remaining_size_;
    return metrics;
  }

  /**
   * Clean up test resources
   */
  void cleanup() {
    // Remove test file if it exists
    if (!current_test_file_.empty()) {
      if (access(current_test_file_.c_str(), F_OK) == 0) {
        unlink(current_test_file_.c_str());
        HLOG(kInfo, "Cleaned up test file: {}", current_test_file_);
      }
    }
  }

 private:
  std::string current_test_file_;
};

}  // end anonymous namespace

//==============================================================================
// BASIC FUNCTIONALITY TESTS
//==============================================================================

TEST_CASE("bdev_container_creation", "[bdev][create]") {
  BdevChimodFixture fixture;

  SECTION("Initialize runtime and client") { REQUIRE(g_initialized); }

  SECTION("Create test file") {
    REQUIRE(fixture.createTestFile(kDefaultFileSize));
  }

  SECTION("Create bdev container with default parameters") {
    chi::PoolId custom_pool_id(100, 0);  // Custom pool ID for this container
    chimaera::bdev::Client client(custom_pool_id);

    bool success = BdevChimodFixture::CreateBdevAsync(
        client, chi::PoolQuery::Dynamic(), fixture.getTestFile(),
        custom_pool_id, chimaera::bdev::BdevType::kFile);
    REQUIRE(success);

    HLOG(kInfo, "Successfully created bdev container with default parameters");
  }
}

TEST_CASE("bdev_block_allocation_4kb", "[bdev][allocate][4kb]") {
  BdevChimodFixture fixture;

  SECTION("Setup") {
    REQUIRE(g_initialized);
    REQUIRE(fixture.createTestFile(kDefaultFileSize));
  }

  SECTION("Create container and allocate 4KB blocks") {
    chi::PoolId custom_pool_id(102, 0);
    chimaera::bdev::Client client(custom_pool_id);

    bool success = BdevChimodFixture::CreateBdevAsync(
        client, chi::PoolQuery::Dynamic(), fixture.getTestFile(),
        custom_pool_id, chimaera::bdev::BdevType::kFile);
    REQUIRE(success);

    // Allocate multiple 4KB blocks using DirectHash for distributed execution
    // Get number of containers from environment variable
    const chi::u32 num_containers = fixture.getNumContainers();
    HLOG(kInfo, "Running test with num_containers={}", num_containers);
    std::vector<chimaera::bdev::Block> blocks;

    for (int i = 0; i < 16; ++i) {
      auto pool_query = chi::PoolQuery::DirectHash(i);
      auto alloc_task = client.AsyncAllocateBlocks(pool_query, k4KB);
      alloc_task.Wait();
      REQUIRE(alloc_task->return_code_ == 0);
      REQUIRE(alloc_task->blocks_.size() > 0);

      chimaera::bdev::Block block = alloc_task->blocks_[0];
      REQUIRE(block.size_ >= k4KB);
      REQUIRE(block.block_type_ == 0);     // 4KB category
      REQUIRE(block.offset_ % 4096 == 0);  // Aligned

      // Verify that completer matches expected value based on DirectHash
      // Formula: expected_container_id = hash_value % num_containers
      chi::ContainerId expected_completer =
          static_cast<chi::ContainerId>(i % num_containers);
      chi::ContainerId actual_completer = alloc_task->GetCompleter();

      HLOG(kInfo,
           "Iteration {}: DirectHash({}) -> completer={}, expected={} "
           "(num_containers={})",
           i, i, actual_completer, expected_completer, num_containers);
      REQUIRE(actual_completer == expected_completer);
      HLOG(kInfo,
           "Allocated 4KB block {}: offset={}, size={}, completer={} "
           "(expected={})",
           i, block.offset_, block.size_, actual_completer, expected_completer);

      blocks.push_back(block);
    }

    // Note: We don't check if blocks overlap because blocks come from different
    // nodes in distributed execution, and each node has its own independent
    // storage space. Blocks from different nodes can have overlapping offsets
    // within their respective storage backends.
  }
}

TEST_CASE("bdev_write_read_basic", "[bdev][io][basic]") {
  BdevChimodFixture fixture;

  SECTION("Setup") {
    REQUIRE(g_initialized);
    REQUIRE(fixture.createTestFile(kDefaultFileSize));
  }

  SECTION("Write and read data verification") {
    chi::PoolId custom_pool_id(103, 0);
    chimaera::bdev::Client client(custom_pool_id);

    bool success = BdevChimodFixture::CreateBdevAsync(
        client, chi::PoolQuery::Dynamic(), fixture.getTestFile(),
        custom_pool_id, chimaera::bdev::BdevType::kFile);
    REQUIRE(success);

    // Run write/read operations using DirectHash for distributed execution
    // Get number of containers from environment variable
    const chi::u32 num_containers = fixture.getNumContainers();
    HLOG(kInfo, "Running test with num_containers={}", num_containers);

    for (int i = 0; i < 16; ++i) {
      auto pool_query = chi::PoolQuery::DirectHash(i);

      // Expected container ID based on DirectHash formula
      chi::ContainerId expected_completer =
          static_cast<chi::ContainerId>(i % num_containers);

      // Allocate a block
      auto alloc_task = client.AsyncAllocateBlocks(pool_query, k4KB);
      alloc_task.Wait();
      REQUIRE(alloc_task->return_code_ == 0);
      REQUIRE(alloc_task->blocks_.size() > 0);
      chimaera::bdev::Block block = alloc_task->blocks_[0];

      // Verify allocate task completer
      HLOG(kInfo,
           "Iteration {}: DirectHash({}) Allocate -> completer={}, expected={} "
           "(num_containers={})",
           i, i, alloc_task->GetCompleter(), expected_completer,
           num_containers);
      REQUIRE(alloc_task->GetCompleter() == expected_completer);

      // Generate test data
      std::vector<hshm::u8> write_data =
          fixture.generateTestData(k4KB, 0xCD + i);

      // Write data - allocate buffer and copy data
      auto write_buffer = CHI_IPC->AllocateBuffer(write_data.size());
      REQUIRE_FALSE(write_buffer.IsNull());
      memcpy(write_buffer.ptr_, write_data.data(), write_data.size());

      auto write_task = client.AsyncWrite(
          pool_query, WrapBlock(block),
          write_buffer.shm_.template Cast<void>().template Cast<void>(),
          write_data.size());
      write_task.Wait();
      REQUIRE(write_task->return_code_ == 0);
      REQUIRE(write_task->bytes_written_ == write_data.size());

      // Verify write task completer
      REQUIRE(write_task->GetCompleter() == expected_completer);
      HLOG(kInfo, "Iteration {}: Write completed by container {} (expected={})",
           i, write_task->GetCompleter(), expected_completer);

      // Read data back - allocate buffer for reading
      auto read_buffer = CHI_IPC->AllocateBuffer(k4KB);
      REQUIRE_FALSE(read_buffer.IsNull());

      auto read_task = client.AsyncRead(
          pool_query, WrapBlock(block),
          read_buffer.shm_.template Cast<void>().template Cast<void>(), k4KB);
      read_task.Wait();
      REQUIRE(read_task->return_code_ == 0);
      REQUIRE(read_task->bytes_read_ == write_data.size());

      // Verify read task completer
      REQUIRE(read_task->GetCompleter() == expected_completer);
      HLOG(kInfo, "Iteration {}: Read completed by container {} (expected={})",
           i, read_task->GetCompleter(), expected_completer);

      // Convert read data back to vector for verification
      std::vector<hshm::u8> read_data(read_task->bytes_read_);
      memcpy(read_data.data(), read_buffer.ptr_, read_task->bytes_read_);

      // Verify data matches
      for (size_t j = 0; j < write_data.size(); ++j) {
        REQUIRE(read_data[j] == write_data[j]);
      }

      // Free buffers
      CHI_IPC->FreeBuffer(write_buffer);
      CHI_IPC->FreeBuffer(read_buffer);

      HLOG(kInfo, "Iteration {}: Successfully wrote and read {} bytes", i,
           write_data.size());
    }
  }
}

TEST_CASE("bdev_async_operations", "[bdev][async][io]") {
  BdevChimodFixture fixture;

  SECTION("Setup") {
    REQUIRE(g_initialized);
    REQUIRE(fixture.createTestFile(kLargeFileSize));
  }

  SECTION("Async allocate, write, and read") {
    chi::PoolId custom_pool_id(104, 0);
    chimaera::bdev::Client client(custom_pool_id);

    bool success = BdevChimodFixture::CreateBdevAsync(
        client, chi::PoolQuery::Dynamic(), fixture.getTestFile(),
        custom_pool_id, chimaera::bdev::BdevType::kFile);
    REQUIRE(success);

    // Run async operations using DirectHash for distributed execution
    for (int i = 0; i < 16; ++i) {
      auto pool_query = chi::PoolQuery::DirectHash(i);

      // Async allocate
      auto alloc_task = client.AsyncAllocateBlocks(pool_query, k64KB);
      alloc_task.Wait();
      REQUIRE(alloc_task->return_code_ == 0);
      REQUIRE(alloc_task->blocks_.size() > 0);
      chimaera::bdev::Block block = alloc_task->blocks_[0];

      // Prepare test data
      std::vector<hshm::u8> write_data =
          fixture.generateTestData(k64KB, 0xEF + i);

      // Async write - allocate buffer and copy data
      auto async_write_buffer = CHI_IPC->AllocateBuffer(write_data.size());
      REQUIRE_FALSE(async_write_buffer.IsNull());
      memcpy(async_write_buffer.ptr_, write_data.data(), write_data.size());

      auto write_task = client.AsyncWrite(
          pool_query, WrapBlock(block),
          async_write_buffer.shm_.template Cast<void>().template Cast<void>(),
          write_data.size());
      write_task.Wait();
      REQUIRE(write_task->return_code_ == 0);
      REQUIRE(write_task->bytes_written_ == write_data.size());

      // Async read - allocate buffer for reading
      auto async_read_buffer = CHI_IPC->AllocateBuffer(k64KB);
      REQUIRE_FALSE(async_read_buffer.IsNull());

      auto read_task = client.AsyncRead(
          pool_query, WrapBlock(block),
          async_read_buffer.shm_.template Cast<void>().template Cast<void>(),
          k64KB);
      read_task.Wait();
      REQUIRE(read_task->return_code_ == 0);
      REQUIRE(read_task->bytes_read_ == write_data.size());

      // Verify data - copy from buffer to check
      std::vector<hshm::u8> async_read_data(read_task->bytes_read_);
      memcpy(async_read_data.data(), async_read_buffer.ptr_,
             read_task->bytes_read_);

      REQUIRE(async_read_data.size() == write_data.size());
      for (size_t j = 0; j < write_data.size(); ++j) {
        REQUIRE(async_read_data[j] == write_data[j]);
      }

      // Free buffers
      CHI_IPC->FreeBuffer(async_write_buffer);
      CHI_IPC->FreeBuffer(async_read_buffer);

      HLOG(kInfo,
           "Iteration {}: Successfully completed async allocate/write/read "
           "cycle",
           i);
    }
  }
}

TEST_CASE("bdev_performance_metrics", "[bdev][performance][metrics]") {
  BdevChimodFixture fixture;

  SECTION("Setup") {
    REQUIRE(g_initialized);
    REQUIRE(fixture.createTestFile(kLargeFileSize));
  }

  SECTION("Track performance metrics during operations") {
    REQUIRE(g_initialized);
    REQUIRE(fixture.createTestFile(kLargeFileSize));

    chi::PoolId custom_pool_id(105, 0);
    chimaera::bdev::Client client(custom_pool_id);

    // Use Broadcast to ensure fresh pool creation with current file
    bool success = BdevChimodFixture::CreateBdevAsync(
        client, chi::PoolQuery::Broadcast(), fixture.getTestFile(),
        custom_pool_id, chimaera::bdev::BdevType::kFile);
    REQUIRE(success);

    // Get initial stats
    chi::u64 initial_remaining;
    auto stats_task = client.AsyncGetStats();
    stats_task.Wait();
    HLOG(kInfo, "Test: GetStats return_code={}, remaining_size={}",
         stats_task->GetReturnCode(), stats_task->remaining_size_);
    chimaera::bdev::PerfMetrics initial_metrics = stats_task->metrics_;
    initial_remaining = stats_task->remaining_size_;

    HLOG(kInfo, "Test: Got initial_remaining={}", initial_remaining);
    REQUIRE(initial_remaining > 0);
    REQUIRE(initial_metrics.read_bandwidth_mbps_ >= 0.0);
    REQUIRE(initial_metrics.write_bandwidth_mbps_ >= 0.0);

    // Perform I/O operations using DirectHash for distributed execution
    for (int i = 0; i < 16; ++i) {
      auto pool_query = chi::PoolQuery::DirectHash(i);

      // Allocate blocks
      auto alloc_task1 = client.AsyncAllocateBlocks(pool_query, k1MB);
      alloc_task1.Wait();
      REQUIRE(alloc_task1->return_code_ == 0);
      REQUIRE(alloc_task1->blocks_.size() > 0);
      chimaera::bdev::Block block1 = alloc_task1->blocks_[0];

      auto alloc_task2 = client.AsyncAllocateBlocks(pool_query, k256KB);
      alloc_task2.Wait();
      REQUIRE(alloc_task2->return_code_ == 0);
      REQUIRE(alloc_task2->blocks_.size() > 0);
      chimaera::bdev::Block block2 = alloc_task2->blocks_[0];

      std::vector<hshm::u8> data1 = fixture.generateTestData(k1MB, 0x12 + i);
      std::vector<hshm::u8> data2 = fixture.generateTestData(k256KB, 0x34 + i);

      // Allocate buffers for data1 write
      auto data1_write_buffer = CHI_IPC->AllocateBuffer(data1.size());
      REQUIRE_FALSE(data1_write_buffer.IsNull());
      memcpy(data1_write_buffer.ptr_, data1.data(), data1.size());

      auto write_task1 = client.AsyncWrite(
          pool_query, WrapBlock(block1),
          data1_write_buffer.shm_.template Cast<void>().template Cast<void>(),
          data1.size());
      write_task1.Wait();
      REQUIRE(write_task1->return_code_ == 0);

      // Allocate buffers for data2 write
      auto data2_write_buffer = CHI_IPC->AllocateBuffer(data2.size());
      REQUIRE_FALSE(data2_write_buffer.IsNull());
      memcpy(data2_write_buffer.ptr_, data2.data(), data2.size());

      auto write_task2 = client.AsyncWrite(
          pool_query, WrapBlock(block2),
          data2_write_buffer.shm_.template Cast<void>().template Cast<void>(),
          data2.size());
      write_task2.Wait();
      REQUIRE(write_task2->return_code_ == 0);

      // Allocate buffers for reads
      auto data1_read_buffer = CHI_IPC->AllocateBuffer(k1MB);
      REQUIRE_FALSE(data1_read_buffer.IsNull());

      auto read_task1 = client.AsyncRead(
          pool_query, WrapBlock(block1),
          data1_read_buffer.shm_.template Cast<void>().template Cast<void>(),
          k1MB);
      read_task1.Wait();
      REQUIRE(read_task1->return_code_ == 0);

      auto data2_read_buffer = CHI_IPC->AllocateBuffer(k256KB);
      REQUIRE_FALSE(data2_read_buffer.IsNull());

      auto read_task2 = client.AsyncRead(
          pool_query, WrapBlock(block2),
          data2_read_buffer.shm_.template Cast<void>().template Cast<void>(),
          k256KB);
      read_task2.Wait();
      REQUIRE(read_task2->return_code_ == 0);

      // Free buffers
      CHI_IPC->FreeBuffer(data1_write_buffer);
      CHI_IPC->FreeBuffer(data2_write_buffer);
      CHI_IPC->FreeBuffer(data1_read_buffer);
      CHI_IPC->FreeBuffer(data2_read_buffer);

      HLOG(kInfo, "Iteration {}: Completed I/O operations", i);
    }

    // Get updated stats
    chi::u64 final_remaining;
    chimaera::bdev::PerfMetrics final_metrics =
        BdevChimodFixture::GetStatsAsync(client, final_remaining);

    // Remaining space should have decreased
    REQUIRE(final_remaining < initial_remaining);

    // Performance metrics should be updated (may be zero for very fast
    // operations)
    REQUIRE(final_metrics.read_bandwidth_mbps_ >= 0.0);
    REQUIRE(final_metrics.write_bandwidth_mbps_ >= 0.0);
    REQUIRE(final_metrics.iops_ >= 0.0);

    HLOG(kInfo, "Initial remaining: {} bytes, Final remaining: {} bytes",
         initial_remaining, final_remaining);
    HLOG(kInfo, "Read BW: {} MB/s", final_metrics.read_bandwidth_mbps_);
    HLOG(kInfo, "Write BW: {} MB/s", final_metrics.write_bandwidth_mbps_);
    HLOG(kInfo, "IOPS: {}", final_metrics.iops_);
  }
}

TEST_CASE("bdev_error_conditions", "[bdev][error][edge_cases]") {
  BdevChimodFixture fixture;

  SECTION("Setup") {
    REQUIRE(g_initialized);
    REQUIRE(fixture.createTestFile(kDefaultFileSize));
  }

  SECTION("Handle invalid file paths") {
    chi::PoolId custom_pool_id(106, 0);
    chimaera::bdev::Client client(custom_pool_id);

    // Try to create with non-existent file
    auto create_task = client.AsyncCreate(
        chi::PoolQuery::Broadcast(), "/nonexistent/path/file.dat",
        custom_pool_id, chimaera::bdev::BdevType::kFile);
    create_task.Wait();
    REQUIRE(create_task->return_code_ != 0);  // Should fail

    HLOG(kInfo, "Correctly handled invalid file path");
  }
}

//==============================================================================
// RAM BACKEND TESTS
//==============================================================================

TEST_CASE("bdev_ram_container_creation", "[bdev][ram][create]") {
  BdevChimodFixture fixture;
  REQUIRE(g_initialized);

  // Admin client is automatically initialized via CHI_ADMIN singleton
  std::this_thread::sleep_for(100ms);

  // Create bdev client for RAM backend
  chi::PoolId custom_pool_id(8001, 0);
  chimaera::bdev::Client bdev_client(custom_pool_id);

  // Create RAM-based bdev container (1MB)
  const chi::u64 ram_size = 1024 * 1024;
  std::string pool_name =
      "ram_test_" + std::to_string(getpid()) + "_" + std::to_string(8001);
  bool bdev_success = BdevChimodFixture::CreateBdevAsync(
      bdev_client, chi::PoolQuery::Dynamic(), pool_name, custom_pool_id,
      chimaera::bdev::BdevType::kRam, ram_size);
  REQUIRE(bdev_success);

  std::this_thread::sleep_for(100ms);

  HLOG(kInfo, "RAM backend container created successfully with size: {} bytes",
       ram_size);
}

TEST_CASE("bdev_ram_allocation_and_io", "[bdev][ram][io]") {
  BdevChimodFixture fixture;
  REQUIRE(g_initialized);

  // Admin client is automatically initialized via CHI_ADMIN singleton
  std::this_thread::sleep_for(100ms);

  // Create bdev client for RAM backend
  chi::PoolId custom_pool_id(8002, 0);
  chimaera::bdev::Client bdev_client(custom_pool_id);

  // Create RAM-based bdev container (1MB)
  const chi::u64 ram_size = 1024 * 1024;
  std::string pool_name =
      "ram_test_" + std::to_string(getpid()) + "_" + std::to_string(8002);
  bool bdev_success = BdevChimodFixture::CreateBdevAsync(
      bdev_client, chi::PoolQuery::Dynamic(), pool_name, custom_pool_id,
      chimaera::bdev::BdevType::kRam, ram_size);
  REQUIRE(bdev_success);
  std::this_thread::sleep_for(100ms);

  // Run I/O operations using DirectHash for distributed execution
  for (int i = 0; i < 16; ++i) {
    auto pool_query = chi::PoolQuery::DirectHash(i);

    // Allocate a 4KB block
    auto alloc_task = bdev_client.AsyncAllocateBlocks(pool_query, k4KB);
    alloc_task.Wait();
    REQUIRE(alloc_task->return_code_ == 0);
    REQUIRE(alloc_task->blocks_.size() > 0);
    chimaera::bdev::Block block = alloc_task->blocks_[0];
    REQUIRE(block.size_ == k4KB);
    REQUIRE(block.offset_ < ram_size);

    // Verify that completer_ is set (can be 0, which is a valid container ID)
    chi::ContainerId completer = alloc_task->GetCompleter();
    HLOG(kInfo, "Iteration {}: Task completed by container {}", i, completer);

    // Prepare test data with pattern
    std::vector<hshm::u8> write_data(k4KB);
    for (size_t j = 0; j < write_data.size(); ++j) {
      write_data[j] = static_cast<hshm::u8>((j + 0xAB + i) % 256);
    }

    // Write data to RAM - allocate buffer and copy data
    auto write_buffer = CHI_IPC->AllocateBuffer(write_data.size());
    REQUIRE_FALSE(write_buffer.IsNull());
    memcpy(write_buffer.ptr_, write_data.data(), write_data.size());

    auto write_task = bdev_client.AsyncWrite(
        pool_query, WrapBlock(block),
        write_buffer.shm_.template Cast<void>().template Cast<void>(),
        write_data.size());
    write_task.Wait();
    REQUIRE(write_task->return_code_ == 0);
    REQUIRE(write_task->bytes_written_ == k4KB);

    // Read data back from RAM - allocate buffer for reading
    auto read_buffer = CHI_IPC->AllocateBuffer(k4KB);
    REQUIRE_FALSE(read_buffer.IsNull());

    auto read_task = bdev_client.AsyncRead(
        pool_query, WrapBlock(block),
        read_buffer.shm_.template Cast<void>().template Cast<void>(), k4KB);
    read_task.Wait();
    REQUIRE(read_task->return_code_ == 0);
    REQUIRE(read_task->bytes_read_ == k4KB);

    // Convert read data back to vector for verification
    std::vector<hshm::u8> read_data(read_task->bytes_read_);
    memcpy(read_data.data(), read_buffer.ptr_, read_task->bytes_read_);

    // Verify data integrity
    bool data_matches =
        std::equal(write_data.begin(), write_data.end(), read_data.begin());
    REQUIRE(data_matches);

    // Free buffers
    CHI_IPC->FreeBuffer(write_buffer);
    CHI_IPC->FreeBuffer(read_buffer);

    // Free the block
    std::vector<chimaera::bdev::Block> free_blocks;
    free_blocks.push_back(block);
    auto free_task = bdev_client.AsyncFreeBlocks(pool_query, free_blocks);
    free_task.Wait();
    REQUIRE(free_task->return_code_ == 0);

    HLOG(kInfo,
         "Iteration {}: RAM backend I/O operations completed successfully", i);
  }
}

TEST_CASE("bdev_ram_large_blocks", "[bdev][ram][large]") {
  BdevChimodFixture fixture;
  REQUIRE(g_initialized);

  // Admin client is automatically initialized via CHI_ADMIN singleton
  std::this_thread::sleep_for(100ms);

  // Create bdev client for RAM backend
  chi::PoolId custom_pool_id(8003, 0);
  chimaera::bdev::Client bdev_client(custom_pool_id);

  // Create RAM-based bdev container (32MB to accommodate multiple 1MB
  // allocations)
  const chi::u64 ram_size = 32 * 1024 * 1024;
  std::string pool_name =
      "ram_test_" + std::to_string(getpid()) + "_" + std::to_string(8003);
  bool bdev_success = BdevChimodFixture::CreateBdevAsync(
      bdev_client, chi::PoolQuery::Dynamic(), pool_name, custom_pool_id,
      chimaera::bdev::BdevType::kRam, ram_size);
  REQUIRE(bdev_success);
  std::this_thread::sleep_for(100ms);

  // Test different block sizes using DirectHash for distributed execution
  std::vector<chi::u64> block_sizes = {k4KB, k64KB, k256KB, k1MB};

  for (chi::u64 block_size : block_sizes) {
    HLOG(kInfo, "Testing RAM backend with block size: {} bytes", block_size);

    for (int i = 0; i < 16; ++i) {
      auto pool_query = chi::PoolQuery::DirectHash(i);

      // Allocate block
      auto alloc_task = bdev_client.AsyncAllocateBlocks(pool_query, block_size);
      alloc_task.Wait();
      REQUIRE(alloc_task->return_code_ == 0);
      REQUIRE(alloc_task->blocks_.size() > 0);

      // Convert hipc::vector to std::vector for validation
      std::vector<chimaera::bdev::Block> blocks;
      for (size_t j = 0; j < alloc_task->blocks_.size(); ++j) {
        blocks.push_back(alloc_task->blocks_[j]);
      }

      REQUIRE(fixture.validateBlockAllocation(blocks, block_size));
      HLOG(kInfo, "Allocated {} blocks for requested size = {}", blocks.size(),
           block_size);

      // Create test pattern
      std::vector<hshm::u8> test_data(block_size);
      for (size_t j = 0; j < test_data.size(); j += 1024) {
        test_data[j] = static_cast<hshm::u8>(((j / 1024) + i) % 256);
      }

      // Write and read - allocate buffers
      auto test_write_buffer = CHI_IPC->AllocateBuffer(test_data.size());
      REQUIRE_FALSE(test_write_buffer.IsNull());
      memcpy(test_write_buffer.ptr_, test_data.data(), test_data.size());

      // Pass all allocated blocks to Write
      auto write_task = bdev_client.AsyncWrite(
          pool_query, ConvertBlocks(blocks),
          test_write_buffer.shm_.template Cast<void>().template Cast<void>(),
          test_data.size());
      write_task.Wait();
      REQUIRE(write_task->return_code_ == 0);
      REQUIRE(write_task->bytes_written_ == block_size);

      auto test_read_buffer = CHI_IPC->AllocateBuffer(block_size);
      REQUIRE_FALSE(test_read_buffer.IsNull());

      // Pass all allocated blocks to Read
      auto read_task = bdev_client.AsyncRead(
          pool_query, ConvertBlocks(blocks),
          test_read_buffer.shm_.template Cast<void>().template Cast<void>(),
          block_size);
      read_task.Wait();
      REQUIRE(read_task->return_code_ == 0);
      REQUIRE(read_task->bytes_read_ == block_size);

      // Convert read data back to vector for verification
      std::vector<hshm::u8> read_data(read_task->bytes_read_);
      memcpy(read_data.data(), test_read_buffer.ptr_, read_task->bytes_read_);

      // Verify critical points in the data
      for (size_t j = 0; j < read_data.size(); j += 1024) {
        REQUIRE(read_data[j] == test_data[j]);
      }

      // Free buffers
      CHI_IPC->FreeBuffer(test_write_buffer);
      CHI_IPC->FreeBuffer(test_read_buffer);

      // Free all allocated blocks
      auto free_task = bdev_client.AsyncFreeBlocks(pool_query, blocks);
      free_task.Wait();
      REQUIRE(free_task->return_code_ == 0);
    }
  }

  HLOG(kInfo, "RAM backend large block tests completed");
}

TEST_CASE("bdev_ram_bounds_checking", "[bdev][ram][bounds]") {
  BdevChimodFixture fixture;
  REQUIRE(g_initialized);

  // Admin client is automatically initialized via CHI_ADMIN singleton
  std::this_thread::sleep_for(100ms);

  // Create bdev client for RAM backend
  chi::PoolId custom_pool_id(8005, 0);
  chimaera::bdev::Client bdev_client(custom_pool_id);

  // Create small RAM-based bdev container (64KB)
  const chi::u64 ram_size = 64 * 1024;
  std::string pool_name =
      "ram_test_" + std::to_string(getpid()) + "_" + std::to_string(8005);
  bool bdev_success = BdevChimodFixture::CreateBdevAsync(
      bdev_client, chi::PoolQuery::Dynamic(), pool_name, custom_pool_id,
      chimaera::bdev::BdevType::kRam, ram_size);
  REQUIRE(bdev_success);
  std::this_thread::sleep_for(100ms);

  // Test bounds checking using DirectHash for distributed execution
  for (int i = 0; i < 16; ++i) {
    auto pool_query = chi::PoolQuery::DirectHash(i);

    // Create a block that would go beyond bounds
    chimaera::bdev::Block out_of_bounds_block;
    out_of_bounds_block.offset_ = ram_size - 1024;  // Near end of buffer
    out_of_bounds_block.size_ = 2048;               // Extends beyond buffer
    out_of_bounds_block.block_type_ = 0;

    // Prepare test data
    std::vector<hshm::u8> test_data(2048, 0xEF + i);

    // Write should fail with bounds check - allocate buffer
    auto error_write_buffer = CHI_IPC->AllocateBuffer(test_data.size());
    REQUIRE_FALSE(error_write_buffer.IsNull());
    memcpy(error_write_buffer.ptr_, test_data.data(), test_data.size());

    auto write_task = bdev_client.AsyncWrite(
        pool_query, WrapBlock(out_of_bounds_block),
        error_write_buffer.shm_.template Cast<void>().template Cast<void>(),
        test_data.size());
    write_task.Wait();
    REQUIRE(write_task->bytes_written_ == 0);  // Should fail

    // Read should also fail with bounds check - allocate buffer
    auto error_read_buffer = CHI_IPC->AllocateBuffer(2048);
    REQUIRE_FALSE(error_read_buffer.IsNull());

    auto read_task = bdev_client.AsyncRead(
        pool_query, WrapBlock(out_of_bounds_block),
        error_read_buffer.shm_.template Cast<void>().template Cast<void>(),
        2048);
    read_task.Wait();
    REQUIRE(read_task->bytes_read_ == 0);  // Should fail

    // Convert read data back to vector (should be empty due to error)
    std::vector<hshm::u8> read_data(read_task->bytes_read_);
    if (read_task->bytes_read_ > 0) {
      memcpy(read_data.data(), error_read_buffer.ptr_, read_task->bytes_read_);
    }
    REQUIRE(read_data.empty());  // Should fail

    // Free buffers
    CHI_IPC->FreeBuffer(error_write_buffer);
    CHI_IPC->FreeBuffer(error_read_buffer);

    HLOG(kInfo, "Iteration {}: RAM backend bounds checking working correctly",
         i);
  }
}

//==============================================================================
// FILE BACKEND TESTS (Enhanced)
//==============================================================================

TEST_CASE("bdev_file_vs_ram_comparison", "[bdev][file][ram][comparison]") {
  BdevChimodFixture fixture;
  REQUIRE(g_initialized);
  REQUIRE(fixture.createTestFile(kDefaultFileSize));

  // Admin client is automatically initialized via CHI_ADMIN singleton
  std::this_thread::sleep_for(100ms);

  // Create two bdev clients - one for file, one for RAM
  chi::PoolId file_pool_id(8006, 0);
  chi::PoolId ram_pool_id(8007, 0);
  chimaera::bdev::Client file_client(file_pool_id);
  chimaera::bdev::Client ram_client(ram_pool_id);

  // Create file-based container
  bool file_success = BdevChimodFixture::CreateBdevAsync(
      file_client, chi::PoolQuery::Dynamic(), fixture.getTestFile(),
      file_pool_id, chimaera::bdev::BdevType::kFile);
  REQUIRE(file_success);
  std::this_thread::sleep_for(100ms);

  // Create RAM-based container (same size as file)
  std::string ram_pool_name =
      "ram_comparison_" + std::to_string(getpid()) + "_" + std::to_string(8007);
  bool ram_success = BdevChimodFixture::CreateBdevAsync(
      ram_client, chi::PoolQuery::Dynamic(), ram_pool_name, ram_pool_id,
      chimaera::bdev::BdevType::kRam, kDefaultFileSize);
  REQUIRE(ram_success);
  std::this_thread::sleep_for(100ms);

  // Test same operations on both backends using DirectHash for distributed
  // execution
  const chi::u64 test_size = k64KB;

  for (int i = 0; i < 16; ++i) {
    auto pool_query = chi::PoolQuery::DirectHash(i);

    // Allocate blocks on both
    auto file_alloc_task =
        file_client.AsyncAllocateBlocks(pool_query, test_size);
    file_alloc_task.Wait();
    REQUIRE(file_alloc_task->return_code_ == 0);
    REQUIRE(file_alloc_task->blocks_.size() > 0);
    chimaera::bdev::Block file_block = file_alloc_task->blocks_[0];

    auto ram_alloc_task = ram_client.AsyncAllocateBlocks(pool_query, test_size);
    ram_alloc_task.Wait();
    REQUIRE(ram_alloc_task->return_code_ == 0);
    REQUIRE(ram_alloc_task->blocks_.size() > 0);
    chimaera::bdev::Block ram_block = ram_alloc_task->blocks_[0];

    REQUIRE(file_block.size_ == test_size);
    REQUIRE(ram_block.size_ == test_size);

    // Create identical test data
    std::vector<hshm::u8> test_data(test_size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);

    for (size_t j = 0; j < test_data.size(); ++j) {
      test_data[j] = static_cast<hshm::u8>(dis(gen));
    }

    // Allocate buffer for file write
    auto file_write_buffer = CHI_IPC->AllocateBuffer(test_data.size());
    REQUIRE_FALSE(file_write_buffer.IsNull());
    memcpy(file_write_buffer.ptr_, test_data.data(), test_data.size());

    // Write to file backend and measure time
    auto file_write_start = std::chrono::high_resolution_clock::now();
    auto file_write_task = file_client.AsyncWrite(
        pool_query, WrapBlock(file_block),
        file_write_buffer.shm_.template Cast<void>().template Cast<void>(),
        test_data.size());
    file_write_task.Wait();
    auto file_write_end = std::chrono::high_resolution_clock::now();

    REQUIRE(file_write_task->return_code_ == 0);
    REQUIRE(file_write_task->bytes_written_ == test_size);

    // Allocate buffer for ram write
    auto ram_write_buffer = CHI_IPC->AllocateBuffer(test_data.size());
    REQUIRE_FALSE(ram_write_buffer.IsNull());
    memcpy(ram_write_buffer.ptr_, test_data.data(), test_data.size());

    auto ram_write_start = std::chrono::high_resolution_clock::now();
    auto ram_write_task = ram_client.AsyncWrite(
        pool_query, WrapBlock(ram_block),
        ram_write_buffer.shm_.template Cast<void>().template Cast<void>(),
        test_data.size());
    ram_write_task.Wait();
    auto ram_write_end = std::chrono::high_resolution_clock::now();

    REQUIRE(ram_write_task->return_code_ == 0);
    REQUIRE(ram_write_task->bytes_written_ == test_size);

    // Allocate buffer for file read
    auto file_read_buffer = CHI_IPC->AllocateBuffer(test_size);
    REQUIRE_FALSE(file_read_buffer.IsNull());

    // Read from file backend and measure time
    auto file_read_start = std::chrono::high_resolution_clock::now();
    auto file_read_task = file_client.AsyncRead(
        pool_query, WrapBlock(file_block),
        file_read_buffer.shm_.template Cast<void>().template Cast<void>(),
        test_size);
    file_read_task.Wait();
    auto file_read_end = std::chrono::high_resolution_clock::now();

    REQUIRE(file_read_task->return_code_ == 0);
    REQUIRE(file_read_task->bytes_read_ == test_size);

    // Convert read data back to vector
    std::vector<hshm::u8> file_read_data(file_read_task->bytes_read_);
    memcpy(file_read_data.data(), file_read_buffer.ptr_,
           file_read_task->bytes_read_);

    // Allocate buffer for ram read
    auto ram_read_buffer = CHI_IPC->AllocateBuffer(test_size);
    REQUIRE_FALSE(ram_read_buffer.IsNull());

    auto ram_read_start = std::chrono::high_resolution_clock::now();
    auto ram_read_task = ram_client.AsyncRead(
        pool_query, WrapBlock(ram_block),
        ram_read_buffer.shm_.template Cast<void>().template Cast<void>(),
        test_size);
    ram_read_task.Wait();
    auto ram_read_end = std::chrono::high_resolution_clock::now();

    REQUIRE(ram_read_task->return_code_ == 0);
    REQUIRE(ram_read_task->bytes_read_ == test_size);

    // Convert read data back to vector
    std::vector<hshm::u8> ram_read_data(ram_read_task->bytes_read_);
    memcpy(ram_read_data.data(), ram_read_buffer.ptr_,
           ram_read_task->bytes_read_);

    REQUIRE(file_read_data.size() == test_size);
    REQUIRE(ram_read_data.size() == test_size);

    // Verify data integrity on both
    bool file_data_ok =
        std::equal(test_data.begin(), test_data.end(), file_read_data.begin());
    bool ram_data_ok =
        std::equal(test_data.begin(), test_data.end(), ram_read_data.begin());

    REQUIRE(file_data_ok);
    REQUIRE(ram_data_ok);

    // Calculate and compare performance
    auto file_write_time = std::chrono::duration<double, std::micro>(
        file_write_end - file_write_start);
    auto ram_write_time = std::chrono::duration<double, std::micro>(
        ram_write_end - ram_write_start);
    auto file_read_time = std::chrono::duration<double, std::micro>(
        file_read_end - file_read_start);
    auto ram_read_time = std::chrono::duration<double, std::micro>(
        ram_read_end - ram_read_start);

    HLOG(kInfo, "Iteration {} - Performance Comparison (64KB operations):", i);
    HLOG(kInfo, "  File Write: {} μs", file_write_time.count());
    HLOG(kInfo, "  RAM Write:  {} μs", ram_write_time.count());
    HLOG(kInfo, "  File Read:  {} μs", file_read_time.count());
    HLOG(kInfo, "  RAM Read:   {} μs", ram_read_time.count());

    // Note: In distributed mode with network overhead, RAM may not always be
    // faster than file due to serialization/network costs. We verify operations
    // complete successfully but don't enforce strict performance requirements
    // in distributed tests.

    // Free buffers
    CHI_IPC->FreeBuffer(file_write_buffer);
    CHI_IPC->FreeBuffer(ram_write_buffer);
    CHI_IPC->FreeBuffer(file_read_buffer);
    CHI_IPC->FreeBuffer(ram_read_buffer);

    // Clean up
    std::vector<chimaera::bdev::Block> file_free_blocks;
    file_free_blocks.push_back(file_block);
    auto file_free_task =
        file_client.AsyncFreeBlocks(pool_query, file_free_blocks);
    file_free_task.Wait();
    REQUIRE(file_free_task->return_code_ == 0);

    std::vector<chimaera::bdev::Block> ram_free_blocks;
    ram_free_blocks.push_back(ram_block);
    auto ram_free_task =
        ram_client.AsyncFreeBlocks(pool_query, ram_free_blocks);
    ram_free_task.Wait();
    REQUIRE(ram_free_task->return_code_ == 0);
  }
}

/**
 * Helper: runs the bdev file explicit backend write/read test.
 * Called by per-mode TEST_CASEs (SHM, TCP, IPC).
 * Each mode must run in a separate process because g_initialized
 * prevents re-initialization with a different CHI_IPC_MODE.
 */
void run_bdev_file_explicit_backend_test(const char *mode_name) {
  HLOG(kInfo, "[bdev_file_explicit_backend_{}] TEST START", mode_name);
  BdevChimodFixture fixture;
  REQUIRE(g_initialized);
  REQUIRE(fixture.createTestFile(kDefaultFileSize));

  std::this_thread::sleep_for(100ms);

  // Create bdev client with explicit file backend
  chi::PoolId custom_pool_id(8008, 0);
  chimaera::bdev::Client bdev_client(custom_pool_id);

  auto create_task = bdev_client.AsyncCreate(
      chi::PoolQuery::Dynamic(), fixture.getTestFile(), custom_pool_id,
      chimaera::bdev::BdevType::kFile, 0, 32, 4096);
  create_task.Wait();
  bdev_client.pool_id_ = create_task->new_pool_id_;
  bdev_client.return_code_ = create_task->return_code_;
  REQUIRE(create_task->GetReturnCode() == 0);
  std::this_thread::sleep_for(100ms);

  const chi::u32 num_containers = fixture.getNumContainers();
  HLOG(kInfo, "[bdev_file_explicit_backend_{}] num_containers={}",
       mode_name, num_containers);

  for (int i = 0; i < 16; ++i) {
    HLOG(kInfo, "[bdev_file_explicit_backend_{}] === ITERATION {} START ===",
         mode_name, i);
    auto pool_query = chi::PoolQuery::DirectHash(i);

    // Allocate block
    auto alloc_task = bdev_client.AsyncAllocateBlocks(pool_query, k4KB);
    alloc_task.Wait();
    REQUIRE(alloc_task->return_code_ == 0);
    REQUIRE(alloc_task->blocks_.size() > 0);
    chimaera::bdev::Block block = alloc_task->blocks_[0];
    REQUIRE(block.size_ == k4KB);

    // Write data
    std::vector<hshm::u8> test_data(k4KB, 0x42 + i);
    auto final_write_buffer = CHI_IPC->AllocateBuffer(test_data.size());
    REQUIRE_FALSE(final_write_buffer.IsNull());
    memcpy(final_write_buffer.ptr_, test_data.data(), test_data.size());

    auto write_task = bdev_client.AsyncWrite(
        pool_query, WrapBlock(block),
        final_write_buffer.shm_.template Cast<void>().template Cast<void>(),
        test_data.size());
    write_task.Wait();
    REQUIRE(write_task->return_code_ == 0);
    REQUIRE(write_task->bytes_written_ == k4KB);

    // Read data back
    auto final_read_buffer = CHI_IPC->AllocateBuffer(k4KB);
    REQUIRE_FALSE(final_read_buffer.IsNull());

    auto read_task = bdev_client.AsyncRead(
        pool_query, WrapBlock(block),
        final_read_buffer.shm_.template Cast<void>().template Cast<void>(),
        k4KB);
    read_task.Wait();
    REQUIRE(read_task->return_code_ == 0);
    REQUIRE(read_task->bytes_read_ == k4KB);

    // Verify data
    std::vector<hshm::u8> read_data(read_task->bytes_read_);
    memcpy(read_data.data(), final_read_buffer.ptr_, read_task->bytes_read_);
    bool data_ok =
        std::equal(test_data.begin(), test_data.end(), read_data.begin());
    HLOG(kInfo,
         "[bdev_file_explicit_backend_{}] Iteration {}: data_ok={}",
         mode_name, i, data_ok);
    REQUIRE(data_ok);

    // Free buffers
    CHI_IPC->FreeBuffer(final_write_buffer);
    CHI_IPC->FreeBuffer(final_read_buffer);

    // Free blocks
    std::vector<chimaera::bdev::Block> free_blocks;
    free_blocks.push_back(block);
    auto free_task = bdev_client.AsyncFreeBlocks(pool_query, free_blocks);
    free_task.Wait();
    REQUIRE(free_task->return_code_ == 0);

    HLOG(kInfo,
         "[bdev_file_explicit_backend_{}] === ITERATION {} COMPLETE ===",
         mode_name, i);
  }
  HLOG(kInfo,
       "[bdev_file_explicit_backend_{}] TEST COMPLETE - All 16 iterations "
       "passed",
       mode_name);
}

TEST_CASE("bdev_file_explicit_backend_shm", "[bdev][file][explicit][shm]") {
  setenv("CHI_IPC_MODE", "SHM", 1);
  run_bdev_file_explicit_backend_test("shm");
}

TEST_CASE("bdev_file_explicit_backend_tcp", "[bdev][file][explicit][tcp]") {
  setenv("CHI_IPC_MODE", "TCP", 1);
  run_bdev_file_explicit_backend_test("tcp");
}

TEST_CASE("bdev_file_explicit_backend_ipc", "[bdev][file][explicit][ipc]") {
  setenv("CHI_IPC_MODE", "IPC", 1);
  run_bdev_file_explicit_backend_test("ipc");
}

TEST_CASE("bdev_error_conditions_enhanced", "[bdev][error][enhanced]") {
  BdevChimodFixture fixture;
  REQUIRE(g_initialized);

  // Admin client is automatically initialized via CHI_ADMIN singleton
  std::this_thread::sleep_for(100ms);

  // Test 1: RAM backend without size specification
  {
    chi::PoolId custom_pool_id(8009, 0);
    chimaera::bdev::Client ram_client_no_size(custom_pool_id);

    // This should fail because RAM backend requires explicit size
    std::string pool_name = "ram_fail_test_" + std::to_string(getpid());
    bool creation_success = BdevChimodFixture::CreateBdevAsync(
        ram_client_no_size, chi::PoolQuery::Dynamic(), pool_name,
        custom_pool_id, chimaera::bdev::BdevType::kRam,
        0);  // Size 0 should fail
    std::this_thread::sleep_for(100ms);

    // Creation should fail for RAM backend with zero size
    bool creation_failed = !creation_success;

    // If creation didn't fail at the Create level, test allocation to see if
    // the container is invalid
    if (!creation_failed) {
      std::vector<chimaera::bdev::Block> blocks =
          BdevChimodFixture::AllocateBlocksAsync(ram_client_no_size,
                                                 chi::PoolQuery::Local(), k4KB);
      creation_failed = (blocks.size() == 0);  // Should be invalid block list
    }

    HLOG(kInfo, "RAM backend properly rejects zero size: {}",
         creation_failed ? "YES" : "NO");
  }

  // Test 2: File backend with non-existent file
  {
    chi::PoolId custom_pool_id(8010, 0);
    chimaera::bdev::Client file_client_bad_path(custom_pool_id);

    // This should fail because the file path doesn't exist
    bool creation_success = BdevChimodFixture::CreateBdevAsync(
        file_client_bad_path, chi::PoolQuery::Dynamic(),
        "/nonexistent/path/file.dat", custom_pool_id,
        chimaera::bdev::BdevType::kFile);
    std::this_thread::sleep_for(100ms);

    // Creation should fail for non-existent file path
    bool creation_failed = !creation_success;

    // If creation didn't fail at the Create level, test allocation to see if
    // the container is invalid
    if (!creation_failed) {
      std::vector<chimaera::bdev::Block> blocks =
          BdevChimodFixture::AllocateBlocksAsync(file_client_bad_path,
                                                 chi::PoolQuery::Local(), k4KB);
      creation_failed = (blocks.size() == 0);
    }

    HLOG(kInfo, "File backend properly handles bad path: {}",
         creation_failed ? "YES" : "NO");
  }
}

//==============================================================================
// PARALLEL OPERATIONS TESTS
//==============================================================================

TEST_CASE("bdev_parallel_io_operations", "[bdev][parallel][io]") {
  BdevChimodFixture fixture;

  SECTION("Setup") {
    REQUIRE(g_initialized);
    REQUIRE(
        fixture.createTestFile(100 * 1024 * 1024));  // 100MB for parallel ops
  }

  SECTION("Parallel allocate/write/free operations") {
    // Test configuration
    const size_t num_threads = 4;
    const size_t ops_per_thread = 100;
    const size_t io_size = 4096;  // 4KB I/O size

    // Create BDev container
    chi::PoolId custom_pool_id(200, 0);
    chimaera::bdev::Client client(custom_pool_id);

    bool success = BdevChimodFixture::CreateBdevAsync(
        client, chi::PoolQuery::Dynamic(), fixture.getTestFile(),
        custom_pool_id, chimaera::bdev::BdevType::kFile);
    REQUIRE(success);
    REQUIRE(client.GetReturnCode() == 0);

    HLOG(kInfo, "Starting parallel I/O operations test:");
    HLOG(kInfo, "  Threads: {}", num_threads);
    HLOG(kInfo, "  Operations per thread: {}", ops_per_thread);
    HLOG(kInfo, "  I/O size: {} bytes", io_size);
    HLOG(kInfo, "  Total operations: {}", num_threads * ops_per_thread);

    // Worker thread function using DirectHash for distributed execution
    auto worker_thread = [&](size_t thread_id) {
      // Create thread-local BDev client
      chimaera::bdev::Client thread_client(custom_pool_id);

      // Allocate write buffer in shared memory
      auto write_buffer = CHI_IPC->AllocateBuffer(io_size);
      std::memset(write_buffer.ptr_, static_cast<int>(thread_id), io_size);

      // Perform I/O operations with DirectHash
      for (size_t i = 0; i < ops_per_thread; i++) {
        // Use DirectHash for distributed execution (cycle through hash values)
        auto pool_query = chi::PoolQuery::DirectHash(i % 16);

        // Allocate block
        auto alloc_task =
            thread_client.AsyncAllocateBlocks(pool_query, io_size);
        alloc_task.Wait();
        REQUIRE(alloc_task->return_code_ == 0);
        REQUIRE(alloc_task->blocks_.size() > 0);

        // Convert hipc::vector to std::vector for FreeBlocks
        std::vector<chimaera::bdev::Block> blocks;
        for (size_t i = 0; i < alloc_task->blocks_.size(); ++i) {
          blocks.push_back(alloc_task->blocks_[i]);
        }

        // Write data
        auto write_task = thread_client.AsyncWrite(
            pool_query, WrapBlock(blocks[0]),
            write_buffer.shm_.template Cast<void>().template Cast<void>(),
            io_size);
        write_task.Wait();
        HLOG(kInfo, "Write task: return code = {}, bytes written = {}",
             write_task->return_code_, write_task->bytes_written_);
        REQUIRE(write_task->return_code_ == 0);
        REQUIRE(write_task->bytes_written_ == io_size);

        // Free block
        auto free_task = thread_client.AsyncFreeBlocks(pool_query, blocks);
        free_task.Wait();
        REQUIRE(free_task->return_code_ == 0);
      }

      // Free buffer after all operations complete
      CHI_IPC->FreeBuffer(write_buffer);
    };

    // Launch worker threads
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    auto start_time = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < num_threads; i++) {
      threads.emplace_back(worker_thread, i);
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
      thread.join();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    // Calculate and log statistics
    size_t total_ops = num_threads * ops_per_thread;
    double ops_per_sec = (total_ops * 1000.0) / elapsed.count();
    double bandwidth_mbps =
        (total_ops * io_size * 1000.0) / (elapsed.count() * 1024 * 1024);

    HLOG(kInfo, "Parallel I/O test completed:");
    HLOG(kInfo, "  Total time: {} ms", elapsed.count());
    HLOG(kInfo, "  IOPS: {:.0f} ops/sec", ops_per_sec);
    HLOG(kInfo, "  Bandwidth: {:.2f} MB/s", bandwidth_mbps);
    HLOG(kInfo, "  Avg latency: {:.3f} us/op",
         (elapsed.count() * 1000.0) / total_ops);

    // Verify all operations completed successfully
    REQUIRE(elapsed.count() > 0);
    REQUIRE(ops_per_sec > 0);
  }
}

//==============================================================================
// TASK_FORCE_NET TESTS
//==============================================================================

/**
 * Test TASK_FORCE_NET flag - forces tasks through network code even for local
 * execution
 */
TEST_CASE("bdev_force_net_flag", "[bdev][network][force_net]") {
  HLOG(kInfo, "[bdev_force_net_flag] TEST START");
  BdevChimodFixture fixture;

  SECTION("Setup") {
    HLOG(kInfo, "[bdev_force_net_flag] Setup section: g_initialized={}",
         g_initialized);
    REQUIRE(g_initialized);
    HLOG(kInfo, "[bdev_force_net_flag] Creating test file...");
    REQUIRE(fixture.createTestFile(kDefaultFileSize));
    HLOG(kInfo, "[bdev_force_net_flag] Test file created: {}",
         fixture.getTestFile());
  }

  SECTION("Write and Read with TASK_FORCE_NET") {
    HLOG(kInfo, "[bdev_force_net_flag] Write/Read section starting");
    chi::PoolId custom_pool_id(105, 0);
    HLOG(kInfo,
         "[bdev_force_net_flag] Creating bdev client with pool_id=(major:{}, "
         "minor:{})",
         custom_pool_id.major_, custom_pool_id.minor_);
    chimaera::bdev::Client client(custom_pool_id);

    HLOG(kInfo,
         "[bdev_force_net_flag] Calling AsyncCreate() with Dynamic pool "
         "query...");
    bool success = BdevChimodFixture::CreateBdevAsync(
        client, chi::PoolQuery::Dynamic(), fixture.getTestFile(),
        custom_pool_id, chimaera::bdev::BdevType::kFile);
    HLOG(kInfo,
         "[bdev_force_net_flag] AsyncCreate() returned success={}, "
         "return_code={}",
         success, client.GetReturnCode());
    REQUIRE(success);
    REQUIRE(client.GetReturnCode() == 0);

    HLOG(
        kInfo,
        "[bdev_force_net_flag] Created BDev container for TASK_FORCE_NET test");

    // Get number of containers for logging
    const chi::u32 num_containers = fixture.getNumContainers();
    HLOG(kInfo, "[bdev_force_net_flag] num_containers={}", num_containers);

    // Run TASK_FORCE_NET test using DirectHash for distributed execution
    for (int i = 0; i < 16; ++i) {
      HLOG(kInfo, "[bdev_force_net_flag] === ITERATION {} START ===", i);
      auto pool_query = chi::PoolQuery::DirectHash(i);
      chi::ContainerId expected_container =
          static_cast<chi::ContainerId>(i % num_containers);
      HLOG(kInfo,
           "[bdev_force_net_flag] Iteration {}: DirectHash({}) -> "
           "expected_container={}",
           i, i, expected_container);

      // Allocate a block
      HLOG(kInfo,
           "[bdev_force_net_flag] Iteration {}: Calling "
           "AsyncAllocateBlocks(k64KB)...",
           i);
      auto alloc_task = client.AsyncAllocateBlocks(pool_query, k64KB);
      HLOG(kInfo,
           "[bdev_force_net_flag] Iteration {}: AsyncAllocateBlocks returned, "
           "calling Wait()...",
           i);
      alloc_task.Wait();
      HLOG(kInfo,
           "[bdev_force_net_flag] Iteration {}: AllocateBlocks Wait() "
           "returned, return_code={}, blocks.size()={}",
           i, alloc_task->return_code_, alloc_task->blocks_.size());
      REQUIRE(alloc_task->return_code_ == 0);
      REQUIRE(alloc_task->blocks_.size() > 0);

      chimaera::bdev::Block block = alloc_task->blocks_[0];
      HLOG(kInfo,
           "[bdev_force_net_flag] Iteration {}: Allocated block: offset={}, "
           "size={}, completer={}",
           i, block.offset_, block.size_, alloc_task->GetCompleter());
      HLOG(kInfo, "[bdev_force_net_flag] Iteration {}: Deleted alloc_task", i);

      // Prepare test data
      std::vector<hshm::u8> write_data =
          fixture.generateTestData(k64KB, 0xAB + i);
      HLOG(kInfo,
           "[bdev_force_net_flag] Iteration {}: Generated test_data of size {}",
           i, write_data.size());

      // Create write task with TASK_FORCE_NET flag
      HLOG(kInfo,
           "[bdev_force_net_flag] Iteration {}: Allocating write buffer...", i);
      auto write_buffer = CHI_IPC->AllocateBuffer(write_data.size());
      REQUIRE_FALSE(write_buffer.IsNull());
      memcpy(write_buffer.ptr_, write_data.data(), write_data.size());
      HLOG(kInfo,
           "[bdev_force_net_flag] Iteration {}: Write buffer allocated and "
           "filled",
           i);

      // Use NewTask directly to create write task
      auto* ipc_manager = CHI_IPC;
      HLOG(kInfo,
           "[bdev_force_net_flag] Iteration {}: Creating WriteTask with "
           "NewTask...",
           i);
      auto write_task_ptr = ipc_manager->NewTask<chimaera::bdev::WriteTask>(
          chi::CreateTaskId(), client.pool_id_, pool_query, WrapBlock(block),
          write_buffer.shm_.template Cast<void>().template Cast<void>(),
          write_data.size());
      HLOG(kInfo,
           "[bdev_force_net_flag] Iteration {}: WriteTask created, "
           "task_id=(pid:{}, tid:{}, major:{}), pool_id=({}, {}), method={}",
           i, write_task_ptr->task_id_.pid_, write_task_ptr->task_id_.tid_,
           write_task_ptr->task_id_.major_, write_task_ptr->pool_id_.major_,
           write_task_ptr->pool_id_.minor_, write_task_ptr->method_);

      // Set TASK_FORCE_NET flag BEFORE enqueueing
      write_task_ptr->SetFlags(TASK_FORCE_NET);
      REQUIRE(write_task_ptr->task_flags_.Any(TASK_FORCE_NET));
      HLOG(kInfo,
           "[bdev_force_net_flag] Iteration {}: Write task TASK_FORCE_NET flag "
           "set, has_flag={}",
           i, write_task_ptr->task_flags_.Any(TASK_FORCE_NET));

      // Enqueue the task
      HLOG(kInfo,
           "[bdev_force_net_flag] Iteration {}: Calling ipc_manager->Send() "
           "for write task...",
           i);
      auto write_task = ipc_manager->Send(write_task_ptr);
      HLOG(kInfo,
           "[bdev_force_net_flag] Iteration {}: Send() returned, calling "
           "Wait()...",
           i);

      // Wait for write to complete
      write_task.Wait();
      HLOG(kInfo,
           "[bdev_force_net_flag] Iteration {}: Write Wait() returned, "
           "return_code={}, bytes_written={}, completer={}",
           i, write_task->return_code_, write_task->bytes_written_,
           write_task->GetCompleter());
      REQUIRE(write_task->return_code_ == 0);
      REQUIRE(write_task->bytes_written_ == write_data.size());
      HLOG(kInfo, "[bdev_force_net_flag] Iteration {}: Deleted write_task", i);

      HLOG(kInfo,
           "[bdev_force_net_flag] Iteration {}: Write with TASK_FORCE_NET "
           "completed successfully",
           i);

      // Create read task with TASK_FORCE_NET flag
      HLOG(kInfo,
           "[bdev_force_net_flag] Iteration {}: Allocating read buffer...", i);
      auto read_buffer = CHI_IPC->AllocateBuffer(k64KB);
      REQUIRE_FALSE(read_buffer.IsNull());
      HLOG(kInfo, "[bdev_force_net_flag] Iteration {}: Read buffer allocated",
           i);

      // Use NewTask directly to create read task
      HLOG(kInfo,
           "[bdev_force_net_flag] Iteration {}: Creating ReadTask with "
           "NewTask...",
           i);
      auto read_task_ptr = ipc_manager->NewTask<chimaera::bdev::ReadTask>(
          chi::CreateTaskId(), client.pool_id_, pool_query, WrapBlock(block),
          read_buffer.shm_.template Cast<void>().template Cast<void>(), k64KB);
      HLOG(kInfo,
           "[bdev_force_net_flag] Iteration {}: ReadTask created, "
           "task_id=(pid:{}, tid:{}, major:{}), pool_id=({}, {}), method={}",
           i, read_task_ptr->task_id_.pid_, read_task_ptr->task_id_.tid_,
           read_task_ptr->task_id_.major_, read_task_ptr->pool_id_.major_,
           read_task_ptr->pool_id_.minor_, read_task_ptr->method_);

      // Set TASK_FORCE_NET flag BEFORE enqueueing
      read_task_ptr->SetFlags(TASK_FORCE_NET);
      REQUIRE(read_task_ptr->task_flags_.Any(TASK_FORCE_NET));
      HLOG(kInfo,
           "[bdev_force_net_flag] Iteration {}: Read task TASK_FORCE_NET flag "
           "set, has_flag={}",
           i, read_task_ptr->task_flags_.Any(TASK_FORCE_NET));

      // Enqueue the task
      HLOG(kInfo,
           "[bdev_force_net_flag] Iteration {}: Calling ipc_manager->Send() "
           "for read task...",
           i);
      auto read_task = ipc_manager->Send(read_task_ptr);
      HLOG(kInfo,
           "[bdev_force_net_flag] Iteration {}: Send() returned, calling "
           "Wait()...",
           i);

      // Wait for read to complete
      read_task.Wait();
      HLOG(kInfo,
           "[bdev_force_net_flag] Iteration {}: Read Wait() returned, "
           "return_code={}, bytes_read={}, completer={}",
           i, read_task->return_code_, read_task->bytes_read_,
           read_task->GetCompleter());
      REQUIRE(read_task->return_code_ == 0);
      REQUIRE(read_task->bytes_read_ == write_data.size());

      HLOG(kInfo,
           "[bdev_force_net_flag] Iteration {}: Read with TASK_FORCE_NET "
           "completed successfully",
           i);

      // Verify data integrity
      std::vector<hshm::u8> read_data(read_task->bytes_read_);
      memcpy(read_data.data(), read_buffer.ptr_, read_task->bytes_read_);
      REQUIRE(read_data.size() == write_data.size());

      for (size_t j = 0; j < write_data.size(); ++j) {
        REQUIRE(read_data[j] == write_data[j]);
      }
      HLOG(kInfo,
           "[bdev_force_net_flag] Iteration {}: Data verification passed", i);

      HLOG(kInfo, "[bdev_force_net_flag] Iteration {}: Deleted read_task", i);

      // Free buffers
      HLOG(kInfo, "[bdev_force_net_flag] Iteration {}: Freeing write buffer...",
           i);
      CHI_IPC->FreeBuffer(write_buffer);
      HLOG(kInfo, "[bdev_force_net_flag] Iteration {}: Freeing read buffer...",
           i);
      CHI_IPC->FreeBuffer(read_buffer);
      HLOG(kInfo, "[bdev_force_net_flag] Iteration {}: Buffers freed", i);

      HLOG(kInfo,
           "[bdev_force_net_flag] === ITERATION {} COMPLETE - data verified "
           "successfully ===",
           i);
    }
    HLOG(kInfo, "[bdev_force_net_flag] All 16 iterations passed");
  }
  HLOG(kInfo, "[bdev_force_net_flag] TEST COMPLETE");
}

//==============================================================================
// MAIN TEST RUNNER
//==============================================================================

SIMPLE_TEST_MAIN()