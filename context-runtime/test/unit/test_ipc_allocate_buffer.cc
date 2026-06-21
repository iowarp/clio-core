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
 * Unit tests for CLIO_IPC->AllocateBuffer functionality
 * Tests the IPC manager's buffer allocation for shared memory operations
 */

#include <clio_runtime/clio_runtime.h>

#include <cstring>
#include <memory>
#include <vector>

#include "../simple_test.h"

namespace {
// Test setup helper - same pattern as other tests
bool initialize_clio() {
  bool success = clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
  if (success) SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;
  return success;
}
}  // namespace

TEST_CASE("CLIO_IPC AllocateBuffer basic functionality",
          "[ipc][allocate_buffer][basic]") {
  // Initialize CLIO Runtime client for testing
  REQUIRE(initialize_clio());

  auto* ipc_manager = CLIO_IPC;
  REQUIRE(ipc_manager != nullptr);

  SECTION("Allocate char buffer") {
    size_t buffer_size = 1024;
    ctp::ipc::FullPtr<char> buffer_ptr =
        ipc_manager->AllocateBuffer(buffer_size);

    REQUIRE_FALSE(buffer_ptr.IsNull());
    REQUIRE(buffer_ptr.ptr_ != nullptr);

    // Test basic memory access
    memset(buffer_ptr.ptr_, 0xAA, buffer_size);

    // Verify data was written
    unsigned char* byte_ptr = reinterpret_cast<unsigned char*>(buffer_ptr.ptr_);
    for (size_t i = 0; i < buffer_size; ++i) {
      REQUIRE(byte_ptr[i] == 0xAA);
    }
  }

  SECTION("Allocate typed buffer") {
    size_t buffer_size = 512;
    ctp::ipc::FullPtr<char> char_buffer =
        ipc_manager->AllocateBuffer(buffer_size);

    REQUIRE_FALSE(char_buffer.IsNull());
    REQUIRE(char_buffer.ptr_ != nullptr);

    // Test typed buffer access
    const char* test_string = "CLIO_IPC AllocateBuffer test";
    size_t test_len = strlen(test_string);
    REQUIRE(test_len < buffer_size);

    strncpy(char_buffer.ptr_, test_string, buffer_size - 1);
    char_buffer.ptr_[buffer_size - 1] = '\0';

    REQUIRE(strcmp(char_buffer.ptr_, test_string) == 0);
  }

  SECTION("Allocate integer buffer") {
    size_t num_ints = 100 * sizeof(int);  // Allocate bytes, not int count
    ctp::ipc::FullPtr<char> buffer = ipc_manager->AllocateBuffer(num_ints);

    REQUIRE_FALSE(buffer.IsNull());
    REQUIRE(buffer.ptr_ != nullptr);

    // Cast to int pointer for testing
    int* int_buffer = reinterpret_cast<int*>(buffer.ptr_);

    // Test integer array operations
    for (size_t i = 0; i < 100; ++i) {
      int_buffer[i] = static_cast<int>(i * 2);
    }

    // Verify data
    for (size_t i = 0; i < 100; ++i) {
      REQUIRE(int_buffer[i] == static_cast<int>(i * 2));
    }
  }
}

TEST_CASE("CLIO_IPC AllocateBuffer return type verification",
          "[ipc][allocate_buffer][types]") {
  REQUIRE(initialize_clio());

  auto* ipc_manager = CLIO_IPC;
  REQUIRE(ipc_manager != nullptr);

  SECTION("Return type is FullPtr<char>, not ctp::ipc::ShmPtr<>") {
    // Compile-time type checking - now always returns FullPtr<char>
    auto buffer1 = ipc_manager->AllocateBuffer(1024);
    static_assert(std::is_same_v<decltype(buffer1), ctp::ipc::FullPtr<char>>,
                  "AllocateBuffer should return FullPtr<char>");

    auto buffer2 = ipc_manager->AllocateBuffer(512);
    static_assert(std::is_same_v<decltype(buffer2), ctp::ipc::FullPtr<char>>,
                  "AllocateBuffer should return FullPtr<char>");

    auto buffer3 = ipc_manager->AllocateBuffer(256);
    static_assert(std::is_same_v<decltype(buffer3), ctp::ipc::FullPtr<char>>,
                  "AllocateBuffer should return FullPtr<char>");

    // Runtime verification
    REQUIRE_FALSE(buffer1.IsNull());
    REQUIRE_FALSE(buffer2.IsNull());
    REQUIRE_FALSE(buffer3.IsNull());
  }
}

TEST_CASE("CLIO_IPC AllocateBuffer size variations",
          "[ipc][allocate_buffer][sizes]") {
  REQUIRE(initialize_clio());

  auto* ipc_manager = CLIO_IPC;
  REQUIRE(ipc_manager != nullptr);

  SECTION("Various buffer sizes") {
    std::vector<size_t> test_sizes = {2, 64, 512, 1024, 4096, 16384};

    for (size_t size : test_sizes) {
      ctp::ipc::FullPtr<char> buffer =
          ipc_manager->AllocateBuffer(size);

      REQUIRE_FALSE(buffer.IsNull());
      REQUIRE(buffer.ptr_ != nullptr);

      // Test memory access at boundaries
      buffer.ptr_[0] = 0x01;         // First byte
      buffer.ptr_[size - 1] = static_cast<char>(0xFF);  // Last byte

      REQUIRE(buffer.ptr_[0] == 0x01);
      REQUIRE(static_cast<unsigned char>(buffer.ptr_[size - 1]) == 0xFF);
    }
  }

  SECTION("Zero size allocation") {
    // Test edge case - zero size should either return null or valid empty
    // buffer
    ctp::ipc::FullPtr<char> buffer = ipc_manager->AllocateBuffer(0);

    // Either null (allocation failed) or valid but zero-sized
    // Both behaviors are acceptable for zero-size allocations
    INFO("Zero-size allocation behavior: "
         << (buffer.IsNull() ? "null" : "valid empty"));
  }
}

TEST_CASE("CLIO_IPC AllocateBuffer multiple allocations",
          "[ipc][allocate_buffer][multiple]") {
  REQUIRE(initialize_clio());

  auto* ipc_manager = CLIO_IPC;
  REQUIRE(ipc_manager != nullptr);

  SECTION("Multiple simultaneous allocations") {
    const size_t num_buffers = 10;
    const size_t buffer_size = 1024;
    std::vector<ctp::ipc::FullPtr<char>> buffers;

    // Allocate multiple buffers
    for (size_t i = 0; i < num_buffers; ++i) {
      auto buffer = ipc_manager->AllocateBuffer(buffer_size);
      REQUIRE_FALSE(buffer.IsNull());
      buffers.push_back(buffer);
    }

    // Write unique data to each buffer
    for (size_t i = 0; i < num_buffers; ++i) {
      std::string test_data = "Buffer " + std::to_string(i);
      strncpy(buffers[i].ptr_, test_data.c_str(), buffer_size - 1);
      buffers[i].ptr_[buffer_size - 1] = '\0';
    }

    // Verify data integrity
    for (size_t i = 0; i < num_buffers; ++i) {
      std::string expected = "Buffer " + std::to_string(i);
      REQUIRE(strcmp(buffers[i].ptr_, expected.c_str()) == 0);
    }

    // Verify all buffers have different addresses
    for (size_t i = 0; i < num_buffers; ++i) {
      for (size_t j = i + 1; j < num_buffers; ++j) {
        REQUIRE(buffers[i].ptr_ != buffers[j].ptr_);
      }
    }
  }
}

TEST_CASE("CLIO_IPC AllocateBuffer client vs runtime behavior",
          "[ipc][allocate_buffer][client_runtime]") {
  REQUIRE(initialize_clio());

  auto* ipc_manager = CLIO_IPC;
  REQUIRE(ipc_manager != nullptr);

  auto* runtime_manager = CLIO_RUNTIME_MANAGER;
  REQUIRE(runtime_manager != nullptr);

  SECTION("Colocated client mode allocation") {
    // In colocated mode (client with runtime), both client and runtime are available
    // AllocateBuffer should work correctly regardless of mode
    REQUIRE(runtime_manager->IsRuntime());  // Colocated mode has runtime
    REQUIRE(runtime_manager->IsClient());   // And client

    ctp::ipc::FullPtr<char> buffer = ipc_manager->AllocateBuffer(100);
    REQUIRE_FALSE(buffer.IsNull());
    REQUIRE(buffer.ptr_ != nullptr);

    // Test basic functionality
    buffer.ptr_[0] = 42;
    buffer.ptr_[99] = 84;

    REQUIRE(buffer.ptr_[0] == 42);
    REQUIRE(buffer.ptr_[99] == 84);
  }
}

TEST_CASE("CLIO_IPC AllocateBuffer memory alignment",
          "[ipc][allocate_buffer][alignment]") {
  REQUIRE(initialize_clio());

  auto* ipc_manager = CLIO_IPC;
  REQUIRE(ipc_manager != nullptr);

  SECTION("Pointer alignment for different types") {
    // Test alignment for various types
    auto char_buffer = ipc_manager->AllocateBuffer(1024);
    auto int_buffer = ipc_manager->AllocateBuffer(256);
    auto double_buffer = ipc_manager->AllocateBuffer(128);

    REQUIRE_FALSE(char_buffer.IsNull());
    REQUIRE_FALSE(int_buffer.IsNull());
    REQUIRE_FALSE(double_buffer.IsNull());

    // Check pointer alignment (implementation dependent)
    uintptr_t char_addr = reinterpret_cast<uintptr_t>(char_buffer.ptr_);
    uintptr_t int_addr = reinterpret_cast<uintptr_t>(int_buffer.ptr_);
    uintptr_t double_addr = reinterpret_cast<uintptr_t>(double_buffer.ptr_);

    // int should be aligned to at least 4 bytes
    REQUIRE((int_addr % alignof(int)) == 0);

    // double should be aligned to at least 8 bytes
    REQUIRE((double_addr % alignof(double)) == 0);

    INFO("char buffer alignment: " << (char_addr % alignof(char)));
    INFO("int buffer alignment: " << (int_addr % alignof(int)));
    INFO("double buffer alignment: " << (double_addr % alignof(double)));
  }
}

TEST_CASE("CLIO_IPC AllocateBuffer documentation examples",
          "[ipc][allocate_buffer][documentation]") {
  REQUIRE(initialize_clio());

  SECTION("MODULE_DEVELOPMENT_GUIDE.md examples") {
    // Test the exact examples from the documentation

    // Get the IPC manager singleton
    auto* ipc_manager = CLIO_IPC;
    REQUIRE(ipc_manager != nullptr);

    // Allocate a buffer in shared memory (returns FullPtr<T>, not
    // ctp::ipc::ShmPtr<>)
    size_t buffer_size = 1024;
    ctp::ipc::FullPtr<char> buffer_ptr =
        ipc_manager->AllocateBuffer(buffer_size);

    REQUIRE_FALSE(buffer_ptr.IsNull());
    REQUIRE(buffer_ptr.ptr_ != nullptr);

    // Use the buffer (example: copy data into it)
    const char* source_data = "Documentation example data";
    size_t data_size = strlen(source_data) + 1;
    REQUIRE(data_size <= buffer_size);

    void* buffer_data = buffer_ptr.ptr_;
    memcpy(buffer_data, source_data, data_size);

    // Verify the copy
    REQUIRE(strcmp(static_cast<const char*>(buffer_data), source_data) == 0);

    // Alternative: Allocate typed buffer
    ctp::ipc::FullPtr<char> char_buffer =
        ipc_manager->AllocateBuffer(buffer_size);
    REQUIRE_FALSE(char_buffer.IsNull());

    strncpy(char_buffer.ptr_, "example data", buffer_size - 1);
    char_buffer.ptr_[buffer_size - 1] = '\0';

    REQUIRE(strcmp(char_buffer.ptr_, "example data") == 0);

    // The buffer will be automatically freed when buffer_ptr goes out of scope
    // or when explicitly deallocated by the framework
  }
}

// Main function to run all tests
SIMPLE_TEST_MAIN()