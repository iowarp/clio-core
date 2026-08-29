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
 * GPU unit test for ShmTransport with GpuShmMmap backend
 *
 * This test verifies that data transfer works correctly with GPU-accessible
 * pinned memory using ShmTransferInfo and ShmTransport::Send/Recv:
 * 1. Allocates pinned host memory using GpuShmMmap backend for copy space
 * 2. Uses ShmTransferInfo for SPSC ring buffer metadata
 * 3. GPU kernel sends data via ShmTransport::Send
 * 4. CPU receives data via ShmTransport::Recv
 * 5. CPU verifies the transferred data
 */

#include <type_traits>
#include <catch2/catch_all.hpp>

#include "clio_ctp/lightbeam/shm_transport.h"
#include "clio_ctp/memory/allocator/buddy_allocator.h"
#include "clio_ctp/memory/backend/gpu_shm_mmap.h"
#include "clio_ctp/util/gpu_api.h"

using ctp::ipc::BuddyAllocator;
using ctp::ipc::GpuShmMmap;
using ctp::ipc::MemoryBackendId;
using ctp::lbm::Bulk;
using ctp::lbm::LbmContext;
using ctp::lbm::LbmMeta;
using ctp::lbm::ShmTransferInfo;
using ctp::lbm::ShmTransport;

using GpuAllocT = ctp::ipc::BuddyAllocator;
using GpuMeta = LbmMeta<GpuAllocT>;

/**
 * Diagnostic struct to capture ContainsPtr results from GPU
 */
struct ContainsPtrDiag {
  size_t data_capacity;        // Value of backend_.data_capacity_ as seen on GPU
  size_t test_offset;          // The offset we tested
  bool contains_result;        // Result of alloc->ContainsPtr(OffsetPtr)
  bool fullptr_offset_null;    // True if FullPtr(alloc, OffsetPtr<T>(off)) is null
  bool fullptr_size_null;      // True if FullPtr(alloc, size_t) is null
};

/**
 * Diagnostic kernel: isolate exactly which ContainsPtr code path fails on GPU.
 * This does NOT go through FinalizeAllocation or AllocateSmall.
 */
__global__ void DiagnoseContainsPtrKernel(GpuAllocT *alloc, size_t test_offset,
                                           ContainsPtrDiag *diag) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    // Read data_capacity_ via the public accessor
    diag->data_capacity = alloc->GetBackendDataCapacity();
    diag->test_offset   = test_offset;

    // Test 1: ContainsPtr(OffsetPtrBase) directly
    ctp::ipc::OffsetPtr<void> ptr(test_offset);
    diag->contains_result = alloc->ContainsPtr(ptr);

    // Test 2: FullPtr via OffsetPtr overload (the original broken path)
    ctp::ipc::FullPtr<void> fp_offset(alloc, ctp::ipc::OffsetPtr<void>(test_offset));
    diag->fullptr_offset_null = fp_offset.IsNull();

    // Test 3: FullPtr via size_t overload (the fixed path)
    ctp::ipc::FullPtr<void> fp_size(alloc, test_offset);
    diag->fullptr_size_null = fp_size.IsNull();
  }
}

/**
 * GPU kernel to fill a buffer with a pattern
 */
__global__ void FillBufferKernel(char *buffer, size_t size, char pattern) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = idx; i < size; i += stride) {
    buffer[i] = pattern;
  }
}

/**
 * GPU kernel that sends data via ShmTransport::Send API.
 * Constructs LbmMeta on device and sends via the SPSC ring buffer.
 * Only thread 0 performs the transfer (single-producer).
 */
__global__ void GpuSendSimpleKernel(GpuAllocT *alloc, const char *src_buffer,
                                     size_t data_size, char *copy_space,
                                     ShmTransferInfo *shm_info) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    GpuMeta meta(alloc);
    Bulk bulk;
    bulk.data.ptr_ = const_cast<char *>(src_buffer);
    bulk.data.shm_.alloc_id_ = ctp::ipc::AllocatorId::GetNull();
    bulk.data.shm_.off_ = 0;
    bulk.size = data_size;
    bulk.flags = ctp::bitfield32_t(BULK_XFER);
    meta.send.push_back(bulk);
    meta.send_bulks = 1;
    LbmContext ctx;
    ctx.copy_space = copy_space;
    ctx.shm_info_ = shm_info;
    ShmTransport::Send(meta, ctx);
  }
}

/**
 * GPU kernel to set a value at a specific location (for simple tests)
 */
__global__ void SetValueKernel(char *buffer, size_t index, char value) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    buffer[index] = value;
  }
}

/**
 * Test GPU to CPU data transfer using ShmTransferInfo SPSC ring buffer
 */
TEST_CASE("ShmTransfer GPU", "[gpu][transfer]") {
  constexpr size_t kBackendSize = 16 * 1024 * 1024;  // 16MB
  constexpr size_t kCopySpaceSize = 16 * 1024;       // 16KB transfer granularity
  constexpr size_t kDataSize = 64 * 1024;            // 64KB buffer
  constexpr int kGpuId = 0;
  const std::string kUrl = "/test_local_transfer_gpu";

  SECTION("BasicGpuToCpuTransfer") {
    // Step 1: Create a GpuShmMmap backend for pinned host memory
    GpuShmMmap backend;
    MemoryBackendId backend_id(0, 0);
    bool init_success =
        backend.shm_init(backend_id, kBackendSize, kUrl, kGpuId);
    REQUIRE(init_success);

    // Step 2: Create an BuddyAllocator on that backend
    using AllocT = ctp::ipc::BuddyAllocator;
    AllocT *alloc_ptr = backend.MakeAlloc<AllocT>();
    REQUIRE(alloc_ptr != nullptr);

    // Step 3: Allocate copy space and ShmTransferInfo from pinned memory
    auto copy_space_ptr = alloc_ptr->AllocateObjs<char>(kCopySpaceSize);
    char *copy_space = copy_space_ptr.ptr_;
    REQUIRE(copy_space != nullptr);

    auto shm_info_ptr = alloc_ptr->AllocateObjs<ShmTransferInfo>(1);
    ShmTransferInfo *shm_info = shm_info_ptr.ptr_;
    REQUIRE(shm_info != nullptr);
    new (shm_info) ShmTransferInfo();
    shm_info->copy_space_size_ = kCopySpaceSize;

    // Step 4: Allocate GPU source buffer (pinned so both GPU and CPU can access)
    char *gpu_buffer;
    gpu_buffer = ctp::GpuApi::MallocHost<std::remove_pointer_t<decltype(gpu_buffer)>>(kDataSize);
    REQUIRE(gpu_buffer != nullptr);

    // Step 5: Fill the buffer with pattern (value = 1) using GPU kernel
    constexpr char kPattern = 1;
    int blockSize = 256;
    int numBlocks = (kDataSize + blockSize - 1) / blockSize;
    FillBufferKernel<<<numBlocks, blockSize>>>(gpu_buffer, kDataSize, kPattern);
    ctp::GpuApi::Synchronize();
    REQUIRE(ctp::GpuApi::LastError() == nullptr);

    // Step 6: GPU sends data via ShmTransport::Send (SPSC ring buffer)
    GpuSendSimpleKernel<<<1, 1>>>(alloc_ptr, gpu_buffer, kDataSize, copy_space,
                                   shm_info);

    // Step 7: CPU receives data via ShmTransport::Recv
    LbmMeta<> recv_meta;
    LbmContext ctx;
    ctx.copy_space = copy_space;
    ctx.shm_info_ = shm_info;
    ShmTransport::Recv(recv_meta, ctx);

    // Wait for GPU kernel to complete
    ctp::GpuApi::Synchronize();
    REQUIRE(ctp::GpuApi::LastError() == nullptr);

    // Step 8: Verify all data was transferred
    REQUIRE(recv_meta.recv.size() == 1);
    REQUIRE(recv_meta.recv[0].size == kDataSize);
    REQUIRE(recv_meta.recv[0].data.ptr_ != nullptr);

    bool all_ones = true;
    for (size_t i = 0; i < kDataSize; ++i) {
      if (recv_meta.recv[0].data.ptr_[i] != kPattern) {
        all_ones = false;
        break;
      }
    }
    REQUIRE(all_ones);

    // Cleanup
    std::free(recv_meta.recv[0].data.ptr_);
    ctp::GpuApi::FreeHost(gpu_buffer);
  }

  SECTION("ChunkedTransferWithPattern") {
    // Test with a more complex pattern to verify data integrity
    GpuShmMmap backend;
    MemoryBackendId backend_id(0, 1);
    bool init_success =
        backend.shm_init(backend_id, kBackendSize, kUrl + "_pattern", kGpuId);
    REQUIRE(init_success);

    using AllocT = ctp::ipc::BuddyAllocator;
    AllocT *alloc_ptr = backend.MakeAlloc<AllocT>();
    REQUIRE(alloc_ptr != nullptr);

    auto copy_space_ptr = alloc_ptr->AllocateObjs<char>(kCopySpaceSize);
    char *copy_space = copy_space_ptr.ptr_;
    REQUIRE(copy_space != nullptr);

    auto shm_info_ptr = alloc_ptr->AllocateObjs<ShmTransferInfo>(1);
    ShmTransferInfo *shm_info = shm_info_ptr.ptr_;
    REQUIRE(shm_info != nullptr);
    new (shm_info) ShmTransferInfo();
    shm_info->copy_space_size_ = kCopySpaceSize;

    // Allocate and initialize GPU buffer with pattern
    char *gpu_buffer;
    gpu_buffer = ctp::GpuApi::MallocHost<std::remove_pointer_t<decltype(gpu_buffer)>>(kDataSize);
    REQUIRE(gpu_buffer != nullptr);

    // Initialize with pattern on CPU (index % 256)
    for (size_t i = 0; i < kDataSize; ++i) {
      gpu_buffer[i] = static_cast<char>(i % 256);
    }

    // GPU sends via SPSC ring buffer
    GpuSendSimpleKernel<<<1, 1>>>(alloc_ptr, gpu_buffer, kDataSize, copy_space,
                                   shm_info);

    // CPU receives via SPSC ring buffer
    LbmMeta<> recv_meta;
    LbmContext ctx;
    ctx.copy_space = copy_space;
    ctx.shm_info_ = shm_info;
    ShmTransport::Recv(recv_meta, ctx);

    ctp::GpuApi::Synchronize();
    REQUIRE(ctp::GpuApi::LastError() == nullptr);

    // Verify data integrity
    REQUIRE(recv_meta.recv.size() == 1);
    REQUIRE(recv_meta.recv[0].size == kDataSize);
    REQUIRE(recv_meta.recv[0].data.ptr_ != nullptr);
    bool pattern_correct = true;
    for (size_t i = 0; i < kDataSize; ++i) {
      if (recv_meta.recv[0].data.ptr_[i] != static_cast<char>(i % 256)) {
        pattern_correct = false;
        break;
      }
    }
    REQUIRE(pattern_correct);

    std::free(recv_meta.recv[0].data.ptr_);
    ctp::GpuApi::FreeHost(gpu_buffer);
  }

  SECTION("DirectGpuMemoryAccess") {
    // Test that GPU can directly read/write to the GpuShmMmap memory
    GpuShmMmap backend;
    MemoryBackendId backend_id(0, 2);
    bool init_success =
        backend.shm_init(backend_id, kBackendSize, kUrl + "_direct", kGpuId);
    REQUIRE(init_success);

    using AllocT = ctp::ipc::BuddyAllocator;
    AllocT *alloc_ptr = backend.MakeAlloc<AllocT>();
    REQUIRE(alloc_ptr != nullptr);

    // Allocate buffer directly from GpuShmMmap
    auto buffer_ptr = alloc_ptr->AllocateObjs<char>(1024);
    char *buffer = buffer_ptr.ptr_;
    REQUIRE(buffer != nullptr);

    // Initialize on CPU
    std::memset(buffer, 0, 1024);

    // GPU sets specific values
    SetValueKernel<<<1, 1>>>(buffer, 0, 'A');
    SetValueKernel<<<1, 1>>>(buffer, 100, 'B');
    SetValueKernel<<<1, 1>>>(buffer, 500, 'C');
    SetValueKernel<<<1, 1>>>(buffer, 1023, 'D');

    ctp::GpuApi::Synchronize();
    REQUIRE(ctp::GpuApi::LastError() == nullptr);

    // CPU reads and verifies
    REQUIRE(buffer[0] == 'A');
    REQUIRE(buffer[100] == 'B');
    REQUIRE(buffer[500] == 'C');
    REQUIRE(buffer[1023] == 'D');

    // Verify untouched locations are still 0
    REQUIRE(buffer[1] == 0);
    REQUIRE(buffer[50] == 0);
    REQUIRE(buffer[1022] == 0);
  }

  SECTION("LargeTransferPerformance") {
    // Test larger transfer (256KB) to verify performance
    constexpr size_t kLargeDataSize = 256 * 1024;  // 256KB

    GpuShmMmap backend;
    MemoryBackendId backend_id(0, 3);
    bool init_success =
        backend.shm_init(backend_id, kBackendSize, kUrl + "_large", kGpuId);
    REQUIRE(init_success);

    using AllocT = ctp::ipc::BuddyAllocator;
    AllocT *alloc_ptr = backend.MakeAlloc<AllocT>();
    REQUIRE(alloc_ptr != nullptr);

    auto copy_space_ptr = alloc_ptr->AllocateObjs<char>(kCopySpaceSize);
    char *copy_space = copy_space_ptr.ptr_;
    REQUIRE(copy_space != nullptr);

    auto shm_info_ptr = alloc_ptr->AllocateObjs<ShmTransferInfo>(1);
    ShmTransferInfo *shm_info = shm_info_ptr.ptr_;
    REQUIRE(shm_info != nullptr);
    new (shm_info) ShmTransferInfo();
    shm_info->copy_space_size_ = kCopySpaceSize;

    // Allocate GPU buffer
    char *gpu_buffer;
    gpu_buffer = ctp::GpuApi::MallocHost<std::remove_pointer_t<decltype(gpu_buffer)>>(kLargeDataSize);
    REQUIRE(gpu_buffer != nullptr);

    // Fill with pattern
    constexpr char kPattern = 0x55;
    int blockSize = 256;
    int numBlocks = (kLargeDataSize + blockSize - 1) / blockSize;
    FillBufferKernel<<<numBlocks, blockSize>>>(gpu_buffer, kLargeDataSize,
                                               kPattern);
    ctp::GpuApi::Synchronize();
    REQUIRE(ctp::GpuApi::LastError() == nullptr);

    // GPU sends via SPSC ring buffer
    GpuSendSimpleKernel<<<1, 1>>>(alloc_ptr, gpu_buffer, kLargeDataSize,
                                   copy_space, shm_info);

    // CPU receives via SPSC ring buffer
    LbmMeta<> recv_meta;
    LbmContext ctx;
    ctx.copy_space = copy_space;
    ctx.shm_info_ = shm_info;
    ShmTransport::Recv(recv_meta, ctx);

    ctp::GpuApi::Synchronize();
    REQUIRE(ctp::GpuApi::LastError() == nullptr);

    // Verify
    REQUIRE(recv_meta.recv.size() == 1);
    REQUIRE(recv_meta.recv[0].size == kLargeDataSize);
    REQUIRE(recv_meta.recv[0].data.ptr_ != nullptr);

    bool pattern_correct = true;
    for (size_t i = 0; i < kLargeDataSize; ++i) {
      if (recv_meta.recv[0].data.ptr_[i] != kPattern) {
        pattern_correct = false;
        break;
      }
    }
    REQUIRE(pattern_correct);

    std::free(recv_meta.recv[0].data.ptr_);
    ctp::GpuApi::FreeHost(gpu_buffer);
  }
}

/**
 * GPU kernel that uses the full ShmTransport::Send API.
 * Constructs LbmMeta on device, attaches bulk data, and sends via the
 * SPSC ring buffer with metadata serialization.
 *
 * @param alloc BuddyAllocator in pinned memory (GPU-accessible)
 * @param data_buf Data buffer in pinned memory to send as bulk
 * @param data_size Size of the data buffer
 * @param copy_space Ring buffer copy space in pinned memory
 * @param shm_info SPSC ring buffer metadata in pinned memory
 * @param send_result Output: 0 on success
 */
__global__ void GpuSendKernel(GpuAllocT *alloc, char *data_buf,
                               size_t data_size, char *copy_space,
                               ShmTransferInfo *shm_info,
                               int *send_result) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    // Build LbmMeta on device with arena allocator
    GpuMeta meta(alloc);

    // Create bulk descriptor for the data buffer (private memory)
    Bulk bulk;
    bulk.data.ptr_ = data_buf;
    bulk.data.shm_.alloc_id_ = ctp::ipc::AllocatorId::GetNull();
    bulk.data.shm_.off_ = 0;
    bulk.size = data_size;
    bulk.flags = ctp::bitfield32_t(BULK_XFER);
    meta.send.push_back(bulk);
    meta.send_bulks = 1;

    // Build context
    LbmContext ctx;
    ctx.copy_space = copy_space;
    ctx.shm_info_ = shm_info;

    // Send metadata + bulk data via SPSC ring buffer
    *send_result = ShmTransport::Send(meta, ctx);
  }
}

/**
 * GPU kernel that uses the full ShmTransport::Recv API.
 * Receives metadata + bulk data through the SPSC ring buffer and copies
 * the first bulk's data into output_buf.
 *
 * @param alloc BuddyAllocator in pinned memory (GPU-accessible)
 * @param output_buf Buffer to copy received data into
 * @param max_size Maximum size of output_buf
 * @param copy_space Ring buffer copy space in pinned memory
 * @param shm_info SPSC ring buffer metadata in pinned memory
 * @param recv_result Output: 0 on success
 * @param recv_size Output: size of received data
 */
__global__ void GpuRecvKernel(GpuAllocT *alloc, char *output_buf,
                               size_t max_size, char *copy_space,
                               ShmTransferInfo *shm_info,
                               int *recv_result, size_t *recv_size) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    GpuMeta meta(alloc);
    LbmContext ctx;
    ctx.copy_space = copy_space;
    ctx.shm_info_ = shm_info;
    auto info = ShmTransport::Recv(meta, ctx);
    *recv_result = info.rc;
    if (info.rc == 0 && meta.recv.size() > 0) {
      *recv_size = meta.recv[0].size;
      size_t copy_size = meta.recv[0].size;
      if (copy_size > max_size) copy_size = max_size;
      for (size_t k = 0; k < copy_size; ++k) {
        output_buf[k] = meta.recv[0].data.ptr_[k];
      }
    }
  }
}

/**
 * Test GPU Send/Recv using the full ShmTransport::Send/Recv API.
 * GPU sends metadata + bulk data, CPU receives and verifies.
 */
TEST_CASE("ShmTransport Send/Recv GPU", "[gpu][transport]") {
  constexpr size_t kBackendSize = 16 * 1024 * 1024;  // 16MB
  constexpr size_t kCopySpaceSize = 16 * 1024;       // 16KB ring buffer
  constexpr int kGpuId = 0;
  const std::string kUrl = "/test_shm_send_recv_gpu";

  SECTION("GpuSendCpuRecv") {
    constexpr size_t kDataSize = 4 * 1024;  // 4KB bulk data

    // Step 1: Create GpuShmMmap backend for pinned host memory
    GpuShmMmap backend;
    MemoryBackendId backend_id(0, 10);
    bool init_success =
        backend.shm_init(backend_id, kBackendSize, kUrl, kGpuId);
    REQUIRE(init_success);

    // Step 2: Create BuddyAllocator on that backend (GPU-accessible)
    using AllocT = ctp::ipc::BuddyAllocator;
    AllocT *alloc_ptr = backend.MakeAlloc<AllocT>();
    REQUIRE(alloc_ptr != nullptr);

    // Step 3: Allocate copy space and ShmTransferInfo from pinned memory
    auto copy_space_ptr = alloc_ptr->AllocateObjs<char>(kCopySpaceSize);
    char *copy_space = copy_space_ptr.ptr_;
    REQUIRE(copy_space != nullptr);

    auto shm_info_ptr = alloc_ptr->AllocateObjs<ShmTransferInfo>(1);
    ShmTransferInfo *shm_info = shm_info_ptr.ptr_;
    REQUIRE(shm_info != nullptr);
    new (shm_info) ShmTransferInfo();
    shm_info->copy_space_size_ = kCopySpaceSize;

    // Step 4: Allocate send result in pinned memory (GPU writes to it)
    auto result_ptr = alloc_ptr->AllocateObjs<int>(1);
    int *send_result = result_ptr.ptr_;
    REQUIRE(send_result != nullptr);
    *send_result = -1;

    // Step 5: Allocate data buffer in pinned memory and fill with pattern
    char *data_buf;
    data_buf = ctp::GpuApi::MallocHost<std::remove_pointer_t<decltype(data_buf)>>(kDataSize);
    REQUIRE(data_buf != nullptr);
    for (size_t i = 0; i < kDataSize; ++i) {
      data_buf[i] = static_cast<char>(i % 251);  // Prime modulus for pattern
    }

    // Step 6: GPU sends metadata + bulk data via SendMsg
    GpuSendKernel<<<1, 1>>>(alloc_ptr, data_buf, kDataSize,
                                copy_space, shm_info, send_result);

    // Step 7: CPU receives via Recv (uses MallocAllocator)
    LbmMeta<> recv_meta;
    LbmContext ctx;
    ctx.copy_space = copy_space;
    ctx.shm_info_ = shm_info;
    auto recv_info = ShmTransport::Recv(recv_meta, ctx);
    int recv_rc = recv_info.rc;

    // Wait for GPU
    ctp::GpuApi::Synchronize();
    REQUIRE(ctp::GpuApi::LastError() == nullptr);

    // Step 8: Verify results
    REQUIRE(*send_result == 0);
    REQUIRE(recv_rc == 0);
    REQUIRE(recv_meta.send.size() == 1);
    REQUIRE(recv_meta.send_bulks == 1);
    REQUIRE(recv_meta.recv.size() == 1);
    REQUIRE(recv_meta.recv[0].size == kDataSize);
    REQUIRE(recv_meta.recv[0].data.ptr_ != nullptr);

    // Verify received bulk data matches the pattern
    bool data_correct = true;
    for (size_t i = 0; i < kDataSize; ++i) {
      if (recv_meta.recv[0].data.ptr_[i] != static_cast<char>(i % 251)) {
        data_correct = false;
        break;
      }
    }
    REQUIRE(data_correct);

    // Cleanup
    std::free(recv_meta.recv[0].data.ptr_);
    ctp::GpuApi::FreeHost(data_buf);
  }

  SECTION("GpuSendCpuRecvLargeData") {
    constexpr size_t kLargeDataSize = 128 * 1024;  // 128KB (larger than ring)

    GpuShmMmap backend;
    MemoryBackendId backend_id(0, 11);
    bool init_success =
        backend.shm_init(backend_id, kBackendSize, kUrl + "_large", kGpuId);
    REQUIRE(init_success);

    using AllocT = ctp::ipc::BuddyAllocator;
    AllocT *alloc_ptr = backend.MakeAlloc<AllocT>();
    REQUIRE(alloc_ptr != nullptr);

    auto copy_space_ptr = alloc_ptr->AllocateObjs<char>(kCopySpaceSize);
    char *copy_space = copy_space_ptr.ptr_;
    REQUIRE(copy_space != nullptr);

    auto shm_info_ptr = alloc_ptr->AllocateObjs<ShmTransferInfo>(1);
    ShmTransferInfo *shm_info = shm_info_ptr.ptr_;
    REQUIRE(shm_info != nullptr);
    new (shm_info) ShmTransferInfo();
    shm_info->copy_space_size_ = kCopySpaceSize;

    auto result_ptr = alloc_ptr->AllocateObjs<int>(1);
    int *send_result = result_ptr.ptr_;
    REQUIRE(send_result != nullptr);
    *send_result = -1;

    // Allocate and fill large data buffer
    char *data_buf;
    data_buf = ctp::GpuApi::MallocHost<std::remove_pointer_t<decltype(data_buf)>>(kLargeDataSize);
    REQUIRE(data_buf != nullptr);
    constexpr char kPattern = 0xAB;
    for (size_t i = 0; i < kLargeDataSize; ++i) {
      data_buf[i] = kPattern;
    }

    // GPU sends via full Send API
    GpuSendKernel<<<1, 1>>>(alloc_ptr, data_buf, kLargeDataSize,
                                copy_space, shm_info, send_result);

    // CPU receives
    LbmMeta<> recv_meta;
    LbmContext ctx;
    ctx.copy_space = copy_space;
    ctx.shm_info_ = shm_info;
    auto recv_info = ShmTransport::Recv(recv_meta, ctx);
    int recv_rc = recv_info.rc;

    ctp::GpuApi::Synchronize();
    REQUIRE(ctp::GpuApi::LastError() == nullptr);

    REQUIRE(*send_result == 0);
    REQUIRE(recv_rc == 0);
    REQUIRE(recv_meta.recv.size() == 1);
    REQUIRE(recv_meta.recv[0].size == kLargeDataSize);
    REQUIRE(recv_meta.recv[0].data.ptr_ != nullptr);

    // Verify all data is correct
    bool data_correct = true;
    for (size_t i = 0; i < kLargeDataSize; ++i) {
      if (recv_meta.recv[0].data.ptr_[i] != kPattern) {
        data_correct = false;
        break;
      }
    }
    REQUIRE(data_correct);

    std::free(recv_meta.recv[0].data.ptr_);
    ctp::GpuApi::FreeHost(data_buf);
  }

  SECTION("GpuSendGpuRecv") {
    constexpr size_t kDataSize = 4 * 1024;  // 4KB bulk data

    // Step 1: Create separate backends for send and recv (separate allocators
    // avoid race conditions since both kernels run concurrently)
    GpuShmMmap send_backend, recv_backend, shared_backend;
    MemoryBackendId send_bid(0, 12), recv_bid(0, 13), shared_bid(0, 14);
    REQUIRE(send_backend.shm_init(send_bid, kBackendSize,
                                   kUrl + "_gpu2gpu_send", kGpuId));
    REQUIRE(recv_backend.shm_init(recv_bid, kBackendSize,
                                   kUrl + "_gpu2gpu_recv", kGpuId));
    REQUIRE(shared_backend.shm_init(shared_bid, kBackendSize,
                                     kUrl + "_gpu2gpu_shared", kGpuId));

    // Step 2: Create allocators
    using AllocT = ctp::ipc::BuddyAllocator;
    AllocT *send_alloc = send_backend.MakeAlloc<AllocT>();
    AllocT *recv_alloc = recv_backend.MakeAlloc<AllocT>();
    AllocT *shared_alloc = shared_backend.MakeAlloc<AllocT>();
    REQUIRE(send_alloc != nullptr);
    REQUIRE(recv_alloc != nullptr);
    REQUIRE(shared_alloc != nullptr);

    // Step 3: Allocate shared copy space and ShmTransferInfo
    auto copy_space_ptr = shared_alloc->AllocateObjs<char>(kCopySpaceSize);
    char *copy_space = copy_space_ptr.ptr_;
    REQUIRE(copy_space != nullptr);

    auto shm_info_ptr = shared_alloc->AllocateObjs<ShmTransferInfo>(1);
    ShmTransferInfo *shm_info = shm_info_ptr.ptr_;
    REQUIRE(shm_info != nullptr);
    new (shm_info) ShmTransferInfo();
    shm_info->copy_space_size_ = kCopySpaceSize;

    // Step 4: Allocate results in shared pinned memory
    auto send_result_ptr = shared_alloc->AllocateObjs<int>(1);
    int *send_result = send_result_ptr.ptr_;
    REQUIRE(send_result != nullptr);
    *send_result = -1;

    auto recv_result_ptr = shared_alloc->AllocateObjs<int>(1);
    int *recv_result = recv_result_ptr.ptr_;
    REQUIRE(recv_result != nullptr);
    *recv_result = -1;

    auto recv_size_ptr = shared_alloc->AllocateObjs<size_t>(1);
    size_t *recv_size = recv_size_ptr.ptr_;
    REQUIRE(recv_size != nullptr);
    *recv_size = 0;

    // Step 5: Allocate data buffer and output buffer in pinned memory
    char *data_buf;
    data_buf = ctp::GpuApi::MallocHost<std::remove_pointer_t<decltype(data_buf)>>(kDataSize);
    REQUIRE(data_buf != nullptr);
    for (size_t i = 0; i < kDataSize; ++i) {
      data_buf[i] = static_cast<char>(i % 251);
    }

    char *output_buf;
    output_buf = ctp::GpuApi::MallocHost<std::remove_pointer_t<decltype(output_buf)>>(kDataSize);
    REQUIRE(output_buf != nullptr);
    std::memset(output_buf, 0, kDataSize);

    // Step 6: Create two CUDA streams for concurrent send/recv
    void *send_stream = ctp::GpuApi::CreateStream();
    void *recv_stream = ctp::GpuApi::CreateStream();

    // Step 7: Launch recv kernel first (will spinwait for data)
    // Uses recv_alloc for its internal allocations
    GpuRecvKernel<<<1, 1, 0, recv_stream>>>(recv_alloc, output_buf, kDataSize,
                                             copy_space, shm_info,
                                             recv_result, recv_size);

    // Step 8: Launch send kernel (produces data)
    // Uses send_alloc for its internal allocations
    GpuSendKernel<<<1, 1, 0, send_stream>>>(send_alloc, data_buf, kDataSize,
                                             copy_space, shm_info,
                                             send_result);

    // Step 9: Synchronize both streams
    ctp::GpuApi::Synchronize(send_stream);
    REQUIRE(ctp::GpuApi::LastError() == nullptr);
    ctp::GpuApi::Synchronize(recv_stream);
    REQUIRE(ctp::GpuApi::LastError() == nullptr);

    // Step 10: Verify results
    REQUIRE(*send_result == 0);
    REQUIRE(*recv_result == 0);
    REQUIRE(*recv_size == kDataSize);

    bool data_correct = true;
    for (size_t i = 0; i < kDataSize; ++i) {
      if (output_buf[i] != static_cast<char>(i % 251)) {
        data_correct = false;
        break;
      }
    }
    REQUIRE(data_correct);

    // Cleanup
    ctp::GpuApi::DestroyStream(send_stream);
    ctp::GpuApi::DestroyStream(recv_stream);
    ctp::GpuApi::FreeHost(data_buf);
    ctp::GpuApi::FreeHost(output_buf);
  }
}

/**
 * Diagnostic test: why did ContainsPtr fail on GPU?
 *
 * This test isolates ContainsPtr behavior by calling it directly from a kernel,
 * NOT through FinalizeAllocation or any allocator path.  It compares:
 *   1. data_capacity_ as read on GPU
 *   2. ContainsPtr(OffsetPtr<void>) result
 *   3. FullPtr via OffsetPtr overload (original broken path)
 *   4. FullPtr via size_t overload (fixed path)
 *
 * Expected: all four should be correct.  If ContainsPtr is broken, test 2
 * fails; if the FullPtr-OffsetPtr path is broken, test 3 fails.
 */
TEST_CASE("ContainsPtr GPU Diagnostic", "[gpu][contains_ptr]") {
  constexpr size_t kBackendSize = 16 * 1024 * 1024;
  constexpr int kGpuId = 0;

  GpuShmMmap backend;
  MemoryBackendId backend_id(9, 9);
  REQUIRE(backend.shm_init(backend_id, kBackendSize, "/diag_contains_ptr", kGpuId));

  GpuAllocT *alloc_ptr = backend.MakeAlloc<GpuAllocT>();
  REQUIRE(alloc_ptr != nullptr);

  // Allocate the diag struct from managed memory so GPU can write to it
  ContainsPtrDiag *diag = nullptr;
  diag = ctp::GpuApi::MallocManaged<std::remove_pointer_t<decltype(diag)>>(
      sizeof(ContainsPtrDiag));
  REQUIRE(diag != nullptr);
  memset(diag, 0, sizeof(ContainsPtrDiag));

  // Use a known-valid offset: sizeof(_BuddyAllocator) is the first allocation
  // offset (allocator object itself starts at 0, data starts after it).
  // Allocate something so we have a known valid offset to test with.
  auto dummy_ptr = alloc_ptr->AllocateObjs<char>(64);
  REQUIRE(dummy_ptr.ptr_ != nullptr);
  size_t valid_offset = dummy_ptr.shm_.off_.load();
  INFO("valid_offset from CPU allocation = " << valid_offset);
  INFO("data_capacity from CPU = " << alloc_ptr->GetBackendDataCapacity());

  DiagnoseContainsPtrKernel<<<1, 1>>>(alloc_ptr, valid_offset, diag);
  REQUIRE((ctp::GpuApi::Synchronize(), true));

  INFO("GPU data_capacity = " << diag->data_capacity);
  INFO("GPU test_offset   = " << diag->test_offset);
  INFO("GPU contains_result       = " << diag->contains_result);
  INFO("GPU fullptr_offset_null   = " << diag->fullptr_offset_null);
  INFO("GPU fullptr_size_null     = " << diag->fullptr_size_null);

  // data_capacity_ must be readable and non-zero from GPU
  REQUIRE(diag->data_capacity > 0);
  REQUIRE(diag->data_capacity == alloc_ptr->GetBackendDataCapacity());

  // ContainsPtr should return true for a valid offset
  REQUIRE(diag->contains_result == true);

  // Both FullPtr overloads should produce non-null results
  REQUIRE(diag->fullptr_offset_null == false);
  REQUIRE(diag->fullptr_size_null   == false);

  ctp::GpuApi::Free(diag);
}
