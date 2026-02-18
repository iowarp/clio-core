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

#include "wrp_cte/uvm/gpu_vmm.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace wrp_cte::uvm;

/** Kernel to read values from GPU memory into an output buffer */
__global__ void readKernel(const int *src, int *dst, size_t count) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    dst[idx] = src[idx];
  }
}

/**
 * Kernel that accesses scattered addresses across the full VA space.
 * Each thread reads one int from a different page's base address and
 * writes it plus a pass/fail flag to the results buffer.
 *
 * page_ptrs[i] = base device pointer to page i (already touched/backed)
 * results[i]   = value read from the first int of page i
 */
__global__ void scatteredReadKernel(const CUdeviceptr *page_ptrs,
                                     int *results, size_t num_pages) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_pages) {
    const int *ptr = (const int *)page_ptrs[idx];
    results[idx] = ptr[0];  // Read first int from the page
  }
}

/** Verify that a GPU page contains the expected fill value */
static bool verifyPage(CUdeviceptr page_ptr, size_t page_size,
                        int expected_value) {
  size_t num_ints = page_size / sizeof(int);

  // Allocate device buffer for readback
  int *d_out = nullptr;
  cudaMalloc(&d_out, page_size);

  int threads = 256;
  int blocks = (int)((num_ints + threads - 1) / threads);
  readKernel<<<blocks, threads>>>((const int *)page_ptr, d_out, num_ints);
  cudaDeviceSynchronize();

  // Copy to host and check
  std::vector<int> host_buf(num_ints);
  cudaMemcpy(host_buf.data(), d_out, page_size, cudaMemcpyDeviceToHost);
  cudaFree(d_out);

  for (size_t i = 0; i < num_ints; ++i) {
    if (host_buf[i] != expected_value) {
      fprintf(stderr, "  MISMATCH at int[%zu]: got %d, expected %d\n", i,
              host_buf[i], expected_value);
      return false;
    }
  }
  return true;
}

static void testBasicDemandPaging() {
  printf("=== Test: Basic Demand Paging ===\n");

  GpuVirtualMemoryManager vmm;
  GpuVmmConfig cfg;
  cfg.va_size_bytes = 64ULL * 1024 * 1024;  // 64 MB for testing
  cfg.fill_value = 5;

  CUresult res = vmm.init(cfg);
  assert(res == CUDA_SUCCESS);

  printf("  Total pages: %zu\n", vmm.getTotalPages());
  printf("  Mapped pages before touch: %zu\n", vmm.getMappedPageCount());

  // Touch page 0
  res = vmm.touchPage(0);
  assert(res == CUDA_SUCCESS);
  assert(vmm.isMapped(0));
  assert(vmm.getMappedPageCount() == 1);
  printf("  After touching page 0: mapped=%zu\n", vmm.getMappedPageCount());

  // Verify page 0 contains fill_value=5
  bool ok = verifyPage(vmm.getPagePtr(0), vmm.getPageSize(), 5);
  assert(ok);
  printf("  Page 0 verification: PASSED\n");

  // Touch page 0 again (should be no-op)
  res = vmm.touchPage(0);
  assert(res == CUDA_SUCCESS);
  assert(vmm.getMappedPageCount() == 1);
  printf("  Re-touch page 0 (no-op): PASSED\n");

  // Touch page 5
  res = vmm.touchPage(5);
  assert(res == CUDA_SUCCESS);
  assert(vmm.isMapped(5));
  assert(vmm.getMappedPageCount() == 2);

  ok = verifyPage(vmm.getPagePtr(5), vmm.getPageSize(), 5);
  assert(ok);
  printf("  Page 5 verification: PASSED\n");

  // Page 1 should still be unmapped
  assert(!vmm.isMapped(1));
  printf("  Page 1 unmapped (no touch): PASSED\n");

  vmm.destroy();
  printf("  Cleanup: PASSED\n\n");
}

static void testEviction() {
  printf("=== Test: Page Eviction ===\n");

  GpuVirtualMemoryManager vmm;
  GpuVmmConfig cfg;
  cfg.va_size_bytes = 64ULL * 1024 * 1024;
  cfg.fill_value = 5;

  CUresult res = vmm.init(cfg);
  assert(res == CUDA_SUCCESS);

  // Map pages 0, 1, 2
  for (size_t i = 0; i < 3; ++i) {
    res = vmm.touchPage(i);
    assert(res == CUDA_SUCCESS);
  }
  assert(vmm.getMappedPageCount() == 3);
  printf("  Mapped 3 pages: PASSED\n");

  // Evict page 1
  res = vmm.evictPage(1);
  assert(res == CUDA_SUCCESS);
  assert(!vmm.isMapped(1));
  assert(vmm.getMappedPageCount() == 2);
  printf("  Evicted page 1: PASSED\n");

  // Re-touch page 1 (should re-allocate and fill)
  res = vmm.touchPage(1);
  assert(res == CUDA_SUCCESS);
  assert(vmm.isMapped(1));

  bool ok = verifyPage(vmm.getPagePtr(1), vmm.getPageSize(), 5);
  assert(ok);
  printf("  Re-touch page 1 after eviction: PASSED\n");

  vmm.destroy();
  printf("  Cleanup: PASSED\n\n");
}

static void testTouchRange() {
  printf("=== Test: Touch Range ===\n");

  GpuVirtualMemoryManager vmm;
  GpuVmmConfig cfg;
  cfg.va_size_bytes = 64ULL * 1024 * 1024;
  cfg.fill_value = 5;

  CUresult res = vmm.init(cfg);
  assert(res == CUDA_SUCCESS);

  // Touch a 6MB range starting at offset 1MB
  // With 2MB pages, this should touch pages 0, 1, 2, 3
  size_t offset = 1ULL * 1024 * 1024;
  size_t size = 6ULL * 1024 * 1024;
  res = vmm.touchRange(offset, size);
  assert(res == CUDA_SUCCESS);

  size_t expected_first = offset / vmm.getPageSize();
  size_t expected_last = (offset + size - 1) / vmm.getPageSize();
  for (size_t i = expected_first; i <= expected_last; ++i) {
    assert(vmm.isMapped(i));
  }
  printf("  touchRange mapped pages [%zu, %zu]: PASSED\n", expected_first,
         expected_last);

  vmm.destroy();
  printf("  Cleanup: PASSED\n\n");
}

static void testLargeVaReservation() {
  printf("=== Test: Large VA Reservation (512 GB) ===\n");

  GpuVirtualMemoryManager vmm;
  GpuVmmConfig cfg;
  cfg.va_size_bytes = 512ULL * 1024 * 1024 * 1024;  // 512 GB

  CUresult res = vmm.init(cfg);
  assert(res == CUDA_SUCCESS);
  printf("  512GB VA reservation: SUCCEEDED\n");
  printf("  Total pages: %zu\n", vmm.getTotalPages());
  printf("  Mapped pages: %zu (should be 0)\n", vmm.getMappedPageCount());

  // Touch just one page to prove demand paging works at scale
  res = vmm.touchPage(0);
  assert(res == CUDA_SUCCESS);
  printf("  Touch page 0 in 512GB space: PASSED\n");

  // Touch a page far into the space (midpoint = ~256GB offset)
  size_t far_page = vmm.getTotalPages() / 2;
  res = vmm.touchPage(far_page);
  assert(res == CUDA_SUCCESS);
  printf("  Touch page %zu (midpoint ~256GB): PASSED\n", far_page);

  bool ok = verifyPage(vmm.getPagePtr(far_page), vmm.getPageSize(), 5);
  assert(ok);
  printf("  Far page verification: PASSED\n");

  // Touch the last page
  size_t last_page = vmm.getTotalPages() - 1;
  res = vmm.touchPage(last_page);
  assert(res == CUDA_SUCCESS);

  ok = verifyPage(vmm.getPagePtr(last_page), vmm.getPageSize(), 5);
  assert(ok);
  printf("  Last page %zu (~512GB): PASSED\n", last_page);

  printf("  Total mapped: %zu pages (%.2f MB physical) out of %zu pages (512GB virtual)\n",
         vmm.getMappedPageCount(),
         (double)(vmm.getMappedPageCount() * vmm.getPageSize()) / (1024.0 * 1024),
         vmm.getTotalPages());

  vmm.destroy();
  printf("  Cleanup: PASSED\n\n");
}

static void testKernelAccessFullVaSpace() {
  printf("=== Test: GPU Kernel Reads Scattered Addresses Across 512GB VA ===\n");

  GpuVirtualMemoryManager vmm;
  GpuVmmConfig cfg;
  cfg.va_size_bytes = 512ULL * 1024 * 1024 * 1024;  // 512 GB
  cfg.fill_value = 5;

  CUresult res = vmm.init(cfg);
  assert(res == CUDA_SUCCESS);

  // Choose pages spread across the entire 512GB space.
  // We can't touch all 262144 pages (would need 512GB VRAM), so we sample
  // pages at evenly-spaced intervals across the full address range.
  const size_t NUM_SAMPLE_PAGES = 256;
  size_t stride = vmm.getTotalPages() / NUM_SAMPLE_PAGES;
  if (stride == 0) stride = 1;

  std::vector<size_t> sampled_indices;
  std::vector<CUdeviceptr> sampled_ptrs;

  printf("  Sampling %zu pages across 512GB (stride=%zu pages = %.2f GB)\n",
         NUM_SAMPLE_PAGES, stride,
         (double)(stride * vmm.getPageSize()) / (1024.0 * 1024 * 1024));

  // Touch each sampled page (demand-page it and fill with 5)
  for (size_t i = 0; i < NUM_SAMPLE_PAGES; ++i) {
    size_t page_idx = i * stride;
    if (page_idx >= vmm.getTotalPages()) break;

    res = vmm.touchPage(page_idx);
    assert(res == CUDA_SUCCESS);
    sampled_indices.push_back(page_idx);
    sampled_ptrs.push_back(vmm.getPagePtr(page_idx));
  }

  size_t num_sampled = sampled_indices.size();
  printf("  Touched %zu pages (%.2f MB physical) across 512GB VA\n",
         num_sampled,
         (double)(num_sampled * vmm.getPageSize()) / (1024.0 * 1024));

  // Print address range covered
  printf("  First sampled page: %zu (VA offset: 0x%llx)\n",
         sampled_indices.front(),
         (unsigned long long)(sampled_indices.front() * vmm.getPageSize()));
  printf("  Last sampled page:  %zu (VA offset: 0x%llx = %.2f GB)\n",
         sampled_indices.back(),
         (unsigned long long)(sampled_indices.back() * vmm.getPageSize()),
         (double)(sampled_indices.back() * vmm.getPageSize()) / (1024.0 * 1024 * 1024));

  // Upload the page pointer array to GPU
  CUdeviceptr *d_page_ptrs = nullptr;
  cudaMalloc(&d_page_ptrs, num_sampled * sizeof(CUdeviceptr));
  cudaMemcpy(d_page_ptrs, sampled_ptrs.data(),
             num_sampled * sizeof(CUdeviceptr), cudaMemcpyHostToDevice);

  // Allocate results buffer on GPU
  int *d_results = nullptr;
  cudaMalloc(&d_results, num_sampled * sizeof(int));
  cudaMemset(d_results, 0, num_sampled * sizeof(int));

  // Launch ONE kernel that reads from all sampled pages simultaneously
  int threads = 256;
  int blocks = (int)((num_sampled + threads - 1) / threads);
  scatteredReadKernel<<<blocks, threads>>>(d_page_ptrs, d_results, num_sampled);

  cudaError_t err = cudaDeviceSynchronize();
  assert(err == cudaSuccess);
  printf("  Kernel launched and completed: PASSED\n");

  // Read results back to host and verify
  std::vector<int> host_results(num_sampled);
  cudaMemcpy(host_results.data(), d_results,
             num_sampled * sizeof(int), cudaMemcpyDeviceToHost);

  size_t failures = 0;
  for (size_t i = 0; i < num_sampled; ++i) {
    if (host_results[i] != 5) {
      fprintf(stderr, "  FAIL: page %zu (VA offset %.2f GB): got %d, expected 5\n",
              sampled_indices[i],
              (double)(sampled_indices[i] * vmm.getPageSize()) / (1024.0 * 1024 * 1024),
              host_results[i]);
      failures++;
    }
  }

  cudaFree(d_page_ptrs);
  cudaFree(d_results);

  if (failures == 0) {
    printf("  All %zu pages verified == 5 by GPU kernel: PASSED\n", num_sampled);
  } else {
    printf("  FAILED: %zu / %zu pages had wrong values\n", failures, num_sampled);
    assert(false);
  }

  vmm.destroy();
  printf("  Cleanup: PASSED\n\n");
}

int main() {
  printf("GPU Virtual Memory Manager -- Demand Paging Tests\n");
  printf("==================================================\n\n");

  // Ensure a CUDA context exists (runtime API creates one lazily)
  cudaFree(0);

  testBasicDemandPaging();
  testEviction();
  testTouchRange();
  testLargeVaReservation();
  testKernelAccessFullVaSpace();

  printf("All tests PASSED.\n");
  return 0;
}
