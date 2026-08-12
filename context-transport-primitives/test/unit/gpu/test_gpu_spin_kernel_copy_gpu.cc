/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * Concurrency invariant: a RESIDENT, SPINNING GPU kernel must not prevent the
 * CPU from completing a device->host copy on its own non-blocking stream.
 *
 * The whole GPU->CPU submit path depends on this. A GPU producer kernel parks
 * on a host-visible flag (a ring-buffer head, a task completion word) while the
 * CPU worker services the submitted task -- and servicing it means a D2H copy
 * (see FsBdevTransport::WriteBlocks -> ctp::DeviceAwareMemcpy). If a spinning
 * kernel could starve that copy, GPU and CPU would deadlock on each other.
 *
 * These cases pin down that DeviceAwareMemcpy's exact pattern (async copy on a
 * dedicated cudaStreamNonBlocking stream, then synchronize) keeps working while
 * a kernel spins on a host flag, across the memory kinds a registered GPU
 * backend can use (kDeviceMem / kManagedUvm) and both destination kinds the
 * bdev can hand us (pinned and pageable -- the bdev stages into a
 * std::vector<char>, which is pageable).
 *
 * NOTE (why this is worth a regression test): this invariant genuinely holds at
 * the CUDA level, so a hang in the GPU submit path is NOT explained by "a
 * spinning kernel starves the copy engine". Keeping this green stops that wrong
 * theory from being re-derived.
 *
 * That theory was the first (wrong) explanation for the real gpu2cpu ring
 * deadlock, whose actual cause is the CPU consumer ceasing to advance the ring's
 * head_ once a GPU producer parks in Emplace's wait-for-space spin. See
 * context-runtime/test/unit/gpu/GPU2CPU_RING_DEADLOCK.md.
 */

#include <catch2/catch_all.hpp>

#include <atomic>
#include <chrono>
#include <thread>

#include "clio_ctp/util/gpu_api.h"

using ctp::GpuApi;

namespace {

/**
 * Spins until the CPU flips `flag` (pinned host memory). This mirrors the GPU
 * producer parking on a ring-buffer head / task completion word: one thread of
 * one block, hammering host memory over PCIe.
 */
__global__ void SpinOnHostFlagKernel(volatile unsigned int *flag,
                                     unsigned long long *iters) {
  if (threadIdx.x != 0) return;
  unsigned long long i = 0;
  while (*flag == 0u) {
    ++i;
  }
  *iters = i;
}

/** Bytes moved per probe copy: same order as a kvhdf5 chunk. */
constexpr size_t kCopyBytes = 1u << 20;

/** How long we let the copy thread try before declaring a starvation bug. */
constexpr int kCopyTimeoutMs = 10000;

/**
 * Launch the spinning kernel, then -- from another thread, while it is still
 * resident -- run DeviceAwareMemcpy's exact pattern from `src` into `dst`.
 * Returns the number of copies that completed. 0 means the copy was starved
 * (the bug this test guards against). Always releases the kernel before
 * returning so the test cannot wedge the process.
 */
int CopiesWhileKernelSpins(void *src, void *dst) {
  volatile unsigned int *flag = GpuApi::MallocHost<unsigned int>(1);
  REQUIRE(flag != nullptr);
  *flag = 0u;
  unsigned long long *iters = GpuApi::Malloc<unsigned long long>(1);

  SpinOnHostFlagKernel<<<1, 256>>>(flag, iters);
  REQUIRE(cudaGetLastError() == cudaSuccess);
  // Give the kernel time to actually become resident on the device.
  std::this_thread::sleep_for(std::chrono::milliseconds(250));

  std::atomic<int> copies{0};
  std::atomic<bool> done{false};
  std::thread copier([&] {
    // Exactly ctp::DeviceAwareMemcpy: a dedicated non-blocking stream, an async
    // copy, then a synchronize.
    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    for (int i = 0; i < 8; ++i) {
      if (cudaMemcpyAsync(dst, src, kCopyBytes, cudaMemcpyDefault, s) !=
          cudaSuccess) {
        break;
      }
      if (cudaStreamSynchronize(s) != cudaSuccess) break;
      copies.fetch_add(1);
    }
    cudaStreamDestroy(s);
    done = true;
  });

  // Watchdog: never block forever, even if the invariant is violated.
  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::milliseconds(kCopyTimeoutMs);
  while (!done && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  const int observed = copies.load();

  // Release the kernel and reap everything, whatever happened above.
  *flag = 1u;
  __sync_synchronize();
  GpuApi::Synchronize();
  copier.join();

  GpuApi::FreeHost(const_cast<unsigned int *>(flag));
  GpuApi::Free(iters);
  return observed;
}

}  // namespace

TEST_CASE("D2H copy completes while a kernel spins: device -> pinned",
          "[gpu][spin][copy]") {
  char *src = GpuApi::Malloc<char>(kCopyBytes);
  char *dst = GpuApi::MallocHost<char>(kCopyBytes);
  REQUIRE(src != nullptr);
  REQUIRE(dst != nullptr);

  REQUIRE(CopiesWhileKernelSpins(src, dst) == 8);

  GpuApi::Free(src);
  GpuApi::FreeHost(dst);
}

TEST_CASE("D2H copy completes while a kernel spins: device -> pageable",
          "[gpu][spin][copy]") {
  // The fs bdev stages into a std::vector<char>, i.e. pageable host memory.
  char *src = GpuApi::Malloc<char>(kCopyBytes);
  std::vector<char> dst(kCopyBytes);
  REQUIRE(src != nullptr);

  REQUIRE(CopiesWhileKernelSpins(src, dst.data()) == 8);

  GpuApi::Free(src);
}

TEST_CASE("D2H copy completes while a kernel spins: managed -> pinned",
          "[gpu][spin][copy]") {
  char *src = GpuApi::MallocManaged<char>(kCopyBytes);
  char *dst = GpuApi::MallocHost<char>(kCopyBytes);
  REQUIRE(src != nullptr);
  REQUIRE(dst != nullptr);

  REQUIRE(CopiesWhileKernelSpins(src, dst) == 8);

  GpuApi::Free(src);
  GpuApi::FreeHost(dst);
}

TEST_CASE("D2H copy completes while a kernel spins: managed -> pageable",
          "[gpu][spin][copy]") {
  char *src = GpuApi::MallocManaged<char>(kCopyBytes);
  std::vector<char> dst(kCopyBytes);
  REQUIRE(src != nullptr);

  REQUIRE(CopiesWhileKernelSpins(src, dst.data()) == 8);

  GpuApi::Free(src);
}
