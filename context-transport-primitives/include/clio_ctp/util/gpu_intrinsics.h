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

#ifndef CTP_UTIL_GPU_INTRINSICS_H
#define CTP_UTIL_GPU_INTRINSICS_H

#include "clio_ctp/constants/macros.h"

#include <cstdint>
#include <cstdio>

/**
 * Backend-conditional GPU device intrinsic wrappers.
 *
 * Use CTP_DEVICE_* from device-side runtime/orchestrator code (e.g.
 * clio::run::gpu::Worker) so the same source compiles under CUDA, ROCm, or SYCL
 * device compilers. Each wrapper expands to:
 *   - the raw nvcc/hipcc intrinsic when CTP_IS_GPU_COMPILER is set
 *   - a sycl::* equivalent when CTP_IS_SYCL_COMPILER is set
 *   - a no-op stub on host-only compilation
 *
 * Scope conventions match CUDA's:
 *   _DEVICE — visible to other threads on the same GPU
 *   _SYSTEM — visible to host CPU and remote GPUs (system-coherent memory)
 */

// =====================================================================
// Memory fences
// =====================================================================

#if CTP_IS_GPU_COMPILER
#define CTP_DEVICE_FENCE_DEVICE() __threadfence()
#define CTP_DEVICE_FENCE_SYSTEM() __threadfence_system()
#elif CTP_IS_SYCL_COMPILER
#define CTP_DEVICE_FENCE_DEVICE()                                            \
  ::sycl::atomic_fence(::sycl::memory_order::seq_cst,                          \
                       ::sycl::memory_scope::device)
#define CTP_DEVICE_FENCE_SYSTEM()                                            \
  ::sycl::atomic_fence(::sycl::memory_order::seq_cst,                          \
                       ::sycl::memory_scope::system)
#else
#define CTP_DEVICE_FENCE_DEVICE() ((void)0)
#define CTP_DEVICE_FENCE_SYSTEM() ((void)0)
#endif

// =====================================================================
// Atomic OR (uint32_t) — used by clio::run::gpu::Worker for FUTURE_COMPLETE flags
// =====================================================================

#if CTP_IS_GPU_COMPILER
#define CTP_DEVICE_ATOMIC_OR_U32_DEVICE(ptr, val)                            \
  ::atomicOr(reinterpret_cast<unsigned int *>(ptr),                            \
             static_cast<unsigned int>(val))
#define CTP_DEVICE_ATOMIC_OR_U32_SYSTEM(ptr, val)                            \
  ::atomicOr_system(reinterpret_cast<unsigned int *>(ptr),                     \
                    static_cast<unsigned int>(val))
#elif CTP_IS_SYCL_COMPILER
namespace ctp::gpu_intr {
inline uint32_t atomic_or_u32_device(uint32_t *ptr, uint32_t val) {
  ::sycl::atomic_ref<uint32_t, ::sycl::memory_order::acq_rel,
                     ::sycl::memory_scope::device>
      ref(*ptr);
  return ref.fetch_or(val);
}
inline uint32_t atomic_or_u32_system(uint32_t *ptr, uint32_t val) {
  ::sycl::atomic_ref<uint32_t, ::sycl::memory_order::acq_rel,
                     ::sycl::memory_scope::system>
      ref(*ptr);
  return ref.fetch_or(val);
}
}  // namespace ctp::gpu_intr
#define CTP_DEVICE_ATOMIC_OR_U32_DEVICE(ptr, val)                            \
  ::ctp::gpu_intr::atomic_or_u32_device(                                      \
      reinterpret_cast<uint32_t *>(ptr), static_cast<uint32_t>(val))
#define CTP_DEVICE_ATOMIC_OR_U32_SYSTEM(ptr, val)                            \
  ::ctp::gpu_intr::atomic_or_u32_system(                                      \
      reinterpret_cast<uint32_t *>(ptr), static_cast<uint32_t>(val))
#else
#define CTP_DEVICE_ATOMIC_OR_U32_DEVICE(ptr, val) ((void)0)
#define CTP_DEVICE_ATOMIC_OR_U32_SYSTEM(ptr, val) ((void)0)
#endif

// =====================================================================
// Atomic ADD — used by free-list bump pointers, tag-id allocation
// =====================================================================
//
// The U64 variant operates on `unsigned long long` for ABI portability with
// CUDA's atomicAdd overload (which only accepts unsigned long long, not
// uint64_t directly).

#if CTP_IS_GPU_COMPILER
#define CTP_DEVICE_ATOMIC_ADD_U32_DEVICE(ptr, val)                           \
  ::atomicAdd(reinterpret_cast<unsigned int *>(ptr),                           \
              static_cast<unsigned int>(val))
#define CTP_DEVICE_ATOMIC_ADD_U64_DEVICE(ptr, val)                           \
  ::atomicAdd(reinterpret_cast<unsigned long long *>(ptr),                     \
              static_cast<unsigned long long>(val))
#elif CTP_IS_SYCL_COMPILER
namespace ctp::gpu_intr {
inline uint32_t atomic_add_u32_device(uint32_t *ptr, uint32_t val) {
  ::sycl::atomic_ref<uint32_t, ::sycl::memory_order::acq_rel,
                     ::sycl::memory_scope::device>
      ref(*ptr);
  return ref.fetch_add(val);
}
inline uint64_t atomic_add_u64_device(uint64_t *ptr, uint64_t val) {
  ::sycl::atomic_ref<uint64_t, ::sycl::memory_order::acq_rel,
                     ::sycl::memory_scope::device>
      ref(*ptr);
  return ref.fetch_add(val);
}
}  // namespace ctp::gpu_intr
#define CTP_DEVICE_ATOMIC_ADD_U32_DEVICE(ptr, val)                           \
  ::ctp::gpu_intr::atomic_add_u32_device(                                     \
      reinterpret_cast<uint32_t *>(ptr), static_cast<uint32_t>(val))
#define CTP_DEVICE_ATOMIC_ADD_U64_DEVICE(ptr, val)                           \
  ::ctp::gpu_intr::atomic_add_u64_device(                                     \
      reinterpret_cast<uint64_t *>(ptr), static_cast<uint64_t>(val))
#else
#define CTP_DEVICE_ATOMIC_ADD_U32_DEVICE(ptr, val) ((void)0)
#define CTP_DEVICE_ATOMIC_ADD_U64_DEVICE(ptr, val) ((void)0)
#endif

// =====================================================================
// Cycle counter — for in-kernel profiling timers
// =====================================================================

#if CTP_IS_GPU_COMPILER
#define CTP_DEVICE_CLOCK64() ::clock64()
#else
// SYCL has no portable cycle counter (DPC++ has experimental
// sycl::ext::oneapi::experimental::this_kernel::get_cycle_count() but it is
// not available on every backend). Profiling is disabled for now under
// SYCL; use SYCL queue events for end-to-end timing instead.
#define CTP_DEVICE_CLOCK64() (static_cast<long long>(0))
#endif

// =====================================================================
// Warp-level synchronization
// =====================================================================
//
// CUDA: __syncwarp() — barrier across the 32 lanes of the active warp.
// ROCm: __syncthreads() at the wavefront level (HIP exposes __syncwarp via
//       the HIP-CUDA emulation header included by macros.h).
// SYCL single_task: no-op — the kernel runs as 1 work-item; there are no
//                   peer lanes to wait for. Phase 3b's nd_range fan-out
//                   replaces this with sycl::group_barrier(sub_group).

#if CTP_IS_GPU_COMPILER
#define CTP_DEVICE_SYNCWARP() __syncwarp()
#else
#define CTP_DEVICE_SYNCWARP() ((void)0)
#endif

// =====================================================================
// printf — device-callable on CUDA/HIP, no-op on SYCL
// =====================================================================
//
// SYCL kernels cannot call variadic functions, so plain ::printf is
// unavailable. The constant-format `sycl::ext::oneapi::experimental::printf`
// extension exists but its constant-address-space format-string requirement
// is awkward to satisfy from generic code. Device prints are debug-only;
// suppress them under SYCL and rely on host-side queue events for diagnostics.
#if CTP_IS_SYCL_COMPILER
#define CTP_DEVICE_PRINTF(...) ((void)0)
#else
#define CTP_DEVICE_PRINTF(...) ::printf(__VA_ARGS__)
#endif

#endif  // CTP_UTIL_GPU_INTRINSICS_H
