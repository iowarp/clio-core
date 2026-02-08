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

#ifndef HSHM_MACROS_H
#define HSHM_MACROS_H

/** For windows */
// #define _CRT_SECURE_NO_DEPRECATE

/** Function content selector for CUDA */
#ifdef __CUDA_ARCH__
#define HSHM_IS_CUDA_GPU
#endif

/** Function content selector for ROCm */
#if __HIP_DEVICE_COMPILE__
#define HSHM_IS_ROCM_GPU
#endif

/** Function content selector for CPU vs GPU */
#if defined(HSHM_IS_CUDA_GPU) || defined(HSHM_IS_ROCM_GPU)
#define HSHM_IS_GPU 1
#define HSHM_IS_HOST 0
#else
#define HSHM_IS_GPU 0
#define HSHM_IS_HOST 1
#endif

/** Import / export flags for MSVC DLLs */
#if HSHM_COMPILER_MSVC
#define HSHM_DLL_EXPORT __declspec(dllexport)
#define HSHM_DLL_IMPORT __declspec(dllimport)
#else
#define HSHM_DLL_EXPORT __attribute__((visibility("default")))
#define HSHM_DLL_IMPORT __attribute__((visibility("default")))
#endif

/** DLL import / export for HSHM code */
#if HSHM_ENABLE_DLL_EXPORT
#define HSHM_DLL HSHM_DLL_EXPORT
#else
#define HSHM_DLL HSHM_DLL_IMPORT
#endif

/** DLL import / export for singletons */
#ifdef HSHM_COMPILING_DLL
#define HSHM_DLL_SINGLETON HSHM_DLL_EXPORT
#else
#define HSHM_DLL_SINGLETON HSHM_DLL_IMPORT
#endif

/**
 * Remove parenthesis surrounding "X" if it has parenthesis
 * Used for helper macros which take templated types as parameters
 * E.g., let's say we have:
 *
 * #define HELPER_MACRO(T) TYPE_UNWRAP(T)
 * HELPER_MACRO( (std::vector<std::pair<int, int>>) )
 * will return std::vector<std::pair<int, int>> without the parenthesis
 * */
#define TYPE_UNWRAP(X) ESC(ISH X)
#define ISH(...) ISH __VA_ARGS__
#define ESC(...) ESC_(__VA_ARGS__)
#define ESC_(...) VAN##__VA_ARGS__
#define VANISH
#define __TU(X) TYPE_UNWRAP(X)

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
#define HSHM_ENABLE_CUDA_OR_ROCM 1
#endif

/** Includes for CUDA and ROCm */
#if HSHM_ENABLE_CUDA && defined(__CUDACC__)
#include <cuda_runtime.h>
#endif

#if HSHM_ENABLE_ROCM && defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#endif

/** Macros for CUDA functions */
#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
#define ROCM_HOST __host__
#define ROCM_DEVICE __device__
#define ROCM_HOST_DEVICE __device__ __host__
#define ROCM_KERNEL __global__
#else
#define ROCM_HOST_DEVICE
#define ROCM_HOST
#define ROCM_DEVICE
#define ROCM_KERNEL
#endif

/** Error checking for ROCM */
#define HIP_ERROR_CHECK(X)                                                  \
  do {                                                                      \
    if (X != hipSuccess) {                                                  \
      hipError_t hipErr = hipGetLastError();                                \
      HLOG(kFatal, "HIP Error {}: {}", hipErr, hipGetErrorString(hipErr)); \
    }                                                                       \
  } while (false)

/** Error checking for CUDA */
#define CUDA_ERROR_CHECK(X)                       \
  do {                                            \
    if (X != cudaSuccess) {                       \
      cudaError_t cudaErr = cudaGetLastError();   \
      HLOG(kFatal, "CUDA Error {}: {}", cudaErr, \
            cudaGetErrorString(cudaErr));         \
    }                                             \
  } while (false)

/**
 * Ensure that the compiler ALWAYS inlines a particular function.
 * */
#if HSHM_COMPILER_MSVC
#define HSHM_INLINE_FLAG __forceinline
#define HSHM_NO_INLINE_FLAG __declspec(noinline)
#define HSHM_FUNC_IS_USED __declspec(selectany)
#elif HSHM_COMPILER_GNU
#define HSHM_INLINE_FLAG __attribute__((always_inline))
#define HSHM_NO_INLINE_FLAG __attribute__((noinline))
#define HSHM_FUNC_IS_USED __attribute__((used))
#else
#define HSHM_INLINE_FLAG inline
#define HSHM_NO_INLINE_FLAG
#define HSHM_FUNC_IS_USED
#endif

#define HSHM_NO_INLINE HSHM_NO_INLINE_FLAG
#ifndef HSHM_DEBUG
#define HSHM_INLINE
#else
#define HSHM_INLINE inline
#endif

/** Macros for gpu/host function + var */
#define HSHM_HOST_FUN ROCM_HOST
#define HSHM_HOST_VAR ROCM_HOST
#define HSHM_GPU_FUN ROCM_DEVICE
#define HSHM_GPU_VAR ROCM_DEVICE
#define HSHM_CROSS_FUN ROCM_HOST_DEVICE
#define HSHM_GPU_KERNEL ROCM_KERNEL

/** Macro for inline gpu/host function + var */
#define HSHM_INLINE_CROSS_FUN HSHM_CROSS_FUN inline
#define HSHM_INLINE_CROSS_VAR HSHM_CROSS_FUN inline
#define HSHM_INLINE_GPU_FUN ROCM_DEVICE HSHM_INLINE
#define HSHM_INLINE_GPU_VAR ROCM_DEVICE inline
#define HSHM_INLINE_HOST_FUN ROCM_HOST HSHM_INLINE
#define HSHM_INLINE_HOST_VAR ROCM_HOST inline

/** Macro for selective cross function */
#if HSHM_IS_HOST
#define HSHM_CROSS_FUN_SEL HSHM_HOST_FUN
#define HSHM_INLINE_CROSS_FUN_SEL HSHM_INLINE_HOST_FUN
#else
#define HSHM_CROSS_FUN_SEL HSHM_GPU_FUN
#define HSHM_INLINE_CROSS_FUN_SEL HSHM_INLINE_GPU_FUN
#endif

/** Test cross functions */
#define HSHM_NO_INLINE_CROSS_FUN HSHM_NO_INLINE HSHM_CROSS_FUN HSHM_FUNC_IS_USED

/** Bitfield macros */
#define MARK_FIRST_BIT_MASK(T) ((T)1 << (sizeof(T) * 8 - 1))
#define MARK_FIRST_BIT(T, X) ((X) | MARK_FIRST_BIT_MASK(T))
#define IS_FIRST_BIT_MARKED(T, X) ((X) & MARK_FIRST_BIT_MASK(T))
#define UNMARK_FIRST_BIT(T, X) ((X) & ~MARK_FIRST_BIT_MASK(T))

/** Class constant macro */
#define CLS_CONST static inline constexpr const
#define CLS_CROSS_CONST CLS_CONST

/** Class constant macro */
#if HSHM_IS_HOST
#define GLOBAL_CONST inline const
#define GLOBAL_CROSS_CONST inline const
#else
#define GLOBAL_CONST inline const
#define GLOBAL_CROSS_CONST inline const __device__ __constant__
#endif

/** Namespace definitions */
namespace hshm {}
namespace hshm::ipc {}
namespace hipc = hshm::ipc;

/** The name of the current device */
#define HSHM_DEV_TYPE_CPU 0
#define HSHM_DEV_TYPE_GPU 1
#if HSHM_IS_HOST
#define kCurrentDevice "cpu"
#define kCurrentDeviceType HSHM_DEV_TYPE_CPU
#define HSHM_GPU_OR_HOST host
#else
#define kCurrentDevice "gpu"
#define kCurrentDeviceType HSHM_DEV_TYPE_GPU
#define HSHM_GPU_OR_HOST gpu
#endif

/***************************************************
 * CUSTOM SETTINGS FOR ALLOCATORS
 * ************************************************* */
/** Define the root allocator class */
#ifndef HSHM_ROOT_ALLOC_T
#define HSHM_ROOT_ALLOC_T hipc::StackAllocator
#endif
#define HSHM_ROOT_ALLOC \
  HSHM_MEMORY_MANAGER->template GetRootAllocator<HSHM_ROOT_ALLOC_T>()

#define HSHM_DEFAULT_ALLOC \
  HSHM_MEMORY_MANAGER->template GetDefaultAllocator<HSHM_DEFAULT_ALLOC_T>()

/** Default memory context macro (no longer used - kept for compatibility) */
#define HSHM_MCTX (void)0

/** Compatability hack for static_assert */
template <bool TRUTH, typename T = int>
class assert_hack {
 public:
  CLS_CONST bool value = TRUTH;
};

/** A hack for static asserts */
#define STATIC_ASSERT(TRUTH, MSG, T) \
  static_assert(assert_hack<TRUTH, __TU(T)>::value, MSG)

#endif  // HSHM_MACROS_H
