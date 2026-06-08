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

#ifndef CTP_MACROS_H
#define CTP_MACROS_H

/** For windows */
// #define _CRT_SECURE_NO_DEPRECATE

/** Function content selector for CUDA */
#ifdef __CUDA_ARCH__
#define CTP_IS_CUDA_GPU
#endif

/** Function content selector for ROCm */
#if __HIP_DEVICE_COMPILE__
#define CTP_IS_ROCM_GPU
#endif

/** Function content selector for CPU vs GPU.
 *
 * CTP_IS_GPU stays gated to CUDA/ROCm device passes only — code under
 * `#if CTP_IS_GPU` uses raw nvcc/hipcc intrinsics (__threadfence,
 * atomicCAS, __shfl_sync) that DPC++ does not provide. SYCL device-side
 * code paths are guarded with CTP_IS_SYCL_DEVICE.
 *
 * CTP_IS_DEVICE_PASS (defined after CTP_IS_SYCL_DEVICE further down
 * this file) is the union: any device-only compilation pass. Use it to
 * elide host-only headers (cout, FILE*, std streams) that would otherwise
 * leak into DPC++'s SYCL device pass parsing. */
#if defined(CTP_IS_CUDA_GPU) || defined(CTP_IS_ROCM_GPU)
#define CTP_IS_GPU 1
#define CTP_IS_HOST 0
#else
#define CTP_IS_GPU 0
#define CTP_IS_HOST 1
#endif

/** Import / export flags for MSVC DLLs */
#if CTP_COMPILER_MSVC
#define CTP_DLL_EXPORT __declspec(dllexport)
#define CTP_DLL_IMPORT __declspec(dllimport)
#else
#define CTP_DLL_EXPORT __attribute__((visibility("default")))
#define CTP_DLL_IMPORT __attribute__((visibility("default")))
#endif

/** DLL import / export for CTP code */
#if CTP_ENABLE_DLL_EXPORT
#define CTP_DLL CTP_DLL_EXPORT
#else
#define CTP_DLL CTP_DLL_IMPORT
#endif

/** DLL import / export for singletons */
#ifdef CTP_COMPILING_DLL
#define CTP_DLL_SINGLETON CTP_DLL_EXPORT
#else
#define CTP_DLL_SINGLETON CTP_DLL_IMPORT
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

#if CTP_ENABLE_CUDA || CTP_ENABLE_ROCM
#define CTP_ENABLE_CUDA_OR_ROCM 1
#endif

#if CTP_ENABLE_CUDA || CTP_ENABLE_ROCM || CTP_ENABLE_SYCL
#define CTP_ENABLE_GPU 1
#endif

/** Detect GPU compilers.
 * These combine the CMake build flag (CTP_ENABLE_CUDA / CTP_ENABLE_ROCM)
 * with the actual compiler detection (__CUDACC__ / __HIPCC__) so that GPU
 * code paths are only compiled when BOTH the build was configured for the
 * GPU backend AND the file is being compiled by the GPU compiler.
 * All other files MUST use these macros instead of raw __CUDACC__ etc. */
#if CTP_ENABLE_CUDA && defined(__CUDACC__)
#define CTP_IS_CUDA_COMPILER 1
#else
#define CTP_IS_CUDA_COMPILER 0
#endif

/** HIP-NVCC: when CLIO_ROCM_HIP_PLATFORM=nvidia, hipcc invokes nvcc and
 *  __CUDACC__ is defined (not __HIPCC__) — but we still want
 *  CTP_IS_ROCM_COMPILER=1 since the build was configured for ROCm and
 *  the headers expose HIP types via the HIP-NVCC shim. */
#if CTP_ENABLE_ROCM && (defined(__HIPCC__) || defined(__CUDACC__))
#define CTP_IS_ROCM_COMPILER 1
#else
#define CTP_IS_ROCM_COMPILER 0
#endif

/** Detect SYCL compiler (Intel oneAPI icpx -fsycl, AdaptiveCpp acpp). */
#if CTP_ENABLE_SYCL && defined(SYCL_LANGUAGE_VERSION)
#define CTP_IS_SYCL_COMPILER 1
#else
#define CTP_IS_SYCL_COMPILER 0
#endif

/** Detect device-side SYCL compilation pass.
 *  __SYCL_DEVICE_ONLY__ is defined by both DPC++ and AdaptiveCpp during the
 *  device pass of single-source SYCL compilation. */
#if CTP_IS_SYCL_COMPILER && defined(__SYCL_DEVICE_ONLY__)
#define CTP_IS_SYCL_DEVICE 1
#else
#define CTP_IS_SYCL_DEVICE 0
#endif

/** Union: any device-only compilation pass. Guard host-only code (std::cout,
 *  FILE*, exceptions) with `#if !CTP_IS_DEVICE_PASS` so DPC++'s device pass
 *  doesn't choke on it during parsing. */
#if CTP_IS_GPU || CTP_IS_SYCL_DEVICE
#define CTP_IS_DEVICE_PASS 1
#else
#define CTP_IS_DEVICE_PASS 0
#endif

/** Throw on host; drop on any device pass. CUDA/ROCm/SYCL device code cannot do
 *  exception handling, so a bare `throw` in CTP_CROSS_FUN data-structure code
 *  makes nvcc/hipcc/DPC++ fail with "device code does not support exception
 *  handling". Use CTP_THROW(expr) instead of `throw expr` in those data
 *  structures. Always terminate the call with a semicolon: on a device pass it
 *  expands to an empty statement, keeping `if (cond) CTP_THROW(...);`
 *  well-formed. */
#if !CTP_IS_DEVICE_PASS
#define CTP_THROW(X) throw X
#else
#define CTP_THROW(X)
#endif

/** CTP_IS_GPU_COMPILER stays gated to CUDA/ROCm: blocks guarded by it use
 *  raw __threadfence / atomicOr / __shfl_sync intrinsics that only exist on
 *  nvcc and hipcc. SYCL device-side code paths use CTP_IS_SYCL_COMPILER /
 *  CTP_IS_SYCL_DEVICE plus the abstractions in clio_ctp/util/gpu_intrinsics.h. */
#if CTP_IS_CUDA_COMPILER || CTP_IS_ROCM_COMPILER
#define CTP_IS_GPU_COMPILER 1
#else
#define CTP_IS_GPU_COMPILER 0
#endif

/** Includes for CUDA and ROCm.
 * nvcc/hipcc get the full runtime header (includes device builtins).
 * Regular g++/clang++ with CTP_ENABLE_CUDA/ROCM get the runtime API header
 * which provides host-callable functions (cudaMalloc, cudaMemcpy, etc.)
 * without device builtins (atomicAdd, threadIdx, etc.). */
#if CTP_IS_CUDA_COMPILER
#include <cuda_runtime.h>
#elif CTP_ENABLE_CUDA
#include <cuda_runtime_api.h>
#endif

#if CTP_IS_ROCM_COMPILER
#include <hip/hip_runtime.h>
#elif CTP_ENABLE_ROCM
// g++/clang++ host TUs with CTP_ENABLE_ROCM get the host-callable HIP
// runtime API header (hipMalloc, hipMemcpy, hipIpcMemHandle_t, etc.)
// without device builtins. Under HIP_PLATFORM=nvidia these forward to
// cudart symbols at link time.
#include <hip/hip_runtime_api.h>
#endif

// Pull <sycl/sycl.hpp> in whenever the SYCL backend is enabled, not only
// when this TU is being compiled with -fsycl. Host TUs in chimaera_cxx
// (compiled by dpcpp without -fsycl) still need sycl::malloc_host /
// sycl::queue / sycl::free declarations because gpu_api.h's sycl::
// branches activate on CTP_ENABLE_SYCL=1. The header is plain C++ —
// safe to parse without the SYCL compiler frontend.
#if CTP_ENABLE_SYCL
#include <sycl/sycl.hpp>
#endif

/** Macros for CUDA functions.
 * CUDA/ROCm keywords (__host__, __device__, etc.) are compiler built-ins
 * that only exist when compiling with nvcc or hipcc.  Defining them
 * unconditionally causes errors when the same header is included in files
 * compiled with a plain C++ compiler (g++/clang++). */
#if CTP_IS_GPU_COMPILER
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
#define HIP_ERROR_CHECK(X)                                                 \
  do {                                                                     \
    if (X != hipSuccess) {                                                 \
      hipError_t hipErr = hipGetLastError();                               \
      HLOG(kFatal, "HIP Error {}: {}", hipErr, hipGetErrorString(hipErr)); \
    }                                                                      \
  } while (false)

/** Error checking for CUDA */
#define CUDA_ERROR_CHECK(X)                                                    \
  do {                                                                         \
    if (X != cudaSuccess) {                                                    \
      cudaError_t cudaErr = cudaGetLastError();                                \
      HLOG(kFatal, "CUDA Error {}: {}", cudaErr, cudaGetErrorString(cudaErr)); \
    }                                                                          \
  } while (false)

/**
 * Ensure that the compiler ALWAYS inlines a particular function.
 * */
#if CTP_COMPILER_MSVC
#define CTP_INLINE_FLAG __forceinline
#define CTP_NO_INLINE_FLAG __declspec(noinline)
#define CTP_FUNC_IS_USED __declspec(selectany)
#elif CTP_COMPILER_GNU
#define CTP_INLINE_FLAG __attribute__((always_inline))
#define CTP_NO_INLINE_FLAG __attribute__((noinline))
#define CTP_FUNC_IS_USED __attribute__((used))
#else
#define CTP_INLINE_FLAG inline
#define CTP_NO_INLINE_FLAG
#define CTP_FUNC_IS_USED
#endif

#define CTP_NO_INLINE CTP_NO_INLINE_FLAG
#ifndef CTP_DEBUG
#define CTP_INLINE
#else
#define CTP_INLINE inline
#endif

/** Macros for gpu/host function + var */
#define CTP_HOST_FUN ROCM_HOST
#define CTP_HOST_VAR ROCM_HOST
#define CTP_GPU_FUN ROCM_DEVICE
#define CTP_GPU_VAR ROCM_DEVICE
#define CTP_CROSS_FUN ROCM_HOST_DEVICE
#define CTP_GPU_KERNEL ROCM_KERNEL

/** Macro for inline gpu/host function + var */
#if CTP_IS_GPU_COMPILER
#define CTP_INLINE_CROSS_FUN CTP_CROSS_FUN __forceinline__
#else
#define CTP_INLINE_CROSS_FUN CTP_CROSS_FUN inline
#endif
#define CTP_INLINE_CROSS_VAR CTP_CROSS_FUN inline
#define CTP_INLINE_GPU_FUN ROCM_DEVICE CTP_INLINE
#define CTP_INLINE_GPU_VAR ROCM_DEVICE inline
#define CTP_INLINE_HOST_FUN ROCM_HOST CTP_INLINE
#define CTP_INLINE_HOST_VAR ROCM_HOST inline

/** Macro for selective cross function */
#if CTP_IS_HOST
#define CTP_CROSS_FUN_SEL CTP_HOST_FUN
#define CTP_INLINE_CROSS_FUN_SEL CTP_INLINE_HOST_FUN
#else
#define CTP_CROSS_FUN_SEL CTP_GPU_FUN
#define CTP_INLINE_CROSS_FUN_SEL CTP_INLINE_GPU_FUN
#endif

/** Test cross functions */
#define CTP_NO_INLINE_CROSS_FUN CTP_NO_INLINE CTP_CROSS_FUN CTP_FUNC_IS_USED

/** Mark a device function whose definition lives in another translation unit.
 *
 *  Use on every CTP_GPU_FUN declaration in chimod headers (e.g. bdev's
 *  AllocateBlocks(), CTE core's PutBlob()) when the body sits in a
 *  separate _gpu.cc / _sycl.cc TU.
 *
 *  - On CUDA/ROCm: expands to nothing; nvcc/hipcc resolve cross-TU device
 *    references through device-side relocatable linking by default
 *    (CUDA_SEPARABLE_COMPILATION = ON).
 *  - On DPC++/AdaptiveCpp: expands to SYCL_EXTERNAL. SYCL kernels can only
 *    call functions that are either (a) defined in the same TU or (b)
 *    marked SYCL_EXTERNAL — without the marker, DPC++ fails the device
 *    compile with "SYCL kernel cannot call an undefined function". The
 *    matching definition's TU must also be compiled with -fsycl. */
#if CTP_IS_SYCL_COMPILER
#define CTP_DEVICE_EXTERN SYCL_EXTERNAL
#else
#define CTP_DEVICE_EXTERN
#endif

/** Mark a function whose address is taken on the device.
 *
 *  Use on every function that gets stored in a function-pointer table the
 *  device side dispatches through (for example chi::gpu::Container::run_,
 *  alloc_task_, save_task_, ...).
 *
 *  - On CUDA/ROCm: expands to nothing; nvcc/hipcc allow taking the address
 *    of any __device__ function.
 *  - On DPC++: emits [[intel::device_indirectly_callable]]. Recent DPC++
 *    nightlies reject `&fn` from device code unless the function is tagged,
 *    even with `-Xclang -fsycl-allow-func-ptr` enabled. The attribute also
 *    makes call sites work under the sycl_ext_oneapi_virtual_functions
 *    extension when CLIO_SYCL_ALLOW_VIRTUAL_FUNCTIONS is on. Override with
 *    -DHSHM_NO_SYCL_INDIRECTLY_CALLABLE=1 to fall back to no-op. */
#if CTP_IS_SYCL_COMPILER && !defined(CTP_NO_SYCL_INDIRECTLY_CALLABLE)
#define CTP_INDIRECTLY_CALLABLE [[intel::device_indirectly_callable]]
#else
#define CTP_INDIRECTLY_CALLABLE
#endif

/** Bitfield macros */
#define MARK_FIRST_BIT_MASK(T) ((T)1 << (sizeof(T) * 8 - 1))
#define MARK_FIRST_BIT(T, X) ((X) | MARK_FIRST_BIT_MASK(T))
#define IS_FIRST_BIT_MARKED(T, X) ((X) & MARK_FIRST_BIT_MASK(T))
#define UNMARK_FIRST_BIT(T, X) ((X) & ~MARK_FIRST_BIT_MASK(T))

/** Class constant macro */
#define CLS_CONST static inline constexpr const
#define CLS_CROSS_CONST CLS_CONST

/** Class constant macro */
#if CTP_IS_HOST
#define GLOBAL_CONST inline const
#define GLOBAL_CROSS_CONST inline const
#else
#define GLOBAL_CONST inline const
#define GLOBAL_CROSS_CONST inline const __device__ __constant__
#endif

/** Namespace definitions */
namespace ctp {}
namespace ctp::ipc {}

/** The name of the current device */
#define CTP_DEV_TYPE_CPU 0
#define CTP_DEV_TYPE_GPU 1
#if CTP_IS_HOST
#define kCurrentDevice "cpu"
#define kCurrentDeviceType CTP_DEV_TYPE_CPU
#define CTP_GPU_OR_HOST host
#else
#define kCurrentDevice "gpu"
#define kCurrentDeviceType CTP_DEV_TYPE_GPU
#define CTP_GPU_OR_HOST gpu
#endif

/***************************************************
 * CUSTOM SETTINGS FOR ALLOCATORS
 * ************************************************* */
/** Define the root allocator class */
#ifndef CTP_ROOT_ALLOC_T
#define CTP_ROOT_ALLOC_T ctp::ipc::StackAllocator
#endif
#define CTP_ROOT_ALLOC \
  CTP_MEMORY_MANAGER->template GetRootAllocator<CTP_ROOT_ALLOC_T>()

#define CTP_DEFAULT_ALLOC \
  CTP_MEMORY_MANAGER->template GetDefaultAllocator<CTP_DEFAULT_ALLOC_T>()

#ifndef CTP_DEFAULT_ALLOC_GPU_T
#define CTP_DEFAULT_ALLOC_GPU_T ctp::ipc::PartitionedAllocator
#endif

/** Default memory context macro (no longer used - kept for compatibility) */
#define CTP_MCTX (void)0

/** Compatability hack for static_assert */
template <bool TRUTH, typename T = int>
class assert_hack {
 public:
  CLS_CONST bool value = TRUTH;
};

/** A hack for static asserts */
#define STATIC_ASSERT(TRUTH, MSG, T) \
  static_assert(assert_hack<TRUTH, __TU(T)>::value, MSG)

#endif  // CTP_MACROS_H
