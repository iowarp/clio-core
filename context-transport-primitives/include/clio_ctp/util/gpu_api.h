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

#ifndef CTP_UTIL_GPU_API_H
#define CTP_UTIL_GPU_API_H

#include <cstring>

#include "clio_ctp/constants/macros.h"
#include "clio_ctp/util/logging.h"

namespace ctp {

struct GpuIpcMemHandle {
#if CTP_ENABLE_CUDA
  cudaIpcMemHandle_t cuda_;
#endif
#if CTP_ENABLE_ROCM
  hipIpcMemHandle_t rocm_;
#endif
#if CTP_ENABLE_SYCL
  void *sycl_ptr_;  // SYCL USM pointers are directly shareable; store base ptr
#endif
};

class GpuApi {
 public:
  static void SetDevice(int gpu_id) {
#if CTP_ENABLE_CUDA
    CUDA_ERROR_CHECK(cudaSetDevice(gpu_id));
#elif CTP_ENABLE_ROCM
    HIP_ERROR_CHECK(hipSetDevice(gpu_id));
#elif CTP_ENABLE_SYCL
    // SYCL device selection is done via queue construction; no global set
    (void)gpu_id;
#endif
  }

  static int GetDeviceCount() {
    int ngpu = 0;
#if CTP_ENABLE_ROCM
    if (hipGetDeviceCount(&ngpu) != hipSuccess) {
      ngpu = 0;
    }
#elif CTP_ENABLE_CUDA
    if (cudaGetDeviceCount(&ngpu) != cudaSuccess) {
      cudaGetLastError();  // Clear the error state
      ngpu = 0;
    }
#elif CTP_ENABLE_SYCL
    auto platforms = sycl::platform::get_platforms();
    for (auto &p : platforms) {
      auto devs = p.get_devices(sycl::info::device_type::gpu);
      ngpu += static_cast<int>(devs.size());
    }
#endif
    return ngpu;
  }

  static void Synchronize() {
#if CTP_ENABLE_ROCM
    HIP_ERROR_CHECK(hipDeviceSynchronize());
#elif CTP_ENABLE_CUDA
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
#elif CTP_ENABLE_SYCL
    SyclQueue().wait_and_throw();
#endif
  }

  /** Synchronize a specific GPU stream instead of the whole device.
   *  Under SYCL, "stream" is a heap-allocated sycl::queue created by
   *  CreateStream(); pass null to fall back to whole-device synchronize. */
  static void Synchronize(void *stream) {
#if CTP_ENABLE_ROCM
    HIP_ERROR_CHECK(hipStreamSynchronize(static_cast<hipStream_t>(stream)));
#endif
#if CTP_ENABLE_CUDA
    CUDA_ERROR_CHECK(
        cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
#endif
#if CTP_ENABLE_SYCL
    if (stream) {
      static_cast<sycl::queue *>(stream)->wait_and_throw();
    } else {
      Synchronize();
    }
#endif
  }

  /** Create a non-blocking GPU stream.
   *  Under SYCL, allocates a heap sycl::queue selected against the default
   *  GPU device with the in_order property so submission order matches
   *  CUDA stream semantics. Caller owns the returned pointer. */
  static void *CreateStream() {
    void *stream = nullptr;
#if CTP_ENABLE_ROCM
    hipStream_t s;
    HIP_ERROR_CHECK(
        hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
    stream = s;
#endif
#if CTP_ENABLE_CUDA
    cudaStream_t s;
    CUDA_ERROR_CHECK(
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    stream = s;
#endif
#if CTP_ENABLE_SYCL
    stream = new sycl::queue(sycl::gpu_selector_v,
                             sycl::property::queue::in_order{});
#endif
    return stream;
  }

  /** Destroy a GPU stream */
  static void DestroyStream(void *stream) {
#if CTP_ENABLE_ROCM
    HIP_ERROR_CHECK(hipStreamDestroy(static_cast<hipStream_t>(stream)));
#endif
#if CTP_ENABLE_CUDA
    CUDA_ERROR_CHECK(
        cudaStreamDestroy(static_cast<cudaStream_t>(stream)));
#endif
#if CTP_ENABLE_SYCL
    delete static_cast<sycl::queue *>(stream);
#endif
  }

  static void GetIpcMemHandle(GpuIpcMemHandle &ipc, void *data) {
#if CTP_ENABLE_ROCM
    HIP_ERROR_CHECK(hipIpcGetMemHandle(&ipc.rocm_, (void *)data));
#endif
#if CTP_ENABLE_CUDA
    CUDA_ERROR_CHECK(cudaIpcGetMemHandle(&ipc.cuda_, (void *)data));
#endif
  }

  template <typename T>
  static void OpenIpcMemHandle(GpuIpcMemHandle &ipc, T **data) {
#if CTP_ENABLE_ROCM
    HIP_ERROR_CHECK(hipIpcOpenMemHandle((void **)data, ipc.rocm_,
                                        hipIpcMemLazyEnablePeerAccess));
#endif
#if CTP_ENABLE_CUDA
    CUDA_ERROR_CHECK(cudaIpcOpenMemHandle((void **)data, ipc.cuda_,
                                          cudaIpcMemLazyEnablePeerAccess));
#endif
  }

  static void CloseIpcMemHandle(void *data) {
#if CTP_ENABLE_ROCM
    HIP_ERROR_CHECK(hipIpcCloseMemHandle(data));
#endif
#if CTP_ENABLE_CUDA
    CUDA_ERROR_CHECK(cudaIpcCloseMemHandle(data));
#endif
  }

  template <typename T>
  static T *Malloc(size_t size) {
#if CTP_ENABLE_ROCM
    T *ptr;
    HIP_ERROR_CHECK(hipMalloc(&ptr, size));
    return ptr;
#elif CTP_ENABLE_CUDA
    void *vptr;
    CUDA_ERROR_CHECK(cudaMalloc(&vptr, size));
    return static_cast<T *>(vptr);
#elif CTP_ENABLE_SYCL
    return static_cast<T *>(sycl::malloc_device(size, SyclQueue()));
#else
    (void)size;
    return nullptr;
#endif
  }

  template <typename T>
  static T *MallocManaged(size_t size) {
#if CTP_ENABLE_ROCM
    void *ptr = nullptr;
    HIP_ERROR_CHECK(hipMallocManaged(&ptr, size));
    return static_cast<T *>(ptr);
#elif CTP_ENABLE_CUDA
    void *ptr = nullptr;
    CUDA_ERROR_CHECK(cudaMallocManaged(&ptr, size));
    return static_cast<T *>(ptr);
#elif CTP_ENABLE_SYCL
    return static_cast<T *>(sycl::malloc_shared(size, SyclQueue()));
#endif
    return nullptr;
  }

  template <typename T>
  static void RegisterHostMemory(T *ptr, size_t size) {
#if CTP_ENABLE_ROCM
    HIP_ERROR_CHECK(
        hipHostRegister((void *)ptr, size, hipHostRegisterPortable | hipHostRegisterMapped));
#elif CTP_ENABLE_CUDA
    CUDA_ERROR_CHECK(
        cudaHostRegister((void *)ptr, size, cudaHostRegisterPortable | cudaHostRegisterMapped));
#elif CTP_ENABLE_SYCL
    // SYCL USM host memory doesn't require explicit registration;
    // use sycl::malloc_host for GPU-accessible host allocations when needed.
    (void)ptr; (void)size;
#endif
  }

  template <typename T>
  static void UnregisterHostMemory(T *ptr) {
#if CTP_ENABLE_ROCM
    HIP_ERROR_CHECK(hipHostUnregister((void *)ptr));
#elif CTP_ENABLE_CUDA
    CUDA_ERROR_CHECK(cudaHostUnregister((void *)ptr));
#elif CTP_ENABLE_SYCL
    (void)ptr;
#endif
  }

  template <typename T>
  static void Memcpy(T *dst, const T *src, size_t size) {
#if CTP_ENABLE_ROCM
    HIP_ERROR_CHECK(hipMemcpy(dst, src, size, hipMemcpyDefault));
#elif CTP_ENABLE_CUDA
    CUDA_ERROR_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDefault));
#elif CTP_ENABLE_SYCL
    SyclQueue().memcpy(dst, src, size).wait_and_throw();
#endif
  }

  template <typename T>
  static bool IsDevicePointer(T *ptr) {
    if (ptr == nullptr) return false;
#if CTP_ENABLE_ROCM
    hipPointerAttribute_t attributes{};
    HIP_ERROR_CHECK(hipPointerGetAttributes(&attributes, (void *)ptr));
#if defined(HIP_VERSION) && HIP_VERSION >= 60000000
    return attributes.type == hipMemoryTypeDevice;
#else
    return attributes.memoryType == hipMemoryTypeDevice;
#endif
#elif CTP_ENABLE_CUDA
    cudaPointerAttributes attributes{};
    CUDA_ERROR_CHECK(cudaPointerGetAttributes(&attributes, (void *)ptr));
    return attributes.type == cudaMemoryTypeDevice;
#elif CTP_ENABLE_SYCL
    auto kind = sycl::get_pointer_type(static_cast<const void *>(ptr),
                                        SyclQueue().get_context());
    return kind == sycl::usm::alloc::device;
#else
    return false;
#endif
  }

  template <typename T>
  static void Memset(T *dst, int value, size_t size) {
    if (IsDevicePointer(dst)) {
#if CTP_ENABLE_ROCM
      HIP_ERROR_CHECK(hipMemset(dst, value, size));
#endif
#if CTP_ENABLE_CUDA
      CUDA_ERROR_CHECK(cudaMemset(dst, value, size));
#endif
    } else {
      memset(dst, value, size);
    }
  }

  template <typename T>
  static void Free(T *ptr) {
#if CTP_ENABLE_ROCM
    HIP_ERROR_CHECK(hipFree(ptr));
#elif CTP_ENABLE_CUDA
    CUDA_ERROR_CHECK(cudaFree(ptr));
#elif CTP_ENABLE_SYCL
    sycl::free(ptr, SyclQueue());
#endif
  }

  /** Allocate pinned host memory accessible by GPU. */
  template <typename T>
  static T *MallocHost(size_t size) {
#if CTP_ENABLE_ROCM
    T *ptr;
    HIP_ERROR_CHECK(hipHostMalloc(&ptr, size, hipHostMallocDefault));
    return ptr;
#endif
#if CTP_ENABLE_CUDA
    void *vptr;
    CUDA_ERROR_CHECK(cudaMallocHost(&vptr, size));
    return static_cast<T *>(vptr);
#endif
#if CTP_ENABLE_SYCL
    return static_cast<T *>(sycl::malloc_host(size, SyclQueue()));
#endif
    return nullptr;
  }

  /** Free pinned host memory. */
  template <typename T>
  static void FreeHost(T *ptr) {
#if CTP_ENABLE_ROCM
    HIP_ERROR_CHECK(hipHostFree(ptr));
#endif
#if CTP_ENABLE_CUDA
    CUDA_ERROR_CHECK(cudaFreeHost(ptr));
#endif
#if CTP_ENABLE_SYCL
    sycl::free(ptr, SyclQueue());
#endif
  }

  /** Async memcpy (uses cudaMemcpyDefault direction). */
  template <typename T>
  static void MemcpyAsync(T *dst, const T *src, size_t size,
                           void *stream = nullptr) {
#if CTP_ENABLE_ROCM
    HIP_ERROR_CHECK(hipMemcpyAsync(dst, src, size, hipMemcpyDefault,
                                    static_cast<hipStream_t>(stream)));
#endif
#if CTP_ENABLE_CUDA
    CUDA_ERROR_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault,
                                      static_cast<cudaStream_t>(stream)));
#endif
  }

  /** Allocate device memory and copy host data into it. */
  template <typename T>
  static T *MallocAndCopy(const T *host_src, size_t copy_size,
                           size_t alloc_size) {
    T *device_ptr = Malloc<T>(alloc_size);
    if (!device_ptr) return nullptr;
    Memcpy(device_ptr, host_src, copy_size);
    return device_ptr;
  }

#if CTP_IS_GPU_COMPILER
  CTP_GPU_FUN static size_t GetGlobalThreadId() {
    return threadIdx.x + blockIdx.x * blockDim.x +
           (threadIdx.y + blockIdx.y * blockDim.y) * (blockDim.x * gridDim.x) +
           (threadIdx.z + blockIdx.z * blockDim.z) *
               (blockDim.x * gridDim.x * blockDim.y * gridDim.y);
  }
#endif

#if CTP_ENABLE_SYCL
  static sycl::queue &SyclQueue() {
    static sycl::queue q{sycl::gpu_selector_v};
    return q;
  }
#endif
};

/**
 * memcpy that transparently handles host and/or device (USM) pointers. A pure
 * host<->host copy uses std::memcpy; any copy touching device memory uses an
 * async copy on a dedicated non-blocking, per-thread stream so it cannot
 * deadlock against a kernel parked on the default stream. On a non-GPU build
 * (or for host-only pointers) it is exactly std::memcpy.
 *
 * Header-only, CUDA-free-at-the-call-site replacement for the old runtime
 * g_device_aware_memcpy hook: CPU-only callers (bdev, worker) just call this.
 */
inline void DeviceAwareMemcpy(void *dst, const void *src, size_t n) {
  if (n == 0) return;
#if CTP_ENABLE_CUDA
  // Fast path: when both pointers are plain host memory, use std::memcpy. The
  // CUDA host->host path tops out at ~1.5-3 GB/s; std::memcpy hits 8-15 GB/s.
  // Managed (UVM)/Device pointers stay on the async path so the driver handles
  // migration / D2H / D2D correctly.
  auto is_host_kind = [](const void *p) {
    cudaPointerAttributes a{};
    cudaError_t rc = cudaPointerGetAttributes(&a, p);
    if (rc == cudaErrorInvalidValue) {
      // Older CUDA returned this for unregistered host memory; clear the
      // sticky error so it doesn't trip the next CUDA call.
      (void)cudaGetLastError();
      return true;
    }
    if (rc != cudaSuccess) return false;
    return a.type == cudaMemoryTypeHost ||
           a.type == cudaMemoryTypeUnregistered;
  };
  if (is_host_kind(dst) && is_host_kind(src)) {
    std::memcpy(dst, src, n);
    return;
  }
  thread_local cudaStream_t s = nullptr;
  if (!s) {
    CUDA_ERROR_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  }
  CUDA_ERROR_CHECK(cudaMemcpyAsync(dst, src, n, cudaMemcpyDefault, s));
  CUDA_ERROR_CHECK(cudaStreamSynchronize(s));
#elif CTP_ENABLE_ROCM
  auto is_host_kind = [](const void *p) {
    hipPointerAttribute_t a{};
    hipError_t rc = hipPointerGetAttributes(&a, p);
    if (rc == hipErrorInvalidValue) {
      (void)hipGetLastError();
      return true;
    }
    if (rc != hipSuccess) return false;
#if defined(HIP_VERSION) && HIP_VERSION >= 60000000
    return a.type == hipMemoryTypeHost ||
           a.type == hipMemoryTypeUnregistered;
#else
    return a.memoryType == hipMemoryTypeHost ||
           a.memoryType == hipMemoryTypeUnregistered;
#endif
  };
  if (is_host_kind(dst) && is_host_kind(src)) {
    std::memcpy(dst, src, n);
    return;
  }
  thread_local hipStream_t s = nullptr;
  if (!s) {
    HIP_ERROR_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
  }
  HIP_ERROR_CHECK(hipMemcpyAsync(dst, src, n, hipMemcpyDefault, s));
  HIP_ERROR_CHECK(hipStreamSynchronize(s));
#else
  std::memcpy(dst, src, n);
#endif
}

/** True if ptr is device (USM) memory the host cannot dereference; false on a
 *  non-GPU build. Header-only replacement for the old g_is_device_pointer
 *  hook. */
inline bool IsDevicePointer(const void *ptr) {
  return GpuApi::IsDevicePointer(const_cast<void *>(ptr));
}

}  // namespace ctp

#endif