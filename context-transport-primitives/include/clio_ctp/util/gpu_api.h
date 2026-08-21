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
#include <thread>
#include <chrono>
#include <atomic>
#include <string>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "clio_ctp/constants/macros.h"
#include "clio_ctp/util/logging.h"

extern "C" void ctp_copy_kernel_launch(char *dst, const char *src, size_t n,
                                       void *stream);

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

#if defined(__CUDACC__) || defined(__HIPCC__)
/** Grid-stride copy kernel: the CPU-launched alternative to cudaMemcpyAsync
 *  for device reads that stall in channel order behind a resident kernel.
 *  Kernels are SM-scheduled, so with SM headroom this executes where the
 *  engine copy cannot. */
template <typename T4>
__global__ void CtpCopyKernel(char *dst, const char *src, size_t n) {
  const size_t tid = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
  const size_t nthreads = (size_t) gridDim.x * blockDim.x;
  // Vector width only when BOTH pointers carry the alignment — task PODs
  // and scratch offsets are frequently unaligned (CUDA error 716 otherwise).
  if (((reinterpret_cast<uintptr_t>(dst) |
        reinterpret_cast<uintptr_t>(src)) & (sizeof(T4) - 1)) == 0) {
    const size_t i0 = tid * sizeof(T4);
    const size_t stride = nthreads * sizeof(T4);
    for (size_t i = i0; i + sizeof(T4) <= n; i += stride) {
      *reinterpret_cast<T4 *>(dst + i) =
          *reinterpret_cast<const T4 *>(src + i);
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      for (size_t i = n - (n % sizeof(T4)); i < n; ++i) dst[i] = src[i];
    }
  } else {
    for (size_t i = tid; i < n; i += nthreads) dst[i] = src[i];
  }
}
#endif

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

  /**
   * Completion wait that NEVER enters a blocking driver sync. Threads parked
   * inside cuStreamSynchronize can hold driver submission resources that
   * every other stream's enqueued work needs — captured as a process-wide
   * async-op stall (0%% DMA on healthy copy engines, fresh processes
   * unaffected). Service paths must poll instead.
   */
  static inline void PollSync(void *stream) {
    // Busy-spin first: most service copies land in <100 us, and a 20 us
    // sleep quantum tripled the fetch path (848 -> 3090 ms/tok measured).
    // Sleep only once the copy is provably long.
    for (int i = 0; i < 4000; ++i) {          // ~150-300 us of spin
      if (StreamQuery(stream)) return;
    }
    while (!StreamQuery(stream)) {
      std::this_thread::sleep_for(std::chrono::microseconds(2));
    }
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

  /** Device the calling thread is currently bound to (0 when not applicable). */
  static int CurrentDevice() {
    int dev = 0;
#if CTP_ENABLE_ROCM
    hipGetDevice(&dev);
#endif
#if CTP_ENABLE_CUDA
    cudaGetDevice(&dev);
#endif
    return dev;
  }

  /** Free streams per device. Function-local so the header stays standalone. */
  static std::unordered_map<int, std::vector<void *>> &StreamPool() {
    static std::unordered_map<int, std::vector<void *>> pool;
    return pool;
  }

  static std::mutex &StreamPoolMutex() {
    static std::mutex mtx;
    return mtx;
  }

  /** Borrow/return accounting. Exhaustion with (borrows - returns) far below
   *  the warmed count means streams are LEAKING; equal to it means they are
   *  legitimately all in flight. Those need opposite fixes, and the stacks
   *  alone could not tell them apart. */
  static std::atomic<long> &StreamBorrows() {
    static std::atomic<long> n{0};
    return n;
  }
  static std::atomic<long> &StreamReturns() {
    static std::atomic<long> n{0};
    return n;
  }
  /** Streams handed out by WarmStreamPool, for comparison against the above. */
  static std::atomic<long> &StreamWarmed() {
    static std::atomic<long> n{0};
    return n;
  }

  /**
   * Per-device free-list sizes, e.g. "dev0=0 dev1=64".
   *
   * Borrow and return BOTH index StreamPool() by CurrentDevice(). If a thread
   * ever returns a stream while bound to a different device than it borrowed
   * from, the stream migrates buckets: the borrowing bucket drains to zero
   * while another fills, and the borrow/return COUNTS stay perfectly balanced
   * the whole time. That is exactly the state observed -- balanced counters
   * (290484/290484, outstanding=0 at idle) yet a bucket that hits zero -- so
   * the distribution, not the total, is what has to be printed.
   * Caller must hold StreamPoolMutex().
   */
  static std::string PoolSizesLocked() {
    std::string out;
    for (auto &kv : StreamPool()) {
      out += "dev" + std::to_string(kv.first) + "=" +
             std::to_string(kv.second.size()) + " ";
    }
    return out.empty() ? std::string("(no buckets)") : out;
  }

  /** Whether a device's pool has been pre-created. */
  static std::unordered_map<int, bool> &PoolWarmed() {
    static std::unordered_map<int, bool> warmed;
    return warmed;
  }

  /**
   * Borrow a stream from a process-wide pool, creating one only if the pool
   * is empty. Return it with ReturnStream when the task's work has completed.
   *
   * Creating and destroying a stream PER I/O deadlocks the runtime under
   * concurrency. cuStreamCreate and cuStreamDestroy both take a write lock on
   * the CUDA context, so every worker doing device I/O serialises on one
   * rwlock inside libcuda; worse, cuStreamDestroy waits for the device. When
   * the device work in question is a kernel that is itself SPINNING on those
   * very I/Os to complete (a demand-paged GPU vector), the two wait on each
   * other and neither ever finishes. Observed with a paged vector at 96 CUDA
   * blocks: every compute worker parked in pthread_rwlock_wrlock under
   * cuStreamCreate/cuStreamDestroy, the runtime reporting "ALL compute workers
   * stalled", and the kernel never returning. 64 blocks happened to stay under
   * the contention threshold, which is why this looked like a block-count bug.
   *
   * A borrowed stream is owned EXCLUSIVELY by its task, so StreamQuery on it
   * still means "my copies are done" and not "the pool is idle" -- which is
   * why this is a pool rather than one shared stream per thread. Pooled
   * streams are never destroyed; they are process-lifetime objects.
   */
  /**
   * Streams RESERVED for callers that cannot yield, one per thread.
   *
   * The shared pool is safe only for callers that release the worker while
   * they wait. Path A (bdev writes) is a coroutine: it CO_AWAITs while HOLDING
   * a stream, so a single worker can carry many suspended stream-holding
   * tasks. Path B (IpcGpu2Cpu::SendOut -> DeviceAwareMemcpy) runs directly
   * under Worker::ExecTask and cannot yield, so its retry loop BLOCKS the
   * worker thread.
   *
   * Those two combine into a resource inversion: path A's suspended
   * coroutines can only be resumed BY A WORKER, and path B blocks a worker
   * waiting for the stream that only that resumption would release. Measured
   * at a live wedge: outstanding == warmed == 64, three workers spinning in
   * BorrowStream, one worker holding 45 undrained tasks, live=0, and the
   * faulting kernel stuck in cuCtxSynchronize forever.
   *
   * A reserved stream removes path B from the pool entirely, so it can never
   * be starved. Exclusive per-thread ownership is sound because
   * DeviceAwareMemcpy is synchronous end to end (launch, PollSync, done) and
   * never yields, so a thread cannot re-enter it and cannot overlap two uses
   * of its own stream.
   */
  static std::vector<void *> &ReservedStreams() {
    static std::vector<void *> v;
    return v;
  }
  static std::atomic<int> &ReservedNext() {
    static std::atomic<int> n{0};
    return n;
  }

  /**
   * This thread's reserved stream, or nullptr when the reserve is used up
   * (then the caller falls back to the shared pool).
   *
   * Assignment is sticky per thread and happens on first use. Streams are
   * NEVER created here: creating one takes the CUDA context write lock, which
   * blocks while a kernel is resident -- the very deadlock the pool exists to
   * avoid. The reserve is pre-created in WarmStreamPool.
   */
  static void *ThreadReservedStream() {
    static thread_local void *mine = nullptr;
    static thread_local bool tried = false;
    if (!tried) {
      tried = true;
      std::lock_guard<std::mutex> lock(StreamPoolMutex());
      const int idx = ReservedNext().fetch_add(1, std::memory_order_relaxed);
      if (idx >= 0 && idx < static_cast<int>(ReservedStreams().size())) {
        mine = ReservedStreams()[idx];
      }
    }
    return mine;
  }

  /** Lock-taking wrapper for PoolSizesLocked. */
  static std::string PoolSizes() {
    std::lock_guard<std::mutex> lock(StreamPoolMutex());
    return PoolSizesLocked();
  }

  static void *BorrowStream() {
    const int dev = CurrentDevice();
    std::lock_guard<std::mutex> lock(StreamPoolMutex());
    auto &free_list = StreamPool()[dev];
    if (!free_list.empty()) {
      void *s = free_list.back();
      free_list.pop_back();
      StreamBorrows().fetch_add(1, std::memory_order_relaxed);
      return s;
    }
    // Exhausted. Do NOT create one here: creating a stream while a kernel is
    // resident blocks for as long as that kernel runs, and the kernels this
    // serves are demand-paged ones that spin until THIS I/O completes. Two
    // shapes of that were measured and both deadlocked -- creating per cold
    // task (hung at 128 blocks) and creating a batch under this mutex (hung at
    // 96, because the holder blocked inside libcuda with everyone queued
    // behind it). The caller yields and retries instead; a stream comes back
    // as soon as any in-flight copy finishes.
    if (!PoolWarmed()[dev]) {
      // Never warmed (no GPU init in this process): bootstrap exactly one so
      // an un-warmed process still makes progress.
      PoolWarmed()[dev] = true;
      StreamBorrows().fetch_add(1, std::memory_order_relaxed);
      StreamWarmed().fetch_add(1, std::memory_order_relaxed);
      return CreateStream();
    }
    return nullptr;
  }

  /**
   * Pre-create the stream pool for the current device.
   *
   * MUST be called during initialization, before any long-running kernel can
   * be resident -- that is the entire point. See BorrowStream.
   */
  static void WarmStreamPool(int count) {
    const int dev = CurrentDevice();
    std::lock_guard<std::mutex> lock(StreamPoolMutex());
    auto &free_list = StreamPool()[dev];
    for (int i = 0; i < count; ++i) {
      free_list.push_back(CreateStream());
    }
    StreamWarmed().fetch_add(count, std::memory_order_relaxed);
    // Reserve for non-yieldable callers. Sized well above the worker count so
    // every thread that can reach DeviceAwareMemcpy gets one; these are never
    // handed to the shared pool, so no coroutine can hold them.
    int reserved = 128;
    if (const char *e = std::getenv("CLIO_GPU_RESERVED_STREAMS")) {
      const int v = std::atoi(e);
      if (v >= 0) reserved = v;
    }
    for (int i = 0; i < reserved; ++i) {
      ReservedStreams().push_back(CreateStream());
    }
    PoolWarmed()[dev] = true;
  }

  /** Give a borrowed stream back. The stream is NOT destroyed. */
  static void ReturnStream(void *stream) {
    if (stream == nullptr) return;
    std::lock_guard<std::mutex> lock(StreamPoolMutex());
    StreamReturns().fetch_add(1, std::memory_order_relaxed);
    StreamPool()[CurrentDevice()].push_back(stream);
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

  /** Non-fatal host registration: pin [ptr, ptr+size) so async DMA against
   *  it stays asynchronous (an unpinned destination silently degrades
   *  MemcpyAsync to a driver-staged SYNCHRONOUS copy at roughly a fifth of
   *  the engine bandwidth). Unlike RegisterHostMemory this reports failure
   *  instead of dying, so callers can pin opportunistically -- e.g. an SHM
   *  tier on a host that may have no GPU driver at all.
   *  @return true if the range is now (or was already) pinned. */
  static bool TryRegisterHostMemory(void *ptr, size_t size) {
#if CTP_ENABLE_ROCM
    hipError_t e = hipHostRegister(ptr, size, hipHostRegisterPortable);
    if (e != hipSuccess) {
      (void) hipGetLastError();   // clear the sticky error
    }
    return e == hipSuccess || e == hipErrorHostMemoryAlreadyRegistered;
#elif CTP_ENABLE_CUDA
    cudaError_t e = cudaHostRegister(ptr, size, cudaHostRegisterPortable);
    if (e != cudaSuccess) {
      (void) cudaGetLastError();   // clear the sticky error
    }
    return e == cudaSuccess || e == cudaErrorHostMemoryAlreadyRegistered;
#else
    (void) ptr;
    (void) size;
    return false;
#endif
  }

  /** Non-fatal partner of TryRegisterHostMemory; `ptr` must be a base
   *  pointer that was passed to it. */
  static void TryUnregisterHostMemory(void *ptr) {
#if CTP_ENABLE_ROCM
    if (hipHostUnregister(ptr) != hipSuccess) {
      (void) hipGetLastError();
    }
#elif CTP_ENABLE_CUDA
    if (cudaHostUnregister(ptr) != cudaSuccess) {
      (void) cudaGetLastError();
    }
#else
    (void) ptr;
#endif
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
    // A failed attribute query means there is no usable GPU (no driver / no
    // device) or the pointer is an unregistered host allocation — either way it
    // is NOT a device pointer. Do not treat this as fatal: a GPU-enabled build
    // must still run entirely on the CPU path on a host without a GPU driver.
    // Clear the sticky error so a later real GPU call is not misattributed.
    hipPointerAttribute_t attributes{};
    if (hipPointerGetAttributes(&attributes, (void *)ptr) != hipSuccess) {
      (void)hipGetLastError();
      return false;
    }
#if defined(HIP_VERSION) && HIP_VERSION >= 60000000
    return attributes.type == hipMemoryTypeDevice;
#else
    return attributes.memoryType == hipMemoryTypeDevice;
#endif
#elif CTP_ENABLE_CUDA
    // See the ROCm note above: a failed query (e.g. CUDA error 35, "driver
    // version is insufficient", on a host with no GPU driver) means this is a
    // host pointer, not a fatal condition. Clear the sticky error and fall back
    // to the CPU path.
    cudaPointerAttributes attributes{};
    if (cudaPointerGetAttributes(&attributes, (void *)ptr) != cudaSuccess) {
      (void)cudaGetLastError();
      return false;
    }
    return attributes.type == cudaMemoryTypeDevice;
#elif CTP_ENABLE_SYCL
    // On a host with no GPU no USM *device* allocation can exist, so any pointer
    // is host memory. Short-circuit before touching SyclQueue(): constructing a
    // sycl::queue throws when no SYCL device is present, and an unhandled throw
    // inside a coroutine task aborts the whole runtime server (mirrors the
    // CUDA/ROCm "failed query -> host pointer" degradation above).
    if (!HasSyclGpuDevice()) return false;
    auto kind = sycl::get_pointer_type(static_cast<const void *>(ptr),
                                        SyclQueue().get_context());
    return kind == sycl::usm::alloc::device;
#else
    return false;
#endif
  }

  /**
   * Like IsDevicePointer, but TRUE for managed (UVM) memory too.
   *
   * Managed memory is addressable by both the host and every context on the
   * device, so for the question "may a GPU touch this?" it is a yes -- while
   * cudaPointerGetAttributes reports it as cudaMemoryTypeManaged, not
   * ...TypeDevice, so the stricter check says no. Use THIS one to decide
   * whether to take a GPU path; use IsDevicePointer only when the memory must
   * be device-resident specifically.
   *
   * gpu_vector's page cache is managed precisely so a codec running in another
   * CUDA context can write into a faulting page. With the strict check those
   * pages read as host memory and every GPU path silently declines.
   */
  template <typename T>
  static bool IsDeviceAccessiblePointer(T *ptr) {
    if (ptr == nullptr) return false;
#if CTP_ENABLE_ROCM
    hipPointerAttribute_t a{};
    if (hipPointerGetAttributes(&a, (void *)ptr) != hipSuccess) {
      (void)hipGetLastError();
      return false;
    }
#if defined(HIP_VERSION) && HIP_VERSION >= 60000000
    return a.type == hipMemoryTypeDevice || a.type == hipMemoryTypeManaged;
#else
    return a.memoryType == hipMemoryTypeDevice ||
           a.memoryType == hipMemoryTypeManaged;
#endif
#elif CTP_ENABLE_CUDA
    cudaPointerAttributes a{};
    if (cudaPointerGetAttributes(&a, (void *)ptr) != cudaSuccess) {
      (void)cudaGetLastError();
      return false;
    }
    return a.type == cudaMemoryTypeDevice || a.type == cudaMemoryTypeManaged;
#else
    (void)ptr;
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
    {
      cudaError_t _rc = cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault,
                                        static_cast<cudaStream_t>(stream));
      if (_rc != cudaSuccess) {
        cudaPointerAttributes _ad{}, _as{};
        cudaError_t _rd = cudaPointerGetAttributes(&_ad, dst);
        cudaError_t _rs = cudaPointerGetAttributes(&_as, src);
        (void)cudaGetLastError();
        fprintf(stderr,
                "[memcpyasync-fail] dst=%p (rc=%d type=%d) src=%p (rc=%d "
                "type=%d) size=%zu stream=%p\n",
                dst, (int)_rd, (int)_ad.type, (const void *)src, (int)_rs,
                (int)_as.type, size, stream);
      }
      CUDA_ERROR_CHECK(_rc);
    }
#endif
  }

  /** Async memset on a stream. */
  static void MemsetAsync(void *dst, int value, size_t size, void *stream) {
#if CTP_ENABLE_ROCM
    HIP_ERROR_CHECK(hipMemsetAsync(dst, value, size,
                                   static_cast<hipStream_t>(stream)));
#endif
#if CTP_ENABLE_CUDA
    CUDA_ERROR_CHECK(cudaMemsetAsync(dst, value, size,
                                     static_cast<cudaStream_t>(stream)));
#endif
#if CTP_ENABLE_SYCL
    if (stream) {
      static_cast<sycl::queue *>(stream)->memset(dst, value, size);
    }
#endif
  }

  /** Non-blocking poll of a stream. Returns true once every operation
   *  submitted to `stream` has completed, false while work is still in flight.
   *  The "not ready" status is cleared so this is a pure query and never trips
   *  a later error check. On a non-GPU build it returns true; under SYCL there
   *  is no cheap non-blocking query, so it waits and then returns true. */
  static bool StreamQuery(void *stream) {
#if CTP_ENABLE_ROCM
    hipError_t rc = hipStreamQuery(static_cast<hipStream_t>(stream));
    if (rc == hipSuccess) return true;
    if (rc == hipErrorNotReady) {
      (void)hipGetLastError();
      return false;
    }
    HIP_ERROR_CHECK(rc);
    return true;
#elif CTP_ENABLE_CUDA
    cudaError_t rc = cudaStreamQuery(static_cast<cudaStream_t>(stream));
    if (rc == cudaSuccess) return true;
    if (rc == cudaErrorNotReady) {
      (void)cudaGetLastError();
      return false;
    }
    CUDA_ERROR_CHECK(rc);
    return true;
#elif CTP_ENABLE_SYCL
    if (stream) {
      static_cast<sycl::queue *>(stream)->wait_and_throw();
    }
    return true;
#else
    (void)stream;
    return true;
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
  /**
   * True if at least one SYCL GPU device is visible. Cached on first use:
   * enumerating platforms on every per-op IsDevicePointer call would be far too
   * slow. This is the guard that keeps a GPU-enabled build running on a host
   * with no GPU — see SyclQueue()/IsDevicePointer() for why constructing a queue
   * unconditionally is fatal there.
   */
  static bool HasSyclGpuDevice() {
    static const bool has = [] {
      try {
        return !sycl::device::get_devices(sycl::info::device_type::gpu).empty();
      } catch (const sycl::exception &) {
        return false;
      }
    }();
    return has;
  }

  /**
   * Shared SYCL queue backing the GPU host-side helpers (Memcpy, Free,
   * Synchronize) and the device-pointer probe. Only ever call this when
   * HasSyclGpuDevice() is true: sycl::gpu_selector_v (and even the default
   * selector on this environment) throws "No device of requested type
   * available" when no SYCL device is present, and an unhandled throw inside a
   * coroutine task aborts the whole runtime server.
   */
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
    if (rc != cudaSuccess) {
      // Any failure — older CUDA's cudaErrorInvalidValue for unregistered host
      // memory, OR a host with no usable GPU driver (e.g. error 35, "driver
      // version is insufficient") — means this is not a confirmed device
      // pointer. Treat it as host so the copy stays on std::memcpy instead of
      // FATAL-ing the device path below. Clear the sticky error so it does not
      // trip the next CUDA call. (With a working driver the query succeeds, so a
      // real device pointer is never misclassified.)
      (void)cudaGetLastError();
      return true;
    }
    return a.type == cudaMemoryTypeHost ||
           a.type == cudaMemoryTypeUnregistered;
  };
  if (is_host_kind(dst) && is_host_kind(src)) {
    std::memcpy(dst, src, n);
    return;
  }
  // NEVER create a stream here. Stream creation takes the context write
  // lock, which BLOCKS while any kernel is resident — and this function
  // runs on the fault-service path while a faulting kernel spins for the
  // very copy being made. A worker thread whose FIRST device copy landed
  // mid-decode froze in cudaStreamCreateWithFlags, killing fault service
  // and wedging the process (intermittent by thread-to-task lottery;
  // copy engines measured healthy from a separate process throughout).
  // Borrow from the pool pre-warmed at init instead.
  // Use this thread's RESERVED stream when it has one. This path cannot
  // yield, so taking from the shared pool lets suspended coroutines starve it
  // and deadlock the runtime (see ThreadReservedStream). With a reserved
  // stream there is nothing to wait for and the spin below is unreachable.
  void *ps = GpuApi::ThreadReservedStream();
  const bool reserved = (ps != nullptr);
  if (ps == nullptr) ps = GpuApi::BorrowStream();
  // THIS SPIN IS UNBOUNDED AND HAS BEEN CAUGHT WEDGING THE RUNTIME. Captured
  // live: two worker threads parked here in sched_yield under
  // Worker::ExecTask -> IpcGpu2Cpu::SendOut, while 8 tasks sat queued in the
  // lanes of two OTHER idle workers and the GPU kernel waited in
  // cuCtxSynchronize for exactly those tasks. Workers consumed by this loop
  // stop draining their lanes, so the fault service the resident kernel is
  // waiting on never runs -- a circular wait no watchdog can see (the
  // scheduler's stall detector only covers workers INSIDE a long ExecTask,
  // and these are "executing" by its definition).
  //
  // The pool is 64 streams against ~9 workers, so ordinary concurrency cannot
  // empty it; reaching nullptr at all means streams are outstanding far longer
  // than expected or are not coming back. Report it instead of spinning
  // silently -- a silent unbounded spin is why this took a live gdb attach to
  // find.
  // BOUNDED, WITH A GUARANTEED WAY OUT. This is the fallback for threads that
  // did not get a reserved stream (more callers than the reserve). It is the
  // path that used to spin forever: captured live with worker threads parked
  // here in sched_yield under Worker::ExecTask -> IpcGpu2Cpu::SendOut, lanes
  // undrained, live=0, and the process wedged. A blocking wait on this path
  // can never be safe -- the streams it waits for are released by coroutines
  // that only a worker can resume, and this IS a worker.
  //
  // So: wait briefly, then fall back to the DEFAULT stream rather than keep
  // waiting. The default stream always exists, so there is no creation (it is
  // cuStreamCreate, not stream use, that takes the context write lock and
  // deadlocks against a resident kernel). Pool streams are created with
  // cudaStreamNonBlocking, so they do not implicitly synchronise against it.
  // Serialising a few service copies is strictly better than a hang.
  if (ps == nullptr) {
    for (int spins = 0; spins < 20000 && ps == nullptr; ++spins) {
      std::this_thread::yield();
      ps = GpuApi::BorrowStream();
    }
    if (ps == nullptr) {
      HLOG(kWarning,
           "[stream-pool] exhausted in DeviceAwareMemcpy(n={}); warmed={} "
           "outstanding={} buckets=[{}]. Falling back to the default stream "
           "so this worker keeps making progress instead of blocking on a "
           "stream only it could release.",
           n, GpuApi::StreamWarmed().load(),
           GpuApi::StreamBorrows().load() - GpuApi::StreamReturns().load(),
           GpuApi::PoolSizes());
    }
  }
  const bool pooled = (!reserved && ps != nullptr);
  cudaStream_t s = static_cast<cudaStream_t>(ps);
  // COPY VIA KERNEL WHEN LEGAL, ENGINE OTHERWISE. Engine copies that read
  // device memory are channel-ordered and stall behind a resident faulting
  // kernel — captured live: the gpu2cpu drain's task-POD D2H froze in
  // exactly this call, wedging fault service. A copy KERNEL is
  // SM-scheduled and immune. But a kernel can only touch device-accessible
  // memory: device, managed, or PINNED host — a pageable pointer (ingest
  // buffers) must stay on the engine path (those copies never run under a
  // resident kernel; converting them crashed ingest outright).
  auto dev_ok = [](const void *p) {
    cudaPointerAttributes a{};
    if (cudaPointerGetAttributes(&a, p) != cudaSuccess) {
      (void)cudaGetLastError();
      return false;
    }
    return a.type == cudaMemoryTypeDevice ||
           a.type == cudaMemoryTypeManaged ||
           (a.type == cudaMemoryTypeHost && a.devicePointer != nullptr);
  };
  if (dev_ok(dst) && dev_ok(src)) {
    ctp_copy_kernel_launch(static_cast<char *>(dst),
                           static_cast<const char *>(src), n, ps);
  } else {
    CUDA_ERROR_CHECK(cudaMemcpyAsync(dst, src, n, cudaMemcpyDefault, s));
  }
  GpuApi::PollSync(ps);   // never block in driver sync on a service path
  // A reserved stream stays with its thread; the default-stream fallback was
  // never borrowed. Only genuinely pooled streams go back.
  if (pooled) GpuApi::ReturnStream(ps);
#elif CTP_ENABLE_ROCM
  auto is_host_kind = [](const void *p) {
    hipPointerAttribute_t a{};
    hipError_t rc = hipPointerGetAttributes(&a, p);
    if (rc != hipSuccess) {
      // See the CUDA branch: any query failure (unregistered host pointer, or a
      // host with no usable GPU driver) means "not a confirmed device pointer".
      // Treat as host and stay on std::memcpy instead of FATAL-ing the device
      // path. Clear the sticky error.
      (void)hipGetLastError();
      return true;
    }
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
/** True if a GPU may touch `ptr`: device OR managed memory. Prefer this over
 *  IsDevicePointer when deciding whether to take a GPU path. */
inline bool IsDeviceAccessible(const void *ptr) {
  return GpuApi::IsDeviceAccessiblePointer(const_cast<void *>(ptr));
}

inline bool IsDevicePointer(const void *ptr) {
  return GpuApi::IsDevicePointer(const_cast<void *>(ptr));
}

}  // namespace ctp

#endif