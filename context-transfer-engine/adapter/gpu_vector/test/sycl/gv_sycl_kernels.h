/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * Launchers for the gpu_vector SYCL kernels.
 *
 * WHY THE TRANSLATION UNIT IS SPLIT AT ALL
 *
 * This is the one structural thing SYCL forces that CUDA does not.
 *
 * In CUDA the kernels are `__global__` functions at namespace scope and the
 * host test is a TEST_CASE beneath them, so the host-only parts can simply
 * be `#if !CTP_IS_DEVICE_PASS`-ed out of the device pass -- which they must
 * be, because gpu_vector.h's host `Vector` class calls host-only client
 * methods by non-dependent name and does not parse there at all.
 *
 * A SYCL kernel is a LAMBDA SUBMITTED TO A QUEUE, so it lives INSIDE the
 * host function that launches it. There is no preprocessor line that keeps
 * the kernel and drops the host code around it -- they are the same
 * statement. And DPC++ fully type-checks the whole translation unit in its
 * device pass, so a TU containing both fails on the host half.
 *
 * The split is the answer: this header's functions are the seam. The .cc
 * that defines them is compiled with -fsycl and includes only device-side
 * headers; the test that calls them is ordinary C++ and includes the host
 * ones. Everything either side touches -- DeviceVector, the coroutines, the
 * lane stack, the relaunch driver -- is the same header the CUDA build uses.
 */
#ifndef CLIO_CTE_GPU_VECTOR_TEST_SYCL_KERNELS_H_
#define CLIO_CTE_GPU_VECTOR_TEST_SYCL_KERNELS_H_

#include <clio_cte/gpu_vector/device_vector.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_ctp/util/sycl_cuda_compat.h>
#include <clio_runtime/gpu/yield_stack.h>
#include <clio_runtime/gpu/yieldable.h>
#include <clio_runtime/types.h>

namespace clio::cte::gpu_vector::sycl_test {

using DevU32 = ::clio::cte::gpu_vector::DeviceVector<clio::run::u32>;
using View = ::clio::run::gpu::YieldableView<>;
using StackView = ::clio::run::gpu::YieldStackView;

/**
 * Allocate and publish the per-block IpcManager array, sized for the widest
 * grid any launch below will use. Call once, after the runtime is up.
 *
 * Lives on this side of the seam because it submits a SYCL kernel, and the
 * host test is not compiled with -fsycl.
 */
void InitBlockIpc(clio::run::u32 max_blocks,
                  const clio::run::IpcManagerGpuInfo &gpu_info);

/** Write v[i] = i*7+1 for i in [0, n), flushing each page as it is written. */
void LaunchFill(dim3 grid, dim3 block, DevU32 v, clio::run::u64 n, View vw,
                StackView sv);

/** Read [0, n) back and count elements that are not i*7+1. */
void LaunchCheck(dim3 grid, dim3 block, DevU32 v, clio::run::u64 n,
                 unsigned long long *bad, View vw, StackView sv);

/** As LaunchFill, but block b owns the slice [b*per, (b+1)*per). */
void LaunchMultiFill(dim3 grid, dim3 block, DevU32 v, clio::run::u64 per,
                     View vw, StackView sv);

/** As LaunchCheck, over the per-block slices LaunchMultiFill wrote. */
void LaunchMultiCheck(dim3 grid, dim3 block, DevU32 v, clio::run::u64 per,
                      unsigned long long *bad, View vw, StackView sv);

}  // namespace clio::cte::gpu_vector::sycl_test

#endif  // CLIO_CTE_GPU_VECTOR_TEST_SYCL_KERNELS_H_
