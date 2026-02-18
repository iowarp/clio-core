/*
 * Probe the maximum virtual address reservation size supported by the GPU.
 * Uses binary search over cuMemAddressReserve to find the largest VA range.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>

int main() {
  cudaFree(0);  // Ensure CUDA context

  CUdevice device;
  cuInit(0);
  cuDeviceGet(&device, 0);

  // Query allocation granularity
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = 0;

  size_t granularity = 0;
  cuMemGetAllocationGranularity(&granularity, &prop,
                                 CU_MEM_ALLOC_GRANULARITY_MINIMUM);

  printf("GPU allocation granularity: %zu bytes (%.2f MB)\n",
         granularity, (double)granularity / (1024.0 * 1024));

  // Binary search: [lo, hi] in units of granularity
  size_t lo = 1;                                        // 1 page
  size_t hi = (16ULL * 1024 * 1024 * 1024 * 1024) / granularity;  // 16 TB

  size_t best = 0;

  printf("\nProbing VA range [%zu pages .. %zu pages] (granularity=%zu)\n",
         lo, hi, granularity);

  while (lo <= hi) {
    size_t mid = lo + (hi - lo) / 2;
    size_t try_size = mid * granularity;

    CUdeviceptr ptr = 0;
    CUresult res = cuMemAddressReserve(&ptr, try_size, granularity, 0, 0);

    if (res == CUDA_SUCCESS) {
      best = mid;
      cuMemAddressFree(ptr, try_size);
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }

  size_t best_bytes = best * granularity;
  printf("\n=== Maximum VA Reservation ===\n");
  printf("  %zu bytes\n", best_bytes);
  printf("  %.2f MB\n", (double)best_bytes / (1024.0 * 1024));
  printf("  %.2f GB\n", (double)best_bytes / (1024.0 * 1024 * 1024));
  printf("  %.4f TB\n", (double)best_bytes / (1024.0 * 1024 * 1024 * 1024));
  printf("  (%zu pages x %zu bytes)\n", best, granularity);

  return 0;
}
