// cdp_hostapi_fail.cu
// ---------------------------------------------------------------------------
// PART 1 of the CDP proof: demonstrates WHY cuSZp / nvcomp cannot be invoked
// from within a CUDA kernel via Dynamic Parallelism.
//
// Both libraries expose only HOST entry points. cuSZp v3's real signature is:
//   void cuSZp_compress(float* d_in, unsigned char* d_cmp, size_t n,
//                       size_t* cmp_size, float eb, ...);   // __host__
// nvcomp's is nvcompBatched*CompressAsync(...), also __host__.
//
// Calling a __host__ function from __global__ device code is rejected by nvcc.
// `cuSZp_compress_like` below mirrors the cuSZp API shape with an explicit
// __host__ qualifier. Compiling this file is EXPECTED TO FAIL with:
//   "calling a __host__ function from a __global__ function is not allowed"
// which is the concrete evidence that "cuSZp via CDP" is not possible.
// ---------------------------------------------------------------------------
#include <cstddef>

// Stand-in for the real cuSZp_compress / nvcompBatched*CompressAsync: a
// host-only library entry point (this is exactly what the real headers declare).
__host__ void cuSZp_compress_like(float* d_in, unsigned char* d_cmp, size_t n,
                                  size_t* cmp_size, float eb);

// A Gray-Scott parent kernel that tries to compress its output in-kernel.
__global__ void GsThenCompressParent(float* v2, unsigned char* comp, size_t n,
                                     size_t* csz, float eb) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    // >>> This line is the whole point: invoke the compressor from device code.
    cuSZp_compress_like(v2, comp, n, csz, eb);
  }
}

int main() { return 0; }
