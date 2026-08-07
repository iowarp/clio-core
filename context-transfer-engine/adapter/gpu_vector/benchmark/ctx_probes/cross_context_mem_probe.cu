// Probe 2: can a kernel / cudaMemcpy in a SEPARATE context write into a buffer
// that was cudaMalloc'd in the PRIMARY context? The compressed-vector decompress
// (context B) must write the decompressed page into the vector's HBM slot, which
// was allocated by the app (context A) via plain cudaMalloc. If cross-context
// access works -> zero-copy integration is possible as-is. If not -> the pages
// would need to be managed / shared memory.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){printf("  cudaerr %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));fflush(stdout);} } while(0)
#define DK(x) do { CUresult r=(x); if(r!=CUDA_SUCCESS){const char*s;cuGetErrorString(r,&s);printf("  drverr %s:%d %s\n",__FILE__,__LINE__,s);fflush(stdout);} } while(0)

__global__ void fill(int* p, int n, int v) {
  int i = blockIdx.x*blockDim.x+threadIdx.x; if (i<n) p[i]=v;
}

int main() {
  CK(cudaFree(0));
  CUcontext primary; DK(cuCtxGetCurrent(&primary));
  CUdevice dev; DK(cuDeviceGet(&dev,0));

  const int N = 256;
  // Allocate a PLAIN cudaMalloc device buffer in the PRIMARY context.
  int* dA = nullptr; CK(cudaMalloc(&dA, N*sizeof(int)));
  fill<<<1,N>>>(dA, N, 111); CK(cudaDeviceSynchronize());   // primary writes 111

  // Switch to a SEPARATE context and try to write dA from there.
  CUcontext ctxB; DK(cuCtxCreate(&ctxB, 0, dev));           // becomes current
  printf("[probe2] separate context created; attempting cross-context writes to a primary-context cudaMalloc ptr...\n");
  fflush(stdout);

  // (a) kernel in ctxB writing dA
  fill<<<1,N>>>(dA, N, 222);
  cudaError_t ek = cudaDeviceSynchronize();
  printf("[probe2] (a) kernel write from ctxB: rc=%d (%s)\n", (int)ek, cudaGetErrorString(ek));

  // (b) cudaMemcpy (D2H) from ctxB reading dA
  int hostbuf[N]; for(int i=0;i<N;i++) hostbuf[i]=-1;
  cudaError_t em = cudaMemcpy(hostbuf, dA, N*sizeof(int), cudaMemcpyDeviceToHost);
  printf("[probe2] (b) D2H copy from ctxB:    rc=%d (%s)  hostbuf[100]=%d\n",
         (int)em, cudaGetErrorString(em), hostbuf[100]);

  // Verdict: if both succeeded and hostbuf reflects 222, cross-context access works.
  bool ok = (ek==cudaSuccess && em==cudaSuccess && hostbuf[100]==222);
  printf("[probe2] => cross-context access to a primary-context cudaMalloc buffer: %s\n",
         ok ? "WORKS" : "DOES NOT WORK (pages would need managed/shared memory)");

  DK(cuCtxSetCurrent(primary));
  return 0;
}
