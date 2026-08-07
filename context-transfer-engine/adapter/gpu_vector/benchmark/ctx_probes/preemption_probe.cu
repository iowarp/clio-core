// Probe: does a kernel launched in a SEPARATE CUDA context make progress while a
// kernel in the PRIMARY context spin-waits forever? This is the exact mechanism
// the compressed-vector on-device fault would rely on: the compressor's
// decompress (context B) must run while the app's fault kernel (context A) spins
// on-device for it. If context B's kernel completes -> compute preemption across
// contexts works -> the separate-context fix is viable. If it hangs -> it isn't.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA err %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));fflush(stdout);} } while(0)
#define DK(x) do { CUresult r=(x); if(r!=CUDA_SUCCESS){const char*s;cuGetErrorString(r,&s);printf("DRV err %s:%d %s\n",__FILE__,__LINE__,s);fflush(stdout);} } while(0)

__global__ void spinKernel(volatile int* flag) {
  // Spin on-device until the host sets *flag (exactly like gpu::Future::Wait).
  while (*flag == 0) {
#if __CUDA_ARCH__ >= 700
    __nanosleep(2000);
#endif
  }
}

__global__ void workKernel(int* out, int n, int iters) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    unsigned x = (unsigned)i + 1u;
    for (int k = 0; k < iters; ++k) x = x * 1103515245u + 12345u;
    out[i] = (int)x;
  }
}

int main() {
  CK(cudaFree(0));                 // init primary context (runtime)
  CUcontext primary; DK(cuCtxGetCurrent(&primary));

  // Managed memory is UVA / process-wide, so it is reachable from BOTH contexts
  // and from the host (the real integration would instead need the vector's HBM
  // pages to be cross-context-accessible -- tested separately below).
  int* flag; CK(cudaMallocManaged(&flag, sizeof(int)));  *flag = 0;
  int* out;  CK(cudaMallocManaged(&out, 256 * sizeof(int)));
  for (int i = 0; i < 256; ++i) out[i] = 0;

  // 1) Launch the spinner in the PRIMARY context; do NOT wait for it.
  cudaStream_t s; CK(cudaStreamCreate(&s));
  spinKernel<<<1, 32, 0, s>>>(flag);
  printf("[probe] spinner launched in primary context (spinning)\n"); fflush(stdout);

  // 2) Create a SEPARATE context and run a worker kernel in it.
  CUdevice dev; DK(cuDeviceGet(&dev, 0));
  CUcontext ctxB; DK(cuCtxCreate(&ctxB, 0, dev));   // becomes current
  printf("[probe] created separate context; launching worker in it...\n"); fflush(stdout);

  clock_t t0 = clock();
  workKernel<<<1, 256>>>(out, 256, 200000);
  cudaError_t sync = cudaDeviceSynchronize();       // waits for ctxB's work only
  double secs = (double)(clock() - t0) / CLOCKS_PER_SEC;

  bool ran = (out[100] != 0);
  printf("[probe] worker sync rc=%d (%s)  time=%.2fs  out[100]=%d  => %s\n",
         (int)sync, cudaGetErrorString(sync), secs, out[100],
         (ran && sync == cudaSuccess)
             ? "PREEMPTION WORKS (context B ran while A spun)"
             : "context B did NOT complete");

  // 3) Release the spinner and clean up.
  DK(cuCtxSetCurrent(primary));
  *flag = 1;
  CK(cudaDeviceSynchronize());
  printf("[probe] spinner released; done.\n");
  return 0;
}
