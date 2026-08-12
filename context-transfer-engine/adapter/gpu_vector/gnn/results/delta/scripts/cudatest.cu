#include <cstdio>
__global__ void k(float*a){a[threadIdx.x]=threadIdx.x*2.0f;}
int main(){
  int n=0; cudaError_t e=cudaGetDeviceCount(&n);
  printf("cudaGetDeviceCount -> %s, n=%d\n", cudaGetErrorString(e), n);
  if(e!=cudaSuccess||n==0) return 1;
  cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
  printf("device0: %s  sm_%d%d  %.1f GB\n",p.name,p.major,p.minor,p.totalGlobalMem/1e9);
  float*d; if(cudaMalloc(&d,128*sizeof(float))!=cudaSuccess){printf("cudaMalloc FAILED\n");return 1;}
  k<<<1,32>>>(d); cudaDeviceSynchronize();
  float h[32]; cudaMemcpy(h,d,32*sizeof(float),cudaMemcpyDeviceToHost);
  printf("kernel result h[5]=%.1f (expect 10.0) -> %s\n",h[5], h[5]==10.0f?"CUDA OK":"CUDA BROKEN");
  return h[5]==10.0f?0:1;
}
