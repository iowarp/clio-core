// How much does one nvcompBatchedLZ4DecompressAsync launch cost, and how does
// batching amortize it? Decides the fault-path architecture.
#include <cuda_runtime.h>
#include <nvcomp/lz4.h>
#include <cstdio>
#include <cstring>
#include <vector>
int main() {
  const size_t kChunk = 64 * 1024, kN = 64;
  cudaStream_t st; cudaStreamCreate(&st);
  // compressible payload
  std::vector<char> h(kChunk);
  for (size_t i = 0; i < kChunk; ++i) h[i] = (char)((i / 64) % 17);
  char *d_in; cudaMalloc(&d_in, kChunk); cudaMemcpy(d_in, h.data(), kChunk, cudaMemcpyHostToDevice);
  // compress one chunk
  const void *cp[1] = {d_in}; size_t cs[1] = {kChunk};
  const void **d_cp; size_t *d_cs; cudaMalloc(&d_cp, 8); cudaMalloc(&d_cs, 8);
  cudaMemcpy(d_cp, cp, 8, cudaMemcpyHostToDevice); cudaMemcpy(d_cs, cs, 8, cudaMemcpyHostToDevice);
  size_t tmpb=0; nvcompBatchedLZ4CompressGetTempSizeAsync(1, kChunk, nvcompBatchedLZ4CompressDefaultOpts, &tmpb, kChunk);
  void *tmp; cudaMalloc(&tmp, tmpb ? tmpb : 8);
  size_t maxout=0; nvcompBatchedLZ4CompressGetMaxOutputChunkSize(kChunk, nvcompBatchedLZ4CompressDefaultOpts, &maxout);
  char *d_out; cudaMalloc(&d_out, maxout);
  void *op[1] = {d_out}; void **d_op; cudaMalloc(&d_op, 8); cudaMemcpy(d_op, op, 8, cudaMemcpyHostToDevice);
  size_t *d_os; cudaMalloc(&d_os, 8);
  nvcompStatus_t *d_cst; cudaMalloc(&d_cst, 4);
  if (nvcompBatchedLZ4CompressAsync(d_cp, d_cs, kChunk, 1, tmp, tmpb, d_op, d_os,
                                    nvcompBatchedLZ4CompressDefaultOpts, d_cst, st) != nvcompSuccess) { printf("comp fail\n"); return 1; }
  cudaStreamSynchronize(st);
  size_t csz; cudaMemcpy(&csz, d_os, 8, cudaMemcpyDeviceToHost);
  printf("chunk %zu -> %zu\n", kChunk, csz);
  // build N identical compressed chunks + N dsts
  std::vector<const void*> hcp(kN, d_out); std::vector<size_t> hcs(kN, csz), hob(kN, kChunk);
  char *d_dst; cudaMalloc(&d_dst, kChunk * kN);
  std::vector<void*> hop(kN); for (size_t i=0;i<kN;++i) hop[i]=d_dst+i*kChunk;
  const void **dcp; size_t *dcs,*dob,*dos; void **dop; nvcompStatus_t *dstt;
  cudaMalloc(&dcp,kN*8); cudaMalloc(&dcs,kN*8); cudaMalloc(&dob,kN*8); cudaMalloc(&dop,kN*8); cudaMalloc(&dos,kN*8); cudaMalloc(&dstt,kN*4);
  cudaMemcpy(dcp,hcp.data(),kN*8,cudaMemcpyHostToDevice); cudaMemcpy(dcs,hcs.data(),kN*8,cudaMemcpyHostToDevice); cudaMemcpy(dob,hob.data(),kN*8,cudaMemcpyHostToDevice); cudaMemcpy(dop,hop.data(),kN*8,cudaMemcpyHostToDevice);
  size_t dtb=0;
  nvcompBatchedLZ4DecompressGetTempSizeSync(dcp,dcs,kN,kChunk,&dtb,kChunk*kN,nvcompBatchedLZ4DecompressDefaultOpts,dstt,st);
  void *dtmp; cudaMalloc(&dtmp, dtb?dtb:8);
  cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  for (int bs : {1, 4, 16, 64}) {
    // warm
    nvcompBatchedLZ4DecompressAsync(dcp,dcs,dob,dos,bs,dtmp,dtb,dop,nvcompBatchedLZ4DecompressDefaultOpts,dstt,st);
    cudaStreamSynchronize(st);
    const int iters = 256 / bs;
    cudaEventRecord(e0, st);
    for (int i = 0; i < iters; ++i)
      nvcompBatchedLZ4DecompressAsync(dcp,dcs,dob,dos,bs,dtmp,dtb,dop,nvcompBatchedLZ4DecompressDefaultOpts,dstt,st);
    cudaEventRecord(e1, st);
    cudaStreamSynchronize(st);
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    printf("batch=%-3d launches=%-3d gpu=%7.3f ms  per-launch=%7.1f us  per-chunk=%6.1f us\n",
           bs, iters, ms, ms*1000/iters, ms*1000/(iters*bs));
  }
  return 0;
}
