/*
 * NVSHMEM multi-GPU k-means baseline (Track B). Same data-parallel k-means as
 * mgpu_kmeans, but the partial (sums, counts, inertia) are reduced across GPUs
 * with an NVSHMEM collective over the symmetric heap. Dataset sharded
 * UNCOMPRESSED across N GPUs (capacity ceiling = sum of physical HBM, 1.0x).
 *
 * Reports pooled GiB, aggregate throughput (MiB/s), peak GPU MiB/PE, inertia.
 */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <mpi.h>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  std::fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD,1);} }while(0)

static const int kDims=128, kClusters=256;

__global__ void AssignKernel(const float* __restrict__ pts, unsigned long long n,
                             const float* __restrict__ cen, double* sums,
                             double* counts, double* inertia){
  unsigned long long i=blockIdx.x*(unsigned long long)blockDim.x+threadIdx.x;
  if(i>=n) return;
  const float* p=pts+i*kDims; float best=3.4e38f; int bc=0;
  for(int c=0;c<kClusters;++c){ const float* q=cen+c*kDims; float d=0.f;
    for(int k=0;k<kDims;++k){ float t=p[k]-q[k]; d+=t*t; } if(d<best){best=d;bc=c;} }
  atomicAdd(&counts[bc],1.0); atomicAdd(inertia,(double)best);
  double* s=sums+(unsigned long long)bc*kDims;
  for(int k=0;k<kDims;++k) atomicAdd(&s[k],(double)p[k]);
}

static unsigned long long LoadSift(const char* path, std::vector<float>& out){
  std::ifstream f(path,std::ios::binary); if(!f) return 0;
  f.seekg(0,std::ios::end); long long b=f.tellg(); f.seekg(0);
  const unsigned long long rec=4+(unsigned long long)kDims*4, n0=(unsigned long long)b/rec;
  out.resize(n0*kDims);
  for(unsigned long long p=0;p<n0;++p){ int dim=0; f.read((char*)&dim,4);
    if(!f||dim!=kDims) return 0; f.read((char*)(out.data()+p*kDims),4ULL*kDims);
    for(int d=0;d<kDims;++d) out[p*kDims+d]*=1.0f/255.0f; } return n0;
}

int main(int argc,char** argv){
  MPI_Init(&argc,&argv);
  int rank=0,np=1; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&np);
  double gib_per_gpu=(argc>1)?atof(argv[1]):8.0;
  int iters=(argc>2)?atoi(argv[2]):4;
  const char* data=(argc>3)?argv[3]:"/u/rpawar/data/sift/sift/sift_base.fvecs";

  CK(cudaSetDevice(rank%8));
  MPI_Comm mc=MPI_COMM_WORLD;
  nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
  attr.mpi_comm=&mc;
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM,&attr);

  const unsigned long long shard_bytes=(unsigned long long)(gib_per_gpu*(1ULL<<30));
  const unsigned long long n_local=shard_bytes/(kDims*sizeof(float));
  std::vector<float> sift; unsigned long long n0=LoadSift(data,sift);
  if(!n0){ if(rank==0) std::fprintf(stderr,"[NVK] cannot load %s\n",data); MPI_Abort(MPI_COMM_WORLD,2);}
  const unsigned long long file_elems=n0*kDims;

  size_t f0,tot; CK(cudaMemGetInfo(&f0,&tot));
  float* d_pts=nullptr; cudaError_t ae=cudaMalloc(&d_pts,n_local*kDims*sizeof(float));
  int ok=(ae==cudaSuccess)?1:0, allok=0; MPI_Allreduce(&ok,&allok,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
  if(!allok){ if(rank==0) std::fprintf(stderr,"[NVK][OOM] gib/gpu=%.1f shard cudaMalloc failed\n",gib_per_gpu);
    nvshmem_finalize(); MPI_Finalize(); return 0; }
  { std::vector<float> h(file_elems); unsigned long long done=0, off=(unsigned long long)rank*n_local*kDims;
    while(done<n_local*kDims){ unsigned long long m=std::min(file_elems,n_local*kDims-done);
      for(unsigned long long j=0;j<m;++j) h[j]=sift[(off+done+j)%file_elems];
      CK(cudaMemcpy(d_pts+done,h.data(),m*sizeof(float),cudaMemcpyHostToDevice)); done+=m; } }
  size_t f1; CK(cudaMemGetInfo(&f1,&tot)); long long peak=(long long)((tot-f1)>>20);

  std::vector<float> cen(kClusters*kDims);
  for(unsigned long long j=0;j<(unsigned long long)kClusters*kDims;++j) cen[j]=sift[j%file_elems];
  float* d_cen; CK(cudaMalloc(&d_cen,kClusters*kDims*sizeof(float)));
  // symmetric-heap reduce buffers
  double* s_sums =(double*)nvshmem_malloc(kClusters*kDims*sizeof(double));
  double* s_cnt  =(double*)nvshmem_malloc(kClusters*sizeof(double));
  double* s_in   =(double*)nvshmem_malloc(sizeof(double));
  double* r_sums =(double*)nvshmem_malloc(kClusters*kDims*sizeof(double));
  double* r_cnt  =(double*)nvshmem_malloc(kClusters*sizeof(double));
  double* r_in   =(double*)nvshmem_malloc(sizeof(double));
  std::vector<double> h_sums(kClusters*kDims), h_cnt(kClusters); double inertia=0;

  MPI_Barrier(MPI_COMM_WORLD); auto t0=std::chrono::steady_clock::now();
  for(int it=0;it<iters;++it){
    CK(cudaMemcpy(d_cen,cen.data(),cen.size()*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMemset(s_sums,0,kClusters*kDims*sizeof(double)));
    CK(cudaMemset(s_cnt,0,kClusters*sizeof(double)));
    CK(cudaMemset(s_in,0,sizeof(double)));
    int th=256; unsigned long long bl=(n_local+th-1)/th;
    AssignKernel<<<(unsigned)std::min(bl,(unsigned long long)2147483647u),th>>>(d_pts,n_local,d_cen,s_sums,s_cnt,s_in);
    CK(cudaDeviceSynchronize());
    nvshmem_double_sum_reduce(NVSHMEM_TEAM_WORLD,r_sums,s_sums,(size_t)kClusters*kDims);
    nvshmem_double_sum_reduce(NVSHMEM_TEAM_WORLD,r_cnt,s_cnt,(size_t)kClusters);
    nvshmem_double_sum_reduce(NVSHMEM_TEAM_WORLD,r_in,s_in,1);
    nvshmem_barrier_all();
    CK(cudaMemcpy(h_sums.data(),r_sums,h_sums.size()*sizeof(double),cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_cnt.data(),r_cnt,h_cnt.size()*sizeof(double),cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&inertia,r_in,sizeof(double),cudaMemcpyDeviceToHost));
    for(int c=0;c<kClusters;++c){ double cnt=h_cnt[c]; if(cnt>0)
      for(int d=0;d<kDims;++d) cen[c*kDims+d]=(float)(h_sums[c*kDims+d]/cnt); }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  double dt=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
  unsigned long long pooled_pts=n_local*(unsigned long long)np;
  double pooled_mib=(double)pooled_pts*kDims*sizeof(float)/(1024.0*1024.0);
  long long maxpeak=0; MPI_Reduce(&peak,&maxpeak,1,MPI_LONG_LONG,MPI_MAX,0,MPI_COMM_WORLD);
  if(rank==0) std::fprintf(stderr,
    "[NVK][RESULT] backend=nvshmem ngpu=%d gib_per_gpu=%.1f pooled_gib=%.1f iters=%d "
    "time_s=%.3f agg_mibps=%.1f peak_gpu_mib_per_gpu=%lld inertia=%.1f\n",
    np,gib_per_gpu,pooled_mib/1024.0,iters,dt,dt>0?pooled_mib*iters/dt:0.0,maxpeak,inertia);
  nvshmem_free(s_sums);nvshmem_free(s_cnt);nvshmem_free(s_in);
  nvshmem_free(r_sums);nvshmem_free(r_cnt);nvshmem_free(r_in);
  cudaFree(d_pts);cudaFree(d_cen);
  nvshmem_finalize(); MPI_Finalize(); return 0;
}
