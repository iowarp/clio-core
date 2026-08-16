/* Multi-GPU GNN feature-store readout (Track B) — pool the N x 128 feature matrix
 * UNCOMPRESSED across N GPUs (one MPI rank/GPU), sum-pool locally, ALLREDUCE the
 * 128-d partial sums. Backend: nccl | mpi. Capacity ceiling = sum of physical HBM
 * (N*40 GiB, 1.0x). Reports pooled GiB, aggregate throughput, peak GPU/rank. */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <mpi.h>
#include <cuda_runtime.h>
#include <nccl.h>
#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ std::fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD,1);} }while(0)
#define NK(x) do{ ncclResult_t r=(x); if(r!=ncclSuccess){ std::fprintf(stderr,"NCCL %s\n",ncclGetErrorString(r)); MPI_Abort(MPI_COMM_WORLD,1);} }while(0)
static const int kDims=128;
__global__ void AggKernel(const float* __restrict__ p, unsigned long long n, double* s){
  unsigned long long i=blockIdx.x*(unsigned long long)blockDim.x+threadIdx.x; if(i>=n) return;
  const float* q=p+i*kDims;
  #pragma unroll
  for(int k=0;k<kDims;++k) atomicAdd(&s[k],(double)q[k]);
}
static unsigned long long LoadSift(const char* path, std::vector<float>& out){
  std::ifstream f(path,std::ios::binary); if(!f) return 0;
  f.seekg(0,std::ios::end); long long b=f.tellg(); f.seekg(0);
  const unsigned long long rec=4+(unsigned long long)kDims*4, n0=(unsigned long long)b/rec; out.resize(n0*kDims);
  for(unsigned long long p=0;p<n0;++p){ int dim=0; f.read((char*)&dim,4); if(!f||dim!=kDims) return 0;
    f.read((char*)(out.data()+p*kDims),4ULL*kDims); for(int d=0;d<kDims;++d) out[p*kDims+d]*=1.0f/255.0f; } return n0;
}

// Dispatch: raw ".f32" file = [N x kDims] float32 (papers100M agg features, no
// normalization); otherwise SIFT .fvecs (with dim headers, /255).
static unsigned long long LoadData(const char* path, std::vector<float>& out){
  std::string pp(path);
  if(pp.size()>4 && pp.substr(pp.size()-4)==".f32"){
    std::ifstream f(path,std::ios::binary); if(!f) return 0;
    f.seekg(0,std::ios::end); long long b=f.tellg(); f.seekg(0);
    unsigned long long n0=(unsigned long long)b/((unsigned long long)kDims*sizeof(float));
    out.resize(n0*kDims); f.read((char*)out.data(),(std::streamsize)(n0*(unsigned long long)kDims*sizeof(float)));
    return n0;
  }
  return LoadSift(path,out);
}
int main(int argc,char** argv){
  MPI_Init(&argc,&argv); int rank,np; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&np);
  std::string bk=argc>1?argv[1]:"nccl"; double gib=argc>2?atof(argv[2]):8.0; int iters=argc>3?atoi(argv[3]):3;
  const char* data=argc>4?argv[4]:"/u/rpawar/data/sift/sift/sift_base.fvecs";
  CK(cudaSetDevice(rank%8));
  ncclComm_t comm; ncclUniqueId id; if(rank==0) NK(ncclGetUniqueId(&id));
  MPI_Bcast(&id,sizeof(id),MPI_BYTE,0,MPI_COMM_WORLD); if(bk=="nccl") NK(ncclCommInitRank(&comm,np,id,rank));
  const unsigned long long shard=(unsigned long long)(gib*(1ULL<<30)), n_local=shard/(kDims*sizeof(float));
  std::vector<float> sift; unsigned long long n0=LoadData(data,sift);
  if(!n0){ if(!rank) std::fprintf(stderr,"[MG] load fail\n"); MPI_Abort(MPI_COMM_WORLD,2);} const unsigned long long fe=n0*kDims;
  size_t f0,tot; CK(cudaMemGetInfo(&f0,&tot));
  float* d=nullptr; cudaError_t ae=cudaMalloc(&d,n_local*kDims*sizeof(float));
  int ok=(ae==cudaSuccess),all=0; MPI_Allreduce(&ok,&all,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
  if(!all){ if(!rank) std::fprintf(stderr,"[MG][RESULT] backend=%s ngpu=%d gib_per_gpu=%.1f pooled_gib=%.1f ran=OOM (shard>40GiB HBM)\n",bk.c_str(),np,gib,gib*np);
    cudaGetLastError(); if(bk=="nccl") ncclCommDestroy(comm); MPI_Finalize(); return 0; }
  { std::vector<float> h(fe); unsigned long long done=0, off=(unsigned long long)rank*n_local*kDims;
    while(done<n_local*kDims){ unsigned long long m=std::min(fe,n_local*kDims-done);
      for(unsigned long long j=0;j<m;++j) h[j]=sift[(off+done+j)%fe];
      CK(cudaMemcpy(d+done,h.data(),m*sizeof(float),cudaMemcpyHostToDevice)); done+=m; } }
  size_t f1; CK(cudaMemGetInfo(&f1,&tot)); long long peak=(long long)((tot-f1)>>20);
  double* ds; CK(cudaMalloc(&ds,kDims*sizeof(double))); std::vector<double> hs(kDims);
  MPI_Barrier(MPI_COMM_WORLD); auto t0=std::chrono::steady_clock::now();
  for(int it=0; it<iters; ++it){ CK(cudaMemset(ds,0,kDims*sizeof(double)));
    int th=256; unsigned long long bl=(n_local+th-1)/th;
    AggKernel<<<(unsigned)std::min(bl,(unsigned long long)2147483647u),th>>>(d,n_local,ds); CK(cudaDeviceSynchronize());
    if(bk=="nccl"){ NK(ncclAllReduce(ds,ds,kDims,ncclDouble,ncclSum,comm,0)); CK(cudaStreamSynchronize(0)); }
    else MPI_Allreduce(MPI_IN_PLACE,ds,kDims,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD); double dt=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
  CK(cudaMemcpy(hs.data(),ds,kDims*sizeof(double),cudaMemcpyDeviceToHost));
  double pooled=(double)n_local*np*kDims*sizeof(float)/(1024.0*1024.0);
  long long mx=0; MPI_Reduce(&peak,&mx,1,MPI_LONG_LONG,MPI_MAX,0,MPI_COMM_WORLD);
  if(!rank) std::fprintf(stderr,"[MG][RESULT] backend=%s ngpu=%d gib_per_gpu=%.1f pooled_gib=%.1f ran=ran iters=%d time_s=%.3f agg_mibps=%.1f peak_gpu_mib_per_gpu=%lld checksum=%.6e\n",
    bk.c_str(),np,gib,pooled/1024.0,iters,dt,dt>0?pooled*iters/dt:0.0,mx,hs[0]);
  cudaFree(d);cudaFree(ds); if(bk=="nccl") ncclCommDestroy(comm); MPI_Finalize(); return 0;
}
