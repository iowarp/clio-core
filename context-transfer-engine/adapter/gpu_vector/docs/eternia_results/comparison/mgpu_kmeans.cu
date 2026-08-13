/*
 * Multi-GPU data-parallel k-means baseline for the "pool across GPUs" (Track B)
 * comparison against Eternia. The dataset is sharded UNCOMPRESSED across N GPUs
 * (one MPI rank per GPU); each iteration does a local assign + partial (sums,
 * counts, inertia) reduction, then an ALLREDUCE across GPUs. The allreduce
 * backend is selected at runtime: "nccl" (ncclAllReduce) or "mpi" (CUDA-aware
 * MPI_Allreduce on device pointers).
 *
 * This is the honest pooling baseline: capacity ceiling = sum of physical HBM
 * (N * 40 GiB, uncompressed 1.0x), vs Eternia holding D/3.4 per GPU + tiering.
 * Same SIFT data (tiled), same k, same iters as the single-GPU capacity test.
 *
 * Reports per run: pooled dataset GiB, aggregate throughput (MiB/s over the
 * points streamed per iter), peak GPU MiB/rank (cudaMemGetInfo), final inertia.
 */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <mpi.h>
#include <cuda_runtime.h>
#include <nccl.h>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  std::fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD,1);} }while(0)
#define NK(x) do{ ncclResult_t r=(x); if(r!=ncclSuccess){ \
  std::fprintf(stderr,"NCCL %s:%d %s\n",__FILE__,__LINE__,ncclGetErrorString(r)); MPI_Abort(MPI_COMM_WORLD,1);} }while(0)

static const int kDims = 128;
static const int kClusters = 256;

__global__ void AssignKernel(const float* __restrict__ pts, unsigned long long n,
                             const float* __restrict__ cen, double* sums,
                             int* counts, double* inertia) {
  unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float* p = pts + i * kDims;
  float best = 3.4e38f; int bc = 0;
  for (int c = 0; c < kClusters; ++c) {
    const float* q = cen + c * kDims; float d = 0.f;
    for (int k = 0; k < kDims; ++k) { float t = p[k]-q[k]; d += t*t; }
    if (d < best) { best = d; bc = c; }
  }
  atomicAdd(&counts[bc], 1);
  atomicAdd(inertia, (double)best);
  double* s = sums + (unsigned long long)bc * kDims;
  for (int k = 0; k < kDims; ++k) atomicAdd(&s[k], (double)p[k]);
}

// Load whole SIFT .fvecs into host [N0*kDims], normalized /255. Returns N0.
static unsigned long long LoadSift(const char* path, std::vector<float>& out){
  std::ifstream f(path, std::ios::binary);
  if(!f){ return 0; }
  f.seekg(0,std::ios::end); long long bytes=f.tellg(); f.seekg(0);
  const unsigned long long rec = 4 + (unsigned long long)kDims*4;
  const unsigned long long n0 = (unsigned long long)bytes/rec;
  out.resize(n0*kDims);
  for(unsigned long long p=0;p<n0;++p){ int dim=0; f.read((char*)&dim,4);
    if(!f||dim!=kDims) return 0;
    f.read((char*)(out.data()+p*kDims), 4ULL*kDims);
    for(int d=0;d<kDims;++d) out[p*kDims+d]*=1.0f/255.0f; }
  return n0;
}

int main(int argc, char** argv){
  MPI_Init(&argc,&argv);
  int rank=0,np=1; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&np);

  // args: backend(nccl|mpi)  gib_per_gpu  iters  data_path
  std::string backend = (argc>1)? argv[1] : "nccl";
  double gib_per_gpu  = (argc>2)? atof(argv[2]) : 8.0;
  int iters           = (argc>3)? atoi(argv[3]) : 4;
  const char* data    = (argc>4)? argv[4] : "/u/rpawar/data/sift/sift/sift_base.fvecs";

  CK(cudaSetDevice(rank % 8));   // one GPU per rank (local)

  // NCCL comm (rank 0 makes id, bcast)
  ncclComm_t comm; ncclUniqueId id;
  if(rank==0) NK(ncclGetUniqueId(&id));
  MPI_Bcast(&id,sizeof(id),MPI_BYTE,0,MPI_COMM_WORLD);
  if(backend=="nccl") NK(ncclCommInitRank(&comm,np,id,rank));

  // Per-GPU shard: gib_per_gpu of points, tiled from SIFT.
  const unsigned long long shard_bytes = (unsigned long long)(gib_per_gpu*(1ULL<<30));
  const unsigned long long n_local = shard_bytes/(kDims*sizeof(float));
  std::vector<float> sift; unsigned long long n0=LoadSift(data,sift);
  if(!n0){ if(rank==0) std::fprintf(stderr,"[MG] cannot load %s\n",data); MPI_Abort(MPI_COMM_WORLD,2);}
  const unsigned long long file_elems=n0*kDims;

  size_t f0,tot; CK(cudaMemGetInfo(&f0,&tot));
  float* d_pts=nullptr;
  cudaError_t ae=cudaMalloc(&d_pts, n_local*kDims*sizeof(float));
  int local_ok = (ae==cudaSuccess)?1:0, all_ok=0;
  MPI_Allreduce(&local_ok,&all_ok,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
  if(!all_ok){ if(rank==0) std::fprintf(stderr,
      "[MG][OOM] backend=%s gib/gpu=%.1f: cudaMalloc shard failed (exceeds 40GiB HBM)\n",
      backend.c_str(),gib_per_gpu);
    cudaGetLastError(); if(backend=="nccl") ncclCommDestroy(comm); MPI_Finalize(); return 0; }

  // Fill shard from tiled SIFT (host staged in chunks).
  { std::vector<float> hbuf; const unsigned long long CH=file_elems;
    unsigned long long done=0, off=(unsigned long long)rank*n_local*kDims;
    hbuf.resize(CH);
    while(done < n_local*kDims){ unsigned long long m=std::min(CH, n_local*kDims-done);
      for(unsigned long long j=0;j<m;++j) hbuf[j]=sift[(off+done+j)%file_elems];
      CK(cudaMemcpy(d_pts+done, hbuf.data(), m*sizeof(float), cudaMemcpyHostToDevice));
      done+=m; }
  }
  size_t f1; CK(cudaMemGetInfo(&f1,&tot)); long long peak_mib=(long long)((tot-f1)>>20);

  // Centroids: first kClusters tiled points (identical on every rank).
  std::vector<float> cen(kClusters*kDims);
  for(unsigned long long j=0;j<(unsigned long long)kClusters*kDims;++j) cen[j]=sift[j%file_elems];
  float* d_cen; CK(cudaMalloc(&d_cen,kClusters*kDims*sizeof(float)));
  double *d_sums,*d_inertia; int* d_counts;
  CK(cudaMalloc(&d_sums,kClusters*kDims*sizeof(double)));
  CK(cudaMalloc(&d_counts,kClusters*sizeof(int)));
  CK(cudaMalloc(&d_inertia,sizeof(double)));
  std::vector<double> h_sums(kClusters*kDims); std::vector<int> h_counts(kClusters);
  double inertia=0;

  MPI_Barrier(MPI_COMM_WORLD);
  auto t0=std::chrono::steady_clock::now();
  for(int it=0; it<iters; ++it){
    CK(cudaMemcpy(d_cen,cen.data(),cen.size()*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMemset(d_sums,0,kClusters*kDims*sizeof(double)));
    CK(cudaMemset(d_counts,0,kClusters*sizeof(int)));
    CK(cudaMemset(d_inertia,0,sizeof(double)));
    int th=256; unsigned long long bl=(n_local+th-1)/th;
    AssignKernel<<<(unsigned)std::min(bl,(unsigned long long)2147483647u),th>>>(d_pts,n_local,d_cen,d_sums,d_counts,d_inertia);
    CK(cudaDeviceSynchronize());
    // allreduce sums(double), counts(int as double), inertia
    if(backend=="nccl"){
      NK(ncclAllReduce(d_sums,d_sums,kClusters*kDims,ncclDouble,ncclSum,comm,0));
      // counts: reduce as int via a double temp is overkill; use NCCL int
      NK(ncclAllReduce(d_counts,d_counts,kClusters,ncclInt,ncclSum,comm,0));
      NK(ncclAllReduce(d_inertia,d_inertia,1,ncclDouble,ncclSum,comm,0));
      CK(cudaStreamSynchronize(0));
    } else { // CUDA-aware MPI on device pointers
      MPI_Allreduce(MPI_IN_PLACE,d_sums,kClusters*kDims,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE,d_counts,kClusters,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE,d_inertia,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    }
    CK(cudaMemcpy(h_sums.data(),d_sums,h_sums.size()*sizeof(double),cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_counts.data(),d_counts,h_counts.size()*sizeof(int),cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&inertia,d_inertia,sizeof(double),cudaMemcpyDeviceToHost));
    for(int c=0;c<kClusters;++c){ int cnt=h_counts[c]; if(cnt>0)
      for(int d=0;d<kDims;++d) cen[c*kDims+d]=(float)(h_sums[c*kDims+d]/cnt); }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  double dt=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();

  unsigned long long pooled_pts=n_local*(unsigned long long)np;
  double pooled_mib=(double)pooled_pts*kDims*sizeof(float)/(1024.0*1024.0);
  double agg_mibps=dt>0? pooled_mib*iters/dt : 0.0;
  long long maxpeak=0; MPI_Reduce(&peak_mib,&maxpeak,1,MPI_LONG_LONG,MPI_MAX,0,MPI_COMM_WORLD);
  if(rank==0){
    std::fprintf(stderr,
      "[MG][RESULT] backend=%s ngpu=%d gib_per_gpu=%.1f pooled_gib=%.1f iters=%d "
      "time_s=%.3f agg_mibps=%.1f peak_gpu_mib_per_gpu=%lld inertia=%.1f\n",
      backend.c_str(),np,gib_per_gpu,pooled_mib/1024.0,iters,dt,agg_mibps,maxpeak,inertia);
  }
  cudaFree(d_pts);cudaFree(d_cen);cudaFree(d_sums);cudaFree(d_counts);cudaFree(d_inertia);
  if(backend=="nccl") ncclCommDestroy(comm);
  MPI_Finalize(); return 0;
}
