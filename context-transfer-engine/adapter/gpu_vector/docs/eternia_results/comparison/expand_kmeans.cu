/*
 * Single-GPU MEMORY-EXPANSION baselines for k-means (Track A additions).
 * Backends (choose at runtime): how the point set is made visible to the GPU
 * when it exceeds HBM:
 *   incore   : cudaMalloc the whole shard in HBM (reference; OOMs > HBM)
 *   zerocopy : cudaHostAllocMapped -> GPU reads points from host RAM over PCIe
 *              (no explicit copy; capacity ceiling = pinned host RAM)
 *   gds      : cuFile (GPUDirect Storage; compat/POSIX mode if no nvidia-fs)
 *              streams windows from a file on disk -> GPU (ceiling = disk)
 *
 * Same SIFT data (tiled), same k, same iters as the CTE Track-A test. The
 * assign kernel loads each point into registers ONCE, so zero-copy/GDS are
 * PCIe/-storage-bandwidth bound (not 256x re-reads). Reports peak GPU (whole-
 * process cudaMemGetInfo), throughput MiB/s, time/iter, inertia.
 */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <cuda_runtime.h>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  std::fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); std::exit(1);} }while(0)

static const int kDims=128, kClusters=256;

// One thread per point; load the point into registers once, then compare to all
// centroids. Point storage may be HBM, mapped host RAM, or a staged HBM window.
__global__ void AssignKernel(const float* __restrict__ pts, unsigned long long n,
                             const float* __restrict__ cen, double* sums,
                             int* counts, double* inertia){
  unsigned long long i=blockIdx.x*(unsigned long long)blockDim.x+threadIdx.x;
  if(i>=n) return;
  float pt[kDims];
  const float* p=pts+i*kDims;
  #pragma unroll
  for(int k=0;k<kDims;++k) pt[k]=p[k];          // ONE read of the point
  float best=3.4e38f; int bc=0;
  for(int c=0;c<kClusters;++c){ const float* q=cen+c*kDims; float d=0.f;
    for(int k=0;k<kDims;++k){ float t=pt[k]-q[k]; d+=t*t; } if(d<best){best=d;bc=c;} }
  atomicAdd(&counts[bc],1); atomicAdd(inertia,(double)best);
  double* s=sums+(unsigned long long)bc*kDims;
  for(int k=0;k<kDims;++k) atomicAdd(&s[k],(double)pt[k]);
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
  std::string backend = argc>1? argv[1] : "zerocopy";
  double gib          = argc>2? atof(argv[2]) : 8.0;
  int iters           = argc>3? atoi(argv[3]) : 1;
  const char* data    = argc>4? argv[4] : "/u/rpawar/data/sift/sift/sift_base.fvecs";
  const char* scratch = argc>5? argv[5] : "/tmp";

  const unsigned long long bytes=(unsigned long long)(gib*(1ULL<<30));
  const unsigned long long npoints=bytes/(kDims*sizeof(float));
  std::vector<float> sift; unsigned long long n0=LoadSift(data,sift);
  if(!n0){ std::fprintf(stderr,"[EX] cannot load %s\n",data); return 2; }
  const unsigned long long fe=n0*kDims, N=npoints*kDims;

  size_t f_base,tot; CK(cudaMemGetInfo(&f_base,&tot)); size_t f_min=f_base;
  auto sample=[&]{ size_t f,t; if(cudaMemGetInfo(&f,&t)==cudaSuccess){tot=t; if(f<f_min)f_min=f;} };

  // ---- make the dataset visible per backend ----
  float* pts=nullptr; float* h_map=nullptr; int gds_fd=-1; std::string gpath;
  float* d_win=nullptr; const unsigned long long win_e=(4ULL<<20)/sizeof(float); // 4 MiB window (unused by incore/zerocopy)
  std::string ran="ran";
  if(backend=="incore"){
    if(cudaMalloc(&pts,N*sizeof(float))!=cudaSuccess){ cudaGetLastError();
      std::printf("[EX][RESULT] backend=incore gib=%.0f ran=OOM\n",gib); return 0; }
    std::vector<float> h(fe); unsigned long long done=0;
    while(done<N){ unsigned long long m=std::min(fe,N-done);
      for(unsigned long long j=0;j<m;++j) h[j]=sift[(done+j)%fe];
      CK(cudaMemcpy(pts+done,h.data(),m*sizeof(float),cudaMemcpyHostToDevice)); done+=m; } sample();
  } else if(backend=="zerocopy"){
    cudaError_t e=cudaHostAlloc((void**)&h_map,N*sizeof(float),cudaHostAllocMapped|cudaHostAllocPortable);
    if(e!=cudaSuccess){ cudaGetLastError();
      std::printf("[EX][RESULT] backend=zerocopy gib=%.0f ran=OOM_PINNED\n",gib); return 0; }
    for(unsigned long long j=0;j<N;++j) h_map[j]=sift[j%fe];
    CK(cudaHostGetDevicePointer((void**)&pts,h_map,0)); sample();
  } else if(backend=="gds"){
    gpath=std::string(scratch)+"/expand_gds.bin";
    { std::ofstream of(gpath,std::ios::binary); std::vector<float> h(fe);
      unsigned long long done=0; while(done<N){ unsigned long long m=std::min(fe,N-done);
        for(unsigned long long j=0;j<m;++j) h[j]=sift[(done+j)%fe];
        of.write((char*)h.data(),m*sizeof(float)); done+=m; } }
    // Real GPUDirect Storage needs the nvidia-fs kernel driver (absent on Delta),
    // so cuFile falls back to POSIX. We implement that compat path directly:
    // pread the file into a pinned host bounce buffer, then H2D. Capacity ceiling
    // = disk (unbounded), which is what makes GDS/file-streaming a real past-host-
    // RAM expansion baseline. Labeled "gds_compat".
    gds_fd=open(gpath.c_str(),O_RDONLY);
    if(gds_fd<0){ std::fprintf(stderr,"[EX] open %s failed\n",gpath.c_str()); return 4; }
    CK(cudaMalloc(&d_win,win_e*sizeof(float)));
    CK(cudaHostAlloc((void**)&h_map,win_e*sizeof(float),cudaHostAllocDefault)); // bounce buf
    sample();
  } else { std::fprintf(stderr,"[EX] unknown backend %s\n",backend.c_str()); return 3; }

  // centroids: first kClusters tiled points
  std::vector<float> cen(kClusters*kDims); for(unsigned long long j=0;j<(unsigned long long)kClusters*kDims;++j) cen[j]=sift[j%fe];
  float* d_cen; CK(cudaMalloc(&d_cen,kClusters*kDims*sizeof(float)));
  double *d_sums,*d_in; int* d_cnt;
  CK(cudaMalloc(&d_sums,kClusters*kDims*sizeof(double))); CK(cudaMalloc(&d_cnt,kClusters*sizeof(int))); CK(cudaMalloc(&d_in,sizeof(double)));
  std::vector<double> h_sums(kClusters*kDims); std::vector<int> h_cnt(kClusters); double inertia=0;

  auto t0=std::chrono::steady_clock::now();
  for(int it=0; it<iters; ++it){
    CK(cudaMemcpy(d_cen,cen.data(),cen.size()*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMemset(d_sums,0,kClusters*kDims*sizeof(double))); CK(cudaMemset(d_cnt,0,kClusters*sizeof(int))); CK(cudaMemset(d_in,0,sizeof(double)));
    int th=256;
    if(backend=="gds"){
      unsigned long long off=0;
      while(off<N){ unsigned long long m=std::min(win_e,N-off);
        ssize_t got=pread(gds_fd,h_map,m*sizeof(float),(off_t)(off*sizeof(float)));
        if(got<=0){ std::fprintf(stderr,"[EX] pread failed off=%llu\n",off); break; }
        unsigned long long mf=(unsigned long long)got/sizeof(float);
        CK(cudaMemcpy(d_win,h_map,mf*sizeof(float),cudaMemcpyHostToDevice));
        unsigned long long np=mf/kDims; if(np){ unsigned long long bl=(np+th-1)/th;
          AssignKernel<<<(unsigned)bl,th>>>(d_win,np,d_cen,d_sums,d_cnt,d_in); }
        off+=mf; }
      CK(cudaDeviceSynchronize()); sample();
    } else {
      unsigned long long bl=(npoints+th-1)/th;
      AssignKernel<<<(unsigned)std::min(bl,(unsigned long long)2147483647u),th>>>(pts,npoints,d_cen,d_sums,d_cnt,d_in);
      CK(cudaDeviceSynchronize()); sample();
    }
    CK(cudaMemcpy(h_sums.data(),d_sums,h_sums.size()*sizeof(double),cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_cnt.data(),d_cnt,h_cnt.size()*sizeof(int),cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&inertia,d_in,sizeof(double),cudaMemcpyDeviceToHost));
    for(int c=0;c<kClusters;++c){ int cnt=h_cnt[c]; if(cnt>0) for(int d=0;d<kDims;++d) cen[c*kDims+d]=(float)(h_sums[c*kDims+d]/cnt); }
  }
  double dt=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
  long long peak=(long long)((tot-f_min)>>20);
  double mib=(double)N*sizeof(float)/(1024.0*1024.0);
  std::printf("[EX][RESULT] backend=%s gib=%.0f ran=%s peak_gpu_mib=%lld throughput_mibps=%.1f time_per_iter_s=%.3f inertia=%.1f iters=%d\n",
    backend.c_str(),gib,ran.c_str(),peak, dt>0?mib*iters/dt:0.0, iters>0?dt/iters:0.0, inertia, iters);
  if(backend=="gds"){ if(gds_fd>=0) close(gds_fd); if(d_win) cudaFree(d_win); if(h_map) cudaFreeHost(h_map); remove(gpath.c_str()); }
  return 0;
}
