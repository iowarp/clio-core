/*
 * Single-GPU MEMORY-EXPANSION baselines for the GNN feature-store readout.
 * Workload = sum-pool the whole N x 128-d float feature matrix into a global
 * per-feature sum (the aggregation the Eternia GNN capacity test performs),
 * repeated `iters` times (epochs). Backends decide how the feature matrix is
 * made visible when it exceeds HBM:
 *   incore   : cudaMalloc the whole matrix in HBM (OOMs > HBM)
 *   zerocopy : cudaHostAllocMapped -> GPU reads over PCIe (ceiling = host RAM)
 *   staged   : host-resident, windowed H2D each epoch (ceiling = host RAM)
 *   gds      : file on disk, windowed pread+H2D each epoch (ceiling = disk)
 * (UVM is measured by the Eternia CTE test harness; Eternia is the compressed+
 *  tiered reference.) Reports whole-process peak GPU, throughput, per-feature
 * checksum (for bit-exactness across backends).
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
static const int kDims=128;

__global__ void AggKernel(const float* __restrict__ pts, unsigned long long n, double* sums){
  unsigned long long i=blockIdx.x*(unsigned long long)blockDim.x+threadIdx.x;
  if(i>=n) return;
  const float* p=pts+i*kDims;
  #pragma unroll
  for(int k=0;k<kDims;++k) atomicAdd(&sums[k],(double)p[k]);
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
  std::string backend = argc>1? argv[1] : "zerocopy";
  double gib          = argc>2? atof(argv[2]) : 8.0;
  int iters           = argc>3? atoi(argv[3]) : 3;      // epochs
  const char* data    = argc>4? argv[4] : "/u/rpawar/data/sift/sift/sift_base.fvecs";
  const char* scratch = argc>5? argv[5] : "/tmp";

  const unsigned long long bytes=(unsigned long long)(gib*(1ULL<<30));
  const unsigned long long npoints=bytes/(kDims*sizeof(float));
  std::vector<float> sift; unsigned long long n0=LoadData(data,sift);
  if(!n0){ std::fprintf(stderr,"[EX] cannot load %s\n",data); return 2; }
  const unsigned long long fe=n0*kDims, N=npoints*kDims;

  size_t f_base,tot; CK(cudaMemGetInfo(&f_base,&tot)); size_t f_min=f_base;
  auto sample=[&]{ size_t f,t; if(cudaMemGetInfo(&f,&t)==cudaSuccess){tot=t; if(f<f_min)f_min=f;} };

  float* pts=nullptr; float* h_map=nullptr; int gds_fd=-1; std::string gpath;
  float* d_win=nullptr; const unsigned long long win_e=(4ULL<<20)/sizeof(float);
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
  } else if(backend=="staged"){
    // host-resident dataset ceiling = host RAM: gate on CLIO_HOST_GIB
    long long host_gib=0; if(const char*h=getenv("CLIO_HOST_GIB")) host_gib=atoll(h);
    if(host_gib>0 && (long long)(bytes>>30)>host_gib){
      std::printf("[EX][RESULT] backend=staged gib=%.0f ran=OOM_HOSTRAM\n",gib); return 0; }
    CK(cudaMalloc(&d_win,win_e*sizeof(float)));
    CK(cudaHostAlloc((void**)&h_map,win_e*sizeof(float),cudaHostAllocDefault)); sample();
  } else if(backend=="gds"){
    gpath=std::string(scratch)+"/expand_gnn.bin";
    { std::ofstream of(gpath,std::ios::binary); std::vector<float> h(fe);
      unsigned long long done=0; while(done<N){ unsigned long long m=std::min(fe,N-done);
        for(unsigned long long j=0;j<m;++j) h[j]=sift[(done+j)%fe];
        of.write((char*)h.data(),m*sizeof(float)); done+=m; } }
    gds_fd=open(gpath.c_str(),O_RDONLY);
    if(gds_fd<0){ std::fprintf(stderr,"[EX] open failed\n"); return 4; }
    CK(cudaMalloc(&d_win,win_e*sizeof(float)));
    CK(cudaHostAlloc((void**)&h_map,win_e*sizeof(float),cudaHostAllocDefault)); sample();
  } else { std::fprintf(stderr,"[EX] unknown backend %s\n",backend.c_str()); return 3; }

  double *d_sums; CK(cudaMalloc(&d_sums,kDims*sizeof(double)));
  std::vector<double> h_sums(kDims);
  auto t0=std::chrono::steady_clock::now();
  for(int it=0; it<iters; ++it){
    CK(cudaMemset(d_sums,0,kDims*sizeof(double)));
    int th=256;
    if(backend=="staged"){
      unsigned long long off=0;
      while(off<N){ unsigned long long m=std::min(win_e,N-off);
        for(unsigned long long j=0;j<m;++j) h_map[j]=sift[(off+j)%fe];
        CK(cudaMemcpy(d_win,h_map,m*sizeof(float),cudaMemcpyHostToDevice));
        unsigned long long np=m/kDims; if(np){ unsigned long long bl=(np+th-1)/th;
          AggKernel<<<(unsigned)bl,th>>>(d_win,np,d_sums); } off+=m; }
      CK(cudaDeviceSynchronize()); sample();
    } else if(backend=="gds"){
      unsigned long long off=0;
      while(off<N){ unsigned long long m=std::min(win_e,N-off);
        ssize_t got=pread(gds_fd,h_map,m*sizeof(float),(off_t)(off*sizeof(float)));
        if(got<=0){ std::fprintf(stderr,"[EX] pread failed\n"); break; }
        unsigned long long mf=(unsigned long long)got/sizeof(float);
        CK(cudaMemcpy(d_win,h_map,mf*sizeof(float),cudaMemcpyHostToDevice));
        unsigned long long np=mf/kDims; if(np){ unsigned long long bl=(np+th-1)/th;
          AggKernel<<<(unsigned)bl,th>>>(d_win,np,d_sums); } off+=mf; }
      CK(cudaDeviceSynchronize()); sample();
    } else {
      unsigned long long bl=(npoints+th-1)/th;
      AggKernel<<<(unsigned)std::min(bl,(unsigned long long)2147483647u),th>>>(pts,npoints,d_sums);
      CK(cudaDeviceSynchronize()); sample();
    }
  }
  double dt=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
  CK(cudaMemcpy(h_sums.data(),d_sums,kDims*sizeof(double),cudaMemcpyDeviceToHost));
  long long peak=(long long)((tot-f_min)>>20);
  double mib=(double)N*sizeof(float)/(1024.0*1024.0);
  std::printf("[EX][RESULT] backend=%s gib=%.0f ran=ran peak_gpu_mib=%lld throughput_mibps=%.1f time_per_iter_s=%.3f checksum=%.6e iters=%d\n",
    backend.c_str(),gib,peak, dt>0?mib*iters/dt:0.0, iters>0?dt/iters:0.0, h_sums[0], iters);
  if(gds_fd>=0){ close(gds_fd); remove(gpath.c_str()); }
  return 0;
}
