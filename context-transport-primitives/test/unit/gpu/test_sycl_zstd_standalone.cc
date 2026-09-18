// Standalone validation + microbench for the SYCL Zstd-STYLE codec (LZ + rANS).
// Not libzstd-compatible, so correctness is bit-exact roundtrip. The headline is
// ratio: entropy-coding the literals should beat a pure LZ codec (lz4/snappy).
//   clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda ... \
//       sycl_zstd_kernels.cc test_sycl_zstd_standalone.cc -o t
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
using u8 = std::uint8_t; using u32 = std::uint32_t;
extern "C" {
void *clio_sycl_zstd_queue_create();
void clio_sycl_zstd_queue_destroy(void *q);
int clio_sycl_zstd_compress(void *q, const void *s, size_t n, void *d, size_t c, size_t *o);
int clio_sycl_zstd_decompress(void *q, const void *s, size_t n, void *d, size_t c, size_t *o);
}
static std::vector<u8> TextData(size_t n, double red) {
  std::vector<u8> v(n); std::mt19937 rng(1234);
  std::uniform_real_distribution<double> U(0,1);
  const char *w[] = {"the ","quick ","brown ","fox ","gpu ","zstd ","sycl ","data ","block ","x "};
  size_t i=0; while(i<n){ if(U(rng)<red){const char*s=w[rng()%10];for(size_t k=0;s[k]&&i<n;++k)v[i++]=(u8)s[k];} else v[i++]=(u8)(rng()&0xFF);} return v;
}
static bool run(const char *name, void *q, std::vector<u8> &orig) {
  size_t n=orig.size(), cap=2*n+4096;
  std::vector<u8> comp(cap), deco(n); size_t cs=0, ds=0;
  if(!clio_sycl_zstd_compress(q,orig.data(),n,comp.data(),cap,&cs)){printf("[%s] compress FAIL\n",name);return false;}
  auto t0=std::chrono::high_resolution_clock::now(); int reps=5;
  for(int r=0;r<reps;r++) clio_sycl_zstd_compress(q,orig.data(),n,comp.data(),cap,&cs);
  double cms=std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now()-t0).count()/reps;
  if(!clio_sycl_zstd_decompress(q,comp.data(),cs,deco.data(),n,&ds)){printf("[%s] decompress FAIL\n",name);return false;}
  auto t2=std::chrono::high_resolution_clock::now();
  for(int r=0;r<reps;r++) clio_sycl_zstd_decompress(q,comp.data(),cs,deco.data(),n,&ds);
  double dms=std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now()-t2).count()/reps;
  bool rt = ds==n && !memcmp(orig.data(),deco.data(),n);
  double gb=n/1e9;
  printf("[%s] %.0fMB roundtrip=%s  ratio=%.3fx  compress=%.2f GB/s  decompress=%.2f GB/s\n",
         name, n/1048576.0, rt?"OK":"BAD", (double)n/cs, gb/(cms/1e3), gb/(dms/1e3));
  return rt;
}
int main() {
  void *q=clio_sycl_zstd_queue_create();
  if(!q){printf("FAIL: no GPU queue\n");return 1;}
  bool all=true;
  { auto d=TextData(32u<<20,0.85); all&=run("text-32M/.85",q,d); }
  { auto d=TextData(64u<<20,0.70); all&=run("text-64M/.70",q,d); }
  { auto d=TextData(8u<<20,0.95);  all&=run("text-8M /.95",q,d); }
  clio_sycl_zstd_queue_destroy(q);
  printf("=== %s ===\n", all?"ALL PASS":"FAILURE");
  return all?0:1;
}
