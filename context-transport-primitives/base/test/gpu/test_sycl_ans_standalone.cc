// Standalone validation + microbench for the SYCL rANS (ANS) codec.
// Unlike LZ4/Snappy there is no ubiquitous reference format, so correctness is
// (a) bit-exact roundtrip and (b) an entropy sanity check: the achieved ratio
// must come within a small margin of the byte-entropy limit 8/H(data).
//   clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda ... \
//       sycl_ans_kernels.cc test_sycl_ans_standalone.cc -o t
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
using u8 = std::uint8_t; using u32 = std::uint32_t;
extern "C" {
void *clio_sycl_ans_queue_create();
void clio_sycl_ans_queue_destroy(void *q);
int clio_sycl_ans_compress(void *q, const void *s, size_t n, void *d, size_t c, size_t *o);
int clio_sycl_ans_decompress(void *q, const void *s, size_t n, void *d, size_t c, size_t *o);
}
static double Entropy(const std::vector<u8> &v) {
  double h[256] = {0};
  for (u8 b : v) h[b] += 1.0;
  double H = 0, n = (double)v.size();
  for (int i = 0; i < 256; ++i) if (h[i] > 0) { double p = h[i]/n; H -= p*std::log2(p); }
  return H;  // bits/byte
}
// text-like: repeated words + noise (skew from repetition)
static std::vector<u8> TextData(size_t n, double red) {
  std::vector<u8> v(n); std::mt19937 rng(1234);
  std::uniform_real_distribution<double> U(0,1);
  const char *w[] = {"the ","quick ","brown ","fox ","gpu ","ans ","sycl ","data ","x ","z "};
  size_t i=0; while(i<n){ if(U(rng)<red){const char*s=w[rng()%10];for(size_t k=0;s[k]&&i<n;++k)v[i++]=(u8)s[k];} else v[i++]=(u8)(rng()&0xFF);} return v;
}
// skewed bytes: a biased distribution (entropy < 8) with little repetition
static std::vector<u8> SkewData(size_t n, double bias) {
  std::vector<u8> v(n); std::mt19937 rng(7);
  std::geometric_distribution<int> g(bias);  // small values dominate
  for (size_t i=0;i<n;++i) v[i]=(u8)(g(rng)&0xFF); return v;
}
static bool run(const char *name, void *q, std::vector<u8> &orig) {
  size_t n = orig.size(); size_t cap = n + 512*((n>>15)+2) + 4096;
  std::vector<u8> comp(cap), deco(n);
  size_t cs=0, ds=0;
  if(!clio_sycl_ans_compress(q,orig.data(),n,comp.data(),cap,&cs)){printf("[%s] compress FAIL\n",name);return false;}
  auto t0=std::chrono::high_resolution_clock::now(); int reps=5;
  for(int r=0;r<reps;r++) clio_sycl_ans_compress(q,orig.data(),n,comp.data(),cap,&cs);
  double cms=std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now()-t0).count()/reps;
  if(!clio_sycl_ans_decompress(q,comp.data(),cs,deco.data(),n,&ds)){printf("[%s] decompress FAIL\n",name);return false;}
  auto t2=std::chrono::high_resolution_clock::now();
  for(int r=0;r<reps;r++) clio_sycl_ans_decompress(q,comp.data(),cs,deco.data(),n,&ds);
  double dms=std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now()-t2).count()/reps;
  bool rt = ds==n && !memcmp(orig.data(),deco.data(),n);
  double H=Entropy(orig), limit = H>0? 8.0/H : 999, ratio=(double)n/cs;
  // achieved ratio should not EXCEED the entropy limit (impossible for a real
  // entropy coder) and should come reasonably close to it.
  bool ent_ok = ratio <= limit*1.02 && ratio >= limit*0.80;
  double gb=n/1e9;
  printf("[%s] %.0fMB roundtrip=%s  H=%.2f b/byte  ratio=%.3fx (limit %.3fx) entropy=%s  c=%.2f d=%.2f GB/s\n",
         name, n/1048576.0, rt?"OK":"BAD", H, ratio, limit, ent_ok?"OK":"OFF",
         gb/(cms/1e3), gb/(dms/1e3));
  return rt && ent_ok;
}
int main() {
  void *q = clio_sycl_ans_queue_create();
  if(!q){printf("FAIL: no GPU queue\n");return 1;}
  bool all=true;
  { auto d=TextData(32u<<20,0.85); all &= run("text-32M",q,d); }
  { auto d=SkewData(32u<<20,0.30); all &= run("skew-32M",q,d); }
  { auto d=SkewData(64u<<20,0.10); all &= run("skew-64M",q,d); }
  { auto d=TextData(8u<<20,0.95);  all &= run("text-8M ",q,d); }
  clio_sycl_ans_queue_destroy(q);
  printf("=== %s ===\n", all?"ALL PASS":"FAILURE");
  return all?0:1;
}
