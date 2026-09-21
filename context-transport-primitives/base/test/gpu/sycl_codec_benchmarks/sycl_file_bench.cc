// SYCL LZ4 benchmark: ratio + throughput in TWO modes on the SAME file nvcomp
// uses. host2host = codec stages H2D/D2H (fair vs nvcomp host2host). gpu-only =
// input/output already device-resident (kernel path; the codec detects the
// device pointers on the shared default context and skips staging).
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>
extern "C" {
void *clio_sycl_lz4_queue_create();
void clio_sycl_lz4_queue_destroy(void *q);
int clio_sycl_lz4_compress(void *q, const void *s, size_t n, void *d, size_t c, size_t *o);
int clio_sycl_lz4_decompress(void *q, const void *s, size_t n, void *d, size_t c, size_t *o);
}
using clk = std::chrono::high_resolution_clock;
static double ms_since(clk::time_point t) {
  return std::chrono::duration<double, std::milli>(clk::now() - t).count();
}
int main(int argc, char **argv) {
  if (argc < 2) { std::printf("usage: %s <file> [reps]\n", argv[0]); return 2; }
  int reps = argc > 2 ? atoi(argv[2]) : 10;
  std::ifstream f(argv[1], std::ios::binary | std::ios::ate);
  if (!f) { std::printf("cannot open %s\n", argv[1]); return 1; }
  size_t n = f.tellg(); f.seekg(0);
  std::vector<uint8_t> orig(n), deco(n);
  f.read((char *)orig.data(), n);
  size_t cap = n + n / 255 + 4096;
  std::vector<uint8_t> comp(cap);
  void *q = clio_sycl_lz4_queue_create();
  if (!q) { std::printf("[sycl-lz4] no GPU queue\n"); return 1; }
  double gb = n / 1e9;
  const char *base = strrchr(argv[1], '/'); base = base ? base + 1 : argv[1];

  // ---------- host-to-host ----------
  size_t csize = 0, dsize = 0;
  if (!clio_sycl_lz4_compress(q, orig.data(), n, comp.data(), cap, &csize)) {
    std::printf("[sycl-lz4] compress failed\n"); return 1; }
  clio_sycl_lz4_decompress(q, comp.data(), csize, deco.data(), n, &dsize);
  bool ok_h = (dsize == n) && !memcmp(orig.data(), deco.data(), n);
  auto t = clk::now();
  for (int r = 0; r < reps; r++) clio_sycl_lz4_compress(q, orig.data(), n, comp.data(), cap, &csize);
  double c_h = ms_since(t) / reps;
  t = clk::now();
  for (int r = 0; r < reps; r++) clio_sycl_lz4_decompress(q, comp.data(), csize, deco.data(), n, &dsize);
  double d_h = ms_since(t) / reps;

  // ---------- gpu-only (device-resident) ----------
  sycl::queue sq{sycl::gpu_selector_v};
  uint8_t *d_in = sycl::malloc_device<uint8_t>(n, sq);
  uint8_t *d_cmp = sycl::malloc_device<uint8_t>(cap, sq);
  uint8_t *d_dec = sycl::malloc_device<uint8_t>(n, sq);
  double c_g = 0, d_g = 0; bool ok_g = false;
  if (d_in && d_cmp && d_dec) {
    sq.memcpy(d_in, orig.data(), n).wait();
    size_t cz = 0, dz = 0;
    bool okc = clio_sycl_lz4_compress(q, d_in, n, d_cmp, cap, &cz);
    bool okd = clio_sycl_lz4_decompress(q, d_cmp, cz, d_dec, n, &dz);
    // verify device roundtrip
    std::vector<uint8_t> back(n);
    sq.memcpy(back.data(), d_dec, n).wait();
    ok_g = okc && okd && dz == n && !memcmp(orig.data(), back.data(), n);
    t = clk::now();
    for (int r = 0; r < reps; r++) clio_sycl_lz4_compress(q, d_in, n, d_cmp, cap, &cz);
    c_g = ms_since(t) / reps;
    t = clk::now();
    for (int r = 0; r < reps; r++) clio_sycl_lz4_decompress(q, d_cmp, cz, d_dec, n, &dz);
    d_g = ms_since(t) / reps;
  }
  if (d_in) sycl::free(d_in, sq);
  if (d_cmp) sycl::free(d_cmp, sq);
  if (d_dec) sycl::free(d_dec, sq);

  std::printf("[sycl-lz4] %s  size=%.0fMB  rt(host)=%s rt(gpu)=%s\n",
              base, n / 1048576.0, ok_h ? "OK" : "BAD", ok_g ? "OK" : "BAD");
  std::printf("  ratio                 = %.3fx\n", (double)n / csize);
  std::printf("  compress   host2host  = %.2f GB/s\n", gb / (c_h / 1e3));
  std::printf("  decompress host2host  = %.2f GB/s\n", gb / (d_h / 1e3));
  std::printf("  compress   GPU-only   = %.2f GB/s\n", c_g > 0 ? gb / (c_g / 1e3) : 0);
  std::printf("  decompress GPU-only   = %.2f GB/s\n", d_g > 0 ? gb / (d_g / 1e3) : 0);
  clio_sycl_lz4_queue_destroy(q);
  return (ok_h && ok_g) ? 0 : 1;
}
