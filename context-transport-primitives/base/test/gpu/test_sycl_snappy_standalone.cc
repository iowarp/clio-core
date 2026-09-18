// Standalone validation + microbench for the SYCL Snappy codec.
// Links libsnappy (CPU) only as a reference oracle to prove the SYCL-produced
// blocks are standard Snappy.
//   clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda ... \
//       sycl_snappy_kernels.cc test_sycl_snappy_standalone.cc -lsnappy -o t
// Checks: (a) SYCL compress -> SYCL decompress bit-exact;
//         (b) every non-raw chunk decodes with snappy::RawUncompress;
//         (c) a snappy::RawCompress block decodes via the SYCL decoder;
//         plus ratio + throughput.
#include <snappy.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
using u8 = std::uint8_t; using u32 = std::uint32_t; using u64 = std::uint64_t;
extern "C" {
void *clio_sycl_snappy_queue_create();
void clio_sycl_snappy_queue_destroy(void *q);
int clio_sycl_snappy_compress(void *q, const void *s, size_t n, void *d, size_t c, size_t *o);
int clio_sycl_snappy_decompress(void *q, const void *s, size_t n, void *d, size_t c, size_t *o);
}
struct Header { u32 magic; u32 chunk_size; u64 orig_size; u32 n_chunks; u32 rsvd; };
constexpr u32 kStoredFlag = 0x80000000u;
static std::vector<u8> MakeData(size_t n, double red) {
  std::vector<u8> v(n); std::mt19937 rng(1234);
  std::uniform_real_distribution<double> U(0, 1);
  const char *w[] = {"the ", "quick ", "brown ", "fox ", "GPU ", "snappy ",
                     "sycl ", "data ", "block ", "gpu "};
  size_t i = 0;
  while (i < n) {
    if (U(rng) < red) { const char *s = w[rng() % 10]; size_t l = strlen(s);
      for (size_t k = 0; k < l && i < n; ++k) v[i++] = (u8)s[k]; }
    else v[i++] = (u8)(rng() & 0xFF);
  }
  return v;
}
int main(int argc, char **argv) {
  size_t n = argc > 1 ? strtoull(argv[1], nullptr, 10) : (32u << 20);
  double red = argc > 2 ? atof(argv[2]) : 0.85;
  std::printf("=== SYCL Snappy standalone: %.1f MB, red=%.2f ===\n", n / 1048576.0, red);
  std::vector<u8> orig = MakeData(n, red), decomp(n);
  size_t cap = n + n / 255 + 4096; std::vector<u8> comp(cap);
  void *q = clio_sycl_snappy_queue_create();
  if (!q) { std::printf("FAIL: no GPU queue\n"); return 1; }
  size_t csize = 0, dsize = 0;
  if (!clio_sycl_snappy_compress(q, orig.data(), n, comp.data(), cap, &csize)) { std::printf("FAIL compress\n"); return 1; }
  auto t0 = std::chrono::high_resolution_clock::now(); int reps = 5;
  for (int r = 0; r < reps; r++) clio_sycl_snappy_compress(q, orig.data(), n, comp.data(), cap, &csize);
  double cms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count() / reps;
  if (!clio_sycl_snappy_decompress(q, comp.data(), csize, decomp.data(), n, &dsize)) { std::printf("FAIL decompress\n"); return 1; }
  auto t2 = std::chrono::high_resolution_clock::now();
  for (int r = 0; r < reps; r++) clio_sycl_snappy_decompress(q, comp.data(), csize, decomp.data(), n, &dsize);
  double dms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t2).count() / reps;
  bool rt = dsize == n && !memcmp(orig.data(), decomp.data(), n);
  std::printf("[a] roundtrip bit-exact ...... %s\n", rt ? "PASS" : "FAIL");
  // (b) each SYCL chunk decodes with reference libsnappy
  const Header *H = (const Header *)comp.data();
  const u32 *csz = (const u32 *)(comp.data() + sizeof(Header));
  size_t hdr = sizeof(Header) + (size_t)H->n_chunks * sizeof(u32);
  const u8 *pay = comp.data() + hdr; u64 off = 0; bool oracle = true; int chk = 0, raws = 0;
  std::vector<char> tmp(H->chunk_size);
  for (u32 c = 0; c < H->n_chunks; ++c) {
    u32 raw = csz[c] & ~kStoredFlag; bool stored = csz[c] & kStoredFlag;
    u32 doff = c * H->chunk_size; u32 dlen = (doff + H->chunk_size <= n) ? H->chunk_size : (u32)(n - doff);
    if (stored) raws++;
    else {
      size_t ul = 0; bool okl = snappy::GetUncompressedLength((const char *)(pay + off), raw, &ul);
      bool oku = okl && ul == dlen && snappy::RawUncompress((const char *)(pay + off), raw, tmp.data())
                 && !memcmp(tmp.data(), orig.data() + doff, dlen);
      if (!oku) { oracle = false; if (chk < 3) std::printf("    chunk %u bad (ul=%zu want=%u)\n", c, ul, dlen); }
      chk++;
    }
    off += raw;
  }
  std::printf("[b] SYCL blocks decode via libsnappy %s (%d checked, %d raw)\n", oracle ? "PASS" : "FAIL", chk, raws);
  // (c) a libsnappy block decodes via the SYCL decoder
  bool dec_ok = true;
  { u32 clen = n < H->chunk_size ? (u32)n : H->chunk_size;
    std::vector<char> sc(snappy::MaxCompressedLength(clen)); size_t sl = 0;
    snappy::RawCompress((const char *)orig.data(), clen, sc.data(), &sl);
    std::vector<u8> cont(sizeof(Header) + sizeof(u32) + sl);
    Header h{H->magic, H->chunk_size, clen, 1, 0};
    memcpy(cont.data(), &h, sizeof(Header));
    u32 cs = (u32)sl; memcpy(cont.data() + sizeof(Header), &cs, sizeof(u32));
    memcpy(cont.data() + sizeof(Header) + sizeof(u32), sc.data(), sl);
    std::vector<u8> out(clen); size_t os = 0;
    dec_ok = clio_sycl_snappy_decompress(q, cont.data(), cont.size(), out.data(), clen, &os)
             && os == clen && !memcmp(out.data(), orig.data(), clen);
  }
  std::printf("[c] libsnappy block decodes via SYCL %s\n", dec_ok ? "PASS" : "FAIL");
  double gb = n / 1e9;
  std::printf("ratio=%.3fx  compress=%.2f GB/s  decompress=%.2f GB/s\n",
              (double)n / csize, gb / (cms / 1e3), gb / (dms / 1e3));
  clio_sycl_snappy_queue_destroy(q);
  bool all = rt && oracle && dec_ok;
  std::printf("=== %s ===\n", all ? "ALL PASS" : "FAILURE");
  return all ? 0 : 1;
}
