// Standalone validation + microbench for the SYCL LZ4 codec.
// Build (DPC++, CUDA target) and run on a GPU node; links liblz4 (CPU) purely
// as a reference oracle to prove the SYCL-produced blocks are standard LZ4.
//
//   clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda \
//       -Xsycl-target-backend --cuda-gpu-arch=sm_80 \
//       sycl_lz4_kernels.cc test_sycl_lz4_standalone.cc -llz4 -o t
//
// Checks: (a) SYCL compress -> SYCL decompress is bit-exact;
//         (b) every non-raw chunk decodes with liblz4 LZ4_decompress_safe
//             (=> the SYCL encoder emits valid standard LZ4 blocks);
//         (c) a liblz4-encoded block decodes via the SYCL decoder
//             (=> the SYCL decoder accepts standard LZ4);
//         plus compress/decompress throughput and ratio.

#include <lz4.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using u8 = std::uint8_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

extern "C" {
void *clio_sycl_lz4_queue_create();
void clio_sycl_lz4_queue_destroy(void *q);
int clio_sycl_lz4_compress(void *q, const void *src, size_t n, void *dst,
                           size_t cap, size_t *out);
int clio_sycl_lz4_decompress(void *q, const void *src, size_t n, void *dst,
                             size_t cap, size_t *out);
}

struct Header {
  u32 magic;
  u32 chunk_size;
  u64 orig_size;
  u32 n_chunks;
  u32 rsvd;
};
constexpr u32 kStoredFlag = 0x80000000u;

// Data with realistic redundancy: runs + repeated tokens + some noise.
static std::vector<u8> MakeData(size_t n, double redundancy) {
  std::vector<u8> v(n);
  std::mt19937 rng(1234);
  std::uniform_real_distribution<double> U(0, 1);
  const char *words[] = {"the ", "quick ", "brown ", "fox ", "GPU ",
                         "compress ", "sycl ", "lz4 ", "data ", "block "};
  size_t i = 0;
  while (i < n) {
    if (U(rng) < redundancy) {
      const char *w = words[rng() % 10];
      size_t l = std::strlen(w);
      for (size_t k = 0; k < l && i < n; ++k) v[i++] = (u8)w[k];
    } else {
      v[i++] = (u8)(rng() & 0xFF);
    }
  }
  return v;
}

int main(int argc, char **argv) {
  size_t n = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : (32u << 20);
  double red = (argc > 2) ? std::atof(argv[2]) : 0.85;
  std::printf("=== SYCL LZ4 standalone: %.1f MB, redundancy=%.2f ===\n",
              n / 1048576.0, red);

  std::vector<u8> orig = MakeData(n, red);
  size_t cap = n + n / 255 + 4096;
  std::vector<u8> comp(cap), decomp(n);

  void *q = clio_sycl_lz4_queue_create();
  if (!q) {
    std::printf("FAIL: no SYCL GPU queue\n");
    return 1;
  }

  // Warm up (JIT + allocations), then time.
  size_t csize = 0;
  if (!clio_sycl_lz4_compress(q, orig.data(), n, comp.data(), cap, &csize)) {
    std::printf("FAIL: compress\n");
    return 1;
  }
  auto t0 = std::chrono::high_resolution_clock::now();
  int reps = 5;
  for (int r = 0; r < reps; ++r)
    clio_sycl_lz4_compress(q, orig.data(), n, comp.data(), cap, &csize);
  auto t1 = std::chrono::high_resolution_clock::now();
  double cms = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;

  size_t dsize = 0;
  if (!clio_sycl_lz4_decompress(q, comp.data(), csize, decomp.data(), n,
                                &dsize)) {
    std::printf("FAIL: decompress\n");
    return 1;
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  for (int r = 0; r < reps; ++r)
    clio_sycl_lz4_decompress(q, comp.data(), csize, decomp.data(), n, &dsize);
  auto t3 = std::chrono::high_resolution_clock::now();
  double dms = std::chrono::duration<double, std::milli>(t3 - t2).count() / reps;

  // (a) bit-exact roundtrip
  bool rt = (dsize == n) && (std::memcmp(orig.data(), decomp.data(), n) == 0);
  std::printf("[a] roundtrip bit-exact ...... %s (out=%zu, expect %zu)\n",
              rt ? "PASS" : "FAIL", dsize, n);

  // (b) every non-raw chunk decodes with reference liblz4
  const Header *H = reinterpret_cast<const Header *>(comp.data());
  const u32 *csz = reinterpret_cast<const u32 *>(comp.data() + sizeof(Header));
  size_t hdr = sizeof(Header) + (size_t)H->n_chunks * sizeof(u32);
  const u8 *payload = comp.data() + hdr;
  u64 off = 0;
  bool oracle = true;
  int checked = 0, raws = 0;
  std::vector<u8> tmp(H->chunk_size);
  for (u32 c = 0; c < H->n_chunks; ++c) {
    u32 raw = csz[c] & ~kStoredFlag;
    bool stored = (csz[c] & kStoredFlag) != 0;
    u32 doff = c * H->chunk_size;
    u32 dlen = (doff + H->chunk_size <= n) ? H->chunk_size : (u32)(n - doff);
    if (stored) {
      ++raws;
    } else {
      int got = LZ4_decompress_safe((const char *)(payload + off),
                                    (char *)tmp.data(), (int)raw, (int)dlen);
      if (got != (int)dlen ||
          std::memcmp(tmp.data(), orig.data() + doff, dlen) != 0) {
        oracle = false;
        if (checked < 3)
          std::printf("    chunk %u: liblz4 got=%d expect=%u\n", c, got, dlen);
      }
      ++checked;
    }
    off += raw;
  }
  std::printf("[b] SYCL blocks decode via liblz4 %s (%d checked, %d raw)\n",
              oracle ? "PASS" : "FAIL", checked, raws);

  // (c) a liblz4-encoded block decodes via the SYCL decoder: wrap one liblz4
  //     block in the container and SYCL-decompress it.
  bool decoder_ok = true;
  {
    u32 clen = (n < H->chunk_size) ? (u32)n : H->chunk_size;
    std::vector<u8> lz(LZ4_compressBound(clen));
    int z = LZ4_compress_default((const char *)orig.data(), (char *)lz.data(),
                                 (int)clen, (int)lz.size());
    if (z <= 0) {
      decoder_ok = false;
    } else {
      std::vector<u8> cont(sizeof(Header) + sizeof(u32) + z);
      Header h{H->magic, H->chunk_size, clen, 1, 0};
      std::memcpy(cont.data(), &h, sizeof(Header));
      u32 cs = (u32)z;
      std::memcpy(cont.data() + sizeof(Header), &cs, sizeof(u32));
      std::memcpy(cont.data() + sizeof(Header) + sizeof(u32), lz.data(), z);
      std::vector<u8> out(clen);
      size_t os = 0;
      decoder_ok = clio_sycl_lz4_decompress(q, cont.data(), cont.size(),
                                            out.data(), clen, &os) &&
                   os == clen &&
                   std::memcmp(out.data(), orig.data(), clen) == 0;
    }
  }
  std::printf("[c] liblz4 block decodes via SYCL %s\n",
              decoder_ok ? "PASS" : "FAIL");

  double ratio = (double)n / (double)csize;
  double cgbps = (n / 1e9) / (cms / 1e3);
  double dgbps = (n / 1e9) / (dms / 1e3);
  std::printf("ratio=%.3fx  compress=%.2f GB/s (%.2f ms)  "
              "decompress=%.2f GB/s (%.2f ms)\n",
              ratio, cgbps, cms, dgbps, dms);

  clio_sycl_lz4_queue_destroy(q);
  bool all = rt && oracle && decoder_ok;
  std::printf("=== %s ===\n", all ? "ALL PASS" : "FAILURE");
  return all ? 0 : 1;
}
