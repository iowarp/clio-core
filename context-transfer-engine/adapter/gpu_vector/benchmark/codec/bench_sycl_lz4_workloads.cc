/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * sycl-lz4 (the portable nvcomp-lz4 equivalent) on the gpu_vector workloads'
 * OWN data.
 *
 * Each dataset is regenerated here, byte-for-byte, from the dependency-free
 * *_math.h headers the paged benchmarks seed from -- the same reason those
 * headers exist (every substrate must start from identical bytes). So the
 * ratios below are the ratios the paged vector's pages would see.
 *
 *   lbann      [W1|b1|W2|b2] fp32, Sym01(Lcg(seed+i))*0.05   (SeedCoro)
 *   weights    u32 Weight(i, flat_pct), flat_pct 0/50/90       (the codec bench)
 *   grayscott  [U|V] fp32 plane-major, InitU/InitV then ReactDiffuse with the
 *              paged kernel's fixed boundary; at the bench's default step
 *              count AND after the pattern has developed
 *   kmeans     fp32 PointVal(i, dims=32, k=16)
 *   gmx        u64 Q40.24 PME charge mesh after SpreadBin (K^3, 64 KiB planes)
 *
 * For each dataset and call size (gpu_vector page sizes, plus one bulk call):
 *   sycl-lz4/dev   src AND dst device USM -- a resident page, which is what
 *                  the vector hands a GPU codec; no PCIe in the timing
 *   sycl-batch/dev every page of the dataset in ONE batched call (one launch
 *                  per phase) -- how CTE's CompressBatch drives a GPU codec
 *   sycl-lz4/host  host buffers (bulk only): includes the H2D/D2H staging
 *   liblz4/1t      reference CPU LZ4, one thread
 *   liblz4/mt      reference CPU LZ4, OpenMP over the pages, all threads
 * Every row is verified: sycl roundtrip bit-exact, every non-raw sycl chunk
 * decodes with liblz4 (=> standard LZ4 blocks), liblz4 roundtrip bit-exact.
 *
 * Output: human table + "CSV," lines. Rank tag from PALS_RANKID so many
 * copies (one per tile, many nodes) can be launched by mpiexec with no MPI.
 */
#include <lz4.h>
#include <omp.h>

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../gmx/gmx_math.h"
#include "../grayscott/grayscott_math.h"
#include "../kmeans/kmeans_math.h"
#include "../lbann/lbann_math.h"
#include "../weights/weights_math.h"

extern "C" {
void *clio_sycl_lz4_queue_create();
void clio_sycl_lz4_queue_destroy(void *q);
int clio_sycl_lz4_compress(void *q, const void *src, size_t n, void *dst,
                           size_t cap, size_t *out);
int clio_sycl_lz4_decompress(void *q, const void *src, size_t n, void *dst,
                             size_t cap, size_t *out);
int clio_sycl_lz4_compress_batch(void *q, size_t n, const void *const *srcs,
                                 const size_t *src_sizes, void *const *dsts,
                                 const size_t *dst_caps, size_t *out_sizes,
                                 int *oks);
int clio_sycl_lz4_decompress_batch(void *q, size_t n, const void *const *srcs,
                                   const size_t *src_sizes, void *const *dsts,
                                   const size_t *dst_caps, size_t *out_sizes,
                                   int *oks);
}

using u8 = std::uint8_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

namespace {

struct Header {  // must match sycl_lz4_kernels.cc
  u32 magic, chunk_size;
  u64 orig_size;
  u32 n_chunks, rsvd;
};
constexpr u32 kStoredFlag = 0x80000000u;

double Now() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

struct Dataset {
  std::string name;
  std::vector<u8> bytes;
};

template <typename T>
std::vector<u8> AsBytes(const std::vector<T> &v) {
  std::vector<u8> b(v.size() * sizeof(T));
  std::memcpy(b.data(), v.data(), b.size());
  return b;
}

// ---- generators (mirror the paged benchmarks' seeds) ----------------------
Dataset Lbann(u64 I, u64 H, u64 O) {
  const u64 n = I * H + H + H * O + O;  // [W1|b1|W2|b2] in one vector
  std::vector<float> w(n);
#pragma omp parallel for
  for (u64 i = 0; i < n; ++i)
    w[i] = clio_lb::Sym01(clio_lb::Lcg(0xB5297A4D3F84D5B5ull + i)) * 0.05f;
  return {"lbann", AsBytes(w)};
}

Dataset Weights(u64 bytes, u32 pct) {
  std::vector<u32> w(bytes / 4);
#pragma omp parallel for
  for (u64 i = 0; i < w.size(); ++i) w[i] = clio_wt::Weight(i, pct);
  return {"weights_flat" + std::to_string(pct), AsBytes(w)};
}

Dataset Kmeans(u64 bytes) {
  std::vector<float> p(bytes / 4);
#pragma omp parallel for
  for (u64 i = 0; i < p.size(); ++i) p[i] = clio_km::PointVal(i, 32, 16);
  return {"kmeans", AsBytes(p)};
}

// The paged StepCoro: 7-point Laplacian, lap = 0 on every boundary face.
std::vector<Dataset> GrayScott(u64 nx, u32 bench_steps, u32 dev_steps) {
  const u64 plane = nx * nx, cells = plane * nx;
  const float Du = 0.2f, Dv = 0.1f, F = 0.02f, K = 0.048f, dt = 0.5f;
  std::vector<float> u(cells), v(cells), un(cells), vn(cells);
#pragma omp parallel for
  for (u64 c = 0; c < cells; ++c) {
    const u64 z = c / plane, i = c % plane;
    u[c] = clio_gs::InitU(i % nx, i / nx, z, nx, nx, nx);
    v[c] = clio_gs::InitV(i % nx, i / nx, z, nx, nx, nx);
  }
  auto pack = [&](const char *tag) {
    std::vector<float> uv(2 * cells);
    std::memcpy(uv.data(), u.data(), cells * 4);
    std::memcpy(uv.data() + cells, v.data(), cells * 4);
    return Dataset{std::string("grayscott_") + tag, AsBytes(uv)};
  };
  std::vector<Dataset> out;
  for (u32 s = 1; s <= dev_steps; ++s) {
#pragma omp parallel for
    for (u64 c = 0; c < cells; ++c) {
      const u64 z = c / plane, i = c % plane, x = i % nx, y = i / nx;
      float lu = 0, lv = 0;
      if (!(x == 0 || x + 1 == nx || y == 0 || y + 1 == nx || z == 0 ||
            z + 1 == nx)) {
        lu = u[c - 1] + u[c + 1] + u[c - nx] + u[c + nx] + u[c - plane] +
             u[c + plane] - 6.0f * u[c];
        lv = v[c - 1] + v[c + 1] + v[c - nx] + v[c + nx] + v[c - plane] +
             v[c + plane] - 6.0f * v[c];
      }
      clio_gs::ReactDiffuse(u[c], v[c], lu, lv, Du, Dv, F, K, dt, &un[c],
                            &vn[c]);
    }
    u.swap(un);
    v.swap(vn);
    if (s == bench_steps)
      out.push_back(pack(("step" + std::to_string(s)).c_str()));
  }
  out.push_back(pack(("step" + std::to_string(dev_steps)).c_str()));
  return out;
}

// SpreadBin, serially: the mesh is integer fixed point, so the order of the
// adds does not change a single bit -- the property the paged gates rely on.
Dataset Gmx(u64 K, u64 atoms) {
  using namespace clio_gmx;
  std::vector<unsigned long long> mesh(K * K * K, 0);
  u64 s = 0x9E3779B97F4A7C15ull;
  const int Ki = static_cast<int>(K);
  for (u64 a = 0; a < atoms; ++a) {
    s = Lcg(s); const float x = Frac01(s) * static_cast<float>(K);
    s = Lcg(s); const float y = Frac01(s) * static_cast<float>(K);
    s = Lcg(s); const float z = Frac01(s) * static_cast<float>(K);
    const long long q = (a & 1) ? -(1ll << 24) : (1ll << 24);
    const int ix0 = static_cast<int>(std::floor(x)) - 1;
    const int iy0 = static_cast<int>(std::floor(y)) - 1;
    const int iz0 = static_cast<int>(std::floor(z)) - 1;
    float wx[4], wy[4], wz[4];
    Spline4(x - std::floor(x), wx);
    Spline4(y - std::floor(y), wy);
    Spline4(z - std::floor(z), wz);
    long long qz4[4], run = 0;
    for (int j = 0; j < 3; ++j) {
      qz4[j] = FxRound(static_cast<double>(q) * wz[j]);
      run += qz4[j];
    }
    qz4[3] = q - run;
    for (int jz = 0; jz < 4; ++jz) {
      const u64 pz = static_cast<u64>((iz0 + jz + Ki) % Ki);
      const long long qz = qz4[jz];
      long long xy_run = 0;
      for (int jy = 0; jy < 4; ++jy) {
        const u64 gy = static_cast<u64>((iy0 + jy + Ki) % Ki);
        for (int jx = 0; jx < 4; ++jx) {
          const u64 gx = static_cast<u64>((ix0 + jx + Ki) % Ki);
          const long long vv =
              (jy == 3 && jx == 3)
                  ? qz - xy_run
                  : FxRound(static_cast<double>(qz) * wy[jy] * wx[jx]);
          xy_run += vv;
          mesh[pz * K * K + gy * K + gx] += static_cast<unsigned long long>(vv);
        }
      }
    }
  }
  return {"gmx_mesh", AsBytes(mesh)};
}

// ---- measurement ----------------------------------------------------------
struct Row {
  std::string ds, codec;
  u64 bytes, call;
  double ratio, c_gbs, d_gbs;
  bool ok;
};

// Check that every non-raw chunk of a sycl-lz4 stream is a standard LZ4 block.
bool OracleOk(const u8 *stream, const u8 *orig, u64 n) {
  const Header *H = reinterpret_cast<const Header *>(stream);
  const u32 *cs = reinterpret_cast<const u32 *>(stream + sizeof(Header));
  const u8 *p = stream + sizeof(Header) + (u64)H->n_chunks * 4;
  std::vector<u8> tmp(H->chunk_size);
  u64 off = 0;
  for (u32 c = 0; c < H->n_chunks; ++c) {
    const u32 raw = cs[c] & ~kStoredFlag;
    const u64 doff = (u64)c * H->chunk_size;
    const u32 dlen = (u32)std::min<u64>(H->chunk_size, n - doff);
    if (!(cs[c] & kStoredFlag)) {
      int got = LZ4_decompress_safe((const char *)p + off, (char *)tmp.data(),
                                    (int)raw, (int)dlen);
      if (got != (int)dlen || std::memcmp(tmp.data(), orig + doff, dlen))
        return false;
    }
    off += raw;
  }
  return true;
}

Row SyclDev(sycl::queue &q, void *codec, const Dataset &d, u64 call,
            int reps) {
  const u64 n = d.bytes.size(), pages = (n + call - 1) / call;
  const u64 slot = call + call / 255 + 64 + (call / 4096 + 1) * 4 + 64;
  u8 *src = sycl::malloc_device<u8>(n, q);
  u8 *cmp = sycl::malloc_device<u8>(pages * slot, q);
  u8 *out = sycl::malloc_device<u8>(n, q);
  q.memcpy(src, d.bytes.data(), n).wait();
  std::vector<size_t> cs(pages);
  bool ok = true;
  auto comp_all = [&] {
    for (u64 p = 0; p < pages; ++p) {
      const u64 len = std::min(call, n - p * call);
      ok &= clio_sycl_lz4_compress(codec, src + p * call, len, cmp + p * slot,
                                   slot, &cs[p]) == 1;
    }
  };
  auto decomp_all = [&] {
    for (u64 p = 0; p < pages; ++p) {
      const u64 len = std::min(call, n - p * call);
      size_t o = 0;
      ok &= clio_sycl_lz4_decompress(codec, cmp + p * slot, cs[p],
                                     out + p * call, len, &o) == 1 &&
            o == len;
    }
  };
  comp_all();  // warm-up: JIT-free (AOT) but first-touch + allocator
  decomp_all();
  double tc = 1e30, td = 1e30;
  for (int r = 0; r < reps; ++r) {
    double t0 = Now();
    comp_all();
    double t1 = Now();
    decomp_all();
    double t2 = Now();
    tc = std::min(tc, t1 - t0);
    td = std::min(td, t2 - t1);
  }
  u64 total = 0;
  for (auto c : cs) total += c;
  std::vector<u8> back(n);
  q.memcpy(back.data(), out, n).wait();
  ok &= std::memcmp(back.data(), d.bytes.data(), n) == 0;
  // Oracle on the first page's stream (bulk => the whole dataset).
  {
    std::vector<u8> s0(cs[0]);
    q.memcpy(s0.data(), cmp, cs[0]).wait();
    ok &= OracleOk(s0.data(), d.bytes.data(), std::min(call, n));
  }
  sycl::free(src, q);
  sycl::free(cmp, q);
  sycl::free(out, q);
  return {d.name, "sycl-lz4/dev", n, call, (double)total / n,
          n / tc / 1e9, n / td / 1e9, ok};
}

Row SyclBatchDev(sycl::queue &q, void *codec, const Dataset &d, u64 call,
                 int reps) {
  const u64 n = d.bytes.size(), pages = (n + call - 1) / call;
  const u64 slot = call + call / 255 + 64 + (call / 4096 + 1) * 4 + 64;
  u8 *src = sycl::malloc_device<u8>(n, q);
  u8 *cmp = sycl::malloc_device<u8>(pages * slot, q);
  u8 *out = sycl::malloc_device<u8>(n, q);
  q.memcpy(src, d.bytes.data(), n).wait();
  std::vector<const void *> cs_src(pages), ds_src(pages);
  std::vector<void *> cs_dst(pages), ds_dst(pages);
  std::vector<size_t> len(pages), cap(pages, slot), csz(pages), dsz(pages);
  std::vector<int> ok1(pages), ok2(pages);
  for (u64 p = 0; p < pages; ++p) {
    len[p] = std::min(call, n - p * call);
    cs_src[p] = src + p * call;
    cs_dst[p] = cmp + p * slot;
    ds_src[p] = cmp + p * slot;
    ds_dst[p] = out + p * call;
  }
  bool ok = true;
  auto comp = [&] {
    ok &= clio_sycl_lz4_compress_batch(codec, pages, cs_src.data(), len.data(),
                                       cs_dst.data(), cap.data(), csz.data(),
                                       ok1.data()) == 1;
  };
  auto decomp = [&] {
    ok &= clio_sycl_lz4_decompress_batch(codec, pages, ds_src.data(),
                                         csz.data(), ds_dst.data(), len.data(),
                                         dsz.data(), ok2.data()) == 1;
  };
  comp();
  decomp();
  double tc = 1e30, td = 1e30;
  for (int r = 0; r < reps; ++r) {
    double t0 = Now();
    comp();
    double t1 = Now();
    decomp();
    double t2 = Now();
    tc = std::min(tc, t1 - t0);
    td = std::min(td, t2 - t1);
  }
  u64 total = 0;
  for (u64 p = 0; p < pages; ++p) {
    total += csz[p];
    ok &= dsz[p] == len[p];
  }
  std::vector<u8> back(n);
  q.memcpy(back.data(), out, n).wait();
  ok &= std::memcmp(back.data(), d.bytes.data(), n) == 0;
  {
    std::vector<u8> s0(csz[0]);
    q.memcpy(s0.data(), cmp, csz[0]).wait();
    ok &= OracleOk(s0.data(), d.bytes.data(), len[0]);
  }
  sycl::free(src, q);
  sycl::free(cmp, q);
  sycl::free(out, q);
  return {d.name, "sycl-batch/dev", n, call, (double)total / n, n / tc / 1e9,
          n / td / 1e9, ok};
}

Row SyclHost(void *codec, const Dataset &d, int reps) {
  const u64 n = d.bytes.size(), cap = n + n / 255 + (n / 4096 + 1) * 4 + 4096;
  std::vector<u8> cmp(cap), out(n);
  size_t cs = 0, o = 0;
  bool ok = clio_sycl_lz4_compress(codec, d.bytes.data(), n, cmp.data(), cap,
                                   &cs) == 1;
  double tc = 1e30, td = 1e30;
  for (int r = 0; r < reps && ok; ++r) {
    double t0 = Now();
    ok &= clio_sycl_lz4_compress(codec, d.bytes.data(), n, cmp.data(), cap,
                                 &cs) == 1;
    double t1 = Now();
    ok &= clio_sycl_lz4_decompress(codec, cmp.data(), cs, out.data(), n, &o) ==
          1;
    double t2 = Now();
    tc = std::min(tc, t1 - t0);
    td = std::min(td, t2 - t1);
  }
  ok &= o == n && std::memcmp(out.data(), d.bytes.data(), n) == 0;
  return {d.name, "sycl-lz4/host", n, n, (double)cs / n, n / tc / 1e9,
          n / td / 1e9, ok};
}

Row Cpu(const Dataset &d, u64 call, int threads, int reps) {
  const u64 n = d.bytes.size(), pages = (n + call - 1) / call;
  const u64 slot = (u64)LZ4_compressBound((int)std::min<u64>(call, n));
  std::vector<u8> cmp(pages * slot), out(n);
  std::vector<int> cs(pages);
  const char *src = (const char *)d.bytes.data();
  double tc = 1e30, td = 1e30;
  bool ok = true;
  for (int r = 0; r <= reps; ++r) {
    double t0 = Now();
#pragma omp parallel for num_threads(threads) schedule(dynamic)
    for (u64 p = 0; p < pages; ++p) {
      const int len = (int)std::min(call, n - p * call);
      cs[p] = LZ4_compress_default(src + p * call, (char *)cmp.data() + p * slot,
                                   len, (int)slot);
    }
    double t1 = Now();
    int bad = 0;
#pragma omp parallel for num_threads(threads) schedule(dynamic) reduction(+ : bad)
    for (u64 p = 0; p < pages; ++p) {
      const int len = (int)std::min(call, n - p * call);
      bad += LZ4_decompress_safe((const char *)cmp.data() + p * slot,
                                 (char *)out.data() + p * call, cs[p],
                                 len) != len;
    }
    double t2 = Now();
    ok &= bad == 0;
    if (r) {  // r == 0 is warm-up
      tc = std::min(tc, t1 - t0);
      td = std::min(td, t2 - t1);
    }
  }
  u64 total = 0;
  for (int c : cs) total += (u64)c;
  ok &= std::memcmp(out.data(), d.bytes.data(), n) == 0;
  return {d.name, threads == 1 ? "liblz4/1t" : "liblz4/mt", n, call,
          (double)total / n, n / tc / 1e9, n / td / 1e9, ok};
}

}  // namespace

int main(int argc, char **argv) {
  u64 mb = 64;            // size of the scalable datasets
  u64 gs_n = 128;         // grayscott edge (2 * n^3 * 4 B)
  u32 gs_dev_steps = 2000;
  int reps = 3;
  std::string only;
  std::vector<u64> calls = {64 << 10, 1 << 20};  // gpu_vector page sizes
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&] { return std::strtoull(argv[++i], nullptr, 10); };
    if (a == "--mb") mb = next();
    else if (a == "--gs-n") gs_n = next();
    else if (a == "--gs-steps") gs_dev_steps = (u32)next();
    else if (a == "--reps") reps = (int)next();
    else if (a == "--only") only = argv[++i];
    else if (a == "--calls-kb") {
      calls.clear();
      std::string s = argv[++i];
      for (size_t p = 0; p < s.size();) {
        size_t e = s.find(',', p);
        calls.push_back(std::strtoull(s.substr(p, e - p).c_str(), nullptr, 10)
                        << 10);
        p = e == std::string::npos ? s.size() : e + 1;
      }
    }
  }
  const char *rk = std::getenv("PALS_RANKID");
  const std::string rank = rk ? rk : "0";
  const int nthr = omp_get_max_threads();

  sycl::queue q{sycl::gpu_selector_v};
  void *codec = clio_sycl_lz4_queue_create();
  if (!codec) {
    std::printf("FAIL: no SYCL GPU queue\n");
    return 1;
  }
  std::printf("[r%s] device: %s | host threads: %d | mask=%s\n", rank.c_str(),
              q.get_device().get_info<sycl::info::device::name>().c_str(), nthr,
              std::getenv("ZE_AFFINITY_MASK") ? std::getenv("ZE_AFFINITY_MASK")
                                              : "-");

  std::vector<Dataset> ds;
  auto want = [&](const char *n) {
    return only.empty() || only.find(n) != std::string::npos;
  };
  // lbann: --in 1024 --hidden 16384 is the README's way to a multi-GB-class
  // footprint; at the defaults (256/4096/64) the vector is only 4 MiB.
  if (want("lbann")) ds.push_back(Lbann(1024, mb * 256, 64));
  if (want("weights"))
    for (u32 p : {0u, 50u, 90u}) ds.push_back(Weights(mb << 20, p));
  if (want("grayscott"))
    for (auto &d : GrayScott(gs_n, 4, gs_dev_steps)) ds.push_back(std::move(d));
  if (want("kmeans")) ds.push_back(Kmeans(mb << 20));
  if (want("gmx")) ds.push_back(Gmx(90, 200000));  // bench defaults

  std::vector<Row> rows;
  for (const auto &d : ds) {
    std::vector<u64> cl = calls;
    cl.push_back(d.bytes.size());  // one bulk call
    for (u64 c : cl) {
      if (c != d.bytes.size()) rows.push_back(SyclBatchDev(q, codec, d, c, reps));
      rows.push_back(SyclDev(q, codec, d, c, reps));
      rows.push_back(Cpu(d, c, 1, reps));
      rows.push_back(Cpu(d, c, nthr, reps));
    }
    rows.push_back(SyclHost(codec, d, reps));
    std::fflush(stdout);
  }

  std::printf("\n%-20s %-14s %9s %9s %7s %9s %9s %s\n", "dataset", "codec",
              "MiB", "call", "ratio", "comp GB/s", "dec GB/s", "ok");
  bool all = true;
  for (const auto &r : rows) {
    std::string call = r.call == r.bytes ? "bulk"
                       : r.call >= (1 << 20)
                           ? std::to_string(r.call >> 20) + "M"
                           : std::to_string(r.call >> 10) + "K";
    std::printf("%-20s %-14s %9.1f %9s %7.3f %9.2f %9.2f %s\n", r.ds.c_str(),
                r.codec.c_str(), r.bytes / 1048576.0, call.c_str(), r.ratio,
                r.c_gbs, r.d_gbs, r.ok ? "PASS" : "FAIL");
    all &= r.ok;
  }
  for (const auto &r : rows)
    std::printf("CSV,%s,%s,%s,%llu,%llu,%.5f,%.4f,%.4f,%d\n", rank.c_str(),
                r.ds.c_str(), r.codec.c_str(), (unsigned long long)r.bytes,
                (unsigned long long)r.call, r.ratio, r.c_gbs, r.d_gbs, r.ok);
  std::printf("[r%s] %s\n", rank.c_str(), all ? "ALL PASS" : "SOME FAILED");
  clio_sycl_lz4_queue_destroy(codec);
  return all ? 0 : 2;
}
