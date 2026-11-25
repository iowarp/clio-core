/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * The SYCL GPU codecs (sycl-lz4, sycl-snappy, sycl-ans, sycl-zstd -- the
 * portable counterparts of nvcomp's LZ4/Snappy/ANS/Zstd) on the gpu_vector
 * workloads' OWN data.
 *
 * Each dataset is regenerated here, byte-for-byte, from the seed functions the
 * paged benchmarks use (the dependency-free *_math.h headers, or a verbatim
 * copy where the seed lives in a benchmark source), so the ratios below are
 * the ratios the paged vector's pages would see:
 *
 *   lbann      [W1|b1|W2|b2] fp32, Sym01(Lcg(seed+i))*0.05        (SeedCoro)
 *   weights    u32 Weight(i, flat_pct), flat_pct 0/50/90           (codec bench)
 *   grayscott  [U|V] fp32 plane-major, fixed boundary; at the bench's default
 *              step count AND after the pattern has developed
 *   kmeans     fp32 PointVal(i, dims=32, k=16)
 *   gmx        u64 Q40.24 PME charge mesh after SpreadBin (K=90, 64 KiB planes)
 *   lammps_md  x and v: FCC lattice in padded bins (BuildInitialState)
 *   gnn        fp32 features FeatureAt(row, f, F=64)
 *
 * Rows, per dataset and codec:
 *   gpu/page   one call per 64 KiB page, src+dst device USM (a resident page,
 *              what the vector hands a GPU codec; no PCIe in the timing)
 *   gpu/batch  every page in ONE batched call (codecs with a batch API)
 *   gpu/bulk   one call over the whole dataset (the codec's ceiling)
 *   cpu1/cpu8  the reference CPU library for the same format (liblz4,
 *              libsnappy; libzstd level 1 for zstd; none for ANS) per 64 KiB
 *              page, 1 thread and the tile's 8-core share
 * Every GPU row is verified: bit-exact roundtrip, and for LZ4/Snappy every
 * non-raw chunk must decode with the reference library (standard blocks).
 *
 * ratio = compressed / original (lower is better).
 * Output: human table + "CSV," lines, rank-tagged from PALS_RANKID so one
 * copy per tile can be launched by mpiexec with no MPI linked.
 */
#include <lz4.h>
#include <omp.h>
#include <snappy.h>
#include <zstd.h>

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

#include <unistd.h>

#include "../gmx/gmx_math.h"
#include "../grayscott/grayscott_math.h"
#include "../kmeans/kmeans_math.h"
#include "../lbann/lbann_math.h"
#include "../weights/weights_math.h"

using u8 = std::uint8_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

#define SYCL_CODEC_DECLS(c)                                                    \
  void *clio_sycl_##c##_queue_create();                                        \
  void clio_sycl_##c##_queue_destroy(void *q);                                 \
  int clio_sycl_##c##_compress(void *q, const void *src, size_t n, void *dst,  \
                               size_t cap, size_t *out);                       \
  int clio_sycl_##c##_decompress(void *q, const void *src, size_t n,           \
                                 void *dst, size_t cap, size_t *out);
extern "C" {
SYCL_CODEC_DECLS(lz4)
SYCL_CODEC_DECLS(snappy)
SYCL_CODEC_DECLS(ans)
SYCL_CODEC_DECLS(zstd)
int clio_sycl_lz4_compress_batch(void *q, size_t n, const void *const *srcs,
                                 const size_t *src_sizes, void *const *dsts,
                                 const size_t *dst_caps, size_t *out_sizes,
                                 int *oks);
int clio_sycl_lz4_decompress_batch(void *q, size_t n, const void *const *srcs,
                                   const size_t *src_sizes, void *const *dsts,
                                   const size_t *dst_caps, size_t *out_sizes,
                                   int *oks);
}

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


// lammps_md: BuildInitialState + InitVelocity from clio_lammps_md_paged_newcoro.cc
// (rho 0.8442, cutoff 2.5, skin 0.3, cap 32, kStride 4: x,y,z,type).
std::vector<Dataset> LammpsMd(u32 lattice) {
  const double rho = 0.8442, cutoff = 2.5, skin = 0.3;
  const u32 cap = 32, kStride = 4;
  const double box = lattice * std::cbrt(4.0 / rho);
  u32 nb = static_cast<u32>(box / (cutoff + skin));
  if (nb == 0) nb = 1;
  const double bin_edge = box / nb;
  const u64 nbins = (u64)nb * nb * nb, nslots = nbins * cap;
  std::vector<float> hx(nslots * kStride, 0.0f), hv(nslots * kStride, 0.0f);
  for (u64 s = 0; s < nslots; ++s) hx[s * kStride + 3] = -1.0f;
  std::vector<u32> cnt(nbins, 0);
  const double cell = box / lattice;
  static const double kB[4][3] = {
      {0.0, 0.0, 0.0}, {0.5, 0.5, 0.0}, {0.5, 0.0, 0.5}, {0.0, 0.5, 0.5}};
  u64 id = 0;
  for (u32 i = 0; i < lattice; ++i)
    for (u32 j = 0; j < lattice; ++j)
      for (u32 k = 0; k < lattice; ++k)
        for (int b = 0; b < 4; ++b) {
          const double px = (i + kB[b][0]) * cell, py = (j + kB[b][1]) * cell,
                       pz = (k + kB[b][2]) * cell;
          u32 bx = std::min<u32>((u32)(px / bin_edge), nb - 1);
          u32 by = std::min<u32>((u32)(py / bin_edge), nb - 1);
          u32 bz = std::min<u32>((u32)(pz / bin_edge), nb - 1);
          const u64 bin = ((u64)bz * nb + by) * nb + bx;
          const u32 slot = cnt[bin]++;
          if (slot >= cap) { std::fprintf(stderr, "md: bin overflow\n"); std::exit(3); }
          const u64 e = (bin * cap + slot) * kStride;
          hx[e] = (float)px; hx[e + 1] = (float)py; hx[e + 2] = (float)pz;
          hx[e + 3] = 1.0f;
          const double a = (double)id++;
          hv[e] = (float)(0.20 * std::sin(0.37 * a + 0.1));
          hv[e + 1] = (float)(0.20 * std::cos(0.53 * a + 0.7));
          hv[e + 2] = (float)(0.20 * std::sin(0.71 * a + 1.3));
        }
  return {{"lammps_md_x", AsBytes(hx)}, {"lammps_md_v", AsBytes(hv)}};
}

// gnn: FeatureAt / Mix64 from gnn_kernels.h, F = 64 (the bench default).
Dataset Gnn(u64 bytes) {
  const u64 F = 64, rows = bytes / 4 / F;
  std::vector<float> x(rows * F);
#pragma omp parallel for
  for (u64 i = 0; i < rows * F; ++i) {
    u64 z = i + 0x9E3779B97F4A7C15ull;  // Mix64(row * F + f), row*F+f == i
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    x[i] = (float)((z >> 40) & 0xFFFFFFull) * (2.0f / 16777216.0f) - 1.0f;
  }
  return {"gnn_features", AsBytes(x)};
}

// ---- codecs -----------------------------------------------------------------
using QCreate = void *(*)();
using QDestroy = void (*)(void *);
using QCall = int (*)(void *, const void *, size_t, void *, size_t, size_t *);
enum class Oracle { kNone, kLz4, kSnappy };
struct GpuCodec {
  const char *name;
  QCreate create;
  QDestroy destroy;
  QCall comp, decomp;
  bool batch;
  Oracle oracle;
  const char *cpu;  // reference CPU format, or nullptr
};
#define GPU_CODEC(c, b, o, cpu)                                                \
  GpuCodec{"sycl-" #c, &clio_sycl_##c##_queue_create,                          \
           &clio_sycl_##c##_queue_destroy, &clio_sycl_##c##_compress,          \
           &clio_sycl_##c##_decompress, b, o, cpu}
const GpuCodec kCodecs[] = {
    GPU_CODEC(lz4, true, Oracle::kLz4, "lz4"),
    GPU_CODEC(snappy, false, Oracle::kSnappy, "snappy"),
    GPU_CODEC(ans, false, Oracle::kNone, nullptr),
    GPU_CODEC(zstd, false, Oracle::kNone, "zstd1"),
};

// Every non-raw chunk of a stream must decode with the reference library.
bool OracleOk(Oracle o, const u8 *stream, const u8 *orig, u64 n) {
  if (o == Oracle::kNone) return true;
  const Header *H = reinterpret_cast<const Header *>(stream);
  const u32 *cs = reinterpret_cast<const u32 *>(stream + sizeof(Header));
  const u8 *p = stream + sizeof(Header) + (u64)H->n_chunks * 4;
  std::vector<char> tmp(H->chunk_size);
  u64 off = 0;
  for (u32 c = 0; c < H->n_chunks; ++c) {
    const u32 raw = cs[c] & ~kStoredFlag;
    const u64 doff = (u64)c * H->chunk_size;
    const u32 dlen = (u32)std::min<u64>(H->chunk_size, n - doff);
    if (!(cs[c] & kStoredFlag)) {
      bool ok;
      if (o == Oracle::kLz4) {
        ok = LZ4_decompress_safe((const char *)p + off, tmp.data(), (int)raw,
                                 (int)dlen) == (int)dlen;
      } else {
        size_t ul = 0;
        ok = snappy::GetUncompressedLength((const char *)p + off, raw, &ul) &&
             ul == dlen &&
             snappy::RawUncompress((const char *)p + off, raw, tmp.data());
      }
      if (!ok || std::memcmp(tmp.data(), orig + doff, dlen)) return false;
    }
    off += raw;
  }
  return true;
}

struct Row {
  std::string ds, codec, mode;
  u64 bytes, call;
  double ratio, c_gbs, d_gbs;
  bool ok;
};

// GPU: pages of `call` bytes (call == n => one bulk call), device-resident.
Row Gpu(sycl::queue &q, const GpuCodec &cd, void *codec, const Dataset &d,
        u64 call, bool batch, int reps) {
  const u64 n = d.bytes.size(), pages = (n + call - 1) / call;
  const u64 slot = call + call / 16 + 4096;
  u8 *src = sycl::malloc_device<u8>(n, q);
  u8 *cmp = sycl::malloc_device<u8>(pages * slot, q);
  u8 *out = sycl::malloc_device<u8>(n, q);
  q.memcpy(src, d.bytes.data(), n).wait();
  q.memset(out, 0, n).wait();
  std::vector<const void *> csrc(pages), dsrc(pages);
  std::vector<void *> cdst(pages), ddst(pages);
  std::vector<size_t> len(pages), cap(pages, slot), csz(pages), dsz(pages);
  std::vector<int> ok1(pages), ok2(pages);
  for (u64 p = 0; p < pages; ++p) {
    len[p] = std::min(call, n - p * call);
    csrc[p] = src + p * call; cdst[p] = cmp + p * slot;
    dsrc[p] = cmp + p * slot; ddst[p] = out + p * call;
  }
  bool ok = true;
  auto comp = [&] {
    if (batch) {
      ok &= clio_sycl_lz4_compress_batch(codec, pages, csrc.data(), len.data(),
                                         cdst.data(), cap.data(), csz.data(),
                                         ok1.data()) == 1;
    } else {
      for (u64 p = 0; p < pages; ++p)
        ok &= cd.comp(codec, csrc[p], len[p], cdst[p], slot, &csz[p]) == 1;
    }
  };
  auto decomp = [&] {
    if (batch) {
      ok &= clio_sycl_lz4_decompress_batch(codec, pages, dsrc.data(),
                                           csz.data(), ddst.data(), len.data(),
                                           dsz.data(), ok2.data()) == 1;
    } else {
      for (u64 p = 0; p < pages; ++p)
        ok &= cd.decomp(codec, dsrc[p], csz[p], ddst[p], len[p], &dsz[p]) == 1;
    }
  };
  comp();  // warm-up
  decomp();
  double tc = 1e30, td = 1e30;
  for (int r = 0; r < reps && ok; ++r) {
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
  if (ok) {
    std::vector<u8> s0(csz[0]);
    q.memcpy(s0.data(), cmp, csz[0]).wait();
    ok &= OracleOk(cd.oracle, s0.data(), d.bytes.data(), len[0]);
  }
  sycl::free(src, q);
  sycl::free(cmp, q);
  sycl::free(out, q);
  const char *mode = call >= n ? "gpu/bulk" : batch ? "gpu/batch" : "gpu/page";
  return {d.name, cd.name, mode, n, call, (double)total / n,
          ok ? n / tc / 1e9 : 0, ok ? n / td / 1e9 : 0, ok};
}

// Reference CPU library for the same format, per page, OpenMP over pages.
Row Cpu(const char *fmt, const Dataset &d, u64 call, int threads, int reps) {
  const u64 n = d.bytes.size(), pages = (n + call - 1) / call;
  const std::string f = fmt;
  const u64 slot = std::max<u64>({(u64)LZ4_compressBound((int)call),
                                  snappy::MaxCompressedLength(call),
                                  ZSTD_compressBound(call)});
  std::vector<char> cmp(pages * slot), out(n);
  std::vector<size_t> cs(pages);
  const char *src = (const char *)d.bytes.data();
  double tc = 1e30, td = 1e30;
  int bad = 0;
  for (int r = 0; r <= reps; ++r) {
    double t0 = Now();
#pragma omp parallel for num_threads(threads) schedule(dynamic)
    for (u64 p = 0; p < pages; ++p) {
      const size_t len = std::min(call, n - p * call);
      char *dst = cmp.data() + p * slot;
      if (f == "lz4") {
        cs[p] = (size_t)LZ4_compress_default(src + p * call, dst, (int)len,
                                             (int)slot);
      } else if (f == "snappy") {
        snappy::RawCompress(src + p * call, len, dst, &cs[p]);
      } else {
        cs[p] = ZSTD_compress(dst, slot, src + p * call, len, 1);
      }
    }
    double t1 = Now();
    int b = 0;
#pragma omp parallel for num_threads(threads) schedule(dynamic) reduction(+ : b)
    for (u64 p = 0; p < pages; ++p) {
      const size_t len = std::min(call, n - p * call);
      const char *s = cmp.data() + p * slot;
      char *o = out.data() + p * call;
      if (f == "lz4") {
        b += LZ4_decompress_safe(s, o, (int)cs[p], (int)len) != (int)len;
      } else if (f == "snappy") {
        b += !snappy::RawUncompress(s, cs[p], o);
      } else {
        b += ZSTD_decompress(o, len, s, cs[p]) != len;
      }
    }
    double t2 = Now();
    bad += b;
    if (r) {
      tc = std::min(tc, t1 - t0);
      td = std::min(td, t2 - t1);
    }
  }
  u64 total = 0;
  for (size_t c : cs) total += c;
  const bool ok = bad == 0 && std::memcmp(out.data(), src, n) == 0;
  return {d.name, f == "zstd1" ? "zstd-1" : fmt,
          threads == 1 ? "cpu1/page" : "cpu8/page", n, call,
          (double)total / n, n / tc / 1e9, n / td / 1e9, ok};
}

}  // namespace

int main(int argc, char **argv) {
  u64 mb = 32, page_kb = 64, gs_n = 128;
  u32 gs_dev_steps = 2000, md_lattice = 48;
  int reps = 3, page_reps = 1;
  std::string only, codecs, dump;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&] { return std::strtoull(argv[++i], nullptr, 10); };
    if (a == "--mb") mb = next();
    else if (a == "--page-kb") page_kb = next();
    else if (a == "--gs-n") gs_n = next();
    else if (a == "--gs-steps") gs_dev_steps = (u32)next();
    else if (a == "--md-lattice") md_lattice = (u32)next();
    else if (a == "--reps") reps = (int)next();
    else if (a == "--page-reps") page_reps = (int)next();
    else if (a == "--only") only = argv[++i];
    else if (a == "--codecs") codecs = argv[++i];
    else if (a == "--dump") dump = argv[++i];
  }
  const u64 page = page_kb << 10;
  const char *rk = std::getenv("PALS_RANKID");
  const std::string rank = rk ? rk : "0";
  const int nthr = omp_get_max_threads();
  sycl::queue q{sycl::gpu_selector_v};
  char host[256] = {0};
  gethostname(host, sizeof(host) - 1);
  std::printf("[r%s] %s device: %s | host threads: %d | mask=%s\n",
              rank.c_str(), host,
              q.get_device().get_info<sycl::info::device::name>().c_str(), nthr,
              std::getenv("ZE_AFFINITY_MASK") ? std::getenv("ZE_AFFINITY_MASK")
                                              : "-");
  auto want = [&](const std::string &filter, const std::string &n) {
    return filter.empty() || filter.find(n) != std::string::npos;
  };
  std::vector<Dataset> ds;
  if (want(only, "lbann")) ds.push_back(Lbann(1024, mb * 256, 64));
  if (want(only, "weights"))
    for (u32 p : {0u, 50u, 90u}) ds.push_back(Weights(mb << 20, p));
  if (want(only, "grayscott"))
    for (auto &d : GrayScott(gs_n, 4, gs_dev_steps)) ds.push_back(std::move(d));
  if (want(only, "kmeans")) ds.push_back(Kmeans(mb << 20));
  if (want(only, "gmx")) ds.push_back(Gmx(90, 200000));
  if (want(only, "lammps_md"))
    for (auto &d : LammpsMd(md_lattice)) ds.push_back(std::move(d));
  if (want(only, "gnn")) ds.push_back(Gnn(mb << 20));

  // Dump each dataset's exact bytes so an external codec (e.g. nvcomp, which
  // is CUDA-only and cannot link here) compresses the SAME data the SYCL
  // codecs see. These are the vector's page bytes, verbatim.
  if (!dump.empty()) {
    for (const auto &d : ds) {
      std::string path = dump + "/" + d.name + ".bin";
      std::FILE *f = std::fopen(path.c_str(), "wb");
      if (!f) { std::printf("FAIL: cannot open %s\n", path.c_str()); return 1; }
      std::fwrite(d.bytes.data(), 1, d.bytes.size(), f);
      std::fclose(f);
      std::printf("DUMP,%s,%llu\n", path.c_str(),
                  (unsigned long long)d.bytes.size());
    }
    std::fflush(stdout);
  }

  std::vector<Row> rows;
  bool all = true;
  auto add = [&](Row r) {
    std::printf("CSV,%s,%s,%s,%s,%s,%llu,%llu,%.5f,%.4f,%.4f,%d\n",
                rank.c_str(), host, r.ds.c_str(), r.codec.c_str(),
                r.mode.c_str(), (unsigned long long)r.bytes,
                (unsigned long long)r.call, r.ratio, r.c_gbs, r.d_gbs, r.ok);
    std::fflush(stdout);
    all &= r.ok;
    rows.push_back(std::move(r));
  };
  std::printf("CSV,rank,host,dataset,codec,mode,bytes,call_bytes,ratio,"
              "comp_GBps,decomp_GBps,ok\n");
  for (const auto &cd : kCodecs) {
    if (!want(codecs, cd.name + 5)) continue;  // filter on "lz4", "ans", ...
    void *codec = cd.create();
    if (!codec) { std::printf("FAIL: %s queue\n", cd.name); return 1; }
    for (const auto &d : ds) {
      add(Gpu(q, cd, codec, d, page, false, page_reps));
      if (cd.batch) add(Gpu(q, cd, codec, d, page, true, reps));
      add(Gpu(q, cd, codec, d, d.bytes.size(), false, reps));
      if (cd.cpu) {
        add(Cpu(cd.cpu, d, page, 1, reps));
        add(Cpu(cd.cpu, d, page, nthr, reps));
      }
    }
    cd.destroy(codec);
  }

  std::printf("\n%-18s %-12s %-10s %8s %7s %9s %9s %s\n", "dataset", "codec",
              "mode", "MiB", "ratio", "comp GB/s", "dec GB/s", "ok");
  for (const auto &r : rows)
    std::printf("%-18s %-12s %-10s %8.1f %7.3f %9.2f %9.2f %s\n", r.ds.c_str(),
                r.codec.c_str(), r.mode.c_str(), r.bytes / 1048576.0, r.ratio,
                r.c_gbs, r.d_gbs, r.ok ? "PASS" : "FAIL");
  std::printf("[r%s] %s\n", rank.c_str(), all ? "ALL PASS" : "SOME FAILED");
  return all ? 0 : 2;
}
