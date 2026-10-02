/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * @file lossy_sweep.cu
 * @brief Benchmark one GPU lossy codec (linked in through lossy_codec.h) on
 * whole fields at several value-range-relative error bounds.
 *
 * Per field: read the file, copy it to the GPU, find its value range (all
 * untimed), then for every variant and bound convert the relative bound to an
 * absolute one, compress and decompress device to device (timed), and verify
 * the reconstruction on the GPU: maximum point-wise error against the bound,
 * PSNR and NRMSE. One CSV row per (field, variant, bound). A field passes
 * the bound when its max error is within the bound plus one float32 ulp at
 * the field's largest magnitude, the rounding every float32 codec does when
 * it reconstructs; err_over_eb records the exact ratio.
 *
 * Timing: *_ms are CUDA events on the legacy default stream around the call,
 * so they cover every blocking stream the codec launches on; *_wall_ms are
 * the host clock around the same call plus a device synchronize. Neither
 * includes file I/O, the H2D copy, the range scan, the untimed Prepare/After
 * hooks or the verification. Each (variant, bound) gets one untimed warm-up
 * on the first field.
 *
 * Usage:
 *   lossy_sweep_<codec> --dir DIR --list FILE --dims X,Y,Z --out CSV
 *                       [--workload NAME] [--ebs 0.01,0.02,0.03]
 *                       [--variants a,b] [--max-files N]
 */

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "lossy_codec.h"

#define CUDA_CHECK(x)                                                     \
  do {                                                                    \
    cudaError_t e_ = (x);                                                 \
    if (e_ != cudaSuccess) {                                              \
      std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x,      \
                   cudaGetErrorString(e_));                               \
      std::exit(2);                                                       \
    }                                                                     \
  } while (0)

namespace {

/** @brief Error statistics of one reconstruction. */
struct ErrStats {
  float max_err = 0;
  double sse = 0;
  unsigned long long nan = 0;
};

/**
 * Max absolute error, sum of squared errors and count of NaNs the
 * reconstruction introduced. Non-negative floats order like their bits, so
 * the max is an atomicMax on the bit pattern.
 */
__global__ void ErrKernel(const float *a, const float *b, size_t n,
                          unsigned int *max_bits, double *sse,
                          unsigned long long *nan) {
  __shared__ float s_max[256];
  __shared__ double s_sse[256];
  float m = 0;
  double acc = 0;
  unsigned long long bad = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x) {
    const float o = a[i], d = b[i];
    if (d != d && o == o) { ++bad; continue; }
    const float e = fabsf(o - d);
    m = fmaxf(m, e);
    acc += (double)e * e;
  }
  s_max[threadIdx.x] = m;
  s_sse[threadIdx.x] = acc;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      s_max[threadIdx.x] = fmaxf(s_max[threadIdx.x], s_max[threadIdx.x + s]);
      s_sse[threadIdx.x] += s_sse[threadIdx.x + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicMax(max_bits, __float_as_uint(s_max[0]));
    atomicAdd(sse, s_sse[0]);
  }
  if (bad) atomicAdd(nan, bad);
}

/** @brief Device scratch reused for every field. */
struct Scratch {
  float *d_field = nullptr, *d_dec = nullptr;
  uint8_t *d_cmp = nullptr;
  size_t field_cap = 0, cmp_cap = 0;
  unsigned int *d_max = nullptr;
  double *d_sse = nullptr;
  unsigned long long *d_nan = nullptr;
  cudaEvent_t e0 = nullptr, e1 = nullptr;
  cudaStream_t s = nullptr;
};

/** @return error statistics of d_dec against d_orig (n floats). */
ErrStats Verify(const float *d_orig, const float *d_dec, size_t n,
                Scratch *w) {
  CUDA_CHECK(cudaMemset(w->d_max, 0, sizeof(unsigned int)));
  CUDA_CHECK(cudaMemset(w->d_sse, 0, sizeof(double)));
  CUDA_CHECK(cudaMemset(w->d_nan, 0, sizeof(unsigned long long)));
  ErrKernel<<<1024, 256>>>(d_orig, d_dec, n, w->d_max, w->d_sse, w->d_nan);
  CUDA_CHECK(cudaGetLastError());
  ErrStats st;
  unsigned int bits = 0;
  CUDA_CHECK(cudaMemcpy(&bits, w->d_max, sizeof(bits), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&st.sse, w->d_sse, sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&st.nan, w->d_nan, sizeof(st.nan), cudaMemcpyDeviceToHost));
  std::memcpy(&st.max_err, &bits, sizeof(float));
  return st;
}

/**
 * Time one call: device idle, events on the legacy default stream around it,
 * then a full device synchronize.
 * @param f       the call; returns false on failure
 * @param gpu_ms  event time, ms
 * @param wall_ms host time, ms
 * @return what f returned
 */
bool Timed(Scratch *w, const std::function<bool()> &f, double *gpu_ms,
           double *wall_ms) {
  using Clock = std::chrono::steady_clock;
  CUDA_CHECK(cudaDeviceSynchronize());
  const auto t0 = Clock::now();
  CUDA_CHECK(cudaEventRecord(w->e0, 0));
  const bool ok = f();
  CUDA_CHECK(cudaEventRecord(w->e1, 0));
  CUDA_CHECK(cudaEventSynchronize(w->e1));
  const cudaError_t sync = cudaDeviceSynchronize();
  const auto t1 = Clock::now();
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, w->e0, w->e1));
  *gpu_ms = ms;
  *wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  if (sync != cudaSuccess) {
    std::fprintf(stderr, "CUDA error in codec: %s\n", cudaGetErrorString(sync));
    std::exit(2);  // a sticky error poisons every later call
  }
  return ok;
}

/** @brief One (field, variant, bound) outcome. */
struct Result {
  double eb_abs = 0;
  size_t bytes = 0;
  double comp_ms = 0, comp_wall_ms = 0, decomp_ms = 0, decomp_wall_ms = 0;
  ErrStats err;
  bool ok = false;
  std::string note;
};

/**
 * Compress, decompress and verify one field with the current variant.
 * @param eb_abs absolute error bound
 */
Result RunOne(LossyCodec *c, const Field &f, double eb_abs, Scratch *w) {
  Result r;
  r.eb_abs = eb_abs;
  const size_t cap = c->MaxCompressedBytes(f);
  if (cap > w->cmp_cap) {
    if (w->d_cmp) cudaFree(w->d_cmp);
    CUDA_CHECK(cudaMalloc(&w->d_cmp, cap));
    w->cmp_cap = cap;
  }
  if (!c->PrepareCompress(w->d_field, f, eb_abs, w->s)) {
    r.note = c->Note();  // why the codec declined (e.g. unsupported shape)
    return r;
  }
  size_t bytes = 0;
  Timed(w, [&] {
    bytes = c->Compress(w->d_field, f, eb_abs, w->d_cmp, cap, w->s);
    return bytes > 0;
  }, &r.comp_ms, &r.comp_wall_ms);
  r.bytes = bytes;
  r.note = c->Note();
  if (bytes == 0 || !c->AfterCompress(w->d_cmp, bytes, w->s)) return r;
  CUDA_CHECK(cudaMemset(w->d_dec, 0xFF, f.n() * sizeof(float)));  // NaNs
  if (!c->PrepareDecompress(w->d_cmp, bytes, f, eb_abs, w->s)) return r;
  const bool dok = Timed(w, [&] {
    return c->Decompress(w->d_cmp, bytes, f, eb_abs, w->d_dec, w->s);
  }, &r.decomp_ms, &r.decomp_wall_ms);
  if (!dok || !c->AfterDecompress(w->d_dec, f, w->s)) return r;
  r.err = Verify(w->d_field, w->d_dec, f.n(), w);
  r.ok = r.err.nan == 0;
  return r;
}

/** @brief Command-line options. */
struct Options {
  std::string dir, list, out, workload = "data";
  Field f;
  std::vector<double> ebs = {0.01, 0.02, 0.03};
  std::vector<std::string> variants;
  size_t max_files = 0;
};

/** @return comma-separated tokens of s */
std::vector<std::string> Split(const std::string &s) {
  std::vector<std::string> v;
  std::stringstream ss(s);
  for (std::string t; std::getline(ss, t, ',');) if (!t.empty()) v.push_back(t);
  return v;
}

/** @return parsed options; exits with usage on error */
Options ParseArgs(int argc, char **argv) {
  Options o;
  auto usage = [] {
    std::fprintf(stderr, "usage: lossy_sweep --dir DIR --list FILE --dims X,Y,Z "
                         "--out CSV [--workload W] [--ebs a,b] [--variants a,b] "
                         "[--max-files N]\n");
    std::exit(1);
  };
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (i + 1 >= argc) usage();
    const std::string v = argv[++i];
    if (a == "--dir") o.dir = v;
    else if (a == "--list") o.list = v;
    else if (a == "--out") o.out = v;
    else if (a == "--workload") o.workload = v;
    else if (a == "--max-files") o.max_files = std::strtoull(v.c_str(), nullptr, 10);
    else if (a == "--variants") o.variants = Split(v);
    else if (a == "--ebs") {
      o.ebs.clear();
      for (auto &t : Split(v)) o.ebs.push_back(std::atof(t.c_str()));
    } else if (a == "--dims") {
      auto d = Split(v);
      if (d.empty() || d.size() > 3) usage();
      o.f.x = std::strtoull(d[0].c_str(), nullptr, 10);
      if (d.size() > 1) o.f.y = std::strtoull(d[1].c_str(), nullptr, 10);
      if (d.size() > 2) o.f.z = std::strtoull(d[2].c_str(), nullptr, 10);
    } else usage();
  }
  if (o.dir.empty() || o.list.empty() || o.out.empty() || o.f.n() <= 1) usage();
  return o;
}

/** @return bytes read into a growing pinned buffer, 0 on failure */
size_t ReadFile(const std::string &path, std::vector<char> *buf) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) return 0;
  const size_t sz = in.tellg();
  buf->resize(sz);
  in.seekg(0);
  return in.read(buf->data(), sz) ? sz : 0;
}

/** @brief Per (variant, bound) totals for the summary. */
struct Totals {
  size_t fields = 0, failed = 0, violations = 0, in = 0, out = 0;
  size_t psnr_fields = 0;  // non-constant fields that reconstructed
  double comp_ms = 0, decomp_ms = 0, psnr = 0;
};

/** Print one summary line per (variant, bound). */
void PrintSummary(const std::string &codec,
                  const std::map<std::pair<std::string, double>, Totals> &t) {
  std::printf("\n%-10s %-14s %6s %6s %6s %6s %9s %10s %10s %8s\n", "codec",
              "variant", "eb", "fields", "failed", "viol", "ratio", "comp_GB/s",
              "decomp_GB/s", "psnr");
  for (const auto &[k, x] : t) {
    std::printf("%-10s %-14s %6g %6zu %6zu %6zu %9.2f %10.2f %10.2f %8.2f\n",
                codec.c_str(), k.first.c_str(), k.second, x.fields, x.failed,
                x.violations, x.out ? double(x.in) / x.out : 0.0,
                x.comp_ms > 0 ? x.in / 1e6 / x.comp_ms : 0.0,
                x.decomp_ms > 0 ? x.in / 1e6 / x.decomp_ms : 0.0,
                x.psnr_fields ? x.psnr / x.psnr_fields : 0.0);
  }
}

/** @return {min, max} of n device floats */
std::pair<float, float> Range(const float *d, size_t n) {
  thrust::device_ptr<const float> p(d);
  auto mm = thrust::minmax_element(p, p + n);
  float lo = 0, hi = 0;
  CUDA_CHECK(cudaMemcpy(&lo, thrust::raw_pointer_cast(mm.first), sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&hi, thrust::raw_pointer_cast(mm.second), sizeof(float),
                        cudaMemcpyDeviceToHost));
  return {lo, hi};
}

/**
 * Run every variant and bound on one field and write their rows.
 * @param warm run each (variant, bound) once untimed first
 */
void SweepField(const std::string &name, LossyCodec *c, const Options &o,
                const std::vector<std::string> &variants, Scratch *w, FILE *csv,
                std::map<std::pair<std::string, double>, Totals> *tot,
                bool warm) {
  const auto [lo, hi] = Range(w->d_field, o.f.n());
  const double range = double(hi) - lo;
  const bool constant = !(range > 0);
  const double maxabs = std::max(std::fabs(lo), std::fabs(hi));
  const double scale = constant ? std::max(maxabs, 1.0) : range;
  const double ulp = maxabs * 1.1920928955078125e-7;  // 2^-23 x |max|
  for (const auto &v : variants) {
    if (!c->SetVariant(v)) continue;
    for (double eb : o.ebs) {
      const double eb_abs = eb * scale;
      if (warm) RunOne(c, o.f, eb_abs, w);
      const Result r = RunOne(c, o.f, eb_abs, w);
      const double mse = r.err.sse / o.f.n();
      const double psnr = (mse > 0 && !constant)
                              ? 20 * std::log10(range) - 10 * std::log10(mse)
                              : 999.0;
      const double over = r.eb_abs > 0 ? r.err.max_err / r.eb_abs : 0;
      const bool bound_ok = r.ok && r.err.max_err <= r.eb_abs + ulp;
      const size_t in = o.f.n() * sizeof(float);
      std::fprintf(csv,
                   "%s,%s,%s,%s,%g,%.9g,%zu,%zu,%zu,%.6f,%.4f,%.4f,%.4f,%.4f,"
                   "%.9g,%.6f,%.4f,%.6g,%d,%d,%d,%s\n",
                   o.workload.c_str(), name.c_str(), c->Name().c_str(), v.c_str(),
                   eb, r.eb_abs, o.f.n(), in, r.bytes,
                   r.bytes ? double(in) / r.bytes : 0.0, r.comp_ms,
                   r.comp_wall_ms, r.decomp_ms, r.decomp_wall_ms, r.err.max_err,
                   over, psnr, constant ? 0.0 : std::sqrt(mse) / range,
                   bound_ok ? 1 : 0, r.ok ? 1 : 0, constant ? 1 : 0,
                   r.note.c_str());
      Totals &t = (*tot)[{v, eb}];
      ++t.fields;
      if (!r.ok) { ++t.failed; continue; }
      if (!bound_ok) ++t.violations;
      t.in += in;
      t.out += r.bytes;
      t.comp_ms += r.comp_ms;
      t.decomp_ms += r.decomp_ms;
      if (!constant) { t.psnr += psnr; ++t.psnr_fields; }
    }
  }
}

/** @return device scratch for fields of n floats */
Scratch MakeScratch(size_t n) {
  Scratch w;
  CUDA_CHECK(cudaMalloc(&w.d_field, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&w.d_dec, n * sizeof(float)));
  w.field_cap = n;
  CUDA_CHECK(cudaMalloc(&w.d_max, sizeof(unsigned int)));
  CUDA_CHECK(cudaMalloc(&w.d_sse, sizeof(double)));
  CUDA_CHECK(cudaMalloc(&w.d_nan, sizeof(unsigned long long)));
  CUDA_CHECK(cudaEventCreate(&w.e0));
  CUDA_CHECK(cudaEventCreate(&w.e1));
  CUDA_CHECK(cudaStreamCreate(&w.s));  // blocking: ordered with stream 0
  return w;
}

}  // namespace

int main(int argc, char **argv) {
  const Options o = ParseArgs(argc, argv);
  auto codec = MakeLossyCodec();
  std::vector<std::string> variants =
      o.variants.empty() ? codec->Variants() : o.variants;
  std::vector<std::string> files;
  {
    std::ifstream in(o.list);
    for (std::string l; std::getline(in, l);) if (!l.empty()) files.push_back(l);
  }
  if (o.max_files && files.size() > o.max_files) files.resize(o.max_files);
  FILE *csv = std::fopen(o.out.c_str(), "w");
  if (!csv) { std::fprintf(stderr, "cannot write %s\n", o.out.c_str()); return 1; }
  std::fprintf(csv, "workload,file,codec,variant,eb_rel,eb_abs,n,bytes_in,"
                    "bytes_out,ratio,comp_ms,comp_wall_ms,decomp_ms,"
                    "decomp_wall_ms,max_err,err_over_eb,psnr_db,nrmse,"
                    "bound_ok,ok,constant,note\n");
  Scratch w = MakeScratch(o.f.n());
  std::map<std::pair<std::string, double>, Totals> tot;
  std::vector<char> buf;
  size_t bad = 0;
  bool warmed = false;
  const auto t0 = std::chrono::steady_clock::now();
  for (size_t i = 0; i < files.size(); ++i) {
    const size_t sz = ReadFile(o.dir + "/" + files[i], &buf);
    if (sz != o.f.n() * sizeof(float)) {
      std::fprintf(stderr, "skip %s: %zu bytes, dims want %zu\n",
                   files[i].c_str(), sz, o.f.n() * sizeof(float));
      ++bad;
      continue;
    }
    CUDA_CHECK(cudaMemcpy(w.d_field, buf.data(), sz, cudaMemcpyHostToDevice));
    SweepField(files[i], codec.get(), o, variants, &w, csv, &tot, !warmed);
    warmed = true;
    std::fflush(csv);
    if ((i + 1) % 50 == 0 || i + 1 == files.size()) {
      std::fprintf(stderr, "%zu/%zu fields, %.0f ms\n", i + 1, files.size(),
                   std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - t0).count());
    }
  }
  std::fclose(csv);
  PrintSummary(codec->Name(), tot);
  if (bad) std::printf("skipped files: %zu\n", bad);
  return bad ? 3 : 0;
}
