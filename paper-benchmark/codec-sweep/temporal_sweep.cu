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
 * @file temporal_sweep.cu
 * @brief LOSSLESS look-ahead (temporal) coding as a pure GPU kernel, outside
 * clio: one chunk across a window of T consecutive dumps, each frame coded
 * either on its own or as a bit-exact residual against a temporal prediction,
 * the residual packed by any lossless GPU codec of gpu_codecs.cuh.
 *
 * ctp::LookaheadCodec (compress/lookahead.h) is lossy by construction: anchor
 * frames go through cuSZp and the in-between frames store a QUANTIZED residual
 * of the two-sided prediction lo + w (hi - lo), w = (t - lo) / (hi - lo). This program keeps the same
 * coding plan (frame 0 intra, frame T-1 forward from frame 0, the frames in
 * between predicted from both sides, midpoint first) but makes the residual
 * exact, so every decoded frame is bit-identical to its input:
 *
 *   indep        every frame coded on its own (the baseline every mode has
 *                to beat; it is what the per-chunk sweep measures)
 *   xor_prev     frame 0 intra; frame t = bits(x_t) XOR bits(x_{t-1})
 *   xor_la       the look-ahead plan; residual = bits(x) XOR bits(pred),
 *                pred = lo (forward) or lo + w (hi - lo) (bidirectional),
 *                w = (t - lo) / (hi - lo) as ctp::LookaheadCodec::Weight,
 *                computed in float exactly as the decoder will
 *   idelta_prev  like xor_prev, but the residual is the zigzag-coded
 *                difference of the sign-magnitude-to-monotone integer map of
 *                the bit patterns (what fpzip does with its prediction):
 *                small prediction errors become small integers
 *   idelta_la    the look-ahead plan with the integer-map residual
 *
 * The prediction uses the ORIGINAL neighbouring frames, which the decoder
 * has exactly because the coding is lossless and the plan reconstructs every
 * reference before it is needed. Because the residual is a bit-level
 * transform, the intra frame is the only frame whose codec input is the data
 * itself; the others hand the codec the residual words.
 *
 * Measurement rules follow corpus_sweep.cu: every input is device resident
 * before anything is timed; comp_ms / decomp_ms are CUDA events around the
 * codec's own compress / decompress call, summed over the window's frames;
 * resid_ms / recon_ms are events around the prediction+residual and the
 * prediction+reconstruction kernels, summed likewise (zero for indep). Each
 * value is the median of --reps timed round trips, comp_bytes must agree
 * across reps, and a row is ok only when every reconstructed frame compares
 * byte for byte with its input on the GPU. One untimed warm-up per
 * (setting, mode) precedes the first timed round.
 *
 * Usage:
 *   temporal_sweep --dir DIR --groups FILE --configs SPECS --out CSV
 *                  [--chunk BYTES] [--reps N] [--elem 4|8] [--modes a,b,...]
 *
 * FILE lists one group per line: "label k path_0 path_1 ... path_{T-1}", the
 * T consecutive dumps of one field (relative to DIR) and the chunk index k.
 * temporal_groups.py writes it from a dump directory. SPECS is a codec spec
 * list as corpus_sweep takes it. --elem 8 treats the data as 64-bit words
 * (double) for the prediction and residual kernels.
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "gpu_codecs.cuh"

namespace {

using gpu_codecs::Codec;
using gpu_codecs::CodecSpec;
using Clock = std::chrono::steady_clock;

/** @brief Command-line options. */
struct Options {
  std::string dir, groups, configs, out;
  int reps = 3;              ///< timed round trips per (group, mode, setting)
  size_t chunk = 4u << 20;   ///< bytes per chunk of each frame
  int elem = 4;              ///< word size: 4 (float) or 8 (double)
  std::string modes = "indep,xor_prev,xor_la,idelta_prev,idelta_la";
};

/** @return the options; exits with usage on a missing or unknown flag. */
Options ParseArgs(int argc, char **argv) {
  Options o;
  auto usage = [] {
    std::fprintf(stderr,
                 "usage: temporal_sweep --dir DIR --groups FILE --configs SPECS "
                 "--out CSV [--chunk BYTES] [--reps N] [--elem 4|8] "
                 "[--modes a,b,...]\n");
    std::exit(1);
  };
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (i + 1 >= argc) usage();
    const char *v = argv[++i];
    if (a == "--dir") o.dir = v;
    else if (a == "--groups") o.groups = v;
    else if (a == "--configs") o.configs = v;
    else if (a == "--out") o.out = v;
    else if (a == "--reps") o.reps = std::atoi(v);
    else if (a == "--chunk") o.chunk = std::strtoull(v, nullptr, 10);
    else if (a == "--elem") o.elem = std::atoi(v);
    else if (a == "--modes") o.modes = v;
    else usage();
  }
  if (o.dir.empty() || o.groups.empty() || o.configs.empty() ||
      o.out.empty() || o.reps < 1 || o.chunk == 0 ||
      (o.elem != 4 && o.elem != 8)) {
    usage();
  }
  return o;
}

/** @return the lines of a file that are neither blank nor '#' comments. */
std::vector<std::string> ReadLines(const std::string &path) {
  std::vector<std::string> v;
  std::ifstream f(path);
  if (!f) {
    std::fprintf(stderr, "cannot read %s\n", path.c_str());
    std::exit(1);
  }
  for (std::string l; std::getline(f, l);) {
    const size_t b = l.find_first_not_of(" \t\r");
    if (b == std::string::npos || l[b] == '#') continue;
    const size_t e = l.find_last_not_of(" \t\r");
    v.push_back(l.substr(b, e - b + 1));
  }
  return v;
}

/** @brief One window: the same chunk of one field at T consecutive dumps. */
struct Group {
  std::string label;
  size_t k = 0;                    ///< chunk index
  std::vector<std::string> paths;  ///< one per frame, relative to dir
};

/** @return the groups listed in path; exits on a malformed line. */
std::vector<Group> ReadGroups(const std::string &path) {
  std::vector<Group> v;
  for (const std::string &line : ReadLines(path)) {
    std::istringstream ss(line);
    Group g;
    ss >> g.label >> g.k;
    for (std::string p; ss >> p;) g.paths.push_back(p);
    if (g.paths.size() < 2) {
      std::fprintf(stderr, "group needs >= 2 frames: %s\n", line.c_str());
      std::exit(3);
    }
    v.push_back(g);
  }
  return v;
}

// ---------------------------------------------------------------- coding plan

/** How one frame of a window is coded. */
enum class Kind { kIntra, kForward, kBidir };

/** One coding step: frame t, its kind and its reference frames. */
struct Step {
  int t = 0;
  Kind kind = Kind::kIntra;
  int lo = -1, hi = -1;
};

/**
 * @return the weight of the later reference, (t - lo) / (hi - lo), computed
 * in float exactly as ctp::LookaheadCodec::Weight does.
 */
float Weight(const Step &st) {
  return static_cast<float>(st.t - st.lo) / static_cast<float>(st.hi - st.lo);
}

/** Residual transform applied to a predicted frame. */
enum class Resid { kNone, kXor, kIdelta };

/** @brief A mode: a plan family plus a residual transform. */
struct Mode {
  std::string name;
  bool lookahead = false;  ///< the B-frame plan; else forward from t-1
  Resid resid = Resid::kNone;
};

/** @return the mode named s; exits on an unknown name. */
Mode ParseMode(const std::string &s) {
  if (s == "indep") return {s, false, Resid::kNone};
  if (s == "xor_prev") return {s, false, Resid::kXor};
  if (s == "xor_la") return {s, true, Resid::kXor};
  if (s == "idelta_prev") return {s, false, Resid::kIdelta};
  if (s == "idelta_la") return {s, true, Resid::kIdelta};
  std::fprintf(stderr, "unknown mode %s\n", s.c_str());
  std::exit(1);
}

/** Midpoint-first bidirectional steps for the open interval (lo, hi). */
void AddMidpoints(int lo, int hi, std::vector<Step> *steps) {
  if (hi - lo < 2) return;
  const int mid = (lo + hi) / 2;
  steps->push_back({mid, Kind::kBidir, lo, hi});
  AddMidpoints(lo, mid, steps);
  AddMidpoints(mid, hi, steps);
}

/**
 * @return the coding order of nf frames under a mode: every frame intra for
 * indep; frame 0 intra then each frame forward from its predecessor for the
 * prev modes; ctp::LookaheadCodec::Plan's order for the look-ahead modes.
 */
std::vector<Step> Plan(int nf, const Mode &m) {
  std::vector<Step> steps;
  if (m.resid == Resid::kNone) {
    for (int t = 0; t < nf; ++t) steps.push_back({t, Kind::kIntra, -1, -1});
    return steps;
  }
  steps.push_back({0, Kind::kIntra, -1, -1});
  if (!m.lookahead) {
    for (int t = 1; t < nf; ++t) steps.push_back({t, Kind::kForward, t - 1, -1});
    return steps;
  }
  if (nf == 1) return steps;
  steps.push_back({nf - 1, Kind::kForward, 0, -1});
  AddMidpoints(0, nf - 1, &steps);
  return steps;
}

// ------------------------------------------------------------------ kernels

/** @brief Word / float pairs the kernels are instantiated for. */
template <int E> struct Word;
template <> struct Word<4> { using U = uint32_t; using F = float; };
template <> struct Word<8> { using U = uint64_t; using F = double; };

/** @return the prediction of x from its references, as a word. */
template <typename U, typename F>
__device__ __forceinline__ U Predict(U lo, U hi, bool bidir, F w) {
  if (!bidir) return lo;
  F a, b;
  memcpy(&a, &lo, sizeof(F));
  memcpy(&b, &hi, sizeof(F));
  const F p = a + w * (b - a);  // ctp TemporalPredictDevice's formula
  U u;
  memcpy(&u, &p, sizeof(U));
  return u;
}

/** Sign-magnitude float bits -> monotone unsigned integer (fpzip's map). */
template <typename U>
__device__ __forceinline__ U ToMono(U u) {
  const U sign = U(1) << (8 * sizeof(U) - 1);
  return (u & sign) ? ~u : (u | sign);
}

/** Inverse of ToMono. */
template <typename U>
__device__ __forceinline__ U FromMono(U m) {
  const U sign = U(1) << (8 * sizeof(U) - 1);
  return (m & sign) ? (m & ~sign) : ~m;
}

/** Zigzag code of a two's-complement difference held in an unsigned word. */
template <typename U>
__device__ __forceinline__ U Zigzag(U d) {
  const U sign = U(1) << (8 * sizeof(U) - 1);
  return (d << 1) ^ ((d & sign) ? ~U(0) : U(0));
}

/** Inverse of Zigzag. */
template <typename U>
__device__ __forceinline__ U Unzigzag(U z) {
  return (z >> 1) ^ (U(0) - (z & U(1)));
}

/**
 * residual[i] of x[i] against the prediction from lo[i] / hi[i].
 * @param xor_mode true: bits XOR; false: zigzag(mono(x) - mono(pred))
 */
template <typename U, typename F>
__global__ void ResidualKernel(const U *x, const U *lo, const U *hi, U *out,
                               size_t n, bool bidir, F w, bool xor_mode) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  for (; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const U p = Predict<U, F>(lo[i], hi[i], bidir, w);
    out[i] = xor_mode ? (x[i] ^ p) : Zigzag<U>(ToMono<U>(x[i]) - ToMono<U>(p));
  }
}

/** x[i] from residual[i] and the prediction; the exact inverse. */
template <typename U, typename F>
__global__ void ReconstructKernel(const U *res, const U *lo, const U *hi,
                                  U *x, size_t n, bool bidir, F w,
                                  bool xor_mode) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  for (; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const U p = Predict<U, F>(lo[i], hi[i], bidir, w);
    x[i] = xor_mode ? (res[i] ^ p)
                    : FromMono<U>(ToMono<U>(p) + Unzigzag<U>(res[i]));
  }
}

/**
 * Launch the residual (forward) or reconstruction (inverse) kernel.
 * @param elem   word size, 4 or 8
 * @param a      input: the frame (forward) or its residual (inverse)
 * @param lo,hi  reference frames (hi unused unless bidir)
 * @param out    the residual (forward) or the frame (inverse)
 * @param bytes  frame bytes, a multiple of elem
 * @param w      weight of hi in a bidirectional prediction
 */
void LaunchTransform(int elem, const uint8_t *a, const uint8_t *lo,
                     const uint8_t *hi, uint8_t *out, size_t bytes, bool bidir,
                     float w, bool xor_mode, bool inverse, cudaStream_t s) {
  const size_t n = bytes / static_cast<size_t>(elem);
  const unsigned blocks = static_cast<unsigned>(std::min<size_t>(4096, (n + 255) / 256));
  if (elem == 4) {
    using U = Word<4>::U;
    using F = Word<4>::F;
    if (inverse) {
      ReconstructKernel<U, F><<<blocks, 256, 0, s>>>(
          reinterpret_cast<const U *>(a), reinterpret_cast<const U *>(lo),
          reinterpret_cast<const U *>(hi), reinterpret_cast<U *>(out), n,
          bidir, static_cast<F>(w), xor_mode);
    } else {
      ResidualKernel<U, F><<<blocks, 256, 0, s>>>(
          reinterpret_cast<const U *>(a), reinterpret_cast<const U *>(lo),
          reinterpret_cast<const U *>(hi), reinterpret_cast<U *>(out), n,
          bidir, static_cast<F>(w), xor_mode);
    }
  } else {
    using U = Word<8>::U;
    using F = Word<8>::F;
    if (inverse) {
      ReconstructKernel<U, F><<<blocks, 256, 0, s>>>(
          reinterpret_cast<const U *>(a), reinterpret_cast<const U *>(lo),
          reinterpret_cast<const U *>(hi), reinterpret_cast<U *>(out), n,
          bidir, static_cast<F>(w), xor_mode);
    } else {
      ResidualKernel<U, F><<<blocks, 256, 0, s>>>(
          reinterpret_cast<const U *>(a), reinterpret_cast<const U *>(lo),
          reinterpret_cast<const U *>(hi), reinterpret_cast<U *>(out), n,
          bidir, static_cast<F>(w), xor_mode);
    }
  }
  CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------- workspace

/** @brief Device buffers for one window of T frames. */
struct Window {
  int nf = 0;
  size_t bytes = 0;                ///< bytes per frame
  std::vector<uint8_t *> x;        ///< the input frames
  std::vector<uint8_t *> rec;      ///< reconstructed frames
  std::vector<uint8_t *> comp;     ///< compressed stream per frame
  std::vector<size_t> comp_bytes;  ///< its length per frame
  size_t cap = 0;                  ///< capacity of each comp buffer
  uint8_t *res = nullptr;          ///< residual scratch
  uint8_t *dec = nullptr;          ///< decoded residual scratch
  unsigned int *d_bad = nullptr;
  cudaEvent_t e0 = nullptr, e1 = nullptr;
};

/** @return milliseconds between the two recorded events. */
double EventMs(const Window &w) {
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, w.e0, w.e1));
  return ms;
}

/** @return milliseconds of host time since t0. */
double MsSince(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

/** Allocate the window for nf frames of bytes each (comp buffers later). */
void AllocWindow(Window *w, int nf, size_t bytes) {
  w->nf = nf;
  w->bytes = bytes;
  w->x.assign(nf, nullptr);
  w->rec.assign(nf, nullptr);
  w->comp.assign(nf, nullptr);
  w->comp_bytes.assign(nf, 0);
  for (int t = 0; t < nf; ++t) {
    CUDA_CHECK(cudaMalloc(&w->x[t], bytes));
    CUDA_CHECK(cudaMalloc(&w->rec[t], bytes));
  }
  CUDA_CHECK(cudaMalloc(&w->res, bytes));
  CUDA_CHECK(cudaMalloc(&w->dec, bytes));
  CUDA_CHECK(cudaMalloc(&w->d_bad, sizeof(unsigned int)));
  CUDA_CHECK(cudaEventCreate(&w->e0));
  CUDA_CHECK(cudaEventCreate(&w->e1));
}

/** Grow every per-frame compressed buffer to cap bytes. */
void FitComp(Window *w, size_t cap) {
  if (cap <= w->cap) return;
  for (int t = 0; t < w->nf; ++t) {
    if (w->comp[t]) CUDA_CHECK(cudaFree(w->comp[t]));
    CUDA_CHECK(cudaMalloc(&w->comp[t], cap));
  }
  w->cap = cap;
}

/**
 * Read one group's frames (chunk k of each path) into the window (untimed).
 * @return false when a frame is missing, short, or the sizes differ
 */
bool LoadGroup(const std::string &dir, const Group &g, size_t chunk, int elem,
               Window *w) {
  std::vector<char> buf(chunk);
  size_t bytes = 0;
  for (size_t t = 0; t < g.paths.size(); ++t) {
    std::ifstream f(dir + "/" + g.paths[t], std::ios::binary | std::ios::ate);
    if (!f) return false;
    const size_t size = static_cast<size_t>(f.tellg());
    const size_t off = g.k * chunk;
    if (off >= size) return false;
    const size_t n = std::min(chunk, size - off);
    if (t == 0) bytes = n; else if (n != bytes) return false;
    f.seekg(static_cast<std::streamoff>(off));
    if (!f.read(buf.data(), static_cast<std::streamsize>(n))) return false;
    CUDA_CHECK(cudaMemcpy(w->x[t], buf.data(), n, cudaMemcpyHostToDevice));
  }
  if (bytes % static_cast<size_t>(elem) != 0) return false;
  w->bytes = bytes;
  return true;
}

// ---------------------------------------------------------------- one round

/** @brief Timings of one coded window (sums over its frames). */
struct Round {
  size_t comp_bytes = 0;
  double comp_ms = 0, decomp_ms = 0, resid_ms = 0, recon_ms = 0;
  double comp_wall_ms = 0, decomp_wall_ms = 0;
  std::vector<size_t> frame_bytes;  ///< compressed bytes per frame, by t
  bool ok = false;
  std::string note;
};

/**
 * Code the window under one mode with one codec, decode it in plan order,
 * and verify every frame on the GPU.
 */
Round RunRound(Codec *c, const Mode &m, const std::vector<Step> &plan,
               int elem, Window *w) {
  Round r;
  r.frame_bytes.assign(w->nf, 0);
  const size_t n = w->bytes;
  if (!c->Accepts(n)) {
    r.note = "size not accepted";
    return r;
  }
  cudaStream_t s = c->stream();
  const bool xor_mode = (m.resid == Resid::kXor);
  try {
    // ---- encode
    for (const Step &st : plan) {
      const uint8_t *src = w->x[st.t];
      if (st.kind != Kind::kIntra) {
        const uint8_t *hi = st.hi >= 0 ? w->x[st.hi] : w->x[st.lo];
        CUDA_CHECK(cudaStreamSynchronize(s));
        CUDA_CHECK(cudaEventRecord(w->e0, s));
        LaunchTransform(elem, w->x[st.t], w->x[st.lo], hi, w->res, n,
                        st.kind == Kind::kBidir,
                        st.kind == Kind::kBidir ? Weight(st) : 0.0f, xor_mode,
                        false, s);
        CUDA_CHECK(cudaEventRecord(w->e1, s));
        CUDA_CHECK(cudaEventSynchronize(w->e1));
        r.resid_ms += EventMs(*w);
        src = w->res;
      }
      c->PrepareCompress(n);
      CUDA_CHECK(cudaStreamSynchronize(s));
      const Clock::time_point w0 = Clock::now();
      CUDA_CHECK(cudaEventRecord(w->e0, s));
      const uint8_t *in = c->Preprocess(src, n);  // pre-shuffle: timed
      c->Compress(in, n, w->comp[st.t], w->cap);
      CUDA_CHECK(cudaEventRecord(w->e1, s));
      CUDA_CHECK(cudaEventSynchronize(w->e1));
      const size_t cb = c->CompressedBytes(w->comp[st.t]);
      r.comp_wall_ms += MsSince(w0);
      r.comp_ms += EventMs(*w);
      if (cb == 0 || cb > w->cap) {
        r.note = "compress failed";
        return r;
      }
      w->comp_bytes[st.t] = cb;
      r.frame_bytes[st.t] = cb;
      r.comp_bytes += cb;
    }
    // ---- decode, in plan order so every reference is reconstructed first
    for (const Step &st : plan) {
      uint8_t *dst = c->DecodeTarget(w->dec);
      CUDA_CHECK(cudaMemsetAsync(w->dec, 0xA5, n, s));
      if (dst != w->dec) CUDA_CHECK(cudaMemsetAsync(dst, 0xA5, n, s));
      CUDA_CHECK(cudaStreamSynchronize(s));
      const Clock::time_point w0 = Clock::now();
      c->PrepareDecompress(w->comp[st.t], w->comp_bytes[st.t]);
      CUDA_CHECK(cudaEventRecord(w->e0, s));
      c->Decompress(w->comp[st.t], w->comp_bytes[st.t], dst, n);
      c->Postprocess(w->dec, n);  // un-shuffle: timed
      CUDA_CHECK(cudaEventRecord(w->e1, s));
      CUDA_CHECK(cudaEventSynchronize(w->e1));
      r.decomp_wall_ms += MsSince(w0);
      r.decomp_ms += EventMs(*w);
      if (!c->DecompressOk()) {
        r.note = "decompress failed";
        return r;
      }
      if (st.kind == Kind::kIntra) {
        CUDA_CHECK(cudaMemcpyAsync(w->rec[st.t], w->dec, n,
                                   cudaMemcpyDeviceToDevice, s));
      } else {
        const uint8_t *hi = st.hi >= 0 ? w->rec[st.hi] : w->rec[st.lo];
        CUDA_CHECK(cudaStreamSynchronize(s));
        CUDA_CHECK(cudaEventRecord(w->e0, s));
        LaunchTransform(elem, w->dec, w->rec[st.lo], hi, w->rec[st.t], n,
                        st.kind == Kind::kBidir,
                        st.kind == Kind::kBidir ? Weight(st) : 0.0f, xor_mode,
                        true, s);
        CUDA_CHECK(cudaEventRecord(w->e1, s));
        CUDA_CHECK(cudaEventSynchronize(w->e1));
        r.recon_ms += EventMs(*w);
      }
    }
    // ---- verify
    r.ok = true;
    for (int t = 0; t < w->nf; ++t) {
      if (!gpu_codecs::BytesEqualOnGpu(w->x[t], w->rec[t], n, w->d_bad, s)) {
        r.ok = false;
        r.note = "frame " + std::to_string(t) + " differs";
        return r;
      }
    }
  } catch (const std::exception &e) {
    r.ok = false;
    r.note = std::string("exception: ") + e.what();
    cudaGetLastError();
  }
  return r;
}

/** @return the median of v (mean of the middle two for an even count). */
double Median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const size_t m = v.size() / 2;
  return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

/** Time reps rounds and reduce them to medians (sizes must agree). */
Round Measure(Codec *c, const Mode &m, const std::vector<Step> &plan,
              int elem, int reps, Window *w) {
  std::vector<double> cm, dm, rm, km, cw, dw;
  Round first;
  for (int i = 0; i < reps; ++i) {
    Round r = RunRound(c, m, plan, elem, w);
    if (!r.ok) return r;
    if (i == 0) first = r;
    if (r.comp_bytes != first.comp_bytes) {
      r.ok = false;
      r.note = "compressed size differs between reps";
      return r;
    }
    cm.push_back(r.comp_ms);
    dm.push_back(r.decomp_ms);
    rm.push_back(r.resid_ms);
    km.push_back(r.recon_ms);
    cw.push_back(r.comp_wall_ms);
    dw.push_back(r.decomp_wall_ms);
  }
  first.comp_ms = Median(cm);
  first.decomp_ms = Median(dm);
  first.resid_ms = Median(rm);
  first.recon_ms = Median(km);
  first.comp_wall_ms = Median(cw);
  first.decomp_wall_ms = Median(dw);
  return first;
}

/** @return s with commas and line breaks replaced, safe in a CSV field. */
std::string CsvSafe(std::string s) {
  for (char &ch : s) {
    if (ch == ',' || ch == '\n' || ch == '\r') ch = ';';
  }
  return s;
}

/** @brief One built codec and whether it has been warmed up. */
struct Setting {
  CodecSpec spec;
  std::unique_ptr<Codec> codec;
  std::string build_error;
  std::map<std::string, bool> warmed;  ///< by mode name
};

}  // namespace

int main(int argc, char **argv) {
  const Options o = ParseArgs(argc, argv);
  std::vector<Mode> modes;
  {
    std::istringstream ss(o.modes);
    for (std::string m; std::getline(ss, m, ',');) {
      if (!m.empty()) modes.push_back(ParseMode(m));
    }
  }
  std::vector<Setting> settings;
  for (const std::string &l : ReadLines(o.configs)) {
    Setting s;
    s.spec = gpu_codecs::ParseSpec(l);
    try {
      s.codec = gpu_codecs::MakeCodec(s.spec);
      if (!s.codec) s.build_error = "unknown codec " + s.spec.base;
    } catch (const std::exception &e) {
      s.build_error = std::string("setup: ") + e.what();
      cudaGetLastError();
    }
    settings.push_back(std::move(s));
  }
  const std::vector<Group> groups = ReadGroups(o.groups);
  if (groups.empty()) {
    std::fprintf(stderr, "no groups\n");
    return 1;
  }
  const int nf = static_cast<int>(groups[0].paths.size());
  for (const Group &g : groups) {
    if (static_cast<int>(g.paths.size()) != nf) {
      std::fprintf(stderr, "every group must have %d frames\n", nf);
      return 1;
    }
  }
  std::fprintf(stderr, "%zu groups of %d frames, %zu modes, %zu settings, %d reps\n",
               groups.size(), nf, modes.size(), settings.size(), o.reps);

  Window w;
  AllocWindow(&w, nf, o.chunk);
  size_t cap = 0;
  for (Setting &s : settings) {
    if (s.codec) cap = std::max(cap, s.codec->Bound(o.chunk));
  }
  FitComp(&w, cap);

  FILE *csv = std::fopen(o.out.c_str(), "w");
  if (!csv) {
    std::fprintf(stderr, "cannot write %s\n", o.out.c_str());
    return 1;
  }
  std::fprintf(csv, "group,chunk,frames,bytes_in,mode,algorithm,settings,"
                    "comp_bytes,ratio,comp_ms,resid_ms,decomp_ms,recon_ms,"
                    "comp_wall_ms,decomp_wall_ms,frame_bytes,reps,ok,note\n");
  const Clock::time_point t0 = Clock::now();
  size_t failed = 0, rows = 0;
  for (size_t gi = 0; gi < groups.size(); ++gi) {
    const Group &g = groups[gi];
    if (!LoadGroup(o.dir, g, o.chunk, o.elem, &w)) {
      std::fprintf(stderr, "skipping %s: unreadable or uneven frames\n",
                   g.label.c_str());
      continue;
    }
    const size_t bytes_in = w.bytes * static_cast<size_t>(nf);
    for (const Mode &m : modes) {
      const std::vector<Step> plan = Plan(nf, m);
      for (Setting &s : settings) {
        Round r;
        if (!s.build_error.empty()) {
          r.note = s.build_error;
        } else {
          if (!s.warmed[m.name]) {
            RunRound(s.codec.get(), m, plan, o.elem, &w);  // untimed
            s.warmed[m.name] = true;
          }
          r = Measure(s.codec.get(), m, plan, o.elem, o.reps, &w);
        }
        std::string fb;
        for (size_t t = 0; t < r.frame_bytes.size(); ++t) {
          fb += (t ? ";" : "") + std::to_string(r.frame_bytes[t]);
        }
        std::fprintf(csv,
                     "%s,%zu,%d,%zu,%s,%s,%s,%zu,%.6f,%.6f,%.6f,%.6f,%.6f,"
                     "%.6f,%.6f,%s,%d,%d,%s\n",
                     g.label.c_str(), g.k, nf, bytes_in, m.name.c_str(),
                     s.spec.base.c_str(), s.spec.Settings().c_str(),
                     r.comp_bytes,
                     r.comp_bytes ? double(bytes_in) / r.comp_bytes : 0.0,
                     r.comp_ms, r.resid_ms, r.decomp_ms, r.recon_ms,
                     r.comp_wall_ms, r.decomp_wall_ms, fb.c_str(), o.reps,
                     r.ok ? 1 : 0, CsvSafe(r.note).c_str());
        ++rows;
        if (!r.ok) ++failed;
      }
    }
    std::fflush(csv);
    std::fprintf(stderr, "[%zu/%zu] %s done (%zu rows, %zu failed, t=%.0f ms)\n",
                 gi + 1, groups.size(), g.label.c_str(), rows, failed,
                 MsSince(t0));
  }
  std::fclose(csv);
  if (failed) std::printf("failed rows: %zu\n", failed);
  return failed ? 4 : 0;
}
