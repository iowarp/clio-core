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
 * @file codec_sweep.cu
 * @brief Standalone GPU codec benchmark: every nvcomp codec, ndzip, GPULZ,
 * SPspeed and SPratio on raw field files, chunk by chunk, with no
 * preprocessing and no clio.
 *
 * Each input file is read whole and copied to the GPU (untimed), then cut into
 * fixed-size chunks. Every codec compresses each chunk device-to-device,
 * decompresses it device-to-device, and the output is compared byte for byte
 * with the original on the GPU. One CSV row per (chunk, codec).
 *
 * Timing. `comp_ms` / `decomp_ms` are CUDA-event times on the codec's own
 * stream around the compress / decompress call. `comp_wall_ms` /
 * `decomp_wall_ms` are the host clock around the same call plus the final
 * sync, so they include launch overhead. Neither includes file I/O, the H2D
 * copy, header parsing (nvcomp configure_*, the SPspeed/SPratio size check)
 * or the verification. GPULZ, SPspeed and SPratio return the compressed size
 * from a synchronous call, so their times include that 4-byte read-back;
 * nvcomp and ndzip read it after the timed region. Each codec gets one
 * untimed warm-up on the first chunk.
 *
 * Usage:
 *   codec_sweep --dir DIR --list FILE --out CSV [--chunk BYTES]
 *               [--codecs a,b,...] [--nvcomp-chunk BYTES]
 * FILE lists one path per line, relative to DIR.
 */

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "gpu_codecs.cuh"

namespace {

using gpu_codecs::Codec;
using gpu_codecs::CodecSpec;

/** @return every codec name, in CSV order. */
std::vector<std::string> AllCodecNames() {
  return {"nvcomp-lz4",      "nvcomp-snappy",  "nvcomp-zstd",
          "nvcomp-gdeflate", "nvcomp-deflate", "nvcomp-ans",
          "nvcomp-cascaded", "nvcomp-bitcomp", "ndzip",
          "gpulz",           "spspeed",        "spratio"};
}

/**
 * The sweep's codecs: nvcomp's default options except Bitcomp, whose default
 * data type (bytes) is set to 4-byte words to match float32.
 * @param name     codec name from AllCodecNames()
 * @param nv_chunk nvcomp's internal chunk size in bytes
 * @return the codec, or nullptr for an unknown name
 */
std::unique_ptr<Codec> MakeCodec(const std::string &name, size_t nv_chunk) {
  CodecSpec spec{name, {}};
  if (name.rfind("nvcomp-", 0) == 0) {
    spec.base = name.substr(7);
    spec.params["chunk"] = std::to_string(nv_chunk);
    if (spec.base == "bitcomp") spec.params["type"] = "uint";
  }
  return gpu_codecs::MakeCodec(spec, name);
}

/** @brief Scratch the sweep reuses for every chunk. */
struct Workspace {
  uint8_t *d_comp = nullptr;  ///< compressed stream, cap bytes
  size_t cap = 0;
  uint8_t *d_dec = nullptr;   ///< decompressed output, one chunk
  unsigned int *d_bad = nullptr;
  cudaEvent_t e0 = nullptr, e1 = nullptr;
};

/**
 * @return whether n bytes at a and b are identical, checked on the GPU.
 * @param s stream to run the comparison on
 */
bool SameOnGpu(const uint8_t *a, const uint8_t *b, size_t n, Workspace *ws,
               cudaStream_t s) {
  return gpu_codecs::BytesEqualOnGpu(a, b, n, ws->d_bad, s);
}

/**
 * Time one enqueued operation on stream s.
 * @param f       enqueues the work
 * @param gpu_ms  CUDA-event time between the two records, ms
 * @param wall_ms host time from before the launch to the end of the sync, ms
 */
void TimeOnStream(cudaStream_t s, Workspace *ws,
                  const std::function<void()> &f, double *gpu_ms,
                  double *wall_ms) {
  using Clock = std::chrono::steady_clock;
  CUDA_CHECK(cudaStreamSynchronize(s));
  const auto w0 = Clock::now();
  CUDA_CHECK(cudaEventRecord(ws->e0, s));
  f();
  CUDA_CHECK(cudaEventRecord(ws->e1, s));
  CUDA_CHECK(cudaEventSynchronize(ws->e1));
  const auto w1 = Clock::now();
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, ws->e0, ws->e1));
  *gpu_ms = ms;
  *wall_ms = std::chrono::duration<double, std::milli>(w1 - w0).count();
}

/** @brief One (chunk, codec) measurement. */
struct Sample {
  size_t comp_bytes = 0;
  double comp_ms = 0, comp_wall_ms = 0, decomp_ms = 0, decomp_wall_ms = 0;
  bool ok = false;
};

/**
 * Compress, decompress and verify one chunk with one codec.
 * @param in device chunk, n bytes
 * @return the measurement; ok=false on any failure or mismatch
 */
Sample RunOne(Codec *c, const uint8_t *in, size_t n, Workspace *ws) {
  Sample r;
  if (!c->Accepts(n)) return r;
  cudaStream_t s = c->stream();
  try {
    c->PrepareCompress(n);
    // A pre-shuffle and its inverse are part of (de)compressing: timed.
    TimeOnStream(s, ws,
                 [&] {
                   const uint8_t *src = c->Preprocess(in, n);
                   c->Compress(src, n, ws->d_comp, ws->cap);
                 },
                 &r.comp_ms, &r.comp_wall_ms);
    r.comp_bytes = c->CompressedBytes(ws->d_comp);
    if (r.comp_bytes == 0 || r.comp_bytes > ws->cap) return r;
    // A codec that writes nothing must not pass on the previous one's output.
    uint8_t *dst = c->DecodeTarget(ws->d_dec);
    CUDA_CHECK(cudaMemsetAsync(ws->d_dec, 0xA5, n, s));
    if (dst != ws->d_dec) CUDA_CHECK(cudaMemsetAsync(dst, 0xA5, n, s));
    c->PrepareDecompress(ws->d_comp, r.comp_bytes);
    TimeOnStream(s, ws,
                 [&] {
                   c->Decompress(ws->d_comp, r.comp_bytes, dst, n);
                   c->Postprocess(ws->d_dec, n);
                 },
                 &r.decomp_ms, &r.decomp_wall_ms);
    if (!c->DecompressOk()) return r;
    r.ok = SameOnGpu(in, ws->d_dec, n, ws, s);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "%s: %s\n", c->name().c_str(), e.what());
    cudaGetLastError();  // clear a non-sticky error so the next codec runs
  }
  return r;
}

/** @brief Command-line options. */
struct Options {
  std::string dir, list, out;
  std::vector<std::string> codecs = AllCodecNames();
  size_t chunk = 4u << 20;      ///< bytes per compressed chunk
  size_t nv_chunk = 64u << 10;  ///< nvcomp's internal chunk size
};

/** @return the options; exits with usage on a missing or unknown flag. */
Options ParseArgs(int argc, char **argv) {
  Options o;
  auto usage = [] {
    std::fprintf(stderr,
                 "usage: codec_sweep --dir DIR --list FILE --out CSV "
                 "[--chunk BYTES] [--codecs a,b] [--nvcomp-chunk BYTES]\n");
    std::exit(1);
  };
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (i + 1 >= argc) usage();
    const char *v = argv[++i];
    if (a == "--dir") o.dir = v;
    else if (a == "--list") o.list = v;
    else if (a == "--out") o.out = v;
    else if (a == "--chunk") o.chunk = std::strtoull(v, nullptr, 10);
    else if (a == "--nvcomp-chunk") o.nv_chunk = std::strtoull(v, nullptr, 10);
    else if (a == "--codecs") {
      o.codecs.clear();
      std::stringstream ss(v);
      for (std::string t; std::getline(ss, t, ',');) o.codecs.push_back(t);
    } else usage();
  }
  if (o.dir.empty() || o.list.empty() || o.out.empty() || o.chunk == 0) {
    usage();
  }
  return o;
}

/** @return the non-empty lines of a file. */
std::vector<std::string> ReadList(const std::string &path) {
  std::vector<std::string> v;
  std::ifstream f(path);
  for (std::string l; std::getline(f, l);) {
    if (!l.empty()) v.push_back(l);
  }
  return v;
}

/**
 * Read a whole file into a growing pinned host buffer.
 * @param path file to read
 * @param buf  pinned buffer, reallocated when too small
 * @param cap  its capacity
 * @return bytes read, or 0 when the file cannot be read
 */
size_t ReadFile(const std::string &path, uint8_t **buf, size_t *cap) {
  FILE *f = std::fopen(path.c_str(), "rb");
  if (!f) return 0;
  std::fseek(f, 0, SEEK_END);
  const long sz = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (sz > 0 && static_cast<size_t>(sz) > *cap) {
    if (*buf) cudaFreeHost(*buf);
    CUDA_CHECK(cudaMallocHost(buf, sz));
    *cap = sz;
  }
  const size_t got = sz > 0 ? std::fread(*buf, 1, sz, f) : 0;
  std::fclose(f);
  return got == static_cast<size_t>(sz) ? got : 0;
}

/** @brief Per-codec running totals for the summary. */
struct Totals {
  size_t chunks = 0, failed = 0, in_bytes = 0, out_bytes = 0;
  double comp_ms = 0, decomp_ms = 0;
};

/** Print aggregate ratio and throughput per codec (successful chunks only). */
void PrintSummary(const std::vector<std::unique_ptr<Codec>> &codecs,
                  const std::vector<Totals> &t) {
  std::printf("\n%-16s %7s %6s %8s %12s %12s %14s %14s\n", "codec", "chunks",
              "failed", "ratio", "comp_GB/s", "decomp_GB/s", "comp_ms_mean",
              "decomp_ms_mean");
  for (size_t i = 0; i < codecs.size(); ++i) {
    const Totals &x = t[i];
    const size_t ok = x.chunks - x.failed;
    const double gb = x.in_bytes / 1e9;
    std::printf("%-16s %7zu %6zu %8.3f %12.2f %12.2f %14.3f %14.3f\n",
                codecs[i]->name().c_str(), x.chunks, x.failed,
                x.out_bytes ? double(x.in_bytes) / x.out_bytes : 0.0,
                x.comp_ms > 0 ? gb / (x.comp_ms / 1e3) : 0.0,
                x.decomp_ms > 0 ? gb / (x.decomp_ms / 1e3) : 0.0,
                ok ? x.comp_ms / ok : 0.0, ok ? x.decomp_ms / ok : 0.0);
  }
}

/**
 * Sweep every chunk of one device-resident file through every codec.
 * @param rel  file name as written to the CSV
 * @param d_in the file on the GPU, size bytes
 * @param warm run one untimed pass per codec on the first chunk first
 */
void SweepFile(const std::string &rel, const uint8_t *d_in, size_t size,
               const Options &o,
               const std::vector<std::unique_ptr<Codec>> &codecs,
               Workspace *ws, std::vector<Totals> *tot, FILE *csv, bool warm) {
  for (size_t off = 0, k = 0; off < size; off += o.chunk, ++k) {
    const size_t n = std::min(o.chunk, size - off);
    if (warm && k == 0) {
      for (auto &c : codecs) RunOne(c.get(), d_in, n, ws);
    }
    for (size_t i = 0; i < codecs.size(); ++i) {
      const Sample r = RunOne(codecs[i].get(), d_in + off, n, ws);
      std::fprintf(csv, "%s,%zu,%zu,%zu,%s,%zu,%.6f,%.4f,%.4f,%.4f,%.4f,%d\n",
                   rel.c_str(), k, off, n, codecs[i]->name().c_str(),
                   r.comp_bytes, r.comp_bytes ? double(n) / r.comp_bytes : 0.0,
                   r.comp_ms, r.comp_wall_ms, r.decomp_ms, r.decomp_wall_ms,
                   r.ok ? 1 : 0);
      Totals &t = (*tot)[i];
      ++t.chunks;
      if (!r.ok) {
        ++t.failed;
        continue;
      }
      t.in_bytes += n;
      t.out_bytes += r.comp_bytes;
      t.comp_ms += r.comp_ms;
      t.decomp_ms += r.decomp_ms;
    }
  }
}

/** @return the codecs named in o, with a workspace sized for all of them. */
std::vector<std::unique_ptr<Codec>> SetUp(const Options &o, Workspace *ws) {
  std::vector<std::unique_ptr<Codec>> codecs;
  for (const auto &name : o.codecs) {
    auto c = MakeCodec(name, o.nv_chunk);
    if (!c) {
      std::fprintf(stderr, "unknown codec %s\n", name.c_str());
      std::exit(1);
    }
    ws->cap = std::max(ws->cap, c->Bound(o.chunk));
    codecs.push_back(std::move(c));
  }
  CUDA_CHECK(cudaMalloc(&ws->d_comp, ws->cap));
  CUDA_CHECK(cudaMalloc(&ws->d_dec, o.chunk));
  CUDA_CHECK(cudaMalloc(&ws->d_bad, sizeof(unsigned int)));
  CUDA_CHECK(cudaEventCreate(&ws->e0));
  CUDA_CHECK(cudaEventCreate(&ws->e1));
  return codecs;
}

}  // namespace

int main(int argc, char **argv) {
  const Options o = ParseArgs(argc, argv);
  const std::vector<std::string> files = ReadList(o.list);
  Workspace ws;
  auto codecs = SetUp(o, &ws);
  FILE *csv = std::fopen(o.out.c_str(), "w");
  if (!csv) {
    std::fprintf(stderr, "cannot write %s\n", o.out.c_str());
    return 1;
  }
  std::fprintf(csv, "file,chunk,offset,bytes,codec,comp_bytes,ratio,comp_ms,"
                    "comp_wall_ms,decomp_ms,decomp_wall_ms,ok\n");
  std::vector<Totals> tot(codecs.size());
  uint8_t *h_buf = nullptr, *d_file = nullptr;
  size_t h_cap = 0, d_cap = 0, unreadable = 0;
  bool warmed = false;
  const auto t0 = std::chrono::steady_clock::now();
  for (size_t f = 0; f < files.size(); ++f) {
    const size_t size = ReadFile(o.dir + "/" + files[f], &h_buf, &h_cap);
    if (size == 0) {
      std::fprintf(stderr, "cannot read %s\n", files[f].c_str());
      ++unreadable;
      continue;
    }
    if (size > d_cap) {
      if (d_file) cudaFree(d_file);
      CUDA_CHECK(cudaMalloc(&d_file, size));
      d_cap = size;
    }
    CUDA_CHECK(cudaMemcpy(d_file, h_buf, size, cudaMemcpyHostToDevice));
    SweepFile(files[f], d_file, size, o, codecs, &ws, &tot, csv, !warmed);
    warmed = true;
    if ((f + 1) % 100 == 0 || f + 1 == files.size()) {
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0).count();
      std::fprintf(stderr, "%zu/%zu files, %.0f ms\n", f + 1, files.size(),
                   ms);
    }
  }
  std::fclose(csv);
  PrintSummary(codecs, tot);
  size_t failed = 0;
  for (const Totals &t : tot) failed += t.failed;
  if (unreadable) std::printf("unreadable files: %zu\n", unreadable);
  if (failed) std::printf("failed round trips: %zu\n", failed);
  // Nonzero on any failure, so a SLURM afterok chain stops at a bad run.
  return unreadable ? 3 : failed ? 4 : 0;
}
