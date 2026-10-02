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

// ndzip.hh uses assert and unqualified size()/data() without including
// their headers, so <cassert> and <iterator> must come first.
#include <cassert>
#include <iterator>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <fpc_api.h>
#include <gpulz_api.h>
#include <ndzip/cuda.hh>
#include <nvcomp/nvcompManagerFactory.hpp>

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

/** @brief One GPU codec behind a device-to-device interface. */
class Codec {
 public:
  /** @param name label written to the CSV. Creates the codec's stream. */
  explicit Codec(std::string name) : name_(std::move(name)) {
    CUDA_CHECK(cudaStreamCreate(&stream_));
  }
  virtual ~Codec() { cudaStreamDestroy(stream_); }

  /** @return worst-case compressed bytes for n input bytes. */
  virtual size_t Bound(size_t n) = 0;
  /** @return false when the codec cannot take an n-byte chunk at all. */
  virtual bool Accepts(size_t n) const { (void)n; return true; }
  /** Untimed setup before Compress (configuration, per-size objects). */
  virtual void PrepareCompress(size_t n) { (void)n; }
  /**
   * Enqueue compression on stream().
   * @param in  device input, n bytes
   * @param n   input bytes
   * @param out device output, cap bytes
   * @param cap capacity of out
   */
  virtual void Compress(const uint8_t *in, size_t n, uint8_t *out,
                        size_t cap) = 0;
  /** @return compressed bytes after the stream is idle, 0 on failure. */
  virtual size_t CompressedBytes(const uint8_t *out) = 0;
  /** Untimed setup before Decompress (e.g. parsing the stream header). */
  virtual void PrepareDecompress(const uint8_t *in, size_t comp_bytes) {
    (void)in;
    (void)comp_bytes;
  }
  /**
   * Enqueue decompression on stream().
   * @param in         device compressed stream
   * @param comp_bytes its length
   * @param out        device output, n bytes
   * @param n          decompressed bytes expected
   */
  virtual void Decompress(const uint8_t *in, size_t comp_bytes, uint8_t *out,
                          size_t n) = 0;
  /** @return whether the last decompression reported success. */
  virtual bool DecompressOk() { return true; }

  const std::string &name() const { return name_; }
  cudaStream_t stream() const { return stream_; }

 protected:
  std::string name_;
  cudaStream_t stream_ = nullptr;
};

using ManagerPtr = std::shared_ptr<nvcomp::nvcompManagerBase>;
using ManagerMaker = std::function<ManagerPtr(size_t, cudaStream_t)>;

/**
 * @brief nvcomp's high-level managers with their default options.
 *
 * The one exception is Bitcomp's data type: its default is UCHAR, which models
 * the buffer as bytes; it is set to 4-byte words to match float32. Cascaded's
 * default type is already 4-byte INT.
 * @return codec name -> manager factory (nvcomp chunk size, stream)
 */
const std::map<std::string, ManagerMaker> &NvcompMakers() {
  static const std::map<std::string, ManagerMaker> m = {
      {"nvcomp-lz4", [](size_t c, cudaStream_t s) -> ManagerPtr {
         return std::make_shared<nvcomp::LZ4Manager>(
             c, nvcompBatchedLZ4CompressDefaultOpts,
             nvcompBatchedLZ4DecompressDefaultOpts, s);
       }},
      {"nvcomp-snappy", [](size_t c, cudaStream_t s) -> ManagerPtr {
         return std::make_shared<nvcomp::SnappyManager>(
             c, nvcompBatchedSnappyCompressDefaultOpts,
             nvcompBatchedSnappyDecompressDefaultOpts, s);
       }},
      {"nvcomp-zstd", [](size_t c, cudaStream_t s) -> ManagerPtr {
         return std::make_shared<nvcomp::ZstdManager>(
             c, nvcompBatchedZstdCompressDefaultOpts,
             nvcompBatchedZstdDecompressDefaultOpts, s);
       }},
      {"nvcomp-gdeflate", [](size_t c, cudaStream_t s) -> ManagerPtr {
         return std::make_shared<nvcomp::GdeflateManager>(
             c, nvcompBatchedGdeflateCompressDefaultOpts,
             nvcompBatchedGdeflateDecompressDefaultOpts, s);
       }},
      {"nvcomp-deflate", [](size_t c, cudaStream_t s) -> ManagerPtr {
         return std::make_shared<nvcomp::DeflateManager>(
             c, nvcompBatchedDeflateCompressDefaultOpts,
             nvcompBatchedDeflateDecompressDefaultOpts, s);
       }},
      {"nvcomp-ans", [](size_t c, cudaStream_t s) -> ManagerPtr {
         return std::make_shared<nvcomp::ANSManager>(
             c, nvcompBatchedANSCompressDefaultOpts,
             nvcompBatchedANSDecompressDefaultOpts, s);
       }},
      {"nvcomp-cascaded", [](size_t c, cudaStream_t s) -> ManagerPtr {
         return std::make_shared<nvcomp::CascadedManager>(
             c, nvcompBatchedCascadedCompressDefaultOpts,
             nvcompBatchedCascadedDecompressDefaultOpts, s);
       }},
      {"nvcomp-bitcomp", [](size_t c, cudaStream_t s) -> ManagerPtr {
         nvcompBatchedBitcompCompressOpts_t o =
             nvcompBatchedBitcompCompressDefaultOpts;
         o.data_type = NVCOMP_TYPE_UINT;
         return std::make_shared<nvcomp::BitcompManager>(
             c, o, nvcompBatchedBitcompDecompressDefaultOpts, s);
       }},
  };
  return m;
}

/** @brief One nvcomp algorithm through its high-level manager. */
class NvcompCodec : public Codec {
 public:
  /**
   * @param name     key of NvcompMakers()
   * @param nv_chunk nvcomp's internal chunk size in bytes
   */
  NvcompCodec(const std::string &name, size_t nv_chunk) : Codec(name) {
    mgr_ = NvcompMakers().at(name)(nv_chunk, stream_);
  }
  size_t Bound(size_t n) override {
    return mgr_->configure_compression(n).max_compressed_buffer_size;
  }
  void PrepareCompress(size_t n) override {
    ccfg_ = mgr_->configure_compression(n);
  }
  void Compress(const uint8_t *in, size_t, uint8_t *out, size_t) override {
    mgr_->compress(in, out, ccfg_);
  }
  size_t CompressedBytes(const uint8_t *out) override {
    if (*ccfg_.get_status() != nvcompSuccess) return 0;
    return mgr_->get_compressed_output_size(out);
  }
  void PrepareDecompress(const uint8_t *in, size_t comp_bytes) override {
    dcfg_ = mgr_->configure_decompression(in, &comp_bytes);
  }
  void Decompress(const uint8_t *in, size_t, uint8_t *out,
                  size_t n) override {
    want_ = n;
    mgr_->decompress(out, in, dcfg_);
  }
  bool DecompressOk() override {
    return *dcfg_.get_status() == nvcompSuccess &&
           dcfg_.decomp_data_size == want_;
  }

 private:
  ManagerPtr mgr_;
  nvcomp::CompressionConfig ccfg_;
  nvcomp::DecompressionConfig dcfg_;
  size_t want_ = 0;
};

/** @brief ndzip's CUDA backend on 1-D float32 data. */
class NdzipCodec : public Codec {
 public:
  NdzipCodec() : Codec("ndzip") {
    CUDA_CHECK(cudaMalloc(&d_len_, sizeof(ndzip::index_type)));
    decomp_ = ndzip::make_cuda_decompressor<float>(1, stream_);
  }
  ~NdzipCodec() override { cudaFree(d_len_); }

  bool Accepts(size_t n) const override { return n % sizeof(float) == 0; }
  size_t Bound(size_t n) override {
    return ndzip::compressed_length_bound<float>(Extent(n)) * sizeof(Word);
  }
  void PrepareCompress(size_t n) override {
    if (comp_ && comp_n_ == n) return;  // one compressor per chunk size
    comp_ = ndzip::make_cuda_compressor<float>(
        ndzip::compressor_requirements{Extent(n)}, stream_);
    comp_n_ = n;
  }
  void Compress(const uint8_t *in, size_t n, uint8_t *out, size_t) override {
    comp_->compress(reinterpret_cast<const float *>(in), Extent(n),
                    reinterpret_cast<Word *>(out), d_len_);
  }
  size_t CompressedBytes(const uint8_t *) override {
    ndzip::index_type words = 0;
    CUDA_CHECK(cudaMemcpy(&words, d_len_, sizeof(words),
                          cudaMemcpyDeviceToHost));
    return static_cast<size_t>(words) * sizeof(Word);
  }
  void Decompress(const uint8_t *in, size_t, uint8_t *out,
                  size_t n) override {
    decomp_->decompress(reinterpret_cast<const Word *>(in),
                        reinterpret_cast<float *>(out), Extent(n));
  }

 private:
  using Word = ndzip::compressed_type<float>;
  /** @return the 1-D extent of n bytes of float32. */
  static ndzip::extent Extent(size_t n) {
    ndzip::extent e(1);
    e[0] = static_cast<ndzip::index_type>(n / sizeof(float));
    return e;
  }
  std::unique_ptr<ndzip::cuda_compressor<float>> comp_;
  std::unique_ptr<ndzip::cuda_decompressor<float>> decomp_;
  size_t comp_n_ = 0;
  ndzip::index_type *d_len_ = nullptr;
};

/** @brief GPULZ (ICS'23) through the gpulz_api.h shim. */
class GpulzCodec : public Codec {
 public:
  GpulzCodec() : Codec("gpulz") {}
  size_t Bound(size_t n) override { return gpulz_compress_bound(n); }
  void Compress(const uint8_t *in, size_t n, uint8_t *out,
                size_t cap) override {
    size_t b = 0;
    bytes_ = gpulz_compress(in, n, out, cap, &b, stream_) == 0 ? b : 0;
  }
  size_t CompressedBytes(const uint8_t *) override { return bytes_; }
  void Decompress(const uint8_t *in, size_t comp_bytes, uint8_t *out,
                  size_t n) override {
    drc_ = gpulz_decompress(in, comp_bytes, out, n, stream_);
  }
  bool DecompressOk() override { return drc_ == 0; }

 private:
  size_t bytes_ = 0;
  int drc_ = 0;
};

/** @brief SPspeed or SPratio (burtscher/FPcompress) through fpc_api.h. */
class FpcCodec : public Codec {
 public:
  /** One algorithm's four fpc_api.h entry points. */
  struct Api {
    size_t (*bound)(size_t);
    int (*compress)(const void *, size_t, void *, size_t, size_t *,
                    cudaStream_t);
    long long (*declared)(const void *, size_t, cudaStream_t);
    int (*decompress)(const void *, size_t, void *, size_t, cudaStream_t);
  };
  /** @param name CSV label; @param api the algorithm's entry points */
  FpcCodec(std::string name, Api api) : Codec(std::move(name)), api_(api) {}
  size_t Bound(size_t n) override { return api_.bound(n); }
  void Compress(const uint8_t *in, size_t n, uint8_t *out,
                size_t cap) override {
    size_t b = 0;
    bytes_ = api_.compress(in, n, out, cap, &b, stream_) == 0 ? b : 0;
  }
  size_t CompressedBytes(const uint8_t *) override { return bytes_; }
  void PrepareDecompress(const uint8_t *in, size_t comp_bytes) override {
    declared_ = api_.declared(in, comp_bytes, stream_);
  }
  void Decompress(const uint8_t *in, size_t comp_bytes, uint8_t *out,
                  size_t n) override {
    // The decoder writes as many bytes as the stream declares; never more
    // than the caller's buffer.
    drc_ = declared_ == static_cast<long long>(n)
               ? api_.decompress(in, comp_bytes, out, n, stream_)
               : -3;
  }
  bool DecompressOk() override { return drc_ == 0; }

 private:
  Api api_;
  size_t bytes_ = 0;
  long long declared_ = -1;
  int drc_ = 0;
};

/** @return every codec name, in CSV order. */
std::vector<std::string> AllCodecNames() {
  return {"nvcomp-lz4",      "nvcomp-snappy",  "nvcomp-zstd",
          "nvcomp-gdeflate", "nvcomp-deflate", "nvcomp-ans",
          "nvcomp-cascaded", "nvcomp-bitcomp", "ndzip",
          "gpulz",           "spspeed",        "spratio"};
}

/**
 * @param name     codec name from AllCodecNames()
 * @param nv_chunk nvcomp's internal chunk size in bytes
 * @return the codec, or nullptr for an unknown name
 */
std::unique_ptr<Codec> MakeCodec(const std::string &name, size_t nv_chunk) {
  if (name == "ndzip") return std::make_unique<NdzipCodec>();
  if (name == "gpulz") return std::make_unique<GpulzCodec>();
  if (name == "spspeed") {
    return std::make_unique<FpcCodec>(
        name, FpcCodec::Api{spspeed_compress_bound, spspeed_compress,
                            spspeed_decompressed_size, spspeed_decompress});
  }
  if (name == "spratio") {
    return std::make_unique<FpcCodec>(
        name, FpcCodec::Api{spratio_compress_bound, spratio_compress,
                            spratio_decompressed_size, spratio_decompress});
  }
  if (NvcompMakers().count(name)) {
    return std::make_unique<NvcompCodec>(name, nv_chunk);
  }
  return nullptr;
}

/** Sets *bad to 1 when a and b differ anywhere in their first n bytes. */
__global__ void DiffKernel(const uint8_t *a, const uint8_t *b, size_t n,
                           unsigned int *bad) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  for (; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    if (a[i] != b[i]) {
      *bad = 1;
      return;
    }
  }
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
  unsigned int bad = 0;
  CUDA_CHECK(cudaMemsetAsync(ws->d_bad, 0, sizeof(bad), s));
  DiffKernel<<<1024, 256, 0, s>>>(a, b, n, ws->d_bad);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpyAsync(&bad, ws->d_bad, sizeof(bad),
                             cudaMemcpyDeviceToHost, s));
  CUDA_CHECK(cudaStreamSynchronize(s));
  return bad == 0;
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
    TimeOnStream(s, ws, [&] { c->Compress(in, n, ws->d_comp, ws->cap); },
                 &r.comp_ms, &r.comp_wall_ms);
    r.comp_bytes = c->CompressedBytes(ws->d_comp);
    if (r.comp_bytes == 0 || r.comp_bytes > ws->cap) return r;
    // A codec that writes nothing must not pass on the previous one's output.
    CUDA_CHECK(cudaMemsetAsync(ws->d_dec, 0xA5, n, s));
    c->PrepareDecompress(ws->d_comp, r.comp_bytes);
    TimeOnStream(s, ws,
                 [&] { c->Decompress(ws->d_comp, r.comp_bytes, ws->d_dec, n); },
                 &r.decomp_ms, &r.decomp_wall_ms);
    r.ok = c->DecompressOk() && SameOnGpu(in, ws->d_dec, n, ws, s);
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
