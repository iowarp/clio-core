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
 * @file gpu_codecs.cuh
 * @brief The lossless GPU codecs the sweeps measure, behind one
 * device-to-device interface: the eight nvcomp algorithms with any of their
 * settings, ndzip, GPULZ, SPspeed and SPratio.
 *
 * A codec is named by a CodecSpec: a base name plus key=value settings, e.g.
 * "gdeflate chunk=65536 level=5". Settings per base (defaults in brackets):
 *   every nvcomp base  chunk=<bytes>          nvcomp's internal chunk [65536]
 *   lz4       type=char|short|int [char], bitshuffle=none|msb|lsb [none]
 *   snappy, zstd       (chunk only)
 *   gdeflate, deflate  level=0..5 [1]  (0 entropy-only ... 5 best ratio)
 *   ans       type=char|float16 [char], subchunks=0|4..64 [0 = auto]
 *   cascaded  type=char|short|int|longlong [int], rle=N [2], delta=N [1],
 *             bp=0|1 [1]
 *   bitcomp   algo=0|1 [0] (1 = sparse), type=char|short|int|longlong [char]
 *   ndzip     shape=<fastest dims> [none = 1-D]: "3600" makes the chunk 2-D
 *             rows of 3600 floats, "128x128" makes it 3-D planes of 128x128
 *             (slowest first). Whole rows/planes go through ndzip's 2-D/3-D
 *             compressor; a partial row/plane at the end goes 1-D.
 *   gpulz, spspeed, spratio   (no settings)
 * Every base also takes shuffle=none|byte|bit [none]: a GPU transform of the
 * input as 4-byte words before compressing (undone after decompressing),
 * counted in the codec's time. byte groups byte k of every word together
 * (4 planes); bit groups bit k of every word together (32 planes). A partial
 * word or a last group of fewer than 32 words is passed through unchanged.
 * An unknown base or setting is an error, so a typo never measures a default.
 */
#pragma once

// ndzip.hh uses assert and unqualified size()/data() without including
// their headers, so <cassert> and <iterator> must come first.
#include <cassert>
#include <iterator>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
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

namespace gpu_codecs {

/** @brief One GPU codec behind a device-to-device interface. */
class Codec {
 public:
  /** @param name label written to the CSV. Creates the codec's stream. */
  explicit Codec(std::string name) : name_(std::move(name)) {
    CUDA_CHECK(cudaStreamCreate(&stream_));
  }
  virtual ~Codec() {
    if (owns_stream_) cudaStreamDestroy(stream_);
  }

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
  /** Setup before Decompress (e.g. parsing the stream header). */
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
  bool owns_stream_ = true;  ///< false for a wrapper borrowing a stream
};

/** @brief A codec base name and its settings. */
struct CodecSpec {
  std::string base;                           ///< lz4, ..., spratio
  std::map<std::string, std::string> params;  ///< setting -> value

  /** @return the settings as "k=v k=v" in key order, "" when none. */
  std::string Settings() const {
    std::string s;
    for (const auto &kv : params) {
      if (!s.empty()) s += ' ';
      s += kv.first + '=' + kv.second;
    }
    return s;
  }
};

/**
 * Parse "base k=v k=v ...".
 * @param line whitespace-separated base name and settings
 * @return the spec; throws std::invalid_argument on a malformed token
 */
inline CodecSpec ParseSpec(const std::string &line) {
  CodecSpec spec;
  std::istringstream ss(line);
  ss >> spec.base;
  for (std::string tok; ss >> tok;) {
    const size_t eq = tok.find('=');
    if (eq == 0 || eq == std::string::npos || eq + 1 == tok.size()) {
      throw std::invalid_argument("bad setting '" + tok + "' in: " + line);
    }
    spec.params[tok.substr(0, eq)] = tok.substr(eq + 1);
  }
  if (spec.base.empty()) throw std::invalid_argument("empty codec spec");
  return spec;
}

/** @brief Reads a spec's settings and rejects any it never asked for. */
class SettingReader {
 public:
  explicit SettingReader(const CodecSpec &spec) : spec_(spec) {}
  /** @return the value of key, or def when it is not set. */
  std::string Str(const std::string &key, const std::string &def) {
    used_.insert(key);
    auto it = spec_.params.find(key);
    return it == spec_.params.end() ? def : it->second;
  }
  /** @return the integer value of key, or def; throws on a non-integer. */
  long long Int(const std::string &key, long long def) {
    const std::string v = Str(key, "");
    if (v.empty()) return def;
    size_t used = 0;
    const long long x = std::stoll(v, &used);
    if (used != v.size()) {
      throw std::invalid_argument(key + "=" + v + " is not an integer");
    }
    return x;
  }
  /** Throw when the spec carries a setting no reader asked for. */
  void CheckAllUsed() const {
    for (const auto &kv : spec_.params) {
      if (!used_.count(kv.first)) {
        throw std::invalid_argument("unknown setting " + kv.first + " for " +
                                    spec_.base);
      }
    }
  }

 private:
  const CodecSpec &spec_;
  std::set<std::string> used_;
};

/** @return the nvcomp data type named t; throws on an unknown name. */
inline nvcompType_t ParseType(const std::string &t) {
  static const std::map<std::string, nvcompType_t> m = {
      {"char", NVCOMP_TYPE_CHAR},         {"uchar", NVCOMP_TYPE_UCHAR},
      {"short", NVCOMP_TYPE_SHORT},       {"ushort", NVCOMP_TYPE_USHORT},
      {"int", NVCOMP_TYPE_INT},           {"uint", NVCOMP_TYPE_UINT},
      {"longlong", NVCOMP_TYPE_LONGLONG}, {"ulonglong", NVCOMP_TYPE_ULONGLONG},
      {"float16", NVCOMP_TYPE_FLOAT16}};
  auto it = m.find(t);
  if (it == m.end()) throw std::invalid_argument("unknown type " + t);
  return it->second;
}

/** @return the LZ4 bitshuffle mode named b (none, msb, lsb). */
inline nvcompBitshuffleMode_t ParseBitshuffle(const std::string &b) {
  if (b == "none") return NVCOMP_BITSHUFFLE_NONE;
  if (b == "msb") return NVCOMP_BITSHUFFLE_MSB_FIRST;
  if (b == "lsb") return NVCOMP_BITSHUFFLE_LSB_FIRST;
  throw std::invalid_argument("unknown bitshuffle " + b);
}

using ManagerPtr = std::shared_ptr<nvcomp::nvcompManagerBase>;

/**
 * Build the nvcomp high-level manager a spec describes.
 * @param spec base name (lz4 ... bitcomp) and settings
 * @param s    stream the manager runs on
 * @return the manager, or nullptr when the base is not an nvcomp algorithm
 */
inline ManagerPtr MakeNvcompManager(const CodecSpec &spec, cudaStream_t s) {
  SettingReader r(spec);
  const size_t chunk = static_cast<size_t>(r.Int("chunk", 65536));
  const std::string &b = spec.base;
  ManagerPtr m;
  if (b == "lz4") {
    auto c = nvcompBatchedLZ4CompressDefaultOpts;
    auto d = nvcompBatchedLZ4DecompressDefaultOpts;
    // The decoder needs the same type and mode to undo the bitshuffle.
    c.data_type = d.data_type = ParseType(r.Str("type", "char"));
    c.bitshuffle_mode = d.bitshuffle_mode =
        ParseBitshuffle(r.Str("bitshuffle", "none"));
    m = std::make_shared<nvcomp::LZ4Manager>(chunk, c, d, s);
  } else if (b == "snappy") {
    m = std::make_shared<nvcomp::SnappyManager>(
        chunk, nvcompBatchedSnappyCompressDefaultOpts,
        nvcompBatchedSnappyDecompressDefaultOpts, s);
  } else if (b == "zstd") {
    m = std::make_shared<nvcomp::ZstdManager>(
        chunk, nvcompBatchedZstdCompressDefaultOpts,
        nvcompBatchedZstdDecompressDefaultOpts, s);
  } else if (b == "gdeflate") {
    auto c = nvcompBatchedGdeflateCompressDefaultOpts;
    c.algorithm = static_cast<int>(r.Int("level", 1));
    m = std::make_shared<nvcomp::GdeflateManager>(
        chunk, c, nvcompBatchedGdeflateDecompressDefaultOpts, s);
  } else if (b == "deflate") {
    auto c = nvcompBatchedDeflateCompressDefaultOpts;
    c.algorithm = static_cast<int>(r.Int("level", 1));
    m = std::make_shared<nvcomp::DeflateManager>(
        chunk, c, nvcompBatchedDeflateDecompressDefaultOpts, s);
  } else if (b == "ans") {
    auto c = nvcompBatchedANSCompressDefaultOpts;
    c.data_type = ParseType(r.Str("type", "char"));
    c.max_sub_chunk_count = static_cast<uint8_t>(r.Int("subchunks", 0));
    m = std::make_shared<nvcomp::ANSManager>(
        chunk, c, nvcompBatchedANSDecompressDefaultOpts, s);
  } else if (b == "cascaded") {
    auto c = nvcompBatchedCascadedCompressDefaultOpts;
    c.type = ParseType(r.Str("type", "int"));
    c.num_RLEs = static_cast<int>(r.Int("rle", 2));
    c.num_deltas = static_cast<int>(r.Int("delta", 1));
    c.use_bp = static_cast<int>(r.Int("bp", 1));
    m = std::make_shared<nvcomp::CascadedManager>(
        chunk, c, nvcompBatchedCascadedDecompressDefaultOpts, s);
  } else if (b == "bitcomp") {
    auto c = nvcompBatchedBitcompCompressDefaultOpts;
    c.algorithm = static_cast<int>(r.Int("algo", 0));
    c.data_type = ParseType(r.Str("type", "char"));
    m = std::make_shared<nvcomp::BitcompManager>(
        chunk, c, nvcompBatchedBitcompDecompressDefaultOpts, s);
  } else {
    return nullptr;
  }
  r.CheckAllUsed();
  return m;
}

/** @brief One nvcomp algorithm through its high-level manager. */
class NvcompCodec : public Codec {
 public:
  /**
   * @param name label for the CSV
   * @param spec nvcomp base name and settings
   */
  NvcompCodec(const std::string &name, const CodecSpec &spec) : Codec(name) {
    mgr_ = MakeNvcompManager(spec, stream_);
    if (!mgr_) throw std::invalid_argument("not an nvcomp codec: " + spec.base);
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

/**
 * @brief ndzip's CUDA backend on float32 data, 1-D or shaped.
 *
 * 1-D (no shape): the chunk is one 1-D array, stream = ndzip's words.
 * Shaped: the chunk's whole rows/planes form one 2-D/3-D array and any
 * remainder a 1-D array. Stream = an 8-byte word count of the shaped part,
 * then the shaped part's words, then the remainder's words.
 */
class NdzipCodec : public Codec {
 public:
  /**
   * @param name  label for the CSV
   * @param inner the fastest dimensions, slowest first; empty = 1-D
   */
  NdzipCodec(const std::string &name, std::vector<ndzip::index_type> inner)
      : Codec(name), inner_(std::move(inner)) {
    CUDA_CHECK(cudaMalloc(&d_len_, sizeof(ndzip::index_type)));
    decomp1_ = ndzip::make_cuda_decompressor<float>(1, stream_);
    if (!inner_.empty()) {
      decompn_ = ndzip::make_cuda_decompressor<float>(
          static_cast<ndzip::dim_type>(inner_.size() + 1), stream_);
    }
  }
  ~NdzipCodec() override { cudaFree(d_len_); }

  bool Accepts(size_t n) const override { return n % sizeof(float) == 0; }
  size_t Bound(size_t n) override {
    if (Shaped1D()) {
      return ndzip::compressed_length_bound<float>(Flat(n / 4)) * kWord;
    }
    const Split p = SplitOf(n);
    size_t b = kHeader;
    if (p.rows) b += ndzip::compressed_length_bound<float>(Shaped(p)) * kWord;
    if (p.tail) b += ndzip::compressed_length_bound<float>(Flat(p.tail)) * kWord;
    return b;
  }
  void PrepareCompress(size_t n) override {
    const Split p = SplitOf(n);
    if (Shaped1D()) {
      cur1_ = Get(&comp1_, n, Flat(n / 4));
      return;
    }
    curn_ = p.rows ? Get(&compn_, n, Shaped(p)) : nullptr;
    cur1_ = p.tail ? Get(&comp1_, n, Flat(p.tail)) : nullptr;
  }
  void Compress(const uint8_t *in, size_t n, uint8_t *out, size_t) override {
    const float *x = reinterpret_cast<const float *>(in);
    if (Shaped1D()) {
      cur1_->compress(x, Flat(n / 4), reinterpret_cast<Word *>(out), d_len_);
      return;
    }
    const Split p = SplitOf(n);
    h_shaped_ = 0;
    if (p.rows) {
      curn_->compress(x, Shaped(p), reinterpret_cast<Word *>(out + kHeader),
                      d_len_);
      // The remainder starts right after the shaped words: read their count
      // (an index_type on the device, widened to the 8-byte header here).
      ndzip::index_type words = 0;
      CUDA_CHECK(cudaMemcpyAsync(&words, d_len_, sizeof(words),
                                 cudaMemcpyDeviceToHost, stream_));
      CUDA_CHECK(cudaStreamSynchronize(stream_));
      h_shaped_ = words;
    }
    CUDA_CHECK(cudaMemcpyAsync(out, &h_shaped_, sizeof(h_shaped_),
                               cudaMemcpyHostToDevice, stream_));
    if (p.tail) {
      cur1_->compress(x + p.rows * Inner(), Flat(p.tail),
                      reinterpret_cast<Word *>(out + kHeader +
                                               h_shaped_ * kWord),
                      d_len_);
    } else {
      CUDA_CHECK(cudaMemsetAsync(d_len_, 0, sizeof(ndzip::index_type),
                                 stream_));
    }
  }
  size_t CompressedBytes(const uint8_t *) override {
    ndzip::index_type words = 0;
    CUDA_CHECK(cudaMemcpy(&words, d_len_, sizeof(words),
                          cudaMemcpyDeviceToHost));
    if (Shaped1D()) return static_cast<size_t>(words) * kWord;
    return kHeader + static_cast<size_t>(h_shaped_ + words) * kWord;
  }
  void PrepareDecompress(const uint8_t *in, size_t) override {
    if (!Shaped1D()) {
      CUDA_CHECK(cudaMemcpy(&h_shaped_dec_, in, sizeof(h_shaped_dec_),
                            cudaMemcpyDeviceToHost));
    }
  }
  void Decompress(const uint8_t *in, size_t, uint8_t *out,
                  size_t n) override {
    float *y = reinterpret_cast<float *>(out);
    if (Shaped1D()) {
      decomp1_->decompress(reinterpret_cast<const Word *>(in), y, Flat(n / 4));
      return;
    }
    const Split p = SplitOf(n);
    if (p.rows) {
      decompn_->decompress(reinterpret_cast<const Word *>(in + kHeader), y,
                           Shaped(p));
    }
    if (p.tail) {
      decomp1_->decompress(
          reinterpret_cast<const Word *>(in + kHeader + h_shaped_dec_ * kWord),
          y + p.rows * Inner(), Flat(p.tail));
    }
  }

 private:
  using Word = ndzip::compressed_type<float>;
  using CompMap =
      std::map<size_t, std::unique_ptr<ndzip::cuda_compressor<float>>>;
  static constexpr size_t kWord = sizeof(Word);
  static constexpr size_t kHeader = sizeof(uint64_t);  ///< shaped words
  /** @brief A chunk cut into whole rows/planes and a 1-D remainder. */
  struct Split {
    size_t rows = 0;  ///< whole rows/planes (slowest dimension)
    size_t tail = 0;  ///< leftover floats, compressed 1-D
  };
  /** @return true when the codec runs plain 1-D (no shape set). */
  bool Shaped1D() const { return inner_.empty(); }
  /** @return floats per row/plane. */
  size_t Inner() const {
    size_t p = 1;
    for (auto d : inner_) p *= static_cast<size_t>(d);
    return p;
  }
  /** @return how an n-byte chunk splits into rows/planes and a remainder. */
  Split SplitOf(size_t n) const {
    Split s;
    const size_t floats = n / sizeof(float);
    if (Shaped1D()) return s;
    s.rows = floats / Inner();
    s.tail = floats - s.rows * Inner();
    return s;
  }
  /** @return the 1-D extent of k floats. */
  static ndzip::extent Flat(size_t k) {
    ndzip::extent e(1);
    e[0] = static_cast<ndzip::index_type>(k);
    return e;
  }
  /** @return the 2-D/3-D extent of the whole rows/planes. */
  ndzip::extent Shaped(const Split &p) const {
    ndzip::extent e(static_cast<ndzip::dim_type>(inner_.size() + 1));
    e[0] = static_cast<ndzip::index_type>(p.rows);
    for (size_t d = 0; d < inner_.size(); ++d) e[d + 1] = inner_[d];
    return e;
  }
  /** @return the compressor for this input size, built once per size. */
  ndzip::cuda_compressor<float> *Get(CompMap *m, size_t n,
                                     const ndzip::extent &e) {
    auto &c = (*m)[n];
    if (!c) {
      c = ndzip::make_cuda_compressor<float>(
          ndzip::compressor_requirements{e}, stream_);
    }
    return c.get();
  }

  std::vector<ndzip::index_type> inner_;
  CompMap comp1_, compn_;
  ndzip::cuda_compressor<float> *cur1_ = nullptr, *curn_ = nullptr;
  std::unique_ptr<ndzip::cuda_decompressor<float>> decomp1_, decompn_;
  ndzip::index_type *d_len_ = nullptr;
  uint64_t h_shaped_ = 0;      ///< words of the shaped part, compress side
  uint64_t h_shaped_dec_ = 0;  ///< same, read back from the stream header
};

/** @brief GPULZ (ICS'23) through the gpulz_api.h shim. */
class GpulzCodec : public Codec {
 public:
  /** @param name label for the CSV */
  explicit GpulzCodec(const std::string &name) : Codec(name) {}
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

/**
 * Byte shuffle of m 4-byte words: out[k*m + i] = byte k of word i
 * (inverse: the reverse map).
 */
__global__ void ByteShuffleKernel(const uint8_t *in, uint8_t *out, size_t m,
                                  bool inverse) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  for (; i < m; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    if (!inverse) {
      const uint32_t w = reinterpret_cast<const uint32_t *>(in)[i];
      for (int k = 0; k < 4; ++k) out[k * m + i] = (w >> (8 * k)) & 0xffu;
    } else {
      uint32_t w = 0;
      for (int k = 0; k < 4; ++k) w |= uint32_t(in[k * m + i]) << (8 * k);
      reinterpret_cast<uint32_t *>(out)[i] = w;
    }
  }
}

/**
 * Bit shuffle of g groups of 32 words: plane p, word w holds bit p of words
 * 32w..32w+31 (bit l = word 32w+l); planes are stored one after another.
 * One warp per group: a ballot per bit forward, a shuffle per bit back.
 */
__global__ void BitShuffleKernel(const uint32_t *in, uint32_t *out, size_t g,
                                 bool inverse) {
  const unsigned lane = threadIdx.x & 31u;
  size_t w = (blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x) / 32;
  const size_t warps = static_cast<size_t>(gridDim.x) * blockDim.x / 32;
  for (; w < g; w += warps) {
    if (!inverse) {
      const uint32_t x = in[32 * w + lane];
      for (unsigned p = 0; p < 32; ++p) {
        const uint32_t plane = __ballot_sync(0xffffffffu, (x >> p) & 1u);
        if (lane == p) out[p * g + w] = plane;
      }
    } else {
      const uint32_t mine = in[lane * g + w];  // plane `lane`, group w
      uint32_t x = 0;
      for (unsigned p = 0; p < 32; ++p) {
        const uint32_t plane = __shfl_sync(0xffffffffu, mine, p);
        x |= ((plane >> lane) & 1u) << p;
      }
      out[32 * w + lane] = x;
    }
  }
}

/** @brief How a ShuffleCodec rearranges its input. */
enum class Shuffle { kNone, kByte, kBit };

/** @return the shuffle named s (none, byte, bit); throws otherwise. */
inline Shuffle ParseShuffle(const std::string &s) {
  if (s == "none") return Shuffle::kNone;
  if (s == "byte") return Shuffle::kByte;
  if (s == "bit") return Shuffle::kBit;
  throw std::invalid_argument("unknown shuffle " + s);
}

/**
 * @brief Any codec with a byte or bit shuffle in front of it. The transform
 * runs on the inner codec's stream inside Compress / Decompress, so it is in
 * the timed region; the scratch buffer is sized in PrepareCompress, untimed.
 */
class ShuffleCodec : public Codec {
 public:
  /**
   * @param name  label for the CSV
   * @param inner the codec that compresses the shuffled words
   * @param mode  kByte or kBit
   */
  ShuffleCodec(std::string name, std::unique_ptr<Codec> inner, Shuffle mode)
      : Codec(std::move(name)), inner_(std::move(inner)), mode_(mode) {
    cudaStreamDestroy(stream_);  // run on the inner codec's stream instead
    stream_ = inner_->stream();
    owns_stream_ = false;
  }
  ~ShuffleCodec() override {
    if (scratch_) cudaFree(scratch_);
  }

  size_t Bound(size_t n) override { return inner_->Bound(n); }
  bool Accepts(size_t n) const override { return inner_->Accepts(n); }
  void PrepareCompress(size_t n) override {
    Fit(n);
    inner_->PrepareCompress(n);
  }
  void Compress(const uint8_t *in, size_t n, uint8_t *out,
                size_t cap) override {
    Transform(in, scratch_, n, false);
    inner_->Compress(scratch_, n, out, cap);
  }
  size_t CompressedBytes(const uint8_t *out) override {
    return inner_->CompressedBytes(out);
  }
  void PrepareDecompress(const uint8_t *in, size_t comp_bytes) override {
    inner_->PrepareDecompress(in, comp_bytes);
  }
  void Decompress(const uint8_t *in, size_t comp_bytes, uint8_t *out,
                  size_t n) override {
    inner_->Decompress(in, comp_bytes, scratch_, n);
    Transform(scratch_, out, n, true);
  }
  bool DecompressOk() override { return inner_->DecompressOk(); }

 private:
  /** Grow the scratch buffer to n bytes (untimed). */
  void Fit(size_t n) {
    if (n <= cap_) return;
    if (scratch_) CUDA_CHECK(cudaFree(scratch_));
    CUDA_CHECK(cudaMalloc(&scratch_, n));
    cap_ = n;
  }
  /**
   * Shuffle (or undo it) n bytes from src to dst on stream_; whatever the
   * transform does not cover (a partial word, a last group under 32 words)
   * is copied through unchanged.
   */
  void Transform(const uint8_t *src, uint8_t *dst, size_t n, bool inverse) {
    const size_t words = n / 4;
    size_t covered = 0;
    if (mode_ == Shuffle::kByte && words) {
      ByteShuffleKernel<<<1024, 256, 0, stream_>>>(src, dst, words, inverse);
      covered = words * 4;
    } else if (mode_ == Shuffle::kBit && words >= 32) {
      const size_t g = words / 32;
      BitShuffleKernel<<<1024, 256, 0, stream_>>>(
          reinterpret_cast<const uint32_t *>(src),
          reinterpret_cast<uint32_t *>(dst), g, inverse);
      covered = g * 32 * 4;
    }
    CUDA_CHECK(cudaGetLastError());
    if (covered < n) {
      CUDA_CHECK(cudaMemcpyAsync(dst + covered, src + covered, n - covered,
                                 cudaMemcpyDeviceToDevice, stream_));
    }
  }

  std::unique_ptr<Codec> inner_;
  Shuffle mode_;
  uint8_t *scratch_ = nullptr;
  size_t cap_ = 0;
};

/** @return the dimensions in "128x128" (slowest first); throws on junk. */
inline std::vector<ndzip::index_type> ParseShape(const std::string &s) {
  std::vector<ndzip::index_type> v;
  std::stringstream ss(s);
  for (std::string tok; std::getline(ss, tok, 'x');) {
    size_t used = 0;
    const long long d = tok.empty() ? 0 : std::stoll(tok, &used);
    if (d <= 0 || used != tok.size()) {
      throw std::invalid_argument("bad ndzip shape " + s);
    }
    v.push_back(static_cast<ndzip::index_type>(d));
  }
  if (v.empty() || v.size() > 2) {
    throw std::invalid_argument("ndzip shape needs 1 or 2 dims: " + s);
  }
  return v;
}

/**
 * Build the codec a spec names.
 * @param spec base name and settings
 * @param name label for the CSV; the base name when empty
 * @return the codec, or nullptr for an unknown base; throws
 *         std::invalid_argument on a bad setting
 */
inline std::unique_ptr<Codec> MakeCodec(const CodecSpec &spec,
                                        std::string name = "") {
  if (name.empty()) name = spec.base;
  auto sh = spec.params.find("shuffle");
  if (sh != spec.params.end()) {
    // The shuffle wraps whatever the rest of the spec builds.
    const Shuffle mode = ParseShuffle(sh->second);
    CodecSpec rest = spec;
    rest.params.erase("shuffle");
    auto inner = MakeCodec(rest, name);
    if (!inner || mode == Shuffle::kNone) return inner;
    return std::make_unique<ShuffleCodec>(name, std::move(inner), mode);
  }
  const std::string &b = spec.base;
  if (b == "ndzip") {
    SettingReader r(spec);
    const std::string shape = r.Str("shape", "");
    r.CheckAllUsed();
    return std::make_unique<NdzipCodec>(
        name, shape.empty() ? std::vector<ndzip::index_type>{}
                            : ParseShape(shape));
  }
  const bool plain = b == "gpulz" || b == "spspeed" || b == "spratio";
  if (plain) {
    SettingReader(spec).CheckAllUsed();  // these take no settings
    if (b == "gpulz") return std::make_unique<GpulzCodec>(name);
    if (b == "spspeed") {
      return std::make_unique<FpcCodec>(
          name, FpcCodec::Api{spspeed_compress_bound, spspeed_compress,
                              spspeed_decompressed_size, spspeed_decompress});
    }
    return std::make_unique<FpcCodec>(
        name, FpcCodec::Api{spratio_compress_bound, spratio_compress,
                            spratio_decompressed_size, spratio_decompress});
  }
  static const std::set<std::string> kNvcomp = {
      "lz4", "snappy", "zstd", "gdeflate", "deflate", "ans", "cascaded",
      "bitcomp"};
  if (!kNvcomp.count(b)) return nullptr;
  return std::make_unique<NvcompCodec>(name, spec);
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

/**
 * @return whether n bytes at a and b are identical, checked on the GPU.
 * @param d_bad one device word of scratch
 * @param s     stream to run the comparison on
 */
inline bool BytesEqualOnGpu(const uint8_t *a, const uint8_t *b, size_t n,
                            unsigned int *d_bad, cudaStream_t s) {
  unsigned int bad = 0;
  CUDA_CHECK(cudaMemsetAsync(d_bad, 0, sizeof(bad), s));
  DiffKernel<<<1024, 256, 0, s>>>(a, b, n, d_bad);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpyAsync(&bad, d_bad, sizeof(bad),
                             cudaMemcpyDeviceToHost, s));
  CUDA_CHECK(cudaStreamSynchronize(s));
  return bad == 0;
}

}  // namespace gpu_codecs
