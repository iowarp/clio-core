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
 * @file cpu_codecs.h
 * @brief Lossless CPU codecs behind one host-memory interface, for
 * cpu_corpus_sweep. Spec lines mirror gpu_codecs.cuh: "base k=v k=v".
 *
 *   zstd    level=N            ZSTD_compressCCtx
 *   lz4     level=0|N          0 = LZ4_compress_default, else LZ4-HC level N
 *   zlib    level=N            compress2
 *   bzip2   level=N            block size N x 100 kB
 *   xz      preset=N           lzma_easy_buffer_encode, no integrity check
 *   brotli  quality=N          generic mode, default window
 *   snappy, lzo                no settings (LZO = lzo1x_1)
 *   store                      no settings: stored uncompressed (memcpy)
 *   blosc2  codec=blosclz|lz4|zstd filter=shuffle|bitshuffle|bytedelta
 *           clevel=N type=float|double     (bytedelta = shuffle + bytedelta)
 *   fpzip   type= shape=       lossless (prec 0)
 *   zfp     type= shape=       reversible (lossless) mode
 *   ndzip   type= shape=       ndzip's CPU path
 * Every codec also takes
 *   shuffle=byte      an HDF5-style byte shuffle in front, at the element
 *                     size of type=float|double (default float)
 *   shuffle=bit       gpu_codecs.cuh's 32-word bit shuffle in front
 *                     Either shuffle runs untimed.
 *   threads=N         N > 1: the input is cut into N equal blocks (whole
 *                     rows/planes for a shape) compressed independently,
 *                     one OpenMP thread each; the block table is in the
 *                     stream. N = 1: one stream, one thread.
 * shape= names the fastest dimensions slowest first ("3600" = 2-D rows of
 * 3600, "256x256" = 3-D planes of 256x256); a chunk becomes whole rows or
 * planes plus a 1-D remainder.
 */

#ifndef PAPER_BENCHMARK_CODEC_SWEEP_CPU_CODECS_H_
#define PAPER_BENCHMARK_CODEC_SWEEP_CPU_CODECS_H_

#include <blosc2.h>
#include <blosc2/filters-registry.h>
#include <brotli/decode.h>
#include <brotli/encode.h>
#include <bzlib.h>
#include <fpzip.h>
#include <lz4.h>
#include <lz4hc.h>
#include <lzma.h>
#include <lzo/lzo1x.h>
#include <omp.h>
#include <snappy-c.h>
#include <zfp.h>
#include <zlib.h>
#include <zstd.h>

// ndzip/ndzip.hh uses assert() and size_t without including their headers.
#include <cassert>
#include <cstddef>
#include <ndzip/ndzip.hh>
#include <ndzip/offload.hh>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cpu_codecs {

/** @brief One lossless CPU codec working on host memory. */
class Codec {
 public:
  virtual ~Codec() = default;
  /** @return worst-case compressed bytes for n input bytes. */
  virtual size_t Bound(size_t n) = 0;
  /**
   * Compress n bytes.
   * @param in  input
   * @param n   input bytes
   * @param out output, cap bytes
   * @param cap capacity of out
   * @return compressed bytes, 0 on failure
   */
  virtual size_t Compress(const uint8_t *in, size_t n, uint8_t *out,
                          size_t cap) = 0;
  /**
   * Decompress a stream back to n bytes.
   * @param in         compressed stream
   * @param comp_bytes its length
   * @param out        output, n bytes
   * @param n          decompressed bytes expected
   * @return true on success
   */
  virtual bool Decompress(const uint8_t *in, size_t comp_bytes, uint8_t *out,
                          size_t n) = 0;
  /**
   * Untimed transform before Compress (a shuffle).
   * @return the buffer to hand to Compress: in itself, or codec scratch
   */
  virtual const uint8_t *Preprocess(const uint8_t *in, size_t n) {
    (void)n;
    return in;
  }
  /** @return where Decompress must write: out itself, or codec scratch. */
  virtual uint8_t *DecodeTarget(uint8_t *out) { return out; }
  /** Untimed transform after Decompress, from DecodeTarget(out) into out. */
  virtual void Postprocess(uint8_t *out, size_t n) {
    (void)out;
    (void)n;
  }
};

/** @brief A codec base name and its settings. */
struct CodecSpec {
  std::string base;
  std::map<std::string, std::string> params;

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

// ---------------------------------------------------------------------------
// General-purpose byte codecs
// ---------------------------------------------------------------------------

/** @brief Zstandard at one level, reusing its contexts. */
class ZstdCodec : public Codec {
 public:
  /** @param level ZSTD compression level */
  explicit ZstdCodec(int level)
      : level_(level), cctx_(ZSTD_createCCtx()), dctx_(ZSTD_createDCtx()) {}
  ~ZstdCodec() override {
    ZSTD_freeCCtx(cctx_);
    ZSTD_freeDCtx(dctx_);
  }
  size_t Bound(size_t n) override { return ZSTD_compressBound(n); }
  size_t Compress(const uint8_t *in, size_t n, uint8_t *out,
                  size_t cap) override {
    const size_t r = ZSTD_compressCCtx(cctx_, out, cap, in, n, level_);
    return ZSTD_isError(r) ? 0 : r;
  }
  bool Decompress(const uint8_t *in, size_t cb, uint8_t *out,
                  size_t n) override {
    return ZSTD_decompressDCtx(dctx_, out, n, in, cb) == n;
  }

 private:
  int level_;
  ZSTD_CCtx *cctx_;
  ZSTD_DCtx *dctx_;
};

/** @brief LZ4 (level 0) or LZ4-HC at a level, with preallocated HC state. */
class Lz4Codec : public Codec {
 public:
  /** @param level 0 = LZ4 default, else the LZ4-HC level */
  explicit Lz4Codec(int level)
      : level_(level), state_(level ? LZ4_sizeofStateHC() : 0) {}
  size_t Bound(size_t n) override { return LZ4_compressBound(int(n)); }
  size_t Compress(const uint8_t *in, size_t n, uint8_t *out,
                  size_t cap) override {
    const char *s = reinterpret_cast<const char *>(in);
    char *d = reinterpret_cast<char *>(out);
    const int r = level_ ? LZ4_compress_HC_extStateHC(state_.data(), s, d,
                                                      int(n), int(cap), level_)
                         : LZ4_compress_default(s, d, int(n), int(cap));
    return r > 0 ? size_t(r) : 0;
  }
  bool Decompress(const uint8_t *in, size_t cb, uint8_t *out,
                  size_t n) override {
    return LZ4_decompress_safe(reinterpret_cast<const char *>(in),
                               reinterpret_cast<char *>(out), int(cb),
                               int(n)) == int(n);
  }

 private:
  int level_;
  std::vector<char> state_;
};

/** @brief zlib (deflate) at one level. */
class ZlibCodec : public Codec {
 public:
  /** @param level zlib level 1-9 */
  explicit ZlibCodec(int level) : level_(level) {}
  size_t Bound(size_t n) override { return compressBound(uLong(n)); }
  size_t Compress(const uint8_t *in, size_t n, uint8_t *out,
                  size_t cap) override {
    uLongf len = uLongf(cap);
    return compress2(out, &len, in, uLong(n), level_) == Z_OK ? len : 0;
  }
  bool Decompress(const uint8_t *in, size_t cb, uint8_t *out,
                  size_t n) override {
    uLongf len = uLongf(n);
    return uncompress(out, &len, in, uLong(cb)) == Z_OK && len == n;
  }

 private:
  int level_;
};

/** @brief bzip2 with a block size of level x 100 kB. */
class Bzip2Codec : public Codec {
 public:
  /** @param level 1-9 */
  explicit Bzip2Codec(int level) : level_(level) {}
  size_t Bound(size_t n) override { return n + n / 100 + 600; }
  size_t Compress(const uint8_t *in, size_t n, uint8_t *out,
                  size_t cap) override {
    unsigned len = unsigned(cap);
    const int r = BZ2_bzBuffToBuffCompress(
        reinterpret_cast<char *>(out), &len,
        const_cast<char *>(reinterpret_cast<const char *>(in)), unsigned(n),
        level_, 0, 30);
    return r == BZ_OK ? len : 0;
  }
  bool Decompress(const uint8_t *in, size_t cb, uint8_t *out,
                  size_t n) override {
    unsigned len = unsigned(n);
    const int r = BZ2_bzBuffToBuffDecompress(
        reinterpret_cast<char *>(out), &len,
        const_cast<char *>(reinterpret_cast<const char *>(in)), unsigned(cb),
        0, 0);
    return r == BZ_OK && len == n;
  }

 private:
  int level_;
};

/** @brief xz (LZMA2) at one preset, without an integrity check. */
class XzCodec : public Codec {
 public:
  /** @param preset 0-9 */
  explicit XzCodec(uint32_t preset) : preset_(preset) {}
  size_t Bound(size_t n) override { return lzma_stream_buffer_bound(n); }
  size_t Compress(const uint8_t *in, size_t n, uint8_t *out,
                  size_t cap) override {
    size_t pos = 0;
    const lzma_ret r = lzma_easy_buffer_encode(preset_, LZMA_CHECK_NONE,
                                               nullptr, in, n, out, &pos, cap);
    return r == LZMA_OK ? pos : 0;
  }
  bool Decompress(const uint8_t *in, size_t cb, uint8_t *out,
                  size_t n) override {
    uint64_t memlimit = UINT64_MAX;
    size_t in_pos = 0, out_pos = 0;
    const lzma_ret r = lzma_stream_buffer_decode(
        &memlimit, 0, nullptr, in, &in_pos, cb, out, &out_pos, n);
    return r == LZMA_OK && out_pos == n;
  }

 private:
  uint32_t preset_;
};

/** @brief Brotli at one quality, generic mode, default window. */
class BrotliCodec : public Codec {
 public:
  /** @param quality 0-11 */
  explicit BrotliCodec(int quality) : quality_(quality) {}
  size_t Bound(size_t n) override { return BrotliEncoderMaxCompressedSize(n); }
  size_t Compress(const uint8_t *in, size_t n, uint8_t *out,
                  size_t cap) override {
    size_t len = cap;
    return BrotliEncoderCompress(quality_, BROTLI_DEFAULT_WINDOW,
                                 BROTLI_MODE_GENERIC, n, in, &len, out)
               ? len
               : 0;
  }
  bool Decompress(const uint8_t *in, size_t cb, uint8_t *out,
                  size_t n) override {
    size_t len = n;
    return BrotliDecoderDecompress(cb, in, &len, out) ==
               BROTLI_DECODER_RESULT_SUCCESS &&
           len == n;
  }

 private:
  int quality_;
};

/** @brief Stored uncompressed: the stream is the input, copied (ratio 1). */
class StoreCodec : public Codec {
 public:
  size_t Bound(size_t n) override { return n; }
  size_t Compress(const uint8_t *in, size_t n, uint8_t *out,
                  size_t cap) override {
    if (cap < n) return 0;
    std::memcpy(out, in, n);
    return n;
  }
  bool Decompress(const uint8_t *in, size_t cb, uint8_t *out,
                  size_t n) override {
    if (cb != n) return false;
    std::memcpy(out, in, n);
    return true;
  }
};

/** @brief Snappy. */
class SnappyCodec : public Codec {
 public:
  size_t Bound(size_t n) override { return snappy_max_compressed_length(n); }
  size_t Compress(const uint8_t *in, size_t n, uint8_t *out,
                  size_t cap) override {
    size_t len = cap;
    return snappy_compress(reinterpret_cast<const char *>(in), n,
                           reinterpret_cast<char *>(out), &len) == SNAPPY_OK
               ? len
               : 0;
  }
  bool Decompress(const uint8_t *in, size_t cb, uint8_t *out,
                  size_t n) override {
    size_t len = n;
    return snappy_uncompress(reinterpret_cast<const char *>(in), cb,
                             reinterpret_cast<char *>(out), &len) ==
               SNAPPY_OK &&
           len == n;
  }
};

/** @brief LZO1X-1 with its own work memory. */
class LzoCodec : public Codec {
 public:
  LzoCodec() : wrk_(LZO1X_1_MEM_COMPRESS) {
    if (lzo_init() != LZO_E_OK) throw std::runtime_error("lzo_init failed");
  }
  size_t Bound(size_t n) override { return n + n / 16 + 64 + 3; }
  size_t Compress(const uint8_t *in, size_t n, uint8_t *out,
                  size_t cap) override {
    (void)cap;
    lzo_uint len = 0;
    return lzo1x_1_compress(in, n, out, &len, wrk_.data()) == LZO_E_OK ? len
                                                                       : 0;
  }
  bool Decompress(const uint8_t *in, size_t cb, uint8_t *out,
                  size_t n) override {
    lzo_uint len = n;
    return lzo1x_decompress_safe(in, cb, out, &len, nullptr) == LZO_E_OK &&
           len == n;
  }

 private:
  std::vector<unsigned char> wrk_;
};

/** @brief Blosc2: a shuffle-family filter in front of an LZ codec. */
class Blosc2Codec : public Codec {
 public:
  /**
   * @param codec    BLOSC_BLOSCLZ, BLOSC_LZ4 or BLOSC_ZSTD
   * @param filter   "shuffle", "bitshuffle" or "bytedelta"
   * @param clevel   0-9
   * @param typesize element bytes (4 float, 8 double)
   */
  Blosc2Codec(int codec, const std::string &filter, int clevel, int typesize) {
    static const int kInit = (blosc2_init(), 0);
    (void)kInit;
    blosc2_cparams cp = BLOSC2_CPARAMS_DEFAULTS;
    cp.compcode = uint8_t(codec);
    cp.clevel = uint8_t(clevel);
    cp.typesize = int32_t(typesize);
    cp.nthreads = 1;
    for (auto &f : cp.filters) f = BLOSC_NOFILTER;
    const int last = BLOSC2_MAX_FILTERS - 1;
    if (filter == "shuffle") {
      cp.filters[last] = BLOSC_SHUFFLE;
    } else if (filter == "bitshuffle") {
      cp.filters[last] = BLOSC_BITSHUFFLE;
    } else if (filter == "bytedelta") {
      cp.filters[last - 1] = BLOSC_SHUFFLE;
      cp.filters[last] = BLOSC_FILTER_BYTEDELTA;
      cp.filters_meta[last] = uint8_t(typesize);
    } else {
      throw std::invalid_argument("unknown blosc2 filter " + filter);
    }
    blosc2_dparams dp = BLOSC2_DPARAMS_DEFAULTS;
    dp.nthreads = 1;
    cctx_ = blosc2_create_cctx(cp);
    dctx_ = blosc2_create_dctx(dp);
  }
  ~Blosc2Codec() override {
    blosc2_free_ctx(cctx_);
    blosc2_free_ctx(dctx_);
  }
  size_t Bound(size_t n) override { return n + BLOSC2_MAX_OVERHEAD; }
  size_t Compress(const uint8_t *in, size_t n, uint8_t *out,
                  size_t cap) override {
    const int r = blosc2_compress_ctx(cctx_, in, int32_t(n), out,
                                      int32_t(cap));
    return r > 0 ? size_t(r) : 0;
  }
  bool Decompress(const uint8_t *in, size_t cb, uint8_t *out,
                  size_t n) override {
    return blosc2_decompress_ctx(dctx_, in, int32_t(cb), out, int32_t(n)) ==
           int(n);
  }

 private:
  blosc2_context *cctx_ = nullptr;
  blosc2_context *dctx_ = nullptr;
};

// ---------------------------------------------------------------------------
// Floating-point codecs: shaped part + 1-D remainder + raw tail bytes
// ---------------------------------------------------------------------------

/** @brief Array dimensions, slowest first, of one float/double block. */
struct Dims {
  int rank = 1;
  size_t n[3] = {0, 1, 1};
  /** @return the element count. */
  size_t Count() const {
    size_t c = 1;
    for (int d = 0; d < rank; ++d) c *= n[d];
    return c;
  }
};

/**
 * @brief A codec for float or double arrays. A chunk becomes whole
 * rows/planes of the shape (one array), a 1-D remainder (a second array) and
 * any bytes past the last whole element, stored raw. Stream: two 8-byte
 * lengths (shaped, remainder), the two streams, then the raw bytes.
 */
class FloatCodec : public Codec {
 public:
  /**
   * @param elem  element bytes, 4 or 8
   * @param inner the fastest dimensions, slowest first; empty = 1-D
   */
  FloatCodec(size_t elem, std::vector<size_t> inner)
      : elem_(elem), inner_(std::move(inner)) {}
  /** @return element bytes. */
  size_t elem() const { return elem_; }
  /** @return bytes per whole row/plane (one element for 1-D). */
  size_t Quantum() const { return Inner() * elem_; }

  size_t Bound(size_t n) override {
    const Split p = SplitOf(n);
    size_t b = kHeader + p.raw;
    if (p.rows) b += BoundDims(Shaped(p));
    if (p.tail) b += BoundDims(Flat(p.tail));
    return b;
  }
  size_t Compress(const uint8_t *in, size_t n, uint8_t *out,
                  size_t cap) override {
    const Split p = SplitOf(n);
    uint64_t len[2] = {0, 0};
    size_t pos = kHeader;
    if (p.rows) {
      len[0] = CompressDims(in, Shaped(p), out + pos, cap - pos);
      if (!len[0]) return 0;
      pos += len[0];
    }
    if (p.tail) {
      len[1] = CompressDims(in + p.rows * Quantum(), Flat(p.tail), out + pos,
                            cap - pos);
      if (!len[1]) return 0;
      pos += len[1];
    }
    if (pos + p.raw > cap) return 0;
    std::memcpy(out + pos, in + n - p.raw, p.raw);
    std::memcpy(out, len, kHeader);
    return pos + p.raw;
  }
  bool Decompress(const uint8_t *in, size_t cb, uint8_t *out,
                  size_t n) override {
    const Split p = SplitOf(n);
    uint64_t len[2];
    if (cb < kHeader) return false;
    std::memcpy(len, in, kHeader);
    size_t pos = kHeader;
    if (pos + len[0] + len[1] + p.raw != cb) return false;
    if (p.rows && !DecompressDims(in + pos, len[0], Shaped(p), out)) {
      return false;
    }
    pos += len[0];
    if (p.tail && !DecompressDims(in + pos, len[1], Flat(p.tail),
                                  out + p.rows * Quantum())) {
      return false;
    }
    pos += len[1];
    std::memcpy(out + n - p.raw, in + pos, p.raw);
    return true;
  }

 protected:
  /** @return worst-case bytes for one array. */
  virtual size_t BoundDims(const Dims &d) = 0;
  /** @return compressed bytes of one array, 0 on failure. */
  virtual size_t CompressDims(const uint8_t *in, const Dims &d, uint8_t *out,
                              size_t cap) = 0;
  /** @return true when one array decompressed into out. */
  virtual bool DecompressDims(const uint8_t *in, size_t cb, const Dims &d,
                              uint8_t *out) = 0;

 private:
  static constexpr size_t kHeader = 2 * sizeof(uint64_t);
  /** @brief A chunk cut into rows/planes, a 1-D remainder and raw bytes. */
  struct Split {
    size_t rows = 0, tail = 0, raw = 0;
  };
  size_t Inner() const {
    size_t p = 1;
    for (size_t d : inner_) p *= d;
    return p;
  }
  Split SplitOf(size_t n) const {
    Split s;
    const size_t elems = n / elem_;
    s.raw = n - elems * elem_;
    if (inner_.empty()) {
      s.tail = elems;
      return s;
    }
    s.rows = elems / Inner();
    s.tail = elems - s.rows * Inner();
    return s;
  }
  static Dims Flat(size_t k) {
    Dims d;
    d.n[0] = k;
    return d;
  }
  Dims Shaped(const Split &p) const {
    Dims d;
    d.rank = int(inner_.size()) + 1;
    d.n[0] = p.rows;
    for (size_t i = 0; i < inner_.size(); ++i) d.n[i + 1] = inner_[i];
    return d;
  }

  size_t elem_;
  std::vector<size_t> inner_;
};

/** @brief fpzip in lossless mode (full precision). */
class FpzipCodec : public FloatCodec {
 public:
  using FloatCodec::FloatCodec;

 protected:
  size_t BoundDims(const Dims &d) override {
    return d.Count() * elem() + d.Count() * elem() / 4 + 1024;
  }
  size_t CompressDims(const uint8_t *in, const Dims &d, uint8_t *out,
                      size_t cap) override {
    FPZ *fpz = fpzip_write_to_buffer(out, cap);
    fpz->type = elem() == 8 ? FPZIP_TYPE_DOUBLE : FPZIP_TYPE_FLOAT;
    fpz->prec = 0;
    SetDims(fpz, d);
    size_t r = 0;
    if (fpzip_write_header(fpz)) r = fpzip_write(fpz, in);
    fpzip_write_close(fpz);
    return r;
  }
  bool DecompressDims(const uint8_t *in, size_t cb, const Dims &d,
                      uint8_t *out) override {
    (void)cb;
    FPZ *fpz = fpzip_read_from_buffer(in);
    bool ok = fpzip_read_header(fpz) && size_t(fpz->nx) * fpz->ny * fpz->nz *
                                                fpz->nf ==
                                            d.Count();
    if (ok) ok = fpzip_read(fpz, out) != 0;
    fpzip_read_close(fpz);
    return ok;
  }

 private:
  /** Set fpzip's nx (fastest) .. nz from dims given slowest first. */
  static void SetDims(FPZ *fpz, const Dims &d) {
    int v[3] = {1, 1, 1};
    for (int i = 0; i < d.rank; ++i) v[i] = int(d.n[d.rank - 1 - i]);
    fpz->nx = v[0];
    fpz->ny = v[1];
    fpz->nz = v[2];
    fpz->nf = 1;
  }
};

/** @brief ZFP in reversible (lossless) mode, header in the stream. */
class ZfpCodec : public FloatCodec {
 public:
  using FloatCodec::FloatCodec;

 protected:
  size_t BoundDims(const Dims &d) override {
    zfp_field *f = Field(nullptr, d);
    zfp_stream *z = zfp_stream_open(nullptr);
    zfp_stream_set_reversible(z);
    const size_t b = zfp_stream_maximum_size(z, f) + ZFP_HEADER_MAX_BITS / 8 + 64;
    zfp_stream_close(z);
    zfp_field_free(f);
    return b;
  }
  size_t CompressDims(const uint8_t *in, const Dims &d, uint8_t *out,
                      size_t cap) override {
    zfp_field *f = Field(const_cast<uint8_t *>(in), d);
    zfp_stream *z = zfp_stream_open(nullptr);
    zfp_stream_set_reversible(z);
    bitstream *bs = stream_open(out, cap);
    zfp_stream_set_bit_stream(z, bs);
    zfp_stream_rewind(z);
    size_t r = 0;
    if (zfp_write_header(z, f, ZFP_HEADER_FULL)) r = zfp_compress(z, f);
    zfp_stream_close(z);
    stream_close(bs);
    zfp_field_free(f);
    return r;
  }
  bool DecompressDims(const uint8_t *in, size_t cb, const Dims &d,
                      uint8_t *out) override {
    zfp_field *f = zfp_field_alloc();
    zfp_stream *z = zfp_stream_open(nullptr);
    bitstream *bs = stream_open(const_cast<uint8_t *>(in), cb);
    zfp_stream_set_bit_stream(z, bs);
    zfp_stream_rewind(z);
    bool ok = zfp_read_header(z, f, ZFP_HEADER_FULL) &&
              zfp_field_size(f, nullptr) == d.Count();
    if (ok) {
      zfp_field_set_pointer(f, out);
      ok = zfp_decompress(z, f) != 0;
    }
    zfp_stream_close(z);
    stream_close(bs);
    zfp_field_free(f);
    return ok;
  }

 private:
  /** @return a zfp field over p with dims given slowest first. */
  zfp_field *Field(uint8_t *p, const Dims &d) const {
    const zfp_type t = elem() == 8 ? zfp_type_double : zfp_type_float;
    if (d.rank == 1) return zfp_field_1d(p, t, d.n[0]);
    if (d.rank == 2) return zfp_field_2d(p, t, d.n[1], d.n[0]);
    return zfp_field_3d(p, t, d.n[2], d.n[1], d.n[0]);
  }
};

/** @brief ndzip's single-threaded CPU path, float or double. */
class NdzipCpuCodec : public FloatCodec {
 public:
  /** @param elem 4 or 8; @param inner fastest dims, slowest first */
  NdzipCpuCodec(size_t elem, std::vector<size_t> inner)
      : FloatCodec(elem, inner) {
    const int nd = int(inner.size()) + 1;
    if (elem == 8) {
      d1_ = ndzip::make_cpu_offloader<double>(1, 1);
      if (nd > 1) dn_ = ndzip::make_cpu_offloader<double>(nd, 1);
    } else {
      f1_ = ndzip::make_cpu_offloader<float>(1, 1);
      if (nd > 1) fn_ = ndzip::make_cpu_offloader<float>(nd, 1);
    }
  }

 protected:
  size_t BoundDims(const Dims &d) override {
    const ndzip::extent e = Extent(d);
    return elem() == 8 ? ndzip::compressed_length_bound<double>(e) * 8
                       : ndzip::compressed_length_bound<float>(e) * 4;
  }
  size_t CompressDims(const uint8_t *in, const Dims &d, uint8_t *out,
                      size_t cap) override {
    (void)cap;
    const ndzip::extent e = Extent(d);
    if (elem() == 8) {
      auto *o = d.rank == 1 ? d1_.get() : dn_.get();
      return size_t(o->compress(reinterpret_cast<const double *>(in), e,
                                reinterpret_cast<uint64_t *>(out))) * 8;
    }
    auto *o = d.rank == 1 ? f1_.get() : fn_.get();
    return size_t(o->compress(reinterpret_cast<const float *>(in), e,
                              reinterpret_cast<uint32_t *>(out))) * 4;
  }
  bool DecompressDims(const uint8_t *in, size_t cb, const Dims &d,
                      uint8_t *out) override {
    const ndzip::extent e = Extent(d);
    if (elem() == 8) {
      auto *o = d.rank == 1 ? d1_.get() : dn_.get();
      o->decompress(reinterpret_cast<const uint64_t *>(in),
                    ndzip::index_type(cb / 8), reinterpret_cast<double *>(out),
                    e);
      return true;
    }
    auto *o = d.rank == 1 ? f1_.get() : fn_.get();
    o->decompress(reinterpret_cast<const uint32_t *>(in),
                  ndzip::index_type(cb / 4), reinterpret_cast<float *>(out), e);
    return true;
  }

 private:
  static ndzip::extent Extent(const Dims &d) {
    ndzip::extent e(d.rank);
    for (int i = 0; i < d.rank; ++i) e[i] = ndzip::index_type(d.n[i]);
    return e;
  }
  std::unique_ptr<ndzip::offloader<float>> f1_, fn_;
  std::unique_ptr<ndzip::offloader<double>> d1_, dn_;
};

// ---------------------------------------------------------------------------
// Wrappers: untimed shuffle, block-parallel threads
// ---------------------------------------------------------------------------

/** @brief How a ShuffleCodec rearranges its input (4-byte words). */
enum class Shuffle { kNone, kByte, kBit };

/** @return the shuffle named s (none, byte, bit); throws otherwise. */
inline Shuffle ParseShuffle(const std::string &s) {
  if (s == "none") return Shuffle::kNone;
  if (s == "byte") return Shuffle::kByte;
  if (s == "bit") return Shuffle::kBit;
  throw std::invalid_argument("unknown shuffle " + s);
}

/**
 * HDF5-style byte shuffle: byte k of every element goes to plane k, the
 * nbytes % elem leftover bytes follow the planes unchanged, and nothing
 * moves for 1-byte elements or a single element.
 * @param in      source, nbytes
 * @param out     destination, nbytes
 * @param nbytes  buffer bytes
 * @param elem    bytes per element (the data type's size)
 * @param inverse undo the shuffle
 */
inline void ByteShuffle(const uint8_t *in, uint8_t *out, size_t nbytes,
                        size_t elem, bool inverse) {
  const size_t m = nbytes / elem;
  if (elem <= 1 || m <= 1) {
    std::memcpy(out, in, nbytes);
    return;
  }
  for (size_t k = 0; k < elem; ++k) {
    const uint8_t *s = inverse ? in + k * m : in + k;
    uint8_t *d = inverse ? out + k : out + k * m;
    for (size_t i = 0; i < m; ++i) {
      if (inverse) d[i * elem] = s[i];
      else d[i] = s[i * elem];
    }
  }
  const size_t done = m * elem;
  std::memcpy(out + done, in + done, nbytes - done);
}

/**
 * Bit shuffle of g groups of 32 words: plane p, word w holds bit p of words
 * 32w..32w+31 (bit l = word 32w+l), as gpu_codecs.cuh's BitShuffleKernel.
 */
inline void BitShuffle(const uint32_t *in, uint32_t *out, size_t g,
                       bool inverse) {
  for (size_t w = 0; w < g; ++w) {
    for (unsigned p = 0; p < 32; ++p) {
      if (!inverse) {
        uint32_t plane = 0;
        for (unsigned l = 0; l < 32; ++l) {
          plane |= ((in[32 * w + l] >> p) & 1u) << l;
        }
        out[p * g + w] = plane;
      } else {
        const uint32_t plane = in[p * g + w];
        for (unsigned l = 0; l < 32; ++l) {
          if (p == 0) out[32 * w + l] = 0;
          out[32 * w + l] |= ((plane >> l) & 1u) << p;
        }
      }
    }
  }
}

/**
 * @brief Any codec with an untimed shuffle in front of it: an HDF5-style
 * byte shuffle at the element size, or gpu_codecs.cuh's 32-word bit shuffle.
 */
class ShuffleCodec : public Codec {
 public:
  /**
   * @param inner the codec compressing the shuffled bytes
   * @param mode  kByte (HDF5-style byte shuffle) or kBit
   * @param elem  element bytes for kByte
   */
  ShuffleCodec(std::unique_ptr<Codec> inner, Shuffle mode, size_t elem)
      : inner_(std::move(inner)), mode_(mode), elem_(elem) {}
  size_t Bound(size_t n) override { return inner_->Bound(n); }
  const uint8_t *Preprocess(const uint8_t *in, size_t n) override {
    Fit(n);
    Transform(in, shuffled_.data(), n, false);
    return shuffled_.data();
  }
  size_t Compress(const uint8_t *in, size_t n, uint8_t *out,
                  size_t cap) override {
    return inner_->Compress(in, n, out, cap);
  }
  uint8_t *DecodeTarget(uint8_t *out) override {
    (void)out;
    return decoded_.data();
  }
  bool Decompress(const uint8_t *in, size_t cb, uint8_t *out,
                  size_t n) override {
    return inner_->Decompress(in, cb, out, n);
  }
  void Postprocess(uint8_t *out, size_t n) override {
    Transform(decoded_.data(), out, n, true);
  }

 private:
  void Fit(size_t n) {
    if (shuffled_.size() < n) {
      shuffled_.assign(n, 0);
      decoded_.assign(n, 0);
    }
  }
  /** Shuffle (or undo) n bytes; what the transform skips is copied as is. */
  void Transform(const uint8_t *src, uint8_t *dst, size_t n, bool inverse) {
    if (mode_ == Shuffle::kByte) {
      ByteShuffle(src, dst, n, elem_, inverse);
      return;
    }
    const size_t words = n / 4;
    size_t covered = 0;
    if (words >= 32) {
      const size_t g = words / 32;
      BitShuffle(reinterpret_cast<const uint32_t *>(src),
                 reinterpret_cast<uint32_t *>(dst), g, inverse);
      covered = g * 32 * 4;
    }
    std::memcpy(dst + covered, src + covered, n - covered);
  }

  std::unique_ptr<Codec> inner_;
  Shuffle mode_;
  size_t elem_;
  std::vector<uint8_t> shuffled_, decoded_;
};

/**
 * @brief Cut the input into up to `threads` equal blocks, each a multiple of
 * `quantum` bytes, and run one codec instance per block on its own OpenMP
 * thread. Stream: a 4-byte block count, 8-byte compressed length per block,
 * then the blocks back to back. Scratch is sized before timing.
 */
class BlockParallelCodec : public Codec {
 public:
  /**
   * @param make    builds one independent codec instance
   * @param threads blocks / threads
   * @param quantum block sizes are multiples of this many bytes
   */
  BlockParallelCodec(const std::function<std::unique_ptr<Codec>()> &make,
                     int threads, size_t quantum)
      : threads_(threads), quantum_(std::max<size_t>(quantum, 1)) {
    for (int i = 0; i < threads; ++i) inst_.push_back(make());
    scratch_.resize(size_t(threads));
  }
  size_t Bound(size_t n) override {
    const auto b = Blocks(n);
    size_t total = Header(b.size());
    for (size_t i = 0; i < b.size(); ++i) {
      const size_t need = inst_[i]->Bound(b[i].second);
      if (scratch_[i].size() < need) scratch_[i].assign(need, 0);
      total += need;
    }
    return total;
  }
  size_t Compress(const uint8_t *in, size_t n, uint8_t *out,
                  size_t cap) override {
    const auto b = Blocks(n);
    const size_t nb = b.size();
    std::vector<uint64_t> len(nb, 0);
#pragma omp parallel for num_threads(threads_) schedule(static, 1)
    for (size_t i = 0; i < nb; ++i) {
      len[i] = inst_[i]->Compress(in + b[i].first, b[i].second,
                                  scratch_[i].data(), scratch_[i].size());
    }
    std::vector<size_t> off(nb + 1, Header(nb));
    for (size_t i = 0; i < nb; ++i) {
      if (!len[i]) return 0;
      off[i + 1] = off[i] + len[i];
    }
    if (off[nb] > cap) return 0;
#pragma omp parallel for num_threads(threads_) schedule(static, 1)
    for (size_t i = 0; i < nb; ++i) {
      std::memcpy(out + off[i], scratch_[i].data(), len[i]);
    }
    const uint32_t count = uint32_t(nb);
    std::memcpy(out, &count, 4);
    std::memcpy(out + 4, len.data(), nb * 8);
    return off[nb];
  }
  bool Decompress(const uint8_t *in, size_t cb, uint8_t *out,
                  size_t n) override {
    const auto b = Blocks(n);
    const size_t nb = b.size();
    uint32_t count = 0;
    if (cb < 4) return false;
    std::memcpy(&count, in, 4);
    if (count != nb || cb < Header(nb)) return false;
    std::vector<uint64_t> len(nb);
    std::memcpy(len.data(), in + 4, nb * 8);
    std::vector<size_t> off(nb + 1, Header(nb));
    for (size_t i = 0; i < nb; ++i) off[i + 1] = off[i] + len[i];
    if (off[nb] != cb) return false;
    int bad = 0;
#pragma omp parallel for num_threads(threads_) schedule(static, 1) \
    reduction(| : bad)
    for (size_t i = 0; i < nb; ++i) {
      bad |= !inst_[i]->Decompress(in + off[i], len[i], out + b[i].first,
                                   b[i].second);
    }
    return !bad;
  }

 private:
  static size_t Header(size_t nb) { return 4 + 8 * nb; }
  /** @return (offset, bytes) of each block; the last takes the remainder. */
  std::vector<std::pair<size_t, size_t>> Blocks(size_t n) const {
    const size_t units = n / quantum_;
    const size_t nb = std::max<size_t>(
        1, std::min<size_t>(size_t(threads_), units));
    std::vector<std::pair<size_t, size_t>> v;
    size_t off = 0;
    for (size_t i = 0; i < nb; ++i) {
      const size_t u = units / nb + (i < units % nb ? 1 : 0);
      const size_t bytes = i + 1 == nb ? n - off : u * quantum_;
      v.emplace_back(off, bytes);
      off += bytes;
    }
    return v;
  }

  int threads_;
  size_t quantum_;
  std::vector<std::unique_ptr<Codec>> inst_;
  std::vector<std::vector<uint8_t>> scratch_;
};

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/** @return the shape "AxB" as sizes, slowest first; throws on bad input. */
inline std::vector<size_t> ParseShape(const std::string &s) {
  std::vector<size_t> v;
  std::stringstream ss(s);
  for (std::string tok; std::getline(ss, tok, 'x');) {
    size_t used = 0;
    const unsigned long long x = std::stoull(tok, &used);
    if (used != tok.size() || x == 0) {
      throw std::invalid_argument("bad shape " + s);
    }
    v.push_back(size_t(x));
  }
  if (v.empty() || v.size() > 2) throw std::invalid_argument("bad shape " + s);
  return v;
}

/** @return element bytes for type=float|double. */
inline size_t ParseType(const std::string &t) {
  if (t == "float") return 4;
  if (t == "double") return 8;
  throw std::invalid_argument("unknown type " + t);
}

/**
 * Build one instance of the codec a spec names, without shuffle or threads.
 * @param r       the spec's settings (shuffle and threads already read)
 * @param base    codec base name
 * @param quantum set to the bytes a block must be a multiple of
 * @return the codec, or nullptr for an unknown base
 */
inline std::unique_ptr<Codec> MakeInner(SettingReader &r,
                                        const std::string &base,
                                        size_t *quantum) {
  *quantum = 4096;
  if (base == "zstd") return std::make_unique<ZstdCodec>(int(r.Int("level", 3)));
  if (base == "lz4") return std::make_unique<Lz4Codec>(int(r.Int("level", 0)));
  if (base == "zlib") return std::make_unique<ZlibCodec>(int(r.Int("level", 6)));
  if (base == "bzip2") {
    return std::make_unique<Bzip2Codec>(int(r.Int("level", 9)));
  }
  if (base == "xz") return std::make_unique<XzCodec>(uint32_t(r.Int("preset", 6)));
  if (base == "brotli") {
    return std::make_unique<BrotliCodec>(int(r.Int("quality", 6)));
  }
  if (base == "store") return std::make_unique<StoreCodec>();
  if (base == "snappy") return std::make_unique<SnappyCodec>();
  if (base == "lzo") return std::make_unique<LzoCodec>();
  if (base == "blosc2") {
    static const std::map<std::string, int> kCodec = {
        {"blosclz", BLOSC_BLOSCLZ}, {"lz4", BLOSC_LZ4}, {"zstd", BLOSC_ZSTD}};
    const auto it = kCodec.find(r.Str("codec", "zstd"));
    if (it == kCodec.end()) throw std::invalid_argument("unknown blosc2 codec");
    return std::make_unique<Blosc2Codec>(
        it->second, r.Str("filter", "shuffle"), int(r.Int("clevel", 5)),
        int(ParseType(r.Str("type", "float"))));
  }
  if (base == "fpzip" || base == "zfp" || base == "ndzip") {
    const size_t elem = ParseType(r.Str("type", "float"));
    const std::string s = r.Str("shape", "");
    const std::vector<size_t> inner = s.empty() ? std::vector<size_t>{}
                                                : ParseShape(s);
    std::unique_ptr<FloatCodec> c;
    if (base == "fpzip") c = std::make_unique<FpzipCodec>(elem, inner);
    else if (base == "zfp") c = std::make_unique<ZfpCodec>(elem, inner);
    else c = std::make_unique<NdzipCpuCodec>(elem, inner);
    *quantum = c->Quantum();
    // ndzip stores whatever does not fill a hypercube (4096 values in 1-D,
    // 64 rows in 2-D, 16 planes in 3-D) raw: blocks must hold whole ones.
    if (base == "ndzip") *quantum *= inner.empty() ? 4096 : inner.size() == 1 ? 64 : 16;
    // zfp pads every 4^d block, so slabs under 4 rows/planes deep inflate.
    if (base == "zfp") *quantum *= 4;
    return c;
  }
  return nullptr;
}

/**
 * Build the codec a spec describes, with its shuffle and threading.
 * @param spec base name and settings
 * @return the codec, or nullptr for an unknown base; throws
 *         std::invalid_argument on a bad setting
 */
inline std::unique_ptr<Codec> MakeCodec(const CodecSpec &spec) {
  SettingReader r(spec);
  const Shuffle sh = ParseShuffle(r.Str("shuffle", "none"));
  const size_t elem =
      sh == Shuffle::kByte ? ParseType(r.Str("type", "float")) : 4;
  const int threads = int(r.Int("threads", 1));
  if (threads < 1) throw std::invalid_argument("threads must be >= 1");
  size_t quantum = 0;
  std::unique_ptr<Codec> c = MakeInner(r, spec.base, &quantum);
  if (!c) return nullptr;
  r.CheckAllUsed();
  if (threads > 1) {
    auto make = [spec, quantum]() {
      SettingReader rr(spec);
      rr.Str("shuffle", "");
      rr.Str("threads", "");
      size_t q = 0;
      return MakeInner(rr, spec.base, &q);
    };
    // Blocks of whole 4-byte words keep a later shuffle's words intact.
    const size_t q = std::max<size_t>(quantum, 4) / 4 * 4;
    c = std::make_unique<BlockParallelCodec>(make, threads, q);
  }
  if (sh != Shuffle::kNone) {
    c = std::make_unique<ShuffleCodec>(std::move(c), sh, elem);
  }
  return c;
}

}  // namespace cpu_codecs

#endif  // PAPER_BENCHMARK_CODEC_SWEEP_CPU_CODECS_H_
