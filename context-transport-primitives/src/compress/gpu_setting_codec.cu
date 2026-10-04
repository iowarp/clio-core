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
 * @file gpu_setting_codec.cu
 * @brief The 45 lossless GPU settings NeuroPress v2 chooses between.
 *
 * The codec layer below is a port of the benchmark's
 * paper-benchmark/codec-sweep/gpu_codecs.cuh, the code every v2 training
 * label was measured with, kept as close to it as possible so a setting here
 * produces the same bytes and runs the same kernels as it did there. The
 * differences are only what a library needs: errors become a `false` return
 * instead of exiting, host buffers are staged to the device, codecs are
 * cached per thread, the FPcompress calls are serialised, and ndzip runs its
 * 1-D path only (no v2 setting has a shape).
 */

#include "clio_ctp/compress/gpu_setting_codec.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if CTP_ENABLE_CUDA
#include <cuda_runtime.h>
#if CTP_ENABLE_NVCOMP
#include <nvcomp/nvcompManagerFactory.hpp>
#endif
#if CTP_ENABLE_NDZIP
// ndzip/ndzip.hh uses assert() and unqualified size()/data() but relies on
// its includer for them.
#include <cassert>
#include <iterator>
#include <ndzip/cuda.hh>
#endif
#if CTP_ENABLE_GPULZ
#include <gpulz_api.h>
#endif
#if CTP_ENABLE_FPCOMPRESS
#include <fpc_api.h>
#endif
#endif  // CTP_ENABLE_CUDA

namespace ctp {

namespace {

/** The frozen table: index -> canonical spec (keys sorted). Append-only. */
constexpr const char *kSpecs[kGpuSettingCount] = {
    "ans",
    "ans shuffle=bit",
    "ans shuffle=byte",
    "ans type=float16",
    "bitcomp algo=1 shuffle=bit type=char",
    "bitcomp algo=1 shuffle=byte type=char",
    "bitcomp algo=1 type=char",
    "bitcomp shuffle=bit type=int",
    "bitcomp shuffle=byte type=int",
    "bitcomp type=int",
    "cascaded bp=1 delta=0 rle=0 shuffle=bit type=int",
    "cascaded bp=1 delta=0 rle=0 shuffle=byte type=int",
    "cascaded bp=1 delta=0 rle=0 type=int",
    "cascaded bp=1 delta=0 rle=1 shuffle=bit type=int",
    "cascaded bp=1 delta=0 rle=1 shuffle=byte type=int",
    "cascaded bp=1 delta=0 rle=1 type=int",
    "deflate level=1",
    "deflate level=1 shuffle=bit",
    "deflate level=1 shuffle=byte",
    "gdeflate level=1",
    "gdeflate level=1 shuffle=bit",
    "gdeflate level=1 shuffle=byte",
    "gpulz",
    "gpulz shuffle=bit",
    "gpulz shuffle=byte",
    "lz4",
    "lz4 bitshuffle=msb type=int",
    "lz4 shuffle=bit",
    "lz4 shuffle=byte",
    "ndzip",
    "ndzip shuffle=bit",
    "ndzip shuffle=byte",
    "snappy",
    "snappy shuffle=bit",
    "snappy shuffle=byte",
    "spratio",
    "spratio shuffle=bit",
    "spratio shuffle=byte",
    "spspeed",
    "spspeed shuffle=bit",
    "spspeed shuffle=byte",
    "store",
    "zstd",
    "zstd shuffle=bit",
    "zstd shuffle=byte",
};

/** @brief A codec base name and its key=value settings. */
struct CodecSpec {
  std::string base;                           ///< lz4, ..., store
  std::map<std::string, std::string> params;  ///< setting -> value
};

/**
 * Parse "base k=v k=v ...".
 * @param line whitespace-separated base name and settings
 * @param spec receives the parsed spec
 * @return false on a malformed token or an empty line
 */
bool ParseSpec(const std::string &line, CodecSpec *spec) {
  std::istringstream ss(line);
  ss >> spec->base;
  for (std::string tok; ss >> tok;) {
    const size_t eq = tok.find('=');
    if (eq == 0 || eq == std::string::npos || eq + 1 == tok.size()) return false;
    spec->params[tok.substr(0, eq)] = tok.substr(eq + 1);
  }
  return !spec->base.empty();
}

/**
 * @param spec any token order
 * @return the canonical form (settings sorted by key), "" when malformed
 */
std::string Canonical(const std::string &spec) {
  CodecSpec s;
  if (!ParseSpec(spec, &s)) return "";
  std::string out = s.base;
  for (const auto &kv : s.params) out += " " + kv.first + "=" + kv.second;
  return out;
}

/** @return the codec family a setting needs: its base name. */
std::string BaseOf(int index) {
  CodecSpec s;
  ParseSpec(kSpecs[index], &s);
  return s.base;
}

#if CTP_ENABLE_CUDA

/** Codec-layer failure; caught at the GpuSettingCodec boundary. */
struct CodecError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

/** Throw CodecError on a CUDA error (the benchmark exited instead). */
#define GSC_CHECK(x)                                                     \
  do {                                                                   \
    cudaError_t e_ = (x);                                                \
    if (e_ != cudaSuccess) {                                             \
      throw CodecError(std::string(#x) + ": " + cudaGetErrorString(e_)); \
    }                                                                    \
  } while (0)

/** @brief One GPU codec behind a device-to-device interface. */
class Codec {
 public:
  /** Creates the codec's stream. */
  Codec() { GSC_CHECK(cudaStreamCreate(&stream_)); }
  virtual ~Codec() {
    if (owns_stream_ && stream_ != nullptr) cudaStreamDestroy(stream_);
  }
  /** @return worst-case compressed bytes for n input bytes. */
  virtual size_t Bound(size_t n) = 0;
  /** @return false when the codec cannot take an n-byte input at all. */
  virtual bool Accepts(size_t n) const {
    (void)n;
    return true;
  }
  /** Untimed setup before Compress (configuration, per-size objects). */
  virtual void PrepareCompress(size_t n) { (void)n; }
  /** Untimed setup before decompressing n bytes (scratch buffers). */
  virtual void PrepareDecode(size_t n) { (void)n; }
  /**
   * Transform before Compress (a shuffle), timed with the compression.
   * @return the buffer to hand to Compress: in itself, or codec scratch
   */
  virtual const uint8_t *Preprocess(const uint8_t *in, size_t n) {
    (void)n;
    return in;
  }
  /** @return where Decompress must write: out itself, or codec scratch. */
  virtual uint8_t *DecodeTarget(uint8_t *out) { return out; }
  /** Transform after Decompress into out, timed with the decompression. */
  virtual void Postprocess(uint8_t *out, size_t n) {
    (void)out;
    (void)n;
  }
  /** Enqueue compression of n device bytes into out (cap bytes). */
  virtual void Compress(const uint8_t *in, size_t n, uint8_t *out,
                        size_t cap) = 0;
  /** @return compressed bytes once the stream is idle, 0 on failure. */
  virtual size_t CompressedBytes(const uint8_t *out) = 0;
  /** Setup before Decompress (e.g. parsing the stream header). */
  virtual void PrepareDecompress(const uint8_t *in, size_t comp_bytes) {
    (void)in;
    (void)comp_bytes;
  }
  /** Enqueue decompression of comp_bytes into n bytes at out. */
  virtual void Decompress(const uint8_t *in, size_t comp_bytes, uint8_t *out,
                          size_t n) = 0;
  /** @return whether the last decompression reported success. */
  virtual bool DecompressOk() { return true; }
  /** @return the stream every call of this codec runs on. */
  cudaStream_t stream() const { return stream_; }
  /**
   * @param n          input bytes
   * @param decompress which direction
   * @return true the first time this codec object sees n bytes in that
   *         direction (and records it), so the caller can run an untimed
   *         warm-up first, as the sweep did: first launches pay for lazy
   *         kernel loading and per-size allocation
   */
  bool FirstUse(size_t n, bool decompress) {
    return (decompress ? warm_d_ : warm_c_).insert(n).second;
  }

 protected:
  cudaStream_t stream_ = nullptr;
  bool owns_stream_ = true;  ///< false for a wrapper borrowing a stream
  std::set<size_t> warm_c_, warm_d_;  ///< sizes already warmed up
};

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
    if (used != v.size()) throw CodecError(key + "=" + v + " is not an integer");
    return x;
  }
  /** Throw when the spec carries a setting no reader asked for. */
  void CheckAllUsed() const {
    for (const auto &kv : spec_.params) {
      if (!used_.count(kv.first)) {
        throw CodecError("unknown setting " + kv.first + " for " + spec_.base);
      }
    }
  }

 private:
  const CodecSpec &spec_;
  std::set<std::string> used_;
};

#if CTP_ENABLE_NVCOMP
/** @return the nvcomp data type named t; throws on an unknown name. */
nvcompType_t ParseType(const std::string &t) {
  static const std::map<std::string, nvcompType_t> m = {
      {"char", NVCOMP_TYPE_CHAR},         {"uchar", NVCOMP_TYPE_UCHAR},
      {"short", NVCOMP_TYPE_SHORT},       {"ushort", NVCOMP_TYPE_USHORT},
      {"int", NVCOMP_TYPE_INT},           {"uint", NVCOMP_TYPE_UINT},
      {"longlong", NVCOMP_TYPE_LONGLONG}, {"ulonglong", NVCOMP_TYPE_ULONGLONG},
      {"float16", NVCOMP_TYPE_FLOAT16}};
  auto it = m.find(t);
  if (it == m.end()) throw CodecError("unknown type " + t);
  return it->second;
}

/** @return the LZ4 bitshuffle mode named b (none, msb, lsb). */
nvcompBitshuffleMode_t ParseBitshuffle(const std::string &b) {
  if (b == "none") return NVCOMP_BITSHUFFLE_NONE;
  if (b == "msb") return NVCOMP_BITSHUFFLE_MSB_FIRST;
  if (b == "lsb") return NVCOMP_BITSHUFFLE_LSB_FIRST;
  throw CodecError("unknown bitshuffle " + b);
}

using ManagerPtr = std::shared_ptr<nvcomp::nvcompManagerBase>;

/**
 * Build the nvcomp manager for the byte-oriented algorithms of a spec.
 * @return the manager, or nullptr when the base is not one of them
 */
ManagerPtr MakeByteManager(const std::string &b, SettingReader *r, size_t chunk,
                           cudaStream_t s) {
  if (b == "lz4") {
    auto c = nvcompBatchedLZ4CompressDefaultOpts;
    auto d = nvcompBatchedLZ4DecompressDefaultOpts;
    // The decoder needs the same type and mode to undo the bitshuffle.
    c.data_type = d.data_type = ParseType(r->Str("type", "char"));
    c.bitshuffle_mode = d.bitshuffle_mode =
        ParseBitshuffle(r->Str("bitshuffle", "none"));
    return std::make_shared<nvcomp::LZ4Manager>(chunk, c, d, s);
  }
  if (b == "snappy") {
    return std::make_shared<nvcomp::SnappyManager>(
        chunk, nvcompBatchedSnappyCompressDefaultOpts,
        nvcompBatchedSnappyDecompressDefaultOpts, s);
  }
  if (b == "zstd") {
    return std::make_shared<nvcomp::ZstdManager>(
        chunk, nvcompBatchedZstdCompressDefaultOpts,
        nvcompBatchedZstdDecompressDefaultOpts, s);
  }
  if (b == "gdeflate") {
    auto c = nvcompBatchedGdeflateCompressDefaultOpts;
    c.algorithm = static_cast<int>(r->Int("level", 1));
    return std::make_shared<nvcomp::GdeflateManager>(
        chunk, c, nvcompBatchedGdeflateDecompressDefaultOpts, s);
  }
  if (b == "deflate") {
    auto c = nvcompBatchedDeflateCompressDefaultOpts;
    c.algorithm = static_cast<int>(r->Int("level", 1));
    return std::make_shared<nvcomp::DeflateManager>(
        chunk, c, nvcompBatchedDeflateDecompressDefaultOpts, s);
  }
  return nullptr;
}

/**
 * Build the nvcomp manager for the typed algorithms (ANS, Cascaded, Bitcomp).
 * @return the manager, or nullptr when the base is not one of them
 */
ManagerPtr MakeTypedManager(const std::string &b, SettingReader *r,
                            size_t chunk, cudaStream_t s) {
  if (b == "ans") {
    auto c = nvcompBatchedANSCompressDefaultOpts;
    c.data_type = ParseType(r->Str("type", "char"));
    c.max_sub_chunk_count = static_cast<uint8_t>(r->Int("subchunks", 0));
    return std::make_shared<nvcomp::ANSManager>(
        chunk, c, nvcompBatchedANSDecompressDefaultOpts, s);
  }
  if (b == "cascaded") {
    auto c = nvcompBatchedCascadedCompressDefaultOpts;
    c.type = ParseType(r->Str("type", "int"));
    c.num_RLEs = static_cast<int>(r->Int("rle", 2));
    c.num_deltas = static_cast<int>(r->Int("delta", 1));
    c.use_bp = static_cast<int>(r->Int("bp", 1));
    return std::make_shared<nvcomp::CascadedManager>(
        chunk, c, nvcompBatchedCascadedDecompressDefaultOpts, s);
  }
  if (b == "bitcomp") {
    auto c = nvcompBatchedBitcompCompressDefaultOpts;
    c.algorithm = static_cast<int>(r->Int("algo", 0));
    c.data_type = ParseType(r->Str("type", "char"));
    return std::make_shared<nvcomp::BitcompManager>(
        chunk, c, nvcompBatchedBitcompDecompressDefaultOpts, s);
  }
  return nullptr;
}

/** @brief One nvcomp algorithm through its high-level manager. */
class NvcompCodec : public Codec {
 public:
  /** @param spec nvcomp base name and settings; throws when it is not one */
  explicit NvcompCodec(const CodecSpec &spec) {
    SettingReader r(spec);
    const size_t chunk = static_cast<size_t>(r.Int("chunk", 65536));
    mgr_ = MakeByteManager(spec.base, &r, chunk, stream_);
    if (!mgr_) mgr_ = MakeTypedManager(spec.base, &r, chunk, stream_);
    if (!mgr_) throw CodecError("not an nvcomp codec: " + spec.base);
    r.CheckAllUsed();
  }
  size_t Bound(size_t n) override {
    return mgr_->configure_compression(n).max_compressed_buffer_size;
  }
  void PrepareCompress(size_t n) override {
    ccfg_ = std::make_unique<nvcomp::CompressionConfig>(
        mgr_->configure_compression(n));
  }
  void Compress(const uint8_t *in, size_t, uint8_t *out, size_t) override {
    mgr_->compress(in, out, *ccfg_);
  }
  size_t CompressedBytes(const uint8_t *out) override {
    if (*ccfg_->get_status() != nvcompSuccess) return 0;
    return mgr_->get_compressed_output_size(out);
  }
  void PrepareDecompress(const uint8_t *in, size_t comp_bytes) override {
    dcfg_ = std::make_unique<nvcomp::DecompressionConfig>(
        mgr_->configure_decompression(in, &comp_bytes));
  }
  void Decompress(const uint8_t *in, size_t, uint8_t *out,
                  size_t n) override {
    want_ = n;
    mgr_->decompress(out, in, *dcfg_);
  }
  bool DecompressOk() override {
    return *dcfg_->get_status() == nvcompSuccess &&
           dcfg_->decomp_data_size == want_;
  }

 private:
  ManagerPtr mgr_;
  std::unique_ptr<nvcomp::CompressionConfig> ccfg_;
  std::unique_ptr<nvcomp::DecompressionConfig> dcfg_;
  size_t want_ = 0;
};
#endif  // CTP_ENABLE_NVCOMP

#if CTP_ENABLE_NDZIP
/** @brief ndzip's CUDA backend on float32 data, 1-D (the unshaped sweep path). */
class NdzipCodec : public Codec {
 public:
  NdzipCodec() {
    GSC_CHECK(cudaMalloc(&d_len_, sizeof(ndzip::index_type)));
    decomp_ = ndzip::make_cuda_decompressor<float>(1, stream_);
  }
  ~NdzipCodec() override {
    if (d_len_ != nullptr) cudaFree(d_len_);
  }
  bool Accepts(size_t n) const override { return n % sizeof(float) == 0; }
  size_t Bound(size_t n) override {
    return ndzip::compressed_length_bound<float>(Flat(n / 4)) * kWord;
  }
  void PrepareCompress(size_t n) override {
    auto &c = comp_[n];
    if (!c) {
      c = ndzip::make_cuda_compressor<float>(
          ndzip::compressor_requirements{Flat(n / 4)}, stream_);
    }
    cur_ = c.get();
  }
  void Compress(const uint8_t *in, size_t n, uint8_t *out, size_t) override {
    cur_->compress(reinterpret_cast<const float *>(in), Flat(n / 4),
                   reinterpret_cast<Word *>(out), d_len_);
  }
  size_t CompressedBytes(const uint8_t *) override {
    ndzip::index_type words = 0;
    GSC_CHECK(cudaMemcpy(&words, d_len_, sizeof(words), cudaMemcpyDeviceToHost));
    return static_cast<size_t>(words) * kWord;
  }
  void Decompress(const uint8_t *in, size_t, uint8_t *out, size_t n) override {
    decomp_->decompress(reinterpret_cast<const Word *>(in),
                        reinterpret_cast<float *>(out), Flat(n / 4));
  }

 private:
  using Word = ndzip::compressed_type<float>;
  static constexpr size_t kWord = sizeof(Word);
  /** @return the 1-D extent of k floats. */
  static ndzip::extent Flat(size_t k) {
    ndzip::extent e(1);
    e[0] = static_cast<ndzip::index_type>(k);
    return e;
  }
  std::map<size_t, std::unique_ptr<ndzip::cuda_compressor<float>>> comp_;
  ndzip::cuda_compressor<float> *cur_ = nullptr;
  std::unique_ptr<ndzip::cuda_decompressor<float>> decomp_;
  ndzip::index_type *d_len_ = nullptr;
};
#endif  // CTP_ENABLE_NDZIP

/** @brief Store uncompressed: one device-to-device copy each way. */
class StoreCodec : public Codec {
 public:
  size_t Bound(size_t n) override { return n; }
  void Compress(const uint8_t *in, size_t n, uint8_t *out,
                size_t cap) override {
    bytes_ = n <= cap ? n : 0;
    if (bytes_) {
      GSC_CHECK(cudaMemcpyAsync(out, in, n, cudaMemcpyDeviceToDevice, stream_));
    }
  }
  size_t CompressedBytes(const uint8_t *) override { return bytes_; }
  void Decompress(const uint8_t *in, size_t comp_bytes, uint8_t *out,
                  size_t n) override {
    ok_ = comp_bytes == n;
    if (ok_) {
      GSC_CHECK(cudaMemcpyAsync(out, in, n, cudaMemcpyDeviceToDevice, stream_));
    }
  }
  bool DecompressOk() override { return ok_; }

 private:
  size_t bytes_ = 0;
  bool ok_ = false;
};

#if CTP_ENABLE_GPULZ
/** @brief GPULZ (ICS'23) through the gpulz_api.h shim. */
class GpulzCodec : public Codec {
 public:
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
#endif  // CTP_ENABLE_GPULZ

#if CTP_ENABLE_FPCOMPRESS
/** fpc_api.h keeps one set of scratch buffers per process: serialise it. */
std::mutex &FpcMutex() {
  static std::mutex m;
  return m;
}

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
  /** @param api the algorithm's entry points */
  explicit FpcCodec(Api api) : api_(api) {}
  size_t Bound(size_t n) override { return api_.bound(n); }
  void Compress(const uint8_t *in, size_t n, uint8_t *out,
                size_t cap) override {
    std::lock_guard<std::mutex> lock(FpcMutex());
    size_t b = 0;
    bytes_ = api_.compress(in, n, out, cap, &b, stream_) == 0 ? b : 0;
    GSC_CHECK(cudaStreamSynchronize(stream_));  // scratch free before unlock
  }
  size_t CompressedBytes(const uint8_t *) override { return bytes_; }
  void PrepareDecompress(const uint8_t *in, size_t comp_bytes) override {
    std::lock_guard<std::mutex> lock(FpcMutex());
    declared_ = api_.declared(in, comp_bytes, stream_);
  }
  void Decompress(const uint8_t *in, size_t comp_bytes, uint8_t *out,
                  size_t n) override {
    // The decoder writes as many bytes as the stream declares; never more
    // than the caller's buffer.
    std::lock_guard<std::mutex> lock(FpcMutex());
    drc_ = declared_ == static_cast<long long>(n)
               ? api_.decompress(in, comp_bytes, out, n, stream_)
               : -3;
    GSC_CHECK(cudaStreamSynchronize(stream_));
  }
  bool DecompressOk() override { return drc_ == 0; }

 private:
  Api api_;
  size_t bytes_ = 0;
  long long declared_ = -1;
  int drc_ = 0;
};
#endif  // CTP_ENABLE_FPCOMPRESS

/**
 * HDF5-style byte shuffle of m elements of e bytes each: out[k*m + i] = byte
 * k of element i (inverse: the reverse map).
 */
__global__ void ByteShuffleKernel(const uint8_t *in, uint8_t *out, size_t m,
                                  unsigned e, bool inverse) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  for (; i < m; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    for (unsigned k = 0; k < e; ++k) {
      if (!inverse) {
        out[k * m + i] = in[i * e + k];
      } else {
        out[i * e + k] = in[k * m + i];
      }
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
enum class Shuffle { kByte, kBit };

/**
 * @brief Any codec with a byte or bit shuffle in front of it, on the inner
 * codec's stream. The forward transform runs in Preprocess and the inverse
 * in Postprocess, both inside the timed region.
 */
class ShuffleCodec : public Codec {
 public:
  /**
   * @param inner the codec that compresses the shuffled words
   * @param mode  kByte or kBit
   * @param elem  element bytes for kByte (4 for float); bit is 32-bit words
   */
  ShuffleCodec(std::unique_ptr<Codec> inner, Shuffle mode, unsigned elem)
      : inner_(std::move(inner)), mode_(mode), elem_(elem) {
    cudaStreamDestroy(stream_);  // run on the inner codec's stream instead
    stream_ = inner_->stream();
    owns_stream_ = false;
  }
  ~ShuffleCodec() override {
    if (shuffled_) cudaFree(shuffled_);
    if (decoded_) cudaFree(decoded_);
  }
  size_t Bound(size_t n) override { return inner_->Bound(n); }
  bool Accepts(size_t n) const override { return inner_->Accepts(n); }
  void PrepareCompress(size_t n) override {
    Fit(n);
    inner_->PrepareCompress(n);
  }
  void PrepareDecode(size_t n) override {
    Fit(n);
    inner_->PrepareDecode(n);
  }
  const uint8_t *Preprocess(const uint8_t *in, size_t n) override {
    Fit(n);
    Transform(in, shuffled_, n, false);
    return shuffled_;
  }
  void Compress(const uint8_t *in, size_t n, uint8_t *out,
                size_t cap) override {
    inner_->Compress(in, n, out, cap);  // in: the Preprocess output
  }
  size_t CompressedBytes(const uint8_t *out) override {
    return inner_->CompressedBytes(out);
  }
  void PrepareDecompress(const uint8_t *in, size_t comp_bytes) override {
    inner_->PrepareDecompress(in, comp_bytes);
  }
  uint8_t *DecodeTarget(uint8_t *out) override {
    (void)out;
    return decoded_;
  }
  void Decompress(const uint8_t *in, size_t comp_bytes, uint8_t *out,
                  size_t n) override {
    inner_->Decompress(in, comp_bytes, out, n);  // out: DecodeTarget()
  }
  void Postprocess(uint8_t *out, size_t n) override {
    Transform(decoded_, out, n, true);
  }
  bool DecompressOk() override { return inner_->DecompressOk(); }

 private:
  /** Grow both scratch buffers to n bytes (untimed). */
  void Fit(size_t n) {
    if (n <= cap_) return;
    if (shuffled_) GSC_CHECK(cudaFree(shuffled_));
    if (decoded_) GSC_CHECK(cudaFree(decoded_));
    shuffled_ = decoded_ = nullptr;
    cap_ = 0;
    GSC_CHECK(cudaMalloc(&shuffled_, n));
    GSC_CHECK(cudaMalloc(&decoded_, n));
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
    if (mode_ == Shuffle::kByte && n / elem_ > 1) {
      const size_t m = n / elem_;
      ByteShuffleKernel<<<1024, 256, 0, stream_>>>(src, dst, m, elem_, inverse);
      covered = m * elem_;
    } else if (mode_ == Shuffle::kBit && words >= 32) {
      const size_t g = words / 32;
      BitShuffleKernel<<<1024, 256, 0, stream_>>>(
          reinterpret_cast<const uint32_t *>(src),
          reinterpret_cast<uint32_t *>(dst), g, inverse);
      covered = g * 32 * 4;
    }
    GSC_CHECK(cudaGetLastError());
    if (covered < n) {
      GSC_CHECK(cudaMemcpyAsync(dst + covered, src + covered, n - covered,
                                cudaMemcpyDeviceToDevice, stream_));
    }
  }

  std::unique_ptr<Codec> inner_;
  Shuffle mode_;
  unsigned elem_;
  uint8_t *shuffled_ = nullptr;  ///< shuffled input, handed to Compress
  uint8_t *decoded_ = nullptr;   ///< Decompress output, still shuffled
  size_t cap_ = 0;
};

/**
 * Build the unshuffled codec a spec names.
 * @return the codec, or nullptr when this build lacks its library
 */
std::unique_ptr<Codec> MakeBaseCodec(const CodecSpec &spec) {
  const std::string &b = spec.base;
  if (b == "store" || b == "gpulz" || b == "spspeed" || b == "spratio" ||
      b == "ndzip") {
    SettingReader(spec).CheckAllUsed();  // these take no settings
  }
  if (b == "store") return std::make_unique<StoreCodec>();
#if CTP_ENABLE_NDZIP
  if (b == "ndzip") return std::make_unique<NdzipCodec>();
#endif
#if CTP_ENABLE_GPULZ
  if (b == "gpulz") return std::make_unique<GpulzCodec>();
#endif
#if CTP_ENABLE_FPCOMPRESS
  if (b == "spspeed") {
    return std::make_unique<FpcCodec>(
        FpcCodec::Api{spspeed_compress_bound, spspeed_compress,
                      spspeed_decompressed_size, spspeed_decompress});
  }
  if (b == "spratio") {
    return std::make_unique<FpcCodec>(
        FpcCodec::Api{spratio_compress_bound, spratio_compress,
                      spratio_decompressed_size, spratio_decompress});
  }
#endif
#if CTP_ENABLE_NVCOMP
  static const std::set<std::string> kNvcomp = {
      "lz4", "snappy", "zstd", "gdeflate", "deflate", "ans", "cascaded",
      "bitcomp"};
  if (kNvcomp.count(b)) return std::make_unique<NvcompCodec>(spec);
#endif
  return nullptr;
}

/**
 * Build the codec of one setting: its base codec, wrapped in a shuffle when
 * the spec asks for one (element 4 for the byte shuffle, as in the sweep).
 * @return the codec, or nullptr when this build lacks its library
 */
std::unique_ptr<Codec> MakeCodec(int index) {
  CodecSpec spec;
  ParseSpec(kSpecs[index], &spec);
  auto sh = spec.params.find("shuffle");
  if (sh == spec.params.end()) return MakeBaseCodec(spec);
  const Shuffle mode = sh->second == "bit" ? Shuffle::kBit : Shuffle::kByte;
  spec.params.erase(sh);
  auto inner = MakeBaseCodec(spec);
  if (!inner) return nullptr;
  return std::make_unique<ShuffleCodec>(std::move(inner), mode, 4u);
}

/**
 * @return this thread's codec for a setting, built on first use. The cache is
 * deliberately never destroyed: its nvcomp managers, streams and device
 * buffers would otherwise be freed at thread or process exit, after the CUDA
 * runtime has already been torn down, which crashes the process on exit.
 */
Codec *CachedCodec(int index) {
  static thread_local auto *cache = new std::map<int, std::unique_ptr<Codec>>();
  auto &c = (*cache)[index];
  if (!c) c = MakeCodec(index);
  if (!c) throw CodecError(std::string("setting unavailable: ") + kSpecs[index]);
  return c.get();
}

/** @return true for device or managed memory. */
bool OnDevice(const void *p) {
  cudaPointerAttributes attr;
  if (cudaPointerGetAttributes(&attr, p) != cudaSuccess) {
    cudaGetLastError();  // reset the sticky error from the failed query
    return false;
  }
  return attr.type == cudaMemoryTypeDevice ||
         attr.type == cudaMemoryTypeManaged;
}

/** @brief A device buffer freed on scope exit. */
struct DeviceBuffer {
  uint8_t *ptr = nullptr;
  ~DeviceBuffer() {
    if (ptr != nullptr) cudaFree(ptr);
  }
  /** Allocate n bytes (n >= 1). */
  uint8_t *Alloc(size_t n) {
    GSC_CHECK(cudaMalloc(&ptr, n > 0 ? n : 1));
    return ptr;
  }
};

/**
 * A device view of an input: itself when it is device memory aligned for the
 * codecs' word loads, else an aligned device copy (untimed staging).
 */
const uint8_t *DeviceInput(const void *in, size_t n, cudaStream_t s,
                           DeviceBuffer *tmp) {
  const bool aligned = reinterpret_cast<uintptr_t>(in) % 8 == 0;
  if (OnDevice(in) && aligned) return static_cast<const uint8_t *>(in);
  const cudaMemcpyKind kind =
      OnDevice(in) ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
  GSC_CHECK(cudaMemcpyAsync(tmp->Alloc(n), in, n, kind, s));
  return tmp->ptr;
}

/**
 * Compress n bytes with one setting.
 * @return compressed bytes written to output (0 never: failures throw)
 */
size_t RunCompress(int index, void *output, size_t cap, const void *input,
                   size_t n) {
  Codec *c = CachedCodec(index);
  if (!c->Accepts(n)) throw CodecError("input size not accepted");
  const cudaStream_t s = c->stream();
  DeviceBuffer in_tmp, out_tmp;
  const uint8_t *d_in = DeviceInput(input, n, s, &in_tmp);
  const size_t bound = c->Bound(n);
  const bool direct = OnDevice(output) && cap >= bound &&
                      reinterpret_cast<uintptr_t>(output) % 8 == 0;
  uint8_t *d_out = direct ? static_cast<uint8_t *>(output) : out_tmp.Alloc(bound);
  c->PrepareCompress(n);
  GSC_CHECK(cudaStreamSynchronize(s));
  if (c->FirstUse(n, false)) {
    // Untimed warm-up on this thread's first call at this size.
    c->Compress(c->Preprocess(d_in, n), n, d_out, bound);
    GSC_CHECK(cudaStreamSynchronize(s));
    c->PrepareCompress(n);
    GSC_CHECK(cudaStreamSynchronize(s));
  }
  {
    // Preprocess + Compress, exactly the region the v2 labels timed.
    CodecKernelTimer timer(s);
    const uint8_t *src = c->Preprocess(d_in, n);
    c->Compress(src, n, d_out, bound);
  }
  GSC_CHECK(cudaStreamSynchronize(s));
  const size_t comp = c->CompressedBytes(d_out);
  if (comp == 0 || comp > cap) throw CodecError("compression failed or overflowed");
  if (!direct) {
    GSC_CHECK(cudaMemcpy(output, d_out, comp, cudaMemcpyDefault));
  }
  return comp;
}

/** Decompress comp bytes into exactly n bytes with one setting. */
void RunDecompress(int index, void *output, size_t n, const void *input,
                   size_t comp) {
  Codec *c = CachedCodec(index);
  const cudaStream_t s = c->stream();
  DeviceBuffer in_tmp, out_tmp;
  const uint8_t *d_in = DeviceInput(input, comp, s, &in_tmp);
  const bool direct = OnDevice(output) &&
                      reinterpret_cast<uintptr_t>(output) % 8 == 0;
  uint8_t *d_out = direct ? static_cast<uint8_t *>(output) : out_tmp.Alloc(n);
  c->PrepareDecode(n);
  GSC_CHECK(cudaStreamSynchronize(s));
  uint8_t *dst = c->DecodeTarget(d_out);
  if (c->FirstUse(n, true)) {
    // Untimed warm-up on this thread's first call at this size.
    c->PrepareDecompress(d_in, comp);
    c->Decompress(d_in, comp, dst, n);
    c->Postprocess(d_out, n);
    GSC_CHECK(cudaStreamSynchronize(s));
  }
  c->PrepareDecompress(d_in, comp);
  {
    // Decompress + Postprocess, exactly the region the v2 labels timed.
    CodecKernelTimer timer(s);
    c->Decompress(d_in, comp, dst, n);
    c->Postprocess(d_out, n);
  }
  GSC_CHECK(cudaStreamSynchronize(s));
  if (!c->DecompressOk()) throw CodecError("decompression failed");
  if (!direct) GSC_CHECK(cudaMemcpy(output, d_out, n, cudaMemcpyDefault));
}

#undef GSC_CHECK
#endif  // CTP_ENABLE_CUDA

}  // namespace

const char *GpuSettingSpec(int index) {
  return (index >= 0 && index < kGpuSettingCount) ? kSpecs[index] : nullptr;
}

int GpuSettingIndex(const std::string &spec) {
  const std::string c = Canonical(spec);
  if (c.empty()) return -1;
  for (int i = 0; i < kGpuSettingCount; ++i) {
    if (c == kSpecs[i]) return i;
  }
  return -1;
}

bool GpuSettingAvailable(int index) {
  if (index < 0 || index >= kGpuSettingCount) return false;
#if CTP_ENABLE_CUDA
  const std::string b = BaseOf(index);
  if (b == "store") return true;
  if (b == "ndzip") return CTP_ENABLE_NDZIP != 0;
  if (b == "gpulz") return CTP_ENABLE_GPULZ != 0;
  if (b == "spspeed" || b == "spratio") return CTP_ENABLE_FPCOMPRESS != 0;
  return CTP_ENABLE_NVCOMP != 0;
#else
  return false;
#endif
}

GpuSettingCodec::GpuSettingCodec(int index) : index_(index) {}

GpuSettingCodec::~GpuSettingCodec() = default;

bool GpuSettingCodec::Compress(void *output, size_t &output_size, void *input,
                               size_t input_size) {
#if CTP_ENABLE_CUDA
  LastCodecKernelMs() = -1.0;
  if (!GpuSettingAvailable(index_) || output == nullptr || input == nullptr) {
    return false;
  }
  try {
    output_size = RunCompress(index_, output, output_size, input, input_size);
    return true;
  } catch (const std::exception &) {
    cudaGetLastError();  // clear a non-sticky error for the next call
    return false;
  }
#else
  (void)output;
  (void)output_size;
  (void)input;
  (void)input_size;
  return false;
#endif
}

bool GpuSettingCodec::Decompress(void *output, size_t &output_size, void *input,
                                 size_t input_size) {
#if CTP_ENABLE_CUDA
  LastCodecKernelMs() = -1.0;
  if (!GpuSettingAvailable(index_) || output == nullptr || input == nullptr) {
    return false;
  }
  try {
    RunDecompress(index_, output, output_size, input, input_size);
    return true;
  } catch (const std::exception &) {
    cudaGetLastError();
    return false;
  }
#else
  (void)output;
  (void)output_size;
  (void)input;
  (void)input_size;
  return false;
#endif
}

size_t GpuSettingCodec::MaxCompressedSize(size_t input_size) {
#if CTP_ENABLE_CUDA
  if (!GpuSettingAvailable(index_)) return 0;
  try {
    return CachedCodec(index_)->Bound(input_size);
  } catch (const std::exception &) {
    return 0;
  }
#else
  (void)input_size;
  return 0;
#endif
}

}  // namespace ctp
