/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

// QuantizeDevice / DequantizeDevice: every float32/float64 chunk quantizes
// under every positive bound, and every element decodes within the bound or
// bit-exact. Checked element by element with exact arithmetic, and against a
// host decoder written from the format.
#include "basic_test.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "clio_ctp/compress/compress_factory.h"
#include "clio_ctp/compress/preprocess/quantization.h"

namespace pp = ctp::compress::preprocess;

namespace {

struct Dev {
  void *p = nullptr;
  explicit Dev(size_t bytes) {
    REQUIRE(cudaMalloc(&p, bytes ? bytes : 1) == cudaSuccess);
  }
  ~Dev() { cudaFree(p); }
};

/** |a - b| <= eb, exactly: TwoSum gives a - b = s + e with no error. */
bool WithinBound(double a, double b, double eb) {
  if (!(eb >= 0.0)) return false;
  const double s = a - b;
  const double bb = s - a;
  const double e = (a - (s - bb)) + (-b - bb);
  if (!std::isfinite(s)) return false;
  const double S = std::fabs(s);
  const double E = (s < 0.0) ? -e : e;
  if (S < eb / 2) return true;
  if (S > 2 * eb) return false;
  return E <= eb - S;  // eb - S is exact here (Sterbenz)
}

bool SameBits(const void *a, const void *b, size_t n) {
  return std::memcmp(a, b, n) == 0;
}

template <typename T>
T HostDecode(const pp::DeviceQuantizeParams &p, const std::vector<char> &buf,
             size_t n, size_t i) {
  const size_t w = pp::PrecisionToBytes(p.precision);
  if (p.escapes && ((static_cast<unsigned char>(buf[n * w + i / 8]) >> (i % 8)) & 1)) {
    T v;
    std::memcpy(&v, buf.data() + i * w, sizeof(T));
    return v;
  }
  int64_t q = 0;
  switch (w) {
    case 1: { int8_t x; std::memcpy(&x, buf.data() + i, 1); q = x; break; }
    case 2: { int16_t x; std::memcpy(&x, buf.data() + i * 2, 2); q = x; break; }
    case 4: { int32_t x; std::memcpy(&x, buf.data() + i * 4, 4); q = x; break; }
    default: { std::memcpy(&q, buf.data() + i * 8, 8); break; }
  }
  return static_cast<T>(
      std::fma(static_cast<double>(q), 1.0 / p.scale, p.data_min));
}

struct Outcome {
  pp::DeviceQuantizeParams p;
  size_t bytes = 0;
  size_t violations = 0;  // bound missed, or a non-finite value changed
  size_t spec_mismatch = 0;  // device decode != host decode
  size_t map_mismatch = 0;   // escape count or escaped bits wrong
};

template <typename T>
Outcome RoundTrip(const std::vector<T> &v, double eb) {
  Outcome o;
  const size_t n = v.size(), elem = sizeof(T);
  const size_t cap = pp::QuantizeCapacity(n, elem);
  Dev in(n * elem), q(cap), dec(n * elem);
  REQUIRE(cudaMemcpy(in.p, v.data(), n * elem, cudaMemcpyHostToDevice) ==
          cudaSuccess);
  const bool q_ok =
      pp::QuantizeDevice(in.p, n, elem, eb, q.p, &o.bytes, &o.p);
  INFO("refusal=" << pp::QuantizeRefusalToken(o.p.refusal));
  REQUIRE(q_ok);
  REQUIRE(o.bytes == pp::QuantizedBytes(n, o.p));
  REQUIRE(o.bytes <= cap);
  REQUIRE(o.p.elem_bytes == static_cast<int>(elem));
  if (o.p.escapes) REQUIRE(pp::PrecisionToBytes(o.p.precision) == elem);
  REQUIRE(pp::DequantizeDevice(q.p, n, o.p, dec.p));

  std::vector<char> packed(o.bytes);
  std::vector<T> out(n);
  REQUIRE(cudaMemcpy(packed.data(), q.p, o.bytes, cudaMemcpyDeviceToHost) ==
          cudaSuccess);
  REQUIRE(cudaMemcpy(out.data(), dec.p, n * elem,
                     cudaMemcpyDeviceToHost) == cudaSuccess);

  const size_t w = pp::PrecisionToBytes(o.p.precision);
  uint64_t set_bits = 0;
  for (size_t i = 0; i < n; ++i) {
    const bool escaped =
        o.p.escapes &&
        ((static_cast<unsigned char>(packed[n * w + i / 8]) >> (i % 8)) & 1);
    set_bits += escaped;
    if (escaped && !SameBits(packed.data() + i * w, &v[i], elem)) {
      ++o.map_mismatch;
    }
    const T h = HostDecode<T>(o.p, packed, n, i);
    if (!SameBits(&h, &out[i], elem)) ++o.spec_mismatch;
    if (std::isfinite(v[i])) {
      if (!WithinBound(out[i], v[i], eb)) ++o.violations;
    } else if (!SameBits(&out[i], &v[i], elem)) {
      ++o.violations;
    }
  }
  if (o.p.escapes && set_bits != o.p.escape_count) ++o.map_mismatch;
  for (size_t i = n * w + (o.p.escapes ? (n + 7) / 8 : 0); i < o.bytes; ++i) {
    if (packed[i] != 0) ++o.map_mismatch;  // padding must be zeros
  }
  REQUIRE(o.bytes % 8 == 0);
  return o;
}

template <typename T>
std::vector<std::pair<std::string, std::vector<T>>> Datasets() {
  using L = std::numeric_limits<T>;
  std::vector<std::pair<std::string, std::vector<T>>> d;
  const size_t n = 1 << 16;
  std::mt19937_64 rng(693);
  for (double a : {1e-3, 1.0, 100.0, 1e4, 1e8, 1.96e12}) {
    std::vector<T> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = static_cast<T>(std::sin(i * 1e-3) * a);
    d.emplace_back("sine*" + std::to_string(a), std::move(v));
  }
  {
    std::vector<T> v(n);
    std::uniform_real_distribution<double> u(-1e6, 1e6);
    for (auto &x : v) x = static_cast<T>(u(rng));
    d.emplace_back("uniform 1e6", std::move(v));
  }
  {
    std::vector<T> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = static_cast<T>(i % 1000);
    d.emplace_back("integers", std::move(v));
  }
  {
    std::vector<T> v(n);
    for (size_t i = 0; i < n; ++i) {
      v[i] = (i % 2) ? static_cast<T>(std::sin(i * 1e-3) * 1.96e12) : T(0);
    }
    d.emplace_back("warpx-like", std::move(v));
  }
  {
    std::vector<T> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = static_cast<T>(std::sin(i * 1e-3));
    v[777] = L::quiet_NaN();
    d.emplace_back("sine + NaN", std::move(v));
  }
  {
    std::vector<T> v(n, T(3.25));
    d.emplace_back("constant", std::move(v));
  }
  {
    // Specials, binade edges and subnormals, repeated through the chunk.
    std::vector<T> s = {T(0), -T(0), L::infinity(), -L::infinity(),
                        L::quiet_NaN(), L::signaling_NaN(), L::max(),
                        L::lowest(), L::min(), L::denorm_min(),
                        -L::denorm_min(), T(1), T(-1)};
    T sig;
    std::memset(&sig, 0xFF, sizeof(T));  // NaN with a full payload
    s.push_back(sig);
    for (int k = L::min_exponent - 1; k < L::max_exponent; k += 7) {
      const T p = std::ldexp(T(1), k);
      s.push_back(p);
      s.push_back(std::nextafter(p, T(0)));
      s.push_back(-std::nextafter(p, L::infinity()));
    }
    std::vector<T> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = s[i % s.size()];
    d.emplace_back("specials", std::move(v));
  }
  {
    std::vector<T> v(n + 5);  // bitmap tail
    for (auto &x : v) {
      const uint64_t r = rng();
      std::memcpy(&x, &r, sizeof(T));
    }
    d.emplace_back("random bits", std::move(v));
  }
  d.emplace_back("one element", std::vector<T>{T(42.5)});
  return d;
}

template <typename T>
void RunMatrix(const std::vector<double> &bounds) {
  size_t cases = 0, escaped_cases = 0;
  for (const auto &[name, v] : Datasets<T>()) {
    for (double eb : bounds) {
      CAPTURE(name, eb, sizeof(T));
      const Outcome o = RoundTrip(v, eb);
      CHECK(o.violations == 0);
      CHECK(o.spec_mismatch == 0);
      CHECK(o.map_mismatch == 0);
      ++cases;
      escaped_cases += o.p.escapes;
    }
  }
  std::printf("[quantize] float%zu: %zu chunk/bound pairs, %zu used escapes\n",
              sizeof(T) * 8, cases, escaped_cases);
}

}  // namespace

TEST_CASE("float32 quantizes within the bound, never declining",
          "[gpu][quantize]") {
  RunMatrix<float>({1e-45, 1e-38, 1e-20, 1e-7, 1e-3, 0.01, 0.05, 0.5, 1.0,
                    1e3, 1e12, 1e30, 3e38, 1e300, 1.7e308,
                    std::numeric_limits<double>::infinity()});
}

TEST_CASE("float64 quantizes within the bound, never declining",
          "[gpu][quantize]") {
  RunMatrix<double>({5e-324, 1e-300, 1e-100, 1e-12, 1e-3, 0.01, 1.0, 1e12,
                     1e100, 1e300, 1.7e308,
                     std::numeric_limits<double>::infinity()});
}

TEST_CASE("chunks that fit keep the compact format", "[gpu][quantize]") {
  std::vector<float> v(1 << 16);
  for (size_t i = 0; i < v.size(); ++i) v[i] = std::sin(i * 1e-3) * 100.0f;
  Outcome o = RoundTrip(v, 0.01);
  CHECK_FALSE(o.p.escapes);
  CHECK(o.p.precision == 16);
  CHECK(o.bytes == v.size() * 2);

  v[777] = std::numeric_limits<float>::quiet_NaN();
  o = RoundTrip(v, 0.01);
  CHECK(o.p.escapes);
  CHECK(o.p.escape_count == 1);
  CHECK(o.violations == 0);

  std::vector<double> ints(1 << 16);
  for (size_t i = 0; i < ints.size(); ++i) ints[i] = static_cast<double>(i % 1000);
  o = RoundTrip(ints, 0.01);
  CHECK_FALSE(o.p.escapes);
  CHECK(o.p.elem_bytes == 8);
  CHECK(o.violations == 0);
}

TEST_CASE("wide float32 chunks escape only what the grid cannot hold",
          "[gpu][quantize]") {
  std::vector<float> v(1 << 16);
  for (size_t i = 0; i < v.size(); ++i) {
    v[i] = (i % 2) ? static_cast<float>(std::sin(i * 1e-3) * 1.96e12)
                   : static_cast<float>(std::sin(i * 1e-3));
  }
  const Outcome o = RoundTrip(v, 0.01);
  CHECK(o.p.escapes);
  CHECK(o.p.precision == 32);
  CHECK(o.violations == 0);
  // The small half stays on the grid.
  CHECK(o.p.escape_count <= v.size() / 2 + 64);
  CHECK(o.p.escape_count > 0);
}

TEST_CASE("QuantizeDevice rejects only invalid arguments", "[gpu][quantize]") {
  std::vector<float> v(64, 1.0f);
  Dev in(64 * 4), q(pp::QuantizeCapacity(64, 4));
  REQUIRE(cudaMemcpy(in.p, v.data(), 64 * 4, cudaMemcpyHostToDevice) ==
          cudaSuccess);
  size_t bytes = 0;
  pp::DeviceQuantizeParams p;
  for (double eb : {0.0, -1.0, std::nan("")}) {
    CAPTURE(eb);
    CHECK_FALSE(pp::QuantizeDevice(in.p, 64, 4, eb, q.p, &bytes, &p));
    CHECK(p.refusal == pp::QuantizeRefusal::kInvalidArgument);
  }
  CHECK_FALSE(pp::QuantizeDevice(in.p, 64, 3, 0.1, q.p, &bytes, &p));
  CHECK(p.refusal == pp::QuantizeRefusal::kInvalidArgument);
  CHECK_FALSE(pp::QuantizeDevice(in.p, 0, 4, 0.1, q.p, &bytes, &p));
}

TEST_CASE("nvcomp codecs round-trip every input length", "[gpu][codec]") {
  const std::vector<std::string> codecs = {
      "nvcomp-lz4",     "nvcomp-snappy", "nvcomp-zstd", "nvcomp-gdeflate",
      "nvcomp-deflate", "nvcomp-ans",    "nvcomp-cascaded", "nvcomp-bitcomp"};
  const std::vector<size_t> sizes = {1,          7,          8,       9,
                                     15,         1023,       65538,   196613,
                                     1920266,    2097152,    4194306, 8388612};
  std::mt19937_64 rng(24);
  for (const auto &name : codecs) {
    auto codec = ctp::CompressionFactory::GetPreset(
        name, ctp::CompressionPreset::BALANCED);
    REQUIRE(codec != nullptr);
    std::string bad;
    for (size_t n : sizes) {
      std::vector<char> in(n);
      for (size_t i = 0; i < n; ++i) in[i] = static_cast<char>((i / 7) % 5 + (rng() % 3));
      std::vector<char> comp(codec->MaxCompressedSize(n) + n / 20 + 1024);
      size_t csize = comp.size();
      REQUIRE(codec->Compress(comp.data(), csize, in.data(), n));
      std::vector<char> out(n + 64, 0);
      size_t dsize = out.size();
      REQUIRE(codec->Decompress(out.data(), dsize, comp.data(), csize));
      if (dsize != n || std::memcmp(out.data(), in.data(), n) != 0) {
        bad += " " + std::to_string(n) + "(got " + std::to_string(dsize) + ")";
      }
      if (n % 8 == 0) CHECK((dsize == n && !std::memcmp(out.data(), in.data(), n)));
    }
    std::printf("[codec] %-16s %s\n", name.c_str(),
                bad.empty() ? "every length round-trips" : ("FAILS at" + bad).c_str());
  }
}
