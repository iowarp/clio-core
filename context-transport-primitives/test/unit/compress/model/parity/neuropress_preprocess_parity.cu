/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file neuropress_preprocess_parity.cu
 * @brief Differential test for the two PREPROCESSORS: byte shuffle and
 *        error-bound quantization.
 *
 * The existing parity harnesses cover the model (inference, ranking, SGD) and
 * the statistics that feed it. They do not touch either preprocessor, and the
 * shuffle one is not hypothetical: every chunk in the 1 GiB end-to-end run
 * selected a shuffled action, so Clio's ShuffleKernel decides the bytes the
 * codec actually sees. If it disagrees with NeuroPress's
 * byte_shuffle_kernels.cu by one byte, every shuffled chunk is encoded from a
 * different buffer than upstream would have encoded -- while still
 * round-tripping perfectly through Clio's own inverse, so no correctness test
 * on Clio alone can see it.
 *
 * Both sides are driven through their REAL production entry points:
 *   - upstream: byte_shuffle_simple / byte_unshuffle_simple at
 *     SHUFFLE_CHUNK_SIZE, the same call gpucompress_compress.cpp and
 *     :1257 make, and quantize_simple / dequantize_simple as called at :434.
 *   - Clio: ByteShuffleDevice / ByteUnshuffleDevice and QuantizeDevice /
 *     DequantizeDevice, the same calls compressor_runtime.cc:1658, :2247,
 *     :2811, :1605 and :2181 make.
 *
 * Four things are checked, all byte-for-byte with a first-divergence report:
 *
 *   1. SHUFFLE. Identical device input -> identical shuffled bytes. Clio's
 *      host ByteShuffle() is compared too, since the blob format has to be
 *      the same whichever path produced it.
 *   2. UNSHUFFLE, and INTERCHANGE: Clio unshuffles upstream's shuffled blob
 *      and vice versa. Round-tripping against your own inverse proves
 *      self-consistency, not compatibility.
 *   3. QUANTIZE. Identical input -> identical precision, scale, data range
 *      and packed bytes.
 *   4. DEQUANTIZE, and the error bound each side actually achieves.
 *
 * The size sweep deliberately straddles SHUFFLE_CHUNK_SIZE (the plane layout
 * is per 256 KiB block, so an off-by-one in the block walk only shows above
 * it), sub-element buffers (upstream has a `num_elements <= 1` early-out that
 * Clio's port does not), and non-multiples of the element size (the trailing
 * partial element is copied verbatim, a path easy to drop).
 *
 * Built only when the NeuroPress source tree is present (see CMakeLists.txt);
 * this is a cross-project check, not part of the default build.
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "clio_ctp/compress/preprocess/byte_shuffle.h"
#include "clio_ctp/compress/preprocess/quantization.h"

// --- Native NeuroPress, compiled from its own source. internal.hpp is
//     included for SHUFFLE_CHUNK_SIZE itself rather than a local 256*1024:
//     the point is to drive upstream with the value ITS compress path uses,
//     so if upstream ever changes it this harness follows instead of
//     silently testing a chunk size nothing runs at.
#include "api/internal.hpp"
#include "preprocessing/byte_shuffle.cuh"
#include "preprocessing/quantization.cuh"

namespace {

int g_failures = 0;
long g_checks = 0;

void Check(bool cond, const std::string &what) {
  ++g_checks;
  if (!cond) {
    ++g_failures;
    std::printf("  [FAIL] %s\n", what.c_str());
  }
}

/**
 * Byte-for-byte comparison that reports WHERE, not just whether.
 *
 * The first differing index is the whole point: a mismatch at the very first
 * plane boundary means a different layout, one at `n - leftover` means the
 * trailing partial element, and one only past 256 KiB means the per-block
 * walk. Reporting "buffers differ" would leave all three looking alike.
 */
bool SameBytes(const std::vector<uint8_t> &a, const std::vector<uint8_t> &b,
               const std::string &what) {
  ++g_checks;
  if (a.size() != b.size()) {
    ++g_failures;
    std::printf("  [FAIL] %s: size %zu vs %zu\n", what.c_str(), a.size(),
                b.size());
    return false;
  }
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i]) {
      ++g_failures;
      size_t ndiff = 0;
      for (size_t j = i; j < a.size(); ++j) {
        if (a[j] != b[j]) ++ndiff;
      }
      std::printf(
          "  [FAIL] %s: first diff at byte %zu (0x%02x vs 0x%02x), "
          "%zu/%zu bytes differ\n",
          what.c_str(), i, a[i], b[i], ndiff, a.size());
      return false;
    }
  }
  return true;
}

/** Deterministic byte pattern -- same generator drives both sides. */
std::vector<uint8_t> MakeBytes(size_t n, uint32_t seed) {
  std::vector<uint8_t> v(n);
  uint32_t s = seed * 2654435761u + 1u;
  for (size_t i = 0; i < n; ++i) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    v[i] = static_cast<uint8_t>(s >> 24);
  }
  return v;
}

std::vector<uint8_t> Download(const void *d, size_t n) {
  std::vector<uint8_t> h(n);
  cudaMemcpy(h.data(), d, n, cudaMemcpyDeviceToHost);
  return h;
}

// ===========================================================================
// Part 1 + 2: byte shuffle / unshuffle
// ===========================================================================

void ShuffleCase(size_t num_bytes, unsigned elem_size) {
  char tag[128];
  std::snprintf(tag, sizeof(tag), "shuffle n=%zu elem=%u", num_bytes,
                elem_size);

  const std::vector<uint8_t> src = MakeBytes(num_bytes, 0x5eed ^ num_bytes);

  uint8_t *d_in = nullptr;
  uint8_t *d_clio = nullptr;
  if (cudaMalloc(&d_in, num_bytes) != cudaSuccess ||
      cudaMalloc(&d_clio, num_bytes) != cudaSuccess) {
    std::printf("  [SKIP] %s: cudaMalloc failed\n", tag);
    cudaFree(d_in);
    cudaFree(d_clio);
    return;
  }
  cudaMemcpy(d_in, src.data(), num_bytes, cudaMemcpyHostToDevice);

  // --- upstream, through its real entry point and real chunk size.
  uint8_t *d_native = byte_shuffle_simple(d_in, num_bytes, elem_size,
                                          gpucompress::SHUFFLE_CHUNK_SIZE, 0);
  // --- Clio device.
  const bool clio_ok = ctp::compress::preprocess::ByteShuffleDevice(
      d_in, d_clio, num_bytes, elem_size);
  // --- Clio host.
  std::vector<uint8_t> host_shuf(num_bytes);
  const bool host_ok = ctp::compress::preprocess::ByteShuffle(
      src.data(), num_bytes, elem_size, host_shuf.data());

  Check(d_native != nullptr, std::string(tag) + ": upstream returned a buffer");
  Check(clio_ok, std::string(tag) + ": Clio device shuffle succeeded");
  Check(host_ok, std::string(tag) + ": Clio host shuffle succeeded");
  if (!d_native || !clio_ok || !host_ok) {
    cudaFree(d_in);
    cudaFree(d_clio);
    if (d_native) cudaFree(d_native);
    return;
  }

  const std::vector<uint8_t> native_shuf = Download(d_native, num_bytes);
  const std::vector<uint8_t> clio_shuf = Download(d_clio, num_bytes);

  SameBytes(native_shuf, clio_shuf, std::string(tag) + ": device bytes");
  SameBytes(native_shuf, host_shuf, std::string(tag) + ": host bytes");

  // --- Unshuffle, and INTERCHANGE: each side inverts the other's blob.
  uint8_t *d_native_shuf = nullptr;
  cudaMalloc(&d_native_shuf, num_bytes);
  cudaMemcpy(d_native_shuf, native_shuf.data(), num_bytes,
             cudaMemcpyHostToDevice);

  uint8_t *d_native_back = byte_unshuffle_simple(
      d_native_shuf, num_bytes, elem_size, gpucompress::SHUFFLE_CHUNK_SIZE, 0);
  uint8_t *d_clio_back = nullptr;
  cudaMalloc(&d_clio_back, num_bytes);
  const bool unshuf_ok = ctp::compress::preprocess::ByteUnshuffleDevice(
      d_native_shuf, d_clio_back, num_bytes, elem_size);

  std::vector<uint8_t> host_back(num_bytes);
  ctp::compress::preprocess::ByteUnshuffle(native_shuf.data(), num_bytes,
                                           elem_size, host_back.data());

  if (d_native_back && unshuf_ok) {
    const std::vector<uint8_t> nb = Download(d_native_back, num_bytes);
    const std::vector<uint8_t> cb = Download(d_clio_back, num_bytes);
    // Both must invert to the ORIGINAL, and agree with each other.
    SameBytes(nb, src, std::string(tag) + ": upstream unshuffle restores src");
    SameBytes(cb, src,
              std::string(tag) + ": Clio unshuffle of UPSTREAM blob restores src");
    SameBytes(host_back, src,
              std::string(tag) + ": Clio host unshuffle of UPSTREAM blob");
  } else {
    Check(false, std::string(tag) + ": unshuffle path ran");
  }

  cudaFree(d_in);
  cudaFree(d_clio);
  cudaFree(d_native);
  cudaFree(d_native_shuf);
  cudaFree(d_clio_back);
  if (d_native_back) cudaFree(d_native_back);
}

// ===========================================================================
// Part 3 + 4: quantize / dequantize
// ===========================================================================

/** Data regimes that exercise different precision selections and the
 *  constant-data corner where upstream clamps the range. */
enum class Regime { kRandom, kSmooth, kConstant, kWideRange, kNegative };

const char *RegimeName(Regime r) {
  switch (r) {
    case Regime::kRandom: return "random";
    case Regime::kSmooth: return "smooth";
    case Regime::kConstant: return "constant";
    case Regime::kWideRange: return "wide";
    default: return "negative";
  }
}

std::vector<float> MakeFloats(size_t n, Regime regime) {
  std::vector<float> v(n);
  uint32_t s = 0x1234567u;
  for (size_t i = 0; i < n; ++i) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    const double u = static_cast<double>(s) / 4294967296.0;  // [0,1)
    switch (regime) {
      case Regime::kRandom:
        v[i] = static_cast<float>(u * 2.0 - 1.0);
        break;
      case Regime::kSmooth:
        v[i] = static_cast<float>(std::sin(static_cast<double>(i) * 0.001) +
                                  0.01 * u);
        break;
      case Regime::kConstant:
        v[i] = 3.5f;
        break;
      case Regime::kWideRange:
        v[i] = static_cast<float>((u - 0.5) * 2.0e6);
        break;
      case Regime::kNegative:
        v[i] = static_cast<float>(-1000.0 - u * 500.0);
        break;
    }
  }
  return v;
}

void QuantizeCase(size_t n, double error_bound, Regime regime) {
  char tag[160];
  std::snprintf(tag, sizeof(tag), "quantize n=%zu eb=%.1e %s", n, error_bound,
                RegimeName(regime));

  const std::vector<float> src = MakeFloats(n, regime);
  const size_t in_bytes = n * sizeof(float);

  float *d_in = nullptr;
  if (cudaMalloc(&d_in, in_bytes) != cudaSuccess) {
    std::printf("  [SKIP] %s: cudaMalloc failed\n", tag);
    return;
  }
  cudaMemcpy(d_in, src.data(), in_bytes, cudaMemcpyHostToDevice);

  // --- upstream.
  QuantizationConfig cfg(QuantizationType::LINEAR, error_bound, n,
                         sizeof(float), QuantizationPrecision::AUTO);
  QuantizationResult nat = quantize_simple(d_in, n, sizeof(float), cfg, 0);

  // --- Clio. Output buffer sized for the worst case (int32).
  void *d_clio_out = nullptr;
  cudaMalloc(&d_clio_out, n * 4);
  size_t clio_bytes = 0;
  ctp::compress::preprocess::DeviceQuantizeParams params;
  const bool clio_ok = ctp::compress::preprocess::QuantizeDevice(
      d_in, n, error_bound, d_clio_out, &clio_bytes, &params);

  Check(nat.d_quantized != nullptr, std::string(tag) + ": upstream quantized");
  Check(clio_ok, std::string(tag) + ": Clio quantized");
  if (!nat.d_quantized || !clio_ok) {
    // Report the asymmetry explicitly -- one side refusing where the other
    // succeeds is itself a divergence, not a skip.
    std::printf("      upstream=%s clio=%s\n",
                nat.d_quantized ? "ok" : "FAILED", clio_ok ? "ok" : "FAILED");
    cudaFree(d_in);
    cudaFree(d_clio_out);
    if (nat.d_quantized && nat.owns_output) cudaFree(nat.d_quantized);
    return;
  }

  // Metadata first: precision decides the packed width, so a mismatch there
  // makes the byte comparison below meaningless rather than merely failing.
  const bool prec_same = (nat.actual_precision == params.precision);
  Check(prec_same,
        std::string(tag) + ": precision " +
            std::to_string(nat.actual_precision) + " vs " +
            std::to_string(params.precision));
  Check(nat.quantized_bytes == clio_bytes,
        std::string(tag) + ": packed bytes " +
            std::to_string(nat.quantized_bytes) + " vs " +
            std::to_string(clio_bytes));
  Check(nat.data_min == params.data_min,
        std::string(tag) + ": data_min");
  Check(nat.data_max == params.data_max,
        std::string(tag) + ": data_max");
  Check(nat.scale_factor == params.scale,
        std::string(tag) + ": scale " + std::to_string(nat.scale_factor) +
            " vs " + std::to_string(params.scale));

  if (prec_same && nat.quantized_bytes == clio_bytes) {
    SameBytes(Download(nat.d_quantized, nat.quantized_bytes),
              Download(d_clio_out, clio_bytes),
              std::string(tag) + ": packed quantized bytes");

    // --- Dequantize both and compare the restored floats, then the error
    //     each side actually achieved against the bound that was asked for.
    void *d_nat_deq = dequantize_simple(nat.d_quantized, nat, 0);
    float *d_clio_deq = nullptr;
    cudaMalloc(&d_clio_deq, in_bytes);
    const bool deq_ok = ctp::compress::preprocess::DequantizeDevice(
        d_clio_out, n, params, d_clio_deq);
    Check(d_nat_deq != nullptr, std::string(tag) + ": upstream dequantized");
    Check(deq_ok, std::string(tag) + ": Clio dequantized");

    if (d_nat_deq && deq_ok) {
      SameBytes(Download(d_nat_deq, in_bytes), Download(d_clio_deq, in_bytes),
                std::string(tag) + ": dequantized bytes");

      std::vector<float> nd(n), cd(n);
      cudaMemcpy(nd.data(), d_nat_deq, in_bytes, cudaMemcpyDeviceToHost);
      cudaMemcpy(cd.data(), d_clio_deq, in_bytes, cudaMemcpyDeviceToHost);
      double nerr = 0.0, cerr = 0.0;
      for (size_t i = 0; i < n; ++i) {
        nerr = std::max(nerr, std::fabs(static_cast<double>(nd[i]) - src[i]));
        cerr = std::max(cerr, std::fabs(static_cast<double>(cd[i]) - src[i]));
      }
      std::printf("      %s: max|err| upstream=%.3e clio=%.3e bound=%.1e "
                  "prec=%d\n",
                  tag, nerr, cerr, error_bound, params.precision);
      // The bound is the contract both sides advertise. Only assert it where
      // upstream itself claims it is achievable -- it prints a warning and
      // deliberately exceeds the bound when the request is below float32
      // resolution, and Clio records the same via bound_achievable.
      if (params.bound_achievable) {
        Check(nerr <= error_bound,
              std::string(tag) + ": upstream within bound");
        Check(cerr <= error_bound, std::string(tag) + ": Clio within bound");
      }
    }
    if (d_nat_deq) cudaFree(d_nat_deq);
    cudaFree(d_clio_deq);
  }

  cudaFree(d_in);
  cudaFree(d_clio_out);
  if (nat.owns_output) cudaFree(nat.d_quantized);
}

}  // namespace

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::printf("No CUDA device -- skipping.\n");
    return 77;
  }

  std::printf("=== Phase 3: byte shuffle / unshuffle ===\n");
  // Straddle every boundary the two implementations could disagree on:
  // sub-element, exact element multiples and not, the 256 KiB block edge,
  // and multi-block including the production 4 MiB chunk.
  const size_t kSizes[] = {
      1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 255, 256, 257,
      1023, 1024, 4095, 4096, 4097,
      262143, 262144, 262145,           // SHUFFLE_CHUNK_SIZE - 1, =, + 1
      262144 + 7, 524288, 524288 + 3,   // two blocks, and a ragged tail
      1048576, 4194304, 4194304 + 5};   // 1 MiB; production 4 MiB chunk
  for (unsigned elem : {2u, 4u, 8u}) {
    for (size_t n : kSizes) {
      ShuffleCase(n, elem);
    }
  }

  std::printf("\n=== Phase 4: quantize / dequantize ===\n");
  for (Regime r : {Regime::kRandom, Regime::kSmooth, Regime::kConstant,
                   Regime::kWideRange, Regime::kNegative}) {
    for (double eb : {1e-1, 1e-2, 1e-3, 1e-5, 1e-7}) {
      QuantizeCase(1 << 16, eb, r);
    }
  }
  // A production-sized chunk at the bound the bridge enumerates.
  QuantizeCase(1 << 20, 1e-3, Regime::kSmooth);

  std::printf("\n===== %ld checks, %d failures =====\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
