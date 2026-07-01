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

#include "basic_test.h"
#include "clio_ctp/compress/compress_factory.h"
#include <utility>

#if CTP_ENABLE_NVCOMP
#include <cuda_runtime.h>
#endif

#if CTP_ENABLE_ZFP_SYCL
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>
#endif

#if CTP_ENABLE_CUSZ || CTP_ENABLE_NDZIP || CTP_ENABLE_CUSZP
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <vector>
#endif

TEST_CASE("TestCompress") {
  std::string raw = "Hello, World!";
  std::vector<char> compressed(1024);
  std::vector<char> decompressed(1024);

  PAGE_DIVIDE("BZIP2") {
    ctp::Bzip2 bzip;
    size_t cmpr_size = 1024, raw_size = 1024;
    bzip.Compress(compressed.data(), cmpr_size,
                  raw.data(), raw.size());
    bzip.Decompress(decompressed.data(), raw_size,
                    compressed.data(), cmpr_size);
    REQUIRE(raw == std::string(decompressed.data(), raw_size));
  }

  PAGE_DIVIDE("LZO") {
    ctp::Lzo lzo;
    size_t cmpr_size = 1024, raw_size = 1024;
    lzo.Compress(compressed.data(), cmpr_size,
                 raw.data(), raw.size());
    lzo.Decompress(decompressed.data(), raw_size,
                   compressed.data(), cmpr_size);
    REQUIRE(raw == std::string(decompressed.data(), raw_size));
  }

  PAGE_DIVIDE("Zstd") {
    ctp::Zstd zstd;
    size_t cmpr_size = 1024, raw_size = 1024;
    zstd.Compress(compressed.data(), cmpr_size,
                  raw.data(), raw.size());
    zstd.Decompress(decompressed.data(), raw_size,
                    compressed.data(), cmpr_size);
    REQUIRE(raw == std::string(decompressed.data(), raw_size));
  }

  PAGE_DIVIDE("LZ4") {
    ctp::Lz4 lz4;
    size_t cmpr_size = 1024, raw_size = 1024;
    lz4.Compress(compressed.data(), cmpr_size,
                 raw.data(), raw.size());
    lz4.Decompress(decompressed.data(), raw_size,
                   compressed.data(), cmpr_size);
    REQUIRE(raw == std::string(decompressed.data(), raw_size));
  }

  PAGE_DIVIDE("Zlib") {
    ctp::Zlib zlib;
    size_t cmpr_size = 1024, raw_size = 1024;
    zlib.Compress(compressed.data(), cmpr_size,
                  raw.data(), raw.size());
    zlib.Decompress(decompressed.data(), raw_size,
                    compressed.data(), cmpr_size);
    REQUIRE(raw == std::string(decompressed.data(), raw_size));
  }

  PAGE_DIVIDE("Lzma") {
    ctp::Lzma lzma;
    size_t cmpr_size = 1024, raw_size = 1024;
    lzma.Compress(compressed.data(), cmpr_size,
                  raw.data(), raw.size());
    lzma.Decompress(decompressed.data(), raw_size,
                    compressed.data(), cmpr_size);
    REQUIRE(raw == std::string(decompressed.data(), raw_size));
  }

  PAGE_DIVIDE("Brotli") {
    ctp::Brotli brotli;
    size_t cmpr_size = 1024, raw_size = 1024;
    brotli.Compress(compressed.data(), cmpr_size,
                    raw.data(), raw.size());
    brotli.Decompress(decompressed.data(), raw_size,
                      compressed.data(), cmpr_size);
    REQUIRE(raw == std::string(decompressed.data(), raw_size));
  }

  PAGE_DIVIDE("Snappy") {
    ctp::Snappy snappy;
    size_t cmpr_size = 1024, raw_size = 1024;
    snappy.Compress(compressed.data(), cmpr_size,
                    raw.data(), raw.size());
    snappy.Decompress(decompressed.data(), raw_size,
                      compressed.data(), cmpr_size);
    REQUIRE(raw == std::string(decompressed.data(), raw_size));
  }

  PAGE_DIVIDE("Blosc2") {
    ctp::Blosc blosc;
    size_t cmpr_size = 1024, raw_size = 1024;
    blosc.Compress(compressed.data(), cmpr_size,
                   raw.data(), raw.size());
    blosc.Decompress(decompressed.data(), raw_size,
                     compressed.data(), cmpr_size);
    REQUIRE(raw == std::string(decompressed.data(), raw_size));
  }
}

#if CTP_ENABLE_ZFP_SYCL
// zfp-sycl is lossy fixed-rate GPU (SYCL) compression. Requires a SYCL device
// at runtime (e.g. ONEAPI_DEVICE_SELECTOR=opencl:cpu) and a SYCL-enabled libzfp
// loaded ahead of any stock libzfp on the loader path.
TEST_CASE("TestSyclZfpRoundTrip") {
  using ctp::CompressionFactory;
  using ctp::CompressionPreset;

  // 4096 floats of a smooth, exactly-representable-ish signal.
  const size_t n = 4096;
  std::vector<float> orig(n), deco(n, 0.0f);
  for (size_t i = 0; i < n; ++i) {
    orig[i] = std::sin(static_cast<float>(i) * 0.01f) * 100.0f;
  }
  const size_t raw_bytes = n * sizeof(float);
  std::vector<char> compressed(raw_bytes + 4096);

  PAGE_DIVIDE("compress (BALANCED) then decompress (FAST) -- self-describing") {
    auto comp = CompressionFactory::GetPreset("zfp-sycl",
                                              CompressionPreset::BALANCED);
    REQUIRE(comp != nullptr);
    size_t cmpr_size = compressed.size();
    REQUIRE(comp->Compress(compressed.data(), cmpr_size, orig.data(),
                           raw_bytes));
    REQUIRE(cmpr_size > 0);
    REQUIRE(cmpr_size < raw_bytes);  // fixed-rate 16 bits/value -> ~2x

    // Decompress with a DIFFERENT preset on purpose: the embedded header makes
    // the stream self-describing, so the rate mismatch must not corrupt output.
    auto dcmp = CompressionFactory::GetPreset("zfp-sycl",
                                              CompressionPreset::FAST);
    REQUIRE(dcmp != nullptr);
    size_t deco_size = raw_bytes;
    REQUIRE(dcmp->Decompress(deco.data(), deco_size, compressed.data(),
                             cmpr_size));
    REQUIRE(deco_size == raw_bytes);

    // Lossy, but fixed-rate 16 bits/float on a [-100,100] signal stays close.
    double max_err = 0.0;
    for (size_t i = 0; i < n; ++i) {
      max_err = std::max(max_err,
                         std::abs(static_cast<double>(orig[i] - deco[i])));
    }
    REQUIRE(max_err < 1.0);
  }
}
#endif  // CTP_ENABLE_SYCL

// Characterization test for the compressor registry's frozen ID mappings.
// These values are the on-disk wire protocol and the ML id scheme; renumbering
// any of them silently breaks stored blobs / trained models. The expected
// numbers are transcribed from the historical source and must never drift.
// (No GPU required: these are pure name<->id lookups.)
TEST_CASE("CompressorRegistryMappings") {
  using ctp::CompressionFactory;
  using ctp::CompressionPreset;

  PAGE_DIVIDE("wire id -> name (CompressionHeader.compress_lib_)") {
    REQUIRE(CompressionFactory::NameForWireId(0) == "brotli");
    REQUIRE(CompressionFactory::NameForWireId(1) == "bzip2");
    REQUIRE(CompressionFactory::NameForWireId(2) == "blosc2");
    REQUIRE(CompressionFactory::NameForWireId(3) == "fpzip");
    REQUIRE(CompressionFactory::NameForWireId(4) == "lz4");
    REQUIRE(CompressionFactory::NameForWireId(5) == "lzma");
    REQUIRE(CompressionFactory::NameForWireId(6) == "snappy");
    REQUIRE(CompressionFactory::NameForWireId(7) == "sz3");
    REQUIRE(CompressionFactory::NameForWireId(8) == "zfp");
    REQUIRE(CompressionFactory::NameForWireId(9) == "zlib");
    REQUIRE(CompressionFactory::NameForWireId(10) == "zstd");
    REQUIRE(CompressionFactory::NameForWireId(11) == "nvcomp-lz4");
    REQUIRE(CompressionFactory::NameForWireId(12) == "nvcomp-snappy");
    REQUIRE(CompressionFactory::NameForWireId(13) == "nvcomp-zstd");
    REQUIRE(CompressionFactory::NameForWireId(14) == "nvcomp-gdeflate");
    REQUIRE(CompressionFactory::NameForWireId(15) == "nvcomp-deflate");
    REQUIRE(CompressionFactory::NameForWireId(16) == "nvcomp-ans");
    REQUIRE(CompressionFactory::NameForWireId(17) == "zfp-sycl");
    REQUIRE(CompressionFactory::NameForWireId(18) == "cusz");
    REQUIRE(CompressionFactory::NameForWireId(19) == "ndzip");
    REQUIRE(CompressionFactory::NameForWireId(20) == "cuszp");
    // Out-of-range falls back to the historical default. (Registry rows are
    // build-independent, so the GPU names above resolve even without nvcomp.)
    REQUIRE(CompressionFactory::NameForWireId(-1) == "zstd");
    REQUIRE(CompressionFactory::NameForWireId(21) == "zstd");
    REQUIRE(CompressionFactory::NameForWireId(9999) == "zstd");
  }

  PAGE_DIVIDE("name + preset -> ML library id (base_id*10 + preset)") {
    const auto FAST = CompressionPreset::FAST;
    const auto BAL = CompressionPreset::BALANCED;
    const auto BEST = CompressionPreset::BEST;

    // Multi-mode lossless: id = base_id*10 + {1,2,3}.
    REQUIRE(CompressionFactory::GetLibraryId("bzip2", FAST) == 11);
    REQUIRE(CompressionFactory::GetLibraryId("bzip2", BAL) == 12);
    REQUIRE(CompressionFactory::GetLibraryId("bzip2", BEST) == 13);
    REQUIRE(CompressionFactory::GetLibraryId("zstd", BAL) == 22);
    REQUIRE(CompressionFactory::GetLibraryId("lz4", BAL) == 32);
    REQUIRE(CompressionFactory::GetLibraryId("zlib", BAL) == 42);
    REQUIRE(CompressionFactory::GetLibraryId("lzma", BAL) == 52);
    REQUIRE(CompressionFactory::GetLibraryId("brotli", BEST) == 63);

    // Single-mode: preset forced to 2 (BALANCED) regardless of request.
    REQUIRE(CompressionFactory::GetLibraryId("snappy", FAST) == 72);
    REQUIRE(CompressionFactory::GetLibraryId("snappy", BEST) == 72);
    REQUIRE(CompressionFactory::GetLibraryId("blosc2", FAST) == 82);
    REQUIRE(CompressionFactory::GetLibraryId("blosc", BEST) == 82);  // alias

    // Lossy (base_id known regardless of LibPressio availability).
    REQUIRE(CompressionFactory::GetLibraryId("zfp", FAST) == 101);
    REQUIRE(CompressionFactory::GetLibraryId("zfp", BAL) == 102);
    REQUIRE(CompressionFactory::GetLibraryId("sz3", BAL) == 112);
    REQUIRE(CompressionFactory::GetLibraryId("fpzip", BEST) == 123);

    // Unknown library -> 0.
    REQUIRE(CompressionFactory::GetLibraryId("does-not-exist", BAL) == 0);

    // GPU compressors: base_ids 13-21. The nvcomp/ndzip rows are single-mode
    // (preset forced to 2); zfp-sycl/cusz are multi-mode. All are build-
    // independent -- the registry rows resolve even without the backend.
    REQUIRE(CompressionFactory::GetLibraryId("nvcomp-lz4", FAST) == 132);
    REQUIRE(CompressionFactory::GetLibraryId("nvcomp-lz4", BEST) == 132);
    REQUIRE(CompressionFactory::GetLibraryId("nvcomp-snappy", BAL) == 142);
    REQUIRE(CompressionFactory::GetLibraryId("nvcomp-zstd", BAL) == 152);
    REQUIRE(CompressionFactory::GetLibraryId("nvcomp-gdeflate", BAL) == 162);
    REQUIRE(CompressionFactory::GetLibraryId("nvcomp-deflate", BAL) == 172);
    REQUIRE(CompressionFactory::GetLibraryId("nvcomp-ans", BAL) == 182);

    // zfp-sycl: multi-mode lossy GPU (base_id 19), preset varies the bit rate.
    REQUIRE(CompressionFactory::GetLibraryId("zfp-sycl", FAST) == 191);
    REQUIRE(CompressionFactory::GetLibraryId("zfp-sycl", BAL) == 192);
    REQUIRE(CompressionFactory::GetLibraryId("zfp-sycl", BEST) == 193);

    // cusz: multi-mode lossy GPU (base_id 20), preset varies the error bound.
    REQUIRE(CompressionFactory::GetLibraryId("cusz", FAST) == 201);
    REQUIRE(CompressionFactory::GetLibraryId("cusz", BAL) == 202);
    REQUIRE(CompressionFactory::GetLibraryId("cusz", BEST) == 203);

    // ndzip: single-mode lossless GPU (base_id 21), preset forced to 2.
    REQUIRE(CompressionFactory::GetLibraryId("ndzip", FAST) == 212);
    REQUIRE(CompressionFactory::GetLibraryId("ndzip", BEST) == 212);

    // cuszp: multi-mode lossy GPU (base_id 22), preset varies the error bound.
    REQUIRE(CompressionFactory::GetLibraryId("cuszp", FAST) == 221);
    REQUIRE(CompressionFactory::GetLibraryId("cuszp", BAL) == 222);
    REQUIRE(CompressionFactory::GetLibraryId("cuszp", BEST) == 223);
  }

  PAGE_DIVIDE("ML library id -> name + preset (reverse)") {
    REQUIRE(CompressionFactory::GetLibraryInfo(11).first == "bzip2");
    REQUIRE(CompressionFactory::GetLibraryInfo(11).second ==
            CompressionPreset::FAST);
    REQUIRE(CompressionFactory::GetLibraryInfo(13).second ==
            CompressionPreset::BEST);
    REQUIRE(CompressionFactory::GetLibraryInfo(22).first == "zstd");
    REQUIRE(CompressionFactory::GetLibraryInfo(72).first == "snappy");
    REQUIRE(CompressionFactory::GetLibraryInfo(82).first == "blosc2");
    REQUIRE(CompressionFactory::GetLibraryInfo(102).first == "zfp");
    REQUIRE(CompressionFactory::GetLibraryInfo(112).first == "sz3");
    REQUIRE(CompressionFactory::GetLibraryInfo(122).first == "fpzip");
    // Unknown ids -> "unknown".
    REQUIRE(CompressionFactory::GetLibraryInfo(0).first == "unknown");
    REQUIRE(CompressionFactory::GetLibraryInfo(999).first == "unknown");
    // GPU compressors (build-independent).
    REQUIRE(CompressionFactory::GetLibraryInfo(132).first == "nvcomp-lz4");
    REQUIRE(CompressionFactory::GetLibraryInfo(142).first == "nvcomp-snappy");
    REQUIRE(CompressionFactory::GetLibraryInfo(152).first == "nvcomp-zstd");
    REQUIRE(CompressionFactory::GetLibraryInfo(162).first == "nvcomp-gdeflate");
    REQUIRE(CompressionFactory::GetLibraryInfo(172).first == "nvcomp-deflate");
    REQUIRE(CompressionFactory::GetLibraryInfo(182).first == "nvcomp-ans");
    REQUIRE(CompressionFactory::GetLibraryInfo(192).first == "zfp-sycl");
    REQUIRE(CompressionFactory::GetLibraryInfo(202).first == "cusz");
    REQUIRE(CompressionFactory::GetLibraryInfo(222).first == "cuszp");
    REQUIRE(CompressionFactory::GetLibraryInfo(201).second ==
            CompressionPreset::FAST);
    REQUIRE(CompressionFactory::GetLibraryInfo(212).first == "ndzip");
  }

  PAGE_DIVIDE("GetPreset constructs known CPU compressors (incl. alias)") {
    REQUIRE(CompressionFactory::GetPreset("bzip2") != nullptr);
    REQUIRE(CompressionFactory::GetPreset("zstd") != nullptr);
    REQUIRE(CompressionFactory::GetPreset("lz4") != nullptr);
    REQUIRE(CompressionFactory::GetPreset("zlib") != nullptr);
    REQUIRE(CompressionFactory::GetPreset("lzma") != nullptr);
    REQUIRE(CompressionFactory::GetPreset("brotli") != nullptr);
    REQUIRE(CompressionFactory::GetPreset("snappy") != nullptr);
    REQUIRE(CompressionFactory::GetPreset("blosc2") != nullptr);
    REQUIRE(CompressionFactory::GetPreset("blosc") != nullptr);  // alias
    REQUIRE(CompressionFactory::GetPreset("does-not-exist") == nullptr);
  }

  // Functional guard for the registry's make() column: every registered CPU
  // compressor must construct AND round-trip through the factory. This catches a
  // future registry row whose make is forgotten/null/throwing (the anti-footgun
  // goal of the registry). It does NOT prove make identity (a same-family swap
  // would still round-trip), which is enforced by inspection of the one-line row.
  PAGE_DIVIDE("factory round-trip over the CPU compressor set") {
    std::string payload;
    for (int i = 0; i < 256; ++i) {
      payload += "registry round-trip payload 0123456789 ";
    }
    const char *cpu_libs[] = {"bzip2", "zstd",   "lz4",    "zlib",
                              "lzma",  "brotli", "snappy", "blosc2"};
    for (const char *lib : cpu_libs) {
      auto comp = CompressionFactory::GetPreset(lib);
      REQUIRE(comp != nullptr);
      std::vector<char> cbuf(payload.size() * 2 + 4096);
      std::vector<char> dbuf(payload.size() + 64);
      size_t csize = cbuf.size();
      REQUIRE(comp->Compress(cbuf.data(), csize, payload.data(),
                             payload.size()));
      auto decomp = CompressionFactory::GetPreset(lib);
      REQUIRE(decomp != nullptr);
      size_t dsize = dbuf.size();
      REQUIRE(decomp->Decompress(dbuf.data(), dsize, cbuf.data(), csize));
      REQUIRE(payload == std::string(dbuf.data(), dsize));
    }
  }

#if CTP_ENABLE_NVCOMP
  // GPU compressor construction needs no device (NvComp touches the GPU only on
  // Compress/Decompress). Confirms every nvcomp registry row has a live make().
  PAGE_DIVIDE("GetPreset constructs GPU compressors (no device needed)") {
    const char *gpu_libs[] = {"nvcomp-lz4",      "nvcomp-snappy",
                              "nvcomp-zstd",     "nvcomp-gdeflate",
                              "nvcomp-deflate",  "nvcomp-ans"};
    for (const char *lib : gpu_libs) {
      INFO("nvcomp format: " << lib);
      REQUIRE(CompressionFactory::GetPreset(lib) != nullptr);
    }
  }
#endif
}

#if CTP_ENABLE_NVCOMP
// GPU compression via nvcomp. Mirrors the CPU compressor coverage above (a
// direct-class Compress/Decompress round-trip) and adds GPU-specific coverage:
// a per-format factory round-trip over every registered nvcomp codec, the
// device-pointer zero-copy path, and the library-id encoding round-trip.
TEST_CASE("TestNvCompGpu") {
  // nvcomp needs a real GPU. Skip gracefully where none is present (CI, laptops,
  // Docker without --gpus) so the suite stays green everywhere.
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    WARN("No CUDA device available; skipping nvcomp GPU compression test");
    return;
  }

  // Parity with the CPU tests: same direct-class round-trip, same small payload
  // and 1024-byte buffers, same REQUIRE-equality check.
  PAGE_DIVIDE("NvCompLZ4 (direct class)") {
    std::string raw = "Hello, World!";
    std::vector<char> compressed(1024);
    std::vector<char> decompressed(1024);
    ctp::NvComp nvcomp(ctp::NvCompAlgo::LZ4);
    size_t cmpr_size = 1024, raw_size = 1024;
    REQUIRE(nvcomp.Compress(compressed.data(), cmpr_size,
                            raw.data(), raw.size()));
    REQUIRE(nvcomp.Decompress(decompressed.data(), raw_size,
                              compressed.data(), cmpr_size));
    REQUIRE(raw == std::string(decompressed.data(), raw_size));
  }

  // Larger, compressible payload exercised through the factory (also proves the
  // factory dispatches "nvcomp-lz4").
  PAGE_DIVIDE("NvCompLZ4 (via factory)") {
    std::string raw;
    for (int i = 0; i < 4096; ++i) {
      raw += "The quick brown fox jumps over the lazy dog 0123456789 ";
    }

    auto compressor = ctp::CompressionFactory::GetPreset(
        "nvcomp-lz4", ctp::CompressionPreset::BALANCED);
    REQUIRE(compressor != nullptr);

    std::vector<char> compressed(raw.size() + raw.size() / 20 + 4096);
    std::vector<char> decompressed(raw.size() + 16);

    size_t cmpr_size = compressed.size();
    REQUIRE(compressor->Compress(compressed.data(), cmpr_size,
                                 raw.data(), raw.size()));
    REQUIRE(cmpr_size > 0);
    REQUIRE(cmpr_size < raw.size());  // large repetitive data must shrink

    auto decompressor = ctp::CompressionFactory::GetPreset("nvcomp-lz4");
    REQUIRE(decompressor != nullptr);
    size_t raw_size = decompressed.size();
    REQUIRE(decompressor->Decompress(decompressed.data(), raw_size,
                                     compressed.data(), cmpr_size));
    REQUIRE(raw == std::string(decompressed.data(), raw_size));
  }

  // Device-pointer (zero-copy) path: hand NvComp cudaMalloc'd pointers for both
  // input and output so it must use them in place rather than staging copies.
  PAGE_DIVIDE("NvCompLZ4 (device pointers, zero-copy)") {
    std::string raw;
    for (int i = 0; i < 1024; ++i) {
      raw += "GPU-resident data exercises the zero-copy path 9876543210 ";
    }

    void *d_raw = nullptr;
    REQUIRE(cudaMalloc(&d_raw, raw.size()) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_raw, raw.data(), raw.size(),
                       cudaMemcpyHostToDevice) == cudaSuccess);

    size_t cap = raw.size() + raw.size() / 20 + 4096;
    void *d_comp = nullptr;
    REQUIRE(cudaMalloc(&d_comp, cap) == cudaSuccess);

    ctp::NvComp nvcomp(ctp::NvCompAlgo::LZ4);
    size_t comp_size = cap;
    REQUIRE(nvcomp.Compress(d_comp, comp_size, d_raw, raw.size()));
    REQUIRE(comp_size < raw.size());

    void *d_decomp = nullptr;
    REQUIRE(cudaMalloc(&d_decomp, raw.size()) == cudaSuccess);
    size_t decomp_size = raw.size();
    REQUIRE(nvcomp.Decompress(d_decomp, decomp_size, d_comp, comp_size));
    REQUIRE(decomp_size == raw.size());

    std::vector<char> host_out(raw.size());
    REQUIRE(cudaMemcpy(host_out.data(), d_decomp, raw.size(),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE(raw == std::string(host_out.data(), raw.size()));

    cudaFree(d_raw);
    cudaFree(d_comp);
    cudaFree(d_decomp);
  }

  // Every general-purpose nvcomp format must round-trip through the factory on
  // real GPU data. Same shape as the CPU "factory round-trip" test above.
  PAGE_DIVIDE("all nvcomp formats (factory round-trip)") {
    std::string raw;
    for (int i = 0; i < 4096; ++i) {
      raw += "The quick brown fox jumps over the lazy dog 0123456789 ";
    }
    const char *gpu_libs[] = {"nvcomp-lz4",      "nvcomp-snappy",
                              "nvcomp-zstd",     "nvcomp-gdeflate",
                              "nvcomp-deflate",  "nvcomp-ans"};
    for (const char *lib : gpu_libs) {
      INFO("nvcomp format: " << lib);
      auto compressor = ctp::CompressionFactory::GetPreset(lib);
      REQUIRE(compressor != nullptr);

      std::vector<char> compressed(raw.size() + raw.size() / 20 + 4096);
      std::vector<char> decompressed(raw.size() + 16);

      size_t cmpr_size = compressed.size();
      REQUIRE(compressor->Compress(compressed.data(), cmpr_size,
                                   raw.data(), raw.size()));
      REQUIRE(cmpr_size > 0);
      REQUIRE(cmpr_size < raw.size());  // repetitive data must shrink

      auto decompressor = ctp::CompressionFactory::GetPreset(lib);
      REQUIRE(decompressor != nullptr);
      size_t raw_size = decompressed.size();
      REQUIRE(decompressor->Decompress(decompressed.data(), raw_size,
                                       compressed.data(), cmpr_size));
      REQUIRE(raw == std::string(decompressed.data(), raw_size));
    }
  }

  PAGE_DIVIDE("NvCompLZ4 library id round-trip") {
    int id = ctp::CompressionFactory::GetLibraryId(
        "nvcomp-lz4", ctp::CompressionPreset::BALANCED);
    REQUIRE(id == 132);  // base id 13, single-mode (BALANCED=2)
    REQUIRE(ctp::CompressionFactory::GetLibraryInfo(id).first == "nvcomp-lz4");
  }
}
#endif  // CTP_ENABLE_NVCOMP

#if CTP_ENABLE_CUSZ
// cuSZ is GPU error-bounded LOSSY float compression. Needs a real GPU; skip
// gracefully where none is present so the suite stays green everywhere.
TEST_CASE("TestCuszGpu") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    WARN("No CUDA device available; skipping cuSZ GPU compression test");
    return;
  }

  // 8192 floats of a smooth signal -- the regime cuSZ's Lorenzo predictor wins.
  const size_t n = 8192;
  std::vector<float> orig(n), deco(n, 0.0f);
  for (size_t i = 0; i < n; ++i) {
    orig[i] = std::sin(static_cast<float>(i) * 0.01f) * 100.0f;
  }
  const size_t raw_bytes = n * sizeof(float);

  // Device-pointer (zero-copy) round-trip with a fixed error bound.
  PAGE_DIVIDE("cusz (device pointers, BALANCED) round-trips within eb") {
    void *d_in = nullptr, *d_comp = nullptr, *d_out = nullptr;
    REQUIRE(cudaMalloc(&d_in, raw_bytes) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_comp, raw_bytes + 4096) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_out, raw_bytes) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_in, orig.data(), raw_bytes,
                       cudaMemcpyHostToDevice) == cudaSuccess);

    auto comp = ctp::CompressionFactory::GetPreset(
        "cusz", ctp::CompressionPreset::BALANCED);
    REQUIRE(comp != nullptr);
    size_t cmpr_size = raw_bytes + 4096;
    REQUIRE(comp->Compress(d_comp, cmpr_size, d_in, raw_bytes));
    REQUIRE(cmpr_size > 0);

    auto dcmp = ctp::CompressionFactory::GetPreset("cusz");
    REQUIRE(dcmp != nullptr);
    size_t deco_size = raw_bytes;
    REQUIRE(dcmp->Decompress(d_out, deco_size, d_comp, cmpr_size));
    REQUIRE(deco_size == raw_bytes);

    REQUIRE(cudaMemcpy(deco.data(), d_out, raw_bytes,
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    // Lossy within the BALANCED relative error bound (1e-3) on a [-100,100]
    // signal -> generous absolute slack.
    double max_err = 0.0;
    for (size_t i = 0; i < n; ++i) {
      max_err = std::max(max_err,
                         std::abs(static_cast<double>(orig[i] - deco[i])));
    }
    REQUIRE(max_err < 1.0);

    cudaFree(d_in);
    cudaFree(d_comp);
    cudaFree(d_out);
  }
}
#endif  // CTP_ENABLE_CUSZ

#if CTP_ENABLE_NDZIP
// ndzip is GPU high-throughput LOSSLESS float compression. Needs a real GPU;
// skip gracefully where none is present.
TEST_CASE("TestNdzipGpu") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    WARN("No CUDA device available; skipping ndzip GPU compression test");
    return;
  }

  const size_t n = 8192;
  std::vector<float> orig(n), deco(n, 0.0f);
  for (size_t i = 0; i < n; ++i) {
    orig[i] = std::sin(static_cast<float>(i) * 0.01f) * 100.0f;
  }
  const size_t raw_bytes = n * sizeof(float);

  // Lossless: device round-trip must reconstruct the input bit-exactly.
  PAGE_DIVIDE("ndzip (device pointers) round-trips bit-exactly") {
    void *d_in = nullptr, *d_comp = nullptr, *d_out = nullptr;
    REQUIRE(cudaMalloc(&d_in, raw_bytes) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_comp, raw_bytes * 2 + 4096) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_out, raw_bytes) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_in, orig.data(), raw_bytes,
                       cudaMemcpyHostToDevice) == cudaSuccess);

    auto comp = ctp::CompressionFactory::GetPreset("ndzip");
    REQUIRE(comp != nullptr);
    size_t cmpr_size = raw_bytes * 2 + 4096;
    REQUIRE(comp->Compress(d_comp, cmpr_size, d_in, raw_bytes));
    REQUIRE(cmpr_size > 0);

    auto dcmp = ctp::CompressionFactory::GetPreset("ndzip");
    REQUIRE(dcmp != nullptr);
    size_t deco_size = raw_bytes;
    REQUIRE(dcmp->Decompress(d_out, deco_size, d_comp, cmpr_size));
    REQUIRE(deco_size == raw_bytes);

    REQUIRE(cudaMemcpy(deco.data(), d_out, raw_bytes,
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE(orig == deco);  // lossless: exact
    cudaFree(d_in);
    cudaFree(d_comp);
    cudaFree(d_out);
  }
}
#endif  // CTP_ENABLE_NDZIP

#if CTP_ENABLE_CUSZP
// cuSZp is GPU ultra-fast error-bounded LOSSY float compression. Needs a real
// GPU; skip gracefully where none is present.
TEST_CASE("TestCuszpGpu") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    WARN("No CUDA device available; skipping cuSZp GPU compression test");
    return;
  }

  // Use a multi-block input (> 32768 elements). cuSZp only reports a complete
  // compressed size for multi-block inputs; smaller single-block inputs still
  // round-trip correctly but are stored uncompressed (see ctp::Cuszp), so a
  // multi-block size exercises the real, compressing path.
  const size_t n = 65536;
  std::vector<float> orig(n), deco(n, 0.0f);
  for (size_t i = 0; i < n; ++i) {
    orig[i] = std::sin(static_cast<float>(i) * 0.01f) * 100.0f;
  }
  const size_t raw_bytes = n * sizeof(float);

  // Device-pointer (zero-copy) round-trip within the absolute error bound.
  PAGE_DIVIDE("cuszp (device pointers, BALANCED) round-trips within eb") {
    void *d_in = nullptr, *d_comp = nullptr, *d_out = nullptr;
    REQUIRE(cudaMalloc(&d_in, raw_bytes) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_comp, raw_bytes + 4096) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_out, raw_bytes) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_in, orig.data(), raw_bytes,
                       cudaMemcpyHostToDevice) == cudaSuccess);

    auto comp = ctp::CompressionFactory::GetPreset(
        "cuszp", ctp::CompressionPreset::BALANCED);
    REQUIRE(comp != nullptr);
    size_t cmpr_size = raw_bytes + 4096;
    REQUIRE(comp->Compress(d_comp, cmpr_size, d_in, raw_bytes));
    REQUIRE(cmpr_size > 0);
    REQUIRE(cmpr_size < raw_bytes);  // multi-block -> actually compresses

    auto dcmp = ctp::CompressionFactory::GetPreset("cuszp");
    REQUIRE(dcmp != nullptr);
    size_t deco_size = raw_bytes;
    REQUIRE(dcmp->Decompress(d_out, deco_size, d_comp, cmpr_size));
    REQUIRE(deco_size == raw_bytes);

    REQUIRE(cudaMemcpy(deco.data(), d_out, raw_bytes,
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    // Lossy within the BALANCED absolute error bound (1e-3) -> generous slack.
    double max_err = 0.0;
    for (size_t i = 0; i < n; ++i) {
      max_err = std::max(max_err,
                         std::abs(static_cast<double>(orig[i] - deco[i])));
    }
    REQUIRE(max_err < 1.0);

    cudaFree(d_in);
    cudaFree(d_comp);
    cudaFree(d_out);
  }
}
#endif  // CTP_ENABLE_CUSZP
