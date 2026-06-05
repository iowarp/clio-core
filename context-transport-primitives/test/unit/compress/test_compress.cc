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
    // Out-of-range falls back to the historical default. (Registry rows are
    // build-independent, so the GPU names above resolve even without nvcomp.)
    REQUIRE(CompressionFactory::NameForWireId(-1) == "zstd");
    REQUIRE(CompressionFactory::NameForWireId(17) == "zstd");
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

    // GPU compressors: single-mode (preset forced to 2), base_ids 13-19. These
    // are build-independent -- the registry rows resolve even without nvcomp.
    REQUIRE(CompressionFactory::GetLibraryId("nvcomp-lz4", FAST) == 132);
    REQUIRE(CompressionFactory::GetLibraryId("nvcomp-lz4", BEST) == 132);
    REQUIRE(CompressionFactory::GetLibraryId("nvcomp-snappy", BAL) == 142);
    REQUIRE(CompressionFactory::GetLibraryId("nvcomp-zstd", BAL) == 152);
    REQUIRE(CompressionFactory::GetLibraryId("nvcomp-gdeflate", BAL) == 162);
    REQUIRE(CompressionFactory::GetLibraryId("nvcomp-deflate", BAL) == 172);
    REQUIRE(CompressionFactory::GetLibraryId("nvcomp-ans", BAL) == 182);
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
