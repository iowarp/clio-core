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

#if CTP_ENABLE_NVCOMP
// GPU compression via nvcomp. Mirrors the CPU compressor coverage above (a
// direct-class Compress/Decompress round-trip) and adds GPU-specific coverage:
// a factory round-trip (validates the "nvcomp-lz4" registration) and the
// library-id encoding round-trip.
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

  PAGE_DIVIDE("NvCompLZ4 library id round-trip") {
    int id = ctp::CompressionFactory::GetLibraryId(
        "nvcomp-lz4", ctp::CompressionPreset::BALANCED);
    REQUIRE(id == 132);  // base id 13, single-mode (BALANCED=2)
    REQUIRE(ctp::CompressionFactory::GetLibraryInfo(id).first == "nvcomp-lz4");
  }
}
#endif  // CTP_ENABLE_NVCOMP
