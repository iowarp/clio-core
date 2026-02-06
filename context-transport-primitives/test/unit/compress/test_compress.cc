/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
* Distributed under BSD 3-Clause license.                                   *
* Copyright by The HDF Group.                                               *
* Copyright by the Illinois Institute of Technology.                        *
* All rights reserved.                                                      *
*                                                                           *
* This file is part of Hermes. The full Hermes copyright notice, including  *
* terms governing use, modification, and redistribution, is contained in    *
* the COPYING file, which can be found at the top directory. If you do not  *
* have access to the file, you may request a copy from help@hdfgroup.org.   *
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "basic_test.h"
#include "hermes_shm/compress/compress_factory.h"
#include <utility>

TEST_CASE("TestCompress") {
  std::string raw = "Hello, World!";
  std::vector<char> compressed(1024);
  std::vector<char> decompressed(1024);

  PAGE_DIVIDE("BZIP2") {
    hshm::Bzip2 bzip;
    size_t cmpr_size = 1024, raw_size = 1024;
    bzip.Compress(compressed.data(), cmpr_size,
                  raw.data(), raw.size());
    bzip.Decompress(decompressed.data(), raw_size,
                    compressed.data(), cmpr_size);
    REQUIRE(raw == std::string(decompressed.data(), raw_size));
  }

  PAGE_DIVIDE("LZO") {
    hshm::Lzo lzo;
    size_t cmpr_size = 1024, raw_size = 1024;
    lzo.Compress(compressed.data(), cmpr_size,
                 raw.data(), raw.size());
    lzo.Decompress(decompressed.data(), raw_size,
                   compressed.data(), cmpr_size);
    REQUIRE(raw == std::string(decompressed.data(), raw_size));
  }

  PAGE_DIVIDE("Zstd") {
    hshm::Zstd zstd;
    size_t cmpr_size = 1024, raw_size = 1024;
    zstd.Compress(compressed.data(), cmpr_size,
                  raw.data(), raw.size());
    zstd.Decompress(decompressed.data(), raw_size,
                    compressed.data(), cmpr_size);
    REQUIRE(raw == std::string(decompressed.data(), raw_size));
  }

  PAGE_DIVIDE("LZ4") {
    hshm::Lz4 lz4;
    size_t cmpr_size = 1024, raw_size = 1024;
    lz4.Compress(compressed.data(), cmpr_size,
                 raw.data(), raw.size());
    lz4.Decompress(decompressed.data(), raw_size,
                   compressed.data(), cmpr_size);
    REQUIRE(raw == std::string(decompressed.data(), raw_size));
  }

  PAGE_DIVIDE("Zlib") {
    hshm::Zlib zlib;
    size_t cmpr_size = 1024, raw_size = 1024;
    zlib.Compress(compressed.data(), cmpr_size,
                  raw.data(), raw.size());
    zlib.Decompress(decompressed.data(), raw_size,
                    compressed.data(), cmpr_size);
    REQUIRE(raw == std::string(decompressed.data(), raw_size));
  }

  PAGE_DIVIDE("Lzma") {
    hshm::Lzma lzma;
    size_t cmpr_size = 1024, raw_size = 1024;
    lzma.Compress(compressed.data(), cmpr_size,
                  raw.data(), raw.size());
    lzma.Decompress(decompressed.data(), raw_size,
                    compressed.data(), cmpr_size);
    REQUIRE(raw == std::string(decompressed.data(), raw_size));
  }

  PAGE_DIVIDE("Brotli") {
    hshm::Brotli brotli;
    size_t cmpr_size = 1024, raw_size = 1024;
    brotli.Compress(compressed.data(), cmpr_size,
                    raw.data(), raw.size());
    brotli.Decompress(decompressed.data(), raw_size,
                      compressed.data(), cmpr_size);
    REQUIRE(raw == std::string(decompressed.data(), raw_size));
  }

  PAGE_DIVIDE("Snappy") {
    hshm::Snappy snappy;
    size_t cmpr_size = 1024, raw_size = 1024;
    snappy.Compress(compressed.data(), cmpr_size,
                    raw.data(), raw.size());
    snappy.Decompress(decompressed.data(), raw_size,
                      compressed.data(), cmpr_size);
    REQUIRE(raw == std::string(decompressed.data(), raw_size));
  }

  PAGE_DIVIDE("Blosc2") {
    hshm::Blosc blosc;
    size_t cmpr_size = 1024, raw_size = 1024;
    blosc.Compress(compressed.data(), cmpr_size,
                   raw.data(), raw.size());
    blosc.Decompress(decompressed.data(), raw_size,
                     compressed.data(), cmpr_size);
    REQUIRE(raw == std::string(decompressed.data(), raw_size));
  }
}
