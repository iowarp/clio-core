/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Distributed under BSD 3-Clause license.                                   *
 * Copyright by The HDF Group.                                               *
 * Copyright by the Illinois Institute of Technology.                        *
 * All rights reserved.                                                      *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef HSHM_SHM_INCLUDE_HSHM_SHM_COMPRESS_LOSSLESS_MODES_H_
#define HSHM_SHM_INCLUDE_HSHM_SHM_COMPRESS_LOSSLESS_MODES_H_

#if HSHM_ENABLE_COMPRESS

#include <zlib.h>
#include <bzlib.h>
#include <zstd.h>
#include <lz4.h>
#include <lz4hc.h>
#include <lzma.h>
#include <brotli/encode.h>
#include <brotli/decode.h>

#include "compress.h"

namespace hshm {

/**
 * Compression mode for lossless compressors.
 * Maps to different compression levels (speed vs ratio tradeoff).
 */
enum class LosslessMode {
  FAST,      // Fast compression, lower ratio (level 1-3)
  BALANCED,  // Balanced speed and ratio (level 5-6)
  BEST       // Best compression, slower (level 9+)
};

/**
 * ZLIB wrapper with compression level support.
 * Levels: 1 (fast) to 9 (best compression)
 */
class ZlibWithModes : public Compressor {
 private:
  int level_;

 public:
  explicit ZlibWithModes(LosslessMode mode) {
    switch (mode) {
      case LosslessMode::FAST:
        level_ = 1;  // Fast compression
        break;
      case LosslessMode::BALANCED:
        level_ = 6;  // Default/balanced
        break;
      case LosslessMode::BEST:
        level_ = 9;  // Maximum compression
        break;
    }
  }

  bool Compress(void *output, size_t &output_size, void *input,
                size_t input_size) override {
    z_stream stream;
    stream.zalloc = Z_NULL;
    stream.zfree = Z_NULL;
    stream.opaque = Z_NULL;

    if (deflateInit(&stream, level_) != Z_OK) {
      return false;
    }

    stream.avail_in = input_size;
    stream.next_in = reinterpret_cast<Bytef *>(input);
    stream.avail_out = output_size;
    stream.next_out = reinterpret_cast<Bytef *>(output);

    if (deflate(&stream, Z_FINISH) != Z_STREAM_END) {
      deflateEnd(&stream);
      return false;
    }

    output_size = stream.total_out;
    deflateEnd(&stream);
    return true;
  }

  bool Decompress(void *output, size_t &output_size, void *input,
                  size_t input_size) override {
    z_stream stream;
    stream.zalloc = Z_NULL;
    stream.zfree = Z_NULL;
    stream.opaque = Z_NULL;

    if (inflateInit(&stream) != Z_OK) {
      return false;
    }

    stream.avail_in = input_size;
    stream.next_in = reinterpret_cast<Bytef *>(input);
    stream.avail_out = output_size;
    stream.next_out = reinterpret_cast<Bytef *>(output);

    if (inflate(&stream, Z_FINISH) != Z_STREAM_END) {
      inflateEnd(&stream);
      return false;
    }

    output_size = stream.total_out;
    inflateEnd(&stream);
    return true;
  }
};

/**
 * BZIP2 wrapper with compression level support.
 * Levels: 1 (fast) to 9 (best compression)
 */
class Bzip2WithModes : public Compressor {
 private:
  int level_;
  int verbosity_ = 0;
  int work_factor_ = 30;

 public:
  explicit Bzip2WithModes(LosslessMode mode) {
    switch (mode) {
      case LosslessMode::FAST:
        level_ = 1;  // Fast compression
        break;
      case LosslessMode::BALANCED:
        level_ = 6;  // Balanced
        break;
      case LosslessMode::BEST:
        level_ = 9;  // Maximum compression
        break;
    }
  }

  bool Compress(void *output, size_t &output_size, void *input,
                size_t input_size) override {
    unsigned int output_size_int = output_size;
    int ret = BZ2_bzBuffToBuffCompress((char *)output, &output_size_int,
                                       (char *)input, input_size, level_,
                                       verbosity_, work_factor_);
    output_size = output_size_int;
    return ret == BZ_OK;
  }

  bool Decompress(void *output, size_t &output_size, void *input,
                  size_t input_size) override {
    unsigned int output_size_int = output_size;
    int small = 0;
    int ret = BZ2_bzBuffToBuffDecompress((char *)output, &output_size_int,
                                         (char *)input, input_size, small,
                                         verbosity_);
    output_size = output_size_int;
    return ret == BZ_OK;
  }
};

/**
 * ZSTD wrapper with compression level support.
 * Levels: 1 (fast) to 22 (best compression), default is 3
 */
class ZstdWithModes : public Compressor {
 private:
  int level_;

 public:
  explicit ZstdWithModes(LosslessMode mode) {
    switch (mode) {
      case LosslessMode::FAST:
        level_ = 1;   // Fast compression
        break;
      case LosslessMode::BALANCED:
        level_ = 3;   // Default level
        break;
      case LosslessMode::BEST:
        level_ = 19;  // High compression (not max to avoid extreme slowdown)
        break;
    }
  }

  bool Compress(void *output, size_t &output_size, void *input,
                size_t input_size) override {
    if (ZSTD_compressBound(input_size) > output_size) {
      return false;
    }
    output_size = ZSTD_compress(output, output_size, input, input_size, level_);
    return output_size != 0;
  }

  bool Decompress(void *output, size_t &output_size, void *input,
                  size_t input_size) override {
    output_size = ZSTD_decompress(output, output_size, input, input_size);
    return output_size != 0;
  }
};

/**
 * LZ4 wrapper with compression mode support.
 * Uses LZ4_compress_default (fast) or LZ4_compress_HC (high compression)
 */
class Lz4WithModes : public Compressor {
 private:
  LosslessMode mode_;

 public:
  explicit Lz4WithModes(LosslessMode mode) : mode_(mode) {}

  bool Compress(void *output, size_t &output_size, void *input,
                size_t input_size) override {
    if ((size_t)LZ4_compressBound(input_size) > output_size) {
      return false;
    }

    int result = 0;
    switch (mode_) {
      case LosslessMode::FAST:
        // Fast mode - use default LZ4
        result = LZ4_compress_default((char *)input, (char *)output,
                                      (int)input_size, (int)output_size);
        break;
      case LosslessMode::BALANCED:
        // Balanced - use HC with moderate level
        result = LZ4_compress_HC((char *)input, (char *)output,
                                 (int)input_size, (int)output_size, 6);
        break;
      case LosslessMode::BEST:
        // Best - use HC with high level
        result = LZ4_compress_HC((char *)input, (char *)output,
                                 (int)input_size, (int)output_size, 12);
        break;
    }

    output_size = result;
    return result != 0;
  }

  bool Decompress(void *output, size_t &output_size, void *input,
                  size_t input_size) override {
    output_size = LZ4_decompress_safe((char *)input, (char *)output,
                                      (int)input_size, (int)output_size);
    return output_size != 0;
  }
};

/**
 * LZMA wrapper with compression level support.
 * Levels: 0 (fast) to 9 (best compression)
 */
class LzmaWithModes : public Compressor {
 private:
  uint32_t preset_;

 public:
  explicit LzmaWithModes(LosslessMode mode) {
    switch (mode) {
      case LosslessMode::FAST:
        preset_ = 0;  // Fast compression
        break;
      case LosslessMode::BALANCED:
        preset_ = 6;  // Default level
        break;
      case LosslessMode::BEST:
        preset_ = 9;  // Maximum compression
        break;
    }
  }

  bool Compress(void *output, size_t &output_size, void *input,
                size_t input_size) override {
    lzma_stream stream = LZMA_STREAM_INIT;
    lzma_ret ret = lzma_easy_encoder(&stream, preset_, LZMA_CHECK_CRC64);
    if (ret != LZMA_OK) {
      return false;
    }

    stream.next_in = (const uint8_t *)input;
    stream.avail_in = input_size;
    stream.next_out = (uint8_t *)output;
    stream.avail_out = output_size;

    ret = lzma_code(&stream, LZMA_FINISH);
    if (ret != LZMA_STREAM_END) {
      lzma_end(&stream);
      return false;
    }

    output_size = stream.total_out;
    lzma_end(&stream);
    return true;
  }

  bool Decompress(void *output, size_t &output_size, void *input,
                  size_t input_size) override {
    lzma_stream stream = LZMA_STREAM_INIT;
    lzma_ret ret = lzma_stream_decoder(&stream, UINT64_MAX, 0);
    if (ret != LZMA_OK) {
      return false;
    }

    stream.next_in = (const uint8_t *)input;
    stream.avail_in = input_size;
    stream.next_out = (uint8_t *)output;
    stream.avail_out = output_size;

    ret = lzma_code(&stream, LZMA_FINISH);
    if (ret != LZMA_STREAM_END) {
      lzma_end(&stream);
      return false;
    }

    output_size = stream.total_out;
    lzma_end(&stream);
    return true;
  }
};

/**
 * Brotli wrapper with compression quality support.
 * Quality levels: 0 (fast) to 11 (best compression)
 */
class BrotliWithModes : public Compressor {
 private:
  int quality_;

 public:
  explicit BrotliWithModes(LosslessMode mode) {
    switch (mode) {
      case LosslessMode::FAST:
        quality_ = 1;   // Fast compression
        break;
      case LosslessMode::BALANCED:
        quality_ = 6;   // Balanced
        break;
      case LosslessMode::BEST:
        quality_ = 11;  // Maximum compression
        break;
    }
  }

  bool Compress(void *output, size_t &output_size, void *input,
                size_t input_size) override {
    size_t encoded_size = output_size;
    int result = BrotliEncoderCompress(
        quality_, BROTLI_DEFAULT_WINDOW, BROTLI_DEFAULT_MODE,
        input_size, (const uint8_t *)input, &encoded_size, (uint8_t *)output);

    if (result == BROTLI_TRUE) {
      output_size = encoded_size;
      return true;
    }
    return false;
  }

  bool Decompress(void *output, size_t &output_size, void *input,
                  size_t input_size) override {
    size_t decoded_size = output_size;
    BrotliDecoderResult result = BrotliDecoderDecompress(
        input_size, (const uint8_t *)input, &decoded_size, (uint8_t *)output);

    if (result == BROTLI_DECODER_RESULT_SUCCESS) {
      output_size = decoded_size;
      return true;
    }
    return false;
  }
};

}  // namespace hshm

#endif  // HSHM_ENABLE_COMPRESS

#endif  // HSHM_SHM_INCLUDE_HSHM_SHM_COMPRESS_LOSSLESS_MODES_H_
