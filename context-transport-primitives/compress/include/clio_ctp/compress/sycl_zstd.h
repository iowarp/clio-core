/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. (BSD-3-Clause; see repository headers.)
 */

#ifndef CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_SYCL_ZSTD_H_
#define CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_SYCL_ZSTD_H_

#if CTP_ENABLE_COMPRESS && CTP_ENABLE_SYCL_ZSTD

#include <cstddef>

#include "compress.h"

// ABI in src/compress/sycl_zstd_kernels.cc (SYCL-compiled). Declared here so the
// header builds with the ORDINARY CTP toolchain -- the sycl_zfp / sycl_lz4
// delegation model.
extern "C" {
void *clio_sycl_zstd_queue_create();
void clio_sycl_zstd_queue_destroy(void *q);
int clio_sycl_zstd_compress(void *q, const void *src, size_t src_size, void *dst,
                            size_t dst_cap, size_t *out);
int clio_sycl_zstd_decompress(void *q, const void *src, size_t src_size,
                              void *dst, size_t dst_cap, size_t *out);
}

namespace ctp {

/**
 * GPU (SYCL) Zstd-STYLE compressor -- a two-stage LZ77 + rANS codec.
 *
 * LOSSLESS. This combines the two paradigms Zstd is built on: an LZ77 stage
 * (like sycl-lz4) emits literals + (litLen, matchLen, offset) sequences, then a
 * static rANS entropy stage (like sycl-ans) codes the literals -- so it beats a
 * pure LZ codec on ratio by entropy-coding the residual literals.
 *
 * IMPORTANT -- NOT libzstd/nvcomp-zstd COMPATIBLE. Standard Zstd uses FSE +
 * Huffman with a specific frame/block format; this is a self-contained,
 * self-describing analog of that architecture, not a byte-compatible Zstd. It
 * interoperates only with itself. Same delegation split as the other SYCL
 * codecs; one queue per instance.
 */
class SyclZstd : public Compressor {
 public:
  SyclZstd() : queue_(clio_sycl_zstd_queue_create()) {}
  ~SyclZstd() override {
    if (queue_) clio_sycl_zstd_queue_destroy(queue_);
  }
  SyclZstd(const SyclZstd &) = delete;
  SyclZstd &operator=(const SyclZstd &) = delete;

  bool Compress(void *output, size_t &output_size, void *input,
                size_t input_size) override {
    if (!queue_) return false;
    size_t out = 0;
    if (!clio_sycl_zstd_compress(queue_, input, input_size, output, output_size,
                                 &out)) {
      return false;
    }
    output_size = out;
    return true;
  }

  bool Decompress(void *output, size_t &output_size, void *input,
                  size_t input_size) override {
    if (!queue_) return false;
    size_t out = 0;
    if (!clio_sycl_zstd_decompress(queue_, input, input_size, output,
                                   output_size, &out)) {
      return false;
    }
    output_size = out;
    return true;
  }

 private:
  void *queue_;
};

}  // namespace ctp

#endif  // CTP_ENABLE_COMPRESS && CTP_ENABLE_SYCL_ZSTD
#endif  // CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_SYCL_ZSTD_H_
