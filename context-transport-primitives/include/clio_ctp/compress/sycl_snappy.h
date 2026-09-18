/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. (BSD-3-Clause; see repository headers.)
 */

#ifndef CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_SYCL_SNAPPY_H_
#define CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_SYCL_SNAPPY_H_

#if CTP_ENABLE_COMPRESS && CTP_ENABLE_SYCL_SNAPPY

#include <cstddef>

#include "compress.h"

// ABI in src/compress/sycl_snappy_kernels.cc (SYCL-compiled). Declared here so
// this header builds with the ORDINARY CTP toolchain -- the sycl_zfp / sycl_lz4
// delegation model.
extern "C" {
void *clio_sycl_snappy_queue_create();
void clio_sycl_snappy_queue_destroy(void *q);
int clio_sycl_snappy_compress(void *q, const void *src, size_t src_size,
                              void *dst, size_t dst_cap, size_t *out);
int clio_sycl_snappy_decompress(void *q, const void *src, size_t src_size,
                                void *dst, size_t dst_cap, size_t *out);
}

namespace ctp {

/**
 * GPU (SYCL) Snappy compressor -- the portable analog of nvcomp's Snappy codec.
 *
 * A general-purpose, byte-oriented LOSSLESS codec. Input is split into 4 KiB
 * chunks, each a standard Snappy block (validated against reference libsnappy),
 * wrapped in a self-describing container; one work-group decodes/encodes each
 * chunk. Same delegation split as ctp::SyclLz4: the device work is in a
 * SYCL-compiled library, this wrapper only marshals the ctp::Compressor calls,
 * and one SYCL queue is created per instance and reused.
 */
class SyclSnappy : public Compressor {
 public:
  SyclSnappy() : queue_(clio_sycl_snappy_queue_create()) {}
  ~SyclSnappy() override {
    if (queue_) clio_sycl_snappy_queue_destroy(queue_);
  }
  SyclSnappy(const SyclSnappy &) = delete;
  SyclSnappy &operator=(const SyclSnappy &) = delete;

  bool Compress(void *output, size_t &output_size, void *input,
                size_t input_size) override {
    if (!queue_) return false;
    size_t out = 0;
    if (!clio_sycl_snappy_compress(queue_, input, input_size, output,
                                   output_size, &out)) {
      return false;
    }
    output_size = out;
    return true;
  }

  bool Decompress(void *output, size_t &output_size, void *input,
                  size_t input_size) override {
    if (!queue_) return false;
    size_t out = 0;
    if (!clio_sycl_snappy_decompress(queue_, input, input_size, output,
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

#endif  // CTP_ENABLE_COMPRESS && CTP_ENABLE_SYCL_SNAPPY
#endif  // CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_SYCL_SNAPPY_H_
