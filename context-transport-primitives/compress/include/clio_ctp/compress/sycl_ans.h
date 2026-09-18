/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. (BSD-3-Clause; see repository headers.)
 */

#ifndef CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_SYCL_ANS_H_
#define CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_SYCL_ANS_H_

#if CTP_ENABLE_COMPRESS && CTP_ENABLE_SYCL_ANS

#include <cstddef>

#include "compress.h"

// ABI in src/compress/sycl_ans_kernels.cc (SYCL-compiled). Declared here so the
// header builds with the ORDINARY CTP toolchain -- the sycl_zfp / sycl_lz4
// delegation model.
extern "C" {
void *clio_sycl_ans_queue_create();
void clio_sycl_ans_queue_destroy(void *q);
int clio_sycl_ans_compress(void *q, const void *src, size_t src_size, void *dst,
                           size_t dst_cap, size_t *out);
int clio_sycl_ans_decompress(void *q, const void *src, size_t src_size,
                             void *dst, size_t dst_cap, size_t *out);
}

namespace ctp {

/**
 * GPU (SYCL) ANS compressor -- a portable static rANS entropy coder, the analog
 * of nvcomp's ANS codec (each vendor's ANS uses its own layout; this is a
 * self-contained, self-describing format).
 *
 * LOSSLESS and general-purpose. Unlike the LZ-family codecs (sycl-lz4 /
 * sycl-snappy) it captures byte-frequency skew, not repetition, so it wins on
 * entropy-limited data and is weaker on highly repetitive data -- the two are
 * complementary. Input is split into 32 KiB chunks, each carrying its own
 * normalized frequency table + rANS stream; one work-group codes each chunk.
 * Same delegation split as the other SYCL codecs; one queue per instance.
 */
class SyclAns : public Compressor {
 public:
  SyclAns() : queue_(clio_sycl_ans_queue_create()) {}
  ~SyclAns() override {
    if (queue_) clio_sycl_ans_queue_destroy(queue_);
  }
  SyclAns(const SyclAns &) = delete;
  SyclAns &operator=(const SyclAns &) = delete;

  bool Compress(void *output, size_t &output_size, void *input,
                size_t input_size) override {
    if (!queue_) return false;
    size_t out = 0;
    if (!clio_sycl_ans_compress(queue_, input, input_size, output, output_size,
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
    if (!clio_sycl_ans_decompress(queue_, input, input_size, output,
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

#endif  // CTP_ENABLE_COMPRESS && CTP_ENABLE_SYCL_ANS
#endif  // CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_SYCL_ANS_H_
