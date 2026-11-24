/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. (BSD-3-Clause; see repository headers.)
 */

#ifndef CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_SYCL_LZ4_H_
#define CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_SYCL_LZ4_H_

#if CTP_ENABLE_COMPRESS && CTP_ENABLE_SYCL_LZ4

#include <cstddef>
#include <vector>

#include "compress.h"

// ABI implemented in src/compress/sycl_lz4_kernels.cc, compiled by a SYCL
// compiler (DPC++/AdaptiveCpp) and linked as clio_ctp_sycl_lz4. Declared here
// so this header compiles with the ORDINARY CTP toolchain -- exactly the
// delegation model sycl_zfp uses for its SYCL-built libzfp.
extern "C" {
void *clio_sycl_lz4_queue_create();
void clio_sycl_lz4_queue_destroy(void *q);
int clio_sycl_lz4_compress(void *q, const void *src, size_t src_size, void *dst,
                           size_t dst_cap, size_t *out);
int clio_sycl_lz4_decompress(void *q, const void *src, size_t src_size,
                             void *dst, size_t dst_cap, size_t *out);
int clio_sycl_lz4_compress_batch(void *q, size_t n, const void *const *srcs,
                                 const size_t *src_sizes, void *const *dsts,
                                 const size_t *dst_caps, size_t *out_sizes,
                                 int *oks);
int clio_sycl_lz4_decompress_batch(void *q, size_t n, const void *const *srcs,
                                   const size_t *src_sizes, void *const *dsts,
                                   const size_t *dst_caps, size_t *out_sizes,
                                   int *oks);
}

namespace ctp {

/**
 * GPU (SYCL) LZ4 compressor -- the portable analog of NvComp's LZ4 codec.
 *
 * A general-purpose, byte-oriented LOSSLESS codec that fits the byte-stream
 * Compressor interface. The device work (a chunked LZ4 encode/decode, one
 * work-item per 64 KiB chunk) lives in a SYCL-compiled kernels library; this
 * wrapper only marshals the synchronous ctp::Compressor calls across the
 * extern "C" boundary, so it builds with the ordinary toolchain and merely
 * *links* the SYCL object -- the same split sycl_zfp uses.
 *
 * The compressed stream is self-describing (its own chunked container wrapping
 * standard LZ4 blocks), so Decompress needs no side information. Buffers may be
 * host or device USM independently; the kernels library detects each pointer
 * and stages the minimum, matching NvComp's adaptive behavior.
 *
 * Unlike NvComp, no CUDA stream is taken: a SYCL queue is created once per
 * instance (device selector) and reused for every call, which is the SYCL
 * equivalent of "streams created once when the owning module is created."
 */
class SyclLz4 : public Compressor {
 public:
  SyclLz4() : queue_(clio_sycl_lz4_queue_create()) {}
  ~SyclLz4() override {
    if (queue_) clio_sycl_lz4_queue_destroy(queue_);
  }
  SyclLz4(const SyclLz4 &) = delete;
  SyclLz4 &operator=(const SyclLz4 &) = delete;

  bool Compress(void *output, size_t &output_size, void *input,
                size_t input_size) override {
    if (!queue_) return false;
    size_t out = 0;
    if (!clio_sycl_lz4_compress(queue_, input, input_size, output, output_size,
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
    if (!clio_sycl_lz4_decompress(queue_, input, input_size, output,
                                  output_size, &out)) {
      return false;
    }
    output_size = out;
    return true;
  }

  /** ONE launch per phase for the whole batch (nvcomp's batched model). */
  bool CompressBatch(CompressJob *jobs, size_t n) override {
    return RunBatch(jobs, n, /*compress=*/true);
  }
  bool DecompressBatch(CompressJob *jobs, size_t n) override {
    return RunBatch(jobs, n, /*compress=*/false);
  }

 private:
  bool RunBatch(CompressJob *jobs, size_t n, bool compress) {
    if (n == 0) return true;
    if (!queue_) {
      for (size_t i = 0; i < n; ++i) { jobs[i].ok = false; jobs[i].out_size = 0; }
      return false;
    }
    std::vector<const void *> src(n);
    std::vector<void *> dst(n);
    std::vector<size_t> ssz(n), cap(n), out(n, 0);
    std::vector<int> ok(n, 0);
    for (size_t i = 0; i < n; ++i) {
      src[i] = jobs[i].src;
      ssz[i] = jobs[i].src_size;
      dst[i] = jobs[i].dst;
      cap[i] = jobs[i].dst_capacity;
    }
    auto fn = compress ? &clio_sycl_lz4_compress_batch
                       : &clio_sycl_lz4_decompress_batch;
    fn(queue_, n, src.data(), ssz.data(), dst.data(), cap.data(), out.data(),
       ok.data());
    bool all = true;
    for (size_t i = 0; i < n; ++i) {
      jobs[i].ok = ok[i] != 0;
      jobs[i].out_size = out[i];
      all &= jobs[i].ok;
    }
    return all;
  }

  void *queue_;  // opaque QueueBox* owned by the kernels library
};

}  // namespace ctp

#endif  // CTP_ENABLE_COMPRESS && CTP_ENABLE_SYCL_LZ4
#endif  // CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_SYCL_LZ4_H_
