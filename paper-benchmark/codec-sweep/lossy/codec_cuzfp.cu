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

/**
 * @file codec_cuzfp.cu
 * @brief cuZFP (LLNL/zfp 1.0.1, CUDA execution policy) for lossy_sweep.
 *
 * zfp's CUDA backend supports only FIXED-RATE mode, which cannot take an
 * error bound. To configure it as well as it can be, PrepareCompress searches
 * (untimed) for the smallest rate whose maximum point-wise error on THIS
 * field stays within the bound: a binary search over the integer bits per
 * zfp block (rate granularity 1/4^d bit per value). The timed calls then run
 * at that rate, recorded in the CSV note. This per-field oracle tuning is an
 * advantage no other codec gets. zfp detects device pointers and works on
 * them in place.
 */
#include <zfp.h>

#include <cstdio>
#include <string>

#include "lossy_codec.h"

namespace {

/** Max |a - b| over n floats into *max_bits (as uint bits). */
__global__ void MaxErrKernel(const float *a, const float *b, size_t n,
                             unsigned int *max_bits) {
  float m = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x) {
    const float e = fabsf(a[i] - b[i]);
    m = (e == e) ? fmaxf(m, e) : __int_as_float(0x7f800000);  // NaN -> inf
  }
  atomicMax(max_bits, __float_as_uint(m));
}

/** @brief zfp fixed-rate on the GPU, rate tuned per field and bound. */
class CuzfpCodec : public LossyCodec {
 public:
  ~CuzfpCodec() override {
    if (d_tmp_cmp_) cudaFree(d_tmp_cmp_);
    if (d_tmp_dec_) cudaFree(d_tmp_dec_);
    if (d_max_) cudaFree(d_max_);
  }
  std::string Name() const override { return "cuzfp"; }
  /** Fixed-rate at 32 bits/value plus zfp's own slack. */
  size_t MaxCompressedBytes(const Field &f) override {
    return f.n() * sizeof(float) + (f.n() / 16 + 1) * 8 + (1 << 16);
  }
  std::string Note() const override { return note_; }

  bool PrepareCompress(const float *d_in, const Field &f, double eb,
                       cudaStream_t) override {
    if (!Scratch(f)) return false;
    const int d = f.dims();
    const unsigned block = 1u << (2 * d);  // 4^d values per block
    unsigned lo = 1, hi = 32 * block;      // bits per block
    if (MaxErr(d_in, f, Rate(hi, block)) > eb) {
      note_ = "no rate meets bound";
      return false;
    }
    while (lo < hi) {  // smallest maxbits with max error <= eb
      const unsigned mid = lo + (hi - lo) / 2;
      if (MaxErr(d_in, f, Rate(mid, block)) <= eb) hi = mid; else lo = mid + 1;
    }
    rate_ = Rate(hi, block);
    char buf[64];
    std::snprintf(buf, sizeof(buf), "rate=%.4f bits/value", rate_);
    note_ = buf;
    return true;
  }
  size_t Compress(const float *d_in, const Field &f, double, uint8_t *d_out,
                  size_t cap, cudaStream_t) override {
    return Run(true, const_cast<float *>(d_in), f, rate_, d_out, cap);
  }
  bool Decompress(const uint8_t *d_cmp, size_t bytes, const Field &f, double,
                  float *d_out, cudaStream_t) override {
    return Run(false, d_out, f, rate_, const_cast<uint8_t *>(d_cmp), bytes) > 0;
  }

 private:
  static double Rate(unsigned maxbits, unsigned block) {
    return double(maxbits) / block;
  }
  /** @return a zfp field over p with f's shape */
  static zfp_field *MakeField(float *p, const Field &f) {
    if (f.dims() == 3) return zfp_field_3d(p, zfp_type_float, f.x, f.y, f.z);
    if (f.dims() == 2) return zfp_field_2d(p, zfp_type_float, f.x, f.y);
    return zfp_field_1d(p, zfp_type_float, f.x);
  }
  /**
   * One zfp call on device memory at a fixed rate.
   * @param compress true: data -> buf; false: buf -> data
   * @return bytes compressed / consumed, 0 on failure
   */
  static size_t Run(bool compress, float *data, const Field &f, double rate,
                    uint8_t *buf, size_t cap) {
    zfp_field *field = MakeField(data, f);
    zfp_stream *zs = zfp_stream_open(nullptr);
    zfp_stream_set_rate(zs, rate, zfp_type_float, f.dims(), zfp_false);
    bitstream *bs = stream_open(buf, cap);
    zfp_stream_set_bit_stream(zs, bs);
    zfp_stream_rewind(zs);
    size_t r = 0;
    if (zfp_stream_set_execution(zs, zfp_exec_cuda)) {
      r = compress ? zfp_compress(zs, field) : zfp_decompress(zs, field);
    }
    stream_close(bs);
    zfp_stream_close(zs);
    zfp_field_free(field);
    return r;
  }
  /** @return max error of a round trip of d_in at rate (inf on failure) */
  double MaxErr(const float *d_in, const Field &f, double rate) {
    const size_t bytes = Run(true, const_cast<float *>(d_in), f, rate,
                             d_tmp_cmp_, tmp_cap_);
    if (!bytes || !Run(false, d_tmp_dec_, f, rate, d_tmp_cmp_, bytes)) {
      return 1e300;
    }
    cudaMemset(d_max_, 0, sizeof(unsigned int));
    MaxErrKernel<<<512, 256>>>(d_in, d_tmp_dec_, f.n(), d_max_);
    unsigned int bits = 0;
    cudaMemcpy(&bits, d_max_, sizeof(bits), cudaMemcpyDeviceToHost);
    return __builtin_bit_cast(float, bits);
  }
  /** Size the search buffers for f. @return success */
  bool Scratch(const Field &f) {
    if (f.n() <= n_) return true;
    if (d_tmp_cmp_) cudaFree(d_tmp_cmp_);
    if (d_tmp_dec_) cudaFree(d_tmp_dec_);
    tmp_cap_ = MaxCompressedBytes(f);
    n_ = 0;
    if (cudaMalloc(&d_tmp_cmp_, tmp_cap_) != cudaSuccess ||
        cudaMalloc(&d_tmp_dec_, f.n() * sizeof(float)) != cudaSuccess ||
        (!d_max_ && cudaMalloc(&d_max_, sizeof(unsigned int)) != cudaSuccess)) {
      return false;
    }
    n_ = f.n();
    return true;
  }

  double rate_ = 32;
  std::string note_;
  uint8_t *d_tmp_cmp_ = nullptr;
  float *d_tmp_dec_ = nullptr;
  unsigned int *d_max_ = nullptr;
  size_t tmp_cap_ = 0, n_ = 0;
};

}  // namespace

std::unique_ptr<LossyCodec> MakeLossyCodec() {
  return std::make_unique<CuzfpCodec>();
}
