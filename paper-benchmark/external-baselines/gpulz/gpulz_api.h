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
 * @file gpulz_api.h
 * @brief A library interface over GPULZ (ICS'23), device buffer to device buffer.
 *
 * GPULZ (https://github.com/hpdps-group/ICS23-GPULZ) ships as one program that
 * compresses and decompresses a file in the same run. This interface exposes
 * its unmodified kernels as two calls on device memory, with a self-describing
 * stream so a compressed chunk can be decompressed later on its own:
 *
 *   [header 24 B][flag offsets, nb+1 x u32][data offsets, nb+1 x u32]
 *   [flag bytes][literal/match bytes]
 *
 * nb = ceil(n / 2048): GPULZ works on 2048-byte blocks of 4-byte symbols. Input
 * that is not a whole number of blocks is zero-padded internally, so any byte
 * length round-trips. LOSSLESS: decompression is bit-exact.
 *
 * Buffers and nothing else are kept per calling thread and only grow, so
 * repeated calls pay no allocation (the fairness rule cuszp.h applies).
 */
#ifndef CLIO_EXTERNAL_GPULZ_API_H_
#define CLIO_EXTERNAL_GPULZ_API_H_

#include <cuda_runtime.h>

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Largest stream gpulz_compress can produce for n input bytes.
 *
 * @param n uncompressed bytes.
 * @return bytes the output buffer must hold in the worst case.
 */
size_t gpulz_compress_bound(size_t n);

/**
 * Compress n bytes of device memory into a self-describing device stream.
 *
 * @param d_in      device input, n bytes.
 * @param n         input bytes (> 0).
 * @param d_out     device output, out_cap bytes, 4-byte aligned.
 * @param out_cap   capacity of d_out; gpulz_compress_bound(n) always fits.
 * @param out_bytes receives the stream length in bytes.
 * @param stream    CUDA stream every step is issued on; synchronized on return.
 * @return 0 on success; negative on a CUDA error (-1), a stream larger than
 *         out_cap or an input over 4 GiB (-2), or an invalid argument (-3).
 */
int gpulz_compress(const void *d_in, size_t n, void *d_out, size_t out_cap,
                   size_t *out_bytes, cudaStream_t stream);

/**
 * Decompress a gpulz_compress stream into device memory.
 *
 * @param d_in     device stream, in_bytes long.
 * @param in_bytes stream length in bytes.
 * @param d_out    device output, n bytes.
 * @param n        decompressed bytes; must equal the length compressed.
 * @param stream   CUDA stream; synchronized on return.
 * @return 0 on success; negative on a CUDA error (-1) or a malformed or
 *         mismatched stream (-3).
 */
int gpulz_decompress(const void *d_in, size_t in_bytes, void *d_out, size_t n,
                     cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif  // CLIO_EXTERNAL_GPULZ_API_H_
