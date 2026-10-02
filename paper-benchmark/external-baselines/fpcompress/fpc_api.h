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
 * @file fpc_api.h
 * @brief Device-buffer library interface to SPspeed and SPratio, the GPU
 * single-precision lossless compressors of Azami, Fallin and Burtscher
 * (ASPLOS '25, github.com/burtscher/FPcompress).
 *
 * Upstream ships each algorithm as two CUDA programs (file in, file out). The
 * implementation compiles those UNMODIFIED programs from a pinned clone and
 * launches their kernels exactly as their main() does; see fpc_encode.cuh.
 * The stream format is upstream's own: a 4-byte uncompressed size, then one
 * 2-byte size per 16 KiB chunk, then the chunks.
 *
 * Every function issues its work on the given stream and synchronizes it
 * before returning. Not thread-safe: each algorithm keeps one set of
 * scratch buffers per process.
 */
#ifndef CLIO_EXTERNAL_FPC_API_H_
#define CLIO_EXTERNAL_FPC_API_H_

#include <cuda_runtime.h>

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @param n uncompressed bytes.
 * @return bytes the compressed buffer must hold in the worst case.
 */
size_t spspeed_compress_bound(size_t n);
size_t spratio_compress_bound(size_t n);

/**
 * Compress n bytes of device memory.
 *
 * @param d_in      device input, n bytes (float32 data; any length works).
 * @param n         input bytes, 0 < n <= 1 GiB.
 * @param d_out     device output, out_cap bytes.
 * @param out_cap   capacity of d_out; must be at least *_compress_bound(n).
 * @param out_bytes receives the compressed length in bytes.
 * @param stream    CUDA stream; synchronized on return.
 * @return 0 on success; -1 on a CUDA error, -2 when out_cap is below the
 *         bound, -3 on an invalid argument.
 */
int spspeed_compress(const void *d_in, size_t n, void *d_out, size_t out_cap,
                     size_t *out_bytes, cudaStream_t stream);
int spratio_compress(const void *d_in, size_t n, void *d_out, size_t out_cap,
                     size_t *out_bytes, cudaStream_t stream);

/**
 * Read the uncompressed size a compressed stream declares. Call it before
 * decompressing a stream that did not come from this process: the decoder
 * writes exactly that many bytes.
 *
 * @param d_in     device compressed stream.
 * @param in_bytes its length.
 * @param stream   CUDA stream; synchronized on return.
 * @return the declared size in bytes, or -1 on a CUDA error or short stream.
 */
long long spspeed_decompressed_size(const void *d_in, size_t in_bytes,
                                    cudaStream_t stream);
long long spratio_decompressed_size(const void *d_in, size_t in_bytes,
                                    cudaStream_t stream);

/**
 * Decompress a stream produced by the matching *_compress.
 *
 * @param d_in     device compressed stream, in_bytes long.
 * @param in_bytes its length.
 * @param d_out    device output, n bytes.
 * @param n        decompressed bytes; must equal the size the stream declares.
 * @param stream   CUDA stream; synchronized on return.
 * @return 0 on success; -1 on a CUDA error, -3 on an invalid argument or when
 *         the decoder produced a size other than n.
 */
int spspeed_decompress(const void *d_in, size_t in_bytes, void *d_out,
                       size_t n, cudaStream_t stream);
int spratio_decompress(const void *d_in, size_t in_bytes, void *d_out,
                       size_t n, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif  // CLIO_EXTERNAL_FPC_API_H_
