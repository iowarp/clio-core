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
 * @file pfpl_shim.cuh
 * @brief Shared pieces of the PFPL wrappers (pfpl_comp.cu, pfpl_decomp.cu):
 * every header the upstream programs include, included FIRST so the `main`
 * rename cannot reach into them, and the launch state their main() builds.
 */
#ifndef PFPL_SHIM_CUH_
#define PFPL_SHIM_CUH_

#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

/**
 * Compress n_bytes of float32 device data with an ABSOLUTE bound, exactly as
 * PFPL's f32_abs_compress_cuda main() launches it (threshold = +inf).
 * @return 0 on success, -1 on a CUDA error, -2 when out_cap is too small
 */
extern "C" int pfpl_abs_compress(const void *d_in, size_t n_bytes, float eb,
                                 void *d_out, size_t out_cap,
                                 size_t *out_bytes, cudaStream_t s);
/** @return the largest stream pfpl_abs_compress can write for n_bytes */
extern "C" size_t pfpl_abs_bound(size_t n_bytes);
/**
 * Decompress a pfpl_abs_compress stream into n_bytes of device memory.
 * @return 0 on success, -1 on a CUDA error, -3 on a size mismatch
 */
extern "C" int pfpl_abs_decompress(const void *d_cmp, void *d_out,
                                   size_t n_bytes, cudaStream_t s);

#endif  // PFPL_SHIM_CUH_
