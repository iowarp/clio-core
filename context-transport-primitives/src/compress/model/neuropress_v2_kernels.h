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
 * @file neuropress_v2_kernels.h
 * @brief Internal: the GPU side of NeuroPressV2Predictor (one kernel that
 * builds the inputs from device statistics, runs the network and ranks the
 * settings). Not installed; only neuropress_v2_predictor.cc includes it.
 */

#ifndef CLIO_CTP_SRC_COMPRESS_MODEL_NEUROPRESS_V2_KERNELS_H_
#define CLIO_CTP_SRC_COMPRESS_MODEL_NEUROPRESS_V2_KERNELS_H_

#include <cstddef>

namespace ctp::compress::model::v2 {

constexpr int kMaxLayers = 8;     ///< linear layers a model may have
constexpr int kMaxWidth = 256;    ///< widest layer (input, hidden or output)
constexpr int kMaxSettings = 64;  ///< settings a ranking can hold

/** @brief Where each tensor sits in the flat parameter vector. */
struct NetDesc {
  int n_layers = 0;
  int dims[kMaxLayers + 1] = {};         ///< input, hidden..., output
  unsigned long long w_off[kMaxLayers] = {};  ///< W[out][in] of each layer
  unsigned long long b_off[kMaxLayers] = {};  ///< b[out] of each layer
  unsigned long long xm_off = 0, xs_off = 0, ym_off = 0, ys_off = 0;
  int n_settings = 0;
};

/** @brief The cost a ranking minimises, and the chunk it is for. */
struct RankArgs {
  double w_ct = 1.0, w_dt = 1.0, w_io = 1.0;
  double bw_bytes_per_ms = 5.0e6;
  double chunk_bytes = 0.0;
};

/** @brief One ranking, copied back to the host in a single transfer. */
struct RankOut {
  float x[4];                ///< the inputs the kernel computed
  float ct[kMaxSettings];    ///< predicted compress ms per setting
  float dt[kMaxSettings];    ///< predicted decompress ms per setting
  float ratio[kMaxSettings]; ///< predicted ratio per setting
  double cost[kMaxSettings]; ///< cost per setting (inf when unusable)
  int order[kMaxSettings];   ///< settings, cheapest first
};

/**
 * Allocate device memory and copy host bytes into it.
 * @param host  source
 * @param bytes size
 * @param dev   receives the device pointer
 * @return false on a CUDA error
 */
bool Upload(const void *host, size_t bytes, void **dev);

/**
 * Overwrite part of a device buffer (synchronously).
 * @return false on a CUDA error
 */
bool CopyToDevice(void *dev, size_t offset_bytes, const void *host,
                  size_t bytes);

/** Free a device buffer from Upload (nullptr is fine). */
void FreeDevice(void *dev);

/**
 * Compute the inputs from ctp::DeviceFeatureStats, run the network and rank
 * the settings on `stream`, then copy the result back and synchronise.
 * @param d_params     flat parameters on the device
 * @param desc         their layout
 * @param d_available  per-setting availability on the device
 * @param device_stats device pointer to a ctp::DeviceFeatureStats
 * @param args         cost weights and chunk size
 * @param stream       cudaStream_t (nullptr = default stream)
 * @param out          receives the ranking
 * @return false on a CUDA error or an unsupported shape
 */
bool RankOnDevice(const float *d_params, const NetDesc &desc,
                  const unsigned char *d_available, const void *device_stats,
                  const RankArgs &args, void *stream, RankOut *out);

}  // namespace ctp::compress::model::v2

#endif  // CLIO_CTP_SRC_COMPRESS_MODEL_NEUROPRESS_V2_KERNELS_H_
