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

/** @file neuropress_bridge.h
 *  @brief Ranks clio_ctp::compress::model predictors into CompressionStats. */
#ifndef CLIO_CTE_COMPRESSOR_MODELS_NEUROPRESS_BRIDGE_H_
#define CLIO_CTE_COMPRESSOR_MODELS_NEUROPRESS_BRIDGE_H_

#include <clio_ctp/compress/model/predictor.h>
#include <clio_runtime/types.h>

#include <vector>

#include "clio_cte/compressor/compressor_runtime.h"
#include "clio_cte/compressor/models/neuropress_cost.h"

namespace clio::cte::compressor {

/** Rank the model's candidate set for one chunk's statistics. compress_lib_ is
 *  the CompressionFactory WIRE id, not the ML base id. Best-first, restricted
 *  to the trained action space: the 8 GPU-lossless nvcomp algorithms. */
std::vector<CompressionStats> NeuroPressCandidateStats(
    ctp::compress::model::CompressionPredictor &predictor,
    clio::run::u64 chunk_size, double entropy, double mad,
    double second_derivative_mean, bool data_type_float,
    double error_bound = 0.0, bool ratio_only = false);

/** Same ranking with the network reading statistics from DEVICE memory. Takes
 *  no entropy/MAD args on purpose: reading them back first is a stream sync
 *  that would reinstate the stall this path removes.
 *
 *  Does NOT fall back to the host: a failed device inference returns empty and
 *  sets out_inference_failed, so the caller declines NeuroPress for the chunk
 *  rather than being handed a plausible ranking the GPU never produced. */
std::vector<CompressionStats> NeuroPressCandidateStatsDevice(
    ctp::compress::model::CompressionPredictor &predictor,
    clio::run::u64 chunk_size, const void *device_stats, void *stream,
    bool data_type_float, double error_bound = 0.0, double min_psnr = 0.0,
    bool *out_inference_failed = nullptr, bool ratio_only = false);

}  // namespace clio::cte::compressor

#endif  // CLIO_CTE_COMPRESSOR_MODELS_NEUROPRESS_BRIDGE_H_
