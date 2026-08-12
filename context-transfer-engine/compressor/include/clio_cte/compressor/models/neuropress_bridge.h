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
 * @file neuropress_bridge.h
 * @brief Bridges the ported clio_ctp::compress::model predictors (issue #693,
 * the NeuroPress NN in particular) into this chimod's own CompressionStats,
 * so EstCompressionStats() can source dynamic-selection candidates -- GPU
 * ones (nvcomp/cusz/ndzip, incl. Cascaded/Bitcomp) included -- from the
 * ranked model instead of only the fixed CPU-only candidate list.
 */
#ifndef CLIO_CTE_COMPRESSOR_MODELS_NEUROPRESS_BRIDGE_H_
#define CLIO_CTE_COMPRESSOR_MODELS_NEUROPRESS_BRIDGE_H_

#include <clio_ctp/compress/model/predictor.h>
#include <clio_runtime/types.h>

#include <vector>

#include "clio_cte/compressor/compressor_runtime.h"

namespace clio::cte::compressor {

/**
 * @brief Rank clio_ctp::compress::model's candidate set for one data
 * buffer's statistics and convert the results into this module's own
 * CompressionStats.
 *
 * compress_lib_ on each returned stat is the CompressionFactory WIRE id
 * (CompressionFactory::WireIdForName), not the model's ML base id, so the
 * result can be used exactly like the existing hardcoded candidate stats --
 * including being passed straight through to the Compress task, which
 * resolves compress_lib_ via NameForWireId().
 *
 * @param predictor Loaded compression-metric predictor (e.g. a ready
 *   NeuroPressNNPredictor). Ranking on a not-ready predictor is safe (each
 *   model's Rank() degrades to finite defaults) but not useful.
 * @param chunk_size Size of the data chunk in bytes.
 * @param entropy Shannon entropy of the chunk (bits/byte).
 * @param mad Mean absolute deviation of the chunk.
 * @param second_derivative_mean Mean second derivative (curvature).
 * @param data_type_float True if the chunk is floating-point data.
 * @param include_gpu Include GPU-backed compressors (nvcomp/cusz/ndzip) in
 *   the candidate set. Candidates whose backend isn't compiled into this
 *   build still rank (the registry is build-independent) but can't
 *   actually execute -- callers driving real compression should filter on
 *   backend availability.
 * @param include_cpu Include CPU-backed compressors. Pass false when the
 *   chunk is a device-resident buffer: the original NeuroPress project's
 *   action space is GPU-native only (see DefaultCandidates), and a CPU
 *   codec would otherwise have to read the device pointer directly on the
 *   host to run at all.
 * @return Candidate stats, best-first (mirrors Rank()'s ordering).
 */
std::vector<CompressionStats> NeuroPressCandidateStats(
    ctp::compress::model::CompressionPredictor &predictor,
    clio::run::u64 chunk_size, double entropy, double mad,
    double second_derivative_mean, bool data_type_float,
    bool include_gpu = true, bool include_cpu = true);

}  // namespace clio::cte::compressor

#endif  // CLIO_CTE_COMPRESSOR_MODELS_NEUROPRESS_BRIDGE_H_
