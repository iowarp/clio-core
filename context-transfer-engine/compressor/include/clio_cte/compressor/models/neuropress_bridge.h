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
 * so EstCompressionStats() can source dynamic-selection candidates from the
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
 * @return Candidate stats, best-first (mirrors Rank()'s ordering). Always
 *   restricted to NeuroPress's actual trained action space -- the 8
 *   GPU-lossless nvcomp algorithms (LZ4/Snappy/Deflate/GDeflate/Zstd/ANS/
 *   Cascaded/Bitcomp). No CPU library and none of zfp-sycl/cuSZ/nDzip/cuSZp
 *   are ever included: none of those were part of the trained action space
 *   in the original NeuroPress project either (see the .cc for the
 *   decodeAction cross-reference), so a "prediction" for one of them would
 *   only ever be an alias of some other algorithm's real prediction, not a
 *   genuine learned opinion. They remain reachable through explicit/static
 *   selection, just not through this ranked, dynamic path.
 */
std::vector<CompressionStats> NeuroPressCandidateStats(
    ctp::compress::model::CompressionPredictor &predictor,
    clio::run::u64 chunk_size, double entropy, double mad,
    double second_derivative_mean, bool data_type_float,
    double error_bound = 0.0);

/**
 * @brief Same ranking, with the network reading the data statistics out of
 * DEVICE memory instead of receiving them as host doubles.
 *
 * Identical candidate set, cost model and tie-breaking -- it shares one body
 * with the function above, and the scoring/sort step is literally the same
 * code. The only difference is where inputs 5-7 of the feature vector come
 * from, and therefore whether the chunk's statistics have to make a round trip
 * through host memory to reach the model.
 *
 * That round trip is what upstream does not have: gpucompress_infer_gpu keeps
 * an AutoStatsGPU on the device and hands the pointer to its inference kernel
 * ("Stats remain on GPU -- NN inference reads d_stats_ptr directly on device",
 * gpucompress_compress.cpp:281). Use this whenever the chunk is
 * device-resident, which is the only situation upstream ever operates in.
 *
 * Falls back to the host ranking internally if device inference fails, so the
 * caller never has to handle a half-finished selection.
 *
 * Deliberately takes no entropy/MAD/second-derivative arguments. They would
 * have to be read back before this call to be passed, and that read is a
 * stream synchronization -- placing it BEFORE the inference would reinstate
 * exactly the stall this path exists to remove. They are also provably
 * unnecessary: RankingWeights::Score consumes only the prediction and the
 * chunk size, and the network takes its copies from `device_stats`. Callers
 * wanting the numbers for a log should call ctp::ReadDeviceFeatureStats()
 * AFTER this returns, when the stream is already idle and the read is free.
 *
 * @param device_stats Device pointer from ctp::ComputeDeviceStatsResident().
 * @param stream The stream those statistics were enqueued on
 *   (ctp::DeviceStatsStream()); the inference chains onto it.
 */
std::vector<CompressionStats> NeuroPressCandidateStatsDevice(
    ctp::compress::model::CompressionPredictor &predictor,
    clio::run::u64 chunk_size, const void *device_stats, void *stream,
    bool data_type_float, double error_bound = 0.0, double min_psnr = 0.0,
    bool *out_inference_failed = nullptr);

}  // namespace clio::cte::compressor

#endif  // CLIO_CTE_COMPRESSOR_MODELS_NEUROPRESS_BRIDGE_H_
