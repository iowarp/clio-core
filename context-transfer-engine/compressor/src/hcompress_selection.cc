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
 * @file hcompress_selection.cc
 * @brief HCompress's Expected-Compression-Cost model, deployed as the
 * chunk's selector.
 *
 * HCompress (Devarajan et al., IPDPS 2020) predicts a codec's cost from the
 * compression library and the data size only. This file is where that model
 * chooses a chunk's codec at write time, so figure 9 can draw it as a bar
 * beside NeuroPress.
 *
 * ONLY THE PREDICTOR DIFFERS FROM A NEUROPRESS SELECTION. HCompress ranks
 * through NeuroPressCandidateStats -- NeuroPress's own 32-action nvcomp
 * candidate set, its NeuroPressCost weights and its tie order -- with
 * HCompress's predictions in place of the network's. Ranking a different
 * candidate set (the legacy CPU-codec path) would compare two different
 * questions, not two predictors.
 *
 * WHAT IT IS GIVEN IS THE LIBRARY AND THE SIZE. The chunk's bytes are never
 * read: no per-chunk statistics (entropy, MAD, second derivative), no data
 * type and no data distribution. hcompress_ccp_predictor.h warns that feeding
 * any of them would quietly repair the very property the comparison exists to
 * expose.
 */

#include <clio_ctp/compress/compress_factory.h>
#include <clio_ctp/compress/model/hcompress_ccp_predictor.h>

#include <chrono>
#include <cstdlib>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "clio_cte/compressor/compressor_runtime.h"
#include "clio_cte/compressor/models/neuropress_bridge.h"
#include "clio_cte/compressor/neuropress_telemetry.h"

namespace clio::cte::compressor {

namespace {

/** Milliseconds since `t0`. */
double HcMsSince(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

std::vector<CompressionStats> Runtime::HCompressRankChunk(
    clio::run::u64 chunk_size, const Context &context) {
  // Rank NeuroPress's candidates with HCompress's predictions. Each candidate
  // is predicted from its library and chunk_size alone. The statistics and
  // the data-type flag are zeros because HCompress takes none of them; its
  // Predict() reads only the library and the size.
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<CompressionStats> stats;
  {
    std::lock_guard<std::mutex> lock(hcompress_mutex_);
    stats = NeuroPressCandidateStats(*hcompress_predictor_, chunk_size, 0.0,
                                     0.0, 0.0, /*data_type_float=*/false,
                                     context.error_bound_,
                                     /*ratio_only=*/false);
  }
  const double rank_ms = HcMsSince(t0);

  // HCompress has no quality output (psnr_db < 0, "not predicted"), so a PSNR
  // target cannot be honoured: filtering on it would drop every candidate.
  if (context.target_psnr_ > 0) {
    static std::once_flag warned;
    std::call_once(warned, [] {
      HLOG(kWarning,
           "HCompress: a PSNR target is set but HCompress predicts no quality "
           "(IPDPS 2020 Sec. IV-D has no quality output); ranking ignores it");
    });
  }

  // Same per-chunk records NeuroPress writes, so the two selectors' costs
  // land in the same phases.csv columns: stats = 0 (HCompress reads nothing
  // from the chunk), nn = prediction + ranking.
  RecordSelectionTiming(rank_ms * 1000.0, /*reused=*/false);
  if (PhaseLogEnabled()) {
    RecordSelectionPhases(/*stats_ms=*/0.0, rank_ms, /*choice_ms=*/0.0,
                          /*reused=*/false);
  }
  if (!stats.empty()) {
    HLOG(kDebug,
         "HCompress selection: chunk_size={} -> wire_id={} ({} candidates; "
         "rank {} ms)",
         chunk_size, stats.front().compress_lib_, stats.size(), rank_ms);
  }
  return stats;
}

void Runtime::HCompressObserve(int wire_lib, int preset_field,
                               clio::run::u64 chunk_size,
                               const Context &context) {
  // Key the EXECUTED action the way the seed keys it: bare algorithm name,
  // then the shuffle and quantize bits that ride in the preset field.
  std::string name = ctp::CompressionFactory::NameForWireId(wire_lib);
  static const std::string kNvcompPrefix = "nvcomp-";
  if (name.compare(0, kNvcompPrefix.size(), kNvcompPrefix) == 0) {
    name.erase(0, kNvcompPrefix.size());
  }
  const uint32_t bits = static_cast<uint32_t>(preset_field);
  const bool byte_shuffle = ((bits >> 8) & 0xFFu) != 0;
  const bool quantize = ((bits >> 24) & 0x1u) != 0;

  ctp::compress::model::CcpObservation row;
  row.library = ctp::compress::model::HCompressCcpPredictor::LibraryKey(
      name, quantize, byte_shuffle);
  // Ranking the v2 settings, the preset field IS the setting index (also
  // for the store setting, wire 0): key it the way the seed keys it.
  if (HCompressRanksV2()) row.library = HCompressV2Key(preset_field);
  row.bytes = static_cast<double>(chunk_size);
  // A non-positive measurement means NOT MEASURED and trains nothing, so an
  // unmeasured decompression time teaches the decompression head nothing
  // rather than teaching it zero.
  row.compress_time_ms = context.actual_compress_time_ms_;
  row.decompress_time_ms = context.actual_decompress_time_ms_ > 0.0
                               ? context.actual_decompress_time_ms_
                               : 0.0;
  row.compression_ratio = context.actual_compression_ratio_;
  std::lock_guard<std::mutex> lock(hcompress_mutex_);
  hcompress_predictor_->Observe(row);
}

bool Runtime::HCompressV2SettingsRequested() {
  static const bool on = [] {
    const char *e = std::getenv("CLIO_HCOMPRESS_V2_SETTINGS");
    return e != nullptr && *e != '\0' && *e != '0';
  }();
  return on;
}

std::string Runtime::HCompressV2Key(int setting) {
  const char *spec = ctp::GpuSettingSpec(setting);
  std::istringstream words(spec != nullptr ? spec : "");
  std::string algorithm, word;
  words >> algorithm;
  bool shuffle = false;
  while (words >> word) {
    shuffle = shuffle || word == "shuffle=byte" || word == "shuffle=bit";
  }
  return ctp::compress::model::HCompressCcpPredictor::LibraryKey(
      algorithm, /*quantize=*/false, shuffle);
}

}  // namespace clio::cte::compressor
