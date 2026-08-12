/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

/**
 * @file ranking.h
 * @brief Bulk-ranking helpers for the unified compression predictors (#693).
 *
 * DefaultCandidates() enumerates the known compressors (optionally crossed with
 * presets and preprocessor combinations) as a ready-made candidate set for
 * CompressionPredictor::Rank(). The base_ids mirror the frozen
 * CompressionFactory registry, but this header is deliberately independent of
 * the compressor backends so the model library builds without them; callers
 * that only want *available* compressors should filter the returned list.
 */
#ifndef CLIO_CTP_COMPRESS_MODEL_RANKING_H_
#define CLIO_CTP_COMPRESS_MODEL_RANKING_H_

#include <string>
#include <vector>

#include "clio_ctp/compress/model/predictor.h"

namespace ctp::compress::model {

/** @brief A known compressor: canonical name + frozen CompressionFactory id. */
struct CompressorEntry {
  const char *name;
  int base_id;
  bool is_gpu;  /**< GPU backend (may be unavailable at runtime). */
};

/** @brief The frozen compressor registry mirror (see compress_factory.h). */
inline const std::vector<CompressorEntry> &KnownCompressors() {
  static const std::vector<CompressorEntry> kEntries = {
      // CPU lossless
      {"brotli", 6, false}, {"bzip2", 1, false}, {"blosc2", 8, false},
      {"lz4", 3, false},    {"lzma", 5, false},  {"snappy", 7, false},
      {"zlib", 4, false},   {"zstd", 2, false},
      // CPU lossy (LibPressio)
      {"fpzip", 12, false}, {"sz3", 11, false},  {"zfp", 10, false},
      // GPU lossless (nvcomp)
      {"nvcomp-lz4", 13, true},   {"nvcomp-snappy", 14, true},
      {"nvcomp-zstd", 15, true},  {"nvcomp-gdeflate", 16, true},
      {"nvcomp-deflate", 17, true}, {"nvcomp-ans", 18, true},
      // GPU lossy / misc
      {"zfp-sycl", 19, true}, {"cusz", 20, true}, {"ndzip", 21, true},
      {"cuszp", 22, true},
      // GPU lossless (nvcomp) -- completes NeuroPress's 8-algorithm action
      // space (LZ4/Snappy/Deflate/Gdeflate/Zstd/ANS/Cascaded/Bitcomp).
      {"nvcomp-cascaded", 23, true}, {"nvcomp-bitcomp", 24, true},
  };
  return kEntries;
}

/**
 * @brief Build a default candidate set for Rank().
 *
 * @param include_gpu       Include GPU-backed compressors.
 * @param presets           Presets to expand each compressor over (1/2/3).
 * @param with_preprocessors Also emit byte-shuffle / quantize / both variants
 *                           (error_bound applied to the quantized variants).
 * @param error_bound       Error bound used for the quantized variants.
 * @param include_cpu       Include CPU-backed compressors. Set false for a
 *                           device-resident input buffer: the original
 *                           NeuroPress project's entire action space is
 *                           GPU-native (nvcomp/cusz/cuszp/ndzip only, see
 *                           gpucompress_compress.cpp's "stats remain on
 *                           GPU" design) -- it never had CPU candidates to
 *                           rank in the first place. Requires include_gpu
 *                           true or the candidate set comes back empty.
 */
inline std::vector<CandidateConfig> DefaultCandidates(
    bool include_gpu = false,
    const std::vector<int> &presets = {1, 2, 3},
    bool with_preprocessors = false, double error_bound = 1e-3,
    bool include_cpu = true) {
  std::vector<CandidateConfig> out;
  for (const auto &e : KnownCompressors()) {
    if (e.is_gpu && !include_gpu) continue;
    if (!e.is_gpu && !include_cpu) continue;
    for (int preset : presets) {
      CandidateConfig base;
      base.base_id = e.base_id;
      base.preset_id = preset;
      base.library_name = e.name;
      out.push_back(base);
      if (with_preprocessors) {
        CandidateConfig shuf = base;
        shuf.byte_shuffle = true;
        out.push_back(shuf);
        CandidateConfig quant = base;
        quant.quantize = true;
        quant.error_bound = error_bound;
        out.push_back(quant);
        CandidateConfig both = base;
        both.byte_shuffle = true;
        both.quantize = true;
        both.error_bound = error_bound;
        out.push_back(both);
      }
    }
  }
  return out;
}

/**
 * @brief Convenience: rank the default candidate set for one data buffer.
 */
inline std::vector<RankedPrediction> RankDefault(
    CompressionPredictor &model, const DataFeatures &data,
    const RankingWeights &weights = RankingWeights(), bool include_gpu = false) {
  return model.Rank(data, DefaultCandidates(include_gpu), weights);
}

}  // namespace ctp::compress::model

#endif  // CLIO_CTP_COMPRESS_MODEL_RANKING_H_
