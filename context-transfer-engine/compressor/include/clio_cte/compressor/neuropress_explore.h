/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file neuropress_explore.h
 * @brief State for an exploration sweep: re-compress one stored chunk with up
 * to K next-best actions (prepare / collect / score) and keep any winner.
 */
#ifndef CLIO_CTE_COMPRESSOR_NEUROPRESS_EXPLORE_H_
#define CLIO_CTE_COMPRESSOR_NEUROPRESS_EXPLORE_H_

#include <clio_ctp/compress/compress.h>
#include <clio_ctp/compress/nvcomp.h>
#include <clio_ctp/compress/preprocess/quality_metrics.h>
#include <clio_ctp/compress/preprocess/quantization.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "clio_cte/compressor/compressor_runtime.h"

namespace clio::cte::compressor {

/** One sweep-log row. Held until scoring ends: `adopted` is settled by the
 *  LAST candidate to beat the running best, so a row written at measurement
 *  time would claim adoption for every successive winner. */
struct ExploreRow {
  std::string lib;
  uint32_t preset_id;
  bool quant;
  uint32_t shuffle;
  /** pred_ct/pred_dt are the NN's own outputs for THIS action, made in the
   *  one inference that ranked the chunk -- not recomputed by the sweep. */
  double pred_ratio, pred_ct, pred_dt, ratio, ct_ms, psnr, cost;
  int rank;
  /** MEASURED decompression time, or <0 when the sweep did not take one. */
  double dt_ms = -1.0;
  /** MEASURED reconstruction quality, valid only when have_quality. Distinct
   *  from `psnr` above, which is ANALYTICAL -- derived from (range, bound) and
   *  therefore unable to see a bound violation at all. */
  ctp::compress::preprocess::QualityMetrics quality;
  bool have_quality = false;
};

/** Best alternative so far, plus what is needed to store it. One object, not
 *  ten locals: kept apart a candidate can be adopted with a field left from an
 *  earlier one, and a mismatched quantize pair is unrecoverable on read. */
struct ExploreWinner {
  bool have = false;
  std::vector<char> payload;
  int lib = 0;
  uint32_t preset_id = 0;
  uint32_t shuffle = 0;
  double ratio = 0.0;
  double time_ms = 0.0;
  /** MEASURED decompression time for THIS candidate, or <0 if not measured.
   *  Carried for the same reason ratio and time_ms are: when a winner is
   *  adopted every field the context reports must describe the WINNER. Leaving
   *  dt behind reported the primary's decompression time beside the winner's
   *  codec name -- a row that looks right and describes two different
   *  algorithms. */
  double dt_ms = -1.0;
  bool quant = false;
  ctp::compress::preprocess::DeviceQuantizeParams quant_params;
  /** Index into the sweep-log rows, so the last adopter carries the flag. */
  int row = -1;
};

/** One measured alternative: what the codec call needs, and what the scoring
 *  step reads back. At file scope because all three sweep steps name it. */
// Only the CODEC CALL is parallel. Prep allocates GPU backends through the
// runtime, and scoring must be order-deterministic -- a tie decided by which
// worker finished first would make the stored blob vary run to run.
struct ExploreSlot {
  const CompressionStats* alt = nullptr;
  std::string name;
  uint32_t preset_id = 0;
  uint32_t applied_shuffle = 0;
  bool applied_quant = false;
  ctp::compress::preprocess::DeviceQuantizeParams quant_params;
  std::unique_ptr<ctp::Compressor> compressor;
  // MOVED in from the prep loop's locals; moving a vector transfers the heap
  // block, so `input`/`out_ptr` stay valid without fixup.
  std::vector<char> device_staging, quant_staging, shuffle_staging,
      output;
  char* input = nullptr;
  size_t compress_size = 0;
  char* out_ptr = nullptr;
  size_t capacity = 0;
  // Filled by the launch/collect phases below.
  bool ok = false;
  bool launched = false;
  /** Set once collected, so a batched sweep never finishes a slot twice. */
  bool collected = false;
  size_t compressed_size = 0;
  double time_ms = 0.0;
  /** MEASURED decompression time, or <0 when it was not measured -- which is
   *  the default and upstream's only behaviour. Scoring falls back to the
   *  primary's predicted dt when this is negative, so the two paths differ in
   *  exactly one term. */
  double decomp_time_ms = -1.0;

  // ---- MEASURED reconstruction quality (opt-in) --------------------------
  // The ORIGINAL device bytes, captured BEFORE this candidate's quantize and
  // shuffle. Measuring against `input` instead would compare the codec with
  // itself: every nvcomp codec is lossless, so the error would be identically
  // zero for every candidate and the column would read as a perfect
  // reconstruction. Quality is only meaningful against the pre-transform
  // data, which is what upstream compares (it hands gpucompress_compress_gpu
  // the raw buffer and lets the API quantize internally).
  const void* orig_device = nullptr;
  size_t orig_bytes = 0;
  /** Valid only when have_quality; left untouched otherwise, never zeroed --
   *  a zeroed QualityMetrics reads as rmse 0, i.e. a perfect round trip. */
  ctp::compress::preprocess::QualityMetrics quality;
  bool have_quality = false;
#if CTP_ENABLE_COMPRESS && CTP_ENABLE_NVCOMP
  ctp::NvComp* gpu = nullptr;   ///< non-null when the async path ran
  ctp::NvComp::AsyncSlot async;  ///< this slot's stream + events
#endif
};

/** Sweep step 2: finish every launched slot. Every slot must be in flight
 *  before any is waited on, or the sweep serializes into K sequential calls. */
void CollectExploreSlots(std::vector<std::unique_ptr<ExploreSlot>> &slots);

/** True when CLIO_NEUROPRESS_EXPLORE_MEASURE_DT asks the sweep to decompress
 *  each candidate back for a real decompression time instead of substituting
 *  the NN's prediction. Read once. Off by default -- upstream never
 *  decompresses at write time, and turning this on makes selection diverge. */
/**
 * @brief Is per-candidate reconstruction quality measured? (env, default OFF)
 *
 * CLIO_NEUROPRESS_MEASURE_QUALITY. Off by default in the LIBRARY, but the
 * paper benchmark turns it ON in every run_config.sh -- an exploration run
 * that reports ratio and speed against a merely PREDICTED psnr_db answers
 * half the lossy question, and psnr_db is easily mistaken for a measurement
 * when meas_psnr_db beside it is empty. Set MEASURE_QUALITY=0 to opt out.
 *
 * Off by default HERE and that is not caution:
 * measuring quality means DECOMPRESSING and then INVERTING both preprocessing
 * transforms for every candidate, so at K=31 the sweep pays 31 full inverse
 * chains per chunk on top of a decompression that already runs 1.2-30x the
 * codec's compression time. Upstream gates the equivalent behind a whole
 * separate VOL mode ("trace"), whose own comment calls it "intentionally
 * sequential and slow -- for offline analysis only".
 *
 * It also changes nothing about selection: the metrics are recorded, never
 * scored on. Turning it on must not move which candidate is adopted.
 */
bool MeasureExploreQuality();

/**
 * @brief Reconstruct one stored chunk and MEASURE it against the original.
 *
 * ONE copy of the inverse chain, shared by the sweep's candidates and by the
 * primary. Two copies would be worse than duplication: both would print
 * plausible numbers, so a divergence between them would be invisible in the
 * results -- the same failure mode the shared QualityFromAccumulators exists
 * to prevent on the derivation side.
 *
 * The chain is codec inverse -> byte unshuffle -> dequantize, and all three
 * steps are required. Decompressing alone returns the QUANTIZED, SHUFFLED
 * bytes; every nvcomp codec is lossless, so comparing those against the
 * codec's own input reports rmse 0 for every candidate at every error bound.
 *
 * @param orig_device   ORIGINAL bytes, device-resident, pre-transform. A host
 *   pointer yields false: quality is measured on the GPU or not at all.
 * @param orig_bytes    size of the original (float32 assumed, as upstream does)
 * @param compressed    device pointer to the codec's output, no Clio header
 * @param compressed_size  bytes at `compressed`
 * @param post_transform_size  what the CODEC was fed -- smaller than
 *   orig_bytes whenever quantization ran
 * @param shuffle       applied shuffle WIDTH, 0 for none
 * @param quant         applied quantizer parameters, or nullptr if none ran
 * @return false on any failure, leaving *out untouched. Never a partial
 *   reconstruction and never a host fallback: unmeasured must read as
 *   unmeasured, not as a perfect round trip.
 */
bool MeasureStoredChunkQuality(
    const void *orig_device, size_t orig_bytes, const void *compressed,
    size_t compressed_size, size_t post_transform_size, uint32_t shuffle,
    const ctp::compress::preprocess::DeviceQuantizeParams *quant,
    ctp::compress::preprocess::QualityMetrics *out);

bool MeasureExploreDecompTime();

}  // namespace clio::cte::compressor

#endif  // CLIO_CTE_COMPRESSOR_NEUROPRESS_EXPLORE_H_
