/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file neuropress_telemetry.h
 * @brief NeuroPress's diagnostic logs, off unless CLIO_NEUROPRESS_SELECTION_LOG
 * / _EXPLORE_LOG name a file. Handles stay private to the .cc.
 */
#ifndef CLIO_CTE_COMPRESSOR_NEUROPRESS_TELEMETRY_H_
#define CLIO_CTE_COMPRESSOR_NEUROPRESS_TELEMETRY_H_

#include <clio_ctp/compress/preprocess/quality_metrics.h>
#include <cstddef>
#include <cstdint>
#include <string>

#include "clio_cte/compressor/compressor_runtime.h"

namespace clio::cte::compressor {

/** Checked before work that only feeds the log, e.g. hashing a chunk. */
bool SelectionLogEnabled();

/** Gates recording sweep candidates, never the sweep itself. */
bool ExploreLogEnabled();

/** packed_preset carries shuffle+quantize as stored in the blob header.
 *  checksum is FNV-1a of the chunk, to prove another impl saw the same bytes. */
void LogNeuroPressSelection(const std::string &blob_name, size_t chunk_size,
                            double entropy, double mad, double second_deriv,
                            int wire_lib, int packed_preset,
                            const CompressionStats *predicted,
                            double actual_ratio, double actual_ct_ms,
                            double actual_psnr, unsigned long long checksum,
                            const char *role = "primary");

/** One row of a sweep. is_primary marks the model's own pick, logged alongside
 *  the alternatives so all actions for a chunk sit in one file. `adopted` marks
 *  whichever row's bytes are stored -- exactly one per chunk. */
/**
 * @brief Park the PRIMARY's measured quality for the site that logs its row.
 *
 * The primary is compressed and measured in Runtime::Compress, but its explore
 * row is written in Runtime::DynamicSchedule -- which AWAITS Compress
 * (compressor_runtime.cc:1388) and logs afterwards. So the measurement always
 * exists by the time the row is written; it just lives in another function.
 * Compress calls Record, the logging site calls Take.
 *
 * Without this the primary's five measured columns were the -1 sentinel on
 * every chunk, while its 31 swept alternatives carried real numbers -- the
 * measurement was being written only to selection.csv.quality.
 */
void RecordPrimaryQuality(
    const std::string &blob_name,
    const ctp::compress::preprocess::QualityMetrics &quality);

/** Retrieve and erase what RecordPrimaryQuality parked for `blob_name`.
 *  False when there is nothing, leaving *out untouched. */
bool TakePrimaryQuality(const std::string &blob_name,
                        ctp::compress::preprocess::QualityMetrics *out);

void LogNeuroPressExplore(const std::string &blob_name, size_t chunk_size,
                          int rank, const std::string &lib_name,
                          uint32_t preset_id, bool quantize, uint32_t shuffle,
                          double pred_ratio, double pred_ct_ms,
                          double pred_dt_ms, double ratio,
                          double ct_ms, double psnr_db, double cost,
                          double primary_cost, bool adopted,
                          bool is_primary = false, double dt_ms = -1.0,
                          /**
                           * The MODEL'S OWN INPUTS for this row, so the eight
                           * numbers NeuroPress was fed are recoverable from
                           * one line instead of a join against selection.csv.
                           *
                           * entropy/mad/second_deriv describe the CHUNK, so
                           * they repeat across a chunk's candidates; that
                           * redundancy is the price of the row being
                           * self-contained.
                           *
                           * eb_encoded is NOT --eb. FeaturesTo8Input feeds
                           * input 3 a SENTINEL of 1e-7 for a lossless
                           * candidate rather than 0, because that is how the
                           * net was trained (configs.py:44,
                           * `eb_val = eb if quant else 1e-7`). Logging the raw
                           * bound would misreport input 3 on every lossless
                           * candidate -- which is every candidate in a
                           * lossless run.
                           */
                          double entropy = -1.0, double mad = -1.0,
                          double second_deriv = -1.0,
                          double eb_encoded = -1.0,
                          /**
                           * MEASURED reconstruction quality, or nullptr when
                           * the candidate was not measured -- which is the
                           * default, since CLIO_NEUROPRESS_MEASURE_QUALITY is
                           * off and measuring costs a full inverse chain per
                           * candidate.
                           *
                           * NOT the same quantity as the psnr_db argument
                           * above, which is ANALYTICAL: derived from
                           * (range, error_bound), capped at 120 dB, and blind
                           * to whether the bound was met. These come from
                           * comparing the reconstruction with the original,
                           * so they can disagree with it -- and a disagreement
                           * is the finding, not an error. The columns are
                           * named meas_* for that reason.
                           */
                          const ctp::compress::preprocess::QualityMetrics
                              *quality = nullptr);

/** Hash of codec output, from Compress(); joined by blob name. stage is
 *  "primary" or "adopted" -- a chunk can appear twice and the LAST row is the
 *  tier, so summing rows under-reports exploration. */
void LogCompressedPayload(const std::string &blob_name, const char *payload,
                          size_t payload_size, bool on_device, bool beneficial,
                          double compress_kernel_ms, const char *stage);

/**
 * @brief One row per STORED chunk: measured reconstruction quality.
 *
 * Its own file (<CLIO_NEUROPRESS_SELECTION_LOG>.quality, the same sibling
 * convention as .payload) rather than a column on an existing one, because
 * none of them is written late enough. The selection log's primary row is
 * emitted in DynamicSchedule, BEFORE Compress runs; explore.csv is written in
 * the same place; and the payload row is written before the reconstruction
 * exists. A measurement that only happens after the codec has run cannot be
 * carried by a file that has already been written.
 *
 * COVERAGE IS THE POINT. The sweep measures only its own candidates, so a
 * chunk whose PRIMARY was kept -- and a static-codec or inference-only run,
 * which never explores at all -- had no quality figure. Measured on a 64^3
 * eb=0.1 run, --check-bound's worst chunk was exactly such a chunk:
 * 9.499836e-02 against the adopted candidates' 9.499615e-02, a disagreement
 * that looked like an arithmetic bug and was a coverage hole.
 *
 * Joins to blobs.csv / selection.csv / explore.csv on `blob`.
 */
void LogMeasuredQuality(const std::string &blob_name, size_t orig_bytes,
                        uint32_t shuffle, bool quantized,
                        const ctp::compress::preprocess::QualityMetrics &q);

}  // namespace clio::cte::compressor

#endif  // CLIO_CTE_COMPRESSOR_NEUROPRESS_TELEMETRY_H_
