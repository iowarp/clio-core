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
                            double actual_psnr, unsigned long long checksum);

/** One row of a sweep. is_primary marks the model's own pick, logged alongside
 *  the alternatives so all actions for a chunk sit in one file. `adopted` marks
 *  whichever row's bytes are stored -- exactly one per chunk. */
void LogNeuroPressExplore(const std::string &blob_name, size_t chunk_size,
                          int rank, const std::string &lib_name,
                          uint32_t preset_id, bool quantize, uint32_t shuffle,
                          double pred_ratio, double pred_ct_ms,
                          double pred_dt_ms, double ratio,
                          double ct_ms, double psnr_db, double cost,
                          double primary_cost, bool adopted,
                          bool is_primary = false, double dt_ms = -1.0);

/** Hash of codec output, from Compress(); joined by blob name. stage is
 *  "primary" or "adopted" -- a chunk can appear twice and the LAST row is the
 *  tier, so summing rows under-reports exploration. */
void LogCompressedPayload(const std::string &blob_name, const char *payload,
                          size_t payload_size, bool on_device, bool beneficial,
                          double compress_kernel_ms, const char *stage);

}  // namespace clio::cte::compressor

#endif  // CLIO_CTE_COMPRESSOR_NEUROPRESS_TELEMETRY_H_
