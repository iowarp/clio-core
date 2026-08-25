/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file neuropress_telemetry.h
 * @brief The three NeuroPress diagnostic logs, and the only five entry points
 * the compressor runtime uses to write them.
 *
 * All three are off unless their environment variable names a file, so the
 * normal path pays one relaxed load per call. They live here rather than in
 * compressor_runtime.cc because none of them is part of compressing a blob:
 * they exist so a run can be audited afterwards, and a selection is otherwise
 * invisible -- it happens inside the compressor runtime, several layers below
 * whatever issued the write, and the caller only learns that the write
 * succeeded.
 *
 * The file handles and their mutexes stay private to the .cc. Nothing outside
 * needs to name a log to write to it.
 *
 *   CLIO_NEUROPRESS_SELECTION_LOG   one row per chunk: what was chosen
 *                                   (plus a `.payload` sidecar: what the
 *                                   codec produced for it)
 *   CLIO_NEUROPRESS_EXPLORE_LOG     one row per candidate measured in a sweep
 */
#ifndef CLIO_CTE_COMPRESSOR_NEUROPRESS_TELEMETRY_H_
#define CLIO_CTE_COMPRESSOR_NEUROPRESS_TELEMETRY_H_

#include <cstddef>
#include <cstdint>
#include <string>

#include "clio_cte/compressor/compressor_runtime.h"

namespace clio::cte::compressor {

/** @brief Is the per-chunk selection trace switched on? Checked before doing
 *  any work that exists only to feed it -- hashing a chunk, for one. */
bool SelectionLogEnabled();

/** @brief Is the per-candidate sweep trace switched on? The sweep measures up
 *  to 31 alternatives per chunk; this gates recording them, never the sweep. */
bool ExploreLogEnabled();

/**
 * @brief Record the configuration NeuroPress chose for one chunk.
 *
 * @param packed_preset Preset with shuffle and quantize state packed in, as
 *   stored in the blob header -- the log decodes it into its own columns.
 * @param predicted What the model expected, so the row carries prediction and
 *   outcome side by side and a MAPE needs no second file.
 * @param checksum FNV-1a of the chunk, so a comparison against another
 *   implementation can prove the two saw the same bytes rather than assume it.
 */
void LogNeuroPressSelection(const std::string &blob_name, size_t chunk_size,
                            double entropy, double mad, double second_deriv,
                            int wire_lib, int packed_preset,
                            const CompressionStats *predicted,
                            double actual_ratio, double actual_ct_ms,
                            double actual_psnr, unsigned long long checksum);

/**
 * @brief Record one alternative measured during an exploration sweep.
 *
 * The selection log holds one row per chunk, which is the right shape for
 * auditing selection and the wrong one for auditing exploration: the
 * interesting question is what the other 30 candidates cost, and by the time
 * a selection row is written the sweep has not run.
 */
void LogNeuroPressExplore(const std::string &blob_name, size_t chunk_size,
                          int rank, const std::string &lib_name,
                          uint32_t preset_id, bool quantize, uint32_t shuffle,
                          double pred_ratio, double pred_ct_ms, double ratio,
                          double ct_ms, double psnr_db, double cost,
                          double primary_cost, bool adopted);

/**
 * @brief Hash the bytes a codec actually produced for one chunk.
 *
 * Written from Compress(), where those bytes exist, rather than passed up to
 * DynamicSchedule: the two run as separate tasks and need not share a thread,
 * and keying on the blob name lets a comparison join them afterwards with no
 * cross-task plumbing.
 *
 * @param stage "primary" for what the selected codec produced, "adopted" for
 *   what exploration replaced it with. A chunk can appear twice and the LAST
 *   row describes what is on the tier -- a consumer that SUMS rows rather than
 *   keying on blob name will double-count and under-report exploration.
 */
void LogCompressedPayload(const std::string &blob_name, const char *payload,
                          size_t payload_size, bool on_device, bool beneficial,
                          double compress_kernel_ms, const char *stage);

}  // namespace clio::cte::compressor

#endif  // CLIO_CTE_COMPRESSOR_NEUROPRESS_TELEMETRY_H_
