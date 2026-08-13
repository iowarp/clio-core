/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file comparison.h
 * @brief Compares two callback traces stage by stage and locates the FIRST
 *        point of divergence.
 *
 * The normalization rule, stated once because it is the part that could
 * quietly turn this harness into a rubber stamp: only entries classified
 * StageKind::kClioArchitectural are removed, that classification is assigned
 * by a closed switch in KindOf(), and every removed entry is reported by name
 * with its justification. No NeuroPress functional stage -- statistics,
 * diagnostics, inference, ranking, selection, quantization, shuffle,
 * compression -- can be normalized away, and a difference in any of them is a
 * failure, never a note.
 */
#ifndef CLIO_CTE_COMPRESSOR_EXAMPLE_NP_EQUIV_COMPARISON_H_
#define CLIO_CTE_COMPRESSOR_EXAMPLE_NP_EQUIV_COMPARISON_H_

#include <string>
#include <vector>

#include "callback_trace.h"
#include "device_probe.h"

namespace npeq {

/** @brief The device buffers one side produced, for byte-level comparison. */
struct SideArtifacts {
  const void *input = nullptr;
  size_t input_bytes = 0;
  const void *quantized = nullptr;
  size_t quantized_bytes = 0;
  const void *shuffled = nullptr;
  size_t shuffled_bytes = 0;

  /** Quantize -> dequantize round trip, and its verdict against the bound. */
  const void *dequantized = nullptr;
  size_t dequantized_bytes = 0;
  bool bound_checked = false;
  unsigned long long bound_violations = 0;
  double requested_error_bound = 0.0;
  const void *payload = nullptr;
  size_t payload_bytes = 0;

  /**
   * The payload from a SECOND run of the same side over the same chunk.
   *
   * Without this, "native and Clio produced different bytes" is ambiguous
   * between a real divergence and a codec that is not reproducible even against
   * itself -- and nvcomp is known to have output that depends on manager reuse
   * history (recorded as D18-2 for ANS). Comparing a side against itself first
   * is what makes the cross-side comparison interpretable.
   */
  const void *payload_repeat = nullptr;
  size_t payload_repeat_bytes = 0;
};

/** @brief One named check within a stage. */
struct CheckResult {
  std::string name;
  bool pass = true;
  /** True when the check could not be run (missing data on one side). */
  bool skipped = false;
  std::string detail;
};

/** @brief Everything compared for one stage. */
struct StageComparison {
  Stage stage = Stage::kStatistics;
  bool present_native = false;
  bool present_clio = false;
  Status status_native = Status::kExecuted;
  Status status_clio = Status::kExecuted;
  std::vector<CheckResult> checks;

  bool Pass() const;
  bool AnyRan() const;
};

/** @brief The comparison for one chunk. */
struct ChunkComparison {
  int chunk_id = -1;
  std::string regime;
  size_t chunk_bytes = 0;
  double error_bound = 0.0;

  std::vector<std::string> native_sequence;
  std::vector<std::string> clio_sequence;
  std::vector<std::string> clio_normalized;
  std::vector<std::string> clio_architectural;
  bool sequence_pass = false;

  std::vector<StageComparison> stages;

  /** Index into `stages` of the first stage that failed, or -1. */
  int first_divergent_stage = -1;
  std::string first_divergence;

  ByteCompareResult input_compare;
  ByteCompareResult quantized_compare;
  ByteCompareResult shuffled_compare;
  ByteCompareResult payload_compare;
  ByteCompareResult dequantized_compare;
  /** Each side against a second run of ITSELF over the same chunk. */
  ByteCompareResult native_self_compare;
  ByteCompareResult clio_self_compare;

  bool native_input_device = false;
  bool clio_input_device = false;

  size_t native_production_d2h = 0;
  size_t clio_production_d2h = 0;
  size_t native_production_h2d = 0;
  size_t clio_production_h2d = 0;
  int native_production_d2h_count = 0;
  int clio_production_d2h_count = 0;
  int native_production_h2d_count = 0;
  int clio_production_h2d_count = 0;
  size_t harness_transfer_bytes = 0;

  /**
   * Production transfers big enough to be the CHUNK rather than a scalar or a
   * header. This is the "no unexpected production D->H->D" requirement made
   * measurable: a stage that moved the payload through host memory would show
   * up here and nowhere else, since the small metadata copies both sides make
   * are legitimate and would otherwise drown it out.
   */
  int native_bulk_transfers = 0;
  int clio_bulk_transfers = 0;
  std::vector<std::string> bulk_transfer_detail;

  int native_kernel_launches = 0;
  int clio_kernel_launches = 0;
  /** A functional stage that ran with zero kernels is a CPU-fallback signal. */
  std::vector<std::string> native_stages_without_kernels;
  std::vector<std::string> clio_stages_without_kernels;

  bool pass = false;
};

/**
 * @brief Compare one chunk's two traces.
 *
 * Byte comparisons are performed ON THE DEVICE (device_probe.h) and are
 * bracketed as harness activity so their transfers cannot be confused with
 * production ones.
 */
ChunkComparison CompareChunk(const ChunkTrace &native, const ChunkTrace &clio,
                             const SideArtifacts &native_art,
                             const SideArtifacts &clio_art);

bool WriteComparisonJson(const std::string &path,
                         const std::vector<ChunkComparison> &comparisons);

/**
 * @brief The human-readable callback trace report.
 *
 * Includes the per-chunk callback walk, the per-callback PASS/FAIL, the first
 * divergence, and the final matrix.
 */
bool WriteReport(const std::string &path,
                 const std::vector<ChunkTrace> &native_traces,
                 const std::vector<ChunkTrace> &clio_traces,
                 const std::vector<ChunkComparison> &comparisons,
                 const std::string &preamble);

/** @brief Same report, to a stream, so the run also prints it. */
void PrintReport(std::ostream &os,
                 const std::vector<ChunkTrace> &native_traces,
                 const std::vector<ChunkTrace> &clio_traces,
                 const std::vector<ChunkComparison> &comparisons,
                 const std::string &preamble);

}  // namespace npeq

#endif  // CLIO_CTE_COMPRESSOR_EXAMPLE_NP_EQUIV_COMPARISON_H_
