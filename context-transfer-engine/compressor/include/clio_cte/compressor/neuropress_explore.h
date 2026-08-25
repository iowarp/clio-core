/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file neuropress_explore.h
 * @brief The state an exploration sweep works on, and the step that collects
 * its results.
 *
 * A sweep re-compresses one already-stored chunk with up to K of the model's
 * next-best actions and keeps the winner if one beats what the primary
 * achieved. It runs in three steps -- prepare every slot, collect every slot,
 * score them in rank order -- and the steps hand these types to one another,
 * so they are declared here rather than inside the function that drives them.
 *
 * These are NOT part of compressing a blob. The compressor runtime reaches
 * them only from its exploration path, which is off unless NeuroPress asked
 * for it.
 */
#ifndef CLIO_CTE_COMPRESSOR_NEUROPRESS_EXPLORE_H_
#define CLIO_CTE_COMPRESSOR_NEUROPRESS_EXPLORE_H_

#include <clio_ctp/compress/compress.h>
#include <clio_ctp/compress/nvcomp.h>
#include <clio_ctp/compress/preprocess/quantization.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "clio_cte/compressor/compressor_runtime.h"

namespace clio::cte::compressor {

/**
 * One row of the per-candidate sweep log.
 *
 * Rows are held until the scoring loop ends rather than written as each
 * candidate is measured, because `adopted` is settled only by the LAST
 * candidate to beat the running best -- a row emitted at measurement time
 * would claim adoption for every successive winner.
 */
struct ExploreRow {
  std::string lib;
  uint32_t preset_id;
  bool quant;
  uint32_t shuffle;
  double pred_ratio, pred_ct, ratio, ct_ms, psnr, cost;
  int rank;
};

/**
 * The best alternative a sweep has found so far, and everything needed to
 * store it if it survives to the end.
 *
 * One object rather than ten locals because these are written together, at a
 * single point in the scoring loop, and read together by the block that
 * commits the winner. Kept apart, a candidate could be adopted with one field
 * left behind from an earlier one -- and the quantization pair is exactly
 * where that goes unnoticed: a blob whose header says "not quantized" while
 * its payload is quantized is unrecoverable on read, and nothing about the
 * write reports a problem.
 */
struct ExploreWinner {
  bool have = false;
  std::vector<char> payload;
  int lib = 0;
  uint32_t preset_id = 0;
  uint32_t shuffle = 0;
  double ratio = 0.0;
  double time_ms = 0.0;
  bool quant = false;
  ctp::compress::preprocess::DeviceQuantizeParams quant_params;
  /** Index into the sweep-log rows, so the last adopter carries the flag. */
  int row = -1;
};

/**
 * One alternative measured during an exploration sweep: everything the
 * codec call needs, and everything the scoring phase reads back from it.
 *
 * At file scope rather than inside DynamicSchedule because the sweep's
 * three phases hand slots to one another, and a type they can all name
 * is easier to follow than one declared ten levels deep inside the single
 * function that uses it.
 */
// Upstream launches its K alternatives on K separate CUDA streams and
// only then syncs and scores them ("Parallel exploration: K
// alternatives on K separate streams", gpucompress_compress.cpp).
// The same three phases are kept here, with one deliberate
// narrowing: only the CODEC CALL is parallel. Preparation stays on
// this thread because it allocates and registers GPU backends
// through the runtime, and scoring stays on it because the winner
// comparison has to be order-deterministic -- a tie decided by which
// worker happened to finish first would make the stored blob vary
// run to run.
struct ExploreSlot {
  const CompressionStats* alt = nullptr;
  std::string name;
  uint32_t preset_id = 0;
  uint32_t applied_shuffle = 0;
  bool applied_quant = false;
  ctp::compress::preprocess::DeviceQuantizeParams quant_params;
  std::unique_ptr<ctp::Compressor> compressor;
  // Buffers the codec call reads or writes. They are MOVED in from
  // the prep loop's locals; moving a std::vector transfers the heap
  // block rather than copying it, so `input`/`out_ptr` keep pointing
  // at the right bytes without any fixup.
  std::vector<char> device_staging, quant_staging, shuffle_staging,
      output;
  char* input = nullptr;
  size_t compress_size = 0;
  char* out_ptr = nullptr;
  size_t capacity = 0;
  // Filled by the launch/collect phases below.
  bool ok = false;
  bool launched = false;
  size_t compressed_size = 0;
  double time_ms = 0.0;
#if CTP_ENABLE_COMPRESS && CTP_ENABLE_NVCOMP
  ctp::NvComp* gpu = nullptr;   ///< non-null when the async path ran
  ctp::NvComp::AsyncSlot async;  ///< this slot's stream + events
#endif
};

/**
 * Sweep step 2: finish every launched slot and record what it produced.
 *
 * Separated from the launch step because the two must not interleave --
 * every slot has to be in flight before any is waited on, or the sweep
 * serializes into K sequential codec calls and stops measuring what it
 * was written to measure.
 *
 * Takes only the slots: what a codec produced is entirely recorded in
 * them, so this step needs nothing about the chunk or the selection.
 */
void CollectExploreSlots(std::vector<std::unique_ptr<ExploreSlot>> &slots);

}  // namespace clio::cte::compressor

#endif  // CLIO_CTE_COMPRESSOR_NEUROPRESS_EXPLORE_H_
