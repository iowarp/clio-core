/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file neuropress_cost.h
 * @brief NeuroPress's cost model: the resolved weights, and the one function
 * that scores an outcome with them.
 *
 * Its own header because two headers need it and neither can include the
 * other -- neuropress_bridge.h already includes compressor_runtime.h, so the
 * runtime cannot reach back for the bridge's declarations. The ranking (in
 * the bridge) and the online-learning gate and exploration sweep (in the
 * runtime) all have to score with IDENTICAL weights, so the type they share
 * belongs to neither of them.
 */
#ifndef CLIO_CTE_COMPRESSOR_MODELS_NEUROPRESS_COST_H_
#define CLIO_CTE_COMPRESSOR_MODELS_NEUROPRESS_COST_H_

#include <clio_runtime/types.h>

#include <algorithm>

namespace clio::cte::compressor {

/**
 * Resolved cost-model parameters. `cap` is the compression-ratio ceiling the
 * cost model applies (upstream's RATIO_CAP, 100). It is exposed here so the
 * adoption cost and the exploration gate use one value rather than each
 * hardcoding it.
 */
struct NeuroPressCostWeights {
  double ct, dt, io, bw, cap;
};

/**
 * @brief The cost-model weights the ranking actually used, after any
 *        CLIO_NEUROPRESS_COST_W_* override.
 *
 * Upstream keeps ONE set of weights (g_rank_w0/w1/w2, g_measured_bw_bytes_per_ms)
 * and reads them both in the ranking kernel and in the cost it gates SGD on
 * (gpucompress_compress.cpp:662-664). Clio had the override reach the ranking
 * only, so a "ratio cost model" run changed what was selected while the gate
 * kept scoring the balanced cost -- training the model against an objective it
 * was not ranking on. This exposes the resolved values so both agree.
 */
NeuroPressCostWeights NeuroPressResolvedCostWeights();

/**
 * NeuroPress's weighted cost of one outcome:
 *   w_ct*compress_ms + w_dt*decompress_ms + w_io*bytes/(min(ratio,cap)*bw)
 *
 * An object rather than loose constants and a lambda because the SGD gate and
 * the exploration sweep have to score with the SAME weights. That is the
 * entire reason NeuroPressResolvedCostWeights() exists: when the
 * CLIO_NEUROPRESS_COST_W_* override reached only the ranking, a "ratio cost
 * model" run selected on ratio while the gate still scored the balanced cost,
 * training the model against an objective it was not ranking on. Passing one
 * callable around makes that mismatch impossible to reintroduce by editing a
 * weight at one site and not the other.
 *
 * Times are floored at 1 ms and the ratio capped, both before weighting,
 * exactly as upstream clamps them (gpucompress_compress.cpp). A non-positive
 * ratio yields 1e30 -- "no measurable result", a sentinel the MAPE gate
 * reads, not an arithmetic accident.
 */
struct NeuroPressCost {
  double w_ct;
  double w_dt;
  double w_io;
  double bandwidth_bytes_per_ms;
  double ratio_cap;
  clio::run::u64 chunk_size;

  double operator()(double compress_ms, double decompress_ms,
                    double ratio) const {
    const double ct = std::max(1.0, compress_ms);
    const double dt = std::max(1.0, decompress_ms);
    const double rc = std::min(ratio_cap, ratio);
    return w_ct * ct + w_dt * dt +
           ((rc > 0.0) ? w_io * static_cast<double>(chunk_size) /
                             (rc * bandwidth_bytes_per_ms)
                       : 1e30);
  }
};

}  // namespace clio::cte::compressor

#endif  // CLIO_CTE_COMPRESSOR_MODELS_NEUROPRESS_COST_H_
