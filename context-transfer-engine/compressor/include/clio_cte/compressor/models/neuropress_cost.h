/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file neuropress_cost.h
 * @brief NeuroPress's cost model. Own header: bridge and runtime both need it,
 * and neuropress_bridge.h already includes compressor_runtime.h.
 */
#ifndef CLIO_CTE_COMPRESSOR_MODELS_NEUROPRESS_COST_H_
#define CLIO_CTE_COMPRESSOR_MODELS_NEUROPRESS_COST_H_

#include <clio_runtime/types.h>

#include <algorithm>
#include <cmath>

namespace clio::cte::compressor {

/** Resolved parameters for cost model weights. */
struct NeuroPressCostWeights {
  double ct, dt, io, bw;
};

/** Weights after any CLIO_NEUROPRESS_COST_W_* override.
 *  Ranking and SGD gate must both read these, or training scores
 *  what it is not ranking on. */
NeuroPressCostWeights NeuroPressResolvedCostWeights();

/** w_ct*ct + w_dt*dt + w_io*bytes/(ratio*bw). No time floor or ratio cap.
 *  Ratio <= 0 or non-finite gives 1e30 (sentinel). */
struct NeuroPressCost {
  double w_ct;
  double w_dt;
  double w_io;
  double bandwidth_bytes_per_ms;
  clio::run::u64 chunk_size;

  double operator()(double compress_ms, double decompress_ms,
                    double ratio) const {
    return Eval(compress_ms, decompress_ms, ratio);
  }

  /** Public so the struct stays an aggregate. */
  double Eval(double compress_ms, double decompress_ms, double ratio) const {
    const double ct = std::max(0.0, compress_ms);
    const double dt = std::max(0.0, decompress_ms);
    // Guard non-finite or <= 0 ratio with sentinel.
    bool is_ratio_valid = std::isfinite(ratio) && ratio > 0.0;
    return w_ct * ct + w_dt * dt +
           (is_ratio_valid ? w_io * static_cast<double>(chunk_size) /
                                 (ratio * bandwidth_bytes_per_ms)
                       : 1e30);
  }
};

}  // namespace clio::cte::compressor

#endif  // CLIO_CTE_COMPRESSOR_MODELS_NEUROPRESS_COST_H_
