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

namespace clio::cte::compressor {

/** Resolved parameters. `cap` is upstream's RATIO_CAP (100). */
struct NeuroPressCostWeights {
  double ct, dt, io, bw, cap;
};

/** Weights after any CLIO_NEUROPRESS_COST_W_* override. Ranking and SGD gate
 *  must both read these, or training scores what it is not ranking on. */
NeuroPressCostWeights NeuroPressResolvedCostWeights();

/** w_ct*ct + w_dt*dt + w_io*bytes/(min(ratio,cap)*bw). Times floored at 1 ms
 *  and ratio capped first, as upstream. Ratio <= 0 gives 1e30, a gate sentinel. */
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
