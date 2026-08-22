/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file neuropress_train_device_stats_parity.cu
 * @brief The SGD must train from the statistics the selection already
 *        computed, as upstream hands one AutoStatsGPU to both.
 *
 * The features carry deliberately WRONG statistics and the weights must still
 * match the host-matrix path: agreement alone cannot tell "read from the
 * device" from "read from a host copy that agreed".
 */

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "clio_ctp/compress/model/neuropress_nn_predictor.h"
#include "clio_ctp/compress/preprocess/data_stats_gpu.h"

using ctp::compress::model::CompressionFeatures;
using ctp::compress::model::NeuroPressNNPredictor;
using ctp::compress::model::TrainingLabels;

namespace {

int g_failures = 0;

void Check(bool cond, const std::string &what) {
  if (cond) return;
  ++g_failures;
  std::printf("  [FAIL] %s\n", what.c_str());
}

/** Deterministic chunk, matching the sibling harnesses' generator. */
std::vector<float> MakeChunk(size_t n, int regime) {
  std::vector<float> v(n);
  uint32_t s = 0x9e3779b9u ^ static_cast<uint32_t>(regime * 2654435761u);
  for (size_t i = 0; i < n; ++i) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    const double u = static_cast<double>(s) / 4294967296.0;
    switch (regime) {
      case 0: v[i] = static_cast<float>(u * 2.0 - 1.0); break;
      case 1: v[i] = static_cast<float>(std::sin(i * 0.0005) + 0.02 * u); break;
      case 2: v[i] = static_cast<float>(i % 97) * 0.25f; break;
      default: v[i] = (i % 1024 < 512) ? 0.0f : static_cast<float>(u); break;
    }
  }
  return v;
}

/** One candidate's features. Slots 5-7 are the ones under test. */
CompressionFeatures MakeFeatures(size_t chunk_bytes, double entropy, double mad,
                                 double deriv, int lib_id) {
  CompressionFeatures f;
  f.chunk_size_bytes = static_cast<double>(chunk_bytes);
  f.shannon_entropy = entropy;
  f.mad = mad;
  f.second_derivative_mean = deriv;
  f.library_config_id = lib_id;
  f.data_type_float = 1;
  f.quantize = 0;
  f.byte_shuffle = 0;
  f.error_bound = 0.0;
  return f;
}

/** Fresh weights, one SGD step, and the resulting parameters. */
bool TrainOnce(const std::vector<CompressionFeatures> &feats,
               const std::vector<TrainingLabels> &labels,
               const void *device_stats, std::vector<float> *out_w,
               std::vector<float> *out_b) {
  NeuroPressNNPredictor p;
  if (!p.Load(CLIO_CTP_NEUROPRESS_WEIGHTS_DIR)) return false;
  if (!p.TrainDeviceStats(feats, labels, device_stats)) return false;
  // The SGD is fire-and-forget, so let it land before reading.
  cudaDeviceSynchronize();
  *out_w = p.DebugWeights();
  *out_b = p.DebugBiases();
  return true;
}

size_t CountDiffs(const std::vector<float> &a, const std::vector<float> &b) {
  if (a.size() != b.size()) return a.size() + b.size();
  size_t n = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    // Bit-exact: same kernel, same inputs -- not a rounding difference.
    if (a[i] != b[i]) ++n;
  }
  return n;
}

}  // namespace

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::printf("No CUDA device -- skipping.\n");
    return 77;
  }
  {
    NeuroPressNNPredictor probe;
    if (!probe.Load(CLIO_CTP_NEUROPRESS_WEIGHTS_DIR)) {
      std::printf("Could not load NeuroPress weights -- skipping.\n");
      return 77;
    }
  }

  const size_t kElems = 1u << 20;  // 4 MiB of float32
  const size_t kBytes = kElems * sizeof(float);

  for (int regime = 0; regime < 4; ++regime) {
    char tag[64];
    std::snprintf(tag, sizeof(tag), "regime %d", regime);

    const std::vector<float> host = MakeChunk(kElems, regime);
    float *d_data = nullptr;
    if (cudaMalloc(&d_data, kBytes) != cudaSuccess) {
      std::printf("  [SKIP] %s: cudaMalloc failed\n", tag);
      continue;
    }
    cudaMemcpy(d_data, host.data(), kBytes, cudaMemcpyHostToDevice);

    // Computed once, left on the device.
    void *stream = ctp::DeviceStatsStream();
    const void *d_stats = ctp::ComputeDeviceStatsResident(
        d_data, kElems, ctp::DataType::FLOAT32, stream);
    Check(d_stats != nullptr, std::string(tag) + ": resident stats ran");
    if (d_stats == nullptr) {
      cudaFree(d_data);
      continue;
    }

    double entropy = 0, mad = 0, deriv = 0;
    Check(ctp::ReadDeviceFeatureStats(d_stats, &entropy, &mad, &deriv, stream),
          std::string(tag) + ": stats readable");

    const std::vector<TrainingLabels> labels = {
        TrainingLabels(3.5f, 120.0f, 8.0f, 4.0f),
        TrainingLabels(2.1f, 120.0f, 3.0f, 2.0f)};

    // Reference: the host-matrix path with the true statistics.
    const std::vector<CompressionFeatures> true_feats = {
        MakeFeatures(kBytes, entropy, mad, deriv, 302),
        MakeFeatures(kBytes, entropy, mad, deriv, 312)};
    std::vector<float> ref_w, ref_b;
    Check(TrainOnce(true_feats, labels, nullptr, &ref_w, &ref_b),
          std::string(tag) + ": host-feature training ran");

    // Features whose statistics are WRONG: if the kernel reads them instead
    // of the device buffer, this fails.
    const std::vector<CompressionFeatures> poisoned = {
        MakeFeatures(kBytes, entropy + 3.0, mad * 7.0 + 1.0, deriv * 11.0 + 1.0,
                     302),
        MakeFeatures(kBytes, entropy - 2.0, mad * 0.1 - 1.0, deriv * 0.2 - 1.0,
                     312)};
    std::vector<float> dev_w, dev_b;
    Check(TrainOnce(poisoned, labels, d_stats, &dev_w, &dev_b),
          std::string(tag) + ": device-stats training ran");

    const size_t wdiff = CountDiffs(ref_w, dev_w);
    const size_t bdiff = CountDiffs(ref_b, dev_b);
    Check(wdiff == 0, std::string(tag) + ": weights identical (" +
                          std::to_string(wdiff) + " differ)");
    Check(bdiff == 0, std::string(tag) + ": biases identical (" +
                          std::to_string(bdiff) + " differ)");

    // Control: they must MATTER as the only source, or a Train() ignoring
    // slots 5-7 would pass everything above.
    std::vector<float> poisoned_w, poisoned_b;
    Check(TrainOnce(poisoned, labels, nullptr, &poisoned_w, &poisoned_b),
          std::string(tag) + ": poisoned host-feature training ran");
    const size_t control = CountDiffs(ref_w, poisoned_w);
    Check(control > 0,
          std::string(tag) +
              ": poisoned statistics change the update when read from the "
              "host (control)");

    std::printf(
        "  %s: entropy %.9f mad %.9f d2 %.9f | dev-vs-host w:%zu b:%zu "
        "| control differs in %zu weights\n",
        tag, entropy, mad, deriv, wdiff, bdiff, control);

    cudaFree(d_data);
  }

  std::printf("\n===== %s =====\n",
              g_failures == 0 ? "0 failures" : "FAILURES PRESENT");
  return g_failures == 0 ? 0 : 1;
}
