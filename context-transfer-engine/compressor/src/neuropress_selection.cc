/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file neuropress_selection.cc
 * @brief Runtime::NeuroPressRankChunk -- everything that happens when
 * NeuroPress, rather than Clio's own heuristics, decides how a chunk is
 * compressed.
 *
 * This is the model's half of EstCompressionStats, which used to hold both
 * halves interleaved: one data_type expression, one features_ok expression and
 * one ranking gate each carrying a `neuropress_active ? ... : ...` branch. The
 * two halves do not share a model, a feature definition, or even a way of
 * reading the same bytes -- NeuroPress reads every chunk as float32 because
 * its statistics kernel is typed that way, while the legacy models use the
 * context's own type mapping -- so interleaving them meant every edit to one
 * had to be checked against the other.
 *
 * Separated, each half states its own preconditions. The caller decides which
 * one runs; an empty return here means "NeuroPress declined", and the legacy
 * heuristics take the chunk with the statistics this pass already computed.
 */

#include <clio_ctp/compress/compress_factory.h>
#include <clio_ctp/compress/preprocess/data_stats.h>
#include <clio_ctp/compress/preprocess/data_stats_gpu.h>
#include <clio_ctp/compress/preprocess/feature_extractor.h>

#include <vector>

#include "clio_cte/compressor/compressor_runtime.h"
#include "clio_cte/compressor/models/neuropress_bridge.h"

namespace clio::cte::compressor {

std::vector<CompressionStats> Runtime::NeuroPressRankChunk(
    const void* chunk, clio::run::u64 chunk_size, const Context& context,
    double* entropy, double* mad, double* second_derivative_mean,
    double* out_entropy, double* out_mad, double* out_second_deriv,
    bool* out_neuropress_gpu_failed, const void** out_device_stats) {
  // NeuroPress reads every chunk as float32: its statistics kernel is typed
  // `const float*` with no data-type parameter, and the shipped model.nnwt was
  // normalized against float32 statistics. Reading the same bytes as uint8 puts
  // MAD hundreds of sigma outside the training range. The legacy qtable/dense-NN
  // path keeps the context-driven mapping -- those models were fit on Clio's own
  // features, not NeuroPress's.
  const ctp::DataType data_type = ctp::DataType::FLOAT32;
  size_t type_size = ctp::DataStatisticsFactory::GetTypeSize(data_type);

  // Whole chunk, not a prefix: NeuroPress grid-strides the entire buffer and
  // the model was normalized against whole-chunk statistics. Prefix statistics
  // paired with the full chunk_size never occur in training, and every
  // candidate is scored from these same three numbers.
  size_t num_elements = static_cast<size_t>(chunk_size / type_size);

  // Calculate compression features. A chunk resolved from a CUDA-IPC device
  // buffer is not host-readable -- ComputeCompressionFeatures detects that
  // and computes entropy/MAD/second-derivative on-device, so the buffer
  // itself never has to be staged through host memory just to feed
  // NeuroPress. Falls through to the existing host path otherwise.
  // A chunk smaller than one element has no statistics; NeuroPress refuses
  // outright, so fall through to the legacy heuristics rather than reading
  // past the allocation.

  // Device-resident chunk with NeuroPress ranking: keep the statistics ON the
  // GPU and let the network read them there, as upstream does. Nothing is
  // synchronized here on purpose: the statistics kernels are enqueued and the
  // ranking below chains its inference onto the SAME stream, so the decision
  // runs as one asynchronous chain with a single wait at the end.
  const void *device_stats = nullptr;
  void *np_stream = nullptr;
  const bool np_device_path =
      num_elements > 0 && ctp::IsDevicePointer(chunk);
  if (np_device_path) {
    np_stream = ctp::DeviceStatsStream();
    /* A device-resident float64 chunk is CONVERTED on the GPU, not
       reinterpreted -- `data_type` above is pinned to FLOAT32 for the model's
       normalisation, and applying that to doubles reads each one as two
       float32 words whose low half is a NaN about one time in 256. Host and
       device now agree, so the features a chunk gets no longer depend on which
       memory the application happened to write it from. */
    if (context.data_type_ == 2) {
      device_stats = ctp::ComputeDeviceStatsResidentF32From64(
          chunk, chunk_size / sizeof(double), np_stream);
    } else {
      device_stats = ctp::ComputeDeviceStatsResident(chunk, num_elements,
                                                     data_type, np_stream);
    }
    /* Reused below instead of re-measuring; upstream passes one d_stats_ptr
       to both inference and runNNSGDCtx. */
    if (out_device_stats) *out_device_stats = device_stats;
  }

  // When the device path was the right one, it is the ONLY one. Computing the
  // same three features on the host instead would run stages upstream only
  // ever runs on the GPU, and would do it invisibly -- the selection would
  // come out looking entirely normal. Upstream propagates the failure rather
  // than substituting anything (a null d_stats_ptr gives
  // GPUCOMPRESS_ERROR_NN_NOT_LOADED, gpucompress_compress.cpp), so this
  // reports it and declines NeuroPress for the chunk.
  /* Host path with NeuroPress: go through ComputeNeuroPressFeatures, which
     honours the element type. `data_type` above is pinned to FLOAT32 for the
     model's normalisation, and applying that to a float64 buffer relabels the
     bytes instead of converting the values -- each double reads as two float32
     words whose low half lands on the NaN encoding about one time in 256, so
     MAD and the second derivative came back NaN for every chunk of every
     float64 dataset and the model was handed NaN for two of its three inputs.
     The other callers keep the plain factory call: they are not feeding
     NeuroPress and their models were fit on Clio's own features. */
  const bool features_ok =
      np_device_path ? device_stats != nullptr
                     : ctp::ComputeNeuroPressFeatures(
                           chunk, chunk_size, context.data_type_, entropy,
                           mad, second_derivative_mean);
  if (np_device_path && device_stats == nullptr) {
    HLOG(kError,
         "EstCompressionStats: device-resident statistics failed for a "
         "NeuroPress chunk (size={}); NOT recomputing them on the host",
         chunk_size);
    if (out_neuropress_gpu_failed) *out_neuropress_gpu_failed = true;
  }
  if (features_ok && device_stats == nullptr) {
    // Report the statistics the ranking below is about to use, so a caller
    // can record what a selection was actually based on.
    if (out_entropy) *out_entropy = *entropy;
    if (out_mad) *out_mad = *mad;
    if (out_second_deriv) *out_second_deriv = *second_derivative_mean;
  }
  if (!features_ok) {
    // Device-resident chunk whose on-device stats could not be computed.
    // Ranking on the zeros left behind would hand NeuroPress a chunk that
    // looks perfectly compressible and pick accordingly, so skip the
    // model entirely and let the legacy heuristics below decide.
    HLOG(kWarning,
         "EstCompressionStats: no usable statistics for this chunk "
         "(size={} elem_size={}); skipping NeuroPress ranking for it",
         chunk_size, type_size);
  }

  // Dynamic mode: NeuroPress takes priority over the legacy qtable/dense-NN
  // heuristics below. It ranks the model candidate set restricted to its own
  // trained 8-algorithm nvcomp action space (see NeuroPressCandidateStats);
  // every candidate is GPU-native, so a device-resident chunk never forces a
  // host round-trip downstream in Compress().
  if (!features_ok) return {};

  {
    bool data_type_float = (context.data_type_ == 1);
    std::vector<CompressionStats> neuropress_stats;
    if (device_stats != nullptr) {
      bool np_infer_failed = false;
      neuropress_stats = NeuroPressCandidateStatsDevice(
          *neuropress_predictor_, chunk_size, device_stats, np_stream,
          data_type_float, context.error_bound_, context.target_psnr_,
          &np_infer_failed, config_.neuropress_best_mode_);
      if (np_infer_failed && out_neuropress_gpu_failed) {
        *out_neuropress_gpu_failed = true;
      }
      // The stream is idle now -- the ranking waited on it once -- so this is
      // a 24-byte copy and no additional stall. Ungated: the per-chunk record
      // needs these on every chunk, not only when the selection log is on.
      // The statistics themselves stay on the device.
      if (!ctp::ReadDeviceFeatureStats(device_stats, entropy, mad,
                                       second_derivative_mean, np_stream)) {
        *entropy = *mad = *second_derivative_mean = 0.0;
      }
      if (out_entropy) *out_entropy = *entropy;
      if (out_mad) *out_mad = *mad;
      if (out_second_deriv) *out_second_deriv = *second_derivative_mean;
    } else {
      neuropress_stats = NeuroPressCandidateStats(
          *neuropress_predictor_, chunk_size, *entropy, *mad,
          *second_derivative_mean, data_type_float, context.error_bound_,
          config_.neuropress_best_mode_);
    }
    // PSNR filtering, HOST path only. The device path applies the floor inside
    // RankKernel where upstream applies it, and the two are not interchangeable:
    // masking to -INFINITY leaves the candidate ranked last but still
    // selectable, whereas removing it here can empty the list. Upstream always
    // returns an action even when every action is masked.
    if (device_stats == nullptr && context.target_psnr_ > 0) {
      std::vector<CompressionStats> filtered;
      filtered.reserve(neuropress_stats.size());
      for (const auto& stat : neuropress_stats) {
        // No `psnr_db_ > 0 &&` guard: upstream masks purely on
        // `psnr < min_psnr` (nn_gpu.cu), and its psnr is already clamped
        // to [0, 120], so a candidate predicted at exactly 0 dB is rejected.
        // The extra guard kept those -- and 0.0 is precisely what a failed
        // inference leaves behind.
        if (stat.psnr_db_ < context.target_psnr_) continue;
        filtered.push_back(stat);
      }
      neuropress_stats = std::move(filtered);
    }
    if (!neuropress_stats.empty()) {
      HLOG(kDebug,
           "NeuroPress dynamic selection: chunk_size={} entropy={} mad={} "
           "-> top pick wire_id={} ({} candidates ranked)",
           chunk_size, *entropy, *mad, neuropress_stats.front().compress_lib_,
           neuropress_stats.size());
      return neuropress_stats;
    }
    // Fall through to the legacy heuristics below if NeuroPress produced no
    // usable candidates. That is a change of MODEL, not a relocation of
    // NeuroPress's own work to the host -- the legacy heuristics are Clio's
    // and have always been host code. It still gets said out loud on the
    // device path, because there the empty result means GPU inference
    // declined, and a selection silently made by a different model is
    // indistinguishable downstream from one NeuroPress made.
    if (device_stats != nullptr) {
      HLOG(kError,
           "EstCompressionStats: NeuroPress produced no candidates for a "
           "device-resident chunk (size={}); NOT re-ranking it on the host",
           chunk_size);
    } else {
      // Host-resident chunks rank through NeuroPress too -- same gate, same
      // RankIntoStats -- so an empty list here is the same model hand-off
      // the device branch above reports, and was previously the only one
      // that happened silently. It is a warning rather than an error
      // because nothing failed: the legacy heuristics are Clio's own and
      // will produce a valid selection. What must not stay invisible is
      // WHICH model chose it, since downstream a legacy pick is
      // indistinguishable from a NeuroPress one.
      HLOG(kWarning,
           "EstCompressionStats: NeuroPress produced no candidates for a "
           "host-resident chunk (size={} entropy={} mad={}); falling back "
           "to Clio's legacy heuristics -- this selection is NOT "
           "NeuroPress's",
           chunk_size, *entropy, *mad);
    }
  }
  return {};
}

}  // namespace clio::cte::compressor
