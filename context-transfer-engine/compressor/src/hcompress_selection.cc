/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * @file hcompress_selection.cc
 * @brief HCompress's Expected-Compression-Cost model, deployed as the
 * chunk's selector.
 *
 * HCompress (Devarajan et al., IPDPS 2020, Sec. IV-D) predicts a codec's cost
 * from four CATEGORICAL inputs -- data type, data format, library, and a data
 * distribution class "deduced by sub-sampling the buffer". This file is where
 * that model chooses a chunk's codec at write time, so figure 9 can draw it as
 * a bar beside NeuroPress.
 *
 * ONLY THE PREDICTOR DIFFERS FROM A NEUROPRESS SELECTION. HCompress ranks
 * through NeuroPressCandidateStats -- NeuroPress's own 32-action nvcomp
 * candidate set, its NeuroPressCost weights and its tie order -- with
 * HCompress's predictions in place of the network's. Ranking a different
 * candidate set (the legacy CPU-codec path) would compare two different
 * questions, not two predictors.
 *
 * WHAT IT IS GIVEN IS WHAT THE PAPER GIVES IT. The per-chunk statistics
 * NeuroPress uses (entropy, MAD, second derivative) are not HCompress inputs
 * and are not passed: hcompress_ccp_predictor.h warns that feeding them would
 * quietly repair the very property the comparison exists to expose.
 */

#include <clio_ctp/compress/compress_factory.h>
#include <clio_ctp/compress/model/hcompress_ccp_predictor.h>
#include <clio_ctp/util/gpu_api.h>

#include <chrono>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "clio_cte/compressor/compressor_runtime.h"
#include "clio_cte/compressor/models/distribution_classifier.h"
#include "clio_cte/compressor/models/neuropress_bridge.h"
#include "clio_cte/compressor/neuropress_telemetry.h"

namespace clio::cte::compressor {

namespace {

/** Elements sub-sampled per chunk. classify_chunks.py's SAMPLE_ELEMS, so a
 *  chunk gets the same class here as it does in the offline accuracy table. */
constexpr size_t kHcSampleElems = 65536;

/** DistributionClassifier::Classify's own default, and classify_chunks.py's
 *  NUM_BINS. */
constexpr size_t kHcNumBins = 64;

/** What the seed's `data_type` column calls a context element type.
 *  The seed profiled float32 only, so any other type is a category it never
 *  saw and contributes no term -- reported, not substituted.
 *
 *  @param context_data_type Context::data_type_ (0 byte, 1 float32, 2 float64)
 *  @return the seed-vocabulary name */
const char *HcDataTypeName(int context_data_type) {
  switch (context_data_type) {
    case 1: return "float32";
    case 2: return "float64";
    default: return "uint8";
  }
}

/** The classifier's element type for a context element type.
 *  @param context_data_type Context::data_type_
 *  @return the matching DataType */
DataType HcClassifierType(int context_data_type) {
  switch (context_data_type) {
    case 1: return DataType::FLOAT32;
    case 2: return DataType::DOUBLE64;
    default: return DataType::UINT8;
  }
}

/** A STRIDED sub-sample of the whole chunk, element for element the one
 *  classify_chunks.py takes: every max(1, n/65536)-th element, at most 65536.
 *  NOT a prefix -- a prefix of a field dump is one corner of the simulation
 *  domain, whose moments can differ from the chunk's by more than the classes
 *  differ from each other.
 *
 *  A device chunk is copied to the host WHOLE and strided there, which is
 *  exactly what the offline port does (it reads the chunk, then strides).
 *  A narrow strided cudaMemcpy2D issues one transfer per element and is
 *  slower than the whole copy; a gather kernel would move only the sample,
 *  and is the follow-up if the measured classification cost warrants it.
 *
 *  @param chunk      chunk bytes, host or device resident
 *  @param chunk_size chunk size in bytes
 *  @param elem_size  bytes per element
 *  @param out        filled with the sampled elements' raw bytes
 *  @return number of elements sampled; 0 when the chunk holds none */
size_t HcStridedSample(const void *chunk, size_t chunk_size, size_t elem_size,
                       std::vector<unsigned char> *out) {
  const size_t count = chunk_size / elem_size;
  if (count == 0) return 0;
  // Per-thread staging: selection runs per chunk on worker threads, and a
  // fresh 8 MiB allocation each time would be a cost HCompress does not have.
  thread_local std::vector<unsigned char> host_copy;
  const unsigned char *src = static_cast<const unsigned char *>(chunk);
  if (ctp::IsDevicePointer(chunk)) {
    host_copy.resize(count * elem_size);
    ctp::DeviceAwareMemcpy(host_copy.data(), chunk, count * elem_size);
    src = host_copy.data();
  }
  const size_t stride = count > kHcSampleElems ? count / kHcSampleElems : 1;
  const size_t n = count > kHcSampleElems ? kHcSampleElems : count;
  out->resize(n * elem_size);
  for (size_t i = 0; i < n; ++i) {
    std::memcpy(out->data() + i * elem_size, src + i * stride * elem_size,
                elem_size);
  }
  return n;
}

/** Milliseconds since `t0`. */
double HcMsSince(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

std::vector<CompressionStats> Runtime::HCompressRankChunk(
    const void *chunk, clio::run::u64 chunk_size, const Context &context,
    std::string *out_distribution) {
  const auto t0 = std::chrono::steady_clock::now();
  const DataType type = HcClassifierType(context.data_type_);
  const size_t elem_size = (type == DataType::FLOAT32 || type == DataType::INT32)
                               ? 4
                               : (type == DataType::DOUBLE64 ? 8 : 1);

  // ---- 1. The distribution class, deduced by sub-sampling the buffer.
  std::vector<unsigned char> sample;
  const size_t n = HcStridedSample(chunk, static_cast<size_t>(chunk_size),
                                   elem_size, &sample);
  if (n == 0) {
    HLOG(kWarning,
         "HCompress: chunk of {} bytes holds no {}-byte element to classify; "
         "declining it (stored uncompressed, NOT handed to another selector)",
         chunk_size, elem_size);
    return {};
  }
  // CONSTANT is passed through as-is, not folded into a paper class: it is
  // not one of HCompress's four, the seed never profiled it, and so it
  // contributes no distribution term -- the offline evaluator's default too.
  const std::string distribution =
      DistributionClassifierFactory::Classify(sample.data(), n, type,
                                              kHcNumBins)
          .ToString();
  const double classify_ms = HcMsSince(t0);

  // ---- 2. Rank NeuroPress's candidates with HCompress's predictions.
  const auto t1 = std::chrono::steady_clock::now();
  std::vector<CompressionStats> stats;
  {
    // SetInputs() and the ranking's reads of those inputs are ONE critical
    // section: see hcompress_mutex_. No suspension happens inside it.
    std::lock_guard<std::mutex> lock(hcompress_mutex_);
    hcompress_predictor_->SetInputs(
        {HcDataTypeName(context.data_type_), /*data_format=*/"",
         /*library=*/"", distribution});
    // The balanced cost model, as the NeuroPress `learn`/`dynamic` arms it is
    // compared with use; the statistics are zeros because HCompress takes
    // none and the cost model does not read them.
    stats = NeuroPressCandidateStats(*hcompress_predictor_, chunk_size, 0.0,
                                     0.0, 0.0, context.data_type_ == 1,
                                     context.error_bound_,
                                     /*ratio_only=*/false);
  }
  const double rank_ms = HcMsSince(t1);

  // HCompress has no quality output (psnr_db < 0, "not predicted"), so a PSNR
  // target cannot be honoured: filtering on it would drop every candidate.
  if (context.target_psnr_ > 0) {
    static std::once_flag warned;
    std::call_once(warned, [] {
      HLOG(kWarning,
           "HCompress: a PSNR target is set but HCompress predicts no quality "
           "(IPDPS 2020 Sec. IV-D has no quality output); ranking ignores it");
    });
  }

  // Same per-chunk records NeuroPress writes, so the two selectors' costs
  // land in the same phases.csv columns: stats = classification (including
  // the device->host copy), nn = prediction + ranking.
  RecordSelectionTiming((classify_ms + rank_ms) * 1000.0, /*reused=*/false);
  if (PhaseLogEnabled()) {
    RecordSelectionPhases(classify_ms, rank_ms, /*choice_ms=*/0.0,
                          /*reused=*/false);
  }
  if (out_distribution) *out_distribution = distribution;
  if (!stats.empty()) {
    HLOG(kDebug,
         "HCompress selection: chunk_size={} distribution={} -> wire_id={} "
         "({} candidates; classify {} ms, rank {} ms)",
         chunk_size, distribution, stats.front().compress_lib_, stats.size(),
         classify_ms, rank_ms);
  }
  return stats;
}

void Runtime::HCompressObserve(const std::string &distribution, int wire_lib,
                               int preset_field, clio::run::u64 chunk_size,
                               const Context &context) {
  // Key the EXECUTED action the way the seed keys it: bare algorithm name,
  // then the shuffle and quantize bits that ride in the preset field.
  std::string name = ctp::CompressionFactory::NameForWireId(wire_lib);
  static const std::string kNvcompPrefix = "nvcomp-";
  if (name.compare(0, kNvcompPrefix.size(), kNvcompPrefix) == 0) {
    name.erase(0, kNvcompPrefix.size());
  }
  const uint32_t bits = static_cast<uint32_t>(preset_field);
  const bool byte_shuffle = ((bits >> 8) & 0xFFu) != 0;
  const bool quantize = ((bits >> 24) & 0x1u) != 0;

  ctp::compress::model::CcpObservation row;
  row.inputs.data_type = HcDataTypeName(context.data_type_);
  row.inputs.data_format = "";
  row.inputs.library = ctp::compress::model::HCompressCcpPredictor::LibraryKey(
      name, quantize, byte_shuffle);
  row.inputs.distribution = distribution;
  row.bytes = static_cast<double>(chunk_size);
  // A non-positive measurement means NOT MEASURED and trains nothing, so an
  // unmeasured decompression time teaches the decompression head nothing
  // rather than teaching it zero.
  row.compress_time_ms = context.actual_compress_time_ms_;
  row.decompress_time_ms = context.actual_decompress_time_ms_ > 0.0
                               ? context.actual_decompress_time_ms_
                               : 0.0;
  row.compression_ratio = context.actual_compression_ratio_;
  std::lock_guard<std::mutex> lock(hcompress_mutex_);
  hcompress_predictor_->Observe(row);
}

}  // namespace clio::cte::compressor
