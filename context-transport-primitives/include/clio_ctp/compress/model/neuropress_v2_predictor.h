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
 * @file neuropress_v2_predictor.h
 * @brief NeuroPress v2: one network pass predicts compress time, decompress
 * time and ratio for every lossless GPU setting of gpu_setting_codec.h.
 *
 * The model (paper-benchmark/model-accuracy/train_nn_v2.py) is an MLP
 * 4 -> 64 x 4 (ReLU) -> 3 * 45 stored as an NNWT container version 3:
 *   u32 magic 'NNWT', u32 version 3, u32 n_layers, u32 dims[n_layers + 1],
 *   u32 feature_set (1), u32 outputs_per_setting (3), u32 n_settings,
 *   u32 table_bytes, the settings ('\n'-separated canonical specs, output
 *   order), f32 x_mean[in], x_std[in], y_mean[out], y_std[out], then per
 *   layer f32 W[out][in] (row-major) and f32 b[out]; little-endian.
 * Load() refuses a file whose settings table is not exactly the codec's
 * frozen table, so a prediction index is always a codec index.
 *
 * Feature set 1, from a chunk's float32 values (the training definition):
 *   x0 = log2(chunk bytes)
 *   x1 = Shannon entropy of the chunk's bytes, bits/byte
 *   x2 = log10(mean |x - mean(x)| / range + 1e-12)
 *   x3 = log10(mean |x[i+2] - 2 x[i+1] + x[i]| / range + 1e-12)
 * with range = max - min (1 when that is not positive). Outputs per setting
 * s: y[3s] = ln(compress ms), y[3s+1] = ln(decompress ms), y[3s+2] =
 * ln(ratio), each de-standardised then exponentiated. Times are GPU kernel
 * times with any shuffle included. No floor, cap or clamp is applied.
 */

#ifndef CLIO_CTP_COMPRESS_MODEL_NEUROPRESS_V2_PREDICTOR_H_
#define CLIO_CTP_COMPRESS_MODEL_NEUROPRESS_V2_PREDICTOR_H_

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace ctp::compress::model {

/** @brief The four v2 inputs of one chunk, before standardisation. */
struct NeuroPressV2Features {
  float x[4] = {0.0f, 0.0f, 0.0f, 0.0f};
};

/** @brief One setting's predicted outcome on one chunk. */
struct NeuroPressV2Prediction {
  int setting = -1;        ///< index into the frozen settings table
  double comp_ms = 0.0;    ///< predicted compress time, ms
  double decomp_ms = 0.0;  ///< predicted decompress time, ms
  double ratio = 0.0;      ///< predicted compression ratio
  /** w_ct * comp + w_dt * decomp + w_io * bytes / (ratio * bw); +inf when
   *  the setting is unavailable in this build or a prediction is not finite. */
  double cost = 0.0;
};

/** @brief The cost a ranking minimises (same form as v1's balanced cost). */
struct NeuroPressV2CostWeights {
  double w_ct = 1.0;               ///< weight of compress ms
  double w_dt = 1.0;               ///< weight of decompress ms
  double w_io = 1.0;               ///< weight of the I/O term
  double bw_bytes_per_ms = 5.0e6;  ///< storage bandwidth (5e6 B/ms = 5 GB/s)
};

/**
 * @brief Raw statistics of a float32 chunk, computed as training computed
 * them (paper-benchmark/codec-sweep/build_corpus_csv.py file_stats), in
 * double precision on the host.
 */
struct NeuroPressV2Stats {
  double entropy = 0.0;  ///< bits per byte over the chunk's bytes
  double mad = 0.0;      ///< mean |x - mean(x)|
  double d2 = 0.0;       ///< mean |x[i+2] - 2 x[i+1] + x[i]|
  double vmin = 0.0;     ///< minimum value
  double vmax = 0.0;     ///< maximum value
};

/**
 * @brief Loads and runs a NeuroPress v2 model; ranks the 45 settings of a
 * chunk by cost and learns online from measured results.
 *
 * Thread safety: ranking may run concurrently from many threads. Training
 * steps are serialised by an internal mutex; each one rewrites the touched
 * output rows on the host and then on the device.
 */
class NeuroPressV2Predictor {
 public:
  static constexpr uint32_t kFeatureSet = 1;      ///< the four inputs above
  static constexpr int kOutputsPerSetting = 3;    ///< ln ct, ln dt, ln ratio

  NeuroPressV2Predictor();
  ~NeuroPressV2Predictor();
  NeuroPressV2Predictor(const NeuroPressV2Predictor &) = delete;
  NeuroPressV2Predictor &operator=(const NeuroPressV2Predictor &) = delete;

  /**
   * @param path_or_dir a .nnwt file, or a directory holding model_v2.nnwt
   * @return the file a Load(path_or_dir) would read
   */
  static std::string ResolvePath(const std::string &path_or_dir);

  /**
   * @param path_or_dir as for ResolvePath
   * @return true when that file is an NNWT container version 3
   */
  static bool IsV2File(const std::string &path_or_dir);

  /**
   * Read, validate and (with CUDA) upload a v2 model.
   * @param path_or_dir as for ResolvePath
   * @return false with LastError() set on any mismatch or I/O error
   */
  bool Load(const std::string &path_or_dir);

  /** @return true after a successful Load(). */
  bool IsReady() const { return ready_; }
  /** @return why the last Load() failed, "" after a success. */
  const std::string &LastError() const { return error_; }
  /** @return the number of settings (45 for the shipped model). */
  int NumSettings() const { return n_settings_; }
  /** @return the number of training steps applied since Load(). */
  uint64_t UpdateCount() const { return updates_; }

  /**
   * Statistics of a host float32 chunk, exactly as the training data was
   * described.
   * @param data float32 values
   * @param n    number of values
   */
  static NeuroPressV2Stats ComputeStats(const float *data, size_t n);

  /**
   * The four network inputs from raw statistics.
   * @param chunk_bytes size of the chunk in bytes
   * @param s           its statistics
   */
  static NeuroPressV2Features MakeFeatures(size_t chunk_bytes,
                                           const NeuroPressV2Stats &s);

  /**
   * Run the network on the host (float32, as the GPU does).
   * @param f   the four inputs
   * @param out receives the de-standardised log targets, 3 per setting
   */
  void ForwardHost(const NeuroPressV2Features &f, std::vector<float> *out) const;

  /**
   * Rank every setting with the host network.
   * @param f           the chunk's inputs
   * @param chunk_bytes its size (for the I/O term)
   * @param w           cost weights
   * @return all settings, cheapest first (ties: lower index first)
   */
  std::vector<NeuroPressV2Prediction> RankHost(
      const NeuroPressV2Features &f, size_t chunk_bytes,
      const NeuroPressV2CostWeights &w) const;

  /**
   * Rank every setting on the GPU from device-resident statistics (as
   * returned by ctp::ComputeDeviceStatsResident): one kernel, one copy, one
   * synchronisation on `stream`.
   * @param device_stats  device pointer to a ctp::DeviceFeatureStats
   * @param chunk_bytes   the chunk's size
   * @param w             cost weights
   * @param stream        cudaStream_t to run on (nullptr = default)
   * @param features_out  optional: receives the inputs the kernel computed
   * @return all settings, cheapest first; empty without CUDA or on error
   */
  std::vector<NeuroPressV2Prediction> RankDevice(
      const void *device_stats, size_t chunk_bytes,
      const NeuroPressV2CostWeights &w, void *stream,
      NeuroPressV2Features *features_out = nullptr);

  /**
   * One online step on the output rows of one setting, from a measured
   * result: a normalised LMS update of each labelled row (gradient of the
   * squared standardised log error, divided by 1 + |h|^2 of the last hidden
   * layer), so a step removes at most a fraction lr (0 < lr <= 1) of that
   * row's error on this input and stays bounded on out-of-range inputs. Only
   * the rows of the labels given (> 0 and finite) change; the hidden layers
   * and every other setting stay as they are.
   * @param f          the chunk's inputs
   * @param setting    the setting that was run
   * @param comp_ms    measured compress ms (<= 0: no label)
   * @param decomp_ms  measured decompress ms (<= 0: no label)
   * @param ratio      measured compression ratio (<= 0: no label)
   * @param lr         fraction of the error removed per step (0, 1]
   * @param abs_err    optional: mean |log error| of the labels before the step
   * @return false when not loaded, the setting is invalid or no label is valid
   */
  bool TrainSetting(const NeuroPressV2Features &f, int setting, double comp_ms,
                    double decomp_ms, double ratio, double lr,
                    double *abs_err = nullptr);

 private:
  /** Parse and validate a version-3 file into the host parameters. */
  bool ParseFile(const std::string &path);
  /** Hidden activations of the last hidden layer and the outputs. */
  void ForwardHostFull(const NeuroPressV2Features &f, std::vector<float> *last,
                       std::vector<float> *out) const;
  /** Upload the whole parameter vector (Load) or a range of it (training). */
  bool SyncDevice(size_t offset, size_t count);
  /** Cost of a setting's prediction under w (inf when unusable). */
  double Cost(int s, double ct, double dt, double ratio, size_t chunk_bytes,
              const NeuroPressV2CostWeights &w) const;

  bool ready_ = false;
  std::string error_;
  int n_layers_ = 0;
  std::vector<int> dims_;            ///< input, hidden..., output
  int n_settings_ = 0;
  std::vector<float> params_;        ///< x_mean, x_std, y_mean, y_std, W/b...
  std::vector<size_t> w_off_, b_off_;
  size_t xm_off_ = 0, xs_off_ = 0, ym_off_ = 0, ys_off_ = 0;
  std::vector<unsigned char> available_;  ///< per setting, from the codec
  float *d_params_ = nullptr;        ///< device copy of params_
  unsigned char *d_available_ = nullptr;
  std::mutex train_mutex_;
  uint64_t updates_ = 0;
};

}  // namespace ctp::compress::model

#endif  // CLIO_CTP_COMPRESS_MODEL_NEUROPRESS_V2_PREDICTOR_H_
