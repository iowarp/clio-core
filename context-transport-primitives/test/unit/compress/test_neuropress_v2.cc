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
 * @file test_neuropress_v2.cc
 * @brief NeuroPress v2 predictor: loading and table check, the host network
 * against the Python model's own outputs, GPU features and ranking against
 * the host, and online learning on one setting's output rows.
 */

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "basic_test.h"
#include "clio_ctp/compress/gpu_setting_codec.h"
#include "clio_ctp/compress/model/neuropress_v2_predictor.h"
#include "clio_ctp/compress/preprocess/data_stats.h"
#include "clio_ctp/compress/preprocess/data_stats_gpu.h"

using ctp::compress::model::NeuroPressV2CostWeights;
using ctp::compress::model::NeuroPressV2Features;
using ctp::compress::model::NeuroPressV2Predictor;

namespace {

/** The shipped v2 weights (set by CMake from the source tree). */
const std::string kWeights = CLIO_CTP_NEUROPRESS_V2_WEIGHTS;

/** @return a float32 field with smooth, noisy and constant regions. */
std::vector<float> Field(size_t n, float scale) {
  std::vector<float> v(n);
  for (size_t i = 0; i < n; ++i) {
    const float sine = std::sin(static_cast<float>(i) * 0.002f) * scale;
    v[i] = (i % 5000 < 300) ? 0.0f
                            : sine + static_cast<float>((i * 7919) % 101);
  }
  return v;
}

/** @return |a - b| / max(|b|, 1e-6). */
double Rel(double a, double b) { return std::fabs(a - b) / std::max(std::fabs(b), 1e-6); }

}  // namespace

TEST_CASE("NeuroPressV2Load", "[neuropress_v2]") {
  NeuroPressV2Predictor p;
  REQUIRE(NeuroPressV2Predictor::IsV2File(kWeights));
  REQUIRE(p.Load(kWeights));
  REQUIRE(p.IsReady());
  REQUIRE(p.NumSettings() == ctp::kGpuSettingCount);
  // A model whose table disagrees with the codec's must be refused.
  std::ifstream in(NeuroPressV2Predictor::ResolvePath(kWeights), std::ios::binary);
  std::string bytes((std::istreambuf_iterator<char>(in)), {});
  const size_t at = bytes.find("zstd shuffle=byte");
  REQUIRE(at != std::string::npos);
  bytes.replace(at, 17, "zstd shuffle=bxte");
  const std::string bad = "/tmp/neuropress_v2_bad_table.nnwt";
  std::ofstream(bad, std::ios::binary) << bytes;
  NeuroPressV2Predictor q;
  REQUIRE_FALSE(q.Load(bad));
  REQUIRE(q.LastError().find("setting 44") != std::string::npos);
  std::remove(bad.c_str());
  REQUIRE_FALSE(q.Load("/nonexistent/model_v2.nnwt"));
}

TEST_CASE("NeuroPressV2HostMatchesPython", "[neuropress_v2]") {
  NeuroPressV2Predictor p;
  REQUIRE(p.Load(kWeights));
  // Inputs and outputs 0, 1, 2, 64, 100, 134 from train_nn_v2.numpy_forward.
  const float in[3][4] = {{22.0f, 6.88f, -0.74f, -2.2f},
                          {16.0f, 2.5f, -1.3f, -4.0f},
                          {20.0f, 7.4f, -0.9f, -0.5f}};
  const float want[3][6] = {
      {-1.52135468f, -1.83832288f, 0.103630334f, 0.978956521f, -0.227885246f,
       0.949283004f},
      {-1.65314198f, -2.1068697f, 1.54213452f, -0.736243129f, -1.29862857f,
       6.51465273f},
      {-1.54602969f, -1.92593765f, -0.016931802f, 1.16417348f, -0.645867646f,
       0.223747969f}};
  const int picks[6] = {0, 1, 2, 64, 100, 134};
  for (int v = 0; v < 3; ++v) {
    NeuroPressV2Features f;
    for (int i = 0; i < 4; ++i) f.x[i] = in[v][i];
    std::vector<float> y;
    p.ForwardHost(f, &y);
    REQUIRE(y.size() == 135);
    for (int k = 0; k < 6; ++k) {
      INFO("vector " << v << " output " << picks[k]);
      REQUIRE(std::fabs(y[picks[k]] - want[v][k]) < 1e-4);
    }
  }
}

TEST_CASE("NeuroPressV2DeviceMatchesHost", "[neuropress_v2]") {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    WARN("no CUDA device; skipping");
    return;
  }
  NeuroPressV2Predictor p;
  REQUIRE(p.Load(kWeights));
  const NeuroPressV2CostWeights w;
  for (float scale : {1.0f, 1e6f}) {
    const auto field = Field(1u << 20, scale);  // 4 MiB
    const size_t bytes = field.size() * sizeof(float);
    void *d = nullptr;
    REQUIRE(cudaMalloc(&d, bytes) == cudaSuccess);
    REQUIRE(cudaMemcpy(d, field.data(), bytes, cudaMemcpyHostToDevice) ==
            cudaSuccess);
    void *stream = ctp::DeviceStatsStream();
    const void *st = ctp::ComputeDeviceStatsResident(
        d, field.size(), ctp::DataType::FLOAT32, stream);
    REQUIRE(st != nullptr);
    NeuroPressV2Features fd;
    const auto dev = p.RankDevice(st, bytes, w, stream, &fd);
    const auto fh = NeuroPressV2Predictor::MakeFeatures(
        bytes, NeuroPressV2Predictor::ComputeStats(field.data(), field.size()));
    const auto host = p.RankHost(fh, bytes, w);
    REQUIRE(dev.size() == 45);
    REQUIRE(host.size() == 45);
    for (int i = 0; i < 4; ++i) {
      INFO("scale " << scale << " feature " << i);
      REQUIRE(std::fabs(fd.x[i] - fh.x[i]) < 1e-4);
    }
    for (int k = 0; k < 45; ++k) {
      INFO("scale " << scale << " rank " << k);
      REQUIRE(Rel(dev[k].cost, host[k].cost) < 1e-3);
      REQUIRE(Rel(dev[k].ratio, host[k].ratio) < 1e-3);
    }
    REQUIRE(dev[0].setting == host[0].setting);
    std::printf("  [neuropress_v2] scale %g: features %.4f %.4f %.4f %.4f, "
                "pick %s (%.3f ms, ratio %.3f)\n", scale, fd.x[0], fd.x[1],
                fd.x[2], fd.x[3], ctp::GpuSettingSpec(dev[0].setting),
                dev[0].comp_ms, dev[0].ratio);
    cudaFree(d);
  }
}

TEST_CASE("NeuroPressV2OnlineLearning", "[neuropress_v2]") {
  NeuroPressV2Predictor p;
  REQUIRE(p.Load(kWeights));
  NeuroPressV2Features f;
  const float x[4] = {22.0f, 6.88f, -0.74f, -2.2f};
  for (int i = 0; i < 4; ++i) f.x[i] = x[i];
  std::vector<float> before, after;
  p.ForwardHost(f, &before);
  const int s = 42;  // zstd
  // Labels the model gets wrong: twice its compress time, half its ratio.
  const double ct = 2.0 * std::exp(before[3 * s]);
  const double ratio = 0.5 * std::exp(before[3 * s + 2]);
  double err0 = 0.0, err = 0.0;
  REQUIRE(p.TrainSetting(f, s, ct, -1.0, ratio, 0.05, &err0));
  for (int it = 0; it < 200; ++it) REQUIRE(p.TrainSetting(f, s, ct, -1.0, ratio, 0.05, &err));
  REQUIRE(err < 0.1 * err0);
  REQUIRE(p.UpdateCount() == 201);
  p.ForwardHost(f, &after);
  REQUIRE(Rel(std::exp(after[3 * s]), ct) < 0.02);
  REQUIRE(Rel(std::exp(after[3 * s + 2]), ratio) < 0.02);
  // The unlabeled decompress head and every other setting are untouched.
  REQUIRE(after[3 * s + 1] == before[3 * s + 1]);
  for (int o = 0; o < 135; ++o) {
    if (o / 3 != s) REQUIRE(after[o] == before[o]);
  }
  REQUIRE_FALSE(p.TrainSetting(f, 45, ct, -1.0, ratio, 0.05));
  REQUIRE_FALSE(p.TrainSetting(f, s, -1.0, -1.0, -1.0, 0.05));
  // The device copy follows: a fresh ranking sees the learned values.
  int devices = 0;
  if (cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0) {
    const auto field = Field(1u << 18, 1.0f);
    void *d = nullptr;
    REQUIRE(cudaMalloc(&d, field.size() * 4) == cudaSuccess);
    REQUIRE(cudaMemcpy(d, field.data(), field.size() * 4,
                       cudaMemcpyHostToDevice) == cudaSuccess);
    void *stream = ctp::DeviceStatsStream();
    const void *st = ctp::ComputeDeviceStatsResident(
        d, field.size(), ctp::DataType::FLOAT32, stream);
    NeuroPressV2Features fd;
    const auto dev = p.RankDevice(st, field.size() * 4, NeuroPressV2CostWeights(), stream, &fd);
    const auto host = p.RankHost(fd, field.size() * 4, NeuroPressV2CostWeights());
    for (int k = 0; k < 45; ++k) REQUIRE(Rel(dev[k].cost, host[k].cost) < 1e-3);
    cudaFree(d);
  }
}
