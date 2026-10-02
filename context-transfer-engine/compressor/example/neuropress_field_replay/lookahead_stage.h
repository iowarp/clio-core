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
 * @file lookahead_stage.h
 * @brief Driver-side look-ahead staging for neuropress_field_replay
 *        (--lookahead N).
 *
 * The replay writes dumps in time order. With --lookahead N the driver holds
 * each chunk (same field, same chunk index) until N timesteps of it have
 * arrived, trial-encodes the middle one both ways (ctp::LookaheadCodec::
 * Trial), and when look-ahead is smaller writes the group as ONE blob encoded
 * by ctp::LookaheadCodec; otherwise the chunks go through NeuroPress's usual
 * per-chunk path, unchanged. The staging lives in the producer for this
 * prototype; the system design moves it to Clio's drain.
 */
#ifndef NEUROPRESS_FIELD_REPLAY_LOOKAHEAD_STAGE_H_
#define NEUROPRESS_FIELD_REPLAY_LOOKAHEAD_STAGE_H_

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <clio_ctp/compress/cuszp.h>
#include <clio_ctp/compress/lookahead.h>
#include <clio_ctp/compress/nvcomp.h>

namespace neuropress_replay {

/** One chunk of one timestep, held until its group is complete. */
struct StagedChunk {
  std::string frame;                                /**< timestep directory */
  std::vector<char> data;                           /**< float32 bytes */
  const std::filesystem::path *src = nullptr;       /**< source file */
  size_t off = 0;                                   /**< chunk offset in src */
};

/** A group stored as one look-ahead blob. */
struct LookaheadRecord {
  std::string name;                                 /**< blob name */
  std::vector<const std::filesystem::path *> src;   /**< per frame */
  std::vector<size_t> off;                          /**< per frame */
  size_t chunk_bytes = 0;                           /**< bytes per frame */
  size_t stored = 0;                                /**< blob bytes */
  bool ok = false;
};

/** nf device frames of n floats, freed on scope exit. */
struct DeviceGroup {
  std::vector<float *> p;
  bool ok = true;
  DeviceGroup(size_t nf, size_t n) : p(nf, nullptr) {
    for (auto &f : p) ok = ok && cudaMalloc(&f, n * sizeof(float)) == cudaSuccess;
  }
  ~DeviceGroup() {
    for (auto *f : p) cudaFree(f);
  }
};

/** See the file comment. */
class LookaheadStage {
 public:
  /**
   * @param frames Timesteps per group (>= 3).
   * @param eb     Absolute error bound (the NeuroPress bound of the run).
   */
  LookaheadStage(int frames, double eb)
      : frames_(frames), eb_(eb), packer_(ctp::NvCompAlgo::BITCOMP),
        codec_(&anchor_, &packer_) {
    ready_ = frames >= 3 && eb > 0.0 && anchor_.SetErrorBound(eb);
  }

  /** False when the stage could not be configured (see the constructor). */
  bool Ready() const { return ready_; }

  /**
   * @brief Holds a chunk under its key (field + chunk index).
   * @return true when the key's group is complete; it is moved to *group.
   */
  bool Add(const std::string &key, StagedChunk c,
           std::vector<StagedChunk> *group) {
    auto &g = open_[key];
    g.push_back(std::move(c));
    if (static_cast<int>(g.size()) < frames_) return false;
    *group = std::move(g);
    open_.erase(key);
    return true;
  }

  /** Moves out every partial group (end of the run). */
  std::vector<std::pair<std::string, std::vector<StagedChunk>>> TakeRemaining() {
    std::vector<std::pair<std::string, std::vector<StagedChunk>>> out(
        open_.begin(), open_.end());
    open_.clear();
    return out;
  }

  /**
   * @brief Trial-encodes the group's middle chunk both ways and, when
   *        look-ahead is smaller, encodes the whole group.
   * @param g    Chunks of one key, oldest first, all the same size.
   * @param blob Receives the look-ahead blob.
   * @return true when look-ahead was chosen and the group encoded.
   */
  bool EncodeIfPays(const std::vector<StagedChunk> &g,
                    std::vector<uint8_t> *blob) {
    const size_t nf = g.size(), bytes = nf ? g[0].data.size() : 0;
    if (!ready_ || nf < 3 || bytes == 0 || bytes % sizeof(float) != 0) {
      return false;
    }
    for (const auto &c : g) {
      if (c.data.size() != bytes) return false;
    }
    const size_t n = bytes / sizeof(float);
    DeviceGroup d(nf, n);
    if (!d.ok) return false;
    for (size_t t = 0; t < nf; ++t) {
      if (cudaMemcpy(d.p[t], g[t].data.data(), bytes, cudaMemcpyHostToDevice) !=
          cudaSuccess) {
        return false;
      }
    }
    const size_t m = nf / 2;
    ctp::LookaheadTrial trial;
    if (!codec_.Trial(d.p[m - 1], d.p[m], d.p[m + 1], n, eb_, &trial)) {
      return false;
    }
    trial_spatial_bytes_ += trial.spatial_bytes;
    trial_lookahead_bytes_ += trial.lookahead_bytes;
    if (!trial.LookaheadPays()) return false;
    return codec_.Encode(d.p.data(), static_cast<int>(nf), n, eb_, blob);
  }

  /**
   * @brief Decodes a stored blob and checks every frame against its source.
   * @return worst |decoded - original| / eb over the group, or -1 on failure.
   */
  double Check(const std::vector<uint8_t> &blob, const LookaheadRecord &r) {
    const size_t nf = r.src.size(), n = r.chunk_bytes / sizeof(float);
    DeviceGroup d(nf, n);
    if (!d.ok || !codec_.Decode(blob.data(), blob.size(), d.p.data())) {
      return -1.0;
    }
    std::vector<float> got(n), want(n);
    double worst = 0.0;
    for (size_t t = 0; t < nf; ++t) {
      std::ifstream in(*r.src[t], std::ios::binary);
      in.seekg(static_cast<std::streamoff>(r.off[t]));
      in.read(reinterpret_cast<char *>(want.data()),
              static_cast<std::streamsize>(r.chunk_bytes));
      if (!in ||
          cudaMemcpy(got.data(), d.p[t], r.chunk_bytes,
                     cudaMemcpyDeviceToHost) != cudaSuccess) {
        return -1.0;
      }
      for (size_t i = 0; i < n; ++i) {
        const double e = std::fabs(static_cast<double>(got[i]) - want[i]);
        worst = std::isnan(e) ? worst : std::max(worst, e);
      }
    }
    return worst / eb_;
  }

  size_t trial_spatial_bytes_ = 0;    /**< summed trial results */
  size_t trial_lookahead_bytes_ = 0;

 private:
  int frames_;
  double eb_;
  bool ready_ = false;
  ctp::Cuszp anchor_;
  ctp::NvComp packer_;
  ctp::LookaheadCodec codec_;
  std::map<std::string, std::vector<StagedChunk>> open_;
};

}  // namespace neuropress_replay

#endif  // NEUROPRESS_FIELD_REPLAY_LOOKAHEAD_STAGE_H_
