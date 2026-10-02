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
 * @file lookahead.h
 * @brief Look-ahead (bidirectional temporal) coding of one chunk across a
 *        group of consecutive timesteps.
 *
 * A group of nf device frames (the same chunk at nf consecutive timesteps,
 * float32, n values each) becomes one self-describing host blob:
 *   - frame 0 is coded on its own with the anchor codec (e.g. cuSZp);
 *   - frame nf-1 is coded as its change from frame 0 with the anchor codec;
 *   - frames in between are predicted from the reconstructed frames on both
 *     sides (midpoint first), and the residual goes through NeuroPress's
 *     device quantizer (QuantizeDevice) and the packer codec (e.g. nvCOMP).
 * Every frame is reconstructed in the encoder exactly as the decoder will,
 * and any value whose float32 reconstruction misses the bound is stored as is
 * (an escape), so every decoded value is within the bound or bit-exact.
 * Groups are independent: decoding one needs nothing outside its blob.
 */
#ifndef CLIO_CTP_COMPRESS_LOOKAHEAD_H_
#define CLIO_CTP_COMPRESS_LOOKAHEAD_H_

#if CTP_ENABLE_COMPRESS && CTP_ENABLE_CUDA

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "clio_ctp/compress/compress.h"
#include "clio_ctp/compress/preprocess/quantization.h"
#include "clio_ctp/compress/preprocess/temporal.h"

namespace ctp {

/** How one frame of a look-ahead group is coded. */
enum class LookaheadKind : uint8_t {
  kIntra = 0,          /**< on its own, anchor codec */
  kForward = 1,        /**< change from frame lo, anchor codec */
  kBidirectional = 2,  /**< residual of the lo/hi prediction, quantizer */
};

/** One coding step: frame t, its kind and its reference frames. */
struct LookaheadStep {
  int t = 0;
  LookaheadKind kind = LookaheadKind::kIntra;
  int lo = -1;  /**< earlier reference (forward and bidirectional) */
  int hi = -1;  /**< later reference (bidirectional) */
};

/** Blob header. */
struct LookaheadGroupHeader {
  uint32_t magic = 0x3148414Cu;  /**< "LAH1" */
  uint32_t version = 1;
  uint32_t frames = 0;
  uint32_t reserved = 0;
  uint64_t elems = 0;            /**< values per frame */
  double error_bound = 0.0;      /**< absolute */
};

/** Per-frame record; payload, escape indices and escape values follow it. */
struct LookaheadFrameHeader {
  uint32_t t = 0;
  uint8_t kind = 0;
  uint8_t pad[3] = {0, 0, 0};
  int32_t lo = -1;
  int32_t hi = -1;
  uint64_t payload_bytes = 0;
  uint64_t escapes = 0;
  uint64_t quant_bytes = 0;  /**< quantized bytes before packing (B only) */
  compress::preprocess::DeviceQuantizeParams quant;  /**< B only */
};

/** Bytes the middle of three frames costs when coded each way. */
struct LookaheadTrial {
  size_t spatial_bytes = 0;    /**< the frame through the anchor codec */
  size_t lookahead_bytes = 0;  /**< its two-sided residual, quantized + packed */
  /** True when coding in-between frames by look-ahead is cheaper. */
  bool LookaheadPays() const { return lookahead_bytes < spatial_bytes; }
};

/**
 * @brief Encoder / decoder for look-ahead groups (see the file comment).
 *
 * The codecs are borrowed, not owned. The anchor codec must already hold
 * the error bound (Compressor::SetErrorBound); the packer must be lossless.
 */
class LookaheadCodec {
 public:
  /**
   * @param anchor Error-bounded codec for anchor frames (e.g. ctp::Cuszp).
   * @param packer Lossless codec for quantized residuals (e.g. ctp::NvComp).
   */
  LookaheadCodec(Compressor *anchor, Compressor *packer)
      : anchor_(anchor), packer_(packer) {}

  /**
   * @brief Coding order for a group of nf frames: frame 0, frame nf-1, then
   *        the frames in between, each midpoint before its halves.
   */
  static std::vector<LookaheadStep> Plan(int nf) {
    std::vector<LookaheadStep> steps;
    if (nf <= 0) return steps;
    steps.push_back({0, LookaheadKind::kIntra, -1, -1});
    if (nf == 1) return steps;
    steps.push_back({nf - 1, LookaheadKind::kForward, 0, -1});
    AddMidpoints(0, nf - 1, &steps);
    return steps;
  }

  /**
   * @brief Encode a group.
   * @param frames    nf device pointers, n floats each (oldest first).
   * @param nf        Frames in the group (>= 1).
   * @param n         Values per frame.
   * @param eb        Absolute error bound.
   * @param out       Receives the blob (host memory).
   * @param recon_out Optional nf device buffers that receive the encoder's
   *                  reconstructions (what Decode will produce).
   * @return false on a codec or CUDA failure.
   */
  bool Encode(const float *const *frames, int nf, size_t n, double eb,
              std::vector<uint8_t> *out, float *const *recon_out = nullptr) {
    if (nf <= 0 || n == 0 || out == nullptr || !(eb > 0.0)) return false;
    Scratch sc;
    if (!sc.Init(n)) return false;
    std::vector<float *> recon(nf);
    for (int t = 0; t < nf; ++t) {
      recon[t] = recon_out != nullptr ? recon_out[t] : sc.Frame(t, n);
      if (recon[t] == nullptr) return false;
    }
    LookaheadGroupHeader gh;
    gh.frames = static_cast<uint32_t>(nf);
    gh.elems = n;
    gh.error_bound = eb;
    out->clear();
    Append(out, &gh, sizeof(gh));
    for (const LookaheadStep &s : Plan(nf)) {
      if (!EncodeFrame(s, frames, recon.data(), n, eb, &sc, out)) return false;
    }
    return true;
  }

  /**
   * @brief Decode a blob written by Encode.
   * @param blob   Host bytes.
   * @param size   Blob size.
   * @param frames nf device buffers of n floats (nf, n as encoded).
   * @return false on a malformed blob or a codec / CUDA failure.
   */
  bool Decode(const uint8_t *blob, size_t size, float *const *frames) {
    LookaheadGroupHeader gh;
    if (blob == nullptr || size < sizeof(gh)) return false;
    std::memcpy(&gh, blob, sizeof(gh));
    if (gh.magic != LookaheadGroupHeader().magic || gh.version != 1) {
      return false;
    }
    Scratch sc;
    if (!sc.Init(gh.elems)) return false;
    size_t pos = sizeof(gh);
    for (uint32_t k = 0; k < gh.frames; ++k) {
      if (!DecodeFrame(blob, size, &pos, gh, frames, &sc)) return false;
    }
    return true;
  }

  /**
   * @brief Trial-encodes the middle of three consecutive frames both ways
   *        with the real codecs, to decide whether look-ahead pays.
   *
   * Spatial: mid through the anchor codec. Look-ahead: mid minus the average
   * of prev and next, through QuantizeDevice and the packer -- what Encode
   * stores for an in-between frame. Costs one extra compression of each kind
   * per decision, so callers run it once per group, not per frame.
   *
   * @param prev, mid, next Device frames of n floats (consecutive timesteps).
   * @param n               Values per frame.
   * @param eb              Absolute error bound (the anchor codec must hold
   *                        the same bound).
   * @param out             Byte counts of both encodings.
   * @return false on a codec or CUDA failure.
   */
  bool Trial(const float *prev, const float *mid, const float *next, size_t n,
             double eb, LookaheadTrial *out) {
    namespace pp = compress::preprocess;
    if (out == nullptr || n == 0 || !(eb > 0.0)) return false;
    Scratch sc;
    if (!sc.Init(n)) return false;
    std::vector<uint8_t> payload;
    if (!CompressTo(anchor_, mid, n * 4, &payload)) return false;
    out->spatial_bytes = payload.size();
    pp::DeviceQuantizeParams qp;
    size_t qbytes = 0;
    if (!pp::TemporalPredictDevice(prev, next, 0.5f, sc.pred, n, nullptr) ||
        !pp::AddDevice(mid, sc.pred, -1.0f, sc.resid, n, nullptr) ||
        !pp::QuantizeDevice(sc.resid, n, 4, eb, sc.quant, &qbytes, &qp,
                            nullptr) ||
        !CompressTo(packer_, sc.quant, qbytes, &payload)) {
      return false;
    }
    out->lookahead_bytes = payload.size() + sizeof(LookaheadFrameHeader);
    return true;
  }

 private:
  /** Device scratch for one Encode / Decode call. */
  struct Scratch {
    float *pred = nullptr, *resid = nullptr, *dec = nullptr;
    void *quant = nullptr;
    uint32_t *esc_idx = nullptr;
    float *esc_val = nullptr;
    std::vector<float *> frames;  /**< internal reconstructions */

    bool Init(size_t n) {
      const size_t qcap = compress::preprocess::QuantizeCapacity(n, 4);
      return cudaMalloc(&pred, n * 4) == cudaSuccess &&
             cudaMalloc(&resid, n * 4) == cudaSuccess &&
             cudaMalloc(&dec, n * 4) == cudaSuccess &&
             cudaMalloc(&quant, qcap) == cudaSuccess &&
             cudaMalloc(&esc_idx, n * 4) == cudaSuccess &&
             cudaMalloc(&esc_val, n * 4) == cudaSuccess;
    }
    float *Frame(int t, size_t n) {
      if (frames.size() <= static_cast<size_t>(t)) frames.resize(t + 1, nullptr);
      if (frames[t] == nullptr && cudaMalloc(&frames[t], n * 4) != cudaSuccess) {
        frames[t] = nullptr;
      }
      return frames[t];
    }
    ~Scratch() {
      cudaFree(pred);
      cudaFree(resid);
      cudaFree(dec);
      cudaFree(quant);
      cudaFree(esc_idx);
      cudaFree(esc_val);
      for (float *f : frames) cudaFree(f);
    }
  };

  /** Appends the frames strictly between lo and hi, midpoint first. */
  static void AddMidpoints(int lo, int hi, std::vector<LookaheadStep> *steps) {
    if (hi - lo < 2) return;
    const int m = (lo + hi) / 2;
    steps->push_back({m, LookaheadKind::kBidirectional, lo, hi});
    AddMidpoints(lo, m, steps);
    AddMidpoints(m, hi, steps);
  }

  /** Weight of the later reference in a bidirectional prediction. */
  static float Weight(int t, int lo, int hi) {
    return static_cast<float>(t - lo) / static_cast<float>(hi - lo);
  }

  /** Appends raw bytes to a blob. */
  static void Append(std::vector<uint8_t> *out, const void *p, size_t bytes) {
    const uint8_t *b = static_cast<const uint8_t *>(p);
    out->insert(out->end(), b, b + bytes);
  }

  /** Compresses a device buffer with codec c into host bytes. */
  static bool CompressTo(Compressor *c, const void *dev, size_t bytes,
                         std::vector<uint8_t> *payload) {
    size_t cap = c->MaxCompressedSize(bytes);
    if (cap == 0) cap = 2 * bytes + 65536;  // cuSZp documents no worst case
    payload->resize(cap);
    size_t size = cap;
    if (!c->Compress(payload->data(), size, const_cast<void *>(dev), bytes)) {
      return false;
    }
    payload->resize(size);
    return true;
  }

  /**
   * Encodes one step: computes its residual, compresses it, reconstructs it
   * exactly as Decode will, and appends the frame record to the blob.
   */
  bool EncodeFrame(const LookaheadStep &s, const float *const *frames,
                   float *const *recon, size_t n, double eb, Scratch *sc,
                   std::vector<uint8_t> *out) {
    namespace pp = compress::preprocess;
    const float *x = frames[s.t];
    const float *pred = nullptr;
    LookaheadFrameHeader fh;
    fh.t = static_cast<uint32_t>(s.t);
    fh.kind = static_cast<uint8_t>(s.kind);
    fh.lo = s.lo;
    fh.hi = s.hi;
    std::vector<uint8_t> payload;
    if (s.kind == LookaheadKind::kBidirectional) {
      pred = sc->pred;
      if (!pp::TemporalPredictDevice(recon[s.lo], recon[s.hi],
                                     Weight(s.t, s.lo, s.hi), sc->pred, n,
                                     nullptr) ||
          !pp::AddDevice(x, pred, -1.0f, sc->resid, n, nullptr)) {
        return false;
      }
      size_t qbytes = 0;
      if (!pp::QuantizeDevice(sc->resid, n, 4, eb, sc->quant, &qbytes,
                              &fh.quant, nullptr) ||
          !pp::DequantizeDevice(sc->quant, n, fh.quant, sc->dec) ||
          !CompressTo(packer_, sc->quant, qbytes, &payload)) {
        return false;
      }
      fh.quant_bytes = qbytes;
    } else {
      const void *in = x;
      if (s.kind == LookaheadKind::kForward) {
        pred = recon[s.lo];
        if (!pp::AddDevice(x, pred, -1.0f, sc->resid, n, nullptr)) return false;
        in = sc->resid;
      }
      size_t dsize = n * 4;
      if (!CompressTo(anchor_, in, n * 4, &payload) ||
          !anchor_->Decompress(sc->dec, dsize, payload.data(), payload.size())) {
        return false;
      }
    }
    uint64_t esc = 0;
    if (!pp::BoundCheckDevice(x, pred, sc->dec, eb, recon[s.t], sc->esc_idx,
                              sc->esc_val, &esc, n, nullptr)) {
      return false;
    }
    fh.payload_bytes = payload.size();
    fh.escapes = esc;
    return AppendFrame(fh, payload, sc, out);
  }

  /** Appends a frame record, its payload and its escape list to the blob. */
  static bool AppendFrame(const LookaheadFrameHeader &fh,
                          const std::vector<uint8_t> &payload, Scratch *sc,
                          std::vector<uint8_t> *out) {
    Append(out, &fh, sizeof(fh));
    Append(out, payload.data(), payload.size());
    if (fh.escapes == 0) return true;
    const size_t pos = out->size();
    out->resize(pos + fh.escapes * 8);
    return cudaMemcpy(out->data() + pos, sc->esc_idx, fh.escapes * 4,
                      cudaMemcpyDeviceToHost) == cudaSuccess &&
           cudaMemcpy(out->data() + pos + fh.escapes * 4, sc->esc_val,
                      fh.escapes * 4, cudaMemcpyDeviceToHost) == cudaSuccess;
  }

  /** Decodes the frame record at *pos into frames[t] and advances *pos. */
  bool DecodeFrame(const uint8_t *blob, size_t size, size_t *pos,
                   const LookaheadGroupHeader &gh, float *const *frames,
                   Scratch *sc) {
    namespace pp = compress::preprocess;
    LookaheadFrameHeader fh;
    if (*pos + sizeof(fh) > size) return false;
    std::memcpy(&fh, blob + *pos, sizeof(fh));
    *pos += sizeof(fh);
    const size_t n = gh.elems;
    if (fh.t >= gh.frames || *pos + fh.payload_bytes + fh.escapes * 8 > size) {
      return false;
    }
    void *payload = const_cast<uint8_t *>(blob + *pos);
    float *dst = frames[fh.t];
    const auto kind = static_cast<LookaheadKind>(fh.kind);
    size_t dsize = n * 4;
    bool ok;
    if (kind == LookaheadKind::kIntra) {
      ok = anchor_->Decompress(dst, dsize, payload, fh.payload_bytes);
    } else if (kind == LookaheadKind::kForward) {
      ok = anchor_->Decompress(sc->dec, dsize, payload, fh.payload_bytes) &&
           pp::AddDevice(frames[fh.lo], sc->dec, 1.0f, dst, n, nullptr);
    } else {
      size_t qsize = compress::preprocess::QuantizeCapacity(n, 4);
      ok = pp::TemporalPredictDevice(frames[fh.lo], frames[fh.hi],
                                     Weight(fh.t, fh.lo, fh.hi), sc->pred, n,
                                     nullptr) &&
           packer_->Decompress(sc->quant, qsize, payload, fh.payload_bytes) &&
           qsize == fh.quant_bytes &&
           pp::DequantizeDevice(sc->quant, n, fh.quant, sc->dec) &&
           pp::AddDevice(sc->pred, sc->dec, 1.0f, dst, n, nullptr);
    }
    *pos += fh.payload_bytes;
    if (ok && fh.escapes > 0) ok = ScatterEscapes(blob + *pos, fh.escapes, dst, sc, n);
    *pos += fh.escapes * 8;
    return ok;
  }

  /** Copies an escape list to the device and applies it to dst. */
  static bool ScatterEscapes(const uint8_t *esc, uint64_t m, float *dst,
                             Scratch *sc, size_t n) {
    if (m > n) return false;
    return cudaMemcpy(sc->esc_idx, esc, m * 4, cudaMemcpyHostToDevice) ==
               cudaSuccess &&
           cudaMemcpy(sc->esc_val, esc + m * 4, m * 4,
                      cudaMemcpyHostToDevice) == cudaSuccess &&
           compress::preprocess::ScatterDevice(sc->esc_idx, sc->esc_val, m,
                                               dst, nullptr);
  }

  Compressor *anchor_;
  Compressor *packer_;
};

}  // namespace ctp

#endif  // CTP_ENABLE_COMPRESS && CTP_ENABLE_CUDA

#endif  // CLIO_CTP_COMPRESS_LOOKAHEAD_H_
