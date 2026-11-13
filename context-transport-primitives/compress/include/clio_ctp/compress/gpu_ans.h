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
 * gpu_ans: interleaved static rANS in the gpu_lz4 page container (magic
 * "GANS"), with ONE decoder for the host and every GPU backend.
 *
 * WHY. Byte-entropy coding is what compresses BF16 weights stored in the esm
 * layout: the exponent plane has a few bits of entropy per byte and almost
 * no repeated strings, so an entropy coder reaches ~1.47x where LZ4 reaches
 * ~1.28x. nvCOMP's ANS is CUDA-only and its format private; sycl-ans is
 * SYCL-only and decodes each chunk on ONE work-item. This format decodes on
 * CUDA, ROCm and SYCL alike, with every lane of a sub-group busy.
 *
 * CHUNK LAYOUT (each chunk of the gpu_lz4 container, <= 64 KiB raw):
 *
 *   u16 freq[256]       normalized frequencies, sum = kProbScale (4096)
 *   u16 soff[kLanes+1]  byte offset of each stream after this table
 *   stream s            u32 final encoder state, then its renormalization
 *                       bytes in DECODE order
 *
 * Stream s codes the chunk's bytes s, s + kLanes, s + 2 kLanes, ... (kLanes =
 * 32 streams), so a sub-group decodes all 32 streams at once with no
 * synchronization between them: a 32-lane CUDA warp one stream per lane, a
 * 16-lane Intel sub-group two per lane. Every stream must end at its encoder's
 * initial state (kRansL): a corrupted stream fails that check instead of
 * decoding to plausible garbage.
 *
 * rANS parameters follow Giesen's rans_byte (32-bit state, byte-wise
 * renormalization, 12-bit probabilities) -- the coder sycl-ans uses.
 */

#ifndef CLIO_CTP_COMPRESS_GPU_ANS_H_
#define CLIO_CTP_COMPRESS_GPU_ANS_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "gpu_lz4.h"

namespace ctp::gpu_lz4 {

inline constexpr uint32_t kAnsLanes = 32;
inline constexpr uint32_t kProbBits = 12;
inline constexpr uint32_t kProbScale = 1u << kProbBits;
inline constexpr uint32_t kRansL = 1u << 23;
/** freq table + stream offsets */
inline constexpr uint32_t kAnsTableBytes = 2 * 256 + 2 * (kAnsLanes + 1);

/** Decode tables one sub-group builds per chunk (local / shared memory). */
struct AnsTables {
  uint16_t freq[256];
  uint16_t cumf[257];
  uint8_t slot[kProbScale];  /**< slot -> symbol */
};
/** Bytes of local memory each sub-group of the decode launch needs. */
inline constexpr uint32_t kDecodeScratch =
    (sizeof(AnsTables) + 15u) / 16u * 16u;

CTP_CROSS_FUN inline uint16_t Rd16(const uint8_t *p) {
  return static_cast<uint16_t>(p[0] | (static_cast<uint16_t>(p[1]) << 8));
}

/**
 * Decode one rANS chunk COOPERATIVELY. Every lane of a sub-group calls this
 * with the same arguments except `lane`; `sync` is a sub-group barrier with
 * memory visibility, reached uniformly (the control flow before each barrier
 * depends only on values every lane reads identically).
 * @param sp     the chunk (from LocateChunk), not stored raw
 * @param lane   this lane's index; nl = lanes in the sub-group
 * @param t      this sub-group's table scratch (local memory)
 * @return this lane's verdict: false if any of ITS streams failed
 */
template <typename SyncT>
CTP_CROSS_FUN inline bool DecodeChunkAnsCoop(const ChunkSpan &sp, uint32_t lane,
                                             uint32_t nl, SyncT &&sync,
                                             AnsTables *t) {
  if (sp.sn < kAnsTableBytes) return false;  // uniform
  for (uint32_t s = lane; s < 256; s += nl) t->freq[s] = Rd16(sp.src + 2 * s);
  sync();
  if (lane == 0) {
    uint32_t acc = 0;
    for (uint32_t s = 0; s < 256; ++s) {
      t->cumf[s] = static_cast<uint16_t>(acc < 65535u ? acc : 65535u);
      acc += t->freq[s];
    }
    t->cumf[256] = static_cast<uint16_t>(acc == kProbScale ? acc : 0u);
  }
  sync();
  if (t->cumf[256] != kProbScale) return false;  // uniform: same shared word
  for (uint32_t s = lane; s < 256; s += nl) {
    for (uint32_t j = t->cumf[s]; j < t->cumf[s + 1]; ++j) {
      t->slot[j] = static_cast<uint8_t>(s);
    }
  }
  sync();
  bool ok = true;
  const uint8_t *streams = sp.src + kAnsTableBytes;
  const uint32_t avail = sp.sn - kAnsTableBytes;
  const uint8_t *soff = sp.src + 2 * 256;
  for (uint32_t st = lane; st < kAnsLanes; st += nl) {
    const uint32_t b = Rd16(soff + 2 * st), e = Rd16(soff + 2 * (st + 1));
    if (e < b || e > avail || e - b < 4) {
      ok = false;
      continue;
    }
    const uint8_t *p = streams + b + 4;
    const uint8_t *end = streams + e;
    const uint8_t *s0 = streams + b;
    uint32_t x = static_cast<uint32_t>(s0[0]) |
                 static_cast<uint32_t>(s0[1]) << 8 |
                 static_cast<uint32_t>(s0[2]) << 16 |
                 static_cast<uint32_t>(s0[3]) << 24;
    for (uint32_t i = st; i < sp.want; i += kAnsLanes) {
      const uint32_t slot = x & (kProbScale - 1);
      const uint32_t s = t->slot[slot];
      sp.out[i] = static_cast<uint8_t>(s);
      x = t->freq[s] * (x >> kProbBits) + slot - t->cumf[s];
      while (x < kRansL && p < end) x = (x << 8) | *p++;
    }
    if (x != kRansL || p != end) ok = false;  // every byte, back to start
  }
  sync();  // the tables are reused for the next chunk
  return ok;
}

/**
 * Decode the chunks of one page -- LZ4 or ANS, by the page's magic -- that
 * belong to sub-group `warp` (warp, warp + nwarps, ...). The kernel entry of
 * every backend.
 * @param job      the page
 * @param lane     lane in the sub-group; nl = lanes per sub-group
 * @param warp     sub-group index in the work-group; nwarps = sub-groups
 * @param sync     sub-group barrier (callable)
 * @param scratch  this sub-group's kDecodeScratch bytes of local memory
 */
template <typename SyncT>
CTP_CROSS_FUN inline void DecodeJobWarpAny(DecodeJob *job, uint32_t lane,
                                           uint32_t nl, uint32_t warp,
                                           uint32_t nwarps, SyncT &&sync,
                                           void *scratch) {
  const uint32_t nch = ChunksOf(job->src_, job->src_n_);
  if (nch == 0) {
    if (lane == 0 && warp == 0) job->ok_ = 0;
    return;
  }
  auto *t = static_cast<AnsTables *>(scratch);
  for (uint32_t c = warp; c < nch; c += nwarps) {
    ChunkSpan sp;
    bool ok = LocateChunk(job->src_, job->src_n_, c, job->dst_, job->dst_cap_,
                          &sp);
    if (ok && sp.raw) {
      for (uint32_t k = lane; k < sp.want; k += nl) sp.out[k] = sp.src[k];
    } else if (ok && sp.magic == kMagicAns) {
      ok = DecodeChunkAnsCoop(sp, lane, nl, sync, t);
    } else if (ok) {
      ok = DecodeBlockCoop(sp.src, sp.sn, sp.out, sp.want, lane, nl, sync) ==
           static_cast<int64_t>(sp.want);
    }
    if (!ok) job->ok_ = 0;  // benign race: same value
  }
}

// ---------------------------------------------------------------- host side

/**
 * Normalize a histogram to frequencies summing to kProbScale, every present
 * symbol >= 1 (a symbol with frequency 0 could not be coded).
 * @param cnt   byte histogram of n bytes (n > 0)
 * @param freq  OUT: normalized frequencies
 */
inline void NormalizeFreqs(const uint32_t *cnt, uint32_t n, uint16_t *freq) {
  int32_t sum = 0;
  for (int s = 0; s < 256; ++s) {
    uint32_t f = 0;
    if (cnt[s] != 0) {
      f = static_cast<uint32_t>(
          (static_cast<uint64_t>(cnt[s]) * kProbScale + n / 2) / n);
      if (f == 0) f = 1;
    }
    freq[s] = static_cast<uint16_t>(f);
    sum += static_cast<int32_t>(f);
  }
  // Fix the rounding drift on the largest entries, never taking one to 0.
  while (sum != static_cast<int32_t>(kProbScale)) {
    int best = -1;
    for (int s = 0; s < 256; ++s) {
      if (freq[s] == 0 || (sum > static_cast<int32_t>(kProbScale) && freq[s] < 2)) {
        continue;
      }
      if (best < 0 || freq[s] > freq[best]) best = s;
    }
    if (sum > static_cast<int32_t>(kProbScale)) {
      --freq[best];
      --sum;
    } else {
      ++freq[best];
      ++sum;
    }
  }
}

/**
 * Encode one chunk (host). Returns 0 when the chunk does not shrink.
 * @param src  chunk bytes; n <= kChunkRaw
 * @param out  receives the encoded chunk (layout in the file comment)
 */
inline size_t EncodeChunkAns(const uint8_t *src, uint32_t n,
                             std::vector<uint8_t> &out) {
  uint32_t cnt[256] = {0};
  for (uint32_t i = 0; i < n; ++i) ++cnt[src[i]];
  uint16_t freq[256];
  NormalizeFreqs(cnt, n, freq);
  uint32_t cumf[257];
  cumf[0] = 0;
  for (int s = 0; s < 256; ++s) cumf[s + 1] = cumf[s] + freq[s];
  out.assign(kAnsTableBytes, 0);
  for (int s = 0; s < 256; ++s) {
    out[2 * s] = static_cast<uint8_t>(freq[s] & 0xFF);
    out[2 * s + 1] = static_cast<uint8_t>(freq[s] >> 8);
  }
  std::vector<uint8_t> rev;  // one stream, bytes emitted back to front
  for (uint32_t st = 0; st < kAnsLanes; ++st) {
    rev.clear();
    uint32_t x = kRansL;
    // Encode this stream's bytes in REVERSE so the decoder runs forward.
    uint32_t last = st;
    if (st < n) last = st + (n - 1 - st) / kAnsLanes * kAnsLanes;
    for (int64_t i = (st < n ? static_cast<int64_t>(last) : -1); i >= 0;
         i -= kAnsLanes) {
      const uint32_t s = src[i];
      const uint32_t f = freq[s];
      const uint32_t x_max = ((kRansL >> kProbBits) << 8) * f;
      while (x >= x_max) {
        rev.push_back(static_cast<uint8_t>(x & 0xFF));
        x >>= 8;
      }
      x = ((x / f) << kProbBits) + (x % f) + cumf[s];
    }
    const size_t at = out.size() - kAnsTableBytes;
    if (at > 0xFFFFu) return 0;
    out[512 + 2 * st] = static_cast<uint8_t>(at & 0xFF);
    out[512 + 2 * st + 1] = static_cast<uint8_t>(at >> 8);
    for (int k = 0; k < 4; ++k) out.push_back(static_cast<uint8_t>(x >> (8 * k)));
    out.insert(out.end(), rev.rbegin(), rev.rend());
    if (out.size() >= n) return 0;  // not worth it: store the chunk raw
  }
  const size_t total = out.size() - kAnsTableBytes;
  if (total > 0xFFFFu) return 0;
  out[512 + 2 * kAnsLanes] = static_cast<uint8_t>(total & 0xFF);
  out[512 + 2 * kAnsLanes + 1] = static_cast<uint8_t>(total >> 8);
  return out.size();
}

/**
 * Encode a page as gpu_ans (host). Chunks that do not shrink are stored raw.
 * @param chunk raw bytes per chunk (<= 64 KiB). 64 KiB keeps the per-chunk
 *              table (578 B) under 1%.
 */
inline void EncodePageAns(const void *src, size_t n, std::vector<uint8_t> &out,
                          uint32_t chunk = kChunkRaw) {
  const auto *s = static_cast<const uint8_t *>(src);
  const uint32_t nch = NumChunks(n, chunk);
  const size_t hb = HeaderBytes(nch);
  out.assign(hb, 0);
  Header h{kMagicAns, kVersion, static_cast<uint64_t>(n), chunk, nch};
  std::memcpy(out.data(), &h, sizeof(h));
  std::vector<uint32_t> off(nch + 1, 0);
  std::vector<uint8_t> enc;
  for (uint32_t c = 0; c < nch; ++c) {
    const size_t at = static_cast<size_t>(c) * chunk;
    const uint32_t len = static_cast<uint32_t>(n - at < chunk ? n - at : chunk);
    off[c] = static_cast<uint32_t>(out.size() - hb);
    if (EncodeChunkAns(s + at, len, enc) != 0) {
      out.insert(out.end(), enc.begin(), enc.end());
    } else {
      off[c] |= kRawBit;
      out.insert(out.end(), s + at, s + at + len);
    }
  }
  off[nch] = static_cast<uint32_t>(out.size() - hb);
  std::memcpy(out.data() + sizeof(Header), off.data(),
              sizeof(uint32_t) * (nch + 1));
}

/**
 * Decode a whole page on the host -- LZ4 or ANS, by its magic -- with the
 * same chunk decoders the GPU runs (one lane, a no-op barrier).
 * @return true when every chunk decoded and the page fits in `cap`
 */
inline bool DecodePageAny(const void *page, size_t page_n, void *dst,
                          size_t cap) {
  const auto *p = static_cast<const uint8_t *>(page);
  const uint64_t raw = RawSizeOf(p, page_n);
  if (raw == 0 || raw > cap) return false;
  DecodeJob job{p, page_n, static_cast<uint8_t *>(dst), cap, 1, 0};
  std::vector<uint8_t> scratch(kDecodeScratch);
  DecodeJobWarpAny(&job, 0, 1, 0, 1, []() {}, scratch.data());
  return job.ok_ == 1u;
}

/** @return true when `page` is a gpu_lz4 or gpu_ans page. */
inline bool IsEncodedPage(const void *page, size_t page_n) {
  return RawSizeOf(static_cast<const uint8_t *>(page), page_n) != 0;
}

}  // namespace ctp::gpu_lz4

#endif  // CLIO_CTP_COMPRESS_GPU_ANS_H_
