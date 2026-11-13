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
 * gpu_lz4: a chunked LZ4 page format whose decoder runs unchanged on the host
 * and on any GPU backend (CUDA, ROCm, SYCL).
 *
 * WHY NOT nvCOMP / sycl-lz4. nvCOMP is CUDA-only, and the sycl-lz4 codec is
 * SYCL-only with its own queue. One decoder function, compiled by whichever
 * GPU toolchain builds the caller, decodes the same bytes everywhere: pages
 * assimilated on one machine decode on an NVIDIA or an Intel GPU alike.
 *
 * FORMAT (one page = one blob):
 *
 *   Header      magic "GLZ4", version, raw_size, chunk_raw, nchunks
 *   u32 off[nchunks + 1]   byte offset of each chunk in the payload; bit 31 of
 *                          off[i] set = chunk i is stored raw (incompressible)
 *   payload     chunk i occupies [off[i] & kOffMask, off[i+1] & kOffMask)
 *
 * Each compressed chunk is ONE standard LZ4 block (liblz4's block format, the
 * end-of-block rules included), so liblz4 can decode any chunk and the format
 * can be cross-checked against it. Chunks are independent: a GPU decodes a
 * page with one work-group, one chunk per work-item, and parses the header
 * itself -- the host never reads a page header back from device memory.
 */

#ifndef CLIO_CTP_COMPRESS_GPU_LZ4_H_
#define CLIO_CTP_COMPRESS_GPU_LZ4_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "clio_ctp/constants/macros.h"

namespace ctp::gpu_lz4 {

/** "GLZ4", little-endian. */
inline constexpr uint32_t kMagic = 0x345A4C47u;
/** "GANS": the same container holding rANS chunks (gpu_ans.h). */
inline constexpr uint32_t kMagicAns = 0x534E4147u;

/** @return true for a magic this container format uses. */
CTP_CROSS_FUN inline bool IsPageMagic(uint32_t m) {
  return m == kMagic || m == kMagicAns;
}
inline constexpr uint32_t kVersion = 1;
/** Raw bytes per chunk. 64 KiB: a 960 KiB page is 15 chunks, and an LZ4
 *  offset (16 bits) can reach anywhere in a chunk. */
inline constexpr uint32_t kChunkRaw = 64u * 1024u;
inline constexpr uint32_t kRawBit = 0x80000000u;
inline constexpr uint32_t kOffMask = 0x7FFFFFFFu;

/** Fixed part of a page header; the offset table follows it. */
struct Header {
  uint32_t magic;
  uint32_t version;
  uint64_t raw_size;
  uint32_t chunk_raw;
  uint32_t nchunks;
};

/** @return bytes of header + offset table for `nchunks` chunks. */
CTP_CROSS_FUN inline size_t HeaderBytes(uint32_t nchunks) {
  return sizeof(Header) + sizeof(uint32_t) * (static_cast<size_t>(nchunks) + 1);
}

/** @return chunk count for a page of `raw` bytes in `chunk`-byte chunks. */
CTP_CROSS_FUN inline uint32_t NumChunks(uint64_t raw,
                                        uint32_t chunk = kChunkRaw) {
  return static_cast<uint32_t>((raw + chunk - 1) / chunk);
}

/** @return worst-case encoded size of a page of `raw` bytes (every chunk
 *  stored raw, plus the header). */
inline size_t MaxEncodedSize(size_t raw) {
  return HeaderBytes(NumChunks(raw)) + raw;
}

/**
 * Decode one LZ4 block. Byte-wise and branch-light so the same code runs on a
 * GPU work-item; every read and write is bounds-checked, so a corrupt block
 * fails instead of writing out of range.
 * @param src   the compressed block
 * @param sn    its length in bytes
 * @param dst   output buffer
 * @param dn    output capacity in bytes
 * @return bytes produced, or -1 when the block is malformed
 */
CTP_CROSS_FUN inline int64_t DecodeBlock(const uint8_t *src, uint32_t sn,
                                         uint8_t *dst, uint32_t dn) {
  uint32_t ip = 0, op = 0;
  while (ip < sn) {
    const uint32_t tok = src[ip++];
    uint32_t ll = tok >> 4;
    if (ll == 15) {
      uint32_t b = 0;
      do {
        if (ip >= sn) return -1;
        b = src[ip++];
        ll += b;
      } while (b == 255);
    }
    if (ll > sn - ip || ll > dn - op) return -1;
    for (uint32_t k = 0; k < ll; ++k) dst[op++] = src[ip++];
    if (ip == sn) break;  // the last sequence carries literals only
    if (sn - ip < 2) return -1;
    const uint32_t off = src[ip] | (static_cast<uint32_t>(src[ip + 1]) << 8);
    ip += 2;
    if (off == 0 || off > op) return -1;
    uint32_t ml = tok & 15u;
    if (ml == 15) {
      uint32_t b = 0;
      do {
        if (ip >= sn) return -1;
        b = src[ip++];
        ml += b;
      } while (b == 255);
    }
    ml += 4;
    if (ml > dn - op) return -1;
    // Overlapping copy (off < ml) is the run-length case: byte order matters.
    for (uint32_t k = 0; k < ml; ++k, ++op) dst[op] = dst[op - off];
  }
  return static_cast<int64_t>(op);
}

/** Where chunk `c` of a page lives, and where it decodes to. */
struct ChunkSpan {
  const uint8_t *src;  /**< the chunk's stored bytes */
  uint32_t sn;         /**< stored length */
  uint8_t *out;        /**< destination of the chunk's raw bytes */
  uint32_t want;       /**< raw length of the chunk */
  bool raw;            /**< stored uncompressed */
  uint32_t magic;      /**< the page's codec (kMagic / kMagicAns) */
};

/**
 * Resolve chunk `c` of an encoded page whose header is at `page`. Every field
 * is read byte-wise (a page in a tier has no alignment guarantee) and every
 * offset is checked against page_n and dst_cap.
 * @param page      encoded page (header first)
 * @param page_n    encoded page size in bytes
 * @param c         chunk index
 * @param dst       start of the decoded PAGE (chunk c lands at c * chunk_raw)
 * @param dst_cap   capacity of dst in bytes
 * @param span      OUT: the chunk's location
 * @return false when the page is malformed or does not fit in dst_cap
 */
CTP_CROSS_FUN inline bool LocateChunk(const uint8_t *page, uint64_t page_n,
                                      uint32_t c, uint8_t *dst,
                                      uint64_t dst_cap, ChunkSpan *span) {
  if (page_n < sizeof(Header)) return false;
  const uint8_t *p = page;
  auto rd32 = [p](size_t at) {
    return static_cast<uint32_t>(p[at]) | static_cast<uint32_t>(p[at + 1]) << 8 |
           static_cast<uint32_t>(p[at + 2]) << 16 |
           static_cast<uint32_t>(p[at + 3]) << 24;
  };
  const uint32_t magic = rd32(0);
  if (!IsPageMagic(magic)) return false;
  uint64_t raw = 0;
  for (int i = 7; i >= 0; --i) raw = (raw << 8) | p[8 + i];
  const uint32_t chunk_raw = rd32(16);
  const uint32_t nch = rd32(20);
  if (c >= nch || chunk_raw == 0 || chunk_raw > kChunkRaw || raw > dst_cap) {
    return false;
  }
  if (HeaderBytes(nch) > page_n) return false;
  const size_t tab = sizeof(Header);
  const uint32_t o0 = rd32(tab + 4 * static_cast<size_t>(c));
  const uint32_t o1 = rd32(tab + 4 * (static_cast<size_t>(c) + 1));
  const uint64_t s = (o0 & kOffMask), e = (o1 & kOffMask);
  const uint64_t base = HeaderBytes(nch);
  if (e < s || base + e > page_n) return false;
  const uint64_t at = static_cast<uint64_t>(c) * chunk_raw;
  if (at >= raw) return false;
  span->src = page + base + s;
  span->sn = static_cast<uint32_t>(e - s);
  span->out = dst + at;
  span->want = static_cast<uint32_t>(raw - at < chunk_raw ? raw - at : chunk_raw);
  span->raw = (o0 & kRawBit) != 0;
  span->magic = magic;
  return !span->raw || span->sn == span->want;
}

/**
 * Decode chunk `c` of an encoded page on ONE thread (host decode, and the
 * reference the cooperative GPU decoder is tested against).
 * @return true when the chunk decoded to exactly its raw length
 */
CTP_CROSS_FUN inline bool DecodeChunk(const uint8_t *page, uint64_t page_n,
                                      uint32_t c, uint8_t *dst,
                                      uint64_t dst_cap) {
  ChunkSpan sp;
  if (!LocateChunk(page, page_n, c, dst, dst_cap, &sp)) return false;
  if (sp.raw) {
    for (uint32_t k = 0; k < sp.want; ++k) sp.out[k] = sp.src[k];
    return true;
  }
  if (sp.magic != kMagic) return false;  // an ANS page: see gpu_ans.h
  return DecodeBlock(sp.src, sp.sn, sp.out, sp.want) ==
         static_cast<int64_t>(sp.want);
}

/** @return the decoded size an encoded page declares, or 0 if not a page. */
CTP_CROSS_FUN inline uint64_t RawSizeOf(const uint8_t *page, uint64_t page_n) {
  if (page_n < sizeof(Header)) return 0;
  const uint32_t m = static_cast<uint32_t>(page[0]) |
                     static_cast<uint32_t>(page[1]) << 8 |
                     static_cast<uint32_t>(page[2]) << 16 |
                     static_cast<uint32_t>(page[3]) << 24;
  if (!IsPageMagic(m)) return 0;
  uint64_t raw = 0;
  for (int i = 7; i >= 0; --i) raw = (raw << 8) | page[8 + i];
  return raw;
}

/** @return the chunk count an encoded page declares, or 0 if not a page. */
CTP_CROSS_FUN inline uint32_t ChunksOf(const uint8_t *page, uint64_t page_n) {
  if (RawSizeOf(page, page_n) == 0) return 0;
  return static_cast<uint32_t>(page[20]) |
         static_cast<uint32_t>(page[21]) << 8 |
         static_cast<uint32_t>(page[22]) << 16 |
         static_cast<uint32_t>(page[23]) << 24;
}

// ------------------------------------------------------------ GPU decoding

/**
 * One page to decode on the GPU. The job array must be device-accessible.
 * ok_ is written by the kernel: 1 when every chunk decoded, 0 otherwise.
 */
struct DecodeJob {
  const uint8_t *src_;   /**< encoded page (device-accessible) */
  uint64_t src_n_;       /**< encoded bytes */
  uint8_t *dst_;         /**< decoded page destination (device-accessible) */
  uint64_t dst_cap_;     /**< destination capacity in bytes */
  uint32_t ok_;          /**< OUT: 1 = decoded, 0 = failed */
  uint32_t pad_;
};

/** Work-items per page in the decode launch. Each sub-group (CUDA warp)
 *  decodes whole chunks cooperatively. 256 = 8 warps of 32: every sub-group
 *  also owns a kDecodeScratch slice of local memory for the rANS tables
 *  (gpu_ans.h), and 8 x 5 KiB stays inside CUDA's 48 KiB static limit. */
inline constexpr uint32_t kDecodeGroup = 256;

/** Read an LZ4 length extension (the 255-continuation bytes). */
CTP_CROSS_FUN inline bool ReadLenExt(const uint8_t *src, uint32_t sn,
                                     uint32_t *ip, uint32_t *len) {
  uint32_t b = 0;
  do {
    if (*ip >= sn) return false;
    b = src[(*ip)++];
    *len += b;
  } while (b == 255);
  return true;
}

/**
 * Decode one LZ4 block COOPERATIVELY: every lane of a sub-group calls this
 * with the same arguments except `lane`. All lanes parse the same sequence
 * headers (identical control flow, so `sync` -- a sub-group barrier -- is
 * reached uniformly) and split the byte copies:
 *   - literals: lane k copies bytes k, k + nl, ...
 *   - a match of offset `off` is the last `off` output bytes repeated, so
 *     output byte i = dst[op - off + i % off]: no byte depends on another
 *     byte of the same match, and every lane copies independently.
 * One barrier per sequence makes the literals (and earlier matches) visible
 * before a match reads them.
 * @param lane  this lane's index in the sub-group
 * @param nl    lanes in the sub-group
 * @param sync  sub-group barrier with memory visibility (callable)
 * @return bytes produced, or -1 when the block is malformed (uniformly)
 */
template <typename SyncT>
CTP_CROSS_FUN inline int64_t DecodeBlockCoop(const uint8_t *src, uint32_t sn,
                                             uint8_t *dst, uint32_t dn,
                                             uint32_t lane, uint32_t nl,
                                             SyncT &&sync) {
  uint32_t ip = 0, op = 0;
  while (ip < sn) {
    const uint32_t tok = src[ip++];
    uint32_t ll = tok >> 4;
    if (ll == 15 && !ReadLenExt(src, sn, &ip, &ll)) return -1;
    if (ll > sn - ip || ll > dn - op) return -1;
    for (uint32_t k = lane; k < ll; k += nl) dst[op + k] = src[ip + k];
    ip += ll;
    op += ll;
    if (ip == sn) break;  // the last sequence carries literals only
    if (sn - ip < 2) return -1;
    const uint32_t off = src[ip] | (static_cast<uint32_t>(src[ip + 1]) << 8);
    ip += 2;
    if (off == 0 || off > op) return -1;
    uint32_t ml = tok & 15u;
    if (ml == 15 && !ReadLenExt(src, sn, &ip, &ml)) return -1;
    ml += 4;
    if (ml > dn - op) return -1;
    sync();  // literals and earlier matches visible before this match reads
    const uint32_t from = op - off;
    if (off >= ml) {
      for (uint32_t k = lane; k < ml; k += nl) dst[op + k] = dst[from + k];
    } else {
      for (uint32_t k = lane; k < ml; k += nl) {
        dst[op + k] = dst[from + k % off];
      }
    }
    op += ml;
  }
  sync();
  return static_cast<int64_t>(op);
}

/**
 * Decode the chunks of one page that belong to sub-group `warp` of the page's
 * work-group (warp, warp + nwarps, ...), each chunk by all lanes of the
 * sub-group. Shared by every backend's kernel.
 * @param job     the page
 * @param lane    lane index in the sub-group
 * @param nl      lanes per sub-group
 * @param warp    sub-group index in the work-group
 * @param nwarps  sub-groups in the work-group
 * @param sync    sub-group barrier (callable)
 */
template <typename SyncT>
CTP_CROSS_FUN inline void DecodeJobWarp(DecodeJob *job, uint32_t lane,
                                        uint32_t nl, uint32_t warp,
                                        uint32_t nwarps, SyncT &&sync) {
  const uint32_t nch = ChunksOf(job->src_, job->src_n_);
  if (nch == 0) {
    if (lane == 0 && warp == 0) job->ok_ = 0;
    return;
  }
  for (uint32_t c = warp; c < nch; c += nwarps) {
    ChunkSpan sp;
    bool ok = LocateChunk(job->src_, job->src_n_, c, job->dst_, job->dst_cap_,
                          &sp);
    if (ok && sp.raw) {
      for (uint32_t k = lane; k < sp.want; k += nl) sp.out[k] = sp.src[k];
    } else if (ok && sp.magic != kMagic) {
      ok = false;  // an ANS page: DecodeJobWarpAny (gpu_ans.h) handles it
    } else if (ok) {
      ok = DecodeBlockCoop(sp.src, sp.sn, sp.out, sp.want, lane, nl, sync) ==
           static_cast<int64_t>(sp.want);
    }
    if (!ok && lane == 0) job->ok_ = 0;  // benign race: same value
  }
}

}  // namespace ctp::gpu_lz4

/**
 * Launch the decode of `njobs` pages (one work-group each) on `stream` (a
 * cudaStream_t / hipStream_t, or a sycl::queue* -- GpuApi's stream type).
 * Asynchronous: completion is the stream's. The caller sets every job's ok_
 * to 1 before the launch. Defined in the runtime's GPU library
 * (context-runtime/src/gpu/gpu_lz4_{gpu,sycl}.cc).
 * @return 0 when launched, nonzero when no GPU backend is built
 */
extern "C" int ctp_gpu_lz4_decode_launch(ctp::gpu_lz4::DecodeJob *jobs,
                                         uint32_t njobs, void *stream);

namespace ctp::gpu_lz4 {

// ---------------------------------------------------------------- host side

/**
 * Encode one LZ4 block (host). Hash-chain matcher: each position's 4-byte
 * prefix is hashed, and up to `depth` earlier positions with the same hash are
 * tried; the longest match wins. Follows liblz4's end-of-block rules (the last
 * 5 bytes are literals, no match starts in the last 12), so liblz4 decodes it.
 * @param src    input
 * @param n      input bytes (<= kChunkRaw, so 16-bit offsets reach all of it)
 * @param dst    output; must hold at least n + n / 255 + 16 bytes
 * @param depth  chain candidates tried per position (ratio vs speed)
 * @return encoded bytes
 */
inline size_t EncodeBlock(const uint8_t *src, uint32_t n, uint8_t *dst,
                          int depth = 32) {
  constexpr uint32_t kMinMatch = 4, kLastLiterals = 5, kMfLimit = 12;
  constexpr uint32_t kHashBits = 15;
  std::vector<int32_t> head(1u << kHashBits, -1);
  std::vector<int32_t> prev(n > 0 ? n : 1, -1);
  auto rd32 = [&](uint32_t at) {
    uint32_t v;
    std::memcpy(&v, src + at, 4);
    return v;
  };
  auto hash = [&](uint32_t at) {
    return (rd32(at) * 2654435761u) >> (32 - kHashBits);
  };
  auto insert = [&](uint32_t at) {
    const uint32_t h = hash(at);
    prev[at] = head[h];
    head[h] = static_cast<int32_t>(at);
  };
  size_t op = 0;
  auto put_len = [&](uint32_t v) {  // the 255-continuation tail of a length
    while (v >= 255) {
      dst[op++] = 255;
      v -= 255;
    }
    dst[op++] = static_cast<uint8_t>(v);
  };
  auto emit = [&](uint32_t lit_at, uint32_t ll, uint32_t off, uint32_t ml) {
    uint8_t *tok = dst + op++;
    *tok = static_cast<uint8_t>((ll >= 15 ? 15 : ll) << 4);
    if (ll >= 15) put_len(ll - 15);
    std::memcpy(dst + op, src + lit_at, ll);
    op += ll;
    if (ml == 0) return;  // last sequence: literals only
    dst[op++] = static_cast<uint8_t>(off & 0xFF);
    dst[op++] = static_cast<uint8_t>(off >> 8);
    const uint32_t mlc = ml - kMinMatch;
    *tok |= static_cast<uint8_t>(mlc >= 15 ? 15 : mlc);
    if (mlc >= 15) put_len(mlc - 15);
  };
  uint32_t ip = 0, anchor = 0;
  if (n > kMfLimit) {
    const uint32_t match_start_limit = n - kMfLimit;   // last legal start
    const uint32_t match_end_limit = n - kLastLiterals;
    // Longest match at `at` among `depth` chain candidates (len 0 if none).
    auto find = [&](uint32_t at, uint32_t *off_out) {
      uint32_t best_len = 0;
      const uint32_t v = rd32(at);
      int32_t cand = head[hash(at)];
      for (int d = 0; d < depth && cand >= 0; ++d, cand = prev[cand]) {
        const uint32_t off = at - static_cast<uint32_t>(cand);
        if (off > 65535) break;
        if (rd32(static_cast<uint32_t>(cand)) != v) continue;
        uint32_t len = kMinMatch;
        while (at + len < match_end_limit &&
               src[cand + len] == src[at + len]) {
          ++len;
        }
        if (len > best_len) {
          best_len = len;
          *off_out = off;
        }
      }
      return best_len;
    };
    while (ip <= match_start_limit) {
      uint32_t best_off = 0;
      uint32_t best_len = find(ip, &best_off);
      // LAZY step: a longer match one byte later beats taking this one now
      // (one literal costs less than the bytes the longer match saves).
      if (best_len >= kMinMatch && ip + 1 <= match_start_limit) {
        insert(ip);
        uint32_t next_off = 0;
        const uint32_t next_len = find(ip + 1, &next_off);
        if (next_len > best_len + 1) {
          ++ip;
          best_len = next_len;
          best_off = next_off;
        } else {
          // Take the match at ip. ip is already in the chain, so insertion
          // resumes at ip + 1.
          emit(anchor, ip - anchor, best_off, best_len);
          for (uint32_t k = 1; k < best_len; ++k) {
            if (ip + k <= match_start_limit) insert(ip + k);
          }
          ip += best_len;
          anchor = ip;
          continue;
        }
      }
      if (best_len >= kMinMatch) {
        emit(anchor, ip - anchor, best_off, best_len);
        for (uint32_t k = 0; k < best_len; ++k) {
          if (ip + k <= match_start_limit) insert(ip + k);
        }
        ip += best_len;
        anchor = ip;
      } else {
        insert(ip);
        ++ip;
      }
    }
  }
  emit(anchor, n - anchor, 0, 0);
  return op;
}

/**
 * Encode a page (host). Chunks that do not shrink are stored raw.
 * @param src  raw page
 * @param n    raw bytes (> 0)
 * @param out  receives the encoded page (resized)
 * @param depth hash-chain depth (see EncodeBlock)
 * @param chunk raw bytes per chunk (<= 64 KiB; smaller = more GPU decode
 *              parallelism, slightly lower ratio). Recorded in the header.
 */
inline void EncodePage(const void *src, size_t n, std::vector<uint8_t> &out,
                       int depth = 32, uint32_t chunk = kChunkRaw) {
  const auto *s = static_cast<const uint8_t *>(src);
  const uint32_t nch = NumChunks(n, chunk);
  const size_t hb = HeaderBytes(nch);
  out.resize(hb + n + nch * (chunk / 255 + 16));
  Header h{kMagic, kVersion, static_cast<uint64_t>(n), chunk, nch};
  std::memcpy(out.data(), &h, sizeof(h));
  std::vector<uint32_t> off(nch + 1, 0);
  size_t pos = 0;  // payload bytes written
  for (uint32_t c = 0; c < nch; ++c) {
    const size_t at = static_cast<size_t>(c) * chunk;
    const uint32_t len = static_cast<uint32_t>(n - at < chunk ? n - at : chunk);
    uint8_t *dst = out.data() + hb + pos;
    const size_t enc = EncodeBlock(s + at, len, dst, depth);
    off[c] = static_cast<uint32_t>(pos);
    if (enc >= len) {
      std::memcpy(dst, s + at, len);
      off[c] |= kRawBit;
      pos += len;
    } else {
      pos += enc;
    }
  }
  off[nch] = static_cast<uint32_t>(pos);
  std::memcpy(out.data() + sizeof(Header), off.data(),
              sizeof(uint32_t) * (nch + 1));
  out.resize(hb + pos);
}

/**
 * Decode a whole page on the host, chunk by chunk (the same DecodeChunk a GPU
 * work-item runs).
 * @return true when every chunk decoded and the page fits in `cap`
 */
inline bool DecodePage(const void *page, size_t page_n, void *dst,
                       size_t cap) {
  const auto *p = static_cast<const uint8_t *>(page);
  const uint64_t raw = RawSizeOf(p, page_n);
  if (raw == 0 || raw > cap) return false;
  const uint32_t nch = ChunksOf(p, page_n);
  for (uint32_t c = 0; c < nch; ++c) {
    if (!DecodeChunk(p, page_n, c, static_cast<uint8_t *>(dst), cap)) {
      return false;
    }
  }
  return true;
}

}  // namespace ctp::gpu_lz4

#endif  // CLIO_CTP_COMPRESS_GPU_LZ4_H_
