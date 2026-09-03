/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. (BSD-3-Clause; see repository headers.)
 *
 * SYCL LZ4 kernels -- the device half of the "nvcomp equivalent for SYCL".
 *
 * This translation unit is compiled by a SYCL compiler (Intel DPC++ /
 * AdaptiveCpp), NOT by the ordinary CTP toolchain. It exposes a small extern
 * "C" ABI (clio_sycl_lz4_*) that the header-only ctp::SyclLz4 wrapper calls,
 * mirroring how sycl_zfp delegates its device work to a SYCL-built libzfp.
 *
 * FORMAT (self-describing, chunked -- the same architecture nvcomp's LZ4 uses):
 *   Header  { u32 magic; u32 chunk_size; u64 orig_size; u32 n_chunks; u32 rsvd }
 *   u32     comp_size[n_chunks]         // high bit set => chunk stored RAW
 *   bytes   payload                     // chunks concatenated, in order
 * Each chunk payload is a STANDARD LZ4 block (LZ4_compress_default-compatible),
 * so a chunk can be cross-validated against the reference liblz4 in either
 * direction. Chunks are independent -> one WORK-GROUP handles each (its hash
 * table and data staged in local memory; decode is cooperative across the
 * sub-group's lanes), thousands of chunks in flight -- nvcomp's parallelism
 * model. See build_and_test.sh for the validated build recipe.
 */

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

constexpr u32 kMagic = 0x5A344C53u;   // "SL4Z"
// 4 KiB chunks with a u16 hash: keeps per-work-group LOCAL memory small
// (src 4 KiB + hash 2 KiB) so 6-8 groups fit per SM concurrently -- the A100
// then processes hundreds of chunks at once instead of ~one-per-SM. A 32 MB
// input becomes 8192 independent chunks. Small chunks give up a little ratio
// (less LZ4 history) for a large concurrency win. Positions stay < 4096, so a
// u16 hash entry is exact.
constexpr u32 kChunkSize = 1u << 12;  // 4 KiB
constexpr u32 kHashLog = 11;          // 2048-entry hash table per chunk
constexpr u32 kHashSize = 1u << kHashLog;
constexpr u16 kHashNull = 0xFFFFu;    // u16 sentinel (chunk < 65536 bytes)
// One SUB-GROUP (32 lanes) per chunk. Decode is cooperative across the lanes:
// lane 0 parses each LZ4 sequence header, all 32 lanes then copy the literals
// and the match body in parallel -- the byte copies are the bulk of decode
// work and this is exactly how nvcomp gets its decode throughput.
constexpr u32 kGroup = 32;
constexpr u32 kStoredFlag = 0x80000000u;  // high bit of comp_size => raw chunk

// LZ4 block-format constants.
constexpr int kMinMatch = 4;
constexpr int kLastLiterals = 5;   // last 5 bytes are always literals
constexpr int kMfLimit = 12;       // no match may start in the last 12 bytes
constexpr u32 kMaxOffset = 65535;  // 16-bit offsets

struct Header {
  u32 magic;
  u32 chunk_size;
  u64 orig_size;
  u32 n_chunks;
  u32 rsvd;
};

// LZ4_compressBound: worst-case size of a compressed chunk.
inline u32 CompressBound(u32 n) { return n + n / 255u + 16u; }

inline u32 Hash4(u32 seq) { return (seq * 2654435761u) >> (32 - kHashLog); }

// ---- device: compress one chunk into a standard LZ4 block --------------------
// src[0..n) -> dst (capacity >= CompressBound(n)); hash is this chunk's local-
// memory scratch (kHashSize u16). Returns compressed byte count, or 0 on
// overflow. Greedy single-pass match finder; byte-exact LZ4 block output.
inline u32 Lz4CompressChunk(const u8 *src, u32 n, u8 *dst, u32 dst_cap,
                            u16 *hash) {
  for (u32 i = 0; i < kHashSize; ++i) hash[i] = kHashNull;
  u32 ip = 0, anchor = 0, op = 0;
  auto load32 = [&](u32 p) -> u32 {
    u32 v;
    v = (u32)src[p] | ((u32)src[p + 1] << 8) | ((u32)src[p + 2] << 16) |
        ((u32)src[p + 3] << 24);
    return v;
  };
  auto emit_len = [&](u32 len) -> bool {
    while (len >= 255) {
      if (op >= dst_cap) return false;
      dst[op++] = 255;
      len -= 255;
    }
    if (op >= dst_cap) return false;
    dst[op++] = (u8)len;
    return true;
  };
  if (n >= (u32)(kMfLimit + kMinMatch)) {
    const u32 mflimit = n - kMfLimit;
    while (ip < mflimit) {
      u32 h = Hash4(load32(ip));
      u32 ref = hash[h];
      hash[h] = (u16)ip;
      bool have = (ref != kHashNull) && (ip - ref) <= kMaxOffset &&
                  load32(ref) == load32(ip);
      if (!have) {
        ++ip;
        continue;
      }
      // Extend the match forward (bounded so the last 5 bytes stay literals).
      u32 mstart = ip, off = ip - ref;
      ip += kMinMatch;
      u32 r = ref + kMinMatch;
      const u32 mend = n - kLastLiterals;
      while (ip < mend && src[ip] == src[r]) {
        ++ip;
        ++r;
      }
      u32 litlen = mstart - anchor;
      u32 matchlen = (ip - mstart) - kMinMatch;
      // token
      if (op >= dst_cap) return 0;
      u32 tok_pos = op++;
      u8 tok = (u8)((litlen >= 15 ? 15 : litlen) << 4) |
               (u8)(matchlen >= 15 ? 15 : matchlen);
      dst[tok_pos] = tok;
      if (litlen >= 15) {
        if (!emit_len(litlen - 15)) return 0;
      }
      // literals
      if (op + litlen > dst_cap) return 0;
      for (u32 k = 0; k < litlen; ++k) dst[op++] = src[anchor + k];
      // offset (little-endian)
      if (op + 2 > dst_cap) return 0;
      dst[op++] = (u8)(off & 0xFF);
      dst[op++] = (u8)((off >> 8) & 0xFF);
      if (matchlen >= 15) {
        if (!emit_len(matchlen - 15)) return 0;
      }
      anchor = ip;
    }
  }
  // Final literals: everything from anchor to end.
  u32 litlen = n - anchor;
  if (op >= dst_cap) return 0;
  u32 tok_pos = op++;
  dst[tok_pos] = (u8)((litlen >= 15 ? 15 : litlen) << 4);
  if (litlen >= 15) {
    u32 l = litlen - 15;
    while (l >= 255) {
      if (op >= dst_cap) return 0;
      dst[op++] = 255;
      l -= 255;
    }
    if (op >= dst_cap) return 0;
    dst[op++] = (u8)l;
  }
  if (op + litlen > dst_cap) return 0;
  for (u32 k = 0; k < litlen; ++k) dst[op++] = src[anchor + k];
  return op;
}

// ---- device: decompress one standard LZ4 block ------------------------------
// src[0..n) -> dst (capacity >= expected out). Returns bytes written, or 0 on
// malformed input / overflow. Overlap-safe byte copy for matches.
inline u32 Lz4DecompressChunk(const u8 *src, u32 n, u8 *dst, u32 dst_cap) {
  u32 ip = 0, op = 0;
  while (ip < n) {
    u8 token = src[ip++];
    u32 litlen = token >> 4;
    if (litlen == 15) {
      u8 b;
      do {
        if (ip >= n) return 0;
        b = src[ip++];
        litlen += b;
      } while (b == 255);
    }
    if (ip + litlen > n || op + litlen > dst_cap) return 0;
    for (u32 k = 0; k < litlen; ++k) dst[op++] = src[ip++];
    if (ip == n) break;  // last sequence: literals only
    if (ip + 2 > n) return 0;
    u32 off = (u32)src[ip] | ((u32)src[ip + 1] << 8);
    ip += 2;
    u32 matchlen = token & 0x0F;
    if (matchlen == 15) {
      u8 b;
      do {
        if (ip >= n) return 0;
        b = src[ip++];
        matchlen += b;
      } while (b == 255);
    }
    matchlen += kMinMatch;
    if (off == 0 || off > op || op + matchlen > dst_cap) return 0;
    u32 mp = op - off;
    for (u32 k = 0; k < matchlen; ++k) dst[op++] = dst[mp++];
  }
  return op;
}

// A GPU queue (device selector), created once and reused. Buffers may be host
// or device USM; we detect and stage minimally, like NvComp does with CUDA.
struct QueueBox {
  sycl::queue q;
  QueueBox() : q(sycl::gpu_selector_v) {}
};

inline bool IsDevicePtr(const sycl::queue &q, const void *p) {
  if (!p) return false;
  auto kind = sycl::get_pointer_type(p, q.get_context());
  return kind == sycl::usm::alloc::device || kind == sycl::usm::alloc::shared;
}

}  // namespace

extern "C" {

void *clio_sycl_lz4_queue_create() {
  try {
    return new QueueBox();
  } catch (...) {
    return nullptr;
  }
}
void clio_sycl_lz4_queue_destroy(void *qb) { delete static_cast<QueueBox *>(qb); }

// Compress src[0..src_size) -> dst (cap dst_cap); *out set to bytes written.
// Returns 1 on success, 0 on failure. src/dst may each be host or device USM.
int clio_sycl_lz4_compress(void *qbv, const void *src, size_t src_size,
                           void *dst, size_t dst_cap, size_t *out) {
  if (!qbv || !src || !dst || !out) return 0;
  QueueBox *qb = static_cast<QueueBox *>(qbv);
  sycl::queue &q = qb->q;
  const u32 chunk_size = kChunkSize;
  const u32 n_chunks =
      (u32)((src_size + chunk_size - 1) / (src_size ? chunk_size : 1));
  const u32 nc = src_size ? n_chunks : 0;
  const size_t hdr = sizeof(Header) + (size_t)nc * sizeof(u32);
  bool ok = false;
  u8 *d_src = nullptr, *d_scratch = nullptr, *d_out = nullptr;
  u32 *d_csize = nullptr, *d_coff = nullptr;
  try {
    const bool src_dev = IsDevicePtr(q, src);
    d_src = src_dev ? const_cast<u8 *>(static_cast<const u8 *>(src))
                    : sycl::malloc_device<u8>(src_size ? src_size : 1, q);
    if (!d_src) throw std::bad_alloc();
    if (!src_dev) q.memcpy(d_src, src, src_size).wait();

    const u32 bound = CompressBound(chunk_size);
    d_scratch = sycl::malloc_device<u8>((size_t)nc * bound + 1, q);
    d_csize = sycl::malloc_device<u32>(nc + 1, q);
    if (!d_scratch || !d_csize) throw std::bad_alloc();

    // One WORK-GROUP per chunk: stage the chunk's source AND its hash table in
    // local memory (fast), then lane 0 runs the serial LZ4 encode against that
    // local state. The group's other lanes cooperatively load the source. This
    // removes the two killers of the naive version: uncached global hash probes
    // and a mere handful of work-items.
    const u32 bound_c = bound;
    const u32 chunk_c = chunk_size;
    q.submit([&](sycl::handler &h) {
       sycl::local_accessor<u8, 1> s_src(sycl::range<1>(chunk_c), h);
       sycl::local_accessor<u16, 1> s_hash(sycl::range<1>(kHashSize), h);
       h.parallel_for(
           sycl::nd_range<1>(sycl::range<1>((size_t)(nc ? nc : 1) * kGroup),
                             sycl::range<1>(kGroup)),
           [=](sycl::nd_item<1> it) {
             u32 c = (u32)it.get_group(0);
             u32 lid = (u32)it.get_local_id(0);
             if (c >= nc) return;
             u32 off = c * chunk_c;
             u32 len = (off + chunk_c <= src_size) ? chunk_c
                                                   : (u32)(src_size - off);
             u8 *ss = &s_src[0];
             for (u32 k = lid; k < len; k += kGroup) ss[k] = d_src[off + k];
             it.barrier(sycl::access::fence_space::local_space);
             if (lid == 0) {
               u8 *cdst = d_scratch + (size_t)c * bound_c;
               u32 z = Lz4CompressChunk(ss, len, cdst, bound_c, &s_hash[0]);
               if (z == 0 || z >= len) {  // incompressible -> store raw
                 for (u32 k = 0; k < len; ++k) cdst[k] = ss[k];
                 d_csize[c] = len | kStoredFlag;
               } else {
                 d_csize[c] = z;
               }
             }
           });
     }).wait();

    // Pull per-chunk sizes to host, compute packed offsets (prefix sum).
    std::vector<u32> csize(nc);
    if (nc) q.memcpy(csize.data(), d_csize, (size_t)nc * sizeof(u32)).wait();
    std::vector<u64> coff(nc + 1, 0);
    for (u32 c = 0; c < nc; ++c) {
      u32 raw = csize[c] & ~kStoredFlag;
      coff[c + 1] = coff[c] + raw;
    }
    const u64 payload = coff[nc];
    const u64 total = hdr + payload;
    if (total > dst_cap) throw std::runtime_error("dst too small");

    // Gather chunks into their packed positions (device kernel), then copy the
    // packed payload out. Offsets/sizes pushed back to device for the gather.
    const bool dst_dev = IsDevicePtr(q, dst);
    d_out = dst_dev ? static_cast<u8 *>(dst)
                    : sycl::malloc_device<u8>(total ? total : 1, q);
    d_coff = sycl::malloc_device<u32>(nc + 1, q);
    if (!d_out || !d_coff) throw std::bad_alloc();
    std::vector<u32> coff32(nc);
    for (u32 c = 0; c < nc; ++c) coff32[c] = (u32)coff[c];
    if (nc) q.memcpy(d_coff, coff32.data(), (size_t)nc * sizeof(u32)).wait();

    const u32 bound_k = bound, chunk_k = chunk_size;
    u8 *payload_base = d_out + hdr;
    q.parallel_for(sycl::range<1>(nc ? nc : 1), [=](sycl::id<1> idx) {
       u32 c = (u32)idx[0];
       if (c >= nc) return;
       u32 raw = d_csize[c] & ~kStoredFlag;
       const u8 *csrc = d_scratch + (size_t)c * bound_k;
       u8 *cdst = payload_base + d_coff[c];
       for (u32 k = 0; k < raw; ++k) cdst[k] = csrc[k];
     }).wait();

    // Write the header + size table into d_out[0..hdr).
    {
      std::vector<u8> h(hdr);
      Header *H = reinterpret_cast<Header *>(h.data());
      H->magic = kMagic;
      H->chunk_size = chunk_size;
      H->orig_size = src_size;
      H->n_chunks = nc;
      H->rsvd = 0;
      std::memcpy(h.data() + sizeof(Header), csize.data(),
                  (size_t)nc * sizeof(u32));
      q.memcpy(d_out, h.data(), hdr).wait();
    }
    if (!dst_dev) q.memcpy(dst, d_out, total).wait();
    *out = total;
    ok = true;
  } catch (...) {
    ok = false;
  }
  if (d_src && !IsDevicePtr(q, src)) sycl::free(d_src, q);
  if (d_scratch) sycl::free(d_scratch, q);
  if (d_csize) sycl::free(d_csize, q);
  if (d_coff) sycl::free(d_coff, q);
  if (d_out && !IsDevicePtr(q, dst)) sycl::free(d_out, q);
  return ok ? 1 : 0;
}

// Decompress src[0..src_size) -> dst (cap dst_cap); *out set to bytes written.
int clio_sycl_lz4_decompress(void *qbv, const void *src, size_t src_size,
                             void *dst, size_t dst_cap, size_t *out) {
  if (!qbv || !src || !dst || !out || src_size < sizeof(Header)) return 0;
  QueueBox *qb = static_cast<QueueBox *>(qbv);
  sycl::queue &q = qb->q;
  bool ok = false;
  u8 *d_src = nullptr, *d_out = nullptr;
  u32 *d_csize = nullptr, *d_coff = nullptr;
  try {
    const bool src_dev = IsDevicePtr(q, src);
    d_src = src_dev ? const_cast<u8 *>(static_cast<const u8 *>(src))
                    : sycl::malloc_device<u8>(src_size, q);
    if (!d_src) throw std::bad_alloc();
    if (!src_dev) q.memcpy(d_src, src, src_size).wait();

    Header H;
    q.memcpy(&H, d_src, sizeof(Header)).wait();
    if (H.magic != kMagic) throw std::runtime_error("bad magic");
    const u32 nc = H.n_chunks;
    const u32 chunk_size = H.chunk_size;
    const u64 orig = H.orig_size;
    if (orig > dst_cap) throw std::runtime_error("dst too small");
    const size_t hdr = sizeof(Header) + (size_t)nc * sizeof(u32);
    if (src_size < hdr) throw std::runtime_error("truncated");

    std::vector<u32> csize(nc);
    if (nc) q.memcpy(csize.data(), d_src + sizeof(Header),
                     (size_t)nc * sizeof(u32)).wait();
    std::vector<u32> coff(nc);
    u64 acc = 0;
    for (u32 c = 0; c < nc; ++c) {
      coff[c] = (u32)acc;
      acc += (csize[c] & ~kStoredFlag);
    }
    if (hdr + acc > src_size) throw std::runtime_error("payload short");

    const bool dst_dev = IsDevicePtr(q, dst);
    d_out = dst_dev ? static_cast<u8 *>(dst)
                    : sycl::malloc_device<u8>(orig ? orig : 1, q);
    d_csize = sycl::malloc_device<u32>(nc + 1, q);
    d_coff = sycl::malloc_device<u32>(nc + 1, q);
    if (!d_out || !d_csize || !d_coff) throw std::bad_alloc();
    if (nc) {
      q.memcpy(d_csize, csize.data(), (size_t)nc * sizeof(u32)).wait();
      q.memcpy(d_coff, coff.data(), (size_t)nc * sizeof(u32)).wait();
    }
    const u8 *payload_base = d_src + hdr;
    const u32 chunk_d = chunk_size;
    // One WORK-GROUP per chunk: stage the compressed block in local memory, lane
    // 0 decodes it into a local output buffer (so the dependent match copies hit
    // local memory, not global), then the group cooperatively flushes the result
    // to the global output.
    q.submit([&](sycl::handler &h) {
       sycl::local_accessor<u8, 1> s_in(sycl::range<1>(chunk_d), h);
       sycl::local_accessor<u8, 1> s_out(sycl::range<1>(chunk_d), h);
       h.parallel_for(
           sycl::nd_range<1>(sycl::range<1>((size_t)(nc ? nc : 1) * kGroup),
                             sycl::range<1>(kGroup)),
           [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
             u32 c = (u32)it.get_group(0);
             u32 lane = (u32)it.get_local_id(0);  // 0..31 (kGroup == 32)
             if (c >= nc) return;
             auto sg = it.get_sub_group();
             u32 raw = d_csize[c] & ~kStoredFlag;
             bool stored = (d_csize[c] & kStoredFlag) != 0;
             u32 doff = c * chunk_d;
             u32 dlen = (doff + chunk_d <= orig) ? chunk_d : (u32)(orig - doff);
             const u8 *csrc = payload_base + d_coff[c];
             u8 *si = &s_in[0];
             u8 *so = &s_out[0];
             for (u32 k = lane; k < raw; k += kGroup) si[k] = csrc[k];
             sycl::group_barrier(it.get_group());
             if (stored) {
               for (u32 k = lane; k < raw; k += kGroup) so[k] = si[k];
             } else {
               // Cooperative LZ4 decode: lane 0 parses each sequence header,
               // the whole sub-group copies the literals and match body.
               u32 ip = 0, op = 0;
               while (true) {
                 // ---- lane 0 parses token + literal length ----
                 u32 litlen = 0;
                 int stop = 0;
                 u8 match_tok = 0;  // lane-0 only: token's match nibble source
                 if (lane == 0) {
                   if (ip >= raw) {
                     stop = 1;
                   } else {
                     u8 token = si[ip++];
                     litlen = token >> 4;
                     if (litlen == 15) {
                       u8 b;
                       do {
                         b = si[ip++];
                         litlen += b;
                       } while (b == 255 && ip < raw);
                     }
                     // stash token's match nibble for the match phase below
                     match_tok = token;
                   }
                 }
                 stop = sycl::group_broadcast(sg, stop, 0);
                 if (stop) break;
                 litlen = sycl::group_broadcast(sg, litlen, 0);
                 u32 lit_ip = sycl::group_broadcast(sg, ip, 0);
                 u32 lit_op = sycl::group_broadcast(sg, op, 0);
                 // ---- all lanes copy literals (no hazard: si is read-only) ----
                 for (u32 base = 0; base < litlen; base += kGroup) {
                   u32 k = base + lane;
                   if (k < litlen) so[lit_op + k] = si[lit_ip + k];
                 }
                 int last = 0;
                 if (lane == 0) {
                   ip = lit_ip + litlen;
                   op = lit_op + litlen;
                   if (ip >= raw) last = 1;  // final sequence: literals only
                 }
                 last = sycl::group_broadcast(sg, last, 0);
                 if (last) break;
                 // ---- lane 0 parses offset + match length ----
                 u32 off = 0, matchlen = 0;
                 if (lane == 0) {
                   off = (u32)si[ip] | ((u32)si[ip + 1] << 8);
                   ip += 2;
                   matchlen = match_tok & 0x0F;
                   if (matchlen == 15) {
                     u8 b;
                     do {
                       b = si[ip++];
                       matchlen += b;
                     } while (b == 255 && ip < raw);
                   }
                   matchlen += kMinMatch;
                 }
                 off = sycl::group_broadcast(sg, off, 0);
                 matchlen = sycl::group_broadcast(sg, matchlen, 0);
                 u32 m_op = sycl::group_broadcast(sg, op, 0);
                 // ---- copy match body ----
                 // If off >= matchlen the source region is entirely before m_op
                 // (already written) -> safe to copy cooperatively. Otherwise the
                 // match self-overlaps (RLE-like); lane 0 copies it serially.
                 if (off >= matchlen) {
                   for (u32 base = 0; base < matchlen; base += kGroup) {
                     u32 k = base + lane;
                     if (k < matchlen) so[m_op + k] = so[m_op - off + k];
                   }
                 } else if (lane == 0) {
                   u32 mp = m_op - off;
                   for (u32 k = 0; k < matchlen; ++k) so[m_op + k] = so[mp++];
                 }
                 if (lane == 0) op = m_op + matchlen;
               }
             }
             sycl::group_barrier(it.get_group());
             for (u32 k = lane; k < dlen; k += kGroup) d_out[doff + k] = so[k];
           });
     }).wait();

    if (!dst_dev) q.memcpy(dst, d_out, orig).wait();
    *out = orig;
    ok = true;
  } catch (...) {
    ok = false;
  }
  if (d_src && !IsDevicePtr(q, src)) sycl::free(d_src, q);
  if (d_csize) sycl::free(d_csize, q);
  if (d_coff) sycl::free(d_coff, q);
  if (d_out && !IsDevicePtr(q, dst)) sycl::free(d_out, q);
  return ok ? 1 : 0;
}

}  // extern "C"
