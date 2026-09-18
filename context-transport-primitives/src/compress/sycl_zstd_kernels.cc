/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. (BSD-3-Clause; see repository headers.)
 *
 * SYCL Zstd-STYLE kernels -- an LZ77 + rANS combined codec (NOT libzstd-compatible).
 *
 * This translation unit is compiled by a SYCL compiler (Intel DPC++ /
 * AdaptiveCpp), NOT by the ordinary CTP toolchain. It exposes a small extern
 * "C" ABI (clio_sycl_zstd_*) that the header-only ctp::SyclZstd wrapper calls,
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

constexpr u32 kMagic = 0x5A535444u;   // "ZSTD"
// Zstd-STYLE combined codec (NOT libzstd-compatible): an LZ77 stage produces
// literals + (litLen, matchLen, offset) sequences, then a static rANS entropy
// stage codes the literals -- the two-stage idea Zstd is built on. 32 KiB chunks
// give the entropy stage enough data and amortize its 512-byte frequency table.
constexpr u32 kChunkSize = 1u << 15;  // 32 KiB
constexpr u32 kHashLog = 13;          // 8192-entry LZ hash table per chunk
constexpr u32 kHashSize = 1u << kHashLog;
constexpr u16 kHashNull = 0xFFFFu;    // u16 sentinel (chunk < 65536 bytes)
constexpr u32 kGroup = 64;            // work-items per chunk (cooperative I/O)
constexpr u32 kStoredFlag = 0x80000000u;  // high bit of comp_size => raw chunk

// LZ stage constants.
constexpr int kMinMatch = 4;
constexpr u32 kMaxOffset = 65535;  // 16-bit offsets within a chunk

// rANS entropy-stage parameters (Giesen rans_byte scheme).
constexpr u32 kProbBits = 12;
constexpr u32 kProbScale = 1u << kProbBits;
constexpr u32 kRansL = 1u << 23;
constexpr u32 kFreqTableBytes = 256 * 2;

struct Header {
  u32 magic;
  u32 chunk_size;
  u64 orig_size;
  u32 n_chunks;
  u32 rsvd;
};

// Worst case: literals rANS (~lit + table) + sequence varints (~n) + header.
inline u32 CompressBound(u32 n) { return 2u * n + kFreqTableBytes + 64u; }

inline u32 Hash4(u32 seq) { return (seq * 2654435761u) >> (32 - kHashLog); }

// ---- varint (LEB128) ---------------------------------------------------------
inline bool PutVarint(u8 *dst, u32 &op, u32 cap, u32 v) {
  while (v >= 0x80) { if (op >= cap) return false; dst[op++] = (u8)(v | 0x80); v >>= 7; }
  if (op >= cap) return false; dst[op++] = (u8)v; return true;
}
inline u32 GetVarint(const u8 *src, u32 &ip, u32 n) {
  u32 v = 0, sh = 0;
  while (ip < n) { u8 b = src[ip++]; v |= (u32)(b & 0x7F) << sh; if (!(b & 0x80)) break; sh += 7; }
  return v;
}

// ---- rANS entropy stage over an arbitrary byte buffer ------------------------
inline void NormalizeFreqs(const u32 *freq, u32 total, u16 *freqN) {
  u32 sum = 0;
  for (int s = 0; s < 256; ++s) {
    if (!freq[s]) { freqN[s] = 0; continue; }
    u32 f = (u32)(((u64)freq[s] * kProbScale) / total); if (!f) f = 1;
    freqN[s] = (u16)f; sum += f;
  }
  while (sum < kProbScale) { int bs = -1; u32 bf = 0;
    for (int s = 0; s < 256; ++s) if (freqN[s] > bf) { bf = freqN[s]; bs = s; }
    ++freqN[bs]; ++sum; }
  while (sum > kProbScale) { int bs = -1; u32 bf = 0;
    for (int s = 0; s < 256; ++s) if (freqN[s] > 1 && freqN[s] > bf) { bf = freqN[s]; bs = s; }
    if (bs < 0) break; --freqN[bs]; --sum; }
}
inline u32 RansEncodeBuf(const u8 *lit, u32 n, u8 *dst, u32 dst_cap,
                         u32 *freq, u16 *freqN, u32 *cumf) {
  if (n == 0) return 0;
  if (dst_cap < kFreqTableBytes + 8) return 0;
  for (int s = 0; s < 256; ++s) freq[s] = 0;
  for (u32 i = 0; i < n; ++i) ++freq[lit[i]];
  NormalizeFreqs(freq, n, freqN);
  cumf[0] = 0; for (int s = 0; s < 256; ++s) cumf[s + 1] = cumf[s] + freqN[s];
  for (int s = 0; s < 256; ++s) { dst[2*s] = (u8)(freqN[s] & 0xFF); dst[2*s+1] = (u8)(freqN[s] >> 8); }
  u32 x = kRansL, wp = dst_cap;
  for (u32 i = n; i-- > 0;) {
    u8 s = lit[i]; u32 fr = freqN[s], st = cumf[s];
    u32 x_max = ((kRansL >> kProbBits) << 8) * fr;
    while (x >= x_max) { if (wp <= kFreqTableBytes) return 0; dst[--wp] = (u8)(x & 0xFF); x >>= 8; }
    x = ((x / fr) << kProbBits) + (x % fr) + st;
  }
  if (wp < kFreqTableBytes + 4) return 0;
  wp -= 4;
  dst[wp]=(u8)(x&0xFF); dst[wp+1]=(u8)((x>>8)&0xFF); dst[wp+2]=(u8)((x>>16)&0xFF); dst[wp+3]=(u8)((x>>24)&0xFF);
  u32 rlen = dst_cap - wp;
  for (u32 k = 0; k < rlen; ++k) dst[kFreqTableBytes + k] = dst[wp + k];
  return kFreqTableBytes + rlen;
}
inline u32 RansDecodeBuf(const u8 *src, u32 n, u8 *lit, u32 dlen,
                         u16 *freqN, u32 *cumf, u16 *slot2sym) {
  if (n < kFreqTableBytes + 4) return 0;
  for (int s = 0; s < 256; ++s) freqN[s] = (u16)(src[2*s] | ((u16)src[2*s+1] << 8));
  cumf[0] = 0; for (int s = 0; s < 256; ++s) cumf[s+1] = cumf[s] + freqN[s];
  if (cumf[256] != kProbScale) return 0;
  for (int s = 0; s < 256; ++s) for (u32 j = cumf[s]; j < cumf[s+1]; ++j) slot2sym[j] = (u16)s;
  const u8 *p = src + kFreqTableBytes, *end = src + n;
  u32 x = (u32)p[0] | ((u32)p[1]<<8) | ((u32)p[2]<<16) | ((u32)p[3]<<24); p += 4;
  for (u32 i = 0; i < dlen; ++i) {
    u32 slot = x & (kProbScale - 1); u16 s = slot2sym[slot]; lit[i] = (u8)s;
    x = freqN[s] * (x >> kProbBits) + slot - cumf[s];
    while (x < kRansL && p < end) x = (x << 8) | *p++;
  }
  return dlen;
}

// ---- Zstd-style combined compress (v2: entropy-code BOTH streams) -----------
// dst: [u32 n_seq][u32 lit_count][u32 seq_raw][u32 seq_comp][u32 lit_comp]
//   [rANS(sequence varints)][rANS(literals)].  seqbuf/litbuf/hash/freq/freqN/cumf
// are per-chunk scratch. Returns size, 0 on overflow.
inline u32 ZstdCompressChunk(const u8 *src, u32 n, u8 *dst, u32 dst_cap,
                             u8 *litbuf, u8 *seqbuf, u32 seq_cap, u16 *hash,
                             u32 *freq, u16 *freqN, u32 *cumf) {
  for (u32 i = 0; i < kHashSize; ++i) hash[i] = kHashNull;
  auto load32 = [&](u32 p) -> u32 {
    return (u32)src[p] | ((u32)src[p+1]<<8) | ((u32)src[p+2]<<16) | ((u32)src[p+3]<<24);
  };
  const u32 kHdr = 20;
  u32 sp = 0, lit_count = 0, n_seq = 0, ip = 0, anchor = 0;  // sp = seqbuf cursor
  while (ip + 4 <= n) {
    u32 h = Hash4(load32(ip));
    u32 ref = hash[h]; hash[h] = (u16)ip;
    bool have = (ref != kHashNull) && (ip - ref) <= kMaxOffset && load32(ref) == load32(ip);
    if (!have) { ++ip; continue; }
    u32 litLen = ip - anchor, off = ip - ref, mstart = ip, r = ref + kMinMatch;
    ip += kMinMatch;
    while (ip < n && src[ip] == src[r]) { ++ip; ++r; }
    u32 matchLen = ip - mstart;
    for (u32 k = 0; k < litLen; ++k) litbuf[lit_count + k] = src[anchor + k];
    lit_count += litLen;
    if (!PutVarint(seqbuf, sp, seq_cap, litLen)) return 0;
    if (!PutVarint(seqbuf, sp, seq_cap, matchLen - kMinMatch)) return 0;
    if (!PutVarint(seqbuf, sp, seq_cap, off)) return 0;
    ++n_seq; anchor = ip;
  }
  u32 tail = n - anchor;
  for (u32 k = 0; k < tail; ++k) litbuf[lit_count + k] = src[anchor + k];
  lit_count += tail;
  u32 seq_raw = sp;
  if (dst_cap < kHdr) return 0;
  // rANS both streams, sequences first.
  u32 seq_comp = RansEncodeBuf(seqbuf, seq_raw, dst + kHdr, dst_cap - kHdr, freq, freqN, cumf);
  if (seq_raw > 0 && seq_comp == 0) return 0;
  u32 lit_comp = RansEncodeBuf(litbuf, lit_count, dst + kHdr + seq_comp,
                               dst_cap - kHdr - seq_comp, freq, freqN, cumf);
  if (lit_count > 0 && lit_comp == 0) return 0;
  u32 hv[5] = {n_seq, lit_count, seq_raw, seq_comp, lit_comp};
  for (int i = 0; i < 5; ++i) {
    dst[4*i]=(u8)hv[i]; dst[4*i+1]=(u8)(hv[i]>>8); dst[4*i+2]=(u8)(hv[i]>>16); dst[4*i+3]=(u8)(hv[i]>>24);
  }
  return kHdr + seq_comp + lit_comp;
}

// ---- Zstd-style combined decompress (v2) ------------------------------------
inline u32 ZstdDecompressChunk(const u8 *src, u32 n, u8 *dst, u32 dlen, u8 *litbuf,
                               u8 *seqbuf, u16 *freqN, u32 *cumf, u16 *slot2sym) {
  const u32 kHdr = 20;
  if (n < kHdr) return 0;
  u32 hv[5];
  for (int i = 0; i < 5; ++i)
    hv[i] = (u32)src[4*i] | ((u32)src[4*i+1]<<8) | ((u32)src[4*i+2]<<16) | ((u32)src[4*i+3]<<24);
  u32 n_seq = hv[0], lit_count = hv[1], seq_raw = hv[2], seq_comp = hv[3], lit_comp = hv[4];
  if ((u64)kHdr + seq_comp + lit_comp > n) return 0;
  if (seq_raw > 0 &&
      RansDecodeBuf(src + kHdr, seq_comp, seqbuf, seq_raw, freqN, cumf, slot2sym) != seq_raw)
    return 0;
  if (lit_count > 0 &&
      RansDecodeBuf(src + kHdr + seq_comp, lit_comp, litbuf, lit_count, freqN, cumf, slot2sym) != lit_count)
    return 0;
  u32 sip = 0, lp = 0, op = 0;
  for (u32 s = 0; s < n_seq; ++s) {
    u32 litLen = GetVarint(seqbuf, sip, seq_raw);
    u32 matchLen = GetVarint(seqbuf, sip, seq_raw) + kMinMatch;
    u32 off = GetVarint(seqbuf, sip, seq_raw);
    if (lp + litLen > lit_count || op + litLen > dlen) return 0;
    for (u32 k = 0; k < litLen; ++k) dst[op++] = litbuf[lp++];
    if (off == 0 || off > op || op + matchLen > dlen) return 0;
    u32 mp = op - off;
    for (u32 k = 0; k < matchLen; ++k) dst[op++] = dst[mp++];
  }
  u32 tail = lit_count - lp;
  if (op + tail > dlen) return 0;
  for (u32 k = 0; k < tail; ++k) dst[op++] = litbuf[lp++];
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

void *clio_sycl_zstd_queue_create() {
  try {
    return new QueueBox();
  } catch (...) {
    return nullptr;
  }
}
void clio_sycl_zstd_queue_destroy(void *qb) { delete static_cast<QueueBox *>(qb); }

// Compress src[0..src_size) -> dst (cap dst_cap); *out set to bytes written.
// Returns 1 on success, 0 on failure. src/dst may each be host or device USM.
int clio_sycl_zstd_compress(void *qbv, const void *src, size_t src_size,
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
  u8 *d_src = nullptr, *d_scratch = nullptr, *d_out = nullptr, *d_lit = nullptr, *d_seq = nullptr;
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
    d_lit = sycl::malloc_device<u8>((size_t)nc * chunk_size + 1, q);  // literals
    d_seq = sycl::malloc_device<u8>((size_t)nc * 2 * chunk_size + 1, q);  // seq varints
    if (!d_scratch || !d_csize || !d_lit || !d_seq) throw std::bad_alloc();

    // One WORK-GROUP per chunk: LZ hash + rANS tables in local memory; lane 0
    // runs the serial LZ parse (collecting literals into the chunk's global
    // literal scratch + writing sequence varints) then rANS-codes the literals.
    const u32 bound_c = bound;
    const u32 chunk_c = chunk_size;
    q.submit([&](sycl::handler &h) {
       sycl::local_accessor<u16, 1> s_hash(sycl::range<1>(kHashSize), h);
       sycl::local_accessor<u32, 1> s_freq(sycl::range<1>(256), h);
       sycl::local_accessor<u16, 1> s_freqN(sycl::range<1>(256), h);
       sycl::local_accessor<u32, 1> s_cumf(sycl::range<1>(257), h);
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
             if (lid == 0) {
               u8 *cdst = d_scratch + (size_t)c * bound_c;
               u8 *lb = d_lit + (size_t)c * chunk_c;
               u8 *sb = d_seq + (size_t)c * 2 * chunk_c;
               u32 z = ZstdCompressChunk(d_src + off, len, cdst, bound_c, lb, sb,
                                         2 * chunk_c, &s_hash[0], &s_freq[0],
                                         &s_freqN[0], &s_cumf[0]);
               if (z == 0 || z >= len) {  // incompressible -> store raw
                 for (u32 k = 0; k < len; ++k) cdst[k] = d_src[off + k];
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
  if (d_lit) sycl::free(d_lit, q);
  if (d_seq) sycl::free(d_seq, q);
  if (d_csize) sycl::free(d_csize, q);
  if (d_coff) sycl::free(d_coff, q);
  if (d_out && !IsDevicePtr(q, dst)) sycl::free(d_out, q);
  return ok ? 1 : 0;
}

// Decompress src[0..src_size) -> dst (cap dst_cap); *out set to bytes written.
int clio_sycl_zstd_decompress(void *qbv, const void *src, size_t src_size,
                             void *dst, size_t dst_cap, size_t *out) {
  if (!qbv || !src || !dst || !out || src_size < sizeof(Header)) return 0;
  QueueBox *qb = static_cast<QueueBox *>(qbv);
  sycl::queue &q = qb->q;
  bool ok = false;
  u8 *d_src = nullptr, *d_out = nullptr, *d_litdec = nullptr, *d_seqdec = nullptr;
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
    d_litdec = sycl::malloc_device<u8>((size_t)nc * chunk_d + 1, q);  // literals
    d_seqdec = sycl::malloc_device<u8>((size_t)nc * 2 * chunk_d + 1, q);  // seq varints
    if (!d_litdec || !d_seqdec) throw std::bad_alloc();
    // One WORK-GROUP per chunk: rANS decode tables in local memory; lane 0
    // rANS-decodes the literals into the chunk's global literal scratch, then
    // walks the sequences to reconstruct the output. Parallelism is across chunks.
    q.submit([&](sycl::handler &h) {
       sycl::local_accessor<u16, 1> s_freqN(sycl::range<1>(256), h);
       sycl::local_accessor<u32, 1> s_cumf(sycl::range<1>(257), h);
       sycl::local_accessor<u16, 1> s_slot(sycl::range<1>(kProbScale), h);
       h.parallel_for(
           sycl::nd_range<1>(sycl::range<1>((size_t)(nc ? nc : 1) * kGroup),
                             sycl::range<1>(kGroup)),
           [=](sycl::nd_item<1> it) {
             u32 c = (u32)it.get_group(0);
             u32 lid = (u32)it.get_local_id(0);
             if (c >= nc) return;
             u32 raw = d_csize[c] & ~kStoredFlag;
             bool stored = (d_csize[c] & kStoredFlag) != 0;
             u32 doff = c * chunk_d;
             u32 dlen = (doff + chunk_d <= orig) ? chunk_d : (u32)(orig - doff);
             const u8 *csrc = payload_base + d_coff[c];
             u8 *ddst = d_out + doff;
             if (lid == 0) {
               if (stored) {
                 for (u32 k = 0; k < raw; ++k) ddst[k] = csrc[k];
               } else {
                 u8 *lb = d_litdec + (size_t)c * chunk_d;
                 u8 *sb = d_seqdec + (size_t)c * 2 * chunk_d;
                 ZstdDecompressChunk(csrc, raw, ddst, dlen, lb, sb, &s_freqN[0],
                                     &s_cumf[0], &s_slot[0]);
               }
             }
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
  if (d_litdec) sycl::free(d_litdec, q);
  if (d_seqdec) sycl::free(d_seqdec, q);
  if (d_out && !IsDevicePtr(q, dst)) sycl::free(d_out, q);
  return ok ? 1 : 0;
}

}  // extern "C"
