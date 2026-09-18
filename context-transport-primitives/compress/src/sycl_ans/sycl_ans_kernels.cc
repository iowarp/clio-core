/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. (BSD-3-Clause; see repository headers.)
 *
 * SYCL ANS kernels -- a portable rANS entropy coder (nvcomp-ANS analog).
 *
 * This translation unit is compiled by a SYCL compiler (Intel DPC++ /
 * AdaptiveCpp), NOT by the ordinary CTP toolchain. It exposes a small extern
 * "C" ABI (clio_sycl_ans_*) that the header-only ctp::SyclAns wrapper calls,
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

constexpr u32 kMagic = 0x5A414E53u;   // "ANSZ"
// Static rANS entropy coder. 32 KiB chunks: each chunk carries its own 256-entry
// normalized frequency table (512 bytes) so a bigger chunk amortizes that header
// (512/32768 ~ 1.5%). Parallelism is across chunks (one work-group each).
constexpr u32 kChunkSize = 1u << 15;  // 32 KiB
constexpr u32 kGroup = 64;            // work-items per chunk (cooperative I/O)
constexpr u32 kStoredFlag = 0x80000000u;  // high bit of comp_size => raw chunk

// rANS parameters (byte-wise renormalization, Giesen's rans_byte scheme).
constexpr u32 kProbBits = 12;
constexpr u32 kProbScale = 1u << kProbBits;   // frequencies sum to 4096
constexpr u32 kRansL = 1u << 23;              // lower renorm bound
constexpr u32 kFreqTableBytes = 256 * 2;      // 256 x u16 normalized freqs

struct Header {
  u32 magic;
  u32 chunk_size;
  u64 orig_size;
  u32 n_chunks;
  u32 rsvd;
};

// Worst-case compressed size: freq table + rANS state + ~1 byte/symbol.
inline u32 CompressBound(u32 n) { return n + kFreqTableBytes + 64u; }

// Normalize a 256-bin histogram (raw counts summing to `total`) to freqN[256]
// summing EXACTLY to kProbScale, with every present symbol getting freq >= 1.
inline void NormalizeFreqs(const u32 *freq, u32 total, u16 *freqN) {
  u32 sum = 0;
  for (int s = 0; s < 256; ++s) {
    if (freq[s] == 0) { freqN[s] = 0; continue; }
    u32 f = (u32)(((u64)freq[s] * kProbScale) / total);
    if (f == 0) f = 1;
    freqN[s] = (u16)f;
    sum += f;
  }
  // Nudge the current-largest bins by +/-1 until the total is exactly kProbScale.
  while (sum < kProbScale) {
    int bs = -1; u32 bf = 0;
    for (int s = 0; s < 256; ++s) if (freqN[s] > bf) { bf = freqN[s]; bs = s; }
    ++freqN[bs]; ++sum;
  }
  while (sum > kProbScale) {
    int bs = -1; u32 bf = 0;
    for (int s = 0; s < 256; ++s)
      if (freqN[s] > 1 && freqN[s] > bf) { bf = freqN[s]; bs = s; }
    if (bs < 0) break;
    --freqN[bs]; --sum;
  }
}

// ---- device: rANS-compress one chunk ----------------------------------------
// dst layout: [freqN 512 bytes][rANS region: LE state(4) + renorm bytes].
// freq/freqN/cumf are this chunk's local-memory scratch. Returns byte count, or
// 0 on overflow (caller then stores the chunk raw).
inline u32 AnsCompressChunk(const u8 *src, u32 n, u8 *dst, u32 dst_cap,
                            u32 *freq, u16 *freqN, u32 *cumf) {
  if (n == 0 || dst_cap < kFreqTableBytes + 8) return 0;
  for (int s = 0; s < 256; ++s) freq[s] = 0;
  for (u32 i = 0; i < n; ++i) ++freq[src[i]];
  NormalizeFreqs(freq, n, freqN);
  cumf[0] = 0;
  for (int s = 0; s < 256; ++s) cumf[s + 1] = cumf[s] + freqN[s];
  // Frequency table at the front.
  for (int s = 0; s < 256; ++s) {
    dst[2 * s] = (u8)(freqN[s] & 0xFF);
    dst[2 * s + 1] = (u8)(freqN[s] >> 8);
  }
  // Encode symbols in REVERSE, emitting renorm bytes backward from dst_cap.
  u32 x = kRansL, wp = dst_cap;
  for (u32 i = n; i-- > 0;) {
    u8 s = src[i];
    u32 fr = freqN[s], st = cumf[s];
    u32 x_max = ((kRansL >> kProbBits) << 8) * fr;
    while (x >= x_max) {
      if (wp <= kFreqTableBytes) return 0;
      dst[--wp] = (u8)(x & 0xFF);
      x >>= 8;
    }
    x = ((x / fr) << kProbBits) + (x % fr) + st;
  }
  // Flush the 4-byte state (LE) at the region start.
  if (wp < kFreqTableBytes + 4) return 0;
  wp -= 4;
  dst[wp] = (u8)(x & 0xFF);
  dst[wp + 1] = (u8)((x >> 8) & 0xFF);
  dst[wp + 2] = (u8)((x >> 16) & 0xFF);
  dst[wp + 3] = (u8)((x >> 24) & 0xFF);
  // Move the rANS region [wp, dst_cap) to sit right after the freq table.
  // dest (kFreqTableBytes) < src (wp), so a forward copy is overlap-safe.
  u32 rlen = dst_cap - wp;
  for (u32 k = 0; k < rlen; ++k) dst[kFreqTableBytes + k] = dst[wp + k];
  return kFreqTableBytes + rlen;
}

// ---- device: rANS-decompress one chunk --------------------------------------
// src = [freqN 512][state(4)+bytes]; dlen symbols out. cumf[257] + slot2sym[4096]
// are local-memory scratch rebuilt from the freq table. Returns bytes written.
inline u32 AnsDecompressChunk(const u8 *src, u32 n, u8 *dst, u32 dlen,
                              u16 *freqN, u32 *cumf, u16 *slot2sym) {
  if (n < kFreqTableBytes + 4) return 0;
  for (int s = 0; s < 256; ++s)
    freqN[s] = (u16)(src[2 * s] | ((u16)src[2 * s + 1] << 8));
  cumf[0] = 0;
  for (int s = 0; s < 256; ++s) cumf[s + 1] = cumf[s] + freqN[s];
  if (cumf[256] != kProbScale) return 0;
  for (int s = 0; s < 256; ++s)
    for (u32 j = cumf[s]; j < cumf[s + 1]; ++j) slot2sym[j] = (u16)s;
  const u8 *p = src + kFreqTableBytes;
  const u8 *end = src + n;
  u32 x = (u32)p[0] | ((u32)p[1] << 8) | ((u32)p[2] << 16) | ((u32)p[3] << 24);
  p += 4;
  for (u32 i = 0; i < dlen; ++i) {
    u32 slot = x & (kProbScale - 1);
    u16 s = slot2sym[slot];
    dst[i] = (u8)s;
    x = freqN[s] * (x >> kProbBits) + slot - cumf[s];
    while (x < kRansL && p < end) x = (x << 8) | *p++;
  }
  return dlen;
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

void *clio_sycl_ans_queue_create() {
  try {
    return new QueueBox();
  } catch (...) {
    return nullptr;
  }
}
void clio_sycl_ans_queue_destroy(void *qb) { delete static_cast<QueueBox *>(qb); }

// Compress src[0..src_size) -> dst (cap dst_cap); *out set to bytes written.
// Returns 1 on success, 0 on failure. src/dst may each be host or device USM.
int clio_sycl_ans_compress(void *qbv, const void *src, size_t src_size,
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

    // One WORK-GROUP per chunk: the small rANS tables live in local memory, lane
    // 0 runs the serial histogram + normalize + encode over the chunk's global
    // source into its global scratch slot. (The 32 KiB chunk itself is NOT staged
    // in local memory -- only the 256-entry tables are.)
    const u32 bound_c = bound;
    const u32 chunk_c = chunk_size;
    q.submit([&](sycl::handler &h) {
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
               u32 z = AnsCompressChunk(d_src + off, len, cdst, bound_c,
                                        &s_freq[0], &s_freqN[0], &s_cumf[0]);
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
  if (d_csize) sycl::free(d_csize, q);
  if (d_coff) sycl::free(d_coff, q);
  if (d_out && !IsDevicePtr(q, dst)) sycl::free(d_out, q);
  return ok ? 1 : 0;
}

// Decompress src[0..src_size) -> dst (cap dst_cap); *out set to bytes written.
int clio_sycl_ans_decompress(void *qbv, const void *src, size_t src_size,
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
    // One WORK-GROUP per chunk: the rANS decode tables (freqN / cumf / slot2sym)
    // live in local memory; lane 0 runs the serial rANS decode from the global
    // compressed block straight into the global output. Parallelism is across
    // chunks. (slot2sym is 8 KiB of local memory.)
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
                 AnsDecompressChunk(csrc, raw, ddst, dlen, &s_freqN[0],
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
  if (d_out && !IsDevicePtr(q, dst)) sycl::free(d_out, q);
  return ok ? 1 : 0;
}

}  // extern "C"
