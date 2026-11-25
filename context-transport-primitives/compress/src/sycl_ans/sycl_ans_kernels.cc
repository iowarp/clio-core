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
 * FORMAT (self-describing, chunked):
 *   Header  { u32 magic; u32 chunk_size; u64 orig_size; u32 n_chunks; u32 rsvd }
 *   u32     comp_size[n_chunks]         // high bit set => chunk stored RAW
 *   bytes   payload                     // chunks concatenated, in order
 *
 * Each compressed chunk is split into W INTERLEAVED rANS sub-streams, one per
 * lane of its work-group, all sharing a single normalized frequency table:
 *   chunk = [freqN: 256 x u16][u32 sub_size[W]][sub_0 | sub_1 | ... | sub_{W-1}]
 * Lane j independently rANS-encodes the chunk's j-th contiguous sub-range into
 * sub_j (a 4-byte rANS state + backward renorm bytes). This is the SIMD/
 * interleaved-rANS layout nvcomp uses: a whole warp works one chunk, the O(n)
 * encode/decode is split W ways, and the histogram is built cooperatively.
 * Earlier revisions ran a single lane per chunk (correct but ~Wx too slow).
 */

#include <sycl/sycl.hpp>

#include <algorithm>

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
// normalized frequency table (512 bytes) so a bigger chunk amortizes that header.
constexpr u32 kChunkSize = 1u << 15;  // 32 KiB
constexpr u32 kStates = 32;           // interleaved rANS sub-streams per chunk
constexpr u32 kGroup = kStates;       // one lane per sub-stream
// Replicated local histograms: lanes split across kHistCopies copies so at most
// kGroup/kHistCopies lanes contend on any one bin (vs all kGroup with a single
// shared histogram). 8 copies = 8 KiB local, ~4-way contention worst case.
constexpr u32 kHistCopies = 8;
constexpr u32 kStoredFlag = 0x80000000u;  // high bit of comp_size => raw chunk

// rANS parameters (byte-wise renormalization, Giesen's rans_byte scheme).
constexpr u32 kProbBits = 12;
constexpr u32 kProbScale = 1u << kProbBits;   // frequencies sum to 4096
constexpr u32 kRansL = 1u << 23;              // lower renorm bound
constexpr u32 kFreqTableBytes = 256 * 2;      // 256 x u16 normalized freqs
constexpr u32 kSizeTableBytes = kStates * 4;  // u32 sub_size[W]

// Per sub-stream and per chunk worst-case byte bounds.
inline u32 SubChunk() { return (kChunkSize + kStates - 1) / kStates; }  // 1024
inline u32 SubBound() { return SubChunk() + 128; }  // state + renorm + margin
inline u32 CompressBound(u32 /*chunk*/) {
  return kFreqTableBytes + kSizeTableBytes + kStates * SubBound() + 64;
}

struct Header {
  u32 magic;
  u32 chunk_size;
  u64 orig_size;
  u32 n_chunks;
  u32 rsvd;
};

using LocalAtomU32 =
    sycl::atomic_ref<u32, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                     sycl::access::address_space::local_space>;

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

// ---- device: rANS-encode ONE sub-range with a shared freq table --------------
// Encodes src[0..n) using shared freqN/cumf into dst (cap bytes): a 4-byte LE
// rANS state followed by backward-emitted renorm bytes, packed to dst[0].
// Returns bytes written (>=4), or 0 on overflow.
// Per-symbol encode constants (ryg_rans): replace the per-symbol integer divide
// x/freq with a 64-bit multiply-high. Built once per chunk from freqN/cumf.
//   x_max = renorm threshold; x/freq == mulhi(x, rcp) >> shift;
//   x' = x + bias + (x/freq)*cmpl, with cmpl = (1<<probBits) - freq.
inline void BuildEncSym(u32 fr, u32 start, u32 &x_max, u32 &rcp, u32 &bias,
                        u32 &cmpl, u32 &shift) {
  x_max = ((kRansL >> kProbBits) << 8) * fr;
  cmpl = kProbScale - fr;
  if (fr < 2) {
    rcp = ~0u;
    shift = 0;
    bias = start + kProbScale - 1;
  } else {
    u32 sh = 0;
    while (fr > (1u << sh)) ++sh;
    rcp = (u32)(((1ull << (sh + 31)) + fr - 1) / fr);
    shift = sh - 1;
    bias = start;
  }
}

inline u32 RansEncodeSub(const u8 *src, u32 n, const u32 *ex_max, const u32 *ercp,
                         const u32 *ebias, const u32 *ecmpl, const u32 *eshift,
                         u8 *dst, u32 cap) {
  u32 x = kRansL, wp = cap;
  for (u32 i = n; i-- > 0;) {
    u8 s = src[i];
    u32 xm = ex_max[s];
    while (x >= xm) {
      if (wp == 0) return 0;
      dst[--wp] = (u8)(x & 0xFF);
      x >>= 8;
    }
    u32 q = (u32)(((u64)x * ercp[s]) >> 32) >> eshift[s];  // x / freq, no divide
    x = x + ebias[s] + q * ecmpl[s];
  }
  if (wp < 4) return 0;
  wp -= 4;
  dst[wp] = (u8)(x & 0xFF);
  dst[wp + 1] = (u8)((x >> 8) & 0xFF);
  dst[wp + 2] = (u8)((x >> 16) & 0xFF);
  dst[wp + 3] = (u8)((x >> 24) & 0xFF);
  u32 rlen = cap - wp;
  for (u32 k = 0; k < rlen; ++k) dst[k] = dst[wp + k];  // dest < src: safe
  return rlen;
}

// ---- device: rANS-decode ONE sub-range with a shared freq table --------------
// Decodes src[0..n) (state + renorm) into dst[0..dlen) using shared tables.
inline void RansDecodeSub(const u8 *src, u32 n, u8 *dst, u32 dlen,
                          const u16 *freqN, const u32 *cumf, const u16 *slot2sym) {
  if (n < 4) return;
  u32 x = (u32)src[0] | ((u32)src[1] << 8) | ((u32)src[2] << 16) |
          ((u32)src[3] << 24);
  const u8 *p = src + 4;
  const u8 *end = src + n;
  for (u32 i = 0; i < dlen; ++i) {
    u32 slot = x & (kProbScale - 1);
    u16 s = slot2sym[slot];
    dst[i] = (u8)s;
    x = freqN[s] * (x >> kProbBits) + slot - cumf[s];
    while (x < kRansL && p < end) x = (x << 8) | *p++;
  }
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
  u8 *d_src = nullptr, *d_scratch = nullptr, *d_enc = nullptr, *d_out = nullptr;
  u32 *d_csize = nullptr, *d_coff = nullptr;
  try {
    const bool src_dev = IsDevicePtr(q, src);
    d_src = src_dev ? const_cast<u8 *>(static_cast<const u8 *>(src))
                    : sycl::malloc_device<u8>(src_size ? src_size : 1, q);
    if (!d_src) throw std::bad_alloc();
    if (!src_dev) q.memcpy(d_src, src, src_size).wait();

    const u32 bound = CompressBound(chunk_size);
    const u32 subbound = SubBound();
    d_scratch = sycl::malloc_device<u8>((size_t)nc * bound + 1, q);
    d_enc = sycl::malloc_device<u8>((size_t)nc * kStates * subbound + 1, q);
    d_csize = sycl::malloc_device<u32>(nc + 1, q);
    if (!d_scratch || !d_enc || !d_csize) throw std::bad_alloc();

    // One WORK-GROUP (W lanes) per chunk. Lanes cooperatively build the chunk's
    // histogram, then each lane rANS-encodes its own contiguous sub-range into a
    // per-lane temp (d_enc), and the group packs the W sub-streams into d_scratch
    // behind the shared freq table + sub-size table. Thousands of chunks in flight.
    const u32 bound_c = bound, chunk_c = chunk_size, subbound_c = subbound;
    q.submit([&](sycl::handler &h) {
       sycl::local_accessor<u32, 1> s_hist(sycl::range<1>(kHistCopies * 256), h);
       sycl::local_accessor<u32, 1> s_freq(sycl::range<1>(256), h);
       sycl::local_accessor<u16, 1> s_freqN(sycl::range<1>(256), h);
       sycl::local_accessor<u32, 1> s_cumf(sycl::range<1>(257), h);
       sycl::local_accessor<u32, 1> s_psum(sycl::range<1>(kGroup), h);
       sycl::local_accessor<u32, 1> s_xmax(sycl::range<1>(256), h);
       sycl::local_accessor<u32, 1> s_rcp(sycl::range<1>(256), h);
       sycl::local_accessor<u32, 1> s_bias(sycl::range<1>(256), h);
       sycl::local_accessor<u32, 1> s_cmpl(sycl::range<1>(256), h);
       sycl::local_accessor<u32, 1> s_shift(sycl::range<1>(256), h);
       sycl::local_accessor<u32, 1> s_size(sycl::range<1>(kStates), h);
       sycl::local_accessor<u32, 1> s_off(sycl::range<1>(kStates + 1), h);
       sycl::local_accessor<u32, 1> s_bad(sycl::range<1>(1), h);
       h.parallel_for(
           sycl::nd_range<1>(sycl::range<1>((size_t)(nc ? nc : 1) * kGroup),
                             sycl::range<1>(kGroup)),
           [=](sycl::nd_item<1> it) {
             const u32 c = (u32)it.get_group(0);
             const u32 lid = (u32)it.get_local_id(0);
             if (c >= nc) return;
             const u32 off = c * chunk_c;
             const u32 len = (off + chunk_c <= src_size) ? chunk_c
                                                         : (u32)(src_size - off);
             const u8 *csrc = d_src + off;
             u8 *cdst = d_scratch + (size_t)c * bound_c;
             // sub-range [a,b) for this lane
             const u32 subn = (len + kStates - 1) / kStates;
             const u32 a = std::min(lid * subn, len);
             const u32 b = std::min(a + subn, len);

             // Cooperative histogram into kHistCopies replicated copies, so each
             // bin is contended by at most kGroup/kHistCopies lanes (not all of
             // them) -- the fix for low-entropy chunks where one bin dominates.
             for (u32 i = lid; i < kHistCopies * 256; i += kGroup) s_hist[i] = 0;
             if (lid == 0) s_bad[0] = 0;
             sycl::group_barrier(it.get_group());
             const u32 hbase = (lid % kHistCopies) * 256;
             for (u32 i = a; i < b; ++i)
               LocalAtomU32(s_hist[hbase + csrc[i]]).fetch_add(1u);
             sycl::group_barrier(it.get_group());
             // Reduce the copies, then cooperatively assign proportional freqs
             // (each lane owns a stride of symbols) while summing per-lane; lane 0
             // runs only the final exact greedy nudge + prefix sum.
             for (u32 s = lid; s < 256; s += kGroup) {
               u32 t = 0;
               for (u32 p = 0; p < kHistCopies; ++p) t += s_hist[p * 256 + s];
               s_freq[s] = t;
             }
             sycl::group_barrier(it.get_group());
             {
               u32 ps = 0;
               for (u32 s = lid; s < 256; s += kGroup) {
                 u32 fq = s_freq[s], f = 0;
                 if (fq) { f = (u32)(((u64)fq * kProbScale) / len); if (!f) f = 1; }
                 s_freqN[s] = (u16)f;
                 ps += f;
               }
               s_psum[lid] = ps;
             }
             sycl::group_barrier(it.get_group());
             if (lid == 0) {
               u32 sum = 0;
               for (u32 j = 0; j < kGroup; ++j) sum += s_psum[j];
               // Distribute the deficit/surplus by +/-1 across present bins in
               // O(256) passes. The old greedy rescanned all 256 bins per unit
               // (O(deficit*256)), which blew up on near-uniform data. Spreading
               // over many bins keeps the ratio essentially identical.
               while (sum < kProbScale) {
                 for (int s = 0; s < 256 && sum < kProbScale; ++s)
                   if (s_freqN[s] > 0) { ++s_freqN[s]; ++sum; }
               }
               while (sum > kProbScale) {
                 bool moved = false;
                 for (int s = 0; s < 256 && sum > kProbScale; ++s)
                   if (s_freqN[s] > 1) { --s_freqN[s]; --sum; moved = true; }
                 if (!moved) break;
               }
               s_cumf[0] = 0;
               for (int s = 0; s < 256; ++s) s_cumf[s + 1] = s_cumf[s] + s_freqN[s];
             }
             sycl::group_barrier(it.get_group());

             // Build the per-symbol reciprocal encode table (cooperative), so the
             // encode loop multiplies instead of dividing -- the real compress win.
             for (u32 s = lid; s < 256; s += kGroup)
               BuildEncSym(s_freqN[s], s_cumf[s], s_xmax[s], s_rcp[s], s_bias[s],
                           s_cmpl[s], s_shift[s]);
             sycl::group_barrier(it.get_group());

             // each lane encodes its sub-range into its own temp slot
             u8 *myenc = d_enc + ((size_t)c * kStates + lid) * subbound_c;
             u32 sz = RansEncodeSub(csrc + a, b - a, &s_xmax[0], &s_rcp[0],
                                    &s_bias[0], &s_cmpl[0], &s_shift[0], myenc,
                                    subbound_c);
             s_size[lid] = sz;
             if (sz == 0) LocalAtomU32(s_bad[0]).fetch_add(1u);
             sycl::group_barrier(it.get_group());

             // lane 0: prefix-sum sizes, decide compressed vs raw
             if (lid == 0) {
               u32 acc = 0;
               for (u32 j = 0; j < kStates; ++j) { s_off[j] = acc; acc += s_size[j]; }
               s_off[kStates] = acc;
             }
             sycl::group_barrier(it.get_group());
             const u32 payload = s_off[kStates];
             const u32 zlen = kFreqTableBytes + kSizeTableBytes + payload;

             if (s_bad[0] != 0 || zlen >= len) {  // store raw
               for (u32 i = lid; i < len; i += kGroup) cdst[i] = csrc[i];
               if (lid == 0) d_csize[c] = len | kStoredFlag;
               return;
             }
             // write freq table + size table (lane 0), pack sub-streams (all lanes)
             if (lid == 0) {
               for (int s = 0; s < 256; ++s) {
                 cdst[2 * s] = (u8)(s_freqN[s] & 0xFF);
                 cdst[2 * s + 1] = (u8)(s_freqN[s] >> 8);
               }
               u8 *st = cdst + kFreqTableBytes;
               for (u32 j = 0; j < kStates; ++j) {
                 u32 v = s_size[j];
                 st[4 * j] = (u8)(v & 0xFF);
                 st[4 * j + 1] = (u8)((v >> 8) & 0xFF);
                 st[4 * j + 2] = (u8)((v >> 16) & 0xFF);
                 st[4 * j + 3] = (u8)((v >> 24) & 0xFF);
               }
               d_csize[c] = zlen;
             }
             u8 *pay = cdst + kFreqTableBytes + kSizeTableBytes + s_off[lid];
             for (u32 k = 0; k < s_size[lid]; ++k) pay[k] = myenc[k];
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

    const bool dst_dev = IsDevicePtr(q, dst);
    d_out = dst_dev ? static_cast<u8 *>(dst)
                    : sycl::malloc_device<u8>(total ? total : 1, q);
    d_coff = sycl::malloc_device<u32>(nc + 1, q);
    if (!d_out || !d_coff) throw std::bad_alloc();
    std::vector<u32> coff32(nc);
    for (u32 c = 0; c < nc; ++c) coff32[c] = (u32)coff[c];
    if (nc) q.memcpy(d_coff, coff32.data(), (size_t)nc * sizeof(u32)).wait();

    const u32 bound_k = bound;
    u8 *payload_base = d_out + hdr;
    // Gather each chunk's packed bytes with one WORK-GROUP per chunk, lanes
    // copying consecutive bytes (coalesced). The earlier one-work-item-per-chunk
    // gather read d_scratch at bound-sized strides, so every byte was a separate
    // uncoalesced transaction -- it, not the encode, was the compress bottleneck.
    constexpr u32 kGatherW = 128;
    q.submit([&](sycl::handler &h) {
       h.parallel_for(
           sycl::nd_range<1>(sycl::range<1>((size_t)(nc ? nc : 1) * kGatherW),
                             sycl::range<1>(kGatherW)),
           [=](sycl::nd_item<1> it) {
             u32 c = (u32)it.get_group(0);
             if (c >= nc) return;
             u32 lid = (u32)it.get_local_id(0);
             u32 raw = d_csize[c] & ~kStoredFlag;
             const u8 *csrc = d_scratch + (size_t)c * bound_k;
             u8 *cdst = payload_base + d_coff[c];
             for (u32 k = lid; k < raw; k += kGatherW) cdst[k] = csrc[k];
           });
     }).wait();

    {
      std::vector<u8> hbuf(hdr);
      Header *H = reinterpret_cast<Header *>(hbuf.data());
      H->magic = kMagic;
      H->chunk_size = chunk_size;
      H->orig_size = src_size;
      H->n_chunks = nc;
      H->rsvd = 0;
      std::memcpy(hbuf.data() + sizeof(Header), csize.data(),
                  (size_t)nc * sizeof(u32));
      q.memcpy(d_out, hbuf.data(), hdr).wait();
    }
    if (!dst_dev) q.memcpy(dst, d_out, total).wait();
    *out = total;
    ok = true;
  } catch (...) {
    ok = false;
  }
  if (d_src && !IsDevicePtr(q, src)) sycl::free(d_src, q);
  if (d_scratch) sycl::free(d_scratch, q);
  if (d_enc) sycl::free(d_enc, q);
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
    // Validate the size table BEFORE launch (reject foreign / corrupt streams).
    if (chunk_size == 0 || chunk_size > kChunkSize ||
        (u64)nc * chunk_size < orig ||
        (nc && (u64)(nc - 1) * chunk_size >= orig))
      throw std::runtime_error("bad geometry");
    std::vector<u32> coff(nc);
    u64 acc = 0;
    for (u32 c = 0; c < nc; ++c) {
      const u32 raw = csize[c] & ~kStoredFlag;
      const u64 dlen = std::min<u64>(chunk_size, orig - (u64)c * chunk_size);
      const u32 cbound = CompressBound(chunk_size);
      if (raw > cbound || ((csize[c] & kStoredFlag) && raw != dlen) ||
          (!(csize[c] & kStoredFlag) && raw < kFreqTableBytes + kSizeTableBytes))
        throw std::runtime_error("bad chunk size");
      coff[c] = (u32)acc;
      acc += raw;
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
    // One WORK-GROUP (W lanes) per chunk. The decode tables (freqN / cumf /
    // slot2sym) are rebuilt cooperatively in local memory, then each lane
    // decodes its own sub-stream into its slice of the output. (slot2sym is
    // 8 KiB of local memory.)
    q.submit([&](sycl::handler &h) {
       sycl::local_accessor<u16, 1> s_freqN(sycl::range<1>(256), h);
       sycl::local_accessor<u32, 1> s_cumf(sycl::range<1>(257), h);
       sycl::local_accessor<u16, 1> s_slot(sycl::range<1>(kProbScale), h);
       sycl::local_accessor<u32, 1> s_off(sycl::range<1>(kStates + 1), h);
       sycl::local_accessor<u32, 1> s_valid(sycl::range<1>(1), h);
       h.parallel_for(
           sycl::nd_range<1>(sycl::range<1>((size_t)(nc ? nc : 1) * kGroup),
                             sycl::range<1>(kGroup)),
           [=](sycl::nd_item<1> it) {
             const u32 c = (u32)it.get_group(0);
             const u32 lid = (u32)it.get_local_id(0);
             if (c >= nc) return;
             const u32 raw = d_csize[c] & ~kStoredFlag;
             const bool stored = (d_csize[c] & kStoredFlag) != 0;
             const u32 doff = c * chunk_d;
             const u32 dlen = (doff + chunk_d <= orig) ? chunk_d
                                                       : (u32)(orig - doff);
             const u8 *csrc = payload_base + d_coff[c];
             u8 *ddst = d_out + doff;

             if (stored) {
               for (u32 k = lid; k < raw; k += kGroup) ddst[k] = csrc[k];
               return;
             }
             // rebuild freqN + cumf (lane 0), validate, then slot2sym (cooperative)
             if (lid == 0) {
               for (int s = 0; s < 256; ++s)
                 s_freqN[s] = (u16)(csrc[2 * s] | ((u16)csrc[2 * s + 1] << 8));
               s_cumf[0] = 0;
               for (int s = 0; s < 256; ++s) s_cumf[s + 1] = s_cumf[s] + s_freqN[s];
               // sub-stream offsets from the size table
               const u8 *st = csrc + kFreqTableBytes;
               u32 a = 0;
               for (u32 j = 0; j < kStates; ++j) {
                 u32 v = (u32)st[4 * j] | ((u32)st[4 * j + 1] << 8) |
                         ((u32)st[4 * j + 2] << 16) | ((u32)st[4 * j + 3] << 24);
                 s_off[j] = a;
                 a += v;
               }
               s_off[kStates] = a;
               s_valid[0] = (s_cumf[256] == kProbScale &&
                             kFreqTableBytes + kSizeTableBytes + a <= raw)
                                ? 1u
                                : 0u;
             }
             sycl::group_barrier(it.get_group());
             if (!s_valid[0]) return;
             for (u32 s = lid; s < 256; s += kGroup)
               for (u32 j = s_cumf[s]; j < s_cumf[s + 1]; ++j) s_slot[j] = (u16)s;
             sycl::group_barrier(it.get_group());

             const u32 subn = (dlen + kStates - 1) / kStates;
             const u32 a = std::min(lid * subn, dlen);
             const u32 b = std::min(a + subn, dlen);
             const u8 *mysrc =
                 csrc + kFreqTableBytes + kSizeTableBytes + s_off[lid];
             const u32 myn = s_off[lid + 1] - s_off[lid];
             RansDecodeSub(mysrc, myn, ddst + a, b - a, &s_freqN[0], &s_cumf[0],
                           &s_slot[0]);
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
