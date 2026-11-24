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
 * model. Batched entry points (clio_sycl_lz4_*_batch) issue ONE launch per
 * phase for every chunk of every job -- see BATCHED EXECUTION below. See
 * build_and_test.sh for the validated build recipe.
 */

#include <sycl/sycl.hpp>

#include <algorithm>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
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

// ============================================================================
// BATCHED EXECUTION -- the nvcomp model, and what ctp::Compressor::CompressBatch
// asks of a GPU codec: ONE launch per phase for EVERY chunk of EVERY job, no
// scratch allocation per call, and a single host synchronization.
//
// Compress (all on one in-order queue, one wait at the end):
//   1. encode   one launch over the global chunk list; chunk c belongs to job
//               chunk_job[c]. Each chunk -> scratch slot c (CompressBound).
//   2. scan     one WORK-GROUP per job: exclusive scan of its chunk sizes ->
//               per-chunk payload offsets; writes the header + size table
//               straight into dst; ok = (total <= dst_capacity).
//   3. gather   one work-group per chunk: cooperative copy of the chunk from
//               scratch into dst at its offset (skipped for failed jobs).
//   4. copy the per-job {out_size, ok} back; wait.
// Decompress:
//   1. read every job's Header on the device, copy back, wait (the only extra
//      sync: the chunk count decides the launch shape).
//   2. scan     per job: validate the size table (a block larger than its
//               chunk would overrun the decoder's local buffers; a stored chunk
//               must be exactly its length) and compute payload offsets.
//   3. decode   one work-group per chunk (skipped for failed jobs).
//   4. copy statuses back; wait.
// Host buffers are staged through grow-only device scratch; device buffers are
// used in place -- the CompressJob contract.
// ============================================================================

struct JobDev {
  const u8 *src;
  u8 *dst;
  u64 src_size;
  u64 dst_cap;
  u64 orig;        // decompress: original size from the header
  u64 chunk_base;  // first global chunk index of this job
  u32 n_chunks;
  u32 chunk_size;
  u32 hdr_bytes;   // sizeof(Header) + 4 * n_chunks
  u32 valid;       // host pre-validation passed
};
struct JobOut {
  u64 out_size;
  u32 ok;
  u32 pad;
};

// Unaligned-safe byte access: dst/src are caller buffers of any alignment.
inline void St32(u8 *p, u32 v) {
  p[0] = (u8)v; p[1] = (u8)(v >> 8); p[2] = (u8)(v >> 16); p[3] = (u8)(v >> 24);
}
inline u32 Ld32(const u8 *p) {
  return (u32)p[0] | ((u32)p[1] << 8) | ((u32)p[2] << 16) | ((u32)p[3] << 24);
}

constexpr u32 kScanWG = 256;

// Grow-only device scratch. Reallocated only when a batch needs more.
struct DevBuf {
  void *p = nullptr;
  size_t cap = 0;
  template <typename T>
  T *Get(sycl::queue &q, size_t n) {
    size_t bytes = std::max<size_t>(n * sizeof(T), 1);
    if (bytes > cap) {
      if (p) sycl::free(p, q);
      size_t want = std::max(bytes, cap + cap / 2);
      p = sycl::malloc_device(want, q);
      if (!p) { cap = 0; throw std::bad_alloc(); }
      cap = want;
    }
    return static_cast<T *>(p);
  }
  void Free(sycl::queue &q) { if (p) sycl::free(p, q); p = nullptr; cap = 0; }
};

enum class Enc { kGroup, kItem };

// A GPU queue (device selector, in-order), created once and reused with its
// scratch. The mutex makes one instance safe to share across threads.
struct QueueBox {
  sycl::queue q;
  std::mutex mu;
  Enc enc;
  DevBuf jobs, outs, chunk_job, csize, coff, scratch, stage_in, stage_out, hdrs;
  QueueBox()
      : q(sycl::gpu_selector_v, sycl::property::queue::in_order{}) {
    const char *e = std::getenv("CLIO_SYCL_LZ4_ENC");
    // Default: one chunk per WORK-GROUP (measured 1.5-2.7x faster on PVC
    // than one chunk per work-item). CLIO_SYCL_LZ4_ENC=item selects the other.
    enc = (e && std::strcmp(e, "item") == 0) ? Enc::kItem : Enc::kGroup;
  }
  ~QueueBox() {
    for (DevBuf *b : {&jobs, &outs, &chunk_job, &csize, &coff, &scratch,
                      &stage_in, &stage_out, &hdrs})
      b->Free(q);
  }
};

inline bool IsDevicePtr(const sycl::queue &q, const void *p) {
  if (!p) return false;
  auto kind = sycl::get_pointer_type(p, q.get_context());
  return kind == sycl::usm::alloc::device || kind == sycl::usm::alloc::shared;
}

// Items per work-group for the one-chunk-per-ITEM encoder. Each item needs a
// private 4 KiB hash table in local memory: 16 x 4 KiB = 64 KiB per group,
// half of a PVC Xe-core's SLM, so two groups can be co-resident.
constexpr u32 kEncItems = 16;

int CompressBatchImpl(QueueBox *qb, size_t n, const void *const *srcs,
                      const size_t *src_sizes, void *const *dsts,
                      const size_t *dst_caps, size_t *out_sizes, int *oks) {
  std::lock_guard<std::mutex> lk(qb->mu);
  sycl::queue &q = qb->q;
  const u32 chunk = kChunkSize, bound = CompressBound(kChunkSize);
  std::vector<JobDev> jobs(n);
  std::vector<u32> chunk_job;
  std::vector<u64> in_off(n, ~0ull), out_off(n, ~0ull);
  u64 stage_in = 0, stage_out = 0, C = 0;
  for (size_t j = 0; j < n; ++j) {
    JobDev &J = jobs[j];
    J.src_size = src_sizes[j];
    J.dst_cap = dst_caps[j];
    // A zero-length job may pass null src (an empty vector's data()).
    J.valid = (srcs[j] || J.src_size == 0) && dsts[j] && J.src_size < (1ull << 40);
    J.n_chunks = J.valid ? (u32)((J.src_size + chunk - 1) / chunk) : 0;
    J.chunk_size = chunk;
    J.hdr_bytes = (u32)(sizeof(Header) + 4ull * J.n_chunks);
    J.chunk_base = C;
    J.orig = J.src_size;
    C += J.n_chunks;
    if (!J.valid) continue;
    if (!IsDevicePtr(q, srcs[j])) { in_off[j] = stage_in; stage_in += J.src_size; }
    if (!IsDevicePtr(q, dsts[j])) { out_off[j] = stage_out; stage_out += J.dst_cap; }
  }
  chunk_job.resize(C);
  for (size_t j = 0; j < n; ++j)
    for (u32 c = 0; c < jobs[j].n_chunks; ++c)
      chunk_job[jobs[j].chunk_base + c] = (u32)j;

  u8 *sin = stage_in ? qb->stage_in.Get<u8>(q, stage_in) : nullptr;
  u8 *sout = stage_out ? qb->stage_out.Get<u8>(q, stage_out) : nullptr;
  for (size_t j = 0; j < n; ++j) {
    if (in_off[j] != ~0ull) {
      if (jobs[j].src_size)
        q.memcpy(sin + in_off[j], srcs[j], jobs[j].src_size);
      jobs[j].src = sin + in_off[j];
    } else {
      jobs[j].src = static_cast<const u8 *>(srcs[j]);
    }
    jobs[j].dst = out_off[j] != ~0ull ? sout + out_off[j]
                                      : static_cast<u8 *>(dsts[j]);
  }
  JobDev *d_jobs = qb->jobs.Get<JobDev>(q, n);
  JobOut *d_out = qb->outs.Get<JobOut>(q, n);
  u32 *d_cj = qb->chunk_job.Get<u32>(q, C);
  u32 *d_cs = qb->csize.Get<u32>(q, C);
  u64 *d_co = qb->coff.Get<u64>(q, C);
  u8 *d_scr = qb->scratch.Get<u8>(q, C * (u64)bound);
  q.memcpy(d_jobs, jobs.data(), n * sizeof(JobDev));
  if (C) q.memcpy(d_cj, chunk_job.data(), C * sizeof(u32));

  // ---- 1. encode ----
  if (C && qb->enc == Enc::kGroup) {
    q.submit([&](sycl::handler &h) {
      sycl::local_accessor<u8, 1> s_src(sycl::range<1>(chunk), h);
      sycl::local_accessor<u16, 1> s_hash(sycl::range<1>(kHashSize), h);
      h.parallel_for(sycl::nd_range<1>(C * kGroup, kGroup),
                     [=](sycl::nd_item<1> it) {
        const u64 c = it.get_group(0);
        const u32 lid = (u32)it.get_local_id(0);
        const JobDev J = d_jobs[d_cj[c]];
        const u64 off = (c - J.chunk_base) * chunk;
        const u32 len = (u32)std::min<u64>(chunk, J.src_size - off);
        u8 *ss = &s_src[0];
        for (u32 k = lid; k < len; k += kGroup) ss[k] = J.src[off + k];
        sycl::group_barrier(it.get_group());
        if (lid == 0) {
          u8 *cd = d_scr + c * bound;
          u32 z = Lz4CompressChunk(ss, len, cd, bound, &s_hash[0]);
          if (z == 0 || z >= len) {
            for (u32 k = 0; k < len; ++k) cd[k] = ss[k];
            d_cs[c] = len | kStoredFlag;
          } else {
            d_cs[c] = z;
          }
        }
      });
    });
  } else if (C) {
    const u64 G = (C + kEncItems - 1) / kEncItems;
    q.submit([&](sycl::handler &h) {
      sycl::local_accessor<u16, 1> s_hash(sycl::range<1>(kEncItems * kHashSize),
                                          h);
      h.parallel_for(sycl::nd_range<1>(G * kEncItems, kEncItems),
                     [=](sycl::nd_item<1> it) {
        const u64 c = it.get_global_id(0);
        if (c >= C) return;
        const JobDev J = d_jobs[d_cj[c]];
        const u64 off = (c - J.chunk_base) * chunk;
        const u32 len = (u32)std::min<u64>(chunk, J.src_size - off);
        const u8 *src = J.src + off;
        u8 *cd = d_scr + c * bound;
        u32 z = Lz4CompressChunk(src, len, cd, bound,
                                 &s_hash[it.get_local_id(0) * kHashSize]);
        if (z == 0 || z >= len) {
          for (u32 k = 0; k < len; ++k) cd[k] = src[k];
          d_cs[c] = len | kStoredFlag;
        } else {
          d_cs[c] = z;
        }
      });
    });
  }

  // ---- 2. per-job scan + header/size table into dst ----
  q.parallel_for(sycl::nd_range<1>(n * kScanWG, kScanWG),
                 [=](sycl::nd_item<1> it) {
    const u64 j = it.get_group(0);
    const u32 t = (u32)it.get_local_id(0);
    const JobDev J = d_jobs[j];
    const u32 nc = J.n_chunks;
    const u32 per = (nc + kScanWG - 1) / kScanWG;
    const u32 b = std::min(nc, t * per), e = std::min(nc, b + per);
    u64 part = 0;
    for (u32 i = b; i < e; ++i) part += d_cs[J.chunk_base + i] & ~kStoredFlag;
    u64 pre = sycl::exclusive_scan_over_group(it.get_group(), part,
                                              sycl::plus<u64>());
    u64 total = sycl::reduce_over_group(it.get_group(), part,
                                        sycl::plus<u64>());
    for (u32 i = b; i < e; ++i) {
      d_co[J.chunk_base + i] = pre;
      pre += d_cs[J.chunk_base + i] & ~kStoredFlag;
    }
    const u64 out = J.hdr_bytes + total;
    const bool ok = J.valid && out <= J.dst_cap;
    if (ok) {
      for (u32 i = b; i < e; ++i)
        St32(J.dst + sizeof(Header) + 4ull * i, d_cs[J.chunk_base + i]);
      if (t == 0) {
        St32(J.dst + 0, kMagic);
        St32(J.dst + 4, kChunkSize);
        St32(J.dst + 8, (u32)J.src_size);
        St32(J.dst + 12, (u32)(J.src_size >> 32));
        St32(J.dst + 16, nc);
        St32(J.dst + 20, 0);
      }
    }
    if (t == 0) d_out[j] = JobOut{ok ? out : 0, ok ? 1u : 0u, 0};
  });

  // ---- 3. gather payloads ----
  if (C) {
    q.parallel_for(sycl::nd_range<1>(C * kGroup, kGroup),
                   [=](sycl::nd_item<1> it) {
      const u64 c = it.get_group(0);
      const u32 lid = (u32)it.get_local_id(0);
      const u32 j = d_cj[c];
      if (!d_out[j].ok) return;
      const JobDev J = d_jobs[j];
      const u32 raw = d_cs[c] & ~kStoredFlag;
      const u8 *s = d_scr + c * bound;
      u8 *d = J.dst + J.hdr_bytes + d_co[c];
      for (u32 k = lid; k < raw; k += kGroup) d[k] = s[k];
    });
  }

  // ---- 4. statuses back, staged outputs back, ONE wait ----
  std::vector<JobOut> res(n);
  q.memcpy(res.data(), d_out, n * sizeof(JobOut)).wait();
  bool any_back = false;
  for (size_t j = 0; j < n; ++j)
    if (out_off[j] != ~0ull && res[j].ok) {
      q.memcpy(dsts[j], sout + out_off[j], res[j].out_size);
      any_back = true;
    }
  if (any_back) q.wait();
  int all = 1;
  for (size_t j = 0; j < n; ++j) {
    out_sizes[j] = res[j].ok ? res[j].out_size : 0;
    oks[j] = (int)res[j].ok;
    all &= oks[j];
  }
  return all;
}

int DecompressBatchImpl(QueueBox *qb, size_t n, const void *const *srcs,
                        const size_t *src_sizes, void *const *dsts,
                        const size_t *dst_caps, size_t *out_sizes, int *oks) {
  std::lock_guard<std::mutex> lk(qb->mu);
  sycl::queue &q = qb->q;
  std::vector<JobDev> jobs(n);
  std::vector<u64> in_off(n, ~0ull);
  u64 stage_in = 0;
  for (size_t j = 0; j < n; ++j) {
    jobs[j].src_size = src_sizes[j];
    jobs[j].dst_cap = dst_caps[j];
    if (!IsDevicePtr(q, srcs[j])) { in_off[j] = stage_in; stage_in += src_sizes[j]; }
  }
  u8 *sin = stage_in ? qb->stage_in.Get<u8>(q, stage_in) : nullptr;
  for (size_t j = 0; j < n; ++j) {
    if (in_off[j] != ~0ull) {
      if (src_sizes[j]) q.memcpy(sin + in_off[j], srcs[j], src_sizes[j]);
      jobs[j].src = sin + in_off[j];
    } else {
      jobs[j].src = static_cast<const u8 *>(srcs[j]);
    }
  }
  // ---- 1. headers (the one extra sync) ----
  u8 *d_h = qb->hdrs.Get<u8>(q, n * sizeof(Header));
  // Host arrays read by ASYNC copies must outlive the wait below -- declare
  // them at function scope, never inside a block.
  std::vector<const u8 *> srcp(n);
  std::vector<u64> ss(n);
  for (size_t j = 0; j < n; ++j) srcp[j] = jobs[j].src;
  {
    const u8 **d_srcp = reinterpret_cast<const u8 **>(
        qb->jobs.Get<JobDev>(q, std::max<size_t>(n, 1)));
    q.memcpy(d_srcp, srcp.data(), n * sizeof(u8 *));
    for (size_t j = 0; j < n; ++j) ss[j] = src_sizes[j];
    u64 *d_ss = qb->coff.Get<u64>(q, n);
    q.memcpy(d_ss, ss.data(), n * sizeof(u64));
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
      const u64 j = i[0];
      for (u32 k = 0; k < sizeof(Header); ++k)
        d_h[j * sizeof(Header) + k] = k < d_ss[j] ? d_srcp[j][k] : 0;
    });
  }
  std::vector<Header> H(n);
  q.memcpy(H.data(), d_h, n * sizeof(Header)).wait();

  std::vector<u64> out_off(n, ~0ull);
  u64 stage_out = 0, C = 0;
  for (size_t j = 0; j < n; ++j) {
    JobDev &J = jobs[j];
    const Header &h = H[j];
    const u64 nc = h.n_chunks, cs = h.chunk_size, orig = h.orig_size;
    // An empty output may have a null dst (an empty vector's data()).
    bool v = srcs[j] && (dsts[j] || orig == 0) && J.src_size >= sizeof(Header) &&
             h.magic == kMagic && cs > 0 && cs <= kChunkSize &&
             nc * cs >= orig && (nc == 0 || (nc - 1) * cs < orig) &&
             orig <= J.dst_cap && sizeof(Header) + 4 * nc <= J.src_size;
    J.valid = v;
    J.n_chunks = v ? (u32)nc : 0;
    J.chunk_size = (u32)cs;
    J.orig = v ? orig : 0;
    J.hdr_bytes = (u32)(sizeof(Header) + 4 * J.n_chunks);
    J.chunk_base = C;
    C += J.n_chunks;
    if (v && !IsDevicePtr(q, dsts[j])) { out_off[j] = stage_out; stage_out += orig; }
  }
  u8 *sout = stage_out ? qb->stage_out.Get<u8>(q, stage_out) : nullptr;
  for (size_t j = 0; j < n; ++j)
    jobs[j].dst = out_off[j] != ~0ull ? sout + out_off[j]
                                      : static_cast<u8 *>(dsts[j]);
  std::vector<u32> chunk_job(C);
  for (size_t j = 0; j < n; ++j)
    for (u32 c = 0; c < jobs[j].n_chunks; ++c)
      chunk_job[jobs[j].chunk_base + c] = (u32)j;
  JobDev *d_jobs = qb->jobs.Get<JobDev>(q, n);
  JobOut *d_out = qb->outs.Get<JobOut>(q, n);
  u32 *d_cj = qb->chunk_job.Get<u32>(q, C);
  u64 *d_co = qb->coff.Get<u64>(q, C);
  q.memcpy(d_jobs, jobs.data(), n * sizeof(JobDev));
  if (C) q.memcpy(d_cj, chunk_job.data(), C * sizeof(u32));

  // ---- 2. per-job validate + scan ----
  q.parallel_for(sycl::nd_range<1>(n * kScanWG, kScanWG),
                 [=](sycl::nd_item<1> it) {
    const u64 j = it.get_group(0);
    const u32 t = (u32)it.get_local_id(0);
    const JobDev J = d_jobs[j];
    const u32 nc = J.n_chunks, per = (nc + kScanWG - 1) / kScanWG;
    const u32 b = std::min(nc, t * per), e = std::min(nc, b + per);
    const u8 *tab = J.src + sizeof(Header);
    u64 part = 0;
    u32 bad = 0;
    for (u32 i = b; i < e; ++i) {
      const u32 v = Ld32(tab + 4ull * i);
      const u32 raw = v & ~kStoredFlag;
      const u64 dlen = std::min<u64>(J.chunk_size, J.orig - (u64)i * J.chunk_size);
      bad |= raw > J.chunk_size || ((v & kStoredFlag) && raw != dlen) || raw == 0;
      part += raw;
    }
    u64 pre = sycl::exclusive_scan_over_group(it.get_group(), part,
                                              sycl::plus<u64>());
    u64 total = sycl::reduce_over_group(it.get_group(), part, sycl::plus<u64>());
    u32 anybad = sycl::reduce_over_group(it.get_group(), bad,
                                         sycl::bit_or<u32>());
    for (u32 i = b; i < e; ++i) {
      d_co[J.chunk_base + i] = pre;
      pre += Ld32(tab + 4ull * i) & ~kStoredFlag;
    }
    if (t == 0) {
      const bool ok = J.valid && !anybad && J.hdr_bytes + total <= J.src_size;
      d_out[j] = JobOut{ok ? J.orig : 0, ok ? 1u : 0u, 0};
    }
  });

  // ---- 3. decode: one work-group (= one 32-lane sub-group) per chunk ----
  if (C) {
    q.submit([&](sycl::handler &h) {
      sycl::local_accessor<u8, 1> s_in(sycl::range<1>(kChunkSize), h);
      sycl::local_accessor<u8, 1> s_out(sycl::range<1>(kChunkSize), h);
      h.parallel_for(
          sycl::nd_range<1>(C * kGroup, kGroup),
          [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const u64 c = it.get_group(0);
            const u32 lane = (u32)it.get_local_id(0);
            const u32 j = d_cj[c];
            // Read once and broadcast: another chunk of this job may clear
            // ok concurrently, and every lane must take the same path here.
            const u32 jok = sycl::group_broadcast(it.get_group(),
                                                  lane == 0 ? d_out[j].ok : 0u,
                                                  0);
            if (!jok) return;
            const JobDev J = d_jobs[j];
            auto sg = it.get_sub_group();
            const u32 i = (u32)(c - J.chunk_base);
            const u32 v = Ld32(J.src + sizeof(Header) + 4ull * i);
            const u32 raw = v & ~kStoredFlag;
            const bool stored = (v & kStoredFlag) != 0;
            const u64 doff = (u64)i * J.chunk_size;
            const u32 dlen = (u32)std::min<u64>(J.chunk_size, J.orig - doff);
            const u8 *csrc = J.src + J.hdr_bytes + d_co[c];
            u8 *si = &s_in[0];
            u8 *so = &s_out[0];
            for (u32 k = lane; k < raw; k += kGroup) si[k] = csrc[k];
            sycl::group_barrier(it.get_group());
            u32 op_final = dlen;
            if (stored) {
              for (u32 k = lane; k < raw; k += kGroup) so[k] = si[k];
            } else {
              u32 ip = 0, op = 0;
              while (true) {
                u32 litlen = 0;
                int stop = 0;
                u8 tok = 0;
                if (lane == 0) {
                  if (ip >= raw) {
                    stop = 1;
                  } else {
                    tok = si[ip++];
                    litlen = tok >> 4;
                    if (litlen == 15) {
                      u8 bb;
                      do { bb = si[ip++]; litlen += bb; } while (bb == 255 && ip < raw);
                    }
                    // Corrupt input: clamp so the copies stay inside the
                    // chunk-sized local buffers (output then fails roundtrip
                    // checks, but never overruns).
                    if (op + litlen > dlen || ip + litlen > raw) { stop = 1; }
                  }
                }
                stop = sycl::group_broadcast(sg, stop, 0);
                if (stop) break;
                litlen = sycl::group_broadcast(sg, litlen, 0);
                const u32 lip = sycl::group_broadcast(sg, ip, 0);
                const u32 lop = sycl::group_broadcast(sg, op, 0);
                for (u32 base = 0; base < litlen; base += kGroup) {
                  u32 k = base + lane;
                  if (k < litlen) so[lop + k] = si[lip + k];
                }
                sycl::group_barrier(sg);
                int last = 0;
                if (lane == 0) {
                  ip = lip + litlen;
                  op = lop + litlen;
                  if (ip >= raw) last = 1;
                }
                last = sycl::group_broadcast(sg, last, 0);
                if (last) break;
                u32 off = 0, ml = 0;
                int badm = 0;
                if (lane == 0) {
                  off = (u32)si[ip] | ((u32)si[ip + 1] << 8);
                  ip += 2;
                  ml = tok & 0x0F;
                  if (ml == 15) {
                    u8 bb;
                    do { bb = si[ip++]; ml += bb; } while (bb == 255 && ip < raw);
                  }
                  ml += kMinMatch;
                  badm = off == 0 || off > op || op + ml > dlen;
                }
                badm = sycl::group_broadcast(sg, badm, 0);
                if (badm) break;
                off = sycl::group_broadcast(sg, off, 0);
                ml = sycl::group_broadcast(sg, ml, 0);
                const u32 mop = sycl::group_broadcast(sg, op, 0);
                if (off >= ml) {
                  for (u32 base = 0; base < ml; base += kGroup) {
                    u32 k = base + lane;
                    if (k < ml) so[mop + k] = so[mop - off + k];
                  }
                } else if (lane == 0) {
                  u32 mp = mop - off;
                  for (u32 k = 0; k < ml; ++k) so[mop + k] = so[mp++];
                }
                if (lane == 0) op = mop + ml;
                sycl::group_barrier(sg);
              }
              op_final = op;  // meaningful on lane 0, which checks it
            }
            // A chunk that did not decode to exactly its length is corrupt:
            // fail the whole job (relaxed store; the host reads it after wait).
            if (lane == 0 && !stored && op_final != dlen) {
              sycl::atomic_ref<u32, sycl::memory_order::relaxed,
                               sycl::memory_scope::device,
                               sycl::access::address_space::global_space>(
                  d_out[j].ok)
                  .store(0u);
            }
            sycl::group_barrier(it.get_group());
            for (u32 k = lane; k < dlen; k += kGroup) J.dst[doff + k] = so[k];
          });
    });
  }

  std::vector<JobOut> res(n);
  q.memcpy(res.data(), d_out, n * sizeof(JobOut)).wait();
  bool any_back = false;
  for (size_t j = 0; j < n; ++j)
    if (out_off[j] != ~0ull && res[j].ok) {
      q.memcpy(dsts[j], sout + out_off[j], res[j].out_size);
      any_back = true;
    }
  if (any_back) q.wait();
  int all = 1;
  for (size_t j = 0; j < n; ++j) {
    out_sizes[j] = res[j].ok ? res[j].out_size : 0;
    oks[j] = (int)res[j].ok;
    all &= oks[j];
  }
  return all;
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

// Batched entry points: n independent jobs, ONE launch per phase. Each job's
// src/dst may be host or device USM independently. Per-job out_sizes/oks;
// returns 1 iff every job succeeded.
int clio_sycl_lz4_compress_batch(void *qbv, size_t n, const void *const *srcs,
                                 const size_t *src_sizes, void *const *dsts,
                                 const size_t *dst_caps, size_t *out_sizes,
                                 int *oks) {
  if (!qbv || (n && (!srcs || !src_sizes || !dsts || !dst_caps || !out_sizes ||
                     !oks)))
    return 0;
  if (n == 0) return 1;
  try {
    return CompressBatchImpl(static_cast<QueueBox *>(qbv), n, srcs, src_sizes,
                             dsts, dst_caps, out_sizes, oks);
  } catch (...) {
    for (size_t j = 0; j < n; ++j) { oks[j] = 0; out_sizes[j] = 0; }
    return 0;
  }
}

int clio_sycl_lz4_decompress_batch(void *qbv, size_t n,
                                   const void *const *srcs,
                                   const size_t *src_sizes, void *const *dsts,
                                   const size_t *dst_caps, size_t *out_sizes,
                                   int *oks) {
  if (!qbv || (n && (!srcs || !src_sizes || !dsts || !dst_caps || !out_sizes ||
                     !oks)))
    return 0;
  if (n == 0) return 1;
  try {
    return DecompressBatchImpl(static_cast<QueueBox *>(qbv), n, srcs,
                               src_sizes, dsts, dst_caps, out_sizes, oks);
  } catch (...) {
    for (size_t j = 0; j < n; ++j) { oks[j] = 0; out_sizes[j] = 0; }
    return 0;
  }
}

// Single-buffer calls: a batch of one (same scratch reuse, same syncs).
int clio_sycl_lz4_compress(void *qbv, const void *src, size_t src_size,
                           void *dst, size_t dst_cap, size_t *out) {
  if (!out) return 0;
  int ok = 0;
  return clio_sycl_lz4_compress_batch(qbv, 1, &src, &src_size, &dst, &dst_cap,
                                      out, &ok);
}
int clio_sycl_lz4_decompress(void *qbv, const void *src, size_t src_size,
                             void *dst, size_t dst_cap, size_t *out) {
  if (!out) return 0;
  int ok = 0;
  return clio_sycl_lz4_decompress_batch(qbv, 1, &src, &src_size, &dst,
                                        &dst_cap, out, &ok);
}

}  // extern "C"
