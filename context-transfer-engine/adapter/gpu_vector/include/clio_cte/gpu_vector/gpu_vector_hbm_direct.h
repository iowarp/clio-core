/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * Zero-task reads of PROMOTED, hbm-resident compressed pages.
 *
 * The score-driven prefetch protocol raises the scores of soon-needed page
 * blobs (>= 0.9) so the CTE's organizer migrates them into the kHbm tier.
 * This header is the access-time half: a co-located reader resolves the
 * blob's cached SHM record straight to DEVICE pointers (the kHbm analog of
 * the kRam TryReadBlobShm fast path), gathers the block extents D2D into a
 * contiguous scratch, and decompresses EVERY frame of a batch with one
 * client-side nvcomp launch. No task, no hop, no host byte.
 *
 * Safety follows the shm fast-path discipline: refuse by default, and
 * re-check placement_gen_ AFTER the payload read -- a frame whose blocks
 * moved mid-read is discarded and refetched through the task path.
 */
#ifndef CLIO_CTE_GPU_VECTOR_HBM_DIRECT_H_
#define CLIO_CTE_GPU_VECTOR_HBM_DIRECT_H_

#include <clio_cte/core/core_client.h>
#include <clio_cte/gpu_vector/gpu_vector.h>
#include <clio_runtime/bdev/transports/mem_bdev_transport.h>
#include <clio_ctp/compress/nvcomp.h>

#include <cuda_runtime.h>

#include <array>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace clio::cte::gpu_vector {

/** One whole storage page the caller wants: decompressed bytes of blob
 *  `gp` into dst[0..len). */
struct DirectWant {
  clio::run::u64 gp;
  uint8_t *dst;
  clio::run::u64 len;
};

struct DirectStats {
  clio::run::u64 hits = 0;
  clio::run::u64 misses = 0;
  clio::run::u64 no_rec = 0;     /**< no shm record for the blob */
  clio::run::u64 cold = 0;       /**< score below the 0.9 threshold */
  clio::run::u64 trunc = 0;      /**< record truncated (too many blocks) */
  clio::run::u64 unresolv = 0;   /**< blocks not hbm / not resolvable */
  clio::run::u64 moved = 0;      /**< placement changed mid-read */
};

/** Growable device scratch for gathered compressed frames (separate from
 *  raw_lz4::DevScratch, which DecompressBatch uses internally). */
inline uint8_t *GatherScratch(size_t bytes) {
  static thread_local uint8_t *buf = nullptr;
  static thread_local size_t cap = 0;
  if (bytes > cap) {
    if (buf) cudaFree(buf);
    if (cudaMalloc(&buf, bytes) != cudaSuccess) {
      buf = nullptr;
      cap = 0;
      return nullptr;
    }
    cap = bytes;
  }
  return buf;
}

/**
 * Serve as many of `want` as possible with zero tasks; frames that cannot
 * be served (score below threshold, not hbm-resident, moved mid-read,
 * malformed) are appended to `leftover` for the task path. Blocking: served
 * frames' bytes are resident in their dst on return.
 */
template <typename VecT>
inline bool HbmDirectFetchBatch(VecT &vec, clio::cte::core::Client &cli,
                                const std::vector<DirectWant> &want,
                                std::vector<DirectWant> *leftover,
                                DirectStats *stats = nullptr,
                                bool device_only = false) {
  using clio::run::bdev::MemBdevTransport;
  struct Pending {
    const DirectWant *w;
    clio::cte::core::ShmBlobRecord rec;
    uint8_t *frame_dev;      // gathered contiguous compressed frame
    clio::run::u64 stored;   // stored (compressed) size
  };
  cudaStream_t stream = ctp::raw_lz4::RawStream();
  if (stream == nullptr) {
    for (const auto &w : want) leftover->push_back(w);
    return false;
  }
  std::vector<Pending> pend;
  pend.reserve(want.size());
  std::vector<size_t> raw_served;
  // Pass 1: resolve + gather (D2D, async on our stream).
  clio::run::u64 gather_bytes = 0;
  std::vector<clio::cte::core::ShmBlobRecord> recs(want.size());
  std::vector<int> ok(want.size(), 0);
  for (size_t i = 0; i < want.size(); ++i) {
    const auto &w = want[i];
    clio::cte::core::ShmBlobRecord &rec = recs[i];
    const std::string name = vec.PageBlobName(w.gp);
    if (!cli.TryGetBlobRecordShm(vec.TagId(), name, &rec)) {
      if (stats) stats->no_rec++;
      continue;
    }
    // No score gate: a page below 0.9 lives in a lower tier, and this
    // path can now serve THOSE too (H2D of the compressed frame). The
    // score still decides PLACEMENT -- it just no longer decides whether
    // a read needs a task.
    if ((rec.flags_ & clio::cte::core::kShmBlobTruncated) != 0) {
      if (stats) stats->trunc++;
      continue;
    }
    if (rec.num_blocks_ == 0 || rec.num_blocks_ > clio::cte::core::kMaxInlineBlocks) continue;
    bool resolvable = true;
    bool from_ram = false;
    for (clio::run::u32 b = 0; b < rec.num_blocks_; ++b) {
      const auto &blk = rec.blocks_[b];
      if (blk.bdev_type_ ==
          (clio::run::u32) clio::run::bdev::BdevType::kRam) {
        // RAM-tier page: the runtime's segment is mappable in-process, so
        // the frame can be pulled H2D and decompressed client-side --
        // still no task, just a PCIe copy of COMPRESSED bytes instead of a
        // D2D copy. This is what makes a miss cheap when the model is far
        // larger than the device tier.
        //
        // device_only: the caller has a DECOMPRESSED DRAM tier, which
        // serves the same bytes over the same PCIe with no decompress at
        // all -- so refuse ram pages and let it use that instead.
        if (device_only) {
          resolvable = false;
          break;
        }
        from_ram = true;
        continue;
      }
      if (blk.bdev_type_ !=
          (clio::run::u32) clio::run::bdev::BdevType::kHbm) {
        resolvable = false;
        break;
      }
      auto *t = MemBdevTransport::LookupHbm(blk.target_pool_);
      if (t == nullptr ||
          t->ResolveHbmSpan(blk.target_offset_, blk.size_) == nullptr) {
        resolvable = false;
        break;
      }
    }
    if (from_ram) {
      // Mixed layouts are not worth the bookkeeping; require all-RAM.
      char *hp = nullptr;
      size_t hsz2 = 0;
      clio::run::u64 hgen = 0;
      if (!cli.TryGetStoredViewShm(vec.TagId(), name, &hp, &hsz2, &hgen) ||
          hsz2 != rec.total_size_) {
        resolvable = false;
      }
    }
    if (!resolvable) {
      if (stats) {
        stats->unresolv++;
        if (stats->unresolv == 1000) {
          std::fprintf(stderr,
                       "[hbm-direct sample] '%s' score=%.2f nblk=%u blk0: "
                       "type=%u pool=%llu off=%llu lookup=%p\n",
                       name.c_str(), rec.score_, rec.num_blocks_,
                       rec.blocks_[0].bdev_type_,
                       (unsigned long long) rec.blocks_[0].target_pool_.ToU64(),
                       (unsigned long long) rec.blocks_[0].target_offset_,
                       (void *) MemBdevTransport::LookupHbm(
                           rec.blocks_[0].target_pool_));
        }
      }
      continue;
    }
    ok[i] = 1;
    gather_bytes += (rec.total_size_ + 255) & ~255ull;
  }
  uint8_t *scratch =
      gather_bytes ? GatherScratch(gather_bytes) : nullptr;
  clio::run::u64 scratch_off = 0;
  for (size_t i = 0; i < want.size(); ++i) {
    if (!ok[i]) {
      leftover->push_back(want[i]);
      if (stats) stats->misses++;
      continue;
    }
    if (scratch == nullptr) {
      leftover->push_back(want[i]);
      if (stats) stats->misses++;
      continue;
    }
    clio::cte::core::ShmBlobRecord &rec = recs[i];
    uint8_t *frame = scratch + scratch_off;
    scratch_off += (rec.total_size_ + 255) & ~255ull;
    clio::run::u64 off = 0;
    bool copied_ok = true;
    for (clio::run::u32 b = 0; b < rec.num_blocks_ && copied_ok; ++b) {
      const auto &blk = rec.blocks_[b];
      if (blk.bdev_type_ ==
          (clio::run::u32) clio::run::bdev::BdevType::kRam) {
        char *hp = nullptr;
        size_t hsz2 = 0;
        clio::run::u64 hgen = 0;
        const std::string nm = vec.PageBlobName(want[i].gp);
        if (!cli.TryGetStoredViewShm(vec.TagId(), nm, &hp, &hsz2, &hgen) ||
            cudaMemcpyAsync(frame + off, hp, blk.size_,
                            cudaMemcpyHostToDevice, stream) != cudaSuccess) {
          copied_ok = false;
          break;
        }
        off += blk.size_;
        continue;
      }
      auto *t = MemBdevTransport::LookupHbm(blk.target_pool_);
      char *src = t->ResolveHbmSpan(blk.target_offset_, blk.size_);
      if (src == nullptr ||
          cudaMemcpyAsync(frame + off, src, blk.size_,
                          cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
        copied_ok = false;
        break;
      }
      off += blk.size_;
    }
    if (!copied_ok || off < rec.total_size_) {
      leftover->push_back(want[i]);
      if (stats) stats->misses++;
      continue;
    }
    if (!rec.IsTransformed()) {
      // RAW blob: the gathered bytes ARE the data -- copy to dst directly,
      // no decompress. (Gathered to scratch first because blocks scatter.)
      const clio::run::u64 n =
          std::min<clio::run::u64>(want[i].len, rec.total_size_);
      if (cudaMemcpyAsync(want[i].dst, frame, n, cudaMemcpyDeviceToDevice,
                          stream) == cudaSuccess) {
        raw_served.push_back(i);
      } else {
        leftover->push_back(want[i]);
        if (stats) stats->misses++;
      }
      continue;
    }
    pend.push_back(Pending{&want[i], rec, frame, rec.total_size_});
  }
  if (pend.empty() && raw_served.empty()) return true;
  // Pass 2: headers + tables (small D2H), one sync.
  // (CompressionHeader wire layout: 3x u32 + pad + 2x u64 = 32 bytes.)
  const size_t kHdr = 32;
  std::vector<std::array<uint8_t, 32 + sizeof(ctp::raw_lz4::Table)>> heads(
      pend.size());
  for (size_t i = 0; i < pend.size(); ++i) {
    cudaMemcpyAsync(heads[i].data(), pend[i].frame_dev, heads[i].size(),
                    cudaMemcpyDeviceToHost, stream);
  }
  if (cudaStreamSynchronize(stream) != cudaSuccess) {
    for (auto &pd : pend) leftover->push_back(*pd.w);
    return false;
  }
  // Pass 3: one batched decompress across every valid frame.
  std::vector<ctp::raw_lz4::FrameRef> frames;
  std::vector<size_t> fidx;
  std::vector<const ctp::raw_lz4::Table *> tbls(pend.size());
  for (size_t i = 0; i < pend.size(); ++i) {
    tbls[i] = reinterpret_cast<const ctp::raw_lz4::Table *>(heads[i].data() +
                                                            kHdr);
    if (tbls[i]->magic != ctp::raw_lz4::kMagic) {
      leftover->push_back(*pend[i].w);
      if (stats) stats->misses++;
      continue;
    }
    frames.push_back(ctp::raw_lz4::FrameRef{
        pend[i].frame_dev + kHdr, tbls[i], pend[i].w->dst,
        (size_t) pend[i].w->len});
    fidx.push_back(i);
  }
  bool all_ok = true;
  if (!frames.empty()) {
    all_ok = ctp::raw_lz4::DecompressBatch(frames.data(), frames.size(),
                                           stream) &&
             cudaStreamSynchronize(stream) == cudaSuccess;
  }
  // Raw-served frames: ensure their D2D copies completed, then gen-check.
  if (!raw_served.empty()) {
    cudaStreamSynchronize(stream);
    for (size_t ri : raw_served) {
      clio::cte::core::ShmBlobRecord after;
      const std::string name = vec.PageBlobName(want[ri].gp);
      const bool still =
          cli.TryGetBlobRecordShm(vec.TagId(), name, &after) &&
          after.placement_gen_ == recs[ri].placement_gen_;
      if (!still) {
        leftover->push_back(want[ri]);
        if (stats) { stats->misses++; stats->moved++; }
      } else if (stats) {
        stats->hits++;
      }
    }
  }
  // Pass 4: placement_gen recheck -- a frame whose blocks moved mid-read is
  // poisoned; refetch it through the task path.
  for (size_t k = 0; k < fidx.size(); ++k) {
    const size_t i = fidx[k];
    clio::cte::core::ShmBlobRecord after;
    const std::string name = vec.PageBlobName(pend[i].w->gp);
    const bool still =
        all_ok && cli.TryGetBlobRecordShm(vec.TagId(), name, &after) &&
        after.placement_gen_ == pend[i].rec.placement_gen_;
    if (!still) {
      leftover->push_back(*pend[i].w);
      if (stats) { stats->misses++; stats->moved++; }
    } else if (stats) {
      stats->hits++;
    }
  }
  return true;
}

/**
 * Install the zero-task read path on `vec`: from here, every whole-page
 * fetch the vector issues is offered to the in-process resolver first, and
 * only refusals become GetBlob tasks. One shared client + stats block per
 * vector. Safe to call once at init; a null install is a no-op.
 */
template <typename VecT>
inline void InstallDirectReads(VecT &vec, DirectStats *stats,
                               bool device_only = false) {
  auto cli = std::make_shared<clio::cte::core::Client>(
      clio::cte::core::kCtePoolId);
  cli->AttachShmCache();
  VecT *vp = &vec;
  using PP = typename VecT::PendingPage;
  const bool dev_only = device_only;
  vec.SetDirectBatchFn([vp, cli, stats, dev_only](const std::vector<PP> &want,
                                                  std::vector<PP> *leftover) {
    std::vector<DirectWant> w;
    w.reserve(want.size());
    for (const auto &p : want) w.push_back(DirectWant{p.gp, p.dst, p.len});
    std::vector<DirectWant> lo;
    HbmDirectFetchBatch(*vp, *cli, w, &lo, stats, dev_only);
    for (const auto &l : lo) leftover->push_back(PP{l.gp, l.dst, l.len});
  });
}

}  // namespace clio::cte::gpu_vector

#endif  // CLIO_CTE_GPU_VECTOR_HBM_DIRECT_H_
