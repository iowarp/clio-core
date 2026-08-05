/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#ifndef CLIO_CTE_CORE_BLOB_BATCH_H_
#define CLIO_CTE_CORE_BLOB_BATCH_H_

/**
 * Centralized batching/vectoring helpers for the CTE blob data path.
 *
 * One implementation shared by every producer and consumer of batched and
 * vectored puts/gets: the core client (encodes MultiPutBlob batches), the
 * core runtime (executes them), and every INTERPOSING chimod that speaks the
 * core task interface (replication, compressor) and must iterate the same
 * records/regions. Before this header each of those hand-rolled its own
 * desc (de)serialization and scalar-vs-segments loops — three drifting
 * copies of wire-format knowledge.
 */

#include <clio_cte/core/core_tasks.h>

#include <string>
#include <vector>

namespace clio::cte::core {

/** Pack a MultiPutBlob batch's descriptors into the wire string carried by
 *  MultiPutBlobTask::descs_. Inverse of DecodeMultiPutDescs. */
inline std::string EncodeMultiPutDescs(const std::vector<MultiPutDesc> &descs) {
  std::vector<char> buf;
  ctp::ipc::GlobalSerialize<std::vector<char>> ar(buf);
  ar(const_cast<std::vector<MultiPutDesc> &>(descs));
  ar.Finalize();
  return std::string(buf.begin(), buf.end());
}

/** Unpack MultiPutBlobTask::descs_. */
inline std::vector<MultiPutDesc> DecodeMultiPutDescs(
    const clio::run::priv::string &packed) {
  std::vector<MultiPutDesc> descs;
  std::string raw = packed.str();
  std::vector<char> buf(raw.begin(), raw.end());
  ctp::ipc::GlobalDeserialize<std::vector<char>> ar(buf);
  ar(descs);
  return descs;
}

/**
 * A decoded, addressable view of a MultiPutBlob batch: descriptors plus the
 * resolved base pointer of the shared staging buffer. Records whose payload
 * slice falls outside the buffer are flagged invalid instead of dropped, so
 * consumers can keep the core's accounting (count them, skip the I/O).
 */
struct MultiPutBatchView {
  std::vector<MultiPutDesc> descs_;
  char *base_ = nullptr;
  clio::run::u64 data_len_ = 0;

  /** Decode the task's descriptors and resolve its staging base. Returns
   *  false when the batch carries no payload buffer or no records. */
  static bool Attach(MultiPutBlobTask &task, MultiPutBatchView *out) {
    out->descs_ = DecodeMultiPutDescs(task.descs_);
    out->data_len_ = task.data_len_;
    out->base_ = nullptr;
    if (!task.data_.IsNull()) {
      auto *ipc_manager = CLIO_CPU_IPC;
      out->base_ =
          ipc_manager->ToFullPtr<char>(task.data_.template Cast<char>()).ptr_;
    }
    return out->base_ != nullptr && !out->descs_.empty();
  }

  size_t size() const { return descs_.size(); }

  /** True when record i's payload slice lies inside the staging buffer. */
  bool RecordValid(size_t i) const {
    const MultiPutDesc &d = descs_[i];
    return d.payload_off_ + d.size_ <= data_len_;
  }

  /** Record i's payload as an absolute in-process pointer (the private-put
   *  zero-copy contract: bdev I/O reads it directly). Only meaningful when
   *  RecordValid(i). */
  ctp::ipc::ShmPtr<> RecordSlice(size_t i) const {
    return ctp::ipc::ShmPtr<>::FromRaw(base_ + descs_[i].payload_off_);
  }
};

/** One contiguous region of a put/get: either the task's scalar triple or a
 *  vectored segment. */
struct BlobRegion {
  clio::run::u64 blob_off_;
  clio::run::u64 size_;
  ctp::ipc::ShmPtr<> data_;
};

/**
 * Invoke fn(const BlobRegion&) for every region a put/get task addresses —
 * each vectored segment IN LIST ORDER (the core's last-writer-wins rule for
 * overlapping segments), or the scalar triple when segments_ is empty. Works
 * for any task with offset_/size_/blob_data_/segments_ (PutBlobTask,
 * GetBlobTask). Returns false if fn returned false (early stop).
 */
template <typename TaskT, typename Fn>
inline bool ForEachBlobRegion(TaskT &task, Fn &&fn) {
  if (!task.segments_.empty()) {
    for (size_t i = 0; i < task.segments_.size(); ++i) {
      const auto &seg = task.segments_[i];
      BlobRegion r{seg.blob_off_, seg.size_, seg.data_};
      if (!fn(static_cast<const BlobRegion &>(r))) {
        return false;
      }
    }
    return true;
  }
  BlobRegion r{task.offset_, task.size_, task.blob_data_};
  return fn(static_cast<const BlobRegion &>(r));
}

/** The byte range a put/get task covers: the scalar [offset, offset+size) or
 *  the union of its vectored segments — the same range the core's
 *  torn-layout guard and every interposer coverage probe must agree on. */
template <typename TaskT>
inline void BlobRequestRange(TaskT &task, clio::run::u64 *lo_out,
                             clio::run::u64 *hi_out) {
  if (!task.segments_.empty()) {
    clio::run::u64 lo = ~0ULL, hi = 0;
    for (size_t i = 0; i < task.segments_.size(); ++i) {
      const auto &seg = task.segments_[i];
      if (seg.blob_off_ < lo) lo = seg.blob_off_;
      if (seg.blob_off_ + seg.size_ > hi) hi = seg.blob_off_ + seg.size_;
    }
    *lo_out = lo;
    *hi_out = hi;
    return;
  }
  *lo_out = task.offset_;
  *hi_out = task.offset_ + task.size_;
}

}  // namespace clio::cte::core

#endif  // CLIO_CTE_CORE_BLOB_BATCH_H_
