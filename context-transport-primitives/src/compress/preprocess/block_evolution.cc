/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file block_evolution.cc
 * @brief Host reference for the block-evolution metric, plus the tracker that
 * retains the previous sampled timestep so the metric has a B1 at all.
 *
 * Always compiled. On a CUDA build the tracker forwards device-resident
 * chunks to block_evolution_gpu_kernels.cu; without CUDA the detail:: hooks
 * below fail cleanly and only the host path is available -- the same split
 * byte_shuffle_cpu_stub.cc uses.
 */

#include <cmath>
#include <cstdint>
#include <cstring>
#include <unordered_map>

#include "clio_ctp/compress/preprocess/block_evolution_gpu.h"

namespace ctp {
namespace {

/** Kahan-free double accumulation, matching the device kernel exactly: the
    kernel's per-block tree reduction and this serial loop differ only in
    summation order, which is why the test compares them with a relative
    tolerance rather than bit-for-bit. */
template <typename T>
void AccumulateTyped(const T *b1, const T *b2, size_t n, double *sum_d2,
                     double *sum_n1, double *sum_n2,
                     unsigned long long *bad) {
  double d2 = 0.0, n1 = 0.0, n2 = 0.0;
  unsigned long long skipped = 0;
  for (size_t i = 0; i < n; ++i) {
    const double x1 = static_cast<double>(b1[i]);
    const double x2 = static_cast<double>(b2[i]);
    if (!std::isfinite(x1) || !std::isfinite(x2)) {
      ++skipped;
      continue;
    }
    const double d = x2 - x1;
    d2 += d * d;
    n1 += x1 * x1;
    n2 += x2 * x2;
  }
  *sum_d2 = d2;
  *sum_n1 = n1;
  *sum_n2 = n2;
  *bad = skipped;
}

}  // namespace

bool ComputeBlockEvolutionHost(const void *prev, const void *curr,
                               size_t num_elements, DataType type,
                               double epsilon, BlockEvolution *out) {
  if (out == nullptr) return false;
  *out = BlockEvolution{};
  if (prev == nullptr || curr == nullptr || num_elements == 0) return false;

  double d2 = 0.0, n1 = 0.0, n2 = 0.0;
  unsigned long long bad = 0;
  switch (type) {
    case DataType::FLOAT32:
      AccumulateTyped(static_cast<const float *>(prev),
                      static_cast<const float *>(curr), num_elements, &d2, &n1,
                      &n2, &bad);
      break;
    case DataType::DOUBLE64:
      AccumulateTyped(static_cast<const double *>(prev),
                      static_cast<const double *>(curr), num_elements, &d2,
                      &n1, &n2, &bad);
      break;
    case DataType::INT32:
      AccumulateTyped(static_cast<const int32_t *>(prev),
                      static_cast<const int32_t *>(curr), num_elements, &d2,
                      &n1, &n2, &bad);
      break;
    case DataType::UINT8:
      AccumulateTyped(static_cast<const uint8_t *>(prev),
                      static_cast<const uint8_t *>(curr), num_elements, &d2,
                      &n1, &n2, &bad);
      break;
    default:
      return false;
  }

  const double d = std::sqrt(d2);
  const double b1n = std::sqrt(n1);
  const double b2n = std::sqrt(n2);
  out->absolute_change = d;
  out->b1_norm = b1n;
  out->b2_norm = b2n;
  out->normalized_change = d / (b1n + b2n + epsilon);
  out->nonfinite_skipped = bad;
  out->elements_compared = num_elements - bad;
  out->status = (out->elements_compared == 0)
                    ? BlockEvolutionStatus::kAllNonFinite
                    : BlockEvolutionStatus::kOk;
  return out->status == BlockEvolutionStatus::kOk;
}

#if !defined(CTP_ENABLE_CUDA) || !CTP_ENABLE_CUDA
namespace detail {
// No CUDA in this build: the tracker still works, for host chunks only.
void *EvoDeviceAlloc(size_t) { return nullptr; }
void EvoDeviceFree(void *) {}
bool EvoDeviceCopyAsync(void *, const void *, size_t, void *) { return false; }
bool EvoDeviceSync(void *) { return true; }
}  // namespace detail

bool ComputeBlockEvolutionDevice(const void *, const void *, size_t, DataType,
                                 double, void *, BlockEvolution *out) {
  if (out) *out = BlockEvolution{};
  return false;
}
#endif  // !CTP_ENABLE_CUDA

// ---------------------------------------------------------------------------
// BlockEvolutionTracker
// ---------------------------------------------------------------------------

namespace {

/** One tracked block's retained previous timestep. At namespace scope rather
    than nested in Impl so the file-local helpers below can name it. */
struct EvoEntry {
  void *buf = nullptr;        // retained B1
  size_t bytes = 0;
  DataType type = DataType::FLOAT32;
  long long timestep = 0;     // the timestep `buf` holds
  bool on_device = false;
};

void ReleaseEntry(EvoEntry &e) {
  if (e.buf == nullptr) return;
  if (e.on_device) {
    detail::EvoDeviceFree(e.buf);
  } else {
    delete[] static_cast<uint8_t *>(e.buf);
  }
  e.buf = nullptr;
  e.bytes = 0;
}

/** Allocate `bytes` on the side `on_device` names, and copy `src` into it. */
bool RetainInto(EvoEntry &e, const void *src, size_t bytes, bool on_device,
                void *stream) {
  if (e.buf != nullptr && (e.bytes != bytes || e.on_device != on_device)) {
    ReleaseEntry(e);
  }
  if (e.buf == nullptr) {
    e.buf = on_device ? detail::EvoDeviceAlloc(bytes)
                      : static_cast<void *>(new (std::nothrow) uint8_t[bytes]);
    if (e.buf == nullptr) return false;
    e.bytes = bytes;
    e.on_device = on_device;
  }
  if (on_device) {
    // Enqueued, then drained before returning. The retained copy must be
    // complete before the caller is free to overwrite its chunk in place --
    // which every in-situ producer does on the next step.
    if (!detail::EvoDeviceCopyAsync(e.buf, src, bytes, stream)) return false;
    return detail::EvoDeviceSync(stream);
  }
  std::memcpy(e.buf, src, bytes);
  return true;
}

}  // namespace

struct BlockEvolutionTracker::Impl {
  std::unordered_map<std::string, EvoEntry> blocks;
};

BlockEvolutionTracker::BlockEvolutionTracker(int sample_interval,
                                             double epsilon,
                                             size_t capacity_bytes)
    : impl_(new Impl()),
      sample_interval_(sample_interval > 0 ? sample_interval : 1),
      epsilon_(epsilon),
      capacity_bytes_(capacity_bytes) {}

BlockEvolutionTracker::~BlockEvolutionTracker() {
  Clear();
  delete impl_;
}

void BlockEvolutionTracker::Clear() {
  for (auto &kv : impl_->blocks) ReleaseEntry(kv.second);
  impl_->blocks.clear();
  retained_bytes_ = 0;
}

size_t BlockEvolutionTracker::tracked_blocks() const {
  return impl_->blocks.size();
}

bool BlockEvolutionTracker::Observe(const std::string &block_key,
                                    long long timestep, const void *chunk,
                                    size_t chunk_bytes, DataType type,
                                    void *stream, BlockEvolution *out) {
  BlockEvolution scratch;
  if (out == nullptr) out = &scratch;
  *out = BlockEvolution{};

  if (chunk == nullptr || chunk_bytes == 0) {
    out->status = BlockEvolutionStatus::kFailed;
    return false;
  }
  const size_t type_size = DataStatisticsFactory::GetTypeSize(type);
  if (type_size == 0 || chunk_bytes < type_size) {
    out->status = BlockEvolutionStatus::kFailed;
    return false;
  }
  const size_t num_elements = chunk_bytes / type_size;

  // Sampling grid. Off-grid timesteps cost one modulo and nothing else -- in
  // particular they do NOT refresh the retained block, or the interval between
  // the two compared timesteps would not be the interval that was configured.
  if (sample_interval_ > 1 && (timestep % sample_interval_) != 0) {
    out->status = BlockEvolutionStatus::kNotSampled;
    return false;
  }

  const bool on_device = GpuApi::IsDevicePointer(chunk);
  auto it = impl_->blocks.find(block_key);

  // --- first sample for this block: retain it, report unavailable ----------
  if (it == impl_->blocks.end()) {
    if (retained_bytes_ + chunk_bytes > capacity_bytes_) {
      out->status = BlockEvolutionStatus::kFailed;
      return false;
    }
    EvoEntry e;
    if (!RetainInto(e, chunk, chunk_bytes, on_device, stream)) {
      ReleaseEntry(e);
      out->status = BlockEvolutionStatus::kFailed;
      return false;
    }
    e.type = type;
    e.timestep = timestep;
    retained_bytes_ += chunk_bytes;
    impl_->blocks.emplace(block_key, e);
    out->status = BlockEvolutionStatus::kFirstTimestep;
    return false;
  }

  EvoEntry &e = it->second;

  // --- the block changed shape, or moved between host and device -----------
  // Element i of a resized block is a different cell, so differencing the
  // overlapping prefix would compare unrelated points and report it as
  // evolution. Re-retain instead and let the next sample produce a value.
  // A residency flip lands here for the same reason: there is no previous
  // block in the memory space the current one lives in.
  if (e.bytes != chunk_bytes || e.type != type || e.on_device != on_device) {
    const size_t old_bytes = e.bytes;
    const bool resized = (e.bytes != chunk_bytes || e.type != type);
    if (!RetainInto(e, chunk, chunk_bytes, on_device, stream)) {
      retained_bytes_ -= old_bytes;
      ReleaseEntry(e);
      impl_->blocks.erase(it);
      out->status = BlockEvolutionStatus::kFailed;
      return false;
    }
    retained_bytes_ = retained_bytes_ - old_bytes + chunk_bytes;
    e.type = type;
    e.timestep = timestep;
    out->status = resized ? BlockEvolutionStatus::kSizeMismatch
                          : BlockEvolutionStatus::kFirstTimestep;
    return false;
  }

  // --- compare, then advance the retained block ----------------------------
  const long long delta_t = timestep - e.timestep;
  const bool ok = on_device
                      ? ComputeBlockEvolutionDevice(e.buf, chunk, num_elements,
                                                    type, epsilon_, stream, out)
                      : ComputeBlockEvolutionHost(e.buf, chunk, num_elements,
                                                  type, epsilon_, out);
  out->delta_t = delta_t;
  // Only meaningful across runs with different intervals; with a fixed one it
  // is normalized_change scaled by a constant.
  out->evolution_rate =
      (delta_t > 0) ? out->normalized_change / static_cast<double>(delta_t)
                    : 0.0;
  if (!ok && out->status == BlockEvolutionStatus::kOk) {
    out->status = BlockEvolutionStatus::kFailed;
  }

  // Advance regardless of whether the metric succeeded: the series must stay
  // on the sampling grid. Retaining the OLD block after a failed comparison
  // would silently widen the next delta_t to 2*interval.
  if (!RetainInto(e, chunk, chunk_bytes, on_device, stream)) {
    retained_bytes_ -= e.bytes;
    ReleaseEntry(e);
    impl_->blocks.erase(it);
    if (out->status == BlockEvolutionStatus::kOk) return true;
    out->status = BlockEvolutionStatus::kFailed;
    return false;
  }
  e.timestep = timestep;
  return out->status == BlockEvolutionStatus::kOk;
}

}  // namespace ctp
