/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#ifndef CLIO_CTP_BLOCK_EVOLUTION_GPU_H
#define CLIO_CTP_BLOCK_EVOLUTION_GPU_H

#include <cstddef>
#include <cstdint>
#include <string>

#include "clio_ctp/compress/preprocess/data_stats.h"
#include "clio_ctp/util/gpu_api.h"

namespace ctp {

/**
 * Temporal evolution of one block between two SAMPLED timesteps.
 *
 *     D(B1,B2) = ||B2 - B1||_2                        (absolute change)
 *     E(B1,B2) = D / (||B1||_2 + ||B2||_2 + epsilon)   (normalized, PRIMARY)
 *
 * One value per block, not one per cell. E is dimensionless and bounded in
 * [0,1] for finite inputs -- the triangle inequality gives
 * ||B2-B1|| <= ||B1||+||B2|| -- which is what makes it comparable across
 * blocks of different magnitude, and across fields with different units.
 * D is kept alongside it because E alone cannot distinguish a quiet block
 * from a small one.
 *
 * Companion to DeviceFeatureStats (data_stats_gpu.h): that describes ONE
 * chunk at ONE timestep (entropy/MAD/second derivative), this describes the
 * SAME chunk across two. The two are meant to be recorded together and
 * correlated against what the compressor then did with the chunk.
 *
 * CAVEAT FOR PARTICLE POSITIONS -- read this before correlating anything.
 * The metric measures what is IN the buffer, which for a periodic MD code is
 * not the same as how far the atoms moved. LAMMPS re-images atoms back into
 * the box on a neighbor rebuild, so a wrapped coordinate jumps by a full box
 * length while the atom itself barely moved. Measured on stock melt (32k
 * atoms, 20^3 fcc, rebuild every 20 steps), block 0, interval 10:
 *
 *     interval   x y z (wrapped)   xu yu zu (unwrapped)
 *      0 -> 10       0.002679            0.002679
 *     10 -> 20       0.199549            0.001914     <-- rebuild at t=20
 *     20 -> 30       0.001864            0.002018
 *     30 -> 40       0.116988            0.002008     <-- rebuild at t=40
 *
 * A ~100x swing that tracks the neighbor-rebuild cadence, not the physics.
 * Feed UNWRAPPED coordinates for particle data, or the evolution score will
 * correlate with the rebuild schedule and nothing else. Mesh/field codes
 * (Nyx, VPIC, Gray-Scott) have no equivalent hazard.
 *
 * CAVEAT FOR ZERO-BACKGROUND FIELDS -- why D is not optional.
 * E is a RELATIVE measure, so a block whose background is exactly zero
 * saturates at E = 1 the instant anything at all appears in it, however
 * small. Nyx Sedov momentum, step 190->200, over the 53 blocks with E > 0.1:
 *
 *     blk94  E = 1.000   D = 2.02e-20      <-- numerically nothing happened
 *     blk59  E = 0.207   D = 5.90e+01      <-- the actual shock front
 *
 * The ranking by E is very nearly the REVERSE of the ranking by real change,
 * because ||B1|| = 0 for gas at rest makes D/(0+||B2||) identically 1. Six of
 * those 53 blocks had D more than 1e6x below the loudest. Always read
 * absolute_change alongside normalized_change on a field whose background is
 * zero (momenta, velocities, reaction products); on a field with a nonzero
 * uniform background (density, temperature) E behaves.
 *
 * CAVEAT FOR LARGE-DC-OFFSET FIELDS.
 * The mirror image: when a field has a large non-zero mean, ||B|| is
 * dominated by the offset rather than the fluctuation, so E measures WHERE
 * the block is more than how much it changed. LAMMPS unwrapped positions,
 * t=200->210: D is constant to 1.2% across blocks (7.588..7.676) while E
 * falls 0.00199 -> 0.00135 purely because mean(B1) rises 11.4 -> 20.4.
 */
enum class BlockEvolutionStatus : int {
  /** Both timesteps present, sizes matched, at least one finite pair. */
  kOk = 0,
  /** This timestep is not on the sampling grid; nothing was computed. */
  kNotSampled,
  /** First sample for this block -- no previous timestep to compare against.
      The block was retained; the NEXT sample produces a value. */
  kFirstTimestep,
  /** The block changed size between the two sampled timesteps. Deliberately
      NOT computed over the overlapping prefix: element i of a resized block
      is a different cell, so the "difference" would be between unrelated
      points. The new size is retained and the next sample recovers. */
  kSizeMismatch,
  /** Every element pair contained a NaN or an Inf, so no finite pair was
      left to reduce over. */
  kAllNonFinite,
  /** No CUDA support compiled in, no device, or a launch/allocation failed. */
  kFailed,
};

struct BlockEvolution {
  /** D = ||B2-B1||_2. Absolute, in the units of the field. */
  double absolute_change = 0.0;
  /** E = D / (||B1|| + ||B2|| + eps). THE primary metric. */
  double normalized_change = 0.0;
  double b1_norm = 0.0;
  double b2_norm = 0.0;
  /** E / delta_t. Only meaningful when comparing runs whose sampling
      intervals differ; with a fixed interval it is E rescaled by a constant
      and normalized_change is the value to use. */
  double evolution_rate = 0.0;
  /** Timesteps between the two samples (current - previous SAMPLED). */
  long long delta_t = 0;
  /** Element pairs that actually entered the reduction. */
  unsigned long long elements_compared = 0;
  /** Element pairs skipped because either side was NaN or Inf. Nonzero here
      means absolute_change is a partial norm -- report it, do not hide it. */
  unsigned long long nonfinite_skipped = 0;
  BlockEvolutionStatus status = BlockEvolutionStatus::kFailed;

  bool ok() const { return status == BlockEvolutionStatus::kOk; }
};

/**
 * Default epsilon.
 *
 * Its ONLY job is to turn 0/0 into 0 for a block that was zero and stayed
 * zero. It is deliberately far smaller than a "small number" instinct
 * suggests, because epsilon does not just guard the singular case -- it sits
 * in the denominator of every block, and any block whose norms are comparable
 * to it is reported as having barely evolved no matter how much it actually
 * changed. At 1e-12, a field of 1e-15-scale values that DOUBLED between
 * timesteps comes back as E ~ 1e-3, i.e. "quiet", which is exactly the
 * small-valued-block failure this metric is supposed to survive.
 *
 * At 1e-30 the guard still removes 0/0, while every field whose values are
 * representable in float32 with room to spare (|x| >~ 1e-20, so norms >~
 * 1e-18) passes through unperturbed to better than one part in 1e12. Blocks
 * genuinely sitting at float32's denormal floor are damped toward 0, which is
 * the right reading for numerical noise.
 */
constexpr double kBlockEvolutionEpsilon = 1e-30;

/**
 * The metric, computed entirely on the GPU from two buffers that are ALREADY
 * device-resident. Nothing is staged through host memory: one fused
 * grid-stride pass reduces all three sums (sum (b2-b1)^2, sum b1^2,
 * sum b2^2) at once, a second one-thread kernel takes the square roots and
 * forms the ratio, and the only transfer is the 48-byte result struct.
 *
 * Reading each element once for all three accumulators is the entire reason
 * this is one kernel rather than three: the reduction is memory-bound, so
 * three separate passes would cost three times the bandwidth to produce
 * numbers that a single pass already has in registers.
 *
 * @param d_prev  B1, device-resident, `num_elements` of `type`
 * @param d_curr  B2, device-resident, same count and type
 * @param stream  CUDA stream to enqueue on; nullptr uses this thread's own
 *                (mirrors DeviceStatsStream()'s per-thread scoping). The
 *                stream IS synchronized before returning, because the caller
 *                needs the scalars.
 * @return false on launch/allocation failure or a build without CUDA; `out`
 *         then carries status kFailed.
 */
bool ComputeBlockEvolutionDevice(const void *d_prev, const void *d_curr,
                                 size_t num_elements, DataType type,
                                 double epsilon, void *stream,
                                 BlockEvolution *out);

/* ComputeBlockEvolutionHost() was REMOVED -- this metric is CUDA-only, like
   the byte shuffle and the quantizer. See block_evolution.cc. */


/**
 * Retains the previous SAMPLED timestep per block, so the kernel above has a
 * B1 to compare against.
 *
 * This retention is structural, not an optimization that could be removed:
 * every in-situ producer Clio integrates with (LAMMPS's Kokkos views, Nyx's
 * AMReX fabs, VPIC's field arrays) advances its state IN PLACE, so by the
 * time step t+dt is handed over, step t no longer exists anywhere. One
 * device-to-device copy per sampled block per sampled timestep is therefore
 * the floor, and it is a copy into a buffer that is allocated once per block
 * and reused -- not a per-timestep temporary.
 *
 * COST, stated plainly: steady-state device memory grows by the total size of
 * the tracked blocks (LAMMPS box 40, 100 chunks x 61,440 B = 6.1 MB; scale
 * with the field, not with the number of timesteps). `capacity_bytes` bounds
 * it; blocks beyond the bound are not tracked rather than silently evicted
 * mid-series, since an evicted block would report kFirstTimestep again and
 * quietly drop a sample out of the series.
 *
 * Thread-safety: none. One tracker per producing thread, or hold the caller's
 * own lock -- the compressor's per-chunk path is already on worker threads
 * and adding a mutex here would serialize them.
 */
/**
 * Thin device-memory hooks the tracker uses to retain previous blocks.
 *
 * They exist so BlockEvolutionTracker itself stays plain C++ in a .cc that
 * compiles with or without CUDA -- the same split byte_shuffle_cpu_stub.cc
 * uses for ByteShuffleDevice. With CUDA they are defined in
 * block_evolution_gpu_kernels.cu; without it, in block_evolution.cc, where
 * they fail cleanly and the tracker keeps working for host chunks.
 *
 * Failure is reported, never fatal: a tracker that cannot allocate simply
 * stops tracking that block instead of aborting a simulation over a metric.
 */
namespace detail {
void *EvoDeviceAlloc(size_t bytes);
void EvoDeviceFree(void *ptr);
/** Enqueue a device-to-device copy on `stream` (nullptr = per-thread). */
bool EvoDeviceCopyAsync(void *dst, const void *src, size_t bytes,
                        void *stream);
/** Block until `stream` (nullptr = per-thread) has drained. */
bool EvoDeviceSync(void *stream);
}  // namespace detail

class BlockEvolutionTracker {
 public:
  /**
   * @param sample_interval Compare every Nth timestep: with 10, timesteps
   *   0,10,20,... are sampled and 0-vs-10, 10-vs-20, ... are the comparisons.
   *   Timesteps off the grid return kNotSampled and cost nothing. 1 samples
   *   every timestep. Held FIXED for a run so the scores are comparable.
   * @param epsilon Denominator guard; see kBlockEvolutionEpsilon.
   * @param capacity_bytes Ceiling on retained previous-timestep bytes.
   */
  explicit BlockEvolutionTracker(int sample_interval = 1,
                                 double epsilon = kBlockEvolutionEpsilon,
                                 size_t capacity_bytes = (size_t)1 << 30);
  ~BlockEvolutionTracker();

  BlockEvolutionTracker(const BlockEvolutionTracker &) = delete;
  BlockEvolutionTracker &operator=(const BlockEvolutionTracker &) = delete;

  /**
   * Offer one block at one timestep.
   *
   * Dispatches on where `chunk` lives (IsDevicePointer), so a device chunk is
   * measured device-side and never touches host memory. The retained copy is
   * updated AFTER the comparison, on the same stream, so no synchronization
   * is needed between the two.
   *
   * @param block_key Identity of the block ACROSS timesteps -- the blob name
   *   with the step component removed (e.g. "position/chunk_37"). Two
   *   different blocks sharing a key would be differenced against each other.
   * @param timestep  Simulation timestep. Used both for the sampling grid and
   *   for delta_t; need not be contiguous.
   * @param out Always written, including for the skip statuses.
   * @return true only when a comparable value was produced (status kOk).
   */
  bool Observe(const std::string &block_key, long long timestep,
               const void *chunk, size_t chunk_bytes, DataType type,
               void *stream, BlockEvolution *out);

  /** Drop all retained blocks and free their device/host buffers. */
  void Clear();

  /** Currently retained previous-timestep bytes. */
  size_t retained_bytes() const { return retained_bytes_; }
  /** Blocks being tracked. */
  size_t tracked_blocks() const;
  int sample_interval() const { return sample_interval_; }

 private:
  struct Impl;
  Impl *impl_;
  int sample_interval_;
  double epsilon_;
  size_t capacity_bytes_;
  size_t retained_bytes_ = 0;
};

}  // namespace ctp

#endif  // CLIO_CTP_BLOCK_EVOLUTION_GPU_H
