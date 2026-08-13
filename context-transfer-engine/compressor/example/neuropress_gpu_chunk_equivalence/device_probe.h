/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file device_probe.h
 * @brief GPU-side probes for the callback-trace harness: hashing, byte
 *        comparison, chunk generation and residency checks.
 *
 * Everything here runs ON the device. The point of the harness is to compare
 * two GPU-resident pipelines, so pulling every intermediate buffer to the host
 * to look at it would destroy the property under test. Only the compact RESULT
 * of a comparison comes back (16 bytes), and every such copy is tagged
 * TEST_HARNESS_TRANSFER in the trace so it can never be mistaken for something
 * NeuroPress or Clio did.
 *
 * Declared in a header with no CUDA, Clio or NeuroPress types in its interface
 * so it can be included from any of the three translation units.
 */
#ifndef CLIO_CTE_COMPRESSOR_EXAMPLE_NP_EQUIV_DEVICE_PROBE_H_
#define CLIO_CTE_COMPRESSOR_EXAMPLE_NP_EQUIV_DEVICE_PROBE_H_

#include <cstddef>
#include <cstdint>

namespace npeq {

/** @brief Where a pointer actually lives, as the CUDA driver reports it. */
enum class PointerKind { kUnknown, kDevice, kHost, kManaged };

/**
 * @brief Ask the driver what a pointer is.
 *
 * Used to VERIFY device residency rather than assume it -- requirement of the
 * harness is that the input chunk is proven device-resident, not asserted to
 * be. Uses cudaPointerGetAttributes, so a plain malloc'd pointer reports
 * kHost rather than failing.
 */
PointerKind ClassifyPointer(const void *ptr);

/**
 * @brief Order-independent, position-sensitive 64-bit hash of a device buffer.
 *
 * Not a cryptographic digest and not FNV: FNV is inherently sequential, and a
 * sequential hash of a multi-gigabyte device buffer would need either one
 * thread or a host copy. This mixes each 8-byte word together with its INDEX
 * through the splitmix64 finalizer and sums the results, so it is insensitive
 * to the order threads happen to accumulate in (which is what makes it
 * parallelizable) while still detecting a permutation of the data -- exactly
 * the property needed, since byte shuffle IS a permutation and a
 * permutation-blind hash would call a shuffled buffer identical to its input.
 *
 * This is the FAST DIAGNOSTIC only. Equivalence assertions use
 * CompareDeviceBuffers() below, which is byte-for-byte.
 *
 * @param device_ptr Device buffer.
 * @param bytes Length.
 * @param out_hash Receives the hash.
 * @return false if a CUDA step failed.
 */
bool HashDeviceBuffer(const void *device_ptr, size_t bytes, uint64_t *out_hash);

/** @brief Byte-level comparison result, as required by the harness report. */
struct ByteCompareResult {
  size_t total_bytes = 0;
  size_t identical_bytes = 0;
  size_t differing_bytes = 0;
  /** Index of the first differing byte, or SIZE_MAX when none differ. */
  size_t first_differing_byte = SIZE_MAX;
  /** first_differing_byte / element_size, or SIZE_MAX when none differ. */
  size_t first_differing_element = SIZE_MAX;
  bool valid = false;

  bool Identical() const { return valid && differing_bytes == 0; }
};

/**
 * @brief Byte-for-byte comparison of two DEVICE buffers, performed on-device.
 *
 * The whole comparison happens in a kernel; only the 24-byte result struct is
 * copied to the host. That copy is the one host transfer this function makes
 * and it is a TEST_HARNESS_TRANSFER.
 *
 * @param element_size Used only to report first_differing_element.
 */
ByteCompareResult CompareDeviceBuffers(const void *a, const void *b,
                                       size_t bytes, size_t element_size);

/**
 * @brief Min and max of a device float32 buffer, computed on-device.
 *
 * Used to size a per-chunk error bound relative to the data's own range. The
 * bound handed to BOTH implementations is still an ABSOLUTE one -- upstream's
 * bound is absolute (and D16-1 was exactly a relative/absolute mix-up), so the
 * range is used only to CHOOSE the number, never to change its meaning.
 */
bool ComputeDeviceRange(const float *device_ptr, size_t num_elements,
                        double *out_min, double *out_max);

/**
 * @brief Copy a small window of a device buffer to the host.
 *
 * For showing the reader the actual differing bytes when a comparison fails --
 * "4 bytes differ at offset 7" is a fact, but it is not yet a diagnosis. Bounded
 * to a handful of bytes and only used on failure, so it never becomes a way of
 * staging the payload. A TEST_HARNESS_TRANSFER.
 */
bool FetchDeviceWindow(const void *device_ptr, size_t offset, size_t bytes,
                       void *host_out);

/**
 * @brief Count elements where |a - b| exceeds an absolute bound, on-device.
 *
 * Used to check the error-bound guarantee after a lossy round trip on both
 * sides, without staging either buffer to the host.
 */
bool CountBoundViolations(const float *a, const float *b, size_t num_elements,
                          double bound, unsigned long long *out_violations);

/**
 * @brief The data regime a generated chunk should follow.
 *
 * Chosen to drive the selector into materially different corners of its action
 * space: the harness needs chunks where shuffle is picked and where it is not,
 * where quantization is picked and where it is not, and chunks that land near
 * the error-bound boundary. What each regime actually produces is measured,
 * not assumed -- the trace reports which action was really selected.
 */
enum class ChunkRegime {
  kConstant,       /**< One repeated value. Maximally compressible. */
  kLinearRamp,     /**< Smooth ramp: tiny second derivative. */
  kSmoothWave,     /**< Low-entropy sinusoid. */
  kStepped,        /**< Piecewise-constant plateaus. */
  kNoisyWave,      /**< Sinusoid plus deterministic noise. */
  kHighEntropy,    /**< Pseudo-random bits: essentially incompressible. */
  kSmallMagnitude, /**< Values ~1e-6, to sit near a typical error bound. */
  kMixed,          /**< Alternating smooth/noisy bands within one chunk. */
};

const char *ChunkRegimeName(ChunkRegime regime);

/**
 * @brief Fill a device float32 buffer with a regime, deterministically.
 *
 * Generated ON the device, so the chunk is GPU-resident from the moment it
 * exists and never had a host original. Deterministic in the element index
 * alone, so the same chunk id always yields the same bytes on both sides and
 * across runs.
 */
bool FillChunk(float *device_buf, size_t num_elements, ChunkRegime regime,
               uint32_t seed);

}  // namespace npeq

#endif  // CLIO_CTE_COMPRESSOR_EXAMPLE_NP_EQUIV_DEVICE_PROBE_H_
