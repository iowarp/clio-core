/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file byte_shuffle_cpu_stub.cc
 * @brief CPU-build definitions of the device byte-shuffle entry points.
 *
 * byte_shuffle.h declares ByteShuffleDevice/ByteUnshuffleDevice
 * unconditionally so call sites can branch on IsDevicePointer() without
 * #if-guards. On a build without CUDA there are no device pointers to hand
 * them, so these just report failure -- which callers must treat as "do not
 * shuffle", never as "fall back to the host routines", since those would
 * dereference whatever pointer they were given.
 */

#include "clio_ctp/compress/preprocess/byte_shuffle.h"
#include "clio_ctp/compress/preprocess/data_stats_gpu.h"
#include "clio_ctp/compress/preprocess/quantization.h"

#if !defined(CTP_ENABLE_CUDA) || !CTP_ENABLE_CUDA

namespace ctp::compress::preprocess {

bool ByteShuffleDevice(const void *, void *, size_t, size_t, void *) {
  return false;
}
bool ByteUnshuffleDevice(const void *, void *, size_t, size_t) { return false; }

/** Same contract: no device, so no device quantization. */
bool QuantizeDevice(const void *, size_t, double, void *, size_t *,
                    DeviceQuantizeParams *, void *) {
  return false;
}
bool DequantizeDevice(const void *, size_t, const DeviceQuantizeParams &,
                      void *) {
  return false;
}

}  // namespace ctp::compress::preprocess

namespace ctp {

/**
 * Same contract as the shuffle stubs above, for the same reason.
 *
 * data_stats_gpu.h documents ComputeDeviceStats as returning false "if CUDA
 * support isn't compiled in", and the inline ComputeCompressionFeatures calls
 * it unconditionally -- but its only definition lives in
 * data_stats_gpu_kernels.cu, which CMake adds solely under
 * CLIO_CORE_ENABLE_CUDA. A CPU-only build of the compressor chimod therefore
 * failed to link, and the host fallback the header describes was unreachable
 * because the device branch's callee did not exist.
 */
bool ComputeDeviceStats(const void *, size_t, DataType, double *out_entropy,
                        double *out_mad, double *out_second_derivative) {
  if (out_entropy) *out_entropy = 0.0;
  if (out_mad) *out_mad = 0.0;
  if (out_second_derivative) *out_second_derivative = 0.0;
  return false;
}

/**
 * Device-resident stats path: same contract as the shuffle stubs above. With
 * no device there is no stream and nothing to compute on it, so the null
 * return sends callers down the host feature path rather than leaving them
 * holding a pointer they cannot dereference.
 */
void *DeviceStatsStream() { return nullptr; }

const void *ComputeDeviceStatsResident(const void *, size_t, DataType, void *) {
  return nullptr;
}

bool ReadDeviceFeatureStats(const void *, double *, double *, double *,
                            void *) {
  return false;
}

}  // namespace ctp

#endif
