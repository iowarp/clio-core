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

#if !defined(CTP_ENABLE_CUDA) || !CTP_ENABLE_CUDA

namespace ctp::compress::preprocess {

bool ByteShuffleDevice(const void *, void *, size_t, size_t) { return false; }
bool ByteUnshuffleDevice(const void *, void *, size_t, size_t) { return false; }

}  // namespace ctp::compress::preprocess

#endif
