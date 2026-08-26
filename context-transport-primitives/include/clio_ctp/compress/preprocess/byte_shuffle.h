/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

/**
 * @file byte_shuffle.h
 * @brief Byte-plane shuffling (AoS to SoA) for compression preprocessing.
 *
 * Reorganizes data by grouping bytes at the same position across elements.
 * This improves compressibility for structured data (e.g., float arrays).
 *
 *   INPUT:  [A0 A1 A2 A3][B0 B1 B2 B3][C0 C1 C2 C3]
 *   OUTPUT: [A0 B0 C0][A1 B1 C1][A2 B2 C2][A3 B3 C3]
 *
 * The regrouping happens within each kShuffleChunkBytes block independently,
 * not across the whole buffer -- see that constant for why.
 *
 * Reversible for elem_size in {2, 4, 8}. Header-only CPU implementation.
 */
#ifndef CLIO_CTP_COMPRESS_PREPROCESS_BYTE_SHUFFLE_H_
#define CLIO_CTP_COMPRESS_PREPROCESS_BYTE_SHUFFLE_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ctp::compress::preprocess {

/**
 * @brief Bytes per independently-shuffled block.
 *
 * NeuroPress does NOT shuffle a buffer as one set of byte planes: it splits
 * the input into SHUFFLE_CHUNK_SIZE pieces and builds planes within each
 * piece (src/api/internal.hpp `constexpr size_t SHUFFLE_CHUNK_SIZE =
 * 256 * 1024`, passed at gpucompress_compress.cpp and :1260, applied per
 * chunk in byte_shuffle_kernels.cu where `num_elements = chunk_size /
 * ElementSize` is the CHUNK's element count).
 *
 * The distinction is invisible for buffers <= 256 KiB (one chunk = the whole
 * buffer) but changes every byte handed to the codec above it, and the
 * shipped model's ratio predictions for shuffle actions were learned against
 * the chunked layout. Matching the constant keeps the codec input -- and so
 * the achieved ratio -- comparable with NeuroPress's.
 */
constexpr size_t kShuffleChunkBytes = 256 * 1024;

/* ByteShuffle() / ByteUnshuffle() -- the CPU implementations -- were REMOVED.
 *
 * NeuroPress preprocessing is CUDA-only, matching upstream, whose
 * src/preprocessing/ contains nothing but byte_shuffle_kernels.cu and
 * quantization_kernels.cu. Keeping a working CPU mirror here meant a
 * host-resident chunk silently ran the transform on the CPU: same bytes out,
 * different hardware, and nothing in the results to reveal it. Their absence
 * is the guarantee -- a caller that needs the transform on host data no longer
 * compiles, rather than quietly getting it.
 *
 * Use ByteShuffleDevice() / ByteUnshuffleDevice() below. Data that is not
 * already device-resident must be refused (see
 * CLIO_NEUROPRESS_REQUIRE_DEVICE) or staged H2D by the caller first.
 */


/**
 * @brief Device byte-shuffle: device pointer in, device pointer out.
 *
 * Defined in src/compress/preprocess/byte_shuffle_gpu_kernels.cu; declared
 * unconditionally so callers can branch on IsDevicePointer() without
 * #if-guarding every call site. Returns false when CUDA support is not
 * compiled in, so a caller that must not fall back to the host routines
 * above (they would dereference the device pointer) can fail cleanly.
 *
 * Produces byte-for-byte the same layout as ByteShuffle(): the buffer is cut
 * into kShuffleChunkBytes blocks, plane `b` of a block starts at
 * `b * (block_bytes / elem_size)` within that block, and a trailing partial
 * element is copied verbatim. Blobs are interchangeable between the two
 * paths (verified by the preprocess unit tests, which cross both).
 *
 * @param device_in  Device buffer to read.
 * @param device_out Device buffer to write (at least num_bytes, no overlap).
 * @param num_bytes  Total bytes.
 * @param elem_size  Bytes per element (2, 4 or 8).
 * @param stream     cudaStream_t to launch on, as an opaque pointer, or
 *                   nullptr for the shared per-thread stream.
 *
 * When a stream is supplied the launch is ASYNCHRONOUS -- the caller is
 * responsible for ordering, which it gets for free by issuing the work that
 * consumes the output on the same stream. That is how upstream shuffles an
 * explored slot: byte_shuffle_simple(..., s.stream) followed immediately by
 * the codec launch on s.stream, with the only wait happening once, later,
 * when every slot is collected (gpucompress_compress.cpp). With nullptr the
 * call synchronizes before returning, which is what every non-sweep caller
 * expects.
 */
bool ByteShuffleDevice(const void *device_in, void *device_out,
                       size_t num_bytes, size_t elem_size,
                       void *stream = nullptr);

/** @brief Device inverse of ByteShuffleDevice(). Same contract. */
bool ByteUnshuffleDevice(const void *device_in, void *device_out,
                         size_t num_bytes, size_t elem_size);

}  // namespace ctp::compress::preprocess

#endif  // CLIO_CTP_COMPRESS_PREPROCESS_BYTE_SHUFFLE_H_
