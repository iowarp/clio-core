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

/**
 * @brief Shuffle data from AoS (Array of Structures) to SoA by byte plane.
 *
 * Reorganizes elem_size bytes of each element into separate planes.
 * Each byte position across all elements gets grouped together.
 *
 * Supported elem_size: 2, 4, 8 bytes.
 *
 * Example (elem_size=4):
 *   Input:  [A0 A1 A2 A3][B0 B1 B2 B3][C0 C1 C2 C3]
 *   Output: [A0 B0 C0] [A1 B1 C1] [A2 B2 C2] [A3 B3 C3]
 *
 * @param input      Input buffer (AoS format)
 * @param num_bytes  Total input size in bytes
 * @param elem_size  Bytes per element (must be 2, 4, or 8)
 * @param output     Output buffer (must be at least num_bytes)
 * @return true on success, false if elem_size unsupported
 */
inline bool ByteShuffle(const uint8_t* input,
                        size_t num_bytes,
                        size_t elem_size,
                        uint8_t* output) {
  if (!input || !output || num_bytes == 0) {
    return false;
  }

  if (elem_size != 2 && elem_size != 4 && elem_size != 8) {
    return false;
  }

  // A buffer shorter than one element is NOT an error: NeuroPress's
  // byte_shuffle_simple handles it as num_elements = 0, leftover = the whole
  // buffer, and copies it verbatim (byte_shuffle_kernels.cu:33-65). The loop
  // below does the same, so nothing special is needed -- but the guard that
  // used to reject it here would have made Compress() decline a shuffle
  // upstream would have performed and recorded.

  // Planes are built WITHIN each kShuffleChunkBytes block, not across the
  // whole buffer -- see the constant's docs. Blocks are independent, so the
  // inverse is the same walk.
  for (size_t base = 0; base < num_bytes; base += kShuffleChunkBytes) {
    const size_t chunk = std::min(kShuffleChunkBytes, num_bytes - base);
    const uint8_t* in = input + base;
    uint8_t* out = output + base;
    const size_t num_elements = chunk / elem_size;
    const size_t leftover = chunk % elem_size;

    for (size_t byte_idx = 0; byte_idx < elem_size; ++byte_idx) {
      for (size_t elem_idx = 0; elem_idx < num_elements; ++elem_idx) {
        out[byte_idx * num_elements + elem_idx] =
            in[elem_idx * elem_size + byte_idx];
      }
    }
    // Trailing partial element is copied verbatim, matching
    // byte_shuffle_kernels.cu:59-65. Dropping it (as this did before) left
    // the caller's tail bytes untouched -- silent corruption on round trip,
    // since the length was still right.
    for (size_t i = 0; i < leftover; ++i) {
      out[elem_size * num_elements + i] = in[elem_size * num_elements + i];
    }
  }

  return true;
}

/**
 * @brief Unshuffle data from SoA (Structure of Arrays) back to AoS.
 *
 * Reverses ByteShuffle: reconstructs AoS layout from byte planes.
 *
 * Supported elem_size: 2, 4, 8 bytes.
 *
 * Example (elem_size=4):
 *   Input:  [A0 B0 C0] [A1 B1 C1] [A2 B2 C2] [A3 B3 C3]
 *   Output: [A0 A1 A2 A3][B0 B1 B2 B3][C0 C1 C2 C3]
 *
 * @param input      Input buffer (SoA format, shuffled)
 * @param num_bytes  Total input size in bytes
 * @param elem_size  Bytes per element (must be 2, 4, or 8)
 * @param output     Output buffer (must be at least num_bytes)
 * @return true on success, false if elem_size unsupported
 */
inline bool ByteUnshuffle(const uint8_t* input,
                          size_t num_bytes,
                          size_t elem_size,
                          uint8_t* output) {
  if (!input || !output || num_bytes == 0) {
    return false;
  }

  if (elem_size != 2 && elem_size != 4 && elem_size != 8) {
    return false;
  }

  // A buffer shorter than one element is NOT an error: NeuroPress's
  // byte_shuffle_simple handles it as num_elements = 0, leftover = the whole
  // buffer, and copies it verbatim (byte_shuffle_kernels.cu:33-65). The loop
  // below does the same, so nothing special is needed -- but the guard that
  // used to reject it here would have made Compress() decline a shuffle
  // upstream would have performed and recorded.

  // Exact inverse of ByteShuffle's per-block walk above.
  for (size_t base = 0; base < num_bytes; base += kShuffleChunkBytes) {
    const size_t chunk = std::min(kShuffleChunkBytes, num_bytes - base);
    const uint8_t* in = input + base;
    uint8_t* out = output + base;
    const size_t num_elements = chunk / elem_size;
    const size_t leftover = chunk % elem_size;

    for (size_t byte_idx = 0; byte_idx < elem_size; ++byte_idx) {
      for (size_t elem_idx = 0; elem_idx < num_elements; ++elem_idx) {
        out[elem_idx * elem_size + byte_idx] =
            in[byte_idx * num_elements + elem_idx];
      }
    }
    for (size_t i = 0; i < leftover; ++i) {
      out[elem_size * num_elements + i] = in[elem_size * num_elements + i];
    }
  }

  return true;
}

/**
 * @brief Shuffle with vector allocation (convenience wrapper).
 *
 * Allocates output vector and performs ByteShuffle.
 *
 * @param input      Input buffer (AoS format)
 * @param num_bytes  Total input size in bytes
 * @param elem_size  Bytes per element (must be 2, 4, or 8)
 * @return Shuffled data vector; empty if elem_size unsupported
 */
inline std::vector<uint8_t> ByteShuffleVector(const uint8_t* input,
                                               size_t num_bytes,
                                               size_t elem_size) {
  std::vector<uint8_t> output(num_bytes);
  if (ByteShuffle(input, num_bytes, elem_size, output.data())) {
    return output;
  }
  return std::vector<uint8_t>();
}

/**
 * @brief Unshuffle with vector allocation (convenience wrapper).
 *
 * Allocates output vector and performs ByteUnshuffle.
 *
 * @param input      Input buffer (SoA format)
 * @param num_bytes  Total input size in bytes
 * @param elem_size  Bytes per element (must be 2, 4, or 8)
 * @return Unshuffled data vector; empty if elem_size unsupported
 */
inline std::vector<uint8_t> ByteUnshuffleVector(const uint8_t* input,
                                                 size_t num_bytes,
                                                 size_t elem_size) {
  std::vector<uint8_t> output(num_bytes);
  if (ByteUnshuffle(input, num_bytes, elem_size, output.data())) {
    return output;
  }
  return std::vector<uint8_t>();
}

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
 */
bool ByteShuffleDevice(const void *device_in, void *device_out,
                       size_t num_bytes, size_t elem_size);

/** @brief Device inverse of ByteShuffleDevice(). Same contract. */
bool ByteUnshuffleDevice(const void *device_in, void *device_out,
                         size_t num_bytes, size_t elem_size);

}  // namespace ctp::compress::preprocess

#endif  // CLIO_CTP_COMPRESS_PREPROCESS_BYTE_SHUFFLE_H_
