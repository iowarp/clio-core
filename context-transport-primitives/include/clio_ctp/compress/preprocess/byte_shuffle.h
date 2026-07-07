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
 * Reversible for elem_size in {2, 4, 8}. Header-only CPU implementation.
 */
#ifndef CLIO_CTP_COMPRESS_PREPROCESS_BYTE_SHUFFLE_H_
#define CLIO_CTP_COMPRESS_PREPROCESS_BYTE_SHUFFLE_H_

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ctp::compress::preprocess {

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

  size_t num_elements = num_bytes / elem_size;
  if (num_elements == 0) {
    return false;
  }

  // Process element by element: distribute bytes to planes
  size_t output_offset = 0;

  for (size_t byte_idx = 0; byte_idx < elem_size; ++byte_idx) {
    for (size_t elem_idx = 0; elem_idx < num_elements; ++elem_idx) {
      size_t input_offset = elem_idx * elem_size + byte_idx;
      output[output_offset] = input[input_offset];
      output_offset++;
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

  size_t num_elements = num_bytes / elem_size;
  if (num_elements == 0) {
    return false;
  }

  // Process byte plane by plane: gather bytes back to elements
  size_t input_offset = 0;

  for (size_t byte_idx = 0; byte_idx < elem_size; ++byte_idx) {
    for (size_t elem_idx = 0; elem_idx < num_elements; ++elem_idx) {
      size_t output_offset = elem_idx * elem_size + byte_idx;
      output[output_offset] = input[input_offset];
      input_offset++;
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

}  // namespace ctp::compress::preprocess

#endif  // CLIO_CTP_COMPRESS_PREPROCESS_BYTE_SHUFFLE_H_
