/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef CLIO_BDEV_FS_STRIPE_LAYOUT_H_
#define CLIO_BDEV_FS_STRIPE_LAYOUT_H_

#include <clio_runtime/types.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

namespace clio::run::bdev {

/**
 * Where one logical byte of a striped backing store lives.
 */
struct StripeLocation {
  size_t file_index;            /**< stripe file holding the byte, in [0, N) */
  clio::run::u64 file_offset;   /**< byte offset of the byte inside that file */
};

/**
 * A piece of a logical extent that lies entirely inside one stripe file.
 */
struct StripePiece {
  size_t file_index;            /**< stripe file, in [0, N) */
  clio::run::u64 file_offset;   /**< offset of the piece inside that file */
  clio::run::u64 logical_offset;  /**< offset of the piece in logical space */
  clio::run::u64 length;        /**< piece length in bytes (> 0) */
};

/**
 * Round-robin striping of a logical block address space over N files.
 *
 * Logical offset L falls in stripe s = L / unit. Stripe s lives in file
 * s % N at file offset (s / N) * unit + L % unit. With N == 1 this is the
 * identity mapping (file 0, offset L), so a single-file device is unchanged.
 * Pure arithmetic: no I/O, safe to call from any thread.
 */
struct StripeLayout {
  /** Number of backing files, in [1, 64]. 1 means the classic single file. */
  clio::run::u32 num_files = 1;
  /** Stripe unit in bytes (> 0). Only meaningful when num_files > 1. */
  clio::run::u64 unit = clio::run::u64(1) << 20;

  /**
   * @return true when the layout spreads data over more than one file
   */
  bool IsStriped() const { return num_files > 1; }

  /**
   * Map a logical offset to the stripe file and offset that hold it.
   * @param logical logical byte offset
   * @return the file index and the offset inside that file
   */
  StripeLocation Locate(clio::run::u64 logical) const {
    if (!IsStriped()) return StripeLocation{0, logical};
    const clio::run::u64 stripe = logical / unit;
    const clio::run::u64 within = logical % unit;
    return StripeLocation{
        static_cast<size_t>(stripe % num_files),
        (stripe / num_files) * unit + within};
  }

  /**
   * Bytes from a logical offset up to the next stripe boundary. A piece of at
   * most this length never crosses into another stripe file.
   * @param logical logical byte offset
   * @return distance to the next multiple of unit; the maximum u64 when the
   *         layout is not striped (no boundary exists)
   */
  clio::run::u64 BytesToBoundary(clio::run::u64 logical) const {
    if (!IsStriped()) return std::numeric_limits<clio::run::u64>::max();
    return unit - logical % unit;
  }

  /**
   * Split the logical extent [logical, logical + length) into pieces that each
   * lie inside a single stripe file, in increasing logical order.
   * @param logical start of the extent
   * @param length extent length in bytes
   * @return the pieces; their lengths sum to `length` (empty when length is 0)
   */
  std::vector<StripePiece> Split(clio::run::u64 logical,
                                 clio::run::u64 length) const {
    std::vector<StripePiece> pieces;
    clio::run::u64 done = 0;
    while (done < length) {
      const clio::run::u64 at = logical + done;
      const clio::run::u64 len =
          std::min(length - done, BytesToBoundary(at));
      const StripeLocation loc = Locate(at);
      pieces.push_back(StripePiece{loc.file_index, loc.file_offset, at, len});
      done += len;
    }
    return pieces;
  }

  /**
   * Size every stripe file must have so that a logical prefix of `prefix`
   * bytes is fully backed. Each file holds ceil(ceil(prefix / unit) / N)
   * stripes, so the per-file size is ceil(prefix / (N * unit)) * unit.
   * @param prefix logical backed prefix in bytes
   * @return the per-file length in bytes (`prefix` itself when not striped)
   */
  clio::run::u64 PerFileBytes(clio::run::u64 prefix) const {
    if (!IsStriped()) return prefix;
    const clio::run::u64 row = static_cast<clio::run::u64>(num_files) * unit;
    return (prefix + row - 1) / row * unit;
  }

  /**
   * Largest logical prefix whose every stripe file is at least as long as it
   * needs, given the smallest existing stripe file. Only whole rows of N
   * stripes are claimed, so no file is asked to shrink to back the prefix.
   * @param min_file_size smallest size among the stripe files, in bytes
   * @return the logical backed prefix in bytes (`min_file_size` when not
   *         striped)
   */
  clio::run::u64 LogicalPrefixFromFileSize(clio::run::u64 min_file_size) const {
    if (!IsStriped()) return min_file_size;
    return (min_file_size / unit) * static_cast<clio::run::u64>(num_files) *
           unit;
  }
};

}  // namespace clio::run::bdev

#endif  // CLIO_BDEV_FS_STRIPE_LAYOUT_H_
