#pragma once

// Shared chunk-geometry core. CROSS_FUN + cuda::std only, so the same code runs
// in a device kernel (Slice 2) and a host unit test. Row-major, dim 0 outermost.
// A "chunk coordinate" is the per-dimension chunk index, not the element index.

#include "defines.h"
#include <cuda/std/span>

namespace kvhdf5::chunking {

// iowarp keeps a device-touched blob name in clio::run::priv::string SSO; longer names
// spill to a device heap alloc that faults in this runtime config. Chunk-coord
// names are tiny, so this is a guard rather than a real limit.
inline constexpr size_t kMaxBlobNameLen = 31;

[[nodiscard]] CROSS_FUN inline uint64_t ChunksInDim(uint64_t dim, uint64_t chunk) {
    KVHDF5_ASSERT(chunk > 0, "chunk dim must be > 0");
    return (dim + chunk - 1) / chunk;
}

[[nodiscard]] CROSS_FUN inline uint64_t ChunkCount(cstd::span<const uint64_t> dims,
                                                   cstd::span<const uint64_t> chunk_dims) {
    KVHDF5_ASSERT(dims.size() == chunk_dims.size(), "dims/chunk_dims rank mismatch");
    uint64_t total = 1;
    for (size_t i = 0; i < dims.size(); ++i) {
        total *= ChunksInDim(dims[i], chunk_dims[i]);
    }
    return total;
}

CROSS_FUN inline void ChunkIndexToCoord(uint64_t index,
                                        cstd::span<const uint64_t> dims,
                                        cstd::span<const uint64_t> chunk_dims,
                                        cstd::span<uint64_t> out_coord) {
    KVHDF5_ASSERT(out_coord.size() == dims.size(), "out_coord rank mismatch");
    for (size_t pos = dims.size(); pos-- > 0;) {
        uint64_t n = ChunksInDim(dims[pos], chunk_dims[pos]);
        out_coord[pos] = index % n;
        index /= n;
    }
}

[[nodiscard]] CROSS_FUN inline uint64_t ChunkCoordToIndex(cstd::span<const uint64_t> coord,
                                                          cstd::span<const uint64_t> dims,
                                                          cstd::span<const uint64_t> chunk_dims) {
    KVHDF5_ASSERT(coord.size() == dims.size(), "coord rank mismatch");
    uint64_t index = 0;
    for (size_t i = 0; i < dims.size(); ++i) {
        uint64_t n = ChunksInDim(dims[i], chunk_dims[i]);
        KVHDF5_ASSERT(coord[i] < n, "chunk coord out of range");
        index = index * n + coord[i];
    }
    return index;
}

// Format a chunk coordinate as a blob name: coords joined by '_', e.g. "12_47".
// Writes a NUL-terminated string into `out` (its size must include the NUL) and
// returns a view of the name (excluding the NUL; out.data() stays a valid
// C-string). Returns an empty span on failure (didn't fit, or exceeds the device
// name limit) — unambiguous since a valid name is always >= 1 char.
[[nodiscard]] CROSS_FUN inline cstd::span<const char> ChunkCoordToName(
        cstd::span<const uint64_t> coord, cstd::span<char> out) {
    size_t cap = out.size();
    if (coord.empty() || cap == 0) return {};
    size_t len = 0;
    for (size_t i = 0; i < coord.size(); ++i) {
        if (i != 0) {
            if (len + 1 >= cap || len + 1 > kMaxBlobNameLen) return {};
            out[len++] = '_';
        }
        char digits[20];  // max decimal digits in a uint64_t
        size_t n = 0;
        uint64_t v = coord[i];
        do { digits[n++] = static_cast<char>('0' + (v % 10)); v /= 10; } while (v != 0);
        if (len + n + 1 > cap || len + n > kMaxBlobNameLen) return {};
        for (size_t j = 0; j < n; ++j) out[len++] = digits[n - 1 - j];
    }
    out[len] = '\0';
    return {out.data(), len};
}

}  // namespace kvhdf5::chunking
