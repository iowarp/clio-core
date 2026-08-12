#pragma once

// The kvhdf5 dataset self-description: the `__meta` blob.
//
// A dataset's chunks live in the KV store as raw bytes under (tag, chunk-coord)
// -- see tag_path.h. That is all that lands: no dims, no chunk geometry, no
// element type. From the store alone a reader can recover the rank and the
// per-dim chunk COUNTS (from the coord names) and the uniform chunk byte size,
// but not dims/chunk_dims/elem_size themselves -- chunk_bytes =
// prod(chunk_dims) * elem_size is one equation in many unknowns. So a dataset
// is NOT reconstructible without out-of-band information.
//
// This header defines that information and where it lives: one extra blob,
// named kMetaBlobName, inside the dataset's OWN tag, holding a fixed-size POD
// describing the layout. Colocating it with the chunks (rather than in some
// per-"file" object) is deliberate:
//
//   - There is no file object to put it in. A path is flattened into ONE tag
//     string; the hierarchy is just '/' characters in the tag name.
//   - It shares the chunks' durability. CTE blobs sit on bdev targets of
//     varying persistence_level_ (the kRam bdev is volatile). Metadata in the
//     same tag rides the same tier and shares the same fate -- it can neither
//     outlive the data it describes nor die before it.
//   - It has exactly one writer. A shared per-file blob would be a
//     read-modify-write race between every dataset in that file.
//   - DelTag removes it with the chunks. No orphan-cleanup path to write.
//
// Store-format note: this is an INTERNAL format, not a portable artifact (the
// exported .h5 is the portable artifact). It is a fixed-layout POD read back on
// the same machine that wrote it, so it does not byte-swap; `magic` + `version`
// exist so a future reader can reject what it does not understand rather than
// silently misparse it.
//
// DEPENDENCY CONTRACT -- this header is a LEAF: <cstdint>/<cstddef>/<cstring>/
// <string> and nothing else. In particular it must never include defines.h (or
// any other kvhdf5 header), because defines.h pulls in <cuda/std/*> and this
// header is included by the CPU-only CAE runtime, which must not acquire a CUDA
// include dependency to read an exported dataset's shape. Keep it that way.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

namespace kvhdf5 {

// Reserved blob name holding the MetaBlob inside a dataset's tag.
//
// COLLISION-SAFE BY CONSTRUCTION, not by luck: chunk blob names are produced
// only by chunking::ChunkCoordToName, which emits exclusively decimal digits
// separated by '_'. A name containing any other character is therefore
// unreachable by the chunk-name generator, so '_'-prefixed-with-letters can
// never collide with a chunk. IsChunkBlobName() below is the matching filter.
// Also comfortably under chunking::kMaxBlobNameLen (31).
inline constexpr const char* kMetaBlobName = "__meta";

// Dimension cap. Mirrors MAX_DIMS in defines.h, which this header cannot
// include (see the dependency contract above); layout.h static_asserts that the
// two agree, so they cannot drift.
inline constexpr size_t kMetaMaxDims = 8;

// Longest chunk-coordinate blob name. Mirrors chunking::kMaxBlobNameLen (31),
// which lives in the CUDA header this leaf cannot include. ChunkCoordToName
// below refuses to emit a name past this, exactly as chunking::ChunkCoordToName
// does, so the two agree on which coordinates are nameable.
inline constexpr size_t kMetaMaxBlobNameLen = 31;

// Element type of a dataset.
//
// Deliberately HDF5-AGNOSTIC. kvhdf5 does not link libhdf5 and must not start:
// it is an HDF5-*like* API over the KV store, and that independence is what
// lets it be compiled into device code. The enum->H5T_NATIVE_* mapping is owned
// by the exporter, which is the only component that needs HDF5.
//
// Values are explicit and MUST remain stable: they are serialized into the
// __meta blob. Append new types; never renumber existing ones.
enum class DType : uint32_t {
    kOpaque = 0,  // untyped bytes; elem_size carries the (arbitrary) width
    kI8     = 1,
    kU8     = 2,
    kI16    = 3,
    kU16    = 4,
    kI32    = 5,
    kU32    = 6,
    kI64    = 7,
    kU64    = 8,
    kF32    = 9,
    kF64    = 10,
};

// Byte width of a typed DType. Returns 0 for kOpaque (whose width is not
// implied by the type -- the Layout's elem_size carries it) and for any
// unrecognized value, so 0 doubles as the "don't know" answer.
[[nodiscard]] inline uint64_t DTypeSize(DType t) {
    switch (t) {
        case DType::kI8:
        case DType::kU8:   return 1;
        case DType::kI16:
        case DType::kU16:  return 2;
        case DType::kI32:
        case DType::kU32:
        case DType::kF32:  return 4;
        case DType::kI64:
        case DType::kU64:
        case DType::kF64:  return 8;
        case DType::kOpaque:
        default:           return 0;
    }
}

[[nodiscard]] inline bool IsKnownDType(uint32_t raw) {
    return raw <= static_cast<uint32_t>(DType::kF64);
}

inline constexpr uint32_t kMetaMagic   = 0x4B564835u;  // "KVH5"
inline constexpr uint32_t kMetaVersion = 1u;

// The serialized dataset description. Fixed size, no padding surprises (all
// members are naturally aligned and the trailing arrays are u64), so the wire
// form is just the object's bytes.
struct MetaBlob {
    uint32_t magic     = kMetaMagic;
    uint32_t version   = kMetaVersion;
    uint32_t rank      = 0;
    uint32_t dtype     = static_cast<uint32_t>(DType::kOpaque);
    uint64_t elem_size = 0;
    uint64_t dims[kMetaMaxDims]       = {};
    uint64_t chunk_dims[kMetaMaxDims] = {};

    [[nodiscard]] DType Dtype() const { return static_cast<DType>(dtype); }

    // Structural validity of a DECODED blob. Enforces the same invariant
    // Layout::Valid() does -- a typed dtype must agree with elem_size -- so a
    // corrupt or hand-rolled blob cannot smuggle in an inconsistent width.
    [[nodiscard]] bool Valid() const {
        if (magic != kMetaMagic || version != kMetaVersion) return false;
        if (rank == 0 || rank > kMetaMaxDims) return false;
        if (elem_size == 0) return false;
        if (!IsKnownDType(dtype)) return false;
        const uint64_t typed = DTypeSize(Dtype());
        if (typed != 0 && typed != elem_size) return false;
        for (uint32_t i = 0; i < rank; ++i) {
            if (dims[i] == 0 || chunk_dims[i] == 0) return false;
            if (chunk_dims[i] > dims[i]) return false;
            // Edge chunks are unsupported end to end: GpuCteDataset refuses to
            // WRITE a non-divisible layout, so a __meta blob describing one
            // could only come from corruption or a hand-rolled writer. Reject
            // it here rather than let the exporter emit a truncated dataset.
            if (dims[i] % chunk_dims[i] != 0) return false;
        }
        return true;
    }

    [[nodiscard]] uint64_t ChunkCount() const {
        uint64_t n = 1;
        for (uint32_t i = 0; i < rank; ++i) n *= dims[i] / chunk_dims[i];
        return n;
    }

    [[nodiscard]] uint64_t ChunkBytes() const {
        uint64_t n = elem_size;
        for (uint32_t i = 0; i < rank; ++i) n *= chunk_dims[i];
        return n;
    }

    // Row-major linear index of a chunk COORDINATE, in [0, ChunkCount()).
    // Mixed-radix over the per-dim chunk counts (dims[i] / chunk_dims[i]), the
    // same ordering chunking::ChunkCoordToIndex uses. Precondition: `coord` has
    // `rank` entries, each in range (the exporter bounds-checks before calling).
    // Kept here rather than reusing chunking.h so this stays a CUDA-free leaf.
    [[nodiscard]] uint64_t ChunkLinearIndex(const uint64_t* coord) const {
        uint64_t idx = 0;
        for (uint32_t i = 0; i < rank; ++i) {
            const uint64_t n = dims[i] / chunk_dims[i];
            idx = idx * n + coord[i];
        }
        return idx;
    }

    // Inverse of ChunkLinearIndex: recover the chunk COORDINATE from its
    // row-major linear index. Walks dims back-to-front because the last dim is
    // the least significant digit of the mixed-radix index. Precondition: `idx`
    // is in [0, ChunkCount()) and `out_coord` has `rank` slots.
    void ChunkCoordFromLinearIndex(uint64_t idx, uint64_t* out_coord) const {
        for (uint32_t i = rank; i-- > 0;) {
            const uint64_t n = dims[i] / chunk_dims[i];
            out_coord[i] = idx % n;
            idx /= n;
        }
    }
};

static_assert(sizeof(MetaBlob) == 24 + 2 * 8 * kMetaMaxDims,
              "MetaBlob must be tightly packed: its byte image IS the wire format");

// Encode into `out` (must be at least sizeof(MetaBlob) bytes).
inline void EncodeMeta(const MetaBlob& m, void* out) {
    std::memcpy(out, &m, sizeof(MetaBlob));
}

// Decode from `in`. Returns false (leaving `out` untouched) if the buffer is
// the wrong size or the contents don't validate -- so a caller that gets `true`
// holds a structurally sound description.
[[nodiscard]] inline bool DecodeMeta(const void* in, size_t len, MetaBlob* out) {
    if (in == nullptr || out == nullptr || len != sizeof(MetaBlob)) return false;
    MetaBlob tmp;
    std::memcpy(&tmp, in, sizeof(MetaBlob));
    if (!tmp.Valid()) return false;
    *out = tmp;
    return true;
}

// True if `name` is a chunk-coordinate blob name (the form ChunkCoordToName
// emits): one or more '_'-separated runs of decimal digits, e.g. "0", "12_47".
// This is the filter that keeps kMetaBlobName -- and any future reserved key --
// out of a chunk enumeration. Written as a whitelist of the generator's output
// rather than a blacklist of known reserved names, so adding a reserved name
// later cannot silently start leaking it into chunk iteration.
[[nodiscard]] inline bool IsChunkBlobName(const std::string& name) {
    if (name.empty()) return false;
    bool digit_in_field = false;
    for (char c : name) {
        if (c == '_') {
            if (!digit_in_field) return false;  // empty field: leading/trailing/dup '_'
            digit_in_field = false;
        } else if (c >= '0' && c <= '9') {
            digit_in_field = true;
        } else {
            return false;
        }
    }
    return digit_in_field;  // must not end on '_'
}

// Emit the chunk-coordinate blob name for `coord` (a `rank`-entry coordinate)
// into `*out`. A CUDA-free twin of chunking::ChunkCoordToName producing the
// IDENTICAL string -- decimal fields in row-major order joined by '_', no
// leading zeros -- so a name minted here addresses the same blob the device
// generator would. Returns false (leaving `*out` untouched) on a bad argument
// or if the name would exceed kMetaMaxBlobNameLen, exactly as the device twin
// refuses it. The inverse of ParseChunkName.
[[nodiscard]] inline bool ChunkCoordToName(const uint64_t* coord, uint32_t rank,
                                           std::string* out) {
    if (coord == nullptr || out == nullptr || rank == 0 || rank > kMetaMaxDims) {
        return false;
    }
    std::string name;
    for (uint32_t i = 0; i < rank; ++i) {
        if (i != 0) name.push_back('_');
        name += std::to_string(coord[i]);
    }
    if (name.size() > kMetaMaxBlobNameLen) return false;
    *out = std::move(name);
    return true;
}

// Parse a chunk-coordinate blob name into `out_coord` (at least `rank` slots).
// The inverse of chunking::ChunkCoordToName. Returns false unless the name is a
// well-formed chunk name of exactly `rank` fields, each of which fits in u64.
[[nodiscard]] inline bool ParseChunkName(const std::string& name, uint32_t rank,
                                         uint64_t* out_coord) {
    if (!IsChunkBlobName(name) || rank == 0 || rank > kMetaMaxDims) return false;
    uint32_t field = 0;
    uint64_t acc = 0;
    bool overflow = false;
    for (size_t i = 0; i <= name.size(); ++i) {
        if (i == name.size() || name[i] == '_') {
            if (field >= rank || overflow) return false;
            out_coord[field++] = acc;
            acc = 0;
            continue;
        }
        const uint64_t d = static_cast<uint64_t>(name[i] - '0');
        if (acc > (UINT64_MAX - d) / 10) { overflow = true; continue; }
        acc = acc * 10 + d;
    }
    return field == rank;
}

}  // namespace kvhdf5
