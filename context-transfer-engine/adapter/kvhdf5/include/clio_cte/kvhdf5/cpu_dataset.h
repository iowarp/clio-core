#pragma once

#include "defines.h"
#include "blob_backend.h"
#include "chunking.h"
#include "tag_path.h"
#include <string>
#include <vector>
#include <cuda/std/span>
#include <cuda/std/expected>

namespace kvhdf5 {

struct Layout {
    std::vector<uint64_t> dims;
    std::vector<uint64_t> chunk_dims;
    uint64_t elem_size = 0;  // bytes per element

    [[nodiscard]] bool Valid() const {
        if (dims.empty() || dims.size() != chunk_dims.size() || dims.size() > MAX_DIMS) return false;
        if (elem_size == 0) return false;
        for (size_t i = 0; i < dims.size(); ++i)
            if (chunk_dims[i] == 0 || chunk_dims[i] > dims[i]) return false;
        return true;
    }
    [[nodiscard]] cstd::span<const uint64_t> Dims() const { return {dims.data(), dims.size()}; }
    [[nodiscard]] cstd::span<const uint64_t> ChunkDims() const { return {chunk_dims.data(), chunk_dims.size()}; }
    [[nodiscard]] uint64_t TotalElems() const { uint64_t n = 1; for (uint64_t d : dims) n *= d; return n; }
    [[nodiscard]] uint64_t TotalBytes() const { return TotalElems() * elem_size; }
    [[nodiscard]] uint64_t ChunkCount() const { return chunking::ChunkCount(Dims(), ChunkDims()); }
    [[nodiscard]] bool IsSingleChunk() const { return ChunkCount() == 1; }
};

struct DatasetMeta {
    std::string path;
    Layout layout;
    // Slice 2: cached iowarp TagId resolved once from `path` on the runtime path.

    // The dataset's CTE tag = its canonicalized path (one slash-joined tag). The
    // blob key within it is the chunk coordinate, so a chunk addresses as
    // (TagName(), ChunkCoordToName(coord)). Empty if `path` has no real segment.
    [[nodiscard]] std::string TagName() const { return tagpath::CanonicalTag(path); }
};

enum class IoError : uint8_t { BadLayout, SizeMismatch, Unsupported, NotFound, BackendFailure };

// Non-owning view: a Dataset is valid only while its originating File is alive
// and not moved (it holds borrowed DatasetMeta* and backend B* into the File).
//
// MVP: one chunk per dataset. `data`/`out` are the full row-major array. The
// multi-chunk gather/scatter is Phase 1+ (returns Unsupported until then).
template<KVHDF5_BLOB_BACKEND B>
class Dataset {
    DatasetMeta* meta_;
    B* backend_;

    static cstd::span<const char> ChunkZeroName(const Layout& L,
            char (&out)[chunking::kMaxBlobNameLen + 1]) {
        uint64_t coord[MAX_DIMS] = {};
        return chunking::ChunkCoordToName({coord, L.dims.size()}, out);
    }

public:
    Dataset(DatasetMeta* meta, B* backend) : meta_(meta), backend_(backend) {}

    [[nodiscard]] const DatasetMeta& Meta() const { return *meta_; }

    [[nodiscard]] cstd::expected<void, IoError> Write(cstd::span<const byte_t> data) {
        const Layout& L = meta_->layout;
        if (!L.Valid()) return cstd::unexpected(IoError::BadLayout);
        if (data.size() != L.TotalBytes()) return cstd::unexpected(IoError::SizeMismatch);
        if (!L.IsSingleChunk()) return cstd::unexpected(IoError::Unsupported);

        char name[chunking::kMaxBlobNameLen + 1];
        auto chunk_name = ChunkZeroName(L, name);
        if (chunk_name.empty()) return cstd::unexpected(IoError::BadLayout);

        if (!backend_->WriteChunk(chunk_name, data))
            return cstd::unexpected(IoError::BackendFailure);
        return {};
    }

    [[nodiscard]] cstd::expected<cstd::span<byte_t>, IoError> Read(cstd::span<byte_t> out) {
        const Layout& L = meta_->layout;
        if (!L.Valid()) return cstd::unexpected(IoError::BadLayout);
        if (out.size() < L.TotalBytes()) return cstd::unexpected(IoError::SizeMismatch);
        if (!L.IsSingleChunk()) return cstd::unexpected(IoError::Unsupported);

        char name[chunking::kMaxBlobNameLen + 1];
        auto chunk_name = ChunkZeroName(L, name);
        if (chunk_name.empty()) return cstd::unexpected(IoError::BadLayout);

        auto r = backend_->ReadChunk(chunk_name, out);
        if (!r) {
            IoError e = r.error() == BlobError::NotFound       ? IoError::NotFound
                      : r.error() == BlobError::NotEnoughSpace ? IoError::SizeMismatch
                                                               : IoError::BackendFailure;
            return cstd::unexpected(e);
        }
        return *r;
    }
};

}  // namespace kvhdf5
