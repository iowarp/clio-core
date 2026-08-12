#pragma once

#include "defines.h"
#include "chunking.h"
#include "meta_blob.h"  // DType (+ the __meta store format it is serialized into)
#include <cstdint>
#include <vector>
#include <cuda/std/span>

namespace kvhdf5 {

// meta_blob.h cannot include defines.h (it must stay CUDA-free; see its
// dependency contract), so it carries its own copy of the dimension cap. This is
// the one place both are visible -- pin them together.
static_assert(MAX_DIMS == kMetaMaxDims, "MAX_DIMS and kMetaMaxDims must agree");

struct Layout {
    std::vector<uint64_t> dims;
    std::vector<uint64_t> chunk_dims;
    uint64_t elem_size = 0;  // bytes per element

    // Element type. Defaults to kOpaque = "untyped bytes of width elem_size",
    // which is what every pre-dtype layout meant, so the existing
    // Layout{dims, chunk_dims, elem_size} aggregate initializations keep their
    // exact meaning. Set it when the data has a real type: an exported .h5 needs
    // one (HDF5 has no "just bytes of width 4"), and elem_size alone cannot
    // distinguish f32 from i32/u32.
    //
    // elem_size stays the single sizing input used everywhere; dtype does NOT
    // silently override it. Instead Valid() REQUIRES the two to agree whenever
    // dtype is typed, so the pair cannot drift out of sync.
    DType dtype = DType::kOpaque;

    [[nodiscard]] bool Valid() const {
        if (dims.empty() || dims.size() != chunk_dims.size() || dims.size() > MAX_DIMS) return false;
        if (elem_size == 0) return false;
        const uint64_t typed = DTypeSize(dtype);
        if (typed != 0 && typed != elem_size) return false;
        for (size_t i = 0; i < dims.size(); ++i)
            if (chunk_dims[i] == 0 || chunk_dims[i] > dims[i]) return false;
        return true;
    }

    // Serialize into the __meta store format. Precondition: Valid(). Callers on
    // the write path additionally require divisible dims (GpuCteDataset rejects
    // edge chunks), which MetaBlob::Valid() also enforces on read-back.
    [[nodiscard]] MetaBlob ToMetaBlob() const {
        MetaBlob m;
        m.rank = static_cast<uint32_t>(dims.size());
        m.dtype = static_cast<uint32_t>(dtype);
        m.elem_size = elem_size;
        for (size_t i = 0; i < dims.size(); ++i) {
            m.dims[i] = dims[i];
            m.chunk_dims[i] = chunk_dims[i];
        }
        return m;
    }
    [[nodiscard]] cstd::span<const uint64_t> Dims() const { return {dims.data(), dims.size()}; }
    [[nodiscard]] cstd::span<const uint64_t> ChunkDims() const { return {chunk_dims.data(), chunk_dims.size()}; }
    [[nodiscard]] uint64_t TotalElems() const { uint64_t n = 1; for (uint64_t d : dims) n *= d; return n; }
    [[nodiscard]] uint64_t TotalBytes() const { return TotalElems() * elem_size; }
    [[nodiscard]] uint64_t ChunkCount() const { return chunking::ChunkCount(Dims(), ChunkDims()); }
    [[nodiscard]] bool IsSingleChunk() const { return ChunkCount() == 1; }
};

}  // namespace kvhdf5
