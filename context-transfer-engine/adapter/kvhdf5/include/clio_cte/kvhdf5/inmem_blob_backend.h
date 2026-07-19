#pragma once

#include "defines.h"
#include "blob_backend.h"
#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

namespace kvhdf5 {

// Host-only, runtime-free chunk store for CPU unit tests. Drives the same
// Dataset<B> read/write logic the GPU backend will, without the iowarp runtime.
class InMemBlobBackend {
    std::unordered_map<std::string, std::vector<byte_t>> blobs_;

    static std::string Key(cstd::span<const char> name) {
        return std::string(name.data(), name.size());
    }

public:
    bool WriteChunk(cstd::span<const char> name, cstd::span<const byte_t> data) {
        blobs_[Key(name)].assign(data.begin(), data.end());
        return true;
    }

    [[nodiscard]] cstd::expected<cstd::span<byte_t>, BlobError> ReadChunk(
            cstd::span<const char> name, cstd::span<byte_t> out) {
        auto it = blobs_.find(Key(name));
        if (it == blobs_.end()) return cstd::unexpected(BlobError::NotFound);
        const auto& v = it->second;
        if (out.size() < v.size()) return cstd::unexpected(BlobError::NotEnoughSpace);
        std::copy(v.begin(), v.end(), out.begin());
        return cstd::span<byte_t>(out.data(), v.size());
    }

    [[nodiscard]] bool ChunkExists(cstd::span<const char> name) {
        return blobs_.find(Key(name)) != blobs_.end();
    }

    [[nodiscard]] size_t ChunkCount() const { return blobs_.size(); }
};

#if defined(__cpp_concepts) && __cpp_concepts >= 201907L
static_assert(BlobBackend<InMemBlobBackend>);
#endif

}  // namespace kvhdf5
