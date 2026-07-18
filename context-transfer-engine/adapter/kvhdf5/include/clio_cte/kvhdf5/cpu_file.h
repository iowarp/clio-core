#pragma once

#include "defines.h"
#include "blob_backend.h"
#include "cpu_dataset.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <cuda/std/expected>

namespace kvhdf5 {

// MVP host control plane: owns the backend + a flat path->DatasetMeta table.
// Group hierarchy is carried in the path string for now; a real Group tree and
// path->TagId resolution land in Slice 2. DatasetMeta is heap-owned so its
// address stays stable across rehash (Dataset holds a DatasetMeta*).
template<KVHDF5_BLOB_BACKEND B>
class File {
    B backend_;
    std::unordered_map<std::string, std::unique_ptr<DatasetMeta>> datasets_;

public:
    explicit File(B backend) : backend_(std::move(backend)) {}

    // A fixed anchor: Datasets borrow &backend_ and DatasetMeta*, so the File
    // must not move out from under them. (Already non-copyable via unique_ptr.)
    File(File&&) = delete;
    File& operator=(File&&) = delete;

    B& Backend() { return backend_; }

    [[nodiscard]] cstd::expected<Dataset<B>, IoError> CreateDataset(std::string path, Layout layout) {
        if (!layout.Valid()) return cstd::unexpected(IoError::BadLayout);
        if (datasets_.count(path)) return cstd::unexpected(IoError::Unsupported);  // already exists
        auto meta = std::make_unique<DatasetMeta>(DatasetMeta{path, std::move(layout)});
        DatasetMeta* ptr = meta.get();
        datasets_.emplace(std::move(path), std::move(meta));
        return Dataset<B>(ptr, &backend_);
    }

    [[nodiscard]] cstd::expected<Dataset<B>, IoError> OpenDataset(const std::string& path) {
        auto it = datasets_.find(path);
        if (it == datasets_.end()) return cstd::unexpected(IoError::NotFound);
        return Dataset<B>(it->second.get(), &backend_);
    }
};

}  // namespace kvhdf5
