#pragma once

#include "layout.h"
#include "tag_path.h"
#include <string>

namespace kvhdf5 {

struct DatasetMeta {
    std::string path;
    Layout layout;
    // Slice 2: cached iowarp TagId resolved once from `path` on the runtime path.

    // The dataset's CTE tag = its canonicalized path (one slash-joined tag). The
    // blob key within it is the chunk coordinate, so a chunk addresses as
    // (TagName(), ChunkCoordToName(coord)). Empty if `path` has no real segment.
    [[nodiscard]] std::string TagName() const { return tagpath::CanonicalTag(path); }
};

}  // namespace kvhdf5
