#pragma once

// Path -> CTE tag scheme. A dataset's full path (file + groups + dataset name) is
// encoded as ONE slash-joined tag, e.g. "results/snapshots/2026/temperature"; the
// blob key within that tag is the chunk coordinate (chunking::ChunkCoordToName).
// So the KV address of a chunk is (tag, name) = (dataset-path, chunk-coord).
//
// HOST-ONLY by design: tags are created on the CPU (no 32-char limit — that limit
// only applies to GPU-created tags), so this is plain std::string, NOT CROSS_FUN,
// and must never be compiled into a device kernel.
//
// Convention (mirrors chunking::ChunkCoordToName): an empty result == failure. A
// valid tag is always non-empty, so the empty string is an unambiguous sentinel.

#include "defines.h"
#include <cuda/std/span>
#include <string>
#include <string_view>

namespace kvhdf5::tagpath {

inline constexpr char kTagSep = '/';

// Normalize a user-supplied path string into a canonical tag: split on '/', drop
// empty segments (collapsing duplicate separators and stripping leading/trailing
// ones), and rejoin with a single '/'. Deterministic and idempotent, so the same
// logical path always resolves to the same tag (CTE get-or-create dedups it).
// Components are kept verbatim (case-sensitive). Returns "" if no segment remains.
[[nodiscard]] inline std::string CanonicalTag(std::string_view raw) {
    std::string out;
    size_t i = 0;
    while (i < raw.size()) {
        while (i < raw.size() && raw[i] == kTagSep) ++i;  // skip separators
        size_t start = i;
        while (i < raw.size() && raw[i] != kTagSep) ++i;   // span one segment
        if (i > start) {
            if (!out.empty()) out.push_back(kTagSep);
            out.append(raw.substr(start, i - start));
        }
    }
    return out;
}

// Build a canonical tag from already-separated hierarchy components (file name,
// group names, dataset name). Strict: returns "" if any component is empty or
// itself contains the separator (an embedded '/' would corrupt the path depth).
[[nodiscard]] inline std::string JoinTag(cstd::span<const std::string_view> components) {
    std::string out;
    for (std::string_view c : components) {
        if (c.empty() || c.find(kTagSep) != std::string_view::npos) return {};
        if (!out.empty()) out.push_back(kTagSep);
        out.append(c);
    }
    return out;
}

}  // namespace kvhdf5::tagpath
