#pragma once

// Runtime side of the path->tag scheme (see tag_path.h): resolve a canonical tag
// string to an iowarp CTE TagId via get-or-create. Host-only (tags are created on
// the CPU). Kept out of cpu_dataset.h so the no-iowarp unit build stays clean;
// only the iowarp-facing layer (GpuCteDataset) pulls this in.

#include <clio_ctp/constants/macros.h>  // CTP_IS_DEVICE_PASS

#if !CTP_IS_DEVICE_PASS

#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>

#include <stdexcept>
#include <string>

namespace kvhdf5 {

namespace cte = clio::cte::core;

// Get-or-create the tag named `tag_name` and return its TagId. Idempotent: the
// same canonical name always maps to the same tag (the dedup/searchability win).
// Throws on a null client, an empty name, or an iowarp failure (matching
// GpuCteDataset's throw-on-failure style).
[[nodiscard]] inline cte::TagId ResolveTagId(cte::Client* client,
                                             const std::string& tag_name) {
    if (client == nullptr)
        throw std::runtime_error("ResolveTagId: null CTE client");
    if (tag_name.empty())
        throw std::runtime_error("ResolveTagId: empty tag name");
    auto task = client->AsyncGetOrCreateTag(tag_name);
    task.Wait();
    if (task->GetReturnCode() != 0)
        throw std::runtime_error("ResolveTagId: AsyncGetOrCreateTag failed");
    return task->tag_id_;
}

}  // namespace kvhdf5

#endif  // !CTP_IS_DEVICE_PASS
