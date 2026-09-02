/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file chunk_lineage.h
 * @brief Blob name -> the identity of a block ACROSS simulation timesteps.
 *
 * Scientific data arrives as the same logical block over and over: the same
 * field, the same chunk of it, one timestep later. Nothing in a blob name
 * says so directly, because every driver folds the timestep INTO the name to
 * keep blobs distinct. Recovering the part that does not change is what lets
 * anything be carried from one timestep to the next.
 *
 * Two conventions ship in this tree and both are supported, because a rule
 * fitted to one of them silently mis-keys the other:
 *
 *     step00010/E_x/chunk_0         field replay  -- timestep FIRST
 *     force/step_100/chunk_2        in-situ       -- timestep SECOND
 *
 * so the timestep component is found by SHAPE, not by position.
 *
 * WHY THE VOCABULARY IS A CLOSED LIST. `chunk_0` and `fab0000_comp00_density`
 * also contain digits, and a field may legitimately be called `E1`. Matching
 * "a component ending in digits" would fold every chunk of a field onto one
 * lineage, which is worse than not tracking it at all -- two different blocks
 * sharing a key get differenced against each other. So a component counts as
 * the timestep only when its non-digit prefix is one of the names below, and
 * anything else is left alone.
 *
 * UNRESOLVED IS A REAL ANSWER, not a failure. A name with no timestep
 * component, or with two, yields resolved=false, and the caller is expected
 * to treat that as "run the model" -- reusing a cached decision under an
 * identity that might belong to a different block is the one outcome that
 * silently corrupts a selection.
 */
#ifndef CLIO_CTP_COMPRESS_PREPROCESS_CHUNK_LINEAGE_H_
#define CLIO_CTP_COMPRESS_PREPROCESS_CHUNK_LINEAGE_H_

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace ctp::compress::preprocess {

/** One blob name, split into what persists and what advances. */
struct BlobLineage {
  /** Identity across timesteps: the name with the timestep component
   *  removed, e.g. "E_x/chunk_0". Empty when unresolved. */
  std::string key;
  /** The parsed simulation timestep, or -1 when unresolved. Need not be
   *  contiguous, and need not start at zero. */
  long long timestep = -1;
  /** False when no single timestep component could be identified. The caller
   *  must then behave as if this block had never been seen. */
  bool resolved = false;
};

namespace detail {

/** Non-digit prefixes that mark a timestep component. Deliberately short and
 *  explicit -- see the header comment on why this is not a pattern. */
inline bool IsTimestepPrefix(std::string_view p) {
  return p == "step" || p == "plt" || p == "frame" || p == "iter" ||
         p == "cycle" || p == "snapshot" || p == "ts";
}

/**
 * Does `comp` look like "<known prefix><digits>"? Writes the value if so.
 *
 * A trailing '_' on the prefix is dropped so `step_00025` and `step00025`
 * are the same shape. Parsed in base 10 explicitly: these components are
 * zero-padded, and the default base would read `00025` as octal 21.
 */
inline bool ParseTimestepComponent(std::string_view comp, long long *out) {
  size_t d = comp.size();
  while (d > 0 && comp[d - 1] >= '0' && comp[d - 1] <= '9') --d;
  if (d == comp.size() || d == 0) return false;  // no digits, or no prefix
  std::string_view prefix = comp.substr(0, d);
  if (!prefix.empty() && prefix.back() == '_') prefix.remove_suffix(1);
  if (!IsTimestepPrefix(prefix)) return false;
  long long v = 0;
  for (size_t i = d; i < comp.size(); ++i) {
    v = v * 10 + (comp[i] - '0');
    if (v > (1LL << 60)) return false;  // absurd; treat as not a timestep
  }
  *out = v;
  return true;
}

}  // namespace detail

/**
 * Split a blob name into its lineage key and its timestep.
 *
 * @param blob_name e.g. "step00010/E_x/chunk_0" or "force/step_100/chunk_2"
 * @return resolved=false when the name carries no timestep component, or more
 *   than one -- both are ambiguous, and the caller must not reuse state.
 */
inline BlobLineage ParseBlobLineage(std::string_view blob_name) {
  BlobLineage out;
  if (blob_name.empty()) return out;

  std::vector<std::string_view> parts;
  size_t start = 0;
  for (size_t i = 0; i <= blob_name.size(); ++i) {
    if (i == blob_name.size() || blob_name[i] == '/') {
      parts.push_back(blob_name.substr(start, i - start));
      start = i + 1;
    }
  }

  size_t step_at = parts.size();
  long long step = -1;
  for (size_t i = 0; i < parts.size(); ++i) {
    long long v = 0;
    if (!detail::ParseTimestepComponent(parts[i], &v)) continue;
    // A second match makes the name ambiguous. Refuse rather than guess:
    // picking one would key two different blocks the same way.
    if (step_at != parts.size()) return BlobLineage{};
    step_at = i;
    step = v;
  }
  if (step_at == parts.size()) return out;  // none found

  std::string key;
  for (size_t i = 0; i < parts.size(); ++i) {
    if (i == step_at) continue;
    if (!key.empty()) key.push_back('/');
    key.append(parts[i]);
  }
  // Every component was the timestep: nothing identifies the block.
  if (key.empty()) return BlobLineage{};

  out.key = std::move(key);
  out.timestep = step;
  out.resolved = true;
  return out;
}

}  // namespace ctp::compress::preprocess

#endif  // CLIO_CTP_COMPRESS_PREPROCESS_CHUNK_LINEAGE_H_
