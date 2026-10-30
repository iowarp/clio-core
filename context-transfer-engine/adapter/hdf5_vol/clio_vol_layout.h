/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/* The clio HDF5 VOL connector's cache layout in the CTE, as a contract.
 *
 * The connector is not the only thing that may fill its cache: the CAE's
 * `hdf5_vol` format stages a file's datasets ahead of the application, so its
 * first read is already served from the tier. Both sides must agree on every
 * name and byte here, or the CAE would stage blobs the connector never finds --
 * or, worse, finds and misreads. So the layout lives in one header and both
 * include it.
 *
 *   tag    "hdf5:" + the file's canonical (realpath) name
 *   blobs  "<dataset path>/chunk_<i>": bytes [i*chunk, (i+1)*chunk) of the
 *          dataset's linear row-major image in the NATIVE form of its file
 *          type, each blob holding only its own bytes at blob offset 0
 *   stamp  kStampBlobName = Stamp(path, chunk_size), the file identity the
 *          image is consistent with; anything else at open drops the tag
 */

#ifndef CLIO_HDF5_VOL_LAYOUT_H_
#define CLIO_HDF5_VOL_LAYOUT_H_

#include <climits>
#include <cstdlib>
#include <string>

#include "adapter/clio_coherence_stamp.h"

namespace clio::adapter::vol_layout {

/* 64 KiB; see CLIO_VOL_DEFAULT_CHUNK_SIZE in clio_vol.h, which is this. */
constexpr size_t kDefaultChunkSize = 64 * 1024;

/* Reserved blob name. The leading "__clio" cannot collide with a dataset path,
   which HDF5 always renders with a leading '/'. */
constexpr const char *kStampBlobName = "__clio_coherence_stamp";

/* CLIO_VOL_CHUNK_SIZE, or the default when unset or 0. */
inline size_t ChunkSizeFromEnv() {
  const char *e = std::getenv("CLIO_VOL_CHUNK_SIZE");
  if (!e) return kDefaultChunkSize;
  size_t n = std::strtoul(e, nullptr, 10);
  return n == 0 ? kDefaultChunkSize : n;
}

/* The name HDF5 was handed, resolved through symlinks and relative segments,
   so one file has one tag however it is spelled: an application that opens
   "datasets/x.h5" through a symlinked directory and a CAE transfer naming the
   real path must land on the same blobs. Falls back to the name as given when
   it cannot be resolved (it does not exist yet, or no realpath here). */
inline std::string CanonicalPath(const std::string &name) {
#ifndef _WIN32
  char buf[PATH_MAX];
  if (!name.empty() && ::realpath(name.c_str(), buf) != nullptr) {
    return std::string(buf);
  }
#endif
  return name;
}

inline std::string TagName(const std::string &file_name) {
  return std::string("hdf5:") + CanonicalPath(file_name);
}

inline std::string ChunkBlobName(const std::string &dataset_path, size_t i) {
  return dataset_path + "/chunk_" + std::to_string(i);
}

/* "3:" is the layout version -- bump it when anything above changes, so the
   mismatch drops the tag. The chunk size is part of the stamp because it is
   part of the layout: chunk_<i> means different bytes under a different size,
   so a reader configured with another CLIO_VOL_CHUNK_SIZE than the writer must
   miss rather than reassemble the wrong offsets. Empty when the file cannot be
   stat'd, which callers treat as "cannot vouch for this". */
inline std::string Stamp(const char *path, size_t chunk_size) {
  const std::string id = stamp::FileIdentity(path);
  if (id.empty()) return std::string();
  return std::string("3:") + std::to_string(chunk_size) + ":" + id;
}

}  // namespace clio::adapter::vol_layout

#endif  // CLIO_HDF5_VOL_LAYOUT_H_
