/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 *
 * The CLIO HDF5 access-telemetry contract.
 *
 * HDF5 has no standard logging schema. It ships logging VOL connectors and
 * H5FD_LOG, but nothing with a documented, versioned event format a tool can
 * consume. We are therefore DEFINING one rather than adopting one, and this
 * header is that definition -- kept next to the code that emits it, because a
 * contract in a separate document drifts from the thing it describes.
 *
 * ---------------------------------------------------------------------------
 * Two producers, two altitudes, one envelope
 * ---------------------------------------------------------------------------
 *
 * Both connectors emit telemetry, and they see genuinely different things:
 *
 *   altitude "semantic" (VOL) -- datasets, selections, datatypes. Knows WHICH
 *       dataset an access touched and what shape the selection was.
 *   altitude "byte" (VFD) -- addresses and lengths. Cannot know which dataset
 *       an access belongs to; byte offsets carry no names, and reconstructing
 *       them would mean reimplementing object-header parsing in the driver.
 *       In exchange it sees something the VOL cannot: H5FD_mem_t, the
 *       metadata-vs-raw-data split, because by the time the VOL sees a request
 *       the metadata traffic HDF5 generates around it is already invisible.
 *
 * Neither is a degraded version of the other. A recommendation engine reading
 * this data should switch on `altitude`, not assume fields.
 *
 * ---------------------------------------------------------------------------
 * Absent means ABSENT
 * ---------------------------------------------------------------------------
 *
 * A field a producer cannot measure is NOT emitted -- not as null, not as 0,
 * not as "". The byte-altitude record simply has no `dataset` key. This is the
 * single most important property of the format: a consumer must never be able
 * to confuse "not measured" with "measured as zero", because those lead to
 * opposite recommendations. `altitude` plus `capabilities` (below) is how a
 * consumer knows WHY a field is missing without guessing.
 *
 * ---------------------------------------------------------------------------
 * Capabilities
 * ---------------------------------------------------------------------------
 *
 * Every summary carries a machine-readable statement of what its producer can
 * and cannot see. This exists so a report can state its own limits without a
 * human remembering to: a tool reading a byte-altitude summary can say
 * "dataset-level recommendations unavailable" because the data says so, not
 * because someone hard-coded it.
 *
 * ---------------------------------------------------------------------------
 * Versioning
 * ---------------------------------------------------------------------------
 *
 * `schema` and `v` are on EVERY record and every summary. A stable contract is
 * a Q1.5 deliverable, and without a version any change silently breaks
 * consumers that cannot tell which format they are holding. Bump `v` for any
 * change that removes or reinterprets a field; adding a new optional field does
 * not require a bump.
 */
#ifndef CLIO_CTE_ADAPTER_CLIO_TRACE_SCHEMA_H_
#define CLIO_CTE_ADAPTER_CLIO_TRACE_SCHEMA_H_

#include <atomic>
#include <string>

namespace clio {
namespace trace {

/** Format identity. On every record and every summary. */
inline constexpr const char *kSchemaName = "clio.hdf5.access";
/* v2: the semantic altitude's write accounting was REINTERPRETED, which is
   exactly the case the versioning rule below says must bump.
   v1 emitted `write_served: {mirrored, native_only}`, where `mirrored` was
   decided by whether the NATIVE write succeeded -- a fact about the
   authoritative file, not about the tier. It therefore counted writes that
   never reached the tier, and any admission ratio built on it measured
   application write volume rather than admitted bytes.
   v2 replaces it with `write_staged: {staged, native_only, bytes_staged,
   bytes_discarded, bytes_resident}` and a matching `staged` value for the
   per-access `served` field. A consumer holding v1 data must not compare it
   with v2 data on these fields; everything else is unchanged. */
inline constexpr int kSchemaVersion = 2;

/** What layer produced this. See the altitude discussion above. */
inline constexpr const char *kAltitudeSemantic = "semantic";  /* VOL */
inline constexpr const char *kAltitudeByte = "byte";          /* VFD */

/**
 * The capabilities block, as a JSON fragment (no enclosing braces).
 *
 * Written as a literal per altitude rather than assembled, so that what a
 * producer CLAIMS to see lives in the same file as the definition of the
 * altitudes -- if a producer gains a field, this is the one place that has to
 * agree with it.
 */
inline std::string CapabilitiesJson(const char *altitude) {
  const std::string a(altitude);
  if (a == kAltitudeSemantic) {
    return "\"capabilities\":{"
           "\"sees\":[\"dataset_identity\",\"selection_shape\",\"datatype\","
           "\"chunk_layout\",\"chunk_alignment\",\"cache_hit_rate\"],"
           "\"cannot_see\":[\"metadata_vs_raw_split\",\"file_byte_offsets\"],"
           "\"limits\":\"Semantic altitude: sees which dataset and which "
           "selection, but not the metadata traffic HDF5 generates beneath it.\""
           "}";
  }
  return "\"capabilities\":{"
         "\"sees\":[\"file_byte_offsets\",\"request_sizes\","
         "\"metadata_vs_raw_split\",\"repeated_ranges\",\"access_locality\"],"
         "\"cannot_see\":[\"dataset_identity\",\"selection_shape\",\"datatype\","
         "\"chunk_indices\"],"
         "\"limits\":\"Byte altitude: sees every access HDF5 makes, including "
         "metadata, but cannot attribute one to a dataset. Dataset-level "
         "recommendations are unavailable from this producer -- run the VOL "
         "connector for those.\""
         "}";
}

/**
 * A monotonic per-process session number.
 *
 * Trace artifacts are keyed by <file>.<pid>, which is unique across processes
 * but NOT across successive opens of the same file within one process -- and
 * the very common shape "create, write, close; open, read, close" is exactly
 * that. Without this the second session truncates the first and the summary
 * describes only the last one, silently: a workload that wrote 4 GiB and then
 * read it back reports zero writes, which is not a small inaccuracy but the
 * opposite of the recommendation it would drive.
 */
inline unsigned NextSessionSeq() {
  static std::atomic<unsigned> seq{0};
  return seq.fetch_add(1);
}

/** `"schema":...,"v":...,"altitude":...` -- the common head of every record. */
inline std::string EnvelopeJson(const char *altitude) {
  return std::string("\"schema\":\"") + kSchemaName + "\",\"v\":" +
         std::to_string(kSchemaVersion) + ",\"altitude\":\"" + altitude + "\"";
}

}  // namespace trace
}  // namespace clio

#endif  // CLIO_CTE_ADAPTER_CLIO_TRACE_SCHEMA_H_
