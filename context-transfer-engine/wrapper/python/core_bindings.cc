/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/chrono.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>

#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/bdev/bdev_tasks.h>

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(clio_cte_core_ext, m) {
  m.doc() = "Python bindings for WRP CTE Core";

  // Bind CteOp enum
  nb::enum_<clio::cte::core::CteOp>(m, "CteOp")
      .value("kPutBlob", clio::cte::core::CteOp::kPutBlob)
      .value("kGetBlob", clio::cte::core::CteOp::kGetBlob)
      .value("kDelBlob", clio::cte::core::CteOp::kDelBlob)
      .value("kGetOrCreateTag", clio::cte::core::CteOp::kGetOrCreateTag)
      .value("kDelTag", clio::cte::core::CteOp::kDelTag)
      .value("kGetTagSize", clio::cte::core::CteOp::kGetTagSize);

  // Bind BdevType enum
  nb::enum_<clio::run::bdev::BdevType>(m, "BdevType")
      .value("kFile", clio::run::bdev::BdevType::kFile)
      .value("kRam", clio::run::bdev::BdevType::kRam);

  // Bind ChimaeraMode enum
  nb::enum_<chi::ChimaeraMode>(m, "ChimaeraMode")
      .value("kClient", chi::ChimaeraMode::kClient)
      .value("kServer", chi::ChimaeraMode::kServer)
      .value("kRuntime", chi::ChimaeraMode::kRuntime);

  // Bind UniqueId type (used by TagId, BlobId, and PoolId)
  // Note: TagId, BlobId, and PoolId are all aliases for chi::UniqueId, so we register the base type
  auto unique_id_class = nb::class_<clio::cte::core::TagId>(m, "UniqueId")
      .def(nb::init<>())
      .def(nb::init<chi::u32, chi::u32>(), "major"_a, "minor"_a,
           "Create UniqueId with major and minor values")
      .def_static("GetNull", &clio::cte::core::TagId::GetNull)
      .def("ToU64", &clio::cte::core::TagId::ToU64)
      .def("IsNull", &clio::cte::core::TagId::IsNull)
      .def_rw("major_", &clio::cte::core::TagId::major_)
      .def_rw("minor_", &clio::cte::core::TagId::minor_);

  // Create aliases for TagId, BlobId, and PoolId (all are UniqueId)
  m.attr("TagId") = unique_id_class;
  m.attr("BlobId") = unique_id_class;
  m.attr("PoolId") = unique_id_class;

  // Note: Timestamp (chrono time_point) is automatically handled by
  // nanobind/stl/chrono.h

  // Bind PoolQuery for routing queries
  nb::class_<chi::PoolQuery>(m, "PoolQuery")
      .def(nb::init<>())
      .def_static("Broadcast", &chi::PoolQuery::Broadcast,
                  "net_timeout"_a = -1.0f,
                  "Create a Broadcast pool query (routes to all nodes)")
      .def_static("Dynamic", &chi::PoolQuery::Dynamic,
                  "net_timeout"_a = -1.0f,
                  "Create a Dynamic pool query (automatic routing optimization)")
      .def_static("Local", &chi::PoolQuery::Local,
                  "parallelism"_a = 32,
                  "Create a Local pool query (routes to local node only). "
                  "parallelism defaults to 32 — matches C++ "
                  "chi::PoolQuery::Local default");

  // Bind SemanticSearchResult — one row in the ranked output of a
  // SemanticSearch task. Exposed as a plain dataclass-style object so
  // Python callers can iterate the returned list and read the fields
  // directly (results[0].tag_id, .blob_name, .score).
  nb::class_<clio::cte::core::SemanticSearchResult>(m, "SemanticSearchResult")
      .def(nb::init<>())
      .def_rw("tag_id", &clio::cte::core::SemanticSearchResult::tag_id_)
      .def_rw("blob_name", &clio::cte::core::SemanticSearchResult::blob_name_)
      .def_rw("score", &clio::cte::core::SemanticSearchResult::score_)
      .def("__repr__", [](const clio::cte::core::SemanticSearchResult &r) {
        return std::string("SemanticSearchResult(tag_id=(") +
               std::to_string(r.tag_id_.major_) + "," +
               std::to_string(r.tag_id_.minor_) + "), blob_name='" +
               r.blob_name_ + "', score=" + std::to_string(r.score_) + ")";
      });

  nb::class_<clio::cte::core::TemporalSearchResult>(m, "TemporalSearchResult")
      .def(nb::init<>())
      .def_rw("tag_id", &clio::cte::core::TemporalSearchResult::tag_id_)
      .def_rw("blob_name", &clio::cte::core::TemporalSearchResult::blob_name_)
      .def_rw("last_modified", &clio::cte::core::TemporalSearchResult::last_modified_)
      .def("__repr__", [](const clio::cte::core::TemporalSearchResult &r) {
        return std::string("TemporalSearchResult(tag_id=(") +
               std::to_string(r.tag_id_.major_) + "," +
               std::to_string(r.tag_id_.minor_) + "), blob_name='" +
               r.blob_name_ + "', last_modified=" +
               std::to_string(r.last_modified_) + ")";
      });

  // Bind CteTelemetry structure
  nb::class_<clio::cte::core::CteTelemetry>(m, "CteTelemetry")
      .def(nb::init<>())
      .def(nb::init<clio::cte::core::CteOp, size_t, size_t,
                    const clio::cte::core::TagId &,
                    const clio::cte::core::Timestamp &,
                    const clio::cte::core::Timestamp &, std::uint64_t>(),
           "op"_a, "off"_a, "size"_a, "tag_id"_a, "mod_time"_a,
           "read_time"_a, "logical_time"_a = 0)
      .def_rw("op_", &clio::cte::core::CteTelemetry::op_)
      .def_rw("off_", &clio::cte::core::CteTelemetry::off_)
      .def_rw("size_", &clio::cte::core::CteTelemetry::size_)
      .def_rw("tag_id_", &clio::cte::core::CteTelemetry::tag_id_)
      .def_rw("mod_time_", &clio::cte::core::CteTelemetry::mod_time_)
      .def_rw("read_time_", &clio::cte::core::CteTelemetry::read_time_)
      .def_rw("logical_time_", &clio::cte::core::CteTelemetry::logical_time_);

  // Bind Client class with async API methods wrapped for synchronous Python use
  // Note: All methods use lambda wrappers to call async methods and wait for completion
  nb::class_<clio::cte::core::Client>(m, "Client")
      .def(nb::init<>())
      .def(nb::init<const chi::PoolId &>())
      .def("PollTelemetryLog",
          [](clio::cte::core::Client &self, std::uint64_t minimum_logical_time) {
            auto task = self.AsyncPollTelemetryLog(minimum_logical_time);
            task.Wait();
            // Convert chi::priv::vector to std::vector for Python
            std::vector<clio::cte::core::CteTelemetry> result;
            for (size_t i = 0; i < task->entries_.size(); ++i) {
              result.push_back(task->entries_[i]);
            }
            return result;
          },
          "minimum_logical_time"_a,
          "Poll telemetry log with minimum logical time filter")
      .def("ReorganizeBlob",
          [](clio::cte::core::Client &self,
             const clio::cte::core::TagId &tag_id, const std::string &blob_name,
             float new_score) {
            auto task = self.AsyncReorganizeBlob(tag_id, blob_name, new_score);
            task.Wait();
            return task->return_code_ == 0;
          },
          "tag_id"_a, "blob_name"_a, "new_score"_a,
          "Reorganize single blob with new score for data placement optimization")
     .def("TagQuery",
         [](clio::cte::core::Client &self,
            const std::string &tag_regex, uint32_t max_tags, const chi::PoolQuery &pool_query) {
           auto task = self.AsyncTagQuery(tag_regex, max_tags, pool_query);
           task.Wait();
           return task->results_;
         },
         "tag_regex"_a, "max_tags"_a = 0, "pool_query"_a,
         "Query tags by regex pattern, returns vector of tag names")
     .def("BlobQuery",
         [](clio::cte::core::Client &self,
            const std::string &tag_regex, const std::string &blob_regex,
            uint32_t max_blobs, const chi::PoolQuery &pool_query) {
           auto task = self.AsyncBlobQuery(tag_regex, blob_regex, max_blobs, pool_query);
           task.Wait();
           // Convert separate tag_names_ and blob_names_ vectors to vector of pairs
           std::vector<std::pair<std::string, std::string>> result;
           size_t count = std::min(task->tag_names_.size(), task->blob_names_.size());
           for (size_t i = 0; i < count; ++i) {
             result.emplace_back(task->tag_names_[i], task->blob_names_[i]);
           }
           return result;
         },
         "tag_regex"_a, "blob_regex"_a, "max_blobs"_a = 0, "pool_query"_a,
         "Query blobs by tag and blob regex patterns, returns vector of (tag_name, blob_name) pairs")
     .def("SemanticSearch",
         [](clio::cte::core::Client &self,
            const std::string &tag_regex, const std::string &blob_regex,
            const std::string &query_text, uint32_t k,
            const chi::PoolQuery &pool_query) {
           // Server-side BM25 over the (tag, blob) pairs that match
           // both regexes. Returns the top-k scored results already
           // sorted descending. Wait() here is fine for the Python
           // binding because the call is synchronous from the caller's
           // perspective; the server runs the heavy lifting as a
           // coroutine.
           auto task = self.AsyncSemanticSearch(tag_regex, blob_regex,
                                                query_text, k, pool_query);
           task.Wait();
           return task->results_;
         },
         "tag_regex"_a, "blob_regex"_a, "query_text"_a, "k"_a = 10,
         "pool_query"_a,
         "BM25 keyword search over blob contents. Filters candidate "
         "blobs by tag_regex AND blob_regex (full-string std::regex_match), "
         "tokenizes each candidate's bytes, scores against query_text, "
         "returns top-k SemanticSearchResult sorted by descending score. "
         "Args: tag_regex (str), blob_regex (str), query_text (str), "
         "k (int, default 10; 0 = no cap), pool_query (PoolQuery; "
         "use PoolQuery.Broadcast() to search all nodes). "
         "Returns: list[SemanticSearchResult]")
     .def("TemporalSearch",
         [](clio::cte::core::Client &self,
            const std::string &tag_regex, const std::string &blob_regex,
            uint64_t time_begin, uint64_t time_end, uint32_t max_entries,
            const chi::PoolQuery &pool_query) {
           auto task = self.AsyncTemporalSearch(tag_regex, blob_regex,
                                                time_begin, time_end,
                                                max_entries, pool_query);
           task.Wait();
           return task->results_;
         },
         "tag_regex"_a, "blob_regex"_a,
         "time_begin"_a = uint64_t{0}, "time_end"_a = uint64_t{0},
         "max_entries"_a = uint32_t{0}, "pool_query"_a,
         "Find blobs by last-modified timestamp. Filters by tag_regex AND "
         "blob_regex (full-string std::regex_match), then returns blobs whose "
         "last_modified metadata falls within [time_begin, time_end] (epoch "
         "nanoseconds; 0 = no bound). Results are sorted ascending by "
         "last_modified and capped at max_entries (0 = unlimited). "
         "Returns: list[TemporalSearchResult]")
     .def("RegisterTarget",
         [](clio::cte::core::Client &self,
            const std::string &target_name, clio::run::bdev::BdevType bdev_type,
            uint64_t total_size, const chi::PoolQuery &target_query, const chi::PoolId &bdev_id) {
           auto task = self.AsyncRegisterTarget(target_name, bdev_type, total_size, target_query, bdev_id);
           task.Wait();
           return task->return_code_;
         },
         "target_name"_a, "bdev_type"_a, "total_size"_a,
         "target_query"_a, "bdev_id"_a,
         "Register a storage target. Returns 0 on success, non-zero on failure")
     .def("RegisterTarget",
         [](clio::cte::core::Client &self,
            const std::string &target_name, clio::run::bdev::BdevType bdev_type,
            uint64_t total_size) {
           auto task = self.AsyncRegisterTarget(target_name, bdev_type, total_size);
           task.Wait();
           return task->return_code_;
         },
         "target_name"_a, "bdev_type"_a, "total_size"_a,
         "Register a storage target with default query and pool ID. Returns 0 on success, non-zero on failure")
     .def("DelBlob",
         [](clio::cte::core::Client &self,
            const clio::cte::core::TagId &tag_id, const std::string &blob_name) {
           auto task = self.AsyncDelBlob(tag_id, blob_name);
           task.Wait();
           return task->return_code_ == 0;
         },
         "tag_id"_a, "blob_name"_a,
         "Delete a blob from a tag. Returns True on success, False otherwise");

  // Bind Tag wrapper class - provides convenient API for tag operations
  // This class wraps tag operations and provides automatic memory management
  nb::class_<clio::cte::core::Tag>(m, "Tag")
      .def(nb::init<const std::string &>(),
           "tag_name"_a,
           "Create or get a tag by name. Calls GetOrCreateTag internally.")
      .def(nb::init<const clio::cte::core::TagId &>(),
           "tag_id"_a,
           "Create tag wrapper from existing TagId")
      .def("PutBlob",
           [](clio::cte::core::Tag &self, const std::string &blob_name,
              nb::bytes data, size_t off) {
             // Use nb::bytes to accept bytes from Python
             // c_str() returns const char*, size() returns size
             self.PutBlob(blob_name, data.c_str(), data.size(), off);
           },
           "blob_name"_a, "data"_a, "off"_a = 0,
           "Put blob data. Automatically allocates shared memory and copies data. "
           "Args: blob_name (str), data (bytes), off (int, optional)")
      .def("GetBlob",
           [](clio::cte::core::Tag &self, const std::string &blob_name,
              size_t data_size, size_t off) -> nb::bytes {
             // Retrieve blob data into a buffer, return it as raw bytes.
             // Must return nb::bytes (not std::string): blobs are arbitrary
             // binary, and nanobind decodes a returned std::string as UTF-8,
             // which raises UnicodeDecodeError on any non-text payload. This
             // also mirrors PutBlob, which already accepts nb::bytes.
             std::string result(data_size, '\0');
             self.GetBlob(blob_name, result.data(), data_size, off);
             return nb::bytes(result.data(), result.size());
           },
           "blob_name"_a, "data_size"_a, "off"_a = 0,
           "Get blob data. Automatically allocates shared memory and copies data. "
           "Args: blob_name (str), data_size (int), off (int, optional). "
           "Returns: bytes containing the raw blob data")
      .def("GetBlobScore", &clio::cte::core::Tag::GetBlobScore,
           "blob_name"_a,
           "Get blob placement score (0.0-1.0). "
           "Args: blob_name (str). Returns: float")
      .def("GetBlobSize", &clio::cte::core::Tag::GetBlobSize,
           "blob_name"_a,
           "Get blob size in bytes. "
           "Args: blob_name (str). Returns: int")
      .def("GetContainedBlobs", &clio::cte::core::Tag::GetContainedBlobs,
           "Get all blob names contained in this tag. "
           "Returns: list of str")
      .def("ReorganizeBlob", &clio::cte::core::Tag::ReorganizeBlob,
           "blob_name"_a, "new_score"_a,
           "Reorganize blob with new score for data placement optimization. "
           "Args: blob_name (str), new_score (float, 0.0-1.0 where higher = faster tier)")
      .def("GetTagId", &clio::cte::core::Tag::GetTagId,
           "Get the TagId for this tag. "
           "Returns: TagId");

  // Module-level convenience functions
  m.def(
      "get_cte_client",
      []() -> clio::cte::core::Client { return *CLIO_CTE_CLIENT; },
      "Get a copy of the global CTE client instance");

  // CLIO Runtime initialization function (unified)
  m.def("chimaera_init", &chi::CHIMAERA_INIT,
        "mode"_a, "default_with_runtime"_a = false, "is_restart"_a = false,
        "Initialize Chimaera with specified mode.\n\n"
        "Args:\n"
        "    mode: ChimaeraMode.kClient or ChimaeraMode.kServer/kRuntime\n"
        "    default_with_runtime: If True, starts runtime in addition to client (default: False)\n"
        "    is_restart: If True, force restart on compose pools and replay WAL (default: False)\n\n"
        "Environment variable CHI_WITH_RUNTIME overrides default_with_runtime:\n"
        "    CHI_WITH_RUNTIME=1 - Start runtime regardless of mode\n"
        "    CHI_WITH_RUNTIME=0 - Don't start runtime (client only)\n\n"
        "Returns:\n"
        "    bool: True if initialization successful, False otherwise");

  // CTE-specific initialization
  // Note: Lambda wrapper used to avoid chi::PoolQuery::Dynamic() evaluation at import
  m.def("initialize_cte",
        [](const std::string &config_path, const chi::PoolQuery &pool_query) {
          return clio::cte::core::CLIO_CTE_CLIENT_INIT(config_path, pool_query);
        },
        "config_path"_a, "pool_query"_a,
        "Initialize the CTE subsystem");
}
