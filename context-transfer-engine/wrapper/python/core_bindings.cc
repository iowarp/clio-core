#include <nanobind/nanobind.h>
#include <nanobind/stl/chrono.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>

#include <wrp_cte/core/core_client.h>
#include <wrp_cte/core/core_tasks.h>
#include <chimaera/chimaera.h>
#include <chimaera/bdev/bdev_tasks.h>

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(wrp_cte_core_ext, m) {
  m.doc() = "Python bindings for WRP CTE Core";

  // Bind CteOp enum
  nb::enum_<wrp_cte::core::CteOp>(m, "CteOp")
      .value("kPutBlob", wrp_cte::core::CteOp::kPutBlob)
      .value("kGetBlob", wrp_cte::core::CteOp::kGetBlob)
      .value("kDelBlob", wrp_cte::core::CteOp::kDelBlob)
      .value("kGetOrCreateTag", wrp_cte::core::CteOp::kGetOrCreateTag)
      .value("kDelTag", wrp_cte::core::CteOp::kDelTag)
      .value("kGetTagSize", wrp_cte::core::CteOp::kGetTagSize);

  // Bind BdevType enum
  nb::enum_<chimaera::bdev::BdevType>(m, "BdevType")
      .value("kFile", chimaera::bdev::BdevType::kFile)
      .value("kRam", chimaera::bdev::BdevType::kRam);

  // Bind ChimaeraMode enum
  nb::enum_<chi::ChimaeraMode>(m, "ChimaeraMode")
      .value("kClient", chi::ChimaeraMode::kClient)
      .value("kServer", chi::ChimaeraMode::kServer)
      .value("kRuntime", chi::ChimaeraMode::kRuntime);

  // Bind UniqueId type (used by TagId, BlobId, and PoolId)
  // Note: TagId, BlobId, and PoolId are all aliases for chi::UniqueId, so we register the base type
  auto unique_id_class = nb::class_<wrp_cte::core::TagId>(m, "UniqueId")
      .def(nb::init<>())
      .def(nb::init<chi::u32, chi::u32>(), "major"_a, "minor"_a,
           "Create UniqueId with major and minor values")
      .def_static("GetNull", &wrp_cte::core::TagId::GetNull)
      .def("ToU64", &wrp_cte::core::TagId::ToU64)
      .def("IsNull", &wrp_cte::core::TagId::IsNull)
      .def_rw("major_", &wrp_cte::core::TagId::major_)
      .def_rw("minor_", &wrp_cte::core::TagId::minor_);

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
                  "Create a Broadcast pool query (routes to all nodes)")
      .def_static("Dynamic", &chi::PoolQuery::Dynamic,
                  "Create a Dynamic pool query (automatic routing optimization)")
      .def_static("Local", &chi::PoolQuery::Local,
                  "Create a Local pool query (routes to local node only)");

  // Bind CteTelemetry structure
  nb::class_<wrp_cte::core::CteTelemetry>(m, "CteTelemetry")
      .def(nb::init<>())
      .def(nb::init<wrp_cte::core::CteOp, size_t, size_t,
                    const wrp_cte::core::TagId &,
                    const wrp_cte::core::Timestamp &,
                    const wrp_cte::core::Timestamp &, std::uint64_t>(),
           "op"_a, "off"_a, "size"_a, "tag_id"_a, "mod_time"_a,
           "read_time"_a, "logical_time"_a = 0)
      .def_rw("op_", &wrp_cte::core::CteTelemetry::op_)
      .def_rw("off_", &wrp_cte::core::CteTelemetry::off_)
      .def_rw("size_", &wrp_cte::core::CteTelemetry::size_)
      .def_rw("tag_id_", &wrp_cte::core::CteTelemetry::tag_id_)
      .def_rw("mod_time_", &wrp_cte::core::CteTelemetry::mod_time_)
      .def_rw("read_time_", &wrp_cte::core::CteTelemetry::read_time_)
      .def_rw("logical_time_", &wrp_cte::core::CteTelemetry::logical_time_);

  // Bind Client class with async API methods wrapped for synchronous Python use
  // Note: All methods use lambda wrappers to call async methods and wait for completion
  nb::class_<wrp_cte::core::Client>(m, "Client")
      .def(nb::init<>())
      .def(nb::init<const chi::PoolId &>())
      .def("PollTelemetryLog",
          [](wrp_cte::core::Client &self, std::uint64_t minimum_logical_time) {
            auto task = self.AsyncPollTelemetryLog(minimum_logical_time);
            task.Wait();
            // Convert chi::priv::vector to std::vector for Python
            std::vector<wrp_cte::core::CteTelemetry> result;
            for (size_t i = 0; i < task->entries_.size(); ++i) {
              result.push_back(task->entries_[i]);
            }
            return result;
          },
          "minimum_logical_time"_a,
          "Poll telemetry log with minimum logical time filter")
      .def("ReorganizeBlob",
          [](wrp_cte::core::Client &self,
             const wrp_cte::core::TagId &tag_id, const std::string &blob_name,
             float new_score) {
            auto task = self.AsyncReorganizeBlob(tag_id, blob_name, new_score);
            task.Wait();
            return task->return_code_ == 0;
          },
          "tag_id"_a, "blob_name"_a, "new_score"_a,
          "Reorganize single blob with new score for data placement optimization")
     .def("TagQuery",
         [](wrp_cte::core::Client &self,
            const std::string &tag_regex, uint32_t max_tags, const chi::PoolQuery &pool_query) {
           auto task = self.AsyncTagQuery(tag_regex, max_tags, pool_query);
           task.Wait();
           return task->results_;
         },
         "tag_regex"_a, "max_tags"_a = 0, "pool_query"_a,
         "Query tags by regex pattern, returns vector of tag names")
     .def("BlobQuery",
         [](wrp_cte::core::Client &self,
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
     .def("RegisterTarget",
         [](wrp_cte::core::Client &self,
            const std::string &target_name, chimaera::bdev::BdevType bdev_type,
            uint64_t total_size, const chi::PoolQuery &target_query, const chi::PoolId &bdev_id) {
           auto task = self.AsyncRegisterTarget(target_name, bdev_type, total_size, target_query, bdev_id);
           task.Wait();
           return task->return_code_;
         },
         "target_name"_a, "bdev_type"_a, "total_size"_a,
         "target_query"_a, "bdev_id"_a,
         "Register a storage target. Returns 0 on success, non-zero on failure")
     .def("RegisterTarget",
         [](wrp_cte::core::Client &self,
            const std::string &target_name, chimaera::bdev::BdevType bdev_type,
            uint64_t total_size) {
           auto task = self.AsyncRegisterTarget(target_name, bdev_type, total_size);
           task.Wait();
           return task->return_code_;
         },
         "target_name"_a, "bdev_type"_a, "total_size"_a,
         "Register a storage target with default query and pool ID. Returns 0 on success, non-zero on failure")
     .def("DelBlob",
         [](wrp_cte::core::Client &self,
            const wrp_cte::core::TagId &tag_id, const std::string &blob_name) {
           auto task = self.AsyncDelBlob(tag_id, blob_name);
           task.Wait();
           return task->return_code_ == 0;
         },
         "tag_id"_a, "blob_name"_a,
         "Delete a blob from a tag. Returns True on success, False otherwise");

  // Bind Tag wrapper class - provides convenient API for tag operations
  // This class wraps tag operations and provides automatic memory management
  nb::class_<wrp_cte::core::Tag>(m, "Tag")
      .def(nb::init<const std::string &>(),
           "tag_name"_a,
           "Create or get a tag by name. Calls GetOrCreateTag internally.")
      .def(nb::init<const wrp_cte::core::TagId &>(),
           "tag_id"_a,
           "Create tag wrapper from existing TagId")
      .def("PutBlob",
           [](wrp_cte::core::Tag &self, const std::string &blob_name,
              nb::bytes data, size_t off) {
             // Use nb::bytes to accept bytes from Python
             // c_str() returns const char*, size() returns size
             self.PutBlob(blob_name, data.c_str(), data.size(), off);
           },
           "blob_name"_a, "data"_a, "off"_a = 0,
           "Put blob data. Automatically allocates shared memory and copies data. "
           "Args: blob_name (str), data (bytes), off (int, optional)")
      .def("GetBlob",
           [](wrp_cte::core::Tag &self, const std::string &blob_name,
              size_t data_size, size_t off) -> std::string {
             // Allocate buffer and retrieve blob data
             std::string result(data_size, '\0');
             self.GetBlob(blob_name, result.data(), data_size, off);
             return result;
           },
           "blob_name"_a, "data_size"_a, "off"_a = 0,
           "Get blob data. Automatically allocates shared memory and copies data. "
           "Args: blob_name (str), data_size (int), off (int, optional). "
           "Returns: str/bytes containing blob data")
      .def("GetBlobScore", &wrp_cte::core::Tag::GetBlobScore,
           "blob_name"_a,
           "Get blob placement score (0.0-1.0). "
           "Args: blob_name (str). Returns: float")
      .def("GetBlobSize", &wrp_cte::core::Tag::GetBlobSize,
           "blob_name"_a,
           "Get blob size in bytes. "
           "Args: blob_name (str). Returns: int")
      .def("GetContainedBlobs", &wrp_cte::core::Tag::GetContainedBlobs,
           "Get all blob names contained in this tag. "
           "Returns: list of str")
      .def("GetTagId", &wrp_cte::core::Tag::GetTagId,
           "Get the TagId for this tag. "
           "Returns: TagId");

  // Module-level convenience functions
  m.def(
      "get_cte_client",
      []() -> wrp_cte::core::Client { return *WRP_CTE_CLIENT; },
      "Get a copy of the global CTE client instance");

  // Chimaera initialization function (unified)
  m.def("chimaera_init", &chi::CHIMAERA_INIT,
        "mode"_a, "default_with_runtime"_a = false,
        "Initialize Chimaera with specified mode.\n\n"
        "Args:\n"
        "    mode: ChimaeraMode.kClient or ChimaeraMode.kServer/kRuntime\n"
        "    default_with_runtime: If True, starts runtime in addition to client (default: False)\n\n"
        "Environment variable CHIMAERA_WITH_RUNTIME overrides default_with_runtime:\n"
        "    CHIMAERA_WITH_RUNTIME=1 - Start runtime regardless of mode\n"
        "    CHIMAERA_WITH_RUNTIME=0 - Don't start runtime (client only)\n\n"
        "Returns:\n"
        "    bool: True if initialization successful, False otherwise");

  // CTE-specific initialization
  // Note: Lambda wrapper used to avoid chi::PoolQuery::Dynamic() evaluation at import
  m.def("initialize_cte",
        [](const std::string &config_path, const chi::PoolQuery &pool_query) {
          return wrp_cte::core::WRP_CTE_CLIENT_INIT(config_path, pool_query);
        },
        "config_path"_a, "pool_query"_a,
        "Initialize the CTE subsystem");
}
