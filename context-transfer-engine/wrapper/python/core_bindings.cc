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

  // Bind MemContext for method calls
  nb::class_<hipc::MemContext>(m, "MemContext")
      .def(nb::init<>());

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

  // Bind Client class with PollTelemetryLog, ReorganizeBlob, and Query methods
  // Note: Query methods use lambda wrappers to avoid evaluating chi::PoolQuery
  // static methods (Broadcast/Dynamic) at module import time, which would
  // cause std::bad_cast errors before runtime initialization
  nb::class_<wrp_cte::core::Client>(m, "Client")
      .def(nb::init<>())
      .def(nb::init<const chi::PoolId &>())
      .def("PollTelemetryLog", &wrp_cte::core::Client::PollTelemetryLog,
           "mctx"_a, "minimum_logical_time"_a,
           "Poll telemetry log with minimum logical time filter")
      .def("ReorganizeBlob", &wrp_cte::core::Client::ReorganizeBlob,
           "mctx"_a, "tag_id"_a, "blob_name"_a, "new_score"_a,
           "Reorganize single blob with new score for data placement optimization")
     .def("TagQuery",
         [](wrp_cte::core::Client &self, const hipc::MemContext &mctx,
            const std::string &tag_regex, uint32_t max_tags, const chi::PoolQuery &pool_query) {
           return self.TagQuery(mctx, tag_regex, max_tags, pool_query);
         },
         "mctx"_a, "tag_regex"_a, "max_tags"_a = 0, "pool_query"_a,
         "Query tags by regex pattern, returns vector of tag names")
     .def("BlobQuery",
         [](wrp_cte::core::Client &self, const hipc::MemContext &mctx,
            const std::string &tag_regex, const std::string &blob_regex,
            uint32_t max_blobs, const chi::PoolQuery &pool_query) {
           return self.BlobQuery(mctx, tag_regex, blob_regex, max_blobs, pool_query);
         },
         "mctx"_a, "tag_regex"_a, "blob_regex"_a, "max_blobs"_a = 0, "pool_query"_a,
         "Query blobs by tag and blob regex patterns, returns vector of (tag_name, blob_name) pairs")
     .def("RegisterTarget",
         [](wrp_cte::core::Client &self, const hipc::MemContext &mctx,
            const std::string &target_name, chimaera::bdev::BdevType bdev_type,
            uint64_t total_size, const chi::PoolQuery &target_query, const chi::PoolId &bdev_id) {
           return self.RegisterTarget(mctx, target_name, bdev_type, total_size, target_query, bdev_id);
         },
         "mctx"_a, "target_name"_a, "bdev_type"_a, "total_size"_a, 
         "target_query"_a, "bdev_id"_a,
         "Register a storage target. Returns 0 on success, non-zero on failure")
     .def("RegisterTarget",
         [](wrp_cte::core::Client &self, const hipc::MemContext &mctx,
            const std::string &target_name, chimaera::bdev::BdevType bdev_type,
            uint64_t total_size) {
           return self.RegisterTarget(mctx, target_name, bdev_type, total_size);
         },
         "mctx"_a, "target_name"_a, "bdev_type"_a, "total_size"_a,
         "Register a storage target with default query and pool ID. Returns 0 on success, non-zero on failure")
     .def("DelBlob",
         [](wrp_cte::core::Client &self, const hipc::MemContext &mctx,
            const wrp_cte::core::TagId &tag_id, const std::string &blob_name) {
           return self.DelBlob(mctx, tag_id, blob_name);
         },
         "mctx"_a, "tag_id"_a, "blob_name"_a,
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

  // Chimaera runtime initialization functions
  m.def("chimaera_runtime_init", &chi::CHIMAERA_RUNTIME_INIT,
        "Initialize the Chimaera runtime");

  m.def("chimaera_client_init", &chi::CHIMAERA_CLIENT_INIT,
        "Initialize the Chimaera client");

  // CTE-specific initialization
  // Note: Lambda wrapper used to avoid chi::PoolQuery::Dynamic() evaluation at import
  m.def("initialize_cte",
        [](const std::string &config_path, const chi::PoolQuery &pool_query) {
          return wrp_cte::core::WRP_CTE_CLIENT_INIT(config_path, pool_query);
        },
        "config_path"_a, "pool_query"_a,
        "Initialize the CTE subsystem");
}