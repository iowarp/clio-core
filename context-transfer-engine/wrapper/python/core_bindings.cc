#include <nanobind/nanobind.h>
#include <nanobind/stl/chrono.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <wrp_cte/core/core_client.h>
#include <wrp_cte/core/core_tasks.h>
#include <chimaera/chimaera.h>

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

  // Bind UniqueId type (used by both TagId and BlobId)
  // Note: Both TagId and BlobId are aliases for chi::UniqueId, so we register the base type
  auto unique_id_class = nb::class_<wrp_cte::core::TagId>(m, "UniqueId")
      .def(nb::init<>())
      .def_static("GetNull", &wrp_cte::core::TagId::GetNull)
      .def("ToU64", &wrp_cte::core::TagId::ToU64)
      .def("IsNull", &wrp_cte::core::TagId::IsNull)
      .def_rw("major_", &wrp_cte::core::TagId::major_)
      .def_rw("minor_", &wrp_cte::core::TagId::minor_);

  // Create aliases for TagId and BlobId
  m.attr("TagId") = unique_id_class;
  m.attr("BlobId") = unique_id_class;

  // Note: Timestamp (chrono time_point) is automatically handled by
  // nanobind/stl/chrono.h

  // Bind MemContext for method calls
  nb::class_<hipc::MemContext>(m, "MemContext")
      .def(nb::init<>());

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
         "Query blobs by tag and blob regex patterns, returns vector of (tag_name, blob_name) pairs");

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