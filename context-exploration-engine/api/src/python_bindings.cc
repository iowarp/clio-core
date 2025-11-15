#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <wrp_cee/api/context_interface.h>
#include <wrp_cae/core/factory/assimilation_ctx.h>

namespace nb = nanobind;

NB_MODULE(wrp_cee, m) {
  m.doc() = "IOWarp Context Exploration Engine API - Python Bindings";

  // Bind AssimilationCtx struct
  nb::class_<wrp_cae::core::AssimilationCtx>(m, "AssimilationCtx",
      "Context for data assimilation operations")
    .def(nb::init<>(),
         "Default constructor")
    .def(nb::init<const std::string&, const std::string&, const std::string&,
                  const std::string&, size_t, size_t, const std::string&, const std::string&>(),
         nb::arg("src"), nb::arg("dst"), nb::arg("format"),
         nb::arg("depends_on") = "", nb::arg("range_off") = 0, nb::arg("range_size") = 0,
         nb::arg("src_token") = "", nb::arg("dst_token") = "",
         "Full constructor")
    .def_rw("src", &wrp_cae::core::AssimilationCtx::src,
            "Source URL (e.g., file::/path/to/file)")
    .def_rw("dst", &wrp_cae::core::AssimilationCtx::dst,
            "Destination URL (e.g., iowarp::tag_name)")
    .def_rw("format", &wrp_cae::core::AssimilationCtx::format,
            "Data format (e.g., binary, hdf5)")
    .def_rw("depends_on", &wrp_cae::core::AssimilationCtx::depends_on,
            "Dependency identifier (empty if none)")
    .def_rw("range_off", &wrp_cae::core::AssimilationCtx::range_off,
            "Byte offset in source file")
    .def_rw("range_size", &wrp_cae::core::AssimilationCtx::range_size,
            "Number of bytes to read")
    .def_rw("src_token", &wrp_cae::core::AssimilationCtx::src_token,
            "Authentication token for source")
    .def_rw("dst_token", &wrp_cae::core::AssimilationCtx::dst_token,
            "Authentication token for destination")
    .def("__repr__", [](const wrp_cae::core::AssimilationCtx& ctx) {
      return "<AssimilationCtx src='" + ctx.src + "' dst='" + ctx.dst +
             "' format='" + ctx.format + "'>";
    });

  // Bind ContextInterface class
  // C++ uses PascalCase (Google style), Python exposes snake_case
  nb::class_<iowarp::ContextInterface>(m, "ContextInterface",
      "High-level API for context exploration and management")
    .def(nb::init<>(),
         "Default constructor - initializes the interface")
    .def("context_bundle", &iowarp::ContextInterface::ContextBundle,
         nb::arg("bundle"),
         "Bundle a group of related objects together and assimilate them\n\n"
         "Parameters:\n"
         "  bundle: List of AssimilationCtx objects to assimilate\n\n"
         "Returns:\n"
         "  0 on success, non-zero error code on failure")
    .def("context_query", &iowarp::ContextInterface::ContextQuery,
         nb::arg("tag_re"), nb::arg("blob_re"), nb::arg("max_results") = 0,
         "Retrieve the identities of objects matching tag and blob patterns\n\n"
         "Parameters:\n"
         "  tag_re: Tag regex pattern to match\n"
         "  blob_re: Blob regex pattern to match\n"
         "  max_results: Maximum number of results to return (0 = unlimited, default: 0)\n\n"
         "Returns:\n"
         "  List of matching blob names")
    .def("context_retrieve", &iowarp::ContextInterface::ContextRetrieve,
         nb::arg("tag_re"), nb::arg("blob_re"),
         nb::arg("max_results") = 1024,
         nb::arg("max_context_size") = 256 * 1024 * 1024,
         nb::arg("batch_size") = 32,
         "Retrieve the identities and data of objects matching patterns\n\n"
         "Queries for blobs matching patterns and retrieves their data into a\n"
         "packed binary buffer. Blobs are retrieved in batches for efficiency.\n\n"
         "Parameters:\n"
         "  tag_re: Tag regex pattern to match\n"
         "  blob_re: Blob regex pattern to match\n"
         "  max_results: Max number of blobs (0=unlimited, default: 1024)\n"
         "  max_context_size: Max total size in bytes (default: 256MB)\n"
         "  batch_size: Concurrent AsyncGetBlob operations (default: 32)\n\n"
         "Returns:\n"
         "  List with one string containing packed binary context data (empty if none)")
    .def("context_splice", &iowarp::ContextInterface::ContextSplice,
         nb::arg("new_ctx"), nb::arg("tag_re"), nb::arg("blob_re"),
         "Split/splice objects into a new context (NOT YET IMPLEMENTED)\n\n"
         "Parameters:\n"
         "  new_ctx: Name of the new context to create\n"
         "  tag_re: Tag regex pattern to match for source objects\n"
         "  blob_re: Blob regex pattern to match for source objects\n\n"
         "Returns:\n"
         "  0 on success, non-zero error code on failure")
    .def("context_destroy", &iowarp::ContextInterface::ContextDestroy,
         nb::arg("context_names"),
         "Destroy contexts by name\n\n"
         "Parameters:\n"
         "  context_names: List of context names to destroy\n\n"
         "Returns:\n"
         "  0 on success, non-zero error code on failure");
}
