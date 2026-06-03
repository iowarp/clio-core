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
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <clio_cee/api/context_interface.h>
#include <clio_cae/core/factory/assimilation_ctx.h>

namespace nb = nanobind;

NB_MODULE(clio_cee, m) {
  m.doc() = "IOWarp Context Exploration Engine API - Python Bindings";

  // Bind AssimilationCtx struct
  nb::class_<clio::cae::core::AssimilationCtx>(m, "AssimilationCtx",
      "Context for data assimilation operations")
    .def(nb::init<>(),
         "Default constructor")
    .def(nb::init<const std::string&, const std::string&, const std::string&,
                  const std::string&, size_t, size_t, const std::string&, const std::string&>(),
         nb::arg("src"), nb::arg("dst"), nb::arg("format"),
         nb::arg("depends_on") = "", nb::arg("range_off") = 0, nb::arg("range_size") = 0,
         nb::arg("src_token") = "", nb::arg("dst_token") = "",
         "Full constructor")
    .def_rw("src", &clio::cae::core::AssimilationCtx::src,
            "Source URL (e.g., file::/path/to/file)")
    .def_rw("dst", &clio::cae::core::AssimilationCtx::dst,
            "Destination URL (e.g., iowarp::tag_name)")
    .def_rw("format", &clio::cae::core::AssimilationCtx::format,
            "Data format (e.g., binary, hdf5)")
    .def_rw("depends_on", &clio::cae::core::AssimilationCtx::depends_on,
            "Dependency identifier (empty if none)")
    .def_rw("range_off", &clio::cae::core::AssimilationCtx::range_off,
            "Byte offset in source file")
    .def_rw("range_size", &clio::cae::core::AssimilationCtx::range_size,
            "Number of bytes to read")
    .def_rw("src_token", &clio::cae::core::AssimilationCtx::src_token,
            "Authentication token for source")
    .def_rw("dst_token", &clio::cae::core::AssimilationCtx::dst_token,
            "Authentication token for destination")
    .def_rw("src_data", &clio::cae::core::AssimilationCtx::src_data,
            "Inline payload bytes for src='string::<blob_name>' sources. "
            "Stored verbatim as a single blob named <blob_name>.")
    .def("__repr__", [](const clio::cae::core::AssimilationCtx& ctx) {
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
         nb::arg("tag_re"), nb::arg("blob_re"),
         nb::arg("max_results") = 0,
         nb::arg("prompt") = "",
         nb::arg("time_begin") = uint64_t{0},
         nb::arg("time_end") = uint64_t{0},
         "Retrieve the identities of objects matching tag and blob patterns\n\n"
         "Three modes, in priority order:\n"
         "  1. Temporal (time_begin or time_end non-zero): returns blobs\n"
         "     whose last_modified falls within [time_begin, time_end]\n"
         "     (epoch nanoseconds; 0 = no bound), sorted ascending by\n"
         "     timestamp, capped at max_results.\n"
         "  2. Semantic (prompt non-empty): BM25 keyword search over the\n"
         "     regex-filtered candidate blobs; top max_results returned\n"
         "     by descending score (0 falls back to top-10).\n"
         "  3. Regex (default): BlobQuery listing all matching blob names.\n\n"
         "Parameters:\n"
         "  tag_re: Tag regex pattern to match\n"
         "  blob_re: Blob regex pattern to match\n"
         "  max_results: Result cap (0 = unlimited for regex/temporal; top-10 for BM25)\n"
         "  prompt: Optional BM25 query text — empty disables semantic search\n"
         "  time_begin: Temporal lower bound, epoch nanoseconds (0 = no bound)\n"
         "  time_end: Temporal upper bound, epoch nanoseconds (0 = no bound)\n\n"
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
