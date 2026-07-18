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

#include "py_wrapper.h"

namespace nb = nanobind;
using namespace nb::literals;

/**
 * Convert C++ results map to Python dict {int: bytes}.
 *
 * Values are returned as bytes because monitor results may contain
 * binary-serialized data (e.g. msgpack blobs).
 */
static nb::dict results_to_dict(
    const std::unordered_map<clio::run::ContainerId, std::string>& results) {
  nb::dict d;
  for (const auto& [k, v] : results) {
    d[nb::int_(k)] = nb::bytes(v.data(), v.size());
  }
  return d;
}

NB_MODULE(clio_runtime_ext, m) {
  m.doc() = "Python bindings for Clio runtime monitoring";

  m.def("clio_init", &py_clio_init, "mode"_a,
        "Initialize the Clio runtime. mode: 0=kClient.");

  m.def("clio_finalize", &py_clio_finalize,
        "Finalize the Clio runtime.");

  nb::class_<PyMonitorTask>(m, "MonitorTask")
      .def("wait", [](PyMonitorTask& self, float max_sec) -> nb::dict {
        // Release the GIL so Flask / timeout threads can run while
        // the C++ Wait() blocks on ZMQ Recv().
        std::unordered_map<clio::run::ContainerId, std::string> results;
        {
          nb::gil_scoped_release release;
          results = self.wait(max_sec);
        }
        return results_to_dict(results);
      }, "max_sec"_a = 0.0f,
      "Block until result is ready, return {container_id: bytes} dict.")
      .def("is_complete", &PyMonitorTask::is_complete,
           "Non-blocking check if the task has completed.")
      .def("get_return_code", &PyMonitorTask::get_return_code,
           "Get task return code. Call after wait(). 0=success, non-zero=error.");

  m.def("async_monitor", &py_async_monitor,
        "pool_query"_a, "query"_a,
        "Submit async monitor query. Returns MonitorTask.");

  m.def("stop_runtime", [](const std::string& pool_query_str,
                            uint32_t grace_period_ms) {
    nb::gil_scoped_release release;
    py_stop_runtime(pool_query_str, grace_period_ms);
  }, "pool_query"_a, "grace_period_ms"_a = 5000,
     "Send stop-runtime command to a node. Fire-and-forget.");

  // ---- Safe-bdev member management (dashboard controls) -------------------
  m.def("safe_bdev_remove_bdev",
        [](const std::string& safe_pool_id, const std::string& target_pool_id,
           uint32_t was_faulty) -> uint32_t {
          nb::gil_scoped_release release;
          return py_safe_bdev_remove_bdev(safe_pool_id, target_pool_id,
                                          was_faulty);
        },
        "safe_pool_id"_a, "target_pool_id"_a, "was_faulty"_a = 1,
        "Remove a member from a safe_bdev. Returns task return code.");

  m.def("safe_bdev_add_bdev",
        [](const std::string& safe_pool_id, const std::string& member_path,
           const std::string& capacity, uint32_t node_id,
           uint32_t as_parity) -> std::string {
          nb::gil_scoped_release release;
          return py_safe_bdev_add_bdev(safe_pool_id, member_path, capacity,
                                       node_id, as_parity);
        },
        "safe_pool_id"_a, "member_path"_a, "capacity"_a = "256MB",
        "node_id"_a = 0, "as_parity"_a = 0,
        "Compose a fresh bdev and add it to a safe_bdev. Returns new pool id.");

  m.def("safe_bdev_replace_bdev",
        [](const std::string& safe_pool_id, const std::string& failed_pool_id,
           const std::string& member_path, const std::string& capacity,
           uint32_t node_id) -> std::string {
          nb::gil_scoped_release release;
          return py_safe_bdev_replace_bdev(safe_pool_id, failed_pool_id,
                                           member_path, capacity, node_id);
        },
        "safe_pool_id"_a, "failed_pool_id"_a, "member_path"_a,
        "capacity"_a = "256MB", "node_id"_a = 0,
        "Replace a failed member and auto-recover onto it. Returns new pool id.");
}
