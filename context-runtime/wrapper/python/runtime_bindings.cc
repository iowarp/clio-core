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
    const std::unordered_map<chi::ContainerId, std::string>& results) {
  nb::dict d;
  for (const auto& [k, v] : results) {
    d[nb::int_(k)] = nb::bytes(v.data(), v.size());
  }
  return d;
}

NB_MODULE(chimaera_runtime_ext, m) {
  m.doc() = "Python bindings for Chimaera runtime monitoring";

  m.def("chimaera_init", &py_chimaera_init, "mode"_a,
        "Initialize the Chimaera runtime. mode: 0=kClient.");

  m.def("chimaera_finalize", &py_chimaera_finalize,
        "Finalize the Chimaera runtime.");

  nb::class_<PyMonitorTask>(m, "MonitorTask")
      .def("wait", [](PyMonitorTask& self) -> nb::dict {
        return results_to_dict(self.wait());
      }, "Block until result is ready, return {container_id: bytes} dict.")
      .def("get_return_code", &PyMonitorTask::get_return_code,
           "Get task return code. Call after wait(). 0=success, non-zero=error.");

  m.def("async_monitor", &py_async_monitor,
        "pool_query"_a, "query"_a,
        "Submit async monitor query. Returns MonitorTask.");

}
