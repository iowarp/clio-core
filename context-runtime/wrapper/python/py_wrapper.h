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

#ifndef PY_WRAPPER_H_
#define PY_WRAPPER_H_

#include <chimaera/admin/admin_client.h>
#include <chimaera/admin/admin_tasks.h>
#include <chimaera/chimaera.h>

#include <string>
#include <thread>
#include <unordered_map>

/**
 * Initialize the Chimaera runtime from Python.
 *
 * Runs CHIMAERA_INIT on a dedicated background thread so that the
 * ZMQ I/O threads it spawns never touch the calling (Python) thread's
 * GIL state.  The caller blocks until initialization is complete.
 *
 * @param mode ChimaeraMode integer (0 = kClient)
 * @return true if initialization succeeded
 */
inline bool py_chimaera_init(int mode) {
  bool result = false;
  std::thread([&result, mode]() {
    result = chi::CHIMAERA_INIT(
        static_cast<chi::ChimaeraMode>(mode), false, false);
  }).join();
  return result;
}

/**
 * Finalize the Chimaera runtime.
 *
 * Closes ZMQ sockets and joins background threads.
 */
inline void py_chimaera_finalize() {
  chi::CHIMAERA_FINALIZE();
}

/**
 * Python-visible wrapper around a MonitorTask future.
 *
 * Owns the chi::Future and exposes a blocking wait() that returns
 * the result map and frees the underlying C++ task.
 */
class PyMonitorTask {
  chi::Future<chimaera::admin::MonitorTask> future_;

 public:
  /** @param f Moved-from future returned by AsyncMonitor */
  explicit PyMonitorTask(chi::Future<chimaera::admin::MonitorTask>&& f)
      : future_(std::move(f)) {}

  PyMonitorTask(const PyMonitorTask&) = delete;
  PyMonitorTask& operator=(const PyMonitorTask&) = delete;
  PyMonitorTask(PyMonitorTask&&) = default;
  PyMonitorTask& operator=(PyMonitorTask&&) = default;

  /**
   * Block until the monitor query completes, return results, free the task.
   * @return map of container-id to serialized result blob
   */
  std::unordered_map<chi::ContainerId, std::string> wait() {
    bool ok = future_.Wait();
    if (!ok) {
      // Recv() failed (server dead, timeout, etc.)
      // Wait() already set return_code_ to -1 on the task.
      return {};
    }
    auto results = future_->results_;
    return results;
  }

  /**
   * Get the return code from the underlying task.
   * Call after wait() to check if the task succeeded.
   * @return 0 on success, non-zero on error
   */
  uint32_t get_return_code() {
    return future_->GetReturnCode();
  }
};

/**
 * Submit an asynchronous monitor query.
 *
 * @param pool_query_str Pool query string (e.g. "local", "broadcast")
 * @param query          Free-form query string (e.g. "status", "worker_stats")
 * @return PyMonitorTask whose wait() returns the result map
 */
inline PyMonitorTask py_async_monitor(const std::string& pool_query_str,
                                      const std::string& query) {
  auto* admin = CHI_ADMIN;
  chi::PoolQuery pq = chi::PoolQuery::FromString(pool_query_str);
  auto future = admin->AsyncMonitor(pq, query);
  return PyMonitorTask(std::move(future));
}


#endif  // PY_WRAPPER_H_
