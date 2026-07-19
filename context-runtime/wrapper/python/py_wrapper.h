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

#include <clio_runtime/admin/admin_client.h>
#include <clio_runtime/admin/admin_tasks.h>
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/config_manager.h>
#include <clio_runtime/safe_bdev/safe_bdev_client.h>

#include <atomic>
#include <string>
#include <thread>
#include <unordered_map>

/**
 * Initialize the CLIO Runtime runtime from Python.
 *
 * Runs CLIO_INIT on a dedicated background thread so that the
 * ZMQ I/O threads it spawns never touch the calling (Python) thread's
 * GIL state.  The caller blocks until initialization is complete.
 *
 * @param mode RuntimeMode integer (0 = kClient)
 * @return true if initialization succeeded
 */
inline bool py_clio_init(int mode) {
  bool result = false;
  std::thread([&result, mode]() {
    result = clio::run::CLIO_INIT(
        static_cast<clio::run::RuntimeMode>(mode), false, false);
  }).join();
  return result;
}

/**
 * Finalize the CLIO Runtime runtime.
 *
 * Closes ZMQ sockets and joins background threads.
 */
inline void py_clio_finalize() {
  clio::run::CLIO_RUNTIME_FINALIZE();
}

/**
 * Python-visible wrapper around a MonitorTask future.
 *
 * Owns the clio::run::Future and exposes a blocking wait() that returns
 * the result map and frees the underlying C++ task.
 */
class PyMonitorTask {
  clio::run::Future<clio::run::admin::MonitorTask> future_;

 public:
  /** @param f Moved-from future returned by AsyncMonitor */
  explicit PyMonitorTask(clio::run::Future<clio::run::admin::MonitorTask>&& f)
      : future_(std::move(f)) {}

  PyMonitorTask(const PyMonitorTask&) = delete;
  PyMonitorTask& operator=(const PyMonitorTask&) = delete;
  PyMonitorTask(PyMonitorTask&&) = default;
  PyMonitorTask& operator=(PyMonitorTask&&) = default;

  /**
   * Block until the monitor query completes, return results, free the task.
   *
   * IMPORTANT: The caller must release the Python GIL before calling this
   * because Wait() calls IpcManager::Recv() which can block for seconds
   * on ZMQ I/O (reconnection, dead nodes).  Holding the GIL here would
   * freeze the entire Python process (Flask event loop, timeout checks, etc.).
   *
   * @param max_sec Maximum seconds to wait (0 = wait indefinitely)
   * @return map of container-id to serialized result blob
   */
  std::unordered_map<clio::run::ContainerId, std::string> wait(float max_sec = 0) {
    bool ok = future_.Wait(max_sec);
    if (!ok) {
      // Recv() failed (server dead, timeout, etc.)
      return {};
    }
    auto results = future_->results_;
    return results;
  }

  /**
   * Non-blocking check if the task has completed.
   *
   * Just reads a flag in shared memory — no IPC, no blocking.
   * Note: the task only becomes complete after Recv() processes the
   * response, so this is useful for checking progress while wait()
   * runs on another thread.
   *
   * @return true if the task is complete
   */
  bool is_complete() const {
    return future_.IsComplete();
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
  auto* admin = CLIO_ADMIN;
  clio::run::PoolQuery pq = clio::run::PoolQuery::FromString(pool_query_str);
  auto future = admin->AsyncMonitor(pq, query);
  return PyMonitorTask(std::move(future));
}


/**
 * Send an asynchronous stop-runtime command to a node.
 *
 * This is the Python equivalent of:
 *   admin_client.AsyncStopRuntime(pool_query, flags, grace_period_ms);
 *
 * The call is fire-and-forget: we submit the task but do not wait for
 * a reply because the target runtime dies before it can respond.
 *
 * @param pool_query_str  Pool query string (e.g. "physical:3", "local")
 * @param grace_period_ms Milliseconds to wait before forced shutdown
 */
inline void py_stop_runtime(const std::string& pool_query_str,
                            uint32_t grace_period_ms = 5000) {
  auto* admin = CLIO_ADMIN;
  clio::run::PoolQuery pq = clio::run::PoolQuery::FromString(pool_query_str);
  admin->AsyncStopRuntime(pq, 0, grace_period_ms);
}

//============================================================================
// Safe-bdev member management (dashboard add / remove / replace controls).
//
// AddBdev and RecoverBdev both require the NEW member bdev pool to already
// exist, so add/replace first compose a fresh file-backed clio_bdev pool and
// then attach/recover onto it. Fresh pool ids come from a high monotonic
// counter to avoid clashing with config-declared ids.
//============================================================================

inline std::string py_pool_id_str(const clio::run::PoolId& id) {
  return std::to_string(id.major_) + "." + std::to_string(id.minor_);
}

/** Compose a fresh file-backed clio_bdev pool at `path`; return its PoolId. */
inline clio::run::PoolId py_compose_bdev_pool(const std::string& path,
                                              const std::string& capacity) {
  static std::atomic<uint32_t> next_major{40000};
  clio::run::PoolId new_id(next_major.fetch_add(1), 0);
  clio::run::PoolConfig pc;
  pc.mod_name_ = "clio_bdev";
  pc.pool_name_ = path;
  pc.pool_id_ = new_id;
  pc.pool_query_ = clio::run::PoolQuery::Local();
  pc.config_ = "bdev_type: file\ncapacity: " + capacity + "\nalloc_log: " +
               path + ".alog\n";
  auto fut = CLIO_ADMIN->AsyncCompose(pc);
  fut.Wait();
  return new_id;
}

/** Mark a member faulty / unlink it. Returns the task return code (0 = ok). */
inline uint32_t py_safe_bdev_remove_bdev(const std::string& safe_pool_id_str,
                                         const std::string& target_pool_id_str,
                                         uint32_t was_faulty) {
  clio::run::safe_bdev::Client safe(
      clio::run::PoolId::FromString(safe_pool_id_str));
  auto fut = safe.AsyncRemoveBdev(
      clio::run::PoolQuery::Dynamic(),
      clio::run::PoolId::FromString(target_pool_id_str), was_faulty);
  fut.Wait();
  return fut->GetReturnCode();
}

/** Grow the array: compose a fresh bdev and attach it (data or parity).
 *  Returns the new member's "major.minor" pool id, or "" on failure. */
inline std::string py_safe_bdev_add_bdev(const std::string& safe_pool_id_str,
                                         const std::string& member_path,
                                         const std::string& capacity,
                                         uint32_t node_id, uint32_t as_parity) {
  clio::run::PoolId member_id = py_compose_bdev_pool(member_path, capacity);
  clio::run::safe_bdev::Client safe(
      clio::run::PoolId::FromString(safe_pool_id_str));
  auto fut = safe.AsyncAddBdev(clio::run::PoolQuery::Dynamic(), member_path,
                               node_id, member_id, as_parity);
  fut.Wait();
  return (fut->GetReturnCode() == 0) ? py_pool_id_str(member_id) : std::string();
}

/** Replace a failed member: mark it faulty, compose a fresh bdev, and recover
 *  the lost shards onto it (the "add another to trigger recovery" flow).
 *  Returns the replacement's pool id, or "" on failure. */
inline std::string py_safe_bdev_replace_bdev(
    const std::string& safe_pool_id_str, const std::string& failed_pool_id_str,
    const std::string& member_path, const std::string& capacity,
    uint32_t node_id) {
  clio::run::safe_bdev::Client safe(
      clio::run::PoolId::FromString(safe_pool_id_str));
  clio::run::PoolId failed = clio::run::PoolId::FromString(failed_pool_id_str);
  // 1. Mark the failed member faulty (idempotent if already faulty).
  {
    auto rm =
        safe.AsyncRemoveBdev(clio::run::PoolQuery::Dynamic(), failed, 1);
    rm.Wait();
  }
  // 2. Compose the replacement bdev.
  clio::run::PoolId new_id = py_compose_bdev_pool(member_path, capacity);
  // 3. Reconstruct onto it. Recovery progress is now visible via Monitor.
  auto rec = safe.AsyncRecoverBdev(clio::run::PoolQuery::Dynamic(), failed,
                                   member_path, node_id, new_id);
  rec.Wait();
  return (rec->GetReturnCode() == 0) ? py_pool_id_str(new_id) : std::string();
}

#endif  // PY_WRAPPER_H_
