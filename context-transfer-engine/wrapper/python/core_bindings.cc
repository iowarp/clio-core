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

#include <functional>
#include <stdexcept>

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
#include <mutex>
#include <memory>
#include <deque>

namespace nb = nanobind;
using namespace nb::literals;

namespace {
/** A copied capacity snapshot; values are bytes, not filesystem free space. */
struct PyCapacityInfo {
  uint64_t total_capacity;
  uint64_t remaining_capacity;
};

/** Copied native target statistics; no TTL or placement eligibility is implied. */
struct PyTargetInfo {
  float target_score;
  uint64_t remaining_space;
  uint64_t bytes_read;
  uint64_t bytes_written;
  uint64_t ops_read;
  uint64_t ops_written;
};

/**
 * Reject failed diagnostic queries instead of reporting their zeroed outputs.
 * @param task the completed native task
 * @param operation the Python operation name used in the error message
 * @throws std::runtime_error if no task or a nonzero return code was received
 */
template <typename TaskT>
void RequireQuerySuccess(TaskT *task, const char *operation) {
  if (task == nullptr) {
    throw std::runtime_error(std::string(operation) + " returned no task");
  }
  const auto rc = task->GetReturnCode();
  if (rc != 0) {
    throw std::runtime_error(std::string(operation) +
                             " failed with return code " + std::to_string(rc));
  }
}

// Python-facing handle for an in-flight CTE task (async client APIs).
//
// Type-erased over the concrete task, but every operation runs against the
// CONCRETE Future<TaskT> captured by MakePyFuture. That matters: Future::Wait
// deserializes the reply through TaskT's SerializeOut and runs TaskT's
// PostWait, and neither is virtual. Waiting on a Future<Task> ran the base
// versions, so query results never arrived (AsyncTagQuery/AsyncBlobQuery
// returned no rows) and a client-mode GetBlob miss never copied its staged
// bytes out (#1096).
//
// Every wait releases the GIL, so a stalled runtime only blocks the calling
// thread (#1096, #793). `extract_` builds the Python result and always runs
// with the GIL held; host buffers the task reads or writes are kept alive by
// being captured in it.
struct PyCteFuture {
  std::function<bool(float)> wait_;  /**< typed Wait; true once complete */
  std::function<bool()> done_;       /**< typed non-blocking completion poll */
  std::function<int()> rc_;          /**< return code; valid once complete */
  std::function<nb::object()> extract_;  /**< Python result (GIL held) */
  bool completed_ = false;           /**< a wait already returned true */

  /**
   * Wait without the GIL.
   * @param max_sec < 0 waits forever, 0 polls once, > 0 is a deadline
   * @return true if the task completed within max_sec
   */
  bool WaitNoGil(float max_sec) {
    if (completed_) return true;
    bool ok;
    {
      nb::gil_scoped_release release;
      ok = wait_(max_sec);
    }
    completed_ = ok;
    return ok;
  }

  /**
   * Block until the task completes, honouring max_sec.
   * @param max_sec < 0 waits forever, 0 polls once
   * @return the task return code (0 = success), or None if max_sec elapsed
   *         first (including when the runtime is gone)
   */
  nb::object Wait(float max_sec) {
    if (!WaitNoGil(max_sec)) return nb::none();
    return nb::int_(rc_());
  }

  /** Non-blocking completion poll. */
  bool Done() { return completed_ || done_(); }

  /**
   * Block, then produce the operation's Python result: bytes for GetBlob, a
   * list or copied snapshot for queries, None for status-only ops.
   * @param max_sec < 0 waits forever
   * @return the result object
   * @throws TimeoutError if max_sec elapsed first
   */
  nb::object Result(float max_sec) {
    if (!WaitNoGil(max_sec)) {
      PyErr_SetString(PyExc_TimeoutError, "CTE future did not complete within max_sec");
      throw nb::python_error();
    }
    return extract_ ? extract_() : nb::none();
  }
};

/**
 * Wrap a concrete future for Python.
 * @param fut the typed future returned by a Client Async* call
 * @param extract builds the Python result from the completed task (null for
 *        an empty, already-satisfied future); omit for status-only ops
 * @return the Python-facing handle
 */
template <typename TaskT>
PyCteFuture MakePyFuture(clio::run::Future<TaskT> fut,
                         std::function<nb::object(TaskT *)> extract = {}) {
  auto sp = std::make_shared<clio::run::Future<TaskT>>(std::move(fut));
  PyCteFuture pf;
  pf.wait_ = [sp](float max_sec) {
    return sp->IsNull() ? true : sp->Wait(max_sec);
  };
  pf.done_ = [sp]() { return sp->IsNull() || sp->IsComplete(); };
  pf.rc_ = [sp]() {
    TaskT *t = sp->get();
    return t ? static_cast<int>(t->GetReturnCode()) : 0;
  };
  if (extract) {
    pf.extract_ = [sp, extract]() { return extract(sp->get()); };
  }
  return pf;
}

/**
 * Submit an existing scalar owner-conditional put from immutable Python bytes.
 * @param self the CTE client whose pool owns the blob
 * @param tag_id blob namespace
 * @param blob_name blob key
 * @param data nonempty immutable payload, retained through completion
 * @param off byte offset within the blob (puts do not truncate)
 * @param op_flag Context::kPutIfAbsent or Context::kPutIfVersion
 * @param version expected version for a compare-and-swap put
 * @return typed Python future with native code and returned/new version
 */
PyCteFuture ConditionalPut(clio::cte::core::Client &self,
                           const clio::cte::core::TagId &tag_id,
                           const std::string &blob_name, nb::bytes data,
                           size_t off, clio::run::u32 op_flag,
                           clio::run::u64 version) {
  if (data.size() == 0) {
    throw std::invalid_argument("conditional puts require nonempty data");
  }
  const char *ptr = data.c_str();
  const size_t len = data.size();
  clio::cte::core::Context context;
  context.op_flags_ = op_flag;
  context.version_ = version;
  using TaskT = clio::cte::core::PutBlobTask;
  clio::run::Future<TaskT> fut;
  {
    nb::gil_scoped_release release;
    fut = self.AsyncPutBlob(tag_id, blob_name, off, len, ptr, -1.0f, context);
  }
  if (fut.IsNull()) {
    throw std::runtime_error("conditional put submission failed");
  }
  return MakePyFuture<TaskT>(std::move(fut), [data](TaskT *task) {
    nb::dict result;
    result["code"] = task->GetReturnCode();
    result["version"] = task->context_.version_;
    return result;
  });
}

/**
 * Send an authoritative get, bypassing the unversioned SHM read fast path.
 * @param self CTE client providing owner routing through its pool
 * @param tag_id blob namespace
 * @param blob_name blob key
 * @param size number of bytes to read
 * @param off byte offset within the blob
 * @param dst retained private destination for the typed PostWait hook
 * @return existing native GetBlobTask future, with context.version_ populated
 */
clio::run::Future<clio::cte::core::GetBlobTask> SubmitVersionedGet(
    clio::cte::core::Client &self, const clio::cte::core::TagId &tag_id,
    const std::string &blob_name, size_t size, size_t off, char *dst) {
  using TaskT = clio::cte::core::GetBlobTask;
  auto *ipc = CLIO_CPU_IPC;
  auto *runtime = CLIO_RUNTIME_MANAGER;
  clio::cte::core::Context context;
  context.op_flags_ = clio::cte::core::Context::kNoFault;
  if (runtime->IsRuntime()) {
    auto task =
        ipc->NewTask<TaskT>(clio::run::CreateTaskId(), self.pool_id_,
                            clio::run::PoolQuery::Dynamic(), tag_id, blob_name,
                            off, size, 0u, ctp::ipc::ShmPtr<>::FromRaw(dst),
                            context);
    return ipc->Send(task);
  }
  auto staging = ipc->AllocateBuffer(size);
  if (staging.IsNull()) {
    throw std::runtime_error("versioned get staging allocation failed");
  }
  auto task = ipc->NewTask<TaskT>(
      clio::run::CreateTaskId(), self.pool_id_, clio::run::PoolQuery::Dynamic(),
      tag_id, blob_name, off, size, 0u, ctp::ipc::ShmPtr<>(staging.shm_), context);
  task->priv_dest_ = dst;
  task->priv_src_ = staging.ptr_;
  auto fut = ipc->Send(task);
  // Match AsyncGetBlob(char*): ownership stays client-local, after Send has
  // serialized flags, and typed Future::Wait delivers bytes via PostWait.
  task->SetFlags(TASK_DATA_OWNER);
  return fut;
}

/**
 * Wrap an authoritative get and its retained private buffer for Python.
 * @param self CTE client
 * @param tag_id blob namespace
 * @param blob_name blob key
 * @param size positive read size, normally a fixed-size coordination record
 * @param off byte offset within the blob
 * @return code, version and bytes (None on a failed read) via Future.result()
 */
PyCteFuture VersionedGet(clio::cte::core::Client &self,
                         const clio::cte::core::TagId &tag_id,
                         const std::string &blob_name, size_t size,
                         size_t off) {
  if (size == 0) {
    throw std::invalid_argument("versioned gets require positive data_size");
  }
  auto buf = std::make_shared<std::string>(size, '\0');
  clio::run::Future<clio::cte::core::GetBlobTask> fut;
  {
    nb::gil_scoped_release release;
    fut = SubmitVersionedGet(self, tag_id, blob_name, size, off, buf->data());
  }
  if (fut.IsNull()) {
    throw std::runtime_error("versioned get submission failed");
  }
  using TaskT = clio::cte::core::GetBlobTask;
  return MakePyFuture<TaskT>(std::move(fut), [buf](TaskT *task) {
    nb::dict result;
    const auto code = task->GetReturnCode();
    result["code"] = code;
    result["version"] = task->context_.version_;
    result["data"] = code == 0 ? nb::object(nb::bytes(buf->data(), buf->size()))
                               : nb::object(nb::none());
    return result;
  });
}

}  // namespace

NB_MODULE(clio_cte_core_ext, m) {
  m.doc() = "Python bindings for WRP CTE Core";

  m.attr("PUT_EXISTS_RC") = clio::cte::core::kPutExistsRc;
  m.attr("PUT_VERSION_MISMATCH_RC") = clio::cte::core::kPutVersionMismatchRc;

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

  // Bind RuntimeMode enum
  nb::enum_<clio::run::RuntimeMode>(m, "RuntimeMode")
      .value("kClient", clio::run::RuntimeMode::kClient)
      .value("kServer", clio::run::RuntimeMode::kServer)
      .value("kRuntime", clio::run::RuntimeMode::kRuntime);

  // Bind UniqueId type (used by TagId, BlobId, and PoolId)
  // Note: TagId, BlobId, and PoolId are all aliases for clio::run::UniqueId, so we register the base type
  auto unique_id_class = nb::class_<clio::cte::core::TagId>(m, "UniqueId")
      .def(nb::init<>())
      .def(nb::init<clio::run::u32, clio::run::u32>(), "major"_a, "minor"_a,
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
  nb::class_<clio::run::PoolQuery>(m, "PoolQuery")
      .def(nb::init<>())
      .def_static("Broadcast", &clio::run::PoolQuery::Broadcast,
                  "net_timeout"_a = -1.0f,
                  "Create a Broadcast pool query (routes to all nodes)")
      .def_static("Dynamic", &clio::run::PoolQuery::Dynamic,
                  "net_timeout"_a = -1.0f,
                  "Create a Dynamic pool query (automatic routing optimization)")
      .def_static("Local", &clio::run::PoolQuery::Local,
                  "parallelism"_a = 32,
                  "Create a Local pool query (routes to local node only). "
                  "parallelism defaults to 32 — matches C++ "
                  "clio::run::PoolQuery::Local default");

  nb::class_<PyCapacityInfo>(m, "CapacityInfo")
      .def_ro("total_capacity", &PyCapacityInfo::total_capacity,
              "Summed configured target capacity in bytes.")
      .def_ro("remaining_capacity", &PyCapacityInfo::remaining_capacity,
              "Native remaining capacity in bytes, including evictable cache.");

  nb::class_<PyTargetInfo>(m, "TargetInfo")
      .def_ro("target_score", &PyTargetInfo::target_score)
      .def_ro("remaining_space", &PyTargetInfo::remaining_space,
              "Native remaining target space in bytes.")
      .def_ro("bytes_read", &PyTargetInfo::bytes_read)
      .def_ro("bytes_written", &PyTargetInfo::bytes_written)
      .def_ro("ops_read", &PyTargetInfo::ops_read)
      .def_ro("ops_written", &PyTargetInfo::ops_written);

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
  // Async handle returned by every Client.Async*() method. Wait for the task,
  // poll it, or fetch its result — while overlapping other work in between.
  nb::class_<PyCteFuture>(m, "Future")
      .def("wait", &PyCteFuture::Wait, "max_sec"_a = -1.0f,
           "Block until the task completes; returns its return code "
           "(0 = success), or None if max_sec elapsed first -- including "
           "when the runtime is gone. max_sec < 0 waits forever, 0 polls "
           "once. The GIL is released while waiting.")
      .def("done", &PyCteFuture::Done,
           "Non-blocking: True if the task has completed.")
      .def("result", &PyCteFuture::Result, "max_sec"_a = -1.0f,
           "Block until done, then return the operation's result: bytes for "
           "AsyncGetBlob, a dict for conditional puts/versioned gets, "
           "a list or snapshot for query operations, None for "
           "status-only ops (put/del/reorganize/register). Raises "
           "TimeoutError if max_sec (> 0) elapses first. The GIL is released "
           "while waiting.");

  nb::class_<clio::cte::core::Client>(m, "Client")
      .def(nb::init<>())
      .def(nb::init<const clio::run::PoolId &>())
      .def("PollTelemetryLog",
          [](clio::cte::core::Client &self, std::uint64_t minimum_logical_time) {
            auto task = self.AsyncPollTelemetryLog(minimum_logical_time);
            {
              nb::gil_scoped_release release;
              task.Wait();
            }
            // Convert clio::run::priv::vector to std::vector for Python
            std::vector<clio::cte::core::CteTelemetry> result;
            for (size_t i = 0; i < task->entries_.size(); ++i) {
              result.push_back(task->entries_[i]);
            }
            return result;
          },
          "minimum_logical_time"_a,
          "Poll telemetry log with minimum logical time filter. "
          "Releases GIL during blocking RPC.")
      .def("ReorganizeBlob",
          [](clio::cte::core::Client &self,
             const clio::cte::core::TagId &tag_id, const std::string &blob_name,
             float new_score) {
            auto task = self.AsyncReorganizeBlob(tag_id, blob_name, new_score);
            {
              nb::gil_scoped_release release;
              task.Wait();
            }
            return task->return_code_ == 0;
          },
          "tag_id"_a, "blob_name"_a, "new_score"_a,
          "Reorganize single blob with new score for data placement optimization. "
          "Releases GIL during blocking RPC.")
     .def("TagQuery",
         [](clio::cte::core::Client &self,
            const std::string &tag_regex, uint32_t max_tags = 0,
            const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Broadcast()) {
           auto task = self.AsyncTagQuery(tag_regex, max_tags, pool_query);
           {
             nb::gil_scoped_release release;
             task.Wait();
           }
           return task->results_;
         },
         "tag_regex"_a, "max_tags"_a = 0,
         "pool_query"_a = clio::run::PoolQuery::Broadcast(),
         "Query tags by regex pattern, returns vector of tag names. "
         "Releases GIL during blocking RPC.")
     .def("BlobQuery",
         [](clio::cte::core::Client &self,
            const std::string &tag_regex, const std::string &blob_regex,
            uint32_t max_blobs = 0,
            const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Broadcast()) {
           auto task = self.AsyncBlobQuery(tag_regex, blob_regex, max_blobs, pool_query);
           {
             nb::gil_scoped_release release;
             task.Wait();
           }
           // Convert separate tag_names_ and blob_names_ vectors to vector of pairs
           std::vector<std::pair<std::string, std::string>> result;
           size_t count = std::min(task->tag_names_.size(), task->blob_names_.size());
           for (size_t i = 0; i < count; ++i) {
             result.emplace_back(task->tag_names_[i], task->blob_names_[i]);
           }
           return result;
         },
         "tag_regex"_a, "blob_regex"_a, "max_blobs"_a = 0,
         "pool_query"_a = clio::run::PoolQuery::Broadcast(),
         "Query blobs by tag and blob regex patterns, returns vector of (tag_name, blob_name) pairs. "
         "Releases GIL during blocking RPC.")
     .def("SemanticSearch",
         [](clio::cte::core::Client &self,
            const std::string &tag_regex, const std::string &blob_regex,
            const std::string &query_text, uint32_t k = 10,
            const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Broadcast()) {
           // Server-side BM25 over the (tag, blob) pairs that match
           // both regexes. Returns the top-k scored results already
           // sorted descending.
           auto task = self.AsyncSemanticSearch(tag_regex, blob_regex,
                                                query_text, k, pool_query);
           {
             nb::gil_scoped_release release;
             task.Wait();
           }
           return task->results_;
         },
         "tag_regex"_a, "blob_regex"_a, "query_text"_a, "k"_a = 10,
         "pool_query"_a = clio::run::PoolQuery::Broadcast(),
         "BM25 keyword search over blob contents. Filters candidate "
         "blobs by tag_regex AND blob_regex (full-string std::regex_match), "
         "tokenizes each candidate's bytes, scores against query_text, "
         "returns top-k SemanticSearchResult sorted by descending score. "
         "Args: tag_regex (str), blob_regex (str), query_text (str), "
         "k (int, default 10; 0 = no cap), pool_query (PoolQuery, default Broadcast). "
         "Returns: list[SemanticSearchResult]. Releases GIL during blocking RPC.")
     .def("TemporalSearch",
         [](clio::cte::core::Client &self,
            const std::string &tag_regex, const std::string &blob_regex,
            uint64_t time_begin = uint64_t{0}, uint64_t time_end = uint64_t{0},
            uint32_t max_entries = uint32_t{0},
            const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Broadcast()) {
           auto task = self.AsyncTemporalSearch(tag_regex, blob_regex,
                                                time_begin, time_end,
                                                max_entries, pool_query);
           {
             nb::gil_scoped_release release;
             task.Wait();
           }
           return task->results_;
         },
         "tag_regex"_a, "blob_regex"_a,
         "time_begin"_a = uint64_t{0}, "time_end"_a = uint64_t{0},
         "max_entries"_a = uint32_t{0}, "pool_query"_a = clio::run::PoolQuery::Broadcast(),
         "Find blobs by last-modified timestamp. Filters by tag_regex AND "
         "blob_regex (full-string std::regex_match), then returns blobs whose "
         "last_modified metadata falls within [time_begin, time_end] (epoch "
         "nanoseconds; 0 = no bound). Results are sorted ascending by "
         "last_modified and capped at max_entries (0 = unlimited). "
         "Returns: list[TemporalSearchResult]. Releases GIL during blocking RPC.")
     .def("RegisterTarget",
         [](clio::cte::core::Client &self,
            const std::string &target_name, clio::run::bdev::BdevType bdev_type,
            uint64_t total_size, const clio::run::PoolQuery &target_query, const clio::run::PoolId &bdev_id) {
           auto task = self.AsyncRegisterTarget(target_name, bdev_type, total_size, target_query, bdev_id);
           {
             nb::gil_scoped_release release;
             task.Wait();
           }
           return task->GetReturnCode();  // a u32: the atomic itself has no Python type
         },
         "target_name"_a, "bdev_type"_a, "total_size"_a,
         "target_query"_a, "bdev_id"_a,
         "Register a storage target. Returns 0 on success, non-zero on failure. "
         "Releases GIL during blocking RPC.")
     .def("RegisterTarget",
         [](clio::cte::core::Client &self,
            const std::string &target_name, clio::run::bdev::BdevType bdev_type,
            uint64_t total_size) {
           auto task = self.AsyncRegisterTarget(target_name, bdev_type, total_size);
           {
             nb::gil_scoped_release release;
             task.Wait();
           }
           return task->GetReturnCode();  // a u32: the atomic itself has no Python type
         },
         "target_name"_a, "bdev_type"_a, "total_size"_a,
         "Register a storage target with default query and pool ID. "
         "Returns 0 on success, non-zero on failure. Releases GIL during blocking RPC.")
     .def("DelBlob",
         [](clio::cte::core::Client &self,
            const clio::cte::core::TagId &tag_id, const std::string &blob_name) {
           auto task = self.AsyncDelBlob(tag_id, blob_name);
           {
             nb::gil_scoped_release release;
             task.Wait();
           }
           return task->return_code_ == 0;
         },
         "tag_id"_a, "blob_name"_a,
         "Delete a blob from a tag. Returns True on success, False otherwise. "
         "Releases GIL during blocking RPC.")
     // ---- Async client APIs -------------------------------------------------
     // Each returns a Future you can overlap other work against: submit N of
     // them, then .wait()/.result() later. The private-memory paths (#823/#830)
     // are used so no manual shared-memory management is needed from Python.
     // Submission runs without the GIL: it can block allocating shared memory
     // (#1096). Python arguments are converted to C++ values before release.
     .def("AsyncGetOrCreateTag",
         [](clio::cte::core::Client &self, const std::string &tag_name,
            const clio::run::PoolQuery &pool_query) {
           using TaskT = clio::cte::core::GetOrCreateTagTask<
               clio::cte::core::CreateParams>;
           clio::run::Future<TaskT> fut;
           {
             nb::gil_scoped_release release;
             fut = self.AsyncGetOrCreateTag(
                 tag_name, clio::cte::core::TagId::GetNull(), pool_query);
           }
           return MakePyFuture<TaskT>(std::move(fut), [](TaskT *t) {
             return t ? nb::cast(t->tag_id_) : nb::none();
           });
         },
         "tag_name"_a, "pool_query"_a = clio::run::PoolQuery::Dynamic(),
         "Asynchronously get or create a tag by name. .result() -> TagId.")
     .def("AsyncGetBlobSize",
         [](clio::cte::core::Client &self, const clio::cte::core::TagId &tag_id,
            const std::string &blob_name,
            const clio::run::PoolQuery &pool_query) {
           using TaskT = clio::cte::core::GetBlobSizeTask;
           clio::run::Future<TaskT> fut;
           {
             nb::gil_scoped_release release;
             fut = self.AsyncGetBlobSize(tag_id, blob_name, pool_query);
           }
           return MakePyFuture<TaskT>(std::move(fut), [](TaskT *t) {
             return t ? nb::cast(t->size_) : nb::none();
           });
         },
         "tag_id"_a, "blob_name"_a,
         "pool_query"_a = clio::run::PoolQuery::Dynamic(),
         "Asynchronously get a blob's size in bytes. .result() -> int.")
     .def("AsyncGetContainedBlobs",
         [](clio::cte::core::Client &self, const clio::cte::core::TagId &tag_id,
            const clio::run::PoolQuery &pool_query) {
           using TaskT = clio::cte::core::GetContainedBlobsTask;
           clio::run::Future<TaskT> fut;
           {
             nb::gil_scoped_release release;
             fut = self.AsyncGetContainedBlobs(tag_id, pool_query);
           }
           return MakePyFuture<TaskT>(std::move(fut), [](TaskT *t) {
             std::vector<std::string> out;
             if (t) {
               for (size_t i = 0; i < t->blob_names_.size(); ++i) {
                 out.emplace_back(t->blob_names_[i]);
               }
             }
             return nb::cast(out);
           });
         },
         "tag_id"_a, "pool_query"_a = clio::run::PoolQuery::Broadcast(),
         "Asynchronously list the blob names in a tag. .result() -> list[str].")
     .def("AsyncPutBlob",
         [](clio::cte::core::Client &self, const clio::cte::core::TagId &tag_id,
            const std::string &blob_name, nb::bytes data, size_t off) {
           // #830 private put. Retain `data` so runtime-mode (co-located,
           // zero-copy) reads stay valid until wait(); client mode stages a
           // TASK_DATA_OWNER copy that the task frees on destruction. bytes
           // are immutable, so reading them without the GIL is safe while
           // this frame holds the reference.
           const char *ptr = data.c_str();
           size_t len = data.size();
           using TaskT = clio::cte::core::PutBlobTask;
           clio::run::Future<TaskT> fut;
           {
             nb::gil_scoped_release release;
             fut = self.AsyncPutBlob(tag_id, blob_name, off, len, ptr);
           }
           return MakePyFuture<TaskT>(std::move(fut), [data](TaskT *) {
             return nb::none();  // status-only; the capture keeps `data` alive
           });
         },
         "tag_id"_a, "blob_name"_a, "data"_a, "off"_a = 0,
         "Asynchronously put blob data from a private buffer (issue #830). "
         "Returns a Future; .wait() -> return code (0 = success).")
     .def("AsyncPutBlobIfAbsent",
         [](clio::cte::core::Client &self,
            const clio::cte::core::TagId &tag_id,
            const std::string &blob_name, nb::bytes data, size_t off) {
           return ConditionalPut(self, tag_id, blob_name, data, off,
                                 clio::cte::core::Context::kPutIfAbsent, 0);
         },
         "tag_id"_a, "blob_name"_a, "data"_a, "off"_a = 0,
         "Atomically create a primary blob if absent. Future.result() -> "
         "{code, version}; PUT_EXISTS_RC means another writer already won. "
         "Nonempty bytes required; existing AsyncPutBlob is unchanged.")
     .def("AsyncPutBlobIfVersion",
         [](clio::cte::core::Client &self,
            const clio::cte::core::TagId &tag_id,
            const std::string &blob_name, nb::bytes data,
            clio::run::u64 version, size_t off) {
           return ConditionalPut(self, tag_id, blob_name, data, off,
                                 clio::cte::core::Context::kPutIfVersion,
                                 version);
         },
         "tag_id"_a, "blob_name"_a, "data"_a, "version"_a, "off"_a = 0,
         "Atomically put only at the expected primary content version "
         "(absent = 0). Future.result() -> {code, version}; "
         "PUT_VERSION_MISMATCH_RC means refetch before retrying. "
         "Puts overwrite a range and never truncate; use fixed-size records.")
     .def("AsyncGetBlobVersioned", &VersionedGet,
         "tag_id"_a, "blob_name"_a, "data_size"_a, "off"_a = 0,
         "Read stored primary bytes through the owner, bypassing the "
         "unversioned SHM cache and lazy fault handlers. "
         "Future.result() -> {code, version, data}; data is None on failure. "
         "A concurrent write can yield newer bytes with an older version; "
         "CAS then conflicts safely. Versions do not promise durable fencing "
         "across runtime restart or blob deletion/recreation.")
     .def("AsyncGetBlob",
         [](clio::cte::core::Client &self, const clio::cte::core::TagId &tag_id,
            const std::string &blob_name, size_t data_size, size_t off) {
           // #823 private get: read straight into a retained heap buffer;
           // .result() returns it as bytes once complete.
           auto buf = std::make_shared<std::string>(data_size, '\0');
           using TaskT = clio::cte::core::GetBlobTask;
           clio::run::Future<TaskT> fut;
           {
             nb::gil_scoped_release release;
             fut = self.AsyncGetBlob(tag_id, blob_name, off, data_size,
                                     /*flags=*/0, buf->data());
           }
           return MakePyFuture<TaskT>(std::move(fut), [buf](TaskT *) {
             return nb::bytes(buf->data(), buf->size());
           });
         },
         "tag_id"_a, "blob_name"_a, "data_size"_a, "off"_a = 0,
         "Asynchronously get blob data into a private buffer (issue #823). "
         "Returns a Future; .result() -> bytes.")
     .def("AsyncPutBlobDefer",
         [](clio::cte::core::Client &self, const clio::cte::core::TagId &tag_id,
            const std::string &blob_name, nb::bytes data, size_t off,
            clio::run::u64 max_inflight_bytes) {
           // The client stages its own SHM copy at submit -- `data` is free
           // the moment this returns, in every mode. Submission may await
           // older puts when shared memory is full, so release the GIL.
           const char *ptr = data.c_str();
           size_t len = data.size();
           int rc;
           {
             nb::gil_scoped_release release;
             rc = self.AsyncPutBlobDefer(
                 tag_id, blob_name, off, len, ptr, -1.0f,
                 clio::cte::core::Context(), 0,
                 clio::run::PoolQuery::Dynamic(), max_inflight_bytes);
           }
           return rc;
         },
         "tag_id"_a, "blob_name"_a, "data"_a, "off"_a = 0,
         "max_inflight_bytes"_a = 0,
         "Deferred async put (issue #862): stages an SHM copy and registers "
         "the put in the client's pending registry; grows until shared memory "
         "is exhausted, then awaits oldest puts (max_inflight_bytes adds an "
         "optional pacing wall; 0 = none). 0 = submitted; poll "
         "DeferErrorCount() for completion failures.")
     .def("AsyncGetBlobDefer",
         [](clio::cte::core::Client &self, const clio::cte::core::TagId &tag_id,
            const std::string &blob_name, size_t data_size, size_t off) {
           auto buf = std::make_shared<std::string>(data_size, '\0');
           using TaskT = clio::cte::core::GetBlobTask;
           clio::run::Future<TaskT> fut;
           {
             nb::gil_scoped_release release;
             fut = self.AsyncGetBlobDefer(tag_id, blob_name, off, data_size,
                                          buf->data());
           }
           return MakePyFuture<TaskT>(std::move(fut), [buf](TaskT *) {
             return nb::bytes(buf->data(), buf->size());
           });
         },
         "tag_id"_a, "blob_name"_a, "data_size"_a, "off"_a = 0,
         "Read-after-write-consistent async get: completes any pending "
         "deferred put(s) for this blob first, then reads. Returns a Future; "
         ".result() -> bytes.")
     .def("AwaitPutsUntilSpace",
         [](clio::cte::core::Client &, clio::run::u64 max_inflight_bytes) {
           nb::gil_scoped_release release;
           return clio::cte::core::Client::AwaitPutsUntilSpace(
               max_inflight_bytes);
         },
         "max_inflight_bytes"_a,
         "Await oldest deferred puts until at most max_inflight_bytes of "
         "payload remain in flight (0 = drain all). Returns in-flight bytes.")
     .def("DeferErrorCount",
         [](clio::cte::core::Client &) {
           return clio::cte::core::Client::DeferErrorCount();
         },
         "Deferred puts that completed with a nonzero return code (sticky).")
     .def("AsyncDelBlob",
         [](clio::cte::core::Client &self, const clio::cte::core::TagId &tag_id,
            const std::string &blob_name) {
           using TaskT = clio::cte::core::DelBlobTask;
           clio::run::Future<TaskT> fut;
           {
             nb::gil_scoped_release release;
             fut = self.AsyncDelBlob(tag_id, blob_name);
           }
           return MakePyFuture<TaskT>(std::move(fut));
         },
         "tag_id"_a, "blob_name"_a,
         "Asynchronously delete a blob. Returns a Future; .wait() -> rc.")
     .def("AsyncReorganizeBlob",
         [](clio::cte::core::Client &self, const clio::cte::core::TagId &tag_id,
            const std::string &blob_name, float new_score) {
           using TaskT = clio::cte::core::ReorganizeBlobTask;
           clio::run::Future<TaskT> fut;
           {
             nb::gil_scoped_release release;
             fut = self.AsyncReorganizeBlob(tag_id, blob_name, new_score);
           }
           return MakePyFuture<TaskT>(std::move(fut));
         },
         "tag_id"_a, "blob_name"_a, "new_score"_a,
         "Asynchronously reorganize a blob. Returns a Future; .wait() -> rc.")
     .def("AsyncRegisterTarget",
         [](clio::cte::core::Client &self, const std::string &target_name,
            clio::run::bdev::BdevType bdev_type, uint64_t total_size,
            const clio::run::PoolQuery &target_query,
            const clio::run::PoolId &bdev_id,
            const clio::run::PoolQuery &pool_query, uint32_t attach_existing,
            uint64_t growth_unit) {
           using TaskT = clio::cte::core::RegisterTargetTask;
           clio::run::Future<TaskT> fut;
           {
             nb::gil_scoped_release release;
             fut = self.AsyncRegisterTarget(target_name, bdev_type,
                                            total_size, target_query, bdev_id,
                                            pool_query, attach_existing,
                                            growth_unit);
           }
           return MakePyFuture<TaskT>(std::move(fut));
         },
         "target_name"_a, "bdev_type"_a, "total_size"_a,
         "target_query"_a = clio::run::PoolQuery::Local(),
         "bdev_id"_a = clio::run::PoolId::GetNull(),
         "pool_query"_a = clio::run::PoolQuery::Dynamic(),
         "attach_existing"_a = uint32_t{0},
         "growth_unit"_a = uint64_t{1} << 30,
         "Asynchronously register a storage target. .wait() -> rc. "
         "The original three-argument call retains native defaults. Supply "
         "an explicit stable bdev_id when creating a managed target; null "
         "does not allocate a new pool ID. Registration does not resize an "
         "existing target or persist its configuration.")
     .def("AsyncGetCapacity",
         [](clio::cte::core::Client &self,
            const clio::run::PoolQuery &pool_query) {
           using TaskT = clio::cte::core::GetCapacityTask;
           clio::run::Future<TaskT> fut;
           {
             nb::gil_scoped_release release;
             fut = self.AsyncGetCapacity(pool_query);
           }
           return MakePyFuture<TaskT>(std::move(fut), [](TaskT *t) {
             RequireQuerySuccess(t, "AsyncGetCapacity");
             return nb::cast(PyCapacityInfo{t->total_capacity_,
                                            t->remaining_capacity_});
           });
         },
         "pool_query"_a = clio::run::PoolQuery::Local(),
         "Read configured and remaining storage bytes. .result() -> "
         "CapacityInfo; raises RuntimeError for a failed query. Local is "
         "this node; Broadcast sums the cluster. Does not report target "
         "health, TTL, or placement eligibility.")
     .def("AsyncListTargets",
         [](clio::cte::core::Client &self,
            const clio::run::PoolQuery &pool_query) {
           using TaskT = clio::cte::core::ListTargetsTask;
           clio::run::Future<TaskT> fut;
           {
             nb::gil_scoped_release release;
             fut = self.AsyncListTargets(pool_query);
           }
           return MakePyFuture<TaskT>(std::move(fut), [](TaskT *t) {
             RequireQuerySuccess(t, "AsyncListTargets");
             return nb::cast(t->target_names_);
           });
         },
         "pool_query"_a = clio::run::PoolQuery::Local(),
         "List registered target names. .result() -> list[str]; raises "
         "RuntimeError for a failed query. Names are not health reports; "
         "Broadcast appends names from replicas without deduplication.")
     .def("AsyncStatTargets",
         [](clio::cte::core::Client &self,
            const clio::run::PoolQuery &pool_query) {
           using TaskT = clio::cte::core::StatTargetsTask;
           clio::run::Future<TaskT> fut;
           {
             nb::gil_scoped_release release;
             fut = self.AsyncStatTargets(pool_query, 0);
           }
           return MakePyFuture<TaskT>(std::move(fut));
         },
         "pool_query"_a = clio::run::PoolQuery::Local(),
         "Request a one-shot target-statistics refresh. .wait() -> rc. "
         "A successful task can retain last-known statistics for a target "
         "that did not answer; it does not prove every target is healthy.")
     .def("AsyncGetTargetInfo",
         [](clio::cte::core::Client &self, const std::string &target_name,
            const clio::run::PoolQuery &pool_query) {
           using TaskT = clio::cte::core::GetTargetInfoTask;
           clio::run::Future<TaskT> fut;
           {
             nb::gil_scoped_release release;
             fut = self.AsyncGetTargetInfo(target_name, pool_query);
           }
           return MakePyFuture<TaskT>(std::move(fut), [](TaskT *t) {
             RequireQuerySuccess(t, "AsyncGetTargetInfo");
             return nb::cast(PyTargetInfo{t->target_score_, t->remaining_space_,
                                          t->bytes_read_, t->bytes_written_,
                                          t->ops_read_, t->ops_written_});
           });
         },
         "target_name"_a,
         "pool_query"_a = clio::run::PoolQuery::Local(),
         "Read a registered target's native statistics. .result() -> "
         "TargetInfo; raises RuntimeError for a failed query, including a "
         "missing target. No TTL, per-target total, or eligibility is "
         "available in this native result; do not infer those fields.")
     .def("AsyncTagQuery",
         [](clio::cte::core::Client &self, const std::string &tag_regex,
            uint32_t max_tags, const clio::run::PoolQuery &pool_query) {
           using TaskT = clio::cte::core::TagQueryTask;
           clio::run::Future<TaskT> fut;
           {
             nb::gil_scoped_release release;
             fut = self.AsyncTagQuery(tag_regex, max_tags, pool_query);
           }
           return MakePyFuture<TaskT>(std::move(fut), [](TaskT *t) {
             std::vector<std::string> out;
             if (t) {
               for (size_t i = 0; i < t->results_.size(); ++i) {
                 out.emplace_back(t->results_[i]);
               }
             }
             return nb::cast(out);
           });
         },
         "tag_regex"_a, "max_tags"_a = 0,
         "pool_query"_a = clio::run::PoolQuery::Broadcast(),
         "Asynchronously query tags by regex. .result() -> list[str].")
     .def("AsyncBlobQuery",
         [](clio::cte::core::Client &self, const std::string &tag_regex,
            const std::string &blob_regex, uint32_t max_blobs,
            const clio::run::PoolQuery &pool_query) {
           using TaskT = clio::cte::core::BlobQueryTask;
           clio::run::Future<TaskT> fut;
           {
             nb::gil_scoped_release release;
             fut = self.AsyncBlobQuery(tag_regex, blob_regex, max_blobs,
                                       pool_query);
           }
           return MakePyFuture<TaskT>(std::move(fut), [](TaskT *t) {
             std::vector<std::pair<std::string, std::string>> out;
             if (t) {
               size_t n = std::min(t->tag_names_.size(),
                                   t->blob_names_.size());
               for (size_t i = 0; i < n; ++i) {
                 out.emplace_back(t->tag_names_[i], t->blob_names_[i]);
               }
             }
             return nb::cast(out);
           });
         },
         "tag_regex"_a, "blob_regex"_a, "max_blobs"_a = 0,
         "pool_query"_a = clio::run::PoolQuery::Broadcast(),
         "Asynchronously query blobs. .result() -> list[(tag, blob)].")
     .def("AsyncSemanticSearch",
         [](clio::cte::core::Client &self, const std::string &tag_regex,
            const std::string &blob_regex, const std::string &query_text,
            uint32_t k, const clio::run::PoolQuery &pool_query) {
           using TaskT = clio::cte::core::SemanticSearchTask;
           clio::run::Future<TaskT> fut;
           {
             nb::gil_scoped_release release;
             fut = self.AsyncSemanticSearch(tag_regex, blob_regex, query_text,
                                            k, pool_query);
           }
           return MakePyFuture<TaskT>(std::move(fut), [](TaskT *t) {
             return t ? nb::cast(t->results_) : nb::none();
           });
         },
         "tag_regex"_a, "blob_regex"_a, "query_text"_a, "k"_a = 10,
         "pool_query"_a = clio::run::PoolQuery::Broadcast(),
         "Asynchronously BM25 keyword search. "
         ".result() -> list[SemanticSearchResult].")
     .def("AsyncTemporalSearch",
         [](clio::cte::core::Client &self, const std::string &tag_regex,
            const std::string &blob_regex, uint64_t time_begin,
            uint64_t time_end, uint32_t max_entries,
            const clio::run::PoolQuery &pool_query) {
           using TaskT = clio::cte::core::TemporalSearchTask;
           clio::run::Future<TaskT> fut;
           {
             nb::gil_scoped_release release;
             fut = self.AsyncTemporalSearch(tag_regex, blob_regex, time_begin,
                                            time_end, max_entries, pool_query);
           }
           return MakePyFuture<TaskT>(std::move(fut), [](TaskT *t) {
             return t ? nb::cast(t->results_) : nb::none();
           });
         },
         "tag_regex"_a, "blob_regex"_a, "time_begin"_a = uint64_t{0},
         "time_end"_a = uint64_t{0}, "max_entries"_a = uint32_t{0},
         "pool_query"_a = clio::run::PoolQuery::Broadcast(),
         "Asynchronously search by last-modified time. "
         ".result() -> list[TemporalSearchResult].")
     .def("AsyncPollTelemetryLog",
         [](clio::cte::core::Client &self,
            std::uint64_t minimum_logical_time) {
           using TaskT = clio::cte::core::PollTelemetryLogTask;
           clio::run::Future<TaskT> fut;
           {
             nb::gil_scoped_release release;
             fut = self.AsyncPollTelemetryLog(minimum_logical_time);
           }
           return MakePyFuture<TaskT>(std::move(fut), [](TaskT *t) {
             std::vector<clio::cte::core::CteTelemetry> out;
             if (t) {
               for (size_t i = 0; i < t->entries_.size(); ++i) {
                 out.push_back(t->entries_[i]);
               }
             }
             return nb::cast(out);
           });
         },
         "minimum_logical_time"_a,
         "Asynchronously poll the telemetry log. "
         ".result() -> list[CteTelemetry].");

  // Bind Tag wrapper class - provides convenient API for tag operations
  // This class wraps tag operations and provides automatic memory management
  nb::class_<clio::cte::core::Tag>(m, "Tag")
      .def(nb::init<const clio::cte::core::TagId &>(),
           "tag_id"_a,
           "Create tag wrapper from existing TagId")
      .def("__init__",
           [](clio::cte::core::Tag *t, const std::string &tag_name) {
             // Tag constructor calls GetOrCreateTag which is blocking,
             // so release the GIL during initialization.
             {
               nb::gil_scoped_release release;
               new (t) clio::cte::core::Tag(tag_name);
             }
           },
           "tag_name"_a,
           "Create or get a tag by name. Calls GetOrCreateTag internally. "
           "Releases GIL during blocking RPC.")
      .def("PutBlob",
           [](clio::cte::core::Tag &self, const std::string &blob_name,
              nb::bytes data, size_t off) {
             // Use nb::bytes to accept arbitrary binary from Python. The bytes'
             // buffer IS private memory, so this writes straight from it via the
             // private-memory path (issue #830): no manual SHM allocate/copy/
             // free, and a zero-copy write in runtime (co-located) mode.
             // Tag::PutBlob(const char*) blocks and throws on a non-zero rc.
             {
               nb::gil_scoped_release release;
               self.PutBlob(blob_name, data.c_str(), data.size(), off);
             }
           },
           "blob_name"_a, "data"_a, "off"_a = 0,
           "Put blob data from a private buffer (issue #830): no manual shared "
           "memory management, and a zero-copy write in runtime mode. "
           "Args: blob_name (str), data (bytes), off (int, optional). "
           "Releases GIL during blocking RPC.")
      .def("GetBlob",
           [](clio::cte::core::Tag &self, const std::string &blob_name,
              size_t data_size, size_t off) -> nb::bytes {
             // Retrieve blob data into a buffer, return it as raw bytes.
             // Must return nb::bytes (not std::string): blobs are arbitrary
             // binary, and nanobind decodes a returned std::string as UTF-8,
             // which raises UnicodeDecodeError on any non-text payload. This
             // also mirrors PutBlob, which already accepts nb::bytes.
             //
             // The std::string's buffer IS private memory, so this reads
             // straight into it via the private-memory path (issue #823): no
             // manual SHM allocate/copy/free, and a zero-IPC copy when the blob
             // is in the shared cache.
             std::string result(data_size, '\0');
             clio::run::Future<clio::cte::core::GetBlobTask> fut;
             {
               nb::gil_scoped_release release;
               fut = self.AsyncGetBlob(blob_name, result.data(), data_size, off);
               fut.Wait();
             }
             // Empty (cache-hit) future -> already succeeded, do not deref.
             // Non-empty (miss) future -> the task carries the return code.
             if (fut.get() != nullptr && fut.get()->GetReturnCode() != 0) {
               throw std::runtime_error("GetBlob operation failed");
             }
             return nb::bytes(result.data(), result.size());
           },
           "blob_name"_a, "data_size"_a, "off"_a = 0,
           "Get blob data into a private buffer (issue #823): no manual shared "
           "memory management, and a zero-IPC copy on a shared-cache hit. "
           "Args: blob_name (str), data_size (int), off (int, optional). "
           "Returns: bytes containing the raw blob data. Releases GIL during blocking RPC.")
      .def("GetBlobScore",
           [](clio::cte::core::Tag &self, const std::string &blob_name) -> float {
             float result;
             {
               nb::gil_scoped_release release;
               result = self.GetBlobScore(blob_name);
             }
             return result;
           },
           "blob_name"_a,
           "Get blob placement score (0.0-1.0). "
           "Args: blob_name (str). Returns: float. Releases GIL during blocking RPC.")
      .def("GetBlobSize",
           [](clio::cte::core::Tag &self, const std::string &blob_name) -> clio::run::u64 {
             clio::run::u64 result;
             {
               nb::gil_scoped_release release;
               result = self.GetBlobSize(blob_name);
             }
             return result;
           },
           "blob_name"_a,
           "Get blob size in bytes. "
           "Args: blob_name (str). Returns: int. Releases GIL during blocking RPC.")
      .def("GetContainedBlobs",
           [](clio::cte::core::Tag &self) -> std::vector<std::string> {
             std::vector<std::string> result;
             {
               nb::gil_scoped_release release;
               result = self.GetContainedBlobs();
             }
             return result;
           },
           "Get all blob names contained in this tag. "
           "Returns: list of str. Releases GIL during blocking RPC.")
      .def("ReorganizeBlob",
           [](clio::cte::core::Tag &self, const std::string &blob_name, float new_score) {
             nb::gil_scoped_release release;
             self.ReorganizeBlob(blob_name, new_score);
           },
           "blob_name"_a, "new_score"_a,
           "Reorganize blob with new score for data placement optimization. "
           "Args: blob_name (str), new_score (float, 0.0-1.0 where higher = faster tier). "
           "Releases GIL during blocking RPC.")
      .def("GetTagId", &clio::cte::core::Tag::GetTagId,
           "Get the TagId for this tag. "
           "Returns: TagId");

  // Module-level convenience functions
  m.def(
      "get_cte_client",
      []() -> clio::cte::core::Client { return *CLIO_CTE_CLIENT; },
      "Get a copy of the global CTE client instance");

  // CLIO Runtime initialization function (unified)
  // Wraps CLIO_INIT to release the GIL during WaitForServerAndReconnect which
  // can block for up to CLIO_WAIT_SERVER (default 30s).
  m.def("clio_init",
        [](clio::run::RuntimeMode mode, bool default_with_runtime, bool is_restart) -> bool {
          bool result;
          {
            nb::gil_scoped_release release;
            result = clio::run::CLIO_INIT(mode, default_with_runtime, is_restart);
          }
          return result;
        },
        "mode"_a, "default_with_runtime"_a = false, "is_restart"_a = false,
        "Initialize Clio with specified mode.\n\n"
        "Args:\n"
        "    mode: RuntimeMode.kClient or RuntimeMode.kServer/kRuntime\n"
        "    default_with_runtime: If True, starts runtime in addition to client (default: False)\n"
        "    is_restart: If True, force restart on compose pools and replay WAL (default: False)\n\n"
        "Environment variable CLIO_WITH_RUNTIME overrides default_with_runtime:\n"
        "    CLIO_WITH_RUNTIME=1 - Start runtime regardless of mode\n"
        "    CLIO_WITH_RUNTIME=0 - Don't start runtime (client only)\n\n"
        "Returns:\n"
        "    bool: True if initialization successful, False otherwise. "
        "Releases GIL during server wait (CLIO_WAIT_SERVER, default 30s).");

  // CTE-specific initialization
  // Note: Lambda wrapper used to avoid clio::run::PoolQuery::Dynamic() evaluation at import
  m.def("initialize_cte",
        [](const std::string &config_path, const clio::run::PoolQuery &pool_query) -> bool {
          bool result;
          {
            nb::gil_scoped_release release;
            result = clio::cte::core::CLIO_CTE_CLIENT_INIT(config_path, pool_query);
          }
          return result;
        },
        "config_path"_a, "pool_query"_a,
        "Initialize the CTE subsystem. Releases GIL during blocking initialization.");
}
