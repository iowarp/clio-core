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

/**
 * The CTE core ChiMod's dashboard endpoints (issue #990).
 *
 * The CTE's dashboard-facing state is its TARGET roster: the block devices it
 * buffers onto, each with a placement score and live capacity. This file gives
 * the page that roster and the two actions the issue asked for -- register a
 * bdev (by name) as a new target, and unregister one -- plus the pool index and
 * the "Add Pool" create form.
 *
 * Handlers run on the dashboard's HTTP thread pool and drive this module's own
 * typed client (ListTargets / GetTargetInfo / RegisterTarget /
 * UnregisterTarget) with bounded waits, exactly like an external client
 * process.
 */

#include <clio_cte/core/core_runtime.h>

#include <string>
#include <vector>

#include "clio_runtime/pool_manager.h"
#include "clio_runtime/viz/viz_json.h"
#include "clio_runtime/viz/viz_server.h"

namespace clio::cte::core {

namespace {

using clio::run::viz::JsonWriter;
using clio::run::viz::Request;
using clio::run::viz::Response;

/** Reads (roster) get a monitor-sized budget. */
constexpr float kReadTimeoutSec = 5.0f;
/** Register creates a bdev pool underneath; give it a create-sized budget. */
constexpr float kActionTimeoutSec = 30.0f;

/** The bdev types a target can be registered as from the dashboard. */
const std::pair<const char *, clio::run::bdev::BdevType> kVizTargetTypes[] = {
    {"ram", clio::run::bdev::BdevType::kRam},
    {"file", clio::run::bdev::BdevType::kFile},
    {"hbm", clio::run::bdev::BdevType::kHbm},
    {"pinned", clio::run::bdev::BdevType::kPinned},
    {"noop", clio::run::bdev::BdevType::kNoop},
};

bool ParseTargetType(const std::string &name,
                     clio::run::bdev::BdevType *out) {
  for (const auto &kv : kVizTargetTypes) {
    if (name == kv.first) {
      *out = kv.second;
      return true;
    }
  }
  return false;
}

/** Parse {pool} into the id of an existing CTE core pool on this node.
 *  @return false after filling resp with the reason. */
bool ResolveCtePool(const Request &req, const std::string &mod_name,
                    clio::run::PoolId *pool_id, Response &resp) {
  const std::string pool_str = req.Var("pool");
  try {
    *pool_id = clio::run::PoolId::FromString(pool_str);
  } catch (const std::exception &e) {
    resp.Error(400, "bad pool id '" + pool_str + "': " + e.what());
    return false;
  }
  auto *pool_manager = CLIO_POOL_MANAGER;
  const auto *info =
      pool_manager ? pool_manager->GetPoolInfo(*pool_id) : nullptr;
  if (!info || info->chimod_name_ != mod_name) {
    resp.Error(404, "no " + mod_name + " pool with id " + pool_str);
    return false;
  }
  return true;
}

}  // namespace

void Runtime::RegisterViz(clio::run::viz::VizServer &viz,
                          const std::string &mod_name) {
  // ---- Pool index ----------------------------------------------------------
  viz.AddRoute(
      {"GET", "/api/mod/" + mod_name + "/pools", mod_name,
       "CTE core pools on this node",
       [mod_name](const Request &, Response &resp) {
         auto *pool_manager = CLIO_POOL_MANAGER;
         if (!pool_manager) {
           resp.Error(503, "pool manager unavailable");
           return;
         }
         JsonWriter w;
         w.BeginObject();
         w.Key("pools").BeginArray();
         for (const auto &pid : pool_manager->GetAllPoolIds()) {
           const auto *info = pool_manager->GetPoolInfo(pid);
           if (!info || info->chimod_name_ != mod_name) {
             continue;
           }
           w.BeginObject();
           w.Field("pool_id", pid.ToString());
           w.Field("pool_name", info->pool_name_);
           w.EndObject();
         }
         w.EndArray();
         w.EndObject();
         resp.Json(w.Str());
       }});

  // ---- Target roster -------------------------------------------------------
  // ListTargets gives the names; GetTargetInfo fills score / capacity / traffic
  // per name. One task per target is fine at dashboard cadence -- the roster is
  // storage tiers, not blobs.
  viz.AddRoute(
      {"GET", "/api/mod/" + mod_name + "/{pool}/targets", mod_name,
       "The pool's registered storage targets, with score and capacity",
       [mod_name](const Request &req, Response &resp) {
         clio::run::PoolId pool_id;
         if (!ResolveCtePool(req, mod_name, &pool_id, resp)) {
           return;
         }
         Client client(pool_id);
         auto list_future = client.AsyncListTargets(clio::run::PoolQuery::Local());
         if (!list_future.Wait(kReadTimeoutSec)) {
           resp.Error(503, "ListTargets timed out");
           return;
         }
         if (list_future->GetReturnCode() != 0) {
           resp.Error(500, "ListTargets failed with rc=" +
                               std::to_string(list_future->GetReturnCode()));
           return;
         }
         JsonWriter w;
         w.BeginObject();
         w.Field("pool_id", pool_id.ToString());
         w.Key("targets").BeginArray();
         for (const std::string &name : list_future->target_names_) {
           w.BeginObject();
           w.Field("name", name);
           auto info_future =
               client.AsyncGetTargetInfo(name, clio::run::PoolQuery::Local());
           if (info_future.Wait(kReadTimeoutSec) &&
               info_future->GetReturnCode() == 0) {
             w.Field("score", info_future->target_score_);
             w.Field("remaining_space", info_future->remaining_space_);
             w.Field("bytes_read", info_future->bytes_read_);
             w.Field("bytes_written", info_future->bytes_written_);
             w.Field("ops_read", info_future->ops_read_);
             w.Field("ops_written", info_future->ops_written_);
           } else {
             // The roster stays useful even when one member won't answer.
             w.Field("error", "target info unavailable");
           }
           w.EndObject();
         }
         w.EndArray();
         w.EndObject();
         resp.Json(w.Str());
       }});

  // ---- Register a bdev as a target (by name) -------------------------------
  // Two shapes, matching RegisterTarget's own two paths:
  //   - name + type + capacity: a fresh bdev pool of that name is created and
  //     registered (the common "add a tier" flow);
  //   - name + attach_pool_id: attach an EXISTING pool that speaks the bdev
  //     task interface (e.g. a safe-bdev array) without creating anything.
  viz.AddRoute(
      {"POST", "/api/mod/" + mod_name + "/{pool}/register_target", mod_name,
       "Register a bdev as a storage target (fields: name[, bdev_type, "
       "capacity | attach_pool_id])",
       [mod_name](const Request &req, Response &resp) {
         clio::run::PoolId pool_id;
         if (!ResolveCtePool(req, mod_name, &pool_id, resp)) {
           return;
         }
         JsonWriter errors;
         errors.BeginObject();
         size_t error_count = 0;
         auto fail = [&errors, &error_count](const std::string &field,
                                             const std::string &message) {
           errors.Field(field, message);
           ++error_count;
         };

         const std::string name = req.Param("name");
         if (name.empty()) {
           fail("name", "required");
         }
         const std::string attach_str = req.Param("attach_pool_id");
         clio::run::PoolId bdev_id;
         clio::run::u32 attach_existing = 0;
         clio::run::bdev::BdevType bdev_type = clio::run::bdev::BdevType::kRam;
         clio::run::u64 capacity = 0;
         if (!attach_str.empty()) {
           attach_existing = 1;
           try {
             bdev_id = clio::run::PoolId::FromString(attach_str);
             if (bdev_id.IsNull()) {
               fail("attach_pool_id", "must not be null");
             }
           } catch (const std::exception &e) {
             fail("attach_pool_id", std::string("unparseable: ") + e.what());
           }
         } else {
           const std::string type_str = req.Param("bdev_type", "ram");
           if (!ParseTargetType(type_str, &bdev_type)) {
             fail("bdev_type", "unknown type '" + type_str + "'");
           }
           const std::string cap_str = req.Param("capacity", "0");
           if (!clio::run::viz::ParseSizeField(cap_str, &capacity)) {
             fail("capacity", "unparseable size '" + cap_str + "'");
           } else if (error_count == 0 && capacity == 0 &&
                      bdev_type != clio::run::bdev::BdevType::kNoop) {
             fail("capacity", "must be > 0");
           }
           // The fresh bdev pool needs an explicit id; suggest-and-take here
           // mirrors what the CTE's own tier registration does with its fixed
           // 512+idx scheme, minus the collision with well-known ids.
           bdev_id = clio::run::PoolId(clio::run::viz::SuggestFreePoolMajor(900), 0);
         }
         errors.EndObject();
         if (error_count > 0) {
           JsonWriter w;
           w.BeginObject();
           w.Field("ok", false);
           w.RawField("errors", errors.Str());
           w.EndObject();
           resp.status = 400;
           resp.Json(w.Str());
           return;
         }

         Client client(pool_id);
         auto future = client.AsyncRegisterTarget(
             name, bdev_type, capacity, clio::run::PoolQuery::Local(), bdev_id,
             clio::run::PoolQuery::Dynamic(), attach_existing);
         if (!future.Wait(kActionTimeoutSec)) {
           resp.Error(503, "RegisterTarget timed out");
           return;
         }
         if (future->GetReturnCode() != 0) {
           resp.Error(500, "RegisterTarget failed with rc=" +
                               std::to_string(future->GetReturnCode()));
           return;
         }
         JsonWriter w;
         w.BeginObject();
         w.Field("ok", true);
         w.Field("name", name);
         w.Field("bdev_pool_id", bdev_id.ToString());
         w.Field("attached", attach_existing != 0);
         w.EndObject();
         resp.Json(w.Str());
       }});

  // ---- Unregister a target -------------------------------------------------
  viz.AddRoute(
      {"POST", "/api/mod/" + mod_name + "/{pool}/unregister_target", mod_name,
       "Remove a storage target from placement (fields: name)",
       [mod_name](const Request &req, Response &resp) {
         clio::run::PoolId pool_id;
         if (!ResolveCtePool(req, mod_name, &pool_id, resp)) {
           return;
         }
         const std::string name = req.Param("name");
         if (name.empty()) {
           resp.Error(400, "missing name");
           return;
         }
         Client client(pool_id);
         auto future =
             client.AsyncUnregisterTarget(name, clio::run::PoolQuery::Dynamic());
         if (!future.Wait(kActionTimeoutSec)) {
           resp.Error(503, "UnregisterTarget timed out");
           return;
         }
         if (future->GetReturnCode() != 0) {
           resp.Error(500, "UnregisterTarget failed with rc=" +
                               std::to_string(future->GetReturnCode()));
           return;
         }
         JsonWriter w;
         w.BeginObject();
         w.Field("ok", true);
         w.Field("name", name);
         w.EndObject();
         resp.Json(w.Str());
       }});

  // ---- Create form ---------------------------------------------------------
  // A bare CTE pool: no tiers, no chain. Storage targets are then added from
  // this page (register_target), which composes better than encoding the whole
  // tier section in form fields; the full compose YAML remains available in
  // the Pools tab's generic editor for anything richer.
  viz.AddRoute(
      {"GET", "/api/mod/" + mod_name + "/create", mod_name,
       "Form spec for creating a CTE core pool",
       [mod_name](const Request &, Response &resp) {
         JsonWriter w;
         w.BeginObject();
         w.Field("mod_name", mod_name);
         w.Field("title", "Context Transfer Engine");
         w.Key("fields").BeginArray();
         w.BeginObject();
         w.Field("name", "pool_name").Field("label", "Pool name");
         w.Field("type", "text").Field("required", true);
         w.EndObject();
         w.BeginObject();
         w.Field("name", "pool_id").Field("label", "Pool ID");
         w.Field("type", "text").Field("required", true);
         w.Field("default",
                 std::to_string(clio::run::viz::SuggestFreePoolMajor(700)) +
                     ".0");
         w.EndObject();
         w.EndArray();
         w.Field("note",
                 "Creates a bare CTE pool. Add storage targets from the CTE "
                 "page afterwards, or use the raw YAML editor for a full "
                 "tier/chain compose.");
         w.EndObject();
         resp.Json(w.Str());
       }});

  viz.AddRoute(
      {"POST", "/api/mod/" + mod_name + "/create", mod_name,
       "Validate (action=validate) or create a CTE core pool",
       [](const Request &req, Response &resp) {
         JsonWriter errors;
         errors.BeginObject();
         size_t error_count = 0;
         auto fail = [&errors, &error_count](const std::string &field,
                                             const std::string &message) {
           errors.Field(field, message);
           ++error_count;
         };
         const std::string pool_name = req.Param("pool_name");
         if (pool_name.empty()) {
           fail("pool_name", "required");
         }
         clio::run::PoolId pool_id;
         try {
           pool_id = clio::run::PoolId::FromString(req.Param("pool_id"));
           if (pool_id.IsNull()) {
             fail("pool_id", "must not be null");
           }
         } catch (const std::exception &e) {
           fail("pool_id", std::string("unparseable: ") + e.what());
         }
         errors.EndObject();
         if (error_count > 0) {
           JsonWriter w;
           w.BeginObject();
           w.Field("ok", false);
           w.RawField("errors", errors.Str());
           w.EndObject();
           resp.status = 400;
           resp.Json(w.Str());
           return;
         }
         if (req.Param("action") == "validate") {
           resp.Json("{\"ok\":true}");
           return;
         }
         Client client(pool_id);
         auto future = client.AsyncCreate(clio::run::PoolQuery::Dynamic(),
                                          pool_name, pool_id);
         if (!future.Wait(kActionTimeoutSec)) {
           resp.Error(503, "CTE create timed out");
           return;
         }
         if (future->GetReturnCode() != 0) {
           resp.Error(500, "CTE create failed with rc=" +
                               std::to_string(future->GetReturnCode()));
           return;
         }
         JsonWriter w;
         w.BeginObject();
         w.Field("ok", true);
         w.Field("pool_name", pool_name);
         w.Field("pool_id", future->new_pool_id_.ToString());
         w.EndObject();
         resp.Json(w.Str());
       }});
}

}  // namespace clio::cte::core
