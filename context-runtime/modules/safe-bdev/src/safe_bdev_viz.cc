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
 * The safe-bdev ChiMod's dashboard endpoints (issue #990).
 *
 * The page (viz/index.html) reads the array's live state -- recovery progress
 * and the member roster -- through the admin's generic Monitor forward
 * (query=stats), so the ONLY endpoints here are the ones a read cannot do:
 *
 *   - the pool index (which pools on this node are safe-bdev arrays),
 *   - add member    (grow the array with a data or parity bdev),
 *   - remove member (take a member out of service),
 *   - the create form (the dashboard's "Add Pool" convention).
 *
 * Everything runs on the dashboard's HTTP thread pool: handlers submit tasks
 * through this module's own Client and wait with a bounded timeout, exactly the
 * way an external client process would.
 */

#include <clio_runtime/bdev/bdev_client.h>
#include <clio_runtime/safe_bdev/safe_bdev_runtime.h>

#include <string>
#include <vector>

#include "clio_runtime/pool_manager.h"
#include "clio_runtime/viz/viz_json.h"
#include "clio_runtime/viz/viz_server.h"

namespace clio::run::safe_bdev {

namespace {

using clio::run::viz::JsonWriter;
using clio::run::viz::Request;
using clio::run::viz::Response;

/** Member management moves data (a remove may trigger rebuilds); give it the
 *  same budget as a create rather than a read's. */
constexpr float kActionTimeoutSec = 30.0f;

/** Reply with {"ok":false,"errors":{...}} at HTTP 400. */
void ReplyErrors(Response &resp, const JsonWriter &errors) {
  JsonWriter w;
  w.BeginObject();
  w.Field("ok", false);
  w.RawField("errors", errors.Str());
  w.EndObject();
  resp.status = 400;
  resp.Json(w.Str());
}

/** Parse {pool} into the id of an EXISTING safe-bdev pool on this node.
 *  @return false after filling resp with the reason. */
bool ResolveSafePool(const Request &req, const std::string &mod_name,
                     clio::run::PoolId *pool_id, Response &resp) {
  const std::string pool_str = req.Var("pool");
  try {
    *pool_id = clio::run::PoolId::FromString(pool_str);
  } catch (const std::exception &e) {
    resp.Error(400, "bad pool id '" + pool_str + "': " + e.what());
    return false;
  }
  auto *pool_manager = CLIO_POOL_MANAGER;
  const auto *info = pool_manager ? pool_manager->GetPoolInfo(*pool_id)
                                  : nullptr;
  if (!info || info->chimod_name_ != mod_name) {
    resp.Error(404, "no " + mod_name + " pool with id " + pool_str);
    return false;
  }
  return true;
}

/**
 * Resolve a member field to the PoolId of an existing bdev pool, creating the
 * bdev when asked to.
 *
 * The array's members are ordinary bdev pools. The form accepts either the
 * NAME of a pool that already exists (created via the bdev form, or composed)
 * or -- when @p capacity_str is non-empty -- a name + capacity + type to create
 * on the spot, which is the one-step flow the Python dashboard offered.
 *
 * @return true and set @p member_id; false after writing a field error.
 */
bool ResolveMemberBdev(const std::string &name, const std::string &type_str,
                       const std::string &capacity_str, JsonWriter &errors,
                       size_t *error_count, clio::run::PoolId *member_id) {
  auto fail = [&errors, error_count](const std::string &field,
                                     const std::string &message) {
    errors.Field(field, message);
    ++*error_count;
  };
  if (name.empty()) {
    fail("member_name", "required");
    return false;
  }
  auto *pool_manager = CLIO_POOL_MANAGER;
  clio::run::PoolId existing = pool_manager
                                   ? pool_manager->FindPoolByName(name)
                                   : clio::run::PoolId::GetNull();
  if (!existing.IsNull()) {
    // A member must speak the bdev task interface; naming, say, a CTE pool
    // here would wire the array to a pool that cannot serve block I/O.
    const auto *info = pool_manager->GetPoolInfo(existing);
    if (!info || info->chimod_name_ != "clio_bdev") {
      fail("member_name",
           "'" + name + "' exists but is not a clio_bdev pool");
      return false;
    }
    *member_id = existing;
    return true;
  }
  if (capacity_str.empty()) {
    fail("member_name",
         "no pool named '" + name +
             "' exists; give a capacity to create it here");
    return false;
  }
  clio::run::u64 capacity = 0;
  if (!clio::run::viz::ParseSizeField(capacity_str, &capacity)) {
    fail("capacity", "unparseable size '" + capacity_str + "'");
    return false;
  }
  if (capacity == 0) {
    fail("capacity", "must be > 0");
    return false;
  }
  clio::run::bdev::BdevType bdev_type = clio::run::bdev::BdevType::kFile;
  if (type_str == "ram") {
    bdev_type = clio::run::bdev::BdevType::kRam;
  } else if (!type_str.empty() && type_str != "file") {
    fail("bdev_type", "member bdevs here are 'file' or 'ram'");
    return false;
  }
  clio::run::PoolId fresh(clio::run::viz::SuggestFreePoolMajor(900), 0);
  clio::run::bdev::Client bdev_client(fresh);
  auto create = bdev_client.AsyncCreate(clio::run::PoolQuery::Dynamic(), name,
                                        fresh, bdev_type, capacity);
  if (!create.Wait(kActionTimeoutSec) || create->return_code_ != 0) {
    fail("member_name", "creating member bdev '" + name + "' failed");
    return false;
  }
  *member_id = create->new_pool_id_;
  return true;
}

}  // namespace

void Runtime::RegisterViz(clio::run::viz::VizServer &viz,
                          const std::string &mod_name) {
  // ---- Pool index ----------------------------------------------------------
  viz.AddRoute(
      {"GET", "/api/mod/" + mod_name + "/pools", mod_name,
       "Safe-bdev arrays on this node",
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

  // ---- Add member ----------------------------------------------------------
  // Grows the array. The member is an ordinary bdev pool: name an existing one,
  // or give capacity (+ bdev_type) and it is created first. as_parity=1 raises
  // the parity level instead of adding data capacity.
  viz.AddRoute(
      {"POST", "/api/mod/" + mod_name + "/{pool}/add_member", mod_name,
       "Add a data or parity member bdev (fields: member_name[, capacity, "
       "bdev_type, node_id, as_parity])",
       [mod_name](const Request &req, Response &resp) {
         clio::run::PoolId pool_id;
         if (!ResolveSafePool(req, mod_name, &pool_id, resp)) {
           return;
         }
         JsonWriter errors;
         errors.BeginObject();
         size_t error_count = 0;
         const std::string member_name = req.Param("member_name");
         clio::run::PoolId member_id;
         ResolveMemberBdev(member_name, req.Param("bdev_type", "file"),
                           req.Param("capacity"), errors, &error_count,
                           &member_id);
         const clio::run::u32 node_id = static_cast<clio::run::u32>(
             std::strtoul(req.Param("node_id", "0").c_str(), nullptr, 10));
         const clio::run::u32 as_parity =
             (req.Param("as_parity", "0") == "1") ? 1u : 0u;
         errors.EndObject();
         if (error_count > 0) {
           ReplyErrors(resp, errors);
           return;
         }

         Client safe(pool_id);
         auto future = safe.AsyncAddBdev(clio::run::PoolQuery::Dynamic(),
                                         member_name, node_id, member_id,
                                         as_parity);
         if (!future.Wait(kActionTimeoutSec)) {
           resp.Error(503, "AddBdev timed out");
           return;
         }
         if (future->GetReturnCode() != 0) {
           resp.Error(500, "AddBdev failed with rc=" +
                               std::to_string(future->GetReturnCode()));
           return;
         }
         JsonWriter w;
         w.BeginObject();
         w.Field("ok", true);
         w.Field("member_name", member_name);
         w.Field("member_pool_id", member_id.ToString());
         w.Field("as_parity", as_parity != 0);
         w.EndObject();
         resp.Json(w.Str());
       }});

  // ---- Replace + recover a failed member -----------------------------------
  // The repair flow: compose a fresh member bdev and rebuild the failed
  // member's rows onto it. Recovery itself runs in the background; the
  // Monitor("stats") recovery_* counters (which the page's Recovery panel
  // polls) report its progress.
  viz.AddRoute(
      {"POST", "/api/mod/" + mod_name + "/{pool}/replace_member", mod_name,
       "Replace a failed member with a fresh bdev and recover onto it "
       "(fields: failed_pool_id, member_name, capacity[, bdev_type, node_id])",
       [mod_name](const Request &req, Response &resp) {
         clio::run::PoolId pool_id;
         if (!ResolveSafePool(req, mod_name, &pool_id, resp)) {
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
         const std::string failed_str = req.Param("failed_pool_id");
         clio::run::PoolId failed_id;
         try {
           failed_id = clio::run::PoolId::FromString(failed_str);
           if (failed_id.IsNull()) {
             fail("failed_pool_id", "must not be null");
           }
         } catch (const std::exception &e) {
           fail("failed_pool_id", std::string("unparseable: ") + e.what());
         }
         const std::string member_name = req.Param("member_name");
         clio::run::PoolId member_id;
         ResolveMemberBdev(member_name, req.Param("bdev_type", "file"),
                           req.Param("capacity"), errors, &error_count,
                           &member_id);
         const clio::run::u32 node_id = static_cast<clio::run::u32>(
             std::strtoul(req.Param("node_id", "0").c_str(), nullptr, 10));
         errors.EndObject();
         if (error_count > 0) {
           ReplyErrors(resp, errors);
           return;
         }

         Client safe(pool_id);
         auto future = safe.AsyncRecoverBdev(clio::run::PoolQuery::Dynamic(),
                                             failed_id, member_name, node_id,
                                             member_id);
         if (!future.Wait(kActionTimeoutSec)) {
           resp.Error(503, "RecoverBdev timed out");
           return;
         }
         if (future->GetReturnCode() != 0) {
           resp.Error(500, "RecoverBdev failed with rc=" +
                               std::to_string(future->GetReturnCode()));
           return;
         }
         JsonWriter w;
         w.BeginObject();
         w.Field("ok", true);
         w.Field("failed_pool_id", failed_str);
         w.Field("member_name", member_name);
         w.Field("member_pool_id", member_id.ToString());
         w.EndObject();
         resp.Json(w.Str());
       }});

  // ---- Remove member -------------------------------------------------------
  viz.AddRoute(
      {"POST", "/api/mod/" + mod_name + "/{pool}/remove_member", mod_name,
       "Take a member out of service (fields: member_pool_id[, was_faulty])",
       [mod_name](const Request &req, Response &resp) {
         clio::run::PoolId pool_id;
         if (!ResolveSafePool(req, mod_name, &pool_id, resp)) {
           return;
         }
         const std::string member_str = req.Param("member_pool_id");
         clio::run::PoolId member_id;
         try {
           member_id = clio::run::PoolId::FromString(member_str);
         } catch (const std::exception &e) {
           resp.Error(400, "bad member_pool_id '" + member_str +
                               "': " + e.what());
           return;
         }
         const clio::run::u32 was_faulty =
             (req.Param("was_faulty", "0") == "1") ? 1u : 0u;

         Client safe(pool_id);
         auto future = safe.AsyncRemoveBdev(clio::run::PoolQuery::Dynamic(),
                                            member_id, was_faulty);
         if (!future.Wait(kActionTimeoutSec)) {
           resp.Error(503, "RemoveBdev timed out");
           return;
         }
         if (future->GetReturnCode() != 0) {
           resp.Error(500, "RemoveBdev failed with rc=" +
                               std::to_string(future->GetReturnCode()));
           return;
         }
         JsonWriter w;
         w.BeginObject();
         w.Field("ok", true);
         w.Field("member_pool_id", member_str);
         w.EndObject();
         resp.Json(w.Str());
       }});

  // ---- Create form ---------------------------------------------------------
  viz.AddRoute(
      {"GET", "/api/mod/" + mod_name + "/create", mod_name,
       "Form spec for creating a safe-bdev array",
       [mod_name](const Request &, Response &resp) {
         JsonWriter w;
         w.BeginObject();
         w.Field("mod_name", mod_name);
         w.Field("title", "Safe block device (EC array)");
         w.Key("fields").BeginArray();
         w.BeginObject();
         w.Field("name", "pool_name").Field("label", "Pool name");
         w.Field("type", "text").Field("required", true);
         w.EndObject();
         w.BeginObject();
         w.Field("name", "pool_id").Field("label", "Pool ID");
         w.Field("type", "text").Field("required", true);
         w.Field("default",
                 std::to_string(clio::run::viz::SuggestFreePoolMajor(850)) +
                     ".0");
         w.EndObject();
         w.BeginObject();
         w.Field("name", "max_failures").Field("label", "Max failures");
         w.Field("type", "text").Field("default", "1");
         w.Field("help", "simultaneous member failures the array survives");
         w.EndObject();
         w.BeginObject();
         w.Field("name", "members").Field("label", "Data members");
         w.Field("type", "text").Field("required", true);
         w.Field("placeholder", "bdev_a, bdev_b, bdev_c");
         w.Field("help",
                 "comma-separated names of EXISTING bdev pools (create them "
                 "with the Block device form first)");
         w.EndObject();
         w.EndArray();
         w.EndObject();
         resp.Json(w.Str());
       }});

  viz.AddRoute(
      {"POST", "/api/mod/" + mod_name + "/create", mod_name,
       "Validate (action=validate) or create a safe-bdev array",
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
         const clio::run::u32 max_failures = static_cast<clio::run::u32>(
             std::strtoul(req.Param("max_failures", "1").c_str(), nullptr, 10));

         // Members: comma-separated names of bdev pools that already exist.
         // Existence is checked HERE so validate reports the missing name,
         // instead of the create failing opaquely inside the module.
         auto *pool_manager = CLIO_POOL_MANAGER;
         std::vector<MemberBdevDesc> members;
         {
           const std::string list = req.Param("members");
           size_t start = 0;
           while (start <= list.size()) {
             size_t comma = list.find(',', start);
             std::string name = (comma == std::string::npos)
                                    ? list.substr(start)
                                    : list.substr(start, comma - start);
             const size_t first = name.find_first_not_of(" \t");
             const size_t last = name.find_last_not_of(" \t");
             name = (first == std::string::npos)
                        ? ""
                        : name.substr(first, last - first + 1);
             if (!name.empty()) {
               clio::run::PoolId member_id =
                   pool_manager ? pool_manager->FindPoolByName(name)
                                : clio::run::PoolId::GetNull();
               const auto *info = member_id.IsNull()
                                      ? nullptr
                                      : pool_manager->GetPoolInfo(member_id);
               if (member_id.IsNull()) {
                 fail("members", "no pool named '" + name + "' exists");
               } else if (!info || info->chimod_name_ != "clio_bdev") {
                 fail("members",
                      "'" + name + "' exists but is not a clio_bdev pool");
               } else {
                 members.emplace_back(name, /*node_id=*/0, member_id);
               }
             }
             if (comma == std::string::npos) break;
             start = comma + 1;
           }
         }
         if (members.empty() && error_count == 0) {
           fail("members", "at least one existing bdev pool name is required");
         }
         if (error_count == 0 && members.size() <= max_failures) {
           fail("max_failures",
                "needs more data members than tolerated failures");
         }
         errors.EndObject();
         if (error_count > 0) {
           ReplyErrors(resp, errors);
           return;
         }
         if (req.Param("action") == "validate") {
           resp.Json("{\"ok\":true}");
           return;
         }

         Client safe(pool_id);
         auto future = safe.AsyncCreate(clio::run::PoolQuery::Dynamic(),
                                        pool_name, pool_id, max_failures,
                                        members);
         if (!future.Wait(kActionTimeoutSec)) {
           resp.Error(503, "safe-bdev create timed out");
           return;
         }
         if (future->GetReturnCode() != 0) {
           resp.Error(500, "safe-bdev create failed with rc=" +
                               std::to_string(future->GetReturnCode()));
           return;
         }
         JsonWriter w;
         w.BeginObject();
         w.Field("ok", true);
         w.Field("pool_name", pool_name);
         w.Field("pool_id", future->new_pool_id_.ToString());
         w.Field("data_members", static_cast<clio::run::u64>(members.size()));
         w.EndObject();
         resp.Json(w.Str());
       }});
}

}  // namespace clio::run::safe_bdev
