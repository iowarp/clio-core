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
 * Issue #990: the in-runtime web dashboard.
 *
 * Two layers, tested at both:
 *
 *   1. The router, driven directly (no sockets): path patterns, MIME types,
 *      JSON emission, static-asset resolution including the traversal guard,
 *      and the build-tree-vs-install-tree asset search that lets the same
 *      binary find its pages either way.
 *
 *   2. The real thing: an in-process runtime whose admin container started an
 *      HTTP server on an ephemeral port, exercised over TCP with an HTTP client
 *      -- so a broken handler, a wedged wait, or an unregistered route fails
 *      here rather than in someone's browser.
 */

#include "../simple_test.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

#include "clio_runtime/clio_runtime.h"
#include "clio_runtime/config_manager.h"
#include "clio_runtime/ipc_manager.h"
#include "clio_runtime/pool_manager.h"
#include "clio_runtime/viz/viz_json.h"
#include "clio_runtime/viz/viz_server.h"

#include <clio_runtime/bdev/bdev_client.h>
#include <clio_runtime/bdev/bdev_tasks.h>

#include <Poco/Net/HTTPClientSession.h>
#include <Poco/Net/StreamSocket.h>
#include <Poco/Net/SocketAddress.h>
#include <Poco/Net/HTTPRequest.h>
#include <Poco/Net/HTTPResponse.h>
#include <Poco/StreamCopier.h>
#include <Poco/Timespan.h>

namespace viz = clio::run::viz;

namespace {

/** A tiny HTTP response, as seen by the test client. */
struct HttpReply {
  int status = 0;
  std::string content_type;
  std::string body;
};

/** GET @p path from the dashboard on 127.0.0.1:@p port. */
HttpReply HttpGet(clio::run::u32 port, const std::string &path) {
  HttpReply reply;
  Poco::Net::HTTPClientSession session("127.0.0.1",
                                       static_cast<Poco::UInt16>(port));
  session.setTimeout(Poco::Timespan(15, 0));
  Poco::Net::HTTPRequest request(Poco::Net::HTTPRequest::HTTP_GET, path,
                                 Poco::Net::HTTPMessage::HTTP_1_1);
  session.sendRequest(request);
  Poco::Net::HTTPResponse response;
  std::istream &stream = session.receiveResponse(response);
  Poco::StreamCopier::copyToString(stream, reply.body);
  reply.status = static_cast<int>(response.getStatus());
  reply.content_type = response.getContentType();
  return reply;
}

/** POST a urlencoded form to the dashboard, the way the pages' fetch() does.
 *  Actions get a longer client timeout than reads: creating a pool composes
 *  nested pools and the handler itself waits up to 30s. */
HttpReply HttpPostForm(clio::run::u32 port, const std::string &path,
                       const std::string &form) {
  HttpReply reply;
  Poco::Net::HTTPClientSession session("127.0.0.1",
                                       static_cast<Poco::UInt16>(port));
  session.setTimeout(Poco::Timespan(60, 0));
  Poco::Net::HTTPRequest request(Poco::Net::HTTPRequest::HTTP_POST, path,
                                 Poco::Net::HTTPMessage::HTTP_1_1);
  request.setContentType("application/x-www-form-urlencoded");
  request.setContentLength(static_cast<std::streamsize>(form.size()));
  session.sendRequest(request) << form;
  Poco::Net::HTTPResponse response;
  std::istream &stream = session.receiveResponse(response);
  Poco::StreamCopier::copyToString(stream, reply.body);
  reply.status = static_cast<int>(response.getStatus());
  reply.content_type = response.getContentType();
  return reply;
}

bool Contains(const std::string &haystack, const std::string &needle) {
  return haystack.find(needle) != std::string::npos;
}

/** Report what a failing body actually said -- a 200 with the wrong shape and a
 *  503 from a timed-out Monitor are very different bugs. */
void Explain(const std::string &what, const HttpReply &reply) {
  std::cout << "  " << what << " -> " << reply.status << " "
            << reply.body.substr(0, 300) << std::endl;
}

/**
 * Bring up the in-process runtime with the dashboard enabled on an ephemeral
 * port, once for the whole binary.
 *
 * Port 0 matters: CI runs many daemons at once, and a fixed port would make this
 * test fail (or worse, talk to another test's daemon) depending on scheduling.
 * The environment is set before CLIO_INIT because the admin container starts the
 * server during its Create, inside the runtime bring-up.
 */
bool InitRuntimeWithViz() {
  static const bool ok = [] {
    SIMPLE_TEST_SETENV("CLIO_VIZ_ENABLE", "1");
    SIMPLE_TEST_SETENV("CLIO_VIZ_PORT", "0");
    SIMPLE_TEST_SETENV("CLIO_VIZ_BIND", "127.0.0.1");
    return clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
  }();
  return ok;
}

/** Create a throwaway asset tree and return its path. */
std::filesystem::path MakeAssetTree() {
  std::filesystem::path root =
      std::filesystem::temp_directory_path() / "clio_viz_test_assets";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root / "sub");
  std::ofstream(root / "index.html") << "<h1>hello</h1>";
  std::ofstream(root / "sub" / "app.js") << "console.log(1);";
  std::ofstream(root / "secret.txt") << "not reachable via ..";
  return root;
}

}  // namespace

TEST_CASE("Viz router matches path patterns", "[viz]") {
  std::map<std::string, std::string> vars;

  REQUIRE(viz::VizServer::MatchPath("/api/health", "/api/health", &vars));
  REQUIRE(vars.empty());

  REQUIRE(viz::VizServer::MatchPath("/api/nodes/{node}/workers",
                                    "/api/nodes/7/workers", &vars));
  REQUIRE(vars["node"] == "7");

  REQUIRE(viz::VizServer::MatchPath("/api/pools/{pool}/monitor",
                                    "/api/pools/301.0/monitor", &vars));
  REQUIRE(vars["pool"] == "301.0");

  // Segment counts must agree: a variable matches exactly one segment, so
  // neither a shorter nor a longer path may sneak through.
  REQUIRE_FALSE(viz::VizServer::MatchPath("/api/nodes/{node}/workers",
                                          "/api/nodes/7", &vars));
  REQUIRE_FALSE(viz::VizServer::MatchPath("/api/nodes/{node}/workers",
                                          "/api/nodes/7/workers/extra", &vars));
  REQUIRE_FALSE(
      viz::VizServer::MatchPath("/api/health", "/api/healthz", &vars));

  // Empty segments are insignificant, so a trailing slash still matches.
  REQUIRE(viz::VizServer::MatchPath("/api/health", "/api/health/", &vars));
}

TEST_CASE("Viz MIME types cover the dashboard's asset kinds", "[viz]") {
  REQUIRE(Contains(viz::VizServer::MimeTypeOf("index.html"), "text/html"));
  REQUIRE(Contains(viz::VizServer::MimeTypeOf("css/style.CSS"), "text/css"));
  REQUIRE(Contains(viz::VizServer::MimeTypeOf("js/api.js"), "javascript"));
  REQUIRE(viz::VizServer::MimeTypeOf("data.json") == "application/json");
  REQUIRE(viz::VizServer::MimeTypeOf("logo.svg") == "image/svg+xml");
  REQUIRE(Contains(viz::VizServer::MimeTypeOf("no_extension"), "text/plain"));
}

TEST_CASE("Viz JSON writer emits valid nested JSON", "[viz]") {
  viz::JsonWriter w;
  w.BeginObject();
  w.Field("name", "pool");
  w.Field("count", 3u);
  w.Field("ok", true);
  w.Key("items").BeginArray();
  w.Value(1);
  w.BeginObject();
  w.Field("inner", "v");
  w.EndObject();
  w.EndArray();
  w.RawField("raw", "[1,2]");
  w.EndObject();
  REQUIRE(w.Str() ==
          "{\"name\":\"pool\",\"count\":3,\"ok\":true,"
          "\"items\":[1,{\"inner\":\"v\"}],\"raw\":[1,2]}");

  // Strings that would otherwise break the document.
  REQUIRE(viz::JsonQuote("a\"b\\c") == "\"a\\\"b\\\\c\"");
  REQUIRE(viz::JsonQuote("line\nbreak") == "\"line\\nbreak\"");
  REQUIRE(viz::JsonQuote(std::string("nul\x01here")) == "\"nul\\u0001here\"");

  // JSON has no NaN/Infinity, and utilization readings do produce them.
  viz::JsonWriter nan_writer;
  nan_writer.BeginObject();
  nan_writer.Field("pct", std::nan(""));
  nan_writer.Field("inf", std::numeric_limits<double>::infinity());
  nan_writer.EndObject();
  REQUIRE(nan_writer.Str() == "{\"pct\":null,\"inf\":null}");

  // An empty Raw() must not produce a syntax error either.
  viz::JsonWriter empty_writer;
  empty_writer.BeginObject();
  empty_writer.RawField("payload", "");
  empty_writer.EndObject();
  REQUIRE(empty_writer.Str() == "{\"payload\":null}");
}

TEST_CASE("Viz router dispatches routes, assets and the home page", "[viz]") {
  viz::VizServer router;
  int calls = 0;

  REQUIRE(router.AddRoute({"GET", "/api/echo/{what}", "testmod", "echoes",
                           [&calls](const viz::Request &req,
                                    viz::Response &resp) {
                             ++calls;
                             resp.Json("{\"what\":\"" + req.Var("what") +
                                       "\",\"q\":\"" + req.Param("q") + "\"}");
                           }}));

  // First registration of a (method, path) wins, so a ChiMod with several
  // containers -- or several pools -- registers idempotently.
  REQUIRE_FALSE(router.AddRoute({"GET", "/api/echo/{what}", "testmod", "dup",
                                 [](const viz::Request &, viz::Response &) {}}));

  viz::Request req;
  viz::Response resp;
  req.method = "GET";
  req.path = "/api/echo/hi";
  req.params["q"] = "42";
  REQUIRE(router.Dispatch(req, resp));
  REQUIRE(resp.status == 200);
  REQUIRE(resp.body == "{\"what\":\"hi\",\"q\":\"42\"}");
  REQUIRE(calls == 1);

  // HEAD is served by the GET route (the transport drops the body).
  viz::Request head = req;
  head.method = "HEAD";
  viz::Response head_resp;
  REQUIRE(router.Dispatch(head, head_resp));
  REQUIRE(head_resp.status == 200);

  // A different verb on the same path is not a match.
  viz::Request post = req;
  post.method = "POST";
  viz::Response post_resp;
  REQUIRE_FALSE(router.Dispatch(post, post_resp));

  // An unknown path is a 404 (Dispatch returns false; the transport renders it).
  viz::Request missing;
  missing.method = "GET";
  missing.path = "/api/nope";
  viz::Response missing_resp;
  REQUIRE_FALSE(router.Dispatch(missing, missing_resp));

  // A throwing handler becomes a 500 rather than killing the HTTP thread.
  REQUIRE(router.AddRoute({"GET", "/api/boom", "testmod", "throws",
                           [](const viz::Request &, viz::Response &) {
                             throw std::runtime_error("kaboom");
                           }}));
  viz::Request boom;
  boom.method = "GET";
  boom.path = "/api/boom";
  viz::Response boom_resp;
  REQUIRE(router.Dispatch(boom, boom_resp));
  REQUIRE(boom_resp.status == 500);
  REQUIRE(Contains(boom_resp.body, "kaboom"));

  // ---- static assets ----
  std::filesystem::path assets = MakeAssetTree();
  REQUIRE(router.MountDir("testmod", assets.string()));
  REQUIRE_FALSE(router.MountDir("testmod", assets.string()));  // already mounted
  REQUIRE_FALSE(router.MountDir("testmod", (assets / "nonexistent").string(),
                                "/viz/other"));

  auto fetch = [&router](const std::string &path) {
    viz::Request r;
    viz::Response out;
    r.method = "GET";
    r.path = path;
    bool handled = router.Dispatch(r, out);
    return std::make_pair(handled, out);
  };

  auto index = fetch("/viz/testmod/index.html");
  REQUIRE(index.first);
  REQUIRE(index.second.status == 200);
  REQUIRE(index.second.body == "<h1>hello</h1>");
  REQUIRE(Contains(index.second.content_type, "text/html"));

  // A bare mount point serves index.html.
  auto bare = fetch("/viz/testmod");
  REQUIRE(bare.first);
  REQUIRE(bare.second.body == "<h1>hello</h1>");

  auto nested = fetch("/viz/testmod/sub/app.js");
  REQUIRE(nested.first);
  REQUIRE(Contains(nested.second.content_type, "javascript"));

  // Traversal must be refused, not resolved: the URL is already
  // percent-decoded by the time the router sees it.
  REQUIRE_FALSE(fetch("/viz/testmod/../secret.txt").first);
  REQUIRE_FALSE(fetch("/viz/testmod/sub/../../secret.txt").first);

  // A prefix must match on a segment boundary.
  REQUIRE_FALSE(fetch("/viz/testmodX/index.html").first);

  // ---- home page ----
  REQUIRE(fetch("/").first == false);
  router.SetHome("/viz/testmod/index.html");
  auto home = fetch("/");
  REQUIRE(home.first);
  REQUIRE(home.second.status == 302);
  bool has_location = false;
  for (const auto &kv : home.second.headers) {
    if (kv.first == "Location" && kv.second == "/viz/testmod/index.html") {
      has_location = true;
    }
  }
  REQUIRE(has_location);

  // ---- introspection ----
  REQUIRE(router.GetRoutes().size() == 2);
  REQUIRE(router.GetMounts().size() == 1);
  REQUIRE(router.GetMounts()[0].url_prefix == "/viz/testmod");

  // ---- teardown ----
  // A ChiMod whose handler captures container state has to be able to take its
  // registrations back, or a request after the container dies would run a
  // handler over freed memory.
  REQUIRE(router.RemoveModule("testmod") == 2);
  REQUIRE(router.GetRoutes().empty());
  REQUIRE(router.GetMounts().empty());
  // The home page pointed into the mount that just went away.
  REQUIRE(router.GetHome().empty());
  REQUIRE_FALSE(fetch("/viz/testmod/index.html").first);
  REQUIRE_FALSE(fetch("/api/echo/hi").first);
  REQUIRE(router.RemoveModule("testmod") == 0);

  std::filesystem::remove_all(assets);
}

TEST_CASE("Viz asset search covers build and install layouts", "[viz]") {
  // The issue's requirement: the same binary must find its pages whether the
  // tree was only built (libraries in <build>/bin, assets staged next to them)
  // or installed (libraries in <prefix>/lib, assets in <prefix>/share/clio/viz).
  // Both candidates are derived from the ChiMod's own library path, so the
  // answer follows whichever copy of the module was loaded.
  std::vector<std::string> dirs =
      viz::VizServer::AssetSearchDirs("clio_admin");
  bool has_build_layout = false;
  bool has_install_layout = false;
  for (const std::string &dir : dirs) {
    if (Contains(dir, std::filesystem::path("viz/clio_admin").string()) &&
        !Contains(dir, "share")) {
      has_build_layout = true;
    }
    if (Contains(dir,
                 std::filesystem::path("share/clio/viz/clio_admin").string())) {
      has_install_layout = true;
    }
  }
  for (const std::string &dir : dirs) {
    std::cout << "  candidate: " << dir << std::endl;
  }
  REQUIRE(has_build_layout);
  REQUIRE(has_install_layout);

  // CLIO_VIZ_PATH takes precedence, so assets can be edited without a rebuild.
  std::filesystem::path root = MakeAssetTree();
  std::filesystem::path viz_root = root.parent_path() / "clio_viz_test_root";
  std::filesystem::remove_all(viz_root);
  std::filesystem::create_directories(viz_root / "clio_admin");
  SIMPLE_TEST_SETENV("CLIO_VIZ_PATH", viz_root.string().c_str());
  dirs = viz::VizServer::AssetSearchDirs("clio_admin");
  REQUIRE(!dirs.empty());
  REQUIRE(dirs[0] == (viz_root / "clio_admin").string());
  REQUIRE(viz::VizServer::FindAssetDir("clio_admin") ==
          (viz_root / "clio_admin").string());
  // Leave no override behind for the HTTP case below, which needs the real
  // build-tree assets.
  SIMPLE_TEST_SETENV("CLIO_VIZ_PATH", "");
  std::filesystem::remove_all(viz_root);
  std::filesystem::remove_all(root);
}

TEST_CASE("Viz HTTP server answers the dashboard API", "[viz]") {
  REQUIRE(InitRuntimeWithViz());

  auto *viz_server = CLIO_VIZ;
  REQUIRE(viz_server != nullptr);
  // The admin container starts the dashboard during its Create, so by the time
  // CLIO_INIT has returned it is already listening.
  REQUIRE(viz_server->IsRunning());
  const clio::run::u32 port = viz_server->GetPort();
  std::cout << "  dashboard on 127.0.0.1:" << port << std::endl;
  REQUIRE(port != 0);
  REQUIRE(viz_server->GetBindAddr() == "127.0.0.1");

  // ---- node-local endpoints ----
  HttpReply health = HttpGet(port, "/api/health");
  Explain("/api/health", health);
  REQUIRE(health.status == 200);
  REQUIRE(Contains(health.content_type, "application/json"));
  REQUIRE(Contains(health.body, "\"ok\":true"));
  REQUIRE(Contains(health.body, "\"node_id\":"));

  HttpReply topology = HttpGet(port, "/api/topology");
  Explain("/api/topology", topology);
  REQUIRE(topology.status == 200);
  REQUIRE(Contains(topology.body, "\"nodes\":["));
  REQUIRE(Contains(topology.body, "\"self_node_id\":"));

  HttpReply pools = HttpGet(port, "/api/pools");
  Explain("/api/pools", pools);
  REQUIRE(pools.status == 200);
  REQUIRE(Contains(pools.body, "\"pools\":["));
  // The admin pool is always composed, so the list is never empty.
  REQUIRE(Contains(pools.body, "clio_admin"));

  HttpReply config = HttpGet(port, "/api/config");
  Explain("/api/config", config);
  REQUIRE(config.status == 200);
  REQUIRE(Contains(config.body, "\"num_threads\":"));
  REQUIRE(Contains(config.body, "\"viz_port\":"));

  // Every ChiMod's routes and mounts are introspectable -- this is what makes a
  // module's own page discoverable.
  HttpReply routes = HttpGet(port, "/api/routes");
  Explain("/api/routes", routes);
  REQUIRE(routes.status == 200);
  REQUIRE(Contains(routes.body, "/api/health"));
  REQUIRE(Contains(routes.body, "/api/nodes/{node}/workers"));
  REQUIRE(Contains(routes.body, "clio_admin"));

  // ---- Monitor-forwarding endpoints (these submit a task and wait) ----
  HttpReply workers = HttpGet(port, "/api/nodes/local/workers");
  Explain("/api/nodes/local/workers", workers);
  REQUIRE(workers.status == 200);
  REQUIRE(Contains(workers.body, "\"workers\":["));
  REQUIRE(Contains(workers.body, "\"worker_id\":"));

  HttpReply sys = HttpGet(port, "/api/nodes/local/system_stats?min_event_id=0");
  Explain("/api/nodes/local/system_stats", sys);
  REQUIRE(sys.status == 200);
  REQUIRE(Contains(sys.body, "\"entries\":"));

  HttpReply containers = HttpGet(port, "/api/nodes/local/containers");
  Explain("/api/nodes/local/containers", containers);
  REQUIRE(containers.status == 200);
  REQUIRE(Contains(containers.body, "\"containers\":["));

  // Addressing this node by its own id must route locally and give the same
  // answer as "local".
  const clio::run::u64 self_node = CLIO_IPC->GetNodeId();
  HttpReply by_id =
      HttpGet(port, "/api/nodes/" + std::to_string(self_node) + "/workers");
  Explain("/api/nodes/<self>/workers", by_id);
  REQUIRE(by_id.status == 200);
  REQUIRE(Contains(by_id.body, "\"workers\":["));

  // The generic escape hatch: any pool's own Monitor() query. The admin pool
  // answers worker_stats, so this exercises the pool_stats:// forward.
  HttpReply forwarded = HttpGet(
      port, "/api/pools/" + clio::run::kAdminPoolId.ToString() +
                "/monitor?query=worker_stats&routing=local");
  Explain("/api/pools/<admin>/monitor", forwarded);
  REQUIRE(forwarded.status == 200);
  REQUIRE(Contains(forwarded.body, "\"results\":"));
  REQUIRE(Contains(forwarded.body, "\"worker_id\":"));

  // ---- error paths ----
  HttpReply bad_node = HttpGet(port, "/api/nodes/notanumber/workers");
  Explain("/api/nodes/notanumber/workers", bad_node);
  REQUIRE(bad_node.status == 400);
  REQUIRE(Contains(bad_node.body, "\"error\":"));

  // A node id this cluster does not have is refused rather than routed
  // somewhere: physical routing to an unknown id still lands on a node, and
  // labelling its numbers as node 9's would be a lie.
  HttpReply absent_node = HttpGet(port, "/api/nodes/9/workers");
  Explain("/api/nodes/9/workers", absent_node);
  REQUIRE(absent_node.status == 400);

  HttpReply no_query =
      HttpGet(port, "/api/pools/" + clio::run::kAdminPoolId.ToString() +
                        "/monitor");
  REQUIRE(no_query.status == 400);

  // Characters that would re-interpret the pool_stats:// URI are refused rather
  // than spliced into it.
  HttpReply injected = HttpGet(
      port, "/api/pools/301.0/monitor?query=worker_stats%3Aextra%3Abits");
  Explain("/api/pools/301.0/monitor (injection)", injected);
  REQUIRE(injected.status == 400);

  HttpReply missing = HttpGet(port, "/api/does-not-exist");
  REQUIRE(missing.status == 404);
  REQUIRE(Contains(missing.body, "\"error\":"));

  // ---- static assets and the home page ----
  // The admin ChiMod's viz/ directory is mounted automatically because its
  // container registered while the module was loaded from the build tree.
  HttpReply index = HttpGet(port, "/viz/clio_admin/index.html");
  Explain("/viz/clio_admin/index.html", index);
  REQUIRE(index.status == 200);
  REQUIRE(Contains(index.content_type, "text/html"));
  REQUIRE(Contains(index.body, "Clio"));

  HttpReply css = HttpGet(port, "/viz/clio_admin/css/style.css");
  REQUIRE(css.status == 200);
  REQUIRE(Contains(css.content_type, "text/css"));

  HttpReply home = HttpGet(port, "/");
  REQUIRE(home.status == 302);

  HttpReply traversal = HttpGet(port, "/viz/clio_admin/../../etc/passwd");
  REQUIRE(traversal.status == 404);

  // The reported port must be the one that was actually bound (this runtime
  // asked for an ephemeral port), not the 0 that was configured.
  REQUIRE(Contains(config.body,
                   "\"viz_port\":" + std::to_string(port)));
}

TEST_CASE("Viz extends itself when another ChiMod comes up", "[viz]") {
  // The extensibility contract: composing a pool of a ChiMod that ships a viz/
  // directory mounts its pages and registers its routes, with no dashboard-side
  // change. bdev is the second module that does this.
  REQUIRE(InitRuntimeWithViz());
  const clio::run::u32 port = CLIO_VIZ->GetPort();

  // NOTE: no "before" assertion on the route being absent. Whether a bdev pool
  // already exists depends on the config's compose section and on any restart
  // log left behind, so asserting a 404 first would make this test depend on
  // the environment rather than on the mechanism.
  clio::run::PoolId pool_id(4990, 0);
  clio::run::bdev::Client bdev_client(pool_id);
  auto create = bdev_client.AsyncCreate(clio::run::PoolQuery::Dynamic(),
                                        "ram::viz_test_bdev", pool_id,
                                        clio::run::bdev::BdevType::kRam,
                                        16 * 1024 * 1024);
  create.Wait();
  REQUIRE(create->return_code_ == 0);
  const std::string bdev_pool = create->new_pool_id_.ToString();

  // The module's own route now answers, and lists the pool we just made.
  HttpReply pools = HttpGet(port, "/api/mod/clio_bdev/pools");
  Explain("/api/mod/clio_bdev/pools", pools);
  REQUIRE(pools.status == 200);
  REQUIRE(Contains(pools.body, "ram::viz_test_bdev"));
  REQUIRE(Contains(pools.body, bdev_pool));

  // Its page is mounted under its own name, and reachable.
  HttpReply routes = HttpGet(port, "/api/routes");
  REQUIRE(routes.status == 200);
  REQUIRE(Contains(routes.body, "/api/mod/clio_bdev/pools"));
  REQUIRE(Contains(routes.body, "/viz/clio_bdev"));

  HttpReply page = HttpGet(port, "/viz/clio_bdev/index.html");
  Explain("/viz/clio_bdev/index.html", page);
  REQUIRE(page.status == 200);
  REQUIRE(Contains(page.content_type, "text/html"));

  // And the generic forward reaches the bdev container's own Monitor("stats"),
  // which is where the page's per-device numbers come from.
  HttpReply stats =
      HttpGet(port, "/api/pools/" + bdev_pool + "/monitor?query=stats");
  Explain("/api/pools/<bdev>/monitor?query=stats", stats);
  REQUIRE(stats.status == 200);
  REQUIRE(Contains(stats.body, "\"total_capacity\":"));
  REQUIRE(Contains(stats.body, "\"pool_name\":\"ram::viz_test_bdev\""));
}

TEST_CASE("Viz form bodies parse like query parameters", "[viz]") {
  std::map<std::string, std::string> params;
  viz::VizServer::ParseFormBody(
      "pool_name=ram%3A%3Aform&capacity=16MB&note=a+b%26c&flag", &params);
  REQUIRE(params["pool_name"] == "ram::form");
  REQUIRE(params["capacity"] == "16MB");
  REQUIRE(params["note"] == "a b&c");
  REQUIRE(params.count("flag") == 1);
  REQUIRE(params["flag"].empty());

  // Existing keys (the query string, parsed first) win over the body.
  std::map<std::string, std::string> merged{{"action", "validate"}};
  viz::VizServer::ParseFormBody("action=create&other=1", &merged);
  REQUIRE(merged["action"] == "validate");
  REQUIRE(merged["other"] == "1");
}

TEST_CASE("Viz Add Pool: module create forms and the generic compose",
          "[viz]") {
  // The "Add Pool" convention end to end: /api/chimods advertises which loaded
  // modules registered a create form, the module's GET answers a field spec,
  // and its POST validates or creates -- while modules without a form are
  // creatable through the admin's generic compose editor.
  REQUIRE(InitRuntimeWithViz());
  const clio::run::u32 port = CLIO_VIZ->GetPort();

  // A bdev pool must exist for bdev's routes to be registered (the previous
  // test made one; make the dependency explicit rather than ordering-implicit).
  {
    clio::run::PoolId seed_id(4991, 0);
    clio::run::bdev::Client seed(seed_id);
    auto create = seed.AsyncCreate(clio::run::PoolQuery::Dynamic(),
                                   "ram::viz_seed_bdev", seed_id,
                                   clio::run::bdev::BdevType::kRam,
                                   16 * 1024 * 1024);
    create.Wait();
    REQUIRE(create->return_code_ == 0);
  }

  // ---- module inventory ----
  HttpReply chimods = HttpGet(port, "/api/chimods");
  Explain("/api/chimods", chimods);
  REQUIRE(chimods.status == 200);
  REQUIRE(Contains(chimods.body,
                   "\"name\":\"clio_bdev\",\"has_create\":true"));

  // ---- the module's own form ----
  HttpReply spec = HttpGet(port, "/api/mod/clio_bdev/create");
  Explain("GET /api/mod/clio_bdev/create", spec);
  REQUIRE(spec.status == 200);
  REQUIRE(Contains(spec.body, "\"fields\":["));
  REQUIRE(Contains(spec.body, "\"name\":\"pool_name\""));
  REQUIRE(Contains(spec.body, "\"name\":\"bdev_type\""));

  // Validation catches every bad field in one round trip, creating nothing.
  HttpReply invalid = HttpPostForm(
      port, "/api/mod/clio_bdev/create",
      "action=validate&pool_name=&pool_id=notanid&capacity=alot");
  Explain("POST create (invalid)", invalid);
  REQUIRE(invalid.status == 400);
  REQUIRE(Contains(invalid.body, "\"ok\":false"));
  REQUIRE(Contains(invalid.body, "\"pool_name\":"));
  REQUIRE(Contains(invalid.body, "\"pool_id\":"));
  REQUIRE(Contains(invalid.body, "\"capacity\":"));

  HttpReply valid = HttpPostForm(
      port, "/api/mod/clio_bdev/create",
      "action=validate&pool_name=ram::viz_form_bdev&pool_id=4992.0"
      "&bdev_type=ram&capacity=16MB");
  Explain("POST create (validate)", valid);
  REQUIRE(valid.status == 200);
  REQUIRE(Contains(valid.body, "\"ok\":true"));

  // Create for real, then confirm the pool is actually there.
  HttpReply created = HttpPostForm(
      port, "/api/mod/clio_bdev/create",
      "pool_name=ram::viz_form_bdev&pool_id=4992.0&bdev_type=ram"
      "&capacity=16MB");
  Explain("POST create", created);
  REQUIRE(created.status == 200);
  REQUIRE(Contains(created.body, "\"ok\":true"));
  REQUIRE(Contains(created.body, "\"pool_id\":\"4992.0\""));
  HttpReply pools = HttpGet(port, "/api/mod/clio_bdev/pools");
  REQUIRE(Contains(pools.body, "ram::viz_form_bdev"));

  // ---- the generic compose editor (any module, YAML params) ----
  HttpReply composed = HttpPostForm(
      port, "/api/pools/compose",
      "mod_name=clio_bdev&pool_name=ram::viz_composed_bdev&pool_id=4993.0"
      "&pool_query=local&config=" +
          std::string("bdev_type%3A%20ram%0Acapacity%3A%2016MB"));
  Explain("POST /api/pools/compose", composed);
  REQUIRE(composed.status == 200);
  REQUIRE(Contains(composed.body, "\"ok\":true"));
  HttpReply pools2 = HttpGet(port, "/api/mod/clio_bdev/pools");
  REQUIRE(Contains(pools2.body, "ram::viz_composed_bdev"));

  // An unknown module is a validation error, not a create attempt.
  HttpReply bad_mod = HttpPostForm(
      port, "/api/pools/compose",
      "mod_name=clio_nonexistent&pool_name=x&pool_id=4994.0");
  REQUIRE(bad_mod.status == 400);
  REQUIRE(Contains(bad_mod.body, "\"mod_name\":"));
}

TEST_CASE("Viz safe-bdev website: create, add and remove members", "[viz]") {
  // The safe-bdev page's whole lifecycle over HTTP: create member bdevs via the
  // bdev form, create the array via safe-bdev's form, watch it through
  // Monitor("stats"), grow it with add_member, shrink it with remove_member.
  REQUIRE(InitRuntimeWithViz());
  const clio::run::u32 port = CLIO_VIZ->GetPort();

  // Three data members, created through the bdev module's own create route.
  for (int i = 0; i < 3; ++i) {
    HttpReply member = HttpPostForm(
        port, "/api/mod/clio_bdev/create",
        "pool_name=ram::viz_safe_member_" + std::to_string(i) +
            "&pool_id=" + std::to_string(4950 + i) +
            ".0&bdev_type=ram&capacity=16MB");
    REQUIRE(member.status == 200);
  }

  // Validation: a member name that does not exist is reported by name.
  HttpReply invalid = HttpPostForm(
      port, "/api/mod/clio_safe_bdev/create",
      "action=validate&pool_name=viz_safe_array&pool_id=4960.0"
      "&max_failures=1&members=ram::viz_safe_member_0,ram::no_such_pool");
  Explain("safe-bdev create (invalid)", invalid);
  REQUIRE(invalid.status == 400);
  REQUIRE(Contains(invalid.body, "no_such_pool"));

  // Create the array over the three members.
  HttpReply created = HttpPostForm(
      port, "/api/mod/clio_safe_bdev/create",
      "pool_name=viz_safe_array&pool_id=4960.0&max_failures=1"
      "&members=ram::viz_safe_member_0,%20ram::viz_safe_member_1,"
      "%20ram::viz_safe_member_2");
  Explain("safe-bdev create", created);
  REQUIRE(created.status == 200);
  REQUIRE(Contains(created.body, "\"ok\":true"));
  REQUIRE(Contains(created.body, "\"data_members\":3"));

  // The module's page and index are up.
  HttpReply page = HttpGet(port, "/viz/clio_safe_bdev/index.html");
  REQUIRE(page.status == 200);
  HttpReply pools = HttpGet(port, "/api/mod/clio_safe_bdev/pools");
  Explain("/api/mod/clio_safe_bdev/pools", pools);
  REQUIRE(pools.status == 200);
  REQUIRE(Contains(pools.body, "viz_safe_array"));

  // Live state through the generic Monitor forward -- what the page polls.
  HttpReply stats =
      HttpGet(port, "/api/pools/4960.0/monitor?query=stats&routing=local");
  Explain("safe-bdev stats", stats);
  REQUIRE(stats.status == 200);
  REQUIRE(Contains(stats.body, "\"data_count\":3"));
  REQUIRE(Contains(stats.body, "\"members\":["));

  // Grow the array: a parity member the handler creates on the spot.
  HttpReply added = HttpPostForm(
      port, "/api/mod/clio_safe_bdev/4960.0/add_member",
      "member_name=ram::viz_safe_parity&capacity=16MB&bdev_type=ram"
      "&as_parity=1");
  Explain("safe-bdev add_member", added);
  REQUIRE(added.status == 200);
  REQUIRE(Contains(added.body, "\"ok\":true"));
  const std::string member_id_key = "\"member_pool_id\":\"";
  const size_t id_at = added.body.find(member_id_key) + member_id_key.size();
  const std::string parity_id =
      added.body.substr(id_at, added.body.find('"', id_at) - id_at);

  HttpReply grown =
      HttpGet(port, "/api/pools/4960.0/monitor?query=stats&routing=local");
  REQUIRE(Contains(grown.body, "\"parity_level\":1"));
  REQUIRE(Contains(grown.body, "viz_safe_parity"));

  // Shrink it again: remove the parity member we just added.
  HttpReply removed = HttpPostForm(
      port, "/api/mod/clio_safe_bdev/4960.0/remove_member",
      "member_pool_id=" + parity_id + "&was_faulty=0");
  Explain("safe-bdev remove_member", removed);
  REQUIRE(removed.status == 200);
  REQUIRE(Contains(removed.body, "\"ok\":true"));

  // Repair flow: mark a data member faulty, then replace + recover onto a
  // fresh bdev the handler creates on the spot. Recovery of an empty array
  // finishes immediately; the roster shows the replacement.
  {
    HttpReply faulted = HttpPostForm(
        port, "/api/mod/clio_safe_bdev/4960.0/remove_member",
        "member_pool_id=4950.0&was_faulty=1");
    Explain("safe-bdev remove (faulty)", faulted);
    REQUIRE(faulted.status == 200);

    HttpReply replaced = HttpPostForm(
        port, "/api/mod/clio_safe_bdev/4960.0/replace_member",
        "failed_pool_id=4950.0&member_name=ram::viz_safe_repl&capacity=16MB"
        "&bdev_type=ram");
    Explain("safe-bdev replace_member", replaced);
    REQUIRE(replaced.status == 200);
    REQUIRE(Contains(replaced.body, "\"ok\":true"));

    HttpReply after = HttpGet(
        port, "/api/pools/4960.0/monitor?query=stats&routing=local");
    REQUIRE(Contains(after.body, "viz_safe_repl"));
  }

  // Error paths: a missing member name, and a pool that is not a safe-bdev.
  HttpReply no_member = HttpPostForm(
      port, "/api/mod/clio_safe_bdev/4960.0/add_member", "member_name=");
  REQUIRE(no_member.status == 400);
  HttpReply wrong_pool = HttpPostForm(
      port, "/api/mod/clio_safe_bdev/4950.0/add_member",
      "member_name=ram::viz_safe_member_0");
  REQUIRE(wrong_pool.status == 404);
}

TEST_CASE("Viz CTE website: targets register and unregister by name",
          "[viz]") {
  // The CTE page's lifecycle over HTTP: create a bare CTE pool via its form,
  // register a bdev by name as a placement target, see it in the roster with
  // score and capacity, then unregister it.
  REQUIRE(InitRuntimeWithViz());
  const clio::run::u32 port = CLIO_VIZ->GetPort();

  HttpReply created = HttpPostForm(
      port, "/api/mod/clio_cte_core/create",
      "pool_name=viz_cte_pool&pool_id=4970.0");
  Explain("CTE create", created);
  REQUIRE(created.status == 200);
  REQUIRE(Contains(created.body, "\"ok\":true"));

  HttpReply page = HttpGet(port, "/viz/clio_cte_core/index.html");
  REQUIRE(page.status == 200);
  HttpReply pools = HttpGet(port, "/api/mod/clio_cte_core/pools");
  REQUIRE(pools.status == 200);
  REQUIRE(Contains(pools.body, "viz_cte_pool"));

  // Register a fresh RAM bdev, by name, as a target.
  HttpReply registered = HttpPostForm(
      port, "/api/mod/clio_cte_core/4970.0/register_target",
      "name=ram::viz_cte_tier&bdev_type=ram&capacity=16MB");
  Explain("CTE register_target", registered);
  REQUIRE(registered.status == 200);
  REQUIRE(Contains(registered.body, "\"ok\":true"));

  HttpReply targets = HttpGet(port, "/api/mod/clio_cte_core/4970.0/targets");
  Explain("CTE targets", targets);
  REQUIRE(targets.status == 200);
  REQUIRE(Contains(targets.body, "ram::viz_cte_tier"));
  REQUIRE(Contains(targets.body, "\"remaining_space\":"));

  // The pool-card summary reads Monitor("stats"); the CTE now answers it with
  // what it is holding.
  HttpReply cte_stats =
      HttpGet(port, "/api/pools/4970.0/monitor?query=stats&routing=local");
  Explain("CTE monitor stats", cte_stats);
  REQUIRE(cte_stats.status == 200);
  REQUIRE(Contains(cte_stats.body, "\"num_targets\":1"));
  REQUIRE(Contains(cte_stats.body, "\"num_tags\":"));
  REQUIRE(Contains(cte_stats.body, "\"num_blobs\":"));

  // And gone again.
  HttpReply unregistered = HttpPostForm(
      port, "/api/mod/clio_cte_core/4970.0/unregister_target",
      "name=ram::viz_cte_tier");
  Explain("CTE unregister_target", unregistered);
  REQUIRE(unregistered.status == 200);
  HttpReply after = HttpGet(port, "/api/mod/clio_cte_core/4970.0/targets");
  REQUIRE_FALSE(Contains(after.body, "ram::viz_cte_tier"));

  // Error paths: capacity is required for a fresh bdev, and a non-CTE pool 404s.
  HttpReply no_cap = HttpPostForm(
      port, "/api/mod/clio_cte_core/4970.0/register_target",
      "name=ram::viz_cte_tier2&bdev_type=ram&capacity=0");
  REQUIRE(no_cap.status == 400);
  HttpReply wrong_pool = HttpPostForm(
      port, "/api/mod/clio_cte_core/4960.0/register_target",
      "name=ram::x&capacity=16MB");
  REQUIRE(wrong_pool.status == 404);
}

TEST_CASE("Viz pools can be shut down from the dashboard", "[viz]") {
  // The Pools tab's per-card "x": create a pool over HTTP, destroy it over
  // HTTP, and verify both the guard rails -- the admin pool is refused (that
  // would be killing the runtime, not managing a pool) and a second destroy
  // reports the pool gone.
  REQUIRE(InitRuntimeWithViz());
  const clio::run::u32 port = CLIO_VIZ->GetPort();

  HttpReply created = HttpPostForm(
      port, "/api/mod/clio_bdev/create",
      "pool_name=ram::viz_doomed_bdev&pool_id=4995.0&bdev_type=ram"
      "&capacity=16MB");
  Explain("create doomed bdev", created);
  REQUIRE(created.status == 200);
  REQUIRE(Contains(HttpGet(port, "/api/mod/clio_bdev/pools").body,
                   "ram::viz_doomed_bdev"));

  HttpReply destroyed = HttpPostForm(port, "/api/pools/4995.0/destroy", "");
  Explain("destroy pool", destroyed);
  REQUIRE(destroyed.status == 200);
  REQUIRE(Contains(destroyed.body, "\"ok\":true"));
  REQUIRE(Contains(destroyed.body, "ram::viz_doomed_bdev"));
  REQUIRE_FALSE(Contains(HttpGet(port, "/api/mod/clio_bdev/pools").body,
                         "ram::viz_doomed_bdev"));

  // Gone means gone: a second destroy is a 404, not a silent success.
  HttpReply again = HttpPostForm(port, "/api/pools/4995.0/destroy", "");
  REQUIRE(again.status == 404);

  // The admin pool is the runtime itself; the dashboard must refuse.
  HttpReply admin_pool = HttpPostForm(
      port, "/api/pools/" + clio::run::kAdminPoolId.ToString() + "/destroy",
      "");
  Explain("destroy admin pool (refused)", admin_pool);
  REQUIRE(admin_pool.status == 403);
  REQUIRE(Contains(HttpGet(port, "/api/health").body, "\"ok\":true"));

  // A BARE POST -- no body, no Content-Length, no Content-Type, which is what
  // `curl -X POST` sends -- must answer promptly. The bridge once handed such
  // a request an unbounded body stream, so every bodyless action sat on
  // Poco's full 60s receive timeout while the action itself had long
  // succeeded. Sent raw because a well-behaved HTTP client library always
  // adds Content-Length and would never exercise this.
  {
    Poco::Net::StreamSocket socket(Poco::Net::SocketAddress(
        "127.0.0.1", static_cast<Poco::UInt16>(port)));
    socket.setReceiveTimeout(Poco::Timespan(10, 0));
    const std::string raw =
        "POST /api/pools/4995.0/destroy HTTP/1.1\r\n"
        "Host: 127.0.0.1\r\nConnection: close\r\n\r\n";
    socket.sendBytes(raw.data(), static_cast<int>(raw.size()));
    char buf[512];
    const auto start = std::chrono::steady_clock::now();
    const int n = socket.receiveBytes(buf, sizeof(buf));  // throws on timeout
    const double sec = std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - start)
                           .count();
    std::cout << "  bare POST answered in " << sec << "s" << std::endl;
    REQUIRE(n > 0);
    REQUIRE(Contains(std::string(buf, static_cast<size_t>(n)), "404"));
    REQUIRE(sec < 5.0);
  }
}

// ===========================================================================
// Dashboard teardown (issue #1020)
//
// Stop() used to deadlock: the HTTPServer(factory, socket, params) overload
// borrows Poco's process-wide ThreadPool::defaultPool(), so tearing the
// dashboard down joined threads it never owned and stopAll() never returned.
// ServerFinalize() stalled behind it and the RequestStop watchdog force-exited
// the daemon with code 2.
//
// These cases drive Stop() directly, on a private VizServer rather than the
// runtime's singleton, so the teardown is exercised as its own unit instead of
// only incidentally at process exit.
// ===========================================================================

TEST_CASE("Viz dashboard starts and stops without wedging", "[viz][shutdown]") {
  clio::run::viz::VizServer server;
  // Port 0: an ephemeral port, so this never collides with the singleton
  // dashboard the other cases in this file are talking to.
  REQUIRE(server.StartOn("127.0.0.1", 0, /*max_threads=*/4));
  REQUIRE(server.IsRunning());
  REQUIRE(server.GetPort() != 0);
  REQUIRE(server.GetBindAddr() == "127.0.0.1");

  // The regression: this call is what used to never return. Time it, and hold
  // it to a budget far below the watchdog's, so a reintroduced deadlock fails
  // the test instead of quietly slowing every shutdown down.
  auto t0 = std::chrono::steady_clock::now();
  server.Stop();
  double ms = std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - t0)
                  .count();
  std::cout << "  Stop() took " << ms << " ms" << std::endl;
  REQUIRE(!server.IsRunning());
  REQUIRE(server.GetPort() == 0);
  REQUIRE(ms < 3000.0);  // the teardown budget; a deadlock would spend it all

  // Idempotent: stopping an already-stopped dashboard is a no-op, not a crash.
  server.Stop();
  REQUIRE(!server.IsRunning());
}

TEST_CASE("Viz teardown abandons a slow dashboard instead of blocking",
          "[viz][shutdown]") {
  // CLIO_VIZ_STOP_BUDGET_MS=0 forces the abandon path: Stop() must give up on
  // the Poco teardown immediately and return, leaving the server logically
  // stopped. This is the backstop that keeps a blocked request handler from
  // holding the whole daemon's shutdown hostage -- without a knob it would be
  // a branch no test could reach.
  SIMPLE_TEST_SETENV("CLIO_VIZ_STOP_BUDGET_MS", "0");

  clio::run::viz::VizServer server;
  REQUIRE(server.StartOn("127.0.0.1", 0, /*max_threads=*/4));
  REQUIRE(server.IsRunning());

  auto t0 = std::chrono::steady_clock::now();
  server.Stop();
  double ms = std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - t0)
                  .count();
  std::cout << "  Stop() with a 0 ms budget took " << ms << " ms" << std::endl;
  REQUIRE(!server.IsRunning());
  REQUIRE(ms < 1000.0);  // must not wait out anything

  SIMPLE_TEST_SETENV("CLIO_VIZ_STOP_BUDGET_MS", "");
}

SIMPLE_TEST_MAIN()
