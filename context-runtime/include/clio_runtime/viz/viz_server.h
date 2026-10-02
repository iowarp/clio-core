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

#ifndef CLIO_RUNTIME_INCLUDE_VIZ_VIZ_SERVER_H_
#define CLIO_RUNTIME_INCLUDE_VIZ_VIZ_SERVER_H_

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "clio_runtime/types.h"

namespace clio::run {
class Container;
}  // namespace clio::run

/**
 * The in-runtime web dashboard ("viz").
 *
 * Every ChiMod may ship a `viz/` directory of HTML/CSS/JS next to its sources
 * and register HTTP routes from its Container. The admin ChiMod starts the
 * server (one per node, no collectives) and mounts its own dashboard as the
 * home page; every other ChiMod's assets mount under /viz/<mod_name>/.
 *
 * Route handlers run on the HTTP server's own thread pool (Poco), NOT on a
 * runtime worker, so a handler MAY block: reading node-local manager state and
 * submitting a task + Future::Wait() are both legal. What a handler must never
 * do is co_await -- it is not a coroutine.
 */
namespace clio::run::viz {

/** A parsed HTTP request, as handed to a route handler. */
struct Request {
  /** Uppercased HTTP verb ("GET", "POST", ...). */
  std::string method;
  /** Percent-decoded path with no query string, e.g. "/api/nodes/0/workers". */
  std::string path;
  /** Values bound by `{name}` segments of the matched route pattern. */
  std::map<std::string, std::string> path_vars;
  /** Decoded ?key=value query parameters. */
  std::map<std::string, std::string> params;
  /** Request body (empty for GET). */
  std::string body;

  /** @return query parameter @p key, or @p dflt when absent. */
  std::string Param(const std::string &key,
                    const std::string &dflt = "") const {
    auto it = params.find(key);
    return (it == params.end()) ? dflt : it->second;
  }

  /** @return path variable @p key, or @p dflt when the route had no such
   *  `{key}` segment. */
  std::string Var(const std::string &key, const std::string &dflt = "") const {
    auto it = path_vars.find(key);
    return (it == path_vars.end()) ? dflt : it->second;
  }
};

/** A response a route handler fills in. Defaults to an empty 200 JSON body. */
struct Response {
  int status = 200;
  std::string content_type = "application/json";
  std::string body;
  /** Extra headers, sent verbatim. */
  std::vector<std::pair<std::string, std::string>> headers;

  /** Reply with a JSON body (the caller has already serialized it). */
  void Json(const std::string &json) {
    content_type = "application/json";
    body = json;
  }

  /** Reply with an error status and a `{"error": "..."}` JSON body. */
  void Error(int code, const std::string &message);
};

/**
 * A route handler. Runs on an HTTP worker thread; see the file comment for what
 * is allowed. Handlers must be safe to call concurrently.
 */
using Handler = std::function<void(const Request &, Response &)>;

/** A route as registered by a Container. */
struct Route {
  /** HTTP verb to match; case-insensitive. */
  std::string method = "GET";
  /**
   * Path pattern. Segments of the form `{name}` match exactly one path segment
   * and bind it into Request::path_vars, e.g.
   * "/api/nodes/{node}/workers".
   */
  std::string path;
  /** ChiMod that owns this route (for /api/routes introspection). */
  std::string mod_name;
  /** One-line human description (for /api/routes introspection). */
  std::string doc;
  /** The handler to invoke. */
  Handler handler;
};

/** Registered-route metadata, without the handler (for introspection). */
struct RouteInfo {
  std::string method;
  std::string path;
  std::string mod_name;
  std::string doc;
};

/** A mounted static-asset directory. */
struct MountInfo {
  /** URL prefix, e.g. "/viz/clio_admin". */
  std::string url_prefix;
  /** Absolute directory on disk. */
  std::string dir;
  /** Owning ChiMod. */
  std::string mod_name;
};

/**
 * The node-local web server and route registry.
 *
 * Singleton, accessed through CLIO_VIZ. Start()/Stop() are driven by the admin
 * Container; AddRoute()/MountAssets() are called by any Container's
 * RegisterViz() override.
 */
class VizServer {
 public:
  // Both are defined in viz_server.cc, not here: Impl is incomplete in this
  // header (it holds the Poco server), and an inline constructor would need it
  // complete to emit the member's cleanup path.
  VizServer();
  ~VizServer();

  VizServer(const VizServer &) = delete;
  VizServer &operator=(const VizServer &) = delete;

  /**
   * Start listening, using the ConfigManager's viz settings.
   *
   * Never fatal: returns false (with a warning logged) when the viz is
   * disabled, when the build has no HTTP transport, or when the port is
   * already taken -- the runtime must come up either way.
   *
   * @return true if the server is now accepting connections.
   */
  bool Start();

  /**
   * Start listening on an explicit endpoint, ignoring the config's
   * enabled/port/bind settings. For tests.
   * @param bind_addr Address to bind ("127.0.0.1", "0.0.0.0", ...)
   * @param port TCP port; 0 binds an ephemeral port (see GetPort()).
   * @param max_threads Size of the HTTP thread pool.
   */
  bool StartOn(const std::string &bind_addr, u32 port, u32 max_threads);

  /** Stop the server and wait for in-flight requests. Idempotent. */
  void Stop();

  /** @return true between a successful Start() and Stop(). */
  bool IsRunning() const;

  /** @return the port actually bound, or 0 when not running. Resolves the
   *  ephemeral (port 0) case. */
  u32 GetPort() const;

  /** @return the address the listener is bound to, or "" when not running. */
  std::string GetBindAddr() const;

  /**
   * Register a route. First registration of a given (method, path) wins, so a
   * pool with several containers -- or several pools of the same ChiMod --
   * registers its routes idempotently.
   * @return true if the route was added, false if that (method, path) existed.
   */
  bool AddRoute(Route route);

  /**
   * Mount a ChiMod's `viz/` directory as static assets.
   *
   * The directory is located next to the ChiMod's loaded shared library, which
   * is what makes a built-but-not-installed tree and an installed tree both
   * work with no configuration: see AssetSearchDirs().
   *
   * @param mod_name ChiMod name as reported by get_chimod_name()
   * @param url_prefix URL to mount at; defaults to "/viz/<mod_name>"
   * @return true if a directory was found and mounted.
   */
  bool MountAssets(const std::string &mod_name,
                   const std::string &url_prefix = "");

  /**
   * Mount an explicit directory, skipping discovery. For a ChiMod whose pages
   * live somewhere only it knows (a config-supplied path), and for tests.
   * @param mod_name Owning ChiMod, for introspection
   * @param dir Directory to serve
   * @param url_prefix URL to mount at; defaults to "/viz/<mod_name>"
   * @return true if mounted; false if @p dir is not a directory or that prefix
   *         is already mounted.
   */
  bool MountDir(const std::string &mod_name, const std::string &dir,
                const std::string &url_prefix = "");

  /**
   * Drop every route and asset mount owned by @p mod_name, and the home
   * redirect if it pointed at that module's pages.
   *
   * A ChiMod whose handler captures container state MUST call this when the
   * container goes away, or a later request would run a handler over a
   * destroyed container. Handlers that only capture by value and read the
   * manager singletons (the recommended shape) do not need it.
   *
   * Stop the server first if handlers may be in flight: Stop() waits for them,
   * this does not.
   * @return number of routes removed.
   */
  size_t RemoveModule(const std::string &mod_name);

  /** Point "/" at @p url (a 302 redirect). The admin dashboard claims this. */
  void SetHome(const std::string &url);

  /** @return the current home redirect target ("" when unset). */
  std::string GetHome() const;

  /** @return metadata for every registered route. */
  std::vector<RouteInfo> GetRoutes() const;

  /** @return every mounted asset directory. */
  std::vector<MountInfo> GetMounts() const;

  /**
   * Resolve @p req against the route table, then the asset mounts, then the
   * home redirect. Called by the HTTP layer; exposed so tests can drive the
   * router without a socket.
   * @return true if something answered (@p resp is filled), false for 404.
   */
  bool Dispatch(const Request &req, Response &resp);

  /**
   * Hook invoked by PoolManager once a container is published, so a ChiMod's
   * routes and assets register themselves without every module needing startup
   * code of its own. Safe to call repeatedly for the same ChiMod.
   */
  void OnContainerRegistered(const std::string &mod_name, Container &container);

  /**
   * Candidate directories for a ChiMod's viz assets, most-specific first:
   *   1. `<dir>/<mod_name>` for every colon-separated entry of $CLIO_VIZ_PATH
   *   2. `<libdir>/viz/<mod_name>`            -- built, not installed
   *      (libraries land in <build>/bin, and the build copies viz trees to
   *      <build>/bin/viz/<mod_name>)
   *   3. `<libdir>/../share/clio/viz/<mod_name>` -- installed
   *      (libraries land in <prefix>/lib, assets in <prefix>/share/clio/viz)
   * where `<libdir>` is the directory of the ChiMod's own loaded library, so
   * the answer follows whichever copy of the module the runtime actually
   * loaded.
   */
  static std::vector<std::string> AssetSearchDirs(const std::string &mod_name);

  /** @return the first entry of AssetSearchDirs() that exists, or "". */
  static std::string FindAssetDir(const std::string &mod_name);

  /** @return the MIME type for a file name's extension (text/plain if
   *  unknown). */
  static std::string MimeTypeOf(const std::string &path);

  /**
   * Match @p pattern (which may contain `{name}` segments) against @p path,
   * binding any variables into @p vars.
   * @return true on a full match.
   */
  static bool MatchPath(const std::string &pattern, const std::string &path,
                        std::map<std::string, std::string> *vars);

  /**
   * Parse an `application/x-www-form-urlencoded` body ("a=1&b=x%20y") into
   * @p out, percent-decoding keys and values. Keys already present in @p out
   * are NOT overwritten, so calling this after the query string has been
   * parsed gives the query string precedence. The HTTP layer applies this to
   * every POST with that content type; exposed so tests (and non-HTTP callers)
   * can build the same Request a form submission produces.
   */
  static void ParseFormBody(const std::string &body,
                            std::map<std::string, std::string> *out);

 private:
  /** Resolve a mount + URL to a file on disk, rejecting traversal. */
  bool ResolveAsset(const Request &req, std::string *file_path,
                    std::string *mime) const;

  mutable std::mutex mutex_;
  std::vector<Route> routes_;
  std::vector<MountInfo> mounts_;
  std::string home_;

  /** Poco HTTP server state; defined in the .cc so no Poco header leaks into
   *  the installed headers (Poco::Foundation's interface definitions collide
   *  with the POSIX interception adapters -- see CTP's src/CMakeLists.txt). */
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/**
 * Pick a pool MAJOR id no existing pool uses, starting at @p base.
 *
 * Explicit pool ids are mandatory (PoolManager rejects null ids), so every
 * dashboard create form needs a sensible default; this is that default. It is a
 * node-local suggestion, not a reservation — two racing creates can still
 * collide, and the create itself reports that — so forms treat it as a
 * prefilled field the user may edit.
 *
 * @param base First candidate major (each ChiMod picks its own range so the
 *             suggestions do not trample the well-known ids)
 * @return A major id such that (major, 0) names no current pool
 */
u32 SuggestFreePoolMajor(u32 base);

/**
 * Parse a human size string ("16MB", "1g", "4096") into bytes, REJECTING
 * garbage instead of dying.
 *
 * ctp::ConfigParse::ParseSize exits the process on an unknown suffix -- fine
 * for a config file the operator wrote, fatal for a dashboard form field an
 * HTTP client typed. Every viz handler that accepts a size must use this.
 *
 * Grammar matches ConfigParse: optional decimal number, optional K/M/G/T/P
 * suffix (case-insensitive, trailing "B"/"b" tolerated), whitespace ignored.
 * @return true and set @p out on success; false on anything else.
 */
bool ParseSizeField(const std::string &text, u64 *out);

}  // namespace clio::run::viz

// Global pointer variable declaration for the viz server singleton
CLIO_RUN_DEFINE_GLOBAL_PTR_VAR_H(clio::run::viz::VizServer, g_viz_server);

/** Access the node-local viz server singleton. */
#define CLIO_VIZ CTP_GET_GLOBAL_PTR_VAR(::clio::run::viz::VizServer, g_viz_server)

#endif  // CLIO_RUNTIME_INCLUDE_VIZ_VIZ_SERVER_H_
