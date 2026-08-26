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
 * Node-local web dashboard: route registry, static-asset resolution, and the
 * Poco HTTP transport that fronts them.
 */

#include "clio_runtime/viz/viz_server.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <memory>
#include <set>
#include <sstream>
#include <thread>

#include "clio_runtime/container.h"
#include "clio_runtime/module_manager.h"
#include "clio_runtime/config_manager.h"
#include "clio_runtime/pool_manager.h"
#include "clio_runtime/viz/viz_json.h"

#if CLIO_RUN_HAS_POCO
#include <Poco/Net/Net.h>
#include <Poco/Net/HTTPRequestHandler.h>
#include <Poco/Net/HTTPRequestHandlerFactory.h>
#include <Poco/Net/HTTPServer.h>
#include <Poco/Net/HTTPServerParams.h>
#include <Poco/Net/HTTPServerRequest.h>
#include <Poco/Net/HTTPServerResponse.h>
#include <Poco/Net/ServerSocket.h>
#include <Poco/Net/SocketAddress.h>
#include <Poco/StreamCopier.h>
#include <Poco/ThreadPool.h>
#include <Poco/URI.h>
#endif

// Global pointer variable definition for the viz server singleton
CLIO_RUN_DEFINE_GLOBAL_PTR_VAR_CC(clio::run::viz::VizServer, g_viz_server);

namespace clio::run::viz {

namespace {

/** Largest static asset the dashboard will serve. Assets are hand-written
 *  HTML/CSS/JS; anything larger is a packaging accident, not a page. */
constexpr size_t kMaxAssetBytes = 16u * 1024u * 1024u;

/** Largest request body accepted (POST route arguments). */
constexpr size_t kMaxBodyBytes = 1u * 1024u * 1024u;

/** Split @p path on '/', dropping empty segments. */
std::vector<std::string> SplitPath(const std::string &path) {
  std::vector<std::string> out;
  size_t start = 0;
  while (start <= path.size()) {
    size_t slash = path.find('/', start);
    std::string seg = (slash == std::string::npos)
                          ? path.substr(start)
                          : path.substr(start, slash - start);
    if (!seg.empty()) {
      out.push_back(seg);
    }
    if (slash == std::string::npos) break;
    start = slash + 1;
  }
  return out;
}

std::string ToUpper(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  return s;
}

/** Split a PATH-style env value on the platform's list separator. */
std::vector<std::string> SplitSearchPath(const std::string &value) {
#ifdef _WIN32
  const char kSep = ';';
#else
  const char kSep = ':';
#endif
  std::vector<std::string> out;
  size_t start = 0;
  while (start <= value.size()) {
    size_t pos = value.find(kSep, start);
    std::string part = (pos == std::string::npos) ? value.substr(start)
                                                  : value.substr(start, pos - start);
    if (!part.empty()) {
      out.push_back(part);
    }
    if (pos == std::string::npos) break;
    start = pos + 1;
  }
  return out;
}

/** Read a whole file, capped at kMaxAssetBytes. */
bool ReadWholeFile(const std::string &path, std::string *out) {
  std::error_code ec;
  auto size = std::filesystem::file_size(path, ec);
  if (ec || size > kMaxAssetBytes) {
    return false;
  }
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return false;
  }
  std::ostringstream buf;
  buf << in.rdbuf();
  *out = buf.str();
  return true;
}

/**
 * @return true if @p rel is safe to append to a mount directory: no absolute
 * paths, no "..", no NUL, no Windows drive letters or backslashes. The URL is
 * already percent-decoded by this point, which is exactly why the check has to
 * run on the decoded form.
 */
bool IsSafeRelativePath(const std::string &rel) {
  if (rel.empty()) return true;
  if (rel.find('\0') != std::string::npos) return false;
  if (rel.find('\\') != std::string::npos) return false;
  if (rel.find(':') != std::string::npos) return false;
  if (rel.front() == '/') return false;
  for (const std::string &seg : SplitPath(rel)) {
    if (seg == "..") return false;
  }
  return true;
}

}  // namespace

void Response::Error(int code, const std::string &message) {
  status = code;
  content_type = "application/json";
  body = std::string("{\"error\":") + JsonQuote(message) + "}";
}

//===========================================================================
// Path matching, MIME types, asset discovery
//===========================================================================

bool VizServer::MatchPath(const std::string &pattern, const std::string &path,
                          std::map<std::string, std::string> *vars) {
  std::vector<std::string> pat = SplitPath(pattern);
  std::vector<std::string> got = SplitPath(path);
  if (pat.size() != got.size()) {
    return false;
  }
  std::map<std::string, std::string> bound;
  for (size_t i = 0; i < pat.size(); ++i) {
    const std::string &p = pat[i];
    if (p.size() >= 2 && p.front() == '{' && p.back() == '}') {
      bound[p.substr(1, p.size() - 2)] = got[i];
    } else if (p != got[i]) {
      return false;
    }
  }
  if (vars) {
    *vars = std::move(bound);
  }
  return true;
}

namespace {

/** Percent-decode a form component ('+' means space in form encoding). */
std::string FormDecode(const std::string &in) {
  std::string out;
  out.reserve(in.size());
  for (size_t i = 0; i < in.size(); ++i) {
    char c = in[i];
    if (c == '+') {
      out.push_back(' ');
    } else if (c == '%' && i + 2 < in.size() && std::isxdigit(in[i + 1]) &&
               std::isxdigit(in[i + 2])) {
      auto hex = [](char h) {
        return (h >= '0' && h <= '9')   ? h - '0'
               : (h >= 'a' && h <= 'f') ? h - 'a' + 10
                                        : h - 'A' + 10;
      };
      out.push_back(static_cast<char>(hex(in[i + 1]) * 16 + hex(in[i + 2])));
      i += 2;
    } else {
      out.push_back(c);
    }
  }
  return out;
}

}  // namespace

void VizServer::ParseFormBody(const std::string &body,
                              std::map<std::string, std::string> *out) {
  size_t start = 0;
  while (start <= body.size()) {
    size_t amp = body.find('&', start);
    std::string pair = (amp == std::string::npos)
                           ? body.substr(start)
                           : body.substr(start, amp - start);
    if (!pair.empty()) {
      size_t eq = pair.find('=');
      std::string key =
          FormDecode(eq == std::string::npos ? pair : pair.substr(0, eq));
      std::string value =
          (eq == std::string::npos) ? "" : FormDecode(pair.substr(eq + 1));
      if (!key.empty()) {
        out->emplace(key, value);  // emplace: existing (query) keys win
      }
    }
    if (amp == std::string::npos) break;
    start = amp + 1;
  }
}

bool ParseSizeField(const std::string &text, u64 *out) {
  // Strip whitespace, split into the numeric run and the suffix.
  std::string number;
  std::string suffix;
  bool in_suffix = false;
  bool seen_dot = false;
  for (char c : text) {
    if (std::isspace(static_cast<unsigned char>(c))) {
      continue;
    }
    if (!in_suffix && (std::isdigit(static_cast<unsigned char>(c)) ||
                       (c == '.' && !seen_dot))) {
      seen_dot = seen_dot || (c == '.');
      number.push_back(c);
    } else {
      in_suffix = true;
      suffix.push_back(c);
    }
  }
  if (number.empty() || number == ".") {
    return false;
  }
  double value = 0.0;
  try {
    value = std::stod(number);
  } catch (const std::exception &) {
    return false;
  }
  // Optional trailing "B"/"b" ("MB", "kb", plain "B") is unit noise.
  if (!suffix.empty() && (suffix.back() == 'B' || suffix.back() == 'b')) {
    suffix.pop_back();
  }
  u64 scale = 1;
  if (suffix.empty()) {
    scale = 1;
  } else if (suffix.size() == 1) {
    switch (std::toupper(static_cast<unsigned char>(suffix[0]))) {
      case 'K': scale = 1ull << 10; break;
      case 'M': scale = 1ull << 20; break;
      case 'G': scale = 1ull << 30; break;
      case 'T': scale = 1ull << 40; break;
      case 'P': scale = 1ull << 50; break;
      default: return false;
    }
  } else {
    return false;
  }
  *out = static_cast<u64>(value * static_cast<double>(scale));
  return true;
}

u32 SuggestFreePoolMajor(u32 base) {
  // Majors this process has already handed out. Without this, two concurrent
  // requests (or one request that creates two pools, like a register-target
  // that first creates its bdev) would both be suggested the same free major,
  // and the second create would silently bind to the FIRST pool (create is
  // get-or-create by id). Static because suggestions are process-wide advice;
  // the set stays tiny (one entry per dashboard-created pool).
  static std::mutex suggest_mutex;
  static std::set<u32> suggested;

  auto *pool_manager = CLIO_POOL_MANAGER;
  std::set<u32> taken;
  if (pool_manager) {
    for (const auto &pid : pool_manager->GetAllPoolIds()) {
      taken.insert(pid.major_);
    }
  }
  std::lock_guard<std::mutex> guard(suggest_mutex);
  u32 major = base;
  while (taken.count(major) || suggested.count(major)) {
    ++major;
  }
  suggested.insert(major);
  return major;
}

std::string VizServer::MimeTypeOf(const std::string &path) {
  static const std::pair<const char *, const char *> kTypes[] = {
      {".html", "text/html; charset=utf-8"},
      {".htm", "text/html; charset=utf-8"},
      {".css", "text/css; charset=utf-8"},
      {".js", "application/javascript; charset=utf-8"},
      {".mjs", "application/javascript; charset=utf-8"},
      {".json", "application/json"},
      {".map", "application/json"},
      {".svg", "image/svg+xml"},
      {".png", "image/png"},
      {".jpg", "image/jpeg"},
      {".jpeg", "image/jpeg"},
      {".gif", "image/gif"},
      {".ico", "image/x-icon"},
      {".woff", "font/woff"},
      {".woff2", "font/woff2"},
      {".ttf", "font/ttf"},
      {".wasm", "application/wasm"},
      {".txt", "text/plain; charset=utf-8"},
      {".md", "text/plain; charset=utf-8"},
  };
  size_t dot = path.rfind('.');
  if (dot != std::string::npos) {
    std::string ext = path.substr(dot);
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    for (const auto &kv : kTypes) {
      if (ext == kv.first) {
        return kv.second;
      }
    }
  }
  return "text/plain; charset=utf-8";
}

std::vector<std::string> VizServer::AssetSearchDirs(
    const std::string &mod_name) {
  std::vector<std::string> dirs;

  // 1. Explicit override: CLIO_VIZ_PATH is a list of viz roots, each holding
  //    one subdirectory per ChiMod. Wins so a developer can point the daemon at
  //    a working copy of the assets without rebuilding.
  if (const char *env = clio::run::env::GetCompat("VIZ_PATH")) {
    for (const std::string &root : SplitSearchPath(env)) {
      dirs.push_back((std::filesystem::path(root) / mod_name).string());
    }
  }

  // 2/3. Relative to the directory the ChiMod's own library was loaded from,
  //      which is what makes build-tree and install-tree layouts both resolve
  //      with no configuration. Fall back to the runtime library's directory
  //      when the module manager has no record (e.g. a module statically
  //      linked into the daemon, or a caller asking before the scan).
  std::vector<std::string> lib_dirs;
  if (auto *module_manager = CLIO_MODULE_MANAGER) {
    std::string lib_path = module_manager->GetChiModLibPath(mod_name);
    if (!lib_path.empty()) {
      lib_dirs.push_back(
          std::filesystem::path(lib_path).parent_path().string());
    }
    std::string self_dir = module_manager->GetModuleDirectory();
    if (!self_dir.empty()) {
      lib_dirs.push_back(self_dir);
    }
  }
  for (const std::string &lib_dir : lib_dirs) {
    std::filesystem::path dir(lib_dir);
    // Built, not installed: libs in <build>/bin, assets in <build>/bin/viz.
    dirs.push_back((dir / "viz" / mod_name).string());
    // Installed: libs in <prefix>/lib, assets in <prefix>/share/clio/viz.
    dirs.push_back(
        (dir.parent_path() / "share" / "clio" / "viz" / mod_name).string());
  }
  return dirs;
}

std::string VizServer::FindAssetDir(const std::string &mod_name) {
  for (const std::string &dir : AssetSearchDirs(mod_name)) {
    std::error_code ec;
    if (std::filesystem::is_directory(dir, ec)) {
      return dir;
    }
  }
  return std::string();
}

//===========================================================================
// Registry
//===========================================================================

bool VizServer::AddRoute(Route route) {
  if (route.path.empty() || !route.handler) {
    HLOG(kError, "Viz: refusing to register an empty or handler-less route ({})",
         route.path);
    return false;
  }
  route.method = ToUpper(route.method);
  std::lock_guard<std::mutex> guard(mutex_);
  for (const Route &existing : routes_) {
    if (existing.method == route.method && existing.path == route.path) {
      return false;
    }
  }
  HLOG(kDebug, "Viz: route {} {} registered by {}", route.method, route.path,
       route.mod_name.empty() ? "?" : route.mod_name);
  routes_.push_back(std::move(route));
  return true;
}

bool VizServer::MountAssets(const std::string &mod_name,
                            const std::string &url_prefix) {
  // Cheap exit before touching the filesystem: this runs once per container, so
  // a pool-heavy workload would otherwise re-stat the candidate directories for
  // a ChiMod whose assets are already mounted.
  const std::string prefix =
      url_prefix.empty() ? ("/viz/" + mod_name) : url_prefix;
  {
    std::lock_guard<std::mutex> guard(mutex_);
    for (const MountInfo &mount : mounts_) {
      if (mount.url_prefix == prefix) {
        return false;
      }
    }
  }
  std::string dir = FindAssetDir(mod_name);
  if (dir.empty()) {
    HLOG(kDebug, "Viz: no viz/ assets found for ChiMod {}", mod_name);
    return false;
  }
  return MountDir(mod_name, dir, url_prefix);
}

bool VizServer::MountDir(const std::string &mod_name, const std::string &dir,
                         const std::string &url_prefix) {
  std::error_code ec;
  if (dir.empty() || !std::filesystem::is_directory(dir, ec)) {
    return false;
  }
  std::string prefix = url_prefix.empty() ? ("/viz/" + mod_name) : url_prefix;
  if (prefix.size() > 1 && prefix.back() == '/') {
    prefix.pop_back();
  }
  std::lock_guard<std::mutex> guard(mutex_);
  for (const MountInfo &mount : mounts_) {
    if (mount.url_prefix == prefix) {
      return false;
    }
  }
  HLOG(kInfo, "Viz: mounted {} assets at {} (from {})", mod_name, prefix, dir);
  mounts_.push_back(MountInfo{prefix, dir, mod_name});
  return true;
}

size_t VizServer::RemoveModule(const std::string &mod_name) {
  if (mod_name.empty()) {
    return 0;
  }
  std::lock_guard<std::mutex> guard(mutex_);
  size_t removed = 0;
  for (auto it = routes_.begin(); it != routes_.end();) {
    if (it->mod_name == mod_name) {
      it = routes_.erase(it);
      ++removed;
    } else {
      ++it;
    }
  }
  for (auto it = mounts_.begin(); it != mounts_.end();) {
    if (it->mod_name == mod_name) {
      // A home page pointing into the mount we are dropping would 404 from now
      // on; better to have no home than a broken one.
      if (!home_.empty() && home_.compare(0, it->url_prefix.size(),
                                          it->url_prefix) == 0) {
        home_.clear();
      }
      it = mounts_.erase(it);
    } else {
      ++it;
    }
  }
  if (removed > 0) {
    HLOG(kDebug, "Viz: removed {} route(s) registered by {}", removed,
         mod_name);
  }
  return removed;
}

void VizServer::SetHome(const std::string &url) {
  std::lock_guard<std::mutex> guard(mutex_);
  home_ = url;
}

std::string VizServer::GetHome() const {
  std::lock_guard<std::mutex> guard(mutex_);
  return home_;
}

std::vector<RouteInfo> VizServer::GetRoutes() const {
  std::lock_guard<std::mutex> guard(mutex_);
  std::vector<RouteInfo> out;
  out.reserve(routes_.size());
  for (const Route &route : routes_) {
    out.push_back(RouteInfo{route.method, route.path, route.mod_name,
                            route.doc});
  }
  return out;
}

std::vector<MountInfo> VizServer::GetMounts() const {
  std::lock_guard<std::mutex> guard(mutex_);
  return mounts_;
}

void VizServer::OnContainerRegistered(const std::string &mod_name,
                                      Container &container) {
  if (mod_name.empty()) {
    return;
  }
  // Assets first: a module's RegisterViz() may want to claim the home page,
  // which only makes sense once its files are reachable.
  MountAssets(mod_name);
  container.RegisterViz(*this, mod_name);
}

//===========================================================================
// Dispatch
//===========================================================================

bool VizServer::ResolveAsset(const Request &req, std::string *file_path,
                             std::string *mime) const {
  std::vector<MountInfo> mounts;
  {
    std::lock_guard<std::mutex> guard(mutex_);
    mounts = mounts_;
  }
  for (const MountInfo &mount : mounts) {
    if (req.path.compare(0, mount.url_prefix.size(), mount.url_prefix) != 0) {
      continue;
    }
    std::string rel = req.path.substr(mount.url_prefix.size());
    if (!rel.empty() && rel.front() != '/') {
      continue;  // e.g. /viz/clio_adminX must not match /viz/clio_admin
    }
    while (!rel.empty() && rel.front() == '/') {
      rel.erase(rel.begin());
    }
    if (rel.empty()) {
      rel = "index.html";
    }
    if (!IsSafeRelativePath(rel)) {
      return false;
    }
    std::filesystem::path candidate = std::filesystem::path(mount.dir) / rel;
    std::error_code ec;
    if (std::filesystem::is_directory(candidate, ec)) {
      candidate /= "index.html";
    }
    if (!std::filesystem::is_regular_file(candidate, ec)) {
      continue;
    }
    *file_path = candidate.string();
    *mime = MimeTypeOf(candidate.string());
    return true;
  }
  return false;
}

bool VizServer::Dispatch(const Request &req, Response &resp) {
  const std::string method = ToUpper(req.method);

  // 1. Registered routes (exact segments and {var} patterns).
  Handler handler;
  Request bound = req;
  {
    std::lock_guard<std::mutex> guard(mutex_);
    for (const Route &route : routes_) {
      std::map<std::string, std::string> vars;
      if (!MatchPath(route.path, req.path, &vars)) {
        continue;
      }
      if (route.method != method &&
          !(route.method == "GET" && method == "HEAD")) {
        continue;
      }
      handler = route.handler;
      bound.path_vars = std::move(vars);
      break;
    }
  }
  if (handler) {
    // Invoked outside the registry lock: handlers block (they submit tasks and
    // wait), and holding the lock would serialize every request behind the
    // slowest one and deadlock a handler that registers a route.
    try {
      handler(bound, resp);
    } catch (const std::exception &e) {
      HLOG(kError, "Viz: handler for {} {} threw: {}", method, req.path,
           e.what());
      resp.Error(500, std::string("handler failed: ") + e.what());
    } catch (...) {
      HLOG(kError, "Viz: handler for {} {} threw a non-standard exception",
           method, req.path);
      resp.Error(500, "handler failed");
    }
    return true;
  }

  // 2. Static assets from a mounted viz/ directory. Reads only: a write verb
  //    aimed at a page is a client bug, not a request to serve it.
  std::string file_path;
  std::string mime;
  if ((method == "GET" || method == "HEAD") &&
      ResolveAsset(req, &file_path, &mime)) {
    std::string contents;
    if (!ReadWholeFile(file_path, &contents)) {
      resp.Error(500, "failed to read " + file_path);
      return true;
    }
    resp.status = 200;
    resp.content_type = mime;
    resp.body = std::move(contents);
    // The dashboard is served from a live daemon; a cached page would show a
    // stale UI against fresh data.
    resp.headers.push_back({"Cache-Control", "no-cache"});
    return true;
  }

  // 3. The home page.
  const std::string home = GetHome();
  if (!home.empty() && (req.path.empty() || req.path == "/")) {
    resp.status = 302;
    resp.content_type = "text/plain; charset=utf-8";
    resp.body = "";
    resp.headers.push_back({"Location", home});
    return true;
  }

  return false;
}

//===========================================================================
// HTTP transport (Poco)
//===========================================================================

#if CLIO_RUN_HAS_POCO

/** Poco server state. Lives here so no Poco type reaches an installed header. */
struct VizServer::Impl {
  Poco::Net::ServerSocket socket;
  // A PRIVATE thread pool, deliberately NOT Poco's ThreadPool::defaultPool().
  //
  // The HTTPServer(factory, socket, params) overload silently borrows the
  // process-wide default pool, and tearing the server down then joins THAT
  // pool -- i.e. it waits on threads this dashboard never owned and cannot
  // finish. stopAll() then never returns, ServerFinalize() stalls behind it,
  // and the RequestStop watchdog force-exits the daemon with code 2. That is
  // the 2026-08-19 regression in cr_shutdown_bt_churn (issue #990's dashboard).
  //
  // Declared BEFORE `server` so it is destroyed after it (members die in
  // reverse declaration order): the server must not outlive its pool.
  std::unique_ptr<Poco::ThreadPool> pool;
  std::unique_ptr<Poco::Net::HTTPServer> server;
  std::string bind_addr;
  u32 port = 0;
};

namespace {

/** Bridges one Poco request onto VizServer::Dispatch. */
class PocoVizHandler : public Poco::Net::HTTPRequestHandler {
 public:
  explicit PocoVizHandler(VizServer *viz) : viz_(viz) {}

  void handleRequest(Poco::Net::HTTPServerRequest &preq,
                     Poco::Net::HTTPServerResponse &presp) override {
    Request req;
    Response resp;
    try {
      req.method = preq.getMethod();
      Poco::URI uri(preq.getURI());
      req.path = uri.getPath();
      for (const auto &kv : uri.getQueryParameters()) {
        req.params[kv.first] = kv.second;
      }
      // Read the body ONLY when the request declares one (Content-Length or
      // chunked). A bare POST -- `curl -X POST` sends neither -- otherwise
      // hands stream() an unbounded body, and copyToString blocks for the
      // server's full 60s receive timeout waiting for bytes that are never
      // coming: the action succeeds instantly server-side while the client
      // stares at a hung connection.
      const bool has_body =
          preq.getChunkedTransferEncoding() ||
          (preq.getContentLength64() !=
               Poco::Net::HTTPMessage::UNKNOWN_CONTENT_LENGTH &&
           preq.getContentLength64() > 0);
      if (req.method != "GET" && req.method != "HEAD" && has_body) {
        Poco::StreamCopier::copyToString(preq.stream(), req.body,
                                         kMaxBodyBytes);
        // A form submission's fields become params, exactly as if they had been
        // query parameters — with the query string winning on a key collision
        // (ParseFormBody never overwrites). Handlers then read one map either
        // way, and pages can use plain <form> posts or fetch+URLSearchParams.
        const std::string &ctype = preq.getContentType();
        if (ctype.compare(0, 33, "application/x-www-form-urlencoded") == 0) {
          VizServer::ParseFormBody(req.body, &req.params);
        }
      }
      if (!viz_->Dispatch(req, resp)) {
        resp.Error(404, "no route or asset for " + req.path);
      }
    } catch (const Poco::Exception &e) {
      resp.Error(400, std::string("bad request: ") + e.displayText());
    } catch (const std::exception &e) {
      resp.Error(500, std::string("internal error: ") + e.what());
    }

    presp.setStatus(
        static_cast<Poco::Net::HTTPResponse::HTTPStatus>(resp.status));
    presp.setContentType(resp.content_type);
    for (const auto &kv : resp.headers) {
      presp.set(kv.first, kv.second);
    }
    // Deliberately NO Access-Control-Allow-Origin. Every page that calls these
    // endpoints is served by this same daemon, so CORS buys nothing -- while a
    // permissive header on a loopback port would let any website the user
    // happens to visit read this node's runtime internals. To iterate on assets
    // without rebuilding, point $CLIO_VIZ_PATH at a working copy; the daemon
    // then serves them from this same origin.
    presp.setContentLength(static_cast<std::streamsize>(resp.body.size()));
    std::ostream &out = presp.send();
    if (req.method != "HEAD") {
      out.write(resp.body.data(),
                static_cast<std::streamsize>(resp.body.size()));
    }
    out.flush();
  }

 private:
  VizServer *viz_;
};

/** Hands every request to a PocoVizHandler. */
class PocoVizHandlerFactory : public Poco::Net::HTTPRequestHandlerFactory {
 public:
  explicit PocoVizHandlerFactory(VizServer *viz) : viz_(viz) {}

  Poco::Net::HTTPRequestHandler *createRequestHandler(
      const Poco::Net::HTTPServerRequest &) override {
    return new PocoVizHandler(viz_);
  }

 private:
  VizServer *viz_;
};

}  // namespace

bool VizServer::StartOn(const std::string &bind_addr, u32 port,
                        u32 max_threads) {
  if (impl_) {
    HLOG(kWarning, "Viz: server already running on {}:{}", impl_->bind_addr,
         impl_->port);
    return true;
  }
  auto impl = std::make_unique<Impl>();
  try {
    // No-op on POSIX; on Windows this is the WSAStartup Poco needs. Poco's
    // per-TU automatic initializer covers this too, but the refcount is cheap
    // and the failure mode without it (every socket call failing with
    // WSANOTINITIALISED) is obscure.
    Poco::Net::initializeNetwork();
    impl->socket = Poco::Net::ServerSocket(
        Poco::Net::SocketAddress(bind_addr, static_cast<Poco::UInt16>(port)));
    auto *params = new Poco::Net::HTTPServerParams();
    params->setMaxThreads(static_cast<int>(std::max<u32>(max_threads, 1)));
    params->setMaxQueued(64);
    params->setKeepAlive(true);
    // Poco serves thread-per-CONNECTION, and a kept-alive connection PARKS its
    // thread between requests -- for the default keepAliveTimeout (15s) if the
    // client sends nothing more. A browser holds ~6 keep-alive sockets, so a
    // small pool starves in 15s waves the moment a dashboard tab is open
    // (measured: with 4 threads, 10 concurrent GETs answer 4x11ms / 4x15s /
    // 2x30s). Park briefly instead, and cap requests per connection so a
    // misbehaving client cannot own a thread forever.
    params->setKeepAliveTimeout(Poco::Timespan(2, 0));
    params->setMaxKeepAliveRequests(100);
    // Own the thread pool (see Impl) so teardown only ever joins our threads.
    const int pool_max = static_cast<int>(std::max<u32>(max_threads, 2));
    impl->pool = std::make_unique<Poco::ThreadPool>(/*minCapacity=*/1, pool_max);
    impl->server = std::make_unique<Poco::Net::HTTPServer>(
        new PocoVizHandlerFactory(this), *impl->pool, impl->socket, params);
    impl->server->start();
    impl->bind_addr = bind_addr;
    impl->port = impl->socket.address().port();
  } catch (const Poco::Exception &e) {
    // Never fatal: a taken port (a second runtime on this node, or an unrelated
    // service) must not stop the daemon from serving data.
    HLOG(kWarning, "Viz: could not listen on {}:{} ({}); dashboard disabled",
         bind_addr, port, e.displayText());
    return false;
  }
  impl_ = std::move(impl);
  // kSuccess (green) rather than kInfo: this line is the address an operator
  // has to copy into a browser, so it should stand out in a startup log that is
  // otherwise a wall of white.
  HLOG(kSuccess, "Viz: dashboard listening at http://{}:{}", impl_->bind_addr,
       impl_->port);
  return true;
}

void VizServer::Stop() {
  if (!impl_) {
    return;
  }
  HLOG(kInfo, "Viz: stopping dashboard on {}:{}", impl_->bind_addr,
       impl_->port);
  // Close the LISTENING SOCKET FIRST, before asking Poco to stop. Poco is
  // thread-per-connection: TCPServer::start() spawns an acceptor parked in
  // ServerSocket::poll() on this socket, and the stop path joins it. Closing
  // first makes that poll return so the acceptor can observe the stop flag.
  //
  // (The old comment here also had Poco's flag backwards: per
  // HTTPServer::stopAll's contract abortCurrent=TRUE shuts the client sockets
  // down, aborting requests, and FALSE lets current requests finish. Waiting
  // is the risky one, so keep true.)
  try {
    impl_->socket.close();
  } catch (const Poco::Exception &e) {
    HLOG(kWarning, "Viz: error closing the dashboard socket: {}",
         e.displayText());
  }

  // Belt and braces. With the private pool (see Impl) stopAll() returns
  // promptly, so this budget is never actually spent -- measured: teardown is
  // instant and cr_shutdown_bt_churn is back to its pre-dashboard 58 s, versus
  // 130 s when every cycle burned the timeout below. It stays as a backstop
  // because a request handler is ChiMod code that can block on a task, and the
  // dashboard is a debugging aid: it must never be the reason a daemon cannot
  // exit. On timeout, abandon the Poco objects (the shared_ptr keeps them
  // alive for the worker, and the process is seconds from exit anyway) and let
  // the rest of ServerFinalize proceed. Losing the dashboard's memory at exit
  // is strictly better than losing the graceful shutdown.
  // Overridable with CLIO_VIZ_STOP_BUDGET_MS, the same env convention as
  // CLIO_VIZ_ENABLE / _PORT / _BIND. An operator can shorten it on a node
  // where a wedged handler must never delay shutdown, and the tests set 0 to
  // exercise the abandon path deterministically instead of leaving it as a
  // branch nothing ever takes.
  int stop_budget_ms = 3000;
  if (const char *env = clio::run::env::GetCompat("VIZ_STOP_BUDGET_MS")) {
    char *end = nullptr;
    long n = std::strtol(env, &end, 10);
    if (end != env && n >= 0) {
      stop_budget_ms = static_cast<int>(n);
    }
  }
  std::shared_ptr<Impl> impl(std::move(impl_));  // impl_ is null from here on
  auto finished = std::make_shared<std::promise<void>>();
  std::future<void> done = finished->get_future();
  std::thread([impl, finished]() mutable {
    try {
      if (impl->server) {
        impl->server->stopAll(true);
        impl->server.reset();
      }
    } catch (const Poco::Exception &e) {
      HLOG(kWarning, "Viz: error while stopping the dashboard: {}",
           e.displayText());
    }
    finished->set_value();
  }).detach();

  if (done.wait_for(std::chrono::milliseconds(stop_budget_ms)) !=
      std::future_status::ready) {
    HLOG(kWarning,
         "Viz: dashboard teardown did not finish within {} ms; abandoning it "
         "so the shutdown can proceed",
         stop_budget_ms);
  }
}

bool VizServer::IsRunning() const { return impl_ != nullptr; }

u32 VizServer::GetPort() const { return impl_ ? impl_->port : 0; }

std::string VizServer::GetBindAddr() const {
  return impl_ ? impl_->bind_addr : std::string();
}

#else  // !CLIO_RUN_HAS_POCO

struct VizServer::Impl {};

bool VizServer::StartOn(const std::string &bind_addr, u32 port,
                        u32 max_threads) {
  (void)bind_addr;
  (void)port;
  (void)max_threads;
  HLOG(kWarning,
       "Viz: this build has no HTTP transport (Poco was not found at configure "
       "time); dashboard disabled");
  return false;
}

void VizServer::Stop() {}
bool VizServer::IsRunning() const { return false; }
u32 VizServer::GetPort() const { return 0; }
std::string VizServer::GetBindAddr() const { return std::string(); }

#endif  // CLIO_RUN_HAS_POCO

VizServer::VizServer() = default;

VizServer::~VizServer() { Stop(); }

bool VizServer::Start() {
  auto *config = CLIO_CONFIG_MANAGER;
  if (!config) {
    return false;
  }
  if (!config->GetVizEnabled()) {
    HLOG(kDebug, "Viz: dashboard disabled by configuration");
    return false;
  }
  return StartOn(config->GetVizBindAddr(), config->GetVizPort(),
                 config->GetVizMaxThreads());
}

}  // namespace clio::run::viz
