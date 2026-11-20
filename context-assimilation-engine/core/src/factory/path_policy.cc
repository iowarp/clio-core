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

#include <clio_cae/core/factory/path_policy.h>

#include <clio_ctp/util/config_parse.h>
#include <clio_ctp/util/msan.h>
#include <clio_runtime/config_manager.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <string_view>
#include <system_error>

#include "clio_ctp/util/logging.h"

namespace clio::cae::core {

namespace fs = std::filesystem;

namespace {

/** Directories that hold credentials; any path through one is refused. */
constexpr std::array<std::string_view, 11> kSecretDirs = {
    ".ssh",   ".aws",  ".gnupg", ".config",         ".clio", ".docker",
    ".kube",  ".azure", ".pki",  ".password-store", ".gcloud"};

/** Files that hold credentials, wherever they are. */
constexpr std::array<std::string_view, 10> kSecretFiles = {
    ".netrc", ".git-credentials", ".pgpass", ".vault-token", ".npmrc",
    ".pypirc", ".boto",           ".s3cfg",  ".my.cnf",      "id_rsa"};

/** Absolute locations refused for reading and writing. */
constexpr std::array<std::string_view, 7> kSystemDeny = {
    "/proc", "/sys", "/dev", "/etc/shadow", "/etc/gshadow", "/etc/sudoers",
    "/etc/ssl/private"};

/** Absolute locations additionally refused for writing. */
constexpr std::array<std::string_view, 9> kSystemWriteDeny = {
    "/etc", "/usr", "/bin", "/sbin", "/lib", "/lib32", "/lib64", "/boot",
    "/root"};

/** Absolute, `..`-free, and resolved through every symlink that exists. */
fs::path Resolve(const std::string &path) {
  std::error_code ec;
  fs::path p = fs::absolute(fs::path(path), ec);
  if (ec) return {};
  fs::path r = fs::weakly_canonical(p, ec);
  if (ec) return {};
  return r;
}

/** Whether `p` is `root` or lies under it, compared component by component. */
bool IsUnder(const fs::path &p, const fs::path &root) {
  auto pi = p.begin();
  for (auto ri = root.begin(); ri != root.end(); ++ri, ++pi) {
    if (ri->empty()) continue;  // trailing separator on the root
    if (pi == p.end() || *pi != *ri) return false;
  }
  return true;
}

/**
 * Whether resolved path `p` lies under system location `loc`, as written or
 * as it resolves. Requests are compared after Resolve(), so a location that
 * is itself a symlink must be resolved the same way: on macOS /etc is a link
 * to /private/etc, and comparing only the literal "/etc" let writes there
 * through. An empty resolution is never treated as a root (IsUnder would
 * match everything).
 * @param p   the resolved request path
 * @param loc a system location from kSystemDeny / kSystemWriteDeny
 * @return true if p is loc or lies under it
 */
bool UnderSystemPath(const fs::path &p, std::string_view loc) {
  const fs::path literal(loc);
  if (IsUnder(p, literal)) return true;
  const fs::path resolved = Resolve(std::string(loc));
  return !resolved.empty() && resolved != literal && IsUnder(p, resolved);
}

/** Why the default policy refuses `p`, or empty if it allows it. */
std::string DenyReason(const fs::path &p, bool write) {
  for (std::string_view s : kSystemDeny) {
    if (UnderSystemPath(p, s)) return std::string("system location ") += s;
  }
  if (write) {
    for (std::string_view s : kSystemWriteDeny) {
      if (UnderSystemPath(p, s)) {
        return std::string("system directory ") += s;
      }
    }
  }
  for (const fs::path &part : p) {
    const std::string name = part.string();
    for (std::string_view d : kSecretDirs) {
      if (name == d) return "credential location " + name;
    }
    for (std::string_view f : kSecretFiles) {
      if (name == f) return "credential file " + name;
    }
    if (write && name.size() > 1 && name[0] == '.') {
      return "hidden file or directory " + name;
    }
  }
  return {};
}

/** `allowed_roots` from every clio_cae_core compose entry, as given. */
std::vector<std::string> ConfiguredRoots() {
  std::vector<std::string> roots;
  auto *config = CLIO_CONFIG_MANAGER;
  if (!config) return roots;
  for (const auto &pool : config->GetComposeConfig().pools_) {
    if (pool.mod_name_ != "clio_cae_core" || pool.config_.empty()) continue;
    try {
      YAML::Node node = YAML::Load(pool.config_);
      ctp::MsanUnpoisonYaml(node);
      YAML::Node list = node["allowed_roots"];
      if (!list) continue;
      if (list.IsScalar()) {
        roots.push_back(list.as<std::string>());
      } else if (list.IsSequence()) {
        for (const auto &item : list) roots.push_back(item.as<std::string>());
      }
    } catch (const std::exception &e) {
      HLOG(kError, "CAE: cannot read allowed_roots for pool {}: {}",
           pool.pool_name_, e.what());
    }
  }
  return roots;
}

}  // namespace

PathPolicy::PathPolicy(const std::vector<std::string> &allowed_roots) {
  for (const std::string &root : allowed_roots) {
    if (root.empty()) continue;
    fs::path r = Resolve(ctp::ConfigParse::ExpandPath(root));
    if (r.empty()) {
      HLOG(kError, "CAE: ignoring allowed root '{}' (cannot resolve it)", root);
      continue;
    }
    roots_.push_back(r.string());
  }
}

const PathPolicy &PathPolicy::Get() {
  static const PathPolicy policy = [] {
    PathPolicy p(ConfiguredRoots());
    if (p.roots_.empty()) {
      HLOG(kInfo,
           "CAE: no allowed_roots configured; local sources and exports may "
           "use any path except credential and system locations");
    } else {
      for (const std::string &r : p.roots_) {
        HLOG(kInfo, "CAE: local sources and exports allowed under {}", r);
      }
    }
    return p;
  }();
  return policy;
}

bool PathPolicy::Allows(const std::string &path, bool write,
                        std::string &reason) const {
  if (path.empty()) {
    reason = "empty path";
    return false;
  }
  fs::path p = Resolve(path);
  if (p.empty()) {
    reason = "cannot resolve the path";
    return false;
  }
  if (!roots_.empty()) {
    bool under = false;
    for (const std::string &root : roots_) {
      if (IsUnder(p, fs::path(root))) {
        under = true;
        break;
      }
    }
    if (!under) {
      reason = "outside allowed_roots";
      return false;
    }
  }
  // Credential and system locations stay refused even inside a root, so a
  // broad root such as the home directory does not re-open ~/.ssh.
  reason = DenyReason(p, write);
  return reason.empty();
}

}  // namespace clio::cae::core
