/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 */

#ifndef CLIO_CAE_CORE_FACTORY_HTTP_AUTH_H_
#define CLIO_CAE_CORE_FACTORY_HTTP_AUTH_H_

// Credentials and per-site request parameters for http:// and https://
// sources, read from a configuration file instead of the OMNI file.
//
// Why a file: the assimilator runs inside the runtime daemon, which a launcher
// (jarvis, ssh, a PBS script) may start without the client's environment, and
// an OMNI file is meant to be shared. Secrets belong in neither. The file is
//
//   $CAE_HTTP_AUTH_FILE, else ~/.clio/cae_http_auth.ini
//
// in the INI syntax of ~/.aws/credentials (aws_creds.h parses it). Each
// section names the URLs it applies to, either by host or by URL prefix:
//
//   [co2datashare.org]                    # every URL on this host
//   query.country     = United States     # appended as ?country=United%20States
//   query.affiliation = The HDF Group
//
//   [https://data.example.org/private/]   # URLs starting with this prefix
//   bearer_token = ${EXAMPLE_TOKEN}       # Authorization: Bearer <token>
//   header.X-Api-Key = abc123             # any request header
//   user = alice                          # HTTP basic auth
//   password = ${EXAMPLE_PASSWORD}
//
// Every matching section applies, host sections first and then prefixes from
// shortest to longest, so a more specific section overrides a key of a broader
// one. ${VAR} in a value is replaced by that environment variable; an unset
// variable is an error rather than an empty credential. An OMNI src_token,
// when present, is sent as "Authorization: Bearer <src_token>" and replaces
// any Authorization the file gives.
//
// A file that holds a secret (bearer_token, password, or any header.*) must
// not be readable by group or others (chmod 600), as ssh requires of its keys.
//
// The resolution itself (ResolveHttpAuthFrom) is a pure function, strings in
// and struct out, so its rules are covered by an offline unit test.

#include <clio_cae/core/factory/aws_creds.h>  // detail::ParseIni, Trim, ...

#include <sys/stat.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace clio::cae::core {

/** What a request to one URL needs, after the config file has been applied. */
struct HttpAuth {
  std::string url;                   ///< source URL plus the query.* params
  std::vector<std::string> headers;  ///< "Name: value", in a stable order
  std::string userpwd;               ///< "user:password" for basic auth, or ""
  std::vector<std::string> sections; ///< config sections that matched
  size_t num_query = 0;              ///< query.* params appended to url
};

/** Outcome of resolution: ok, or an error naming the section and key. */
struct HttpAuthResult {
  bool ok = false;
  std::string error;
  HttpAuth auth;
};

/** Environment lookup ("" when unset), injected so a test can pin it. */
using HttpEnvLookup = std::function<std::string(const std::string &)>;

namespace http_detail {

inline std::string Lower(std::string s) {
  for (char &c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

/** Scheme and host[:port] of a URL ("" when it has no "://"). */
inline void SplitUrl(const std::string &url, std::string *scheme,
                     std::string *host) {
  scheme->clear();
  host->clear();
  size_t p = url.find("://");
  if (p == std::string::npos) return;
  *scheme = Lower(url.substr(0, p));
  size_t b = p + 3;
  size_t e = url.find_first_of("/?#", b);
  std::string h = url.substr(b, e == std::string::npos ? std::string::npos : e - b);
  size_t at = h.rfind('@');  // drop any user:pass@
  if (at != std::string::npos) h = h.substr(at + 1);
  *host = Lower(h);
}

/** Percent-encode everything but RFC 3986 unreserved characters. */
inline std::string UrlEncode(const std::string &s) {
  static const char *kHex = "0123456789ABCDEF";
  std::string out;
  for (unsigned char c : s) {
    if (std::isalnum(c) || c == '-' || c == '.' || c == '_' || c == '~') {
      out += static_cast<char>(c);
    } else {
      out += '%';
      out += kHex[c >> 4];
      out += kHex[c & 15];
    }
  }
  return out;
}

/** True if the URL's query string already has parameter `name`. */
inline bool HasQueryParam(const std::string &url, const std::string &name) {
  size_t q = url.find('?');
  if (q == std::string::npos) return false;
  size_t h = url.find('#', q);
  std::string qs = url.substr(q + 1, h == std::string::npos ? std::string::npos : h - q - 1);
  const std::string enc = UrlEncode(name);
  size_t pos = 0;
  while (pos <= qs.size()) {
    size_t amp = qs.find('&', pos);
    std::string kv = qs.substr(pos, amp == std::string::npos ? std::string::npos : amp - pos);
    std::string k = kv.substr(0, kv.find('='));
    if (k == name || k == enc) return true;
    if (amp == std::string::npos) break;
    pos = amp + 1;
  }
  return false;
}

/**
 * Does config section `sec` apply to the URL? Returns its rank: 0 = no,
 * 1 = host section, 2 + prefix length = URL-prefix section.
 */
inline size_t SectionRank(const std::string &sec, const std::string &url,
                          const std::string &host) {
  if (sec.find("://") != std::string::npos) {
    // Scheme and host compare case-insensitively, the path exactly.
    std::string ss, sh;
    SplitUrl(sec, &ss, &sh);
    std::string us, uh;
    SplitUrl(url, &us, &uh);
    if (ss != us || sh != uh) return 0;
    size_t sp = sec.find('/', sec.find("://") + 3);
    size_t up = url.find('/', url.find("://") + 3);
    std::string spath = sp == std::string::npos ? "" : sec.substr(sp);
    std::string upath = up == std::string::npos ? "" : url.substr(up);
    if (upath.compare(0, spath.size(), spath) != 0) return 0;
    return 2 + sec.size();
  }
  return Lower(sec) == host ? 1 : 0;
}

/** Replace each ${VAR} in `v`; false (with *missing set) if a VAR is unset. */
inline bool ExpandVars(const std::string &v, const HttpEnvLookup &env,
                       std::string *out, std::string *missing) {
  out->clear();
  size_t i = 0;
  while (i < v.size()) {
    if (v[i] == '$' && i + 1 < v.size() && v[i + 1] == '{') {
      size_t close = v.find('}', i + 2);
      if (close == std::string::npos) {
        out->append(v, i, std::string::npos);
        return true;
      }
      std::string name = v.substr(i + 2, close - i - 2);
      std::string val = env ? env(name) : std::string();
      if (val.empty()) {
        *missing = name;
        return false;
      }
      *out += val;
      i = close + 1;
    } else {
      *out += v[i++];
    }
  }
  return true;
}

/** Keys that carry a secret, which the file-permission check protects. */
inline bool IsSecretKey(const std::string &key) {
  return key == "bearer_token" || key == "password" ||
         key.compare(0, 7, "header.") == 0;
}

}  // namespace http_detail

/**
 * Resolve what a request to `url` needs from the config file's text.
 *
 * @param url         Source URL (http:// or https://).
 * @param src_token   OMNI src_token, already env-expanded (may be empty).
 * @param config_text Contents of the config file (may be empty).
 * @param env         Environment lookup for ${VAR} expansion.
 */
inline HttpAuthResult ResolveHttpAuthFrom(const std::string &url,
                                          const std::string &src_token,
                                          const std::string &config_text,
                                          const HttpEnvLookup &env) {
  HttpAuthResult r;
  std::string scheme, host;
  http_detail::SplitUrl(url, &scheme, &host);
  if ((scheme != "http" && scheme != "https") || host.empty()) {
    r.error = "not an http(s) URL: " + url;
    return r;
  }

  // Matching sections, broadest first.
  detail::IniSections ini = detail::ParseIni(config_text);
  std::vector<std::pair<size_t, std::string>> matches;
  for (const auto &kv : ini) {
    size_t rank = http_detail::SectionRank(kv.first, url, host);
    if (rank > 0) matches.emplace_back(rank, kv.first);
  }
  std::stable_sort(matches.begin(), matches.end());

  // Merge their keys; a later (more specific) section wins a key.
  std::map<std::string, std::pair<std::string, std::string>> keys;  // key -> (value, section)
  std::vector<std::string> order;  // first-seen order, for stable output
  for (const auto &m : matches) {
    r.auth.sections.push_back(m.second);
    for (const auto &kv : ini[m.second]) {
      if (keys.find(kv.first) == keys.end()) order.push_back(kv.first);
      keys[kv.first] = {kv.second, m.second};
    }
  }

  std::string query;
  std::string user, password;
  std::string bearer;
  std::vector<std::pair<std::string, std::string>> headers;  // name -> value
  for (const std::string &key : order) {
    const auto &vs = keys[key];
    std::string val, missing;
    if (!http_detail::ExpandVars(vs.first, env, &val, &missing)) {
      r.error = "[" + vs.second + "] " + key + ": environment variable " +
                missing + " is not set";
      return r;
    }
    if (key.compare(0, 6, "query.") == 0 && key.size() > 6) {
      const std::string name = key.substr(6);
      if (http_detail::HasQueryParam(url, name)) continue;  // the URL's own wins
      query += (query.empty() ? "" : "&") + http_detail::UrlEncode(name) + "=" +
               http_detail::UrlEncode(val);
      ++r.auth.num_query;
    } else if (key.compare(0, 7, "header.") == 0 && key.size() > 7) {
      headers.emplace_back(key.substr(7), val);
    } else if (key == "bearer_token") {
      bearer = val;
    } else if (key == "user") {
      user = val;
    } else if (key == "password") {
      password = val;
    } else {
      r.error = "[" + vs.second + "] unknown key '" + key +
                "' (expected query.<name>, header.<Name>, bearer_token, user, "
                "password)";
      return r;
    }
  }
  if (!password.empty() && user.empty()) {
    r.error = "password given without user for " + host;
    return r;
  }

  // An explicit token (OMNI src_token, then bearer_token) replaces any
  // Authorization header from the file.
  const std::string token = !src_token.empty() ? src_token : bearer;
  if (!token.empty()) {
    headers.erase(std::remove_if(headers.begin(), headers.end(),
                                 [](const auto &h) {
                                   return http_detail::Lower(h.first) ==
                                          "authorization";
                                 }),
                  headers.end());
    headers.emplace_back("Authorization", "Bearer " + token);
  }
  for (const auto &h : headers) r.auth.headers.push_back(h.first + ": " + h.second);
  if (!user.empty()) r.auth.userpwd = user + ":" + password;

  r.auth.url = url;
  if (!query.empty()) {
    size_t hash = r.auth.url.find('#');
    if (hash != std::string::npos) r.auth.url.erase(hash);
    char sep = r.auth.url.find('?') == std::string::npos ? '?' : '&';
    if (r.auth.url.back() == '?' || r.auth.url.back() == '&') sep = '\0';
    if (sep != '\0') r.auth.url += sep;
    r.auth.url += query;
  }
  r.ok = true;
  return r;
}

/** Path of the config file: $CAE_HTTP_AUTH_FILE, else ~/.clio/cae_http_auth.ini. */
inline std::string HttpAuthFilePath() {
  std::string p = detail::Env("CAE_HTTP_AUTH_FILE");
  if (!p.empty()) return p;
  std::string home = detail::Env("HOME");
  return home.empty() ? std::string() : home + "/.clio/cae_http_auth.ini";
}

/**
 * Resolve from the live environment and config file. A missing file is not an
 * error (public URLs need none); a file holding secrets that group or others
 * can read is.
 */
inline HttpAuthResult ResolveHttpAuth(const std::string &url,
                                      const std::string &src_token) {
  const std::string path = HttpAuthFilePath();
  const std::string text = detail::SlurpFile(path);
  if (!text.empty()) {
    struct stat st;
    if (::stat(path.c_str(), &st) == 0 && (st.st_mode & 077) != 0) {
      for (const auto &sec : detail::ParseIni(text)) {
        for (const auto &kv : sec.second) {
          if (http_detail::IsSecretKey(kv.first)) {
            HttpAuthResult r;
            r.error = path + " holds a secret ([" + sec.first + "] " +
                      kv.first + ") but group/others can read it; run chmod 600 " +
                      path;
            return r;
          }
        }
      }
    }
  }
  return ResolveHttpAuthFrom(url, src_token, text,
                             [](const std::string &n) { return detail::Env(n.c_str()); });
}

}  // namespace clio::cae::core

#endif  // CLIO_CAE_CORE_FACTORY_HTTP_AUTH_H_
