/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 */

#ifndef CLIO_CAE_CORE_FACTORY_AWS_CREDS_H_
#define CLIO_CAE_CORE_FACTORY_AWS_CREDS_H_

// AWS credential + region resolution for the in-process S3 assimilator, as pure
// functions with no Poco, no network and no globals -- strings in, struct out,
// so the precedence rules are provable in an ordinary offline unit test.
//
// Why this exists: the S3 REST signer (s3_rest.h) reads credentials from the
// environment ONLY. The read benchmark and pipelines forward the *names*
// AWS_PROFILE / AWS_DEFAULT_REGION (never raw secrets) through AssimilationCtx,
// and the ssh-launched runtime daemon inherits no AWS_* at all. The old
// fork+exec path let the AWS SDK child walk the credential-provider chain and
// read ~/.aws/credentials itself; an env-only signer cannot. This resolver
// restores that: it prefers real environment keys (so a forward_env deployment
// keeps working and the bdev's contract is unchanged), and otherwise reads the
// [profile] section of ~/.aws/credentials. Secrets never travel in a task
// payload -- only the profile NAME does.

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>

namespace clio::cae::core {

/** Resolved credentials + region handed to S3RestClient's S3Config. */
struct AwsCredentials {
  std::string access_key;
  std::string secret_key;
  std::string session_token;  ///< empty unless STS/temporary credentials
  std::string region;
};

/** Outcome of resolution: ok, or an error naming the profile and file tried. */
struct AwsCredResult {
  bool ok = false;
  std::string error;
  AwsCredentials creds;
};

/** The environment inputs resolution consults; injected so a test can pin them. */
struct AwsCredEnv {
  std::string access_key;        ///< AWS_ACCESS_KEY_ID
  std::string secret_key;        ///< AWS_SECRET_ACCESS_KEY
  std::string session_token;     ///< AWS_SESSION_TOKEN
  std::string profile;           ///< AWS_PROFILE
  std::string region;            ///< AWS_DEFAULT_REGION
  std::string region_alt;        ///< AWS_REGION (newer SDK spelling)
  std::string shared_creds_file; ///< AWS_SHARED_CREDENTIALS_FILE
  std::string config_file;       ///< AWS_CONFIG_FILE
};

namespace detail {

/** Trim ASCII whitespace from both ends. */
inline std::string Trim(const std::string &s) {
  size_t b = 0, e = s.size();
  while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
  return s.substr(b, e - b);
}

/**
 * Parse an AWS-style INI file: [section] headers and `key = value` lines, with
 * `#`/`;` comments. Only top-level (non-indented) keys are captured -- nested
 * sub-settings (e.g. an `s3 =` block) are ignored, which is all this resolver
 * needs. Returns section -> (key -> value). Duplicate keys: last wins.
 */
using IniSections = std::map<std::string, std::map<std::string, std::string>>;
inline IniSections ParseIni(const std::string &text) {
  IniSections out;
  std::istringstream in(text);
  std::string line;
  std::string section;
  while (std::getline(in, line)) {
    // Strip inline comments only when they start the trimmed line; values may
    // legitimately contain '#'/';' so we do not split mid-value.
    std::string t = Trim(line);
    if (t.empty() || t[0] == '#' || t[0] == ';') continue;
    if (t.front() == '[' && t.back() == ']') {
      section = Trim(t.substr(1, t.size() - 2));
      out[section];  // ensure the section exists even if empty
      continue;
    }
    // A key line must be non-indented in the original to count as top-level.
    if (!line.empty() && std::isspace(static_cast<unsigned char>(line[0]))) {
      continue;  // indented => a sub-setting of a nested block; skip
    }
    size_t eq = t.find('=');
    if (eq == std::string::npos || section.empty()) continue;
    std::string key = Trim(t.substr(0, eq));
    std::string val = Trim(t.substr(eq + 1));
    if (!key.empty()) out[section][key] = val;
  }
  return out;
}

/** Look up a key in a section, or "" if section/key absent. */
inline std::string IniGet(const IniSections &ini, const std::string &section,
                          const std::string &key) {
  auto s = ini.find(section);
  if (s == ini.end()) return "";
  auto k = s->second.find(key);
  return k == s->second.end() ? "" : k->second;
}

}  // namespace detail

/**
 * Resolve credentials + region from injected inputs (the testable core).
 *
 * Precedence -- credentials:
 *   1. env access+secret (+token) if BOTH key and secret are present;
 *   2. else the `[profile]` section of the credentials file, where profile is
 *      ctx_profile, else env AWS_PROFILE, else "default".
 * Precedence -- region:
 *   ctx_region, else AWS_DEFAULT_REGION, else AWS_REGION, else the profile's
 *   `region` in the config file (section "default" or "profile <name>").
 * A missing profile, missing keys, or unresolved region each fail loudly,
 * naming the profile and (for credentials) the file searched. No silent
 * us-east-1 default: a wrong region is a 301 the signer will not follow.
 *
 * @param ctx_profile AssimilationCtx.s3_profile (may be empty).
 * @param ctx_region  AssimilationCtx.s3_region (may be empty).
 * @param env         Environment inputs.
 * @param credentials_text Contents of ~/.aws/credentials (may be empty).
 * @param config_text      Contents of ~/.aws/config (may be empty).
 * @return Resolved credentials, or ok=false with a diagnostic error.
 */
inline AwsCredResult ResolveAwsCredentialsFrom(const std::string &ctx_profile,
                                               const std::string &ctx_region,
                                               const AwsCredEnv &env,
                                               const std::string &credentials_text,
                                               const std::string &config_text) {
  AwsCredResult r;
  const std::string profile =
      !ctx_profile.empty() ? ctx_profile
                           : (!env.profile.empty() ? env.profile : "default");

  // --- credentials ---
  if (!env.access_key.empty() && !env.secret_key.empty()) {
    r.creds.access_key = env.access_key;
    r.creds.secret_key = env.secret_key;
    r.creds.session_token = env.session_token;
  } else {
    detail::IniSections creds = detail::ParseIni(credentials_text);
    if (creds.find(profile) == creds.end()) {
      r.error = "AWS credentials: no profile [" + profile +
                "] in the credentials file (set AWS_ACCESS_KEY_ID/"
                "AWS_SECRET_ACCESS_KEY, or AWS_SHARED_CREDENTIALS_FILE / "
                "s3_profile correctly)";
      return r;
    }
    r.creds.access_key = detail::IniGet(creds, profile, "aws_access_key_id");
    r.creds.secret_key = detail::IniGet(creds, profile, "aws_secret_access_key");
    r.creds.session_token =
        detail::IniGet(creds, profile, "aws_session_token");
    if (r.creds.access_key.empty() || r.creds.secret_key.empty()) {
      r.error = "AWS credentials: profile [" + profile +
                "] is missing aws_access_key_id / aws_secret_access_key";
      return r;
    }
  }

  // --- region ---
  std::string region = !ctx_region.empty() ? ctx_region
                       : !env.region.empty() ? env.region
                       : !env.region_alt.empty() ? env.region_alt
                                                 : std::string();
  if (region.empty() && !config_text.empty()) {
    detail::IniSections cfg = detail::ParseIni(config_text);
    // ~/.aws/config names non-default profiles "[profile foo]"; default stays
    // "[default]".
    region = detail::IniGet(cfg, profile, "region");
    if (region.empty() && profile != "default") {
      region = detail::IniGet(cfg, "profile " + profile, "region");
    }
  }
  if (region.empty()) {
    r.error = "AWS region unresolved for profile [" + profile +
              "]: set s3_region, AWS_DEFAULT_REGION, AWS_REGION, or a region "
              "in the config file. A wrong/absent region yields HTTP 301.";
    return r;
  }
  r.creds.region = region;
  r.ok = true;
  return r;
}

namespace detail {

/** Read a whole file into a string; "" if it cannot be opened. */
inline std::string SlurpFile(const std::string &path) {
  if (path.empty()) return "";
  std::ifstream f(path, std::ios::binary);
  if (!f) return "";
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

/** getenv as std::string ("" when unset/empty). */
inline std::string Env(const char *name) {
  const char *v = std::getenv(name);
  return (v && *v) ? std::string(v) : std::string();
}

/** Default path under $HOME, or "" if HOME is unset. */
inline std::string HomeAws(const char *leaf) {
  std::string home = Env("HOME");
  if (home.empty()) return "";
  return home + "/.aws/" + leaf;
}

}  // namespace detail

/**
 * Resolve credentials from the live process environment and ~/.aws files.
 * Thin wrapper over ResolveAwsCredentialsFrom that gathers env vars and reads
 * the credentials/config files (honouring AWS_SHARED_CREDENTIALS_FILE /
 * AWS_CONFIG_FILE overrides). See that function for precedence.
 *
 * @param ctx_profile AssimilationCtx.s3_profile.
 * @param ctx_region  AssimilationCtx.s3_region.
 */
inline AwsCredResult ResolveAwsCredentials(const std::string &ctx_profile,
                                           const std::string &ctx_region) {
  AwsCredEnv env;
  env.access_key = detail::Env("AWS_ACCESS_KEY_ID");
  env.secret_key = detail::Env("AWS_SECRET_ACCESS_KEY");
  env.session_token = detail::Env("AWS_SESSION_TOKEN");
  env.profile = detail::Env("AWS_PROFILE");
  env.region = detail::Env("AWS_DEFAULT_REGION");
  env.region_alt = detail::Env("AWS_REGION");
  env.shared_creds_file = detail::Env("AWS_SHARED_CREDENTIALS_FILE");
  env.config_file = detail::Env("AWS_CONFIG_FILE");

  const std::string creds_path = !env.shared_creds_file.empty()
                                     ? env.shared_creds_file
                                     : detail::HomeAws("credentials");
  const std::string config_path =
      !env.config_file.empty() ? env.config_file : detail::HomeAws("config");

  return ResolveAwsCredentialsFrom(ctx_profile, ctx_region, env,
                                   detail::SlurpFile(creds_path),
                                   detail::SlurpFile(config_path));
}

}  // namespace clio::cae::core

#endif  // CLIO_CAE_CORE_FACTORY_AWS_CREDS_H_
