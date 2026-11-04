/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 */

// Offline tests of the http(s) source's config-file rules (http_auth.h) and
// of the pieces of the download engine that need no network (http_fetch.h):
// the curl config it writes and the probe-header parser.

#include <clio_cae/core/factory/http_auth.h>
#include <clio_cae/core/factory/http_fetch.h>

#include <map>
#include <string>

#include "simple_test.h"

namespace cae = clio::cae::core;

namespace {

const char* kConfig =
    "# comment\n"
    "[co2datashare.org]\n"
    "query.country = United States\n"
    "query.affiliation = The HDF Group\n"
    "\n"
    "[https://data.example.org/private/]\n"
    "bearer_token = ${EX_TOKEN}\n"
    "header.X-Api-Key = k1\n"
    "\n"
    "[data.example.org]\n"
    "header.X-Api-Key = k0\n"
    "user = alice\n"
    "password = ${EX_PW}\n";

cae::HttpEnvLookup Env(std::map<std::string, std::string> m) {
  return [m](const std::string& n) {
    auto it = m.find(n);
    return it == m.end() ? std::string() : it->second;
  };
}

bool Has(const std::vector<std::string>& v, const std::string& s) {
  for (const auto& x : v) {
    if (x == s) return true;
  }
  return false;
}

}  // namespace

TEST_CASE("http_auth_host_section_appends_query", "[http_auth]") {
  auto r = cae::ResolveHttpAuthFrom(
      "https://co2datashare.org/dataset/x/resource/y/download/a.zip", "",
      kConfig, Env({}));
  REQUIRE(r.ok);
  REQUIRE(r.auth.url ==
          "https://co2datashare.org/dataset/x/resource/y/download/a.zip"
          "?affiliation=The%20HDF%20Group&country=United%20States");
  REQUIRE(r.auth.num_query == 2);
  REQUIRE(r.auth.headers.empty());
  REQUIRE(r.auth.sections.size() == 1);
}

TEST_CASE("http_auth_host_match_is_case_insensitive", "[http_auth]") {
  auto r = cae::ResolveHttpAuthFrom("https://CO2DataShare.org/a.zip", "",
                                    kConfig, Env({}));
  REQUIRE(r.ok);
  REQUIRE(r.auth.num_query == 2);
}

TEST_CASE("http_auth_existing_query_is_kept_and_wins", "[http_auth]") {
  auto r = cae::ResolveHttpAuthFrom(
      "https://co2datashare.org/a.zip?country=Norway#frag", "", kConfig,
      Env({}));
  REQUIRE(r.ok);
  REQUIRE(r.auth.url ==
          "https://co2datashare.org/a.zip?country=Norway"
          "&affiliation=The%20HDF%20Group");
  REQUIRE(r.auth.num_query == 1);
}

TEST_CASE("http_auth_no_match_is_plain", "[http_auth]") {
  auto r = cae::ResolveHttpAuthFrom("https://other.org/f.bin", "", kConfig,
                                    Env({}));
  REQUIRE(r.ok);
  REQUIRE(r.auth.url == "https://other.org/f.bin");
  REQUIRE(r.auth.headers.empty());
  REQUIRE(r.auth.userpwd.empty());
  REQUIRE(r.auth.sections.empty());
}

TEST_CASE("http_auth_prefix_overrides_host", "[http_auth]") {
  auto r = cae::ResolveHttpAuthFrom("https://data.example.org/private/f.h5",
                                    "", kConfig,
                                    Env({{"EX_TOKEN", "T0K"}, {"EX_PW", "pw"}}));
  REQUIRE(r.ok);
  REQUIRE(r.auth.sections.size() == 2);
  REQUIRE(r.auth.sections[0] == "data.example.org");  // broadest first
  REQUIRE(Has(r.auth.headers, "X-Api-Key: k1"));      // prefix wins
  REQUIRE(!Has(r.auth.headers, "X-Api-Key: k0"));
  REQUIRE(Has(r.auth.headers, "Authorization: Bearer T0K"));
  REQUIRE(r.auth.userpwd == "alice:pw");
}

TEST_CASE("http_auth_prefix_needs_path_match", "[http_auth]") {
  auto r = cae::ResolveHttpAuthFrom("https://data.example.org/public/f.h5", "",
                                    kConfig, Env({{"EX_PW", "pw"}}));
  REQUIRE(r.ok);
  REQUIRE(r.auth.sections.size() == 1);
  REQUIRE(Has(r.auth.headers, "X-Api-Key: k0"));
  // http:// is a different origin from the https:// prefix section.
  auto h = cae::ResolveHttpAuthFrom("http://data.example.org/private/f.h5", "",
                                    kConfig, Env({{"EX_PW", "pw"}}));
  REQUIRE(h.ok);
  REQUIRE(h.auth.sections.size() == 1);
}

TEST_CASE("http_auth_unset_variable_fails", "[http_auth]") {
  auto r = cae::ResolveHttpAuthFrom("https://data.example.org/private/f.h5",
                                    "", kConfig, Env({{"EX_PW", "pw"}}));
  REQUIRE_FALSE(r.ok);
  REQUIRE(r.error.find("EX_TOKEN") != std::string::npos);
}

TEST_CASE("http_auth_src_token_replaces_authorization", "[http_auth]") {
  const char* cfg =
      "[h.org]\n"
      "header.Authorization = Basic Zm9v\n"
      "bearer_token = fromfile\n";
  auto r = cae::ResolveHttpAuthFrom("https://h.org/x", "omni", cfg, Env({}));
  REQUIRE(r.ok);
  REQUIRE(r.auth.headers.size() == 1);
  REQUIRE(r.auth.headers[0] == "Authorization: Bearer omni");
  auto f = cae::ResolveHttpAuthFrom("https://h.org/x", "", cfg, Env({}));
  REQUIRE(f.ok);
  REQUIRE(f.auth.headers.size() == 1);
  REQUIRE(f.auth.headers[0] == "Authorization: Bearer fromfile");
}

TEST_CASE("http_auth_rejects_unknown_key_and_bad_url", "[http_auth]") {
  auto r = cae::ResolveHttpAuthFrom("https://h.org/x", "", "[h.org]\ntoken = x\n",
                                    Env({}));
  REQUIRE_FALSE(r.ok);
  REQUIRE(r.error.find("unknown key") != std::string::npos);
  auto b = cae::ResolveHttpAuthFrom("s3://bucket/key", "", "", Env({}));
  REQUIRE_FALSE(b.ok);
  auto p = cae::ResolveHttpAuthFrom("https://h.org/x", "",
                                    "[h.org]\npassword = x\n", Env({}));
  REQUIRE_FALSE(p.ok);
}

TEST_CASE("http_auth_port_is_part_of_host", "[http_auth]") {
  const char* cfg = "[h.org:8443]\nquery.k = v\n";
  auto r = cae::ResolveHttpAuthFrom("https://h.org:8443/x", "", cfg, Env({}));
  REQUIRE(r.ok);
  REQUIRE(r.auth.url == "https://h.org:8443/x?k=v");
  auto n = cae::ResolveHttpAuthFrom("https://h.org/x", "", cfg, Env({}));
  REQUIRE(n.ok);
  REQUIRE(n.auth.url == "https://h.org/x");
}

TEST_CASE("http_fetch_curl_config_quotes_and_rejects_newlines", "[http_fetch]") {
  cae::HttpAuth a;
  a.url = "https://h.org/a\"b\\c";
  a.headers = {"X-K: v"};
  a.userpwd = "u:p";
  std::string err;
  std::string c = cae::http_detail::CurlConfig(a, &err);
  REQUIRE(c ==
          "url = \"https://h.org/a\\\"b\\\\c\"\n"
          "header = \"X-K: v\"\n"
          "user = \"u:p\"\n");
  a.headers = {"X-K: v\r\nX-Evil: 1"};
  REQUIRE(cae::http_detail::CurlConfig(a, &err).empty());
  REQUIRE(!err.empty());
}

TEST_CASE("http_fetch_probe_parses_ranges_and_redirects", "[http_fetch]") {
  // A 302 to the data, then the ranged answer (curl -L -D - prints both).
  auto p = cae::http_detail::ParseProbeHeaders(
      "HTTP/2 302 \r\nlocation: https://x/y\r\ncontent-length: 591\r\n\r\n"
      "HTTP/2 206 \r\ncontent-type: application/zip\r\n"
      "content-range: bytes 0-0/37781751137\r\ncontent-length: 1\r\n\r\n");
  REQUIRE(p.status == 206);
  REQUIRE(p.ranges);
  REQUIRE(p.total == 37781751137LL);
  REQUIRE(p.content_type == "application/zip");

  auto n = cae::http_detail::ParseProbeHeaders(
      "HTTP/1.1 200 OK\r\nContent-Length: 1234\r\n\r\n");
  REQUIRE(n.status == 200);
  REQUIRE_FALSE(n.ranges);
  REQUIRE(n.total == 1234);

  auto e = cae::http_detail::ParseProbeHeaders(
      "HTTP/1.1 416 Range Not Satisfiable\r\nContent-Range: bytes */0\r\n\r\n");
  REQUIRE(e.ranges);
  REQUIRE(e.total == 0);

  auto bad = cae::http_detail::ParseProbeHeaders(
      "HTTP/2 400 \r\ncontent-type: text/html; charset=utf-8\r\n\r\n");
  REQUIRE(bad.status == 400);
  REQUIRE(bad.total == -1);
}

SIMPLE_TEST_MAIN()
