/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * Unit tests for the summarizer chimod's Ollama client (label_client.cc).
 *
 * OllamaGenerate talks HTTP to an inference server; these tests cover every
 * branch without a real model: argument validation, connection failure, and
 * — via a tiny in-process HTTP fixture — non-200 status, malformed JSON,
 * JSON missing the 'response' field, and the success path.
 */

#include "simple_test.h"

#include <clio_cae/summarizer/label_client.h>

#include "summarizer_http_stub.h"

#include <clio_ctp/introspect/system_info.h>

#include <string>

using clio::cae::summarizer::OllamaGenerate;
using clio::cae::summarizer::detail::LoopbackNoProxyList;
using clio_cae_test::OneShotHttpServer;

TEST_CASE("LabelClient - argument validation", "[summarizer][label][args]") {
  std::string out;

  SECTION("Empty endpoint rejected");
  out = "stale";
  REQUIRE_FALSE(OllamaGenerate("", "model", "prompt", 0, 0, out));
  REQUIRE(out.empty());

  SECTION("Empty model rejected");
  REQUIRE_FALSE(OllamaGenerate("http://127.0.0.1:11434", "", "prompt", 0, 0,
                               out));
}

TEST_CASE("LabelClient - transport error on unreachable endpoint",
          "[summarizer][label][transport]") {
  std::string out;
  // Port 1 on localhost: connection refused almost immediately.
  REQUIRE_FALSE(OllamaGenerate("http://127.0.0.1:1", "m", "p", 128, 16, out));
  REQUIRE(out.empty());
}

TEST_CASE("LabelClient - HTTP error status", "[summarizer][label][http]") {
  OneShotHttpServer server("HTTP/1.1 500 Internal Server Error\r\n",
                           "{\"error\":\"boom\"}");
  std::string out;
  REQUIRE_FALSE(OllamaGenerate(server.Endpoint(), "m", "p", 0, 0, out));
  server.Stop();
}

TEST_CASE("LabelClient - malformed JSON body", "[summarizer][label][badjson]") {
  OneShotHttpServer server("HTTP/1.1 200 OK\r\n", "this is not json {{{");
  std::string out;
  REQUIRE_FALSE(OllamaGenerate(server.Endpoint(), "m", "p", 0, 0, out));
  server.Stop();
}

TEST_CASE("LabelClient - JSON missing response field",
          "[summarizer][label][nofield]") {
  OneShotHttpServer server("HTTP/1.1 200 OK\r\n", "{\"done\":true}");
  std::string out;
  REQUIRE_FALSE(OllamaGenerate(server.Endpoint(), "m", "p", 0, 0, out));
  server.Stop();
}

TEST_CASE("LabelClient - success path", "[summarizer][label][success]") {
  OneShotHttpServer server("HTTP/1.1 200 OK\r\n",
                           "{\"response\":\"a fine label\",\"done\":true}");
  std::string out;

  SECTION("Plain request succeeds");
  REQUIRE(OllamaGenerate(server.Endpoint(), "m", "p", 0, 0, out));
  REQUIRE(out == "a fine label");

  SECTION("Trailing slash and options (num_ctx/num_predict) accepted");
  REQUIRE(OllamaGenerate(server.Endpoint() + "/", "m", "p", 2048, 64, out));
  REQUIRE(out == "a fine label");

  server.Stop();
}

/**
 * A site-wide http_proxy must not capture a loopback endpoint.
 *
 * Ollama runs on the same host, so the endpoint is 127.0.0.1. Where the
 * environment exports an HTTP proxy -- every ALCF login and compute node
 * does, and container images often do -- libcurl would hand even a loopback
 * request to the proxy, which refuses to relay to 127.0.0.1 and returns its
 * own error page. The symptom is an HTTP 503 and an empty label with no sign
 * that a proxy was involved at all.
 *
 * The proxy named here is deliberately unroutable: if the bypass regresses,
 * the request goes to the proxy and this fails rather than silently
 * succeeding against something real.
 */
TEST_CASE("LabelClient - loopback bypasses a configured proxy",
          "[summarizer][label][proxy]") {
  OneShotHttpServer server("HTTP/1.1 200 OK\r\n",
                           "{\"response\":\"a fine label\",\"done\":true}");

  ctp::SystemInfo::Setenv("http_proxy", "http://127.0.0.1:9/", 1);
  ctp::SystemInfo::Setenv("https_proxy", "http://127.0.0.1:9/", 1);
  ctp::SystemInfo::Unsetenv("no_proxy");
  ctp::SystemInfo::Unsetenv("NO_PROXY");

  std::string out;
  const bool ok = OllamaGenerate(server.Endpoint(), "m", "p", 0, 0, out);

  ctp::SystemInfo::Unsetenv("http_proxy");
  ctp::SystemInfo::Unsetenv("https_proxy");

  REQUIRE(ok);
  REQUIRE(out == "a fine label");
  server.Stop();
}

/**
 * The bypass ADDS loopback to the environment's list rather than replacing it.
 *
 * CURLOPT_NOPROXY overrides libcurl's own reading of no_proxy outright, so a
 * client that listed only loopback would silently push a site's internal
 * hosts back through the proxy. Asserted on the list itself: a request to a
 * loopback endpoint succeeds either way, so behaviour alone cannot tell the
 * two apart.
 *
 * Both spellings matter -- libcurl reads lowercase first, so NO_PROXY has to
 * be picked up when no_proxy is unset or empty.
 */
TEST_CASE("LabelClient - the no_proxy list keeps the environment's entries",
          "[summarizer][label][proxy]") {
  const std::string kLoopback = "localhost,127.0.0.1,::1";

  SECTION("neither variable set: loopback only");
  ctp::SystemInfo::Unsetenv("no_proxy");
  ctp::SystemInfo::Unsetenv("NO_PROXY");
  REQUIRE(LoopbackNoProxyList() == kLoopback);

  SECTION("lowercase no_proxy is appended, not dropped");
  ctp::SystemInfo::Setenv("no_proxy", "example.invalid,10.0.0.0/8", 1);
  REQUIRE(LoopbackNoProxyList() == kLoopback + ",example.invalid,10.0.0.0/8");

  SECTION("uppercase NO_PROXY is used when no_proxy is unset");
  ctp::SystemInfo::Unsetenv("no_proxy");
  ctp::SystemInfo::Setenv("NO_PROXY", "upper.invalid", 1);
  REQUIRE(LoopbackNoProxyList() == kLoopback + ",upper.invalid");

  SECTION("an empty no_proxy falls through to NO_PROXY");
  ctp::SystemInfo::Setenv("no_proxy", "", 1);
  REQUIRE(LoopbackNoProxyList() == kLoopback + ",upper.invalid");

  SECTION("both empty: loopback only");
  ctp::SystemInfo::Setenv("NO_PROXY", "", 1);
  REQUIRE(LoopbackNoProxyList() == kLoopback);

  ctp::SystemInfo::Unsetenv("no_proxy");
  ctp::SystemInfo::Unsetenv("NO_PROXY");
}

SIMPLE_TEST_MAIN()
