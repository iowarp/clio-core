/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * Unit tests for the summarizer chimod's inference client (label_client.cc).
 *
 * OllamaGenerate and OpenAIChatComplete talk HTTP to an inference server
 * (a local Ollama, or a hosted OpenAI-compatible service such as the ALCF
 * Metis endpoint); these tests cover every
 * branch without a real model: argument validation, connection failure, and
 * — via a tiny in-process HTTP fixture — non-200 status, malformed JSON,
 * JSON missing the 'response' field, and the success path.
 */

#include "simple_test.h"

#include <clio_cae/summarizer/label_client.h>

#include "summarizer_http_stub.h"

#include <clio_ctp/introspect/system_info.h>

#include <filesystem>
#include <fstream>
#include <string>

using clio::cae::summarizer::InferenceTarget;
using clio::cae::summarizer::LabelApi;
using clio::cae::summarizer::LabelGenerate;
using clio::cae::summarizer::OllamaGenerate;
using clio::cae::summarizer::OpenAIChatComplete;
using clio::cae::summarizer::ParseLabelApi;
using clio::cae::summarizer::ResolveApiKey;
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

// ---------------------------------------------------------------------------
// OpenAI-compatible protocol (label_api: openai), e.g. ALCF Metis
// ---------------------------------------------------------------------------

namespace {

/** A minimal successful chat/completions response. */
const char *kChatOk =
    "{\"id\":\"x\",\"object\":\"chat.completion\",\"choices\":[{\"index\":0,"
    "\"message\":{\"role\":\"assistant\",\"content\":\"a hosted label\"},"
    "\"finish_reason\":\"stop\"}]}";

/**
 * Write a token file in the temp directory.
 * @param name File name.
 * @param contents File contents.
 * @return Full path of the written file.
 */
std::string WriteTokenFile(const std::string &name,
                           const std::string &contents) {
  auto path = std::filesystem::temp_directory_path() / name;
  std::ofstream(path) << contents;
  return path.string();
}

}  // namespace

TEST_CASE("OpenAIChat - argument validation", "[summarizer][label][openai]") {
  std::string out = "stale";
  REQUIRE_FALSE(OpenAIChatComplete("", "tok", "m", "p", 0, out));
  REQUIRE(out.empty());
  REQUIRE_FALSE(OpenAIChatComplete("http://127.0.0.1:1", "tok", "", "p", 0,
                                   out));
}

TEST_CASE("OpenAIChat - transport error on unreachable endpoint",
          "[summarizer][label][openai]") {
  std::string out;
  REQUIRE_FALSE(OpenAIChatComplete("http://127.0.0.1:1", "", "m", "p", 0, out));
  REQUIRE(out.empty());
}

TEST_CASE("OpenAIChat - success sends model, messages, token, max_tokens",
          "[summarizer][label][openai]") {
  OneShotHttpServer server("HTTP/1.1 200 OK\r\n", kChatOk);
  std::string out;
  REQUIRE(OpenAIChatComplete(server.Endpoint() + "/", "secret-token",
                             "Mistral-Large-3-675B-Instruct-2512",
                             "Summarize: hello", 64, out));
  REQUIRE(out == "a hosted label");
  server.Stop();

  auto heads = server.RequestHeads();
  auto bodies = server.RequestBodies();
  REQUIRE(heads.size() == 1);
  REQUIRE(heads[0].rfind("POST /chat/completions ", 0) == 0);
  REQUIRE(heads[0].find("Authorization: Bearer secret-token") !=
          std::string::npos);
  REQUIRE(bodies[0].find("\"model\":\"Mistral-Large-3-675B-Instruct-2512\"") !=
          std::string::npos);
  REQUIRE(bodies[0].find("\"role\":\"user\"") != std::string::npos);
  REQUIRE(bodies[0].find("Summarize: hello") != std::string::npos);
  REQUIRE(bodies[0].find("\"max_tokens\":64") != std::string::npos);
}

TEST_CASE("OpenAIChat - no token sends no Authorization, no cap no max_tokens",
          "[summarizer][label][openai]") {
  OneShotHttpServer server("HTTP/1.1 200 OK\r\n", kChatOk);
  std::string out;
  REQUIRE(OpenAIChatComplete(server.Endpoint(), "", "m", "p", 0, out));
  server.Stop();
  REQUIRE(server.RequestHeads()[0].find("Authorization") == std::string::npos);
  REQUIRE(server.RequestBodies()[0].find("max_tokens") == std::string::npos);
}

TEST_CASE("OpenAIChat - error responses yield no label",
          "[summarizer][label][openai]") {
  std::string out;

  SECTION("HTTP 401 (expired token)");
  {
    OneShotHttpServer server("HTTP/1.1 401 Unauthorized\r\n",
                             "{\"detail\":\"expired\"}");
    REQUIRE_FALSE(OpenAIChatComplete(server.Endpoint(), "t", "m", "p", 0, out));
  }

  SECTION("malformed JSON");
  {
    OneShotHttpServer server("HTTP/1.1 200 OK\r\n", "not json {{{");
    REQUIRE_FALSE(OpenAIChatComplete(server.Endpoint(), "t", "m", "p", 0, out));
  }

  SECTION("missing choices");
  {
    OneShotHttpServer server("HTTP/1.1 200 OK\r\n", "{\"id\":\"x\"}");
    REQUIRE_FALSE(OpenAIChatComplete(server.Endpoint(), "t", "m", "p", 0, out));
  }

  SECTION("empty choices");
  {
    OneShotHttpServer server("HTTP/1.1 200 OK\r\n", "{\"choices\":[]}");
    REQUIRE_FALSE(OpenAIChatComplete(server.Endpoint(), "t", "m", "p", 0, out));
  }

  SECTION("choice without message");
  {
    OneShotHttpServer server("HTTP/1.1 200 OK\r\n",
                             "{\"choices\":[{\"index\":0}]}");
    REQUIRE_FALSE(OpenAIChatComplete(server.Endpoint(), "t", "m", "p", 0, out));
  }

  SECTION("null content");
  {
    OneShotHttpServer server(
        "HTTP/1.1 200 OK\r\n",
        "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":null}}]}");
    REQUIRE_FALSE(OpenAIChatComplete(server.Endpoint(), "t", "m", "p", 0, out));
  }
  REQUIRE(out.empty());
}

TEST_CASE("ParseLabelApi - accepted and rejected spellings",
          "[summarizer][label][api]") {
  LabelApi api = LabelApi::kOpenAI;
  REQUIRE(ParseLabelApi("", api));
  REQUIRE(api == LabelApi::kOllama);
  REQUIRE(ParseLabelApi("OpenAI", api));
  REQUIRE(api == LabelApi::kOpenAI);
  REQUIRE(ParseLabelApi("ollama", api));
  REQUIRE(api == LabelApi::kOllama);
  api = LabelApi::kOpenAI;
  REQUIRE_FALSE(ParseLabelApi("open-ai", api));
  REQUIRE(api == LabelApi::kOpenAI);  // untouched on failure
}

TEST_CASE("ResolveApiKey - file wins over env, whitespace trimmed",
          "[summarizer][label][token]") {
  const char *kEnv = "CLIO_TEST_SUMMARIZER_TOKEN";
  InferenceTarget t;

  SECTION("no sources: empty");
  REQUIRE(ResolveApiKey(t).empty());

  SECTION("env var only");
  ctp::SystemInfo::Setenv(kEnv, "  env-token\n", 1);
  t.api_key_env_ = kEnv;
  REQUIRE(ResolveApiKey(t) == "env-token");

  SECTION("file overrides env");
  t.api_key_file_ = WriteTokenFile("clio_summarizer_token.txt", "file-token\n");
  REQUIRE(ResolveApiKey(t) == "file-token");

  SECTION("${VAR} in the file path is expanded");
  ctp::SystemInfo::Setenv("CLIO_TEST_TOKEN_DIR",
                          std::filesystem::temp_directory_path().string(), 1);
  InferenceTarget expanded = t;
  expanded.api_key_file_ = "${CLIO_TEST_TOKEN_DIR}/clio_summarizer_token.txt";
  REQUIRE(ResolveApiKey(expanded) == "file-token");
  ctp::SystemInfo::Unsetenv("CLIO_TEST_TOKEN_DIR");

  SECTION("whitespace-only file falls back to env");
  WriteTokenFile("clio_summarizer_token.txt", " \n\t");
  REQUIRE(ResolveApiKey(t) == "env-token");

  SECTION("missing file falls back to env");
  std::filesystem::remove(t.api_key_file_);
  REQUIRE(ResolveApiKey(t) == "env-token");

  SECTION("unset env var: empty");
  ctp::SystemInfo::Unsetenv(kEnv);
  REQUIRE(ResolveApiKey(t).empty());
}

TEST_CASE("LabelGenerate - dispatches on the target's protocol",
          "[summarizer][label][api]") {
  std::string out;

  SECTION("ollama target hits /api/generate");
  {
    OneShotHttpServer server("HTTP/1.1 200 OK\r\n",
                             "{\"response\":\"local label\"}");
    InferenceTarget t;
    t.endpoint_ = server.Endpoint();
    REQUIRE(LabelGenerate(t, "m", "p", 2048, 0, out));
    REQUIRE(out == "local label");
    server.Stop();
    REQUIRE(server.RequestHeads()[0].rfind("POST /api/generate ", 0) == 0);
  }

  SECTION("openai target hits /chat/completions with the resolved token");
  {
    OneShotHttpServer server("HTTP/1.1 200 OK\r\n", kChatOk);
    InferenceTarget t;
    t.endpoint_ = server.Endpoint();
    t.api_ = LabelApi::kOpenAI;
    t.api_key_file_ = WriteTokenFile("clio_summarizer_dispatch_token.txt",
                                     "dispatch-token\n");
    REQUIRE(LabelGenerate(t, "m", "p", 2048, 0, out));
    REQUIRE(out == "a hosted label");
    server.Stop();
    REQUIRE(server.RequestHeads()[0].find("Bearer dispatch-token") !=
            std::string::npos);
    std::filesystem::remove(t.api_key_file_);
  }
}

SIMPLE_TEST_MAIN()
