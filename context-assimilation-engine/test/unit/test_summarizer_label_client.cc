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

#include <string>

using clio::cae::summarizer::OllamaGenerate;
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

SIMPLE_TEST_MAIN()
