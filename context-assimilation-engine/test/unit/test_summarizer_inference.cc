/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * test_summarizer_inference.cc
 *
 * Pure tests (no CLIO runtime, no CTE, no real model) for GenerateSummary --
 * the chunk-and-concatenate loop. An in-process HTTP stub plays the role of
 * the inference server, so the semantics that a live-Ollama test can only
 * observe indirectly get asserted directly here:
 *
 *   - a single chunk returns the model's text verbatim;
 *   - multiple chunks are joined by a blank line, one request per chunk;
 *   - a non-200 status, a malformed body, or an empty `response` yields ""
 *     (every chunk failed) rather than a bogus summary;
 *   - the prompt template is prepended to each chunk's request.
 */

#include "simple_test.h"
#include "summarizer_http_stub.h"

#include <clio_cae/summarizer/summarizer_inference.h>

#include <string>
#include <vector>

using clio::cae::summarizer::GenerateSummary;
using clio::cae::summarizer::LabelMatch;
using clio_cae_test::OneShotHttpServer;

namespace {

/** A rule with the given context budget; model/prompt names are incidental. */
LabelMatch MakeRule(int context_length) {
  LabelMatch r;
  r.tag_re_ = ".*";
  r.blob_re_ = ".*";
  r.model_ = "stub-model";
  r.prompt_ = "summarize";
  r.context_length_ = context_length;
  r.num_predict_ = 0;
  return r;
}

const char *kPrompt = "Summarize this:";

}  // namespace

TEST_CASE("GenerateSummary - single chunk returns the model text",
          "[summarizer][inference]") {
  OneShotHttpServer stub("HTTP/1.1 200 OK\r\n", "{\"response\":\"a summary\"}");
  std::string out = GenerateSummary(stub.Endpoint(), MakeRule(4096), kPrompt,
                                    "a short document", "tag", "blob");
  REQUIRE(out == "a summary");
  REQUIRE(stub.RequestCount() == 1);
  stub.Stop();
}

TEST_CASE("GenerateSummary - the prompt template precedes the payload",
          "[summarizer][inference]") {
  OneShotHttpServer stub("HTTP/1.1 200 OK\r\n", "{\"response\":\"ok\"}");
  GenerateSummary(stub.Endpoint(), MakeRule(4096), kPrompt, "PAYLOAD_MARKER",
                  "tag", "blob");
  std::vector<std::string> bodies = stub.RequestBodies();
  REQUIRE(bodies.size() == 1);
  size_t prompt_at = bodies[0].find("Summarize this:");
  size_t payload_at = bodies[0].find("PAYLOAD_MARKER");
  REQUIRE(prompt_at != std::string::npos);
  REQUIRE(payload_at != std::string::npos);
  REQUIRE(prompt_at < payload_at);
  stub.Stop();
}

TEST_CASE("GenerateSummary - chunks are joined by a blank line",
          "[summarizer][inference]") {
  OneShotHttpServer stub("HTTP/1.1 200 OK\r\n", "{\"response\":\"S\"}");
  // ctx 1 => the 256-byte sanity floor => 1000 bytes splits into 4 chunks.
  std::string payload(1000, 'x');
  std::string out =
      GenerateSummary(stub.Endpoint(), MakeRule(1), kPrompt, payload, "t", "b");
  REQUIRE(stub.RequestCount() == 4);
  REQUIRE(out == "S\n\nS\n\nS\n\nS");
  stub.Stop();
}

TEST_CASE("GenerateSummary - a non-200 status yields no summary",
          "[summarizer][inference]") {
  OneShotHttpServer stub("HTTP/1.1 500 Internal Server Error\r\n", "{}");
  std::string out = GenerateSummary(stub.Endpoint(), MakeRule(4096), kPrompt,
                                    "doc", "t", "b");
  REQUIRE(out.empty());
  stub.Stop();
}

TEST_CASE("GenerateSummary - a malformed body yields no summary",
          "[summarizer][inference]") {
  OneShotHttpServer stub("HTTP/1.1 200 OK\r\n", "{not json");
  std::string out = GenerateSummary(stub.Endpoint(), MakeRule(4096), kPrompt,
                                    "doc", "t", "b");
  REQUIRE(out.empty());
  stub.Stop();
}

TEST_CASE("GenerateSummary - an empty response field yields no summary",
          "[summarizer][inference]") {
  // OllamaGenerate succeeds but hands back "", which must count as a failed
  // chunk rather than being concatenated as empty text.
  OneShotHttpServer stub("HTTP/1.1 200 OK\r\n", "{\"response\":\"\"}");
  std::string out = GenerateSummary(stub.Endpoint(), MakeRule(4096), kPrompt,
                                    "doc", "t", "b");
  REQUIRE(out.empty());
  stub.Stop();
}

TEST_CASE("GenerateSummary - an unreachable endpoint yields no summary",
          "[summarizer][inference]") {
  // Port 1 on localhost refuses immediately. This is the shape a
  // misconfigured label_endpoint takes in production.
  std::string out = GenerateSummary("http://127.0.0.1:1", MakeRule(4096),
                                    kPrompt, "doc", "t", "b");
  REQUIRE(out.empty());
}

SIMPLE_TEST_MAIN()
