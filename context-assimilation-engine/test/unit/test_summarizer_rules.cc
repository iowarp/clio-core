/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * test_summarizer_rules.cc
 *
 * Pure unit tests (no CLIO runtime, no server, no inference endpoint) for the
 * two decisions the summarizer makes before it ever prompts a model:
 *
 *   - FindLabelMatch — rule precedence, regex_search (not _match) semantics,
 *     the invalid-regex skip, and the no-match path.
 *   - SplitPayload — the chunk-boundary arithmetic, including the branches a
 *     live-model test never reaches: chunking disabled, empty payload, the
 *     prompt-template deduction, and the sanity floor.
 *
 * These live in summarizer_rules.h precisely so they can be tested here
 * rather than only through a running PutBlob path.
 */

#include "simple_test.h"

#include <clio_cae/summarizer/summarizer_rules.h>

#include <string>
#include <string_view>
#include <vector>

using clio::cae::summarizer::FindLabelMatch;
using clio::cae::summarizer::LabelMatch;
using clio::cae::summarizer::SplitPayload;

namespace {

/** Build a rule; only the fields these tests care about. */
LabelMatch MakeRule(const std::string &tag_re, const std::string &blob_re,
                    const std::string &model = "m") {
  LabelMatch r;
  r.tag_re_ = tag_re;
  r.blob_re_ = blob_re;
  r.model_ = model;
  r.prompt_ = "summarize";
  return r;
}

/** Total bytes across all chunks — must always equal the payload size. */
size_t TotalBytes(const std::vector<std::string_view> &chunks) {
  size_t n = 0;
  for (const auto &c : chunks) n += c.size();
  return n;
}

}  // namespace

//==============================================================================
// FindLabelMatch
//==============================================================================

TEST_CASE("FindLabelMatch - returns null when there are no rules",
          "[summarizer][rules][match]") {
  std::vector<LabelMatch> rules;
  REQUIRE(FindLabelMatch(rules, "any_tag", "any_blob") == nullptr);
}

TEST_CASE("FindLabelMatch - both sides must match",
          "[summarizer][rules][match]") {
  std::vector<LabelMatch> rules{MakeRule("^docs$", "^doc_.*")};
  // Tag matches, blob does not.
  REQUIRE(FindLabelMatch(rules, "docs", "other") == nullptr);
  // Blob matches, tag does not.
  REQUIRE(FindLabelMatch(rules, "images", "doc_a") == nullptr);
  // Both match.
  REQUIRE(FindLabelMatch(rules, "docs", "doc_a") == &rules[0]);
}

TEST_CASE("FindLabelMatch - uses regex_search, not regex_match",
          "[summarizer][rules][match]") {
  // An unanchored suffix pattern must hit a longer name. This is the whole
  // reason the implementation uses regex_search.
  std::vector<LabelMatch> rules{MakeRule(".*\\.txt", ".*")};
  REQUIRE(FindLabelMatch(rules, "corpus/notes.txt", "blob") == &rules[0]);
}

TEST_CASE("FindLabelMatch - first matching rule wins",
          "[summarizer][rules][match]") {
  std::vector<LabelMatch> rules{MakeRule(".*", ".*", "first"),
                                MakeRule(".*", ".*", "second")};
  const LabelMatch *hit = FindLabelMatch(rules, "t", "b");
  REQUIRE(hit != nullptr);
  REQUIRE(hit->model_ == "first");
}

TEST_CASE("FindLabelMatch - an invalid regex skips only that rule",
          "[summarizer][rules][match]") {
  // "[" is an unterminated character class => std::regex_error. The rule is
  // skipped (logged) and matching CONTINUES, so one bad line in a config
  // cannot silence every rule after it.
  std::vector<LabelMatch> rules{MakeRule("[", ".*", "bad"),
                                MakeRule(".*", ".*", "good")};
  const LabelMatch *hit = FindLabelMatch(rules, "t", "b");
  REQUIRE(hit != nullptr);
  REQUIRE(hit->model_ == "good");
}

TEST_CASE("FindLabelMatch - an invalid blob regex is caught too",
          "[summarizer][rules][match]") {
  std::vector<LabelMatch> rules{MakeRule(".*", "(", "bad")};
  REQUIRE(FindLabelMatch(rules, "t", "b") == nullptr);
}

TEST_CASE("FindLabelMatch - an empty tag name can still match .*",
          "[summarizer][rules][match]") {
  // An unresolvable tag id yields ""; a .* rule must still fire on it.
  std::vector<LabelMatch> rules{MakeRule(".*", ".*")};
  REQUIRE(FindLabelMatch(rules, "", "blob") == &rules[0]);
  // ...but a rule demanding real tag text must not.
  std::vector<LabelMatch> strict{MakeRule("something", ".*")};
  REQUIRE(FindLabelMatch(strict, "", "blob") == nullptr);
}

//==============================================================================
// SplitPayload
//==============================================================================

TEST_CASE("SplitPayload - ctx_tokens <= 0 disables chunking",
          "[summarizer][rules][chunk]") {
  std::string payload(10000, 'x');
  for (int ctx : {0, -1, -4096}) {
    auto chunks = SplitPayload(payload, "prompt", ctx);
    REQUIRE(chunks.size() == 1);
    REQUIRE(chunks[0].size() == payload.size());
  }
}

TEST_CASE("SplitPayload - an empty payload yields one empty chunk",
          "[summarizer][rules][chunk]") {
  std::string payload;
  auto chunks = SplitPayload(payload, "prompt", 4096);
  REQUIRE(chunks.size() == 1);
  REQUIRE(chunks[0].empty());
}

TEST_CASE("SplitPayload - a payload inside the budget is one chunk",
          "[summarizer][rules][chunk]") {
  // ctx 4096 => 3072 tokens => 9216 bytes, minus the template.
  std::string payload(1024, 'a');
  auto chunks = SplitPayload(payload, "prompt", 4096);
  REQUIRE(chunks.size() == 1);
  REQUIRE(chunks[0].size() == payload.size());
}

TEST_CASE("SplitPayload - an oversized payload splits and loses nothing",
          "[summarizer][rules][chunk]") {
  const std::string prompt = "Summarize:";
  // ctx 512 => 384 tokens => 1152 bytes; 1152 > 10 + 256 so the template is
  // deducted => 1142 bytes per chunk.
  const size_t kExpectedBudget = (static_cast<size_t>(512) * 3 / 4) * 3
                                 - prompt.size();
  std::string payload(5000, 'z');
  auto chunks = SplitPayload(payload, prompt, 512);

  REQUIRE(chunks.size() > 1);
  // No byte is dropped or duplicated.
  REQUIRE(TotalBytes(chunks) == payload.size());
  // Every chunk but the last is exactly the budget; none exceeds it.
  for (size_t i = 0; i + 1 < chunks.size(); ++i) {
    REQUIRE(chunks[i].size() == kExpectedBudget);
  }
  REQUIRE(chunks.back().size() <= kExpectedBudget);
  REQUIRE(!chunks.back().empty());
}

TEST_CASE("SplitPayload - chunks reassemble to the original bytes",
          "[summarizer][rules][chunk]") {
  std::string payload;
  for (size_t i = 0; i < 4000; ++i) {
    payload.push_back(static_cast<char>('a' + (i % 26)));
  }
  auto chunks = SplitPayload(payload, "p", 512);
  std::string rejoined;
  for (const auto &c : chunks) rejoined.append(c.data(), c.size());
  REQUIRE(rejoined == payload);
}

TEST_CASE("SplitPayload - a huge template does not underflow the budget",
          "[summarizer][rules][chunk]") {
  // budget_bytes (1152) is NOT > template size + 256, so the deduction is
  // skipped rather than wrapping around on the unsigned subtraction. This is
  // the branch that would produce a gigantic budget if it were mishandled.
  std::string prompt(4000, 'p');
  std::string payload(5000, 'q');
  auto chunks = SplitPayload(payload, prompt, 512);
  const size_t kUndeducted = (static_cast<size_t>(512) * 3 / 4) * 3;
  REQUIRE(chunks.size() > 1);
  REQUIRE(TotalBytes(chunks) == payload.size());
  REQUIRE(chunks[0].size() == kUndeducted);
}

TEST_CASE("SplitPayload - a tiny context falls back to the sanity floor",
          "[summarizer][rules][chunk]") {
  // ctx 1 => 0 tokens => 0 bytes; the floor keeps it at 256 instead of
  // looping forever on a zero stride.
  std::string payload(1000, 'w');
  auto chunks = SplitPayload(payload, "p", 1);
  REQUIRE(chunks.size() == 4);  // ceil(1000 / 256)
  REQUIRE(chunks[0].size() == 256);
  REQUIRE(TotalBytes(chunks) == payload.size());
}

SIMPLE_TEST_MAIN()
