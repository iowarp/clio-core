/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * AggregateOut contract lint (issue #915).
 *
 * `Task::AggregateOut` merges a REPLICA's OUT fields into the ORIGIN task
 * (an N->1 gather). `Copy()` is a WHOLE-TASK assignment. Implementing the
 * former with the latter does two illegal things:
 *
 *   1. It runs `Task::Copy`, overwriting the ORIGIN's task_id_, pool_query_,
 *      method_, task_flags_, completer_ and task_group_ with the REPLICA's --
 *      destroying the origin's identity while send_map_, the replica
 *      accounting and the completion path still reference it. The replica's
 *      task_id_ carries net_key/replica_id, so the origin adopts a subtask's
 *      identity mid-flight.
 *   2. It re-assigns IN fields, including priv::string / shm-allocated
 *      members, from the replica's shared-memory segment -- freeing the
 *      origin's buffer through the wrong allocator.
 *
 * That is the `free(): invalid pointer` / SIGSEGV that killed leader-election
 * recovery in #856, plus silently wrong OUT values (last-replica-wins instead
 * of a real merge) everywhere else.
 *
 * The shape is trivially copy-pasteable and was present in 77 task types
 * across the tree, so a runtime assertion alone is not enough: a latent
 * instance only detonates once someone routes that task Broadcast/multi-
 * replica, which may be years later and in a different module. This test is
 * the cheap, total guard -- it reads the SOURCE and fails the build's test
 * suite the moment any AggregateOut body calls Copy() again, including in
 * modules that no test binary links.
 *
 * The complementary runtime checks live in
 * test/unit/autogen_more/test_autogen_methods_sweep.cc (identity survives a
 * real container-dispatched aggregation) and in RecvOut's contract guard in
 * src/ipc/ipc_run2run.cc (catches out-of-tree chimods at run time).
 */

#include "simple_test.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

/** Directory names that are never part of this repository's own sources. */
bool IsSkippedDir(const std::string &name) {
  static const std::vector<std::string> kSkipped = {
      ".git",   "external", "third-party", "third_party",
      "node_modules", "venv", ".venv", "docs"};
  if (std::find(kSkipped.begin(), kSkipped.end(), name) != kSkipped.end()) {
    return true;
  }
  // build/, build-cpu/, cmake-build-debug/, ...
  return name.rfind("build", 0) == 0 || name.rfind("cmake-build", 0) == 0;
}

bool IsSourceFile(const fs::path &p) {
  const std::string ext = p.extension().string();
  return ext == ".h" || ext == ".hpp" || ext == ".cc" || ext == ".cpp" ||
         ext == ".cu" || ext == ".cuh";
}

/** Replace // and comments with spaces so the scan only sees real code. */
std::string StripComments(const std::string &src) {
  std::string out = src;
  for (size_t i = 0; i + 1 < out.size(); ++i) {
    if (out[i] == '/' && out[i + 1] == '/') {
      while (i < out.size() && out[i] != '\n') {
        out[i++] = ' ';
      }
    } else if (out[i] == '/' && out[i + 1] == '*') {
      out[i] = out[i + 1] = ' ';
      i += 2;
      while (i + 1 < out.size() && !(out[i] == '*' && out[i + 1] == '/')) {
        if (out[i] != '\n') {
          out[i] = ' ';
        }
        ++i;
      }
      if (i + 1 < out.size()) {
        out[i] = out[i + 1] = ' ';
        ++i;
      }
    }
  }
  return out;
}

bool IsIdentChar(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
         (c >= '0' && c <= '9') || c == '_';
}

/**
 * True if `body` contains a call to Copy(...) that is not a qualified name
 * (Task::Copy / Base::Copy) and not the tail of a longer identifier
 * (NewCopyTask, DeepCopy, ...).
 */
bool CallsCopy(const std::string &body) {
  const std::string kNeedle = "Copy";
  size_t pos = 0;
  while ((pos = body.find(kNeedle, pos)) != std::string::npos) {
    size_t after = pos + kNeedle.size();
    // Skip whitespace between the name and '('.
    size_t paren = after;
    while (paren < body.size() && std::isspace((unsigned char)body[paren])) {
      ++paren;
    }
    const bool is_call = paren < body.size() && body[paren] == '(';
    bool qualified = false;
    bool suffix_of_ident = false;
    if (pos > 0) {
      suffix_of_ident = IsIdentChar(body[pos - 1]);
    }
    if (pos >= 2) {
      qualified = body[pos - 1] == ':' && body[pos - 2] == ':';
    }
    if (is_call && !qualified && !suffix_of_ident) {
      return true;
    }
    pos = after;
  }
  return false;
}

/** Extract the {...} block starting at or after `from`. Empty if none. */
std::string BraceBlock(const std::string &src, size_t from) {
  size_t open = src.find('{', from);
  if (open == std::string::npos) {
    return "";
  }
  int depth = 0;
  for (size_t i = open; i < src.size(); ++i) {
    if (src[i] == '{') {
      ++depth;
    } else if (src[i] == '}') {
      --depth;
      if (depth == 0) {
        return src.substr(open, i - open + 1);
      }
    }
  }
  return "";
}

size_t LineOf(const std::string &src, size_t pos) {
  return 1 + (size_t)std::count(src.begin(), src.begin() + (long)pos, '\n');
}

/** One offending AggregateOut implementation. */
struct Violation {
  std::string file;
  size_t line;
};

void ScanFile(const fs::path &path, const std::string &rel,
              std::vector<Violation> &out) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return;
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  const std::string raw = ss.str();
  if (raw.find("AggregateOut") == std::string::npos) {
    return;
  }
  const std::string src = StripComments(raw);

  size_t pos = 0;
  const std::string kName = "AggregateOut";
  while ((pos = src.find(kName, pos)) != std::string::npos) {
    size_t after = pos + kName.size();
    size_t paren = after;
    while (paren < src.size() && std::isspace((unsigned char)src[paren])) {
      ++paren;
    }
    // A definition: `... AggregateOut(...) { ... }`, not a call and not a
    // qualified invocation such as `Task::AggregateOut(x);`.
    const bool qualified =
        pos >= 2 && src[pos - 1] == ':' && src[pos - 2] == ':';
    if (paren >= src.size() || src[paren] != '(' || qualified) {
      pos = after;
      continue;
    }
    // Match the parameter list's closing paren (parameters may themselves
    // contain parens, e.g. a defaulted argument).
    size_t close = std::string::npos;
    int pdepth = 0;
    for (size_t i = paren; i < src.size(); ++i) {
      if (src[i] == '(') {
        ++pdepth;
      } else if (src[i] == ')') {
        if (--pdepth == 0) {
          close = i;
          break;
        }
      }
    }
    if (close == std::string::npos) {
      break;
    }
    size_t brace = close + 1;
    while (brace < src.size() && std::isspace((unsigned char)src[brace])) {
      ++brace;
    }
    if (brace >= src.size() || src[brace] != '{') {
      pos = after;  // declaration, or a call -- not a definition
      continue;
    }
    const std::string body = BraceBlock(src, close);
    if (!body.empty() && CallsCopy(body)) {
      out.push_back({rel, LineOf(src, pos)});
    }
    pos = brace + body.size();
  }
}

}  // namespace

TEST_CASE("AggregateOut never delegates to Copy (issue #915)",
          "[aggregate_out][contract][lint]") {
#ifndef CLIO_REPO_SOURCE_DIR
  INFO("CLIO_REPO_SOURCE_DIR not defined - skipping source lint");
  REQUIRE(true);
#else
  const fs::path root(CLIO_REPO_SOURCE_DIR);
  if (!fs::is_directory(root)) {
    // Installed/relocated test binary: the source tree is not present. The
    // lint is a build-time guard, so there is nothing to check here.
    INFO("Source tree not present at " + root.string() + " - skipping lint");
    REQUIRE(true);
    return;
  }

  std::vector<Violation> violations;
  size_t files_scanned = 0;
  for (auto it = fs::recursive_directory_iterator(
           root, fs::directory_options::skip_permission_denied);
       it != fs::recursive_directory_iterator(); ++it) {
    const fs::path &p = it->path();
    if (it->is_directory()) {
      if (IsSkippedDir(p.filename().string())) {
        it.disable_recursion_pending();
      }
      continue;
    }
    if (!it->is_regular_file() || !IsSourceFile(p)) {
      continue;
    }
    ++files_scanned;
    ScanFile(p, fs::relative(p, root).generic_string(), violations);
  }

  // Sanity: the walk must actually have found this repository's sources. A
  // silently empty scan would make the lint vacuously pass forever.
  INFO("scanned " + std::to_string(files_scanned) + " source files");
  REQUIRE(files_scanned > 100);

  if (!violations.empty()) {
    std::ostringstream msg;
    msg << violations.size()
        << " AggregateOut implementation(s) delegate to Copy(), which is a "
           "WHOLE-TASK assignment and corrupts the origin task's identity "
           "(issue #915). Merge OUT fields only:\n";
    for (const Violation &v : violations) {
      msg << "  " << v.file << ":" << v.line << "\n";
    }
    INFO(msg.str());
  }
  REQUIRE(violations.empty());
#endif
}

SIMPLE_TEST_MAIN()
