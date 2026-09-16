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

#ifndef CLIO_CAE_SUMMARIZER_SUMMARIZER_RULES_H_
#define CLIO_CAE_SUMMARIZER_SUMMARIZER_RULES_H_

/**
 * The two pure decisions the summarizer makes before it ever talks to a model:
 * WHICH rule applies to a blob, and HOW the payload is split to fit the
 * model's context.
 *
 * They live in this header, out of summarizer_runtime.cc, for the same reason
 * factory/hdf5_export.h exists in the CAE core: inside the PutBlob coroutine
 * these mechanics cannot be tested without a runtime, a server, and an
 * inference endpoint. Here they are ordinary functions over ordinary types --
 * the chunk-boundary arithmetic and the regex edge cases get direct unit
 * tests. Keep this header a LEAF: no runtime, no CTE, no task types.
 */

#include <clio_cae/summarizer/summarizer_tasks.h>
#include <clio_ctp/util/logging.h>

#include <algorithm>
#include <regex>
#include <string>
#include <string_view>
#include <vector>

namespace clio::cae::summarizer {

/**
 * Walk the configured rules and return the first (if any) whose tag_re_
 * matches `tag_name` and blob_re_ matches `blob_name`.
 *
 * Matching uses std::regex_search, not regex_match, so ".*\\.txt" matches any
 * name ENDING in .txt without needing an explicit anchor. An invalid regex on
 * either side disables that rule -- it is logged at kWarning and matching
 * continues with the next rule, so one bad line in a config cannot silence
 * every rule after it.
 *
 * @param rules The configured summarization rules, in priority order.
 * @param tag_name Resolved name of the blob's tag ("" when unresolvable).
 * @param blob_name Name of the blob being written.
 * @return Pointer to the winning rule (owned by `rules`), or nullptr when
 *         nothing matches.
 */
inline const LabelMatch *FindLabelMatch(const std::vector<LabelMatch> &rules,
                                        const std::string &tag_name,
                                        const std::string &blob_name) {
  for (const auto &rule : rules) {
    try {
      std::regex tag_rx(rule.tag_re_);
      std::regex blob_rx(rule.blob_re_);
      if (std::regex_search(tag_name, tag_rx) &&
          std::regex_search(blob_name, blob_rx)) {
        return &rule;
      }
    } catch (const std::regex_error &e) {
      HLOG(kWarning,
           "FindLabelMatch: invalid regex in rule (tag='{}' blob='{}'): {}",
           rule.tag_re_, rule.blob_re_, e.what());
    }
  }
  return nullptr;
}

/**
 * Split a blob payload into per-request slices that fit the model's context.
 *
 * The Ollama API counts both prompt and generated tokens against num_ctx.
 * Reserve ~25% of context for the prompt template plus the response budget;
 * the remaining 75% is available for blob payload. Tokens become bytes via a
 * conservative ~3 bytes/token English ratio -- binary blobs run closer to
 * 1 byte/token, so this errs on splitting MORE rather than overflowing.
 *
 * @param payload The blob bytes to summarize.
 * @param prompt_template The template prepended to each chunk; its size comes
 *        out of the per-request byte budget.
 * @param ctx_tokens The rule's context_length. <= 0 disables chunking
 *        entirely -- the caller then takes Ollama's own default (~2048) and
 *        accepts whatever truncation it does.
 * @return One view per request, each borrowing from `payload`, so `payload`
 *         must outlive the result. Always at least one element (an empty
 *         payload yields a single empty view).
 */
inline std::vector<std::string_view> SplitPayload(
    const std::string &payload, const std::string &prompt_template,
    int ctx_tokens) {
  std::vector<std::string_view> chunks;
  if (ctx_tokens <= 0 || payload.empty()) {
    chunks.emplace_back(payload);
    return chunks;
  }
  size_t budget_tokens = static_cast<size_t>(ctx_tokens) * 3 / 4;
  size_t budget_bytes = budget_tokens * 3;
  if (budget_bytes > prompt_template.size() + 256) {
    budget_bytes -= prompt_template.size();
  }
  if (budget_bytes == 0) budget_bytes = 256;  // sanity floor
  for (size_t off = 0; off < payload.size(); off += budget_bytes) {
    size_t take = std::min(budget_bytes, payload.size() - off);
    chunks.emplace_back(payload.data() + off, take);
  }
  return chunks;
}

}  // namespace clio::cae::summarizer

#endif  // CLIO_CAE_SUMMARIZER_SUMMARIZER_RULES_H_
