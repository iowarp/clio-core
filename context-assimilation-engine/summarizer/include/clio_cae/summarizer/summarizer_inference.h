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

#ifndef CLIO_CAE_SUMMARIZER_SUMMARIZER_INFERENCE_H_
#define CLIO_CAE_SUMMARIZER_SUMMARIZER_INFERENCE_H_

/**
 * The chunk-and-concatenate loop that turns one blob payload into one summary.
 *
 * Split out of summarizer_runtime.cc for the same testability reason as
 * summarizer_rules.h: inside the PutBlob coroutine this loop needs a runtime,
 * a CTE chain, and a live model to reach at all, so its partial-failure
 * semantics went untested. Here it takes an endpoint and returns a string, and
 * a test can point it at an in-process HTTP stub.
 *
 * This header is a leaf over the HTTP client and the chunker -- no runtime, no
 * CTE, no task dispatch.
 */

#include <clio_cae/summarizer/label_client.h>
#include <clio_cae/summarizer/summarizer_rules.h>
#include <clio_ctp/util/logging.h>

#include <string>
#include <string_view>
#include <vector>

namespace clio::cae::summarizer {

/**
 * Run the rule's prompt over every chunk of the payload and concatenate the
 * responses, separated by a blank line.
 *
 * A failure on any ONE chunk does not abort the whole summary — that chunk is
 * skipped and logged, and the caller still gets a partial summary from the
 * rest. Only when EVERY chunk fails is the result empty, which is the caller's
 * signal to store nothing. A production deploy would dispatch each chunk to a
 * dedicated inference worker pool; this runs them inline on the calling worker.
 *
 * @param endpoint Base URL of the Ollama-compatible inference server.
 * @param rule The matched rule (supplies model, context_length, num_predict).
 * @param prompt_template Prompt text prepended to each chunk.
 * @param payload The blob bytes to summarize.
 * @param tag_name Tag name, for log context only.
 * @param blob_name Blob name, for log context only.
 * @return The concatenated summary, or "" when every chunk failed.
 */
inline std::string GenerateSummary(const std::string &endpoint,
                                   const LabelMatch &rule,
                                   const std::string &prompt_template,
                                   const std::string &payload,
                                   const std::string &tag_name,
                                   const std::string &blob_name) {
  std::vector<std::string_view> chunks =
      SplitPayload(payload, prompt_template, rule.context_length_);
  std::string summary;
  size_t successful_chunks = 0;
  for (size_t i = 0; i < chunks.size(); ++i) {
    std::string full_prompt = prompt_template;
    full_prompt.append("\n\n");
    full_prompt.append(chunks[i].data(), chunks[i].size());

    std::string chunk_summary;
    bool ok = OllamaGenerate(endpoint, rule.model_, full_prompt,
                             rule.context_length_, rule.num_predict_,
                             chunk_summary);
    if (!ok || chunk_summary.empty()) {
      HLOG(kWarning,
           "Summarizer: chunk {} of {} failed for tag='{}' blob='{}' "
           "model='{}'",
           i + 1, chunks.size(), tag_name, blob_name, rule.model_);
      continue;
    }
    if (!summary.empty()) summary.append("\n\n");
    summary.append(chunk_summary);
    ++successful_chunks;
  }
  if (successful_chunks == 0) summary.clear();
  return summary;
}

}  // namespace clio::cae::summarizer

#endif  // CLIO_CAE_SUMMARIZER_SUMMARIZER_INFERENCE_H_
