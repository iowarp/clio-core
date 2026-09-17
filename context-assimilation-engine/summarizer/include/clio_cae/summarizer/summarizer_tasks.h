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

#ifndef CLIO_CAE_SUMMARIZER_SUMMARIZER_TASKS_H_
#define CLIO_CAE_SUMMARIZER_SUMMARIZER_TASKS_H_

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/task.h>
#include <clio_runtime/admin/admin_tasks.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cae/summarizer/autogen/summarizer_methods.h>

#include <string>
#include <unordered_map>
#include <vector>
#include <yaml-cpp/yaml.h>

namespace clio::cae::summarizer {

/** The summarizer chimod speaks the CTE core's vocabulary. */
using Context = clio::cte::core::Context;
using TagId = clio::cte::core::TagId;
using PutBlobTask = clio::cte::core::PutBlobTask;

/**
 * Well-known default pool id/name. The summarizer is an interposer: it sits
 * above whatever pool it forwards to (`next_pool_id`, normally the CTE core
 * at 512.0) and below whatever addresses it (the CAE core at 400.0, a
 * clio::cte::core::Client, or a longer chain).
 */
static constexpr clio::run::PoolId kSummarizerPoolId(401, 0);
static constexpr const char *kSummarizerPoolName = "clio_cae_summarizer";

/**
 * One summarization rule from compose YAML.
 *
 * When the summarizer intercepts a PutBlob and the inbound tag name matches
 * `tag_re` AND the blob name matches `blob_re`, it sends the blob payload to
 * `model` on the configured `label_endpoint` (see SummarizerConfig) using
 * the prompt template registered in label_prompts_[prompt]. The LLM response
 * is stored alongside the original blob as `{blob_name}_label`.
 *
 * `context_length_` is the per-request token budget passed to Ollama as
 * `options.num_ctx`. It also drives chunking: when the blob payload exceeds
 * the effective byte budget for one prompt, the handler splits the blob into
 * chunks each sized to fit, runs the prompt on every chunk, and concatenates
 * the per-chunk responses into the final summary.
 *
 * Regexes are matched with std::regex_search (so `.*` matches everything;
 * `.*\\.txt` matches a .txt suffix). Globs are not converted.
 */
struct LabelMatch {
  std::string tag_re_;
  std::string blob_re_;
  std::string model_;
  std::string prompt_;  // key into label_prompts_
  // Per-request Ollama context window in tokens. Also drives chunk
  // sizing — see summarizer_runtime.cc::PutBlob. 0 means "use Ollama
  // default" (typically 2048) and disables chunking. A safe production
  // value matches the model's architectural max (e.g. 32768 for
  // gemma3:1b, 131072 for gemma3:4b+).
  int context_length_ = 4096;
  // Hard cap on the LLM response length (Ollama `num_predict`). 0
  // means "no cap" — Ollama generates until EOS or context fills.
  // Setting a value caps each per-chunk summary; with chunking the
  // final concatenated summary is roughly num_predict_ × (#chunks).
  int num_predict_ = 0;

  template <class Archive>
  void serialize(Archive &ar) {
    ar(tag_re_, blob_re_, model_, prompt_, context_length_, num_predict_);
  }
};

/**
 * Container creation params for the summarizer chimod.
 *
 * With no rules configured the module is a pure interposer: every core verb,
 * kPutBlob included, forwards to next_pool_id_ unchanged. The compose keys
 * are the `label_*` ones this configuration used when it lived on
 * clio_cae_core, so an existing deployment only moves them onto the
 * summarizer pool entry.
 */
struct SummarizerConfig {
  // Required: chimod library name for the module manager.
  static constexpr const char *chimod_lib_name = "clio_cae_summarizer";

  // Pool this interposer forwards to — normally the CTE core (512.0).
  // Null falls back to the canonical CTE core pool (CoreInterposer).
  clio::run::PoolId next_pool_id_;

  // Summarization rules. Empty by default — the module behaves as a pure
  // passthrough. See LabelMatch above for matching semantics.
  std::vector<LabelMatch> label_matches_;
  // Named prompt templates referenced by LabelMatch::prompt_. The full LLM
  // input becomes "{prompt}\n\n{blob_text}".
  std::unordered_map<std::string, std::string> label_prompts_;
  // HTTP(S) endpoint of the inference server (Ollama-compatible). The
  // handler POSTs to "{label_endpoint_}/api/generate".
  std::string label_endpoint_;

  SummarizerConfig() : next_pool_id_(clio::run::PoolId::GetNull()) {}

  SummarizerConfig(const SummarizerConfig &other) = default;

  // Compose pool-id ctor (matches the other interposer chimods).
  SummarizerConfig(const clio::run::PoolId &pool_id,
                   const SummarizerConfig &other)
      : next_pool_id_(other.next_pool_id_),
        label_matches_(other.label_matches_),
        label_prompts_(other.label_prompts_),
        label_endpoint_(other.label_endpoint_) {
    (void)pool_id;
  }

  template <class Archive>
  void serialize(Archive &ar) {
    // EVERY field round-trips (the cache/replication silently-dropped-field
    // audit rule).
    ar(next_pool_id_, label_matches_, label_prompts_, label_endpoint_);
  }

  /**
   * Load configuration from compose YAML. Parses:
   *   - `next_pool_id`        ("major.minor")
   *   - `label_endpoint`      (LLM HTTP endpoint base URL)
   *   - `label_prompts`       (map of prompt-name → prompt template)
   *   - `label_matches`       (list of {tag_re, blob_re, model, prompt,
   *                            context_length, num_predict})
   * @param pool_config The compose entry for this pool.
   */
  void LoadConfig(const clio::run::PoolConfig &pool_config) {
    if (pool_config.config_.empty()) return;
    try {
      YAML::Node node = YAML::Load(pool_config.config_);
      if (node["next_pool_id"]) {
        std::string next_str = node["next_pool_id"].as<std::string>();
        auto dot = next_str.find('.');
        if (dot != std::string::npos) {
          clio::run::u32 major = std::stoul(next_str.substr(0, dot));
          clio::run::u32 minor = std::stoul(next_str.substr(dot + 1));
          next_pool_id_ = clio::run::PoolId(major, minor);
        }
      }
      if (node["label_endpoint"]) {
        label_endpoint_ = node["label_endpoint"].as<std::string>();
      }
      if (node["label_prompts"] && node["label_prompts"].IsMap()) {
        for (const auto &kv : node["label_prompts"]) {
          label_prompts_[kv.first.as<std::string>()] =
              kv.second.as<std::string>();
        }
      }
      if (node["label_matches"] && node["label_matches"].IsSequence()) {
        for (const auto &entry : node["label_matches"]) {
          LabelMatch m;
          if (entry["tag_re"]) m.tag_re_ = entry["tag_re"].as<std::string>();
          if (entry["blob_re"]) m.blob_re_ = entry["blob_re"].as<std::string>();
          if (entry["model"]) m.model_ = entry["model"].as<std::string>();
          if (entry["prompt"]) m.prompt_ = entry["prompt"].as<std::string>();
          if (entry["context_length"]) {
            m.context_length_ = entry["context_length"].as<int>();
          }
          if (entry["num_predict"]) {
            m.num_predict_ = entry["num_predict"].as<int>();
          }
          label_matches_.push_back(std::move(m));
        }
      }
    } catch (...) {
      // Config parsing is best-effort
    }
  }
};

/** Standard pool creation. */
using CreateTask = clio::run::admin::GetOrCreatePoolTask<SummarizerConfig>;

/** Cleanup the summarizer container. */
struct DestroyTask : public clio::run::Task {
  DestroyTask() : clio::run::Task() {}

  explicit DestroyTask(const clio::run::TaskId &task_id,
                       const clio::run::PoolId &pool_id,
                       const clio::run::PoolQuery &pool_query)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kDestroy) {}

  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    // OUT fields ONLY -- never Copy() (issue #915): a whole-task assignment
    // destroys this ORIGIN's identity and re-assigns IN shm members across
    // allocator segments. See Task::AggregateOut for the full contract.
    // This task declares no OUT fields, so the base call above (return code +
    // completer) is the entire merge.
  }

  void Copy(const ctp::ipc::FullPtr<DestroyTask> &other) {
    Task::Copy(other.template Cast<clio::run::Task>());
  }

  template <typename Ar> void SerializeIn(Ar &ar) { Task::SerializeIn(ar); }
  template <typename Ar> void SerializeOut(Ar &ar) { Task::SerializeOut(ar); }
};

using MonitorTask = clio::run::admin::MonitorTask;

static_assert(Method::kPutBlob == clio::cte::core::Method::kPutBlob,
              "summarizer kPutBlob must match clio::cte::core::Method::kPutBlob "
              "for transparent CTE→summarizer task dispatch");

}  // namespace clio::cae::summarizer

#endif  // CLIO_CAE_SUMMARIZER_SUMMARIZER_TASKS_H_
