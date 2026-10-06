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

#ifndef CLIO_CAE_SUMMARIZER_SUMMARIZER_RUNTIME_H_
#define CLIO_CAE_SUMMARIZER_SUMMARIZER_RUNTIME_H_

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_interposer.h>
#include <clio_cae/summarizer/summarizer_client.h>
#include <clio_cae/summarizer/summarizer_tasks.h>

namespace clio::cae::summarizer {

/**
 * Summarizer chimod runtime — the transparent LLM summarization that used to
 * be inlined in the CAE core's PutBlob handler. An interposer on the CTE
 * core's task interface (summarizer -> next_pool_id, normally the core):
 *
 *  - PutBlob forwards down `next` FIRST, so the user's write succeeds and
 *    acks with the same return code whatever the model does. Only then, and
 *    only when a configured rule matches the blob's tag and blob names, does
 *    the handler prompt the model and store the response as
 *    `{blob_name}_label` in the same tag. Every summarization failure is
 *    logged and swallowed: it must never flip a PutBlob return code.
 *  - Every OTHER core verb — GetBlob, GetOrCreateTag, SemanticSearch, the
 *    metadata verbs — forwards verbatim through the dispatch defaults in
 *    autogen/summarizer_lib_exec.cc. A clio::cte::core::Client pointed at
 *    this pool therefore works unchanged.
 *  - With no rules configured (the default) the module is a pure
 *    passthrough and costs one extra container hop.
 *
 * The summary blob is written through the CTE client at `next_pool_id`, i.e.
 * BELOW this container, so it never re-enters this handler — there is no
 * summarize-the-summary loop even when a rule's blob_re is `.*`.
 *
 * Inference is blocking (libcurl easy, see label_client.h) and runs on the
 * worker that owns the task. A rule that matches a hot write path will
 * therefore serialize that path behind the model; scope `tag_re`/`blob_re`
 * accordingly.
 */
class Runtime : public clio::cte::core::CoreInterposer {
 public:
  // CreateParams type used by the CLIO_TASK_CC macro for lib_name access.
  using CreateParams = SummarizerConfig;

  Runtime() = default;
  ~Runtime() override = default;

  // ---- Method handlers ----
  clio::run::TaskResume Create(clio::run::shared_ptr<CreateTask> &task);
  clio::run::TaskResume Destroy(clio::run::shared_ptr<DestroyTask> &task);
  clio::run::TaskResume Monitor(clio::run::shared_ptr<MonitorTask> &task);
  clio::run::TaskResume PutBlob(clio::run::shared_ptr<PutBlobTask> &task);

  // ---- Container virtuals (defined in autogen/summarizer_lib_exec.cc) ----
  void Init(const clio::run::PoolId &pool_id, const std::string &pool_name,
            clio::run::u32 container_id = 0) override;
  clio::run::TaskResume Run(
      clio::run::u32 method,
      clio::run::shared_ptr<clio::run::Task> task_ptr) override;
  clio::run::u64 GetWorkRemaining() const override;
  void SaveTask(clio::run::u32 method, clio::run::SaveTaskArchive &archive,
                clio::run::shared_ptr<clio::run::Task> &task_ptr) override;
  void LoadTask(clio::run::u32 method, clio::run::LoadTaskArchive &archive,
                clio::run::shared_ptr<clio::run::Task> &task_ptr) override;
  clio::run::shared_ptr<clio::run::Task> AllocLoadTask(
      clio::run::u32 method, clio::run::LoadTaskArchive &archive) override;
  void LocalLoadTask(clio::run::u32 method,
                     clio::run::DefaultLoadArchive &archive,
                     clio::run::shared_ptr<clio::run::Task> &task_ptr) override;
  clio::run::shared_ptr<clio::run::Task> LocalAllocLoadTask(
      clio::run::u32 method, clio::run::DefaultLoadArchive &archive) override;
  void LocalSaveTask(clio::run::u32 method,
                     clio::run::DefaultSaveArchive &archive,
                     clio::run::shared_ptr<clio::run::Task> &task_ptr) override;
  clio::run::shared_ptr<clio::run::Task> NewCopyTask(
      clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> &orig_task_ptr,
      bool deep) override;
  clio::run::shared_ptr<clio::run::Task> NewTask(clio::run::u32 method) override;
  void AggregateOut(
      clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> &orig_task,
      const clio::run::shared_ptr<clio::run::Task> &replica_task) override;
  void AggregateIn(
      clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> &agg_task,
      const clio::run::shared_ptr<clio::run::Task> &member_task) override;

 private:
  /**
   * Resolve a TagId to its tag name, which the rules match `tag_re` against.
   * PutBlobTask carries only the id. Answers from a local cache, else asks
   * the chain below via GetTagName and memoizes the answer.
   * @param tag_id Tag to resolve.
   * @param name_out Set to the resolved name, or cleared if unresolvable.
   */
  clio::run::TaskResume ResolveTagName(const TagId &tag_id,
                                       std::string *name_out);

  /**
   * Store a generated summary as `{blob_name}_label` in the same tag, via
   * the pool below this one. Failures are logged, not propagated.
   * @param tag_id Tag the source blob lives in.
   * @param blob_name Name of the source blob; "_label" is appended.
   * @param score Placement score to inherit from the source blob.
   * @param text The summary text to store.
   */
  clio::run::TaskResume StoreSummary(const TagId &tag_id,
                                     const std::string &blob_name, float score,
                                     const std::string &text);

  /** Lazily construct the client addressing the pool below. */
  clio::cte::core::Client *GetNextClient();

  SummarizerConfig config_;
  std::unique_ptr<clio::cte::core::Client> next_client_;

  // tag_id → tag_name memo for rule matching. Read and written from PutBlob
  // on any worker, so guarded.
  std::unordered_map<TagId, std::string> tag_names_;
  std::mutex tag_names_mu_;
};

}  // namespace clio::cae::summarizer

#endif  // CLIO_CAE_SUMMARIZER_SUMMARIZER_RUNTIME_H_
