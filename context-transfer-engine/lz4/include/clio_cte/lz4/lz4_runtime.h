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

#ifndef CLIO_CTE_LZ4_LZ4_RUNTIME_H_
#define CLIO_CTE_LZ4_LZ4_RUNTIME_H_

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_interposer.h>
#include <clio_cte/lz4/lz4_client.h>
#include <clio_cte/lz4/lz4_tasks.h>

namespace clio::cte::lz4 {

/**
 * LZ4 page compressor in front of the CTE core (see clio_mod.yaml).
 *
 * PUT: a whole-blob put whose Context asks for kGpuLz4Lib (LZ4) or kGpuAnsLib
 * (interleaved rANS) is encoded on the
 * host (ctp::gpu_lz4::EncodePage) and stored marked kBlobTransformCompressed.
 *
 * GET into DEVICE memory (gpu_vector faults, PodGetBlob/PodMultiGetBlob): every
 * page of the request decodes in ONE kernel launch, straight into the
 * caller's frames. A page whose stored bytes are already in a device tier
 * (hbm) is decoded from the tier in place -- no task, no copy. A page on a
 * host or file tier is first copied into this module's device staging arena
 * by an ordinary core GetBlob, then decoded with the rest.
 *
 * GET into host memory: decoded on the host by the same decoder.
 */
class Runtime : public clio::cte::core::CoreInterposer {
 public:
  using CreateParams = Lz4Config;  // required by CLIO_TASK_CC

  Runtime() = default;
  ~Runtime() override;

  clio::run::TaskStat GetTaskStats(const clio::run::Task *task) const override {
    return clio::cte::core::CoreInterposer::GetTaskStats(task);
  }

  // ---- Method handlers ----
  clio::run::TaskResume Create(clio::run::shared_ptr<CreateTask> &task);
  clio::run::TaskResume Destroy(clio::run::shared_ptr<DestroyTask> &task);
  clio::run::TaskResume Monitor(clio::run::shared_ptr<MonitorTask> &task);
  clio::run::TaskResume PutBlob(
      clio::run::shared_ptr<clio::cte::core::PutBlobTask> &task);
  clio::run::TaskResume GetBlob(
      clio::run::shared_ptr<clio::cte::core::GetBlobTask> &task);
  clio::run::TaskResume GetBlobSize(
      clio::run::shared_ptr<clio::cte::core::GetBlobSizeTask> &task);
  clio::run::TaskResume PodPutBlob(
      clio::run::shared_ptr<clio::cte::core::PodPutBlobTask> &task);
  clio::run::TaskResume PodGetBlob(
      clio::run::shared_ptr<clio::cte::core::PodGetBlobTask> &task);
  clio::run::TaskResume PodMultiGetBlob(
      clio::run::shared_ptr<clio::cte::core::PodMultiGetBlobTask> &task);

  // ---- Container virtuals (defined in autogen/lz4_lib_exec.cc) ----
  void Init(const clio::run::PoolId &pool_id, const std::string &pool_name,
            clio::run::u32 container_id = 0) override;
  clio::run::TaskResume Run(clio::run::u32 method,
                      clio::run::shared_ptr<clio::run::Task> task_ptr) override;
  clio::run::u64 GetWorkRemaining() const override;
  void LocalLoadTask(clio::run::u32 method, clio::run::DefaultLoadArchive &archive,
                     clio::run::shared_ptr<clio::run::Task>& task_ptr) override;
  clio::run::shared_ptr<clio::run::Task> LocalAllocLoadTask(
      clio::run::u32 method, clio::run::DefaultLoadArchive &archive) override;
  void LocalSaveTask(clio::run::u32 method, clio::run::DefaultSaveArchive &archive,
                     clio::run::shared_ptr<clio::run::Task>& task_ptr) override;
  void AggregateOut(clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> &orig_task,
                    const clio::run::shared_ptr<clio::run::Task> &replica_task) override;
  void AggregateIn(clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> &agg_task,
                   const clio::run::shared_ptr<clio::run::Task> &member_task) override;
  void SaveTask(clio::run::u32 method, clio::run::SaveTaskArchive &archive,
                clio::run::shared_ptr<clio::run::Task>& task_ptr) override;
  void LoadTask(clio::run::u32 method, clio::run::LoadTaskArchive &archive,
                clio::run::shared_ptr<clio::run::Task>& task_ptr) override;
  clio::run::shared_ptr<clio::run::Task> AllocLoadTask(clio::run::u32 method,
                                             clio::run::LoadTaskArchive &archive) override;
  clio::run::shared_ptr<clio::run::Task> NewCopyTask(clio::run::u32 method,
                                           clio::run::shared_ptr<clio::run::Task> &orig,
                                           bool deep) override;
  clio::run::shared_ptr<clio::run::Task> NewTask(clio::run::u32 method) override;

  /** One record of a device-destination read (internal; public for the
   *  helpers in lz4_runtime.cc). */
  struct DevRead {
    std::string name;   /**< normalized blob name */
    clio::run::u64 offset = 0;
    clio::run::u64 size = 0;
    ctp::ipc::ShmPtr<> data;  /**< the caller's destination */
    char *dst = nullptr;      /**< resolved device pointer */
    clio::run::u32 rc = 0;    /**< OUT: 0 = served */
  };

 private:
  /** @return true when a context asks this module to compress. */
  static bool WantsLz4(const Context &ctx);

  /** Encode `n` host bytes and put them to the next pool, marked
   *  compressed; a page that does not shrink is stored raw. */
  clio::run::TaskResume EncodeAndPut(const TagId &tag_id,
                                     const std::string &name, const char *src,
                                     clio::run::u64 n, float score,
                                     Context ctx, clio::run::u32 flags,
                                     clio::run::u32 &rc, Context &out_ctx);

  /** Decode every record into its device destination (one launch). */
  clio::run::TaskResume ReadToDevice(const TagId &tag_id, const Context &ctx,
                                     clio::run::u32 flags,
                                     std::vector<DevRead> &recs);

  /** Copy the stored bytes of `recs[idx]` into the staging arena. */
  clio::run::TaskResume StageRecords(const TagId &tag_id, const Context &ctx,
                                     std::vector<DevRead> &recs,
                                     const std::vector<size_t> &idx,
                                     const std::vector<clio::run::u64> &stored,
                                     std::vector<const char *> &src,
                                     clio::run::u64 &arena_off,
                                     clio::run::u64 &arena_len);

  /** Serve one record the slow way: read the stored bytes to the host,
   *  decode there (or accept them raw), copy into the destination. */
  clio::run::TaskResume ReadViaHost(const TagId &tag_id, const Context &ctx,
                                    clio::run::u32 flags, DevRead &rec);

  /** Launch the decode of `njobs` jobs (pinned job array) and wait for it. */
  clio::run::TaskResume LaunchAndWait(void *jobs, clio::run::u32 njobs,
                                      bool &launched);

  /** Device staging arena: first-fit ranges, 512-byte aligned. */
  clio::run::u64 ArenaAcquire(clio::run::u64 bytes);
  void ArenaRelease(clio::run::u64 off, clio::run::u64 bytes);

  /** Pinned DecodeJob arrays of kPodMultiMax entries, pooled. */
  void *AcquireJobs();
  void ReleaseJobs(void *jobs);

  clio::cte::core::Client *GetNextClient();

  Lz4Config config_;
  std::unique_ptr<clio::cte::core::Client> next_client_;

  std::mutex arena_mu_;
  char *arena_ = nullptr;                             /**< device memory */
  clio::run::u64 arena_bytes_ = 0;
  std::map<clio::run::u64, clio::run::u64> arena_free_;  /**< off -> len */

  /** Path counters (CLIO_LZ4_STATS=1 logs them): which route pages take. */
  struct Stats {
    std::atomic<clio::run::u64> encoded{0};     /**< puts stored encoded */
    std::atomic<clio::run::u64> put_raw{0};     /**< puts that did not shrink */
    std::atomic<clio::run::u64> raw_in{0};      /**< raw bytes of encoded puts */
    std::atomic<clio::run::u64> enc_out{0};     /**< their encoded bytes */
    std::atomic<clio::run::u64> dev_tier{0};    /**< decoded from a device tier */
    std::atomic<clio::run::u64> dev_staged{0};  /**< staged, then GPU-decoded */
    std::atomic<clio::run::u64> host{0};        /**< served via the host path */
  } stats_;
  /** Log the counters every `every` device reads when CLIO_LZ4_STATS is set. */
  void MaybeLogStats(clio::run::u64 every);

  std::mutex jobs_mu_;
  std::vector<void *> jobs_free_;  /**< pinned DecodeJob arrays */
  std::vector<void *> jobs_all_;
};

}  // namespace clio::cte::lz4

#endif  // CLIO_CTE_LZ4_LZ4_RUNTIME_H_
