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

#ifndef CLIO_CTE_DTSCHEDULE_DTSCHEDULE_RUNTIME_H_
#define CLIO_CTE_DTSCHEDULE_DTSCHEDULE_RUNTIME_H_

#include <memory>
#include <atomic>
#include <mutex>
#include <fstream>
#include <chrono>
#include <deque>
#include <optional>
#include <unordered_map>
#include <vector>

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_interposer.h>
#include <clio_cte/compressor/compression_header.h>
#include <clio_cte/dtschedule/dtschedule_tasks.h>
#include <clio_cte/dtschedule/dtschedule_client.h>
#include <clio_cte/dtschedule/ccm/ccm_manager.h>
#include <clio_cte/dtschedule/dag_spec.h>

namespace clio::cte::dtschedule {

/**
 * DTSchedule chimod runtime (phase 1: skeleton + fixed codec).
 *
 * The dtschedule module INTERPOSES on the CTE core's data verbs (PutBlob,
 * GetBlob, GetBlobSize, MultiPutBlob) to add compression, tiering, and
 * workflow-aware placement decisions. In phase 1, it supports fixed-codec
 * compression and basic tracing.
 *
 * Callers can point a clio::cte::core::Client at this pool (or set
 * CLIO_CTE_POOL=565.0) and every operation keeps working. PutBlob/GetBlob
 * carry the CORE's task structs and method ids. Other core methods are
 * forwarded untouched to the next pool.
 *
 * @see DESIGN.md for architecture and decision logic (phases 2-5)
 */
/** Result of one compression attempt inside PutBlob. */
struct CompressOutcome {
  bool used = false;        ///< compressed bytes were forwarded
  bool attempted = false;   ///< a codec ran on the data
  double ctime_ms = 0.0;    ///< measured compression time (ms)
  double ratio = 1.0;       ///< original / compressed (1 when not attempted)
  size_t comp_size = 0;     ///< codec bytes (without header)
};

/** Where a put's data goes relative to the nodes involved (trace + cost). */
struct Placement {
  uint32_t consumer_node = UINT32_MAX;  ///< tracked consumer, if any
  double consumer_cpu = -1.0;           ///< its last CPU sample (-1 = unknown)
  uint32_t owner_node = UINT32_MAX;     ///< hash owner of the blob
  std::string tier;                     ///< chosen tier name
  float tier_score = -1.0f;             ///< its device score (-1 = unset)
  double tier_bw_mb_ms = 0.0;           ///< its bandwidth (MB/ms)
  std::string order;                    ///< decision order used
  int scenario = 1;                     ///< 1, 2, or 3
  double cost1_ms = -1.0;               ///< S1 cost estimate
  double cost2_ms = -1.0;               ///< S2 cost estimate
  double cost3_ms = -1.0;               ///< S3 cost estimate (Phase 4c)
  double store_ms = -1.0;               ///< consumer-copy wall time
  bool dag_hit = false;                 ///< consumer came from the DAG spec
  int fanout = -1;                      ///< fan-out copies attempted (-1 = none)
  std::vector<uint32_t> dag_consumers;  ///< all consumer nodes from the DAG
};

class Runtime : public clio::cte::core::CoreInterposer {
 public:
  using CreateParams = DtscheduleConfig;  // required by CLIO_TASK_CC

  Runtime() = default;
  ~Runtime() override;

  /**
   * Route data verbs with Dynamic queries to the submitter's node.
   *
   * Phase 4a: Override ScheduleTask to route Dynamic PutBlob/GetBlob/
   * GetBlobSize/MultiPutBlob to Local() so they run on the submitter's node.
   * All other tasks use the default CoreInterposer routing.
   *
   * @param task Task to route
   * @return Local() for Dynamic data verbs, otherwise default routing
   */
  clio::run::PoolQuery ScheduleTask(
      const clio::run::shared_ptr<clio::run::Task> &task) override;

  /**
   * Per-task cost estimate for the scheduler (see Container::GetTaskStats).
   *
   * compute_ is the feature Container::InferCpuTime multiplies its learned
   * per-method coefficient by. wall_time_ seeds InferWallClockTime at the
   * ~500 MB/s house convention. Both coefficients are then learned from real
   * completions.
   */
  clio::run::TaskStat GetTaskStats(const clio::run::Task *task) const override;

  // ---- Method handlers for core data verbs ----

  /**
   * Create: initialize module state.
   *
   * Parses configuration, loads tier tables from core's ListTargets,
   * opens trace file if configured. Inherits next_pool_id_ for base routing.
   */
  clio::run::TaskResume Create(clio::run::shared_ptr<CreateTask> &task);

  /**
   * Destroy: cleanup module state.
   *
   * Flushes and closes trace file, cleans up resources.
   */
  clio::run::TaskResume Destroy(clio::run::shared_ptr<DestroyTask> &task);

  /**
   * Monitor: periodic housekeeping (refresh target table, sample load).
   *
   * In phase 1: just forwards to core's Monitor to keep target table current.
   */
  clio::run::TaskResume Monitor(clio::run::shared_ptr<MonitorTask> &task);

  /**
   * PutBlob (interposed core verb): compression + storage decision.
   *
   * Phase 1: if size >= min_compress_bytes and ccm allows compression,
   * compress with fixed codec, prepend CompressionHeader, and forward to
   * next pool (core). Otherwise forward raw.
   *
   * Records decision and timing in trace CSV (if enabled).
   */
  clio::run::TaskResume PutBlob(
      clio::run::shared_ptr<clio::cte::core::PutBlobTask> &task);

  /**
   * GetBlob (interposed core verb): decompression on read.
   *
   * Phase 1: forward to next pool (core), then if blob is compressed,
   * decompress in-place into caller's buffer. Record observed decompress
   * time in trace.
   */
  clio::run::TaskResume GetBlob(
      clio::run::shared_ptr<clio::cte::core::GetBlobTask> &task);

  /**
   * GetBlobSize (interposed core verb): return uncompressed size.
   *
   * Phase 1: forward to next pool (core). If blob is compressed, return
   * the uncompressed size from CompressionHeader, otherwise return the
   * stored size.
   */
  clio::run::TaskResume GetBlobSize(
      clio::run::shared_ptr<clio::cte::core::GetBlobSizeTask> &task);

  /**
   * MultiPutBlob (interposed core verb): batched put with compression.
   *
   * Phase 1: forward untouched to next pool. Phase 3+ will apply
   * compression to each record in the batch.
   */
  clio::run::TaskResume MultiPutBlob(
      clio::run::shared_ptr<clio::cte::core::MultiPutBlobTask> &task);

  // ---- Method handlers for dtschedule-specific verbs ----

  /**
   * PollNodeLoad: sample local node CPU% and worker queue stats.
   *
   * Phase 1: real implementation using ctp::SystemInfo::GetCpuTimes.
   * Used by phases 3+ for load-aware decisions.
   */
  clio::run::TaskResume PollNodeLoad(
      clio::run::shared_ptr<PollNodeLoadTask> &task);

  /**
   * RegisterConsumer: record (tag, consumer_node) mapping at tag owner.
   *
   * Phase 1: stub; stores nothing. Phase 4 will use this to pick scenario 2/3.
   */
  clio::run::TaskResume RegisterConsumer(
      clio::run::shared_ptr<RegisterConsumerTask> &task);

  /**
   * CompressAt: ship raw blob to node N for scenario 3 compression.
   *
   * Phase 1: returns kEnosys. Phase 4 will implement remote compression.
   */
  clio::run::TaskResume CompressAt(
      clio::run::shared_ptr<CompressAtTask> &task);

  /**
   * GetDecisionStats: return counters for tests and trace analysis.
   *
   * Phase 1: real implementation returning aggregated decision stats.
   */
  clio::run::TaskResume GetDecisionStats(
      clio::run::shared_ptr<GetDecisionStatsTask> &task);

  /**
   * SetKnobs: change ablation knobs at runtime without restart.
   *
   * Phase 1: updates knobs_. Used by load-response experiments.
   */
  clio::run::TaskResume SetKnobs(
      clio::run::shared_ptr<SetKnobsTask> &task);

  /**
   * SampleLoad: periodic self-sampling of local node CPU% and worker load.
   *
   * Phase 3: samples CPU utilization and queued task counts, stores in ring
   * buffer for load-aware decision making.
   */
  clio::run::TaskResume SampleLoad(
      clio::run::shared_ptr<SampleLoadTask> &task);

  // ---- Container virtuals (defined in autogen/dtschedule_lib_exec.cc) ----
  void Init(const clio::run::PoolId &pool_id, const std::string &pool_name,
            clio::run::u32 container_id = 0) override;
  clio::run::TaskResume Run(clio::run::u32 method,
                            clio::run::shared_ptr<clio::run::Task> task_ptr) override;
  clio::run::u64 GetWorkRemaining() const override;
  void LocalLoadTask(clio::run::u32 method, clio::run::DefaultLoadArchive &archive,
                     clio::run::shared_ptr<clio::run::Task> &task_ptr) override;
  clio::run::shared_ptr<clio::run::Task> LocalAllocLoadTask(
      clio::run::u32 method, clio::run::DefaultLoadArchive &archive) override;
  void LocalSaveTask(clio::run::u32 method, clio::run::DefaultSaveArchive &archive,
                     clio::run::shared_ptr<clio::run::Task> &task_ptr) override;
  void SaveTask(clio::run::u32 method, clio::run::SaveTaskArchive &archive,
                clio::run::shared_ptr<clio::run::Task> &task_ptr) override;
  void LoadTask(clio::run::u32 method, clio::run::LoadTaskArchive &archive,
                clio::run::shared_ptr<clio::run::Task> &task_ptr) override;
  clio::run::shared_ptr<clio::run::Task> AllocLoadTask(
      clio::run::u32 method, clio::run::LoadTaskArchive &archive) override;
  clio::run::shared_ptr<clio::run::Task> NewCopyTask(
      clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> &orig,
      bool deep) override;
  void AggregateOut(clio::run::u32 method,
                    clio::run::shared_ptr<clio::run::Task> &orig_task,
                    const clio::run::shared_ptr<clio::run::Task> &replica_task) override;
  void AggregateIn(clio::run::u32 method,
                   clio::run::shared_ptr<clio::run::Task> &agg_task,
                   const clio::run::shared_ptr<clio::run::Task> &member_task) override;
  clio::run::shared_ptr<clio::run::Task> NewTask(clio::run::u32 method) override;

 protected:
  /** Lazily construct the client to the next pool in the chain. */
  void EnsureCoreClient();
  /** Decompress header+codec bytes into a fresh SHM buffer (null on error). */
  ctp::ipc::FullPtr<char> DecompressStored(const char *stored,
                                           clio::run::u64 stored_size,
                                           clio::run::u64 *out_size,
                                           std::string *lib_name);
  /** Slice each requested region of a GetBlob out of the original bytes. */
  bool CopyRegionsFromOriginal(clio::cte::core::GetBlobTask &task,
                               const char *original, clio::run::u64 out_size);
  /** Append a decompression observation row to the trace. */
  void WriteDecompressTraceRow(const std::string &tag, const std::string &blob,
                               clio::run::u64 original_size,
                               const std::string &lib, double dtime_ms);
  DtscheduleConfig config_;              ///< Parsed configuration
  DecisionStats stats_;                  ///< Aggregated decision counters
  Knobs knobs_;                          ///< Ablation knobs (mutable)
  std::mutex stats_lock_;                ///< Protect stats_ updates
  std::mutex knobs_lock_;                ///< Protect knobs_ updates
  std::unique_ptr<ccm::CcmManager> ccm_manager_;  ///< Codec Characteristic Model manager

  std::unique_ptr<clio::cte::core::Client> core_client_;  ///< Client to cache/core pool
  std::unique_ptr<clio::cte::core::Client> compressed_client_;  ///< Client for compressed puts (phase 3+)
  std::unique_ptr<clio::cte::core::Client> copy_client_;  ///< core pool, consumer copies
  std::unique_ptr<Client> dtschedule_client_;  ///< Client to dtschedule pool for S3 CompressAt (phase 4c)
  std::ofstream trace_file_;             ///< CSV trace file (if enabled)
  std::mutex trace_lock_;                ///< Protect concurrent trace writes

  // Phase 3+: load tracking (per-node ring buffers of recent samples)
  struct LoadRing {
    std::deque<NodeLoadSample> samples_;   ///< Last 16 samples per node
    std::mutex lock_;
  };
  std::unordered_map<uint32_t, LoadRing> load_rings_;  ///< node_id -> LoadRing
  std::mutex load_rings_lock_;
  ctp::CpuTimes prev_cpu_times_;                       ///< For delta calculation

  // Phase 4+: consumer tracking (per-tag, keep last 32 nodes, most recent last)
  struct ConsumerNodes {
    std::deque<uint32_t> nodes_;  ///< Last 32 consumer nodes for this tag
    std::mutex lock_;
  };
  std::unordered_map<std::string, ConsumerNodes> tag_consumers_;  // tag -> nodes
  std::mutex consumer_lock_;

  // Tag id -> resolved tag name (adapter traffic carries ids; the DAG and
  // the QoS stage regexes are keyed by file name). "" caches a miss.
  std::unordered_map<std::string, std::string> tag_names_;
  std::mutex tag_names_lock_;
  /** Cached tag name for tag_key, or nullopt when never resolved. */
  std::optional<std::string> CachedTagName(const std::string &tag_key);
  /** Remember the resolved name of a tag. */
  void CacheTagName(const std::string &tag_key, const std::string &name);
  /** True when the DAG spec or a QoS stage override needs file names. */
  bool NeedsTagName() const;
  /** True when blob_name is a bare chunk index (filesystem traffic). */
  static bool IsChunkIndexName(const std::string &blob_name);

  // Phase 5+: workflow-aware DAG spec loader
  std::unique_ptr<DagSpecLoader> dag_spec_;  ///< DAG spec for workflow-aware placement
  uint32_t dag_files_ = 0;                    ///< Number of files in DAG spec
  uint32_t dag_nodes_ = 0;                    ///< Number of unique nodes in DAG spec
  uint64_t copies_pushed_ = 0;                ///< Successful fan-out copies pushed
  uint64_t copy_refused_ = 0;                 ///< Fan-out copies refused (rc != 0)
  uint64_t copy_skipped_local_ = 0;           ///< Fan-out copies skipped (local/owner)
  std::mutex dag_lock_;                       ///< Protect DAG-related counters

  /**
   * Store a load sample for the given node in its ring buffer (keeps last 16).
   *
   * @param node_id Node identifier
   * @param sample Load sample to store
   */
  void StoreLoadSample(uint32_t node_id, const NodeLoadSample &sample);

  /**
   * Get the latest load sample for a node, or default if unknown/stale.
   *
   * Returns {cpu_pct, age_ms, known} where:
   * - cpu_pct: CPU utilization (0-100%), clamped at 1.0 if unknown
   * - age_ms: milliseconds since sample was taken
   * - known: true if sample is fresh (age <= 5*load_period_ms)
   *
   * @param node_id Node to query
   * @return tuple of {cpu_pct, age_ms, known}
   */
  std::tuple<double, uint64_t, bool> GetLoad(uint32_t node_id);

  /**
   * Compute load multiplier for codec selection.
   *
   * m_X = clamp(cpu_X / max(cpu_other, 1), 1/load_cap, load_cap)
   * Both clamped at 1.0 when load_aware is off or either sample is unknown/stale.
   *
   * @param node_a First node (e.g., producer)
   * @param node_b Second node (e.g., consumer)
   * @return Load multiplier to apply to compress+store terms
   */
 public:
  static double LoadMultiplier(double cpu_a, double cpu_b,
                               bool load_aware, double load_cap);

 protected:

  /**
   * Check if this container has already registered as a consumer for a tag.
   *
   * Phase 4: used in GetBlob to deduplicate RegisterConsumer sends.
   * Returns true if the tag is already in tag_consumers_.
   *
   * @param tag_id Tag to check
   * @return true if consumer registration already exists
   */
  bool IsConsumerRegistered(const TagId &tag_id);

  /**
   * Get the most recent consumer node for a tag, or UINT32_MAX if unknown.
   *
   * Phase 4: used by decision logic to pick the consumer node for scenario 2/3.
   * Returns the most recently registered node (last in the deque).
   *
   * @param tag_id Tag to query
   * @return Most recent consumer node ID, or UINT32_MAX if unknown
   */
  uint32_t PickConsumer(const TagId &tag_id);

  /**
   * Pick a consumer node from DAG spec for a blob.
   *
   * Phase 5: when workflow_aware == "dag" and dag_spec is loaded,
   * returns the first consumer node from the spec that is not the
   * producer node (or UINT32_MAX if no consumers found or DAG not loaded).
   * Logs "dag:1" in the forced column.
   *
   * @param tag_or_blob_name File name to look up in DAG spec
   * @param producer_node The producer node (local for PutBlob)
   * @param[out] forced_reason Set to "dag:1" for trace recording
   * @return Consumer node ID or UINT32_MAX if not found/no consumers
   */
  uint32_t PickConsumerFromDag(const std::string &tag_or_blob_name,
                               uint32_t producer_node,
                               std::string &forced_reason);

  /**
   * Push fan-out copies to multiple consumer nodes (Phase 5).
   *
   * When a blob has >= replicate_fanout_min consumer nodes, loop through
   * and push copies (up to replicate_max) to each consumer, excluding the
   * producer and owner. Uses PushConsumerCopy internally and counts
   * copies_pushed_/copy_refused_/copy_skipped_local_ per node.
   *
   * @param tag_id Tag being stored
   * @param blob_name Blob name
   * @param raw_data Pointer to original (uncompressed) blob data
   * @param size Size of original data (bytes)
   * @param consumer_nodes Set of consumer nodes from DAG spec
   * @param producer_node Producer node (excluded)
   * @param owner_node Hash owner (excluded)
   * @param tier_name Tier for replica copies
   * @param version Version for cache replica registration
   * @param place Placement info (optional, for stats)
   * @return Number of copies successfully pushed
   */
  /** Push raw copies to every DAG consumer node (fan-out replication). */
  clio::run::TaskResume PushFanoutCopies(const TagId &tag_id,
                                         const std::string &blob_name,
                                         ctp::ipc::ShmPtr<> raw_data,
                                         uint64_t size, uint32_t owner_node,
                                         uint32_t already_covered,
                                         uint64_t version, Placement *place);

  /**
   * Initialize trace file with header row (if trace_path configured).
   *
   * Header format (DESIGN.md §9): ts_ms, node, tag, blob, size, ...
   */
  void OpenTraceFile();

  /**
   * Close and flush trace file.
   */
  void CloseTraceFile();

  /**
   * Write one CSV row to trace file (decision record).
   *
   * Phase 2: records all decision fields including predictions and statistics.
   * Phase 3+: includes producer_cpu and load_mult in trace columns.
   *
   * chosen_lib is "raw" when the blob was stored uncompressed.
   * load_mult is appended to forced column as lm:<value>.
   */
  void WriteTraceRow(const std::string &tag_id, const std::string &blob_name,
                     size_t original_size, const ccm::Features &features,
                     const ccm::Decision &decision, const CompressOutcome &out,
                     double select_ms, double load_mult, double producer_cpu,
                     const Placement &place);

  /**
   * Write candidate evaluation records to .cand file (task 3).
   *
   * For each candidate in the decision, writes one row per candidate to a
   * parallel CSV file with suffix .cand.NODE.csv containing:
   * ts_ms,blob,lib,preset,pred_ctime_ms,pred_dtime_ms,pred_ratio,cost,reason
   */
  void WriteCandidatesTrace(const std::string &tag_id,
                            const std::string &blob_name,
                            const ccm::Decision &decision);
  /** True when a put of this size goes through codec selection. */
  bool ShouldSelect(size_t size) const;
  /** Whether the CCM may compress at all (ccm is not fixed:none). */
  bool CompressionEnabled() const;
  /**
   * Compress a blob with the decision's codec into a fresh SHM buffer with
   * the shared header in front. Returns a null pointer (and out->used ==
   * false) when the codec is unavailable, fails, or does not reach
   * kMinUsefulRatio; the measured time and ratio are still reported.
   */
  ctp::ipc::FullPtr<char> CompressWithDecision(const char *src, size_t size,
                                               const ccm::Decision &decision,
                                               CompressOutcome *out);
  /** Forward a compressed put below the cache (hash-routed to the owner). */
  /** True when this node is the blob's hash owner. */
  bool OwnerIsLocal(const TagId &tag_id, const std::string &blob_name);
  /** Raw put: local core when this node owns the blob, else the owner. */
  clio::run::TaskResume ForwardRawPut(
      clio::run::shared_ptr<clio::cte::core::PutBlobTask> &task);
  /** Raw get: local core when this node owns the blob, else the owner. */
  clio::run::TaskResume ForwardRawGet(
      clio::run::shared_ptr<clio::cte::core::GetBlobTask> &task);
  clio::run::TaskResume ForwardCompressedPut(
      clio::run::shared_ptr<clio::cte::core::PutBlobTask> &task);
  /** Scenario 3: ship the raw bytes to the consumer and let it compress. */
  clio::run::TaskResume CompressAtConsumer(
      clio::run::shared_ptr<clio::cte::core::PutBlobTask> &task,
      const ccm::Decision &decision, size_t original_size,
      CompressOutcome *out, Placement *place);
  /** Feed the observation back to the CCM and write the trace rows. */
  /**
   * Feed observation back to CCM and record decision in trace.
   *
   * Phase 3+: includes load_mult and producer_cpu for trace recording.
   */
  void RecordDecision(const std::string &tag_id, const std::string &blob_name,
                      size_t original_size, const char *src,
                      const ccm::Decision &decision, const CompressOutcome &out,
                      double select_ms, double load_mult, double producer_cpu,
                      const Placement &place);
  /** Node that owns (tag, blob) by the core's hash; self when unknown. */
  uint32_t OwnerNode(const TagId &tag_id, const std::string &blob_name);

  /**
   * Open the candidates trace file (task 3).
   */
  void OpenCandidatesTraceFile();

  // Phase 4b: Tier selection and scenario selection

  /**
   * Optimistic per-node tier reservations: bytes this container has sent to
   * each tier of each owner node since that node's last load sample (the
   * sample's free bytes are stale until the next one arrives). Reset when a
   * new sample for the node is stored.
   */
  std::unordered_map<uint32_t, std::vector<uint64_t>> tier_reserved_;
  std::mutex tier_cap_lock_;

  /**
   * Read this node's free bytes per configured tier from the core's local
   * targets (ListTargets + GetTargetInfo, Local). A target belongs to the
   * configured tier whose score is closest to the target's score.
   *
   * @param out Free bytes per tier, in config_.tiers_ order
   */
  clio::run::TaskResume CollectTierRemaining(std::vector<uint64_t> *out);
  // ---- dtschedule-owned placement (DtscheduleConfig::placement_) ----
  TagId loc_tag_;                          ///< Tag holding location records
  bool loc_tag_ready_ = false;             ///< loc_tag_ resolved
  /** A codec decision shared by the next chunks of one tag. */
  struct CachedDecision {
    ccm::Decision decision;   ///< Decision without its candidate records
    int uses_left = 0;        ///< Chunks that may still reuse it
    uint64_t made_ms = 0;     ///< When it was made (steady clock, ms)
  };
  std::unordered_map<std::string, CachedDecision> decision_cache_;
  std::mutex decision_cache_lock_;         ///< Guards decision_cache_
  /**
   * Codec decision for a chunk: reuse the tag's recent decision when one is
   * fresh, else select (features + ranking) and cache the result.
   *
   * @param tag_id Tag of the chunk (the file under the POSIX adapter)
   * @param src Chunk bytes
   * @param size Chunk size
   * @param match_name Name the QoS rules and candidate filter match
   * @param load_mult Load multiplier for the ranker
   * @param rank_bw_mb_ms Store bandwidth for the ranker
   * @param store_bw Per-size store bandwidth for the ranker
   * @param reused Out: true when a cached decision was returned
   * @return The decision
   */
  ccm::Decision DecideCodec(const TagId &tag_id, const char *src, size_t size,
                            const std::string &match_name, double load_mult,
                            double rank_bw_mb_ms,
                            const std::function<double(uint64_t)> *store_bw,
                            bool *reused);
  std::unordered_map<std::string, uint32_t> loc_cache_;  ///< blob -> node
  std::mutex loc_lock_;                    ///< Guards loc_cache_
  /** True when dtschedule (not the core's hash) places data chunks. */
  bool PlacesData() const {
    return config_.placement_ == "dtschedule" || config_.placement_ == "local";
  }
  /** True when dtschedule places this blob: placement mode on and the
   *  name is a chunk index (clio-fs data pages; metadata keeps hash routing). */
  bool IsPlaced(const std::string &blob) const {
    return PlacesData() && IsChunkIndexName(blob);
  }
  /** Decode a compressed read from the bytes the first read returned. */
  bool TryDecompressInPlace(clio::cte::core::GetBlobTask &task);
  /** Cheapest node to store bytes on (tier + network); reserves the tier. */
  uint32_t BestStoreNode(uint64_t bytes);
  /** Query that runs on node (Local when it is this node). */
  clio::run::PoolQuery NodeQuery(uint32_t node) const;
  /** Key of a blob in the location cache / record tag. */
  static std::string LocKey(const TagId &tag_id, const std::string &blob);
  /** Resolve (create once) the `_dtschedule_loc` tag. */
  clio::run::TaskResume EnsureLocTag();
  /** Record that blob lives on node (no record when node is its hash
   *  owner and no other location was recorded). */
  clio::run::TaskResume WriteLocation(const TagId &tag_id,
                                      const std::string &blob, uint32_t node);
  /** Node holding blob: cached or recorded location, else its hash owner. */
  clio::run::TaskResume ResolveLocation(const TagId &tag_id,
                                        const std::string &blob,
                                        uint32_t *node);
  /** Put the task's bytes (all regions) on node through the core pool. */
  clio::run::TaskResume PutAtNode(
      clio::run::shared_ptr<clio::cte::core::PutBlobTask> &task,
      uint32_t node);
  /** Get the task's regions from node through the core pool. */
  clio::run::TaskResume GetAtNode(
      clio::run::shared_ptr<clio::cte::core::GetBlobTask> &task,
      uint32_t node);

  // ---- Background compress-on-demote (DtscheduleConfig::demote_*) ----
  /** A chunk this node just read, kept for possible demotion. */
  struct DemoteItem {
    TagId tag_id_;              ///< Tag of the blob
    std::string blob_;          ///< Blob name
    uint32_t owner_ = 0;        ///< Node that owns the blob
    std::vector<char> data_;    ///< Raw bytes as read
  };
  std::deque<DemoteItem> demote_q_;        ///< FIFO of read chunks
  size_t demote_q_bytes_ = 0;              ///< Bytes held in demote_q_
  std::mutex demote_lock_;                 ///< Guards demote_q_
  std::unordered_map<uint32_t, uint64_t> tier0_capacity_;  ///< Max fastest-
                                           ///< tier free bytes seen per node
  uint64_t demoted_ = 0;                   ///< Chunks demoted compressed
  uint64_t demote_in_bytes_ = 0;           ///< Raw bytes demoted
  uint64_t demote_out_bytes_ = 0;          ///< Compressed bytes written

  /** Queue a chunk this node read (bounded; oldest entries are dropped). */
  void EnqueueDemote(const TagId &tag_id, const std::string &blob,
                     uint32_t node, const char *data, size_t size);
  /** True while the owner's fastest tier has less free space than the
   *  demote watermark (fraction of the largest free space seen there). */
  bool OwnerUnderPressure(uint32_t owner);
  /** Pop the oldest queued chunk whose owner is under pressure. */
  bool PopDemoteCandidate(DemoteItem *item);
  /** Highest-score tier below the fastest with room for bytes on owner. */
  std::string ChooseLowerTier(uint64_t bytes, uint32_t owner);
  /** Demote queued chunks within the per-tick byte budget, in batches. */
  clio::run::TaskResume DemoteTick();
  /** Compress one chunk for demotion; false when no codec is useful. */
  bool CompressForDemote(DemoteItem *item, ctp::ipc::FullPtr<char> *comp,
                         size_t *stored, std::string *tier);
  /** Count a demoted chunk (and log periodically). */
  void RecordDemotion(size_t raw_bytes, size_t stored_bytes,
                      const std::string &tier);

  /** False for another node's device registered here as a neighbour. */
  bool IsOwnTarget(const std::string &target_name) const;
  /** Configured tier index for a core target (name match, else score). */
  size_t TierIndexForTarget(const std::string &target_name,
                            float target_score) const;
  /** Log per-tier free bytes when they change by more than 1 GiB. */
  void LogTierRemaining(const std::vector<uint64_t> &free_bytes);
  std::vector<uint64_t> logged_tier_free_;  ///< Last logged free bytes

  /**
   * Choose the tier a blob of blob_size bytes will land in on owner node.
   *
   * Highest-score configured tier whose free bytes on that node (latest
   * load sample minus this container's reservations since) can hold the
   * blob, else the lowest tier. Without a sample for the node, the fastest
   * tier. DESIGN.md §7 tier selection.
   *
   * @param blob_size Bytes to store
   * @param owner_node Node that will store them
   * @param reserve Also reserve the bytes on the chosen tier
   * @return Chosen tier name, or empty string if no tiers are configured
   */
  std::string ChooseTier(uint64_t blob_size, uint32_t owner_node,
                         bool reserve = false);

  /**
   * Store bandwidth (MB/ms) for writing bytes to owner_node: the bandwidth
   * of the tier those bytes would land in, capped by the network when the
   * owner or a known consumer is another node.
   *
   * @param bytes Bytes to store (raw or compressed size)
   * @param owner_node Node that owns the blob
   * @param consumer_node Known consumer, or UINT32_MAX
   * @return Bandwidth in MB/ms
   */
  double StoreBwFor(uint64_t bytes, uint32_t owner_node,
                    uint32_t consumer_node);

  /**
   * Scenario selection result.
   */
  struct ScenarioChoice {
    int chosen_scenario = 1;   ///< 1 (S1), 2 (S2), or 3 (S3)
    double cost1_ms = 0.0;     ///< S1 cost (ms)
    double cost2_ms = 0.0;     ///< S2 cost (ms)
    double cost3_ms = 0.0;     ///< S3 cost (ms)
  };

  /**
   * Select scenario 1, 2, or 3 based on cost model.
   *
   * Computes:
   * - cost1 = T_c·m_P + net(z, P→O) + net(z, O→C) + T_d(C) + tier_cost1
   * - cost2 = T_c·m_P + net(z, P→O) + net(raw, P→C) + tier_cost2
   * - cost3 = net(raw, P→C) + T_c·m_C + net(z, C→O) + tier_cost3
   *
   * Where z = size / pred_ratio (compressed size), m_C is the load multiplier
   * for consumer C (when known, else 1), and tier costs are computed as
   * z / tier_bw_mbps (or raw / tier_bw_mbps for S2/S3).
   *
   * Picks minimum unless force_scenario overrides (1, 2, or 3).
   * S3 requires a known consumer (workflow_aware != "none" and place.consumer_node != UINT32_MAX).
   *
   * DESIGN.md §3: "The minimum-cost scenario wins; force_scenario overrides".
   *
   * @param size Original blob size (bytes)
   * @param pred_ratio Predicted compression ratio (original/compressed)
   * @param pred_ctime_ms Predicted compression time
   * @param pred_dtime_ms Predicted decompression time
   * @param load_mult Load multiplier for producer
   * @param place Placement info (producer, consumer, owner nodes)
   * @return Scenario choice (scenario 1/2/3, with all costs)
   */
  ScenarioChoice SelectScenario(uint64_t size, double pred_ratio,
                                double pred_ctime_ms, double pred_dtime_ms,
                                double load_mult, const Placement &place);

  /**
   * Push a consumer copy to node C for scenario 2.
   *
   * Writes raw blob bytes to the core pool with Physical(C) query and
   * kCacheReplica replica flags, then registers with the owner via
   * RegisterReplicaContainer. On success counts copies_pushed_; on
   * error counts copy_refused_ and logs the rc.
   *
   * @param tag_id Tag being stored
   * @param blob_name Blob name
   * @param src Pointer to original (uncompressed) blob data
   * @param size Size of original data (bytes)
   * @param consumer_node Target consumer node C
   * @param owner_node Hash owner node O (for registration)
   * @param tier_name Tier where the copy will be stored
   * @param[out] store_ms Wall time of the put operation
   * @return Return code from the put (0 = success)
   */
  clio::run::TaskResume PushConsumerCopy(
      const TagId &tag_id, const std::string &blob_name,
      ctp::ipc::ShmPtr<> raw_data, uint64_t size, uint32_t consumer_node,
      uint32_t owner_node, uint64_t version, Placement *place);
  /** Score configured for a tier name (-1 when unknown). */
  float TierScore(const std::string &tier) const;
  /** Bandwidth configured for a tier name in MB/ms (net bw when unknown). */
  double TierBwMbPerMs(const std::string &tier) const;
  /** Apply decision_order: pick the tier for a put and the ranking bandwidth. */
  void PlanTier(uint64_t size, Placement *place) const;

  /** File handle for candidates trace (.cand.NODE.csv). */
  std::ofstream candidates_trace_file_;
};

}  // namespace clio::cte::dtschedule

#endif  // CLIO_CTE_DTSCHEDULE_DTSCHEDULE_RUNTIME_H_
