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

#include <clio_cte/dtschedule/dtschedule_runtime.h>
#include <clio_cte/core/blob_batch.h>

#include <clio_ctp/compress/compress_factory.h>
#include <clio_ctp/util/logging.h>
#include <clio_ctp/introspect/system_info.h>
#include <clio_cte/dtschedule/ccm/data_stats.h>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <functional>
#include <cctype>
#include <chrono>
#include <vector>
#include <set>
#include <sstream>

namespace clio::cte::dtschedule {

/** Smallest chunk worth demoting (independent of min_compress_bytes, so
 *  writes can stay raw while demotion still compresses). */
static constexpr clio::run::u64 kDemoteMinBytes = 64ull << 10;


static std::string JoinCsv(const std::vector<std::string> &fields);
static std::string TraceNum(double v);
static uint64_t TraceNowMs();

Runtime::~Runtime() {
  CloseTraceFile();
}

clio::run::TaskStat Runtime::GetTaskStats(const clio::run::Task *task) const {
  clio::run::TaskStat stat;
  if (task == nullptr) {
    return stat;
  }

  switch (task->method_) {
    case Method::kPutBlob:
      stat.compute_ = 100;  // Compression cost estimate
      stat.wall_time_ = 1000.0f;  // Network + storage
      return stat;
    case Method::kGetBlob:
      stat.compute_ = 50;   // Decompression cost estimate
      stat.wall_time_ = 500.0f;
      return stat;
    case Method::kPollNodeLoad:
      stat.compute_ = 5;
      stat.wall_time_ = 10.0f;
      return stat;
    default:
      return clio::cte::core::CoreInterposer::GetTaskStats(task);
  }
}

clio::run::PoolQuery Runtime::ScheduleTask(
    const clio::run::shared_ptr<clio::run::Task> &task) {
  /**
   * Route data verbs with Dynamic queries to the submitter's node.
   *
   * dtschedule runs on the submitter's node for Dynamic queries on PutBlob,
   * GetBlob, GetBlobSize, and MultiPutBlob so it can make local compression
   * and placement decisions. All other tasks use the default CoreInterposer
   * routing, which delegates to the core's hash-based routing.
   *
   * See DESIGN.md §1 for motivation and cache_runtime.cc §59 for pattern.
   */
  if (!task->pool_query_.IsDynamicMode()) {
    // Non-Dynamic queries keep their explicit routing
    return clio::cte::core::CoreInterposer::ScheduleTask(task);
  }

  switch (task->method_) {
    case clio::cte::core::Method::kPutBlob:
    case clio::cte::core::Method::kGetBlob:
    case clio::cte::core::Method::kGetBlobSize:
    case clio::cte::core::Method::kMultiPutBlob:
      // Dynamic data verbs: run on submitter's node (Local)
      return clio::run::PoolQuery::Local();
    default:
      // All other methods (dtschedule verbs + core non-data methods):
      // use default routing
      return clio::cte::core::CoreInterposer::ScheduleTask(task);
  }
}

clio::run::TaskResume Runtime::Create(clio::run::shared_ptr<CreateTask> &task) {
  CLIO_TASK_BODY_BEGIN

  // Load configuration from task params
  config_ = task->GetParams();
  interposer_next_pool_ = config_.next_pool_id_;

  // Initialize core client
  if (!config_.next_pool_id_.IsNull()) {
    core_client_ = std::make_unique<clio::cte::core::Client>(CorePoolId());
  }

  // Open trace file if configured
  if (!config_.trace_path_.empty()) {
    OpenTraceFile();
  }

  // Initialize knobs from config
  knobs_.ccm_ = config_.ccm_;
  knobs_.load_aware_ = config_.load_aware_;
  knobs_.workflow_aware_ = config_.workflow_aware_ != "none";
  knobs_.force_scenario_ = config_.force_scenario_;
  knobs_.decision_order_ = config_.decision_order_;
  knobs_.ratio_noise_sigma_ = config_.ratio_noise_sigma_;

  // Initialize CCM manager
  ccm_manager_ = std::make_unique<ccm::CcmManager>();
  std::string model_dir = config_.qtable_model_path_;
  if (model_dir.empty()) {
    // Use compile-time default model directory
    model_dir = DTSCHEDULE_MODEL_DIR;
  }
  if (!ccm_manager_->Init(config_, model_dir)) {
    HLOG(kWarning, "dtschedule: CCM initialization failed");
  }

  // Phase 5: Initialize DAG spec loader if configured
  if (!config_.dag_path_.empty() && config_.workflow_aware_ == "dag") {
    dag_spec_ = std::make_unique<DagSpecLoader>();
    bool dag_ok = dag_spec_->Initialize(
        config_.dag_path_, CLIO_IPC, config_.dag_.colocate_fanin_,
        config_.dag_.replicate_fanout_min_,
        config_.dag_.replicate_max_);
    if (dag_ok) {
      dag_spec_->GetStats(dag_files_, dag_nodes_);
      HLOG(kInfo, "dtschedule: DAG spec loaded ({} files, {} nodes)",
           dag_files_, dag_nodes_);
    } else {
      HLOG(kWarning, "dtschedule: failed to load DAG spec from {}",
           config_.dag_path_);
      dag_spec_.reset();  // Clear on error
    }
  }

  HLOG(kInfo, "dtschedule: Create complete (ccm={}, trace={}, load_aware={}, workflow_aware={})",
       config_.ccm_, config_.trace_path_.empty() ? "off" : config_.trace_path_,
       config_.load_aware_, config_.workflow_aware_);

  // Phase 3: spawn periodic load sampling task
  if (config_.load_aware_) {
    int period_us = config_.load_period_ms_ * 1000;
    HLOG(kDebug, "dtschedule: Starting load sampling (period {}ms)",
         config_.load_period_ms_);
    // Spawn periodic SampleLoad task that samples local node load every load_period_ms
    auto *ipc_manager = CLIO_IPC;
    auto sample_task = ipc_manager->NewTask<SampleLoadTask>(
        clio::run::CreateTaskId(), pool_id_, clio::run::PoolQuery::Local());
    if (!sample_task.IsNull()) {
      sample_task->SetPeriod(period_us, clio::run::kMicro);
      sample_task->SetFlags(TASK_PERIODIC);
      ipc_manager->Send(sample_task);
    }
  }

  task->SetReturnCode(0);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Destroy(clio::run::shared_ptr<DestroyTask> &task) {
  CLIO_TASK_BODY_BEGIN

  CloseTraceFile();
  HLOG(kInfo, "dtschedule: Destroy complete (stats: {} puts, {} compressed)",
       stats_.puts_, stats_.compressed_);

  task->SetReturnCode(0);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Monitor(clio::run::shared_ptr<MonitorTask> &task) {
  CLIO_TASK_BODY_BEGIN

  // Phase 1: just forward to core's Monitor to keep target table current
  if (core_client_) {
    HLOG(kDebug, "dtschedule: Monitor forwarding to core");
  }

  task->SetReturnCode(0);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

bool Runtime::ShouldSelect(size_t size) const {
  // With placement: dtschedule a raw-only CCM still runs the placement
  // decision (consumer node, tier); only the codec selection is skipped.
  return size >= static_cast<size_t>(config_.min_compress_bytes_) &&
         ccm_manager_ != nullptr &&
         (CompressionEnabled() || config_.placement_ == "dtschedule");
}

bool Runtime::CompressionEnabled() const {
  return !knobs_.ccm_.empty() && knobs_.ccm_ != "fixed:none";
}

ctp::ipc::FullPtr<char> Runtime::CompressWithDecision(
    const char *src, size_t size, const ccm::Decision &decision,
    CompressOutcome *out) {
  auto codec = ctp::CompressionFactory::GetPreset(decision.chosen_lib_,
                                                  decision.chosen_preset_);
  if (!codec) {
    HLOG(kWarning, "dtschedule: codec {} unavailable; storing raw",
         decision.chosen_lib_);
    return ctp::ipc::FullPtr<char>();
  }
  constexpr size_t kHdr = sizeof(compressor::CompressionHeader);
  const size_t bound = kHdr + size + (size >> 3) + (64 << 10);
  auto buf = CLIO_IPC->AllocateBuffer(bound);
  if (buf.IsNull()) {
    HLOG(kWarning, "dtschedule: AllocateBuffer({}) failed; storing raw", bound);
    return ctp::ipc::FullPtr<char>();
  }
  size_t comp_size = bound - kHdr;
  auto t0 = std::chrono::steady_clock::now();
  bool ok = codec->Compress(buf.ptr_ + kHdr, comp_size,
                            const_cast<char *>(src), size);
  out->attempted = true;
  out->ctime_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
          .count();
  if (!ok || comp_size == 0) {
    CLIO_IPC->FreeBuffer(buf);
    return ctp::ipc::FullPtr<char>();
  }
  out->comp_size = comp_size;
  out->ratio = static_cast<double>(size) / static_cast<double>(comp_size);
  if (out->ratio < ccm::kMinUsefulRatio) {
    CLIO_IPC->FreeBuffer(buf);  // not worth it: store raw, keep the observation
    return ctp::ipc::FullPtr<char>();
  }
  auto *header = reinterpret_cast<compressor::CompressionHeader *>(buf.ptr_);
  *header = compressor::CompressionHeader(
      ctp::CompressionFactory::GetWireId(decision.chosen_lib_),
      compressor::ToWirePreset(decision.chosen_preset_), size, comp_size);
  out->used = true;
  return buf;
}

void Runtime::RecordDecision(const std::string &tag_id,
                             const std::string &blob_name,
                             size_t original_size, const char *src,
                             const ccm::Decision &decision,
                             const CompressOutcome &out, double select_ms,
                             double load_mult, double producer_cpu,
                             const Placement &place) {
  if (out.attempted && !decision.chosen_lib_.empty()) {
    ccm_manager_->Observe(src, original_size, blob_name, decision.chosen_lib_,
                          decision.chosen_preset_, decision.pred_ctime_ms_,
                          decision.pred_ratio_, out.ctime_ms, 0.0, out.ratio,
                          &decision.features_);
  }
  if (!trace_file_.is_open()) {
    return;
  }
  WriteTraceRow(tag_id, blob_name, original_size, decision.features_, decision,
                out, select_ms, load_mult, producer_cpu, place);
  WriteCandidatesTrace(tag_id, blob_name, decision);
}

clio::run::TaskResume Runtime::PutBlob(
    clio::run::shared_ptr<clio::cte::core::PutBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  const size_t original_size = task->size_;
  const auto orig_data = task->blob_data_;
  ccm::Decision decision{};
  CompressOutcome out{};
  ctp::ipc::FullPtr<char> comp_buf;
  ctp::ipc::FullPtr<char> src_ptr;
  bool selected = false;
  double select_ms = 0.0;
  {
    std::lock_guard<std::mutex> lock(stats_lock_);
    stats_.puts_++;
  }
  // Replica-slot writes (cache copies must stay raw) and already-compressed
  // bytes pass through untouched.
  if (task->context_.replica_ != 0 ||
      (task->context_.transform_flags_ &
       clio::cte::core::kBlobTransformCompressed)) {
    {
      std::lock_guard<std::mutex> lock(stats_lock_);
      stats_.passthrough_++;
    }
    if (task->context_.replica_ != 0) {
      CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kPutBlob,
                                  task.template Cast<clio::run::Task>()));
    } else {
      CLIO_CO_AWAIT(ForwardRawPut(task));
    }
    CLIO_CO_RETURN;
  }
  if (IsPlaced(task->blob_name_.str()) &&
      (task->offset_ != 0 || !task->segments_.empty())) {
    // A partial write lands where the chunk already lives (raw).
    uint32_t node = CLIO_IPC->GetNodeId();
    CLIO_CO_AWAIT(ResolveLocation(task->tag_id_, task->blob_name_.str(), &node));
    CLIO_CO_AWAIT(PutAtNode(task, node));
    CLIO_CO_RETURN;
  }
    // Phase 3: initialize load info (may be overridden if ShouldSelect)
  double load_mult = 1.0;
  double producer_cpu = 0.0;
  Placement place{};
  // Name used for DAG lookups, QoS stage regexes and the trace: the tag's
  // resolved name when the adapter only gave us an id, else the id itself.
  std::string tag_name = task->tag_id_.ToString();
  // Name the QoS regexes and the candidate filter match against: the blob
  // name when it carries one (direct API traffic), else the tag's resolved
  // name (adapter traffic: the tag is the file, blobs are chunk indices).
  std::string match_name = task->blob_name_.str();
  const bool blob_is_chunk_index = IsChunkIndexName(match_name);

  if (ShouldSelect(original_size)) {
    src_ptr = CLIO_IPC->ToFullPtr<char>(task->blob_data_.template Cast<char>());
    if (src_ptr.ptr_ != nullptr && NeedsTagName()) {
      auto cached = CachedTagName(tag_name);
      if (!cached.has_value()) {
        if (!core_client_) {
          core_client_ =
              std::make_unique<clio::cte::core::Client>(CorePoolId());
        }
        auto name_task = core_client_->AsyncGetTagName(task->tag_id_);
        CLIO_CO_AWAIT(name_task);
        std::string resolved =
            name_task->found_ ? name_task->tag_name_.str() : std::string();
        CacheTagName(tag_name, resolved);
        cached = resolved;
      }
      if (!cached->empty()) {
        tag_name = *cached;
        if (blob_is_chunk_index) {
          match_name = tag_name;
        }
      }
    }
    if (src_ptr.ptr_ != nullptr) {
      // Phase 3: compute load multiplier for this node
      uint32_t local_node_id = CLIO_IPC->GetNodeId();
      auto [cpu_a, age_a, known_a] = GetLoad(local_node_id);
      producer_cpu = cpu_a;
      // The tracked consumer's load (when known) is the other side of the
      // multiplier; with no consumer the put is local-only and m = 1.
      double cpu_b = cpu_a;
      place.consumer_node = PickConsumer(task->tag_id_);
      if (config_.workflow_aware_ == "dag" && dag_spec_ && dag_spec_->IsLoaded() &&
          dag_spec_->Unresolved() > 0 && !CLIO_IPC->GetAllHosts().empty()) {
        // The host table was empty when the pool was composed: resolve again.
        dag_spec_->Initialize(config_.dag_path_, CLIO_IPC,
                              config_.dag_.colocate_fanin_,
                              config_.dag_.replicate_fanout_min_,
                              config_.dag_.replicate_max_);
        dag_spec_->GetStats(dag_files_, dag_nodes_);
      }
      if (config_.workflow_aware_ == "dag" && dag_spec_ && dag_spec_->IsLoaded()) {
        // The DAG names every file's consumers before the first byte is
        // written: prefer it over the live map (blob name, then tag name).
        auto info = dag_spec_->LookupFile(tag_name);
        if (info.producer_node == UINT32_MAX) {
          info = dag_spec_->LookupFile(task->blob_name_.str());
        }
        if (info.producer_node != UINT32_MAX) {
          place.dag_hit = true;
          place.dag_consumers = info.consumer_nodes;
          for (uint32_t node : info.consumer_nodes) {
            if (node != UINT32_MAX && node != local_node_id) {
              place.consumer_node = node;
              break;
            }
          }
          if (place.consumer_node == UINT32_MAX && !info.consumer_nodes.empty()) {
            place.consumer_node = info.consumer_nodes.front();
          }
        }
      }
      if (place.consumer_node != UINT32_MAX) {
        auto [cpu_c, age_c, known_c] = GetLoad(place.consumer_node);
        if (known_c) {
          cpu_b = cpu_c;
          place.consumer_cpu = cpu_c;
        }
      }
      place.owner_node = OwnerNode(task->tag_id_, task->blob_name_.str());
      load_mult = LoadMultiplier(cpu_a, cpu_b, knobs_.load_aware_,
                                 config_.load_cap_);
      PlanTier(original_size, &place);
      // The ranker's store cost: the tier's bandwidth when the bytes stay
      // on this node, else the slower of tier and network. The payload
      // crosses the wire when the owner is another node, or when a known
      // consumer sits on a node other than the owner (it reads the stored
      // bytes over the network: compressing here shrinks that transfer).
      double rank_bw_mb_ms = place.tier_bw_mb_ms;
      {
        const uint32_t owner = place.owner_node == UINT32_MAX
                                   ? local_node_id : place.owner_node;
        const bool consumer_remote = place.consumer_node != UINT32_MAX &&
                                     place.consumer_node != owner;
        if (owner != local_node_id || consumer_remote) {
          const double net_mb_ms = std::max(config_.net_bw_gbps_, 0.01) / 8.0;
          rank_bw_mb_ms = rank_bw_mb_ms > 0.0
                              ? std::min(rank_bw_mb_ms, net_mb_ms) : net_mb_ms;
        }
      }

      const uint32_t owner_for_tier =
          (IsPlaced(task->blob_name_.str()) || place.owner_node == UINT32_MAX)
              ? local_node_id : place.owner_node;
      // Each candidate is priced at the tier its own stored size lands in
      // on the owner (raw may spill to NVMe while compressed fits in RAM).
      const uint32_t consumer_for_bw = place.consumer_node;
      std::function<double(uint64_t)> store_bw =
          [this, owner_for_tier, consumer_for_bw](uint64_t bytes) {
            return StoreBwFor(bytes, owner_for_tier, consumer_for_bw);
          };
      auto t0 = std::chrono::steady_clock::now();
      bool reused = false;
      if (CompressionEnabled()) {
        decision = DecideCodec(task->tag_id_, src_ptr.ptr_, original_size,
                               match_name, load_mult, rank_bw_mb_ms,
                               &store_bw, &reused);
      } else {
        decision.chosen_lib_.clear();  // raw; placement is still decided
        decision.pred_ratio_ = 1.0;
      }
      select_ms = std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - t0)
                      .count();
      selected = true;
      if (place.order != "tier_first" && !decision.chosen_lib_.empty()) {
        // joint / codec_first: the compressed size may fit a faster tier.
        const uint64_t z = static_cast<uint64_t>(
            original_size / std::max(decision.pred_ratio_, 1.0));
        place.tier = ChooseTier(z, owner_for_tier);
        place.tier_score = TierScore(place.tier);
        place.tier_bw_mb_ms = TierBwMbPerMs(place.tier);
      }
      {
        // Reserve what will actually be stored on the owner's chosen tier.
        const uint64_t stored =
            decision.chosen_lib_.empty()
                ? original_size
                : static_cast<uint64_t>(original_size /
                                        std::max(decision.pred_ratio_, 1.0));
        place.tier = ChooseTier(stored, owner_for_tier, /*reserve=*/true);
        place.tier_score = TierScore(place.tier);
        place.tier_bw_mb_ms = TierBwMbPerMs(place.tier);
      }
      if (task->score_ < 0.0f && place.tier_score >= 0.0f) {
        task->score_ = place.tier_score;  // never override an explicit score
      }
    }
  }
  // ---- Scenario (DESIGN §3) is decided BEFORE any bytes move: with a
  // consumer on another node, scenario 3 ships the raw bytes there instead
  // of compressing here.
  if (selected) {
    auto choice = SelectScenario(original_size, decision.pred_ratio_,
                                 decision.pred_ctime_ms_, decision.pred_dtime_ms_,
                                 load_mult, place);
    place.scenario = choice.chosen_scenario;
    place.cost1_ms = choice.cost1_ms;
    place.cost2_ms = choice.cost2_ms;
    place.cost3_ms = choice.cost3_ms;
  }
  const uint32_t self_node = CLIO_IPC->GetNodeId();
  const bool placed = IsPlaced(task->blob_name_.str());
  // dtschedule decides the store node: the consumer for scenarios 2/3 (its
  // node is idle and will read the chunk), else this (writer) node.
  const bool to_consumer = placed && selected &&
                           (place.scenario == 2 || place.scenario == 3) &&
                           place.consumer_node != UINT32_MAX &&
                           place.consumer_node != self_node;
  const bool remote_s3 = selected && place.scenario == 3 &&
                         !decision.chosen_lib_.empty() &&
                         place.consumer_node != UINT32_MAX &&
                         place.consumer_node != self_node;
  if (remote_s3) {
    CLIO_CO_AWAIT(CompressAtConsumer(task, decision, original_size, &out, &place));
    if (placed && task->GetReturnCode() == 0) {
      CLIO_CO_AWAIT(WriteLocation(task->tag_id_, task->blob_name_.str(),
                                  place.consumer_node));
    }
  } else if (to_consumer) {
    // Scenario 2 under dtschedule placement: compress here (when the codec
    // pays) and store the chunk on the consumer's node, which reads it
    // locally.
    if (!decision.chosen_lib_.empty()) {
      comp_buf = CompressWithDecision(src_ptr.ptr_, original_size, decision, &out);
    }
    if (out.used) {
      task->blob_data_ = comp_buf.shm_.template Cast<void>();
      task->size_ = sizeof(compressor::CompressionHeader) + out.comp_size;
      task->context_.transform_flags_ |= clio::cte::core::kBlobTransformCompressed;
    }
    {
      std::lock_guard<std::mutex> lock(stats_lock_);
      stats_.bytes_in_ += original_size;
      stats_.bytes_out_ += out.used ? out.comp_size : original_size;
      if (out.used) {
        stats_.compressed_++;
        stats_.per_lib_count_[decision.chosen_lib_]++;
      }
    }
    CLIO_CO_AWAIT(PutAtNode(task, place.consumer_node));
    task->blob_data_ = orig_data;
    task->size_ = original_size;
    if (!comp_buf.IsNull()) {
      CLIO_IPC->FreeBuffer(comp_buf);
    }
  } else {
    if (selected && place.scenario == 3) {
      std::lock_guard<std::mutex> lock(stats_lock_);
      stats_.s3_local_++;  // consumer is this node: compress here
    }
    if (selected && !decision.chosen_lib_.empty()) {
      comp_buf = CompressWithDecision(src_ptr.ptr_, original_size, decision, &out);
    }
    if (out.used) {
      task->blob_data_ = comp_buf.shm_.template Cast<void>();
      task->size_ = sizeof(compressor::CompressionHeader) + out.comp_size;
      task->context_.transform_flags_ |= clio::cte::core::kBlobTransformCompressed;
      {
        std::lock_guard<std::mutex> lock(stats_lock_);
        stats_.compressed_++;
        stats_.bytes_in_ += original_size;
        stats_.bytes_out_ += out.comp_size;
        stats_.per_lib_count_[decision.chosen_lib_]++;
      }
      if (placed) {
        CLIO_CO_AWAIT(PutAtNode(task, BestStoreNode(task->size_)));
      } else {
        CLIO_CO_AWAIT(ForwardCompressedPut(task));
      }
    } else {
      {
        std::lock_guard<std::mutex> lock(stats_lock_);
        stats_.bytes_in_ += original_size;
        stats_.bytes_out_ += original_size;
      }
      if (placed) {
        CLIO_CO_AWAIT(PutAtNode(task, BestStoreNode(task->size_)));
      } else {
        CLIO_CO_AWAIT(ForwardRawPut(task));
      }
    }
    task->blob_data_ = orig_data;
    task->size_ = original_size;
    if (!comp_buf.IsNull()) {
      CLIO_IPC->FreeBuffer(comp_buf);
    }
    if (!placed && selected && task->GetReturnCode() == 0 &&
        place.scenario == 2) {
      CLIO_CO_AWAIT(PushConsumerCopy(task->tag_id_, task->blob_name_.str(),
                                     orig_data, original_size,
                                     place.consumer_node, place.owner_node,
                                     task->context_.version_, &place));
    }
  }
  if (!placed && selected && task->GetReturnCode() == 0 && place.dag_hit &&
      dag_spec_ &&
      place.dag_consumers.size() >= dag_spec_->ReplicateFanoutMin()) {
    CLIO_CO_AWAIT(PushFanoutCopies(
        task->tag_id_, task->blob_name_.str(), orig_data, original_size,
        place.owner_node,
        place.scenario != 1 ? place.consumer_node : UINT32_MAX,
        task->context_.version_, &place));
  }
  if (selected) {
    RecordDecision(tag_name, task->blob_name_.str(),
                   original_size, src_ptr.ptr_, decision, out, select_ms,
                   load_mult, producer_cpu, place);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

ccm::Decision Runtime::DecideCodec(
    const TagId &tag_id, const char *src, size_t size,
    const std::string &match_name, double load_mult, double rank_bw_mb_ms,
    const std::function<double(uint64_t)> *store_bw, bool *reused) {
  // Chunks of one file share their data class, so the features and ranking
  // of one chunk hold for its neighbours. Deciding per chunk cost ~1.2 ms of
  // CPU per MiB on the writer: on a node whose cores the application holds,
  // that alone capped writes at ~170 MB/s.
  const uint64_t now_ms = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
  // Size is part of the key: the store cost the ranker weighs scales with it.
  const std::string key = tag_id.ToString() + "|" + std::to_string(size);
  const uint64_t max_age_ms =
      static_cast<uint64_t>(std::max(config_.load_period_ms_, 1));
  {
    std::lock_guard<std::mutex> lock(decision_cache_lock_);
    auto it = decision_cache_.find(key);
    if (it != decision_cache_.end() && it->second.uses_left > 0 &&
        now_ms - it->second.made_ms <= max_age_ms) {
      --it->second.uses_left;
      *reused = true;
      return it->second.decision;
    }
  }
  ccm::Decision decision = ccm_manager_->SelectCodec(
      src, size, match_name, knobs_.ratio_noise_sigma_, load_mult,
      rank_bw_mb_ms, store_bw);
  *reused = false;
  if (config_.decision_reuse_chunks_ > 1) {
    CachedDecision entry;
    entry.decision = decision;
    entry.decision.candidates_.clear();  // traced once, with the fresh one
    entry.uses_left = config_.decision_reuse_chunks_ - 1;
    entry.made_ms = now_ms;
    std::lock_guard<std::mutex> lock(decision_cache_lock_);
    if (decision_cache_.size() >= (1u << 16)) {
      decision_cache_.clear();
    }
    decision_cache_[key] = std::move(entry);
  }
  return decision;
}

clio::run::TaskResume Runtime::ForwardCompressedPut(
    clio::run::shared_ptr<clio::cte::core::PutBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  // Compressed bytes skip the cache (its local copies must stay raw) and go
  // to the pool below it, hash-routed to the owner.
  if (!config_.compressed_next_pool_id_.IsNull()) {
    if (!compressed_client_) {
      compressed_client_ = std::make_unique<clio::cte::core::Client>(
          config_.compressed_next_pool_id_);
    }
    clio::cte::core::Context comp_ctx = task->context_;
    auto comp_put = compressed_client_->AsyncPutBlob(
        task->tag_id_, task->blob_name_.str(), 0, task->size_,
        task->blob_data_, /*score=*/task->score_, comp_ctx, /*flags=*/0,
        clio::run::PoolQuery::Dynamic());
    CLIO_CO_AWAIT(comp_put);
    task->SetReturnCode(comp_put->GetReturnCode());
    task->context_ = comp_put->context_;
  } else {
    CLIO_CO_AWAIT(ForwardRawPut(task));
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

bool Runtime::OwnerIsLocal(const TagId &tag_id, const std::string &blob_name) {
  /**
   * Whether this node owns the blob's authoritative copy (core hash).
   *
   * @param tag_id Tag of the blob
   * @param blob_name Blob name
   * @return true when the hash owner is this node
   */
  return OwnerNode(tag_id, blob_name) == CLIO_IPC->GetNodeId();
}

clio::run::TaskResume Runtime::ForwardRawPut(
    clio::run::shared_ptr<clio::cte::core::PutBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  // dtschedule runs data verbs on the submitter's node (ScheduleTask) so it
  // can read the bytes and the local load. ForwardToCore executes the next
  // pool's method on THIS node's container, which is only right when this
  // node owns the blob (or the next pool is an interposer that hops to the
  // owner itself). A raw put whose owner is elsewhere goes through the
  // chain client with Dynamic(): the core's ScheduleTask hash-routes it;
  // an interposer below keeps its own routing. Without this, every raw
  // chunk (the filesystem's directory blocks included) stayed on the
  // writer's node and the owner's readers saw ENOENT (issue #886 family).
  if (OwnerIsLocal(task->tag_id_, task->blob_name_.str())) {
    CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kPutBlob,
                                task.template Cast<clio::run::Task>()));
    CLIO_CO_RETURN;
  }
  EnsureCoreClient();
  std::vector<clio::cte::core::BlobRegion> regions;
  clio::cte::core::ForEachBlobRegion(
      *task, [&regions](const clio::cte::core::BlobRegion &r) {
        regions.push_back(r);
        return true;
      });
  clio::run::u32 rc = 0;
  clio::cte::core::Context out_ctx = task->context_;
  for (size_t i = 0; i < regions.size() && rc == 0; ++i) {
    auto put = core_client_->AsyncPutBlob(
        task->tag_id_, task->blob_name_.str(), regions[i].blob_off_,
        regions[i].size_, regions[i].data_, task->score_, task->context_,
        task->flags_, clio::run::PoolQuery::Dynamic());
    CLIO_CO_AWAIT(put);
    rc = put->GetReturnCode();
    out_ctx = put->context_;
  }
  task->context_ = out_ctx;
  task->SetReturnCode(rc);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::ForwardRawGet(
    clio::run::shared_ptr<clio::cte::core::GetBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  // Same owner hop as ForwardRawPut, for reads: explicit replica reads are
  // node-local by definition; everything else is served by the owner.
  if (task->context_.replica_ != 0 ||
      OwnerIsLocal(task->tag_id_, task->blob_name_.str())) {
    CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kGetBlob,
                                task.template Cast<clio::run::Task>()));
    CLIO_CO_RETURN;
  }
  EnsureCoreClient();
  std::vector<clio::cte::core::BlobRegion> regions;
  clio::cte::core::ForEachBlobRegion(
      *task, [&regions](const clio::cte::core::BlobRegion &r) {
        regions.push_back(r);
        return true;
      });
  clio::run::u32 rc = 0;
  clio::run::u32 out_tflags = 0;
  clio::run::u64 version = 0;
  for (size_t i = 0; i < regions.size() && rc == 0; ++i) {
    auto get = core_client_->AsyncGetBlob(
        task->tag_id_, task->blob_name_.str(), regions[i].blob_off_,
        regions[i].size_, task->flags_, regions[i].data_,
        clio::run::PoolQuery::Dynamic(), task->context_);
    CLIO_CO_AWAIT(get);
    rc = get->GetReturnCode();
    out_tflags = get->context_.transform_flags_;
    if (i == 0) {
      version = get->context_.version_;
    }
  }
  task->context_.transform_flags_ = out_tflags;
  task->context_.version_ = version;
  task->SetReturnCode(rc);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::CompressAtConsumer(
    clio::run::shared_ptr<clio::cte::core::PutBlobTask> &task,
    const ccm::Decision &decision, size_t original_size, CompressOutcome *out,
    Placement *place) {
  CLIO_TASK_BODY_BEGIN
  // Scenario 3: the raw bytes travel once (P -> C); C compresses, writes its
  // own raw cache copy, stores the compressed primary at the owner and
  // registers itself as the copy holder.
  if (!dtschedule_client_) {
    dtschedule_client_ = std::make_unique<Client>(
        pool_id_, config_.core_pool_id_.IsNull() ? clio::cte::core::kCtePoolId
                                                 : config_.core_pool_id_);
  }
  {
    auto comp_at = dtschedule_client_->AsyncCompressAt(
        task->tag_id_, task->blob_name_.str(), task->blob_data_, original_size,
        task->score_, place->owner_node, decision.chosen_lib_,
        ctp::CompressionFactory::GetPresetName(decision.chosen_preset_),
        task->context_, clio::run::PoolQuery::Physical(place->consumer_node));
    CLIO_CO_AWAIT(comp_at);
    task->SetReturnCode(comp_at->GetReturnCode());
    if (comp_at->GetReturnCode() == 0) {
      task->context_ = comp_at->context_;
      place->store_ms = comp_at->store_ms_;
      out->attempted = true;
      out->ctime_ms = comp_at->ctime_ms_;
      out->comp_size = comp_at->comp_size_;
      out->used = comp_at->comp_size_ > 0 && comp_at->comp_size_ < original_size;
      out->ratio = out->used ? static_cast<double>(original_size) / comp_at->comp_size_
                             : 1.0;
      std::lock_guard<std::mutex> lock(stats_lock_);
      stats_.s3_sent_++;
      stats_.bytes_in_ += original_size;
      stats_.bytes_out_ += out->used ? comp_at->comp_size_ : original_size;
      if (out->used) {
        stats_.compressed_++;
        stats_.per_lib_count_[decision.chosen_lib_]++;
      }
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::GetBlob(
    clio::run::shared_ptr<clio::cte::core::GetBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  // Serve the read as-is first: a raw blob costs nothing extra and the core
  // reports the blob's transform state OUT through the context either way.
  // Chunks dtschedule placed are read from the node that holds them.
  uint32_t data_node = CLIO_IPC->GetNodeId();
  const bool placed_get = IsPlaced(task->blob_name_.str());
  if (placed_get) {
    CLIO_CO_AWAIT(ResolveLocation(task->tag_id_, task->blob_name_.str(),
                                  &data_node));
    CLIO_CO_AWAIT(GetAtNode(task, data_node));
  } else {
    CLIO_CO_AWAIT(ForwardRawGet(task));
  }

  // Phase 4: Consumer tracking for workflow_aware mode
  // Register this node as a consumer of the tag on first read (regardless of compression)
  if (task->GetReturnCode() == 0 && config_.workflow_aware_ == "consumer" &&
      !IsConsumerRegistered(task->tag_id_)) {
    auto *ipc_manager = CLIO_IPC;
    auto my_node_id = ipc_manager->GetNodeId();
    auto reg_task = ipc_manager->NewTask<RegisterConsumerTask>(
        clio::run::CreateTaskId(), pool_id_,
        clio::run::PoolQuery::Broadcast());
    if (!reg_task.IsNull()) {
      reg_task->tag_id_ = task->tag_id_;
      reg_task->consumer_node_ = my_node_id;
      // Send and await the registration
      CLIO_CO_AWAIT(ipc_manager->Send(reg_task.template Cast<clio::run::Task>()));
      if (reg_task->GetReturnCode() == 0) {
        // Only mark as registered after successful broadcast
        std::string tag_key = task->tag_id_.ToString();
        std::lock_guard<std::mutex> lock(consumer_lock_);
        auto &consumer_nodes = tag_consumers_[tag_key];
        std::lock_guard<std::mutex> inner_lock(consumer_nodes.lock_);
        // Keep last 32 nodes per tag (FIFO order, most recent at the end)
        if (consumer_nodes.nodes_.size() >= 32) {
          consumer_nodes.nodes_.pop_front();
        }
        consumer_nodes.nodes_.push_back(my_node_id);
        {
          std::lock_guard<std::mutex> stats_lock(stats_lock_);
          stats_.consumer_tags_++;
        }
        HLOG(kDebug, "dtschedule: RegisterConsumer successful (tag={}, node={})",
             task->tag_id_.ToString(), my_node_id);
      }
    }
  }

  if (task->GetReturnCode() == 0 && task->context_.replica_ == 0 &&
      !(task->context_.transform_flags_ &
        clio::cte::core::kBlobTransformCompressed) &&
      config_.demote_watermark_ > 0.0 && task->segments_.empty() &&
      task->offset_ == 0 && task->size_ >= kDemoteMinBytes) {
    // A whole raw chunk this node just consumed: a demotion candidate.
    auto bytes = CLIO_IPC->ToFullPtr<char>(task->blob_data_.template Cast<char>());
    if (bytes.ptr_ != nullptr) {
      EnqueueDemote(task->tag_id_, task->blob_name_.str(),
                    placed_get ? data_node
                               : OwnerNode(task->tag_id_,
                                           task->blob_name_.str()),
                    bytes.ptr_, task->size_);
    }
  }
  if (task->GetReturnCode() != 0 || task->context_.replica_ != 0 ||
      !(task->context_.transform_flags_ &
        clio::cte::core::kBlobTransformCompressed)) {
    CLIO_CO_RETURN;
  }
  if (TryDecompressInPlace(*task)) {
    CLIO_CO_RETURN;  // the first read already held the whole stored blob
  }
  {
    // The forwarded read handed back CODEC bytes (and the stored size is
    // not the logical size). Fetch the whole stored blob, decompress once,
    // then slice every requested region out of the original.
    auto dtime_start = std::chrono::high_resolution_clock::now();
    EnsureCoreClient();
    clio::run::u64 stored_size = 0;
    {
      clio::cte::core::Client *cc =
          placed_get ? copy_client_.get() : core_client_.get();
      auto sz = cc->AsyncGetBlobSize(
          task->tag_id_, task->GetBlobName(),
          placed_get ? NodeQuery(data_node) : clio::run::PoolQuery::Dynamic());
      CLIO_CO_AWAIT(sz);
      if (sz->GetReturnCode() != 0 || sz->size_ == 0) {
        task->SetReturnCode(10 + sz->GetReturnCode());
        CLIO_CO_RETURN;
      }
      stored_size = sz->size_;
    }
    auto stored = CLIO_IPC->AllocateBuffer(stored_size);
    if (stored.IsNull()) {
      task->SetReturnCode(2);
      CLIO_CO_RETURN;
    }
    {
      clio::cte::core::Client *gc =
          placed_get ? copy_client_.get() : core_client_.get();
      auto get = gc->AsyncGetBlob(
          task->tag_id_, task->GetBlobName(), 0, stored_size,
          /*flags=*/0, stored.shm_.template Cast<void>(),
          placed_get ? NodeQuery(data_node) : clio::run::PoolQuery::Dynamic());
      CLIO_CO_AWAIT(get);
      if (get->GetReturnCode() != 0) {
        CLIO_IPC->FreeBuffer(stored);
        task->SetReturnCode(10 + get->GetReturnCode());
        CLIO_CO_RETURN;
      }
    }
    clio::run::u64 out_size = 0;
    std::string library_name;
    auto scratch = DecompressStored(stored.ptr_, stored_size, &out_size,
                                    &library_name);
    CLIO_IPC->FreeBuffer(stored);
    if (scratch.IsNull()) {
      task->SetReturnCode(5);
      CLIO_CO_RETURN;
    }
    bool region_ok = CopyRegionsFromOriginal(*task, scratch.ptr_, out_size);
    CLIO_IPC->FreeBuffer(scratch);
    if (!region_ok) {
      task->SetReturnCode(3);
      CLIO_CO_RETURN;
    }
    // The caller now holds ORIGINAL bytes: clear the transform report.
    task->context_.transform_flags_ &=
        ~(clio::cte::core::kBlobTransformed |
          clio::cte::core::kBlobTransformCompressed);
    double dtime_ms = std::chrono::duration<double, std::milli>(
                          std::chrono::high_resolution_clock::now() -
                          dtime_start)
                          .count();
    WriteDecompressTraceRow(task->tag_id_.ToString(), task->blob_name_.str(),
                            out_size, library_name, dtime_ms);
    task->SetReturnCode(0);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

/**
 * Lazily construct the client used to talk to the next pool in the chain.
 */
void Runtime::EnsureCoreClient() {
  if (!core_client_) {
    core_client_ = std::make_unique<clio::cte::core::Client>(CorePoolId());
  }
}

/**
 * Decompress a stored (header + codec bytes) blob into a fresh SHM buffer.
 *
 * @param stored      pointer to the stored bytes (header first)
 * @param stored_size number of stored bytes
 * @param out_size    receives the number of original bytes produced
 * @param lib_name    receives the codec name taken from the header
 * @return the scratch buffer holding the original bytes, or a null pointer
 *         on a malformed header, missing codec, or codec failure
 */
bool Runtime::TryDecompressInPlace(clio::cte::core::GetBlobTask &task) {
  /**
   * Fast path for reading a compressed blob: when the first (forwarded) read
   * was a single region from offset 0 that already holds the whole stored
   * blob (header + codec bytes, which is smaller than the original), decode
   * from those bytes instead of asking for the stored size and re-reading
   * the blob. One round trip instead of three.
   *
   * @param task The read; on success its buffer holds the original bytes and
   *        the transform flags are cleared
   * @return false when the fast path does not apply (caller falls back)
   */
  constexpr size_t kHdr = sizeof(compressor::CompressionHeader);
  if (!task.segments_.empty() || task.offset_ != 0 || task.size_ < kHdr) {
    return false;
  }
  auto buf = CLIO_IPC->ToFullPtr<char>(task.blob_data_.template Cast<char>());
  if (buf.ptr_ == nullptr) {
    return false;
  }
  const auto *header =
      reinterpret_cast<const compressor::CompressionHeader *>(buf.ptr_);
  if (!header->IsValid() || header->compressed_size_ == 0 ||
      kHdr + header->compressed_size_ > task.size_) {
    return false;
  }
  auto start = std::chrono::high_resolution_clock::now();
  const clio::run::u64 stored = kHdr + header->compressed_size_;
  std::vector<char> copy(buf.ptr_, buf.ptr_ + stored);  // buf is the target
  clio::run::u64 out_size = 0;
  std::string lib;
  auto scratch = DecompressStored(copy.data(), stored, &out_size, &lib);
  if (scratch.IsNull()) {
    return false;
  }
  const bool ok = CopyRegionsFromOriginal(task, scratch.ptr_, out_size);
  CLIO_IPC->FreeBuffer(scratch);
  if (!ok) {
    return false;
  }
  task.context_.transform_flags_ &=
      ~(clio::cte::core::kBlobTransformed |
        clio::cte::core::kBlobTransformCompressed);
  WriteDecompressTraceRow(task.tag_id_.ToString(), task.blob_name_.str(),
                          out_size, lib,
                          std::chrono::duration<double, std::milli>(
                              std::chrono::high_resolution_clock::now() - start)
                              .count());
  task.SetReturnCode(0);
  return true;
}

ctp::ipc::FullPtr<char> Runtime::DecompressStored(const char *stored,
                                                  clio::run::u64 stored_size,
                                                  clio::run::u64 *out_size,
                                                  std::string *lib_name) {
  constexpr size_t kHdr = sizeof(compressor::CompressionHeader);
  if (stored_size < kHdr) {
    return ctp::ipc::FullPtr<char>();
  }
  const auto *header =
      reinterpret_cast<const compressor::CompressionHeader *>(stored);
  if (!header->IsValid()) {
    return ctp::ipc::FullPtr<char>();
  }
  *lib_name = ctp::CompressionFactory::NameForWireId(header->compress_lib_);
  ctp::CompressionPreset preset =
      compressor::FromWirePreset(header->compress_preset_);
  auto codec = ctp::CompressionFactory::GetPreset(*lib_name, preset);
  if (!codec) {
    HLOG(kWarning, "dtschedule: no codec '{}' for stored blob", *lib_name);
    return ctp::ipc::FullPtr<char>();
  }
  auto scratch = CLIO_IPC->AllocateBuffer(header->original_size_);
  if (scratch.IsNull()) {
    return ctp::ipc::FullPtr<char>();
  }
  size_t produced = header->original_size_;
  // The codec API takes a mutable input pointer but does not write to it.
  bool ok = codec->Decompress(scratch.ptr_, produced,
                              const_cast<char *>(stored) + kHdr,
                              stored_size - kHdr);
  if (!ok) {
    HLOG(kWarning, "dtschedule: {} decompression failed (stored={} orig={})",
         *lib_name, stored_size, header->original_size_);
    CLIO_IPC->FreeBuffer(scratch);
    return ctp::ipc::FullPtr<char>();
  }
  *out_size = produced;
  return scratch;
}

/**
 * Copy each region requested by a GetBlob task out of the original bytes.
 * Regions past the end keep the core's short-read semantics (untouched).
 *
 * @return false when a destination pointer could not be resolved
 */
bool Runtime::CopyRegionsFromOriginal(clio::cte::core::GetBlobTask &task,
                                      const char *original,
                                      clio::run::u64 out_size) {
  bool region_ok = true;
  clio::cte::core::ForEachBlobRegion(
      task, [&](const clio::cte::core::BlobRegion &r) {
        if (r.blob_off_ >= out_size) {
          return true;
        }
        clio::run::u64 n = r.size_;
        if (r.blob_off_ + n > out_size) {
          n = out_size - r.blob_off_;
        }
        auto dst = CLIO_IPC->ToFullPtr<char>(r.data_.template Cast<char>());
        if (dst.ptr_ == nullptr) {
          region_ok = false;
          return false;
        }
        std::memcpy(dst.ptr_, original + r.blob_off_, n);
        return true;
      });
  return region_ok;
}

/**
 * Append a decompression observation to the trace (obs_dtime_ms column).
 */
void Runtime::WriteDecompressTraceRow(const std::string &tag,
                                      const std::string &blob,
                                      clio::run::u64 original_size,
                                      const std::string &lib,
                                      double dtime_ms) {
  if (!trace_file_.is_open()) {
    return;
  }
  // 28 columns: ts,node,tag,blob,size | 9 empty | chosen_lib | 11 empty |
  // obs_dtime_ms | select_ms(empty).
  std::vector<std::string> f = {std::to_string(TraceNowMs()),
                                std::to_string(container_id_), tag, blob,
                                std::to_string(original_size)};
  f.insert(f.end(), 9, "");
  f.push_back(lib);
  f.insert(f.end(), 11, "");
  f.push_back(TraceNum(dtime_ms));
  f.push_back("");
  std::lock_guard<std::mutex> lock(trace_lock_);
  trace_file_ << JoinCsv(f);
  trace_file_.flush();
}

clio::run::TaskResume Runtime::GetBlobSize(
    clio::run::shared_ptr<clio::cte::core::GetBlobSizeTask> &task) {
  CLIO_TASK_BODY_BEGIN

  // Forward to core to get the blob size (at the owner when it is remote,
  // at the node dtschedule placed it on for placed chunks).
  uint32_t size_node = CLIO_IPC->GetNodeId();
  const bool placed_size = IsPlaced(task->blob_name_.str());
  if (placed_size) {
    if (!copy_client_) {
      copy_client_ = std::make_unique<clio::cte::core::Client>(
          config_.core_pool_id_.IsNull() ? clio::cte::core::kCtePoolId
                                         : config_.core_pool_id_);
    }
    CLIO_CO_AWAIT(ResolveLocation(task->tag_id_, task->blob_name_.str(),
                                  &size_node));
    auto sz = copy_client_->AsyncGetBlobSize(task->tag_id_,
                                             task->blob_name_.str(),
                                             NodeQuery(size_node),
                                             task->replica_);
    CLIO_CO_AWAIT(sz);
    task->size_ = sz->size_;
    task->lost_bytes_ = sz->lost_bytes_;
    task->SetReturnCode(sz->GetReturnCode());
  } else if (OwnerIsLocal(task->tag_id_, task->blob_name_.str())) {
    CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kGetBlobSize,
                                task.template Cast<clio::run::Task>()));
  } else {
    EnsureCoreClient();
    auto sz = core_client_->AsyncGetBlobSize(task->tag_id_,
                                             task->blob_name_.str(),
                                             clio::run::PoolQuery::Dynamic(),
                                             task->replica_);
    CLIO_CO_AWAIT(sz);
    task->size_ = sz->size_;
    task->lost_bytes_ = sz->lost_bytes_;
    task->SetReturnCode(sz->GetReturnCode());
  }

  // If blob is compressed, query the header to return the original size
  if (task->GetReturnCode() == 0 && task->size_ >= sizeof(compressor::CompressionHeader)) {
    EnsureCoreClient();

    // Fetch header to determine if blob is compressed and get original size
    auto hdr_buf = CLIO_IPC->AllocateBuffer(sizeof(compressor::CompressionHeader));
    if (!hdr_buf.IsNull()) {
      clio::cte::core::Client *hc =
          placed_size ? copy_client_.get() : core_client_.get();
      auto get = hc->AsyncGetBlob(
          task->tag_id_, task->blob_name_.str(), 0,
          sizeof(compressor::CompressionHeader),
          /*flags=*/0, hdr_buf.shm_.template Cast<void>(),
          placed_size ? NodeQuery(size_node) : clio::run::PoolQuery::Dynamic());
      CLIO_CO_AWAIT(get);
      if (get->GetReturnCode() == 0 &&
          (get->context_.transform_flags_ &
           clio::cte::core::kBlobTransformCompressed)) {
        const auto *header =
            reinterpret_cast<const compressor::CompressionHeader *>(hdr_buf.ptr_);
        if (header->IsValid()) {
          task->size_ = header->original_size_;
        }
      }
      CLIO_IPC->FreeBuffer(hdr_buf);
    }
  }

  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::MultiPutBlob(
    clio::run::shared_ptr<clio::cte::core::MultiPutBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN

  // Forward untouched to core. Phase 3+ will apply compression to each record
  // in the batch individually.
  CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kMultiPutBlob,
                              task.template Cast<clio::run::Task>()));

  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::PollNodeLoad(
    clio::run::shared_ptr<PollNodeLoadTask> &task) {
  CLIO_TASK_BODY_BEGIN

  // Sample local node load: CPU% and worker queue
  // GetCpuTimes() returns a struct with 8 fields:
  // user, nice, system, idle, iowait, irq, softirq, steal
  static ctp::CpuTimes prev_times;
  auto curr_times = ctp::SystemInfo::GetCpuTimes();
  float cpu_util = ctp::SystemInfo::ComputeCpuUtilization(prev_times, curr_times);
  prev_times = curr_times;

  uint64_t ts_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                       .count();

  // Create result (NodeLoadSample will be set in the task output)
  // Phase 3+: also store in the load ring for load-aware decisions
  uint32_t local_node_id = CLIO_IPC->GetNodeId();
  NodeLoadSample sample(ts_ms, cpu_util, 0);  // queued_tasks=0 for now
  CLIO_CO_AWAIT(CollectTierRemaining(&sample.tier_remaining_));
  StoreLoadSample(local_node_id, sample);

  task->result_ = sample;

  task->SetReturnCode(0);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::RegisterConsumer(
    clio::run::shared_ptr<RegisterConsumerTask> &task) {
  CLIO_TASK_BODY_BEGIN

  // Phase 4: store (tag, consumer_node) mapping in tag_consumers_
  // Every container gets the Broadcast, so all see the same set of consumers.
  // Keep a ring of last 32 nodes per tag.
  std::string tag_key = task->tag_id_.ToString();
  {
    std::lock_guard<std::mutex> lock(consumer_lock_);
    auto &consumer_nodes = tag_consumers_[tag_key];
    std::lock_guard<std::mutex> inner_lock(consumer_nodes.lock_);
    // Keep only the last 32 nodes; remove oldest when full
    if (consumer_nodes.nodes_.size() >= 32) {
      consumer_nodes.nodes_.pop_front();
    }
    consumer_nodes.nodes_.push_back(task->consumer_node_);
    {
      std::lock_guard<std::mutex> stats_lock(stats_lock_);
      stats_.consumer_tags_++;
    }
    HLOG(kDebug, "dtschedule: RegisterConsumer registered (tag={}, node={})",
         task->tag_id_.ToString(), task->consumer_node_);
  }

  task->SetReturnCode(0);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

std::optional<std::string> Runtime::CachedTagName(const std::string &tag_key) {
  /**
   * Look up the resolved name of a tag.
   *
   * @param tag_key TagId::ToString() of the tag
   * @return The cached name ("" when the lookup missed), or nullopt when the
   *         tag was never resolved
   */
  std::lock_guard<std::mutex> lock(tag_names_lock_);
  auto it = tag_names_.find(tag_key);
  if (it == tag_names_.end()) {
    return std::nullopt;
  }
  return it->second;
}

void Runtime::CacheTagName(const std::string &tag_key,
                           const std::string &name) {
  /**
   * Remember a tag's resolved name (bounded: the map is cleared when it
   * grows past kMaxTagNames so a long-lived runtime cannot leak).
   *
   * @param tag_key TagId::ToString() of the tag
   * @param name Resolved name, or "" to cache a miss
   */
  constexpr size_t kMaxTagNames = 1 << 16;
  std::lock_guard<std::mutex> lock(tag_names_lock_);
  if (tag_names_.size() >= kMaxTagNames) {
    tag_names_.clear();
  }
  tag_names_[tag_key] = name;
}

bool Runtime::IsChunkIndexName(const std::string &blob_name) {
  /**
   * Whether a blob name is a bare chunk index (what the filesystem chimod
   * names a file's blobs), i.e. non-empty and all decimal digits.
   *
   * @param blob_name Blob name to test
   * @return true for names such as "0" or "17"
   */
  if (blob_name.empty()) {
    return true;
  }
  return std::all_of(blob_name.begin(), blob_name.end(),
                     [](unsigned char c) { return std::isdigit(c) != 0; });
}

bool Runtime::NeedsTagName() const {
  /**
   * Whether any policy keyed by file name is active: the DAG spec or a QoS
   * stage override. Without one the resolve round trip is skipped.
   *
   * @return true when PutBlob should resolve the tag's name
   */
  if (config_.workflow_aware_ == "dag" && dag_spec_ && dag_spec_->IsLoaded()) {
    return true;
  }
  return !config_.stages_.empty() || !config_.lossy_allowlist_.empty();
}

bool Runtime::IsConsumerRegistered(const TagId &tag_id) {
  /**
   * Check if this container has already registered as a consumer for a tag.
   *
   * Used in GetBlob to deduplicate RegisterConsumer sends. A tag is registered
   * when its entry is in tag_consumers_, which is populated by
   * RegisterConsumer via Broadcast (every container gets a copy).
   *
   * @param tag_id Tag to check
   * @return true if consumer registration already exists
   */
  std::string tag_key = tag_id.ToString();
  std::lock_guard<std::mutex> lock(consumer_lock_);
  auto it = tag_consumers_.find(tag_key);
  if (it == tag_consumers_.end()) {
    return false;
  }
  std::lock_guard<std::mutex> inner_lock(it->second.lock_);
  return !it->second.nodes_.empty();
}

clio::run::TaskResume Runtime::CompressAt(
    clio::run::shared_ptr<CompressAtTask> &task) {
  CLIO_TASK_BODY_BEGIN
  // Phase 4c: Compress a blob at consumer node C (scenario 3).
  // (a) Write raw cache copy locally at C with Local() on core pool.
  // (b) Compress with the task's lib/preset via CompressWithDecision.
  // (c) Put compressed result at owner through compressed_client_.
  // (d) Register C as copy holder with AsyncRegisterReplicaContainer.
  // (e) Clean up on failure; return owner's rc + OUT context.

  uint64_t tag_version = task->context_.version_;
  uint32_t C = CLIO_IPC->GetNodeId();  // This runs at consumer C

  // (a) Write raw cache copy locally at C using core pool with Local query.
  if (!copy_client_) {
    copy_client_ = std::make_unique<clio::cte::core::Client>(
        config_.core_pool_id_.IsNull() ? clio::cte::core::kCtePoolId
                                       : config_.core_pool_id_);
  }

  auto cache_start = std::chrono::steady_clock::now();
  clio::cte::core::Context cache_ctx;
  cache_ctx.replica_ = clio::cte::core::kCacheReplica;
  cache_ctx.replica_flags_ |= clio::cte::core::REPLICA_CACHE;
  cache_ctx.transform_flags_ = 0;
  cache_ctx.min_persistence_level_ = 0;
  cache_ctx.version_ = tag_version;

  // The raw copy at C is a read accelerator, not the data: keep it only
  // while C's fastest tier has room, and never fail the put over it. Under
  // memory pressure C just compresses and the bytes go down the tiers.
  bool raw_cached = false;
  const bool placed_here = IsPlaced(task->blob_name_.str());
  if (!placed_here && !config_.tiers_.empty() &&
      ChooseTier(task->size_, C) == config_.tiers_.front().name_) {
    ChooseTier(task->size_, C, /*reserve=*/true);
    auto cache_put = copy_client_->AsyncPutBlob(
        task->tag_id_, task->blob_name_.str(), 0, task->size_,
        task->blob_data_, config_.tiers_.front().score_, cache_ctx,
        /*flags=*/0, clio::run::PoolQuery::Local());
    CLIO_CO_AWAIT(cache_put);
    raw_cached = cache_put->GetReturnCode() == 0;
  }
  double cache_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - cache_start)
                        .count();

  // (b) Compress raw data with the task's codec.
  auto src_ptr = CLIO_IPC->ToFullPtr<char>(task->blob_data_.template Cast<char>());
  if (src_ptr.ptr_ == nullptr) {
    task->SetReturnCode(EFAULT);
    CLIO_CO_RETURN;
  }

  // Build a Decision from task's lib/preset for CompressWithDecision.
  ccm::Decision decision;
  decision.chosen_lib_ = task->lib_;
  // Convert preset string back to enum
  ctp::CompressionPreset preset = ctp::CompressionPreset::BALANCED;
  if (task->preset_ == "fast") {
    preset = ctp::CompressionPreset::FAST;
  } else if (task->preset_ == "balanced") {
    preset = ctp::CompressionPreset::BALANCED;
  } else if (task->preset_ == "best") {
    preset = ctp::CompressionPreset::BEST;
  } else if (task->preset_ == "default") {
    preset = ctp::CompressionPreset::DEFAULT;
  }
  decision.chosen_preset_ = preset;
  // Use default pred_ratio/ctime/dtime since we're actually compressing.
  decision.pred_ratio_ = 1.0;
  decision.pred_ctime_ms_ = 0.0;
  decision.pred_dtime_ms_ = 0.0;

  auto comp_start = std::chrono::steady_clock::now();
  CompressOutcome out{};
  ctp::ipc::FullPtr<char> comp_buf = CompressWithDecision(
      src_ptr.ptr_, task->size_, decision, &out);
  double comp_ms = std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - comp_start)
                       .count();

  if (!out.used) {
    // Compression did not pay: the raw bytes are the primary, at the owner
    // (the cache copy above is only a replica).
    auto raw_start = std::chrono::steady_clock::now();
    auto raw_put = copy_client_->AsyncPutBlob(
        task->tag_id_, task->blob_name_.str(), 0, task->size_,
        task->blob_data_, task->score_, task->context_, /*flags=*/0,
        placed_here ? clio::run::PoolQuery::Local()
                    : clio::run::PoolQuery::Dynamic());
    CLIO_CO_AWAIT(raw_put);
    task->comp_size_ = task->size_;
    task->ctime_ms_ = comp_ms;
    task->store_ms_ = cache_ms + std::chrono::duration<double, std::milli>(
                                     std::chrono::steady_clock::now() -
                                     raw_start).count();
    task->context_ = raw_put->context_;
    {
      std::lock_guard<std::mutex> lock(stats_lock_);
      stats_.s3_local_copies_++;
    }
    task->SetReturnCode(raw_put->GetReturnCode());
    CLIO_CO_RETURN;
  }

  // (c) Put compressed data at owner with the compressed flag set.
  if (!compressed_client_) {
    compressed_client_ = std::make_unique<clio::cte::core::Client>(
        config_.compressed_next_pool_id_.IsNull() ? clio::cte::core::kCtePoolId
                                                  : config_.compressed_next_pool_id_);
  }

  auto store_start = std::chrono::steady_clock::now();
  clio::cte::core::Context comp_ctx = task->context_;
  comp_ctx.transform_flags_ |= clio::cte::core::kBlobTransformCompressed;
  comp_ctx.version_ = tag_version;

  size_t comp_total_size = sizeof(compressor::CompressionHeader) + out.comp_size;
  // dtschedule-placed chunks are stored here, on the consumer's node (the
  // producer records the location); others go to their hash owner.
  auto owner_put = compressed_client_->AsyncPutBlob(
      task->tag_id_, task->blob_name_.str(), 0, comp_total_size,
      comp_buf.shm_.template Cast<void>(), task->score_, comp_ctx,
      /*flags=*/0,
      placed_here ? clio::run::PoolQuery::Local()
                  : clio::run::PoolQuery::Dynamic());
  CLIO_CO_AWAIT(owner_put);
  double store_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - store_start)
                        .count();

  if (owner_put->GetReturnCode() != 0) {
    // Owner put failed; drop the local raw copy if one was kept.
    if (raw_cached) {
      auto del = copy_client_->AsyncDelBlob(
          task->tag_id_, task->blob_name_.str(), clio::run::PoolQuery::Local(),
          clio::cte::core::kDelCacheCopyOnly);
      CLIO_CO_AWAIT(del);
    }
    task->SetReturnCode(owner_put->GetReturnCode());
    CLIO_CO_RETURN;
  }

  // (d) Register C as copy holder at the owner (only when it holds one; a
  // failed registration just means reads go to the owner).
  if (raw_cached) {
    auto reg = copy_client_->AsyncRegisterReplicaContainer(
        task->tag_id_, task->blob_name_.str(), C,
        clio::run::PoolQuery::Dynamic(), tag_version);
    CLIO_CO_AWAIT(reg);
    if (reg->GetReturnCode() != 0) {
      auto del = copy_client_->AsyncDelBlob(
          task->tag_id_, task->blob_name_.str(), clio::run::PoolQuery::Local(),
          clio::cte::core::kDelCacheCopyOnly);
      CLIO_CO_AWAIT(del);
    }
  }

  // (e) Return owner's result and OUT fields.
  task->SetReturnCode(owner_put->GetReturnCode());
  task->context_ = owner_put->context_;
  task->comp_size_ = out.comp_size;
  task->ctime_ms_ = comp_ms;
  task->store_ms_ = cache_ms + store_ms;

  {
    std::lock_guard<std::mutex> lock(stats_lock_);
    stats_.s3_local_copies_++;
  }

  if (!comp_buf.IsNull()) {
    CLIO_IPC->FreeBuffer(comp_buf);
  }

  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::GetDecisionStats(
    clio::run::shared_ptr<GetDecisionStatsTask> &task) {
  CLIO_TASK_BODY_BEGIN

  {
    std::lock_guard<std::mutex> lock(stats_lock_);
    task->result_ = stats_;
    task->result_.dag_files_ = dag_files_;
    task->result_.dag_nodes_ = dag_nodes_;

    // Count number of nodes with fresh load samples (age <= 5*load_period_ms)
    uint64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::system_clock::now().time_since_epoch())
                          .count();
    uint64_t stale_threshold_ms = 5ULL * config_.load_period_ms_;
    uint32_t fresh_sample_count = 0;
    {
      std::lock_guard<std::mutex> rings_lock(load_rings_lock_);
      for (auto &[node_id, ring] : load_rings_) {
        std::lock_guard<std::mutex> ring_lock(ring.lock_);
        if (!ring.samples_.empty()) {
          const auto &latest = ring.samples_.back();
          uint64_t age_ms = (now_ms >= latest.ts_ms_) ? (now_ms - latest.ts_ms_) : 0;
          if (age_ms <= stale_threshold_ms) {
            fresh_sample_count++;
          }
        }
      }
    }
    task->result_.load_samples_ = fresh_sample_count;
  }

  task->SetReturnCode(0);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::SetKnobs(
    clio::run::shared_ptr<SetKnobsTask> &task) {
  CLIO_TASK_BODY_BEGIN

  {
    std::lock_guard<std::mutex> lock(knobs_lock_);
    // **Phase 2 Deliverable 4: Switch CCM predictor at runtime**
    std::string old_ccm = knobs_.ccm_;
    knobs_ = task->knobs_;

    // If CCM spec changed, update the predictor
    if (ccm_manager_ && knobs_.ccm_ != old_ccm) {
      std::string model_dir = config_.qtable_model_path_;
      if (model_dir.empty()) {
        model_dir = DTSCHEDULE_MODEL_DIR;
      }
      if (ccm_manager_->SetPredictor(knobs_.ccm_, model_dir)) {
        HLOG(kInfo, "dtschedule: CCM predictor switched to {}", knobs_.ccm_);
      } else {
        HLOG(kWarning, "dtschedule: CCM predictor switch failed, reverting to {}",
             old_ccm);
        knobs_.ccm_ = old_ccm;
      }
    }
  }

  HLOG(kInfo, "dtschedule: Knobs updated (ccm={}, load_aware={})",
       knobs_.ccm_, knobs_.load_aware_);

  task->SetReturnCode(0);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::SampleLoad(
    clio::run::shared_ptr<SampleLoadTask> &task) {
  CLIO_TASK_BODY_BEGIN

  // Sample local node CPU% using delta from previous sample
  auto curr_times = ctp::SystemInfo::GetCpuTimes();
  double cpu_util = ctp::SystemInfo::ComputeCpuUtilization(prev_cpu_times_, curr_times);
  prev_cpu_times_ = curr_times;

  uint64_t ts_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                       .count();

  // Get local node ID (all nodes in the local pool have the same node id)
  uint32_t local_node_id = CLIO_IPC->GetNodeId();

  // Store sample in the ring buffer for this node
  NodeLoadSample sample(ts_ms, cpu_util, 0);  // queued_tasks=0 for phase 3
  CLIO_CO_AWAIT(CollectTierRemaining(&sample.tier_remaining_));
  StoreLoadSample(local_node_id, sample);

  HLOG(kDebug, "dtschedule: SampleLoad node={} cpu={:.1f}%",
       local_node_id, cpu_util);

  // Phase 3+: Poll remote consumer nodes for load information
  // Collect all consumer nodes from tag_consumers_ + load_peers
  std::set<uint32_t> poll_nodes;
  {
    std::lock_guard<std::mutex> lock(consumer_lock_);
    for (auto &[tag, consumer_nodes] : tag_consumers_) {
      std::lock_guard<std::mutex> inner_lock(consumer_nodes.lock_);
      for (uint32_t node : consumer_nodes.nodes_) {
        poll_nodes.insert(node);
      }
    }
  }
  // Add extra load_peers from config (useful for tests)
  for (uint32_t peer : config_.load_peers_) {
    poll_nodes.insert(peer);
  }
  // With several tiers, every node may own a blob: poll them all so tier
  // choice sees each owner's free bytes.
  if (config_.tiers_.size() > 1) {
    auto *pool_manager = CLIO_POOL_MANAGER;
    const clio::run::PoolInfo *info = pool_manager->GetPoolInfo(pool_id_);
    const uint32_t n = info == nullptr ? 0 : info->num_containers_;
    for (uint32_t c = 0; c < n; ++c) {
      poll_nodes.insert(static_cast<uint32_t>(
          pool_manager->GetContainerNodeId(pool_id_, c)));
    }
  }

  // Poll each remote node (bounded; skip nodes whose previous poll never returned)
  auto *ipc_manager = CLIO_IPC;
  for (uint32_t remote_node : poll_nodes) {
    if (remote_node == local_node_id) {
      continue;  // Skip self
    }
    auto poll_task = ipc_manager->NewTask<PollNodeLoadTask>(
        clio::run::CreateTaskId(), pool_id_,
        clio::run::PoolQuery::Physical(remote_node));
    if (!poll_task.IsNull()) {
      // Send and await the poll task
      CLIO_CO_AWAIT(ipc_manager->Send(
          poll_task.template Cast<clio::run::Task>()));
      if (poll_task->GetReturnCode() == 0) {
        // Store the remote load sample
        StoreLoadSample(remote_node, poll_task->result_);
      }
    }
  }
  CLIO_CO_AWAIT(DemoteTick());

  task->SetReturnCode(0);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

void Runtime::OpenTraceFile() {
  if (config_.trace_path_.empty()) {
    return;
  }

  std::string trace_file =
      config_.trace_path_ + "." + std::to_string(container_id_) + ".csv";
  {
    std::lock_guard<std::mutex> lock(trace_lock_);
    trace_file_.open(trace_file, std::ios::app);
    if (trace_file_.is_open()) {
      // Write header row if file is new (empty)
      trace_file_.seekp(0, std::ios::end);
      if (trace_file_.tellp() == 0) {
        trace_file_ << "ts_ms,node,tag,blob,size,dtype,entropy,mad,d2,"
                       "producer_cpu,consumer_node,consumer_cpu,owner_node,"
                       "n_candidates,chosen_lib,chosen_preset,chosen_scenario,"
                       "chosen_tier,pred_ctime_ms,pred_dtime_ms,pred_ratio,"
                       "obs_ctime_ms,obs_ratio,store_ms,forced,knobs_hash,"
                       "obs_dtime_ms,select_ms\n";
        trace_file_.flush();
      }
      HLOG(kDebug, "dtschedule: Trace file opened: {}", trace_file);
    }
  }

  // Also open candidates trace file (task 3)
  OpenCandidatesTraceFile();
}

void Runtime::OpenCandidatesTraceFile() {
  if (config_.trace_path_.empty() || !config_.trace_candidates_) {
    return;
  }

  std::string cand_file =
      config_.trace_path_ + ".cand." + std::to_string(container_id_) + ".csv";
  {
    std::lock_guard<std::mutex> lock(trace_lock_);
    candidates_trace_file_.open(cand_file, std::ios::app);
    if (candidates_trace_file_.is_open()) {
      // Write header row if file is new (empty)
      candidates_trace_file_.seekp(0, std::ios::end);
      if (candidates_trace_file_.tellp() == 0) {
        candidates_trace_file_ << "ts_ms,node,tag,blob,lib,preset,pred_ctime_ms,"
                                  "pred_dtime_ms,pred_ratio,cost_ms,reason\n";
        candidates_trace_file_.flush();
      }
      HLOG(kDebug, "dtschedule: Candidates trace file opened: {}", cand_file);
    }
  }
}

void Runtime::CloseTraceFile() {
  std::lock_guard<std::mutex> lock(trace_lock_);
  if (trace_file_.is_open()) {
    trace_file_.flush();
    trace_file_.close();
  }
  if (candidates_trace_file_.is_open()) {
    candidates_trace_file_.flush();
    candidates_trace_file_.close();
  }
}

/** Join fields into one CSV line. */
static std::string JoinCsv(const std::vector<std::string> &fields) {
  std::string line;
  for (size_t i = 0; i < fields.size(); ++i) {
    if (i) line += ',';
    line += fields[i];
  }
  return line + "\n";
}

/** Format a double for the trace (empty string when negative = unset). */
static std::string TraceNum(double v) {
  if (v < 0.0) return "";
  std::ostringstream os;
  os << v;
  return os.str();
}

/** Milliseconds since the epoch, as the trace timestamp. */
static uint64_t TraceNowMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

void Runtime::WriteTraceRow(const std::string &tag_id,
                            const std::string &blob_name,
                            size_t original_size, const ccm::Features &features,
                            const ccm::Decision &decision,
                            const CompressOutcome &out, double select_ms,
                            double load_mult, double producer_cpu,
                            const Placement &place) {
  if (!trace_file_.is_open()) {
    return;
  }
  const bool chose = !decision.chosen_lib_.empty();

  // Build forced column: stage:<i> and/or lm:<value>
  std::string forced;
  if (decision.qos_stage_index_ >= 0) {
    forced = "stage:" + std::to_string(decision.qos_stage_index_);
  }
  // Always include load multiplier in forced column (phase 3+)
  if (!forced.empty()) {
    forced += ";";
  }
  forced += "lm:" + TraceNum(load_mult);
  if (place.cost1_ms >= 0.0) {
    forced += ";c1:" + TraceNum(place.cost1_ms) + ";c2:" + TraceNum(place.cost2_ms);
    if (place.cost3_ms >= 0.0) {
      forced += ";c3:" + TraceNum(place.cost3_ms);
    }
  }
  if (!place.order.empty()) {
    forced += ";order:" + place.order;
  }
  if (place.dag_hit) {
    forced += ";dag:1";
  }
  if (place.fanout >= 0) {
    forced += ";fanout:" + std::to_string(place.fanout);
  }

  std::vector<std::string> f = {
      std::to_string(TraceNowMs()), std::to_string(container_id_), tag_id,
      blob_name, std::to_string(original_size),
      std::to_string(features.dtype_), TraceNum(features.entropy_),
      TraceNum(features.mad_), TraceNum(features.d2_),
      TraceNum(producer_cpu),                          // producer_cpu (phase 3+)
      place.consumer_node == UINT32_MAX ? "" : std::to_string(place.consumer_node),
      place.consumer_cpu < 0.0 ? "" : TraceNum(place.consumer_cpu),
      place.owner_node == UINT32_MAX ? "" : std::to_string(place.owner_node),
      std::to_string(decision.n_candidates_),
      out.used ? decision.chosen_lib_ : "raw",
      chose ? std::to_string(compressor::ToWirePreset(decision.chosen_preset_))
            : "",
      std::to_string(place.scenario),
      place.tier.empty() ? decision.chosen_tier_ : place.tier,
      chose ? TraceNum(decision.pred_ctime_ms_) : "",
      chose ? TraceNum(decision.pred_dtime_ms_) : "",
      chose ? TraceNum(decision.pred_ratio_) : "",
      out.attempted ? TraceNum(out.ctime_ms) : "",
      TraceNum(out.used ? out.ratio : 1.0),
      TraceNum(place.store_ms),
      forced,
      std::to_string(knobs_.Hash()),
      "",                                               // obs_dtime_ms
      TraceNum(select_ms)};
  std::lock_guard<std::mutex> lock(trace_lock_);
  trace_file_ << JoinCsv(f);
  trace_file_.flush();
}

void Runtime::WriteCandidatesTrace(const std::string &tag_id,
                                   const std::string &blob_name,
                                   const ccm::Decision &decision) {
  if (!candidates_trace_file_.is_open() || decision.candidates_.empty()) {
    return;
  }
  const std::string ts = std::to_string(TraceNowMs());
  std::lock_guard<std::mutex> lock(trace_lock_);
  for (const auto &c : decision.candidates_) {
    const bool ranked = c.reason_ == ccm::kReasonOk;
    candidates_trace_file_ << JoinCsv(
        {ts, std::to_string(container_id_), tag_id, blob_name, c.lib_,
         std::to_string(compressor::ToWirePreset(c.preset_)),
         ranked || c.reason_ == ccm::kReasonSkipRatio ? TraceNum(c.pred_ctime_ms_) : "",
         ranked || c.reason_ == ccm::kReasonSkipRatio ? TraceNum(c.pred_dtime_ms_) : "",
         ranked || c.reason_ == ccm::kReasonSkipRatio ? TraceNum(c.pred_ratio_) : "",
         ranked ? TraceNum(c.cost_) : "", c.reason_});
  }
  candidates_trace_file_.flush();
}

void Runtime::StoreLoadSample(uint32_t node_id, const NodeLoadSample &sample) {
  std::lock_guard<std::mutex> outer_lock(load_rings_lock_);
  auto &ring = load_rings_[node_id];
  {
    std::lock_guard<std::mutex> lock(ring.lock_);
    // Keep only the last 16 samples; remove oldest when full
    if (ring.samples_.size() >= 16) {
      ring.samples_.pop_front();
    }
    ring.samples_.push_back(sample);
  }
  if (!sample.tier_remaining_.empty()) {
    std::lock_guard<std::mutex> lock(tier_cap_lock_);
    tier_reserved_.erase(node_id);  // the sample already counts those bytes
    uint64_t &cap = tier0_capacity_[node_id];
    cap = std::max(cap, sample.tier_remaining_.front());
  }
}

std::tuple<double, uint64_t, bool> Runtime::GetLoad(uint32_t node_id) {
  uint64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch())
                        .count();

  std::lock_guard<std::mutex> outer_lock(load_rings_lock_);
  auto it = load_rings_.find(node_id);
  if (it == load_rings_.end() || it->second.samples_.empty()) {
    // No sample for this node; return default (cpu=1.0, age=infinite, unknown)
    return {1.0, UINT64_MAX, false};
  }

  LoadRing &ring = it->second;
  std::lock_guard<std::mutex> lock(ring.lock_);
  if (ring.samples_.empty()) {
    return {1.0, UINT64_MAX, false};
  }

  const NodeLoadSample &latest = ring.samples_.back();
  uint64_t age_ms = (now_ms >= latest.ts_ms_) ? (now_ms - latest.ts_ms_) : 0;

  // Clamp CPU% to [0, 100]
  double cpu_pct = std::max(0.0, std::min(100.0, latest.cpu_util_));

  return {cpu_pct, age_ms, true};
}

double Runtime::LoadMultiplier(double cpu_a, double cpu_b,
                               bool load_aware, double load_cap) {
  // When load awareness is off, always return 1.0
  if (!load_aware) {
    return 1.0;
  }

  // Both CPUs must be clamped at 1.0 when unknown/stale
  // (This is handled by the caller using GetLoad and checking the 'known' flag)

  // m_X = clamp(cpu_X / max(cpu_other, 1), 1/load_cap, load_cap)
  double divisor = std::max(1.0, cpu_b);
  double mult = cpu_a / divisor;

  // Clamp to [1/load_cap, load_cap]
  double min_mult = 1.0 / load_cap;
  double mult_clamped = std::max(min_mult, std::min(load_cap, mult));

  return mult_clamped;
}

uint32_t Runtime::OwnerNode(const TagId &tag_id, const std::string &blob_name) {
  // Same hash as the core's HashBlobToContainer and the cache's
  // IsBlobOwnerLocal, resolved against this pool's container map (every
  // pool in the chain shares the layout).
  auto *pool_manager = CLIO_POOL_MANAGER;
  const clio::run::PoolInfo *info = pool_manager->GetPoolInfo(pool_id_);
  if (info == nullptr || info->num_containers_ == 0) {
    return CLIO_IPC->GetNodeId();
  }
  std::hash<std::string> string_hasher;
  std::hash<clio::run::u32> u32_hasher;
  clio::run::u32 h = u32_hasher(tag_id.major_);
  h ^= u32_hasher(tag_id.minor_) + 0x9e3779b9 + (h << 6) + (h >> 2);
  h ^= static_cast<clio::run::u32>(string_hasher(blob_name)) + 0x9e3779b9 +
       (h << 6) + (h >> 2);
  clio::run::ContainerId cid = h % info->num_containers_;
  if (pool_manager->HasContainer(pool_id_, cid)) {
    return CLIO_IPC->GetNodeId();
  }
  return pool_manager->GetContainerNodeId(pool_id_, cid);
}

uint32_t Runtime::PickConsumer(const TagId &tag_id) {
  /**
   * Get the most recent consumer node for a tag, or UINT32_MAX if unknown.
   *
   * Phase 4: used by decision logic to pick the consumer node for scenario 2/3.
   * Returns the most recently registered node (last in the deque).
   *
   * @param tag_id Tag to query
   * @return Most recent consumer node ID, or UINT32_MAX if unknown
   */
  std::string tag_key = tag_id.ToString();
  std::lock_guard<std::mutex> lock(consumer_lock_);
  auto it = tag_consumers_.find(tag_key);
  if (it == tag_consumers_.end()) {
    return UINT32_MAX;
  }
  std::lock_guard<std::mutex> inner_lock(it->second.lock_);
  if (it->second.nodes_.empty()) {
    return UINT32_MAX;
  }
  return it->second.nodes_.back();  // Most recent (last) node
}

std::string Runtime::ChooseTier(uint64_t blob_size, uint32_t owner_node,
                                bool reserve) {
  if (config_.tiers_.empty()) {
    return "";
  }
  std::vector<uint64_t> free_bytes;
  {
    std::lock_guard<std::mutex> outer(load_rings_lock_);
    auto it = load_rings_.find(owner_node);
    if (it != load_rings_.end()) {
      std::lock_guard<std::mutex> lock(it->second.lock_);
      if (!it->second.samples_.empty()) {
        free_bytes = it->second.samples_.back().tier_remaining_;
      }
    }
  }
  if (free_bytes.size() != config_.tiers_.size()) {
    return config_.tiers_.front().name_;  // no capacity known yet
  }
  std::lock_guard<std::mutex> lock(tier_cap_lock_);
  auto &reserved = tier_reserved_[owner_node];
  reserved.resize(config_.tiers_.size(), 0);
  size_t pick = config_.tiers_.size() - 1;
  for (size_t i = 0; i < free_bytes.size(); ++i) {
    const uint64_t avail =
        free_bytes[i] > reserved[i] ? free_bytes[i] - reserved[i] : 0;
    if (avail >= blob_size) {
      pick = i;
      break;
    }
  }
  if (reserve) {
    reserved[pick] += blob_size;
  }
  return config_.tiers_[pick].name_;
}

double Runtime::StoreBwFor(uint64_t bytes, uint32_t owner_node,
                           uint32_t consumer_node) {
  const uint32_t self = CLIO_IPC->GetNodeId();
  const uint32_t owner = owner_node == UINT32_MAX ? self : owner_node;
  double bw = TierBwMbPerMs(ChooseTier(bytes, owner));
  const bool crosses = owner != self ||
                       (consumer_node != UINT32_MAX && consumer_node != owner);
  if (crosses) {
    bw = std::min(bw, std::max(config_.net_bw_gbps_, 0.01) / 8.0);
  }
  return bw;
}

uint32_t Runtime::BestStoreNode(uint64_t bytes) {
  /**
   * Node whose storage takes `bytes` cheapest: the time to store them on
   * the tier they would land in there, plus the network transfer when the
   * node is not this one. This (writer) node wins ties, so data stays local
   * until local fast tiers fill; then an idle node's RAM beats a local HDD.
   * The chosen tier on the chosen node is reserved.
   *
   * @param bytes Bytes to store
   * @return Node id
   */
  const uint32_t self = CLIO_IPC->GetNodeId();
  if (config_.placement_ == "local") {
    ChooseTier(bytes, self, /*reserve=*/true);
    return self;  // baseline: always the writer's node
  }
  std::vector<uint32_t> nodes;
  {
    std::lock_guard<std::mutex> lock(load_rings_lock_);
    for (const auto &kv : load_rings_) nodes.push_back(kv.first);
  }
  const double mb = static_cast<double>(bytes) / 1e6;
  const double net = std::max(config_.net_bw_gbps_, 0.01) / 8.0;
  auto cost = [&](uint32_t node) {
    const double bw = std::max(TierBwMbPerMs(ChooseTier(bytes, node)), 1e-6);
    return mb / bw + (node == self ? 0.0 : mb / net);
  };
  uint32_t best = self;
  double best_cost = cost(self);
  for (uint32_t node : nodes) {
    if (node == self) continue;
    const double c = cost(node);
    if (c < best_cost * 0.9) {  // a clear win only
      best = node;
      best_cost = c;
    }
  }
  ChooseTier(bytes, best, /*reserve=*/true);
  return best;
}

clio::run::PoolQuery Runtime::NodeQuery(uint32_t node) const {
  /**
   * Query that executes on a given node.
   *
   * @param node Node id
   * @return Local() for this node, Physical(node) otherwise
   */
  if (node == CLIO_IPC->GetNodeId()) {
    return clio::run::PoolQuery::Local();
  }
  return clio::run::PoolQuery::Physical(node);
}

std::string Runtime::LocKey(const TagId &tag_id, const std::string &blob) {
  /**
   * Location-record key of a blob.
   *
   * @param tag_id Tag of the blob
   * @param blob Blob name
   * @return "<major>.<minor>/<blob>"
   */
  return tag_id.ToString() + "/" + blob;
}

clio::run::TaskResume Runtime::EnsureLocTag() {
  CLIO_TASK_BODY_BEGIN
  // Location records live in one CTE tag; the core hash-owns each record,
  // so there is no central directory.
  if (!loc_tag_ready_) {
    if (!copy_client_) {
      copy_client_ = std::make_unique<clio::cte::core::Client>(
          config_.core_pool_id_.IsNull() ? clio::cte::core::kCtePoolId
                                         : config_.core_pool_id_);
    }
    auto t = copy_client_->AsyncGetOrCreateTag("_dtschedule_loc");
    CLIO_CO_AWAIT(t);
    if (t->GetReturnCode() == 0) {
      loc_tag_ = t->tag_id_;
      loc_tag_ready_ = true;
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::WriteLocation(const TagId &tag_id,
                                             const std::string &blob,
                                             uint32_t node) {
  CLIO_TASK_BODY_BEGIN
  {
    const std::string key = LocKey(tag_id, blob);
    bool had_other = false;
    {
      std::lock_guard<std::mutex> lock(loc_lock_);
      auto it = loc_cache_.find(key);
      if (it != loc_cache_.end()) {
        if (it->second == node) {
          CLIO_CO_RETURN;  // already recorded here
        }
        had_other = true;
      }
    }
    // A chunk stored at its hash owner is found without a record unless an
    // earlier write recorded another node.
    if (node != OwnerNode(tag_id, blob) || had_other) {
      CLIO_CO_AWAIT(EnsureLocTag());
      if (!loc_tag_ready_) {
        CLIO_CO_RETURN;
      }
      const uint32_t value = node;
      auto put = copy_client_->AsyncPutBlob(
          loc_tag_, key, 0, sizeof(value),
          reinterpret_cast<const char *>(&value), -1.0f,
          clio::cte::core::Context(), 0, clio::run::PoolQuery::Dynamic());
      CLIO_CO_AWAIT(put);
      if (put->GetReturnCode() != 0) {
        HLOG(kWarning, "dtschedule: location record for {} failed ({})", key,
             put->GetReturnCode());
        CLIO_CO_RETURN;
      }
    }
    std::lock_guard<std::mutex> lock(loc_lock_);
    if (loc_cache_.size() >= (1u << 20)) {
      loc_cache_.clear();
    }
    loc_cache_[key] = node;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::ResolveLocation(const TagId &tag_id,
                                               const std::string &blob,
                                               uint32_t *node) {
  CLIO_TASK_BODY_BEGIN
  {
    const std::string key = LocKey(tag_id, blob);
    *node = OwnerNode(tag_id, blob);
    {
      std::lock_guard<std::mutex> lock(loc_lock_);
      auto it = loc_cache_.find(key);
      if (it != loc_cache_.end()) {
        *node = it->second;
        CLIO_CO_RETURN;
      }
    }
    CLIO_CO_AWAIT(EnsureLocTag());
    if (!loc_tag_ready_) {
      CLIO_CO_RETURN;
    }
    auto buf = CLIO_IPC->AllocateBuffer(sizeof(uint32_t));
    if (buf.IsNull()) {
      CLIO_CO_RETURN;
    }
    auto get = copy_client_->AsyncGetBlob(
        loc_tag_, key.c_str(), 0, sizeof(uint32_t), 0,
        buf.shm_.template Cast<void>(), clio::run::PoolQuery::Dynamic());
    CLIO_CO_AWAIT(get);
    if (get->GetReturnCode() == 0) {
      uint32_t value = 0;
      std::memcpy(&value, buf.ptr_, sizeof(value));
      *node = value;
      std::lock_guard<std::mutex> lock(loc_lock_);
      loc_cache_[key] = value;  // only positive answers are cached
    }
    CLIO_IPC->FreeBuffer(buf);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::PutAtNode(
    clio::run::shared_ptr<clio::cte::core::PutBlobTask> &task, uint32_t node) {
  CLIO_TASK_BODY_BEGIN
  {
    if (!copy_client_) {
      copy_client_ = std::make_unique<clio::cte::core::Client>(
          config_.core_pool_id_.IsNull() ? clio::cte::core::kCtePoolId
                                         : config_.core_pool_id_);
    }
    std::vector<clio::cte::core::BlobRegion> regions;
    clio::cte::core::ForEachBlobRegion(
        *task, [&regions](const clio::cte::core::BlobRegion &r) {
          regions.push_back(r);
          return true;
        });
    clio::run::u32 rc = 0;
    clio::cte::core::Context out_ctx = task->context_;
    for (size_t i = 0; i < regions.size() && rc == 0; ++i) {
      auto put = copy_client_->AsyncPutBlob(
          task->tag_id_, task->blob_name_.str(), regions[i].blob_off_,
          regions[i].size_, regions[i].data_, task->score_, task->context_,
          task->flags_, NodeQuery(node));
      CLIO_CO_AWAIT(put);
      rc = put->GetReturnCode();
      out_ctx = put->context_;
    }
    task->context_ = out_ctx;
    task->SetReturnCode(rc);
    if (rc == 0) {
      CLIO_CO_AWAIT(WriteLocation(task->tag_id_, task->blob_name_.str(), node));
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::GetAtNode(
    clio::run::shared_ptr<clio::cte::core::GetBlobTask> &task, uint32_t node) {
  CLIO_TASK_BODY_BEGIN
  {
    if (!copy_client_) {
      copy_client_ = std::make_unique<clio::cte::core::Client>(
          config_.core_pool_id_.IsNull() ? clio::cte::core::kCtePoolId
                                         : config_.core_pool_id_);
    }
    std::vector<clio::cte::core::BlobRegion> regions;
    clio::cte::core::ForEachBlobRegion(
        *task, [&regions](const clio::cte::core::BlobRegion &r) {
          regions.push_back(r);
          return true;
        });
    clio::run::u32 rc = 0;
    clio::run::u32 tflags = 0;
    clio::run::u64 version = 0;
    for (size_t i = 0; i < regions.size() && rc == 0; ++i) {
      auto get = copy_client_->AsyncGetBlob(
          task->tag_id_, task->blob_name_.str().c_str(), regions[i].blob_off_,
          regions[i].size_, task->flags_, regions[i].data_, NodeQuery(node),
          task->context_);
      CLIO_CO_AWAIT(get);
      rc = get->GetReturnCode();
      tflags = get->context_.transform_flags_;
      if (i == 0) version = get->context_.version_;
    }
    task->context_.transform_flags_ = tflags;
    task->context_.version_ = version;
    task->SetReturnCode(rc);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

void Runtime::EnqueueDemote(const TagId &tag_id, const std::string &blob,
                            uint32_t node, const char *data, size_t size) {
  /**
   * Keep a copy of a chunk this node just read for possible demotion. The
   * queue is bounded by demote_queue_mb; the oldest entries are dropped.
   *
   * @param tag_id Tag of the blob
   * @param blob Blob name
   * @param node Node holding the chunk
   * @param data Raw bytes read
   * @param size Byte count
   */
  const size_t cap = static_cast<size_t>(config_.demote_queue_mb_) << 20;
  if (size > cap) {
    return;
  }
  DemoteItem item;
  item.tag_id_ = tag_id;
  item.blob_ = blob;
  item.owner_ = node;
  item.data_.assign(data, data + size);
  std::lock_guard<std::mutex> lock(demote_lock_);
  while (!demote_q_.empty() && demote_q_bytes_ + size > cap) {
    demote_q_bytes_ -= demote_q_.front().data_.size();
    demote_q_.pop_front();
  }
  demote_q_bytes_ += size;
  demote_q_.push_back(std::move(item));
}

bool Runtime::OwnerUnderPressure(uint32_t owner) {
  /**
   * Whether the owner's fastest tier is short of room: its free bytes in
   * the latest sample (minus this container's reservations since) are below
   * demote_watermark x the largest free space ever seen there.
   *
   * @param owner Node id
   * @return true when demoting this owner's chunks would free needed room
   */
  uint64_t free_bytes = 0;
  {
    std::lock_guard<std::mutex> outer(load_rings_lock_);
    auto it = load_rings_.find(owner);
    if (it == load_rings_.end()) {
      return false;
    }
    std::lock_guard<std::mutex> lock(it->second.lock_);
    if (it->second.samples_.empty() ||
        it->second.samples_.back().tier_remaining_.empty()) {
      return false;
    }
    free_bytes = it->second.samples_.back().tier_remaining_.front();
  }
  std::lock_guard<std::mutex> lock(tier_cap_lock_);
  const uint64_t cap = tier0_capacity_[owner];
  auto res = tier_reserved_.find(owner);
  const uint64_t reserved =
      (res == tier_reserved_.end() || res->second.empty()) ? 0
                                                           : res->second[0];
  const uint64_t avail = free_bytes > reserved ? free_bytes - reserved : 0;
  return cap > 0 &&
         static_cast<double>(avail) <
             config_.demote_watermark_ * static_cast<double>(cap);
}

bool Runtime::PopDemoteCandidate(DemoteItem *item) {
  /**
   * Take the oldest queued chunk whose owner is under pressure; chunks of
   * owners with room stay queued.
   *
   * @param item Output
   * @return false when no queued chunk qualifies
   */
  std::lock_guard<std::mutex> lock(demote_lock_);
  for (auto it = demote_q_.begin(); it != demote_q_.end(); ++it) {
    if (OwnerUnderPressure(it->owner_)) {
      demote_q_bytes_ -= it->data_.size();
      *item = std::move(*it);
      demote_q_.erase(it);
      return true;
    }
  }
  return false;
}

std::string Runtime::ChooseLowerTier(uint64_t bytes, uint32_t owner) {
  /**
   * The highest-score tier below the fastest whose free bytes on owner can
   * hold bytes, else the lowest tier.
   *
   * @param bytes Bytes to place
   * @param owner Node id
   * @return Tier name (empty with fewer than two tiers)
   */
  if (config_.tiers_.size() < 2) {
    return "";
  }
  std::vector<uint64_t> free_bytes;
  {
    std::lock_guard<std::mutex> outer(load_rings_lock_);
    auto it = load_rings_.find(owner);
    if (it != load_rings_.end()) {
      std::lock_guard<std::mutex> lock(it->second.lock_);
      if (!it->second.samples_.empty()) {
        free_bytes = it->second.samples_.back().tier_remaining_;
      }
    }
  }
  for (size_t i = 1; i < config_.tiers_.size(); ++i) {
    if (i < free_bytes.size() && free_bytes[i] >= bytes) {
      return config_.tiers_[i].name_;
    }
  }
  return config_.tiers_.back().name_;
}

bool Runtime::CompressForDemote(DemoteItem *item, ctp::ipc::FullPtr<char> *comp,
                                size_t *stored, std::string *tier) {
  /**
   * Compress a queued chunk for demotion. Under memory pressure the goal is
   * to free the fast tier and write fewer bytes to the shared slow one, so
   * any codec with a useful ratio is taken: the CCM's choice when it makes
   * one, else the first preferred codec (lz4 when no preference is set).
   *
   * @param item Chunk to compress
   * @param comp Output: header + compressed bytes (caller frees)
   * @param stored Output: bytes to store
   * @param tier Output: lower tier the bytes go to
   * @return false when no codec gives a useful ratio
   */
  const uint32_t owner = item->owner_;
  std::function<double(uint64_t)> store_bw = [this, owner](uint64_t b) {
    return TierBwMbPerMs(ChooseLowerTier(b, owner));
  };
  ccm::Decision decision = ccm_manager_->SelectCodec(
      item->data_.data(), item->data_.size(), item->blob_, 0.0, 1.0, 0.0,
      &store_bw);
  if (decision.chosen_lib_.empty()) {
    decision.chosen_lib_ = config_.compression_preference_.empty()
                               ? std::string("lz4")
                               : config_.compression_preference_.front();
    decision.chosen_preset_ = ctp::CompressionPreset::FAST;
  }
  CompressOutcome out{};
  *comp = CompressWithDecision(item->data_.data(), item->data_.size(),
                               decision, &out);
  if (!out.used) {
    return false;
  }
  *stored = sizeof(compressor::CompressionHeader) + out.comp_size;
  *tier = ChooseLowerTier(*stored, owner);
  return true;
}

clio::run::TaskResume Runtime::DemoteTick() {
  CLIO_TASK_BODY_BEGIN
  // Runs on the periodic SampleLoad task. Pops up to kBatch chunks of
  // owners under pressure (within demote_budget_mb), compresses them here,
  // then deletes and rewrites them with all I/O of a batch in flight at
  // once. Raw bytes are restored if a compressed rewrite fails.
  if (config_.demote_watermark_ > 0.0 && config_.tiers_.size() >= 2 &&
      ccm_manager_ != nullptr) {
    if (!copy_client_) {
      copy_client_ = std::make_unique<clio::cte::core::Client>(
          config_.core_pool_id_.IsNull() ? clio::cte::core::kCtePoolId
                                         : config_.core_pool_id_);
    }
    constexpr size_t kBatch = 16;
    uint64_t budget = static_cast<uint64_t>(config_.demote_budget_mb_) << 20;
    while (budget > 0) {
      std::vector<DemoteItem> items;
      std::vector<ctp::ipc::FullPtr<char>> comps;
      std::vector<size_t> sizes;
      std::vector<std::string> tiers;
      while (items.size() < kBatch && budget > 0) {
        DemoteItem item;
        if (!PopDemoteCandidate(&item)) break;
        budget -= std::min<uint64_t>(budget, item.data_.size());
        ctp::ipc::FullPtr<char> comp;
        size_t stored = 0;
        std::string tier;
        if (CompressForDemote(&item, &comp, &stored, &tier)) {
          items.push_back(std::move(item));
          comps.push_back(comp);
          sizes.push_back(stored);
          tiers.push_back(tier);
        }
      }
      if (items.empty()) break;
      std::vector<clio::run::Future<clio::cte::core::DelBlobTask>> dels;
      for (size_t i = 0; i < items.size(); ++i) {
        dels.push_back(copy_client_->AsyncDelBlob(
            items[i].tag_id_, items[i].blob_,
            IsPlaced(items[i].blob_) ? NodeQuery(items[i].owner_)
                                     : clio::run::PoolQuery::Dynamic()));
      }
      std::vector<bool> deleted(items.size(), false);
      for (size_t i = 0; i < dels.size(); ++i) {
        CLIO_CO_AWAIT(dels[i]);
        deleted[i] = dels[i]->GetReturnCode() == 0;
      }
      std::vector<clio::run::Future<clio::cte::core::PutBlobTask>> puts;
      std::vector<size_t> put_idx;
      for (size_t i = 0; i < items.size(); ++i) {
        if (!deleted[i]) continue;
        clio::cte::core::Context ctx;
        ctx.transform_flags_ |= clio::cte::core::kBlobTransformCompressed;
        puts.push_back(copy_client_->AsyncPutBlob(
            items[i].tag_id_, items[i].blob_, 0, sizes[i],
            comps[i].shm_.template Cast<void>(), TierScore(tiers[i]), ctx,
            /*flags=*/0,
            IsPlaced(items[i].blob_) ? NodeQuery(items[i].owner_)
                                     : clio::run::PoolQuery::Dynamic()));
        put_idx.push_back(i);
      }
      for (size_t k = 0; k < puts.size(); ++k) {
        CLIO_CO_AWAIT(puts[k]);
        const size_t i = put_idx[k];
        if (puts[k]->GetReturnCode() != 0) {
          HLOG(kWarning, "dtschedule demote: compressed put of {} failed ({});"
               " restoring raw", items[i].blob_, puts[k]->GetReturnCode());
          auto raw = copy_client_->AsyncPutBlob(
              items[i].tag_id_, items[i].blob_, 0, items[i].data_.size(),
              items[i].data_.data(), -1.0f, clio::cte::core::Context(), 0,
              IsPlaced(items[i].blob_) ? NodeQuery(items[i].owner_)
                                       : clio::run::PoolQuery::Dynamic());
          CLIO_CO_AWAIT(raw);
          continue;
        }
        RecordDemotion(items[i].data_.size(), sizes[i], tiers[i]);
      }
      for (auto &c : comps) CLIO_IPC->FreeBuffer(c);
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

void Runtime::RecordDemotion(size_t raw_bytes, size_t stored_bytes,
                             const std::string &tier) {
  /**
   * Count one demoted chunk; log at the 1st and every 256th.
   *
   * @param raw_bytes Raw size of the chunk
   * @param stored_bytes Bytes written to the lower tier
   * @param tier Tier written to
   */
  std::lock_guard<std::mutex> lock(stats_lock_);
  ++demoted_;
  demote_in_bytes_ += raw_bytes;
  demote_out_bytes_ += stored_bytes;
  if (demoted_ == 1 || demoted_ % 256 == 0) {
    HLOG(kInfo, "dtschedule demote: {} chunks, {} MiB -> {} MiB (to {})",
         demoted_, demote_in_bytes_ >> 20, demote_out_bytes_ >> 20, tier);
  }
}

bool Runtime::IsOwnTarget(const std::string &target_name) const {
  /**
   * Whether a core target is this node's own device. The core registers
   * "neighborhood" targets: each container also registers its neighbours'
   * devices as "<path>_node<k>", so the local target list holds other
   * nodes' NVMe/HDD files too and their space must not count here.
   *
   * @param target_name Core target name
   * @return false only when the name ends in "_node<k>" for another node
   */
  const size_t pos = target_name.rfind("_node");
  if (pos == std::string::npos || pos + 5 >= target_name.size()) {
    return true;
  }
  const std::string digits = target_name.substr(pos + 5);
  if (digits.find_first_not_of("0123456789") != std::string::npos) {
    return true;
  }
  return std::stoul(digits) == CLIO_IPC->GetNodeId();
}

size_t Runtime::TierIndexForTarget(const std::string &target_name,
                                  float target_score) const {
  /**
   * Map a core target to a configured tier: by name first (tier names come
   * from the device paths, e.g. "ram" in "ram::cte_ram_tier1_node0", "nvme"
   * in "/mnt/nvme/.../cte_target.bin"), else the closest score. The core's
   * reported score can be a measured bandwidth score that does not match
   * the configured one, so score alone mislabels NVMe as RAM.
   *
   * @param target_name Core target name
   * @param target_score Score the core reports for it
   * @return Index into config_.tiers_
   */
  for (size_t t = 0; t < config_.tiers_.size(); ++t) {
    const std::string &tier = config_.tiers_[t].name_;
    if (!tier.empty() && target_name.find(tier) != std::string::npos) {
      return t;
    }
  }
  size_t best = 0;
  for (size_t t = 1; t < config_.tiers_.size(); ++t) {
    if (std::fabs(config_.tiers_[t].score_ - target_score) <
        std::fabs(config_.tiers_[best].score_ - target_score)) {
      best = t;
    }
  }
  return best;
}

void Runtime::LogTierRemaining(const std::vector<uint64_t> &free_bytes) {
  /**
   * Log this node's free bytes per tier the first time and whenever a tier
   * moved by more than 1 GiB since the last log line.
   *
   * @param free_bytes Free bytes per tier, in config_.tiers_ order
   */
  constexpr uint64_t kStep = 1ull << 30;
  std::lock_guard<std::mutex> lock(tier_cap_lock_);
  bool changed = logged_tier_free_.size() != free_bytes.size();
  for (size_t i = 0; !changed && i < free_bytes.size(); ++i) {
    const uint64_t a = free_bytes[i], b = logged_tier_free_[i];
    changed = (a > b ? a - b : b - a) > kStep;
  }
  if (!changed) {
    return;
  }
  logged_tier_free_ = free_bytes;
  std::string msg;
  for (size_t i = 0; i < free_bytes.size(); ++i) {
    msg += " " + config_.tiers_[i].name_ + "=" +
           std::to_string(free_bytes[i] >> 20) + "MiB";
  }
  HLOG(kInfo, "dtschedule: node {} tier free:{}", CLIO_IPC->GetNodeId(), msg);
}

clio::run::TaskResume Runtime::CollectTierRemaining(
    std::vector<uint64_t> *out) {
  CLIO_TASK_BODY_BEGIN
  out->assign(config_.tiers_.size(), 0);
  if (config_.tiers_.size() < 2) {
    out->clear();  // one tier: nothing to choose between
    CLIO_CO_RETURN;
  }
  if (!copy_client_) {
    copy_client_ = std::make_unique<clio::cte::core::Client>(
        config_.core_pool_id_.IsNull() ? clio::cte::core::kCtePoolId
                                       : config_.core_pool_id_);
  }
  {
    auto list = copy_client_->AsyncListTargets(clio::run::PoolQuery::Local());
    CLIO_CO_AWAIT(list);
    if (list->GetReturnCode() != 0) {
      out->clear();
      CLIO_CO_RETURN;
    }
    std::vector<std::string> names(list->target_names_.begin(),
                                   list->target_names_.end());
    for (size_t i = 0; i < names.size(); ++i) {
      if (!IsOwnTarget(names[i])) {
        continue;  // a neighbour's device registered here (core neighborhood)
      }
      auto info = copy_client_->AsyncGetTargetInfo(
          names[i], clio::run::PoolQuery::Local());
      CLIO_CO_AWAIT(info);
      if (info->GetReturnCode() != 0) {
        continue;
      }
      (*out)[TierIndexForTarget(names[i], info->target_score_)] +=
          info->remaining_space_;
    }
  }
  LogTierRemaining(*out);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

float Runtime::TierScore(const std::string &tier) const {
  for (const auto &t : config_.tiers_) {
    if (t.name_ == tier) return t.score_;
  }
  return -1.0f;
}

double Runtime::TierBwMbPerMs(const std::string &tier) const {
  // tier_bw_mbps is MB/s in the config; the cost model works in MB/ms.
  auto it = config_.tier_bw_mbps_.find(tier);
  if (it != config_.tier_bw_mbps_.end() && it->second > 0.0) {
    return it->second / 1000.0;
  }
  return config_.net_bw_gbps_ / 8.0;  // Gbit/s / 8 = GB/s = MB/ms
}

void Runtime::PlanTier(uint64_t size, Placement *place) const {
  // tier_first / joint: the tier is chosen for the raw size before ranking
  // (joint re-checks it against the compressed size afterwards);
  // codec_first: rank with the fastest tier's bandwidth, choose the tier later.
  place->order = knobs_.decision_order_.empty() ? "joint" : knobs_.decision_order_;
  if (place->order == "codec_first") {
    place->tier = config_.tiers_.empty() ? "" : config_.tiers_.front().name_;
  } else {
    place->tier = const_cast<Runtime *>(this)->ChooseTier(
        size, place->owner_node == UINT32_MAX ? CLIO_IPC->GetNodeId()
                                              : place->owner_node);
  }
  place->tier_score = TierScore(place->tier);
  place->tier_bw_mb_ms = TierBwMbPerMs(place->tier);
}

Runtime::ScenarioChoice Runtime::SelectScenario(uint64_t size,
                                                double pred_ratio,
                                                double pred_ctime_ms,
                                                double pred_dtime_ms,
                                                double load_mult,
                                                const Placement &place) {
  // DESIGN §3. Units: MB and ms. net(b, A->B) = 0 when A == B.
  ScenarioChoice choice;
  const double raw_mb = static_cast<double>(size) / 1e6;
  const double z_mb = raw_mb / std::max(pred_ratio, 1.0);
  const double net_mb_ms = std::max(config_.net_bw_gbps_, 0.01) / 8.0;
  const double tier_mb_ms = place.tier_bw_mb_ms > 0.0 ? place.tier_bw_mb_ms
                                                       : net_mb_ms;
  const double store_ms = (z_mb / tier_mb_ms) * load_mult;
  const uint32_t P = CLIO_IPC->GetNodeId();
  const uint32_t C = place.consumer_node;
  const uint32_t O = place.owner_node == UINT32_MAX ? P : place.owner_node;
  auto net = [&](double mb, uint32_t a, uint32_t b) {
    return a == b ? 0.0 : mb / net_mb_ms;
  };
  choice.cost1_ms = pred_ctime_ms * load_mult + net(z_mb, P, O) + store_ms;
  choice.cost2_ms = choice.cost1_ms;
  choice.cost3_ms = 0.0;

  if (C == UINT32_MAX || config_.workflow_aware_ == "none") {
    choice.chosen_scenario = 1;
    return choice;
  }
  choice.cost1_ms += net(z_mb, O, C) + pred_dtime_ms;
  choice.cost2_ms += net(raw_mb, P, C);

  // Phase 4c: Scenario 3 cost model.
  // S3: net(raw, P->C) + T_c(C)·m_C + net(z, C->O) + store(z)·m_C
  // m_C is the load multiplier for consumer C (when known, else 1).
  double load_mult_c = 1.0;
  if (place.consumer_cpu >= 0.0) {
    // Consumer CPU load is known; compute multiplier from producer/consumer CPU.
    load_mult_c = LoadMultiplier(place.consumer_cpu, place.consumer_cpu,
                                 knobs_.load_aware_, config_.load_cap_);
  }
  choice.cost3_ms = net(raw_mb, P, C) + pred_ctime_ms * load_mult_c +
                    net(z_mb, C, O) + store_ms * load_mult_c;

  const std::string &force = knobs_.force_scenario_;
  if (force == "1") {
    choice.chosen_scenario = 1;
  } else if (force == "2") {
    choice.chosen_scenario = 2;
  } else if (force == "3") {
    choice.chosen_scenario = 3;
  } else {
    // Pick the minimum cost scenario
    if (choice.cost1_ms <= choice.cost2_ms && choice.cost1_ms <= choice.cost3_ms) {
      choice.chosen_scenario = 1;
    } else if (choice.cost2_ms <= choice.cost3_ms) {
      choice.chosen_scenario = 2;
    } else {
      choice.chosen_scenario = 3;
    }
  }
  return choice;
}

clio::run::TaskResume Runtime::PushConsumerCopy(
    const TagId &tag_id, const std::string &blob_name,
    ctp::ipc::ShmPtr<> raw_data, uint64_t size, uint32_t consumer_node,
    uint32_t owner_node, uint64_t version, Placement *place) {
  CLIO_TASK_BODY_BEGIN
  // A copy at the producer (the cache below keeps one) or at the owner (the
  // authoritative copy lives there) protects nothing: skip.
  if (consumer_node == CLIO_IPC->GetNodeId() || consumer_node == owner_node) {
    std::lock_guard<std::mutex> lock(stats_lock_);
    stats_.copy_skipped_local_++;
    CLIO_CO_RETURN;
  }
  if (!copy_client_) {
    copy_client_ = std::make_unique<clio::cte::core::Client>(
        config_.core_pool_id_.IsNull() ? clio::cte::core::kCtePoolId
                                       : config_.core_pool_id_);
  }
  {
    auto t0 = std::chrono::steady_clock::now();
    // The cache's populate write, aimed at the consumer: raw bytes in the
    // cache-replica slot, RAM score, registered at the owner with our
    // put's version so a concurrent overwrite rejects a stale copy.
    clio::cte::core::Context ctx;
    ctx.replica_ = clio::cte::core::kCacheReplica;
    ctx.replica_flags_ |= clio::cte::core::REPLICA_CACHE;
    ctx.transform_flags_ = 0;
    ctx.min_persistence_level_ = 0;
    const float score = config_.tiers_.empty() ? -1.0f : config_.tiers_.front().score_;
    auto put = copy_client_->AsyncPutBlob(
        tag_id, blob_name, 0, size, raw_data, score, ctx, /*flags=*/0,
        clio::run::PoolQuery::Physical(consumer_node));
    CLIO_CO_AWAIT(put);
    bool ok = put->GetReturnCode() == 0;
    if (ok) {
      auto reg = copy_client_->AsyncRegisterReplicaContainer(
          tag_id, blob_name, consumer_node, clio::run::PoolQuery::Dynamic(),
          version);
      CLIO_CO_AWAIT(reg);
      ok = reg->GetReturnCode() == 0;
    }
    if (!ok) {
      auto del = copy_client_->AsyncDelBlob(
          tag_id, blob_name, clio::run::PoolQuery::Physical(consumer_node),
          clio::cte::core::kDelCacheCopyOnly);
      CLIO_CO_AWAIT(del);
    }
    place->store_ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0)
                          .count();
    std::lock_guard<std::mutex> lock(stats_lock_);
    if (ok) {
      stats_.copies_pushed_++;
    } else {
      stats_.copy_refused_++;
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

uint32_t Runtime::PickConsumerFromDag(const std::string &tag_or_blob_name,
                                      uint32_t producer_node,
                                      std::string &forced_reason) {
  /**
   * Pick a consumer node from DAG spec for a blob.
   *
   * Returns the first consumer node from the spec that is not the
   * producer node, or UINT32_MAX if no consumers or DAG not loaded.
   */
  if (!dag_spec_ || !dag_spec_->IsLoaded()) {
    return UINT32_MAX;
  }

  auto info = dag_spec_->LookupFile(tag_or_blob_name);
  if (info.producer_node == UINT32_MAX || info.consumer_nodes.empty()) {
    return UINT32_MAX;
  }

  // Find first consumer that is not the producer
  for (auto consumer : info.consumer_nodes) {
    if (consumer != producer_node && consumer != UINT32_MAX) {
      forced_reason = "dag:1";
      return consumer;
    }
  }

  forced_reason = "dag:1";
  return UINT32_MAX;
}

clio::run::TaskResume Runtime::PushFanoutCopies(
    const TagId &tag_id, const std::string &blob_name,
    ctp::ipc::ShmPtr<> raw_data, uint64_t size, uint32_t owner_node,
    uint32_t already_covered, uint64_t version, Placement *place) {
  CLIO_TASK_BODY_BEGIN
  {
    const uint32_t self = CLIO_IPC->GetNodeId();
    const uint32_t cap = dag_spec_ ? dag_spec_->ReplicateMax() : 0;
    std::vector<uint32_t> targets;
    for (uint32_t node : place->dag_consumers) {
      if (node == UINT32_MAX || node == already_covered) continue;
      if (std::find(targets.begin(), targets.end(), node) != targets.end()) continue;
      targets.push_back(node);
      if (cap > 0 && targets.size() >= cap) break;
    }
    place->fanout = static_cast<int>(targets.size());
    for (size_t i = 0; i < targets.size(); ++i) {
      // Local and owner targets are counted as skipped inside the push.
      CLIO_CO_AWAIT(PushConsumerCopy(tag_id, blob_name, raw_data, size,
                                     targets[i], owner_node, version, place));
    }
    (void)self;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

}  // namespace clio::cte::dtschedule

// Define the chimod entry points (alloc_chimod, new_chimod, get_chimod_name,
// destroy_chimod) that the module manager dlopen()s.
CLIO_TASK_CC(clio::cte::dtschedule::Runtime)
