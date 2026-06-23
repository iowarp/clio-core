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

#ifndef CLIO_RUNTIME_INCLUDE_POOL_QUERY_H_
#define CLIO_RUNTIME_INCLUDE_POOL_QUERY_H_

#include <type_traits>

#include "clio_runtime/types.h"

namespace clio::run {

/**
 * Routing algorithm modes for PoolQuery
 */
enum class RoutingMode {
  Local,          /**< Route to local node only */
  DirectId,       /**< Route to specific container by ID */
  DirectHash,     /**< Route using hash-based load balancing */
  Range,          /**< Route to range of containers */
  Broadcast,      /**< Broadcast to all containers */
  Physical,       /**< Route to specific physical node by ID */
  Dynamic,        /**< Dynamic routing with cache optimization (routes to Monitor) */
  ToLocalCpu,     /**< GPU → CPU direction (the only GPU-related mode) */
  Null,           /**< Do nothing */
  ManyToOne,      /**< Batch+aggregate matching tasks at the neighborhood leader */
  AllToOne        /**< Like ManyToOne, but the aggregate runs only once tasks
                       from ALL containers in the pool have arrived (a collective
                       barrier), instead of after a time window */
};

/**
 * Result of routing a task via RouteTask / RouteLocal / RouteGlobal
 */
enum class RouteResult {
  ExecHere,  /**< Execute on this worker directly (caller runs ExecTask) */
  Local,     /**< Enqueued to a different local worker */
  Network,   /**< Enqueued to net_queue_ for remote dispatch */
  Retry,     /**< Container is plugged — caller should add to retry queue */
  Dne        /**< Container doesn't exist (pool not found) */
};

/**
 * Pool query class for determining task execution location and routing
 *
 * Provides methods to query different container addresses and routing modes
 * for load balancing and task distribution to containers.
 */
class PoolQuery {
 public:
  /**
   * Default constructor
   */
  CTP_CROSS_FUN PoolQuery()
      : routing_mode_(RoutingMode::Local), hash_value_(0),
        container_id_(kInvalidContainerId),
        range_offset_(0), range_count_(0), node_id_(0), ret_node_(0),
        net_timeout_(-1.0f), parallelism_(32),
        batch_key_(0), batch_for_ns_(0) {}

  // Copy/move/destructor are intentionally left implicit (compiler-generated)
  // so PoolQuery stays *trivially copyable*. It is raw-byte serialized and
  // compared (e.g. the CTE transaction log WriteRaw/ReadRaw over
  // sizeof(PoolQuery) and a memcmp roundtrip check), and is bytewise-copied
  // across the GPU/host boundary. A hand-written field-wise copy would NOT
  // copy padding, so two "equal" queries could differ in padding bytes and
  // fail those raw-byte comparisons. (see static_assert below the class)

  // Static factory methods to create different types of PoolQuery

  /**
   * Create a local routing pool query
   * @return PoolQuery configured for local container routing
   */
  static CTP_CROSS_FUN PoolQuery Local(u32 parallelism = 32) {
    PoolQuery query;
    query.routing_mode_ = RoutingMode::Local;
    query.hash_value_ = 0;
    query.container_id_ = kInvalidContainerId;
    query.range_offset_ = 0;
    query.parallelism_ = parallelism;
    return query;
  }

  /**
   * Create a direct ID routing pool query
   * @param container_id Specific container ID to route to
   * @return PoolQuery configured for direct container ID routing
   */
  static PoolQuery DirectId(ContainerId container_id, float net_timeout = -1);

  /**
   * Create a direct hash routing pool query
   * @param hash Hash value for container selection
   * @param net_timeout Per-task network timeout in seconds (-1 = default)
   * @return PoolQuery configured for hash-based routing to specific container
   */
  static PoolQuery DirectHash(u32 hash, float net_timeout = -1);

  /**
   * Create a range routing pool query
   * @param offset Starting offset in the container range
   * @param count Number of containers in the range
   * @param net_timeout Per-task network timeout in seconds (-1 = default)
   * @return PoolQuery configured for range-based routing
   */
  static PoolQuery Range(u32 offset, u32 count, float net_timeout = -1);

  /**
   * Create a broadcast routing pool query
   * @param net_timeout Per-task network timeout in seconds (-1 = default)
   * @return PoolQuery configured for broadcast to all containers
   */
  static PoolQuery Broadcast(float net_timeout = -1);

  /**
   * Create a physical routing pool query
   * @param node_id Specific node ID to route to
   * @param net_timeout Per-task network timeout in seconds (-1 = default)
   * @return PoolQuery configured for physical node routing
   */
  static PoolQuery Physical(u32 node_id, float net_timeout = -1);

  /**
   * Create a dynamic routing pool query (recommended for Create operations)
   * Routes to Monitor with kGlobalSchedule for automatic cache checking
   * @param net_timeout Per-task network timeout in seconds (-1 = default)
   * @return PoolQuery configured for dynamic routing with cache optimization
   */
  static PoolQuery Dynamic(float net_timeout = -1);

  /**
   * Create a many-to-one collective routing pool query.
   *
   * Routes the task to the neighborhood leader, which briefly batches all
   * tasks sharing the same (pool_id, method, container_hash, batch_key) for
   * a batch_for window, aggregates them into a single task, runs it once,
   * and broadcasts the result back to every batched task.
   *
   * @param container_hash Logical container key (stored in hash_value_);
   *        also selects the container the aggregate task executes on.
   * @param batch_key Sub-key to separate concurrent collectives on the same
   *        (pool_id, method, container_hash). Default 0.
   * @param batch_for_ns Batching window in nanoseconds. Default 10000 (10us).
   * @return PoolQuery configured for many-to-one batch aggregation
   */
  static PoolQuery ManyToOne(u32 container_hash, u64 batch_key = 0,
                             u64 batch_for_ns = 10000);

  /**
   * Create an all-to-one collective routing pool query.
   *
   * Identical to ManyToOne except for the flush condition: the neighborhood
   * leader batches all tasks sharing the same (pool_id, method, container_hash,
   * batch_key) and does NOT run the aggregate until tasks from EVERY container
   * in the pool have arrived (a collective barrier). The aggregate tracks how
   * many tasks have been combined into it; when that count reaches the pool's
   * container count, it runs once and the result is broadcast back to every
   * batched task (same 1->N result path as ManyToOne).
   *
   * Use this when the "one" must observe the contribution of all "many" before
   * it can compute (e.g. a global reduction / barrier); use ManyToOne when a
   * best-effort time-windowed batch is sufficient.
   *
   * @param container_hash Logical container key (stored in hash_value_); also
   *        selects the container the aggregate task executes on.
   * @param batch_key Sub-key to separate concurrent collectives on the same
   *        (pool_id, method, container_hash). Default 0.
   * @return PoolQuery configured for all-to-one barrier aggregation
   */
  static PoolQuery AllToOne(u32 container_hash, u64 batch_key = 0);

  /**
   * Create a pool query for GPU → CPU direction
   * @return PoolQuery configured for routing from GPU back to CPU
   */
  static CTP_CROSS_FUN PoolQuery ToLocalCpu(u32 parallelism = 32) {
    PoolQuery query;
    query.routing_mode_ = RoutingMode::ToLocalCpu;
    query.parallelism_ = parallelism;
    return query;
  }

  /**
   * Create a null pool query (do nothing)
   * @return PoolQuery configured to skip execution
   */
  static CTP_CROSS_FUN PoolQuery Null() {
    PoolQuery query;
    query.routing_mode_ = RoutingMode::Null;
    return query;
  }

  /**
   * Parse PoolQuery from string (supports "local" and "dynamic")
   * @param str String representation of pool query mode
   * @return PoolQuery configured based on string value
   */
  static PoolQuery FromString(const std::string& str);

  /**
   * Convert PoolQuery to string representation
   * @return String representation (e.g., "local", "broadcast", "direct_id:5")
   */
  std::string ToString() const;

  // Getter methods for internal query parameters (used by routing logic)

  /**
   * Get the hash value for hash-based routing modes
   * @return Hash value used for container routing
   */
  CTP_CROSS_FUN u32 GetHash() const { return hash_value_; }

  /**
   * Get the container ID for direct ID routing mode
   * @return Container ID for direct routing
   */
  CTP_CROSS_FUN ContainerId GetContainerId() const { return container_id_; }

  /**
   * Check if a specific container ID has been set
   * @return true if container_id is not kInvalidContainerId
   */
  CTP_CROSS_FUN bool HasContainerId() const {
    return container_id_ != kInvalidContainerId;
  }

  /**
   * Get the range offset for range routing mode
   * @return Starting offset in the container range
   */
  CTP_CROSS_FUN u32 GetRangeOffset() const { return range_offset_; }

  /**
   * Get the range count for range routing mode
   * @return Number of containers in the range
   */
  CTP_CROSS_FUN u32 GetRangeCount() const { return range_count_; }

  /**
   * Get the node ID for physical routing mode
   * @return Node ID for physical routing
   */
  CTP_CROSS_FUN u32 GetNodeId() const { return node_id_; }

  /**
   * Determine the routing mode of this pool query
   * @return RoutingMode enum indicating how this query should be routed
   */
  CTP_CROSS_FUN RoutingMode GetRoutingMode() const { return routing_mode_; }

  /**
   * Check if pool query is in Local routing mode
   * @return true if routing mode is Local
   */
  CTP_CROSS_FUN bool IsLocalMode() const {
    return routing_mode_ == RoutingMode::Local;
  }

  /**
   * Check if pool query is in DirectId routing mode
   * @return true if routing mode is DirectId
   */
  CTP_CROSS_FUN bool IsDirectIdMode() const {
    return routing_mode_ == RoutingMode::DirectId;
  }

  /**
   * Check if pool query is in DirectHash routing mode
   * @return true if routing mode is DirectHash
   */
  CTP_CROSS_FUN bool IsDirectHashMode() const {
    return routing_mode_ == RoutingMode::DirectHash;
  }

  /**
   * Check if pool query is in Range routing mode
   * @return true if routing mode is Range
   */
  CTP_CROSS_FUN bool IsRangeMode() const {
    return routing_mode_ == RoutingMode::Range;
  }

  /**
   * Check if pool query is in Broadcast routing mode
   * @return true if routing mode is Broadcast
   */
  CTP_CROSS_FUN bool IsBroadcastMode() const {
    return routing_mode_ == RoutingMode::Broadcast;
  }

  /**
   * Check if pool query is in Physical routing mode
   * @return true if routing mode is Physical
   */
  CTP_CROSS_FUN bool IsPhysicalMode() const {
    return routing_mode_ == RoutingMode::Physical;
  }

  /**
   * Check if pool query is in Dynamic routing mode
   * @return true if routing mode is Dynamic
   */
  CTP_CROSS_FUN bool IsDynamicMode() const {
    return routing_mode_ == RoutingMode::Dynamic;
  }

  /**
   * Check if pool query is in ToLocalCpu routing mode
   * @return true if routing mode is ToLocalCpu
   */
  CTP_CROSS_FUN bool IsToLocalCpuMode() const {
    return routing_mode_ == RoutingMode::ToLocalCpu;
  }

  /**
   * Check if pool query is in Null routing mode
   * @return true if routing mode is Null
   */
  CTP_CROSS_FUN bool IsNullMode() const {
    return routing_mode_ == RoutingMode::Null;
  }

  /**
   * Set the return node ID for distributed task responses
   * @param ret_node Node ID where task results should be returned
   */
  CTP_CROSS_FUN void SetReturnNode(u32 ret_node) {
    ret_node_ = ret_node;
  }

  /**
   * Get the return node ID for distributed task responses
   * @return Node ID where task results should be returned
   */
  CTP_CROSS_FUN u32 GetReturnNode() const {
    return ret_node_;
  }

  /**
   * Get the per-task network timeout
   * @return Network timeout in seconds, or -1 if unset (use global default)
   */
  CTP_CROSS_FUN float GetNetTimeout() const { return net_timeout_; }

  /**
   * Set the per-task network timeout
   * @param t Timeout in seconds. Use -1 for default, 0 for immediate fail.
   */
  CTP_CROSS_FUN void SetNetTimeout(float t) { net_timeout_ = t; }

  /**
   * Get the parallelism level for GPU task dispatch
   * @return Number of threads (1 = lane 0 only, 32 = full warp, >32 = multi-warp)
   */
  CTP_CROSS_FUN u32 GetParallelism() const { return parallelism_; }

  /** Set the parallelism level for GPU task dispatch */
  CTP_CROSS_FUN void SetParallelism(u32 parallelism) { parallelism_ = parallelism; }

  /**
   * Check if pool query is in ManyToOne routing mode
   * @return true if routing mode is ManyToOne
   */
  CTP_CROSS_FUN bool IsManyToOneMode() const {
    return routing_mode_ == RoutingMode::ManyToOne;
  }

  /**
   * Check if pool query is in AllToOne routing mode
   * @return true if routing mode is AllToOne
   */
  CTP_CROSS_FUN bool IsAllToOneMode() const {
    return routing_mode_ == RoutingMode::AllToOne;
  }

  /**
   * Check if pool query is in any collective batch+aggregate mode
   * (ManyToOne or AllToOne). Both route to the neighborhood leader and park in
   * the BatchManager; they differ only in the flush condition.
   * @return true if routing mode is ManyToOne or AllToOne
   */
  CTP_CROSS_FUN bool IsCollectiveMode() const {
    return routing_mode_ == RoutingMode::ManyToOne ||
           routing_mode_ == RoutingMode::AllToOne;
  }

  /**
   * Get the container hash (ManyToOne). Aliases the hash value: it both
   * keys the batch and selects the container the aggregate runs on.
   * @return Container hash for many-to-one batching
   */
  CTP_CROSS_FUN u32 GetContainerHash() const { return hash_value_; }

  /**
   * Get the batch sub-key (ManyToOne)
   * @return Batch key separating concurrent collectives
   */
  CTP_CROSS_FUN u64 GetBatchKey() const { return batch_key_; }

  /**
   * Get the batching window in nanoseconds (ManyToOne)
   * @return Batch window in ns
   */
  CTP_CROSS_FUN u64 GetBatchForNs() const { return batch_for_ns_; }

  /**
   * Serialization support for any archive type
   * @param ar Archive for serialization
   */
  template <class Archive>
  CTP_CROSS_FUN void serialize(Archive& ar) {
    ar.range(routing_mode_, hash_value_, container_id_, range_offset_,
             range_count_, node_id_, ret_node_, net_timeout_, parallelism_,
             batch_key_, batch_for_ns_);
  }

 private:
  RoutingMode routing_mode_; /**< The routing mode for this query */
  u32 hash_value_;           /**< Hash value for hash-based routing */
  ContainerId container_id_; /**< Container ID for direct ID routing */
  u32 range_offset_;         /**< Starting offset for range routing */
  u32 range_count_;          /**< Number of containers for range routing */
  u32 node_id_;              /**< Node ID for physical routing */
  u32 ret_node_;             /**< Return node ID for distributed responses */
  float net_timeout_;        /**< Per-task network timeout in seconds (-1 = use default) */
  u32 parallelism_;          /**< GPU parallelism: 1 (lane 0), 32 (full warp), >32 (multi-warp) */
  u64 batch_key_;            /**< ManyToOne: batch sub-key */
  u64 batch_for_ns_;         /**< ManyToOne: batch window in nanoseconds */
};

// PoolQuery is raw-byte serialized and memcmp-compared (CTE transaction log,
// GPU/host bytewise task copies). Keep it trivially copyable so copies preserve
// every byte (including padding) — a hand-written field-wise copy would not,
// breaking those raw-byte roundtrips.
static_assert(std::is_trivially_copyable<PoolQuery>::value,
              "PoolQuery must remain trivially copyable (raw-byte serialized)");

}  // namespace clio::run

#endif  // CLIO_RUNTIME_INCLUDE_POOL_QUERY_H_