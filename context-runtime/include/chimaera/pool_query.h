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

#ifndef CHIMAERA_INCLUDE_CHIMAERA_POOL_QUERY_H_
#define CHIMAERA_INCLUDE_CHIMAERA_POOL_QUERY_H_

#include "chimaera/types.h"

namespace chi {

/**
 * Routing algorithm modes for PoolQuery
 */
enum class RoutingMode {
  Local,      /**< Route to local node only */
  DirectId,   /**< Route to specific container by ID */
  DirectHash, /**< Route using hash-based load balancing */
  Range,      /**< Route to range of containers */
  Broadcast,  /**< Broadcast to all containers */
  Physical,   /**< Route to specific physical node by ID */
  Dynamic     /**< Dynamic routing with cache optimization (routes to Monitor) */
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
  HSHM_CROSS_FUN PoolQuery()
      : routing_mode_(RoutingMode::Local), hash_value_(0),
        container_id_(kInvalidContainerId),
        range_offset_(0), range_count_(0), node_id_(0), ret_node_(0),
        net_timeout_(-1.0f) {}

  /**
   * Copy constructor
   */
  HSHM_CROSS_FUN PoolQuery(const PoolQuery& other)
      : routing_mode_(other.routing_mode_),
        hash_value_(other.hash_value_),
        container_id_(other.container_id_),
        range_offset_(other.range_offset_),
        range_count_(other.range_count_),
        node_id_(other.node_id_),
        ret_node_(other.ret_node_),
        net_timeout_(other.net_timeout_) {}

  /**
   * Assignment operator
   */
  HSHM_CROSS_FUN PoolQuery& operator=(const PoolQuery& other) {
    if (this != &other) {
      routing_mode_ = other.routing_mode_;
      hash_value_ = other.hash_value_;
      container_id_ = other.container_id_;
      range_offset_ = other.range_offset_;
      range_count_ = other.range_count_;
      node_id_ = other.node_id_;
      ret_node_ = other.ret_node_;
      net_timeout_ = other.net_timeout_;
    }
    return *this;
  }

  /**
   * Destructor
   */
  HSHM_CROSS_FUN ~PoolQuery() {}

  // Static factory methods to create different types of PoolQuery

  /**
   * Create a local routing pool query
   * @return PoolQuery configured for local container routing
   */
  static HSHM_CROSS_FUN PoolQuery Local() {
    PoolQuery query;
    query.routing_mode_ = RoutingMode::Local;
    query.hash_value_ = 0;
    query.container_id_ = kInvalidContainerId;
    query.range_offset_ = 0;
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
  HSHM_CROSS_FUN u32 GetHash() const { return hash_value_; }

  /**
   * Get the container ID for direct ID routing mode
   * @return Container ID for direct routing
   */
  HSHM_CROSS_FUN ContainerId GetContainerId() const { return container_id_; }

  /**
   * Check if a specific container ID has been set
   * @return true if container_id is not kInvalidContainerId
   */
  HSHM_CROSS_FUN bool HasContainerId() const {
    return container_id_ != kInvalidContainerId;
  }

  /**
   * Get the range offset for range routing mode
   * @return Starting offset in the container range
   */
  HSHM_CROSS_FUN u32 GetRangeOffset() const { return range_offset_; }

  /**
   * Get the range count for range routing mode
   * @return Number of containers in the range
   */
  HSHM_CROSS_FUN u32 GetRangeCount() const { return range_count_; }

  /**
   * Get the node ID for physical routing mode
   * @return Node ID for physical routing
   */
  HSHM_CROSS_FUN u32 GetNodeId() const { return node_id_; }

  /**
   * Determine the routing mode of this pool query
   * @return RoutingMode enum indicating how this query should be routed
   */
  HSHM_CROSS_FUN RoutingMode GetRoutingMode() const { return routing_mode_; }

  /**
   * Check if pool query is in Local routing mode
   * @return true if routing mode is Local
   */
  HSHM_CROSS_FUN bool IsLocalMode() const {
    return routing_mode_ == RoutingMode::Local;
  }

  /**
   * Check if pool query is in DirectId routing mode
   * @return true if routing mode is DirectId
   */
  HSHM_CROSS_FUN bool IsDirectIdMode() const {
    return routing_mode_ == RoutingMode::DirectId;
  }

  /**
   * Check if pool query is in DirectHash routing mode
   * @return true if routing mode is DirectHash
   */
  HSHM_CROSS_FUN bool IsDirectHashMode() const {
    return routing_mode_ == RoutingMode::DirectHash;
  }

  /**
   * Check if pool query is in Range routing mode
   * @return true if routing mode is Range
   */
  HSHM_CROSS_FUN bool IsRangeMode() const {
    return routing_mode_ == RoutingMode::Range;
  }

  /**
   * Check if pool query is in Broadcast routing mode
   * @return true if routing mode is Broadcast
   */
  HSHM_CROSS_FUN bool IsBroadcastMode() const {
    return routing_mode_ == RoutingMode::Broadcast;
  }

  /**
   * Check if pool query is in Physical routing mode
   * @return true if routing mode is Physical
   */
  HSHM_CROSS_FUN bool IsPhysicalMode() const {
    return routing_mode_ == RoutingMode::Physical;
  }

  /**
   * Check if pool query is in Dynamic routing mode
   * @return true if routing mode is Dynamic
   */
  HSHM_CROSS_FUN bool IsDynamicMode() const {
    return routing_mode_ == RoutingMode::Dynamic;
  }

  /**
   * Set the return node ID for distributed task responses
   * @param ret_node Node ID where task results should be returned
   */
  HSHM_CROSS_FUN void SetReturnNode(u32 ret_node) {
    ret_node_ = ret_node;
  }

  /**
   * Get the return node ID for distributed task responses
   * @return Node ID where task results should be returned
   */
  HSHM_CROSS_FUN u32 GetReturnNode() const {
    return ret_node_;
  }

  /**
   * Get the per-task network timeout
   * @return Network timeout in seconds, or -1 if unset (use global default)
   */
  HSHM_CROSS_FUN float GetNetTimeout() const { return net_timeout_; }

  /**
   * Set the per-task network timeout
   * @param t Timeout in seconds. Use -1 for default, 0 for immediate fail.
   */
  HSHM_CROSS_FUN void SetNetTimeout(float t) { net_timeout_ = t; }

  /**
   * Serialization support for any archive type
   * @param ar Archive for serialization
   */
  template <class Archive>
  HSHM_CROSS_FUN void serialize(Archive& ar) {
    ar(routing_mode_, hash_value_, container_id_, range_offset_, range_count_, node_id_, ret_node_, net_timeout_);
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
};


}  // namespace chi

#endif  // CHIMAERA_INCLUDE_CHIMAERA_POOL_QUERY_H_