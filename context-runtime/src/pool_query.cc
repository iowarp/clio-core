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

/**
 * Pool query implementation
 */

#include "chimaera/pool_query.h"
#include <algorithm>
#include <stdexcept>

namespace chi {

// Constructor, copy constructor, assignment operator, and destructor
// are now inline in pool_query.h for GPU compatibility

// Static factory methods
// Note: PoolQuery::Local() is now inline in pool_query.h for GPU compatibility

PoolQuery PoolQuery::DirectId(ContainerId container_id) {
  PoolQuery query;
  query.routing_mode_ = RoutingMode::DirectId;
  query.hash_value_ = 0;
  query.container_id_ = container_id;
  query.range_offset_ = 0;
  query.range_count_ = 0;
  query.node_id_ = 0;
  return query;
}

PoolQuery PoolQuery::DirectHash(u32 hash) {
  PoolQuery query;
  query.routing_mode_ = RoutingMode::DirectHash;
  query.hash_value_ = hash;
  query.container_id_ = 0;
  query.range_offset_ = 0;
  query.range_count_ = 0;
  query.node_id_ = 0;
  return query;
}

PoolQuery PoolQuery::Range(u32 offset, u32 count) {
  PoolQuery query;
  query.routing_mode_ = RoutingMode::Range;
  query.hash_value_ = 0;
  query.container_id_ = 0;
  query.range_offset_ = offset;
  query.range_count_ = count;
  query.node_id_ = 0;
  return query;
}

PoolQuery PoolQuery::Broadcast() {
  PoolQuery query;
  query.routing_mode_ = RoutingMode::Broadcast;
  query.hash_value_ = 0;
  query.container_id_ = 0;
  query.range_offset_ = 0;
  query.range_count_ = 0;
  query.node_id_ = 0;
  return query;
}

PoolQuery PoolQuery::Physical(u32 node_id) {
  PoolQuery query;
  query.routing_mode_ = RoutingMode::Physical;
  query.hash_value_ = 0;
  query.container_id_ = 0;
  query.range_offset_ = 0;
  query.range_count_ = 0;
  query.node_id_ = node_id;
  return query;
}

PoolQuery PoolQuery::Dynamic() {
  PoolQuery query;
  query.routing_mode_ = RoutingMode::Dynamic;
  query.hash_value_ = 0;
  query.container_id_ = 0;
  query.range_offset_ = 0;
  query.range_count_ = 0;
  query.node_id_ = 0;
  return query;
}

PoolQuery PoolQuery::FromString(const std::string& str) {
  // Convert to lowercase for case-insensitive comparison
  std::string lower_str = str;
  std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  if (lower_str == "local") {
    return PoolQuery::Local();
  } else if (lower_str == "dynamic") {
    return PoolQuery::Dynamic();
  } else {
    throw std::invalid_argument("Invalid PoolQuery string, expected 'local' or 'dynamic'");
  }
}

// Getter methods are now inline in pool_query.h for GPU compatibility

}  // namespace chi