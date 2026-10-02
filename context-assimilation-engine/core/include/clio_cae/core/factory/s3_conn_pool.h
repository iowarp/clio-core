/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 */

#ifndef CLIO_CAE_CORE_FACTORY_S3_CONN_POOL_H_
#define CLIO_CAE_CORE_FACTORY_S3_CONN_POOL_H_

// A lease pool of keep-alive S3 connections for the in-process CAE assimilator.
//
// Why a checkout pool and not the bdev's per-worker vector: the bdev caches one
// S3Connection per runtime worker and never suspends inside an S3 op, so a
// worker's socket is only ever touched by that worker. The CAE assimilator DOES
// suspend mid-object (it co_awaits PutBlob inside the download loop), and a
// parked task can be migrated to a different rescuer worker (issue #785). Keying
// a socket by worker would then let two workers touch one non-thread-safe Poco
// session. Instead a task LEASES a connection for the whole object and returns
// it on completion -- exclusive ownership across suspends, which is exactly why
// the GCS assimilator's stack-local client is safe today, made persistent.
//
// Sockets are keyed by "scheme://host:port" (S3RestClient::ConnectionKey),
// because on real AWS the bucket and region live in the hostname, so different
// imports may target different endpoints.

#ifdef CLIO_ENABLE_S3_REST

#include <clio_ctp/util/logging.h>

#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "clio_runtime/bdev/transports/s3_rest.h"

namespace clio::cae::core {

/** Lease pool of reusable keep-alive S3 connections, keyed by endpoint. */
class S3ConnectionPool {
 public:
  using S3Connection = clio::run::bdev::s3::S3Connection;

  S3ConnectionPool() = default;

  /**
   * Lease a connection for `conn_key`, reusing an idle keep-alive socket when
   * one is available, else handing back a fresh (unconnected) handle. The
   * caller owns it exclusively until Release. Never null.
   *
   * @param conn_key "scheme://host:port" from S3RestClient::ConnectionKey.
   * @return An owned connection handle to use, then Release.
   */
  std::unique_ptr<S3Connection> Acquire(const std::string &conn_key) {
    std::lock_guard<std::mutex> lk(mtx_);
    auto it = idle_.find(conn_key);
    if (it != idle_.end() && !it->second.empty()) {
      std::unique_ptr<S3Connection> c = std::move(it->second.back());
      it->second.pop_back();
      return c;
    }
    return std::make_unique<S3Connection>();
  }

  /**
   * Return a leased connection. A retired handle (its session was dropped after
   * a failure or a server `Connection: close`) is not pooled; a live one is
   * kept for reuse up to a per-key cap, above which it is closed. Either way its
   * lifetime socket/request counters fold into the pool totals so a later tally
   * counts sockets that have already been dropped.
   */
  void Release(const std::string &conn_key, std::unique_ptr<S3Connection> conn) {
    if (!conn) return;
    std::lock_guard<std::mutex> lk(mtx_);
    std::vector<std::unique_ptr<S3Connection>> &bucket = idle_[conn_key];
    const bool live = (conn->session != nullptr);
    if (live && bucket.size() < max_idle_per_key_) {
      bucket.push_back(std::move(conn));
      return;
    }
    // Dropped here (retired, or over the idle cap): bank its counters now,
    // because after this it is gone and LogTally can only see idle sockets.
    dropped_connects_ += conn->connects;
    dropped_reuses_ += conn->reuses;
    // conn's dtor closes the socket.
  }

  /**
   * Emit one compact keep-alive tally (sockets / requests / reuse_ratio),
   * mirroring the bdev's line but tagged "CAE S3" so a pipeline gate can grep
   * for it. `sockets == 1` with `requests >> 1` is reuse working; `sockets ==
   * requests` means every request reconnected and the mechanism is dead. Call
   * once at Runtime::Destroy, after every task has released its lease.
   */
  void LogTally() const {
    std::lock_guard<std::mutex> lk(mtx_);
    uint64_t sockets = dropped_connects_;
    uint64_t requests = dropped_connects_ + dropped_reuses_;
    for (const auto &kv : idle_) {
      for (const auto &c : kv.second) {
        sockets += c->connects;
        requests += c->connects + c->reuses;
      }
    }
    if (sockets == 0) return;  // no S3 I/O happened; stay quiet
    // Round here, not in the format string: ctp::Formatter only understands a
    // bare "{}" (see s3_bdev_transport.cc), so "{:.2f}" would emit trailing
    // garbage a parser might pick up.
    char ratio[32];
    std::snprintf(ratio, sizeof(ratio), "%.2f",
                  static_cast<double>(requests) /
                      static_cast<double>(sockets));
    HLOG(kInfo, "CAE S3 keepalive TOTAL sockets={} requests={} reuse_ratio={}",
         sockets, requests, ratio);
  }

 private:
  mutable std::mutex mtx_;
  std::unordered_map<std::string, std::vector<std::unique_ptr<S3Connection>>>
      idle_;
  /// Counters of sockets already closed, so the tally survives their drop.
  uint64_t dropped_connects_ = 0;
  uint64_t dropped_reuses_ = 0;
  /// Cap on idle sockets kept per endpoint; a burst above this simply closes
  /// its surplus on Release rather than parking sockets forever.
  size_t max_idle_per_key_ = 64;
};

}  // namespace clio::cae::core

#endif  // CLIO_ENABLE_S3_REST

#endif  // CLIO_CAE_CORE_FACTORY_S3_CONN_POOL_H_
