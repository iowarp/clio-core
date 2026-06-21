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

#pragma once
#if CTP_ENABLE_THALLIUM
#include <clio_ctp/util/env_compat.h>
#include <arpa/inet.h>

#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <thallium.hpp>
#include <thallium/serialization/stl/string.hpp>
#include <thallium/serialization/stl/vector.hpp>

#include "clio_ctp/util/logging.h"
#include "lightbeam.h"

namespace ctp::lbm {

namespace tl = thallium;

// Process-singleton thallium engine. The engine spawns:
//   - 1 progress thread (mercury network progress; without it the user
//     thread has to drive progress and synchronous RPCs deadlock).
//   - rpc_thread_count argobots ES for running RPC handler ULTs. With v2
//     (bulk RDMA pull happens inside the handler), more handler threads
//     means more peers can pull in parallel. Set via
//     CLIO_LBM_THALLIUM_RPC_THREADS (default 4).
//
// Protocol is set via CLIO_LBM_THALLIUM_PROTOCOL (default "ofi+tcp;ofi_rxm";
// use "ofi+verbs;ofi_rxm" + FI_PROVIDER=verbs + FI_DOMAIN=<roce_domain>
// for the RDMA path).
class ThalliumEngine {
 public:
  static tl::engine &Get() {
    static std::once_flag once;
    static std::unique_ptr<tl::engine> engine;
    std::call_once(once, []() {
      const char *proto_env = ctp::env::GetCompat("LBM_THALLIUM_PROTOCOL");
      std::string proto = (proto_env && *proto_env) ? proto_env
                                                    : "ofi+tcp;ofi_rxm";
      const char *rpc_env = ctp::env::GetCompat("LBM_THALLIUM_RPC_THREADS");
      int rpc_threads = (rpc_env && *rpc_env) ? std::atoi(rpc_env) : 4;
      if (rpc_threads < 1) rpc_threads = 1;
      engine = std::make_unique<tl::engine>(
          proto, THALLIUM_SERVER_MODE,
          true /*use_progress_thread*/,
          rpc_threads /*rpc_thread_count*/);
      HLOG(kInfo,
           "ThalliumEngine: protocol={} progress_thread=1 rpc_threads={}",
           proto, rpc_threads);
    });
    return *engine;
  }
};

inline const char *ThalliumLbmRpcName() { return "ctp_lbm_send"; }

// A single received message after the RPC handler has pulled all BULK_XFER
// payloads via RDMA. The meta_blob still needs deserialization in Recv.
// local_bulks[i] is malloc'd; the receiver's ClearRecvHandles frees them.
struct ThalliumRecvEntry {
  std::string meta_blob;
  std::vector<std::pair<char *, size_t>> local_bulks;
};

struct ThalliumRecvQueue {
  std::mutex mtx;
  std::deque<ThalliumRecvEntry> q;
  std::atomic<bool> shutdown{false};
};

class ThalliumTransport : public Transport {
 public:
  explicit ThalliumTransport(TransportMode mode, const std::string &addr,
                             const std::string &protocol = "tcp",
                             int port = 0)
      : Transport(mode),
        addr_(addr),
        protocol_(protocol),
        port_(port),
        recv_q_(std::make_shared<ThalliumRecvQueue>()) {
    type_ = TransportType::kThallium;
    auto &eng = ThalliumEngine::Get();

    if (mode == TransportMode::kServer) {
      auto q = recv_q_;
      auto &eng_ref = eng;
      // Handler runs on one of the rpc_thread_count argobots ES. It
      // synchronously pulls each remote bulk via RDMA into a malloc'd
      // local buffer, then queues the metadata + buffers for the user
      // thread's Recv() to pop. Pulling inside the handler keeps the
      // sender's memory alive: Send blocks on the RPC response, which
      // we don't issue until the pulls complete.
      auto handler = [q, &eng_ref](const tl::request &req,
                                   std::string &meta_blob,
                                   std::vector<tl::bulk> &remote_bulks) {
        ThalliumRecvEntry entry;
        entry.meta_blob = std::move(meta_blob);
        entry.local_bulks.reserve(remote_bulks.size());
        try {
          tl::endpoint sender_ep = req.get_endpoint();
          for (auto &remote : remote_bulks) {
            size_t sz = remote.size();
            char *buf = static_cast<char *>(std::malloc(sz));
            if (!buf) {
              for (auto &lb : entry.local_bulks) std::free(lb.first);
              req.respond(ENOMEM);
              return;
            }
            std::vector<std::pair<void *, size_t>> segs{
                {static_cast<void *>(buf), sz}};
            tl::bulk local = eng_ref.expose(segs, tl::bulk_mode::write_only);
            // RDMA pull: remote → local. Synchronous.
            remote.on(sender_ep) >> local;
            entry.local_bulks.emplace_back(buf, sz);
          }
        } catch (const std::exception &e) {
          HLOG(kError, "ThalliumTransport handler: pull failed: {}", e.what());
          for (auto &lb : entry.local_bulks) std::free(lb.first);
          req.respond(EIO);
          return;
        }

        if (!q->shutdown.load(std::memory_order_acquire)) {
          std::lock_guard<std::mutex> lock(q->mtx);
          q->q.emplace_back(std::move(entry));
        } else {
          for (auto &lb : entry.local_bulks) std::free(lb.first);
        }
        req.respond(0);
      };
      rpc_ = eng.define(ThalliumLbmRpcName(), handler);
      HLOG(kInfo,
           "ThalliumTransport(server) listening on {} (engine self = {})",
           addr_, std::string(eng.self()));
    } else {
      try {
        peer_ = eng.lookup(addr_);
      } catch (const std::exception &e) {
        throw std::runtime_error(
            std::string("ThalliumTransport: lookup failed for ") + addr_ +
            ": " + e.what());
      }
      rpc_ = eng.define(ThalliumLbmRpcName());
      HLOG(kDebug, "ThalliumTransport(client) connected to {}", addr_);
    }
  }

  ~ThalliumTransport() {
    if (recv_q_) {
      recv_q_->shutdown.store(true, std::memory_order_release);
      std::lock_guard<std::mutex> lock(recv_q_->mtx);
      for (auto &entry : recv_q_->q) {
        for (auto &lb : entry.local_bulks) std::free(lb.first);
      }
      recv_q_->q.clear();
    }
  }

  // ----- Shared APIs -----

  // Register the buffer with thallium's engine so the peer can RDMA-read
  // from it. The tl::bulk handle is heap-allocated and stowed in
  // Bulk::desc; Send() takes ownership and frees it after the RPC.
  Bulk Expose(const ctp::ipc::FullPtr<char> &ptr, size_t data_size, u32 flags) {
    Bulk bulk;
    bulk.data = ptr;
    bulk.size = data_size;
    bulk.flags = ctp::bitfield32_t(flags);
    if (bulk.flags.Any(BULK_XFER) && ptr.ptr_ != nullptr && data_size > 0) {
      auto &eng = ThalliumEngine::Get();
      std::vector<std::pair<void *, size_t>> segs{
          {static_cast<void *>(ptr.ptr_), data_size}};
      auto *handle =
          new tl::bulk(eng.expose(segs, tl::bulk_mode::read_only));
      bulk.desc = handle;
    }
    return bulk;
  }

  void ClearRecvHandles(LbmMeta<> &meta) {
    for (auto &bulk : meta.recv) {
      if (bulk.data.ptr_) {
        std::free(bulk.data.ptr_);
        bulk.data.ptr_ = nullptr;
      }
    }
  }

  std::string GetAddress() const {
    if (IsServer()) {
      return std::string(ThalliumEngine::Get().self());
    }
    return addr_;
  }

  void RegisterEventManager(EventManager &em) { (void)em; }

  bool IsServerAlive(const LbmContext &ctx = LbmContext()) const {
    (void)ctx;
    if (IsServer()) return true;
    try {
      (void)ThalliumEngine::Get().lookup(addr_);
      return true;
    } catch (...) {
      return false;
    }
  }

  // ----- Templated Send / Recv -----

  template <typename MetaT>
  int Send(MetaT &meta, const LbmContext &ctx = LbmContext()) {
    (void)ctx;
    if (!IsClient()) {
      HLOG(kError, "ThalliumTransport::Send called on server-mode");
      return -1;
    }

    // Serialize the lightbeam metadata into a tiny blob. With v2 the
    // bulk payload bytes are not in this blob — they travel via RDMA.
    std::vector<char> meta_buf;
    {
      ctp::ipc::GlobalSerialize<std::vector<char>> ar(meta_buf);
      ar(meta);
      ar.Finalize();
    }
    std::string meta_blob(meta_buf.data(), meta_buf.size());

    // Gather the tl::bulk handles for BULK_XFER entries. The receiver
    // pulls each one via RDMA inside its handler. We free the heap
    // tl::bulk objects after the RPC returns successfully.
    std::vector<tl::bulk> bulks;
    bulks.reserve(meta.send.size());
    for (auto &b : meta.send) {
      if (b.flags.Any(BULK_XFER)) {
        if (b.desc == nullptr) {
          HLOG(kError,
               "ThalliumTransport::Send: BULK_XFER bulk has no tl::bulk "
               "handle (call Expose first)");
          return -1;
        }
        bulks.push_back(*static_cast<tl::bulk *>(b.desc));
      }
    }

    int rc = 0;
    try {
      rc = rpc_.on(peer_)(meta_blob, bulks);
    } catch (const std::exception &e) {
      HLOG(kError, "ThalliumTransport::Send RPC failed: {}", e.what());
      rc = -1;
    }

    // Free the tl::bulk handles allocated in Expose. The receiver has
    // already pulled (the handler did `remote.on(ep) >> local` before
    // responding), so it's safe to deregister now.
    for (auto &b : meta.send) {
      if (b.desc) {
        delete static_cast<tl::bulk *>(b.desc);
        b.desc = nullptr;
      }
    }
    return rc;
  }

  template <typename MetaT>
  ClientInfo Recv(MetaT &meta, const LbmContext &ctx = LbmContext()) {
    (void)ctx;
    ClientInfo info;
    if (!IsServer()) {
      info.rc = EAGAIN;
      return info;
    }

    ThalliumRecvEntry entry;
    {
      std::lock_guard<std::mutex> lock(recv_q_->mtx);
      if (recv_q_->q.empty()) {
        info.rc = EAGAIN;
        return info;
      }
      entry = std::move(recv_q_->q.front());
      recv_q_->q.pop_front();
    }

    try {
      std::vector<char> meta_buf(entry.meta_blob.begin(),
                                 entry.meta_blob.end());
      ctp::ipc::GlobalDeserialize<std::vector<char>> ar(meta_buf);
      ar(meta);
    } catch (const std::exception &e) {
      HLOG(kError, "ThalliumTransport::Recv deserialize failed: {}",
           e.what());
      for (auto &lb : entry.local_bulks) std::free(lb.first);
      info.rc = -1;
      return info;
    }

    // Attach already-pulled local buffers to meta.recv in the same order
    // they appeared as BULK_XFER entries in meta.send.
    size_t bulk_idx = 0;
    for (const auto &send_bulk : meta.send) {
      Bulk recv_bulk;
      recv_bulk.size = send_bulk.size;
      recv_bulk.flags = send_bulk.flags;
      recv_bulk.data = ctp::ipc::FullPtr<char>::GetNull();
      if (send_bulk.flags.Any(BULK_XFER)) {
        if (bulk_idx >= entry.local_bulks.size()) {
          HLOG(kError,
               "ThalliumTransport::Recv: BULK_XFER count mismatch "
               "(meta says >{}, handler pulled {})",
               bulk_idx, entry.local_bulks.size());
          for (size_t i = bulk_idx; i < entry.local_bulks.size(); ++i) {
            std::free(entry.local_bulks[i].first);
          }
          info.rc = -1;
          return info;
        }
        char *buf = entry.local_bulks[bulk_idx].first;
        size_t sz = entry.local_bulks[bulk_idx].second;
        recv_bulk.data.ptr_ = buf;
        recv_bulk.data.shm_.alloc_id_ = ctp::ipc::AllocatorId::GetNull();
        recv_bulk.data.shm_.off_ = reinterpret_cast<size_t>(buf);
        recv_bulk.size = sz;
        ++bulk_idx;
      }
      meta.recv.push_back(recv_bulk);
    }

    info.rc = 0;
    return info;
  }

 private:
  std::string addr_;
  std::string protocol_;
  int port_;
  tl::remote_procedure rpc_;
  tl::endpoint peer_;
  std::shared_ptr<ThalliumRecvQueue> recv_q_;
};

}  // namespace ctp::lbm
#endif  // CTP_ENABLE_THALLIUM
