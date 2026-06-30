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
#if CTP_ENABLE_ZMQ
#include <clio_ctp/util/env_compat.h>
#ifndef _WIN32
#include <unistd.h>
#endif
#include <zmq.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>

#include "clio_ctp/introspect/system_info.h"
#include "clio_ctp/util/logging.h"
#include "lightbeam.h"
#include "posix_socket.h"

namespace ctp::lbm {

// Async send-completion machinery (SendCompletion + release_bulk +
// zmq_noop_free + CompletionGuard + on_send_complete callback) was
// removed: Send is now always synchronous and copy-based.  ZMQ takes
// its own copy of the bulk inside the call, so the caller's buffer is
// safe to free / reuse the instant Send returns; no callback fires,
// no per-Send heap allocation, no atomic refcount to chase.

/** Free zmq_msg_t handles stored in Bulk::desc from zero-copy recv */
static inline void ClearZmqRecvHandles(LbmMeta<> &meta) {
  for (auto &bulk : meta.recv) {
    if (bulk.desc) {
      zmq_msg_t *msg = static_cast<zmq_msg_t*>(bulk.desc);
      zmq_msg_close(msg);
      delete msg;
      bulk.desc = nullptr;
    }
  }
}

/** Action that reads ZMQ_EVENTS when epoll fires on ZMQ_FD.
 *  Required by ZMQ docs: the FD is edge-triggered and won't
 *  re-arm until the application reads ZMQ_EVENTS. */
class ZmqFiredAction : public EventAction {
 public:
  void *socket_;
  explicit ZmqFiredAction(void *socket) : socket_(socket) {}
  void Run(const EventInfo &event) override {
    (void)event;
    int zmq_events = 0;
    ::size_t opt_len = sizeof(zmq_events);  // LCOV_EXCL_LINE
    zmq_getsockopt(socket_, ZMQ_EVENTS, &zmq_events, &opt_len);
  }
};

class ZeroMqTransport : public Transport {
 private:
  static void* GetSharedContext() {
    // CtxOwner holds the shared ZMQ context and destroys it at program exit,
    // ensuring libzmq releases its internal resources and LeakSanitizer is clean.
    struct CtxOwner {
      void* ctx = nullptr;
      std::mutex mtx;
      ~CtxOwner() {
        if (ctx) {
#ifdef _WIN32
          // libzmq 4.3.x's signaler aborts inside both zmq_ctx_shutdown
          // and zmq_ctx_destroy on Windows during static-destructor
          // teardown (assertion "WSASTARTUP not yet performed" in
          // signaler.cpp). The signaler's wakeup `send()` runs on a
          // mailbox thread whose Winsock state is already torn down by
          // the time the global ctx destructor runs, and there's no
          // user-side knob to fix it (we tried pinning the WSAStartup
          // refcount). The process is exiting anyway, so we leak the
          // context: the OS reclaims its sockets at process exit. Tracked
          // under a follow-up GH issue.
          (void)ctx;
#else
          // zmq_ctx_shutdown() causes all blocking ZMQ calls on open sockets
          // to return immediately with ETERM, unblocking background receive
          // threads (e.g. RecvZmqClientThread) so they can exit.
          //
          // We deliberately do NOT call zmq_ctx_destroy()/zmq_ctx_term() here:
          // it blocks until every socket opened on the context has been closed
          // with zmq_close(), and the CLIO Runtime singleton is heap-allocated
          // with a destructor (ClientFinalize, which closes the socket) that is
          // never invoked. Some workloads (force-net tag/query/compose,
          // assimilation) reach process exit with a DEALER socket on this
          // shared context still open, so zmq_ctx_term would hang forever
          // (observed: main thread parked in poll(-1) inside ctx_t::terminate
          // at atexit). The process is exiting anyway, so we leak the context
          // and let the OS reclaim it — a one-time leak at exit is strictly
          // better than a shutdown deadlock. Mirrors the Windows path above.
          zmq_ctx_shutdown(ctx);
#endif
          ctx = nullptr;
        }
      }
    };
    static CtxOwner owner;
    std::lock_guard<std::mutex> lock(owner.mtx);
    if (!owner.ctx) {
      owner.ctx = zmq_ctx_new();
      // I/O thread count. Default 2 saturated at 64+ nodes during SWIM
      // probe rounds + cross-node SendIn fan-out (each clio daemon
      // talks to N-1 peers; N²=4096 connections at N=64 is enough to
      // bottleneck 2 I/O threads). 8 scales comfortably to ~512.
      // Override at runtime via CLIO_ZMQ_IO_THREADS env if needed.
      const char *iot_env = ctp::env::GetCompat("ZMQ_IO_THREADS");
      int iot = (iot_env && *iot_env) ? std::atoi(iot_env) : 8;
      if (iot < 1) iot = 1;
      zmq_ctx_set(owner.ctx, ZMQ_IO_THREADS, iot);
      // Non-blocking termination semantics: new sockets default to LINGER=0 so
      // a close never waits for unsent messages, and blocking calls return
      // ETERM promptly on shutdown. Belt-and-suspenders alongside the leak in
      // ~CtxOwner (we never call the blocking zmq_ctx_term).
      zmq_ctx_set(owner.ctx, ZMQ_BLOCKY, 0);
      HLOG(kInfo, "[ZeroMqTransport] Created shared context with {} I/O threads", iot);
    }
    return owner.ctx;
  }

 public:
  // Wire topology is fixed: ROUTER on the server, DEALER on each client,
  // with identity + empty-delimiter frames around every multipart
  // logical message.  The ROUTER → identity prefix is what lets SendIn /
  // RecvIn on one side and SendOut / RecvOut on the other share a
  // single bidirectional socket pair (DEALER ↔ ROUTER routes responses
  // back to the originating client by identity).  The previous
  // kPushPull alternative (PUSH/PULL, unidirectional) has been removed.
  explicit ZeroMqTransport(TransportMode mode, const std::string& addr,
                           const std::string& protocol = "tcp", int port = 8192,
                           bool use_shared_ctx = false)
      : Transport(mode),
        addr_(addr),
        protocol_(protocol),
        port_(port),
        use_shared_ctx_(use_shared_ctx),
        zmq_fired_action_(nullptr) {
    type_ = TransportType::kZeroMq;
    sock::InitSocketLib();

    // Optional: pin TCP traffic to a specific local network interface on
    // multi-rail fabrics (e.g. Aurora's Slingshot HSN where each compute
    // node has hsn0..hsn7). When LIGHTBEAM_TCP_DEVICE is set:
    //   - ROUTER (server) binds the device's IP directly: tcp://hsn0:port
    //   - DEALER (client) source-binds outbound traffic to the device
    //     using ZMQ's `<source>;<destination>` endpoint syntax, so
    //     connect() routes via the chosen NIC regardless of how the
    //     destination FQDN resolves. Both ends pinned to the same device
    //     -> symmetric routing -> no asymmetric drops at scale.
    const char *bind_device_env = std::getenv("LIGHTBEAM_TCP_DEVICE");
    std::string bind_device =
        (bind_device_env && *bind_device_env) ? bind_device_env : "";

    // Loopback endpoints must stay on the host's lo interface — clients
    // connecting to 127.0.0.1 / ::1 / localhost can't reach a socket
    // pinned to a routed NIC, and the server's local-only ROUTER (used
    // by same-host clio_cte / runtime clients) MUST keep listening on
    // 127.0.0.1 even when LIGHTBEAM_TCP_DEVICE is set for cross-node
    // traffic. Otherwise every same-host client times out in
    // WaitForLocalServer.
    auto is_loopback_addr = [](const std::string &a) {
      return a == "127.0.0.1" || a == "::1" || a == "localhost";
    };

    std::string full_url;
    if (protocol_ == "ipc") {
      full_url = "ipc://" + addr_;
    } else if (!bind_device.empty() && !is_loopback_addr(addr_) &&
               mode == TransportMode::kClient) {
      // Client: source-bind outbound traffic to the chosen NIC so
      // connect() routes via that rail regardless of how the
      // destination FQDN resolves. ZMQ syntax:
      //   tcp://<src_dev>:0;<dst_addr>:<dst_port>
      full_url = protocol_ + "://" + bind_device + ":0;" + addr_ + ":" +
                 std::to_string(port_);
    } else {
      // Server: keep the original bind address. Aurora HSN FQDNs are
      // multi-A (one entry per rail); pinning the server to a single
      // rail's IP makes peer DEALERs whose FQDN lookup landed on a
      // different rail unable to reach us. Listening on 0.0.0.0
      // (every interface) lets any rail accept connections; the
      // client-side source-bind above is what makes outbound
      // routing symmetric.
      full_url = protocol_ + "://" + addr_ + ":" + std::to_string(port_);
    }

    if (mode == TransportMode::kClient) {
      // Client socket: DEALER, connect-style.
      ctx_ = GetSharedContext();
      owns_ctx_ = false;
      socket_ = zmq_socket(ctx_, ZMQ_DEALER);

      // ZMQ_IDENTITY: the server's ROUTER uses this as the response routing
      // prefix. It MUST be unique per DEALER socket: when several DEALERs in one
      // process (e.g. per-thread client send sockets) connect to the same ROUTER
      // with an identical identity, the ROUTER coalesces them onto one
      // connection and silently drops every sender's frames but one. A
      // per-process atomic sequence guarantees uniqueness; the host part (before
      // the first ':') stays parseable for response dial-back.
      {
        static std::atomic<uint64_t> dealer_seq{0};
        std::string hostname = ctp::SystemInfo::GetHostname();
        uint32_t pid = static_cast<uint32_t>(ctp::SystemInfo::GetPid());
        uint64_t seq = dealer_seq.fetch_add(1, std::memory_order_relaxed);
        std::string identity = hostname + ":" + std::to_string(pid) + ":" +
                               std::to_string(seq);
        zmq_setsockopt(socket_, ZMQ_IDENTITY, identity.data(),
                        identity.size());
      }

      HLOG(kDebug, "ZeroMqTransport(DEALER) connecting to URL: {}", full_url);

      int immediate = 0;
      zmq_setsockopt(socket_, ZMQ_IMMEDIATE, &immediate, sizeof(immediate));

      int timeout = 5000;
      zmq_setsockopt(socket_, ZMQ_SNDTIMEO, &timeout, sizeof(timeout));

      int sndbuf = 4 * 1024 * 1024;
      zmq_setsockopt(socket_, ZMQ_SNDBUF, &sndbuf, sizeof(sndbuf));

      int rcvbuf = 4 * 1024 * 1024;
      zmq_setsockopt(socket_, ZMQ_RCVBUF, &rcvbuf, sizeof(rcvbuf));

      // ZMQ socket-level high water marks. Default = 1000 messages.
      // With 1 MiB CTE blob payloads, that's only ~1 GB of queue
      // headroom. When a single FUSE-driven workload submits tens of
      // thousands of cross-node PutBlob tasks (24 ranks * 256 pages
      // / 2 nodes ~= 3 k cross-node tasks/node), the queue saturates
      // and zmq_send blocks (or drops with EAGAIN if DONTWAIT) for
      // long enough that the net worker can't reach Recv() and the
      // bidirectional flow deadlocks. Bump both HWMs to 100 k so the
      // ZMQ I/O thread + TCP path (4 MiB SNDBUF/RCVBUF) is the
      // bottleneck, not the application-side queue.
      int sndhwm = 100000;
      zmq_setsockopt(socket_, ZMQ_SNDHWM, &sndhwm, sizeof(sndhwm));
      int rcvhwm = 100000;
      zmq_setsockopt(socket_, ZMQ_RCVHWM, &rcvhwm, sizeof(rcvhwm));

      // ZMTP heartbeat: detect dead connections within seconds
      // ZMTP heartbeat: at scale (>=64 nodes) the daemon's ROUTER
      // gets busy with cross-node SWIM probes and can't respond to a
      // local DEALER's ZMTP greeting within the prior 3s window —
      // observed at iow_s64 as a HANDSHAKE_FAILED_NO_DETAIL value=32
      // (EPIPE) loop on tcp://127.0.0.1:9416. Bump to 30s so the
      // greeting can complete even under heavy probe traffic.
      int hb_ivl = 5000;     // ZMTP PING every 5 s
      zmq_setsockopt(socket_, ZMQ_HEARTBEAT_IVL, &hb_ivl, sizeof(hb_ivl));
      int hb_timeout = 30000;  // dead after 30 s of no traffic
      zmq_setsockopt(socket_, ZMQ_HEARTBEAT_TIMEOUT, &hb_timeout, sizeof(hb_timeout));
      int hb_ttl = 30000;
      zmq_setsockopt(socket_, ZMQ_HEARTBEAT_TTL, &hb_ttl, sizeof(hb_ttl));

      int rc = zmq_connect(socket_, full_url.c_str());
      if (rc == -1) {
        std::string err = "ZeroMqTransport(DEALER) failed to connect to URL '" +
                          full_url + "': " + zmq_strerror(zmq_errno());
        zmq_close(socket_);
        throw std::runtime_error(err);
      }

      zmq_pollitem_t poll_item = {socket_, 0, ZMQ_POLLOUT, 0};
      int poll_timeout_ms = 5000;
      int poll_rc = zmq_poll(&poll_item, 1, poll_timeout_ms);

      if (poll_rc < 0) {
        HLOG(kError, "[ZeroMqTransport] Poll failed for {}: {}", full_url,
             zmq_strerror(zmq_errno()));
      } else if (poll_rc == 0) {
        HLOG(kWarning,
             "[ZeroMqTransport] Poll timeout - connection to {} may not be ready",
             full_url);
      }

      HLOG(kDebug, "ZeroMqTransport(DEALER) connected to {}", full_url);
      zmq_fired_action_.socket_ = socket_;
    } else {
      // Server socket: ROUTER, bind-style.
      if (use_shared_ctx_) {
        // Bind on the process-wide shared context, which is intentionally
        // leaked at exit (see GetSharedContext / CtxOwner) and never
        // zmq_ctx_destroy'd. On Windows that destroy aborts inside libzmq's
        // signaler ("WSASTARTUP not yet performed"); a ROUTER on its own owned
        // context in a clean-exit process (e.g. the client response listener)
        // would trip it. IO_THREADS are already configured on the shared
        // context, so don't re-set them here.
        ctx_ = GetSharedContext();
        owns_ctx_ = false;
      } else {
        ctx_ = zmq_ctx_new();
        owns_ctx_ = true;
        const char *iot_env = ctp::env::GetCompat("ZMQ_IO_THREADS");
        int iot = (iot_env && *iot_env) ? std::atoi(iot_env) : 8;
        if (iot < 1) iot = 1;
        zmq_ctx_set(ctx_, ZMQ_IO_THREADS, iot);
      }
      socket_ = zmq_socket(ctx_, ZMQ_ROUTER);

      // Mandatory routing makes zmq_send fail loudly if the destination
      // identity isn't connected (instead of silently dropping); handover
      // hot-swaps a stale connection when a client with the same identity
      // reconnects (typical after a restart).
      int mandatory = 1;
      zmq_setsockopt(socket_, ZMQ_ROUTER_MANDATORY, &mandatory,
                      sizeof(mandatory));
      int handover = 1;
      zmq_setsockopt(socket_, ZMQ_ROUTER_HANDOVER, &handover,
                      sizeof(handover));

      int rcvbuf = 4 * 1024 * 1024;
      zmq_setsockopt(socket_, ZMQ_RCVBUF, &rcvbuf, sizeof(rcvbuf));

      int sndbuf = 4 * 1024 * 1024;
      zmq_setsockopt(socket_, ZMQ_SNDBUF, &sndbuf, sizeof(sndbuf));

      // Mirror the DEALER side: lift socket-level HWMs so the queue
      // doesn't become the bottleneck under burst cross-node CTE traffic.
      int sndhwm = 100000;
      zmq_setsockopt(socket_, ZMQ_SNDHWM, &sndhwm, sizeof(sndhwm));
      int rcvhwm = 100000;
      zmq_setsockopt(socket_, ZMQ_RCVHWM, &rcvhwm, sizeof(rcvhwm));

      // The ROUTER's send had no SNDTIMEO, which defaults to -1
      // (block forever). Combined with a single net worker that also
      // owns Recv(), this is what dead-locked SendOut: once the peer's
      // RX queue back-pressured, this side's zmq_send waited
      // indefinitely and Recv() never ran on the other half of the
      // flow. Cap blocking at 1 s; callers re-queue on EAGAIN.
      int sndtimeo = 1000;
      zmq_setsockopt(socket_, ZMQ_SNDTIMEO, &sndtimeo, sizeof(sndtimeo));

      // Bound how long a blocking frame read (the flags=0 delim/meta/bulk reads
      // in RecvMetadata/RecvBulks) can wait. The identity frame is read
      // DONTWAIT, but if a multipart message is ever truncated on the wire the
      // follow-on frames would block forever — wedging the recv thread so it
      // can't honor recv_shutdown_ at teardown (the leaked Windows context
      // sends no ETERM to wake it). 1 s lets the read fail with EAGAIN and the
      // recv loop re-check the shutdown flag; well-formed messages arrive
      // atomically so this never trips in steady state.
      int rcvtimeo = 1000;
      zmq_setsockopt(socket_, ZMQ_RCVTIMEO, &rcvtimeo, sizeof(rcvtimeo));

      // ZMTP heartbeat — same scale-friendly window as DEALER side.
      int hb_ivl = 5000;
      zmq_setsockopt(socket_, ZMQ_HEARTBEAT_IVL, &hb_ivl, sizeof(hb_ivl));
      int hb_timeout = 30000;
      zmq_setsockopt(socket_, ZMQ_HEARTBEAT_TIMEOUT, &hb_timeout, sizeof(hb_timeout));
      int hb_ttl = 30000;
      zmq_setsockopt(socket_, ZMQ_HEARTBEAT_TTL, &hb_ttl, sizeof(hb_ttl));

      HLOG(kDebug, "ZeroMqTransport(ROUTER) binding to URL: {}", full_url);
      int rc = zmq_bind(socket_, full_url.c_str());
      if (rc == -1) {
        std::string err = std::string("ZeroMqTransport(ROUTER) failed to "
                                       "bind to URL '") +
                          full_url + "': " + zmq_strerror(zmq_errno());
        zmq_close(socket_);
        zmq_ctx_destroy(ctx_);
        throw std::runtime_error(err);
      }
      HLOG(kDebug, "ZeroMqTransport(ROUTER) bound successfully to {}",
           full_url);

      // Resolve the actual bound TCP port. When the caller asked for an
      // ephemeral port (port_ == 0), the kernel assigns one; ZMQ_LAST_ENDPOINT
      // reports the concrete "tcp://addr:port" it bound, whose trailing port is
      // what clients advertise for dial-back. For a fixed bind, that port is
      // simply port_.
      if (protocol_ == "tcp") {
        if (port_ != 0) {
          bound_port_ = port_;
        } else {
          char endpoint[256];
          size_t endpoint_len = sizeof(endpoint);
          if (zmq_getsockopt(socket_, ZMQ_LAST_ENDPOINT, endpoint,
                             &endpoint_len) == 0) {
            std::string ep(endpoint);
            size_t colon = ep.find_last_of(':');
            if (colon != std::string::npos) {
              bound_port_ = std::atoi(ep.c_str() + colon + 1);
            }
          }
        }
      }
      zmq_fired_action_.socket_ = socket_;
    }
  }

  ~ZeroMqTransport() {
    HLOG(kDebug,
         "ZeroMqTransport destructor - closing socket to {}:{} "
         "(shutdown={} owns_ctx={})",
         addr_, port_, sock::IsSocketLibShutdown(), owns_ctx_);

    // During process shutdown on Windows, tearing down a ZMQ socket OR a
    // ZMQ context trips libzmq's signaler WSASTARTUP assertion: libzmq has
    // already torn down its own Winsock state, so the signaler's wakeup send()
    // aborts (signaler.cpp:163). The shared context is already leaked for this
    // reason (see GetSharedContext's CtxOwner); do the same for every transport
    // once shutdown has begun — skip both zmq_close and zmq_ctx_destroy and let
    // the OS reclaim the socket (and any private context) at process exit. On
    // POSIX IsSocketLibShutdown() is always false, so teardown is unchanged.
    const bool leak = sock::IsSocketLibShutdown();
    if (!leak) {
      int linger = 0;  // Close immediately; don't wait for unsent messages
      zmq_setsockopt(socket_, ZMQ_LINGER, &linger, sizeof(linger));
      zmq_close(socket_);
      if (owns_ctx_) {
        zmq_ctx_destroy(ctx_);
      }
    }
    HLOG(kDebug, "ZeroMqTransport destructor - socket closed (leak={})", leak);
  }

  Bulk Expose(const ctp::ipc::FullPtr<char>& ptr, size_t data_size,
              u32 flags) {
    Bulk bulk;
    bulk.data = ptr;
    bulk.size = data_size;
    bulk.flags = ctp::bitfield32_t(flags);
    return bulk;
  }

  // EINTR retry wrappers. ZMQ's documentation calls EINTR "interrupted
  // system call - just retry"; without retrying, any signal delivered
  // to the calling process (SIGCHLD from a child exit, SIGALRM, etc.)
  // poisons the multipart message in-flight and the DEALER socket is
  // wedged: every subsequent Send returns EINTR or a desync error
  // because the receiver is still expecting frames from the half-sent
  // message. wfbench's stress-ng subprocesses spam SIGCHLD on exit,
  // which deterministically triggers this path on every Put/Get blob.
  static inline int zmq_send_eintr(void *sock, const void *buf,
                                   size_t len, int flags) {
    while (true) {
      int rc = zmq_send(sock, buf, len, flags);
      if (rc != -1 || zmq_errno() != EINTR) return rc;
    }
  }

  static inline int zmq_msg_send_eintr(zmq_msg_t *msg, void *sock,
                                       int flags) {
    while (true) {
      int rc = zmq_msg_send(msg, sock, flags);
      if (rc != -1 || zmq_errno() != EINTR) return rc;
    }
  }

  static inline int zmq_msg_recv_eintr(zmq_msg_t *msg, void *sock,
                                       int flags) {
    while (true) {
      int rc = zmq_msg_recv(msg, sock, flags);
      if (rc != -1 || zmq_errno() != EINTR) return rc;
    }
  }

  template <typename MetaT>
  int Send(MetaT& meta, const LbmContext& ctx = LbmContext()) {
    // Serialize ALL socket access (send + recv share one mutex): the
    // multipart send must not interleave with another send's frames NOR
    // run concurrently with a recv on this same non-thread-safe socket.
    std::lock_guard<std::mutex> lock(sock_mtx_);

    // Compute send_bulks before serialization so receiver knows how many
    meta.send_bulks = 0;
    for (size_t i = 0; i < meta.send.size(); ++i) {
      if (meta.send[i].flags.Any(BULK_XFER)) {
        meta.send_bulks++;
      }
    }

    std::vector<char> meta_buf;
    {
      ctp::ipc::GlobalSerialize<std::vector<char>> ar(meta_buf);
      ar(meta);
      ar.Finalize();
    }
    std::string meta_str(meta_buf.begin(), meta_buf.end());
    const size_t write_bulk_count = meta.send_bulks;

    // Send is always synchronous and copy-based.  Bulks go through plain
    // zmq_send which copies into ZMQ's own frame inside the call, so the
    // caller's buffer is safe to free / reuse the instant this returns.
    // The 1-second ZMQ_SNDTIMEO socket option caps how long any single
    // frame can block under back-pressure; callers handle the EAGAIN.
    if (IsServer() && !meta.client_info_.identity_.empty()) {
      // ROUTER: send identity + delim frames before meta.
      int rc = zmq_send_eintr(socket_, meta.client_info_.identity_.data(),
                              meta.client_info_.identity_.size(),
                              ZMQ_SNDMORE);
      if (rc == -1) {
        HLOG(kError, "ZeroMqTransport::Send(ROUTER) - identity frame FAILED: {}",
             zmq_strerror(zmq_errno()));
        return zmq_errno();
      }
      rc = zmq_send_eintr(socket_, "", 0, ZMQ_SNDMORE);
      if (rc == -1) {
        HLOG(kError, "ZeroMqTransport::Send(ROUTER) - delimiter frame FAILED: {}",
             zmq_strerror(zmq_errno()));
        return zmq_errno();
      }
    } else if (IsClient()) {
      // DEALER: empty delimiter frame, no identity.
      int rc = zmq_send_eintr(socket_, "", 0, ZMQ_SNDMORE);
      if (rc == -1) {
        HLOG(kError, "ZeroMqTransport::Send(DEALER) - delimiter frame FAILED: {}",
             zmq_strerror(zmq_errno()));
        return zmq_errno();
      }
    }

    int flags = (write_bulk_count > 0) ? ZMQ_SNDMORE : 0;
    int rc = zmq_send_eintr(socket_, meta_str.data(), meta_str.size(), flags);
    if (rc == -1) {
      HLOG(kError, "ZeroMqTransport::Send - meta FAILED: {}",
           zmq_strerror(zmq_errno()));
      return zmq_errno();
    }

    size_t sent_count = 0;
    for (size_t i = 0; i < meta.send.size(); ++i) {
      if (!meta.send[i].flags.Any(BULK_XFER)) continue;

      sent_count++;
      const int bulk_flags = (sent_count < write_bulk_count) ? ZMQ_SNDMORE : 0;

      rc = zmq_send_eintr(socket_, meta.send[i].data.ptr_,
                           meta.send[i].size, bulk_flags);
      if (rc == -1) {
        HLOG(kError, "ZeroMqTransport::Send - bulk {} FAILED: {}", i,
             zmq_strerror(zmq_errno()));
        return zmq_errno();
      }
    }
    return 0;
  }

  template <typename MetaT>
  ClientInfo Recv(MetaT& meta, const LbmContext& ctx = LbmContext()) {
    // Same single socket mutex as Send: recv must not run concurrently
    // with a send (or another recv) on this non-thread-safe socket, and
    // a multipart recv must consume id/delim/meta/bulk frames atomically.
    std::lock_guard<std::mutex> lock(sock_mtx_);

    ClientInfo info;
    info.rc = RecvMetadata(meta, ctx);
    if (info.rc != 0) {
      return info;
    }
#if !CTP_IS_GPU
    // identity_ only exists off-device (it's a std::string; ZMQ is host-only).
    // Guard the use so nvcc's device pass — which still parses this host-only
    // template body — doesn't trip over the #if !CTP_IS_GPU'd-out member.
    info.identity_ = meta.client_info_.identity_;
#endif
    // Set up recv entries from send descriptors.
    for (const auto& send_bulk : meta.send) {
      Bulk recv_bulk;
      recv_bulk.size = send_bulk.size;
      recv_bulk.flags = send_bulk.flags;
      recv_bulk.data = ctp::ipc::FullPtr<char>::GetNull();
      meta.recv.push_back(recv_bulk);
    }
    info.rc = RecvBulks(meta, ctx);
    return info;
  }

 private:
  // After a partial/truncated multipart read, discard any frames still pending
  // for the current logical message so the next Recv() starts cleanly on a
  // message boundary (prevents a one-time wire glitch from desyncing the
  // stream permanently). Non-blocking throughout.
  void DrainToMessageBoundary() {
    while (true) {
      int more = 0;
      size_t more_sz = sizeof(more);
      if (zmq_getsockopt(socket_, ZMQ_RCVMORE, &more, &more_sz) != 0 || !more) {
        break;
      }
      zmq_msg_t junk;
      zmq_msg_init(&junk);
      int rc = zmq_msg_recv(&junk, socket_, ZMQ_DONTWAIT);
      zmq_msg_close(&junk);
      if (rc == -1) break;
    }
  }

  template <typename MetaT>
  int RecvMetadata(MetaT& meta, const LbmContext& ctx = LbmContext()) {
    (void)ctx;
    zmq_msg_t msg;
    zmq_msg_init(&msg);
    int rc;
    if (IsServer()) {
      // ROUTER: [identity, delim, meta]. The identity is read DONTWAIT so a
      // polling caller sees EAGAIN cleanly when nothing is queued. ZMQ delivers
      // a multipart message atomically, so once the identity is in hand the
      // delim and meta are already buffered — read them DONTWAIT too. Staying
      // non-blocking is what lets the recv thread always honor its shutdown
      // flag; a blocking flags=0 read on a truncated message would wedge it.
      zmq_msg_t identity_msg;
      zmq_msg_init(&identity_msg);
      int rc_id = zmq_msg_recv(&identity_msg, socket_, ZMQ_DONTWAIT);
      if (rc_id == -1) {
        int err = zmq_errno();
        zmq_msg_close(&identity_msg);
        zmq_msg_close(&msg);
        return err;
      }
      meta.client_info_.identity_ = std::string(
          static_cast<char*>(zmq_msg_data(&identity_msg)),
          zmq_msg_size(&identity_msg));
      zmq_msg_close(&identity_msg);

      zmq_msg_t delim_msg;
      zmq_msg_init(&delim_msg);
      int rc_d = zmq_msg_recv(&delim_msg, socket_, ZMQ_DONTWAIT);
      zmq_msg_close(&delim_msg);
      if (rc_d == -1) {
        DrainToMessageBoundary();
        zmq_msg_close(&msg);
        return zmq_errno();
      }

      rc = zmq_msg_recv(&msg, socket_, ZMQ_DONTWAIT);
    } else {
      // DEALER: [delim, meta]. Delim DONTWAIT to detect "no message"; meta is
      // buffered with it (atomic multipart), so read it DONTWAIT as well.
      zmq_msg_t delim_msg;
      zmq_msg_init(&delim_msg);
      int rc_d = zmq_msg_recv(&delim_msg, socket_, ZMQ_DONTWAIT);
      if (rc_d == -1) {
        int err = zmq_errno();
        zmq_msg_close(&delim_msg);
        zmq_msg_close(&msg);
        return err;
      }
      zmq_msg_close(&delim_msg);
      rc = zmq_msg_recv(&msg, socket_, ZMQ_DONTWAIT);
    }

    if (rc == -1) {
      int err = zmq_errno();
      DrainToMessageBoundary();
      zmq_msg_close(&msg);
      return err;
    }

    size_t msg_size = zmq_msg_size(&msg);
    try {
      std::vector<char> meta_buf(static_cast<char*>(zmq_msg_data(&msg)),
                                  static_cast<char*>(zmq_msg_data(&msg)) + msg_size);
      ctp::ipc::GlobalDeserialize<std::vector<char>> ar(meta_buf);
      ar(meta);
    } catch (const std::exception& e) {
      HLOG(kFatal,
           "ZeroMQ RecvMetadata: Deserialization failed - {} (msg_size={})",
           e.what(), msg_size);
      zmq_msg_close(&msg);
      return -1;
    }
    zmq_msg_close(&msg);
    return 0;
  }

  template <typename MetaT>
  int RecvBulks(MetaT& meta, const LbmContext& ctx = LbmContext()) {
    (void)ctx;
    size_t recv_count = 0;
    for (size_t i = 0; i < meta.recv.size(); ++i) {
      if (!meta.recv[i].flags.Any(BULK_XFER)) {
        continue;
      }
      recv_count++;
      // Bulks ride in the same atomic multipart message as the meta, so they're
      // already buffered by the time we get here — read DONTWAIT to stay
      // non-blocking. (The previous ZMQ_RCVMORE-as-recv-flag was a no-op-ish
      // misuse: ZMQ_RCVMORE is a getsockopt option, not a recv flag.)
      int flags = ZMQ_DONTWAIT;

      if (meta.recv[i].data.ptr_) {
        zmq_msg_t zmq_msg;
        zmq_msg_init(&zmq_msg);
        int rc = zmq_msg_recv_eintr(&zmq_msg, socket_, flags);
        if (rc == -1) {
          int err = zmq_errno();
          zmq_msg_close(&zmq_msg);
          return err;
        }
        memcpy(meta.recv[i].data.ptr_,
               zmq_msg_data(&zmq_msg), meta.recv[i].size);
        zmq_msg_close(&zmq_msg);
      } else {
        zmq_msg_t *zmq_msg = new zmq_msg_t;
        zmq_msg_init(zmq_msg);
        int rc = zmq_msg_recv_eintr(zmq_msg, socket_, flags);
        if (rc == -1) {
          int err = zmq_errno();
          zmq_msg_close(zmq_msg);
          delete zmq_msg;
          return err;
        }
        char *zmq_data = static_cast<char*>(zmq_msg_data(zmq_msg));
        meta.recv[i].data.ptr_ = zmq_data;
        meta.recv[i].data.shm_.alloc_id_ = ctp::ipc::AllocatorId::GetNull();
        meta.recv[i].data.shm_.off_ = reinterpret_cast<size_t>(zmq_data);
        meta.recv[i].desc = zmq_msg;
      }
    }
    return 0;
  }

 public:
  void ClearRecvHandles(LbmMeta<>& meta) {
    ClearZmqRecvHandles(meta);
  }

  void RegisterEventManager(EventManager &em) {
    int fd;
    size_t fd_size = sizeof(fd);
    zmq_getsockopt(socket_, ZMQ_FD, &fd, reinterpret_cast<::size_t *>(&fd_size));
    if (fd >= 0) {
      em.AddEvent(fd, kDefaultReadEvent, nullptr);
    }
  }

  /** Block up to timeout_ms until this socket is readable, using ZMQ's native
   *  zmq_poll. This is the correct readiness primitive for a ZMQ socket: it
   *  checks ZMQ_EVENTS rather than the raw ZMQ_FD, so it works on Windows where
   *  WSAEventSelect/epoll cannot watch ZMQ_FD. Holds sock_mtx_ so it never runs
   *  concurrently with Send/Recv on this non-thread-safe socket; zmq_poll
   *  returns the instant data is ready, so it only blocks a would-be sender for
   *  at most timeout_ms while the socket is genuinely idle. Returns >0 if
   *  readable, 0 on timeout, <0 on error. */
  int PollRecv(int timeout_ms) {
    std::lock_guard<std::mutex> lock(sock_mtx_);
    zmq_pollitem_t item = {socket_, 0, ZMQ_POLLIN, 0};
    return zmq_poll(&item, 1, timeout_ms);
  }

  std::string GetAddress() const { return addr_; }

  /** Check if the server is still alive via a connect probe. The actual
   *  socket API lives in socket_{win,posix}.cc so no Windows header leaks in
   *  through this header. */
  bool IsServerAlive(const LbmContext& ctx = LbmContext()) const {
    (void)ctx;
    return sock::IsServerAlive(addr_, port_, protocol_);
  }

 public:
  /** Actual bound TCP port for a server socket. Equals port_ for a fixed
   *  bind; resolved from ZMQ_LAST_ENDPOINT when bound to ephemeral port 0.
   *  0 for client sockets / ipc:// servers. */
  int GetBoundPort() const { return bound_port_; }

 private:
  std::string addr_;
  std::string protocol_;
  int port_;
  int bound_port_ = 0;
  bool use_shared_ctx_ = false;
  void* ctx_;
  bool owns_ctx_;
  void* socket_;
  ZmqFiredAction zmq_fired_action_;
  // ZMQ sockets are not thread-safe (per the ZeroMQ guide) — and the
  // constraint is the WHOLE socket, not send vs recv separately:
  // concurrent send-on-one-thread + recv-on-another is still undefined
  // behavior and corrupts libzmq's internal pipe_t/msg_t state
  // (observed: SIGSEGV in zmq::pipe_t::read, "double free or
  // corruption", zmq object.cpp:142 assertion under client
  // connect/disconnect churn — the daemon's dedicated client-recv
  // thread races the periodic AsyncClientSend worker task on this same
  // ROUTER socket). A SINGLE mutex therefore guards ALL socket access
  // (every multipart send AND recv), so the socket is only ever touched
  // by one thread at a time. One non-recursive mutex, never nested with
  // any other lock => deadlock-free by construction.
  std::mutex sock_mtx_;
};

}  // namespace ctp::lbm

#endif  // CTP_ENABLE_ZMQ
