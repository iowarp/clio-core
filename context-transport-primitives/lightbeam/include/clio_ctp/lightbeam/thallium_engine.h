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
#include <unistd.h>

#include <atomic>
#include <cctype>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <thallium.hpp>

#include "clio_ctp/util/logging.h"
#include "posix_socket.h"

namespace ctp::lbm {

namespace tl = thallium;

/**
 * Read a CLIO_LBM_THALLIUM_<suffix> string setting.
 *
 * @param suffix Setting name without the CLIO_LBM_THALLIUM_ prefix.
 * @param dflt Value returned when the variable is unset or empty.
 * @return The setting's value.
 */
inline std::string ThalliumEnvStr(const char *suffix, const std::string &dflt) {
  std::string name = std::string("LBM_THALLIUM_") + suffix;
  const char *v = ctp::env::GetCompat(name.c_str());
  return (v != nullptr && *v != '\0') ? std::string(v) : dflt;
}

/**
 * Read a CLIO_LBM_THALLIUM_<suffix> size/count setting. Accepts a plain
 * number or one with a binary suffix (k, m, g; case-insensitive, an optional
 * trailing "b"/"ib" is ignored), e.g. "65536", "64k", "1g".
 *
 * @param suffix Setting name without the CLIO_LBM_THALLIUM_ prefix.
 * @param dflt Value returned when the variable is unset or malformed.
 * @return The parsed value.
 */
inline long long ThalliumEnvSize(const char *suffix, long long dflt) {
  std::string s = ThalliumEnvStr(suffix, "");
  if (s.empty()) return dflt;
  char *end = nullptr;
  long long v = std::strtoll(s.c_str(), &end, 10);
  if (end == s.c_str()) return dflt;
  switch (std::tolower(static_cast<unsigned char>(*end))) {
    case 'k': v <<= 10; break;
    case 'm': v <<= 20; break;
    case 'g': v <<= 30; break;
    default: break;
  }
  return v;
}

/**
 * Process-wide Thallium settings, read once from the environment.
 * Every field maps to one CLIO_LBM_THALLIUM_* variable (see
 * the networking section of context-runtime/config/clio_default.yaml).
 */
struct ThalliumConfig {
  std::string protocol_;   /**< PROTOCOL: Mercury protocol, e.g. ofi+cxi */
  std::string listen_;     /**< LISTEN: text after "://" in the listen address */
  std::string auth_key_;   /**< AUTH_KEY: NA authorization key ("" = default) */
  int rpc_threads_;        /**< RPC_THREADS: handler execution streams */
  int progress_threads_;   /**< PROGRESS_THREADS: 1 dedicated ES, 0 shared */
  size_t bulk_threshold_;  /**< BULK_THRESHOLD: segments >= this use RDMA pull */
  int rpc_timeout_ms_;     /**< TIMEOUT_MS: per-send RPC timeout, 0 = none */
  size_t queue_msgs_;      /**< RECV_QUEUE_MSGS: receive queue bound, messages */
  size_t queue_bytes_;     /**< RECV_QUEUE_BYTES: receive queue bound, bytes */
  int queue_wait_ms_;      /**< RECV_QUEUE_WAIT_MS: handler wait on a full queue */
  int resolve_timeout_ms_; /**< RESOLVE_TIMEOUT_MS: address exchange reply wait */

  /**
   * Parse the settings from the environment.
   * @return A fully populated configuration with defaults applied.
   */
  static ThalliumConfig Load() {
    ThalliumConfig c;
    c.protocol_ = ThalliumEnvStr("PROTOCOL", "ofi+tcp;ofi_rxm");
    c.listen_ = ThalliumEnvStr("LISTEN", "");
    c.auth_key_ = ThalliumEnvStr("AUTH_KEY", "");
    c.rpc_threads_ = static_cast<int>(ThalliumEnvSize("RPC_THREADS", 4));
    if (c.rpc_threads_ < 1) c.rpc_threads_ = 1;
    c.progress_threads_ =
        static_cast<int>(ThalliumEnvSize("PROGRESS_THREADS", 1));
    if (c.progress_threads_ > 1) c.progress_threads_ = 1;
    if (c.progress_threads_ < 0) c.progress_threads_ = 0;
    c.bulk_threshold_ =
        static_cast<size_t>(ThalliumEnvSize("BULK_THRESHOLD", 64 * 1024));
    c.rpc_timeout_ms_ = static_cast<int>(ThalliumEnvSize("TIMEOUT_MS", 10000));
    c.queue_msgs_ =
        static_cast<size_t>(ThalliumEnvSize("RECV_QUEUE_MSGS", 1024));
    if (c.queue_msgs_ < 1) c.queue_msgs_ = 1;
    c.queue_bytes_ =
        static_cast<size_t>(ThalliumEnvSize("RECV_QUEUE_BYTES", 1LL << 30));
    c.queue_wait_ms_ =
        static_cast<int>(ThalliumEnvSize("RECV_QUEUE_WAIT_MS", 1000));
    c.resolve_timeout_ms_ =
        static_cast<int>(ThalliumEnvSize("RESOLVE_TIMEOUT_MS", 5000));
    return c;
  }

  /** @return The process-wide configuration (parsed on first use). */
  static const ThalliumConfig &Get() {
    static const ThalliumConfig c = Load();
    return c;
  }

  /**
   * Whether the protocol addresses endpoints by IP host (tcp, sockets, verbs,
   * udp providers) rather than by fabric device (cxi, opx, psm2, shm).
   * @return true for IP-addressed protocols.
   */
  bool IsIpProtocol() const {
    for (const char *p : {"tcp", "sockets", "verbs", "udp"}) {
      if (protocol_.find(p) != std::string::npos) return true;
    }
    return false;
  }

  /**
   * Build the address the engine listens on.
   *
   * LISTEN, when set, is used verbatim after "://" (e.g. "cxi0" for a
   * Slingshot device, "hsn0" or an IP for an IP protocol). Otherwise an IP
   * protocol binds the interface that owns host_hint (the address the runtime
   * listens on, so Mercury uses the same NIC as the hostfile), and any other
   * protocol is left to the provider's default device.
   *
   * @param host_hint Host/IP of the first server transport ("" if none).
   * @return A Mercury listen address such as "ofi+cxi" or
   *         "ofi+tcp;ofi_rxm://10.0.0.1".
   */
  std::string ListenAddress(const std::string &host_hint) const {
    if (!listen_.empty()) return protocol_ + "://" + listen_;
    bool wildcard = host_hint.empty() || host_hint == "0.0.0.0" ||
                    host_hint == "*";
    if (IsIpProtocol() && !wildcard) return protocol_ + "://" + host_hint;
    return protocol_;
  }

  /**
   * Build the Margo JSON configuration for the thread layout.
   * progress_threads_ == 1: one dedicated progress ES plus rpc_threads_
   * handler ESs. progress_threads_ == 0: a single ES runs both the progress
   * loop and the handlers (rpc_threads_ is ignored).
   * @return JSON text for margo_init_ext.
   */
  std::string MargoJson() const {
    int rpc = progress_threads_ == 0 ? -1 : rpc_threads_;
    return std::string("{ \"use_progress_thread\" : true, ") +
           "\"rpc_thread_count\" : " + std::to_string(rpc) + " }";
  }
};

/**
 * Process-singleton Thallium engine plus the peer address exchange.
 *
 * Threads (defaults; see ThalliumConfig):
 *   - 1 Argobots ES running Mercury's progress loop (network progress and
 *     completion callbacks). Nothing in this transport blocks it.
 *   - 4 Argobots ESs running RPC handler ULTs. A handler admits the message
 *     to its listener's bounded queue, pulls the bulk payload and responds.
 *   - 1 plain thread ("address server") answering the TCP address exchange;
 *     it sleeps in poll() and never touches Mercury.
 * The thread that first calls Get() initialises Argobots and becomes its
 * primary ES; Finalize() should run on that same thread.
 *
 * Address exchange: Mercury addresses on native fabrics are not host:port
 * (ofi+cxi is NIC address + PID, known only once the endpoint is open), while
 * the runtime identifies a peer listener by host:port. Every listener
 * therefore also opens a plain TCP socket on its port (the port the ZeroMQ
 * listener would have bound) and answers each connection with the engine's
 * Mercury address; a client dials host:port over TCP once, then caches and
 * looks up the returned address. A restarted peer is found again because a
 * failed send drops the cached address and the next send re-asks.
 */
class ThalliumEngine {
 public:
  /**
   * Get the engine, creating it on first use.
   * @param host_hint Host/IP the caller listens on; used to pick the listen
   *        interface for IP protocols when the engine is created here.
   * @return The process-wide engine.
   * @throws std::exception if Margo cannot be initialised (e.g. the protocol
   *         or device is unavailable).
   */
  static tl::engine &Get(const std::string &host_hint = "") {
    State &s = S();
    std::lock_guard<std::mutex> lock(s.mtx_);
    if (!s.engine_) Create(s, host_hint);
    return *s.engine_;
  }

  /** @return true if the engine currently exists. */
  static bool IsUp() {
    State &s = S();
    std::lock_guard<std::mutex> lock(s.mtx_);
    return s.engine_ != nullptr;
  }

  /** @return This process's Mercury address ("" if the engine is down). */
  static std::string SelfAddress() {
    State &s = S();
    std::lock_guard<std::mutex> lock(s.mtx_);
    return s.self_addr_;
  }

  /**
   * Start answering the TCP address exchange on a listener's port.
   * @param port TCP port to listen on (all interfaces).
   * @throws std::runtime_error if the port cannot be bound.
   */
  static void Publish(int port) {
    State &s = S();
    sock::InitSocketLib();
    sock::socket_t fd = sock::Listen("", port, "tcp");
    if (fd == sock::kInvalidSocket) {
      throw std::runtime_error(
          "ThalliumEngine: cannot bind address-exchange port " +
          std::to_string(port) + ": " + sock::GetErrorString());
    }
    sock::SetNonBlocking(fd, true);
    std::lock_guard<std::mutex> lock(s.mtx_);
    if (s.wake_fds_[0] < 0 && ::pipe(s.wake_fds_) != 0) {
      sock::Close(fd);
      throw std::runtime_error("ThalliumEngine: cannot create wake pipe");
    }
    s.listeners_[port] = fd;
    if (!s.addr_thread_.joinable()) {
      sock::SetNonBlocking(s.wake_fds_[0], true);
      s.addr_stop_.store(false, std::memory_order_release);
      s.addr_thread_ = std::thread([&s]() { AddrServerLoop(s); });
    }
    WakeAddrServer(s);
  }

  /**
   * Stop answering the address exchange on a port and wait until its socket
   * is closed, so the port can be bound again at once. The socket is closed
   * by the address server thread itself, never while that thread may be
   * polling it (a closed descriptor number can be reused by another socket).
   * @param port Port previously passed to Publish.
   */
  static void Unpublish(int port) {
    State &s = S();
    std::unique_lock<std::mutex> lock(s.mtx_);
    auto it = s.listeners_.find(port);
    if (it == s.listeners_.end()) return;
    sock::socket_t fd = it->second;
    s.listeners_.erase(it);
    if (!s.addr_thread_.joinable()) {
      sock::Close(fd);
      return;
    }
    s.retired_.push_back(fd);
    WakeAddrServer(s);
    s.retired_cv_.wait(lock, [&s]() {
      return s.retired_.empty() ||
             s.addr_stop_.load(std::memory_order_acquire);
    });
  }

  /**
   * Resolve a peer listener's Mercury address through the address exchange.
   * @param host Peer host name or IP.
   * @param port Peer listener port.
   * @param refresh true to ignore (and replace) a cached answer.
   * @param why Receives the failure reason when "" is returned (may be
   *        null), e.g. "connect: Connection refused" or "no reply within
   *        5000 ms".
   * @return The peer's Mercury address, or "" if the peer did not answer.
   */
  static std::string Resolve(const std::string &host, int port, bool refresh,
                             std::string *why = nullptr) {
    State &s = S();
    std::string key = host + ":" + std::to_string(port);
    if (!refresh) {
      std::lock_guard<std::mutex> lock(s.cache_mtx_);
      auto it = s.addr_cache_.find(key);
      if (it != s.addr_cache_.end()) return it->second;
    }
    std::string reason;
    std::string addr = AskPeer(host, port, &reason);
    if (why != nullptr) *why = reason;
    std::lock_guard<std::mutex> lock(s.cache_mtx_);
    if (addr.empty()) {
      s.addr_cache_.erase(key);
    } else {
      s.addr_cache_[key] = addr;
    }
    return addr;
  }

  /**
   * Mutex that serializes RPC registration (engine.define) across threads.
   * Peer connections are dialed concurrently, and Margo's registration
   * table is not documented as safe for concurrent writers.
   * @return The process-wide registration mutex.
   */
  static std::mutex &DefineMutex() {
    static std::mutex m;
    return m;
  }

  /**
   * Stop the address server and finalize Margo (progress loop, handler ESs,
   * Mercury, Argobots). Every ThalliumTransport must be destroyed first.
   * Idempotent; a later Get() creates a fresh engine.
   */
  static void Finalize() { FinalizeState(S()); }

 private:
  /** Engine, address server and address cache state. */
  struct State {
    std::mutex mtx_;                      /**< guards everything but the cache */
    std::unique_ptr<tl::engine> engine_;
    std::string self_addr_;
    std::map<int, sock::socket_t> listeners_;  /**< port -> listening fd */
    std::vector<sock::socket_t> retired_;      /**< fds awaiting close */
    std::thread addr_thread_;
    std::condition_variable retired_cv_;       /**< retired_ was drained */
    int wake_fds_[2] = {-1, -1};               /**< pipe waking the server */
    std::atomic<bool> addr_stop_{false};
    std::mutex cache_mtx_;
    std::unordered_map<std::string, std::string> addr_cache_;

    /** Finalize at process exit if the owner never did. */
    ~State() { ThalliumEngine::FinalizeState(*this); }
  };

  /** @return The singleton state. */
  static State &S() {
    static State s;
    return s;
  }

  /**
   * Implementation of Finalize() on an explicit state object (also run by
   * the state's destructor at process exit).
   * @param s Singleton state.
   */
  static void FinalizeState(State &s) {
    StopAddrServer(s);
    std::unique_ptr<tl::engine> engine;
    {
      std::lock_guard<std::mutex> lock(s.mtx_);
      engine = std::move(s.engine_);
      s.self_addr_.clear();
    }
    {
      std::lock_guard<std::mutex> lock(s.cache_mtx_);
      s.addr_cache_.clear();
    }
    if (!engine) return;
    // No logging here: this also runs from a static destructor at process
    // exit, when the logger may already be gone.
    try {
      engine->finalize();
    } catch (const std::exception &) {
    }
    engine.reset();
  }

  /**
   * Create the engine. Caller holds s.mtx_.
   * @param s Singleton state.
   * @param host_hint See Get().
   */
  static void Create(State &s, const std::string &host_hint) {
    const ThalliumConfig &cfg = ThalliumConfig::Get();
    std::string listen = cfg.ListenAddress(host_hint);
    std::string json = cfg.MargoJson();
    struct hg_init_info hii = HG_INIT_INFO_INITIALIZER;
    if (!cfg.auth_key_.empty()) {
      hii.na_init_info.auth_key = cfg.auth_key_.c_str();
    }
    s.engine_ = std::make_unique<tl::engine>(listen, THALLIUM_SERVER_MODE,
                                             json.c_str(), &hii);
    s.self_addr_ = static_cast<std::string>(s.engine_->self());
    HLOG(kInfo,
         "ThalliumEngine: listen={} self={} progress_threads={} "
         "rpc_threads={} bulk_threshold={}",
         listen, s.self_addr_, cfg.progress_threads_,
         cfg.progress_threads_ == 0 ? 0 : cfg.rpc_threads_,
         cfg.bulk_threshold_);
  }

  /**
   * Send this process's Mercury address on an accepted connection, framed as
   * a 4-byte big-endian length followed by the address text, then close it.
   * @param fd Accepted TCP connection.
   * @param addr Address text to send.
   */
  static void AnswerPeer(sock::socket_t fd, const std::string &addr) {
    uint32_t len = sock::HostToNet32(static_cast<uint32_t>(addr.size()));
    sock::IoBuffer iov[2];
    iov[0].base = &len;
    iov[0].len = sizeof(len);
    iov[1].base = const_cast<char *>(addr.data());
    iov[1].len = addr.size();
    sock::SendV(fd, iov, 2);
    sock::Close(fd);
  }

  /**
   * Interrupt the address server's poll so it re-reads the listener set.
   * Caller holds s.mtx_.
   * @param s Singleton state.
   */
  static void WakeAddrServer(State &s) {
    if (s.wake_fds_[1] < 0) return;
    char byte = 1;
    ssize_t n = ::write(s.wake_fds_[1], &byte, 1);
    (void)n;
  }

  /**
   * Address server thread body: block in poll() on the wake pipe and every
   * published port, answer each connection, close retired sockets.
   * @param s Singleton state.
   */
  static void AddrServerLoop(State &s) {
    static constexpr int kPollMs = 1000;
    while (!s.addr_stop_.load(std::memory_order_acquire)) {
      std::vector<sock::socket_t> fds;
      std::string addr;
      {
        std::lock_guard<std::mutex> lock(s.mtx_);
        for (sock::socket_t fd : s.retired_) sock::Close(fd);
        s.retired_.clear();
        s.retired_cv_.notify_all();
        fds.push_back(s.wake_fds_[0]);
        for (auto &kv : s.listeners_) fds.push_back(kv.second);
        addr = s.self_addr_;
      }
      int idx = sock::PollReadMulti(fds.data(), static_cast<int>(fds.size()),
                                    kPollMs);
      if (idx < 0) continue;
      if (idx == 0) {
        char drain[64];
        while (::read(fds[0], drain, sizeof(drain)) > 0) {
        }
        continue;
      }
      AnswerPending(fds[idx], addr);
    }
  }

  /**
   * Answer every connection already queued on a (non-blocking) listener, so
   * a burst of peers dialing at once is served in one wakeup rather than one
   * per poll round.
   * @param listen_fd Listening socket that poll() reported readable.
   * @param addr Address text to send to each peer.
   */
  static void AnswerPending(sock::socket_t listen_fd, const std::string &addr) {
    static constexpr int kMaxPerWakeup = 1024;
    for (int i = 0; i < kMaxPerWakeup; ++i) {
      sock::socket_t fd = sock::Accept(listen_fd);
      if (fd == sock::kInvalidSocket) return;
      // Accepted sockets do not inherit O_NONBLOCK on Linux, but do on
      // BSD/macOS; the reply is a few bytes, so a blocking send is safe.
      sock::SetNonBlocking(fd, false);
      AnswerPeer(fd, addr);
    }
  }

  /**
   * Stop and join the address server thread and close its sockets.
   * @param s Singleton state.
   */
  static void StopAddrServer(State &s) {
    std::thread t;
    {
      std::lock_guard<std::mutex> lock(s.mtx_);
      s.addr_stop_.store(true, std::memory_order_release);
      WakeAddrServer(s);
      t = std::move(s.addr_thread_);
    }
    if (t.joinable()) t.join();
    std::lock_guard<std::mutex> lock(s.mtx_);
    for (auto &kv : s.listeners_) sock::Close(kv.second);
    for (sock::socket_t fd : s.retired_) sock::Close(fd);
    s.listeners_.clear();
    s.retired_.clear();
    s.retired_cv_.notify_all();
    for (int &fd : s.wake_fds_) {
      if (fd >= 0) ::close(fd);
      fd = -1;
    }
  }

  /**
   * Ask a peer listener for its Mercury address over TCP.
   * @param host Peer host name or IP.
   * @param port Peer listener port.
   * @param why Receives the failure reason when "" is returned.
   * @return The address, or "" on lookup/connect failure, timeout or a bad
   *         reply.
   */
  static std::string AskPeer(const std::string &host, int port,
                             std::string *why) {
    sock::InitSocketLib();
    int wait_ms = ThalliumConfig::Get().resolve_timeout_ms_;
    sock::ConnectError cerr;
    sock::socket_t fd = sock::ConnectTimed(host, port, wait_ms, &cerr);
    if (fd == sock::kInvalidSocket) {
      *why = cerr.ToString();
      return "";
    }
    std::string addr = ReadAnswer(fd, wait_ms, why);
    sock::Close(fd);
    return addr;
  }

  /**
   * Read the framed address reply from an address-exchange connection.
   * @param fd Connected socket.
   * @param wait_ms Longest wait for each part of the reply.
   * @param why Receives the failure reason when "" is returned.
   * @return The address, or "" on timeout, a closed connection or a bad
   *         length.
   */
  static std::string ReadAnswer(sock::socket_t fd, int wait_ms,
                                std::string *why) {
    static constexpr uint32_t kMaxAddrLen = 4096;
    uint32_t net_len = 0;
    if (sock::PollRead(fd, wait_ms) <= 0) {
      *why = "no reply within " + std::to_string(wait_ms) + " ms";
      return "";
    }
    if (sock::RecvExact(fd, reinterpret_cast<char *>(&net_len),
                        sizeof(net_len)) != 0) {
      *why = "connection closed before the reply";
      return "";
    }
    uint32_t len = sock::NetToHost32(net_len);
    if (len == 0 || len > kMaxAddrLen) {
      *why = "bad reply length " + std::to_string(len);
      return "";
    }
    std::string addr(len, '\0');
    if (sock::PollRead(fd, wait_ms) <= 0 ||
        sock::RecvExact(fd, addr.data(), len) != 0) {
      *why = "truncated reply";
      return "";
    }
    return addr;
  }
};

}  // namespace ctp::lbm
#endif  // CTP_ENABLE_THALLIUM
