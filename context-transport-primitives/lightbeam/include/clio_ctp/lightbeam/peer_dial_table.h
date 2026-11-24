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

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

namespace ctp::lbm {

/** Timing policy of a PeerDialTable. */
struct PeerDialParams {
  /** Backoff after the first failed dial to a peer; doubles per failure. */
  std::chrono::milliseconds backoff_base_{50};
  /** Largest backoff between dials to a failing peer. */
  std::chrono::milliseconds backoff_cap_{2000};
  /** How long a caller waits for another thread's dial of the same key
   *  before giving up with nullptr (the caller retries later). */
  std::chrono::milliseconds inflight_wait_{100};
  /** A failing peer's error is reported at most once per this interval. */
  std::chrono::milliseconds log_interval_{5000};
};

/** What a PeerDialTable::GetOrDial call did. */
enum class PeerDialStatus {
  kCached,    /**< an existing connection was returned */
  kDialed,    /**< this call dialed and the dial succeeded */
  kFailed,    /**< this call dialed and the dial failed */
  kBackoff,   /**< the peer is in its backoff window; nothing was dialed */
  kInFlight,  /**< another thread is dialing this key and did not finish */
};

/** Detail of a GetOrDial call, for the caller's (rate-limited) logging. */
struct PeerDialReport {
  PeerDialStatus status_ = PeerDialStatus::kCached; /**< outcome */
  /** kFailed: true if this failure should be logged (first of a streak or
   *  first after log_interval_). kDialed: true if it ends a failure streak. */
  bool log_ = false;
  uint32_t streak_ = 0;      /**< consecutive failed dials to this peer */
  uint64_t suppressed_ = 0;  /**< failures not logged since the last log */
  std::chrono::milliseconds backoff_{0}; /**< kFailed: wait before redial */
  std::string error_;        /**< kFailed: the dialer's reason */
};

/**
 * Connection table with per-key single-flight dialing and per-peer
 * exponential backoff.
 *
 * - The table lock is never held while dialing, so dials to different peers
 *   run in parallel and a slow or unresponsive peer cannot stall the rest.
 * - At most one dial per key is in flight; a concurrent caller for the same
 *   key waits up to inflight_wait_ for it, then gets nullptr.
 * - A failed dial puts the peer (several keys may share one peer, e.g. one
 *   per lane or send thread) into a backoff window, base doubling to cap;
 *   inside the window every key of that peer returns nullptr without dialing.
 *   A successful dial resets the peer.
 *
 * Returned pointers stay owned by the table; they remain valid until the key
 * is erased or the table is cleared (same contract as a plain map).
 *
 * @tparam T Connection type.
 * @tparam Deleter Deleter of the owning unique_ptr.
 */
template <typename T, typename Deleter = std::default_delete<T>>
class PeerDialTable {
 public:
  using Ptr = std::unique_ptr<T, Deleter>;
  using Clock = std::chrono::steady_clock;
  /** Dial function: return the connection, or null and set *error. May
   *  throw std::exception (treated as a failure with e.what()). */
  using Dialer = std::function<Ptr(std::string *error)>;

  /** @param params Timing policy. */
  explicit PeerDialTable(PeerDialParams params = PeerDialParams())
      : params_(params), now_([]() { return Clock::now(); }) {}

  /**
   * Replace the clock (tests drive backoff with a fake clock).
   * @param now Function returning the current time.
   */
  void SetClock(std::function<Clock::time_point()> now) {
    std::lock_guard<std::mutex> lock(mtx_);
    now_ = std::move(now);
  }

  /**
   * Return the connection for @p key, dialing it if needed and allowed.
   * @param key Connection key (peer plus lane / thread suffix).
   * @param peer Backoff key (the remote listener, e.g. "host:port").
   * @param dial Dial function, called without the table lock held.
   * @param report Receives what happened (may be null).
   * @return The connection, or nullptr (failed, backing off, or another
   *         thread's dial still in flight); the caller retries later.
   */
  T *GetOrDial(const std::string &key, const std::string &peer,
               const Dialer &dial, PeerDialReport *report = nullptr) {
    PeerDialReport local;
    if (report == nullptr) report = &local;
    *report = PeerDialReport();
    std::unique_lock<std::mutex> lock(mtx_);
    T *found = nullptr;
    if (!AdmitDial(lock, key, peer, report, &found)) return found;
    lock.unlock();
    std::string error;
    Ptr conn = RunDialer(dial, &error);
    lock.lock();
    T *raw = FinishDial(key, peer, std::move(conn), error, report);
    lock.unlock();
    cv_.notify_all();
    return raw;
  }

  /**
   * Drop the connection for @p key (a dial in flight for it still lands).
   * @param key Connection key.
   */
  void Erase(const std::string &key) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = entries_.find(key);
    if (it == entries_.end()) return;
    if (it->second.dialing_) {
      it->second.conn_.reset();
    } else {
      entries_.erase(it);
    }
  }

  /** Drop every connection and all backoff state. */
  void Clear() {
    std::lock_guard<std::mutex> lock(mtx_);
    for (auto it = entries_.begin(); it != entries_.end();) {
      if (it->second.dialing_) {
        it->second.conn_.reset();
        ++it;
      } else {
        it = entries_.erase(it);
      }
    }
    peers_.clear();
  }

  /** @return Number of established connections. */
  size_t Size() const {
    std::lock_guard<std::mutex> lock(mtx_);
    size_t n = 0;
    for (const auto &kv : entries_) n += kv.second.conn_ ? 1 : 0;
    return n;
  }

  /** @return Total dialer invocations since construction. */
  uint64_t DialAttempts() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return attempts_;
  }

 private:
  /** One connection key. */
  struct Entry {
    Ptr conn_;              /**< established connection, or null */
    bool dialing_ = false;  /**< a thread is dialing this key */
  };
  /** Backoff state of one peer. */
  struct Peer {
    uint32_t streak_ = 0;          /**< consecutive failed dials */
    Clock::time_point retry_at_{}; /**< no dial before this */
    Clock::time_point last_log_{}; /**< last reported failure */
    uint64_t suppressed_ = 0;      /**< failures since last_log_ */
  };

  /**
   * Decide under the lock whether this caller dials. Caller holds @p lock.
   * @param lock Held table lock (released while waiting).
   * @param key Connection key.
   * @param peer Backoff key.
   * @param report Receives the outcome when not dialing.
   * @param found Receives the connection when one exists.
   * @return true if the caller must dial now (key marked in flight).
   */
  bool AdmitDial(std::unique_lock<std::mutex> &lock, const std::string &key,
                 const std::string &peer, PeerDialReport *report, T **found) {
    Entry &e = entries_[key];
    if (e.conn_) {
      *found = e.conn_.get();
      report->status_ = PeerDialStatus::kCached;
      return false;
    }
    if (e.dialing_) {
      cv_.wait_for(lock, params_.inflight_wait_, [&]() {
        auto it = entries_.find(key);
        return it == entries_.end() || !it->second.dialing_;
      });
      auto it = entries_.find(key);
      if (it != entries_.end() && it->second.conn_) {
        *found = it->second.conn_.get();
        report->status_ = PeerDialStatus::kCached;
      } else {
        report->status_ = PeerDialStatus::kInFlight;
      }
      return false;
    }
    auto pit = peers_.find(peer);
    if (pit != peers_.end() && now_() < pit->second.retry_at_) {
      report->status_ = PeerDialStatus::kBackoff;
      report->streak_ = pit->second.streak_;
      return false;
    }
    e.dialing_ = true;
    ++attempts_;
    return true;
  }

  /**
   * Run the dial function, converting an exception into a failure.
   * @param dial Dial function.
   * @param error Receives the reason on failure.
   * @return The connection, or null.
   */
  static Ptr RunDialer(const Dialer &dial, std::string *error) {
    try {
      Ptr conn = dial(error);
      if (!conn && error->empty()) *error = "dial returned no connection";
      return conn;
    } catch (const std::exception &ex) {
      *error = ex.what();
    } catch (...) {
      *error = "unknown exception";
    }
    return Ptr();
  }

  /**
   * Record a dial's outcome under the lock.
   * @param key Connection key.
   * @param peer Backoff key.
   * @param conn The new connection, or null on failure.
   * @param error Failure reason.
   * @param report Receives the outcome.
   * @return The stored connection, or nullptr.
   */
  T *FinishDial(const std::string &key, const std::string &peer, Ptr conn,
                const std::string &error, PeerDialReport *report) {
    Entry &e = entries_[key];
    e.dialing_ = false;
    Peer &p = peers_[peer];
    if (conn) {
      report->status_ = PeerDialStatus::kDialed;
      report->log_ = p.streak_ > 0;
      report->streak_ = p.streak_;
      report->suppressed_ = p.suppressed_;
      p = Peer();
      e.conn_ = std::move(conn);
      return e.conn_.get();
    }
    Clock::time_point now = now_();
    p.streak_++;
    uint32_t shift = std::min<uint32_t>(p.streak_ - 1, 30);
    auto backoff = params_.backoff_base_ * (int64_t{1} << shift);
    if (backoff > params_.backoff_cap_ || backoff.count() <= 0) {
      backoff = params_.backoff_cap_;
    }
    p.retry_at_ = now + backoff;
    report->status_ = PeerDialStatus::kFailed;
    report->streak_ = p.streak_;
    report->backoff_ = backoff;
    report->error_ = error;
    if (p.streak_ == 1 || now - p.last_log_ >= params_.log_interval_) {
      report->log_ = true;
      report->suppressed_ = p.suppressed_;
      p.suppressed_ = 0;
      p.last_log_ = now;
    } else {
      p.suppressed_++;
    }
    return nullptr;
  }

  PeerDialParams params_;                        /**< timing policy */
  std::function<Clock::time_point()> now_;       /**< clock */
  mutable std::mutex mtx_;                       /**< guards all below */
  std::condition_variable cv_;                   /**< a dial finished */
  std::unordered_map<std::string, Entry> entries_; /**< key -> connection */
  std::unordered_map<std::string, Peer> peers_;  /**< peer -> backoff */
  uint64_t attempts_ = 0;                        /**< dialer invocations */
};

/**
 * Per-key log rate limiter: allows the first event of a key, then at most
 * one per interval, counting what it suppressed.
 */
class LogRateLimiter {
 public:
  /** @param interval Minimum time between allowed events of one key. */
  explicit LogRateLimiter(
      std::chrono::milliseconds interval = std::chrono::milliseconds(5000))
      : interval_(interval) {}

  /**
   * Decide whether to log an event.
   * @param key Event key (e.g. the peer).
   * @param suppressed Receives how many events of this key were suppressed
   *        since the last allowed one (when allowed).
   * @return true if the caller should log.
   */
  bool Allow(const std::string &key, uint64_t *suppressed) {
    auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = state_.find(key);
    if (it == state_.end()) {
      state_.emplace(key, State{now, 0});
      *suppressed = 0;
      return true;
    }
    if (now - it->second.last_ >= interval_) {
      *suppressed = it->second.suppressed_;
      it->second = State{now, 0};
      return true;
    }
    it->second.suppressed_++;
    return false;
  }

 private:
  /** Per-key state. */
  struct State {
    std::chrono::steady_clock::time_point last_; /**< last allowed event */
    uint64_t suppressed_;                        /**< events since last_ */
  };
  std::chrono::milliseconds interval_;            /**< min gap per key */
  std::mutex mtx_;                                /**< guards state_ */
  std::unordered_map<std::string, State> state_;  /**< key -> state */
};

}  // namespace ctp::lbm
