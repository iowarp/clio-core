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

#include "clio_ctp/lightbeam/socket_engine.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>

namespace ctp::lbm::sock {

namespace {

/** Kernel socket buffer size for engine connections (as ZeroMQ uses). */
constexpr int kSockBufBytes = 4 * 1024 * 1024;
/** Poll timeout: bounds how long a newly added socket or newly queued
 *  output waits before its I/O thread re-polls. */
constexpr int kPollMs = 1;
/** Frames one connection may deliver per wake before others get a turn. */
constexpr int kMaxFramesPerWake = 64;

long EnvLong(const char* name, long dflt) {
  const char* e = std::getenv(name);
  if (e == nullptr || *e == '\0') return dflt;
  return std::strtol(e, nullptr, 10);
}

template <typename T>
void EraseOne(std::vector<std::shared_ptr<T>>* v, const T* p) {
  for (auto it = v->begin(); it != v->end(); ++it) {
    if (it->get() == p) {
      v->erase(it);
      return;
    }
  }
}

}  // namespace

// ---------------------------------------------------------------- EngineConn

EngineConn::~EngineConn() { Close(fd); }

EngineListener::~EngineListener() { Close(fd); }

// --------------------------------------------------------------------- Inbox

int Inbox::Register(const std::shared_ptr<EngineConn>& c) {
  std::lock_guard<std::mutex> lk(conns_mu);
  c->id = next_id++;
  if (next_id <= 0) next_id = 1;  // stay positive on wrap
  conns[c->id] = c;
  return c->id;
}

void Inbox::Unregister(int id) {
  std::lock_guard<std::mutex> lk(conns_mu);
  conns.erase(id);
}

std::shared_ptr<EngineConn> Inbox::Find(int id) {
  std::lock_guard<std::mutex> lk(conns_mu);
  auto it = conns.find(id);
  return it == conns.end() ? nullptr : it->second;
}

std::vector<std::shared_ptr<EngineConn>> Inbox::TakeAll() {
  std::vector<std::shared_ptr<EngineConn>> out;
  std::lock_guard<std::mutex> lk(conns_mu);
  for (auto& kv : conns) out.push_back(kv.second);
  conns.clear();
  return out;
}

void Inbox::Push(int id, Frame&& f) {
  {
    std::lock_guard<std::mutex> lk(mu);
    frames.emplace_back(id, std::move(f));
  }
  cv.notify_one();
}

bool Inbox::Pop(int* id, Frame* f) {
  std::lock_guard<std::mutex> lk(mu);
  if (frames.empty()) return false;
  *id = frames.front().first;
  *f = std::move(frames.front().second);
  frames.pop_front();
  return true;
}

bool Inbox::Wait(int timeout_ms) {
  std::unique_lock<std::mutex> lk(mu);
  if (!frames.empty()) return true;
  if (timeout_ms == 0) return false;
  if (timeout_ms < 0) {
    cv.wait(lk, [this] { return !frames.empty(); });
    return true;
  }
  return cv.wait_for(lk, std::chrono::milliseconds(timeout_ms),
                     [this] { return !frames.empty(); });
}

// ------------------------------------------------------------------ IoEngine

IoEngine& IoEngine::Instance() {
  // Leaked on purpose: transports may outlive static destruction order, and
  // the I/O threads must never see a destroyed engine.
  static IoEngine* engine = new IoEngine();
  return *engine;
}

bool IoEngine::Enabled() {
  static const bool on = EnvLong("CLIO_SOCK_ENGINE", 1) != 0;
  return on;
}

IoEngine::IoEngine() {
  long n = EnvLong("CLIO_SOCK_IO_THREADS", 4);
  if (n < 1) n = 1;
  direct_send_ = EnvLong("CLIO_SOCK_DIRECT_SEND", 1) != 0;
  max_queued_ = static_cast<size_t>(
                    std::max(1L, EnvLong("CLIO_SOCK_MAX_QUEUED_MB", 512)))
                << 20;
  for (long i = 0; i < n; ++i) {
    workers_.emplace_back(new Worker());
  }
  for (auto& w : workers_) {
    Worker* wp = w.get();
    wp->th = std::thread([this, wp] { Loop(wp); });
  }
}

IoEngine::~IoEngine() {
  stop_.store(true);
  for (auto& w : workers_) {
    if (w->th.joinable()) w->th.join();
  }
}

void IoEngine::AddConn(const std::shared_ptr<EngineConn>& c) {
  if (c->tcp) SetTcpNoDelay(c->fd);
  SetSendBuf(c->fd, kSockBufBytes);
  SetRecvBuf(c->fd, kSockBufBytes);
  SetNonBlocking(c->fd, true);
  c->inbox->Register(c);
  int w = static_cast<int>(rr_.fetch_add(1) % workers_.size());
  c->worker = w;
  std::lock_guard<std::mutex> lk(workers_[w]->mu);
  workers_[w]->conns.push_back(c);
  workers_[w]->gen.fetch_add(1);
}

void IoEngine::RemoveConn(const std::shared_ptr<EngineConn>& c) {
  if (!c) return;
  {
    std::lock_guard<std::mutex> lk(c->mu);
    if (!c->broken) Fail(c.get());
  }
  if (c->worker < 0) return;
  Worker* w = workers_[c->worker].get();
  std::lock_guard<std::mutex> lk(w->mu);
  EraseOne(&w->conns, c.get());
  w->gen.fetch_add(1);
}

std::shared_ptr<EngineListener> IoEngine::AddListener(
    socket_t fd, const std::shared_ptr<Inbox>& inbox, bool tcp) {
  auto l = std::make_shared<EngineListener>();
  l->fd = fd;
  l->tcp = tcp;
  l->inbox = inbox;
  SetNonBlocking(fd, true);
  int w = static_cast<int>(rr_.fetch_add(1) % workers_.size());
  l->worker = w;
  std::lock_guard<std::mutex> lk(workers_[w]->mu);
  workers_[w]->listeners.push_back(l);
  workers_[w]->gen.fetch_add(1);
  return l;
}

void IoEngine::RemoveListener(const std::shared_ptr<EngineListener>& l) {
  if (!l || l->worker < 0) return;
  Worker* w = workers_[l->worker].get();
  std::lock_guard<std::mutex> lk(w->mu);
  EraseOne(&w->listeners, l.get());
  w->gen.fetch_add(1);
}

void IoEngine::Fail(EngineConn* c) {
  c->broken = true;
  c->cur.clear();
  c->cur_off = 0;
  c->small_q.clear();
  c->big_q.clear();
  c->queued = 0;
  c->want_out.store(false);
  Shutdown(c->fd);  // the peer drops any partial frame
  c->cv.notify_all();
}

int IoEngine::Send(const std::shared_ptr<EngineConn>& c, const IoBuffer* iov,
                   size_t count) {
  size_t total = 0;
  for (size_t i = 0; i < count; ++i) total += iov[i].len;

  std::unique_lock<std::mutex> lk(c->mu);
  // Back-pressure: wait while this connection already has a lot queued.
  c->cv.wait(lk, [&] { return c->broken || c->queued < max_queued_; });
  if (c->broken) return EPIPE;

  size_t done = 0;
  size_t idx = 0;  // first iov entry not fully written
  size_t off = 0;  // bytes of iov[idx] already written
  if (!c->HasOutput() && direct_send_) {
    while (idx < count) {
      IoBuffer win[kSendSomeMaxIov];
      int k = 0;
      size_t o = off;
      for (size_t i = idx; i < count && k < kSendSomeMaxIov; ++i) {
        win[k++] = {static_cast<char*>(iov[i].base) + o, iov[i].len - o};
        o = 0;
      }
      ssize_t n = SendSomeV(c->fd, win, k);
      if (n < 0) {
        Fail(c.get());
        return EPIPE;
      }
      if (n == 0) break;  // socket full: queue the rest
      done += static_cast<size_t>(n);
      size_t left = static_cast<size_t>(n);
      while (idx < count && left >= iov[idx].len - off) {
        left -= iov[idx].len - off;
        off = 0;
        ++idx;
      }
      if (idx < count) off += left;
    }
  }
  if (done < total) {
    std::vector<char> rest;
    rest.reserve(total - done);
    for (size_t i = idx; i < count; ++i) {
      const char* p = static_cast<const char*>(iov[i].base);
      size_t o = (i == idx) ? off : 0;
      rest.insert(rest.end(), p + o, p + iov[i].len);
    }
    c->queued += rest.size();
    if (done > 0) {
      c->cur = std::move(rest);  // partly on the wire: must finish first
      c->cur_off = 0;
    } else if (total < kSmallFrameBytes) {
      c->small_q.push_back(std::move(rest));
    } else {
      c->big_q.push_back(std::move(rest));
    }
    c->want_out.store(true);
  }
  return 0;
}

void IoEngine::Flush(const std::shared_ptr<EngineConn>& c) {
  std::lock_guard<std::mutex> lk(c->mu);
  while (c->HasOutput()) {
    // Wire order: the partly written frame, then small frames, then bulk.
    IoBuffer win[kSendSomeMaxIov];
    int k = 0;
    if (c->cur_off < c->cur.size()) {
      win[k++] = {c->cur.data() + c->cur_off, c->cur.size() - c->cur_off};
    }
    for (auto& f : c->small_q) {
      if (k == kSendSomeMaxIov) break;
      win[k++] = {f.data(), f.size()};
    }
    for (auto& f : c->big_q) {
      if (k == kSendSomeMaxIov) break;
      win[k++] = {f.data(), f.size()};
    }
    ssize_t n = SendSomeV(c->fd, win, k);
    if (n < 0) {
      Fail(c.get());
      return;
    }
    if (n == 0) break;
    size_t left = static_cast<size_t>(n);
    c->queued -= left;
    if (c->cur_off < c->cur.size()) {
      size_t r = c->cur.size() - c->cur_off;
      if (left < r) {
        c->cur_off += left;
        continue;
      }
      left -= r;
      c->cur.clear();
      c->cur_off = 0;
    }
    // Consume in the order the window was built; a frame cut short becomes
    // cur so that nothing is sent ahead of its remainder.
    for (auto* q : {&c->small_q, &c->big_q}) {
      while (left > 0 && !q->empty()) {
        size_t r = q->front().size();
        if (left >= r) {
          left -= r;
          q->pop_front();
        } else {
          c->cur = std::move(q->front());
          q->pop_front();
          c->cur_off = left;
          left = 0;
        }
      }
    }
  }
  c->want_out.store(c->HasOutput());
  if (c->queued < max_queued_) c->cv.notify_all();
}

void IoEngine::ReadConn(const std::shared_ptr<EngineConn>& c) {
  for (int i = 0; i < kMaxFramesPerWake; ++i) {
    Frame f;
    int r = c->reader.Read(c->fd, &f);
    if (r == 1) {
      c->inbox->Push(c->id, std::move(f));
      continue;
    }
    if (r < 0) Drop(c);
    return;
  }
}

void IoEngine::Drop(const std::shared_ptr<EngineConn>& c) {
  {
    std::lock_guard<std::mutex> lk(c->mu);
    if (!c->broken) Fail(c.get());
  }
  c->inbox->Unregister(c->id);
  Worker* w = workers_[c->worker].get();
  std::lock_guard<std::mutex> lk(w->mu);
  EraseOne(&w->conns, c.get());
  w->gen.fetch_add(1);
}

void IoEngine::Accept(const std::shared_ptr<EngineListener>& l) {
  while (true) {
    socket_t fd = sock::Accept(l->fd);
    if (fd == kInvalidSocket) return;
    auto c = std::make_shared<EngineConn>();
    c->fd = fd;
    c->tcp = l->tcp;
    c->inbox = l->inbox;
    AddConn(c);
  }
}

void IoEngine::Loop(Worker* w) {
  std::vector<std::shared_ptr<EngineConn>> conns;
  std::vector<std::shared_ptr<EngineListener>> ls;
  std::vector<PollFd> pfds;
  uint64_t seen = ~0ull;
  while (!stop_.load(std::memory_order_relaxed)) {
    uint64_t g = w->gen.load();
    if (g != seen) {
      std::lock_guard<std::mutex> lk(w->mu);
      conns = w->conns;
      ls = w->listeners;
      seen = w->gen.load();
    }
    if (conns.empty() && ls.empty()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kPollMs));
      continue;
    }
    pfds.clear();
    for (auto& l : ls) pfds.push_back({l->fd, kPollIn, 0});
    for (auto& c : conns) {
      short ev = kPollIn;
      if (c->want_out.load(std::memory_order_relaxed)) ev |= kPollOut;
      pfds.push_back({c->fd, ev, 0});
    }
    int n = PollFds(pfds.data(), static_cast<int>(pfds.size()), kPollMs);
    if (n <= 0) continue;
    size_t i = 0;
    for (auto& l : ls) {
      if (pfds[i++].revents != 0) Accept(l);
    }
    for (auto& c : conns) {
      short r = pfds[i++].revents;
      if (r == 0) continue;
      if (r & kPollOut) Flush(c);
      if (r & (kPollIn | kPollErr)) ReadConn(c);
    }
  }
}

}  // namespace ctp::lbm::sock
