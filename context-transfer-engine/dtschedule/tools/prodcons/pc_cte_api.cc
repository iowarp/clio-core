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
 * @file pc_cte_api.cc
 * CTE-API I/O backend of the producer-consumer workload (see pc_cte_api.h).
 */

#include "pc_cte_api.h"

#include <clio_cte/core/core_client.h>
#include <clio_runtime/clio_runtime.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <deque>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace core = clio::cte::core;

/** Chunk size: one CTE blob per MiB, named by its index like clio-fs pages. */
constexpr size_t kChunk = 1u << 20;
/** Marker blob put into a file's tag once all its chunks are stored. */
constexpr const char *kMarker = "done";
/** Marker put instead when any chunk put failed (the consumer counts it bad
 *  rather than waiting for a file that will never complete). */
constexpr const char *kFailedMarker = "failed";

/** A file whose chunk puts are in flight. */
struct PendingFile {
  std::string name;                                   ///< Tag name
  core::TagId tag;                                    ///< Its tag
  ctp::ipc::FullPtr<char> buf;                        ///< Shared-memory copy
  std::vector<clio::run::Future<core::PutBlobTask>> puts;  ///< Chunk puts
};

std::deque<PendingFile> g_pending;  ///< Files in flight, oldest first
int g_failed = 0;                   ///< Failed chunk/marker puts

/**
 * Tag of a file by name (created on first use; same id on every node).
 * @param name Tag name
 * @param tag Output tag id
 * @return true on success
 */
bool TagOf(const std::string &name, core::TagId *tag) {
  auto *cte = CLIO_CTE_CLIENT;
  auto t = cte->AsyncGetOrCreateTag(name);
  t.Wait();
  if (t->GetReturnCode() != 0) return false;
  *tag = t->tag_id_;
  return true;
}

/**
 * Whether every chunk put of a file has completed.
 * @param f File in flight
 * @return true when all are complete
 */
bool AllDone(const PendingFile &f) {
  for (const auto &p : f.puts) {
    if (!p.IsComplete()) return false;
  }
  return true;
}

/**
 * Finish a file: collect its chunk results, put its marker, free its copy.
 * @param f File in flight (its puts are complete or are waited for here)
 */
void Finish(PendingFile &f) {
  auto *cte = CLIO_CTE_CLIENT;
  int bad = 0;
  for (auto &p : f.puts) {
    p.Wait();
    if (p->GetReturnCode() != 0) ++bad;
  }
  const uint64_t one = 1;
  auto m = cte->AsyncPutBlob(f.tag, std::string(bad == 0 ? kMarker : kFailedMarker),
                             0, sizeof(one),
                             reinterpret_cast<const char *>(&one));
  m.Wait();
  if (m->GetReturnCode() != 0) ++bad;
  if (bad != 0) {
    fprintf(stderr, "pc_cte_api: %s: %d failed puts\n", f.name.c_str(), bad);
  }
  g_failed += bad;
  CLIO_IPC->FreeBuffer(f.buf);
}

}  // namespace

extern "C" int PcApiInit(void) {
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false)) {
    return -1;
  }
  return core::CLIO_CTE_CLIENT_INIT() ? 0 : -1;
}

extern "C" int PcApiPutFile(const char *name, const void *data, size_t bytes,
                            int max_pending) {
  while (static_cast<int>(g_pending.size()) >= std::max(max_pending, 1)) {
    Finish(g_pending.front());
    g_pending.pop_front();
  }
  PendingFile f;
  f.name = name;
  if (!TagOf(f.name, &f.tag)) return EIO;
  f.buf = CLIO_IPC->AllocateBuffer(bytes);
  if (f.buf.IsNull()) return ENOMEM;
  std::memcpy(f.buf.ptr_, data, bytes);
  auto *cte = CLIO_CTE_CLIENT;
  ctp::ipc::ShmPtr<> base = f.buf.shm_.template Cast<void>();
  for (size_t off = 0, i = 0; off < bytes; off += kChunk, ++i) {
    const size_t n = std::min(kChunk, bytes - off);
    f.puts.push_back(cte->AsyncPutBlob(f.tag, std::to_string(i), 0, n,
                                       base + off));
  }
  g_pending.push_back(std::move(f));
  return 0;
}

extern "C" int PcApiDrain(int block) {
  while (!g_pending.empty() && (block || AllDone(g_pending.front()))) {
    Finish(g_pending.front());
    g_pending.pop_front();
  }
  return g_failed;
}

extern "C" int PcApiWaitFile(const char *name, double timeout_s) {
  core::TagId tag;
  if (!TagOf(name, &tag)) return EIO;
  auto *cte = CLIO_CTE_CLIENT;
  const auto end = std::chrono::steady_clock::now() +
                   std::chrono::duration<double>(timeout_s);
  while (std::chrono::steady_clock::now() < end) {
    auto s = cte->AsyncGetBlobSize(tag, std::string(kMarker));
    s.Wait();
    if (s->GetReturnCode() == 0 && s->size_ > 0) return 0;
    auto f = cte->AsyncGetBlobSize(tag, std::string(kFailedMarker));
    f.Wait();
    if (f->GetReturnCode() == 0 && f->size_ > 0) return EIO;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  return ETIMEDOUT;
}

extern "C" int PcApiGetFile(const char *name, void *out, size_t bytes,
                            size_t *got) {
  *got = 0;
  core::TagId tag;
  if (!TagOf(name, &tag)) return EIO;
  auto buf = CLIO_IPC->AllocateBuffer(bytes);
  if (buf.IsNull()) return ENOMEM;
  auto *cte = CLIO_CTE_CLIENT;
  ctp::ipc::ShmPtr<> base = buf.shm_.template Cast<void>();
  std::vector<clio::run::Future<core::GetBlobTask>> gets;
  for (size_t off = 0, i = 0; off < bytes; off += kChunk, ++i) {
    const size_t n = std::min(kChunk, bytes - off);
    gets.push_back(cte->AsyncGetBlob(tag, std::to_string(i), 0, n, 0,
                                     base + off));
  }
  int rc = 0;
  for (auto &g : gets) {
    g.Wait();
    if (g->GetReturnCode() != 0) rc = EIO;
  }
  if (rc == 0) {
    std::memcpy(out, buf.ptr_, bytes);
    *got = bytes;
  }
  CLIO_IPC->FreeBuffer(buf);
  return rc;
}
