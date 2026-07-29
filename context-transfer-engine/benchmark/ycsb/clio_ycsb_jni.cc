/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * JNI shim backing the YCSB Clio binding (site.ycsb.db.ClioClient, issue
 * #862). Maps the YCSB key/value model onto the CTE blob store: one tag
 * ("ycsb") holds every record, the YCSB key is the blob name, and the
 * serialized field map (built Java-side) is the blob payload.
 *
 * The runtime daemon must already be running (clio_run runtime start); this
 * shim connects as a SHM client. Init is process-wide and idempotent — YCSB
 * constructs one DB adapter per client thread, and every adapter's init()
 * funnels through the same std::once.
 */

#include <jni.h>
#include <signal.h>

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>

namespace {

clio::cte::core::Tag *g_tag = nullptr;
std::once_flag g_init_once;
int g_init_rc = -100;  // set by the once-body; <0 means init failed

// ---- async write window (issue #862 follow-up) ----------------------------
// Writes were fully synchronous: one blocking PutBlob round trip per update,
// which caps update-heavy workloads (A/F) at the depth-1 rate. This window
// pipelines puts through the private-memory AsyncPutBlob (issue #830): in
// client mode the private bytes are STAGED (copied) into a task-owned SHM
// buffer during the submit call itself, so the JNI array can be released
// immediately and the task frees its own staging on completion. The window is
// shared by all YCSB client threads (mutex-guarded deque); when it exceeds
// CLIO_YCSB_WINDOW (default 64, 0 = synchronous), the submitting thread reaps
// the oldest future OUTSIDE the lock. Completion errors are sticky
// (g_async_err) and surfaced on the next put and at drain.
//
// Semantics note: a read may observe the pre-put value of a key whose put is
// still in flight (YCSB does not verify read values in the core workloads);
// load/run phases are separate processes and cleanup() drains, so no put
// escapes its phase.
std::mutex g_win_mtx;
std::deque<clio::run::Future<clio::cte::core::PutBlobTask>> g_inflight;
std::atomic<int> g_async_err{0};
size_t g_window = 64;

void InitOnce() {
  // The Clio SHM transport wakes waiters with tgkill(SIGUSR1); clio-aware
  // threads block the signal and consume it via signalfd, but JVM threads
  // keep the default mask, and SIGUSR1's default action TERMINATES the
  // process if a wake ever lands on one of them. A no-op handler makes such
  // strays harmless without disturbing signalfd delivery on blocked threads.
  struct sigaction sa;
  memset(&sa, 0, sizeof(sa));
  sa.sa_handler = [](int) {};
  sigemptyset(&sa.sa_mask);
  sigaction(SIGUSR1, &sa, nullptr);

  // default_with_runtime=false: connect to the externally-started clio_run
  // daemon (true would embed a runtime in the JVM and fight the daemon for
  // port 9413).
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false)) {
    g_init_rc = -1;
    return;
  }
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    g_init_rc = -2;
    return;
  }
  auto *cte_client = CLIO_CTE_CLIENT;
  if (cte_client == nullptr) {
    g_init_rc = -3;
    return;
  }
  cte_client->Init(clio::cte::core::kCtePoolId);
  clio::cte::core::CreateParams params;
  auto create_task = cte_client->AsyncCreate(
      clio::run::PoolQuery::Dynamic(), clio::cte::core::kCtePoolName,
      clio::cte::core::kCtePoolId, params);
  create_task.Wait();
  if (create_task->GetReturnCode() != 0) {
    g_init_rc = -4;
    return;
  }
  g_tag = new clio::cte::core::Tag("ycsb");
  if (const char *w = std::getenv("CLIO_YCSB_WINDOW")) {
    g_window = static_cast<size_t>(std::strtoul(w, nullptr, 10));
  }
  g_init_rc = 0;
}

// Reap futures until at most `keep` remain. Waits happen outside the lock so
// submitting threads only contend on deque ops, not on put completion.
void ReapDownTo(size_t keep) {
  while (true) {
    clio::run::Future<clio::cte::core::PutBlobTask> fut;
    {
      std::lock_guard<std::mutex> lk(g_win_mtx);
      if (g_inflight.size() <= keep) return;
      fut = std::move(g_inflight.front());
      g_inflight.pop_front();
    }
    fut.Wait();
    auto *t = fut.get();
    if (t == nullptr || t->GetReturnCode() != 0) {
      g_async_err.fetch_add(1);
    }
  }
}

}  // namespace

extern "C" {

JNIEXPORT jint JNICALL Java_site_ycsb_db_ClioClient_nativeInit(JNIEnv *,
                                                               jclass) {
  std::call_once(g_init_once, InitOnce);
  return static_cast<jint>(g_init_rc);
}

JNIEXPORT jint JNICALL Java_site_ycsb_db_ClioClient_nativePut(
    JNIEnv *env, jclass, jstring key, jbyteArray value) {
  if (g_tag == nullptr) return -1;
  const char *key_chars = env->GetStringUTFChars(key, nullptr);
  if (key_chars == nullptr) return -2;
  jsize len = env->GetArrayLength(value);
  jbyte *bytes = env->GetByteArrayElements(value, nullptr);
  if (bytes == nullptr) {
    env->ReleaseStringUTFChars(key, key_chars);
    return -3;
  }
  int rc = 0;
  try {
    g_tag->PutBlob(key_chars, reinterpret_cast<const char *>(bytes),
                   static_cast<size_t>(len));
  } catch (const std::exception &) {
    // Tag::PutBlob throws on a failed put (e.g. daemon死/capacity); an
    // exception escaping a JNI boundary terminates the JVM, so convert to a
    // status the Java adapter can surface as Status.ERROR.
    rc = -10;
  }
  env->ReleaseByteArrayElements(value, bytes, JNI_ABORT);
  env->ReleaseStringUTFChars(key, key_chars);
  return rc;
}

JNIEXPORT jint JNICALL Java_site_ycsb_db_ClioClient_nativePutAsync(
    JNIEnv *env, jclass, jstring key, jbyteArray value) {
  if (g_tag == nullptr) return -1;
  const char *key_chars = env->GetStringUTFChars(key, nullptr);
  if (key_chars == nullptr) return -2;
  jsize len = env->GetArrayLength(value);
  jbyte *bytes = env->GetByteArrayElements(value, nullptr);
  if (bytes == nullptr) {
    env->ReleaseStringUTFChars(key, key_chars);
    return -3;
  }
  std::string key_str(key_chars);
  env->ReleaseStringUTFChars(key, key_chars);

  auto *cte_client = CLIO_CTE_CLIENT;
  auto fut = cte_client->AsyncPutBlob(
      g_tag->GetTagId(), key_str, 0, static_cast<clio::run::u64>(len),
      reinterpret_cast<const char *>(bytes));
  // Client mode staged (copied) the bytes during the call; the array can go.
  env->ReleaseByteArrayElements(value, bytes, JNI_ABORT);
  if (fut.IsNull()) return -11;  // degenerate request or staging alloc failed

  if (g_window == 0) {  // synchronous fallback (CLIO_YCSB_WINDOW=0)
    fut.Wait();
    auto *t = fut.get();
    return (t == nullptr || t->GetReturnCode() != 0) ? -10 : 0;
  }
  {
    std::lock_guard<std::mutex> lk(g_win_mtx);
    g_inflight.push_back(std::move(fut));
  }
  ReapDownTo(g_window);
  return g_async_err.load() != 0 ? -12 : 0;
}

JNIEXPORT jint JNICALL Java_site_ycsb_db_ClioClient_nativeDrain(JNIEnv *,
                                                                jclass) {
  ReapDownTo(0);
  return g_async_err.load();
}

JNIEXPORT jbyteArray JNICALL Java_site_ycsb_db_ClioClient_nativeGet(
    JNIEnv *env, jclass, jstring key) {
  if (g_tag == nullptr) return nullptr;
  const char *key_chars = env->GetStringUTFChars(key, nullptr);
  if (key_chars == nullptr) return nullptr;
  std::string key_str(key_chars);
  env->ReleaseStringUTFChars(key, key_chars);

  clio::run::u64 size = 0;
  std::vector<char> buf;
  try {
    size = g_tag->GetBlobSize(key_str);
    if (size == 0) return nullptr;  // absent key
    buf.resize(size);
    g_tag->GetBlob(key_str, buf.data(), size, 0);
  } catch (const std::exception &) {
    return nullptr;  // surfaced as Status.NOT_FOUND/ERROR by the adapter
  }

  jbyteArray out = env->NewByteArray(static_cast<jsize>(size));
  if (out == nullptr) return nullptr;
  env->SetByteArrayRegion(out, 0, static_cast<jsize>(size),
                          reinterpret_cast<const jbyte *>(buf.data()));
  return out;
}

}  // extern "C"
