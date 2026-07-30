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
#include <set>
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
// In CLIENT mode the private AsyncPutBlob STAGES (copies) the bytes into a
// task-owned SHM buffer during the submit call, so the source can be released
// immediately. In RUNTIME (co-located, CLIO_WITH_RUNTIME=1) mode there is no
// staging — the bdev write reads DIRECTLY from the caller's buffer until the
// task completes — so the window must own a copy of the bytes and free it at
// reap. g_runtime_mode selects the ownership scheme.
struct InflightPut {
  clio::run::Future<clio::cte::core::PutBlobTask> fut_;
  char *owned_buf_;  // runtime mode only; nullptr in client mode
};
std::mutex g_win_mtx;
std::deque<InflightPut> g_inflight;
std::atomic<int> g_async_err{0};
size_t g_window = 64;
bool g_runtime_mode = false;

// ---- scan key index (workloads D/E) ---------------------------------------
// CTE has no ordered key scan, so the binding maintains its own sorted index
// of blob names — the same approach the stock YCSB Redis binding takes (it
// keeps a zset index for exactly this reason). Seeded from
// Tag::GetContainedBlobs() at init (run phases start in a fresh process
// against a loaded store) and updated on every put; scan = lower_bound +
// walk. Guarded by its own mutex; scans copy the keys out under the lock and
// fetch values outside it.
std::mutex g_keys_mtx;
std::set<std::string> g_keys;

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
  g_runtime_mode = CLIO_RUNTIME_MANAGER->IsRuntime();
  {
    std::lock_guard<std::mutex> lk(g_keys_mtx);
    for (auto &name : g_tag->GetContainedBlobs()) {
      g_keys.insert(name);
    }
  }
  if (const char *w = std::getenv("CLIO_YCSB_WINDOW")) {
    g_window = static_cast<size_t>(std::strtoul(w, nullptr, 10));
  }
  g_init_rc = 0;
}

// Reap futures until at most `keep` remain. Waits happen outside the lock so
// submitting threads only contend on deque ops, not on put completion.
void ReapDownTo(size_t keep) {
  while (true) {
    InflightPut entry;
    {
      std::lock_guard<std::mutex> lk(g_win_mtx);
      if (g_inflight.size() <= keep) return;
      entry = std::move(g_inflight.front());
      g_inflight.pop_front();
    }
    entry.fut_.Wait();
    auto *t = entry.fut_.get();
    if (t == nullptr || t->GetReturnCode() != 0) {
      g_async_err.fetch_add(1);
    }
    std::free(entry.owned_buf_);  // no-op in client mode (nullptr)
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
    std::lock_guard<std::mutex> lk(g_keys_mtx);
    g_keys.insert(key_chars);
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
  const char *src = reinterpret_cast<const char *>(bytes);
  char *owned = nullptr;
  if (g_runtime_mode) {
    // Co-located: the put reads the source buffer until completion — the JNI
    // array cannot be released while the task is in flight, so the window
    // owns a heap copy instead (freed at reap).
    owned = static_cast<char *>(std::malloc(static_cast<size_t>(len)));
    if (owned == nullptr) {
      env->ReleaseByteArrayElements(value, bytes, JNI_ABORT);
      return -13;
    }
    std::memcpy(owned, src, static_cast<size_t>(len));
    src = owned;
  }
  auto fut = cte_client->AsyncPutBlob(
      g_tag->GetTagId(), key_str, 0, static_cast<clio::run::u64>(len), src);
  // Client mode staged (copied) the bytes during the call; runtime mode uses
  // the owned copy above — either way the array can go now.
  env->ReleaseByteArrayElements(value, bytes, JNI_ABORT);
  if (fut.IsNull()) {
    std::free(owned);
    return -11;  // degenerate request or staging alloc failed
  }
  {
    std::lock_guard<std::mutex> lk(g_keys_mtx);
    g_keys.insert(key_str);
  }

  if (g_window == 0) {  // synchronous fallback (CLIO_YCSB_WINDOW=0)
    fut.Wait();
    auto *t = fut.get();
    std::free(owned);
    return (t == nullptr || t->GetReturnCode() != 0) ? -10 : 0;
  }
  {
    std::lock_guard<std::mutex> lk(g_win_mtx);
    g_inflight.push_back(InflightPut{std::move(fut), owned});
  }
  ReapDownTo(g_window);
  return g_async_err.load() != 0 ? -12 : 0;
}

JNIEXPORT jint JNICALL Java_site_ycsb_db_ClioClient_nativeDrain(JNIEnv *,
                                                                jclass) {
  ReapDownTo(0);
  return g_async_err.load();
}

JNIEXPORT jobjectArray JNICALL Java_site_ycsb_db_ClioClient_nativeScanKeys(
    JNIEnv *env, jclass, jstring start, jint count) {
  if (g_tag == nullptr || count <= 0) return nullptr;
  const char *start_chars = env->GetStringUTFChars(start, nullptr);
  if (start_chars == nullptr) return nullptr;
  std::string start_str(start_chars);
  env->ReleaseStringUTFChars(start, start_chars);

  std::vector<std::string> keys;
  keys.reserve(static_cast<size_t>(count));
  {
    std::lock_guard<std::mutex> lk(g_keys_mtx);
    for (auto it = g_keys.lower_bound(start_str);
         it != g_keys.end() && keys.size() < static_cast<size_t>(count);
         ++it) {
      keys.push_back(*it);
    }
  }
  jclass str_cls = env->FindClass("java/lang/String");
  jobjectArray out =
      env->NewObjectArray(static_cast<jsize>(keys.size()), str_cls, nullptr);
  if (out == nullptr) return nullptr;
  for (jsize i = 0; i < static_cast<jsize>(keys.size()); ++i) {
    jstring js = env->NewStringUTF(keys[static_cast<size_t>(i)].c_str());
    env->SetObjectArrayElement(out, i, js);
    env->DeleteLocalRef(js);
  }
  return out;
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
