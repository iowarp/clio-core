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
#include <limits>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>

namespace {

clio::cte::core::Tag *g_tag = nullptr;
std::once_flag g_init_once;
int g_init_rc = -100;  // set by the once-body; <0 means init failed

// ---- writes ---------------------------------------------------------------
// Every put is Client::PutBlob: synchronous, from the caller's buffer. With
// the CTE shm cache connected (CLIO_CTE_CLIENT_INIT connects it when the
// compose has one) it is a copy into the cache segment -- no task, no wait;
// otherwise a task the call waits on. Either way the source is reusable when
// it returns and a read issued afterwards sees the write.

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
  // CLIO_CTE_CLIENT_INIT already bound the singleton's pool (and honored a
  // CLIO_CTE_POOL interposer override, issue #886) and get-or-created the
  // pool — re-binding here would undo the override.
  g_tag = new clio::cte::core::Tag("ycsb");
  {
    std::lock_guard<std::mutex> lk(g_keys_mtx);
    for (auto &name : g_tag->GetContainedBlobs()) {
      g_keys.insert(name);
    }
  }
  g_init_rc = 0;
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
  if (CLIO_CTE_CLIENT->PutBlob(g_tag->GetTagId(), key_chars, 0,
                               static_cast<clio::run::u64>(len),
                               reinterpret_cast<const char *>(bytes)) == 0) {
    std::lock_guard<std::mutex> lk(g_keys_mtx);
    g_keys.insert(key_chars);
  } else {
    rc = -10;  // surfaced by the Java adapter as Status.ERROR
  }
  env->ReleaseByteArrayElements(value, bytes, JNI_ABORT);
  env->ReleaseStringUTFChars(key, key_chars);
  return rc;
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

/**
 * Direct-ByteBuffer put: the source is a Java direct buffer, so there is no
 * JNI array copy. Client::PutBlob is synchronous, so the buffer may be
 * reused the moment this returns.
 * @return 0 on success, -1/-2/-3 on bad arguments, -11 when the put failed
 */
JNIEXPORT jint JNICALL Java_site_ycsb_db_ClioClient_nativePutDirect(
    JNIEnv *env, jclass, jstring key, jobject buf, jint len) {
  if (g_tag == nullptr || len <= 0) return -1;
  const char *key_chars = env->GetStringUTFChars(key, nullptr);
  if (key_chars == nullptr) return -2;
  std::string key_str(key_chars);
  env->ReleaseStringUTFChars(key, key_chars);
  const char *src = static_cast<const char *>(env->GetDirectBufferAddress(buf));
  if (src == nullptr) return -3;
  if (CLIO_CTE_CLIENT->PutBlob(g_tag->GetTagId(), key_str, 0,
                               static_cast<clio::run::u64>(len), src) != 0) {
    return -11;
  }
  std::lock_guard<std::mutex> lk(g_keys_mtx);
  g_keys.insert(key_str);
  return 0;
}

// Direct-ByteBuffer read (item 2): copy the payload once, store -> the Java
// direct buffer, skipping the byte[] allocation + SetByteArrayRegion copy.
// Returns payload size, 0 = absent, -1 = error, -2 = buffer too small.
JNIEXPORT jint JNICALL Java_site_ycsb_db_ClioClient_nativeGetDirect(
    JNIEnv *env, jclass, jstring key, jobject buf) {
  if (g_tag == nullptr) return -1;
  const char *key_chars = env->GetStringUTFChars(key, nullptr);
  if (key_chars == nullptr) return -1;
  std::string key_str(key_chars);
  env->ReleaseStringUTFChars(key, key_chars);
  char *dst = static_cast<char *>(env->GetDirectBufferAddress(buf));
  jlong cap = env->GetDirectBufferCapacity(buf);
  if (dst == nullptr) return -1;
  try {
    clio::run::u64 size = g_tag->GetBlobSize(key_str);
    if (size == 0) return 0;
    if (static_cast<jlong>(size) > cap) return -2;
    g_tag->GetBlob(key_str, dst, size, 0);
    return static_cast<jint>(size);
  } catch (const std::exception &) {
    return -1;
  }
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
