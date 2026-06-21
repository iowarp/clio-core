#include "clio_llm/kvcache/kvcache_manager.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>

// IOWarp CTE client API — Tag class wraps GetOrCreateTag + PutBlob + GetBlob.
#include "clio_cte/core/core_client.h"

// CUDA for D2H / H2D copies when kv_data is on-device.
// Guarded so that the translation unit still compiles on CPU-only builds.
#ifdef IOWARP_LLM_ENABLE_CUDA
#include <cuda_runtime.h>
#define CUDA_CHECK(expr)                                                 \
  do {                                                                   \
    cudaError_t _e = (expr);                                            \
    if (_e != cudaSuccess)                                              \
      fprintf(stderr, "CUDA error %s at %s:%d\n",                      \
              cudaGetErrorString(_e), __FILE__, __LINE__);              \
  } while (0)
#endif

namespace clio_llm {
namespace kvcache {

// ---------------------------------------------------------------------------
// Index entry
// ---------------------------------------------------------------------------
struct IndexEntry {
  std::string blob_key;       // CTE blob name (hex hash of token prefix)
  size_t blob_size;           // stored size in bytes
  size_t token_count;         // number of tokens this block covers
  std::chrono::steady_clock::time_point stored_at;
};

// ---------------------------------------------------------------------------
// Pimpl
// ---------------------------------------------------------------------------
struct KVCacheManager::Impl {
  KVCacheManager::Config cfg;
  bool ready = false;

  // High-level CTE tag handle.  Null until Init() succeeds.
  std::unique_ptr<clio::cte::core::Tag> tag;

  // In-process index: hash → entry.
  // Protected by a mutex because llama.cpp may call us from worker threads.
  mutable std::mutex mu;
  std::unordered_map<std::string, IndexEntry> index;

  // Staging buffer for GPU↔CPU copies.
  // Resized on demand; protected by mu when accessed.
  std::vector<uint8_t> staging;

  bool EnsureStaging(size_t sz) {
    if (staging.size() < sz) staging.resize(sz);
    return true;
  }

  // D2H copy using a pinned staging buffer; caller must hold no lock.
  bool DeviceToHost(const void* dev_ptr, size_t size) {
#ifdef IOWARP_LLM_ENABLE_CUDA
    EnsureStaging(size);
    CUDA_CHECK(cudaMemcpy(staging.data(), dev_ptr, size, cudaMemcpyDeviceToHost));
    return true;
#else
    (void)dev_ptr; (void)size;
    fprintf(stderr, "kvcache_manager: on_gpu=true but CUDA not compiled in\n");
    return false;
#endif
  }

  // H2D copy from staging buffer to device; caller must hold no lock.
  bool HostToDevice(void* dev_ptr, size_t size) {
#ifdef IOWARP_LLM_ENABLE_CUDA
    CUDA_CHECK(cudaMemcpy(dev_ptr, staging.data(), size, cudaMemcpyHostToDevice));
    return true;
#else
    (void)dev_ptr; (void)size;
    fprintf(stderr, "kvcache_manager: on_gpu=true but CUDA not compiled in\n");
    return false;
#endif
  }

  void TrimIndex() {
    // Evict oldest entries when we exceed max_index_entries.
    if (index.size() <= cfg.max_index_entries) return;
    auto oldest_time = std::chrono::steady_clock::time_point::max();
    std::string oldest_key;
    for (auto& [k, v] : index) {
      if (v.stored_at < oldest_time) {
        oldest_time = v.stored_at;
        oldest_key = k;
      }
    }
    if (!oldest_key.empty()) index.erase(oldest_key);
  }
};

// ---------------------------------------------------------------------------
// KVCacheManager implementation
// ---------------------------------------------------------------------------

KVCacheManager::KVCacheManager()
    : cfg_(Config()), impl_(std::make_unique<Impl>()) {
  impl_->cfg = cfg_;
}

KVCacheManager::KVCacheManager(const Config& cfg)
    : cfg_(cfg), impl_(std::make_unique<Impl>()) {
  impl_->cfg = cfg;
}

KVCacheManager::~KVCacheManager() { Shutdown(); }

KVCacheManager::KVCacheManager(KVCacheManager&&) noexcept = default;
KVCacheManager& KVCacheManager::operator=(KVCacheManager&&) noexcept = default;

bool KVCacheManager::Init() {
  if (impl_->ready) return true;

  // Initialize the global CTE client if not already done.
  // CLIO_CTE_CLIENT_INIT is idempotent — safe to call multiple times.
  try {
    if (!clio::cte::core::CLIO_CTE_CLIENT_INIT("", clio::run::PoolQuery::Local())) {
      fprintf(stderr, "kvcache_manager: CLIO_CTE_CLIENT_INIT() failed — "
                      "is the IOWarp runtime running?\n");
      return false;
    }
    // Open or create the KV cache tag (GetOrCreateTag is synchronous here).
    impl_->tag = std::make_unique<clio::cte::core::Tag>(cfg_.tag_name);
  } catch (const std::exception& e) {
    fprintf(stderr, "kvcache_manager: Init failed: %s\n", e.what());
    impl_->tag.reset();
    return false;
  }

  impl_->ready = true;
  return true;
}

void KVCacheManager::Shutdown() {
  if (!impl_->ready) return;
  impl_->tag.reset();
  impl_->ready = false;
}

bool KVCacheManager::IsReady() const { return impl_->ready; }

// ---------------------------------------------------------------------------
// Raw-key API
// ---------------------------------------------------------------------------
bool KVCacheManager::StoreBlockRaw(const std::string& key,
                                    const void* data, size_t size) {
  if (!impl_->ready || key.empty() || !data || size == 0) return false;

  try {
    impl_->tag->PutBlob(key,
                        reinterpret_cast<const char*>(data), size,
                        /*off=*/0, cfg_.evicted_score);
  } catch (const std::exception& e) {
    fprintf(stderr, "kvcache_manager: StoreBlockRaw failed: %s\n", e.what());
    return false;
  }

  {
    std::lock_guard<std::mutex> lk(impl_->mu);
    impl_->index[key] = IndexEntry{key, size, 0,
                                    std::chrono::steady_clock::now()};
    impl_->TrimIndex();
  }
  return true;
}

bool KVCacheManager::LookupRaw(const std::string& key,
                                 std::vector<uint8_t>& out_data) {
  if (!impl_->ready || key.empty()) return false;

  size_t blob_size = 0;
  {
    std::lock_guard<std::mutex> lk(impl_->mu);
    auto it = impl_->index.find(key);
    if (it == impl_->index.end()) return false;
    blob_size = it->second.blob_size;
  }

  out_data.resize(blob_size);
  try {
    impl_->tag->GetBlob(key,
                        reinterpret_cast<char*>(out_data.data()), blob_size,
                        /*off=*/0);
  } catch (const std::exception& e) {
    fprintf(stderr, "kvcache_manager: LookupRaw failed: %s\n", e.what());
    out_data.clear();
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Hash utility
// ---------------------------------------------------------------------------
// FNV-1a 64-bit hash, then formatted as a 16-char hex string.
std::string KVCacheManager::HashTokens(const std::vector<int32_t>& tokens) {
  uint64_t hash = 14695981039346656037ULL;
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(tokens.data());
  size_t n = tokens.size() * sizeof(int32_t);
  for (size_t i = 0; i < n; ++i) {
    hash ^= bytes[i];
    hash *= 1099511628211ULL;
  }
  char buf[17];
  snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(hash));
  return std::string(buf, 16);
}

// ---------------------------------------------------------------------------
// StoreBlock
// ---------------------------------------------------------------------------
bool KVCacheManager::StoreBlock(const std::vector<int32_t>& tokens,
                                 const void* kv_data, size_t size,
                                 bool on_gpu) {
  if (!impl_->ready || tokens.empty() || !kv_data || size == 0) return false;

  const char* host_ptr = reinterpret_cast<const char*>(kv_data);
  if (on_gpu) {
    if (!impl_->DeviceToHost(kv_data, size)) return false;
    host_ptr = reinterpret_cast<const char*>(impl_->staging.data());
  }

  std::string key = HashTokens(tokens);

  try {
    impl_->tag->PutBlob(key, host_ptr, size, /*off=*/0, cfg_.evicted_score);
  } catch (const std::exception& e) {
    fprintf(stderr, "kvcache_manager: StoreBlock failed: %s\n", e.what());
    return false;
  }

  {
    std::lock_guard<std::mutex> lk(impl_->mu);
    impl_->index[key] = IndexEntry{
        key, size, tokens.size(),
        std::chrono::steady_clock::now()};
    impl_->TrimIndex();
  }
  return true;
}

// ---------------------------------------------------------------------------
// LookupPrefix
// ---------------------------------------------------------------------------
bool KVCacheManager::LookupPrefix(const std::vector<int32_t>& tokens,
                                   void* dst, size_t dst_size, bool on_gpu,
                                   size_t& matched_len) {
  if (!impl_->ready || tokens.empty() || !dst) {
    matched_len = 0;
    return false;
  }

  // Try progressively shorter prefixes, longest first.
  for (size_t len = tokens.size(); len > 0; --len) {
    std::string key = HashTokens(
        std::vector<int32_t>(tokens.begin(), tokens.begin() + len));

    std::string blob_key;
    size_t blob_size = 0;
    {
      std::lock_guard<std::mutex> lk(impl_->mu);
      auto it = impl_->index.find(key);
      if (it == impl_->index.end()) continue;
      blob_key = it->second.blob_key;
      blob_size = it->second.blob_size;
    }

    if (blob_size > dst_size) continue;  // would overflow destination

    char* host_dst = on_gpu
                         ? reinterpret_cast<char*>(impl_->staging.data())
                         : reinterpret_cast<char*>(dst);
    if (on_gpu) impl_->EnsureStaging(blob_size);

    try {
      impl_->tag->GetBlob(blob_key, host_dst, blob_size, /*off=*/0);
    } catch (const std::exception&) {
      continue;  // blob not available — try shorter prefix
    }

    if (on_gpu) {
      if (!impl_->HostToDevice(dst, blob_size)) continue;
    }

    matched_len = len;
    return true;
  }

  matched_len = 0;
  return false;
}

// ---------------------------------------------------------------------------
// OnEvict
// ---------------------------------------------------------------------------
void KVCacheManager::OnEvict(const std::vector<int32_t>& tokens,
                              const void* kv_data, size_t size, bool on_gpu) {
  if (!impl_->ready || tokens.empty()) return;

  if (kv_data && size > 0) {
    StoreBlock(tokens, kv_data, size, on_gpu);
    return;
  }

  // kv_data == nullptr: just remove from index (data was already gone).
  std::string key = HashTokens(tokens);
  std::lock_guard<std::mutex> lk(impl_->mu);
  impl_->index.erase(key);
}

}  // namespace kvcache
}  // namespace clio_llm
