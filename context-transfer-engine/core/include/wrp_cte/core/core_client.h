#ifndef WRPCTE_CORE_CLIENT_H_
#define WRPCTE_CORE_CLIENT_H_

#include <chimaera/chimaera.h>
#include <hermes_shm/util/singleton.h>
#include <wrp_cte/core/core_tasks.h>

namespace wrp_cte::core {

class Client : public chi::ContainerClient {
public:
  Client() = default;
  explicit Client(const chi::PoolId &pool_id) { Init(pool_id); }

  /**
   * Synchronous container creation - waits for completion
   */
  void Create(const hipc::MemContext &mctx, const chi::PoolQuery &pool_query,
              const std::string &pool_name, const chi::PoolId &custom_pool_id,
              const CreateParams &params = CreateParams()) {
    auto task =
        AsyncCreate(mctx, pool_query, pool_name, custom_pool_id, params);
    task->Wait();

    // CRITICAL: Update client pool_id_ with the actual pool ID from the task
    pool_id_ = task->new_pool_id_;

    CHI_IPC->DelTask(task);
  }

  /**
   * Asynchronous container creation - returns immediately
   */
  hipc::FullPtr<CreateTask>
  AsyncCreate(const hipc::MemContext &mctx, const chi::PoolQuery &pool_query,
              const std::string &pool_name, const chi::PoolId &custom_pool_id,
              const CreateParams &params = CreateParams()) {
    (void)mctx; // Suppress unused parameter warning
    auto *ipc_manager = CHI_IPC;

    // CRITICAL: CreateTask MUST use admin pool for GetOrCreatePool processing
    auto task = ipc_manager->NewTask<CreateTask>(
        chi::CreateTaskId(),
        chi::kAdminPoolId, // Always use admin pool for CreateTask
        pool_query,
        CreateParams::chimod_lib_name, // ChiMod name from CreateParams
        pool_name,                     // Pool name from parameter
        custom_pool_id,                // Explicit pool ID from parameter
        params);                       // CreateParams with configuration

    // Submit to runtime
    ipc_manager->Enqueue(task);
    return task;
  }

  /**
   * Synchronous target registration - waits for completion
   */
  chi::u32 RegisterTarget(const hipc::MemContext &mctx,
                          const std::string &target_name,
                          chimaera::bdev::BdevType bdev_type,
                          chi::u64 total_size,
                          const chi::PoolQuery &target_query = chi::PoolQuery::Local(),
                          const chi::PoolId &bdev_id = chi::PoolId::GetNull()) {
    auto task = AsyncRegisterTarget(mctx, target_name, bdev_type, total_size, target_query, bdev_id);
    task->Wait();
    chi::u32 result = task->return_code_.load();
    CHI_IPC->DelTask(task);
    return result;
  }

  /**
   * Asynchronous target registration - returns immediately
   */
  hipc::FullPtr<RegisterTargetTask>
  AsyncRegisterTarget(const hipc::MemContext &mctx,
                      const std::string &target_name,
                      chimaera::bdev::BdevType bdev_type, chi::u64 total_size,
                      const chi::PoolQuery &target_query = chi::PoolQuery::Local(),
                      const chi::PoolId &bdev_id = chi::PoolId::GetNull()) {
    (void)mctx; // Suppress unused parameter warning
    auto *ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<RegisterTargetTask>(
        chi::CreateTaskId(), pool_id_, chi::PoolQuery::Dynamic(), target_name,
        bdev_type, total_size, target_query, bdev_id);

    ipc_manager->Enqueue(task);
    return task;
  }

  /**
   * Synchronous target unregistration - waits for completion
   */
  chi::u32 UnregisterTarget(const hipc::MemContext &mctx,
                            const std::string &target_name) {
    auto task = AsyncUnregisterTarget(mctx, target_name);
    task->Wait();
    chi::u32 result = task->return_code_.load();
    CHI_IPC->DelTask(task);
    return result;
  }

  /**
   * Asynchronous target unregistration - returns immediately
   */
  hipc::FullPtr<UnregisterTargetTask>
  AsyncUnregisterTarget(const hipc::MemContext &mctx,
                        const std::string &target_name) {
    (void)mctx; // Suppress unused parameter warning
    auto *ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<UnregisterTargetTask>(
        chi::CreateTaskId(), pool_id_, chi::PoolQuery::Dynamic(), target_name);

    ipc_manager->Enqueue(task);
    return task;
  }

  /**
   * Synchronous target listing - waits for completion
   */
  std::vector<std::string> ListTargets(const hipc::MemContext &mctx) {
    auto task = AsyncListTargets(mctx);
    task->Wait();

    // Convert HSHM vector to standard vector for client use
    std::vector<std::string> result;
    result.reserve(task->target_names_.size());
    for (const auto &target_name : task->target_names_) {
      result.push_back(target_name.str());
    }

    CHI_IPC->DelTask(task);
    return result;
  }

  /**
   * Asynchronous target listing - returns immediately
   */
  hipc::FullPtr<ListTargetsTask>
  AsyncListTargets(const hipc::MemContext &mctx) {
    (void)mctx; // Suppress unused parameter warning
    auto *ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<ListTargetsTask>(
        chi::CreateTaskId(), pool_id_, chi::PoolQuery::Dynamic());

    ipc_manager->Enqueue(task);
    return task;
  }

  /**
   * Synchronous target stats update - waits for completion
   */
  chi::u32 StatTargets(const hipc::MemContext &mctx) {
    auto task = AsyncStatTargets(mctx);
    task->Wait();
    chi::u32 result = task->return_code_.load();
    CHI_IPC->DelTask(task);
    return result;
  }

  /**
   * Asynchronous target stats update - returns immediately
   */
  hipc::FullPtr<StatTargetsTask>
  AsyncStatTargets(const hipc::MemContext &mctx) {
    (void)mctx; // Suppress unused parameter warning
    auto *ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<StatTargetsTask>(
        chi::CreateTaskId(), pool_id_, chi::PoolQuery::Dynamic());

    ipc_manager->Enqueue(task);
    return task;
  }

  /**
   * Synchronous get or create tag - waits for completion
   */
  TagId GetOrCreateTag(const hipc::MemContext &mctx,
                       const std::string &tag_name,
                       const TagId &tag_id = TagId::GetNull()) {
    auto task = AsyncGetOrCreateTag(mctx, tag_name, tag_id);
    task->Wait();

    TagId result = task->tag_id_;
    CHI_IPC->DelTask(task);
    return result;
  }

  /**
   * Asynchronous get or create tag - returns immediately
   */
  hipc::FullPtr<GetOrCreateTagTask<CreateParams>>
  AsyncGetOrCreateTag(const hipc::MemContext &mctx, const std::string &tag_name,
                      const TagId &tag_id = TagId::GetNull()) {
    (void)mctx; // Suppress unused parameter warning
    auto *ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<GetOrCreateTagTask<CreateParams>>(
        chi::CreateTaskId(), pool_id_, chi::PoolQuery::Dynamic(), tag_name,
        tag_id);

    ipc_manager->Enqueue(task);
    return task;
  }

  /**
   * Synchronous put blob - waits for completion
   */
  bool PutBlob(const hipc::MemContext &mctx, const TagId &tag_id,
               const std::string &blob_name, chi::u64 offset, chi::u64 size,
               hipc::Pointer blob_data, float score, chi::u32 flags) {
    auto task = AsyncPutBlob(mctx, tag_id, blob_name, offset, size, blob_data,
                             score, flags);
    task->Wait();
    bool result = (task->return_code_.load() == 0);
    if (!result) {
      HELOG(kError, "PutBlob failed: {}", task->return_code_.load());
    }
    CHI_IPC->DelTask(task);
    return result;
  }

  /**
   * Asynchronous put blob - returns immediately (unimplemented for now)
   */
  hipc::FullPtr<PutBlobTask>
  AsyncPutBlob(const hipc::MemContext &mctx, const TagId &tag_id,
               const std::string &blob_name, chi::u64 offset, chi::u64 size,
               hipc::Pointer blob_data, float score, chi::u32 flags) {
    (void)mctx; // Suppress unused parameter warning
    auto *ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<PutBlobTask>(
        chi::CreateTaskId(), pool_id_, chi::PoolQuery::Dynamic(), tag_id,
        blob_name, offset, size, blob_data, score, flags);

    ipc_manager->Enqueue(task);
    return task;
  }

  /**
   * Synchronous get blob - waits for completion
   */
  bool GetBlob(const hipc::MemContext &mctx, const TagId &tag_id,
               const std::string &blob_name, chi::u64 offset, chi::u64 size,
               chi::u32 flags, hipc::Pointer blob_data) {
    auto task =
        AsyncGetBlob(mctx, tag_id, blob_name, offset, size, flags, blob_data);
    task->Wait();
    bool result = (task->return_code_.load() == 0);
    CHI_IPC->DelTask(task);
    return result;
  }

  /**
   * Asynchronous get blob - returns immediately
   */
  hipc::FullPtr<GetBlobTask>
  AsyncGetBlob(const hipc::MemContext &mctx, const TagId &tag_id,
               const std::string &blob_name, chi::u64 offset, chi::u64 size,
               chi::u32 flags, hipc::Pointer blob_data) {
    (void)mctx; // Suppress unused parameter warning
    auto *ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<GetBlobTask>(
        chi::CreateTaskId(), pool_id_, chi::PoolQuery::Dynamic(), tag_id,
        blob_name, offset, size, flags, blob_data);

    ipc_manager->Enqueue(task);
    return task;
  }

  /**
   * Synchronous reorganize blob - waits for completion
   */
  chi::u32 ReorganizeBlob(const hipc::MemContext &mctx, const TagId &tag_id,
                          const std::string &blob_name, float new_score) {
    auto task = AsyncReorganizeBlob(mctx, tag_id, blob_name, new_score);
    task->Wait();
    chi::u32 result = task->return_code_.load();
    CHI_IPC->DelTask(task);
    return result;
  }

  /**
   * Asynchronous reorganize blob - returns immediately
   */
  hipc::FullPtr<ReorganizeBlobTask>
  AsyncReorganizeBlob(const hipc::MemContext &mctx, const TagId &tag_id,
                      const std::string &blob_name, float new_score) {
    (void)mctx; // Suppress unused parameter warning
    auto *ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<ReorganizeBlobTask>(
        chi::CreateTaskId(), pool_id_, chi::PoolQuery::Dynamic(), tag_id,
        blob_name, new_score);

    ipc_manager->Enqueue(task);
    return task;
  }

  /**
   * Synchronous delete blob - waits for completion
   */
  bool DelBlob(const hipc::MemContext &mctx, const TagId &tag_id,
               const std::string &blob_name) {
    auto task = AsyncDelBlob(mctx, tag_id, blob_name);
    task->Wait();
    bool result = (task->return_code_.load() == 0);
    CHI_IPC->DelTask(task);
    return result;
  }

  /**
   * Asynchronous delete blob - returns immediately
   */
  hipc::FullPtr<DelBlobTask> AsyncDelBlob(const hipc::MemContext &mctx,
                                          const TagId &tag_id,
                                          const std::string &blob_name) {
    (void)mctx; // Suppress unused parameter warning
    auto *ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<DelBlobTask>(chi::CreateTaskId(), pool_id_,
                                                  chi::PoolQuery::Dynamic(),
                                                  tag_id, blob_name);

    ipc_manager->Enqueue(task);
    return task;
  }

  /**
   * Synchronous delete tag by tag ID - waits for completion
   */
  bool DelTag(const hipc::MemContext &mctx, const TagId &tag_id) {
    auto task = AsyncDelTag(mctx, tag_id);
    task->Wait();
    bool result = (task->return_code_.load() == 0);
    CHI_IPC->DelTask(task);
    return result;
  }

  /**
   * Synchronous delete tag by tag name - waits for completion
   */
  bool DelTag(const hipc::MemContext &mctx, const std::string &tag_name) {
    auto task = AsyncDelTag(mctx, tag_name);
    task->Wait();
    bool result = (task->return_code_.load() == 0);
    CHI_IPC->DelTask(task);
    return result;
  }

  /**
   * Asynchronous delete tag by tag ID - returns immediately
   */
  hipc::FullPtr<DelTagTask> AsyncDelTag(const hipc::MemContext &mctx,
                                        const TagId &tag_id) {
    (void)mctx; // Suppress unused parameter warning
    auto *ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<DelTagTask>(
        chi::CreateTaskId(), pool_id_, chi::PoolQuery::Dynamic(), tag_id);

    ipc_manager->Enqueue(task);
    return task;
  }

  /**
   * Asynchronous delete tag by tag name - returns immediately
   */
  hipc::FullPtr<DelTagTask> AsyncDelTag(const hipc::MemContext &mctx,
                                        const std::string &tag_name) {
    (void)mctx; // Suppress unused parameter warning
    auto *ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<DelTagTask>(
        chi::CreateTaskId(), pool_id_, chi::PoolQuery::Dynamic(), tag_name);

    ipc_manager->Enqueue(task);
    return task;
  }

  /**
   * Synchronous get tag size - waits for completion
   */
  size_t GetTagSize(const hipc::MemContext &mctx, const TagId &tag_id) {
    auto task = AsyncGetTagSize(mctx, tag_id);
    task->Wait();
    size_t result = (task->return_code_.load() == 0) ? task->tag_size_ : 0;
    CHI_IPC->DelTask(task);
    return result;
  }

  /**
   * Asynchronous get tag size - returns immediately
   */
  hipc::FullPtr<GetTagSizeTask> AsyncGetTagSize(const hipc::MemContext &mctx,
                                                const TagId &tag_id) {
    (void)mctx; // Suppress unused parameter warning
    auto *ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<GetTagSizeTask>(
        chi::CreateTaskId(), pool_id_, chi::PoolQuery::Dynamic(), tag_id);

    ipc_manager->Enqueue(task);
    return task;
  }

  /**
   * Synchronous poll telemetry log - waits for completion
   */
  std::vector<CteTelemetry>
  PollTelemetryLog(const hipc::MemContext &mctx,
                   std::uint64_t minimum_logical_time) {
    auto task = AsyncPollTelemetryLog(mctx, minimum_logical_time);
    task->Wait();

    // Convert HSHM vector to standard vector for client use
    std::vector<CteTelemetry> result;
    result.reserve(task->entries_.size());
    for (const auto &entry : task->entries_) {
      result.push_back(entry);
    }

    CHI_IPC->DelTask(task);
    return result;
  }

  /**
   * Asynchronous poll telemetry log - returns immediately
   */
  hipc::FullPtr<PollTelemetryLogTask>
  AsyncPollTelemetryLog(const hipc::MemContext &mctx,
                        std::uint64_t minimum_logical_time) {
    (void)mctx; // Suppress unused parameter warning
    auto *ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<PollTelemetryLogTask>(
        chi::CreateTaskId(), pool_id_, chi::PoolQuery::Dynamic(),
        minimum_logical_time);

    ipc_manager->Enqueue(task);
    return task;
  }

  /**
   * Synchronous get blob score - waits for completion
   */
  float GetBlobScore(const hipc::MemContext &mctx, const TagId &tag_id,
                     const std::string &blob_name) {
    auto task = AsyncGetBlobScore(mctx, tag_id, blob_name);
    task->Wait();
    float result = (task->return_code_.load() == 0) ? task->score_ : 0.0f;
    CHI_IPC->DelTask(task);
    return result;
  }

  /**
   * Asynchronous get blob score - returns immediately
   */
  hipc::FullPtr<GetBlobScoreTask>
  AsyncGetBlobScore(const hipc::MemContext &mctx, const TagId &tag_id,
                    const std::string &blob_name) {
    (void)mctx; // Suppress unused parameter warning
    auto *ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<GetBlobScoreTask>(
        chi::CreateTaskId(), pool_id_, chi::PoolQuery::Dynamic(), tag_id,
        blob_name);

    ipc_manager->Enqueue(task);
    return task;
  }

  /**
   * Synchronous get blob size - waits for completion
   */
  chi::u64 GetBlobSize(const hipc::MemContext &mctx, const TagId &tag_id,
                       const std::string &blob_name) {
    auto task = AsyncGetBlobSize(mctx, tag_id, blob_name);
    task->Wait();
    chi::u64 result = (task->return_code_.load() == 0) ? task->size_ : 0;
    CHI_IPC->DelTask(task);
    return result;
  }

  /**
   * Asynchronous get blob size - returns immediately
   */
  hipc::FullPtr<GetBlobSizeTask>
  AsyncGetBlobSize(const hipc::MemContext &mctx, const TagId &tag_id,
                   const std::string &blob_name) {
    (void)mctx; // Suppress unused parameter warning
    auto *ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<GetBlobSizeTask>(
        chi::CreateTaskId(), pool_id_, chi::PoolQuery::Dynamic(), tag_id,
        blob_name);

    ipc_manager->Enqueue(task);
    return task;
  }

  /**
   * Synchronous get contained blobs - waits for completion
   */
  std::vector<std::string> GetContainedBlobs(const hipc::MemContext &mctx,
                                             const TagId &tag_id) {
    auto task = AsyncGetContainedBlobs(mctx, tag_id);
    task->Wait();
    std::vector<std::string> result;
    if (task->return_code_.load() == 0) {
      for (const auto &blob_name : task->blob_names_) {
        result.emplace_back(blob_name.str());
      }
    }
    CHI_IPC->DelTask(task);
    return result;
  }

  /**
   * Asynchronous get contained blobs - returns immediately
   */
  hipc::FullPtr<GetContainedBlobsTask>
  AsyncGetContainedBlobs(const hipc::MemContext &mctx, const TagId &tag_id) {
    (void)mctx; // Suppress unused parameter warning
    auto *ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<GetContainedBlobsTask>(
        chi::CreateTaskId(), pool_id_, chi::PoolQuery::Dynamic(), tag_id);

    ipc_manager->Enqueue(task);
    return task;
  }

  /**
   * Synchronous tag query - waits for completion
   * Queries tags by regex pattern
   * @param mctx Memory context
   * @param tag_regex Tag regex pattern to match
   * @param max_tags Maximum number of tags to return (0 = no limit)
   * @param pool_query Pool query for routing (default: Broadcast)
   * @return Vector of matching tag names
   */
  std::vector<std::string> TagQuery(const hipc::MemContext &mctx,
                                     const std::string &tag_regex,
                                     chi::u32 max_tags = 0,
                                     const chi::PoolQuery &pool_query = chi::PoolQuery::Broadcast()) {
    auto task = AsyncTagQuery(mctx, tag_regex, max_tags, pool_query);
    task->Wait();
    std::vector<std::string> result;
    if (task->return_code_.load() == 0) {
      for (const auto &tag_name : task->results_) {
        result.emplace_back(tag_name.str());
      }
    }
    CHI_IPC->DelTask(task);
    return result;
  }

  /**
   * Asynchronous tag query - returns immediately
   * @param mctx Memory context
   * @param tag_regex Tag regex pattern to match
   * @param pool_query Pool query for routing (default: Broadcast)
   * @return Task pointer for async operation
   */
  hipc::FullPtr<TagQueryTask>
  AsyncTagQuery(const hipc::MemContext &mctx,
                const std::string &tag_regex,
                chi::u32 max_tags = 0,
                const chi::PoolQuery &pool_query = chi::PoolQuery::Broadcast()) {
    (void)mctx; // Suppress unused parameter warning
    auto *ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<TagQueryTask>(
        chi::CreateTaskId(), pool_id_, pool_query, tag_regex, max_tags);

    ipc_manager->Enqueue(task);
    return task;
  }

  /**
   * Synchronous blob query - waits for completion
   * Queries blobs by tag and blob regex patterns
   * @param mctx Memory context
   * @param tag_regex Tag regex pattern to match
   * @param blob_regex Blob regex pattern to match
   * @param max_blobs Maximum number of blobs to return (0 = no limit)
   * @param pool_query Pool query for routing (default: Broadcast)
   * @return Vector of pairs (tag_name, blob_name) for matching blobs
   */
  std::vector<std::pair<std::string, std::string>> BlobQuery(const hipc::MemContext &mctx,
                                      const std::string &tag_regex,
                                      const std::string &blob_regex,
                                      chi::u32 max_blobs = 0,
                                      const chi::PoolQuery &pool_query = chi::PoolQuery::Broadcast()) {
    auto task = AsyncBlobQuery(mctx, tag_regex, blob_regex, max_blobs, pool_query);
    task->Wait();
    std::vector<std::pair<std::string, std::string>> result;
    if (task->return_code_.load() == 0) {
      for (const auto &pair : task->results_) {
        result.emplace_back(pair.first_->str(), pair.second_->str());
      }
    }
    CHI_IPC->DelTask(task);
    return result;
  }

  /**
   * Asynchronous blob query - returns immediately
   * @param mctx Memory context
   * @param tag_regex Tag regex pattern to match
   * @param blob_regex Blob regex pattern to match
   * @param pool_query Pool query for routing (default: Broadcast)
   * @return Task pointer for async operation
   */
  hipc::FullPtr<BlobQueryTask>
  AsyncBlobQuery(const hipc::MemContext &mctx,
                 const std::string &tag_regex,
                 const std::string &blob_regex,
                 chi::u32 max_blobs = 0,
                 const chi::PoolQuery &pool_query = chi::PoolQuery::Broadcast()) {
    (void)mctx; // Suppress unused parameter warning
    auto *ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<BlobQueryTask>(
        chi::CreateTaskId(), pool_id_, pool_query, tag_regex, blob_regex, max_blobs);

    ipc_manager->Enqueue(task);
    return task;
  }
};

// Global pointer-based singleton for CTE client with lazy initialization
HSHM_DEFINE_GLOBAL_PTR_VAR_H(wrp_cte::core::Client, g_cte_client);

/**
 * Initialize CTE client and configuration subsystem
 * @param config_path Optional path to configuration file
 * @param pool_query Pool query type for CTE container creation (default: Dynamic)
 * @return true if initialization succeeded, false otherwise
 */
bool WRP_CTE_CLIENT_INIT(const std::string &config_path = "",
                         const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic());

/**
 * Tag wrapper class - provides convenient API for tag operations
 */
class Tag {
private:
  TagId tag_id_;
  std::string tag_name_;

public:
  /**
   * Constructor - Call the WRP_CTE client GetOrCreateTag function
   * @param tag_name Tag name to get or create
   */
  explicit Tag(const std::string &tag_name);

  /**
   * Constructor - Does not call WRP_CTE client function, just sets the TagId
   * variable
   * @param tag_id Tag ID to use directly
   */
  explicit Tag(const TagId &tag_id);

  /**
   * PutBlob - Allocates a SHM pointer and then calls PutBlob (SHM)
   * @param blob_name Name of the blob
   * @param data Raw data pointer
   * @param data_size Size of data
   * @param off Offset within blob (default 0)
   */
  void PutBlob(const std::string &blob_name, const char *data, size_t data_size,
               size_t off = 0);

  /**
   * PutBlob (SHM) - Direct shared memory version
   * @param blob_name Name of the blob
   * @param data Shared memory pointer to data
   * @param data_size Size of data
   * @param off Offset within blob (default 0)
   * @param score Blob score for placement decisions (default 1.0)
   */
  void PutBlob(const std::string &blob_name, const hipc::Pointer &data,
               size_t data_size, size_t off = 0, float score = 1.0f);

  /**
   * Asynchronous PutBlob (SHM) - Caller must manage shared memory lifecycle
   * @param blob_name Name of the blob
   * @param data Shared memory pointer to data (must remain valid until task
   * completes)
   * @param data_size Size of data
   * @param off Offset within blob (default 0)
   * @param score Blob score for placement decisions (default 1.0)
   * @return Task pointer for async operation
   * @note For raw data, caller must allocate shared memory using
   * CHI_IPC->AllocateBuffer<void>() and keep the FullPtr alive until the async
   * task completes
   */
  hipc::FullPtr<PutBlobTask> AsyncPutBlob(const std::string &blob_name,
                                          const hipc::Pointer &data,
                                          size_t data_size, size_t off = 0,
                                          float score = 1.0f);

  /**
   * GetBlob - Allocates shared memory, retrieves blob data, copies to output
   * buffer
   * @param blob_name Name of the blob to retrieve
   * @param data Output buffer to copy blob data into (must be pre-allocated by
   * caller)
   * @param data_size Size of data to retrieve (must be > 0)
   * @param off Offset within blob (default 0)
   * @note Automatically handles shared memory allocation/deallocation
   */
  void GetBlob(const std::string &blob_name, char *data, size_t data_size,
               size_t off = 0);

  /**
   * GetBlob (SHM) - Retrieves blob data into pre-allocated shared memory buffer
   * @param blob_name Name of the blob to retrieve
   * @param data Pre-allocated shared memory pointer for output data (must not
   * be null)
   * @param data_size Size of data to retrieve (must be > 0)
   * @param off Offset within blob (default 0)
   * @note Caller must pre-allocate shared memory using
   * CHI_IPC->AllocateBuffer<void>(data_size)
   */
  void GetBlob(const std::string &blob_name, hipc::Pointer data,
               size_t data_size, size_t off = 0);

  /**
   * Get blob score
   * @param blob_name Name of the blob
   * @return Blob score (0.0-1.0)
   */
  float GetBlobScore(const std::string &blob_name);

  /**
   * Get blob size
   * @param blob_name Name of the blob
   * @return Blob size in bytes
   */
  chi::u64 GetBlobSize(const std::string &blob_name);

  /**
   * Get all blob names contained in this tag
   * @return Vector of blob names in this tag
   */
  std::vector<std::string> GetContainedBlobs();

  /**
   * Get the TagId for this tag
   * @return TagId of this tag
   */
  const TagId &GetTagId() const { return tag_id_; }
};

} // namespace wrp_cte::core

// Global singleton macro for CTE client access (returns pointer, not reference)
#define WRP_CTE_CLIENT                                                         \
  (&(*HSHM_GET_GLOBAL_PTR_VAR(wrp_cte::core::Client,                           \
                              wrp_cte::core::g_cte_client)))

#endif // WRPCTE_CORE_CLIENT_H_