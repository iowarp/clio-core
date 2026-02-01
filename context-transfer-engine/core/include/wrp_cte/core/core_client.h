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
   * Asynchronous container creation - returns immediately
   * @param pool_query Pool query for task routing
   * @param pool_name Name of the pool
   * @param custom_pool_id Explicit pool ID
   * @param params Create parameters
   */
  chi::Future<CreateTask> AsyncCreate(
      const chi::PoolQuery &pool_query, const std::string &pool_name,
      const chi::PoolId &custom_pool_id,
      const CreateParams &params = CreateParams()) {
    auto *ipc_manager = CHI_IPC;

    // CRITICAL: CreateTask MUST use admin pool for GetOrCreatePool processing
    // Pass 'this' as client pointer for PostWait callback
    auto task = ipc_manager->NewTask<CreateTask>(
        chi::CreateTaskId(),
        chi::kAdminPoolId,  // Always use admin pool for CreateTask
        pool_query,
        CreateParams::chimod_lib_name,  // ChiMod name from CreateParams
        pool_name,                      // Pool name from parameter
        custom_pool_id,                 // Explicit pool ID from parameter
        this,                           // Client pointer for PostWait
        params);                        // CreateParams with configuration

    // Submit to runtime
    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous target registration - returns immediately
   * @param target_name Name of the target to register
   * @param bdev_type Block device type
   * @param total_size Total size of the target
   * @param target_query Pool query for target routing
   * @param bdev_id Block device ID
   * @param pool_query Pool query for task routing (default: Dynamic)
   */
  chi::Future<RegisterTargetTask> AsyncRegisterTarget(
      const std::string &target_name, chimaera::bdev::BdevType bdev_type,
      chi::u64 total_size,
      const chi::PoolQuery &target_query = chi::PoolQuery::Local(),
      const chi::PoolId &bdev_id = chi::PoolId::GetNull(),
      const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic()) {
    auto *ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<RegisterTargetTask>(
        chi::CreateTaskId(), pool_id_, pool_query, target_name,
        bdev_type, total_size, target_query, bdev_id);

    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous target unregistration - returns immediately
   * @param target_name Name of the target to unregister
   * @param pool_query Pool query for task routing (default: Dynamic)
   */
  chi::Future<UnregisterTargetTask> AsyncUnregisterTarget(
      const std::string &target_name,
      const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic()) {
    auto *ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<UnregisterTargetTask>(
        chi::CreateTaskId(), pool_id_, pool_query, target_name);

    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous target listing - returns immediately
   * @param pool_query Pool query for task routing (default: Dynamic)
   */
  chi::Future<ListTargetsTask> AsyncListTargets(
      const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic()) {
    auto *ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<ListTargetsTask>(
        chi::CreateTaskId(), pool_id_, pool_query);

    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous target stats update - returns immediately
   * @param pool_query Pool query for task routing (default: Dynamic)
   */
  chi::Future<StatTargetsTask> AsyncStatTargets(
      const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic()) {
    auto *ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<StatTargetsTask>(
        chi::CreateTaskId(), pool_id_, pool_query);

    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous get target info - returns target score, capacity, and stats
   * @param target_name Name of the target
   * @param pool_query Pool query for task routing (default: Dynamic)
   */
  chi::Future<GetTargetInfoTask> AsyncGetTargetInfo(
      const std::string &target_name,
      const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic()) {
    auto *ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<GetTargetInfoTask>(
        chi::CreateTaskId(), pool_id_, pool_query, target_name);

    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous get or create tag - returns immediately
   * @param tag_name Name of the tag
   * @param tag_id Optional tag ID
   * @param pool_query Pool query for task routing (default: Dynamic)
   */
  chi::Future<GetOrCreateTagTask<CreateParams>> AsyncGetOrCreateTag(
      const std::string &tag_name,
      const TagId &tag_id = TagId::GetNull(),
      const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic()) {
    auto *ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<GetOrCreateTagTask<CreateParams>>(
        chi::CreateTaskId(), pool_id_, pool_query, tag_name,
        tag_id);

    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous put blob with optional compression context - returns immediately
   * @param tag_id Tag ID
   * @param blob_name Name of the blob
   * @param offset Offset within blob
   * @param size Size of data
   * @param blob_data Shared memory pointer to data
   * @param score Blob score for placement
   * @param context Compression context
   * @param flags Operation flags
   * @param pool_query Pool query for task routing (default: Dynamic)
   */
  chi::Future<PutBlobTask> AsyncPutBlob(
      const TagId &tag_id,
      const std::string &blob_name,
      chi::u64 offset, chi::u64 size,
      hipc::ShmPtr<> blob_data, float score,
      const Context &context = Context(),
      chi::u32 flags = 0,
      const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic()) {
    auto *ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<PutBlobTask>(
        chi::CreateTaskId(), pool_id_, pool_query, tag_id,
        blob_name, offset, size, blob_data, score, context, flags);

    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous get blob - returns immediately
   * @param tag_id Tag ID
   * @param blob_name Name of the blob
   * @param offset Offset within blob
   * @param size Size of data
   * @param flags Operation flags
   * @param blob_data Shared memory pointer for output
   * @param pool_query Pool query for task routing (default: Dynamic)
   */
  chi::Future<GetBlobTask> AsyncGetBlob(
      const TagId &tag_id,
      const std::string &blob_name,
      chi::u64 offset, chi::u64 size,
      chi::u32 flags,
      hipc::ShmPtr<> blob_data,
      const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic()) {
    auto *ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<GetBlobTask>(
        chi::CreateTaskId(), pool_id_, pool_query, tag_id,
        blob_name, offset, size, flags, blob_data);

    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous reorganize blob - returns immediately
   * @param tag_id Tag ID
   * @param blob_name Name of the blob
   * @param new_score New placement score
   * @param pool_query Pool query for task routing (default: Dynamic)
   */
  chi::Future<ReorganizeBlobTask> AsyncReorganizeBlob(
      const TagId &tag_id, const std::string &blob_name, float new_score,
      const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic()) {
    auto *ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<ReorganizeBlobTask>(
        chi::CreateTaskId(), pool_id_, pool_query, tag_id,
        blob_name, new_score);

    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous delete blob - returns immediately
   * @param tag_id Tag ID
   * @param blob_name Name of the blob
   * @param pool_query Pool query for task routing (default: Dynamic)
   */
  chi::Future<DelBlobTask> AsyncDelBlob(
      const TagId &tag_id,
      const std::string &blob_name,
      const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic()) {
    auto *ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<DelBlobTask>(chi::CreateTaskId(), pool_id_,
                                                  pool_query,
                                                  tag_id, blob_name);

    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous delete tag by tag ID - returns immediately
   * @param tag_id Tag ID to delete
   * @param pool_query Pool query for task routing (default: Dynamic)
   */
  chi::Future<DelTagTask> AsyncDelTag(
      const TagId &tag_id,
      const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic()) {
    auto *ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<DelTagTask>(
        chi::CreateTaskId(), pool_id_, pool_query, tag_id);

    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous delete tag by tag name - returns immediately
   * @param tag_name Tag name to delete
   * @param pool_query Pool query for task routing (default: Dynamic)
   */
  chi::Future<DelTagTask> AsyncDelTag(
      const std::string &tag_name,
      const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic()) {
    auto *ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<DelTagTask>(
        chi::CreateTaskId(), pool_id_, pool_query, tag_name);

    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous get tag size - returns immediately
   * @param tag_id Tag ID
   * @param pool_query Pool query for task routing (default: Dynamic)
   */
  chi::Future<GetTagSizeTask> AsyncGetTagSize(
      const TagId &tag_id,
      const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic()) {
    auto *ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<GetTagSizeTask>(
        chi::CreateTaskId(), pool_id_, pool_query, tag_id);

    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous poll telemetry log - returns immediately
   * @param minimum_logical_time Minimum logical time filter
   * @param pool_query Pool query for task routing (default: Dynamic)
   */
  chi::Future<PollTelemetryLogTask> AsyncPollTelemetryLog(
      std::uint64_t minimum_logical_time,
      const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic()) {
    auto *ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<PollTelemetryLogTask>(
        chi::CreateTaskId(), pool_id_, pool_query,
        minimum_logical_time);

    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous get blob score - returns immediately
   * @param tag_id Tag ID
   * @param blob_name Name of the blob
   * @param pool_query Pool query for task routing (default: Dynamic)
   */
  chi::Future<GetBlobScoreTask> AsyncGetBlobScore(
      const TagId &tag_id, const std::string &blob_name,
      const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic()) {
    auto *ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<GetBlobScoreTask>(
        chi::CreateTaskId(), pool_id_, pool_query, tag_id,
        blob_name);

    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous get blob size - returns immediately
   * @param tag_id Tag ID
   * @param blob_name Name of the blob
   * @param pool_query Pool query for task routing (default: Dynamic)
   */
  chi::Future<GetBlobSizeTask> AsyncGetBlobSize(
      const TagId &tag_id,
      const std::string &blob_name,
      const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic()) {
    auto *ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<GetBlobSizeTask>(
        chi::CreateTaskId(), pool_id_, pool_query, tag_id,
        blob_name);

    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous get contained blobs - returns immediately
   * @param tag_id Tag ID
   * @param pool_query Pool query for task routing (default: Dynamic)
   */
  chi::Future<GetContainedBlobsTask> AsyncGetContainedBlobs(
      const TagId &tag_id,
      const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic()) {
    auto *ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<GetContainedBlobsTask>(
        chi::CreateTaskId(), pool_id_, pool_query, tag_id);

    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous tag query - returns immediately
   * @param tag_regex Tag regex pattern to match
   * @param max_tags Maximum number of tags to return (0 = no limit)
   * @param pool_query Pool query for routing (default: Broadcast)
   * @return Future for async operation
   */
  chi::Future<TagQueryTask> AsyncTagQuery(
      const std::string &tag_regex, chi::u32 max_tags = 0,
      const chi::PoolQuery &pool_query = chi::PoolQuery::Broadcast()) {
    auto *ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<TagQueryTask>(
        chi::CreateTaskId(), pool_id_, pool_query, tag_regex, max_tags);

    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous blob query - returns immediately
   * @param tag_regex Tag regex pattern to match
   * @param blob_regex Blob regex pattern to match
   * @param max_blobs Maximum number of blobs to return (0 = no limit)
   * @param pool_query Pool query for routing (default: Broadcast)
   * @return Future for async operation
   */
  chi::Future<BlobQueryTask> AsyncBlobQuery(
      const std::string &tag_regex, const std::string &blob_regex,
      chi::u32 max_blobs = 0,
      const chi::PoolQuery &pool_query = chi::PoolQuery::Broadcast()) {
    auto *ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<BlobQueryTask>(
        chi::CreateTaskId(), pool_id_, pool_query, tag_regex, blob_regex,
        max_blobs);

    return ipc_manager->Send(task);
  }
};

// Global pointer-based singleton for CTE client with lazy initialization
HSHM_DEFINE_GLOBAL_PTR_VAR_H(wrp_cte::core::Client, g_cte_client);

/**
 * Initialize CTE client and configuration subsystem
 * @param config_path Optional path to configuration file
 * @param pool_query Pool query type for CTE container creation (default:
 * Dynamic)
 * @return true if initialization succeeded, false otherwise
 */
bool WRP_CTE_CLIENT_INIT(
    const std::string &config_path = "",
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
   * @param score Blob score for placement decisions (default 1.0)
   * @param context Compression context for workflow-aware decisions (default empty)
   */
  void PutBlob(const std::string &blob_name, const char *data, size_t data_size,
               size_t off = 0, float score = 1.0f, const Context &context = Context());

  /**
   * PutBlob (SHM) - Direct shared memory version
   * @param blob_name Name of the blob
   * @param data Shared memory pointer to data
   * @param data_size Size of data
   * @param off Offset within blob (default 0)
   * @param score Blob score for placement decisions (default 1.0)
   * @param context Compression context for workflow-aware decisions (default empty)
   */
  void PutBlob(const std::string &blob_name, const hipc::ShmPtr<> &data,
               size_t data_size, size_t off = 0, float score = 1.0f,
               const Context &context = Context());

  /**
   * Asynchronous PutBlob (SHM) - Caller must manage shared memory lifecycle
   * @param blob_name Name of the blob
   * @param data Shared memory pointer to data (must remain valid until task
   * completes)
   * @param data_size Size of data
   * @param off Offset within blob (default 0)
   * @param score Blob score for placement decisions (default 1.0)
   * @param context Compression context for workflow-aware decisions (default empty)
   * @return Task pointer for async operation
   * @note For raw data, caller must allocate shared memory using
   * CHI_IPC->AllocateBuffer<void>() and keep the FullPtr alive until the async
   * task completes
   */
  chi::Future<PutBlobTask> AsyncPutBlob(const std::string &blob_name,
                                        const hipc::ShmPtr<> &data,
                                        size_t data_size, size_t off = 0,
                                        float score = 1.0f,
                                        const Context &context = Context());

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
  void GetBlob(const std::string &blob_name, hipc::ShmPtr<> data,
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
   * Reorganize blob with new score for data placement optimization
   * @param blob_name Name of the blob to reorganize
   * @param new_score New placement score (0.0-1.0, higher = faster tier)
   */
  void ReorganizeBlob(const std::string &blob_name, float new_score);

  /**
   * Get the TagId for this tag
   * @return TagId of this tag
   */
  const TagId &GetTagId() const { return tag_id_; }
};

}  // namespace wrp_cte::core

// Global singleton macro for CTE client access (returns pointer, not reference)
#define WRP_CTE_CLIENT                               \
  (&(*HSHM_GET_GLOBAL_PTR_VAR(wrp_cte::core::Client, \
                              wrp_cte::core::g_cte_client)))

#endif  // WRPCTE_CORE_CLIENT_H_
