#ifndef WRPCTE_CORE_RUNTIME_H_
#define WRPCTE_CORE_RUNTIME_H_

#include <atomic>
#include <chimaera/chimaera.h>
#include <chimaera/comutex.h>
#include <chimaera/corwlock.h>
#include <chimaera/unordered_map_ll.h>
#include <hermes_shm/data_structures/ipc/ring_queue.h>
#include <wrp_cte/core/core_client.h>
#include <wrp_cte/core/core_config.h>
#include <wrp_cte/core/core_tasks.h>

// Forward declarations to avoid circular dependency
namespace wrp_cte::core {
class Config;
}

namespace wrp_cte::core {

/**
 * CTE Core Runtime Container
 * Implements target management and tag/blob operations
 */
class Runtime : public chi::Container {
public:
  using CreateParams = wrp_cte::core::CreateParams; // Required for CHI_TASK_CC

  Runtime() = default;
  ~Runtime() override = default;

  /**
   * Create the container (Method::kCreate)
   * This method both creates and initializes the container
   */
  void Create(hipc::FullPtr<CreateTask> task, chi::RunContext &ctx);

  /**
   * Destroy the container (Method::kDestroy)
   */
  void Destroy(hipc::FullPtr<DestroyTask> task, chi::RunContext &ctx);

  /**
   * Register a target (Method::kRegisterTarget)
   */
  void RegisterTarget(hipc::FullPtr<RegisterTargetTask> task,
                      chi::RunContext &ctx);

  /**
   * Unregister a target (Method::kUnregisterTarget)
   */
  void UnregisterTarget(hipc::FullPtr<UnregisterTargetTask> task,
                        chi::RunContext &ctx);

  /**
   * List registered targets (Method::kListTargets)
   */
  void ListTargets(hipc::FullPtr<ListTargetsTask> task, chi::RunContext &ctx);

  /**
   * Update target statistics (Method::kStatTargets)
   */
  void StatTargets(hipc::FullPtr<StatTargetsTask> task, chi::RunContext &ctx);

  /**
   * Get or create a tag (Method::kGetOrCreateTag)
   */
  template <typename CreateParamsT = CreateParams>
  void GetOrCreateTag(hipc::FullPtr<GetOrCreateTagTask<CreateParamsT>> task,
                      chi::RunContext &ctx);

  /**
   * Put blob (Method::kPutBlob) - unimplemented for now
   */
  void PutBlob(hipc::FullPtr<PutBlobTask> task, chi::RunContext &ctx);

  /**
   * Get blob (Method::kGetBlob) - unimplemented for now
   */
  void GetBlob(hipc::FullPtr<GetBlobTask> task, chi::RunContext &ctx);

  /**
   * Reorganize single blob (Method::kReorganizeBlob) - update score for single
   * blob
   */
  void ReorganizeBlob(hipc::FullPtr<ReorganizeBlobTask> task,
                      chi::RunContext &ctx);

  /**
   * Delete blob operation - removes blob and decrements tag size
   */
  void DelBlob(hipc::FullPtr<DelBlobTask> task, chi::RunContext &ctx);

  /**
   * Delete tag operation - removes all blobs from tag and removes tag
   */
  void DelTag(hipc::FullPtr<DelTagTask> task, chi::RunContext &ctx);

  /**
   * Get tag size operation - returns total size of all blobs in tag
   */
  void GetTagSize(hipc::FullPtr<GetTagSizeTask> task, chi::RunContext &ctx);

  // Pure virtual methods - implementations are in autogen/core_lib_exec.cc
  void Init(const chi::PoolId &pool_id, const std::string &pool_name,
            chi::u32 container_id = 0) override;
  void Run(chi::u32 method, hipc::FullPtr<chi::Task> task_ptr,
           chi::RunContext &rctx) override;
  void Monitor(chi::MonitorModeId mode, chi::u32 method,
               hipc::FullPtr<chi::Task> task_ptr, chi::RunContext &rctx);
  void Del(chi::u32 method, hipc::FullPtr<chi::Task> task_ptr) override;
  chi::u64 GetWorkRemaining() const override;
  void SaveTask(chi::u32 method, chi::SaveTaskArchive &archive,
                hipc::FullPtr<chi::Task> task_ptr) override;
  void LoadTask(chi::u32 method, chi::LoadTaskArchive &archive,
                hipc::FullPtr<chi::Task> &task_ptr) override;
  void NewCopy(chi::u32 method, const hipc::FullPtr<chi::Task> &orig_task,
               hipc::FullPtr<chi::Task> &dup_task, bool deep) override;
  void Aggregate(chi::u32 method, hipc::FullPtr<chi::Task> origin_task,
                 hipc::FullPtr<chi::Task> replica_task) override;

private:
  // Queue ID constants (REQUIRED: Use semantic names, not raw integers)
  static const chi::QueueId kTargetManagementQueue = 0;
  static const chi::QueueId kTagManagementQueue = 1;
  static const chi::QueueId kBlobOperationsQueue = 2;
  static const chi::QueueId kStatsQueue = 3;

  // Client for this ChiMod
  Client client_;

  // Target management data structures (using chi::unordered_map_ll for
  // thread-safe concurrent access)
  chi::unordered_map_ll<chi::PoolId, TargetInfo> registered_targets_;
  chi::unordered_map_ll<std::string, chi::PoolId>
      target_name_to_id_; // reverse lookup: target_name -> target_id

  // Tag management data structures (using chi::unordered_map_ll for thread-safe
  // concurrent access)
  chi::unordered_map_ll<std::string, TagId>
      tag_name_to_id_;                                   // tag_name -> tag_id
  chi::unordered_map_ll<TagId, TagInfo> tag_id_to_info_; // tag_id -> TagInfo
  chi::unordered_map_ll<std::string, BlobInfo>
      tag_blob_name_to_info_; // "tag_id.blob_name" -> BlobInfo

  // Atomic counters for thread-safe ID generation
  std::atomic<chi::u32>
      next_tag_id_minor_; // Minor counter for TagId UniqueId generation

  // Synchronization primitives for thread-safe access to data structures
  // Use a set of locks based on maximum number of lanes for better concurrency
  static const size_t kMaxLocks =
      64; // Maximum number of locks (matches max lanes)
  std::vector<std::unique_ptr<chi::CoRwLock>>
      target_locks_; // For registered_targets_
  std::vector<std::unique_ptr<chi::CoRwLock>>
      tag_locks_; // For tag management structures

  // Storage configuration (parsed from config file)
  std::vector<StorageDeviceConfig> storage_devices_;

  // CTE configuration (replaces ConfigManager singleton)
  Config config_;

  // Telemetry ring buffer for performance monitoring
  static const size_t kTelemetryRingSize = 1024; // Ring buffer size
  hipc::circular_mpsc_queue<CteTelemetry> telemetry_log_;
  std::atomic<std::uint64_t>
      telemetry_counter_; // Atomic counter for logical time

  /**
   * Get access to configuration manager
   */
  const Config &GetConfig() const;

  /**
   * Helper function to update target performance statistics
   */
  void UpdateTargetStats(const chi::PoolId &target_id, TargetInfo &target_info);

  /**
   * Helper function to get manual score for a target from storage device config
   * @param target_name Name of the target to look up
   * @return Manual score (0.0-1.0) if configured, -1.0f if not set (use
   * automatic)
   */
  float GetManualScoreForTarget(const std::string &target_name);

  /**
   * Helper function to get or assign a tag ID
   */
  TagId GetOrAssignTagId(const std::string &tag_name,
                         const TagId &preferred_id = TagId::GetNull());

  /**
   * Helper function to generate a new TagId using node_id as major and atomic
   * counter as minor
   */
  TagId GenerateNewTagId();

  /**
   * Get target lock index based on TargetId hash
   */
  size_t GetTargetLockIndex(const chi::PoolId &target_id) const;

  /**
   * Get tag lock index based on tag name hash
   */
  size_t GetTagLockIndex(const std::string &tag_name) const;

  /**
   * Get tag lock index based on tag ID hash
   */
  size_t GetTagLockIndex(const TagId &tag_id) const;

  /**
   * Allocate space from a target for new blob data
   * @param target_info Target to allocate from
   * @param size Size to allocate
   * @param allocated_offset Output parameter for allocated offset
   * @return True if allocation succeeded, false otherwise
   */
  bool AllocateFromTarget(TargetInfo &target_info, chi::u64 size,
                          chi::u64 &allocated_offset);

  /**
   * Free all blocks from a blob back to their respective targets
   * @param blob_info BlobInfo containing blocks to free
   * @return 0 on success, non-zero on error
   */
  chi::u32 FreeAllBlobBlocks(BlobInfo &blob_info);

  /**
   * Check if blob exists and return pointer to BlobInfo if found
   * @param blob_name Blob name to search for (required)
   * @param tag_id Tag ID to search within
   * @return Pointer to BlobInfo if found, nullptr if not found
   */
  BlobInfo *CheckBlobExists(const std::string &blob_name, const TagId &tag_id);

  /**
   * Create new blob with given parameters
   * @param blob_name Name for the new blob (required)
   * @param tag_id Tag ID to associate blob with
   * @param blob_score Score/priority for the blob
   * @return Pointer to created BlobInfo, nullptr on failure
   */
  BlobInfo *CreateNewBlob(const std::string &blob_name, const TagId &tag_id,
                          float blob_score);

  /**
   * Allocate new data blocks for blob expansion
   * @param blob_info Blob to extend with new data blocks
   * @param offset Offset where data starts (for determining required size)
   * @param size Size of data to accommodate
   * @param blob_score Score for target selection
   * @return Error code: 0 for success, 1 for failure
   */
  chi::u32 AllocateNewData(BlobInfo &blob_info, chi::u64 offset, chi::u64 size,
                           float blob_score);

  /**
   * Write data to existing blob blocks
   * @param blob_info Blob containing the blocks to write to
   * @param offset Offset within blob where data starts
   * @param size Size of data to write
   * @param blob_data Pointer to data to write
   * @return Error code: 0 for success, 1 for failure
   */
  chi::u32 ModifyExistingData(const std::vector<BlobBlock> &blocks,
                              hipc::Pointer data, size_t data_size,
                              size_t data_offset_in_blob);

  /**
   * Read existing blob data from blocks
   * @param blocks Vector of blob blocks to read from
   * @param data Output buffer to read data into
   * @param data_size Size of data to read
   * @param data_offset_in_blob Offset within blob where reading starts
   * @return Error code: 0 for success, 1 for failure
   */
  chi::u32 ReadData(const std::vector<BlobBlock> &blocks, hipc::Pointer data,
                    size_t data_size, size_t data_offset_in_blob);

  /**
   * Log telemetry data for CTE operations
   * @param op Operation type
   * @param off Offset within blob
   * @param size Size of operation
   * @param tag_id Tag ID involved
   * @param mod_time Last modification time
   * @param read_time Last read time
   */
  void LogTelemetry(CteOp op, size_t off, size_t size, const TagId &tag_id,
                    const Timestamp &mod_time, const Timestamp &read_time);

  /**
   * Get telemetry queue size for monitoring
   * @return Current number of entries in telemetry queue
   */
  size_t GetTelemetryQueueSize();

  /**
   * Parse capacity string to bytes
   * @param capacity_str Capacity string (e.g., "1TB", "500GB", "100MB")
   * @return Capacity in bytes
   */
  chi::u64 ParseCapacityToBytes(const std::string &capacity_str);

  /**
   * Retrieve telemetry entries for analysis (non-destructive peek)
   * @param entries Vector to store retrieved entries
   * @param max_entries Maximum number of entries to retrieve
   * @return Number of entries actually retrieved
   */
  size_t GetTelemetryEntries(std::vector<CteTelemetry> &entries,
                             size_t max_entries = 100);

  /**
   * Poll telemetry log (Method::kPollTelemetryLog)
   * @param task PollTelemetryLog task containing parameters and results
   * @param ctx Runtime context for task execution
   */
  void PollTelemetryLog(hipc::FullPtr<PollTelemetryLogTask> task,
                        chi::RunContext &ctx);

  /**
   * Get blob score operation - returns the score of a blob
   * @param task GetBlobScore task containing blob lookup parameters and results
   * @param ctx Runtime context for task execution
   */
  void GetBlobScore(hipc::FullPtr<GetBlobScoreTask> task, chi::RunContext &ctx);

  /**
   * Get blob size operation - returns the size of a blob in bytes
   * @param task GetBlobSize task containing blob lookup parameters and results
   * @param ctx Runtime context for task execution
   */
  void GetBlobSize(hipc::FullPtr<GetBlobSizeTask> task, chi::RunContext &ctx);

  /**
   * Get contained blobs operation - returns all blob names in a tag
   * @param task GetContainedBlobs task containing tag ID and results
   * @param ctx Runtime context for task execution
   */
  void GetContainedBlobs(hipc::FullPtr<GetContainedBlobsTask> task,
                         chi::RunContext &ctx);

  /**
   * Query tags by regex pattern (Method::kTagQuery)
   * @param task TagQuery task containing regex pattern and results
   * @param ctx Runtime context for task execution
   */
  void TagQuery(hipc::FullPtr<TagQueryTask> task, chi::RunContext &ctx);

  /**
   * Query blobs by tag and blob regex patterns (Method::kBlobQuery)
   * @param task BlobQuery task containing regex patterns and results
   * @param ctx Runtime context for task execution
   */
  void BlobQuery(hipc::FullPtr<BlobQueryTask> task, chi::RunContext &ctx);

private:
  /**
   * Helper function to compute hash-based pool query for blob operations
   * @param tag_id Tag ID for the blob
   * @param blob_name Blob name
   * @return PoolQuery with DirectHash based on tag_id and blob_name
   */
  chi::PoolQuery HashBlobToContainer(const TagId &tag_id,
                                     const std::string &blob_name);
};

} // namespace wrp_cte::core

#endif // WRPCTE_CORE_RUNTIME_H_