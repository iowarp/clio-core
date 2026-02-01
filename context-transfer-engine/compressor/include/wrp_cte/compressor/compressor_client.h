// Copyright 2024 IOWarp contributors
#ifndef WRP_CTE_COMPRESSOR_CLIENT_H_
#define WRP_CTE_COMPRESSOR_CLIENT_H_

#include <chimaera/chimaera.h>
#include <hermes_shm/util/singleton.h>
#include <wrp_cte/compressor/compressor_tasks.h>

namespace wrp_cte::compressor {

/**
 * Compressor client for asynchronous compression operations
 * Provides async API for DynamicSchedule, Compress, and Decompress tasks
 */
class Client : public chi::ContainerClient {
 public:
  Client() = default;
  explicit Client(const chi::PoolId &pool_id) { Init(pool_id); }

  /**
   * Create the compressor container
   * @param pool_query Task routing strategy
   * @param pool_name Name of the pool
   * @param custom_pool_id Explicit pool ID for the container
   * @return Future for CreateTask
   */
  chi::Future<CreateTask> AsyncCreate(const chi::PoolQuery& pool_query,
                                       const std::string& pool_name,
                                       const chi::PoolId& custom_pool_id) {
    auto* ipc_manager = CHI_IPC;
    auto task = ipc_manager->NewTask<CreateTask>(
        chi::CreateTaskId(),
        chi::kAdminPoolId,  // Always use admin pool for CreateTask
        pool_query,
        "wrp_cte_compressor",  // ChiMod library name
        pool_name,
        custom_pool_id,
        this);  // Client pointer for PostWait callback
    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous monitor - polls core for target information
   * @param core_pool_id Pool ID of core chimod to monitor
   * @param pool_query Pool query for task routing (default: Dynamic)
   * @return Future for MonitorTask
   */
  chi::Future<MonitorTask> AsyncMonitor(
      const chi::PoolId &core_pool_id,
      const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic()) {
    auto *ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<MonitorTask>(
        chi::CreateTaskId(), pool_id_, pool_query, core_pool_id);

    return ipc_manager->Send(task);
  }

  /**
   * Synchronous monitor - blocks until complete
   */
  void Monitor(
      const chi::PoolId &core_pool_id,
      const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic()) {
    auto future = AsyncMonitor(core_pool_id, pool_query);
    future.Wait();
  }

  /**
   * Asynchronous dynamic scheduling - analyzes data, compresses, and stores via PutBlob
   * Has same inputs as PutBlobTask for seamless integration
   * @param pool_query Pool query for task routing
   * @param tag_id Tag ID for blob grouping
   * @param blob_name Blob name
   * @param offset Offset within blob
   * @param size Size of blob data
   * @param blob_data Blob data (shared memory pointer)
   * @param score Score 0-1 for placement decisions
   * @param context Compression context (updated with predictions)
   * @param flags Operation flags
   * @param core_pool_id Pool ID of core chimod for PutBlob
   * @return Future for DynamicScheduleTask
   */
  chi::Future<DynamicScheduleTask> AsyncDynamicSchedule(
      const chi::PoolQuery &pool_query,
      const wrp_cte::core::TagId &tag_id,
      const std::string &blob_name,
      chi::u64 offset, chi::u64 size,
      hipc::ShmPtr<> blob_data,
      float score, const Context &context,
      chi::u32 flags,
      const chi::PoolId &core_pool_id) {
    auto *ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<DynamicScheduleTask>(
        chi::CreateTaskId(), pool_id_, pool_query,
        tag_id, blob_name, offset, size, blob_data,
        score, context, flags, core_pool_id);

    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous compression - compresses data and stores via PutBlob
   * Has same inputs as PutBlobTask for seamless integration
   * @param pool_query Pool query for task routing
   * @param tag_id Tag ID for blob grouping
   * @param blob_name Blob name
   * @param offset Offset within blob
   * @param size Size of blob data
   * @param blob_data Blob data (shared memory pointer)
   * @param score Score 0-1 for placement decisions
   * @param context Compression context (library, preset, etc.)
   * @param flags Operation flags
   * @param core_pool_id Pool ID of core chimod for PutBlob
   * @return Future for CompressTask
   */
  chi::Future<CompressTask> AsyncCompress(
      const chi::PoolQuery &pool_query,
      const wrp_cte::core::TagId &tag_id,
      const std::string &blob_name,
      chi::u64 offset, chi::u64 size,
      hipc::ShmPtr<> blob_data,
      float score, const Context &context,
      chi::u32 flags,
      const chi::PoolId &core_pool_id) {
    auto *ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<CompressTask>(
        chi::CreateTaskId(), pool_id_, pool_query,
        tag_id, blob_name, offset, size, blob_data,
        score, context, flags, core_pool_id);

    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous decompression - retrieves via GetBlob and decompresses
   * Has same inputs as GetBlobTask for seamless integration
   * @param pool_query Pool query for task routing
   * @param tag_id Tag ID for blob lookup
   * @param blob_name Blob name
   * @param offset Offset within blob
   * @param size Size of decompressed data to retrieve
   * @param flags Operation flags
   * @param blob_data Output buffer for decompressed data
   * @param core_pool_id Pool ID of core chimod for GetBlob
   * @return Future for DecompressTask
   */
  chi::Future<DecompressTask> AsyncDecompress(
      const chi::PoolQuery &pool_query,
      const wrp_cte::core::TagId &tag_id,
      const std::string &blob_name,
      chi::u64 offset, chi::u64 size,
      chi::u32 flags, hipc::ShmPtr<> blob_data,
      const chi::PoolId &core_pool_id) {
    auto *ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<DecompressTask>(
        chi::CreateTaskId(), pool_id_, pool_query,
        tag_id, blob_name, offset, size, flags, blob_data, core_pool_id);

    return ipc_manager->Send(task);
  }
};

}  // namespace wrp_cte::compressor

#endif  // WRP_CTE_COMPRESSOR_CLIENT_H_
