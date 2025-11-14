#ifndef WRPCTE_CORE_TASKS_H_
#define WRPCTE_CORE_TASKS_H_

#include <chimaera/chimaera.h>
#include <wrp_cte/core/autogen/core_methods.h>
#include <wrp_cte/core/core_config.h>
// Include admin tasks for GetOrCreatePoolTask
#include <chimaera/admin/admin_tasks.h>
// Include bdev tasks for BdevType enum
#include <chimaera/bdev/bdev_tasks.h>
// Include bdev client for TargetInfo
#include <chimaera/bdev/bdev_client.h>
#include <yaml-cpp/yaml.h>
#include <chrono>
// Include cereal for serialization
#include <cereal/cereal.hpp>
#include <cereal/archives/binary.hpp>

namespace wrp_cte::core {

// CTE Core Pool ID constant (major: 512, minor: 0)
static constexpr chi::PoolId kCtePoolId(512, 0);

// CTE Core Pool Name constant
static constexpr const char* kCtePoolName = "wrp_cte_core";

// Timestamp type definition
using Timestamp = std::chrono::time_point<std::chrono::steady_clock>;

/**
 * CreateParams for CTE Core chimod
 * Contains configuration parameters for CTE container creation
 */
struct CreateParams {
  // CTE configuration object (not serialized, loaded from pool_config)
  Config config_;

  // Required: chimod library name for module manager
  static constexpr const char *chimod_lib_name = "wrp_cte_core";

  // Default constructor
  CreateParams() {}

  // Constructor with allocator and parameters
  CreateParams(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc)
      : config_() {
    (void)alloc; // Suppress unused parameter warning
  }

  // Copy constructor with allocator (required for task creation)
  CreateParams(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc,
               const CreateParams &other)
      : config_(other.config_) {
    (void)alloc; // Suppress unused parameter warning
  }

  // Constructor with allocator, pool_id, and CreateParams (required for admin
  // task creation)
  CreateParams(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc,
               const chi::PoolId &pool_id, const CreateParams &other)
      : config_(other.config_) {
    // pool_id is used by the admin task framework, but we don't need to store it
    (void)pool_id; // Suppress unused parameter warning
    (void)alloc; // Suppress unused parameter warning
  }

  // Serialization support for cereal
  template <class Archive> void serialize(Archive &ar) {
    // Config is not serialized - it's loaded from pool_config.config_ in LoadConfig
    (void)ar;
  }

  /**
   * Load configuration from PoolConfig (for compose mode)
   * Required for compose feature support
   * @param pool_config Pool configuration from compose section
   */
  void LoadConfig(const chi::PoolConfig& pool_config) {
    // The pool_config.config_ contains the full CTE configuration YAML
    // in the format of config/cte_config.yaml (targets, storage, dpe sections).
    // Parse it directly into the Config object
    if (!pool_config.config_.empty()) {
      config_.LoadFromString(pool_config.config_);
    }
  }
};

/**
 * CreateTask - Initialize the CTE Core container
 * Type alias for GetOrCreatePoolTask with CreateParams (uses kGetOrCreatePool
 * method) Non-admin modules should use GetOrCreatePoolTask instead of
 * BaseCreateTask
 */
using CreateTask = chimaera::admin::GetOrCreatePoolTask<CreateParams>;

/**
 * DestroyTask - Destroy the CTE Core container
 * Type alias for DestroyPoolTask (uses kDestroy method)
 */
using DestroyTask = chimaera::admin::DestroyTask;

/**
 * Target information structure
 */
struct TargetInfo {
  std::string target_name_;
  std::string bdev_pool_name_;
  chimaera::bdev::Client bdev_client_; // Bdev client for this target
  chi::PoolQuery target_query_;        // Target pool query for bdev API calls
  chi::u64 bytes_read_;
  chi::u64 bytes_written_;
  chi::u64 ops_read_;
  chi::u64 ops_written_;
  float target_score_;       // Target score (0-1, normalized log bandwidth)
  chi::u64 remaining_space_; // Remaining allocatable space in bytes
  chimaera::bdev::PerfMetrics perf_metrics_; // Performance metrics from bdev

  TargetInfo() = default;

  explicit TargetInfo(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc)
      : bytes_read_(0), bytes_written_(0), ops_read_(0), ops_written_(0),
        target_score_(0.0f), remaining_space_(0) {
    // std::string doesn't need allocator, chi::u64 and float are POD types
    (void)alloc; // Suppress unused parameter warning
  }
};

/**
 * RegisterTarget task - Get/create bdev locally, create Target struct
 */
struct RegisterTargetTask : public chi::Task {
  // Task-specific data using HSHM macros
  IN hipc::string target_name_; // Name and file path of the target to register
  IN chimaera::bdev::BdevType bdev_type_; // Block device type enum
  IN chi::u64 total_size_;                // Total size for allocation
  IN chi::PoolQuery target_query_;        // Target pool query for bdev API calls
  IN chi::PoolId bdev_id_;                // PoolId to create for the underlying bdev

  // SHM constructor
  explicit RegisterTargetTask(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc)
      : chi::Task(alloc), target_name_(alloc),
        bdev_type_(chimaera::bdev::BdevType::kFile), total_size_(0),
        bdev_id_(chi::PoolId::GetNull()) {}

  // Emplace constructor
  explicit RegisterTargetTask(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc,
                              const chi::TaskId &task_id,
                              const chi::PoolId &pool_id,
                              const chi::PoolQuery &pool_query,
                              const std::string &target_name,
                              chimaera::bdev::BdevType bdev_type,
                              chi::u64 total_size,
                              const chi::PoolQuery &target_query,
                              const chi::PoolId &bdev_id)
      : chi::Task(alloc, task_id, pool_id, pool_query, Method::kRegisterTarget),
        target_name_(alloc, target_name), bdev_type_(bdev_type),
        total_size_(total_size), target_query_(target_query), bdev_id_(bdev_id) {
    task_id_ = task_id;
    pool_id_ = pool_id;
    method_ = Method::kRegisterTarget;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /**
   * Serialize IN and INOUT parameters
   */
  template <typename Archive> void SerializeIn(Archive &ar) {
    ar(target_name_, bdev_type_, total_size_, target_query_, bdev_id_);
  }

  /**
   * Serialize OUT and INOUT parameters
   */
  template <typename Archive> void SerializeOut(Archive &ar) {}

  /**
   * Copy from another RegisterTargetTask
   * Used when creating replicas for remote execution
   */
  void Copy(const hipc::FullPtr<RegisterTargetTask> &other) {
    // Copy RegisterTargetTask-specific fields
    target_name_ = other->target_name_;
    bdev_type_ = other->bdev_type_;
    total_size_ = other->total_size_;
    target_query_ = other->target_query_;
    bdev_id_ = other->bdev_id_;
  }
};

/**
 * UnregisterTarget task - Unlink bdev from container (don't destroy bdev
 * container)
 */
struct UnregisterTargetTask : public chi::Task {
  IN hipc::string target_name_; // Name of the target to unregister

  // SHM constructor
  explicit UnregisterTargetTask(
      const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc)
      : chi::Task(alloc), target_name_(alloc) {}

  // Emplace constructor
  explicit UnregisterTargetTask(
      const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc,
      const chi::TaskId &task_id, const chi::PoolId &pool_id,
      const chi::PoolQuery &pool_query, const std::string &target_name)
      : chi::Task(alloc, task_id, pool_id, pool_query,
                  Method::kUnregisterTarget),
        target_name_(alloc, target_name) {
    task_id_ = task_id;
    pool_id_ = pool_id;
    method_ = Method::kUnregisterTarget;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /**
   * Serialize IN and INOUT parameters
   */
  template <typename Archive> void SerializeIn(Archive &ar) {
    ar(target_name_);
  }

  /**
   * Serialize OUT and INOUT parameters
   */
  template <typename Archive> void SerializeOut(Archive &ar) {
    // No output parameters (return_code_ handled by base class)
  }

  /**
   * Copy from another UnregisterTargetTask
   */
  void Copy(const hipc::FullPtr<UnregisterTargetTask> &other) {
    target_name_ = other->target_name_;
  }
};

/**
 * ListTargets task - Return set of registered target names on this node
 */
struct ListTargetsTask : public chi::Task {
  OUT hipc::vector<hipc::string>
      target_names_; // List of registered target names

  // SHM constructor
  explicit ListTargetsTask(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc)
      : chi::Task(alloc), target_names_(alloc) {}

  // Emplace constructor
  explicit ListTargetsTask(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc,
                           const chi::TaskId &task_id,
                           const chi::PoolId &pool_id,
                           const chi::PoolQuery &pool_query)
      : chi::Task(alloc, task_id, pool_id, pool_query, Method::kListTargets),
        target_names_(alloc) {
    task_id_ = task_id;
    pool_id_ = pool_id;
    method_ = Method::kListTargets;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /**
   * Serialize IN and INOUT parameters
   */
  template <typename Archive> void SerializeIn(Archive &ar) {
    // No input parameters
  }

  /**
   * Serialize OUT and INOUT parameters
   */
  template <typename Archive> void SerializeOut(Archive &ar) {
    ar(target_names_);
  }

  /**
   * Copy from another ListTargetsTask
   */
  void Copy(const hipc::FullPtr<ListTargetsTask> &other) {
    target_names_ = other->target_names_;
  }

  /**
   * Aggregate entries from another ListTargetsTask
   * Appends all target names from the other task to this one
   */
  void Aggregate(const hipc::FullPtr<ListTargetsTask> &other) {
    for (const auto &target_name : other->target_names_) {
      target_names_.emplace_back(target_name);
    }
  }
};

/**
 * StatTargets task - Poll each target in vector, update performance stats
 */
struct StatTargetsTask : public chi::Task {
  // SHM constructor
  explicit StatTargetsTask(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc)
      : chi::Task(alloc) {}

  // Emplace constructor
  explicit StatTargetsTask(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc,
                           const chi::TaskId &task_id,
                           const chi::PoolId &pool_id,
                           const chi::PoolQuery &pool_query)
      : chi::Task(alloc, task_id, pool_id, pool_query, Method::kStatTargets) {
    task_id_ = task_id;
    pool_id_ = pool_id;
    method_ = Method::kStatTargets;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /**
   * Serialize IN and INOUT parameters
   */
  template <typename Archive> void SerializeIn(Archive &ar) {
    // No input parameters
  }

  /**
   * Serialize OUT and INOUT parameters
   */
  template <typename Archive> void SerializeOut(Archive &ar) {
    // No output parameters (return_code_ handled by base class)
  }

  /**
   * Copy from another StatTargetsTask
   */
  void Copy(const hipc::FullPtr<StatTargetsTask> &other) {
    // No task-specific fields to copy
    (void)other; // Suppress unused parameter warning
  }
};

/**
 * TagId type definition
 * Uses chi::UniqueId with node_id as major and atomic counter as minor
 */
using TagId = chi::UniqueId;

} // namespace wrp_cte::core

// Hash specialization for TagId (TagId uses same hash as chi::UniqueId)
namespace hshm {
template <> struct hash<wrp_cte::core::TagId> {
  std::size_t operator()(const wrp_cte::core::TagId &id) const {
    std::hash<chi::u32> hasher;
    return hasher(id.major_) ^ (hasher(id.minor_) << 1);
  }
};
} // namespace hshm

namespace wrp_cte::core {

/**
 * Tag information structure for blob grouping
 */
struct TagInfo {
  std::string tag_name_;
  TagId tag_id_;
  std::atomic<size_t> total_size_;       // Total size of all blobs in this tag
  Timestamp last_modified_; // Last modification time
  Timestamp last_read_;     // Last read time

  TagInfo()
      : tag_name_(), tag_id_(TagId::GetNull()), total_size_(0),
        last_modified_(std::chrono::steady_clock::now()),
        last_read_(std::chrono::steady_clock::now()) {}

  explicit TagInfo(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc)
      : tag_name_(), tag_id_(TagId::GetNull()), total_size_(0),
        last_modified_(std::chrono::steady_clock::now()),
        last_read_(std::chrono::steady_clock::now()) {
    (void)alloc; // Suppress unused parameter warning
  }

  TagInfo(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc,
          const std::string &tag_name, const TagId &tag_id)
      : tag_name_(tag_name), tag_id_(tag_id), total_size_(0),
        last_modified_(std::chrono::steady_clock::now()),
        last_read_(std::chrono::steady_clock::now()) {
    (void)alloc; // Suppress unused parameter warning
  }

  // Copy constructor
  TagInfo(const TagInfo &other)
      : tag_name_(other.tag_name_), tag_id_(other.tag_id_),
        total_size_(other.total_size_.load()),
        last_modified_(other.last_modified_),
        last_read_(other.last_read_) {}

  // Copy assignment operator
  TagInfo& operator=(const TagInfo &other) {
    if (this != &other) {
      tag_name_ = other.tag_name_;
      tag_id_ = other.tag_id_;
      total_size_.store(other.total_size_.load());
      last_modified_ = other.last_modified_;
      last_read_ = other.last_read_;
    }
    return *this;
  }
};

/**
 * Block structure for blob management
 * Each block represents a portion of a blob stored in a target
 */
struct BlobBlock {
  chimaera::bdev::Client bdev_client_; // Bdev client for this block's target
  chi::PoolQuery target_query_;        // Target pool query for bdev API calls
  chi::u64 target_offset_; // Offset within target where this block is stored
  chi::u64 size_;          // Size of this block in bytes

  BlobBlock() = default;

  BlobBlock(const chimaera::bdev::Client &client,
            const chi::PoolQuery &target_query, chi::u64 offset, chi::u64 size)
      : bdev_client_(client), target_query_(target_query),
        target_offset_(offset), size_(size) {}
};

/**
 * Blob information structure with block-based management
 */
struct BlobInfo {
  std::string blob_name_;
  std::vector<BlobBlock>
      blocks_;              // Vector of blocks that make up this blob (ordered)
  float score_;             // 0-1 score for reorganization
  Timestamp last_modified_; // Last modification time
  Timestamp last_read_;     // Last read time

  BlobInfo()
      : blob_name_(), blocks_(), score_(0.0f),
        last_modified_(std::chrono::steady_clock::now()),
        last_read_(std::chrono::steady_clock::now()) {}

  explicit BlobInfo(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc)
      : blob_name_(), blocks_(), score_(0.0f),
        last_modified_(std::chrono::steady_clock::now()),
        last_read_(std::chrono::steady_clock::now()) {
    (void)alloc; // Suppress unused parameter warning
  }

  BlobInfo(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc,
           const std::string &blob_name, float score)
      : blob_name_(blob_name), blocks_(), score_(score),
        last_modified_(std::chrono::steady_clock::now()),
        last_read_(std::chrono::steady_clock::now()) {
    (void)alloc; // Suppress unused parameter warning
  }

  /**
   * Get total size of blob by summing all block sizes
   */
  chi::u64 GetTotalSize() const {
    chi::u64 total = 0;
    for (size_t i = 0; i < blocks_.size(); ++i) {
      total += blocks_[i].size_;
    }
    return total;
  }
};

/**
 * CTE Operation types for telemetry
 */
enum class CteOp : chi::u32 {
  kPutBlob = 0,
  kGetBlob = 1,
  kDelBlob = 2,
  kGetOrCreateTag = 3,
  kDelTag = 4,
  kGetTagSize = 5
};

/**
 * CTE Telemetry data structure for performance monitoring
 */
struct CteTelemetry {
  CteOp op_;                   // Operation type
  size_t off_;                 // Offset within blob (for Put/Get operations)
  size_t size_;                // Size of operation (for Put/Get operations)
  TagId tag_id_;               // Tag ID involved
  Timestamp mod_time_;         // Last modification time
  Timestamp read_time_;        // Last read time
  std::uint64_t logical_time_; // Logical time for ordering telemetry entries

  CteTelemetry()
      : op_(CteOp::kPutBlob), off_(0), size_(0),
        tag_id_(TagId::GetNull()), mod_time_(std::chrono::steady_clock::now()),
        read_time_(std::chrono::steady_clock::now()), logical_time_(0) {}

  CteTelemetry(CteOp op, size_t off, size_t size,
               const TagId &tag_id, const Timestamp &mod_time,
               const Timestamp &read_time, std::uint64_t logical_time = 0)
      : op_(op), off_(off), size_(size), tag_id_(tag_id),
        mod_time_(mod_time), read_time_(read_time),
        logical_time_(logical_time) {}

  // Serialization support for cereal
  template <class Archive> void serialize(Archive &ar) {
    // Convert timestamps to duration counts for serialization
    auto mod_count = mod_time_.time_since_epoch().count();
    auto read_count = read_time_.time_since_epoch().count();
    ar(op_, off_, size_, tag_id_, mod_count, read_count,
       logical_time_);
    // Note: On deserialization, timestamps will be reconstructed from counts
    if (Archive::is_loading::value) {
      mod_time_ = Timestamp(Timestamp::duration(mod_count));
      read_time_ = Timestamp(Timestamp::duration(read_count));
    }
  }
};

/**
 * GetOrCreateTag task - Get or create a tag for blob grouping
 * Template parameter allows different CreateParams types
 */
template <typename CreateParamsT = CreateParams>
struct GetOrCreateTagTask : public chi::Task {
  IN hipc::string tag_name_; // Tag name (required)
  INOUT TagId tag_id_;       // Tag unique ID (default null, output on creation)

  // SHM constructor
  explicit GetOrCreateTagTask(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc)
      : chi::Task(alloc), tag_name_(alloc), tag_id_(TagId::GetNull()) {}

  // Emplace constructor
  explicit GetOrCreateTagTask(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc,
                              const chi::TaskId &task_id,
                              const chi::PoolId &pool_id,
                              const chi::PoolQuery &pool_query,
                              const std::string &tag_name,
                              const TagId &tag_id = TagId::GetNull())
      : chi::Task(alloc, task_id, pool_id, pool_query, Method::kGetOrCreateTag),
        tag_name_(alloc, tag_name), tag_id_(tag_id) {
    task_id_ = task_id;
    pool_id_ = pool_id;
    method_ = Method::kGetOrCreateTag;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /**
   * Serialize IN and INOUT parameters
   */
  template <typename Archive> void SerializeIn(Archive &ar) {
    ar(tag_name_, tag_id_);
  }

  /**
   * Serialize OUT and INOUT parameters
   */
  template <typename Archive> void SerializeOut(Archive &ar) {
    ar(tag_id_);
  }

  /**
   * Copy from another GetOrCreateTagTask
   */
  void Copy(const hipc::FullPtr<GetOrCreateTagTask> &other) {
    tag_name_ = other->tag_name_;
    tag_id_ = other->tag_id_;
  }
};

/**
 * PutBlob task - Store a blob (unimplemented for now)
 */
struct PutBlobTask : public chi::Task {
  IN TagId tag_id_;              // Tag ID for blob grouping
  INOUT hipc::string blob_name_; // Blob name (required)
  IN chi::u64 offset_;           // Offset within blob
  IN chi::u64 size_;             // Size of blob data
  IN hipc::Pointer blob_data_;   // Blob data (shared memory pointer)
  IN float score_;               // Score 0-1 for placement decisions
  IN chi::u32 flags_;            // Operation flags

  // SHM constructor
  explicit PutBlobTask(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc)
      : chi::Task(alloc), tag_id_(TagId::GetNull()), blob_name_(alloc),
        offset_(0), size_(0),
        blob_data_(hipc::Pointer::GetNull()), score_(0.5f), flags_(0) {}

  // Emplace constructor
  explicit PutBlobTask(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc,
                       const chi::TaskId &task_id, const chi::PoolId &pool_id,
                       const chi::PoolQuery &pool_query, const TagId &tag_id,
                       const std::string &blob_name,
                       chi::u64 offset, chi::u64 size, hipc::Pointer blob_data,
                       float score, chi::u32 flags)
      : chi::Task(alloc, task_id, pool_id, pool_query, Method::kPutBlob),
        tag_id_(tag_id), blob_name_(alloc, blob_name),
        offset_(offset), size_(size), blob_data_(blob_data), score_(score),
        flags_(flags) {
    task_id_ = task_id;
    pool_id_ = pool_id;
    method_ = Method::kPutBlob;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /**
   * Serialize IN and INOUT parameters
   */
  template <typename Archive> void SerializeIn(Archive &ar) {
    ar(tag_id_, blob_name_, offset_, size_, score_, flags_);
    // Use BULK_XFER to transfer blob data from client to runtime
    ar.bulk(blob_data_, size_, BULK_XFER);
  }

  /**
   * Serialize OUT and INOUT parameters
   */
  template <typename Archive> void SerializeOut(Archive &ar) {
    ar(blob_name_);
    // No bulk transfer needed for PutBlob output (metadata only)
  }

  /**
   * Copy from another PutBlobTask
   */
  void Copy(const hipc::FullPtr<PutBlobTask> &other) {
    tag_id_ = other->tag_id_;
    blob_name_ = other->blob_name_;
    offset_ = other->offset_;
    size_ = other->size_;
    blob_data_ = other->blob_data_;
    score_ = other->score_;
    flags_ = other->flags_;
  }
};

/**
 * GetBlob task - Retrieve a blob (unimplemented for now)
 */
struct GetBlobTask : public chi::Task {
  IN TagId tag_id_;              // Tag ID for blob lookup
  IN hipc::string blob_name_;    // Blob name (required)
  IN chi::u64 offset_;           // Offset within blob
  IN chi::u64 size_;             // Size of data to retrieve
  IN chi::u32 flags_;            // Operation flags
  IN hipc::Pointer
      blob_data_; // Input buffer for blob data (shared memory pointer)

  // SHM constructor
  explicit GetBlobTask(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc)
      : chi::Task(alloc), tag_id_(TagId::GetNull()), blob_name_(alloc),
        offset_(0), size_(0), flags_(0),
        blob_data_(hipc::Pointer::GetNull()) {}

  // Emplace constructor
  explicit GetBlobTask(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc,
                       const chi::TaskId &task_id, const chi::PoolId &pool_id,
                       const chi::PoolQuery &pool_query, const TagId &tag_id,
                       const std::string &blob_name,
                       chi::u64 offset, chi::u64 size, chi::u32 flags,
                       hipc::Pointer blob_data)
      : chi::Task(alloc, task_id, pool_id, pool_query, Method::kGetBlob),
        tag_id_(tag_id), blob_name_(alloc, blob_name),
        offset_(offset), size_(size), flags_(flags), blob_data_(blob_data) {
    task_id_ = task_id;
    pool_id_ = pool_id;
    method_ = Method::kGetBlob;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /**
   * Serialize IN and INOUT parameters
   */
  template <typename Archive> void SerializeIn(Archive &ar) {
    ar(tag_id_, blob_name_, offset_, size_, flags_);
    // Use BULK_EXPOSE - metadata only, runtime will allocate buffer for read
    // data
    ar.bulk(blob_data_, size_, BULK_EXPOSE);
  }

  /**
   * Serialize OUT and INOUT parameters
   */
  template <typename Archive> void SerializeOut(Archive &ar) {
    // Use BULK_XFER to transfer read data back to client
    ar.bulk(blob_data_, size_, BULK_XFER);
  }

  /**
   * Copy from another GetBlobTask
   */
  void Copy(const hipc::FullPtr<GetBlobTask> &other) {
    tag_id_ = other->tag_id_;
    blob_name_ = other->blob_name_;
    offset_ = other->offset_;
    size_ = other->size_;
    flags_ = other->flags_;
    blob_data_ = other->blob_data_;
  }
};

/**
 * ReorganizeBlob task - Change score for a single blob
 */
struct ReorganizeBlobTask : public chi::Task {
  IN TagId tag_id_;              // Tag ID containing blob
  IN hipc::string blob_name_;    // Blob name to reorganize
  IN float new_score_;           // New score for the blob (0-1)

  // SHM constructor
  explicit ReorganizeBlobTask(
      const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc)
      : chi::Task(alloc), tag_id_(TagId::GetNull()), blob_name_(alloc),
        new_score_(0.0f) {}

  // Emplace constructor
  explicit ReorganizeBlobTask(
      const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc,
      const chi::TaskId &task_id, const chi::PoolId &pool_id,
      const chi::PoolQuery &pool_query, const TagId &tag_id,
      const std::string &blob_name, float new_score)
      : chi::Task(alloc, task_id, pool_id, pool_query,
                  Method::kReorganizeBlob),
        tag_id_(tag_id), blob_name_(alloc, blob_name), new_score_(new_score) {
    task_id_ = task_id;
    pool_id_ = pool_id;
    method_ = Method::kReorganizeBlob;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /**
   * Serialize IN and INOUT parameters
   */
  template <typename Archive> void SerializeIn(Archive &ar) {
    ar(tag_id_, blob_name_, new_score_);
  }

  /**
   * Serialize OUT and INOUT parameters
   */
  template <typename Archive> void SerializeOut(Archive &ar) {
    // No output parameters (return_code_ handled by base class)
  }

  /**
   * Copy from another ReorganizeBlobTask
   */
  void Copy(const hipc::FullPtr<ReorganizeBlobTask> &other) {
    tag_id_ = other->tag_id_;
    blob_name_ = other->blob_name_;
    new_score_ = other->new_score_;
  }
};

/**
 * DelBlob task - Remove blob and decrement tag size
 */
struct DelBlobTask : public chi::Task {
  IN TagId tag_id_;           // Tag ID for blob lookup
  IN hipc::string blob_name_; // Blob name (required)

  // SHM constructor
  explicit DelBlobTask(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc)
      : chi::Task(alloc), tag_id_(TagId::GetNull()), blob_name_(alloc) {}

  // Emplace constructor
  explicit DelBlobTask(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc,
                       const chi::TaskId &task_id, const chi::PoolId &pool_id,
                       const chi::PoolQuery &pool_query, const TagId &tag_id,
                       const std::string &blob_name)
      : chi::Task(alloc, task_id, pool_id, pool_query, Method::kDelBlob),
        tag_id_(tag_id), blob_name_(alloc, blob_name) {
    task_id_ = task_id;
    pool_id_ = pool_id;
    method_ = Method::kDelBlob;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /**
   * Serialize IN and INOUT parameters
   */
  template <typename Archive> void SerializeIn(Archive &ar) {
    ar(tag_id_, blob_name_);
  }

  /**
   * Serialize OUT and INOUT parameters
   */
  template <typename Archive> void SerializeOut(Archive &ar) {
    // No output parameters (return_code_ handled by base class)
  }

  /**
   * Copy from another DelBlobTask
   */
  void Copy(const hipc::FullPtr<DelBlobTask> &other) {
    tag_id_ = other->tag_id_;
    blob_name_ = other->blob_name_;
  }
};

/**
 * DelTag task - Remove all blobs from tag and remove tag
 * Supports lookup by either tag ID or tag name
 */
struct DelTagTask : public chi::Task {
  INOUT TagId tag_id_;       // Tag ID to delete (input or lookup result)
  IN hipc::string tag_name_; // Tag name for lookup (optional)

  // SHM constructor
  explicit DelTagTask(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc)
      : chi::Task(alloc), tag_id_(TagId::GetNull()), tag_name_(alloc) {}

  // Emplace constructor with tag ID
  explicit DelTagTask(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc,
                      const chi::TaskId &task_id, const chi::PoolId &pool_id,
                      const chi::PoolQuery &pool_query, const TagId &tag_id)
      : chi::Task(alloc, task_id, pool_id, pool_query, Method::kDelTag),
        tag_id_(tag_id), tag_name_(alloc) {
    task_id_ = task_id;
    pool_id_ = pool_id;
    method_ = Method::kDelTag;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  // Emplace constructor with tag name
  explicit DelTagTask(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc,
                      const chi::TaskId &task_id, const chi::PoolId &pool_id,
                      const chi::PoolQuery &pool_query,
                      const std::string &tag_name)
      : chi::Task(alloc, task_id, pool_id, pool_query, Method::kDelTag),
        tag_id_(TagId::GetNull()), tag_name_(alloc, tag_name) {
    task_id_ = task_id;
    pool_id_ = pool_id;
    method_ = Method::kDelTag;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /**
   * Serialize IN and INOUT parameters
   */
  template <typename Archive> void SerializeIn(Archive &ar) {
    ar(tag_id_, tag_name_);
  }

  /**
   * Serialize OUT and INOUT parameters
   */
  template <typename Archive> void SerializeOut(Archive &ar) {
    ar(tag_id_);
  }

  /**
   * Copy from another DelTagTask
   */
  void Copy(const hipc::FullPtr<DelTagTask> &other) {
    tag_id_ = other->tag_id_;
    tag_name_ = other->tag_name_;
  }
};

/**
 * GetTagSize task - Get the total size of a tag
 */
struct GetTagSizeTask : public chi::Task {
  IN TagId tag_id_;     // Tag ID to query
  OUT size_t tag_size_; // Total size of all blobs in tag

  // SHM constructor
  explicit GetTagSizeTask(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc)
      : chi::Task(alloc), tag_id_(TagId::GetNull()), tag_size_(0) {}

  // Emplace constructor
  explicit GetTagSizeTask(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc,
                          const chi::TaskId &task_id,
                          const chi::PoolId &pool_id,
                          const chi::PoolQuery &pool_query, const TagId &tag_id)
      : chi::Task(alloc, task_id, pool_id, pool_query, Method::kGetTagSize),
        tag_id_(tag_id), tag_size_(0) {
    task_id_ = task_id;
    pool_id_ = pool_id;
    method_ = Method::kGetTagSize;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /**
   * Serialize IN and INOUT parameters
   */
  template <typename Archive> void SerializeIn(Archive &ar) { ar(tag_id_); }

  /**
   * Serialize OUT and INOUT parameters
   */
  template <typename Archive> void SerializeOut(Archive &ar) {
    ar(tag_size_);
  }

  /**
   * Copy from another GetTagSizeTask
   */
  void Copy(const hipc::FullPtr<GetTagSizeTask> &other) {
    tag_id_ = other->tag_id_;
    tag_size_ = other->tag_size_;
  }

  /**
   * Aggregate results from a replica task
   * Sums the tag_size_ values from multiple nodes
   */
  void Aggregate(const hipc::FullPtr<GetTagSizeTask> &replica) {
    tag_size_ += replica->tag_size_;
  }
};

/**
 * PollTelemetryLog task - Poll telemetry log with minimum logical time filter
 */
struct PollTelemetryLogTask : public chi::Task {
  IN std::uint64_t minimum_logical_time_;  // Minimum logical time filter
  OUT std::uint64_t last_logical_time_;    // Last logical time scanned
  OUT hipc::vector<CteTelemetry> entries_; // Retrieved telemetry entries

  // SHM constructor
  explicit PollTelemetryLogTask(
      const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc)
      : chi::Task(alloc), minimum_logical_time_(0), last_logical_time_(0),
        entries_(alloc) {}

  // Emplace constructor
  explicit PollTelemetryLogTask(
      const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc,
      const chi::TaskId &task_id, const chi::PoolId &pool_id,
      const chi::PoolQuery &pool_query, std::uint64_t minimum_logical_time)
      : chi::Task(alloc, task_id, pool_id, pool_query,
                  Method::kPollTelemetryLog),
        minimum_logical_time_(minimum_logical_time), last_logical_time_(0),
        entries_(alloc) {
    task_id_ = task_id;
    pool_id_ = pool_id;
    method_ = Method::kPollTelemetryLog;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /**
   * Serialize IN and INOUT parameters
   */
  template <typename Archive> void SerializeIn(Archive &ar) {
    ar(minimum_logical_time_);
  }

  /**
   * Serialize OUT and INOUT parameters
   */
  template <typename Archive> void SerializeOut(Archive &ar) {
    ar(last_logical_time_, entries_);
  }

  /**
   * Copy from another PollTelemetryLogTask
   */
  void Copy(const hipc::FullPtr<PollTelemetryLogTask> &other) {
    minimum_logical_time_ = other->minimum_logical_time_;
    last_logical_time_ = other->last_logical_time_;
    entries_ = other->entries_;
  }
};

/**
 * GetBlobScore task - Get the score of a blob
 */
struct GetBlobScoreTask : public chi::Task {
  IN TagId tag_id_;           // Tag ID for blob lookup
  IN hipc::string blob_name_; // Blob name (required)
  OUT float score_;           // Blob score (0-1)

  // SHM constructor
  explicit GetBlobScoreTask(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc)
      : chi::Task(alloc), tag_id_(TagId::GetNull()), blob_name_(alloc),
        score_(0.0f) {}

  // Emplace constructor
  explicit GetBlobScoreTask(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc,
                            const chi::TaskId &task_id,
                            const chi::PoolId &pool_id,
                            const chi::PoolQuery &pool_query,
                            const TagId &tag_id, const std::string &blob_name)
      : chi::Task(alloc, task_id, pool_id, pool_query, Method::kGetBlobScore),
        tag_id_(tag_id), blob_name_(alloc, blob_name),
        score_(0.0f) {
    task_id_ = task_id;
    pool_id_ = pool_id;
    method_ = Method::kGetBlobScore;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /**
   * Serialize IN and INOUT parameters
   */
  template <typename Archive> void SerializeIn(Archive &ar) {
    ar(tag_id_, blob_name_);
  }

  /**
   * Serialize OUT and INOUT parameters
   */
  template <typename Archive> void SerializeOut(Archive &ar) {
    ar(score_);
  }

  /**
   * Copy from another GetBlobScoreTask
   */
  void Copy(const hipc::FullPtr<GetBlobScoreTask> &other) {
    tag_id_ = other->tag_id_;
    blob_name_ = other->blob_name_;
    score_ = other->score_;
  }
};

/**
 * GetBlobSize task - Get the size of a blob
 */
struct GetBlobSizeTask : public chi::Task {
  IN TagId tag_id_;           // Tag ID for blob lookup
  IN hipc::string blob_name_; // Blob name (required)
  OUT chi::u64 size_;         // Blob size in bytes

  // SHM constructor
  explicit GetBlobSizeTask(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc)
      : chi::Task(alloc), tag_id_(TagId::GetNull()), blob_name_(alloc),
        size_(0) {}

  // Emplace constructor
  explicit GetBlobSizeTask(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc,
                           const chi::TaskId &task_id,
                           const chi::PoolId &pool_id,
                           const chi::PoolQuery &pool_query,
                           const TagId &tag_id, const std::string &blob_name)
      : chi::Task(alloc, task_id, pool_id, pool_query, Method::kGetBlobSize),
        tag_id_(tag_id), blob_name_(alloc, blob_name),
        size_(0) {
    task_id_ = task_id;
    pool_id_ = pool_id;
    method_ = Method::kGetBlobSize;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /**
   * Serialize IN and INOUT parameters
   */
  template <typename Archive> void SerializeIn(Archive &ar) {
    ar(tag_id_, blob_name_);
  }

  /**
   * Serialize OUT and INOUT parameters
   */
  template <typename Archive> void SerializeOut(Archive &ar) {
    ar(size_);
  }

  /**
   * Copy from another GetBlobSizeTask
   */
  void Copy(const hipc::FullPtr<GetBlobSizeTask> &other) {
    tag_id_ = other->tag_id_;
    blob_name_ = other->blob_name_;
    size_ = other->size_;
  }
};

/**
 * GetContainedBlobs task - Get all blob names contained in a tag
 */
struct GetContainedBlobsTask : public chi::Task {
  IN TagId tag_id_; // Tag ID to query
  OUT chi::ipc::vector<chi::ipc::string>
      blob_names_; // Vector of blob names in the tag

  // SHM constructor
  explicit GetContainedBlobsTask(
      const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc)
      : chi::Task(alloc), tag_id_(TagId::GetNull()), blob_names_(alloc) {}

  // Emplace constructor
  explicit GetContainedBlobsTask(
      const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc,
      const chi::TaskId &task_id, const chi::PoolId &pool_id,
      const chi::PoolQuery &pool_query, const TagId &tag_id)
      : chi::Task(alloc, task_id, pool_id, pool_query,
                  Method::kGetContainedBlobs),
        tag_id_(tag_id), blob_names_(alloc) {
    task_id_ = task_id;
    pool_id_ = pool_id;
    method_ = Method::kGetContainedBlobs;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /**
   * Serialize IN and INOUT parameters
   */
  template <typename Archive> void SerializeIn(Archive &ar) { ar(tag_id_); }

  /**
   * Serialize OUT and INOUT parameters
   */
  template <typename Archive> void SerializeOut(Archive &ar) {
    ar(blob_names_);
  }

  /**
   * Copy from another GetContainedBlobsTask
   */
  void Copy(const hipc::FullPtr<GetContainedBlobsTask> &other) {
    tag_id_ = other->tag_id_;
    blob_names_ = other->blob_names_;
  }

  /**
   * Aggregate results from a replica task
   * Merges the blob_names_ vectors from multiple nodes
   */
  void Aggregate(const hipc::FullPtr<GetContainedBlobsTask> &replica) {
    // Merge blob names from replica into this task's blob_names_
    for (size_t i = 0; i < replica->blob_names_.size(); ++i) {
      blob_names_.emplace_back(replica->blob_names_[i]);
    }
  }
};

/**
 * TagQuery task - Query tags by regex pattern
 * New behavior:
 * - Accepts an input maximum number of tags to store (max_tags_). 0 means no
 *   limit.
 * - Returns a vector of tag names matching the query.
 * - total_tags_matched_ sums the total number of tags that matched the
 *   pattern across replicas during Aggregate.
 */
struct TagQueryTask : public chi::Task {
  IN hipc::string tag_regex_;
  IN chi::u32 max_tags_;
  OUT chi::u64 total_tags_matched_;
  OUT hipc::vector<hipc::string> results_;

  // SHM constructor
  explicit TagQueryTask(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc)
      : chi::Task(alloc), tag_regex_(alloc), max_tags_(0),
        total_tags_matched_(0), results_(alloc) {}

  // Emplace constructor
  explicit TagQueryTask(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc,
                        const chi::TaskId &task_id,
                        const chi::PoolId &pool_id,
                        const chi::PoolQuery &pool_query,
                        const std::string &tag_regex,
                        chi::u32 max_tags = 0)
      : chi::Task(alloc, task_id, pool_id, pool_query, Method::kTagQuery),
        tag_regex_(alloc, tag_regex), max_tags_(max_tags),
        total_tags_matched_(0), results_(alloc) {
    task_id_ = task_id;
    pool_id_ = pool_id;
    method_ = Method::kTagQuery;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /**
   * Serialize IN and INOUT parameters
   */
  template <typename Archive> void SerializeIn(Archive &ar) {
    ar(tag_regex_, max_tags_);
  }

  /**
   * Serialize OUT and INOUT parameters
   */
  template <typename Archive> void SerializeOut(Archive &ar) {
    ar(total_tags_matched_, results_);
  }

  /**
   * Copy from another TagQueryTask
   */
  void Copy(const hipc::FullPtr<TagQueryTask> &other) {
    tag_regex_ = other->tag_regex_;
    max_tags_ = other->max_tags_;
    total_tags_matched_ = other->total_tags_matched_;
    results_ = other->results_;
  }

  /**
   * Aggregate results from multiple nodes
   */
  void Aggregate(const hipc::FullPtr<TagQueryTask> &other) {
    // Sum total matched tags across replicas
    total_tags_matched_ += other->total_tags_matched_;

    // Append results up to max_tags_ (if non-zero)
    for (const auto &tag_name : other->results_) {
      if (max_tags_ != 0 && results_.size() >= static_cast<size_t>(max_tags_))
        break;
      results_.emplace_back(tag_name);
    }
  }
};

/**
 * BlobQuery task - Query blobs by tag and blob regex patterns
 * New behavior:
 * - Accepts an input maximum number of blobs to store (max_blobs_). 0 means no
 *   limit.
 * - Returns a vector of pairs where each pair contains (tag_name, blob_name)
 *   for blobs matching the query.
 * - total_blobs_matched_ sums the total number of blobs that matched across
 *   replicas during Aggregate.
 */
struct BlobQueryTask : public chi::Task {
  IN hipc::string tag_regex_;
  IN hipc::string blob_regex_;
  IN chi::u32 max_blobs_;
  OUT chi::u64 total_blobs_matched_;
  OUT hipc::vector<hipc::pair<hipc::string, hipc::string>> results_;

  // SHM constructor
  explicit BlobQueryTask(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc)
      : chi::Task(alloc), tag_regex_(alloc), blob_regex_(alloc), max_blobs_(0),
        total_blobs_matched_(0), results_(alloc) {}

  // Emplace constructor
  explicit BlobQueryTask(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc,
                         const chi::TaskId &task_id,
                         const chi::PoolId &pool_id,
                         const chi::PoolQuery &pool_query,
                         const std::string &tag_regex,
                         const std::string &blob_regex,
                         chi::u32 max_blobs = 0)
      : chi::Task(alloc, task_id, pool_id, pool_query, Method::kBlobQuery),
        tag_regex_(alloc, tag_regex), blob_regex_(alloc, blob_regex),
        max_blobs_(max_blobs), total_blobs_matched_(0), results_(alloc) {
    task_id_ = task_id;
    pool_id_ = pool_id;
    method_ = Method::kBlobQuery;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /**
   * Serialize IN and INOUT parameters
   */
  template <typename Archive> void SerializeIn(Archive &ar) {
    ar(tag_regex_, blob_regex_, max_blobs_);
  }

  /**
   * Serialize OUT and INOUT parameters
   */
  template <typename Archive> void SerializeOut(Archive &ar) {
    ar(total_blobs_matched_, results_);
  }

  /**
   * Copy from another BlobQueryTask
   */
  void Copy(const hipc::FullPtr<BlobQueryTask> &other) {
    tag_regex_ = other->tag_regex_;
    blob_regex_ = other->blob_regex_;
    max_blobs_ = other->max_blobs_;
    total_blobs_matched_ = other->total_blobs_matched_;
    results_ = other->results_;
  }

  /**
   * Aggregate results from multiple nodes
   */
  void Aggregate(const hipc::FullPtr<BlobQueryTask> &other) {
    // Sum total matched blobs across replicas
    total_blobs_matched_ += other->total_blobs_matched_;

    // Append results up to max_blobs_ (if non-zero)
    for (const auto &pair : other->results_) {
      if (max_blobs_ != 0 && results_.size() >= static_cast<size_t>(max_blobs_))
        break;
      results_.emplace_back(pair);
    }
  }
};

} // namespace wrp_cte::core

// Cereal serialization support for hipc::pair
namespace cereal {

template <class Archive, typename FirstT, typename SecondT>
void serialize(Archive &ar, hipc::pair<FirstT, SecondT> &pair) {
  ar(pair.first_, pair.second_);
}

} // namespace cereal

#endif // WRPCTE_CORE_TASKS_H_