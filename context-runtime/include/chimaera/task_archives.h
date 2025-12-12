#ifndef CHIMAERA_INCLUDE_CHIMAERA_TASK_ARCHIVES_H_
#define CHIMAERA_INCLUDE_CHIMAERA_TASK_ARCHIVES_H_

#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

// Include cereal for serialization
#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>

// Include Lightbeam for networking
#include <hermes_shm/lightbeam/zmq_transport.h>

#include "chimaera/types.h"

namespace chi {

// Forward declaration
class Task;

/**
 * Data transfer object for handling bulk data transfers
 * Stores hipc::FullPtr for network transfer with pointer serialization
 */
struct DataTransfer {
  hipc::FullPtr<char> data; /**< Full pointer for transfer */
  size_t size;              /**< Size of data */
  uint32_t flags;           /**< Transfer flags (BULK_XFER, BULK_EXPOSE) */

  DataTransfer() : size(0), flags(0) {}
  DataTransfer(const hipc::FullPtr<char> &d, size_t s, uint32_t f)
      : data(d), size(s), flags(f) {}

  // Serialization support for cereal
  template <class Archive> void serialize(Archive &ar) {
    ar(data, size, flags);
    // FullPtr can be serialized - it stores allocator ID and offset
  }
};

/**
 * Bulk transfer metadata for handling large data transfers (deprecated - use
 * hshm::lbm::Bulk) Kept for compatibility with old
 * TaskSaveInArchive/TaskLoadInArchive/TaskSaveOutArchive/TaskLoadOutArchive
 */
struct BulkTransferInfo {
  hipc::ShmPtr<> ptr; /**< Pointer to bulk data */
  size_t size;       /**< Size of bulk data */
  uint32_t flags;    /**< Transfer flags (BULK_XFER, BULK_EXPOSE) */

  BulkTransferInfo() : size(0), flags(0) {}
  BulkTransferInfo(hipc::ShmPtr<> p, size_t s, uint32_t f)
      : ptr(p), size(s), flags(f) {}
};

/**
 * Archive for deserializing task inputs (uses cereal::BinaryInputArchive)
 * Used when receiving tasks from remote nodes for execution
 */
class TaskLoadInArchive {
private:
  std::string data_;
  std::unique_ptr<std::istringstream> stream_;
  std::unique_ptr<cereal::BinaryInputArchive> archive_;
  std::vector<DataTransfer> data_transfers_;
  size_t task_count_;

public:
  /** Constructor from serialized data */
  explicit TaskLoadInArchive(const std::string &data)
      : data_(data), stream_(std::make_unique<std::istringstream>(data_)),
        archive_(std::make_unique<cereal::BinaryInputArchive>(*stream_)),
        task_count_(0) {}

  /** Constructor from const char* and size */
  TaskLoadInArchive(const char *data, size_t size)
      : data_(data, size), stream_(std::make_unique<std::istringstream>(data_)),
        archive_(std::make_unique<cereal::BinaryInputArchive>(*stream_)),
        task_count_(0) {}

  /** Default constructor */
  TaskLoadInArchive()
      : stream_(std::make_unique<std::istringstream>("")),
        archive_(std::make_unique<cereal::BinaryInputArchive>(*stream_)),
        task_count_(0) {}

  /** Move constructor */
  TaskLoadInArchive(TaskLoadInArchive &&other) noexcept
      : data_(std::move(other.data_)), stream_(std::move(other.stream_)),
        archive_(std::move(other.archive_)),
        data_transfers_(std::move(other.data_transfers_)),
        task_count_(other.task_count_) {}

  /** Move assignment operator */
  TaskLoadInArchive &operator=(TaskLoadInArchive &&other) noexcept {
    if (this != &other) {
      data_ = std::move(other.data_);
      stream_ = std::move(other.stream_);
      archive_ = std::move(other.archive_);
      data_transfers_ = std::move(other.data_transfers_);
      task_count_ = other.task_count_;
    }
    return *this;
  }

  /** Delete copy constructor and assignment */
  TaskLoadInArchive(const TaskLoadInArchive &) = delete;
  TaskLoadInArchive &operator=(const TaskLoadInArchive &) = delete;

  /** Deserialize operator for input archives */
  template <typename T> TaskLoadInArchive &operator>>(T &value) {
    if constexpr (std::is_base_of_v<Task, T>) {
      // Automatically call BaseSerializeIn + SerializeIn for Task-derived
      // objects
      value.BaseSerializeIn(*this);
      value.SerializeIn(*this);
    } else {
      (*archive_)(value);
    }
    return *this;
  }

  /** Bidirectional serialization - acts as input for this archive type */
  template <typename... Args> void operator()(Args &...args) {
    ((*this >> args), ...);
  }

  /** Bulk transfer support - records transfer for later allocation */
  void bulk(hipc::ShmPtr<> &ptr, size_t size, uint32_t flags);

  /** Get data transfer information */
  const std::vector<DataTransfer> &GetDataTransfers() const {
    return data_transfers_;
  }

  /** Get task count */
  size_t GetTaskCount() const { return task_count_; }

  /** Load archive from BuildMessage() output */
  void LoadFromMessage(const std::string &message) {
    // Deserialize the message using cereal
    std::istringstream msg_stream(message);
    cereal::BinaryInputArchive msg_archive(msg_stream);

    // Deserialize components: task_count, archive_data, data_transfers
    std::string archive_data;
    msg_archive(task_count_);
    msg_archive(archive_data);
    msg_archive(data_transfers_);

    // Reconstruct the internal archive from the archive_data
    data_ = archive_data;
    stream_ = std::make_unique<std::istringstream>(data_);
    archive_ = std::make_unique<cereal::BinaryInputArchive>(*stream_);
  }

  /** Access underlying cereal archive */
  cereal::BinaryInputArchive &GetArchive() { return *archive_; }
};

/**
 * Archive for serializing task inputs (uses cereal::BinaryOutputArchive)
 * Used when sending tasks to remote nodes for execution
 */
class TaskSaveInArchive {
private:
  std::ostringstream stream_;
  std::unique_ptr<cereal::BinaryOutputArchive> archive_;
  std::vector<DataTransfer> data_transfers_;
  size_t task_count_;

public:
  /** Constructor with task count - serializes task count first */
  explicit TaskSaveInArchive(size_t task_count)
      : archive_(std::make_unique<cereal::BinaryOutputArchive>(stream_)),
        task_count_(task_count) {
    // Serialize task count first
    (*archive_)(task_count_);
  }

  /** Move constructor */
  TaskSaveInArchive(TaskSaveInArchive &&other) noexcept
      : stream_(std::move(other.stream_)), archive_(std::move(other.archive_)),
        data_transfers_(std::move(other.data_transfers_)),
        task_count_(other.task_count_) {}

  /** Move assignment operator */
  TaskSaveInArchive &operator=(TaskSaveInArchive &&other) noexcept {
    if (this != &other) {
      stream_ = std::move(other.stream_);
      archive_ = std::move(other.archive_);
      data_transfers_ = std::move(other.data_transfers_);
      task_count_ = other.task_count_;
    }
    return *this;
  }

  /** Delete copy constructor and assignment */
  TaskSaveInArchive(const TaskSaveInArchive &) = delete;
  TaskSaveInArchive &operator=(const TaskSaveInArchive &) = delete;

  /** Serialize operator for output archives */
  template <typename T> TaskSaveInArchive &operator<<(const T &value) {
    if constexpr (std::is_base_of_v<Task, T>) {
      // Automatically call BaseSerializeIn + SerializeIn for Task-derived
      // objects
      const_cast<T &>(value).BaseSerializeIn(*this);
      const_cast<T &>(value).SerializeIn(*this);
    } else {
      (*archive_)(value);
    }
    return *this;
  }

  /** Bidirectional serialization - acts as output for this archive type */
  template <typename... Args> void operator()(const Args &...args) {
    ((*this << args), ...);
  }

  /** Bulk transfer support */
  void bulk(hipc::ShmPtr<> ptr, size_t size, uint32_t flags);

  /** Get serialized data */
  std::string GetData() const { return stream_.str(); }

  /** Get data transfer information */
  const std::vector<DataTransfer> &GetDataTransfers() const {
    return data_transfers_;
  }

  /** Get task count */
  size_t GetTaskCount() const { return task_count_; }

  /** Build message for network transfer */
  std::string BuildMessage() {
    // First, finalize the main archive by ensuring all data is flushed
    archive_.reset(); // Flush and close the archive

    // Create a new stream for the complete message
    std::ostringstream msg_stream;
    cereal::BinaryOutputArchive msg_archive(msg_stream);

    // Serialize the components:
    // 1. Task count
    msg_archive(task_count_);

    // 2. Binary archive data (as string)
    std::string archive_data = stream_.str();
    msg_archive(archive_data);

    // 3. DataTransfer vector
    msg_archive(data_transfers_);

    return msg_stream.str();
  }

  /** Access underlying cereal archive */
  cereal::BinaryOutputArchive &GetArchive() { return *archive_; }
};

/**
 * Archive for deserializing task outputs (uses cereal::BinaryInputArchive)
 * Used when receiving completed task results from remote nodes
 */
class TaskLoadOutArchive {
private:
  std::string data_;
  std::unique_ptr<std::istringstream> stream_;
  std::unique_ptr<cereal::BinaryInputArchive> archive_;
  std::vector<BulkTransferInfo> bulk_transfers_;
  std::vector<TaskId> task_ids_; // Task IDs being deserialized

public:
  /** Constructor from serialized data */
  explicit TaskLoadOutArchive(const std::string &data)
      : data_(data), stream_(std::make_unique<std::istringstream>(data_)),
        archive_(std::make_unique<cereal::BinaryInputArchive>(*stream_)) {}

  /** Constructor from const char* and size */
  TaskLoadOutArchive(const char *data, size_t size)
      : data_(data, size), stream_(std::make_unique<std::istringstream>(data_)),
        archive_(std::make_unique<cereal::BinaryInputArchive>(*stream_)) {}

  /** Default constructor */
  TaskLoadOutArchive()
      : stream_(std::make_unique<std::istringstream>("")),
        archive_(std::make_unique<cereal::BinaryInputArchive>(*stream_)) {}

  /** Move constructor */
  TaskLoadOutArchive(TaskLoadOutArchive &&other) noexcept
      : data_(std::move(other.data_)), stream_(std::move(other.stream_)),
        archive_(std::move(other.archive_)),
        bulk_transfers_(std::move(other.bulk_transfers_)),
        task_ids_(std::move(other.task_ids_)) {}

  /** Move assignment operator */
  TaskLoadOutArchive &operator=(TaskLoadOutArchive &&other) noexcept {
    if (this != &other) {
      data_ = std::move(other.data_);
      stream_ = std::move(other.stream_);
      archive_ = std::move(other.archive_);
      bulk_transfers_ = std::move(other.bulk_transfers_);
      task_ids_ = std::move(other.task_ids_);
    }
    return *this;
  }

  /** Delete copy constructor and assignment */
  TaskLoadOutArchive(const TaskLoadOutArchive &) = delete;
  TaskLoadOutArchive &operator=(const TaskLoadOutArchive &) = delete;

  /** Deserialize operator for input archives */
  template <typename T> TaskLoadOutArchive &operator>>(T &value) {
    if constexpr (std::is_base_of_v<Task, T>) {
      // Automatically call BaseSerializeOut + SerializeOut for Task-derived
      // objects
      value.BaseSerializeOut(*this);
      value.SerializeOut(*this);
    } else {
      (*archive_)(value);
    }
    return *this;
  }

  /** Bidirectional serialization - acts as input for this archive type */
  template <typename... Args> void operator()(Args &...args) {
    ((*this >> args), ...);
  }

  /** Bulk transfer support */
  void bulk(hipc::ShmPtr<> ptr, size_t size, uint32_t flags) {
    bulk_transfers_.emplace_back(ptr, size, flags);
  }

  /** Get bulk transfer information */
  const std::vector<BulkTransferInfo> &GetBulkTransfers() const {
    return bulk_transfers_;
  }

  /** Load archive from message with task IDs */
  void LoadFromMessage(const std::string &message,
                       const std::vector<TaskId> &ids) {
    task_ids_ = ids;
    // Reconstruct the archive from the message
    data_ = message;
    stream_ = std::make_unique<std::istringstream>(data_);
    archive_ = std::make_unique<cereal::BinaryInputArchive>(*stream_);
  }

  /** Get task IDs being deserialized */
  const std::vector<TaskId> &GetTaskIds() const { return task_ids_; }

  /** Access underlying cereal archive */
  cereal::BinaryInputArchive &GetArchive() { return *archive_; }
};

/**
 * Archive for serializing task outputs (uses cereal::BinaryOutputArchive)
 * Used when sending completed task results to remote nodes
 */
class TaskSaveOutArchive {
private:
  std::ostringstream stream_;
  std::unique_ptr<cereal::BinaryOutputArchive> archive_;
  std::vector<BulkTransferInfo> bulk_transfers_;
  std::vector<TaskId> task_ids_; // Unique task IDs being serialized

public:
  /** Default constructor */
  TaskSaveOutArchive()
      : archive_(std::make_unique<cereal::BinaryOutputArchive>(stream_)) {}

  /** Move constructor */
  TaskSaveOutArchive(TaskSaveOutArchive &&other) noexcept
      : stream_(std::move(other.stream_)), archive_(std::move(other.archive_)),
        bulk_transfers_(std::move(other.bulk_transfers_)),
        task_ids_(std::move(other.task_ids_)) {}

  /** Move assignment operator */
  TaskSaveOutArchive &operator=(TaskSaveOutArchive &&other) noexcept {
    if (this != &other) {
      stream_ = std::move(other.stream_);
      archive_ = std::move(other.archive_);
      bulk_transfers_ = std::move(other.bulk_transfers_);
      task_ids_ = std::move(other.task_ids_);
    }
    return *this;
  }

  /** Delete copy constructor and assignment */
  TaskSaveOutArchive(const TaskSaveOutArchive &) = delete;
  TaskSaveOutArchive &operator=(const TaskSaveOutArchive &) = delete;

  /** Serialize operator for output archives */
  template <typename T> TaskSaveOutArchive &operator<<(const T &value) {
    if constexpr (std::is_base_of_v<Task, T>) {
      // Automatically call BaseSerializeOut + SerializeOut for Task-derived
      // objects
      const_cast<T &>(value).BaseSerializeOut(*this);
      const_cast<T &>(value).SerializeOut(*this);
    } else {
      (*archive_)(value);
    }
    return *this;
  }

  /** Bidirectional serialization - acts as output for this archive type */
  template <typename... Args> void operator()(const Args &...args) {
    ((*this << args), ...);
  }

  /** Bulk transfer support */
  void bulk(hipc::ShmPtr<> ptr, size_t size, uint32_t flags) {
    bulk_transfers_.emplace_back(ptr, size, flags);
    // For output archives with BULK_XFER, we record the bulk transfer
    // The actual data transfer would be handled by the networking layer
    // which can convert the pointer to actual data when needed
  }

  /** Get serialized data */
  std::string GetData() const { return stream_.str(); }

  /** Get bulk transfer information */
  const std::vector<BulkTransferInfo> &GetBulkTransfers() const {
    return bulk_transfers_;
  }

  /** Get task IDs being serialized */
  const std::vector<TaskId> &GetTaskIds() const { return task_ids_; }

  /** Add a task ID to the serialization */
  void AddTaskId(const TaskId &id) { task_ids_.push_back(id); }

  /** Access underlying cereal archive */
  cereal::BinaryOutputArchive &GetArchive() { return *archive_; }
};

/**
 * Message type enum for task archives
 * Defines the type of message being sent/received
 */
enum class MsgType : uint8_t {
  kSerializeIn = 0,   /**< Serialize task inputs for remote execution */
  kSerializeOut = 1,  /**< Serialize task outputs back to origin */
  kHeartbeat = 2      /**< Heartbeat message (no task data) */
};

/**
 * Common task information structure used by both SaveTaskArchive and
 * LoadTaskArchive
 */
struct TaskInfo {
  TaskId task_id_;
  PoolId pool_id_;
  u32 method_id_;

  template <class Archive> void serialize(Archive &ar) {
    ar(task_id_, pool_id_, method_id_);
  }
};

/**
 * Archive for saving tasks (inputs or outputs) for network transfer
 * Unified archive that replaces TaskSaveInArchive and TaskSaveOutArchive
 * Inherits from LbmMeta to integrate with Lightbeam networking
 */
class SaveTaskArchive : public hshm::lbm::LbmMeta {
private:
  friend class cereal::access;

public:
  std::vector<TaskInfo> task_infos_;
  MsgType msg_type_; // Message type: kSerializeIn, kSerializeOut, or kHeartbeat

private:
  std::ostringstream stream_;
  std::unique_ptr<cereal::BinaryOutputArchive> archive_;
  hshm::lbm::Client *lbm_client_; // Lightbeam client for Expose calls

public:
  /** Constructor with message type and Lightbeam client */
  explicit SaveTaskArchive(MsgType msg_type,
                           hshm::lbm::Client *lbm_client = nullptr)
      : msg_type_(msg_type),
        archive_(std::make_unique<cereal::BinaryOutputArchive>(stream_)),
        lbm_client_(lbm_client) {}

  /** Move constructor */
  SaveTaskArchive(SaveTaskArchive &&other) noexcept
      : hshm::lbm::LbmMeta(std::move(other)),
        task_infos_(std::move(other.task_infos_)), msg_type_(other.msg_type_),
        stream_(std::move(other.stream_)), archive_(std::move(other.archive_)),
        lbm_client_(other.lbm_client_) {
    other.lbm_client_ = nullptr;
  }

  /** Move assignment operator */
  SaveTaskArchive &operator=(SaveTaskArchive &&other) noexcept {
    if (this != &other) {
      hshm::lbm::LbmMeta::operator=(std::move(other));
      stream_ = std::move(other.stream_);
      archive_ = std::move(other.archive_);
      task_infos_ = std::move(other.task_infos_);
      msg_type_ = other.msg_type_;
      lbm_client_ = other.lbm_client_;
      other.lbm_client_ = nullptr;
    }
    return *this;
  }

  /** Delete copy constructor and assignment */
  SaveTaskArchive(const SaveTaskArchive &) = delete;
  SaveTaskArchive &operator=(const SaveTaskArchive &) = delete;

  /** Serialize operator - handles Task-derived types specially */
  template <typename T> SaveTaskArchive &operator<<(T &value) {
    if constexpr (std::is_base_of_v<Task, T>) {
      // Record task information
      TaskInfo info{value.task_id_, value.pool_id_, value.method_};
      task_infos_.push_back(info);

      // Serialize task based on mode
      if (msg_type_ == MsgType::kSerializeIn) {
        // SerializeIn mode - serialize input parameters
        value.BaseSerializeIn(*this);
        value.SerializeIn(*this);
      } else if (msg_type_ == MsgType::kSerializeOut) {
        // SerializeOut mode - serialize output parameters
        value.BaseSerializeOut(*this);
        value.SerializeOut(*this);
      }
      // kHeartbeat has no task data to serialize
    } else {
      (*archive_)(value);
    }
    return *this;
  }

  /** Bidirectional serialization - acts as output for this archive type */
  template <typename... Args> void operator()(Args &...args) {
    (SerializeArg(args), ...);
  }

private:
  /** Helper to serialize individual arguments - handles Tasks specially */
  template <typename T> void SerializeArg(T &arg) {
    if constexpr (std::is_base_of_v<Task,
                                    std::remove_pointer_t<std::decay_t<T>>>) {
      // This is a Task or Task pointer - use operator<< which handles tasks
      *this << arg;
    } else {
      // Regular type - serialize directly with cereal
      (*archive_)(arg);
    }
  }

public:
  /** Bulk transfer support - uses Lightbeam's send vector and automatically
   * calls Expose */
  void bulk(hipc::ShmPtr<> ptr, size_t size, uint32_t flags);

  /** Get task information */
  const std::vector<TaskInfo> &GetTaskInfos() const { return task_infos_; }

  /** Get message type */
  MsgType GetMsgType() const { return msg_type_; }

  /** Get serialized data as string */
  std::string GetData() const { return stream_.str(); }

  /** Access underlying cereal archive */
  cereal::BinaryOutputArchive &GetArchive() { return *archive_; }

  /** Cereal save function - serializes stream contents as string */
  template <class Archive> void save(Archive &ar) const {
    std::string stream_data = stream_.str();
    // Manually serialize base class members (send, recv) and derived class
    // members
    ar(send, recv, task_infos_, msg_type_, stream_data);
  }

  /** Cereal load function - not applicable for output archive */
  template <class Archive> void load(Archive &ar) {
    throw std::runtime_error("SaveTaskArchive::load should not be called - use "
                             "LoadTaskArchive instead");
  }
};

/**
 * Archive for loading tasks (inputs or outputs) from network transfer
 * Unified archive that replaces TaskLoadInArchive and TaskLoadOutArchive
 * Inherits from LbmMeta to integrate with Lightbeam networking
 */
class LoadTaskArchive : public hshm::lbm::LbmMeta {
private:
  friend class cereal::access;

public:
  std::vector<TaskInfo> task_infos_;
  MsgType msg_type_; // Message type: kSerializeIn, kSerializeOut, or kHeartbeat

private:
  std::string data_;
  std::unique_ptr<std::istringstream> stream_;
  std::unique_ptr<cereal::BinaryInputArchive> archive_;
  size_t current_task_index_;
  size_t current_bulk_index_; // Track bulk transfer index for recv vector
  hshm::lbm::Server *lbm_server_; // Lightbeam server for exposing buffers in output mode

public:
  /** Default constructor */
  LoadTaskArchive()
      : msg_type_(MsgType::kSerializeIn), stream_(std::make_unique<std::istringstream>("")),
        archive_(std::make_unique<cereal::BinaryInputArchive>(*stream_)),
        current_task_index_(0), current_bulk_index_(0), lbm_server_(nullptr) {}

  /** Constructor from serialized data */
  explicit LoadTaskArchive(const std::string &data)
      : msg_type_(MsgType::kSerializeIn), data_(data),
        stream_(std::make_unique<std::istringstream>(data_)),
        archive_(std::make_unique<cereal::BinaryInputArchive>(*stream_)),
        current_task_index_(0), current_bulk_index_(0), lbm_server_(nullptr) {}

  /** Move constructor */
  LoadTaskArchive(LoadTaskArchive &&other) noexcept
      : hshm::lbm::LbmMeta(std::move(other)),
        task_infos_(std::move(other.task_infos_)), msg_type_(other.msg_type_),
        data_(std::move(other.data_)), stream_(std::move(other.stream_)),
        archive_(std::move(other.archive_)),
        current_task_index_(other.current_task_index_),
        current_bulk_index_(other.current_bulk_index_),
        lbm_server_(other.lbm_server_) {
    other.lbm_server_ = nullptr;
  }

  /** Move assignment operator */
  LoadTaskArchive &operator=(LoadTaskArchive &&other) noexcept {
    if (this != &other) {
      hshm::lbm::LbmMeta::operator=(std::move(other));
      data_ = std::move(other.data_);
      stream_ = std::move(other.stream_);
      archive_ = std::move(other.archive_);
      task_infos_ = std::move(other.task_infos_);
      current_task_index_ = other.current_task_index_;
      current_bulk_index_ = other.current_bulk_index_;
      msg_type_ = other.msg_type_;
      lbm_server_ = other.lbm_server_;
      other.lbm_server_ = nullptr;
    }
    return *this;
  }

  /** Delete copy constructor and assignment */
  LoadTaskArchive(const LoadTaskArchive &) = delete;
  LoadTaskArchive &operator=(const LoadTaskArchive &) = delete;

  /** Deserialize operator - handles Task-derived types specially */
  /** Deserialize regular (non-pointer) types using underlying cereal archive */
  template <typename T> LoadTaskArchive &operator>>(T &value) {
    if constexpr (std::is_base_of_v<Task, T>) {
      // Automatically call BaseSerialize* + Serialize* for Task-derived objects
      if (msg_type_ == MsgType::kSerializeIn) {
        value.BaseSerializeIn(*this);
        value.SerializeIn(*this);
      } else if (msg_type_ == MsgType::kSerializeOut) {
        value.BaseSerializeOut(*this);
        value.SerializeOut(*this);
      }
      // kHeartbeat has no task data to deserialize
    } else {
      (*archive_)(value);
    }
    return *this;
  }

  /** Deserialize task pointers */
  template <typename T> LoadTaskArchive &operator>>(T *&value) {
    if constexpr (std::is_base_of_v<Task, T>) {
      // value must be pre-allocated by caller using CHI_IPC->NewTask
      // Deserialize task based on mode
      if (msg_type_ == MsgType::kSerializeIn) {
        // SerializeIn mode - deserialize input parameters
        value->BaseSerializeIn(*this);
        value->SerializeIn(*this);
      } else if (msg_type_ == MsgType::kSerializeOut) {
        // SerializeOut mode - deserialize output parameters
        value->BaseSerializeOut(*this);
        value->SerializeOut(*this);
      }
      // kHeartbeat has no task data to deserialize
      current_task_index_++;
    } else {
      (*archive_)(value);
    }
    return *this;
  }

  /** Bidirectional serialization - acts as input for this archive type */
  template <typename... Args> void operator()(Args &...args) {
    (DeserializeArg(args), ...);
  }

private:
  /** Helper to deserialize individual arguments - handles Tasks specially */
  template <typename T> void DeserializeArg(T &arg) {
    if constexpr (std::is_base_of_v<Task,
                                    std::remove_pointer_t<std::decay_t<T>>>) {
      // This is a Task or Task pointer - use operator>> which handles tasks
      *this >> arg;
    } else {
      // Regular type - deserialize directly with cereal
      (*archive_)(arg);
    }
  }

public:
  /** Bulk transfer support - handles both input and output modes */
  void bulk(hipc::ShmPtr<> &ptr, size_t size, uint32_t flags);

  /** Get task information */
  const std::vector<TaskInfo> &GetTaskInfos() const { return task_infos_; }

  /** Get current task info */
  const TaskInfo &GetCurrentTaskInfo() const {
    return task_infos_[current_task_index_];
  }

  /** Get message type */
  MsgType GetMsgType() const { return msg_type_; }

  /** Reset task index for iteration */
  void ResetTaskIndex() { current_task_index_ = 0; }

  /** Reset bulk index for iteration */
  void ResetBulkIndex() { current_bulk_index_ = 0; }

  /** Set Lightbeam server for output mode bulk transfers */
  void SetLbmServer(hshm::lbm::Server *lbm_server) { lbm_server_ = lbm_server; }

  /** Access underlying cereal archive */
  cereal::BinaryInputArchive &GetArchive() { return *archive_; }

  /** Cereal save function - not applicable for input archive */
  template <class Archive> void save(Archive &ar) const {
    throw std::runtime_error("LoadTaskArchive::save should not be called - use "
                             "SaveTaskArchive instead");
  }

  /** Cereal load function - deserializes stream data from string and
   * reinitializes stream */
  template <class Archive> void load(Archive &ar) {
    std::string stream_data;
    // Manually deserialize base class members (send, recv) and derived class
    // members
    ar(send, recv, task_infos_, msg_type_, stream_data);

    // Reinitialize stream with deserialized data
    data_ = stream_data;
    stream_ = std::make_unique<std::istringstream>(data_);
    archive_ = std::make_unique<cereal::BinaryInputArchive>(*stream_);
  }
};

} // namespace chi

// Cereal specialization to disable inherited serialize function from LbmMeta
// This tells cereal to use the member load/save functions instead of the
// inherited serialize
namespace cereal {
template <class Archive>
struct specialize<Archive, chi::SaveTaskArchive,
                  cereal::specialization::member_load_save> {};

template <class Archive>
struct specialize<Archive, chi::LoadTaskArchive,
                  cereal::specialization::member_load_save> {};
} // namespace cereal

#endif // CHIMAERA_INCLUDE_CHIMAERA_TASK_ARCHIVES_H_