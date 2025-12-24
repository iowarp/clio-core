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
#include <hermes_shm/lightbeam/lightbeam.h>

#include "chimaera/types.h"

namespace chi {

// Forward declaration
class Task;

/**
 * Message type enum for task archives
 * Defines the type of message being sent/received
 */
enum class MsgType : uint8_t {
  kSerializeIn = 0,  /**< Serialize task inputs for remote execution */
  kSerializeOut = 1, /**< Serialize task outputs back to origin */
  kHeartbeat = 2     /**< Heartbeat message (no task data) */
};

/**
 * Common task information structure used by network task archives
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
 * Base class for network task archives
 * Inherits from LbmMeta to integrate with Lightbeam networking
 * Provides common functionality for both SaveTaskArchive and LoadTaskArchive
 *
 * LbmMeta provides:
 * - send: vector<Bulk> for sender's bulk descriptors
 * - recv: vector<Bulk> for receiver's bulk descriptors
 *
 * NetTaskArchive adds:
 * - task_infos_: vector of TaskInfo for task metadata
 * - msg_type_: MsgType for message type (SerializeIn, SerializeOut, Heartbeat)
 */
class NetTaskArchive : public hshm::lbm::LbmMeta {
public:
  std::vector<TaskInfo> task_infos_; /**< Task metadata for each serialized task */
  MsgType msg_type_;                 /**< Message type: kSerializeIn, kSerializeOut, or kHeartbeat */

  /**
   * Default constructor
   */
  NetTaskArchive() : msg_type_(MsgType::kSerializeIn) {}

  /**
   * Constructor with message type
   * @param msg_type The type of message (SerializeIn, SerializeOut, Heartbeat)
   */
  explicit NetTaskArchive(MsgType msg_type) : msg_type_(msg_type) {}

  /**
   * Virtual destructor
   */
  virtual ~NetTaskArchive() = default;

  /**
   * Move constructor
   */
  NetTaskArchive(NetTaskArchive &&other) noexcept
      : hshm::lbm::LbmMeta(std::move(other)),
        task_infos_(std::move(other.task_infos_)),
        msg_type_(other.msg_type_) {}

  /**
   * Move assignment operator
   */
  NetTaskArchive &operator=(NetTaskArchive &&other) noexcept {
    if (this != &other) {
      hshm::lbm::LbmMeta::operator=(std::move(other));
      task_infos_ = std::move(other.task_infos_);
      msg_type_ = other.msg_type_;
    }
    return *this;
  }

  /**
   * Delete copy constructor and assignment
   */
  NetTaskArchive(const NetTaskArchive &) = delete;
  NetTaskArchive &operator=(const NetTaskArchive &) = delete;

  /**
   * Get task information
   * @return Reference to the vector of TaskInfo
   */
  const std::vector<TaskInfo> &GetTaskInfos() const { return task_infos_; }

  /**
   * Get message type
   * @return The message type
   */
  MsgType GetMsgType() const { return msg_type_; }

  /**
   * Get number of bulk transfers in send vector
   * @return Number of bulk transfers
   */
  size_t GetSendBulkCount() const { return send.size(); }

  /**
   * Get number of bulk transfers in recv vector
   * @return Number of bulk transfers
   */
  size_t GetRecvBulkCount() const { return recv.size(); }
};

/**
 * Archive for saving tasks (inputs or outputs) for network transfer
 * Unified archive that handles both SerializeIn and SerializeOut modes
 * Inherits from NetTaskArchive to integrate with Lightbeam networking
 */
class SaveTaskArchive : public NetTaskArchive {
private:
  friend class cereal::access;

  std::ostringstream stream_;
  std::unique_ptr<cereal::BinaryOutputArchive> archive_;
  hshm::lbm::Client *lbm_client_; /**< Lightbeam client for Expose calls */

public:
  /**
   * Constructor with message type and optional Lightbeam client
   * @param msg_type The type of message (SerializeIn, SerializeOut, Heartbeat)
   * @param lbm_client Optional Lightbeam client for bulk transfer Expose calls
   */
  explicit SaveTaskArchive(MsgType msg_type,
                           hshm::lbm::Client *lbm_client = nullptr)
      : NetTaskArchive(msg_type),
        archive_(std::make_unique<cereal::BinaryOutputArchive>(stream_)),
        lbm_client_(lbm_client) {}

  /**
   * Move constructor
   */
  SaveTaskArchive(SaveTaskArchive &&other) noexcept
      : NetTaskArchive(std::move(other)),
        stream_(std::move(other.stream_)),
        archive_(std::move(other.archive_)),
        lbm_client_(other.lbm_client_) {
    other.lbm_client_ = nullptr;
  }

  /**
   * Move assignment operator
   */
  SaveTaskArchive &operator=(SaveTaskArchive &&other) noexcept {
    if (this != &other) {
      NetTaskArchive::operator=(std::move(other));
      stream_ = std::move(other.stream_);
      archive_ = std::move(other.archive_);
      lbm_client_ = other.lbm_client_;
      other.lbm_client_ = nullptr;
    }
    return *this;
  }

  /**
   * Delete copy constructor and assignment
   */
  SaveTaskArchive(const SaveTaskArchive &) = delete;
  SaveTaskArchive &operator=(const SaveTaskArchive &) = delete;

  /**
   * Serialize operator - handles Task-derived types specially
   * For Task types, records task info and calls SerializeIn or SerializeOut
   * @param value The value to serialize
   * @return Reference to this archive
   */
  template <typename T> SaveTaskArchive &operator<<(T &value) {
    if constexpr (std::is_base_of_v<Task, T>) {
      // Record task information
      TaskInfo info{value.task_id_, value.pool_id_, value.method_};
      task_infos_.push_back(info);

      // Serialize task based on mode
      if (msg_type_ == MsgType::kSerializeIn) {
        value.SerializeIn(*this);
      } else if (msg_type_ == MsgType::kSerializeOut) {
        value.SerializeOut(*this);
      }
      // kHeartbeat has no task data to serialize
    } else {
      (*archive_)(value);
    }
    return *this;
  }

  /**
   * Bidirectional serialization - acts as output for this archive type
   * @param args Values to serialize
   */
  template <typename... Args> void operator()(Args &...args) {
    (SerializeArg(args), ...);
  }

private:
  /**
   * Helper to serialize individual arguments - handles Tasks specially
   * @param arg The argument to serialize
   */
  template <typename T> void SerializeArg(T &arg) {
    if constexpr (std::is_base_of_v<Task, std::remove_pointer_t<std::decay_t<T>>>) {
      *this << arg;
    } else {
      (*archive_)(arg);
    }
  }

public:
  /**
   * Bulk transfer support - adds bulk descriptor to send vector
   * Uses Lightbeam's Expose if lbm_client is provided
   * @param ptr Shared memory pointer to the data
   * @param size Size of the data in bytes
   * @param flags Transfer flags (BULK_XFER or BULK_EXPOSE)
   */
  void bulk(hipc::ShmPtr<> ptr, size_t size, uint32_t flags);

  /**
   * Get serialized data as string
   * @return The serialized data
   */
  std::string GetData() const { return stream_.str(); }

  /**
   * Access underlying cereal archive
   * @return Reference to the cereal archive
   */
  cereal::BinaryOutputArchive &GetArchive() { return *archive_; }

  /**
   * Set the Lightbeam client for bulk transfers
   * @param lbm_client Pointer to the Lightbeam client
   */
  void SetLbmClient(hshm::lbm::Client *lbm_client) { lbm_client_ = lbm_client; }

  /**
   * Cereal save function - serializes archive contents
   * @param ar The cereal archive
   */
  template <class Archive> void save(Archive &ar) const {
    std::string stream_data = stream_.str();
    ar(send, recv, task_infos_, msg_type_, stream_data);
  }

  /**
   * Cereal load function - not applicable for output archive
   * @param ar The cereal archive
   */
  template <class Archive> void load(Archive &ar) {
    throw std::runtime_error(
        "SaveTaskArchive::load should not be called - use LoadTaskArchive instead");
  }
};

/**
 * Archive for loading tasks (inputs or outputs) from network transfer
 * Unified archive that handles both SerializeIn and SerializeOut modes
 * Inherits from NetTaskArchive to integrate with Lightbeam networking
 */
class LoadTaskArchive : public NetTaskArchive {
private:
  friend class cereal::access;

  std::string data_;
  std::unique_ptr<std::istringstream> stream_;
  std::unique_ptr<cereal::BinaryInputArchive> archive_;
  size_t current_task_index_;   /**< Index of current task being deserialized */
  size_t current_bulk_index_;   /**< Index of current bulk transfer in recv vector */
  hshm::lbm::Server *lbm_server_; /**< Lightbeam server for output mode bulk transfers */

public:
  /**
   * Default constructor
   */
  LoadTaskArchive()
      : NetTaskArchive(MsgType::kSerializeIn),
        stream_(std::make_unique<std::istringstream>("")),
        archive_(std::make_unique<cereal::BinaryInputArchive>(*stream_)),
        current_task_index_(0),
        current_bulk_index_(0),
        lbm_server_(nullptr) {}

  /**
   * Constructor from serialized data
   * @param data The serialized data string
   */
  explicit LoadTaskArchive(const std::string &data)
      : NetTaskArchive(MsgType::kSerializeIn),
        data_(data),
        stream_(std::make_unique<std::istringstream>(data_)),
        archive_(std::make_unique<cereal::BinaryInputArchive>(*stream_)),
        current_task_index_(0),
        current_bulk_index_(0),
        lbm_server_(nullptr) {}

  /**
   * Constructor from const char* and size
   * @param data Pointer to the serialized data
   * @param size Size of the data in bytes
   */
  LoadTaskArchive(const char *data, size_t size)
      : NetTaskArchive(MsgType::kSerializeIn),
        data_(data, size),
        stream_(std::make_unique<std::istringstream>(data_)),
        archive_(std::make_unique<cereal::BinaryInputArchive>(*stream_)),
        current_task_index_(0),
        current_bulk_index_(0),
        lbm_server_(nullptr) {}

  /**
   * Move constructor
   */
  LoadTaskArchive(LoadTaskArchive &&other) noexcept
      : NetTaskArchive(std::move(other)),
        data_(std::move(other.data_)),
        stream_(std::move(other.stream_)),
        archive_(std::move(other.archive_)),
        current_task_index_(other.current_task_index_),
        current_bulk_index_(other.current_bulk_index_),
        lbm_server_(other.lbm_server_) {
    other.lbm_server_ = nullptr;
  }

  /**
   * Move assignment operator
   */
  LoadTaskArchive &operator=(LoadTaskArchive &&other) noexcept {
    if (this != &other) {
      NetTaskArchive::operator=(std::move(other));
      data_ = std::move(other.data_);
      stream_ = std::move(other.stream_);
      archive_ = std::move(other.archive_);
      current_task_index_ = other.current_task_index_;
      current_bulk_index_ = other.current_bulk_index_;
      lbm_server_ = other.lbm_server_;
      other.lbm_server_ = nullptr;
    }
    return *this;
  }

  /**
   * Delete copy constructor and assignment
   */
  LoadTaskArchive(const LoadTaskArchive &) = delete;
  LoadTaskArchive &operator=(const LoadTaskArchive &) = delete;

  /**
   * Deserialize operator - handles Task-derived types specially
   * For regular (non-pointer) types
   * @param value The value to deserialize into
   * @return Reference to this archive
   */
  template <typename T> LoadTaskArchive &operator>>(T &value) {
    if constexpr (std::is_base_of_v<Task, T>) {
      if (msg_type_ == MsgType::kSerializeIn) {
        value.SerializeIn(*this);
      } else if (msg_type_ == MsgType::kSerializeOut) {
        value.SerializeOut(*this);
      }
      // kHeartbeat has no task data to deserialize
    } else {
      (*archive_)(value);
    }
    return *this;
  }

  /**
   * Deserialize task pointers
   * @param value The task pointer to deserialize into
   * @return Reference to this archive
   */
  template <typename T> LoadTaskArchive &operator>>(T *&value) {
    if constexpr (std::is_base_of_v<Task, T>) {
      // value must be pre-allocated by caller using CHI_IPC->NewTask
      if (msg_type_ == MsgType::kSerializeIn) {
        value->SerializeIn(*this);
      } else if (msg_type_ == MsgType::kSerializeOut) {
        value->SerializeOut(*this);
      }
      // kHeartbeat has no task data to deserialize
      current_task_index_++;
    } else {
      (*archive_)(value);
    }
    return *this;
  }

  /**
   * Bidirectional serialization - acts as input for this archive type
   * @param args Values to deserialize
   */
  template <typename... Args> void operator()(Args &...args) {
    (DeserializeArg(args), ...);
  }

private:
  /**
   * Helper to deserialize individual arguments - handles Tasks specially
   * @param arg The argument to deserialize
   */
  template <typename T> void DeserializeArg(T &arg) {
    if constexpr (std::is_base_of_v<Task, std::remove_pointer_t<std::decay_t<T>>>) {
      *this >> arg;
    } else {
      (*archive_)(arg);
    }
  }

public:
  /**
   * Bulk transfer support - handles both input and output modes
   * For SerializeIn mode: gets pointer from recv vector at current index
   * For SerializeOut mode: exposes pointer using lbm_server and adds to recv
   * @param ptr Reference to shared memory pointer (output parameter for SerializeIn)
   * @param size Size of the data in bytes
   * @param flags Transfer flags (BULK_XFER or BULK_EXPOSE)
   */
  void bulk(hipc::ShmPtr<> &ptr, size_t size, uint32_t flags);

  /**
   * Get current task info
   * @return Reference to the current TaskInfo
   */
  const TaskInfo &GetCurrentTaskInfo() const {
    return task_infos_[current_task_index_];
  }

  /**
   * Reset task index for iteration
   */
  void ResetTaskIndex() { current_task_index_ = 0; }

  /**
   * Reset bulk index for iteration
   */
  void ResetBulkIndex() { current_bulk_index_ = 0; }

  /**
   * Set Lightbeam server for output mode bulk transfers
   * @param lbm_server Pointer to the Lightbeam server
   */
  void SetLbmServer(hshm::lbm::Server *lbm_server) { lbm_server_ = lbm_server; }

  /**
   * Access underlying cereal archive
   * @return Reference to the cereal archive
   */
  cereal::BinaryInputArchive &GetArchive() { return *archive_; }

  /**
   * Cereal save function - not applicable for input archive
   * @param ar The cereal archive
   */
  template <class Archive> void save(Archive &ar) const {
    throw std::runtime_error(
        "LoadTaskArchive::save should not be called - use SaveTaskArchive instead");
  }

  /**
   * Cereal load function - deserializes archive contents
   * @param ar The cereal archive
   */
  template <class Archive> void load(Archive &ar) {
    std::string stream_data;
    ar(send, recv, task_infos_, msg_type_, stream_data);

    // Reinitialize stream with deserialized data
    data_ = stream_data;
    stream_ = std::make_unique<std::istringstream>(data_);
    archive_ = std::make_unique<cereal::BinaryInputArchive>(*stream_);
  }
};

} // namespace chi

// Cereal specialization to disable inherited serialize function from LbmMeta
namespace cereal {
template <class Archive>
struct specialize<Archive, chi::SaveTaskArchive,
                  cereal::specialization::member_load_save> {};

template <class Archive>
struct specialize<Archive, chi::LoadTaskArchive,
                  cereal::specialization::member_load_save> {};
} // namespace cereal

#endif // CHIMAERA_INCLUDE_CHIMAERA_TASK_ARCHIVES_H_
