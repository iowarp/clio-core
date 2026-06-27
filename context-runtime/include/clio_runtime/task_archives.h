/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef CLIO_RUNTIME_INCLUDE_TASK_ARCHIVES_H_
#define CLIO_RUNTIME_INCLUDE_TASK_ARCHIVES_H_

#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

// Include GlobalSerialize for architecture-portable serialization
#include <clio_ctp/data_structures/serialization/global_serialize.h>

// Include Lightbeam for networking
#include <clio_ctp/lightbeam/lightbeam.h>

#include "clio_runtime/types.h"

namespace clio::run {

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
class NetTaskArchive : public ctp::lbm::LbmMeta<> {
public:
  std::vector<TaskInfo> task_infos_; /**< Task metadata for each serialized task */
  MsgType msg_type_;                 /**< Message type: kSerializeIn, kSerializeOut, or kHeartbeat */
  /**
   * Ephemeral port on the requesting client where responses should be
   * returned. Set on the SerializeIn (request) path; the receiver pairs it
   * with the sender's transport identity to open/cache a dedicated dial-back
   * connection for the SerializeOut (response). 0 = unset (legacy path).
   */
  int client_port_ = 0;

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
      : ctp::lbm::LbmMeta<>(std::move(other)),
        task_infos_(std::move(other.task_infos_)),
        msg_type_(other.msg_type_),
        client_port_(other.client_port_) {}

  /**
   * Move assignment operator
   */
  NetTaskArchive &operator=(NetTaskArchive &&other) noexcept {
    if (this != &other) {
      ctp::lbm::LbmMeta<>::operator=(std::move(other));
      task_infos_ = std::move(other.task_infos_);
      msg_type_ = other.msg_type_;
      client_port_ = other.client_port_;
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
public:
  using is_saving = std::true_type;
  using is_loading = std::false_type;
  using supports_range_ops = std::true_type;
private:
  std::vector<char> buffer_;
  ctp::ipc::GlobalSerialize<std::vector<char>> serializer_;
  ctp::lbm::Transport *lbm_transport_;
  bool is_pod_ = false;

public:
  void PushPod(bool val) { is_pod_ = val; }
  void PopPod() { is_pod_ = false; }

  explicit SaveTaskArchive(MsgType msg_type,
                           ctp::lbm::Transport *lbm_transport = nullptr)
      : NetTaskArchive(msg_type),
        serializer_(buffer_),
        lbm_transport_(lbm_transport) {
    buffer_.reserve(256);
  }

  SaveTaskArchive(SaveTaskArchive &&other) noexcept
      : NetTaskArchive(std::move(other)),
        buffer_(std::move(other.buffer_)),
        serializer_(buffer_, true),
        lbm_transport_(other.lbm_transport_) {
    other.lbm_transport_ = nullptr;
  }

  SaveTaskArchive &operator=(SaveTaskArchive &&other) noexcept = delete;
  SaveTaskArchive(const SaveTaskArchive &) = delete;
  SaveTaskArchive &operator=(const SaveTaskArchive &) = delete;

  template <typename T> SaveTaskArchive &operator<<(T &value) {
    if constexpr (std::is_base_of_v<Task, T>) {
      TaskInfo info{value.task_id_, value.pool_id_, value.method_};
      task_infos_.push_back(info);
      if (msg_type_ == MsgType::kSerializeIn) {
        value.SerializeIn(*this);
      } else if (msg_type_ == MsgType::kSerializeOut) {
        value.SerializeOut(*this);
      }
    } else {
      serializer_ << value;
    }
    return *this;
  }

  template <typename... Args> void operator()(Args &...args) {
    if (is_pod_) {
      range(args...);
    } else {
      (SerializeArg(args), ...);
    }
  }

private:
  template <typename T> void SerializeArg(T &arg) {
    if constexpr (std::is_base_of_v<Task, std::remove_pointer_t<std::decay_t<T>>>) {
      *this << arg;
    } else {
      serializer_ << arg;
    }
  }

public:
  void bulk(ctp::ipc::ShmPtr<> ptr, size_t size, uint32_t flags);

  void write_binary(const char *data, size_t size) {
    serializer_.write_binary(data, size);
  }

  /** range() — per-field serialization (architecture-portable) */
  template <typename... Args>
  void range(Args &...args) {
    serializer_.range(args...);
  }

  std::string GetData() {
    serializer_.Finalize();
    return std::string(buffer_.begin(), buffer_.end());
  }

  const std::vector<char> &GetBuffer() {
    serializer_.Finalize();
    return buffer_;
  }

  void SetTransport(ctp::lbm::Transport *lbm_transport) { lbm_transport_ = lbm_transport; }

  template <typename Ar>
  void serialize(Ar &ar) {
    ar(send, recv, send_bulks, recv_bulks);
    ar(task_infos_, msg_type_, client_port_);
    serializer_.Finalize();
    ar(buffer_);
  }

};

/**
 * Archive for loading tasks (inputs or outputs) from network transfer
 * Unified archive that handles both SerializeIn and SerializeOut modes
 * Inherits from NetTaskArchive to integrate with Lightbeam networking
 */
class LoadTaskArchive : public NetTaskArchive {
public:
  using is_saving = std::false_type;
  using is_loading = std::true_type;
  using supports_range_ops = std::true_type;
private:
  std::vector<char> data_;
  ctp::ipc::GlobalDeserialize<std::vector<char>> deserializer_;
  size_t current_task_index_;
  size_t current_bulk_index_;
  ctp::lbm::Transport *lbm_transport_;
  bool is_pod_ = false;

public:
  // Number of bulk buffers that LoadTaskArchive::bulk() allocated locally via
  // CLIO_IPC->AllocateBuffer (BULK_EXPOSE on receive). The receiver owns these
  // and must FreeBuffer them after the task completes — otherwise cross-node
  // GetBlob responses leak the 1 MiB output buffer per call and the daemon's
  // SHM segment fills up after a few thousand cross-node reads.
  // RecvIn promotes this to a TASK_DATA_OWNER flag on the task so the task's
  // destructor frees the buffer; SendOut skips clearing the flag for tasks
  // that actually have owned buffers.
  size_t daemon_allocated_bulk_count_ = 0;
private:

  static const std::vector<char> &GetEmptyBuffer() {
    static const std::vector<char> empty;
    return empty;
  }

public:
  LoadTaskArchive()
      : NetTaskArchive(MsgType::kSerializeIn),
        deserializer_(GetEmptyBuffer()),
        current_task_index_(0),
        current_bulk_index_(0),
        lbm_transport_(nullptr) {}

  explicit LoadTaskArchive(const std::string &data)
      : NetTaskArchive(MsgType::kSerializeIn),
        data_(data.begin(), data.end()),
        deserializer_(data_),
        current_task_index_(0),
        current_bulk_index_(0),
        lbm_transport_(nullptr) {}

  LoadTaskArchive(const char *data, size_t size)
      : NetTaskArchive(MsgType::kSerializeIn),
        data_(data, data + size),
        deserializer_(data_),
        current_task_index_(0),
        current_bulk_index_(0),
        lbm_transport_(nullptr) {}

  explicit LoadTaskArchive(std::vector<char> &&data)
      : NetTaskArchive(MsgType::kSerializeIn),
        data_(std::move(data)),
        deserializer_(data_),
        current_task_index_(0),
        current_bulk_index_(0),
        lbm_transport_(nullptr) {}

  LoadTaskArchive(LoadTaskArchive &&other) noexcept
      : NetTaskArchive(std::move(other)),
        data_(std::move(other.data_)),
        deserializer_(data_),
        current_task_index_(other.current_task_index_),
        current_bulk_index_(other.current_bulk_index_),
        lbm_transport_(other.lbm_transport_) {
    other.lbm_transport_ = nullptr;
  }

  LoadTaskArchive &operator=(LoadTaskArchive &&other) noexcept = delete;
  LoadTaskArchive(const LoadTaskArchive &) = delete;
  LoadTaskArchive &operator=(const LoadTaskArchive &) = delete;

  void PushPod(bool val) { is_pod_ = val; }
  void PopPod() { is_pod_ = false; }

  template <typename T> LoadTaskArchive &operator>>(T &value) {
    if constexpr (std::is_base_of_v<Task, T>) {
      if (msg_type_ == MsgType::kSerializeIn) {
        value.SerializeIn(*this);
      } else if (msg_type_ == MsgType::kSerializeOut) {
        value.SerializeOut(*this);
      }
    } else {
      deserializer_ >> value;
    }
    return *this;
  }

  template <typename T> LoadTaskArchive &operator>>(T *&value) {
    if constexpr (std::is_base_of_v<Task, T>) {
      if (msg_type_ == MsgType::kSerializeIn) {
        value->SerializeIn(*this);
      } else if (msg_type_ == MsgType::kSerializeOut) {
        value->SerializeOut(*this);
      }
      current_task_index_++;
    } else {
      deserializer_ >> value;
    }
    return *this;
  }

  template <typename... Args> void operator()(Args &...args) {
    if (is_pod_) {
      range(args...);
    } else {
      (DeserializeArg(args), ...);
    }
  }

private:
  template <typename T> void DeserializeArg(T &arg) {
    if constexpr (std::is_base_of_v<Task, std::remove_pointer_t<std::decay_t<T>>>) {
      *this >> arg;
    } else {
      deserializer_ >> arg;
    }
  }

public:
  void bulk(ctp::ipc::ShmPtr<> &ptr, size_t size, uint32_t flags);

  const TaskInfo &GetCurrentTaskInfo() const {
    return task_infos_[current_task_index_];
  }

  void ResetTaskIndex() { current_task_index_ = 0; }
  void ResetBulkIndex() { current_bulk_index_ = 0; }

  void SetTransport(ctp::lbm::Transport *lbm_transport) { lbm_transport_ = lbm_transport; }

  void read_binary(char *data, size_t size) {
    deserializer_.read_binary(data, size);
  }

  /** range() — per-field deserialization (architecture-portable) */
  template <typename... Args>
  void range(Args &...args) {
    deserializer_.range(args...);
  }

  template <typename Ar>
  void serialize(Ar &ar) {
    ar(send, recv, send_bulks, recv_bulks);
    ar(task_infos_, msg_type_, client_port_);
    ar(data_);
    // Reinitialize deserializer with new data
    new (&deserializer_)
        ctp::ipc::GlobalDeserialize<std::vector<char>>(data_);
  }

};

}  // namespace clio::run

#endif // CLIO_RUNTIME_INCLUDE_TASK_ARCHIVES_H_
