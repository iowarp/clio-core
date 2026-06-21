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

#ifndef CLIO_RUNTIME_INCLUDE_LOCAL_TASK_ARCHIVES_H_
#define CLIO_RUNTIME_INCLUDE_LOCAL_TASK_ARCHIVES_H_

#include <type_traits>
#include <utility>
#include <vector>

// Include LocalSerialize for serialization
#include <clio_ctp/data_structures/priv/vector.h>
#include <clio_ctp/data_structures/priv/wrap_vector.h>
#include <clio_ctp/data_structures/serialization/local_serialize.h>
#include <clio_ctp/lightbeam/lightbeam.h>

#include "clio_runtime/types.h"

namespace clio::run {

// Forward declaration
class Task;

/**
 * Message type enum for local task archives
 * Defines the type of message being sent/received
 */
enum class LocalMsgType : uint8_t {
  kSerializeIn = 0,  /**< Serialize task inputs for execution */
  kSerializeOut = 1, /**< Serialize task outputs */
};

/**
 * Common task information structure used by both LocalSaveTaskArchive and
 * LocalLoadTaskArchive
 */
struct LocalTaskInfo {
  TaskId task_id_;
  PoolId pool_id_;
  u32 method_id_;

  /**
   * Cereal serialization support for network transfer
   * @tparam Archive Archive type
   * @param ar Archive instance
   */
  template <class Archive>
  CTP_CROSS_FUN void serialize(Archive &ar) {
    ar(task_id_.pid_, task_id_.tid_, task_id_.major_, task_id_.replica_id_,
       task_id_.unique_, task_id_.node_id_, task_id_.net_key_);
    ar(pool_id_.major_, pool_id_.minor_);
    ar(method_id_);
  }
};

}  // namespace clio::run

// Add local serialization support for LocalTaskInfo in ctp::ipc namespace
namespace ctp::ipc {

/**
 * Save LocalTaskInfo to local archive
 * @param ar Archive to save to
 * @param info LocalTaskInfo to serialize
 */
template <typename Ar>
CTP_CROSS_FUN void save(Ar &ar, const clio::run::LocalTaskInfo &info) {
  // Serialize TaskId fields
  ar << info.task_id_.pid_;
  ar << info.task_id_.tid_;
  ar << info.task_id_.major_;
  ar << info.task_id_.replica_id_;
  ar << info.task_id_.unique_;
  ar << info.task_id_.node_id_;
  ar << info.task_id_.net_key_;
  // Serialize PoolId (UniqueId) fields
  ar << info.pool_id_.major_;
  ar << info.pool_id_.minor_;
  // Serialize method_id
  ar << info.method_id_;
}

/**
 * Load LocalTaskInfo from local archive
 * @param ar Archive to load from
 * @param info LocalTaskInfo to deserialize into
 */
template <typename Ar>
CTP_CROSS_FUN void load(Ar &ar, clio::run::LocalTaskInfo &info) {
  // Deserialize TaskId fields
  ar >> info.task_id_.pid_;
  ar >> info.task_id_.tid_;
  ar >> info.task_id_.major_;
  ar >> info.task_id_.replica_id_;
  ar >> info.task_id_.unique_;
  ar >> info.task_id_.node_id_;
  ar >> info.task_id_.net_key_;
  // Deserialize PoolId (UniqueId) fields
  ar >> info.pool_id_.major_;
  ar >> info.pool_id_.minor_;
  // Deserialize method_id
  ar >> info.method_id_;
}

}  // namespace ctp::ipc

namespace clio::run {

// Base type for LbmMeta inheritance: use CLIO_PRIV_ALLOC_T so that
// ShmTransport::Recv can allocate internal buffers on GPU (BuddyAllocator)
// and on host (MallocAllocator).
using LocalLbmBase = ctp::lbm::LbmMeta<CLIO_PRIV_ALLOC_T>;

using LocalTaskInfoVec = clio::run::priv::vector<LocalTaskInfo>;

/**
 * Dry-run archive that computes serialized size for a task without copying.
 * Implements the same API surface as LocalSaveTaskArchive so that
 * SerializeIn / SerializeOut code paths work unchanged.
 * Used by LocalSaveTaskArchive to pre-size the buffer before serialization.
 */
class CalculateSizeTaskArchive {
 public:
  using is_saving = std::true_type;
  using is_loading = std::false_type;
  using supports_range_ops = std::true_type;

 private:
  ctp::ipc::CalculateSizeArchive calc_;
  bool is_pod_ = false;

 public:
  CTP_CROSS_FUN void PushPod(bool val) { is_pod_ = val; }
  CTP_CROSS_FUN void PopPod() { is_pod_ = false; }

  CTP_CROSS_FUN explicit CalculateSizeTaskArchive(LocalMsgType /*msg_type*/)
      {}

  /** Get the total computed size */
  CTP_INLINE_CROSS_FUN size_t size() const { return calc_.size(); }

  /** Serialize operator — for non-Task types, delegates to CalculateSizeArchive */
  template <typename T>
  CTP_CROSS_FUN CalculateSizeTaskArchive &operator<<(const T &obj) {
    calc_ << obj;
    return *this;
  }

  /** & operator */
  template <typename T>
  CTP_CROSS_FUN CalculateSizeTaskArchive &operator&(const T &obj) {
    calc_ << obj;
    return *this;
  }

  /** Call operator */
  template <typename... Args>
  CTP_CROSS_FUN void operator()(Args &...args) {
    if (is_pod_) {
      calc_.range(args...);
    } else {
      (calc_.base(args), ...);
    }
  }

  /** Write raw binary data — just accumulate size */
  CTP_CROSS_FUN void write_binary(const char *data, size_t size) {
    calc_.write_binary(data, size);
  }

  /** write_range — compute span size */
  template <typename FirstT, typename LastT>
  CTP_INLINE_CROSS_FUN void write_range(const FirstT *first,
                                          const LastT *last) {
    calc_.write_range(first, last);
  }

  /** range() — compute span size of contiguous POD fields */
  template <typename... Args>
  CTP_INLINE_CROSS_FUN void range(Args &...args) {
    calc_.range(args...);
  }

  /** Fused string save — just accumulate size */
  CTP_CROSS_FUN void save_string_fused(const char *str_data, size_t len) {
    calc_.save_string_fused(str_data, len);
  }

  /** Bulk transfer size for ShmPtr */
  template <typename T>
  CTP_CROSS_FUN void bulk(ctp::ipc::ShmPtr<T> ptr, size_t size, uint32_t flags) {
    if (!ptr.alloc_id_.IsNull()) {
      // mode=0: uint8_t + size_t + u32 + u32
      calc_.cur_off_ += sizeof(uint8_t) + sizeof(size_t) +
                         sizeof(uint32_t) + sizeof(uint32_t);
    } else if (ptr.off_.load() != 0) {
      // mode=3: uint8_t + size_t
      calc_.cur_off_ += sizeof(uint8_t) + sizeof(size_t);
    } else if (flags & BULK_XFER) {
      // mode=1: uint8_t + data bytes
      calc_.cur_off_ += sizeof(uint8_t) + size;
    } else {
      // mode=2: uint8_t
      calc_.cur_off_ += sizeof(uint8_t);
    }
  }

  /** Bulk transfer size for FullPtr */
  template <typename T>
  CTP_CROSS_FUN void bulk(const ctp::ipc::FullPtr<T> &ptr, size_t size,
                           uint32_t flags) {
    if (!ptr.shm_.alloc_id_.IsNull()) {
      calc_.cur_off_ += sizeof(uint8_t) + sizeof(size_t) +
                         sizeof(uint32_t) + sizeof(uint32_t);
    } else if (flags & BULK_XFER) {
      calc_.cur_off_ += sizeof(uint8_t) + size;
    } else {
      calc_.cur_off_ += sizeof(uint8_t);
    }
  }

  /** Bulk transfer size for raw pointer */
  template <typename T>
  CTP_CROSS_FUN void bulk(T *ptr, size_t size, uint32_t flags) {
    (void)ptr; (void)flags;
    calc_.cur_off_ += size;
  }
};

/**
 * Archive for saving tasks (inputs or outputs) using LocalSerialize.
 * Templated on BufferT so callers can use any vector-like container
 * (e.g., clio::run::priv::vector<char>, ctp::priv::wrap_vector, etc.).
 *
 * Does NOT own the buffer — callers provide a reference.
 * Inherits from LbmMeta for ShmTransport::Send compatibility.
 *
 * @tparam BufferT Buffer type (must support data(), size(), resize(), reserve())
 */
template <typename BufferT>
class LocalSaveTaskArchive : public LocalLbmBase {
  using Base = LocalLbmBase;

 public:
  using is_saving = std::true_type;
  using is_loading = std::false_type;
  using supports_range_ops = std::true_type;
  using buffer_type = BufferT;
  LocalTaskInfoVec task_infos_;
  LocalMsgType msg_type_; /**< Message type: kSerializeIn or kSerializeOut */

 private:
  BufferT &buffer_;
  ctp::ipc::LocalSerialize<BufferT> serializer_;
  bool is_pod_ = false;

 public:
  CTP_CROSS_FUN void PushPod(bool val) { is_pod_ = val; }
  CTP_CROSS_FUN void PopPod() { is_pod_ = false; }

  /** Mark this archive for warp-parallel write_binary calls */
  CTP_INLINE_CROSS_FUN void SetWarpConverged(bool v) {
    serializer_.warp_converged_ = v;
  }

  /** Reset for reuse — avoids re-allocating the buffer between tasks */
  CTP_CROSS_FUN void Reset(LocalMsgType msg_type) {
    msg_type_ = msg_type;
    task_infos_.resize(0);
    serializer_.cur_off_ = 0;
  }

  /** Get the number of bytes actually serialized (not the buffer capacity) */
  CTP_INLINE_CROSS_FUN size_t GetSerializedSize() const {
    return serializer_.cur_off_;
  }

  /**
   * Serialize for ShmTransport compatibility.
   * Wire-compatible with SaveTaskArchive::serialize /
   * LoadTaskArchive::serialize.
   */
  template <typename Ar>
  CTP_CROSS_FUN void serialize(Ar &ar) {
    if constexpr (Ar::is_saving::value) {
      serializer_.Finalize();
    }
    ar(this->send, this->recv, this->send_bulks, this->recv_bulks);
    ar(task_infos_, msg_type_);
    ar(buffer_);
  }

  /**
   * Constructor with allocator and external buffer reference.
   * The archive does NOT own the buffer.
   *
   * @param msg_type Message type (kSerializeIn or kSerializeOut)
   * @param alloc Allocator for internal vectors (task_infos_, LbmMeta bulks)
   * @param buffer External buffer to serialize into
   */
  template <typename AllocPtrT>
  CTP_CROSS_FUN LocalSaveTaskArchive(LocalMsgType msg_type,
                                       AllocPtrT *alloc,
                                       BufferT &buffer)
      : Base(alloc),
        task_infos_(alloc),
        msg_type_(msg_type),
        buffer_(buffer),
        serializer_(buffer) {
    task_infos_.reserve(4);
  }

  /**
   * Constructor with default allocator and external buffer reference.
   */
  CTP_CROSS_FUN LocalSaveTaskArchive(LocalMsgType msg_type,
                                       BufferT &buffer)
      : Base(CLIO_PRIV_ALLOC),
        task_infos_(CLIO_PRIV_ALLOC),
        msg_type_(msg_type),
        buffer_(buffer),
        serializer_(buffer) {
    task_infos_.reserve(4);
  }

  /** Delete copy/move — reference member prevents safe copy/move */
  LocalSaveTaskArchive(const LocalSaveTaskArchive &) = delete;
  LocalSaveTaskArchive &operator=(const LocalSaveTaskArchive &) = delete;
  LocalSaveTaskArchive(LocalSaveTaskArchive &&) = delete;
  LocalSaveTaskArchive &operator=(LocalSaveTaskArchive &&) = delete;

 public:
  /**
   * Serialize operator - handles Task-derived types specially
   */
  template <typename T>
  CTP_CROSS_FUN LocalSaveTaskArchive &operator<<(T &value) {
    if constexpr (std::is_base_of_v<Task, T>) {
      // Record task information
      LocalTaskInfo info{value.task_id_, value.pool_id_, value.method_};
      task_infos_.push_back(info);

      // Serialize task based on mode
      if (msg_type_ == LocalMsgType::kSerializeIn) {
        value.SerializeIn(*this);
      } else if (msg_type_ == LocalMsgType::kSerializeOut) {
        value.SerializeOut(*this);
      }
    } else {
      serializer_ << value;
    }
    return *this;
  }

  template <typename T>
  CTP_CROSS_FUN LocalSaveTaskArchive &operator&(T &value) {
    return *this << value;
  }

  template <typename... Args>
  CTP_CROSS_FUN void operator()(Args &...args) {
    if (is_pod_) {
      serializer_.range(args...);
    } else {
      (SerializeArg(args), ...);
    }
  }

 private:
  template <typename T>
  CTP_CROSS_FUN void SerializeArg(T &arg) {
    if constexpr (std::is_base_of_v<Task,
                                    std::remove_pointer_t<std::decay_t<T>>>) {
      *this << arg;
    } else {
      serializer_ << arg;
    }
  }

 public:
  CTP_CROSS_FUN void write_binary(const char *data, size_t size) {
    serializer_.write_binary(data, size);
  }

  CTP_CROSS_FUN void save_string_fused(const char *str_data, size_t len) {
    serializer_.save_string_fused(str_data, len);
  }

  template <typename FirstT, typename LastT>
  CTP_INLINE_CROSS_FUN void write_range(const FirstT *first,
                                          const LastT *last) {
    serializer_.write_range(first, last);
  }

  template <typename... Args>
  CTP_INLINE_CROSS_FUN void range(Args &...args) {
    serializer_.range(args...);
  }

  template <typename T>
  CTP_CROSS_FUN void bulk(ctp::ipc::ShmPtr<T> ptr, size_t size, uint32_t flags) {
    if (!ptr.alloc_id_.IsNull()) {
      uint8_t mode = 0;
      serializer_ << mode;
      size_t off = ptr.off_.load();
      serializer_ << off << ptr.alloc_id_.major_ << ptr.alloc_id_.minor_;
    } else if (ptr.off_.load() != 0) {
      uint8_t mode = 3;
      serializer_ << mode;
      size_t raw_ptr = ptr.off_.load();
      serializer_ << raw_ptr;
    } else if (flags & BULK_XFER) {
      uint8_t mode = 1;
      serializer_ << mode;
      char *raw_ptr = reinterpret_cast<char *>(ptr.off_.load());
      serializer_.write_binary(raw_ptr, size);
    } else {
      uint8_t mode = 2;
      serializer_ << mode;
    }
  }

  template <typename T>
  CTP_CROSS_FUN void bulk(const ctp::ipc::FullPtr<T> &ptr, size_t size,
                           uint32_t flags) {
    if (!ptr.shm_.alloc_id_.IsNull()) {
      uint8_t mode = 0;
      serializer_ << mode;
      size_t off = ptr.shm_.off_.load();
      serializer_ << off << ptr.shm_.alloc_id_.major_
                  << ptr.shm_.alloc_id_.minor_;
    } else if (flags & BULK_XFER) {
      uint8_t mode = 1;
      serializer_ << mode;
      serializer_.write_binary(reinterpret_cast<const char *>(ptr.ptr_), size);
    } else {
      uint8_t mode = 2;
      serializer_ << mode;
    }
  }

  template <typename T>
  CTP_CROSS_FUN void bulk(T *ptr, size_t size, uint32_t flags) {
    (void)flags;
    serializer_.write_binary(reinterpret_cast<const char *>(ptr), size);
  }

  const LocalTaskInfoVec &GetTaskInfos() const { return task_infos_; }
  CTP_CROSS_FUN LocalMsgType GetMsgType() const { return msg_type_; }

  CTP_CROSS_FUN size_t GetSize() {
    serializer_.Finalize();
    return buffer_.size();
  }

  CTP_CROSS_FUN const BufferT &GetData() {
    serializer_.Finalize();
    return buffer_;
  }

  CTP_CROSS_FUN BufferT &GetMutableData() {
    serializer_.Finalize();
    return buffer_;
  }
};

/**
 * Archive for loading tasks (inputs or outputs) using LocalDeserialize.
 * Templated on BufferT so callers can use any vector-like container.
 *
 * Does NOT own the buffer — callers provide a const reference.
 * Inherits from LbmMeta for ShmTransport::Recv compatibility.
 *
 * @tparam BufferT Buffer type (must support data(), size())
 */
template <typename BufferT>
class LocalLoadTaskArchive : public LocalLbmBase {
  using Base = LocalLbmBase;

 public:
  using is_saving = std::false_type;
  using is_loading = std::true_type;
  using supports_range_ops = std::true_type;
  using buffer_type = BufferT;
  LocalTaskInfoVec task_infos_;
  LocalMsgType msg_type_; /**< Message type: kSerializeIn or kSerializeOut */

 private:
  BufferT &data_;
  ctp::ipc::LocalDeserialize<BufferT> deserializer_;
  size_t current_task_index_;
  bool is_pod_ = false;

 public:
  CTP_CROSS_FUN void PushPod(bool val) { is_pod_ = val; }
  CTP_CROSS_FUN void PopPod() { is_pod_ = false; }

  /** Mark this archive for warp-parallel read_binary calls */
  CTP_INLINE_CROSS_FUN void SetWarpConverged(bool v) {
    deserializer_.warp_converged_ = v;
  }

  /** Reset for reuse with the same buffer */
  CTP_CROSS_FUN void Reset(LocalMsgType msg_type) {
    msg_type_ = msg_type;
    new (&deserializer_) ctp::ipc::LocalDeserialize<BufferT>(data_);
    current_task_index_ = 0;
    task_infos_.resize(0);
  }

  /**
   * Serialize for ShmTransport compatibility.
   */
  template <typename Ar>
  CTP_CROSS_FUN void serialize(Ar &ar) {
    ar(this->send, this->recv, this->send_bulks, this->recv_bulks);
    ar(task_infos_, msg_type_);
    ar(data_);
    new (&deserializer_) ctp::ipc::LocalDeserialize<BufferT>(data_);
  }

  /**
   * Constructor with allocator and external buffer reference.
   * The archive does NOT own the buffer.
   */
  template <typename AllocPtrT>
  CTP_CROSS_FUN LocalLoadTaskArchive(AllocPtrT *alloc,
                                       BufferT &buffer)
      : Base(alloc),
        task_infos_(alloc),
        msg_type_(LocalMsgType::kSerializeIn),
        data_(buffer),
        deserializer_(buffer),
        current_task_index_(0) {}

  /**
   * Constructor with default allocator and external buffer reference.
   */
  CTP_CROSS_FUN explicit LocalLoadTaskArchive(BufferT &buffer)
      : Base(CLIO_PRIV_ALLOC),
        task_infos_(CLIO_PRIV_ALLOC),
        msg_type_(LocalMsgType::kSerializeIn),
        data_(buffer),
        deserializer_(buffer),
        current_task_index_(0) {}

  /** Delete copy/move — reference member prevents safe copy/move */
  LocalLoadTaskArchive(const LocalLoadTaskArchive &) = delete;
  LocalLoadTaskArchive &operator=(const LocalLoadTaskArchive &) = delete;
  LocalLoadTaskArchive(LocalLoadTaskArchive &&) = delete;
  LocalLoadTaskArchive &operator=(LocalLoadTaskArchive &&) = delete;

 public:
  template <typename T>
  CTP_CROSS_FUN LocalLoadTaskArchive &operator>>(T &value) {
    if constexpr (std::is_base_of_v<Task, T>) {
      if (msg_type_ == LocalMsgType::kSerializeIn) {
        value.SerializeIn(*this);
      } else if (msg_type_ == LocalMsgType::kSerializeOut) {
        value.SerializeOut(*this);
      }
    } else {
      deserializer_ >> value;
    }
    return *this;
  }

  template <typename T>
  CTP_CROSS_FUN LocalLoadTaskArchive &operator&(T &value) {
    return *this >> value;
  }

  template <typename T>
  LocalLoadTaskArchive &operator>>(T *&value) {
    if constexpr (std::is_base_of_v<Task, T>) {
      if (msg_type_ == LocalMsgType::kSerializeIn) {
        value->SerializeIn(*this);
      } else if (msg_type_ == LocalMsgType::kSerializeOut) {
        value->SerializeOut(*this);
      }
#if CTP_IS_HOST
      current_task_index_++;
#endif
    } else {
      deserializer_ >> value;
    }
    return *this;
  }

  template <typename... Args>
  CTP_CROSS_FUN void operator()(Args &...args) {
    if (is_pod_) {
      deserializer_.range(args...);
    } else {
      (DeserializeArg(args), ...);
    }
  }

 private:
  template <typename T>
  CTP_CROSS_FUN void DeserializeArg(T &arg) {
    if constexpr (std::is_base_of_v<Task,
                                    std::remove_pointer_t<std::decay_t<T>>>) {
      *this >> arg;
    } else {
      deserializer_ >> arg;
    }
  }

 public:
  CTP_CROSS_FUN void read_binary(char *data, size_t size) {
    deserializer_.read_binary(data, size);
  }

  template <typename FirstT, typename LastT>
  CTP_INLINE_CROSS_FUN void read_range(FirstT *first, LastT *last) {
    deserializer_.read_range(first, last);
  }

  template <typename... Args>
  CTP_INLINE_CROSS_FUN void range(Args &...args) {
    deserializer_.range(args...);
  }

  template <typename T>
  CTP_CROSS_FUN void bulk(ctp::ipc::ShmPtr<T> &ptr, size_t size, uint32_t flags) {
    (void)flags;
    uint8_t mode = 0;
    deserializer_ >> mode;
    if (mode == 1) {
#if CTP_IS_GPU
      ptr.alloc_id_.SetNull();
      ptr.off_ = 0;
#else
      ctp::ipc::FullPtr<char> buf = CTP_MALLOC->AllocateObjs<char>(size);
      deserializer_.read_binary(buf.ptr_, size);
      ptr.off_ = buf.shm_.off_.load();
      ptr.alloc_id_ = buf.shm_.alloc_id_;
#endif
    } else if (mode == 2) {
#if CTP_IS_GPU
      ptr.alloc_id_.SetNull();
      ptr.off_ = 0;
#else
      ctp::ipc::FullPtr<char> buf = CTP_MALLOC->AllocateObjs<char>(size);
      ptr.off_ = buf.shm_.off_.load();
      ptr.alloc_id_ = buf.shm_.alloc_id_;
#endif
    } else if (mode == 3) {
      size_t raw_ptr = 0;
      deserializer_ >> raw_ptr;
      ptr.off_ = raw_ptr;
      ptr.alloc_id_.SetNull();
    } else {
      size_t off = 0;
      u32 major = 0, minor = 0;
      deserializer_ >> off >> major >> minor;
      ptr.off_ = off;
      ptr.alloc_id_ = ctp::ipc::AllocatorId(major, minor);
    }
  }

  template <typename T>
  void bulk(ctp::ipc::FullPtr<T> &ptr, size_t size, uint32_t flags) {
    (void)flags;
    uint8_t mode = 0;
    deserializer_ >> mode;
    if (mode == 1) {
      ctp::ipc::FullPtr<char> buf = CTP_MALLOC->AllocateObjs<char>(size);
      deserializer_.read_binary(buf.ptr_, size);
      ptr.shm_.off_ = buf.shm_.off_.load();
      ptr.shm_.alloc_id_ = buf.shm_.alloc_id_;
      ptr.ptr_ = reinterpret_cast<T *>(buf.ptr_);
    } else if (mode == 2) {
      ctp::ipc::FullPtr<char> buf = CTP_MALLOC->AllocateObjs<char>(size);
      ptr.shm_.off_ = buf.shm_.off_.load();
      ptr.shm_.alloc_id_ = buf.shm_.alloc_id_;
      ptr.ptr_ = reinterpret_cast<T *>(buf.ptr_);
    } else {
      size_t off = 0;
      u32 major = 0, minor = 0;
      deserializer_ >> off >> major >> minor;
      ptr.shm_.off_ = off;
      ptr.shm_.alloc_id_ = ctp::ipc::AllocatorId(major, minor);
    }
  }

  template <typename T>
  CTP_CROSS_FUN void bulk(T *ptr, size_t size, uint32_t flags) {
    (void)flags;
    deserializer_.read_binary(reinterpret_cast<char *>(ptr), size);
  }

  const LocalTaskInfoVec &GetTaskInfos() const { return task_infos_; }
  const LocalTaskInfo &GetCurrentTaskInfo() const {
    return task_infos_[current_task_index_];
  }
  CTP_CROSS_FUN LocalMsgType GetMsgType() const { return msg_type_; }
  void ResetTaskIndex() { current_task_index_ = 0; }
  CTP_CROSS_FUN void SetMsgType(LocalMsgType msg_type) {
    msg_type_ = msg_type;
  }
};

/** Default archive type aliases using clio::run::priv::vector<char> buffer */
using DefaultSaveArchive = LocalSaveTaskArchive<clio::run::priv::vector<char>>;
using DefaultLoadArchive = LocalLoadTaskArchive<clio::run::priv::vector<char>>;

/** Wrap-vector archive aliases for zero-copy GPU transport */
using WrapSaveArchive = LocalSaveTaskArchive<ctp::priv::wrap_vector>;
using WrapLoadArchive = LocalLoadTaskArchive<ctp::priv::wrap_vector>;

/**
 * Lightweight GPU save archive — no LbmMeta, no task_infos vectors.
 * Serializes directly into a wrap_vector bound to copy_space.
 * ShmPtr/FullPtr are copied as-is (same GPU, same address space).
 */
class GpuSaveTaskArchive {
 public:
  using is_saving = std::true_type;
  using is_loading = std::false_type;
  using supports_range_ops = std::true_type;

  LocalMsgType msg_type_;
  ctp::priv::wrap_vector &buffer_;
  ctp::ipc::LocalSerialize<ctp::priv::wrap_vector> serializer_;
  bool is_pod_ = false;

  CTP_CROSS_FUN void PushPod(bool val) { is_pod_ = val; }
  CTP_CROSS_FUN void PopPod() { is_pod_ = false; }

  CTP_CROSS_FUN GpuSaveTaskArchive(LocalMsgType msg_type,
                                      ctp::priv::wrap_vector &buffer)
      : msg_type_(msg_type), buffer_(buffer), serializer_(buffer) {}

  CTP_CROSS_FUN void Reset(LocalMsgType msg_type) {
    msg_type_ = msg_type;
    serializer_.cur_off_ = 0;
  }

  CTP_INLINE_CROSS_FUN void SetWarpConverged(bool v) {
    serializer_.warp_converged_ = v;
  }

  CTP_INLINE_CROSS_FUN size_t GetSerializedSize() const {
    return serializer_.cur_off_;
  }

  CTP_CROSS_FUN LocalMsgType GetMsgType() const { return msg_type_; }

  template <typename T>
  CTP_CROSS_FUN GpuSaveTaskArchive &operator<<(T &value) {
    serializer_ << value;
    return *this;
  }

  template <typename T>
  CTP_CROSS_FUN GpuSaveTaskArchive &operator&(T &value) {
    return *this << value;
  }

  template <typename... Args>
  CTP_CROSS_FUN void operator()(Args &...args) {
    if (is_pod_) {
      serializer_.range(args...);
    } else {
      (void)((serializer_ << args), ...);
    }
  }

  CTP_CROSS_FUN void write_binary(const char *data, size_t size) {
    serializer_.write_binary(data, size);
  }

  CTP_CROSS_FUN void save_string_fused(const char *str_data, size_t len) {
    serializer_.save_string_fused(str_data, len);
  }

  template <typename FirstT, typename LastT>
  CTP_INLINE_CROSS_FUN void write_range(const FirstT *first,
                                          const LastT *last) {
    serializer_.write_range(first, last);
  }

  template <typename... Args>
  CTP_INLINE_CROSS_FUN void range(Args &...args) {
    serializer_.range(args...);
  }

  /** GPU bulk: ShmPtr copied as-is (same address space) */
  template <typename T>
  CTP_CROSS_FUN void bulk(ctp::ipc::ShmPtr<T> ptr, size_t size, uint32_t flags) {
    (void)size; (void)flags;
    serializer_.write_binary(reinterpret_cast<const char *>(&ptr), sizeof(ptr));
  }

  /** GPU bulk: FullPtr copied as-is */
  template <typename T>
  CTP_CROSS_FUN void bulk(const ctp::ipc::FullPtr<T> &ptr, size_t size,
                           uint32_t flags) {
    (void)size; (void)flags;
    serializer_.write_binary(reinterpret_cast<const char *>(&ptr), sizeof(ptr));
  }

  /** GPU bulk: raw pointer — copy data inline */
  template <typename T>
  CTP_CROSS_FUN void bulk(T *ptr, size_t size, uint32_t flags) {
    (void)flags;
    serializer_.write_binary(reinterpret_cast<const char *>(ptr), size);
  }
};

/**
 * Lightweight GPU load archive — no LbmMeta, no task_infos vectors.
 * Deserializes from a wrap_vector bound to copy_space.
 * ShmPtr/FullPtr are copied as-is (same GPU, same address space).
 */
class GpuLoadTaskArchive {
 public:
  using is_saving = std::false_type;
  using is_loading = std::true_type;
  using supports_range_ops = std::true_type;

  LocalMsgType msg_type_;
  ctp::priv::wrap_vector &data_;
  ctp::ipc::LocalDeserialize<ctp::priv::wrap_vector> deserializer_;
  bool is_pod_ = false;

  CTP_CROSS_FUN void PushPod(bool val) { is_pod_ = val; }
  CTP_CROSS_FUN void PopPod() { is_pod_ = false; }

  CTP_CROSS_FUN GpuLoadTaskArchive(ctp::priv::wrap_vector &buffer)
      : msg_type_(LocalMsgType::kSerializeIn), data_(buffer),
        deserializer_(buffer) {}

  CTP_CROSS_FUN void Reset(LocalMsgType msg_type) {
    msg_type_ = msg_type;
    deserializer_.cur_off_ = 0;
  }

  CTP_CROSS_FUN void SetMsgType(LocalMsgType msg_type) {
    msg_type_ = msg_type;
  }

  CTP_INLINE_CROSS_FUN void SetWarpConverged(bool v) {
    deserializer_.warp_converged_ = v;
  }

  CTP_CROSS_FUN LocalMsgType GetMsgType() const { return msg_type_; }

  template <typename T>
  CTP_CROSS_FUN GpuLoadTaskArchive &operator>>(T &value) {
    deserializer_ >> value;
    return *this;
  }

  template <typename T>
  CTP_CROSS_FUN GpuLoadTaskArchive &operator&(T &value) {
    return *this >> value;
  }

  template <typename... Args>
  CTP_CROSS_FUN void operator()(Args &...args) {
    if (is_pod_) {
      deserializer_.range(args...);
    } else {
      (void)((deserializer_ >> args), ...);
    }
  }

  CTP_CROSS_FUN void read_binary(char *data, size_t size) {
    deserializer_.read_binary(data, size);
  }

  template <typename FirstT, typename LastT>
  CTP_INLINE_CROSS_FUN void read_range(FirstT *first, LastT *last) {
    deserializer_.read_range(first, last);
  }

  template <typename... Args>
  CTP_INLINE_CROSS_FUN void range(Args &...args) {
    deserializer_.range(args...);
  }

  /** GPU bulk: ShmPtr copied as-is (same address space) */
  template <typename T>
  CTP_CROSS_FUN void bulk(ctp::ipc::ShmPtr<T> &ptr, size_t size,
                           uint32_t flags) {
    (void)size; (void)flags;
    deserializer_.read_binary(reinterpret_cast<char *>(&ptr), sizeof(ptr));
  }

  /** GPU bulk: FullPtr copied as-is */
  template <typename T>
  CTP_CROSS_FUN void bulk(ctp::ipc::FullPtr<T> &ptr, size_t size,
                           uint32_t flags) {
    (void)size; (void)flags;
    deserializer_.read_binary(reinterpret_cast<char *>(&ptr), sizeof(ptr));
  }

  /** GPU bulk: raw pointer — read data inline */
  template <typename T>
  CTP_CROSS_FUN void bulk(T *ptr, size_t size, uint32_t flags) {
    (void)flags;
    deserializer_.read_binary(reinterpret_cast<char *>(ptr), size);
  }
};


}  // namespace clio::run

#endif  // CLIO_RUNTIME_INCLUDE_LOCAL_TASK_ARCHIVES_H_
