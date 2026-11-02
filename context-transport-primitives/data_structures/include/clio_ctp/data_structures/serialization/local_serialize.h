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

//
// Created by llogan on 11/27/24.
//

#ifndef CTP_SHM_INCLUDE_HSHM_SHM_DATA_STRUCTURES_SERIALIZATION_LOCAL_SERIALIZE_H_
#define CTP_SHM_INCLUDE_HSHM_SHM_DATA_STRUCTURES_SERIALIZATION_LOCAL_SERIALIZE_H_

#include "clio_ctp/constants/macros.h"
#include "clio_ctp/types/argpack.h"
#include "clio_ctp/util/logging.h"
// #include "clio_ctp/data_structures/all.h"  // Deleted during hard
// refactoring
#include <cstring>
#include <list>
#include <string>
#include <unordered_map>
#include <vector>

#include "serialize_common.h"

namespace ctp::ipc {


/** Save string */
template <typename Ar>
CTP_INLINE_CROSS_FUN void save(Ar &ar, const std::string &str) {
  save_string(ar, str);
}

/** Load string */
template <typename Ar>
CTP_INLINE_CROSS_FUN void load(Ar &ar, std::string &str) {
  load_string(ar, str);
}

/** Save vector */
template <typename Ar, typename T>
CTP_INLINE_CROSS_FUN void save(Ar &ar, const std::vector<T> &data) {
  save_vec<Ar, std::vector<T>, T>(ar, data);
}

/** Load vector */
template <typename Ar, typename T>
CTP_INLINE_CROSS_FUN void load(Ar &ar, std::vector<T> &data) {
  load_vec<Ar, std::vector<T>, T>(ar, data);
}

/** Save list */
template <typename Ar, typename T>
CTP_INLINE_CROSS_FUN void save(Ar &ar, const std::list<T> &data) {
  save_list<Ar, std::list<T>, T>(ar, data);
}

/** Load list */
template <typename Ar, typename T>
CTP_INLINE_CROSS_FUN void load(Ar &ar, std::list<T> &data) {
  load_list<Ar, std::list<T>, T>(ar, data);
}

/** Save unordered_map */
template <typename Ar, typename KeyT, typename T>
CTP_INLINE_CROSS_FUN void save(Ar &ar, const std::unordered_map<KeyT, T> &data) {
  save_map<Ar, std::unordered_map<KeyT, T>, KeyT, T>(ar, data);
}

/** Load unordered_map */
template <typename Ar, typename KeyT, typename T>
CTP_INLINE_CROSS_FUN void load(Ar &ar, std::unordered_map<KeyT, T> &data) {
  load_map<Ar, std::unordered_map<KeyT, T>, KeyT, T>(ar, data);
}

/**
 * Dry-run archive that computes the serialized size without copying data.
 * Implements the same save-side API as LocalSerialize so that the same
 * SerializeIn / SerializeOut code paths work unchanged.
 */
class CalculateSizeArchive {
 public:
  using is_loading = std::false_type;
  using is_saving = std::true_type;
  using supports_range_ops = std::true_type;

  size_t cur_off_ = 0;

 public:
  CTP_CROSS_FUN CalculateSizeArchive() = default;

  /** Get the total computed size */
  CTP_INLINE_CROSS_FUN size_t size() const { return cur_off_; }

  /** left shift operator */
  template <typename T>
  CTP_INLINE_CROSS_FUN CalculateSizeArchive &operator<<(const T &obj) {
    return base(obj);
  }

  /** & operator */
  template <typename T>
  CTP_INLINE_CROSS_FUN CalculateSizeArchive &operator&(const T &obj) {
    return base(obj);
  }

  /** Call operator */
  template <typename... Args>
  CTP_INLINE_CROSS_FUN CalculateSizeArchive &operator()(Args &&...args) {
    ctp::ForwardIterateArgpack::Apply(
        ctp::make_argpack(std::forward<Args>(args)...),
        [this](auto i, auto &arg) { this->base(arg); });
    return *this;
  }

  /** range() — compute span size from first to last arg (contiguous POD) */
  template <typename... Args>
  CTP_INLINE_CROSS_FUN CalculateSizeArchive &range(Args &...args) {
    const char *begin = reinterpret_cast<const char *>(
        &std::get<0>(std::forward_as_tuple(args...)));
    auto &last = std::get<sizeof...(Args) - 1>(std::forward_as_tuple(args...));
    const char *end = reinterpret_cast<const char *>(&last) +
                       sizeof(std::decay_t<decltype(last)>);
    cur_off_ += static_cast<size_t>(end - begin);
    return *this;
  }

  /** Size-compute function */
  template <typename T>
  CTP_INLINE_CROSS_FUN CalculateSizeArchive &base(const T &obj) {
    if constexpr (std::is_arithmetic<T>::value) {
      cur_off_ += sizeof(T);
    } else if constexpr (std::is_enum<T>::value) {
      cur_off_ += sizeof(std::underlying_type_t<T>);
    } else if constexpr (has_serialize_fun_v<CalculateSizeArchive, T>) {
      serialize(*this, const_cast<T &>(obj));
    } else if constexpr (has_save_fun_v<CalculateSizeArchive, T>) {
      save(*this, obj);
    } else if constexpr (has_serialize_cls_v<CalculateSizeArchive, T>) {
      const_cast<T &>(obj).serialize(*this);
    } else if constexpr (has_save_cls_v<CalculateSizeArchive, T>) {
      obj.save(*this);
    }
    return *this;
  }

  /** write_range — compute span size */
  template <typename FirstT, typename LastT>
  CTP_INLINE_CROSS_FUN void write_range(const FirstT *first,
                                          const LastT *last) {
    const char *begin = reinterpret_cast<const char *>(first);
    const char *end = reinterpret_cast<const char *>(last) + sizeof(LastT);
    cur_off_ += static_cast<size_t>(end - begin);
  }

  /** write_binary — just accumulate size */
  CTP_INLINE_CROSS_FUN
  CalculateSizeArchive &write_binary(const char *data, size_t size) {
    (void)data;
    cur_off_ += size;
    return *this;
  }

  /** Fused string save — just accumulate sizeof(size_t) + len */
  CTP_INLINE_CROSS_FUN
  void save_string_fused(const char *str_data, size_t len) {
    (void)str_data;
    cur_off_ += sizeof(size_t) + len;
  }
};

/** Save string (CalculateSizeArchive overload) */
CTP_INLINE_CROSS_FUN void save(CalculateSizeArchive &ar,
                                 const std::string &str) {
  save_string(ar, str);
}

/** Save vector (CalculateSizeArchive overload) */
template <typename T>
CTP_INLINE_CROSS_FUN void save(CalculateSizeArchive &ar,
                                 const std::vector<T> &data) {
  save_vec<CalculateSizeArchive, std::vector<T>, T>(ar, data);
}

/** Save list (CalculateSizeArchive overload) */
template <typename T>
CTP_INLINE_CROSS_FUN void save(CalculateSizeArchive &ar,
                                 const std::list<T> &data) {
  save_list<CalculateSizeArchive, std::list<T>, T>(ar, data);
}

/** Save unordered_map (CalculateSizeArchive overload) */
template <typename KeyT, typename T>
CTP_INLINE_CROSS_FUN void save(CalculateSizeArchive &ar,
                                 const std::unordered_map<KeyT, T> &data) {
  save_map<CalculateSizeArchive, std::unordered_map<KeyT, T>, KeyT, T>(ar,
                                                                        data);
}

/** A class for serializing simple objects into private memory */
template <typename DataT = std::vector<char>>
class LocalSerialize {
 public:
  using is_loading = std::false_type;
  using is_saving = std::true_type;
  using supports_range_ops = std::true_type;

  DataT &data_;
  size_t cur_off_ = 0;
  bool warp_converged_ = false;

 public:
  CTP_CROSS_FUN LocalSerialize(DataT &data) : data_(data), cur_off_(0), warp_converged_(false) {}
  CTP_CROSS_FUN LocalSerialize(DataT &data, bool) : data_(data), cur_off_(data.size()), warp_converged_(false) {}

  /** Commit the local offset to the vector's size. Must be called when
   *  serialization is complete so that data_.size() reflects what was written. */
  CTP_INLINE_CROSS_FUN void Finalize() {
    data_.resize(cur_off_);
  }

  /** left shift operator */
  template <typename T>
  CTP_INLINE_CROSS_FUN LocalSerialize &operator<<(const T &obj) {
    return base(obj);
  }

  /** & operator */
  template <typename T>
  CTP_INLINE_CROSS_FUN LocalSerialize &operator&(const T &obj) {
    return base(obj);
  }

  /** Call operator */
  template <typename... Args>
  CTP_INLINE_CROSS_FUN LocalSerialize &operator()(Args &&...args) {
    ctp::ForwardIterateArgpack::Apply(
        ctp::make_argpack(std::forward<Args>(args)...),
        [this](auto i, auto &arg) { this->base(arg); });
    return *this;
  }

  /** range() — batch memcpy from &first_arg through &last_arg (inclusive).
   *  All args must be contiguous POD fields in memory.
   *  The first parameter is the beginning and the last is the end. */
  template <typename... Args>
  CTP_INLINE_CROSS_FUN LocalSerialize &range(Args &...args) {
    const char *begin = reinterpret_cast<const char *>(
        &std::get<0>(std::forward_as_tuple(args...)));
    auto &last = std::get<sizeof...(Args) - 1>(std::forward_as_tuple(args...));
    const char *end = reinterpret_cast<const char *>(&last) +
                       sizeof(std::decay_t<decltype(last)>);
    write_binary(begin, static_cast<size_t>(end - begin));
    return *this;
  }

  /** Save function */
  template <typename T>
  CTP_INLINE_CROSS_FUN LocalSerialize &base(const T &obj) {
    STATIC_ASSERT((is_serializeable_v<LocalSerialize, T>),
                  "Cannot serialize object", void);
    if constexpr (std::is_arithmetic<T>::value) {
      write_binary(reinterpret_cast<const char *>(&obj), sizeof(T));
    } else if constexpr (std::is_enum<T>::value) {
      using UnderlyingType = std::underlying_type_t<T>;
      UnderlyingType value = static_cast<UnderlyingType>(obj);
      write_binary(reinterpret_cast<const char *>(&value),
                   sizeof(UnderlyingType));
    } else if constexpr (has_serialize_fun_v<LocalSerialize, T>) {
      serialize(*this, const_cast<T &>(obj));
    } else if constexpr (has_load_save_fun_v<LocalSerialize, T>) {
      save(*this, obj);
    } else if constexpr (has_serialize_cls_v<LocalSerialize, T>) {
      const_cast<T &>(obj).serialize(*this);
    } else if constexpr (has_load_save_cls_v<LocalSerialize, T>) {
      obj.save(*this);
    }
    return *this;
  }

  /** Batch-serialize a contiguous range of POD fields in one memcpy.
   *  @param first Pointer to the first field
   *  @param last  Pointer to the last field
   *  The range is [first, last] inclusive (i.e. last field IS included).
   *  Example: ar.write_range(&obj.field_a_, &obj.field_z_);
   */
  template <typename FirstT, typename LastT>
  CTP_INLINE_CROSS_FUN LocalSerialize &write_range(const FirstT *first,
                                                     const LastT *last) {
    const char *begin = reinterpret_cast<const char *>(first);
    const char *end = reinterpret_cast<const char *>(last) + sizeof(LastT);
    write_binary(begin, static_cast<size_t>(end - begin));
    return *this;
  }

  /** Fused string save: size prefix + character data in one capacity check.
   *  Uses direct store for size prefix to avoid memcpy overhead on GPU. */
  CTP_INLINE_CROSS_FUN
  void save_string_fused(const char *str_data, size_t len) {
    size_t total = sizeof(size_t) + len;
    size_t new_off = cur_off_ + total;
    if (new_off > data_.size()) {
      size_t new_cap = data_.size();
      if (new_cap == 0) new_cap = 64;
      while (new_cap < new_off) new_cap *= 2;
      data_.resize(new_cap);
    }
    char *dst = data_.data() + cur_off_;
    memcpy(dst, &len, sizeof(size_t));
    if (len > 0) {
      memcpy(dst + sizeof(size_t), str_data, len);
    }
    cur_off_ = new_off;
  }

  /** Save function (binary data).
   *  GPU: warp-parallel when warp_converged_, sequential otherwise. */
  CTP_INLINE_CROSS_FUN
  LocalSerialize &write_binary(const char *data, size_t size) {
#if CTP_IS_GPU
    if (warp_converged_) {
      uint32_t lane_id = threadIdx.x & 31;
      if (lane_id == 0) {
        size_t new_off = cur_off_ + size;
        if (new_off > data_.size()) {
          size_t new_cap = data_.size();
          if (new_cap == 0) new_cap = 64;
          while (new_cap < new_off) new_cap *= 2;
          data_.resize(new_cap);
        }
      }
      __syncwarp();
      char *dst = data_.data() + cur_off_;
      for (size_t i = lane_id * 8; i < size; i += 32 * 8) {
        size_t chunk = (i + 8 <= size) ? 8 : (size - i);
        memcpy(dst + i, data + i, chunk);
      }
      __syncwarp();
      if (lane_id == 0) {
        cur_off_ += size;
      }
      __syncwarp();
    } else
#endif
    {
      size_t new_off = cur_off_ + size;
      if (new_off > data_.size()) {
        size_t new_cap = data_.size();
        if (new_cap == 0) new_cap = 64;
        while (new_cap < new_off) new_cap *= 2;
        data_.resize(new_cap);
      }
      if (size > 0) {
        memcpy(data_.data() + cur_off_, data, size);
      }
      cur_off_ = new_off;
    }
    return *this;
  }
};

/** A class for serializing simple objects into private memory */
template <typename DataT = std::vector<char>>
class LocalDeserialize {
 public:
  using is_loading = std::true_type;
  using is_saving = std::false_type;
  using supports_range_ops = std::true_type;

  const DataT &data_;
  size_t cur_off_ = 0;
  bool warp_converged_ = false;

 public:
  CTP_CROSS_FUN LocalDeserialize(const DataT &data) : data_(data), cur_off_(0), warp_converged_(false) {}

  /** right shift operator */
  template <typename T>
  CTP_INLINE_CROSS_FUN LocalDeserialize &operator>>(T &obj) {
    return base(obj);
  }

  /** & operator */
  template <typename T>
  CTP_INLINE_CROSS_FUN LocalDeserialize &operator&(T &obj) {
    return base(obj);
  }

  /** Call operator */
  template <typename... Args>
  CTP_INLINE_CROSS_FUN LocalDeserialize &operator()(Args &&...args) {
    ctp::ForwardIterateArgpack::Apply(
        ctp::make_argpack(std::forward<Args>(args)...),
        [this](auto i, auto &arg) { this->base(arg); });
    return *this;
  }

  /** range() — batch memcpy from &first_arg through &last_arg (inclusive).
   *  All args must be contiguous POD fields in memory.
   *  The first parameter is the beginning and the last is the end. */
  template <typename... Args>
  CTP_INLINE_CROSS_FUN LocalDeserialize &range(Args &...args) {
    char *begin = reinterpret_cast<char *>(
        &std::get<0>(std::forward_as_tuple(args...)));
    auto &last = std::get<sizeof...(Args) - 1>(std::forward_as_tuple(args...));
    char *end = reinterpret_cast<char *>(&last) +
                 sizeof(std::decay_t<decltype(last)>);
    read_binary(begin, static_cast<size_t>(end - begin));
    return *this;
  }

  /** Load function */
  template <typename T>
  CTP_INLINE_CROSS_FUN LocalDeserialize &base(T &obj) {
    STATIC_ASSERT((is_serializeable_v<LocalDeserialize, T>),
                  "Cannot serialize object", void);
    if constexpr (std::is_arithmetic<T>::value) {
      read_binary(reinterpret_cast<char *>(&obj), sizeof(T));
    } else if constexpr (std::is_enum<T>::value) {
      using UnderlyingType = std::underlying_type_t<T>;
      UnderlyingType value{};
      read_binary(reinterpret_cast<char *>(&value), sizeof(UnderlyingType));
      obj = static_cast<T>(value);
    } else if constexpr (has_serialize_fun_v<LocalDeserialize, T>) {
      serialize(*this, obj);
    } else if constexpr (has_load_save_fun_v<LocalDeserialize, T>) {
      load(*this, obj);
    } else if constexpr (has_serialize_cls_v<LocalDeserialize, T>) {
      obj.serialize(*this);
    } else if constexpr (has_load_save_cls_v<LocalDeserialize, T>) {
      obj.load(*this);
    }
    return *this;
  }

  /** Batch-deserialize a contiguous range of POD fields in one memcpy.
   *  @param first Pointer to the first field
   *  @param last  Pointer to the last field
   *  The range is [first, last] inclusive (i.e. last field IS included).
   *  Example: ar.read_range(&obj.field_a_, &obj.field_z_);
   */
  template <typename FirstT, typename LastT>
  CTP_INLINE_CROSS_FUN LocalDeserialize &read_range(FirstT *first,
                                                      LastT *last) {
    char *begin = reinterpret_cast<char *>(first);
    char *end = reinterpret_cast<char *>(last) + sizeof(LastT);
    read_binary(begin, static_cast<size_t>(end - begin));
    return *this;
  }

  /** Load function (binary data).
   *  GPU: warp-parallel when warp_converged_, sequential otherwise. */
  CTP_INLINE_CROSS_FUN
  LocalDeserialize &read_binary(char *data, size_t size) {
#if CTP_IS_GPU
    if (warp_converged_) {
      uint32_t lane_id = threadIdx.x & 31;
      if (lane_id == 0) {
        if (cur_off_ + size > data_.size()) {
          size = 0;
        }
      }
      __syncwarp();
      const char *src = data_.data() + cur_off_;
      for (size_t i = lane_id * 8; i < size; i += 32 * 8) {
        size_t chunk = (i + 8 <= size) ? 8 : (size - i);
        memcpy(data + i, src + i, chunk);
      }
      __syncwarp();
      if (lane_id == 0) {
        cur_off_ += size;
      }
      __syncwarp();
    } else
#endif
    {
      if (cur_off_ + size > data_.size()) {
#if CTP_IS_HOST
        HLOG(kError,
             "LocalDeserialize::read_binary: Attempted to read beyond end of "
             "data");
#endif
        return *this;
      }
      if (size > 0) {
        memcpy(data, data_.data() + cur_off_, size);
      }
      cur_off_ += size;
    }
    return *this;
  }
};

}  // namespace ctp::ipc

#endif  // CTP_SHM_INCLUDE_HSHM_SHM_DATA_STRUCTURES_SERIALIZATION_LOCAL_SERIALIZE_H_
