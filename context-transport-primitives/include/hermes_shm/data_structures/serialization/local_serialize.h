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

#ifndef HSHM_SHM_INCLUDE_HSHM_SHM_DATA_STRUCTURES_SERIALIZATION_LOCAL_SERIALIZE_H_
#define HSHM_SHM_INCLUDE_HSHM_SHM_DATA_STRUCTURES_SERIALIZATION_LOCAL_SERIALIZE_H_

#include "hermes_shm/constants/macros.h"
#include "hermes_shm/types/argpack.h"
#include "hermes_shm/util/logging.h"
// #include "hermes_shm/data_structures/all.h"  // Deleted during hard
// refactoring
#include <cstring>
#include <list>
#include <string>
#include <unordered_map>
#include <vector>

#include "serialize_common.h"

namespace hshm::ipc {

#if HSHM_ENABLE_CEREAL
/** Save cereal binary archive */
template <typename Ar, typename T>
void save(Ar &ar, const cereal::BinaryData<T> &data) {
  ar.write_binary((const char *)data.data, (size_t)data.size);
}

/** Load cereal binary archive */
template <typename Ar, typename T>
void load(Ar &ar, cereal::BinaryData<T> &data) {
  ar.read_binary((char *)data.data, (size_t)data.size);
}
#endif

/** Save string */
template <typename Ar>
void save(Ar &ar, const std::string &str) {
  save_string(ar, str);
}

/** Load string */
template <typename Ar>
void load(Ar &ar, std::string &str) {
  load_string(ar, str);
}

/** Save vector */
template <typename Ar, typename T>
void save(Ar &ar, const std::vector<T> &data) {
  save_vec<Ar, std::vector<T>, T>(ar, data);
}

/** Load vector */
template <typename Ar, typename T>
void load(Ar &ar, std::vector<T> &data) {
  load_vec<Ar, std::vector<T>, T>(ar, data);
}

/** Save list */
template <typename Ar, typename T>
void save(Ar &ar, const std::list<T> &data) {
  save_list<Ar, std::list<T>, T>(ar, data);
}

/** Load list */
template <typename Ar, typename T>
void load(Ar &ar, std::list<T> &data) {
  load_list<Ar, std::list<T>, T>(ar, data);
}

/** Save unordered_map */
template <typename Ar, typename KeyT, typename T>
void save(Ar &ar, const std::unordered_map<KeyT, T> &data) {
  save_map<Ar, std::unordered_map<KeyT, T>, KeyT, T>(ar, data);
}

/** Load list */
template <typename Ar, typename KeyT, typename T>
void load(Ar &ar, std::unordered_map<KeyT, T> &data) {
  load_map<Ar, std::unordered_map<KeyT, T>, KeyT, T>(ar, data);
}

/** A class for serializing simple objects into private memory */
template <typename DataT = std::vector<char>>
class LocalSerialize {
 public:
  using is_loading = std::false_type;
  using is_saving = std::true_type;

  DataT &data_;

 public:
  HSHM_CROSS_FUN LocalSerialize(DataT &data) : data_(data) { data_.resize(0); }
  HSHM_CROSS_FUN LocalSerialize(DataT &data, bool) : data_(data) {}

  /** left shift operator */
  template <typename T>
  HSHM_INLINE_CROSS_FUN LocalSerialize &operator<<(const T &obj) {
    return base(obj);
  }

  /** & operator */
  template <typename T>
  HSHM_INLINE_CROSS_FUN LocalSerialize &operator&(const T &obj) {
    return base(obj);
  }

  /** Call operator */
  template <typename... Args>
  HSHM_INLINE_CROSS_FUN LocalSerialize &operator()(Args &&...args) {
    hshm::ForwardIterateArgpack::Apply(
        hshm::make_argpack(std::forward<Args>(args)...),
        [this](auto i, auto &arg) { this->base(arg); });
    return *this;
  }

  /** Save function */
  template <typename T>
  HSHM_INLINE_CROSS_FUN LocalSerialize &base(const T &obj) {
    STATIC_ASSERT((is_serializeable_v<LocalSerialize, T>),
                  "Cannot serialize object", void);
    if constexpr (std::is_arithmetic<T>::value) {
      write_binary(reinterpret_cast<const char *>(&obj), sizeof(T));
    } else if constexpr (std::is_enum<T>::value) {
      // Serialize enums as their underlying type
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

  /** Save function (binary data) */
  HSHM_INLINE_CROSS_FUN
  LocalSerialize &write_binary(const char *data, size_t size) {
    size_t off = data_.size();
    data_.resize(off + size);
    if (size > 0) {
      memcpy(data_.data() + off, data, size);
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

  const DataT &data_;
  size_t cur_off_ = 0;

 public:
  HSHM_CROSS_FUN LocalDeserialize(const DataT &data) : data_(data) { cur_off_ = 0; }

  /** right shift operator */
  template <typename T>
  HSHM_INLINE_CROSS_FUN LocalDeserialize &operator>>(T &obj) {
    return base(obj);
  }

  /** & operator */
  template <typename T>
  HSHM_INLINE_CROSS_FUN LocalDeserialize &operator&(T &obj) {
    return base(obj);
  }

  /** Call operator */
  template <typename... Args>
  HSHM_INLINE_CROSS_FUN LocalDeserialize &operator()(Args &&...args) {
    hshm::ForwardIterateArgpack::Apply(
        hshm::make_argpack(std::forward<Args>(args)...),
        [this](auto i, auto &arg) { this->base(arg); });
    return *this;
  }

  /** Load function */
  template <typename T>
  HSHM_INLINE_CROSS_FUN LocalDeserialize &base(T &obj) {
    STATIC_ASSERT((is_serializeable_v<LocalDeserialize, T>),
                  "Cannot serialize object", void);
    if constexpr (std::is_arithmetic<T>::value) {
      read_binary(reinterpret_cast<char *>(&obj), sizeof(T));
    } else if constexpr (std::is_enum<T>::value) {
      // Deserialize enums from their underlying type
      using UnderlyingType = std::underlying_type_t<T>;
      UnderlyingType value;
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

  /** Load function (binary data) */
  HSHM_INLINE_CROSS_FUN
  LocalDeserialize &read_binary(char *data, size_t size) {
    if (cur_off_ + size > data_.size()) {
#if HSHM_IS_HOST
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
    return *this;
  }
};

}  // namespace hshm::ipc

#endif  // HSHM_SHM_INCLUDE_HSHM_SHM_DATA_STRUCTURES_SERIALIZATION_LOCAL_SERIALIZE_H_
