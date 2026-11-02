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

#ifndef CTP_SHM_INCLUDE_HSHM_SHM_DATA_STRUCTURES_SERIALIZATION_GLOBAL_SERIALIZE_H_
#define CTP_SHM_INCLUDE_HSHM_SHM_DATA_STRUCTURES_SERIALIZATION_GLOBAL_SERIALIZE_H_

#include "clio_ctp/constants/macros.h"
#include "clio_ctp/types/argpack.h"
#include "clio_ctp/util/logging.h"
#include <cstring>
#include <list>
#include <string>
#include <unordered_map>
#include <vector>

#include "serialize_common.h"

namespace ctp::ipc {

/**
 * Architecture-portable binary serializer.
 *
 * Serializes each field individually (no batch memcpy across struct fields)
 * so that the wire format is independent of struct layout, padding, and
 * alignment on any particular architecture.
 *
 * Uses std::vector<char> as the output buffer (host-only).
 */
template <typename DataT = std::vector<char>>
class GlobalSerialize {
 public:
  using is_loading = std::false_type;
  using is_saving = std::true_type;
  using supports_range_ops = std::true_type;

  DataT &data_;
  size_t cur_off_ = 0;

 public:
  GlobalSerialize(DataT &data) : data_(data), cur_off_(0) {
    data_.resize(0);
  }
  GlobalSerialize(DataT &data, bool) : data_(data), cur_off_(data.size()) {}

  /** Commit the local offset to the vector's size. */
  CTP_INLINE void Finalize() {
    data_.resize(cur_off_);
  }

  /** left shift operator */
  template <typename T>
  CTP_INLINE GlobalSerialize &operator<<(const T &obj) {
    return base(obj);
  }

  /** & operator */
  template <typename T>
  CTP_INLINE GlobalSerialize &operator&(const T &obj) {
    return base(obj);
  }

  /** Call operator */
  template <typename... Args>
  CTP_INLINE GlobalSerialize &operator()(Args &&...args) {
    ctp::ForwardIterateArgpack::Apply(
        ctp::make_argpack(std::forward<Args>(args)...),
        [this](auto i, auto &arg) { this->base(arg); });
    return *this;
  }

  /** range() — portable: serialize each field individually */
  template <typename... Args>
  CTP_INLINE GlobalSerialize &range(Args &&...args) {
    return (*this)(std::forward<Args>(args)...);
  }

  /** Save function */
  template <typename T>
  CTP_INLINE GlobalSerialize &base(const T &obj) {
    if constexpr (std::is_arithmetic<T>::value) {
      write_binary(reinterpret_cast<const char *>(&obj), sizeof(T));
    } else if constexpr (std::is_enum<T>::value) {
      using UnderlyingType = std::underlying_type_t<T>;
      UnderlyingType value = static_cast<UnderlyingType>(obj);
      write_binary(reinterpret_cast<const char *>(&value),
                   sizeof(UnderlyingType));
    } else if constexpr (has_serialize_fun_v<GlobalSerialize, T>) {
      serialize(*this, const_cast<T &>(obj));
    } else if constexpr (has_save_fun_v<GlobalSerialize, T>) {
      save(*this, obj);
    } else if constexpr (has_serialize_cls_v<GlobalSerialize, T>) {
      const_cast<T &>(obj).serialize(*this);
    } else if constexpr (has_save_cls_v<GlobalSerialize, T>) {
      obj.save(*this);
    }
    return *this;
  }

  /** write_range — portable: delegates to per-field range() */
  template <typename FirstT, typename LastT>
  CTP_INLINE void write_range(const FirstT *first, const LastT *last) {
    // GlobalSerialize does NOT batch — write_range should not be called
    // directly. Use range() instead which serializes per-field.
    // Fallback: treat as raw binary for backwards compatibility.
    const char *begin = reinterpret_cast<const char *>(first);
    const char *end = reinterpret_cast<const char *>(last) + sizeof(LastT);
    write_binary(begin, static_cast<size_t>(end - begin));
  }

  /** Save function (binary data) */
  CTP_INLINE
  GlobalSerialize &write_binary(const char *data, size_t size) {
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
    return *this;
  }

  /** Fused string save: size prefix + character data in one capacity check.
   *  Uses direct store for size prefix to avoid memcpy overhead. */
  CTP_INLINE
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
};

/**
 * Architecture-portable binary deserializer.
 *
 * Deserializes each field individually (matching GlobalSerialize's format).
 */
template <typename DataT = std::vector<char>>
class GlobalDeserialize {
 public:
  using is_loading = std::true_type;
  using is_saving = std::false_type;
  using supports_range_ops = std::true_type;

  const DataT &data_;
  size_t cur_off_ = 0;

 public:
  GlobalDeserialize(const DataT &data) : data_(data) { cur_off_ = 0; }

  /** right shift operator */
  template <typename T>
  CTP_INLINE GlobalDeserialize &operator>>(T &obj) {
    return base(obj);
  }

  /** & operator */
  template <typename T>
  CTP_INLINE GlobalDeserialize &operator&(T &obj) {
    return base(obj);
  }

  /** Call operator */
  template <typename... Args>
  CTP_INLINE GlobalDeserialize &operator()(Args &&...args) {
    ctp::ForwardIterateArgpack::Apply(
        ctp::make_argpack(std::forward<Args>(args)...),
        [this](auto i, auto &arg) { this->base(arg); });
    return *this;
  }

  /** range() — portable: deserialize each field individually */
  template <typename... Args>
  CTP_INLINE GlobalDeserialize &range(Args &&...args) {
    return (*this)(std::forward<Args>(args)...);
  }

  /** Load function */
  template <typename T>
  CTP_INLINE GlobalDeserialize &base(T &obj) {
    if constexpr (std::is_arithmetic<T>::value) {
      read_binary(reinterpret_cast<char *>(&obj), sizeof(T));
    } else if constexpr (std::is_enum<T>::value) {
      using UnderlyingType = std::underlying_type_t<T>;
      UnderlyingType value{};
      read_binary(reinterpret_cast<char *>(&value), sizeof(UnderlyingType));
      obj = static_cast<T>(value);
    } else if constexpr (has_serialize_fun_v<GlobalDeserialize, T>) {
      serialize(*this, obj);
    } else if constexpr (has_load_fun_v<GlobalDeserialize, T>) {
      load(*this, obj);
    } else if constexpr (has_serialize_cls_v<GlobalDeserialize, T>) {
      obj.serialize(*this);
    } else if constexpr (has_load_cls_v<GlobalDeserialize, T>) {
      obj.load(*this);
    }
    return *this;
  }

  /** read_range — portable: delegates to per-field range() */
  template <typename FirstT, typename LastT>
  CTP_INLINE void read_range(FirstT *first, LastT *last) {
    char *begin = reinterpret_cast<char *>(first);
    char *end = reinterpret_cast<char *>(last) + sizeof(LastT);
    read_binary(begin, static_cast<size_t>(end - begin));
  }

  /** Load function (binary data) */
  CTP_INLINE
  GlobalDeserialize &read_binary(char *data, size_t size) {
    if (cur_off_ + size > data_.size()) {
      HLOG(kError,
           "GlobalDeserialize::read_binary: Attempted to read beyond end of "
           "data");
      return *this;
    }
    if (size > 0) {
      memcpy(data, data_.data() + cur_off_, size);
    }
    cur_off_ += size;
    return *this;
  }
};

/** Save string */
template <typename DataT>
CTP_INLINE void save(GlobalSerialize<DataT> &ar, const std::string &str) {
  save_string(ar, str);
}

/** Load string */
template <typename DataT>
CTP_INLINE void load(GlobalDeserialize<DataT> &ar, std::string &str) {
  load_string(ar, str);
}

/** Save vector */
template <typename DataT, typename T>
CTP_INLINE void save(GlobalSerialize<DataT> &ar, const std::vector<T> &data) {
  save_vec<GlobalSerialize<DataT>, std::vector<T>, T>(ar, data);
}

/** Load vector */
template <typename DataT, typename T>
CTP_INLINE void load(GlobalDeserialize<DataT> &ar, std::vector<T> &data) {
  load_vec<GlobalDeserialize<DataT>, std::vector<T>, T>(ar, data);
}

/** Save list */
template <typename DataT, typename T>
CTP_INLINE void save(GlobalSerialize<DataT> &ar, const std::list<T> &data) {
  save_list<GlobalSerialize<DataT>, std::list<T>, T>(ar, data);
}

/** Load list */
template <typename DataT, typename T>
CTP_INLINE void load(GlobalDeserialize<DataT> &ar, std::list<T> &data) {
  load_list<GlobalDeserialize<DataT>, std::list<T>, T>(ar, data);
}

/** Save unordered_map */
template <typename DataT, typename KeyT, typename T>
CTP_INLINE void save(GlobalSerialize<DataT> &ar,
                       const std::unordered_map<KeyT, T> &data) {
  save_map<GlobalSerialize<DataT>, std::unordered_map<KeyT, T>, KeyT, T>(ar, data);
}

/** Load unordered_map */
template <typename DataT, typename KeyT, typename T>
CTP_INLINE void load(GlobalDeserialize<DataT> &ar,
                       std::unordered_map<KeyT, T> &data) {
  load_map<GlobalDeserialize<DataT>, std::unordered_map<KeyT, T>, KeyT, T>(ar, data);
}

}  // namespace ctp::ipc

#endif  // CTP_SHM_INCLUDE_HSHM_SHM_DATA_STRUCTURES_SERIALIZATION_GLOBAL_SERIALIZE_H_
