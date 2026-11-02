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

#ifndef CTP_SHM_SERIALIZE_COMMON_H_
#define CTP_SHM_SERIALIZE_COMMON_H_

#include <stddef.h>

// #include "clio_ctp/data_structures/ipc/hash.h"  // Deleted during hard refactoring

namespace ctp::ipc {

// Detect if serialization function exists
template <typename, typename, typename = void>
struct has_serialize_fun : std::false_type {};
template <typename Ar, typename T>
struct has_serialize_fun<
    Ar, T,
    std::void_t<decltype(serialize(std::declval<Ar &>(), std::declval<T &>()))>>
    : std::true_type {};
template <typename Ar, typename T>
inline constexpr bool has_serialize_fun_v = has_serialize_fun<Ar, T>::value;

// Detect if save function exists
template <typename, typename, typename = void>
struct has_save_fun : std::false_type {};
template <typename Ar, typename T>
struct has_save_fun<Ar, T,
                    std::void_t<decltype(save(std::declval<Ar &>(),
                                              std::declval<const T &>()))>>
    : std::true_type {};
template <typename Ar, typename T>
inline constexpr bool has_save_fun_v = has_save_fun<Ar, T>::value;

// Detect if load function exists
template <typename, typename, typename = void>
struct has_load_fun : std::false_type {};
template <typename Ar, typename T>
struct has_load_fun<
    Ar, T,
    std::void_t<decltype(load(std::declval<Ar &>(), std::declval<T &>()))>>
    : std::true_type {};
template <typename Ar, typename T>
inline constexpr bool has_load_fun_v = has_load_fun<Ar, T>::value;

// Has both load and save functions
template <typename Ar, typename T>
inline constexpr bool has_load_save_fun_v =
    has_load_fun_v<Ar, T> && has_save_fun_v<Ar, T>;

// Detect if serialize method exists
template <typename, typename, typename = void>
struct has_serialize_cls : std::false_type {};
template <typename Ar, typename CLS>
struct has_serialize_cls<
    Ar, CLS,
    std::void_t<decltype(std::declval<CLS>().template serialize<Ar>(
        std::declval<Ar &>()))>> : std::true_type {};
template <typename Ar, typename CLS>
inline constexpr bool has_serialize_cls_v = has_serialize_cls<Ar, CLS>::value;

// Detect if save method exists
template <typename, typename, typename = void>
struct has_save_cls : std::false_type {};
template <typename Ar, typename CLS>
struct has_save_cls<Ar, CLS,
                    std::void_t<decltype(std::declval<CLS>().template save<Ar>(
                        std::declval<Ar &>()))>> : std::true_type {};
template <typename Ar, typename CLS>
inline constexpr bool has_save_cls_v = has_save_cls<Ar, CLS>::value;

// Detect if load method exists
template <typename, typename, typename = void>
struct has_load_cls : std::false_type {};
template <typename Ar, typename CLS>
struct has_load_cls<Ar, CLS,
                    std::void_t<decltype(std::declval<CLS>().template load<Ar>(
                        std::declval<Ar &>()))>> : std::true_type {};
template <typename Ar, typename CLS>
inline constexpr bool has_load_cls_v = has_load_cls<Ar, CLS>::value;

// Has both load and save methods
template <typename Ar, typename T>
inline constexpr bool has_load_save_cls_v =
    has_load_cls_v<Ar, T> && has_save_cls_v<Ar, T>;

// Detect if an archive supports batch write_range/read_range operations
template <typename Ar, typename = void>
struct has_range_ops : std::false_type {};
template <typename Ar>
struct has_range_ops<Ar, std::void_t<typename Ar::supports_range_ops>>
    : std::true_type {};
template <typename Ar>
constexpr bool has_range_ops_v = has_range_ops<Ar>::value;

// Detect if a class is serializable
template <typename Ar, typename T>
inline constexpr bool is_serializeable_v =
    has_serialize_fun_v<Ar, T> || has_load_save_fun_v<Ar, T> ||
    has_serialize_cls_v<Ar, T> || has_load_save_cls_v<Ar, T> ||
    std::is_arithmetic_v<T> || std::is_enum<T>::value;

// Detect if resize_no_init exists
template <typename, typename = void>
struct has_resize_no_init : std::false_type {};
template <typename T>
struct has_resize_no_init<T,
    std::void_t<decltype(std::declval<T&>().resize_no_init(size_t{}))>>
    : std::true_type {};

/** Resize container without zero-filling when possible. */
template <typename ContainerT>
CTP_INLINE_CROSS_FUN void resize_for_overwrite(ContainerT &c, size_t n) {
  if constexpr (has_resize_no_init<ContainerT>::value) {
    c.resize_no_init(n);
  } else {
    c.resize(n);
  }
}

template <typename Ar, typename T>
CTP_INLINE_CROSS_FUN void write_binary(Ar &ar, const T *data, size_t size) {
  ar.write_binary((const char *)data, size);
}
template <typename Ar, typename T>
CTP_INLINE_CROSS_FUN void read_binary(Ar &ar, T *data, size_t size) {
  ar.read_binary((char *)data, size);
}

/** Serialize a generic string. */
template <typename Ar, typename StringT>
CTP_INLINE_CROSS_FUN void save_string(Ar &ar, const StringT &text) {
  ar.save_string_fused(text.data(), text.size());
}
/** Deserialize a generic string.
 *  Uses load_string_fused when the archive supports it (fused size+data read).
 *  Otherwise falls back to separate read operations. */
template <typename Ar, typename StringT>
CTP_INLINE_CROSS_FUN void load_string(Ar &ar, StringT &text) {
  size_t size = 0;
  ar >> size;
  resize_for_overwrite(text, size);
  read_binary(ar, text.data(), size);
}

/** Serialize a generic vector */
template <typename Ar, typename ContainerT, typename T>
CTP_INLINE_CROSS_FUN void save_vec(Ar &ar, const ContainerT &obj) {
  ar << obj.size();
  if constexpr (std::is_arithmetic_v<T>) {
    write_binary(ar, (char *)obj.data(), obj.size() * sizeof(T));
  } else {
    for (auto iter = obj.cbegin(); iter != obj.cend(); ++iter) {
      ar << (*iter);
    }
  }
}
/** Deserialize a generic vector */
template <typename Ar, typename ContainerT, typename T>
CTP_INLINE_CROSS_FUN void load_vec(Ar &ar, ContainerT &obj) {
  size_t size = 0;
  ar >> size;
  if constexpr (std::is_arithmetic_v<T>) {
    resize_for_overwrite(obj, size);
    read_binary(ar, (char *)obj.data(), size * sizeof(T));
  } else {
    obj.resize(size);
    for (size_t i = 0; i < size; ++i) {
      ar >> (obj[i]);
    }
  }
}

/** Serialize a generic list */
template <typename Ar, typename ContainerT, typename T>
CTP_INLINE_CROSS_FUN void save_list(Ar &ar, const ContainerT &obj) {
  ar << obj.size();
  for (auto iter = obj.cbegin(); iter != obj.cend(); ++iter) {
    ar << (*iter);
  }
}
/** Deserialize a generic list */
template <typename Ar, typename ContainerT, typename T>
CTP_INLINE_CROSS_FUN void load_list(Ar &ar, ContainerT &obj) {
  size_t size = 0;
  ar >> size;
  for (size_t i = 0; i < size; ++i) {
    obj.emplace_back();
    auto &last = obj.back();
    ar >> last;
  }
}

/** Serialize a generic list */
template <typename Ar, typename ContainerT, typename KeyT, typename T>
CTP_INLINE_CROSS_FUN void save_map(Ar &ar, const ContainerT &obj) {
  ar << obj.size();
  for (auto iter = obj.cbegin(); iter != obj.cend(); ++iter) {
    ar << (*iter).first;
    ar << (*iter).second;
  }
}
/** Deserialize a generic list */
template <typename Ar, typename ContainerT, typename KeyT, typename T>
CTP_INLINE_CROSS_FUN void load_map(Ar &ar, ContainerT &obj) {
  size_t size = 0;
  ar >> size;
  for (size_t i = 0; i < size; ++i) {
    KeyT key{};
    T val{};
    ar >> key;
    ar >> val;
    obj[key] = val;
  }
}

}  // namespace ctp::ipc

#endif  // CTP_SHM_SERIALIZE_COMMON_H_
