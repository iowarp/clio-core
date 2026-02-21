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

#ifndef HSHM_SHM_SERIALIZE_COMMON_H_
#define HSHM_SHM_SERIALIZE_COMMON_H_

#include <stddef.h>

#if HSHM_ENABLE_CEREAL
#include <cereal/archives/binary.hpp>
#endif

// #include "hermes_shm/data_structures/ipc/hash.h"  // Deleted during hard refactoring

namespace hshm::ipc {

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

// Detect if a class is serializable
template <typename Ar, typename T>
inline constexpr bool is_serializeable_v =
    has_serialize_fun_v<Ar, T> || has_load_save_fun_v<Ar, T> ||
    has_serialize_cls_v<Ar, T> || has_load_save_cls_v<Ar, T> ||
    std::is_arithmetic_v<T> || std::is_enum<T>::value;

template <typename Ar, typename T>
HSHM_CROSS_FUN void write_binary(Ar &ar, const T *data, size_t size) {
#if HSHM_ENABLE_CEREAL
  auto binary = cereal::binary_data(data, size);
  ar(binary);
#else
  ar.write_binary((const char *)data, size);
#endif
}
template <typename Ar, typename T>
HSHM_CROSS_FUN void read_binary(Ar &ar, T *data, size_t size) {
#if HSHM_ENABLE_CEREAL
  auto binary = cereal::binary_data(data, size);
  ar(binary);
#else
  ar.read_binary((char *)data, size);
#endif
}

/** Serialize a generic string. */
template <typename Ar, typename StringT>
HSHM_CROSS_FUN void save_string(Ar &ar, const StringT &text) {
  ar << text.size();
  // ar.write(text.data(), text.size());
  write_binary(ar, text.data(), text.size());
}
/** Deserialize a generic string. */
template <typename Ar, typename StringT>
HSHM_CROSS_FUN void load_string(Ar &ar, StringT &text) {
  size_t size = 0;
  ar >> size;
  text.resize(size);
  read_binary(ar, text.data(), text.size());
}

/** Serialize a generic vector */
template <typename Ar, typename ContainerT, typename T>
HSHM_CROSS_FUN void save_vec(Ar &ar, const ContainerT &obj) {
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
HSHM_CROSS_FUN void load_vec(Ar &ar, ContainerT &obj) {
  size_t size = 0;
  ar >> size;
  obj.resize(size);
  if constexpr (std::is_arithmetic_v<T>) {
    read_binary(ar, (char *)obj.data(), obj.size() * sizeof(T));
  } else {
    for (size_t i = 0; i < size; ++i) {
      ar >> (obj[i]);
    }
  }
}

/** Serialize a generic list */
template <typename Ar, typename ContainerT, typename T>
HSHM_CROSS_FUN void save_list(Ar &ar, const ContainerT &obj) {
  ar << obj.size();
  for (auto iter = obj.cbegin(); iter != obj.cend(); ++iter) {
    ar << (*iter);
  }
}
/** Deserialize a generic list */
template <typename Ar, typename ContainerT, typename T>
HSHM_CROSS_FUN void load_list(Ar &ar, ContainerT &obj) {
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
HSHM_CROSS_FUN void save_map(Ar &ar, const ContainerT &obj) {
  ar << obj.size();
  for (auto iter = obj.cbegin(); iter != obj.cend(); ++iter) {
    ar << (*iter).first;
    ar << (*iter).second;
  }
}
/** Deserialize a generic list */
template <typename Ar, typename ContainerT, typename KeyT, typename T>
HSHM_CROSS_FUN void load_map(Ar &ar, ContainerT &obj) {
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

}  // namespace hshm::ipc

#if HSHM_ENABLE_CEREAL
// Forward declarations for hshm::priv types
namespace hshm::priv {
template <typename T, typename AllocT, size_t SSOSize> class basic_string;
template <typename T, typename AllocT> class vector;
}  // namespace hshm::priv

// Tell cereal to use member load/save functions for hshm::priv types
namespace cereal {
template<typename T, typename AllocT, size_t SSOSize>
struct specialize<cereal::BinaryOutputArchive, hshm::priv::basic_string<T, AllocT, SSOSize>, cereal::specialization::member_load_save> {};

template<typename T, typename AllocT, size_t SSOSize>
struct specialize<cereal::BinaryInputArchive, hshm::priv::basic_string<T, AllocT, SSOSize>, cereal::specialization::member_load_save> {};

template<typename T, typename AllocT>
struct specialize<cereal::BinaryOutputArchive, hshm::priv::vector<T, AllocT>, cereal::specialization::member_load_save> {};

template<typename T, typename AllocT>
struct specialize<cereal::BinaryInputArchive, hshm::priv::vector<T, AllocT>, cereal::specialization::member_load_save> {};
}  // namespace cereal
#endif  // HSHM_ENABLE_CEREAL

#endif  // HSHM_SHM_SERIALIZE_COMMON_H_
