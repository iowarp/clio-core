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

#ifndef HERMES_SHM_SERIALIZE_MSGPACK_WRAPPER_H
#define HERMES_SHM_SERIALIZE_MSGPACK_WRAPPER_H

/**
 * Lightweight C++ wrapper around the msgpack-c (pure C) library.
 * Provides an API compatible with the subset of msgpack-cxx used in IOWarp,
 * without pulling in Boost.
 */

#include <msgpack.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace msgpack {

namespace type {
constexpr msgpack_object_type NIL              = MSGPACK_OBJECT_NIL;
constexpr msgpack_object_type BOOLEAN          = MSGPACK_OBJECT_BOOLEAN;
constexpr msgpack_object_type POSITIVE_INTEGER = MSGPACK_OBJECT_POSITIVE_INTEGER;
constexpr msgpack_object_type NEGATIVE_INTEGER = MSGPACK_OBJECT_NEGATIVE_INTEGER;
constexpr msgpack_object_type FLOAT32          = MSGPACK_OBJECT_FLOAT32;
constexpr msgpack_object_type FLOAT64          = MSGPACK_OBJECT_FLOAT64;
constexpr msgpack_object_type STR              = MSGPACK_OBJECT_STR;
constexpr msgpack_object_type ARRAY            = MSGPACK_OBJECT_ARRAY;
constexpr msgpack_object_type MAP              = MSGPACK_OBJECT_MAP;
constexpr msgpack_object_type BIN              = MSGPACK_OBJECT_BIN;
constexpr msgpack_object_type EXT              = MSGPACK_OBJECT_EXT;
}  // namespace type

// ---- object / object_kv ---------------------------------------------------

struct object;
struct object_kv;

struct object_array {
  uint32_t size;
  object* ptr;
};

struct object_map {
  uint32_t size;
  object_kv* ptr;
};

struct object_str {
  uint32_t size;
  const char* ptr;
};

struct object_bin {
  uint32_t size;
  const char* ptr;
};

union object_union {
  bool boolean;
  uint64_t u64;
  int64_t i64;
  double f64;
  object_array array;
  object_map map;
  object_str str;
  object_bin bin;
};

struct object {
  msgpack_object_type type;
  object_union via;

  template <typename T>
  void convert(T& out) const;
};

struct object_kv {
  object key;
  object val;
};

// Verify layout compatibility with C structs so we can reinterpret_cast.
static_assert(sizeof(object) == sizeof(msgpack_object),
              "object layout mismatch");
static_assert(sizeof(object_kv) == sizeof(msgpack_object_kv),
              "object_kv layout mismatch");
static_assert(offsetof(object, type) == offsetof(msgpack_object, type),
              "object::type offset mismatch");
static_assert(offsetof(object, via) == offsetof(msgpack_object, via),
              "object::via offset mismatch");

// ---- convert specialisations ----------------------------------------------

template <>
inline void object::convert(bool& out) const {
  if (type == MSGPACK_OBJECT_BOOLEAN) {
    out = via.boolean;
  } else if (type == MSGPACK_OBJECT_POSITIVE_INTEGER) {
    out = via.u64 != 0;
  } else if (type == MSGPACK_OBJECT_NEGATIVE_INTEGER) {
    out = via.i64 != 0;
  } else {
    throw std::runtime_error("msgpack::object::convert<bool>: type mismatch");
  }
}

template <>
inline void object::convert(uint8_t& out) const {
  if (type == MSGPACK_OBJECT_POSITIVE_INTEGER)
    out = static_cast<uint8_t>(via.u64);
  else if (type == MSGPACK_OBJECT_NEGATIVE_INTEGER)
    out = static_cast<uint8_t>(via.i64);
  else
    throw std::runtime_error("msgpack::object::convert<uint8_t>: type mismatch");
}

template <>
inline void object::convert(uint16_t& out) const {
  if (type == MSGPACK_OBJECT_POSITIVE_INTEGER)
    out = static_cast<uint16_t>(via.u64);
  else if (type == MSGPACK_OBJECT_NEGATIVE_INTEGER)
    out = static_cast<uint16_t>(via.i64);
  else
    throw std::runtime_error("msgpack::object::convert<uint16_t>: type mismatch");
}

template <>
inline void object::convert(uint32_t& out) const {
  if (type == MSGPACK_OBJECT_POSITIVE_INTEGER)
    out = static_cast<uint32_t>(via.u64);
  else if (type == MSGPACK_OBJECT_NEGATIVE_INTEGER)
    out = static_cast<uint32_t>(via.i64);
  else
    throw std::runtime_error("msgpack::object::convert<uint32_t>: type mismatch");
}

template <>
inline void object::convert(uint64_t& out) const {
  if (type == MSGPACK_OBJECT_POSITIVE_INTEGER)
    out = via.u64;
  else if (type == MSGPACK_OBJECT_NEGATIVE_INTEGER)
    out = static_cast<uint64_t>(via.i64);
  else
    throw std::runtime_error("msgpack::object::convert<uint64_t>: type mismatch");
}

template <>
inline void object::convert(int8_t& out) const {
  if (type == MSGPACK_OBJECT_POSITIVE_INTEGER)
    out = static_cast<int8_t>(via.u64);
  else if (type == MSGPACK_OBJECT_NEGATIVE_INTEGER)
    out = static_cast<int8_t>(via.i64);
  else
    throw std::runtime_error("msgpack::object::convert<int8_t>: type mismatch");
}

template <>
inline void object::convert(int16_t& out) const {
  if (type == MSGPACK_OBJECT_POSITIVE_INTEGER)
    out = static_cast<int16_t>(via.u64);
  else if (type == MSGPACK_OBJECT_NEGATIVE_INTEGER)
    out = static_cast<int16_t>(via.i64);
  else
    throw std::runtime_error("msgpack::object::convert<int16_t>: type mismatch");
}

template <>
inline void object::convert(int32_t& out) const {
  if (type == MSGPACK_OBJECT_POSITIVE_INTEGER)
    out = static_cast<int32_t>(via.u64);
  else if (type == MSGPACK_OBJECT_NEGATIVE_INTEGER)
    out = static_cast<int32_t>(via.i64);
  else
    throw std::runtime_error("msgpack::object::convert<int32_t>: type mismatch");
}

template <>
inline void object::convert(int64_t& out) const {
  if (type == MSGPACK_OBJECT_POSITIVE_INTEGER)
    out = static_cast<int64_t>(via.u64);
  else if (type == MSGPACK_OBJECT_NEGATIVE_INTEGER)
    out = via.i64;
  else
    throw std::runtime_error("msgpack::object::convert<int64_t>: type mismatch");
}

template <>
inline void object::convert(float& out) const {
  if (type == MSGPACK_OBJECT_FLOAT32 || type == MSGPACK_OBJECT_FLOAT64)
    out = static_cast<float>(via.f64);
  else if (type == MSGPACK_OBJECT_POSITIVE_INTEGER)
    out = static_cast<float>(via.u64);
  else if (type == MSGPACK_OBJECT_NEGATIVE_INTEGER)
    out = static_cast<float>(via.i64);
  else
    throw std::runtime_error("msgpack::object::convert<float>: type mismatch");
}

template <>
inline void object::convert(double& out) const {
  if (type == MSGPACK_OBJECT_FLOAT32 || type == MSGPACK_OBJECT_FLOAT64)
    out = via.f64;
  else if (type == MSGPACK_OBJECT_POSITIVE_INTEGER)
    out = static_cast<double>(via.u64);
  else if (type == MSGPACK_OBJECT_NEGATIVE_INTEGER)
    out = static_cast<double>(via.i64);
  else
    throw std::runtime_error("msgpack::object::convert<double>: type mismatch");
}

template <>
inline void object::convert(std::string& out) const {
  if (type == MSGPACK_OBJECT_STR)
    out.assign(via.str.ptr, via.str.size);
  else if (type == MSGPACK_OBJECT_BIN)
    out.assign(via.bin.ptr, via.bin.size);
  else
    throw std::runtime_error("msgpack::object::convert<string>: type mismatch");
}

// ---- sbuffer --------------------------------------------------------------

class sbuffer {
 public:
  sbuffer() { msgpack_sbuffer_init(&buf_); }
  ~sbuffer() { msgpack_sbuffer_destroy(&buf_); }

  sbuffer(const sbuffer&) = delete;
  sbuffer& operator=(const sbuffer&) = delete;

  sbuffer(sbuffer&& o) noexcept : buf_(o.buf_) {
    o.buf_.data = nullptr;
    o.buf_.size = 0;
    o.buf_.alloc = 0;
  }
  sbuffer& operator=(sbuffer&& o) noexcept {
    if (this != &o) {
      msgpack_sbuffer_destroy(&buf_);
      buf_ = o.buf_;
      o.buf_.data = nullptr;
      o.buf_.size = 0;
      o.buf_.alloc = 0;
    }
    return *this;
  }

  const char* data() const { return buf_.data; }
  size_t size() const { return buf_.size; }
  void clear() { msgpack_sbuffer_clear(&buf_); }

  msgpack_sbuffer* c_ptr() { return &buf_; }

 private:
  msgpack_sbuffer buf_;
};

// ---- packer ---------------------------------------------------------------

template <typename Buffer>
class packer {
 public:
  explicit packer(Buffer& buf) : buf_(buf) {
    msgpack_packer_init(&pk_, buf_.c_ptr(), msgpack_sbuffer_write);
  }

  // Container helpers
  void pack_array(size_t n) { msgpack_pack_array(&pk_, n); }
  void pack_map(size_t n) { msgpack_pack_map(&pk_, n); }
  void pack_nil() { msgpack_pack_nil(&pk_); }

  // Overloaded pack()
  void pack(bool v) {
    if (v) msgpack_pack_true(&pk_);
    else   msgpack_pack_false(&pk_);
  }
  void pack(int8_t v)   { msgpack_pack_int8(&pk_, v); }
  void pack(int16_t v)  { msgpack_pack_int16(&pk_, v); }
  void pack(int32_t v)  { msgpack_pack_int32(&pk_, v); }
  void pack(int64_t v)  { msgpack_pack_int64(&pk_, v); }
  void pack(uint8_t v)  { msgpack_pack_uint8(&pk_, v); }
  void pack(uint16_t v) { msgpack_pack_uint16(&pk_, v); }
  void pack(uint32_t v) { msgpack_pack_uint32(&pk_, v); }
  void pack(uint64_t v) { msgpack_pack_uint64(&pk_, v); }
  void pack(float v)    { msgpack_pack_float(&pk_, v); }
  void pack(double v)   { msgpack_pack_double(&pk_, v); }

  void pack(const std::string& v) {
    msgpack_pack_str(&pk_, v.size());
    msgpack_pack_str_body(&pk_, v.data(), v.size());
  }

  void pack(const char* v) {
    size_t len = std::char_traits<char>::length(v);
    msgpack_pack_str(&pk_, len);
    msgpack_pack_str_body(&pk_, v, len);
  }

  void pack_raw_msgpack(const char* data, size_t len) {
    pk_.callback(pk_.data, data, len);
  }
  void pack_raw_msgpack(const std::string& data) {
    pack_raw_msgpack(data.data(), data.size());
  }

 private:
  Buffer& buf_;
  msgpack_packer pk_;
};

// ---- object_handle (unpacking) --------------------------------------------

class object_handle {
 public:
  object_handle() { msgpack_unpacked_init(&unpacked_); }
  ~object_handle() { msgpack_unpacked_destroy(&unpacked_); }

  object_handle(const object_handle&) = delete;
  object_handle& operator=(const object_handle&) = delete;

  object_handle(object_handle&& o) noexcept : unpacked_(o.unpacked_) {
    o.unpacked_.zone = nullptr;
  }
  object_handle& operator=(object_handle&& o) noexcept {
    if (this != &o) {
      msgpack_unpacked_destroy(&unpacked_);
      unpacked_ = o.unpacked_;
      o.unpacked_.zone = nullptr;
    }
    return *this;
  }

  const object& get() const {
    return reinterpret_cast<const object&>(unpacked_.data);
  }

  msgpack_unpacked* c_ptr() { return &unpacked_; }

 private:
  msgpack_unpacked unpacked_;
};

// ---- free function: unpack ------------------------------------------------

inline object_handle unpack(const char* data, size_t len) {
  object_handle oh;
  size_t off = 0;
  msgpack_unpack_return ret = msgpack_unpack_next(oh.c_ptr(), data, len, &off);
  if (ret < 0) {
    throw std::runtime_error("msgpack::unpack: parse error");
  }
  return oh;
}

}  // namespace msgpack

#endif  // HERMES_SHM_SERIALIZE_MSGPACK_WRAPPER_H
