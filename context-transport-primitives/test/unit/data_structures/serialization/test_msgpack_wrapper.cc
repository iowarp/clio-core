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

#include "../../../../../context-runtime/test/simple_test.h"
#include "clio_ctp/serialize/msgpack_wrapper.h"

#include <cstdint>
#include <string>
#include <utility>

// Helper: pack a single value and unpack it back as an object_handle
template <typename T>
static msgpack::object_handle PackUnpack(const T &value,
                                         msgpack::sbuffer &buf) {
  msgpack::packer<msgpack::sbuffer> pk(buf);
  pk.pack(value);
  return msgpack::unpack(buf.data(), buf.size());
}

// ============================================================================
// Integer round trips (positive integer object branches)
// ============================================================================

TEST_CASE("MsgpackWrapper: unsigned integer round trips",
          "[msgpack_wrapper]") {
  {
    msgpack::sbuffer buf;
    auto oh = PackUnpack<uint8_t>(200, buf);
    uint8_t out = 0;
    oh.get().convert(out);
    REQUIRE(out == 200);
  }
  {
    msgpack::sbuffer buf;
    auto oh = PackUnpack<uint16_t>(60000, buf);
    uint16_t out = 0;
    oh.get().convert(out);
    REQUIRE(out == 60000);
  }
  {
    msgpack::sbuffer buf;
    auto oh = PackUnpack<uint32_t>(4000000000u, buf);
    uint32_t out = 0;
    oh.get().convert(out);
    REQUIRE(out == 4000000000u);
  }
  {
    msgpack::sbuffer buf;
    auto oh = PackUnpack<uint64_t>(123456789012345ull, buf);
    uint64_t out = 0;
    oh.get().convert(out);
    REQUIRE(out == 123456789012345ull);
  }
}

TEST_CASE("MsgpackWrapper: signed integer round trips", "[msgpack_wrapper]") {
  {
    msgpack::sbuffer buf;
    auto oh = PackUnpack<int8_t>(-100, buf);
    int8_t out = 0;
    oh.get().convert(out);
    REQUIRE(out == -100);
  }
  {
    msgpack::sbuffer buf;
    auto oh = PackUnpack<int16_t>(-30000, buf);
    int16_t out = 0;
    oh.get().convert(out);
    REQUIRE(out == -30000);
  }
  {
    msgpack::sbuffer buf;
    auto oh = PackUnpack<int32_t>(-2000000000, buf);
    int32_t out = 0;
    oh.get().convert(out);
    REQUIRE(out == -2000000000);
  }
  {
    msgpack::sbuffer buf;
    auto oh = PackUnpack<int64_t>(-123456789012345ll, buf);
    int64_t out = 0;
    oh.get().convert(out);
    REQUIRE(out == -123456789012345ll);
  }
}

TEST_CASE("MsgpackWrapper: positive values into signed converters",
          "[msgpack_wrapper]") {
  // Packed positive values exercise the POSITIVE_INTEGER branch of the
  // signed convert specializations.
  msgpack::sbuffer buf;
  auto oh = PackUnpack<int64_t>(42, buf);
  REQUIRE(oh.get().type == msgpack::type::POSITIVE_INTEGER);

  int8_t i8 = 0;
  oh.get().convert(i8);
  REQUIRE(i8 == 42);
  int16_t i16 = 0;
  oh.get().convert(i16);
  REQUIRE(i16 == 42);
  int32_t i32 = 0;
  oh.get().convert(i32);
  REQUIRE(i32 == 42);
  int64_t i64 = 0;
  oh.get().convert(i64);
  REQUIRE(i64 == 42);
}

TEST_CASE("MsgpackWrapper: negative values into unsigned converters",
          "[msgpack_wrapper]") {
  // Packed negative values exercise the NEGATIVE_INTEGER branch of the
  // unsigned convert specializations.
  msgpack::sbuffer buf;
  auto oh = PackUnpack<int64_t>(-1, buf);
  REQUIRE(oh.get().type == msgpack::type::NEGATIVE_INTEGER);

  uint8_t u8 = 0;
  oh.get().convert(u8);
  REQUIRE(u8 == static_cast<uint8_t>(-1));
  uint16_t u16 = 0;
  oh.get().convert(u16);
  REQUIRE(u16 == static_cast<uint16_t>(-1));
  uint32_t u32 = 0;
  oh.get().convert(u32);
  REQUIRE(u32 == static_cast<uint32_t>(-1));
  uint64_t u64 = 0;
  oh.get().convert(u64);
  REQUIRE(u64 == static_cast<uint64_t>(-1));
}

// ============================================================================
// Bool conversions
// ============================================================================

TEST_CASE("MsgpackWrapper: bool round trips", "[msgpack_wrapper]") {
  {
    msgpack::sbuffer buf;
    auto oh = PackUnpack<bool>(true, buf);
    bool out = false;
    oh.get().convert(out);
    REQUIRE(out);
  }
  {
    msgpack::sbuffer buf;
    auto oh = PackUnpack<bool>(false, buf);
    bool out = true;
    oh.get().convert(out);
    REQUIRE_FALSE(out);
  }
  {
    // Positive integer to bool
    msgpack::sbuffer buf;
    auto oh = PackUnpack<uint32_t>(7, buf);
    bool out = false;
    oh.get().convert(out);
    REQUIRE(out);
  }
  {
    // Negative integer to bool
    msgpack::sbuffer buf;
    auto oh = PackUnpack<int32_t>(-7, buf);
    bool out = false;
    oh.get().convert(out);
    REQUIRE(out);
  }
}

// ============================================================================
// Float / double conversions
// ============================================================================

TEST_CASE("MsgpackWrapper: float and double round trips",
          "[msgpack_wrapper]") {
  {
    msgpack::sbuffer buf;
    auto oh = PackUnpack<float>(3.5f, buf);
    float fout = 0.0f;
    oh.get().convert(fout);
    REQUIRE(fout == 3.5f);
  }
  {
    msgpack::sbuffer buf;
    auto oh = PackUnpack<double>(2.25, buf);
    double dout = 0.0;
    oh.get().convert(dout);
    REQUIRE(dout == 2.25);
  }
  {
    // Integers convert into float/double
    msgpack::sbuffer buf;
    auto oh = PackUnpack<uint32_t>(10, buf);
    float fout = 0.0f;
    oh.get().convert(fout);
    REQUIRE(fout == 10.0f);
    double dout = 0.0;
    oh.get().convert(dout);
    REQUIRE(dout == 10.0);
  }
  {
    // Negative integers convert into float/double
    msgpack::sbuffer buf;
    auto oh = PackUnpack<int32_t>(-10, buf);
    float fout = 0.0f;
    oh.get().convert(fout);
    REQUIRE(fout == -10.0f);
    double dout = 0.0;
    oh.get().convert(dout);
    REQUIRE(dout == -10.0);
  }
}

// ============================================================================
// String conversions
// ============================================================================

TEST_CASE("MsgpackWrapper: string round trips", "[msgpack_wrapper]") {
  {
    msgpack::sbuffer buf;
    std::string in("hello msgpack");
    auto oh = PackUnpack(in, buf);
    REQUIRE(oh.get().type == msgpack::type::STR);
    std::string out;
    oh.get().convert(out);
    REQUIRE(out == in);
  }
  {
    // const char* pack overload
    msgpack::sbuffer buf;
    msgpack::packer<msgpack::sbuffer> pk(buf);
    pk.pack("c string");
    auto oh = msgpack::unpack(buf.data(), buf.size());
    std::string out;
    oh.get().convert(out);
    REQUIRE(out == "c string");
  }
}

// ============================================================================
// Type mismatch errors (throw branches in convert specializations)
// ============================================================================

template <typename T>
static bool ConvertThrows(const msgpack::object &obj) {
  T out{};
  try {
    obj.convert(out);
  } catch (const std::runtime_error &) {
    return true;
  }
  return false;
}

TEST_CASE("MsgpackWrapper: convert type mismatches throw",
          "[msgpack_wrapper]") {
  // A STR object mismatches all numeric/bool converters
  msgpack::sbuffer buf;
  std::string in("not a number");
  auto oh = PackUnpack(in, buf);
  const msgpack::object &obj = oh.get();

  REQUIRE(ConvertThrows<bool>(obj));
  REQUIRE(ConvertThrows<uint8_t>(obj));
  REQUIRE(ConvertThrows<uint16_t>(obj));
  REQUIRE(ConvertThrows<uint32_t>(obj));
  REQUIRE(ConvertThrows<uint64_t>(obj));
  REQUIRE(ConvertThrows<int8_t>(obj));
  REQUIRE(ConvertThrows<int16_t>(obj));
  REQUIRE(ConvertThrows<int32_t>(obj));
  REQUIRE(ConvertThrows<int64_t>(obj));
  REQUIRE(ConvertThrows<float>(obj));
  REQUIRE(ConvertThrows<double>(obj));

  // A numeric object mismatches the string converter
  msgpack::sbuffer buf2;
  auto oh2 = PackUnpack<uint32_t>(5, buf2);
  REQUIRE(ConvertThrows<std::string>(oh2.get()));
}

// ============================================================================
// Arrays, maps, nil
// ============================================================================

TEST_CASE("MsgpackWrapper: array, map and nil", "[msgpack_wrapper]") {
  msgpack::sbuffer buf;
  msgpack::packer<msgpack::sbuffer> pk(buf);

  // [1, "two", nil]
  pk.pack_array(3);
  pk.pack(static_cast<uint32_t>(1));
  pk.pack(std::string("two"));
  pk.pack_nil();

  auto oh = msgpack::unpack(buf.data(), buf.size());
  const msgpack::object &obj = oh.get();
  REQUIRE(obj.type == msgpack::type::ARRAY);
  REQUIRE(obj.via.array.size == 3);

  uint32_t first = 0;
  obj.via.array.ptr[0].convert(first);
  REQUIRE(first == 1);

  std::string second;
  obj.via.array.ptr[1].convert(second);
  REQUIRE(second == "two");

  REQUIRE(obj.via.array.ptr[2].type == msgpack::type::NIL);

  // Map {"key": 9}
  msgpack::sbuffer buf2;
  msgpack::packer<msgpack::sbuffer> pk2(buf2);
  pk2.pack_map(1);
  pk2.pack(std::string("key"));
  pk2.pack(static_cast<int32_t>(9));

  auto oh2 = msgpack::unpack(buf2.data(), buf2.size());
  const msgpack::object &mobj = oh2.get();
  REQUIRE(mobj.type == msgpack::type::MAP);
  REQUIRE(mobj.via.map.size == 1);
  std::string key;
  mobj.via.map.ptr[0].key.convert(key);
  REQUIRE(key == "key");
  int32_t val = 0;
  mobj.via.map.ptr[0].val.convert(val);
  REQUIRE(val == 9);
}

// ============================================================================
// sbuffer semantics
// ============================================================================

TEST_CASE("MsgpackWrapper: sbuffer move and clear", "[msgpack_wrapper]") {
  msgpack::sbuffer buf;
  msgpack::packer<msgpack::sbuffer> pk(buf);
  pk.pack(static_cast<uint64_t>(99));
  REQUIRE(buf.size() > 0);

  // Move constructor
  msgpack::sbuffer moved(std::move(buf));
  REQUIRE(moved.size() > 0);
  REQUIRE(buf.size() == 0);
  REQUIRE(buf.data() == nullptr);

  // Move assignment
  msgpack::sbuffer target;
  target = std::move(moved);
  REQUIRE(target.size() > 0);
  REQUIRE(moved.size() == 0);

  // Self move-assignment is a no-op
  msgpack::sbuffer &self = target;
  target = std::move(self);
  REQUIRE(target.size() > 0);

  // Round trip after moves
  auto oh = msgpack::unpack(target.data(), target.size());
  uint64_t out = 0;
  oh.get().convert(out);
  REQUIRE(out == 99);

  // clear() resets size
  target.clear();
  REQUIRE(target.size() == 0);
}

// ============================================================================
// pack_raw_msgpack
// ============================================================================

TEST_CASE("MsgpackWrapper: pack_raw_msgpack", "[msgpack_wrapper]") {
  // First, serialize a value into its own buffer
  msgpack::sbuffer inner;
  msgpack::packer<msgpack::sbuffer> ipk(inner);
  ipk.pack(static_cast<uint32_t>(77));

  // Embed the raw bytes via both overloads
  msgpack::sbuffer outer;
  msgpack::packer<msgpack::sbuffer> opk(outer);
  opk.pack_array(2);
  opk.pack_raw_msgpack(inner.data(), inner.size());
  opk.pack_raw_msgpack(std::string(inner.data(), inner.size()));

  auto oh = msgpack::unpack(outer.data(), outer.size());
  const msgpack::object &obj = oh.get();
  REQUIRE(obj.type == msgpack::type::ARRAY);
  REQUIRE(obj.via.array.size == 2);
  uint32_t a = 0, b = 0;
  obj.via.array.ptr[0].convert(a);
  obj.via.array.ptr[1].convert(b);
  REQUIRE(a == 77);
  REQUIRE(b == 77);
}

// ============================================================================
// object_handle move semantics and unpack errors
// ============================================================================

TEST_CASE("MsgpackWrapper: object_handle move", "[msgpack_wrapper]") {
  msgpack::sbuffer buf;
  auto oh = PackUnpack<uint32_t>(123, buf);

  // Move constructor
  msgpack::object_handle moved(std::move(oh));
  uint32_t out = 0;
  moved.get().convert(out);
  REQUIRE(out == 123);

  // Move assignment
  msgpack::object_handle target;
  target = std::move(moved);
  out = 0;
  target.get().convert(out);
  REQUIRE(out == 123);

  // Self move-assignment is a no-op
  msgpack::object_handle &self = target;
  target = std::move(self);
  out = 0;
  target.get().convert(out);
  REQUIRE(out == 123);
}

TEST_CASE("MsgpackWrapper: unpack parse error throws", "[msgpack_wrapper]") {
  // 0xc1 is an invalid (never used) msgpack type byte
  const char bad[] = {static_cast<char>(0xc1)};
  bool threw = false;
  try {
    auto oh = msgpack::unpack(bad, sizeof(bad));
    (void)oh;
  } catch (const std::runtime_error &) {
    threw = true;
  }
  REQUIRE(threw);
}

SIMPLE_TEST_MAIN()
