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

#include "../../../../context-runtime/test/simple_test.h"
#include "clio_ctp/data_structures/serialization/local_serialize.h"
#include <string>
#include <vector>
#include <list>
#include <unordered_map>

// ============================================================================
// Basic Arithmetic Types Tests
// ============================================================================

TEST_CASE("LocalSerialize: int", "[local_serialize][basic]") {
  int original = 42;

  // Serialize
  std::vector<char> buffer;
  ctp::ipc::LocalSerialize<> serializer(buffer);
  serializer << original;

  // Deserialize
  int restored = 0;
  ctp::ipc::LocalDeserialize<> deserializer(buffer);
  deserializer >> restored;

  // Verify
  REQUIRE(restored == original);
  REQUIRE(restored == 42);
}

TEST_CASE("LocalSerialize: multiple ints", "[local_serialize][basic]") {
  int val1 = 10;
  int val2 = 20;
  int val3 = 30;

  // Serialize
  std::vector<char> buffer;
  ctp::ipc::LocalSerialize<> serializer(buffer);
  serializer << val1 << val2 << val3;

  // Deserialize
  int restored1 = 0, restored2 = 0, restored3 = 0;
  ctp::ipc::LocalDeserialize<> deserializer(buffer);
  deserializer >> restored1 >> restored2 >> restored3;

  // Verify
  REQUIRE(restored1 == 10);
  REQUIRE(restored2 == 20);
  REQUIRE(restored3 == 30);
}

TEST_CASE("LocalSerialize: float", "[local_serialize][basic]") {
  float original = 3.14159f;

  // Serialize
  std::vector<char> buffer;
  ctp::ipc::LocalSerialize<> serializer(buffer);
  serializer << original;

  // Deserialize
  float restored = 0.0f;
  ctp::ipc::LocalDeserialize<> deserializer(buffer);
  deserializer >> restored;

  // Verify
  REQUIRE(restored == original);
}

TEST_CASE("LocalSerialize: double", "[local_serialize][basic]") {
  double original = 2.718281828459045;

  // Serialize
  std::vector<char> buffer;
  ctp::ipc::LocalSerialize<> serializer(buffer);
  serializer << original;

  // Deserialize
  double restored = 0.0;
  ctp::ipc::LocalDeserialize<> deserializer(buffer);
  deserializer >> restored;

  // Verify
  REQUIRE(restored == original);
}

TEST_CASE("LocalSerialize: bool", "[local_serialize][basic]") {
  bool original_true = true;
  bool original_false = false;

  // Serialize
  std::vector<char> buffer;
  ctp::ipc::LocalSerialize<> serializer(buffer);
  serializer << original_true << original_false;

  // Deserialize
  bool restored_true = false;
  bool restored_false = true;
  ctp::ipc::LocalDeserialize<> deserializer(buffer);
  deserializer >> restored_true >> restored_false;

  // Verify
  REQUIRE(restored_true == true);
  REQUIRE(restored_false == false);
}

TEST_CASE("LocalSerialize: char", "[local_serialize][basic]") {
  char original = 'X';

  // Serialize
  std::vector<char> buffer;
  ctp::ipc::LocalSerialize<> serializer(buffer);
  serializer << original;

  // Deserialize
  char restored = '\0';
  ctp::ipc::LocalDeserialize<> deserializer(buffer);
  deserializer >> restored;

  // Verify
  REQUIRE(restored == original);
  REQUIRE(restored == 'X');
}

TEST_CASE("LocalSerialize: unsigned types", "[local_serialize][basic]") {
  unsigned int uint_val = 4294967295u;
  unsigned long ulong_val = 18446744073709551615ul;

  // Serialize
  std::vector<char> buffer;
  ctp::ipc::LocalSerialize<> serializer(buffer);
  serializer << uint_val << ulong_val;

  // Deserialize
  unsigned int restored_uint = 0;
  unsigned long restored_ulong = 0;
  ctp::ipc::LocalDeserialize<> deserializer(buffer);
  deserializer >> restored_uint >> restored_ulong;

  // Verify
  REQUIRE(restored_uint == uint_val);
  REQUIRE(restored_ulong == ulong_val);
}

TEST_CASE("LocalSerialize: size_t", "[local_serialize][basic]") {
  size_t original = 1234567890;

  // Serialize
  std::vector<char> buffer;
  ctp::ipc::LocalSerialize<> serializer(buffer);
  serializer << original;

  // Deserialize
  size_t restored = 0;
  ctp::ipc::LocalDeserialize<> deserializer(buffer);
  deserializer >> restored;

  // Verify
  REQUIRE(restored == original);
}

// ============================================================================
// std::string Tests
// ============================================================================

TEST_CASE("LocalSerialize: std::string", "[local_serialize][string]") {
  std::string original = "Hello, World!";

  // Serialize
  std::vector<char> buffer;
  ctp::ipc::LocalSerialize<> serializer(buffer);
  serializer << original;

  // Deserialize
  std::string restored;
  ctp::ipc::LocalDeserialize<> deserializer(buffer);
  deserializer >> restored;

  // Verify
  REQUIRE(restored == original);
  REQUIRE(restored == "Hello, World!");
}

TEST_CASE("LocalSerialize: empty std::string", "[local_serialize][string]") {
  std::string original = "";

  // Serialize
  std::vector<char> buffer;
  ctp::ipc::LocalSerialize<> serializer(buffer);
  serializer << original;

  // Deserialize
  std::string restored;
  ctp::ipc::LocalDeserialize<> deserializer(buffer);
  deserializer >> restored;

  // Verify
  REQUIRE(restored.empty());
  REQUIRE(restored == "");
}

TEST_CASE("LocalSerialize: large std::string", "[local_serialize][string]") {
  std::string original;
  for (int i = 0; i < 1000; ++i) {
    original += "This is a test string. ";
  }

  // Serialize
  std::vector<char> buffer;
  ctp::ipc::LocalSerialize<> serializer(buffer);
  serializer << original;

  // Deserialize
  std::string restored;
  ctp::ipc::LocalDeserialize<> deserializer(buffer);
  deserializer >> restored;

  // Verify
  REQUIRE(restored.size() == original.size());
  REQUIRE(restored == original);
}

TEST_CASE("LocalSerialize: multiple std::strings", "[local_serialize][string]") {
  std::string str1 = "first";
  std::string str2 = "second";
  std::string str3 = "third";

  // Serialize
  std::vector<char> buffer;
  ctp::ipc::LocalSerialize<> serializer(buffer);
  serializer << str1 << str2 << str3;

  // Deserialize
  std::string restored1, restored2, restored3;
  ctp::ipc::LocalDeserialize<> deserializer(buffer);
  deserializer >> restored1 >> restored2 >> restored3;

  // Verify
  REQUIRE(restored1 == "first");
  REQUIRE(restored2 == "second");
  REQUIRE(restored3 == "third");
}

// ============================================================================
// std::vector Tests
// ============================================================================

TEST_CASE("LocalSerialize: std::vector<int>", "[local_serialize][vector]") {
  std::vector<int> original = {1, 2, 3, 4, 5};

  // Serialize
  std::vector<char> buffer;
  ctp::ipc::LocalSerialize<> serializer(buffer);
  serializer << original;

  // Deserialize
  std::vector<int> restored;
  ctp::ipc::LocalDeserialize<> deserializer(buffer);
  deserializer >> restored;

  // Verify
  REQUIRE(restored.size() == original.size());
  for (size_t i = 0; i < original.size(); ++i) {
    REQUIRE(restored[i] == original[i]);
  }
}

TEST_CASE("LocalSerialize: empty std::vector", "[local_serialize][vector]") {
  std::vector<int> original;

  // Serialize
  std::vector<char> buffer;
  ctp::ipc::LocalSerialize<> serializer(buffer);
  serializer << original;

  // Deserialize
  std::vector<int> restored;
  ctp::ipc::LocalDeserialize<> deserializer(buffer);
  deserializer >> restored;

  // Verify
  REQUIRE(restored.size() == 0);
  REQUIRE(restored.empty());
}

TEST_CASE("LocalSerialize: std::vector<double>", "[local_serialize][vector]") {
  std::vector<double> original = {1.1, 2.2, 3.3, 4.4, 5.5};

  // Serialize
  std::vector<char> buffer;
  ctp::ipc::LocalSerialize<> serializer(buffer);
  serializer << original;

  // Deserialize
  std::vector<double> restored;
  ctp::ipc::LocalDeserialize<> deserializer(buffer);
  deserializer >> restored;

  // Verify
  REQUIRE(restored.size() == original.size());
  for (size_t i = 0; i < original.size(); ++i) {
    REQUIRE(restored[i] == original[i]);
  }
}

TEST_CASE("LocalSerialize: std::vector<std::string>", "[local_serialize][vector]") {
  std::vector<std::string> original = {"hello", "world", "test"};

  // Serialize
  std::vector<char> buffer;
  ctp::ipc::LocalSerialize<> serializer(buffer);
  serializer << original;

  // Deserialize
  std::vector<std::string> restored;
  ctp::ipc::LocalDeserialize<> deserializer(buffer);
  deserializer >> restored;

  // Verify
  REQUIRE(restored.size() == original.size());
  for (size_t i = 0; i < original.size(); ++i) {
    REQUIRE(restored[i] == original[i]);
  }
}

TEST_CASE("LocalSerialize: large std::vector", "[local_serialize][vector]") {
  std::vector<int> original;
  for (int i = 0; i < 10000; ++i) {
    original.push_back(i);
  }

  // Serialize
  std::vector<char> buffer;
  ctp::ipc::LocalSerialize<> serializer(buffer);
  serializer << original;

  // Deserialize
  std::vector<int> restored;
  ctp::ipc::LocalDeserialize<> deserializer(buffer);
  deserializer >> restored;

  // Verify
  REQUIRE(restored.size() == 10000);
  for (int i = 0; i < 10000; ++i) {
    REQUIRE(restored[i] == i);
  }
}

// ============================================================================
// std::list Tests
// ============================================================================

TEST_CASE("LocalSerialize: std::list<int>", "[local_serialize][list]") {
  std::list<int> original = {10, 20, 30, 40, 50};

  // Serialize
  std::vector<char> buffer;
  ctp::ipc::LocalSerialize<> serializer(buffer);
  serializer << original;

  // Deserialize
  std::list<int> restored;
  ctp::ipc::LocalDeserialize<> deserializer(buffer);
  deserializer >> restored;

  // Verify
  REQUIRE(restored.size() == original.size());
  auto orig_it = original.begin();
  auto rest_it = restored.begin();
  while (orig_it != original.end()) {
    REQUIRE(*rest_it == *orig_it);
    ++orig_it;
    ++rest_it;
  }
}

TEST_CASE("LocalSerialize: empty std::list", "[local_serialize][list]") {
  std::list<int> original;

  // Serialize
  std::vector<char> buffer;
  ctp::ipc::LocalSerialize<> serializer(buffer);
  serializer << original;

  // Deserialize
  std::list<int> restored;
  ctp::ipc::LocalDeserialize<> deserializer(buffer);
  deserializer >> restored;

  // Verify
  REQUIRE(restored.size() == 0);
  REQUIRE(restored.empty());
}

// ============================================================================
// std::unordered_map Tests
// ============================================================================

TEST_CASE("LocalSerialize: std::unordered_map<int, int>", "[local_serialize][map]") {
  std::unordered_map<int, int> original;
  original[1] = 10;
  original[2] = 20;
  original[3] = 30;

  // Serialize
  std::vector<char> buffer;
  ctp::ipc::LocalSerialize<> serializer(buffer);
  serializer << original;

  // Deserialize
  std::unordered_map<int, int> restored;
  ctp::ipc::LocalDeserialize<> deserializer(buffer);
  deserializer >> restored;

  // Verify
  REQUIRE(restored.size() == original.size());
  REQUIRE(restored[1] == 10);
  REQUIRE(restored[2] == 20);
  REQUIRE(restored[3] == 30);
}

TEST_CASE("LocalSerialize: std::unordered_map<std::string, std::string>", "[local_serialize][map]") {
  std::unordered_map<std::string, std::string> original;
  original["key1"] = "value1";
  original["key2"] = "value2";
  original["key3"] = "value3";

  // Serialize
  std::vector<char> buffer;
  ctp::ipc::LocalSerialize<> serializer(buffer);
  serializer << original;

  // Deserialize
  std::unordered_map<std::string, std::string> restored;
  ctp::ipc::LocalDeserialize<> deserializer(buffer);
  deserializer >> restored;

  // Verify
  REQUIRE(restored.size() == original.size());
  REQUIRE(restored["key1"] == "value1");
  REQUIRE(restored["key2"] == "value2");
  REQUIRE(restored["key3"] == "value3");
}

TEST_CASE("LocalSerialize: empty std::unordered_map", "[local_serialize][map]") {
  std::unordered_map<int, int> original;

  // Serialize
  std::vector<char> buffer;
  ctp::ipc::LocalSerialize<> serializer(buffer);
  serializer << original;

  // Deserialize
  std::unordered_map<int, int> restored;
  ctp::ipc::LocalDeserialize<> deserializer(buffer);
  deserializer >> restored;

  // Verify
  REQUIRE(restored.size() == 0);
  REQUIRE(restored.empty());
}

// ============================================================================
// Mixed Type Tests
// ============================================================================

TEST_CASE("LocalSerialize: mixed types", "[local_serialize][mixed]") {
  int int_val = 42;
  double double_val = 3.14159;
  std::string str_val = "test";
  std::vector<int> vec_val = {1, 2, 3};

  // Serialize
  std::vector<char> buffer;
  ctp::ipc::LocalSerialize<> serializer(buffer);
  serializer << int_val << double_val << str_val << vec_val;

  // Deserialize
  int restored_int = 0;
  double restored_double = 0.0;
  std::string restored_str;
  std::vector<int> restored_vec;
  ctp::ipc::LocalDeserialize<> deserializer(buffer);
  deserializer >> restored_int >> restored_double >> restored_str >> restored_vec;

  // Verify
  REQUIRE(restored_int == 42);
  REQUIRE(restored_double == 3.14159);
  REQUIRE(restored_str == "test");
  REQUIRE(restored_vec.size() == 3);
  REQUIRE(restored_vec[0] == 1);
  REQUIRE(restored_vec[1] == 2);
  REQUIRE(restored_vec[2] == 3);
}

TEST_CASE("LocalSerialize: operator() with multiple types", "[local_serialize][mixed]") {
  int int_val = 100;
  std::string str_val = "operator test";
  double double_val = 2.71828;

  // Serialize using operator()
  std::vector<char> buffer;
  ctp::ipc::LocalSerialize<> serializer(buffer);
  serializer(int_val, str_val, double_val);

  // Deserialize using operator()
  int restored_int = 0;
  std::string restored_str;
  double restored_double = 0.0;
  ctp::ipc::LocalDeserialize<> deserializer(buffer);
  deserializer(restored_int, restored_str, restored_double);

  // Verify
  REQUIRE(restored_int == 100);
  REQUIRE(restored_str == "operator test");
  REQUIRE(restored_double == 2.71828);
}

TEST_CASE("LocalSerialize: nested containers", "[local_serialize][nested]") {
  std::vector<std::vector<int>> original = {
    {1, 2, 3},
    {4, 5},
    {6, 7, 8, 9}
  };

  // Serialize
  std::vector<char> buffer;
  ctp::ipc::LocalSerialize<> serializer(buffer);
  serializer << original;

  // Deserialize
  std::vector<std::vector<int>> restored;
  ctp::ipc::LocalDeserialize<> deserializer(buffer);
  deserializer >> restored;

  // Verify
  REQUIRE(restored.size() == original.size());
  for (size_t i = 0; i < original.size(); ++i) {
    REQUIRE(restored[i].size() == original[i].size());
    for (size_t j = 0; j < original[i].size(); ++j) {
      REQUIRE(restored[i][j] == original[i][j]);
    }
  }
}

SIMPLE_TEST_MAIN()
