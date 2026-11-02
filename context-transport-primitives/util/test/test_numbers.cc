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

#include "basic_test.h"
#include "clio_ctp/types/numbers.h"

//------------------------------------------------------------------------------
// BitWidth Tests
//------------------------------------------------------------------------------

TEST_CASE("BitWidth - Zero") {
  REQUIRE(ctp::BitWidth(0) == 0);
}

TEST_CASE("BitWidth - Powers of two") {
  REQUIRE(ctp::BitWidth(1) == 1);
  REQUIRE(ctp::BitWidth(2) == 2);
  REQUIRE(ctp::BitWidth(4) == 3);
  REQUIRE(ctp::BitWidth(8) == 4);
  REQUIRE(ctp::BitWidth(16) == 5);
  REQUIRE(ctp::BitWidth(32) == 6);
  REQUIRE(ctp::BitWidth(64) == 7);
  REQUIRE(ctp::BitWidth(128) == 8);
  REQUIRE(ctp::BitWidth(256) == 9);
  REQUIRE(ctp::BitWidth(512) == 10);
  REQUIRE(ctp::BitWidth(1024) == 11);
  REQUIRE(ctp::BitWidth(1ULL << 20) == 21);
  REQUIRE(ctp::BitWidth(1ULL << 32) == 33);
}

TEST_CASE("BitWidth - Non-powers of two") {
  REQUIRE(ctp::BitWidth(3) == 2);
  REQUIRE(ctp::BitWidth(5) == 3);
  REQUIRE(ctp::BitWidth(6) == 3);
  REQUIRE(ctp::BitWidth(7) == 3);
  REQUIRE(ctp::BitWidth(9) == 4);
  REQUIRE(ctp::BitWidth(10) == 4);
  REQUIRE(ctp::BitWidth(15) == 4);
  REQUIRE(ctp::BitWidth(17) == 5);
  REQUIRE(ctp::BitWidth(100) == 7);
  REQUIRE(ctp::BitWidth(1000) == 10);
}

TEST_CASE("BitWidth - equals FloorLog2 plus one for positive values") {
  size_t values[] = {1, 2, 3, 5, 7, 9, 15, 17, 63, 64, 100, 1000,
                     1ULL << 20, 1ULL << 32};
  for (size_t v : values) {
    REQUIRE(ctp::BitWidth(v) == ctp::FloorLog2(v) + 1);
  }
}

//------------------------------------------------------------------------------
// FloorLog2 Tests
//------------------------------------------------------------------------------

TEST_CASE("FloorLog2 - Powers of two") {
  REQUIRE(ctp::FloorLog2(1) == 0);
  REQUIRE(ctp::FloorLog2(2) == 1);
  REQUIRE(ctp::FloorLog2(4) == 2);
  REQUIRE(ctp::FloorLog2(8) == 3);
  REQUIRE(ctp::FloorLog2(16) == 4);
  REQUIRE(ctp::FloorLog2(32) == 5);
  REQUIRE(ctp::FloorLog2(64) == 6);
  REQUIRE(ctp::FloorLog2(128) == 7);
  REQUIRE(ctp::FloorLog2(256) == 8);
  REQUIRE(ctp::FloorLog2(512) == 9);
  REQUIRE(ctp::FloorLog2(1024) == 10);
  REQUIRE(ctp::FloorLog2(1ULL << 20) == 20);
  REQUIRE(ctp::FloorLog2(1ULL << 32) == 32);
}

TEST_CASE("FloorLog2 - Non-powers of two") {
  REQUIRE(ctp::FloorLog2(3) == 1);
  REQUIRE(ctp::FloorLog2(5) == 2);
  REQUIRE(ctp::FloorLog2(6) == 2);
  REQUIRE(ctp::FloorLog2(7) == 2);
  REQUIRE(ctp::FloorLog2(9) == 3);
  REQUIRE(ctp::FloorLog2(10) == 3);
  REQUIRE(ctp::FloorLog2(15) == 3);
  REQUIRE(ctp::FloorLog2(17) == 4);
  REQUIRE(ctp::FloorLog2(100) == 6);
  REQUIRE(ctp::FloorLog2(1000) == 9);
}

TEST_CASE("FloorLog2 - Equals CeilLog2 for powers of two") {
  size_t powers[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
                            1ULL << 20, 1ULL << 32};
  for (size_t p : powers) {
    REQUIRE(ctp::FloorLog2(p) == ctp::CeilLog2(p));
  }
}

//------------------------------------------------------------------------------
// CeilLog2 Tests
//------------------------------------------------------------------------------

TEST_CASE("CeilLog2 - Edge cases: 0 and 1") {
  REQUIRE(ctp::CeilLog2(0) == 0);
  REQUIRE(ctp::CeilLog2(1) == 0);
}

TEST_CASE("CeilLog2 - Powers of two") {
  REQUIRE(ctp::CeilLog2(1) == 0);
  REQUIRE(ctp::CeilLog2(2) == 1);
  REQUIRE(ctp::CeilLog2(4) == 2);
  REQUIRE(ctp::CeilLog2(8) == 3);
  REQUIRE(ctp::CeilLog2(16) == 4);
  REQUIRE(ctp::CeilLog2(32) == 5);
  REQUIRE(ctp::CeilLog2(64) == 6);
  REQUIRE(ctp::CeilLog2(128) == 7);
  REQUIRE(ctp::CeilLog2(256) == 8);
  REQUIRE(ctp::CeilLog2(512) == 9);
  REQUIRE(ctp::CeilLog2(1024) == 10);
  REQUIRE(ctp::CeilLog2(1ULL << 20) == 20);
  REQUIRE(ctp::CeilLog2(1ULL << 32) == 32);
}

TEST_CASE("CeilLog2 - Non-powers of two") {
  REQUIRE(ctp::CeilLog2(3) == 2);
  REQUIRE(ctp::CeilLog2(5) == 3);
  REQUIRE(ctp::CeilLog2(6) == 3);
  REQUIRE(ctp::CeilLog2(7) == 3);
  REQUIRE(ctp::CeilLog2(9) == 4);
  REQUIRE(ctp::CeilLog2(10) == 4);
  REQUIRE(ctp::CeilLog2(15) == 4);
  REQUIRE(ctp::CeilLog2(17) == 5);
  REQUIRE(ctp::CeilLog2(100) == 7);
  REQUIRE(ctp::CeilLog2(1000) == 10);
}

TEST_CASE("CeilLog2 - Ceil is at least Floor for all values") {
  size_t values[] = {1, 2, 3, 5, 7, 9, 15, 17, 63, 64, 100, 1000,
                            1ULL << 20, 1ULL << 32};
  for (size_t v : values) {
    REQUIRE(ctp::CeilLog2(v) >= ctp::FloorLog2(v));
  }
}

TEST_CASE("CeilLog2 - Ceil exceeds Floor only for non-powers-of-two") {
  size_t non_powers[] = {3, 5, 6, 7, 9, 10, 15, 17, 100, 1000};
  for (size_t v : non_powers) {
    REQUIRE(ctp::CeilLog2(v) == ctp::FloorLog2(v) + 1);
  }
}

TEST_CASE("CeilLog2 - Large value 1ULL << 20") {
  size_t v = 1ULL << 20;
  REQUIRE(ctp::CeilLog2(v) == 20);
  REQUIRE(ctp::FloorLog2(v) == 20);
}

TEST_CASE("CeilLog2 - Large value 1ULL << 32") {
  size_t v = 1ULL << 32;
  REQUIRE(ctp::CeilLog2(v) == 32);
  REQUIRE(ctp::FloorLog2(v) == 32);
}
