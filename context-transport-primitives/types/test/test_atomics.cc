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
// All clio_ctp headers are now included via clio_ctp.h in basic_test.h

template <typename AtomicT>
class AtomicTest {
 public:
  AtomicTest() {}

  void Test() {
    // Test arithmetic methods
    PAGE_DIVIDE("Arithmetic") {
      AtomicT atomic = 0;
      REQUIRE(atomic == 0);
      REQUIRE(atomic != 1);
      REQUIRE(atomic + 1 == 1);
      REQUIRE(atomic - 1 == -1);
      atomic += 1;
      REQUIRE(atomic == 1);
      atomic -= 1;
      REQUIRE(atomic == 0);
    }
    // Test constructors
    PAGE_DIVIDE("Constructors") {
      AtomicT atomic = 0;
      REQUIRE(atomic == 0);
      AtomicT atomic2 = atomic;
      REQUIRE(atomic2 == 0);
      AtomicT atomic3(atomic);
      REQUIRE(atomic3 == 0);
      AtomicT atomic4(std::move(atomic));
      REQUIRE(atomic4 == 0);
    }

    // Test assignment operators
    PAGE_DIVIDE("Assignment") {
      AtomicT atomic = 0;
      REQUIRE(atomic == 0);
      AtomicT atomic2;
      atomic2 = atomic;
      REQUIRE(atomic2 == 0);
      AtomicT atomic3;
      atomic3 = std::move(atomic);
      REQUIRE(atomic3 == 0);
    }
  }
};

TEST_CASE("NonAtomic") {
  AtomicTest<ctp::ipc::nonatomic<int>> test;
  test.Test();
}

TEST_CASE("Atomic") {
  AtomicTest<ctp::ipc::atomic<int>> test;
  test.Test();
}
