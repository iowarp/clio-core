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

#include <utility>

#include "basic_test.h"
#include "clio_ctp/data_structures/ipc/tuple_base.h"

void test_argpack0_pass() { std::cout << "HERE0" << std::endl; }

void test_argpack0() {
  ctp::PassArgPack::Call(ctp::ArgPack<>(), test_argpack0_pass);
}

template <typename T1, typename T2, typename T3>
void test_argpack3_pass(T1 x, T2 y, T3 z) {
  REQUIRE(x == 0);
  REQUIRE(y == 1);
  REQUIRE(z == 0);
  std::cout << "HERE3" << std::endl;
}

void test_product1(int b, int c) {
  REQUIRE(b == 1);
  REQUIRE(c == 2);
}

void test_product2(double d, double e) {
  REQUIRE(d == 3);
  REQUIRE(e == 4);
}

template <typename Pack1, typename Pack2>
void test_product(int a, Pack1 &&pack1, int a2, Pack2 &&pack2) {
  REQUIRE(a == 0);
  REQUIRE(a2 == 0);
  ctp::PassArgPack::Call(std::forward<Pack1>(pack1), test_product1);
  ctp::PassArgPack::Call(std::forward<Pack2>(pack2), test_product2);
}

template <typename T1, typename T2, typename T3>
void verify_tuple3(ctp::tuple<T1, T2, T3> &x) {
  REQUIRE(x.Size() == 3);
  REQUIRE(x.template Get<0>() == 0);
  REQUIRE(x.template Get<1>() == 1);
  REQUIRE(x.template Get<2>() == 0);
#ifdef TEST_COMPILER_ERROR
  std::cout << x.Get<3>() << std::endl;
#endif
}

template <typename T1, typename T2, typename T3>
void test_argpack3() {
  // Pass an argpack to a function
  PAGE_DIVIDE("") {
    ctp::PassArgPack::Call(ctp::make_argpack(T1(0), T2(1), T3(0)),
                            test_argpack3_pass<T1, T2, T3>);
  }

  // Pass an argpack containing references to a function
  PAGE_DIVIDE("") {
    T2 y = 1;
    ctp::PassArgPack::Call(ctp::make_argpack(T1(0), T2(y), T3(0)),
                            test_argpack3_pass<T1, T2, T3>);
  }

  // Create a 3-tuple
  PAGE_DIVIDE("") {
    ctp::tuple<T1, T2, T3> x(T1(0), T2(1), T3(0));
    verify_tuple3(x);
  }

  // Copy a tuple
  PAGE_DIVIDE("") {
    ctp::tuple<T1, T2, T3> y(T1(0), T2(1), T3(0));
    ctp::tuple<T1, T2, T3> x(y);
    verify_tuple3(x);
  }

  // Copy assign tuple
  PAGE_DIVIDE("") {
    ctp::tuple<T1, T2, T3> y(T1(0), T2(1), T3(0));
    ctp::tuple<T1, T2, T3> x;
    x = y;
    verify_tuple3(x);
  }

  // Move tuple
  PAGE_DIVIDE("") {
    ctp::tuple<T1, T2, T3> y(T1(0), T2(1), T3(0));
    ctp::tuple<T1, T2, T3> x(std::move(y));
    verify_tuple3(x);
  }

  // Move assign tuple
  PAGE_DIVIDE("") {
    ctp::tuple<T1, T2, T3> y(T1(0), T2(1), T3(0));
    ctp::tuple<T1, T2, T3> x;
    x = std::move(y);
    verify_tuple3(x);
  }

  // Iterate over a tuple
  PAGE_DIVIDE("") {
    ctp::tuple<T1, T2, T3> x(T1(0), T2(1), T3(0));
    ctp::ForwardIterateTuple::Apply(x, [](auto i, auto &arg) constexpr {
      std::cout << "lambda: " << i.Get() << std::endl;
    });
  }

  // Merge two argpacks into a single pack
  PAGE_DIVIDE("") {
    size_t y = ctp::MergeArgPacks::Merge(ctp::make_argpack(T1(0)),
                                          ctp::make_argpack(T2(1), T2(0)))
                   .Size();
    REQUIRE(y == 3);
  }

  // Pass a merged argpack to a function
  PAGE_DIVIDE("") {
    ctp::PassArgPack::Call(
        ctp::MergeArgPacks::Merge(ctp::make_argpack(T1(0)),
                                   ctp::make_argpack(T2(1), T3(0))),
        test_argpack3_pass<T1, T2, T3>);
  }

  // Construct tuple from argpack
  PAGE_DIVIDE("") {
    ctp::tuple<int, int, int> x(ctp::make_argpack(int(10), int(11), int(12)));
    REQUIRE(x.Get<0>() == 10);
    REQUIRE(x.Get<1>() == 11);
    REQUIRE(x.Get<2>() == 12);
  }

  // Product an argpack
  PAGE_DIVIDE("") {
    auto &&pack = ctp::ProductArgPacks::Product(
        0, ctp::make_argpack(1, 2), ctp::make_argpack<double, double>(3, 4));
    REQUIRE(pack.Size() == 4);
  }

  // Product an argpack
  PAGE_DIVIDE("") {
    ctp::PassArgPack::Call(
        ctp::ProductArgPacks::Product(0, ctp::make_argpack(1, 2),
                                       ctp::make_argpack(3.0, 4.0)),
        test_product<ctp::ArgPack<int &&, int &&>,
                     ctp::ArgPack<double &&, double &&>>);
  }
}

TEST_CASE("TestArgpack") {
  test_argpack0();
  test_argpack3<int, double, float>();
}

class DetectCopy {
 public:
  bool is_copied_ = false;
  bool is_moved_ = false;

 public:
  DetectCopy() = default;
  DetectCopy(const DetectCopy &) {
    is_copied_ = true;
    std::cout << "Copy" << std::endl;
  }
  DetectCopy(DetectCopy &&) {
    is_moved_ = true;
    std::cout << "Move" << std::endl;
  }
};

template <typename ArgPackT>
void test_argpack_copy(ArgPackT &&pack) {
  REQUIRE(pack.template Forward<0>().is_copied_ == false);
  REQUIRE(pack.template Forward<0>().is_moved_ == false);
  REQUIRE(pack.template Forward<1>().is_copied_ == false);
  REQUIRE(pack.template Forward<1>().is_moved_ == false);
}

TEST_CASE("TestArgpackCopy") {
  DetectCopy x;
  test_argpack_copy(ctp::make_argpack(x, DetectCopy()));
}