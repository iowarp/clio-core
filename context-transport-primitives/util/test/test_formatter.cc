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

/**
 * Tests for ctp::Formatter, the formatter behind HLOG and ctp::Error.
 *
 * Issue #1172: placeholders were assumed to be exactly "{}", so "{:.1f}"
 * printed the value and then ".1f}" ("silent for 8.25792.1f}s"). The older
 * TestFormatter cases live in test/unit/types, which is not built, so the
 * formatter had no running test at all.
 */

#include "basic_test.h"

#include <clio_ctp/util/formatter.h>

#include <cstddef>
#include <string>

using ctp::Formatter;

TEST_CASE("FormatterPlaceholders") {
  int rank = 0;
  int i = 0;
  REQUIRE(Formatter::format("bucket{}_{}", rank, i) == "bucket0_0");
  REQUIRE(Formatter::format("{}bucket{}", rank, i) == "0bucket0");
  REQUIRE(Formatter::format("bucket{}{}", rank, i) == "bucket00");
  // A count mismatch returns the format string unchanged.
  REQUIRE(Formatter::format("bucket", rank, i) == "bucket");
  REQUIRE(Formatter::format("bucket{}_{}", rank) == "bucket{}_{}");
}

TEST_CASE("FormatterPlainOutputUnchanged") {
  // "{}" must keep producing exactly what operator<< produces.
  REQUIRE(Formatter::format("{}", 8.25792) == "8.25792");
  REQUIRE(Formatter::format("{}", 'c') == "c");
  REQUIRE(Formatter::format("{}", true) == "1");
  REQUIRE(Formatter::format("{} {}", std::string("s"), 3u) == "s 3");
  REQUIRE(Formatter::format("{}", "text") == "text");
}

TEST_CASE("FormatterFixedPrecision") {
  REQUIRE(Formatter::format("silent for {:.1f}s", 8.25792) ==
          "silent for 8.3s");
  REQUIRE(Formatter::format("{:.0f} ops/sec", 1234.5678) == "1235 ops/sec");
  REQUIRE(Formatter::format("{:.3f} us", 0.5) == "0.500 us");
  REQUIRE(Formatter::format("{:.2f}/{:.2f}", 1.0f, -2.346) == "1.00/-2.35");
  REQUIRE(Formatter::format("{:+.1f}", 2.0) == "+2.0");
}

TEST_CASE("FormatterScientificAndGeneral") {
  REQUIRE(Formatter::format("{:.2e}", 12345.0) == "1.23e+04");
  REQUIRE(Formatter::format("{:.3g}", 0.000123456) == "0.000123");
}

TEST_CASE("FormatterIntegers") {
  REQUIRE(Formatter::format("{:x}", 255) == "ff");
  REQUIRE(Formatter::format("{:#X}", 255) == "0XFF");
  REQUIRE(Formatter::format("{:#x}", static_cast<size_t>(4096)) == "0x1000");
  REQUIRE(Formatter::format("{:b}", 5) == "101");
  REQUIRE(Formatter::format("{:o}", 8) == "10");
  REQUIRE(Formatter::format("{:+d}", 7) == "+7");
  REQUIRE(Formatter::format("{:d}", -42) == "-42");
  REQUIRE(Formatter::format("{:x}", -255) == "-ff");
  REQUIRE(Formatter::format("{:d}", 'A') == "65");
}

TEST_CASE("FormatterWidthAndAlignment") {
  REQUIRE(Formatter::format("[{:5}]", 42) == "[   42]");
  REQUIRE(Formatter::format("[{:<5}]", 42) == "[42   ]");
  REQUIRE(Formatter::format("[{:^6}]", 42) == "[  42  ]");
  REQUIRE(Formatter::format("[{:*>4}]", 7) == "[***7]");
  REQUIRE(Formatter::format("[{:05d}]", -42) == "[-0042]");
  REQUIRE(Formatter::format("[{:#06x}]", 255) == "[0x00ff]");
  REQUIRE(Formatter::format("[{:8.2f}]", 3.14159) == "[    3.14]");
  REQUIRE(Formatter::format("[{:6}]", std::string("ab")) == "[ab    ]");
  REQUIRE(Formatter::format("[{:>6}]", "ab") == "[    ab]");
  REQUIRE(Formatter::format("{:.3}", std::string("abcdef")) == "abc");
}

TEST_CASE("FormatterBraces") {
  REQUIRE(Formatter::format("{{}} {}", 1) == "{} 1");
  REQUIRE(Formatter::format("set {{{}}}", 5) == "set {5}");
  REQUIRE(Formatter::format("a { b {}", 2) == "a { b 2");
  REQUIRE(Formatter::format("json {\"k\": {}}", 3) == "json {\"k\": 3}");
  REQUIRE(Formatter::format("no args, a } here") == "no args, a } here");
  // Positional fields are not supported: printed literally, so the count no
  // longer matches and the format string comes back unchanged.
  REQUIRE(Formatter::format("{0}", 1) == "{0}");
}
