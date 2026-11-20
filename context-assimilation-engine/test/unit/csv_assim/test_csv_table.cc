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
 * test_csv_table.cc - CSV parsing and column encoding, no runtime.
 *
 * Covers the RFC 4180 cases the CSV assimilator relies on (quotes, escaped
 * quotes, embedded commas and line breaks, CRLF, BOM, trailing newline and
 * blank lines, empty header names), the error cases, and type inference
 * (int64, float64 with NaN for empty, char for text), checking the encoded
 * bytes against values written out by hand.
 */

#include <clio_cae/core/factory/csv_table.h>
#include <clio_cae/core/factory/glob_match.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using clio::cae::core::CsvColumnTensor;
using clio::cae::core::CsvTable;
using clio::cae::core::CsvType;
using clio::cae::core::EncodeCsvColumn;
using clio::cae::core::MatchesDatasetFilter;
using clio::cae::core::ParseCsv;

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what);
  if (!ok) ++g_failures;
}

bool Parses(const std::string& text, CsvTable* t) {
  std::string err;
  bool ok = ParseCsv(text, t, &err);
  if (!ok) std::printf("       parse error: %s\n", err.c_str());
  return ok;
}

bool Fails(const std::string& text, const std::string& want_in_err) {
  CsvTable t;
  std::string err;
  bool failed = !ParseCsv(text, &t, &err);
  if (failed) std::printf("       error: %s\n", err.c_str());
  return failed && err.find(want_in_err) != std::string::npos;
}

void TestBasicAndQuoting() {
  CsvTable t;
  bool ok = Parses(
      "\xEF\xBB\xBFid,name,note\r\n"
      "1,\"Smith, J\",\"say \"\"hi\"\"\"\r\n"
      "2,plain,\"two\nlines\"\r\n",
      &t);
  Check(ok && t.names == std::vector<std::string>({"id", "name", "note"}),
        "BOM dropped, CRLF header split into 3 names");
  Check(ok && t.NumRows() == 2, "2 data rows (final CRLF adds none)");
  Check(ok && t.fields[1][0] == "Smith, J", "quoted comma kept");
  Check(ok && t.fields[2][0] == "say \"hi\"", "doubled quote -> one quote");
  Check(ok && t.fields[2][1] == "two\nlines", "quoted line break kept");
}

void TestBlankLinesAndEmptyNames() {
  CsvTable t;
  bool ok = Parses("a,,c\n1,2,3\n\n\n", &t);
  Check(ok && t.names[1] == "column_1", "empty header name -> column_1");
  Check(ok && t.NumRows() == 1, "trailing blank lines dropped");
  ok = Parses("v\n1\n\n3\n", &t);
  Check(ok && t.NumRows() == 3 && t.fields[0][1].empty(),
        "one-column table: interior blank line is an empty value");
  ok = Parses("x,y\n1,2", &t);
  Check(ok && t.NumRows() == 1 && t.fields[1][0] == "2",
        "last record without a line break");
  ok = Parses("x,y\n", &t);
  Check(ok && t.NumRows() == 0 && t.names.size() == 2, "header only");
}

void TestErrors() {
  Check(Fails("", "no header"), "empty file rejected");
  Check(Fails("a,a\n1,2\n", "duplicate column name 'a'"), "duplicate name");
  Check(Fails("a,b\n1,2\n3\n", "record 3: 1 field(s), header has 2"),
        "short record reported with its record number");
  Check(Fails("a,b\n1,2\n\n4,5\n", "record 3"),
        "interior blank line in a 2-column table");
  Check(Fails("a\n\"open\n", "unterminated"), "unterminated quote");
  Check(Fails("a\n\"x\"y\n", "record 2"), "text after closing quote");
  Check(Fails("a\nx\"y\n", "record 2"), "quote inside unquoted field");
}

template <typename T>
bool BytesEqual(const CsvColumnTensor& t, const std::vector<T>& want) {
  return t.bytes.size() == want.size() * sizeof(T) &&
         std::memcmp(t.bytes.data(), want.data(), t.bytes.size()) == 0;
}

void TestTypes() {
  CsvColumnTensor t = EncodeCsvColumn({"1", "-2", "+3", "9223372036854775807"});
  Check(t.type == CsvType::kInt64 && t.description == "tensor<int64, 4>" &&
            BytesEqual<int64_t>(t, {1, -2, 3, INT64_MAX}),
        "int64 column");
  t = EncodeCsvColumn({"1", "", "2.5"});
  bool f_ok = t.type == CsvType::kFloat64 &&
              t.description == "tensor<float64, 3>" &&
              t.bytes.size() == 3 * sizeof(double);
  double got[3] = {0, 0, 0};
  if (f_ok) std::memcpy(got, t.bytes.data(), sizeof(got));
  Check(f_ok && got[0] == 1.0 && std::isnan(got[1]) && got[2] == 2.5,
        "int with an empty field -> float64 with NaN");
  t = EncodeCsvColumn({"1e3", "nan", "-inf", "0.1"});
  Check(t.type == CsvType::kFloat64 && t.rows == 4, "exponent/nan/inf parse");
  t = EncodeCsvColumn({"9223372036854775808"});
  Check(t.type == CsvType::kFloat64, "int64 overflow -> float64");
  t = EncodeCsvColumn({"", ""});
  Check(t.type == CsvType::kFloat64 && t.description == "tensor<float64, 2>",
        "all-empty column -> float64 NaN");
  t = EncodeCsvColumn({"ab", "", "xyz", " 1"});
  std::vector<char> chars = {'a', 'b', 0, 0, 0, 0, 'x', 'y', 'z', ' ', '1', 0};
  Check(t.type == CsvType::kChar && t.width == 3 &&
            t.description == "tensor<char, 4, 3>" && t.bytes == chars,
        "text column -> NUL-padded char[N][W]");
  t = EncodeCsvColumn({});
  Check(t.type == CsvType::kFloat64 && t.description == "tensor<float64, 0>" &&
            t.bytes.empty(),
        "zero-row column");
}

void TestFilter() {
  std::vector<std::string> inc = {"/solar_*"};
  std::vector<std::string> exc = {"/solar_Minute"};
  Check(MatchesDatasetFilter("/solar_GHI", inc, exc), "include glob selects");
  Check(!MatchesDatasetFilter("/solar_Minute", inc, exc), "exclude wins");
  Check(!MatchesDatasetFilter("/load_mw", inc, exc), "unmatched rejected");
  Check(MatchesDatasetFilter("/load_mw", {}, exc), "empty include = all");
}

}  // namespace

int main() {
  TestBasicAndQuoting();
  TestBlankLinesAndEmptyNames();
  TestErrors();
  TestTypes();
  TestFilter();
  std::printf("%s (%d failure(s))\n",
              g_failures ? "CSV TABLE TEST FAILED" : "CSV TABLE TEST PASSED",
              g_failures);
  return g_failures ? 1 : 0;
}
