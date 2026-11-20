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

#ifndef CLIO_CAE_CORE_CSV_TABLE_H_
#define CLIO_CAE_CORE_CSV_TABLE_H_

#include <cstddef>
#include <string>
#include <vector>

namespace clio::cae::core {

/**
 * A CSV file split into named columns of raw field text.
 *
 * Parsing follows RFC 4180: the first record is the header; fields are
 * separated by ','; a field may be wrapped in double quotes, inside which
 * ',' and line breaks are literal and '""' is one quote. Records end at LF
 * or CRLF. A UTF-8 byte-order mark before the header is dropped, and a
 * final line break or blank trailing lines do not create records.
 *
 * No dependencies beyond the standard library, so it is unit-testable
 * without the runtime.
 */
struct CsvTable {
  std::vector<std::string> names;                // header, one per column
  std::vector<std::vector<std::string>> fields;  // fields[col][row]

  /** Number of data rows (records after the header). */
  size_t NumRows() const { return fields.empty() ? 0 : fields[0].size(); }
};

/**
 * Parse CSV text into a CsvTable.
 *
 * Errors (return false, `err` says where): no header, a duplicate column
 * name, a record with a different field count than the header, an
 * unterminated quoted field, or text after a closing quote. An empty header
 * name becomes "column_<index>" (0-based).
 *
 * @param text Whole file contents.
 * @param table Output table.
 * @param err Output: human-readable error with a 1-based record number.
 * @return true on success.
 */
bool ParseCsv(const std::string& text, CsvTable* table, std::string* err);

/** Element type a CSV column is stored as. */
enum class CsvType { kInt64, kFloat64, kChar };

/**
 * One column encoded as a tensor, ready to store as blobs.
 *
 * Type inference over the column's fields:
 * - kInt64 when every field is a base-10 integer that fits in int64;
 * - else kFloat64 when every non-empty field parses as a double
 *   (std::from_chars, so locale-independent; "nan"/"inf" accepted) and empty
 *   fields become quiet NaN -- an all-empty column is float64 NaN;
 * - else kChar: an N x W array of chars, W = longest field in bytes
 *   (at least 1), each field NUL-padded to W.
 * Leading/trailing spaces are kept for kChar and rejected for numbers, so
 * " 1" makes a column kChar.
 *
 * `description` follows the HDF5 assimilator's tensor string:
 * "tensor<int64, N>", "tensor<float64, N>" or "tensor<char, N, W>".
 * `bytes` is the element image in host byte order (row-major), as the HDF5
 * assimilator stores a dataset read with its native type.
 */
struct CsvColumnTensor {
  CsvType type = CsvType::kFloat64;
  size_t rows = 0;
  size_t width = 0;  // kChar only: bytes per element
  std::string description;
  std::vector<char> bytes;
};

/**
 * Infer a type for one column and encode it.
 * @param fields The column's field texts, one per row.
 * @return The encoded tensor.
 */
CsvColumnTensor EncodeCsvColumn(const std::vector<std::string>& fields);

}  // namespace clio::cae::core

#endif  // CLIO_CAE_CORE_CSV_TABLE_H_
