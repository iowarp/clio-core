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

#include <clio_cae/core/factory/csv_table.h>

#include <charconv>
#include <cstdint>
#include <cstring>
#include <limits>
#include <set>

namespace clio::cae::core {

namespace {

/** Incremental RFC 4180 tokenizer state for one record. */
struct RecordState {
  std::vector<std::string> fields;
  std::string field;
  bool in_quotes = false;
  bool after_quote = false;  // just saw the closing quote of a field
  bool any_quote = false;    // the record had a quoted field
};

/** A record is blank when it is one empty, unquoted field. */
bool IsBlank(const RecordState& r) {
  return r.fields.size() == 1 && r.fields[0].empty() && !r.any_quote;
}

/**
 * Accepts finished records: the first is the header, the rest become rows.
 * Blank lines are held back: dropped at the end of the file, and between
 * records kept as an empty value only when the table has one column
 * (otherwise they fail the field-count check).
 */
class TableBuilder {
 public:
  explicit TableBuilder(CsvTable* t) : t_(t) {}

  bool Add(std::vector<std::string>&& rec, bool blank, size_t recno,
           std::string* err) {
    if (t_->names.empty() && !have_header_) {
      if (blank) {
        *err = "record 1: empty header";
        return false;
      }
      return SetHeader(std::move(rec), err);
    }
    if (blank) {
      ++pending_blank_;
      return true;
    }
    for (; pending_blank_ > 0; --pending_blank_) {
      if (!AddRow({std::string()}, recno - pending_blank_, err)) return false;
    }
    return AddRow(std::move(rec), recno, err);
  }

  bool Finish(std::string* err) {
    if (!have_header_) {
      *err = "no header record";
      return false;
    }
    return true;
  }

 private:
  bool SetHeader(std::vector<std::string>&& rec, std::string* err) {
    std::set<std::string> seen;
    for (size_t i = 0; i < rec.size(); ++i) {
      if (rec[i].empty()) rec[i] = "column_" + std::to_string(i);
      if (!seen.insert(rec[i]).second) {
        *err = "record 1: duplicate column name '" + rec[i] + "'";
        return false;
      }
    }
    t_->names = std::move(rec);
    t_->fields.assign(t_->names.size(), {});
    have_header_ = true;
    return true;
  }

  bool AddRow(std::vector<std::string>&& rec, size_t recno, std::string* err) {
    if (rec.size() != t_->names.size()) {
      *err = "record " + std::to_string(recno) + ": " +
             std::to_string(rec.size()) + " field(s), header has " +
             std::to_string(t_->names.size());
      return false;
    }
    for (size_t c = 0; c < rec.size(); ++c) {
      t_->fields[c].push_back(std::move(rec[c]));
    }
    return true;
  }

  CsvTable* t_;
  bool have_header_ = false;
  size_t pending_blank_ = 0;
};

/**
 * Feed one character outside quotes. Returns false on a syntax error.
 * Sets *end_record when the character ends the record.
 */
bool StepUnquoted(char c, RecordState& r, bool* end_record) {
  if (c == '"') {
    if (!r.field.empty() || r.after_quote) return false;  // quote mid-field
    r.in_quotes = true;
    r.any_quote = true;
    return true;
  }
  if (c == ',') {
    r.fields.push_back(std::move(r.field));
    r.field.clear();
    r.after_quote = false;
    return true;
  }
  if (c == '\n') {
    *end_record = true;
    return true;
  }
  if (r.after_quote) return false;  // text after a closing quote
  r.field += c;
  return true;
}

}  // namespace

bool ParseCsv(const std::string& text, CsvTable* table, std::string* err) {
  *table = CsvTable();
  TableBuilder builder(table);
  RecordState r;
  size_t recno = 1;
  size_t i = 0;
  if (text.compare(0, 3, "\xEF\xBB\xBF") == 0) i = 3;  // UTF-8 BOM
  auto finish_record = [&]() {
    r.fields.push_back(std::move(r.field));
    bool ok = builder.Add(std::move(r.fields), IsBlank(r), recno, err);
    r = RecordState();
    ++recno;
    return ok;
  };
  for (; i < text.size(); ++i) {
    char c = text[i];
    if (r.in_quotes) {
      if (c != '"') {
        r.field += c;
      } else if (i + 1 < text.size() && text[i + 1] == '"') {
        r.field += '"';
        ++i;
      } else {
        r.in_quotes = false;
        r.after_quote = true;
      }
      continue;
    }
    if (c == '\r' && i + 1 < text.size() && text[i + 1] == '\n') continue;
    bool end_record = false;
    if (!StepUnquoted(c, r, &end_record)) {
      *err = "record " + std::to_string(recno) +
             ": stray '\"' or text after a closing quote";
      return false;
    }
    if (end_record && !finish_record()) return false;
  }
  if (r.in_quotes) {
    *err = "record " + std::to_string(recno) + ": unterminated quoted field";
    return false;
  }
  if ((!r.field.empty() || !r.fields.empty() || r.any_quote) &&
      !finish_record()) {
    return false;
  }
  return builder.Finish(err);
}

namespace {

/** Parse a whole field as int64 (optional leading '+'). */
bool ParseInt64(const std::string& s, int64_t* out) {
  const char* b = s.data();
  const char* e = b + s.size();
  if (b != e && *b == '+') ++b;
  if (b == e) return false;
  auto res = std::from_chars(b, e, *out);
  return res.ec == std::errc() && res.ptr == e;
}

/** Parse a whole field as double (optional leading '+'; nan/inf ok). */
bool ParseFloat64(const std::string& s, double* out) {
  const char* b = s.data();
  const char* e = b + s.size();
  if (b != e && *b == '+') ++b;
  if (b == e) return false;
  auto res = std::from_chars(b, e, *out);
  return res.ec == std::errc() && res.ptr == e;
}

template <typename T>
void AppendRaw(const std::vector<T>& v, std::vector<char>* bytes) {
  bytes->resize(v.size() * sizeof(T));
  if (!v.empty()) std::memcpy(bytes->data(), v.data(), bytes->size());
}

bool TryInt64(const std::vector<std::string>& fields, CsvColumnTensor* t) {
  std::vector<int64_t> v(fields.size());
  for (size_t i = 0; i < fields.size(); ++i) {
    if (!ParseInt64(fields[i], &v[i])) return false;
  }
  t->type = CsvType::kInt64;
  t->description = "tensor<int64, " + std::to_string(fields.size()) + ">";
  AppendRaw(v, &t->bytes);
  return true;
}

bool TryFloat64(const std::vector<std::string>& fields, CsvColumnTensor* t) {
  std::vector<double> v(fields.size());
  for (size_t i = 0; i < fields.size(); ++i) {
    if (fields[i].empty()) {
      v[i] = std::numeric_limits<double>::quiet_NaN();
    } else if (!ParseFloat64(fields[i], &v[i])) {
      return false;
    }
  }
  t->type = CsvType::kFloat64;
  t->description = "tensor<float64, " + std::to_string(fields.size()) + ">";
  AppendRaw(v, &t->bytes);
  return true;
}

void EncodeChar(const std::vector<std::string>& fields, CsvColumnTensor* t) {
  size_t w = 1;
  for (const auto& f : fields) w = f.size() > w ? f.size() : w;
  t->type = CsvType::kChar;
  t->width = w;
  t->description = "tensor<char, " + std::to_string(fields.size()) + ", " +
                   std::to_string(w) + ">";
  t->bytes.assign(fields.size() * w, '\0');
  for (size_t i = 0; i < fields.size(); ++i) {
    if (!fields[i].empty()) {
      std::memcpy(t->bytes.data() + i * w, fields[i].data(), fields[i].size());
    }
  }
}

}  // namespace

void EncodeCsvColumn(const std::vector<std::string>& fields,
                     CsvColumnTensor* out) {
  *out = CsvColumnTensor();
  out->rows = fields.size();
  bool all_empty = true;
  for (const auto& f : fields) all_empty = all_empty && f.empty();
  if (!fields.empty() && !all_empty && TryInt64(fields, out)) return;
  if (TryFloat64(fields, out)) return;
  EncodeChar(fields, out);
}

}  // namespace clio::cae::core
