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

#ifndef CLIO_RUNTIME_INCLUDE_VIZ_VIZ_JSON_H_
#define CLIO_RUNTIME_INCLUDE_VIZ_VIZ_JSON_H_

#include <cmath>
#include <cstdio>
#include <string>
#include <type_traits>
#include <vector>

/**
 * A minimal JSON writer for viz route handlers.
 *
 * Deliberately dependency-free and header-only: a ChiMod that ships a `viz/`
 * directory should be able to answer its own routes without inheriting the
 * dashboard's HTTP dependencies. It emits, it never parses.
 */
namespace clio::run::viz {

/** @return @p s as a quoted, escaped JSON string literal. */
inline std::string JsonQuote(const std::string &s) {
  std::string out;
  out.reserve(s.size() + 2);
  out.push_back('"');
  for (unsigned char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          // Control characters must be escaped; anything >= 0x20 (including
          // UTF-8 continuation bytes) passes through untouched.
          char buf[7];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out.push_back(static_cast<char>(c));
        }
    }
  }
  out.push_back('"');
  return out;
}

/** @return @p v rendered as a JSON number, or `null` when not finite (JSON has
 *  no NaN / Infinity, and stats readings do occasionally produce them). */
inline std::string JsonNumber(double v) {
  if (!std::isfinite(v)) {
    return "null";
  }
  char buf[40];
  std::snprintf(buf, sizeof(buf), "%.10g", v);
  return std::string(buf);
}

/**
 * Streaming JSON builder.
 *
 * Tracks nesting so commas and key/value ordering are handled for the caller:
 *
 *   JsonWriter w;
 *   w.BeginObject();
 *   w.Field("node_id", 0);
 *   w.Key("workers").BeginArray();
 *   w.Value("a").Value("b");
 *   w.EndArray();
 *   w.EndObject();
 *   resp.Json(w.Str());
 */
class JsonWriter {
 public:
  JsonWriter &BeginObject() {
    Separator();
    out_ += '{';
    stack_.push_back(Frame{true, 0});
    return *this;
  }

  JsonWriter &EndObject() {
    if (!stack_.empty()) stack_.pop_back();
    out_ += '}';
    return *this;
  }

  JsonWriter &BeginArray() {
    Separator();
    out_ += '[';
    stack_.push_back(Frame{false, 0});
    return *this;
  }

  JsonWriter &EndArray() {
    if (!stack_.empty()) stack_.pop_back();
    out_ += ']';
    return *this;
  }

  /** Emit an object key. The next call emits its value. */
  JsonWriter &Key(const std::string &key) {
    Separator();
    out_ += JsonQuote(key);
    out_ += ':';
    expect_value_ = true;
    return *this;
  }

  /** Emit a string value. */
  JsonWriter &Value(const std::string &v) {
    Separator();
    out_ += JsonQuote(v);
    return *this;
  }

  JsonWriter &Value(const char *v) { return Value(std::string(v ? v : "")); }

  JsonWriter &Value(bool v) {
    Separator();
    out_ += v ? "true" : "false";
    return *this;
  }

  /** Emit an integral or floating-point value. */
  template <typename T,
            typename = std::enable_if_t<std::is_arithmetic_v<T> &&
                                        !std::is_same_v<T, bool>>>
  JsonWriter &Value(T v) {
    Separator();
    if constexpr (std::is_floating_point_v<T>) {
      out_ += JsonNumber(static_cast<double>(v));
    } else if constexpr (std::is_signed_v<T>) {
      out_ += std::to_string(static_cast<long long>(v));
    } else {
      out_ += std::to_string(static_cast<unsigned long long>(v));
    }
    return *this;
  }

  JsonWriter &Null() {
    Separator();
    out_ += "null";
    return *this;
  }

  /** Splice in an already-serialized JSON fragment (e.g. a converted payload).
   *  An empty fragment becomes `null` so the output stays valid JSON. */
  JsonWriter &Raw(const std::string &json) {
    Separator();
    out_ += json.empty() ? "null" : json;
    return *this;
  }

  /** Key + value in one call. */
  template <typename T>
  JsonWriter &Field(const std::string &key, const T &value) {
    Key(key);
    Value(value);
    return *this;
  }

  /** Key + pre-serialized value in one call. */
  JsonWriter &RawField(const std::string &key, const std::string &json) {
    Key(key);
    Raw(json);
    return *this;
  }

  /** @return the JSON built so far. */
  const std::string &Str() const { return out_; }

 private:
  struct Frame {
    bool is_object;
    size_t count;
  };

  /** Emit the ',' that precedes every element after the first. Inside an
   *  object, a value that follows its key must NOT be preceded by a comma,
   *  which is what expect_value_ tracks. */
  void Separator() {
    if (stack_.empty()) {
      return;
    }
    Frame &frame = stack_.back();
    if (frame.is_object && expect_value_) {
      expect_value_ = false;
      return;
    }
    if (frame.count > 0) {
      out_ += ',';
    }
    ++frame.count;
  }

  std::string out_;
  std::vector<Frame> stack_;
  bool expect_value_ = false;
};

}  // namespace clio::run::viz

#endif  // CLIO_RUNTIME_INCLUDE_VIZ_VIZ_JSON_H_
