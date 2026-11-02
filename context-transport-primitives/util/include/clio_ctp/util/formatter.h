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

#ifndef CTP_ERROR_SERIALIZER_H
#define CTP_ERROR_SERIALIZER_H

#include <cmath>
#include <cstring>
#include <iomanip>
#include <list>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "clio_ctp/types/argpack.h"

// MSan: unpoison strings produced by uninstrumented libstdc++ stringstream
#include "clio_ctp/util/msan.h"


namespace ctp {

/**
 * Minimal std::format-style formatter used by HLOG and the error types.
 *
 * Placeholders are `{}` or `{:spec}`, filled in order. The spec follows the
 * std::format grammar subset [[fill]align][sign][#][0][width][.precision]
 * [type], with align one of `<` `>` `^` and type one of
 * `f F e E g G x X o b B d s c`. `{{` and `}}` produce literal braces.
 *
 * A `{` that does not open a valid placeholder (no closing `}`, or text that
 * is not a spec) is printed literally. If the number of placeholders differs
 * from the number of arguments the format string is returned unchanged, so a
 * mismatched call degrades to an unformatted but readable message.
 */
class Formatter {
 public:
  /** A parsed `{:spec}` placeholder. */
  struct Spec {
    char fill_ = ' ';     /**< Padding character */
    char align_ = '\0';   /**< '<', '>', '^', or '\0' for the default */
    char sign_ = '\0';    /**< '+', ' ', or '\0' (only '-' shown) */
    bool alt_ = false;    /**< '#': 0x / 0 / 0b prefix for integers */
    bool zero_ = false;   /**< '0': pad numbers with zeros after the sign */
    int width_ = -1;      /**< Minimum field width, -1 if unset */
    int precision_ = -1;  /**< Precision, -1 if unset */
    char type_ = '\0';    /**< Presentation type, '\0' if unset */

    /** @return true when the placeholder was a plain `{}`. */
    bool IsDefault() const {
      return align_ == '\0' && sign_ == '\0' && !alt_ && !zero_ &&
             width_ < 0 && precision_ < 0 && type_ == '\0';
    }
  };

  /**
   * Format @p fmt, replacing each placeholder with the next argument.
   *
   * @param fmt the format string
   * @param args the values to substitute, in order
   * @return the formatted string, or @p fmt unchanged if the number of
   *         placeholders does not match the number of arguments
   */
  template <typename... Args>
  static std::string format(const std::string &fmt, Args &&...args) {
    std::vector<std::string> literals;
    std::vector<Spec> specs;
    Parse(fmt, literals, specs);
    if (specs.size() != sizeof...(Args)) {
      std::string unchanged = fmt;
      CTP_MSAN_UNPOISON_STRING(unchanged);
      return unchanged;
    }
    std::string result = literals[0];
    size_t i = 0;
    auto emit = [&](auto &&arg) {
      result += FormatArg(specs[i], arg);
      result += literals[i + 1];
      ++i;
    };
    (emit(std::forward<Args>(args)), ...);
    (void)emit;
    CTP_MSAN_UNPOISON_STRING(result);
    return result;
  }

  /**
   * Split @p fmt into literal text and placeholder specs.
   *
   * @param fmt the format string
   * @param literals receives N+1 literal segments (escapes already resolved)
   * @param specs receives the N placeholder specs, in order
   */
  static void Parse(const std::string &fmt, std::vector<std::string> &literals,
                    std::vector<Spec> &specs) {
    literals.emplace_back();
    size_t i = 0;
    const size_t n = fmt.size();
    while (i < n) {
      char c = fmt[i];
      if (c == '{' && i + 1 < n && fmt[i + 1] == '{') {
        literals.back() += '{';
        i += 2;
        continue;
      }
      if (c == '}' && i + 1 < n && fmt[i + 1] == '}') {
        literals.back() += '}';
        i += 2;
        continue;
      }
      if (c == '{') {
        size_t close = fmt.find('}', i + 1);
        Spec spec;
        if (close != std::string::npos &&
            ParseSpec(std::string_view(fmt).substr(i + 1, close - i - 1),
                      spec)) {
          specs.push_back(spec);
          literals.emplace_back();
          i = close + 1;
          continue;
        }
      }
      literals.back() += c;
      ++i;
    }
  }

  /**
   * Parse the text between `{` and `}`.
   *
   * @param field the text inside the braces: empty, or `:` followed by a spec
   * @param spec receives the parsed spec
   * @return true if @p field is a placeholder this formatter supports
   */
  static bool ParseSpec(std::string_view field, Spec &spec) {
    if (field.empty()) {
      return true;
    }
    if (field[0] != ':') {
      return false;
    }
    std::string_view s = field.substr(1);
    size_t pos = 0;
    auto is_align = [](char a) { return a == '<' || a == '>' || a == '^'; };
    if (s.size() >= 2 && is_align(s[1])) {
      spec.fill_ = s[0];
      spec.align_ = s[1];
      pos = 2;
    } else if (!s.empty() && is_align(s[0])) {
      spec.align_ = s[0];
      pos = 1;
    }
    if (pos < s.size() && (s[pos] == '+' || s[pos] == '-' || s[pos] == ' ')) {
      spec.sign_ = s[pos] == '-' ? '\0' : s[pos];
      ++pos;
    }
    if (pos < s.size() && s[pos] == '#') {
      spec.alt_ = true;
      ++pos;
    }
    if (pos < s.size() && s[pos] == '0') {
      spec.zero_ = true;
      ++pos;
    }
    spec.width_ = ParseInt(s, pos);
    if (pos < s.size() && s[pos] == '.') {
      ++pos;
      spec.precision_ = ParseInt(s, pos);
      if (spec.precision_ < 0) {
        return false;
      }
    }
    if (pos < s.size()) {
      if (std::string_view("fFeEgGxXobBdsc").find(s[pos]) ==
          std::string_view::npos) {
        return false;
      }
      spec.type_ = s[pos++];
    }
    return pos == s.size();
  }

 private:
  /**
   * Read a run of decimal digits.
   *
   * @param s the text being parsed
   * @param pos the read position; advanced past the digits
   * @return the value, or -1 if there were no digits
   */
  static int ParseInt(std::string_view s, size_t &pos) {
    int value = -1;
    while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') {
      value = (value < 0 ? 0 : value * 10) + (s[pos] - '0');
      ++pos;
    }
    return value;
  }

  /** True for the character types, which print as characters by default. */
  template <typename T>
  static constexpr bool kIsCharType =
      std::is_same_v<T, char> || std::is_same_v<T, signed char> ||
      std::is_same_v<T, unsigned char> || std::is_same_v<T, wchar_t> ||
      std::is_same_v<T, char16_t> || std::is_same_v<T, char32_t>;

  /**
   * Render one argument according to its spec.
   *
   * @param spec the placeholder spec
   * @param arg the argument
   * @return the rendered, padded text
   */
  template <typename T>
  static std::string FormatArg(const Spec &spec, T &&arg) {
    using D = std::decay_t<T>;
    std::ostringstream ss;
    if (spec.IsDefault()) {
      ss << arg;  // Exactly the historical `{}` output.
      return ss.str();
    }
    bool numeric = false;
    std::string body;
    if constexpr (std::is_floating_point_v<D>) {
      numeric = true;
      body = FormatFloat(spec, static_cast<double>(arg));
    } else if constexpr (std::is_integral_v<D> && !std::is_same_v<D, bool>) {
      if (kIsCharType<D> && (spec.type_ == '\0' || spec.type_ == 'c')) {
        ss << arg;
        body = ss.str();
      } else {
        numeric = true;
        body = FormatInteger(spec, arg);
      }
    } else {
      ss << arg;
      body = ss.str();
      if constexpr (std::is_convertible_v<const D &, std::string_view>) {
        if (spec.precision_ >= 0 &&
            body.size() > static_cast<size_t>(spec.precision_)) {
          body.resize(static_cast<size_t>(spec.precision_));
        }
      }
    }
    return Pad(spec, body, numeric);
  }

  /**
   * Render a floating-point value for the f/e/g presentation types.
   *
   * @param spec the placeholder spec
   * @param value the value
   * @return the text, without padding
   */
  static std::string FormatFloat(const Spec &spec, double value) {
    std::ostringstream ss;
    char t = spec.type_;
    if (t == 'f' || t == 'F') {
      ss << std::fixed;
    } else if (t == 'e' || t == 'E') {
      ss << std::scientific;
    }
    if (t == 'F' || t == 'E' || t == 'G') {
      ss << std::uppercase;
    }
    if (spec.precision_ >= 0) {
      ss << std::setprecision(spec.precision_);
    }
    if (spec.sign_ == '+') {
      ss << std::showpos;
    }
    ss << value;
    std::string out = ss.str();
    if (spec.sign_ == ' ' && !std::signbit(value)) {
      out.insert(out.begin(), ' ');
    }
    return out;
  }

  /**
   * Render an integer for the d/x/X/o/b/B presentation types.
   *
   * @param spec the placeholder spec
   * @param value the value
   * @return the text, without padding
   */
  template <typename I>
  static std::string FormatInteger(const Spec &spec, I value) {
    bool negative = false;
    unsigned long long mag = 0;
    if constexpr (std::is_signed_v<I>) {
      negative = value < 0;
      mag = negative ? 0ULL - static_cast<unsigned long long>(value)
                     : static_cast<unsigned long long>(value);
    } else {
      mag = static_cast<unsigned long long>(value);
    }
    unsigned base = 10;
    const char *prefix = "";
    const char *digits = "0123456789abcdef";
    switch (spec.type_) {
      case 'x': base = 16; prefix = "0x"; break;
      case 'X': base = 16; prefix = "0X"; digits = "0123456789ABCDEF"; break;
      case 'o': base = 8; prefix = "0"; break;
      case 'b': base = 2; prefix = "0b"; break;
      case 'B': base = 2; prefix = "0B"; break;
      default: break;
    }
    std::string num;
    do {
      num.insert(num.begin(), digits[mag % base]);
      mag /= base;
    } while (mag != 0);
    std::string out;
    if (negative) {
      out += '-';
    } else if (spec.sign_ == '+' || spec.sign_ == ' ') {
      out += spec.sign_;
    }
    if (spec.alt_ && !(spec.type_ == 'o' && num == "0")) {
      out += prefix;
    }
    return out + num;
  }

  /**
   * Pad @p body to the spec's width.
   *
   * @param spec the placeholder spec
   * @param body the rendered value
   * @param numeric whether the value is a number (right-aligned by default,
   *        and eligible for '0' padding after its sign and prefix)
   * @return the padded text
   */
  static std::string Pad(const Spec &spec, const std::string &body,
                         bool numeric) {
    if (spec.width_ < 0 || body.size() >= static_cast<size_t>(spec.width_)) {
      return body;
    }
    size_t pad = static_cast<size_t>(spec.width_) - body.size();
    if (numeric && spec.zero_ && spec.align_ == '\0') {
      size_t lead = 0;
      while (lead < body.size() &&
             std::string_view("+- ").find(body[lead]) != std::string_view::npos) {
        ++lead;
      }
      if (body.size() > lead + 1 && body[lead] == '0' &&
          std::string_view("xXbB").find(body[lead + 1]) !=
              std::string_view::npos) {
        lead += 2;
      }
      return body.substr(0, lead) + std::string(pad, '0') + body.substr(lead);
    }
    char align = spec.align_ != '\0' ? spec.align_ : (numeric ? '>' : '<');
    if (align == '<') {
      return body + std::string(pad, spec.fill_);
    }
    if (align == '>') {
      return std::string(pad, spec.fill_) + body;
    }
    size_t left = pad / 2;
    return std::string(left, spec.fill_) + body +
           std::string(pad - left, spec.fill_);
  }
};

}  // namespace ctp

#endif  // CTP_ERROR_SERIALIZER_H
