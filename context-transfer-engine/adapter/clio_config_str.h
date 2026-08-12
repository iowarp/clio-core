/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 *
 * One config-string grammar for both HDF5 connectors.
 *
 *     key=value;key=value;under_info={nested string}
 *
 * This is NOT a house style we invented. It is the dialect the registered HDF5
 * VOL connectors already share, originating in the reference pass-through
 * connector -- semicolon-separated key=value pairs, with braces around a nested
 * connector's own config. Verbatim from the Cache VOL's documentation:
 *
 *   HDF5_VOL_CONNECTOR="cache_ext config=cfg;under_vol=512;under_info={under_vol=0;under_info={}};"
 *
 * Matching it means a CLIO connector can be dropped into a stack a user already
 * knows how to spell. Diverging -- comma separators, say -- would have made a
 * third dialect for no gain. The VFD side has more latitude (HDF5's own drivers
 * do not agree on a format; ROS3 uses a parenthesised form) but uses this same
 * grammar so there is ONE CLIO config dialect rather than two.
 *
 * Delivery differs by connector type and is worth stating, because it is not
 * symmetric:
 *   - VOL: HDF5 PUSHES the string to the connector's info_cls.from_str.
 *   - VFD: HDF5 stores the string on the FAPL and the driver PULLS it with
 *     H5Pget_driver_config_str. There is no H5FD_class_t callback for it.
 */
#ifndef CLIO_CTE_ADAPTER_CLIO_CONFIG_STR_H_
#define CLIO_CTE_ADAPTER_CLIO_CONFIG_STR_H_

#include <cstddef>
#include <map>
#include <string>

namespace clio {
namespace cte {
namespace adapter {

/** Trim ASCII whitespace from both ends. */
inline std::string ConfigTrim(const std::string &s) {
  const char *ws = " \t\n\r\f\v";
  const size_t b = s.find_first_not_of(ws);
  if (b == std::string::npos) return std::string();
  const size_t e = s.find_last_not_of(ws);
  return s.substr(b, e - b + 1);
}

/**
 * Parse a config string into key -> value.
 *
 * Brace-aware: a ';' inside {} belongs to the nested value, so
 * "under_info={under_vol=0;under_info={}}" yields ONE pair whose value is
 * "under_vol=0;under_info={}" with the outer braces removed. Nesting is
 * counted, not merely matched, so a doubly-nested stack survives.
 *
 * Returns false and sets `err` on malformed input. Malformed means: an
 * unbalanced brace, a pair with no '=', or an empty key. Being strict here is
 * deliberate -- a config string is something a user typed, and the failure mode
 * of a lenient parser is that a mistyped knob is silently ignored and the user
 * believes they configured something they did not. That specific illusion is a
 * bug class this connector has already had to fix more than once.
 *
 * An EMPTY string is valid and yields no pairs: "no configuration" is a
 * legitimate thing to say.
 */
inline bool ParseConfigStr(const std::string &input,
                           std::map<std::string, std::string> *out,
                           std::string *err) {
  out->clear();
  err->clear();

  std::string cur;
  int depth = 0;
  auto flush = [&](void) -> bool {
    const std::string pair = ConfigTrim(cur);
    cur.clear();
    if (pair.empty()) return true;  /* tolerate ";;" and a trailing ';' */
    const size_t eq = pair.find('=');
    if (eq == std::string::npos) {
      *err = "config entry '" + pair + "' has no '=' (expected key=value)";
      return false;
    }
    const std::string key = ConfigTrim(pair.substr(0, eq));
    std::string val = ConfigTrim(pair.substr(eq + 1));
    if (key.empty()) {
      *err = "config entry '" + pair + "' has an empty key";
      return false;
    }
    /* Strip ONE layer of braces; whatever is inside is the nested connector's
       own config string and is its business, not ours. */
    if (val.size() >= 2 && val.front() == '{' && val.back() == '}') {
      val = val.substr(1, val.size() - 2);
    }
    (*out)[key] = val;
    return true;
  };

  for (size_t i = 0; i < input.size(); ++i) {
    const char c = input[i];
    if (c == '{') {
      ++depth;
    } else if (c == '}') {
      if (--depth < 0) {
        *err = "unbalanced '}' in config string";
        return false;
      }
    }
    if (c == ';' && depth == 0) {
      if (!flush()) return false;
      continue;
    }
    cur.push_back(c);
  }
  if (depth != 0) {
    *err = "unbalanced '{' in config string";
    return false;
  }
  return flush();
}

/**
 * Interpret a value as a boolean. Accepts the spellings the CLIO_*_CACHE
 * environment variables already accept, so a user does not need different
 * muscle memory for the same switch in a different place.
 */
inline bool ConfigParseBool(const std::string &v, bool *out) {
  if (v == "1" || v == "on" || v == "true" || v == "yes") { *out = true; return true; }
  if (v == "0" || v == "off" || v == "false" || v == "no") { *out = false; return true; }
  return false;
}

/** Interpret a value as a size. Plain digits only -- no unit suffixes, because
 *  half-supported units ("1M" but not "1MiB") are worse than none. */
inline bool ConfigParseSize(const std::string &v, size_t *out) {
  if (v.empty()) return false;
  size_t acc = 0;
  for (char c : v) {
    if (c < '0' || c > '9') return false;
    acc = acc * 10 + static_cast<size_t>(c - '0');
  }
  *out = acc;
  return true;
}

}  // namespace adapter
}  // namespace cte
}  // namespace clio

#endif  // CLIO_CTE_ADAPTER_CLIO_CONFIG_STR_H_
