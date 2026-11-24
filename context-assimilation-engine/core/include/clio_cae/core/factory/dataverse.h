/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 */

#ifndef CLIO_CAE_CORE_FACTORY_DATAVERSE_H_
#define CLIO_CAE_CORE_FACTORY_DATAVERSE_H_

// Dataverse sources for CAE: a whole dataset, named by its persistent id,
// with OMNI dataset_filter patterns that pick files and members of zip files.
//
//   - src: "dataverse://dataverse.harvard.edu/doi:10.7910/DVN/LCCLLW"
//     dst: "iowarp::golf"            # or file::/some/dir
//     format: "hdf5"                 # or binary
//     dataset_filter:
//       include_patterns:
//         - "Documentation.zip/*"         # members of one zip
//         - "P*.zip/*_stream_data.hdf5"   # one kind of member of many zips
//       exclude_patterns:
//         - "*.mp4"
//
// Every selected object has a path:
//   <directoryLabel/>filename              a file of the dataset
//   <directoryLabel/>name.zip/<member>     a member of a zip file
// A pattern that contains ".zip/" selects members: the part up to and including
// ".zip" is matched against the zip file's path and the rest against the member
// name, and only zip files some member pattern names are opened. Any other
// pattern selects whole files. Patterns are fnmatch(3) globs in which '*' also
// matches '/'. No include pattern selects every file (zips are not opened).
// An exclude pattern drops an object whose path it matches. With an iowarp::
// destination and format hdf5, patterns that start with "hdf5:" select
// datasets inside the HDF5 files instead (dataverse_file_assimilator.h).
//
// Members are read straight out of the remote zip with HTTP range requests
// (the central directory, then each selected member), so a few HDF5 files can
// be taken from a 2 GB archive of video without downloading the video.
//
// This header holds the parts with no network and no runtime -- the URL
// parser, a small JSON reader for the Dataverse API, the pattern rules and the
// ZIP/ZIP64 directory parser -- so an offline unit test covers them.

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fnmatch.h>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace clio::cae::core {

// --------------------------------------------------------------------------
// Source URL
// --------------------------------------------------------------------------

/** A dataset on a Dataverse installation. */
struct DataverseSource {
  std::string base;  ///< "https://dataverse.harvard.edu"
  std::string pid;   ///< "doi:10.7910/DVN/LCCLLW"
  /** Set for a plain web file (an http(s) URL, no Dataverse API): the one
   *  file of the "dataset", whose zip members can be selected the same way. */
  std::string url;
};

/** The file name a plain web URL names: its last path segment. */
inline std::string WebFileName(const std::string &url) {
  std::string p = url.substr(0, url.find_first_of("?#"));
  while (!p.empty() && p.back() == '/') p.pop_back();
  size_t s = p.rfind('/');
  return s == std::string::npos ? p : p.substr(s + 1);
}

/**
 * Parse "dataverse://<host>/<persistent id>" (https is implied) or
 * "dataverse::https://<host>/<persistent id>". The persistent id keeps its
 * slashes ("doi:10.7910/DVN/LCCLLW", "hdl:1902.1/12345").
 *
 * A plain "http(s)://<host>/<path>" is accepted too: a single web file (for
 * example a data release's zip), staged like a one-file dataset. Its pid is
 * "web:<host>/<path>", which names its cache directory.
 */
inline bool ParseDataverseUrl(const std::string &src, DataverseSource *out,
                              std::string *err) {
  for (const char *s : {"https://", "http://"}) {
    const size_t n = std::strlen(s);
    if (src.compare(0, n, s) != 0) continue;
    const std::string rest = src.substr(n, src.find_first_of("?#") - n);
    const size_t slash = rest.find('/');
    if (slash == std::string::npos || slash == 0 || WebFileName(src).empty()) {
      *err = "expected http(s)://<host>/<path to a file>, got " + src;
      return false;
    }
    out->base = s + rest.substr(0, slash);
    out->pid = "web:" + rest;
    out->url = src;
    return true;
  }
  std::string rest;
  std::string scheme = "https://";
  if (src.compare(0, 12, "dataverse://") == 0) {
    rest = src.substr(12);
  } else if (src.compare(0, 11, "dataverse::") == 0) {
    rest = src.substr(11);
    for (const char *s : {"https://", "http://"}) {
      if (rest.compare(0, std::strlen(s), s) == 0) {
        scheme = s;
        rest = rest.substr(std::strlen(s));
        break;
      }
    }
  } else {
    *err = "not a dataverse source: " + src;
    return false;
  }
  size_t slash = rest.find('/');
  if (slash == std::string::npos || slash == 0 || slash + 1 >= rest.size()) {
    *err = "expected dataverse://<host>/<persistent id>, got " + src;
    return false;
  }
  out->base = scheme + rest.substr(0, slash);
  out->pid = rest.substr(slash + 1);
  if (out->pid.find(':') == std::string::npos) {
    *err = "persistent id '" + out->pid +
           "' has no scheme (expected doi:... or hdl:...)";
    return false;
  }
  return true;
}

/** RFC 3986 percent-encoding of a query value. */
inline std::string DvUrlEncode(const std::string &s) {
  static const char *kHex = "0123456789ABCDEF";
  std::string o;
  for (unsigned char c : s) {
    if (std::isalnum(c) || c == '-' || c == '.' || c == '_' || c == '~') {
      o += static_cast<char>(c);
    } else {
      o += '%';
      o += kHex[c >> 4];
      o += kHex[c & 15];
    }
  }
  return o;
}

inline std::string DataverseDatasetApiUrl(const DataverseSource &s) {
  return s.base + "/api/datasets/:persistentId/?persistentId=" +
         DvUrlEncode(s.pid);
}

inline std::string DataverseFileUrl(const DataverseSource &s, int64_t id) {
  if (!s.url.empty()) return s.url;
  return s.base + "/api/access/datafile/" + std::to_string(id);
}

// --------------------------------------------------------------------------
// JSON (just enough for the Dataverse native API)
// --------------------------------------------------------------------------

struct Json {
  enum Kind { kNull, kBool, kNum, kStr, kArr, kObj } kind = kNull;
  bool b = false;
  double num = 0;
  std::string str;  ///< string value, or the literal text of a number
  std::vector<Json> arr;   ///< array items, or an object's values
  std::vector<std::string> keys;  ///< an object's keys, parallel to arr

  const Json *Get(const std::string &k) const {
    if (kind != kObj) return nullptr;
    for (size_t i = 0; i < keys.size(); ++i)
      if (keys[i] == k) return &arr[i];
    return nullptr;
  }
  std::string Str(const std::string &k, const std::string &dflt = "") const {
    const Json *v = Get(k);
    return v && v->kind == kStr ? v->str : dflt;
  }
  int64_t Int(const std::string &k, int64_t dflt = -1) const {
    const Json *v = Get(k);
    if (!v || v->kind != kNum) return dflt;
    return std::strtoll(v->str.c_str(), nullptr, 10);
  }
};

namespace dv_detail {

class JsonParser {
 public:
  explicit JsonParser(const std::string &t) : t_(t) {}
  bool Parse(Json *out, std::string *err) {
    if (!Value(out, 0)) {
      *err = "JSON parse error at byte " + std::to_string(p_) + ": " + why_;
      return false;
    }
    Ws();
    if (p_ != t_.size()) {
      *err = "trailing bytes after JSON value at byte " + std::to_string(p_);
      return false;
    }
    return true;
  }

 private:
  void Ws() {
    while (p_ < t_.size() &&
           (t_[p_] == ' ' || t_[p_] == '\t' || t_[p_] == '\n' || t_[p_] == '\r'))
      ++p_;
  }
  bool Fail(const char *w) {
    why_ = w;
    return false;
  }
  bool Lit(const char *s) {
    size_t n = std::strlen(s);
    if (t_.compare(p_, n, s) != 0) return false;
    p_ += n;
    return true;
  }
  static void Utf8(uint32_t cp, std::string *o) {
    if (cp < 0x80) {
      *o += static_cast<char>(cp);
    } else if (cp < 0x800) {
      *o += static_cast<char>(0xC0 | (cp >> 6));
      *o += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
      *o += static_cast<char>(0xE0 | (cp >> 12));
      *o += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      *o += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
      *o += static_cast<char>(0xF0 | (cp >> 18));
      *o += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
      *o += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      *o += static_cast<char>(0x80 | (cp & 0x3F));
    }
  }
  bool Hex4(uint32_t *v) {
    if (p_ + 4 > t_.size()) return false;
    *v = 0;
    for (int i = 0; i < 4; ++i) {
      char c = t_[p_++];
      *v <<= 4;
      if (c >= '0' && c <= '9') *v |= c - '0';
      else if (c >= 'a' && c <= 'f') *v |= c - 'a' + 10;
      else if (c >= 'A' && c <= 'F') *v |= c - 'A' + 10;
      else return false;
    }
    return true;
  }
  bool String(std::string *o) {
    ++p_;  // opening quote
    while (p_ < t_.size()) {
      char c = t_[p_++];
      if (c == '"') return true;
      if (c != '\\') {
        *o += c;
        continue;
      }
      if (p_ >= t_.size()) break;
      char e = t_[p_++];
      switch (e) {
        case '"': *o += '"'; break;
        case '\\': *o += '\\'; break;
        case '/': *o += '/'; break;
        case 'b': *o += '\b'; break;
        case 'f': *o += '\f'; break;
        case 'n': *o += '\n'; break;
        case 'r': *o += '\r'; break;
        case 't': *o += '\t'; break;
        case 'u': {
          uint32_t cp;
          if (!Hex4(&cp)) return Fail("bad \\u escape");
          if (cp >= 0xD800 && cp < 0xDC00 && t_.compare(p_, 2, "\\u") == 0) {
            p_ += 2;
            uint32_t lo;
            if (!Hex4(&lo)) return Fail("bad \\u escape");
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
          }
          Utf8(cp, o);
          break;
        }
        default: return Fail("bad escape");
      }
    }
    return Fail("unterminated string");
  }
  bool Value(Json *v, int depth) {
    if (depth > 64) return Fail("nesting too deep");
    Ws();
    if (p_ >= t_.size()) return Fail("unexpected end");
    char c = t_[p_];
    if (c == '{') {
      v->kind = Json::kObj;
      ++p_;
      Ws();
      if (p_ < t_.size() && t_[p_] == '}') { ++p_; return true; }
      for (;;) {
        Ws();
        if (p_ >= t_.size() || t_[p_] != '"') return Fail("expected key");
        std::string k;
        if (!String(&k)) return false;
        Ws();
        if (p_ >= t_.size() || t_[p_] != ':') return Fail("expected ':'");
        ++p_;
        Json child;
        if (!Value(&child, depth + 1)) return false;
        v->keys.push_back(std::move(k));
        v->arr.push_back(std::move(child));
        Ws();
        if (p_ < t_.size() && t_[p_] == ',') { ++p_; continue; }
        if (p_ < t_.size() && t_[p_] == '}') { ++p_; return true; }
        return Fail("expected ',' or '}'");
      }
    }
    if (c == '[') {
      v->kind = Json::kArr;
      ++p_;
      Ws();
      if (p_ < t_.size() && t_[p_] == ']') { ++p_; return true; }
      for (;;) {
        Json child;
        if (!Value(&child, depth + 1)) return false;
        v->arr.push_back(std::move(child));
        Ws();
        if (p_ < t_.size() && t_[p_] == ',') { ++p_; continue; }
        if (p_ < t_.size() && t_[p_] == ']') { ++p_; return true; }
        return Fail("expected ',' or ']'");
      }
    }
    if (c == '"') {
      v->kind = Json::kStr;
      return String(&v->str);
    }
    if (Lit("true")) { v->kind = Json::kBool; v->b = true; return true; }
    if (Lit("false")) { v->kind = Json::kBool; return true; }
    if (Lit("null")) { v->kind = Json::kNull; return true; }
    size_t b = p_;
    while (p_ < t_.size() && std::strchr("+-0123456789.eE", t_[p_])) ++p_;
    if (b == p_) return Fail("unexpected character");
    v->kind = Json::kNum;
    v->str = t_.substr(b, p_ - b);
    v->num = std::strtod(v->str.c_str(), nullptr);
    return true;
  }

  const std::string &t_;
  size_t p_ = 0;
  std::string why_;
};

}  // namespace dv_detail

inline bool ParseJson(const std::string &text, Json *out, std::string *err) {
  return dv_detail::JsonParser(text).Parse(out, err);
}

// --------------------------------------------------------------------------
// Dataset file list
// --------------------------------------------------------------------------

/** One file of a Dataverse dataset version. */
struct DataverseFile {
  int64_t id = -1;
  std::string path;  ///< "<directoryLabel>/<filename>" or "<filename>"
  int64_t size = -1;
  std::string content_type;
  bool restricted = false;
};

/** True if a relative path is safe to create under a staging directory. */
inline bool SafeRelPath(const std::string &p) {
  if (p.empty() || p[0] == '/' || p.find('\\') != std::string::npos ||
      p.find('\0') != std::string::npos)
    return false;
  size_t b = 0;
  while (b <= p.size()) {
    size_t e = p.find('/', b);
    if (e == std::string::npos) e = p.size();
    std::string part = p.substr(b, e - b);
    if (part.empty() || part == "." || part == "..") return false;
    b = e + 1;
  }
  return true;
}

/**
 * Files of the latest version, from the body of
 * GET /api/datasets/:persistentId/?persistentId=<pid>.
 */
inline bool ParseDataverseFiles(const std::string &body,
                                std::vector<DataverseFile> *files,
                                std::string *version, std::string *err) {
  Json root;
  if (!ParseJson(body, &root, err)) return false;
  if (root.Str("status") != "OK") {
    *err = "Dataverse API status '" + root.Str("status") + "': " +
           root.Str("message");
    return false;
  }
  const Json *data = root.Get("data");
  const Json *ver = data ? data->Get("latestVersion") : nullptr;
  const Json *list = ver ? ver->Get("files") : nullptr;
  if (!list || list->kind != Json::kArr) {
    *err = "Dataverse API reply has no data.latestVersion.files";
    return false;
  }
  if (version) {
    int64_t maj = ver->Int("versionNumber", -1);
    int64_t min = ver->Int("versionMinorNumber", -1);
    *version = ver->Str("versionState") +
               (maj >= 0 ? " v" + std::to_string(maj) + "." +
                               std::to_string(min < 0 ? 0 : min)
                         : "");
  }
  for (const Json &f : list->arr) {
    const Json *df = f.Get("dataFile");
    if (!df) continue;
    DataverseFile o;
    o.id = df->Int("id");
    std::string name = df->Str("filename", f.Str("label"));
    std::string dir = f.Str("directoryLabel");
    o.path = dir.empty() ? name : dir + "/" + name;
    o.size = df->Int("filesize");
    o.content_type = df->Str("contentType");
    const Json *r = f.Get("restricted");
    o.restricted = r && r->kind == Json::kBool && r->b;
    if (o.id < 0 || !SafeRelPath(o.path)) {
      *err = "Dataverse file entry with no id or an unsafe path: '" + o.path +
             "'";
      return false;
    }
    files->push_back(o);
  }
  return true;
}

// --------------------------------------------------------------------------
// Selection
// --------------------------------------------------------------------------

inline bool DvGlob(const std::string &pat, const std::string &s) {
  return ::fnmatch(pat.c_str(), s.c_str(), 0) == 0;  // '*' crosses '/'
}

/** OMNI include/exclude patterns, split into whole-file and member rules. */
class DataverseSelection {
 public:
  DataverseSelection(const std::vector<std::string> &include,
                     const std::vector<std::string> &exclude)
      : exclude_(exclude) {
    for (const auto &p : include) {
      size_t z = p.find(".zip/");
      if (z == std::string::npos) {
        whole_.push_back(p);
      } else {
        members_.emplace_back(p.substr(0, z + 4), p.substr(z + 5));
      }
    }
    all_ = include.empty();
  }

  /** Take this file of the dataset as it is. */
  bool WholeFile(const std::string &path) const {
    if (Excluded(path)) return false;
    if (all_) return true;
    for (const auto &p : whole_)
      if (DvGlob(p, path)) return true;
    return false;
  }

  /** Open this zip file and look at its members. */
  bool OpenZip(const std::string &zip_path) const {
    if (Excluded(zip_path)) return false;
    for (const auto &m : members_)
      if (DvGlob(m.first, zip_path)) return true;
    return false;
  }

  /** Take this member of a zip file. */
  bool Member(const std::string &zip_path, const std::string &member) const {
    if (Excluded(zip_path + "/" + member)) return false;
    for (const auto &m : members_)
      if (DvGlob(m.first, zip_path) && DvGlob(m.second, member)) return true;
    return false;
  }

 private:
  bool Excluded(const std::string &path) const {
    for (const auto &p : exclude_)
      if (DvGlob(p, path)) return true;
    return false;
  }

  bool all_ = false;
  std::vector<std::string> whole_;
  std::vector<std::pair<std::string, std::string>> members_;
  std::vector<std::string> exclude_;
};

// --------------------------------------------------------------------------
// ZIP / ZIP64 directory
// --------------------------------------------------------------------------

/** Where the central directory is, from the end-of-central-directory records. */
struct ZipDirLocation {
  uint64_t cd_offset = 0;
  uint64_t cd_size = 0;
  uint64_t entries = 0;
  bool zip64 = false;
  uint64_t zip64_eocd_offset = 0;  ///< set when the ZIP64 record must be read
  bool need_zip64_record = false;
};

/** One member, from the central directory. */
struct ZipEntry {
  std::string name;
  uint16_t flags = 0;
  uint16_t method = 0;  ///< 0 stored, 8 deflated
  uint32_t crc32 = 0;
  uint64_t csize = 0;
  uint64_t usize = 0;
  uint64_t local_header_offset = 0;
  bool IsDir() const { return !name.empty() && name.back() == '/'; }
};

namespace dv_detail {
inline uint16_t U16(const unsigned char *p) { return p[0] | (p[1] << 8); }
inline uint32_t U32(const unsigned char *p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) |
         (static_cast<uint32_t>(p[3]) << 24);
}
inline uint64_t U64(const unsigned char *p) {
  return static_cast<uint64_t>(U32(p)) |
         (static_cast<uint64_t>(U32(p + 4)) << 32);
}
}  // namespace dv_detail

/** Read the ZIP64 end-of-central-directory record (56+ bytes). */
inline bool ParseZip64Record(const std::string &rec, ZipDirLocation *loc,
                             std::string *err);

/** Bytes to read from the end of a zip to be sure of holding its EOCD. */
constexpr uint64_t kZipTailBytes = 22 + 65535 + 20;

/**
 * Find the EOCD (and the ZIP64 locator before it) in the last bytes of a zip.
 * @param tail   the last tail.size() bytes of the file
 * @param total  size of the whole file
 */
inline bool ParseZipTail(const std::string &tail, uint64_t total,
                         ZipDirLocation *loc, std::string *err) {
  using namespace dv_detail;
  const auto *b = reinterpret_cast<const unsigned char *>(tail.data());
  const uint64_t tail_off = total - tail.size();
  if (tail.size() < 22) {
    *err = "file too short to be a zip";
    return false;
  }
  long e = static_cast<long>(tail.size()) - 22;
  for (; e >= 0; --e) {
    if (U32(b + e) == 0x06054b50 &&
        static_cast<size_t>(e) + 22 + U16(b + e + 20) <= tail.size())
      break;
  }
  if (e < 0) {
    *err = "no end-of-central-directory record (not a zip?)";
    return false;
  }
  loc->entries = U16(b + e + 10);
  loc->cd_size = U32(b + e + 12);
  loc->cd_offset = U32(b + e + 16);
  if (loc->entries == 0xFFFF || loc->cd_size == 0xFFFFFFFFu ||
      loc->cd_offset == 0xFFFFFFFFu) {
    long l = e - 20;
    if (l < 0 || U32(b + l) != 0x07064b50) {
      *err = "ZIP64 sizes but no ZIP64 locator";
      return false;
    }
    loc->zip64 = true;
    loc->zip64_eocd_offset = U64(b + l + 8);
    if (loc->zip64_eocd_offset >= tail_off &&
        loc->zip64_eocd_offset + 56 <= total) {
      return ParseZip64Record(tail.substr(loc->zip64_eocd_offset - tail_off),
                              loc, err);
    }
    loc->need_zip64_record = true;
  }
  return true;
}

inline bool ParseZip64Record(const std::string &rec, ZipDirLocation *loc,
                             std::string *err) {
  using namespace dv_detail;
  const auto *b = reinterpret_cast<const unsigned char *>(rec.data());
  if (rec.size() < 56 || U32(b) != 0x06064b50) {
    *err = "bad ZIP64 end-of-central-directory record";
    return false;
  }
  loc->entries = U64(b + 32);
  loc->cd_size = U64(b + 40);
  loc->cd_offset = U64(b + 48);
  loc->need_zip64_record = false;
  return true;
}

/** Parse the central directory into its entries. */
inline bool ParseCentralDirectory(const std::string &cd,
                                  std::vector<ZipEntry> *out,
                                  std::string *err) {
  using namespace dv_detail;
  const auto *b = reinterpret_cast<const unsigned char *>(cd.data());
  size_t p = 0;
  while (p + 46 <= cd.size() && U32(b + p) == 0x02014b50) {
    ZipEntry z;
    z.flags = U16(b + p + 8);
    z.method = U16(b + p + 10);
    z.crc32 = U32(b + p + 16);
    z.csize = U32(b + p + 20);
    z.usize = U32(b + p + 24);
    const uint16_t nlen = U16(b + p + 28);
    const uint16_t xlen = U16(b + p + 30);
    const uint16_t clen = U16(b + p + 32);
    z.local_header_offset = U32(b + p + 42);
    if (p + 46 + nlen + xlen + clen > cd.size()) {
      *err = "central directory entry runs past its end";
      return false;
    }
    z.name.assign(cd, p + 46, nlen);
    // ZIP64 extra field: the 64-bit values of whichever fields are saturated.
    size_t x = p + 46 + nlen, xe = x + xlen;
    while (x + 4 <= xe) {
      uint16_t id = U16(b + x), sz = U16(b + x + 2);
      if (x + 4 + sz > xe) break;
      if (id == 0x0001) {
        size_t q = x + 4, qe = q + sz;
        if (z.usize == 0xFFFFFFFFu && q + 8 <= qe) { z.usize = U64(b + q); q += 8; }
        if (z.csize == 0xFFFFFFFFu && q + 8 <= qe) { z.csize = U64(b + q); q += 8; }
        if (z.local_header_offset == 0xFFFFFFFFu && q + 8 <= qe) {
          z.local_header_offset = U64(b + q);
        }
      }
      x += 4 + sz;
    }
    out->push_back(std::move(z));
    p += 46 + nlen + xlen + clen;
  }
  if (out->empty() && !cd.empty()) {
    *err = "no central directory entries where the EOCD points";
    return false;
  }
  return true;
}

/**
 * Offset of a member's data relative to its local header, from the first 30
 * bytes of that header; 0 if they are not a local header.
 */
inline uint64_t ZipLocalDataOffset(const std::string &hdr) {
  using namespace dv_detail;
  const auto *b = reinterpret_cast<const unsigned char *>(hdr.data());
  if (hdr.size() < 30 || U32(b) != 0x04034b50) return 0;
  return 30ull + U16(b + 26) + U16(b + 28);
}

// --------------------------------------------------------------------------
// Staging (network; dataverse_fetch.cc)
// --------------------------------------------------------------------------

/** One object staged to disk. */
struct DataverseObject {
  std::string path;      ///< relative path (see the header comment)
  std::string local;     ///< absolute path of the staged copy
  uint64_t size = 0;
  int64_t file_id = -1;  ///< Dataverse data file it came from
  std::string member;    ///< zip member name, or "" for a whole file
  bool skipped = false;  ///< already staged with these bytes
};

struct DataverseStageResult {
  bool ok = false;
  std::string error;
  std::string version;  ///< dataset version state and number
  size_t dataset_files = 0;
  std::vector<DataverseObject> objects;
  uint64_t bytes = 0;      ///< bytes of the staged objects
  uint64_t fetched = 0;    ///< bytes received over HTTP
  int requests = 0;        ///< HTTP GETs made
  double seconds = 0;
};

using DataverseLog = std::function<void(const std::string &)>;

/** Name of the manifest the stager writes in the staging directory. */
constexpr const char *kDataverseManifest = ".cae_dataverse_manifest.tsv";

/**
 * List the dataset, select objects with the patterns, and stage each one at
 * <stage_dir>/<path>. Credentials and headers for the Dataverse host come from
 * the http config file (http_auth.h), e.g. header.X-Dataverse-key for a draft
 * or restricted file; `token`, when set, is sent as that header. Objects
 * already staged with the right size are not fetched again unless
 * CAE_HTTP_OVERWRITE=1. CAE_DATAVERSE_JOBS (default 8) objects are fetched at
 * a time. `list_only` stops after selection.
 */
DataverseStageResult DataverseStage(const std::string &src,
                                    const std::vector<std::string> &include,
                                    const std::vector<std::string> &exclude,
                                    const std::string &stage_dir,
                                    const std::string &token, bool list_only,
                                    const DataverseLog &log);

/** Default staging directory for a source: $CAE_DATAVERSE_CACHE or
 *  ~/.clio/cae_dataverse, plus the persistent id with '/' and ':' made '_'. */
std::string DataverseCacheDir(const DataverseSource &s);

}  // namespace clio::cae::core

#endif  // CLIO_CAE_CORE_FACTORY_DATAVERSE_H_
