/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 */

// DataverseStage end to end, without the network: a fake Dataverse server on
// 127.0.0.1 inside this process (dataset API, access API redirects, ranged
// "storage" reads) and the real curl the http engine spawns.
//
// Covers dataverse_fetch.cc: the listing and its retries, redirect-once link
// resolution, ranged zip directory and member reads (stored, deflated, ZIP64,
// a directory outside the tail), whole files, size probes, the API key staying
// on the Dataverse host, throttling and expired-link retries, skip on rerun,
// the manifest, and the error paths. Needs curl on PATH.

#include <clio_cae/core/factory/dataverse.h>
#include <clio_cae/core/factory/http_auth.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <cstdlib>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "simple_test.h"
#include "zip_builder.h"

namespace cae = clio::cae::core;
using zip_builder::MakeZip;
using zip_builder::Member;

namespace {

/** Bytes no two nearby offsets share; `noisy` makes them incompressible. */
std::string Pattern(size_t n, unsigned seed, bool noisy = false) {
  std::string s(n, '\0');
  uint32_t x = seed * 2654435761u + 1;
  for (size_t i = 0; i < n; ++i) {
    if (noisy) {
      x ^= x << 13;
      x ^= x >> 17;
      x ^= x << 5;
      s[i] = static_cast<char>(x & 0xff);
    } else {
      s[i] = static_cast<char>((i * 31 + seed + (i >> 10)) % 97);
    }
  }
  return s;
}

struct DvFile {
  int64_t id;
  std::string dir, name;
  bool with_size;  ///< list "filesize" (else the stager probes for it)
};

std::string DatasetJson(const std::vector<DvFile> &fs,
                        const std::map<int64_t, std::string> &blobs) {
  std::ostringstream o;
  o << R"({"status":"OK","data":{"latestVersion":{"versionState":"RELEASED",)"
    << R"("versionNumber":2,"versionMinorNumber":1,"files":[)";
  for (size_t i = 0; i < fs.size(); ++i) {
    const auto &f = fs[i];
    o << (i ? "," : "") << R"({"label":")" << f.name << '"';
    if (!f.dir.empty()) o << R"(,"directoryLabel":")" << f.dir << '"';
    o << R"(,"dataFile":{"id":)" << f.id << R"(,"filename":")" << f.name << '"';
    if (f.with_size) o << R"(,"filesize":)" << blobs.at(f.id).size();
    o << "}}";
  }
  o << "]}}}";
  return o.str();
}

/**
 * A fake Dataverse, one thread per connection, Connection: close.
 *
 *   GET /api/datasets/:persistentId/?persistentId=<pid>   the dataset JSON
 *   GET /api/access/datafile/<id>   303 to http://localhost:<port>/s3/<id>,
 *                                   or the bytes themselves for `direct` ids
 *   GET /s3/<id>                    the bytes, honouring Range (except for
 *                                   `norange` ids)
 *   GET /web/<id>/<name>            the bytes, honouring Range: a plain web
 *                                   file (no Dataverse API)
 *
 * The storage redirect names "localhost" while the Dataverse base is
 * "127.0.0.1", so they are different hosts to the stager, as on S3.
 */
class FakeDataverse {
 public:
  FakeDataverse() {
    ::signal(SIGPIPE, SIG_IGN);
    lfd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    ::setsockopt(lfd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    ::bind(lfd_, reinterpret_cast<sockaddr *>(&a), sizeof(a));
    ::listen(lfd_, 64);
    socklen_t len = sizeof(a);
    ::getsockname(lfd_, reinterpret_cast<sockaddr *>(&a), &len);
    port_ = ntohs(a.sin_port);
    th_ = std::thread([this] { Loop(); });
  }

  ~FakeDataverse() {
    stop_ = true;
    th_.join();
    ::close(lfd_);
    std::lock_guard<std::mutex> g(mu_);
    for (auto &t : conns_) t.join();
  }

  int Port() const { return port_; }

  std::string Src(const std::string &pid) const {
    return "dataverse::http://127.0.0.1:" + std::to_string(port_) + "/" + pid;
  }

  void AddDataset(const std::string &pid, const std::vector<DvFile> &fs,
                  const std::map<int64_t, std::string> &blobs) {
    std::lock_guard<std::mutex> g(mu_);
    for (const auto &b : blobs) blobs_[b.first] = b.second;
    datasets_[cae::DvUrlEncode(pid)] = DatasetJson(fs, blobs_);
  }
  void AddRawDataset(const std::string &pid, const std::string &json) {
    std::lock_guard<std::mutex> g(mu_);
    datasets_[cae::DvUrlEncode(pid)] = json;
  }

  std::set<int64_t> direct, norange;
  std::atomic<int> fail_dataset{0};      ///< next N dataset GETs: 503
  std::atomic<int> throttle_resolve{0};  ///< next N access API GETs: 429
  std::atomic<int> expire_s3{0};         ///< next N storage requests: 403
  std::atomic<int> expire_data{0};       ///< next N storage data GETs: 403
  std::atomic<int> resolves{0};          ///< access API requests
  std::atomic<int> api_without_key{0};   ///< API requests lacking the key
  std::atomic<int> storage_with_key{0};  ///< storage requests carrying it
  std::string want_key;
  std::map<int64_t, int> resolves_by_id;

  void Reset() {
    std::lock_guard<std::mutex> g(mu_);
    resolves = 0;
    api_without_key = 0;
    storage_with_key = 0;
    resolves_by_id.clear();
  }
  int ResolvesOf(int64_t id) {
    std::lock_guard<std::mutex> g(mu_);
    return resolves_by_id.count(id) ? resolves_by_id[id] : 0;
  }

 private:
  void Loop() {
    while (!stop_) {
      pollfd p{lfd_, POLLIN, 0};
      if (::poll(&p, 1, 50) <= 0) continue;
      int c = ::accept(lfd_, nullptr, nullptr);
      if (c < 0) continue;
      std::lock_guard<std::mutex> g(mu_);
      conns_.emplace_back([this, c] { Serve(c); });
    }
  }

  static void SendAll(int fd, const char *p, size_t n) {
    while (n > 0) {
      ssize_t k = ::send(fd, p, n, 0);
      if (k <= 0) return;
      p += k;
      n -= static_cast<size_t>(k);
    }
  }

  static void Reply(int fd, const std::string &status, const std::string &hdrs,
                    const std::string &body, bool head = false) {
    std::string h = "HTTP/1.1 " + status + "\r\n" + hdrs +
                    "Content-Length: " + std::to_string(body.size()) +
                    "\r\nConnection: close\r\n\r\n";
    SendAll(fd, h.data(), h.size());
    if (!head) SendAll(fd, body.data(), body.size());
  }

  static std::string Header(const std::string &req, const std::string &name) {
    std::istringstream in(req);
    std::string line;
    while (std::getline(in, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      size_t c = line.find(':');
      if (c == std::string::npos) continue;
      if (cae::http_detail::Lower(line.substr(0, c)) == name) {
        size_t v = line.find_first_not_of(' ', c + 1);
        return v == std::string::npos ? "" : line.substr(v);
      }
    }
    return "";
  }

  /** The bytes of `body`, honouring a Range header when `ranges`. */
  static void ServeBytes(int fd, const std::string &body,
                         const std::string &range, bool ranges, bool head) {
    const size_t total = body.size();
    if (ranges && range.compare(0, 6, "bytes=") == 0) {
      size_t dash = range.find('-');
      size_t a = std::strtoull(range.c_str() + 6, nullptr, 10);
      size_t b = dash + 1 < range.size()
                     ? std::strtoull(range.c_str() + dash + 1, nullptr, 10)
                     : total - 1;
      if (a >= total) {
        Reply(fd, "416 Range Not Satisfiable",
              "Content-Range: bytes */" + std::to_string(total) + "\r\n", "");
        return;
      }
      b = std::min(b, total - 1);
      Reply(fd, "206 Partial Content",
            "Content-Type: application/octet-stream\r\nContent-Range: bytes " +
                std::to_string(a) + "-" + std::to_string(b) + "/" +
                std::to_string(total) + "\r\n",
            body.substr(a, b - a + 1), head);
      return;
    }
    Reply(fd, "200 OK", "Content-Type: application/octet-stream\r\n", body, head);
  }

  void Serve(int fd) {
    std::string req;
    char buf[4096];
    while (req.find("\r\n\r\n") == std::string::npos && req.size() < 65536) {
      ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
      if (n <= 0) break;
      req.append(buf, static_cast<size_t>(n));
    }
    std::istringstream sl(req);
    std::string method, target;
    sl >> method >> target;
    const size_t q = target.find('?');
    const std::string path = target.substr(0, q);
    const std::string query = q == std::string::npos ? "" : target.substr(q + 1);
    const std::string range = Header(req, "range");
    const std::string key = Header(req, "x-dataverse-key");
    const bool head = method == "HEAD";
    const bool api = path.compare(0, 5, "/api/") == 0;
    if (api && !want_key.empty() && key != want_key) ++api_without_key;
    if (!api && !key.empty()) ++storage_with_key;

    const std::string kDs = "/api/datasets/:persistentId/";
    const std::string kAccess = "/api/access/datafile/";
    if (path == kDs) {
      if (fail_dataset.fetch_sub(1) > 0) {
        Reply(fd, "503 Service Unavailable", "", "");
      } else {
        const std::string pid = query.substr(query.find('=') + 1);
        std::string json;
        {
          std::lock_guard<std::mutex> g(mu_);
          auto it = datasets_.find(pid);
          if (it != datasets_.end()) json = it->second;
        }
        if (json.empty()) {
          Reply(fd, "404 Not Found", "Content-Type: application/json\r\n",
                R"({"status":"ERROR","message":"Dataset not found"})", head);
        } else {
          Reply(fd, "200 OK", "Content-Type: application/json\r\n", json, head);
        }
      }
    } else if (path.compare(0, kAccess.size(), kAccess) == 0) {
      const int64_t id = std::atoll(path.c_str() + kAccess.size());
      ++resolves;
      std::string body;
      bool found;
      {
        std::lock_guard<std::mutex> g(mu_);
        ++resolves_by_id[id];
        found = blobs_.count(id) > 0;
        if (found) body = blobs_[id];
      }
      if (throttle_resolve.fetch_sub(1) > 0) {
        Reply(fd, "429 Too Many Requests", "", "");
      } else if (!found) {
        Reply(fd, "404 Not Found", "", "");
      } else if (direct.count(id)) {
        ServeBytes(fd, body, range, true, head);
      } else {
        Reply(fd, "303 See Other",
              "Location: http://localhost:" + std::to_string(port_) + "/s3/" +
                  std::to_string(id) + "?X-Amz-Signature=sig\r\n",
              "");
      }
    } else if (path.compare(0, 4, "/s3/") == 0) {
      const int64_t id = std::atoll(path.c_str() + 4);
      std::string body;
      bool found;
      {
        std::lock_guard<std::mutex> g(mu_);
        found = blobs_.count(id) > 0;
        if (found) body = blobs_[id];
      }
      if (!found) {
        Reply(fd, "404 Not Found", "", "");
      } else if (expire_s3.fetch_sub(1) > 0 ||
                 (range != "bytes=0-0" && expire_data.fetch_sub(1) > 0)) {
        Reply(fd, "403 Forbidden", "Content-Type: application/xml\r\n",
              "<Error><Code>AccessDenied</Code></Error>");
      } else {
        ServeBytes(fd, body, range, norange.count(id) == 0, head);
      }
    } else if (path.compare(0, 5, "/web/") == 0) {
      const int64_t id = std::atoll(path.c_str() + 5);
      std::string body;
      bool found;
      {
        std::lock_guard<std::mutex> g(mu_);
        found = blobs_.count(id) > 0;
        if (found) body = blobs_[id];
      }
      if (found) {
        ServeBytes(fd, body, range, true, head);
      } else {
        Reply(fd, "404 Not Found", "", "");
      }
    } else {
      Reply(fd, "404 Not Found", "", "");
    }
    ::close(fd);
  }

  int lfd_ = -1;
  int port_ = 0;
  std::atomic<bool> stop_{false};
  std::thread th_;
  std::mutex mu_;
  std::vector<std::thread> conns_;
  std::map<std::string, std::string> datasets_;
  std::map<int64_t, std::string> blobs_;
};

std::string TmpDir() {
  static std::string d = [] {
    const char *t = std::getenv("TMPDIR");
    std::string tmpl = std::string(t && *t ? t : "/tmp") + "/cae_dv_XXXXXX";
    std::vector<char> b(tmpl.begin(), tmpl.end());
    b.push_back('\0');
    return std::string(::mkdtemp(b.data()));
  }();
  return d;
}

std::string Slurp(const std::string &p) {
  std::ifstream f(p, std::ios::binary);
  std::ostringstream s;
  s << f.rdbuf();
  return s.str();
}

bool Exists(const std::string &p) {
  struct stat st;
  return ::stat(p.c_str(), &st) == 0;
}

// The golf-like dataset used by most cases.
const std::string kGood = "doi:10.5072/FK2/GOOD";
const std::string kHdf1 = Pattern(200000, 1);          // compressible
const std::string kVideo = Pattern(150000, 2, true);   // incompressible
const std::string kHdf2 = Pattern(40000, 3);
const std::string kHdf3 = Pattern(90000, 4);
const std::string kCsv = "Participant Number,Height (cm)\nP01,155\nP02,169\n";
const std::string kReadme = "MultiSenseGolf-like test data\n";

FakeDataverse &Server() {
  static FakeDataverse *s = [] {
    for (const char *v : {"http_proxy", "https_proxy", "HTTP_PROXY",
                          "HTTPS_PROXY", "ALL_PROXY", "all_proxy"}) {
      ::unsetenv(v);
    }
    ::setenv("no_proxy", "*", 1);
    ::setenv("NO_PROXY", "*", 1);
    ::setenv("CAE_HTTP_AUTH_FILE", (TmpDir() + "/no_such_auth.ini").c_str(), 1);
    ::setenv("CAE_HTTP_BACKOFF_MS", "1", 1);
    ::setenv("CAE_HTTP_PART_RETRIES", "3", 1);
    ::setenv("CAE_HTTP_PROGRESS_S", "0", 1);
    ::setenv("CAE_DATAVERSE_JOBS", "3", 1);
    auto *srv = new FakeDataverse();

    std::map<int64_t, std::string> b;
    b[101] = MakeZip({{"Swing01/", "", false},
                      {"Swing01/P01_Swing01_stream_data.hdf5", kHdf1, true},
                      {"Swing01/FPV_RGB.mp4", kVideo, false},
                      {"Swing02/P01_Swing02_stream_data.hdf5", kHdf2, false}},
                     false);
    // ZIP64, no "filesize" in the listing, and a comment long enough to put
    // the central directory before the tail the stager reads first.
    b[102] = MakeZip({{"Swing01/P02_Swing01_stream_data.hdf5", kHdf3, true}},
                     true, std::string(65000, 'c'));
    b[103] = MakeZip({{"Participant Metadata.csv", kCsv, false},
                      {"Annotation Data.xlsx", "xlsx", false}},
                     false);
    b[104] = kReadme;
    b[105] = MakeZip({{"big.mp4", kVideo, false}}, false);
    srv->AddDataset(kGood,
                    {{101, "", "P01.zip", true},
                     {102, "", "P02.zip", false},
                     {103, "", "Documentation.zip", true},
                     {104, "docs", "README.txt", true},
                     {105, "", "Video.zip", true}},
                    b);
    srv->direct.insert(103);  // served by the API itself, no redirect
    return srv;
  }();
  return *s;
}

const std::vector<std::string> kInclude = {
    "P*.zip/*_stream_data.hdf5", "Documentation.zip/*.csv", "docs/*",
    "Video.zip/*"};
const std::vector<std::string> kExclude = {"*Swing02*", "Video.zip"};

struct Logged {
  std::vector<std::string> lines;
  cae::DataverseLog fn() {
    return [this](const std::string &m) { lines.push_back(m); };
  }
  bool Has(const std::string &s) const {
    for (const auto &l : lines)
      if (l.find(s) != std::string::npos) return true;
    return false;
  }
};

cae::DataverseStageResult Stage(const std::string &pid, const std::string &dir,
                                bool list_only = false,
                                const std::string &token = "",
                                Logged *log = nullptr) {
  return cae::DataverseStage(Server().Src(pid), kInclude, kExclude, dir, token,
                             list_only, log ? log->fn() : cae::DataverseLog());
}

/** A dataset with a single zip holding `ms`, staged with "*.zip/*". */
cae::DataverseStageResult StageOneZip(const std::string &pid,
                                      const std::vector<Member> &ms,
                                      int64_t id) {
  std::map<int64_t, std::string> b;
  b[id] = MakeZip(ms, false);
  Server().AddDataset(pid, {{id, "", "Z.zip", true}}, b);
  return cae::DataverseStage(Server().Src(pid), {"*.zip/*"}, {},
                             TmpDir() + "/" + std::to_string(id), "", false,
                             nullptr);
}

}  // namespace

TEST_CASE("dataverse_stage_list_only", "[dataverse][stage]") {
  const std::string dir = TmpDir() + "/list";
  Logged log;
  auto r = Stage(kGood, dir, true, "", &log);
  INFO(r.error);
  REQUIRE(r.ok);
  REQUIRE(r.version == "RELEASED v2.1");
  REQUIRE(r.dataset_files == 5);
  REQUIRE(r.objects.size() == 4);
  REQUIRE(r.objects[0].path == "Documentation.zip/Participant Metadata.csv");
  REQUIRE(r.objects[1].path == "P01.zip/Swing01/P01_Swing01_stream_data.hdf5");
  REQUIRE(r.objects[1].size == kHdf1.size());
  REQUIRE(r.objects[1].member == "Swing01/P01_Swing01_stream_data.hdf5");
  REQUIRE(r.objects[2].path == "P02.zip/Swing01/P02_Swing01_stream_data.hdf5");
  REQUIRE(r.objects[3].path == "docs/README.txt");
  REQUIRE(r.objects[3].member.empty());
  REQUIRE(!Exists(dir + "/docs/README.txt"));
  REQUIRE(!Exists(dir + "/" + cae::kDataverseManifest));
  REQUIRE(log.Has("ZIP64"));
  REQUIRE(log.Has("4 objects selected"));
}

TEST_CASE("dataverse_stage_fetch_and_rerun", "[dataverse][stage]") {
  auto &s = Server();
  const std::string dir = TmpDir() + "/stage/";  // a trailing '/' is fine
  s.Reset();
  s.want_key = "secret-key";
  auto r = Stage(kGood, dir, false, "secret-key");
  s.want_key.clear();
  INFO(r.error);
  REQUIRE(r.ok);
  REQUIRE(r.objects.size() == 4);
  REQUIRE(Slurp(dir + "P01.zip/Swing01/P01_Swing01_stream_data.hdf5") == kHdf1);
  REQUIRE(Slurp(dir + "P02.zip/Swing01/P02_Swing01_stream_data.hdf5") == kHdf3);
  REQUIRE(Slurp(dir + "Documentation.zip/Participant Metadata.csv") == kCsv);
  REQUIRE(Slurp(dir + "docs/README.txt") == kReadme);
  REQUIRE(!Exists(dir + "P01.zip/Swing01/FPV_RGB.mp4"));
  REQUIRE(!Exists(dir + "P01.zip/Swing02"));
  REQUIRE(!Exists(dir + ".cae-tmp"));
  REQUIRE(r.bytes == kHdf1.size() + kHdf3.size() + kCsv.size() + kReadme.size());
  REQUIRE(r.fetched > 0);
  REQUIRE(r.fetched < kVideo.size());  // the video was never read
  REQUIRE(r.requests > 0);
  // One resolve per data file, the key on every API request and never on a
  // storage request, and the excluded zip never touched.
  REQUIRE(s.ResolvesOf(101) == 1);
  REQUIRE(s.ResolvesOf(102) == 1);
  REQUIRE(s.ResolvesOf(104) == 1);
  REQUIRE(s.ResolvesOf(105) == 0);
  REQUIRE(s.api_without_key == 0);
  REQUIRE(s.storage_with_key == 0);
  const std::string m = Slurp(dir + cae::kDataverseManifest);
  REQUIRE(m.find("# path\tsize\tfile_id\tmember\n") == 0);
  REQUIRE(m.find("P01.zip/Swing01/P01_Swing01_stream_data.hdf5\t200000\t101\t"
                 "Swing01/P01_Swing01_stream_data.hdf5\n") != std::string::npos);
  REQUIRE(m.find("docs/README.txt\t" + std::to_string(kReadme.size()) +
                 "\t104\t\n") != std::string::npos);

  // A rerun finds every object staged: it reads the zip directories again
  // but no member.
  s.Reset();
  auto again = Stage(kGood, dir);
  INFO(again.error);
  REQUIRE(again.ok);
  for (const auto &o : again.objects) REQUIRE(o.skipped);
  REQUIRE(again.fetched < r.fetched);
  REQUIRE(again.fetched < kHdf1.size());
  REQUIRE(s.ResolvesOf(101) == 1);

  // CAE_HTTP_OVERWRITE=1 fetches them again.
  ::setenv("CAE_HTTP_OVERWRITE", "1", 1);
  auto over = Stage(kGood, dir);
  ::unsetenv("CAE_HTTP_OVERWRITE");
  INFO(over.error);
  REQUIRE(over.ok);
  for (const auto &o : over.objects) REQUIRE(!o.skipped);
  REQUIRE(Slurp(dir + "P01.zip/Swing01/P01_Swing01_stream_data.hdf5") == kHdf1);
}

TEST_CASE("dataverse_stage_retries", "[dataverse][stage]") {
  auto &s = Server();
  const std::string dir = TmpDir() + "/retry";
  s.Reset();
  s.fail_dataset = 1;      // the listing: 503 once
  s.throttle_resolve = 2;  // the access API: 429 twice
  s.expire_s3 = 1;         // storage: 403 (an expired link) on a probe
  Logged log;
  auto r = Stage(kGood, dir, false, "", &log);
  s.fail_dataset = s.throttle_resolve = s.expire_s3 = 0;
  INFO(r.error);
  REQUIRE(r.ok);
  REQUIRE(Slurp(dir + "/P01.zip/Swing01/P01_Swing01_stream_data.hdf5") == kHdf1);
  REQUIRE(Slurp(dir + "/P02.zip/Swing01/P02_Swing01_stream_data.hdf5") == kHdf3);
  REQUIRE(Slurp(dir + "/docs/README.txt") == kReadme);
  REQUIRE(log.Has("retrying"));
  REQUIRE(log.Has("HTTP 429"));
  REQUIRE(log.Has("HTTP 403"));
  // Expired links were resolved again.
  REQUIRE(s.resolves > 4 + 2);

  // Data GETs refused more often than the engine retries a part: the stager
  // sees curl's exit 22, resolves the link again and retries the read.
  const std::string dir2 = TmpDir() + "/retry2";
  s.Reset();
  ::setenv("CAE_DATAVERSE_JOBS", "1", 1);
  s.expire_data = 4;  // CAE_HTTP_PART_RETRIES=3: the first read's 4 tries
  Logged log2;
  auto r2 = Stage(kGood, dir2, false, "", &log2);
  s.expire_data = 0;
  ::setenv("CAE_DATAVERSE_JOBS", "3", 1);
  INFO(r2.error);
  REQUIRE(r2.ok);
  REQUIRE(Slurp(dir2 + "/P01.zip/Swing01/P01_Swing01_stream_data.hdf5") == kHdf1);
  REQUIRE(Slurp(dir2 + "/docs/README.txt") == kReadme);
  REQUIRE(log2.Has("curl 22"));
  REQUIRE(s.resolves > 4);
}

TEST_CASE("dataverse_stage_errors", "[dataverse][stage]") {
  auto &s = Server();
  std::string err;

  auto bad_url = cae::DataverseStage("dataverse://", {}, {}, TmpDir() + "/e0",
                                     "", true, nullptr);
  REQUIRE(!bad_url.ok);
  REQUIRE(!bad_url.error.empty());

  auto no_dir = cae::DataverseStage(s.Src(kGood), {}, {}, "/proc/cae_dv_no/x",
                                    "", true, nullptr);
  REQUIRE(!no_dir.ok);
  REQUIRE(no_dir.error.find("cannot create") != std::string::npos);

  auto missing = Stage("doi:10.5072/FK2/NONE", TmpDir() + "/e1", true);
  REQUIRE(!missing.ok);
  REQUIRE(missing.error.find("404") != std::string::npos);

  s.AddRawDataset("doi:10.5072/FK2/ERR", R"({"status":"ERROR","message":"nope"})");
  auto api_err = Stage("doi:10.5072/FK2/ERR", TmpDir() + "/e2", true);
  REQUIRE(!api_err.ok);
  REQUIRE(api_err.error.find("nope") != std::string::npos);

  {
    std::map<int64_t, std::string> b;
    b[301] = "this is not a zip file at all";
    s.AddDataset("doi:10.5072/FK2/NOTZIP", {{301, "", "P09.zip", true}}, b);
    auto r = Stage("doi:10.5072/FK2/NOTZIP", TmpDir() + "/e3", true);
    REQUIRE(!r.ok);
    REQUIRE(r.error.find("P09.zip") != std::string::npos);
  }
  {
    // The listing names a data file the access API does not know.
    s.AddRawDataset(
        "doi:10.5072/FK2/GONE",
        R"({"status":"OK","data":{"latestVersion":{"files":[{"dataFile":)"
        R"({"id":999,"filename":"P10.zip","filesize":100}}]}}})");
    auto r = Stage("doi:10.5072/FK2/GONE", TmpDir() + "/e4", true);
    REQUIRE(!r.ok);
    REQUIRE(r.error.find("HTTP 404") != std::string::npos);
  }
  {
    // No "filesize" and storage that ignores Range: the size probe refuses.
    std::map<int64_t, std::string> b;
    b[302] = MakeZip({{"a_stream_data.hdf5", "x", false}}, false);
    s.norange.insert(302);
    s.AddDataset("doi:10.5072/FK2/NORANGE", {{302, "", "P11.zip", false}}, b);
    auto r = Stage("doi:10.5072/FK2/NORANGE", TmpDir() + "/e5", true);
    REQUIRE(!r.ok);
    REQUIRE(r.error.find("ranged reads") != std::string::npos);
  }

  auto enc = StageOneZip("doi:10.5072/FK2/ENC", {{"a.bin", "abc", false, 1}}, 310);
  REQUIRE(!enc.ok);
  REQUIRE(enc.error.find("encrypted") != std::string::npos);

  auto lzma = StageOneZip("doi:10.5072/FK2/LZMA",
                          {{"a.bin", "abc", false, 0, 14}}, 311);
  REQUIRE(!lzma.ok);
  REQUIRE(lzma.error.find("compression method 14") != std::string::npos);

  auto crc = StageOneZip("doi:10.5072/FK2/CRC",
                         {{"a.bin", "abc", false, 0, -1, true}}, 312);
  REQUIRE(!crc.ok);
  REQUIRE(crc.error.find("CRC-32") != std::string::npos);

  // Stored bytes recorded as deflate: inflation fails.
  auto infl = StageOneZip("doi:10.5072/FK2/INFL",
                          {{"a.bin", Pattern(5000, 9, true), false, 0, 8}}, 313);
  REQUIRE(!infl.ok);
  REQUIRE(infl.error.find("inflate failed") != std::string::npos);

  auto unsafe = StageOneZip("doi:10.5072/FK2/UNSAFE",
                            {{"../evil.bin", "abc", false}}, 314);
  REQUIRE(!unsafe.ok);
  REQUIRE(unsafe.error.find("unsafe member") != std::string::npos);

  {
    // A zip member and a file of the dataset that land on the same path.
    std::map<int64_t, std::string> b;
    b[320] = MakeZip({{"m.txt", "zip member", false}}, false);
    b[321] = "plain file";
    s.AddDataset("doi:10.5072/FK2/DUP",
                 {{320, "", "Z.zip", true}, {321, "Z.zip", "m.txt", true}}, b);
    auto r = cae::DataverseStage(s.Src("doi:10.5072/FK2/DUP"),
                                 {"*.txt", "Z.zip/*"}, {}, TmpDir() + "/e6",
                                 "", true, nullptr);
    REQUIRE(!r.ok);
    REQUIRE(r.error.find("share the path") != std::string::npos);
  }
}

TEST_CASE("dataverse_stage_web_zip", "[dataverse][stage]") {
  // A plain http(s) URL is a dataset of one file: no Dataverse API request,
  // a zip's members selected and range-read as for a Dataverse zip, and no
  // Dataverse key sent to the web host.
  auto &s = Server();
  std::map<int64_t, std::string> b;
  b[701] = MakeZip({{"posterior.h5", kHdf1, true},
                    {"example.py", "print('hi')\n", false},
                    {"__MACOSX/._posterior.h5", "junk", false},
                    {"big.mp4", kVideo, false}},
                   false);
  b[702] = kReadme;
  s.AddDataset("doi:10.5072/FK2/WEB", {}, b);
  s.Reset();
  const std::string base = "http://127.0.0.1:" + std::to_string(s.Port());
  const std::string dir = TmpDir() + "/web";
  auto r = cae::DataverseStage(base + "/web/701/release.zip?dl=1",
                               {"*.zip/posterior.h5", "*.zip/*.py"}, {}, dir,
                               "secret-key", false, nullptr);
  INFO(r.error);
  REQUIRE(r.ok);
  REQUIRE(r.version == "web");
  REQUIRE(r.dataset_files == 1);
  REQUIRE(r.objects.size() == 2);
  REQUIRE(r.objects[0].path == "release.zip/example.py");
  REQUIRE(r.objects[1].path == "release.zip/posterior.h5");
  REQUIRE(r.objects[1].member == "posterior.h5");
  REQUIRE(Slurp(dir + "/release.zip/posterior.h5") == kHdf1);
  REQUIRE(Slurp(dir + "/release.zip/example.py") == "print('hi')\n");
  REQUIRE(!Exists(dir + "/release.zip/__MACOSX"));
  REQUIRE(r.fetched < kVideo.size());
  REQUIRE(s.resolves == 0);
  REQUIRE(s.storage_with_key == 0);

  // A web file that is not a zip, selected whole.
  auto w = cae::DataverseStage(base + "/web/702/README.txt", {"README.txt"},
                               {}, dir, "", false, nullptr);
  INFO(w.error);
  REQUIRE(w.ok);
  REQUIRE(w.objects.size() == 1);
  REQUIRE(Slurp(dir + "/README.txt") == kReadme);

  // A missing web file.
  auto m = cae::DataverseStage(base + "/web/799/gone.zip", {"*.zip/*"}, {},
                               dir, "", false, nullptr);
  REQUIRE(!m.ok);
}

TEST_CASE("dataverse_parse_web_url", "[dataverse]") {
  cae::DataverseSource s;
  std::string err;
  REQUIRE(cae::ParseDataverseUrl("https://h.org/a/b/rel.zip?x=1#f", &s, &err));
  REQUIRE(s.base == "https://h.org");
  REQUIRE(s.pid == "web:h.org/a/b/rel.zip");
  REQUIRE(s.url == "https://h.org/a/b/rel.zip?x=1#f");
  REQUIRE(cae::DataverseFileUrl(s, 0) == s.url);
  REQUIRE(cae::WebFileName(s.url) == "rel.zip");
  REQUIRE(cae::WebFileName("http://h/d/") == "d");
  ::setenv("CAE_DATAVERSE_CACHE", "/c", 1);
  REQUIRE(cae::DataverseCacheDir(s) == "/c/web_h.org_a_b_rel.zip");
  ::unsetenv("CAE_DATAVERSE_CACHE");
  REQUIRE(cae::ParseDataverseUrl("http://h.org/f.zip", &s, &err));
  REQUIRE(s.base == "http://h.org");
  REQUIRE(!cae::ParseDataverseUrl("https://h.org", &s, &err));
  REQUIRE(err.find("http(s)://<host>/<path") != std::string::npos);
  REQUIRE(!cae::ParseDataverseUrl("https:///f.zip", &s, &err));
  // A Dataverse source after a web one carries no web URL.
  cae::DataverseSource d;
  REQUIRE(cae::ParseDataverseUrl("dataverse://h.org/doi:10.1/X", &d, &err));
  REQUIRE(d.url.empty());
}

TEST_CASE("dataverse_cache_dir", "[dataverse]") {
  cae::DataverseSource s;
  s.pid = "doi:10.7910/DVN/LCCLLW";
  ::setenv("CAE_DATAVERSE_CACHE", "/data/cache", 1);
  REQUIRE(cae::DataverseCacheDir(s) == "/data/cache/doi_10.7910_DVN_LCCLLW");
  ::unsetenv("CAE_DATAVERSE_CACHE");
  const char *home = std::getenv("HOME");
  std::string saved = home ? home : "";
  ::setenv("HOME", "/home/u", 1);
  REQUIRE(cae::DataverseCacheDir(s) ==
          "/home/u/.clio/cae_dataverse/doi_10.7910_DVN_LCCLLW");
  ::unsetenv("HOME");
  REQUIRE(cae::DataverseCacheDir(s) ==
          "/tmp/.clio/cae_dataverse/doi_10.7910_DVN_LCCLLW");
  if (!saved.empty()) ::setenv("HOME", saved.c_str(), 1);
}

SIMPLE_TEST_MAIN()
