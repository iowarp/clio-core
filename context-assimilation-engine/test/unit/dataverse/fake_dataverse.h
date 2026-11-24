/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 */

// A fake Dataverse (and plain web file) server on 127.0.0.1 for the
// dataverse:// and http(s) zip tests. Header-only, test use only.

#ifndef CAE_TEST_UNIT_DATAVERSE_FAKE_DATAVERSE_H_
#define CAE_TEST_UNIT_DATAVERSE_FAKE_DATAVERSE_H_

#include <clio_cae/core/factory/dataverse.h>
#include <clio_cae/core/factory/http_auth.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fake_dataverse {

namespace cae = clio::cae::core;

struct DvFile {
  int64_t id;
  std::string dir, name;
  bool with_size;  ///< list "filesize" (else the stager probes for it)
};

inline std::string DatasetJson(const std::vector<DvFile> &fs,
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

}  // namespace fake_dataverse

#endif  // CAE_TEST_UNIT_DATAVERSE_FAKE_DATAVERSE_H_
