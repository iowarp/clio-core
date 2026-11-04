/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 */

// The http(s) source end to end, without the network: an HTTP/1.1 server on
// 127.0.0.1 inside this process, and the real curl the engine spawns.
//
// Covers the download engine (http_fetch.cc: probe, parallel ranged GETs,
// whole-body GETs, retries that resume, skip/overwrite, range_off/range_size,
// error paths), HttpFileAssimilator::Run (destination check, the config file,
// error codes) and the factory's http(s) dispatch. Needs curl on PATH.

#include <clio_cae/core/factory/assimilator_factory.h>
#include <clio_cae/core/factory/http_auth.h>
#include <clio_cae/core/factory/http_fetch.h>
#include <clio_cae/core/factory/http_file_assimilator.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "simple_test.h"

namespace cae = clio::cae::core;

namespace {

/** Bytes the server serves: a pattern no two nearby offsets share. */
std::string Pattern(size_t n) {
  std::string s(n, '\0');
  for (size_t i = 0; i < n; ++i) {
    s[i] = static_cast<char>((i * 131 + (i >> 9) * 7 + 3) & 0xff);
  }
  return s;
}

/**
 * A small HTTP/1.1 server, one thread per connection, Connection: close.
 *
 *   /data.bin      honours Range (206), 200 without one
 *   /norange.bin   ignores Range (200 + Content-Length)
 *   /nolen.bin     ignores Range, no Content-Length (body ends at close)
 *   /empty.bin     an empty resource (416 bytes * /0 for a Range)
 *   /redirect.bin  302 to /data.bin
 *   /needq.bin     400 text/html unless the query has key=val
 *   /page.html     /data.bin as text/html
 *   /flaky.bin     /data.bin, but the first `flaky` data GETs stop half way
 *   /liar.bin      206 to the probe, then the whole body for every range
 *   /bearer.bin    401 unless "Authorization: Bearer tok"
 *   /basic.bin     401 unless basic auth u:p
 *   /hdr.bin       401 unless "X-Key: k1"
 *   /slow.bin      /data.bin, each data GET waits 1.2 s first
 */
class TestServer {
 public:
  TestServer() : data_(Pattern((3u << 20) + 12345)) {
    ::signal(SIGPIPE, SIG_IGN);
    lfd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    ::setsockopt(lfd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    a.sin_port = 0;
    ::bind(lfd_, reinterpret_cast<sockaddr*>(&a), sizeof(a));
    ::listen(lfd_, 64);
    socklen_t len = sizeof(a);
    ::getsockname(lfd_, reinterpret_cast<sockaddr*>(&a), &len);
    port_ = ntohs(a.sin_port);
    th_ = std::thread([this] { Loop(); });
  }

  ~TestServer() {
    stop_ = true;
    th_.join();
    ::close(lfd_);
    std::lock_guard<std::mutex> g(mu_);
    for (auto& t : conns_) t.join();
  }

  int port() const { return port_; }
  const std::string& data() const { return data_; }
  std::string Url(const std::string& path) const {
    return "http://127.0.0.1:" + std::to_string(port_) + path;
  }
  std::atomic<int> flaky{0};

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

  static void SendAll(int fd, const char* p, size_t n) {
    while (n > 0) {
      ssize_t k = ::send(fd, p, n, 0);
      if (k <= 0) return;
      p += k;
      n -= static_cast<size_t>(k);
    }
  }

  static void Reply(int fd, const std::string& status, const std::string& hdrs,
                    const std::string& body, size_t send_bytes) {
    std::string h = "HTTP/1.1 " + status + "\r\n" + hdrs +
                    "Connection: close\r\n\r\n";
    SendAll(fd, h.data(), h.size());
    SendAll(fd, body.data(), std::min(send_bytes, body.size()));
  }

  static std::string Header(const std::string& req, const std::string& name) {
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
    const bool probe = range == "bytes=0-0";
    const std::string auth = Header(req, "authorization");

    std::string ctype = "application/octet-stream";
    const std::string* body = &data_;
    static const std::string kEmpty;
    bool ranges = true, length = true, liar = false, half = false;

    if (path == "/norange.bin") {
      ranges = false;
    } else if (path == "/nolen.bin") {
      ranges = false;
      length = false;
    } else if (path == "/empty.bin") {
      body = &kEmpty;
    } else if (path == "/redirect.bin") {
      Reply(fd, "302 Found", "Location: /data.bin\r\nContent-Length: 0\r\n", "", 0);
      ::close(fd);
      return;
    } else if (path == "/needq.bin") {
      if (query.find("key=val") == std::string::npos) {
        std::string page = "<html>fill in the form</html>";
        Reply(fd, "400 Bad Request",
              "Content-Type: text/html; charset=utf-8\r\nContent-Length: " +
                  std::to_string(page.size()) + "\r\n",
              page, page.size());
        ::close(fd);
        return;
      }
    } else if (path == "/page.html") {
      ctype = "text/html";
    } else if (path == "/flaky.bin") {
      half = !probe && flaky.fetch_sub(1) > 0;
    } else if (path == "/liar.bin") {
      liar = !probe;
    } else if (path == "/slow.bin") {
      if (!probe) std::this_thread::sleep_for(std::chrono::milliseconds(1200));
    } else if ((path == "/bearer.bin" && auth != "Bearer tok") ||
               (path == "/basic.bin" && auth != "Basic dTpw") ||
               (path == "/hdr.bin" && Header(req, "x-key") != "k1")) {
      Reply(fd, "401 Unauthorized", "Content-Length: 0\r\n", "", 0);
      ::close(fd);
      return;
    } else if (path != "/data.bin" && path != "/bearer.bin" &&
               path != "/basic.bin" && path != "/hdr.bin") {
      Reply(fd, "404 Not Found", "Content-Length: 0\r\n", "", 0);
      ::close(fd);
      return;
    }

    const size_t total = body->size();
    std::string hdrs = "Content-Type: " + ctype + "\r\n";
    if (ranges && !liar && range.compare(0, 6, "bytes=") == 0) {
      size_t dash = range.find('-');
      size_t a = std::strtoull(range.c_str() + 6, nullptr, 10);
      size_t b = dash + 1 < range.size()
                     ? std::strtoull(range.c_str() + dash + 1, nullptr, 10)
                     : total - 1;
      if (a >= total) {
        Reply(fd, "416 Range Not Satisfiable",
              hdrs + "Content-Range: bytes */" + std::to_string(total) +
                  "\r\nContent-Length: 0\r\n",
              "", 0);
        ::close(fd);
        return;
      }
      b = std::min(b, total - 1);
      std::string part = body->substr(a, b - a + 1);
      Reply(fd, "206 Partial Content",
            hdrs + "Content-Range: bytes " + std::to_string(a) + "-" +
                std::to_string(b) + "/" + std::to_string(total) +
                "\r\nContent-Length: " + std::to_string(part.size()) + "\r\n",
            part, half ? part.size() / 2 : part.size());
    } else {
      if (length) hdrs += "Content-Length: " + std::to_string(total) + "\r\n";
      Reply(fd, "200 OK", hdrs, *body, total);
    }
    ::close(fd);
  }

  std::string data_;
  int lfd_ = -1;
  int port_ = 0;
  std::atomic<bool> stop_{false};
  std::thread th_;
  std::mutex mu_;
  std::vector<std::thread> conns_;
};

TestServer& Server() {
  static TestServer* s = [] {
    // curl must go straight to 127.0.0.1, never through a proxy.
    for (const char* v : {"http_proxy", "https_proxy", "HTTP_PROXY",
                          "HTTPS_PROXY", "ALL_PROXY", "all_proxy"}) {
      ::unsetenv(v);
    }
    ::setenv("no_proxy", "*", 1);
    ::setenv("NO_PROXY", "*", 1);
    return new TestServer();
  }();
  return *s;
}

std::string TmpDir() {
  static std::string d = [] {
    const char* t = std::getenv("TMPDIR");
    std::string tmpl = std::string(t && *t ? t : "/tmp") + "/cae_http_XXXXXX";
    std::vector<char> b(tmpl.begin(), tmpl.end());
    b.push_back('\0');
    return std::string(::mkdtemp(b.data()));
  }();
  return d;
}

std::string Slurp(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  std::ostringstream s;
  s << f.rdbuf();
  return s.str();
}

bool Exists(const std::string& p) {
  struct stat st;
  return ::stat(p.c_str(), &st) == 0;
}

cae::HttpFetchConfig Cfg() {
  cae::HttpFetchConfig c;
  c.streams = 4;
  c.part_bytes = 256 << 10;
  c.part_retries = 3;
  c.backoff_ms = 1;
  c.progress_s = 0;
  return c;
}

cae::HttpFetchRequest Req(const std::string& path, const std::string& dst) {
  cae::HttpFetchRequest r;
  r.auth.url = Server().Url(path);
  r.dst_path = TmpDir() + "/" + dst;
  return r;
}

/** A port nothing listens on. */
int DeadPort() {
  int s = ::socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  ::bind(s, reinterpret_cast<sockaddr*>(&a), sizeof(a));
  socklen_t len = sizeof(a);
  ::getsockname(s, reinterpret_cast<sockaddr*>(&a), &len);
  ::close(s);
  return ntohs(a.sin_port);
}

void WriteFile(const std::string& p, const std::string& text, mode_t mode) {
  std::ofstream(p, std::ios::trunc) << text;
  ::chmod(p.c_str(), mode);
}

}  // namespace

// --- probe ------------------------------------------------------------------

TEST_CASE("http_probe_ranged_plain_and_redirect", "[http_fetch]") {
  auto cfg = Cfg();
  cae::HttpAuth a;
  a.url = Server().Url("/data.bin");
  auto p = cae::HttpProbeUrl(a, cfg);
  REQUIRE(p.error.empty());
  REQUIRE(p.status == 206);
  REQUIRE(p.ranges);
  REQUIRE(p.total == static_cast<int64_t>(Server().data().size()));

  a.url = Server().Url("/norange.bin");
  p = cae::HttpProbeUrl(a, cfg);
  REQUIRE(p.status == 200);
  REQUIRE_FALSE(p.ranges);
  REQUIRE(p.total == static_cast<int64_t>(Server().data().size()));

  a.url = Server().Url("/redirect.bin");
  p = cae::HttpProbeUrl(a, cfg);
  REQUIRE(p.status == 206);
  REQUIRE(p.ranges);
}

TEST_CASE("http_probe_errors", "[http_fetch]") {
  auto cfg = Cfg();
  cae::HttpAuth a;
  a.url = "http://127.0.0.1:" + std::to_string(DeadPort()) + "/x";
  auto p = cae::HttpProbeUrl(a, cfg);
  REQUIRE(p.status == 0);
  REQUIRE(!p.error.empty());

  a.url = Server().Url("/data.bin");
  cfg.curl = "/nonexistent/curl-for-cae-test";
  p = cae::HttpProbeUrl(a, cfg);
  REQUIRE(p.error.find("cannot run") != std::string::npos);

  cfg = Cfg();
  a.url = Server().Url("/data.bin\nx");
  p = cae::HttpProbeUrl(a, cfg);
  REQUIRE(p.error.find("line break") != std::string::npos);
}

// --- fetch ------------------------------------------------------------------

TEST_CASE("http_fetch_parallel_ranges_then_skip_then_overwrite", "[http_fetch]") {
  auto cfg = Cfg();
  auto req = Req("/data.bin", "sub/dir/data.bin");  // parents are created
  std::vector<std::string> log;
  auto r = cae::HttpFetch(req, cfg, [&](const std::string& m) { log.push_back(m); });
  REQUIRE(r.ok);
  REQUIRE_FALSE(r.skipped);
  REQUIRE(r.bytes == Server().data().size());
  REQUIRE(r.parts == 13);  // 3 MiB + 12345 in 256 KiB parts
  REQUIRE(Slurp(req.dst_path) == Server().data());
  REQUIRE_FALSE(Exists(req.dst_path + ".cae-part"));
  REQUIRE(!log.empty());
  REQUIRE(log[0].find("ranged parts") != std::string::npos);

  auto again = cae::HttpFetch(req, cfg, nullptr);
  REQUIRE(again.ok);
  REQUIRE(again.skipped);

  cfg.overwrite = true;
  auto over = cae::HttpFetch(req, cfg, nullptr);
  REQUIRE(over.ok);
  REQUIRE_FALSE(over.skipped);
  REQUIRE(Slurp(req.dst_path) == Server().data());
}

TEST_CASE("http_fetch_whole_body_without_ranges", "[http_fetch]") {
  auto cfg = Cfg();
  std::vector<std::string> log;
  auto req = Req("/norange.bin", "norange.bin");
  auto r = cae::HttpFetch(req, cfg, [&](const std::string& m) { log.push_back(m); });
  REQUIRE(r.ok);
  REQUIRE(r.parts == 1);
  REQUIRE(log[0].find("no range support") != std::string::npos);
  REQUIRE(Slurp(req.dst_path) == Server().data());

  // No Content-Length either: the size is whatever arrives before close.
  auto nolen = Req("/nolen.bin", "nolen.bin");
  r = cae::HttpFetch(nolen, cfg, nullptr);
  REQUIRE(r.ok);
  REQUIRE(Slurp(nolen.dst_path) == Server().data());
}

TEST_CASE("http_fetch_empty_resource", "[http_fetch]") {
  auto req = Req("/empty.bin", "empty.bin");
  auto r = cae::HttpFetch(req, Cfg(), nullptr);
  REQUIRE(r.ok);
  REQUIRE(r.bytes == 0);
  REQUIRE(Exists(req.dst_path));
  REQUIRE(Slurp(req.dst_path).empty());
}

TEST_CASE("http_fetch_range_off_and_size", "[http_fetch]") {
  auto cfg = Cfg();
  auto req = Req("/data.bin", "slice.bin");
  req.range_off = 300000;
  req.range_size = 700001;
  auto r = cae::HttpFetch(req, cfg, nullptr);
  REQUIRE(r.ok);
  REQUIRE(Slurp(req.dst_path) == Server().data().substr(300000, 700001));

  req.range_off = Server().data().size() + 1;
  req.dst_path = TmpDir() + "/past.bin";
  r = cae::HttpFetch(req, cfg, nullptr);
  REQUIRE_FALSE(r.ok);
  REQUIRE(r.error.find("past the end") != std::string::npos);

  auto nr = Req("/norange.bin", "nr_slice.bin");
  nr.range_off = 10;
  r = cae::HttpFetch(nr, cfg, nullptr);
  REQUIRE_FALSE(r.ok);
  REQUIRE(r.error.find("does not support ranged") != std::string::npos);
}

TEST_CASE("http_fetch_http_errors_point_at_the_config_file", "[http_fetch]") {
  auto cfg = Cfg();
  auto req = Req("/needq.bin", "needq.bin");
  auto r = cae::HttpFetch(req, cfg, nullptr);
  REQUIRE_FALSE(r.ok);
  REQUIRE(r.probe.status == 400);
  REQUIRE(r.error.find("HTTP 400") != std::string::npos);
  REQUIRE(r.error.find("request parameters") != std::string::npos);
  REQUIRE_FALSE(Exists(req.dst_path));

  req.auth.url += "?key=val";
  r = cae::HttpFetch(req, cfg, nullptr);
  REQUIRE(r.ok);
  REQUIRE(Slurp(req.dst_path) == Server().data());

  auto missing = Req("/nope.bin", "nope.bin");
  r = cae::HttpFetch(missing, cfg, nullptr);
  REQUIRE_FALSE(r.ok);
  REQUIRE(r.probe.status == 404);
  REQUIRE(r.error == "HTTP 404");
}

TEST_CASE("http_fetch_credentials", "[http_fetch]") {
  auto cfg = Cfg();
  auto req = Req("/bearer.bin", "bearer.bin");
  auto r = cae::HttpFetch(req, cfg, nullptr);
  REQUIRE_FALSE(r.ok);
  REQUIRE(r.probe.status == 401);
  req.auth.headers = {"Authorization: Bearer tok"};
  r = cae::HttpFetch(req, cfg, nullptr);
  REQUIRE(r.ok);
  REQUIRE(Slurp(req.dst_path) == Server().data());

  auto basic = Req("/basic.bin", "basic.bin");
  basic.auth.userpwd = "u:p";
  r = cae::HttpFetch(basic, cfg, nullptr);
  REQUIRE(r.ok);

  auto bad = Req("/data.bin", "bad.bin");
  bad.auth.userpwd = "u\r\n:p";
  r = cae::HttpFetch(bad, cfg, nullptr);
  REQUIRE_FALSE(r.ok);
  REQUIRE(r.error.find("line break") != std::string::npos);
}

TEST_CASE("http_fetch_html_answer_is_flagged", "[http_fetch]") {
  std::vector<std::string> log;
  auto req = Req("/page.html", "page.html");
  auto r = cae::HttpFetch(req, Cfg(), [&](const std::string& m) { log.push_back(m); });
  REQUIRE(r.ok);
  REQUIRE(!log.empty());
  REQUIRE(log[0].find("text/html") != std::string::npos);
}

TEST_CASE("http_fetch_retry_resumes_a_short_part", "[http_fetch]") {
  auto cfg = Cfg();
  cfg.streams = 1;  // the failures land on known parts, in order
  Server().flaky = 2;
  std::vector<std::string> log;
  std::mutex mu;
  auto req = Req("/flaky.bin", "flaky.bin");
  auto r = cae::HttpFetch(req, cfg, [&](const std::string& m) {
    std::lock_guard<std::mutex> g(mu);
    log.push_back(m);
  });
  REQUIRE(r.ok);
  REQUIRE(r.retries == 2);
  REQUIRE(Slurp(req.dst_path) == Server().data());
  bool saw_retry = false;
  for (const auto& m : log) saw_retry |= m.find("retry 1/3") != std::string::npos;
  REQUIRE(saw_retry);
}

TEST_CASE("http_fetch_gives_up_after_the_retries", "[http_fetch]") {
  auto cfg = Cfg();
  cfg.part_retries = 0;
  Server().flaky = 1000;
  auto req = Req("/flaky.bin", "giveup.bin");
  auto r = cae::HttpFetch(req, cfg, nullptr);
  Server().flaky = 0;
  REQUIRE_FALSE(r.ok);
  REQUIRE(r.error.find("failed after 1 tries") != std::string::npos);
  REQUIRE_FALSE(Exists(req.dst_path));
  REQUIRE_FALSE(Exists(req.dst_path + ".cae-part"));
}

TEST_CASE("http_fetch_server_that_ignores_ranges_after_the_probe", "[http_fetch]") {
  // Retries allowed: this failure must not be retried, nor taken as success
  // (the truncated bodies would put the wrong bytes in every part but the
  // first).
  auto cfg = Cfg();
  auto req = Req("/liar.bin", "liar.bin");
  auto r = cae::HttpFetch(req, cfg, nullptr);
  REQUIRE_FALSE(r.ok);
  REQUIRE(r.error.find("more than the requested range") != std::string::npos);
  REQUIRE(r.error.find("after 1 tries") != std::string::npos);
  REQUIRE(r.retries == 0);
  REQUIRE_FALSE(Exists(req.dst_path));
}

TEST_CASE("http_fetch_reports_progress", "[http_fetch]") {
  auto cfg = Cfg();
  cfg.progress_s = 1;
  cfg.part_bytes = 2u << 20;
  cfg.streams = 1;  // two parts, 1.2 s each
  std::vector<std::string> log;
  std::mutex mu;
  auto req = Req("/slow.bin", "slow.bin");
  auto r = cae::HttpFetch(req, cfg, [&](const std::string& m) {
    std::lock_guard<std::mutex> g(mu);
    log.push_back(m);
  });
  REQUIRE(r.ok);
  bool saw = false;
  for (const auto& m : log) saw |= m.find("MB/s") != std::string::npos;
  REQUIRE(saw);
}

TEST_CASE("http_fetch_local_errors", "[http_fetch]") {
  auto cfg = Cfg();
  // A parent that is a file: mkdir fails.
  auto req = Req("/data.bin", "x");
  WriteFile(TmpDir() + "/plainfile", "x", 0644);
  req.dst_path = TmpDir() + "/plainfile/sub/out.bin";
  auto r = cae::HttpFetch(req, cfg, nullptr);
  REQUIRE_FALSE(r.ok);
  REQUIRE(r.error.find("mkdir") != std::string::npos);

  // A directory that cannot be written: open fails (root can write anyway).
  if (::geteuid() != 0) {
    const std::string ro = TmpDir() + "/ro";
    ::mkdir(ro.c_str(), 0555);
    req.dst_path = ro + "/out.bin";
    r = cae::HttpFetch(req, cfg, nullptr);
    REQUIRE_FALSE(r.ok);
    REQUIRE(r.error.find("open") != std::string::npos);
  }

  // Nothing listening.
  req.auth.url = "http://127.0.0.1:" + std::to_string(DeadPort()) + "/x";
  req.dst_path = TmpDir() + "/dead.bin";
  r = cae::HttpFetch(req, cfg, nullptr);
  REQUIRE_FALSE(r.ok);
  REQUIRE(r.error.find("probe failed") != std::string::npos);
}

TEST_CASE("http_fetch_config_from_env", "[http_fetch]") {
  ::setenv("CAE_HTTP_STREAMS", "999", 1);   // clamped to 64
  ::setenv("CAE_HTTP_PART_MB", "2", 1);
  ::setenv("CAE_HTTP_PART_RETRIES", "x", 1);  // not a number: default
  ::setenv("CAE_HTTP_BACKOFF_MS", "7", 1);
  ::setenv("CAE_HTTP_OVERWRITE", "1", 1);
  ::setenv("CAE_HTTP_PROGRESS_S", "0", 1);
  ::setenv("CAE_HTTP_CURL", "/usr/bin/curl", 1);
  auto c = cae::HttpFetchConfig::FromEnv();
  REQUIRE(c.streams == 64);
  REQUIRE(c.part_bytes == (2ull << 20));
  REQUIRE(c.part_retries == 5);
  REQUIRE(c.backoff_ms == 7);
  REQUIRE(c.overwrite);
  REQUIRE(c.progress_s == 0);
  REQUIRE(c.curl == "/usr/bin/curl");
  for (const char* v : {"CAE_HTTP_STREAMS", "CAE_HTTP_PART_MB",
                        "CAE_HTTP_PART_RETRIES", "CAE_HTTP_BACKOFF_MS",
                        "CAE_HTTP_OVERWRITE", "CAE_HTTP_PROGRESS_S",
                        "CAE_HTTP_CURL"}) {
    ::unsetenv(v);
  }
  auto d = cae::HttpFetchConfig::FromEnv();
  REQUIRE(d.streams == 8);
  REQUIRE(d.part_bytes == (64ull << 20));
  REQUIRE_FALSE(d.overwrite);
  REQUIRE(d.curl == "curl");
}

// --- the assimilator and the config file ------------------------------------

namespace {

cae::AssimilationCtx Ctx(const std::string& src, const std::string& dst) {
  cae::AssimilationCtx c{};
  c.src = src;
  c.dst = dst;
  c.format = "binary";
  c.range_off = 0;
  c.range_size = 0;
  return c;
}

/** Point the assimilator at a config file for this server. */
std::string UseConfig(const std::string& body, mode_t mode) {
  const std::string p = TmpDir() + "/cae_http_auth.ini";
  WriteFile(p, "[127.0.0.1:" + std::to_string(Server().port()) + "]\n" + body,
            mode);
  ::setenv("CAE_HTTP_AUTH_FILE", p.c_str(), 1);
  ::setenv("CAE_HTTP_PART_MB", "1", 1);
  ::setenv("CAE_HTTP_BACKOFF_MS", "1", 1);
  ::setenv("CAE_HTTP_PROGRESS_S", "0", 1);
  return p;
}

}  // namespace

TEST_CASE("http_assimilator_needs_a_file_destination", "[http_assim]") {
  const std::string src = Server().Url("/data.bin");
  REQUIRE(cae::HttpFileAssimilator::Run(Ctx(src, "iowarp::tag")) == -3);
  REQUIRE(cae::HttpFileAssimilator::Run(Ctx(src, "file::")) == -3);
  REQUIRE(cae::HttpFileAssimilator::Run(Ctx(src, "/no/scheme")) == -3);
}

TEST_CASE("http_assimilator_uses_the_config_file", "[http_assim]") {
  UseConfig("query.key = val\nheader.X-Key = k1\n", 0600);
  const std::string dst = TmpDir() + "/assim/needq.bin";
  REQUIRE(cae::HttpFileAssimilator::Run(
              Ctx(Server().Url("/needq.bin"), "file::" + dst)) == 0);
  REQUIRE(Slurp(dst) == Server().data());
  // Again: already there, skipped.
  REQUIRE(cae::HttpFileAssimilator::Run(
              Ctx(Server().Url("/needq.bin"), "file::" + dst)) == 0);
  // The header from the same section.
  const std::string hdst = TmpDir() + "/assim/hdr.bin";
  REQUIRE(cae::HttpFileAssimilator::Run(
              Ctx(Server().Url("/hdr.bin"), "file::" + hdst)) == 0);
  REQUIRE(Slurp(hdst) == Server().data());
  // An OMNI src_token is a bearer token.
  auto bc = Ctx(Server().Url("/bearer.bin"), "file::" + TmpDir() + "/assim/b.bin");
  bc.src_token = "tok";
  REQUIRE(cae::HttpFileAssimilator::Run(bc) == 0);
}

TEST_CASE("http_assimilator_error_codes", "[http_assim]") {
  UseConfig("query.key = val\n", 0600);
  const std::string dir = TmpDir() + "/assim_err/";
  // No section for "localhost": the server wants the parameter -> -400.
  const std::string local =
      "http://localhost:" + std::to_string(Server().port()) + "/needq.bin";
  REQUIRE(cae::HttpFileAssimilator::Run(Ctx(local, "file::" + dir + "a")) == -400);
  // Transport failure, no HTTP status -> -13.
  const std::string dead =
      "http://127.0.0.1:" + std::to_string(DeadPort()) + "/x";
  REQUIRE(cae::HttpFileAssimilator::Run(Ctx(dead, "file::" + dir + "b")) == -13);
  // A bad config file -> -1.
  UseConfig("token = x\n", 0600);
  REQUIRE(cae::HttpFileAssimilator::Run(
              Ctx(Server().Url("/data.bin"), "file::" + dir + "c")) == -1);
  // A secret in a file others can read -> -1, even for another host.
  UseConfig("bearer_token = s3cret\n", 0644);
  REQUIRE(cae::HttpFileAssimilator::Run(Ctx(local, "file::" + dir + "d")) == -1);
  auto r = cae::ResolveHttpAuth(local, "");
  REQUIRE_FALSE(r.ok);
  REQUIRE(r.error.find("chmod 600") != std::string::npos);
  // No secret: a readable file is fine.
  UseConfig("query.key = val\n", 0644);
  REQUIRE(cae::ResolveHttpAuth(Server().Url("/x"), "").ok);
  // No file at all: plain requests.
  ::setenv("CAE_HTTP_AUTH_FILE", (dir + "missing.ini").c_str(), 1);
  auto plain = cae::ResolveHttpAuth(Server().Url("/x"), "");
  REQUIRE(plain.ok);
  REQUIRE(plain.auth.sections.empty());
  ::unsetenv("CAE_HTTP_AUTH_FILE");
  const char* home = std::getenv("HOME");
  if (home && *home) {
    REQUIRE(cae::HttpAuthFilePath() ==
            std::string(home) + "/.clio/cae_http_auth.ini");
  }
}

TEST_CASE("http_factory_dispatches_http_and_https", "[http_assim]") {
  cae::AssimilatorFactory factory(std::shared_ptr<clio::cte::core::Client>(nullptr));
  REQUIRE(factory.Get("http://example.org/a.bin") != nullptr);
  REQUIRE(factory.Get("https://example.org/a.bin") != nullptr);
  auto a = factory.Get("https://example.org/a.bin");
  REQUIRE(dynamic_cast<cae::HttpFileAssimilator*>(a.get()) != nullptr);
}

SIMPLE_TEST_MAIN()
