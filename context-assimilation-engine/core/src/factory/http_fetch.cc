/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 */

#include <clio_cae/core/factory/http_fetch.h>

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>

extern char **environ;

namespace clio::cae::core {

namespace {

using Clock = std::chrono::steady_clock;

long EnvLong(const char *name, long dflt, long lo, long hi) {
  const char *v = std::getenv(name);
  if (v == nullptr || *v == '\0') return dflt;
  char *end = nullptr;
  long x = std::strtol(v, &end, 10);
  if (end == v) return dflt;
  return std::max(lo, std::min(hi, x));
}

/** A running curl: its pid and the read end of its stdout. */
struct CurlProc {
  pid_t pid = -1;
  int out = -1;
};

/**
 * Start curl with `args`, feed `config` to its stdin (it reads -K -), and
 * return its stdout. stderr is inherited, so curl's own error messages land in
 * the caller's log. posix_spawn, not fork: the runtime process is large and
 * multithreaded.
 */
/** pipe() with both ends close-on-exec, atomically where the OS allows. */
int PipeCloexec(int fds[2]) {
#ifdef __linux__
  return ::pipe2(fds, O_CLOEXEC);
#else
  if (::pipe(fds) != 0) return -1;
  ::fcntl(fds[0], F_SETFD, FD_CLOEXEC);
  ::fcntl(fds[1], F_SETFD, FD_CLOEXEC);
  return 0;
#endif
}

bool SpawnCurl(const std::string &curl, const std::vector<std::string> &args,
               const std::string &config, CurlProc *p, std::string *err) {
  int in[2], out[2];
  if (PipeCloexec(in) != 0) {
    *err = std::string("pipe: ") + std::strerror(errno);
    return false;
  }
  if (PipeCloexec(out) != 0) {
    *err = std::string("pipe: ") + std::strerror(errno);
    ::close(in[0]);
    ::close(in[1]);
    return false;
  }
  posix_spawn_file_actions_t fa;
  posix_spawn_file_actions_init(&fa);
  posix_spawn_file_actions_adddup2(&fa, in[0], STDIN_FILENO);
  posix_spawn_file_actions_adddup2(&fa, out[1], STDOUT_FILENO);

  std::vector<char *> argv;
  argv.push_back(const_cast<char *>(curl.c_str()));
  for (const auto &a : args) argv.push_back(const_cast<char *>(a.c_str()));
  argv.push_back(nullptr);

  pid_t pid = -1;
  int rc = posix_spawnp(&pid, curl.c_str(), &fa, nullptr, argv.data(), environ);
  posix_spawn_file_actions_destroy(&fa);
  ::close(in[0]);
  ::close(out[1]);
  if (rc != 0) {
    *err = "cannot run " + curl + ": " + std::strerror(rc);
    ::close(in[1]);
    ::close(out[0]);
    return false;
  }
  // The config is a few hundred bytes, well under a pipe's buffer, so this
  // write does not wait for curl. EPIPE (curl already gone) shows up as a curl
  // failure when the caller reaps it.
  size_t off = 0;
  while (off < config.size()) {
    ssize_t n = ::write(in[1], config.data() + off, config.size() - off);
    if (n < 0) {
      if (errno == EINTR) continue;
      break;
    }
    off += static_cast<size_t>(n);
  }
  ::close(in[1]);
  p->pid = pid;
  p->out = out[0];
  return true;
}

/** Reap curl; returns its exit code (or 128 + signal, -1 on waitpid error). */
int ReapCurl(CurlProc *p) {
  if (p->out >= 0) {
    ::close(p->out);
    p->out = -1;
  }
  int status = 0;
  pid_t r;
  do {
    r = ::waitpid(p->pid, &status, 0);
  } while (r < 0 && errno == EINTR);
  p->pid = -1;
  if (r < 0) return -1;
  if (WIFEXITED(status)) return WEXITSTATUS(status);
  if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
  return -1;
}

std::string CurlExitText(int code) {
  switch (code) {
    case 0: return "ok";
    case 6: return "curl 6: could not resolve host (proxy settings?)";
    case 7: return "curl 7: could not connect";
    case 18: return "curl 18: transfer ended short";
    case 22: return "curl 22: HTTP error (>= 400)";
    case 28: return "curl 28: timed out or too slow";
    case 35: return "curl 35: TLS handshake failed";
    case 52: return "curl 52: empty reply";
    case 56: return "curl 56: connection reset while receiving";
    case 63: return "curl 63: larger than --max-filesize";
    default: return "curl exit " + std::to_string(code);
  }
}

/** Escape a value for a double-quoted curl config string. */
std::string Quote(const std::string &v) {
  std::string out = "\"";
  for (char c : v) {
    if (c == '\\' || c == '"') out += '\\';
    out += c;
  }
  return out + "\"";
}

bool MkdirP(const std::string &dir, std::string *err) {
  if (dir.empty()) return true;
  std::string cur;
  size_t i = 0;
  while (i <= dir.size()) {
    size_t slash = dir.find('/', i);
    cur = dir.substr(0, slash == std::string::npos ? dir.size() : slash);
    if (!cur.empty() && ::mkdir(cur.c_str(), 0755) != 0 && errno != EEXIST) {
      *err = "mkdir " + cur + ": " + std::strerror(errno);
      return false;
    }
    if (slash == std::string::npos) break;
    i = slash + 1;
  }
  return true;
}

std::string Mb(uint64_t b) {
  std::ostringstream s;
  s.setf(std::ios::fixed);
  s.precision(1);
  s << static_cast<double>(b) / 1e6 << " MB";
  return s.str();
}

/** One ranged GET's worth of the file. */
struct Part {
  uint64_t src_off;  ///< offset in the source
  uint64_t dst_off;  ///< offset in the output file
  uint64_t len;
};

/**
 * Fetch `len` bytes (or, when len == 0 and !ranged, the whole body) into fd at
 * dst_off. Returns bytes written; *code gets curl's exit code and *why a
 * reason when fewer bytes arrived than asked for.
 */
uint64_t FetchInto(const HttpFetchConfig &cfg, const std::string &curl_cfg,
                   bool ranged, uint64_t src_off, uint64_t dst_off,
                   uint64_t len, int fd, std::atomic<uint64_t> *done,
                   int *code, std::string *why) {
  std::vector<std::string> args = {
      "-sS", "-L", "--fail", "--connect-timeout", "30",
      // Abort a stream that moves under 10 KB/s for two minutes, so a dead
      // proxy connection is retried instead of hanging the transfer.
      "--speed-limit", "10240", "--speed-time", "120",
      "-o", "-", "-K", "-"};
  if (ranged) {
    args.push_back("-r");
    args.push_back(std::to_string(src_off) + "-" +
                   std::to_string(src_off + len - 1));
  }
  CurlProc p;
  std::string err;
  if (!SpawnCurl(cfg.curl, args, curl_cfg, &p, &err)) {
    *code = -1;
    *why = err;
    return 0;
  }
  std::vector<char> buf(4 << 20);
  uint64_t got = 0;
  bool over = false, werr = false;
  for (;;) {
    ssize_t n = ::read(p.out, buf.data(), buf.size());
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) break;
    size_t take = static_cast<size_t>(n);
    if (ranged && got + take > len) {  // server ignored Range
      take = static_cast<size_t>(len - got);
      over = true;
    }
    size_t w = 0;
    while (w < take) {
      ssize_t k = ::pwrite(fd, buf.data() + w, take - w,
                           static_cast<off_t>(dst_off + got + w));
      if (k < 0) {
        if (errno == EINTR) continue;
        *why = std::string("write: ") + std::strerror(errno);
        werr = true;
        break;
      }
      w += static_cast<size_t>(k);
    }
    got += w;
    if (done) done->fetch_add(w, std::memory_order_relaxed);
    if (over || werr) {
      ::kill(p.pid, SIGTERM);
      break;
    }
  }
  *code = ReapCurl(&p);
  if (over) {
    *why = "server sent more than the requested range";
    *code = -2;
  } else if (!werr && *code != 0) {
    *why = CurlExitText(*code);
  }
  if (werr) *code = -3;
  return got;
}

/**
 * A private temporary file holding `text`, removed when this object goes out
 * of scope. mkstemp creates it 0600, so other users cannot read the secrets a
 * wgetrc holds.
 */
class PrivateTempFile {
 public:
  /**
   * @param text bytes to write
   * @param err  set on failure; path() is then ""
   */
  PrivateTempFile(const std::string &text, std::string *err) {
    const char *tmp = std::getenv("TMPDIR");
    std::string tmpl = std::string(tmp != nullptr && *tmp != '\0' ? tmp : "/tmp") +
                       "/cae_http_XXXXXX";
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    int fd = ::mkstemp(buf.data());
    if (fd < 0) {
      *err = "mkstemp " + tmpl + ": " + std::strerror(errno);
      return;
    }
    path_ = buf.data();
    size_t off = 0;
    while (off < text.size()) {
      ssize_t n = ::write(fd, text.data() + off, text.size() - off);
      if (n < 0 && errno == EINTR) continue;
      if (n < 0) {
        *err = "write " + path_ + ": " + std::strerror(errno);
        break;
      }
      off += static_cast<size_t>(n);
    }
    ::close(fd);
    if (off < text.size()) {
      ::unlink(path_.c_str());
      path_.clear();
    }
  }
  ~PrivateTempFile() {
    if (!path_.empty()) ::unlink(path_.c_str());
  }
  PrivateTempFile(const PrivateTempFile &) = delete;
  PrivateTempFile &operator=(const PrivateTempFile &) = delete;

  /** Path of the file, or "" if it could not be created. */
  const std::string &path() const { return path_; }

 private:
  std::string path_;
};

/** What wget's exit status means (wget(1), EXIT STATUS). */
std::string WgetExitText(int code) {
  switch (code) {
    case 0: return "ok";
    case 3: return "wget 3: file I/O error";
    case 4: return "wget 4: network failure";
    case 5: return "wget 5: TLS verification failed";
    case 6: return "wget 6: username/password authentication failed";
    case 8: return "wget 8: the server answered with an error";
    default: return "wget exit " + std::to_string(code);
  }
}

/** The private wgetrc and URL list one wget run reads. */
struct WgetFiles {
  std::unique_ptr<PrivateTempFile> rc;
  std::unique_ptr<PrivateTempFile> urls;
};

/**
 * Write the wgetrc and the one-line URL list for `auth`.
 *
 * @return true when both files exist; otherwise false with *err set
 */
bool MakeWgetFiles(const HttpAuth &auth, WgetFiles *f, std::string *err) {
  std::string rc = http_detail::WgetConfig(auth, err);
  if (rc.empty()) return false;
  f->rc = std::make_unique<PrivateTempFile>(rc, err);
  if (f->rc->path().empty()) return false;
  f->urls = std::make_unique<PrivateTempFile>(auth.url + "\n", err);
  return !f->urls->path().empty();
}

/** Run wget with `args` and the private files; returns stdout and exit code. */
int RunWget(const HttpFetchConfig &cfg, const WgetFiles &f,
            std::vector<std::string> args, std::string *out,
            std::string *err) {
  // --config replaces ~/.wgetrc; -i reads the URL from the private file.
  args.insert(args.begin(), "--config=" + f.rc->path());
  args.push_back("-i");
  args.push_back(f.urls->path());
  CurlProc p;
  if (!SpawnCurl(cfg.wget, args, "", &p, err)) return -1;
  char buf[8192];
  for (;;) {
    ssize_t n = ::read(p.out, buf, sizeof(buf));
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) break;
    if (out) out->append(buf, static_cast<size_t>(n));
  }
  return ReapCurl(&p);
}

/** HttpProbeUrl for client = wget: a HEAD request (--spider) with -S. */
HttpProbe WgetProbe(const HttpAuth &auth, const HttpFetchConfig &cfg) {
  HttpProbe p;
  WgetFiles f;
  std::string err, dump;
  if (!MakeWgetFiles(auth, &f, &err)) {
    p.error = err;
    return p;
  }
  // -o /dev/stdout: wget writes its log (with -S, the response headers) there.
  int code = RunWget(cfg, f,
                     {"--spider", "-S", "-o", "/dev/stdout", "--tries=1",
                      "--timeout=30"},
                     &dump, &err);
  if (code < 0) {
    p.error = err;
    return p;
  }
  p = http_detail::ParseProbeHeaders(dump);
  p.ranges = false;  // fetched in one stream
  if (code != 0 && p.status == 0) p.error = WgetExitText(code);
  return p;
}

/**
 * Download the whole body with wget into `part_path`.
 *
 * @return "" on success, else why it failed
 */
std::string WgetFetchInto(const HttpAuth &auth, const HttpFetchConfig &cfg,
                          const std::string &part_path) {
  WgetFiles f;
  std::string err;
  if (!MakeWgetFiles(auth, &f, &err)) return err;
  int code = RunWget(cfg, f,
                     {"-q", "-O", part_path,
                      "--tries=" + std::to_string(cfg.part_retries + 1),
                      "--waitretry=" + std::to_string(std::max(1, cfg.backoff_ms / 1000)),
                      "--retry-connrefused", "--timeout=30",
                      "--read-timeout=120"},
                     nullptr, &err);
  if (code < 0) return err;
  return code == 0 ? "" : WgetExitText(code);
}

}  // namespace

HttpFetchConfig HttpFetchConfig::FromEnv() {
  HttpFetchConfig c;
  c.streams = static_cast<size_t>(EnvLong("CAE_HTTP_STREAMS", 8, 1, 64));
  c.part_bytes =
      static_cast<uint64_t>(EnvLong("CAE_HTTP_PART_MB", 64, 1, 4096)) << 20;
  c.part_retries = static_cast<int>(EnvLong("CAE_HTTP_PART_RETRIES", 5, 0, 100));
  c.backoff_ms = static_cast<int>(EnvLong("CAE_HTTP_BACKOFF_MS", 1000, 0, 60000));
  c.overwrite = EnvLong("CAE_HTTP_OVERWRITE", 0, 0, 1) == 1;
  c.progress_s = static_cast<int>(EnvLong("CAE_HTTP_PROGRESS_S", 30, 0, 3600));
  const char *curl = std::getenv("CAE_HTTP_CURL");
  if (curl != nullptr && *curl != '\0') c.curl = curl;
  const char *wget = std::getenv("CAE_HTTP_WGET");
  if (wget != nullptr && *wget != '\0') c.wget = wget;
  return c;
}

namespace http_detail {

std::string CurlConfig(const HttpAuth &auth, std::string *err) {
  auto bad = [](const std::string &v) {
    return v.find_first_of("\r\n") != std::string::npos;
  };
  if (bad(auth.url) || bad(auth.userpwd)) {
    *err = "URL or user contains a line break";
    return "";
  }
  std::string c = "url = " + Quote(auth.url) + "\n";
  for (const auto &h : auth.headers) {
    if (bad(h)) {
      *err = "a header contains a line break";
      return "";
    }
    c += "header = " + Quote(h) + "\n";
  }
  if (!auth.userpwd.empty()) c += "user = " + Quote(auth.userpwd) + "\n";
  return c;
}

std::string WgetConfig(const HttpAuth &auth, std::string *err) {
  // A wgetrc value runs to the end of its line and loses surrounding blanks.
  auto bad = [](const std::string &v) {
    return v.find_first_of("\r\n") != std::string::npos ||
           (!v.empty() && (std::isspace(static_cast<unsigned char>(v.front())) ||
                           std::isspace(static_cast<unsigned char>(v.back()))));
  };
  std::string c = "# cae http request\n";
  for (const auto &h : auth.headers) {
    if (bad(h)) {
      *err = "a header cannot be passed to wget (line break or outer blank)";
      return "";
    }
    c += "header = " + h + "\n";
  }
  if (!auth.userpwd.empty()) {
    size_t colon = auth.userpwd.find(':');
    std::string user = auth.userpwd.substr(0, colon);
    std::string pass = colon == std::string::npos ? "" : auth.userpwd.substr(colon + 1);
    if (bad(user) || bad(pass)) {
      *err = "user or password cannot be passed to wget (line break or outer "
             "blank)";
      return "";
    }
    c += "http_user = " + user + "\n";
    if (!pass.empty()) c += "http_password = " + pass + "\n";
    c += "auth_no_challenge = off\n";
  }
  return c;
}

HttpProbe ParseProbeHeaders(const std::string &dump) {
  HttpProbe p;
  std::istringstream in(dump);
  std::string line;
  int64_t clen = -1;
  std::string crange;
  while (std::getline(in, line)) {
    line = detail::Trim(line);  // wget -S indents; also drops the '\r'
    if (line.compare(0, 5, "HTTP/") == 0) {  // a new response (redirects)
      std::istringstream sl(line);
      std::string ver;
      sl >> ver >> p.status;
      p.content_type.clear();
      clen = -1;
      crange.clear();
      continue;
    }
    size_t colon = line.find(':');
    if (colon == std::string::npos) continue;
    std::string name = http_detail::Lower(line.substr(0, colon));
    std::string val = detail::Trim(line.substr(colon + 1));
    if (name == "content-type") {
      p.content_type = val;
    } else if (name == "content-length") {
      clen = std::strtoll(val.c_str(), nullptr, 10);
    } else if (name == "content-range") {
      crange = val;
    }
  }
  size_t slash = crange.rfind('/');
  if (slash != std::string::npos && crange.compare(slash + 1, 1, "*") != 0) {
    p.total = std::strtoll(crange.c_str() + slash + 1, nullptr, 10);
  }
  if (p.status == 206 && p.total >= 0) {
    p.ranges = true;
  } else if (p.status == 416 && p.total >= 0) {  // "bytes */0": empty resource
    p.ranges = true;
  } else if (p.status == 200) {
    p.ranges = false;  // the server ignored Range
    p.total = clen;
  }
  return p;
}

}  // namespace http_detail

HttpProbe HttpProbeUrl(const HttpAuth &auth, const HttpFetchConfig &cfg) {
  if (auth.client == "wget") return WgetProbe(auth, cfg);
  HttpProbe p;
  std::string err;
  std::string conf = http_detail::CurlConfig(auth, &err);
  if (conf.empty()) {
    p.error = err;
    return p;
  }
  // -D - puts every response's headers on stdout. --max-filesize stops a
  // server that ignores Range from sending the whole body to /dev/null; the
  // headers have arrived by then (curl exits 63, which is fine here).
  CurlProc proc;
  if (!SpawnCurl(cfg.curl,
                 {"-sS", "-L", "-r", "0-0", "-o", "/dev/null", "-D", "-",
                  "--max-filesize", "1048576", "--connect-timeout", "30",
                  "--max-time", "120", "-K", "-"},
                 conf, &proc, &err)) {
    p.error = err;
    return p;
  }
  std::string dump;
  char buf[8192];
  for (;;) {
    ssize_t n = ::read(proc.out, buf, sizeof(buf));
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) break;
    dump.append(buf, static_cast<size_t>(n));
  }
  int code = ReapCurl(&proc);
  p = http_detail::ParseProbeHeaders(dump);
  if (code != 0 && code != 63 && p.status == 0) p.error = CurlExitText(code);
  return p;
}

namespace {

/**
 * The client = wget path of HttpFetch, after the probe and the skip check:
 * one wget run into "<dst>.cae-part", renamed to dst once complete.
 *
 * @param req  the download
 * @param cfg  tuning (wget executable, retries)
 * @param want bytes expected (meaningful when r->probe.total >= 0)
 * @param r    result to fill: ok, bytes, error
 */
void FetchWithWget(const HttpFetchRequest &req, const HttpFetchConfig &cfg,
                   uint64_t want, HttpFetchResult *r) {
  std::string err;
  size_t slash = req.dst_path.rfind('/');
  if (slash != std::string::npos && !MkdirP(req.dst_path.substr(0, slash), &err)) {
    r->error = err;
    return;
  }
  const std::string part_path = req.dst_path + ".cae-part";
  r->parts = 1;
  std::string why = WgetFetchInto(req.auth, cfg, part_path);
  struct stat st;
  const bool have = ::stat(part_path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
  r->bytes = have ? static_cast<uint64_t>(st.st_size) : 0;
  if (why.empty() && !have) why = "wget wrote no file";
  if (why.empty() && r->probe.total >= 0 && r->bytes != want) {
    why = "got " + std::to_string(r->bytes) + " bytes, expected " +
          std::to_string(want);
  }
  if (why.empty() && ::rename(part_path.c_str(), req.dst_path.c_str()) != 0) {
    why = "rename to " + req.dst_path + ": " + std::strerror(errno);
  }
  if (!why.empty()) {
    ::unlink(part_path.c_str());
    r->error = why;
    return;
  }
  r->ok = true;
}

}  // namespace

HttpFetchResult HttpFetch(const HttpFetchRequest &req,
                          const HttpFetchConfig &cfg, const HttpLog &log) {
  HttpFetchResult r;
  const auto t0 = Clock::now();
  auto say = [&](const std::string &s) {
    if (log) log(s);
  };
  // curl writes to a pipe we may close early; that must not kill this process.
  ::signal(SIGPIPE, SIG_IGN);

  std::string err;
  const std::string curl_cfg = http_detail::CurlConfig(req.auth, &err);
  if (curl_cfg.empty()) {
    r.error = err;
    return r;
  }

  if (req.auth.client == "wget" && (req.range_off > 0 || req.range_size > 0)) {
    r.error = "range_off/range_size are not supported with client = wget";
    return r;
  }

  // 1. Size and range support.
  r.probe = HttpProbeUrl(req.auth, cfg);
  const HttpProbe &pr = r.probe;
  if (!pr.error.empty()) {
    r.error = "probe failed: " + pr.error;
    return r;
  }
  // 416 to "bytes=0-0" with "bytes */0" is an empty resource, not an error.
  const bool empty_resource = pr.status == 416 && pr.ranges && pr.total == 0;
  if ((pr.status >= 400 && !empty_resource) || pr.status == 0) {
    r.error = "HTTP " + std::to_string(pr.status) +
              (pr.status == 401 || pr.status == 403 || pr.status == 400
                   ? " (the server wants credentials or request parameters; "
                     "see " + HttpAuthFilePath() + ")"
                   : "");
    return r;
  }
  if (pr.content_type.compare(0, 9, "text/html") == 0) {
    // Download URLs that answer with a web page are usually a login or
    // licence page, not the data. Allowed, but worth a line in the log.
    say("warning: the server answered with text/html; check that this is the "
        "data, not a login or licence page");
  }
  const bool ranged = pr.ranges && cfg.streams > 0;
  if ((req.range_off > 0 || req.range_size > 0) && !pr.ranges) {
    r.error = "range_off/range_size given, but the server does not support "
              "ranged requests";
    return r;
  }
  uint64_t want = 0;  // bytes to fetch; 0 with total unknown = whole body
  if (pr.total >= 0) {
    const uint64_t total = static_cast<uint64_t>(pr.total);
    if (req.range_off > total) {
      r.error = "range_off " + std::to_string(req.range_off) +
                " is past the end (" + std::to_string(total) + " bytes)";
      return r;
    }
    want = total - req.range_off;
    if (req.range_size > 0) want = std::min(want, req.range_size);
  }

  // 2. Already there?
  struct stat st;
  if (!cfg.overwrite && pr.total >= 0 &&
      ::stat(req.dst_path.c_str(), &st) == 0 && S_ISREG(st.st_mode) &&
      static_cast<uint64_t>(st.st_size) == want) {
    r.ok = true;
    r.skipped = true;
    r.bytes = want;
    return r;
  }

  if (req.auth.client == "wget") {
    say("GET " + Mb(want) + " in one stream with wget");
    FetchWithWget(req, cfg, want, &r);
    r.seconds = std::chrono::duration<double>(Clock::now() - t0).count();
    return r;
  }

  // 3. Output file.
  size_t slash = req.dst_path.rfind('/');
  if (slash != std::string::npos && !MkdirP(req.dst_path.substr(0, slash), &err)) {
    r.error = err;
    return r;
  }
  const std::string part_path = req.dst_path + ".cae-part";
  int fd = ::open(part_path.c_str(), O_CREAT | O_TRUNC | O_WRONLY | O_CLOEXEC,
                  0644);
  if (fd < 0) {
    r.error = "open " + part_path + ": " + std::strerror(errno);
    return r;
  }
  if (pr.total >= 0 && ::ftruncate(fd, static_cast<off_t>(want)) != 0) {
    r.error = "ftruncate " + part_path + ": " + std::strerror(errno);
    ::close(fd);
    ::unlink(part_path.c_str());
    return r;
  }

  // 4. Parts.
  std::vector<Part> parts;
  if (ranged && want > 0) {
    for (uint64_t o = 0; o < want; o += cfg.part_bytes) {
      parts.push_back({req.range_off + o, o, std::min(cfg.part_bytes, want - o)});
    }
  } else if (!ranged) {
    parts.push_back({0, 0, want});  // one GET of the whole body
  }
  r.parts = static_cast<int>(parts.size());
  const size_t nthreads = std::max<size_t>(1, std::min(cfg.streams, parts.size()));
  say("GET " + Mb(want) + (ranged ? " in " + std::to_string(parts.size()) +
                                        " ranged parts over " +
                                        std::to_string(nthreads) + " streams"
                                  : " in one stream (no range support)"));

  std::atomic<size_t> next{0};
  std::atomic<uint64_t> done{0};
  std::atomic<int> retries{0};
  std::atomic<bool> failed{false};
  std::mutex emu;
  std::string first_error;
  auto worker = [&]() {
    for (;;) {
      if (failed.load()) return;
      size_t i = next.fetch_add(1);
      if (i >= parts.size()) return;
      Part pt = parts[i];
      uint64_t have = 0;  // bytes of this part already written
      for (int attempt = 0;; ++attempt) {
        int code = 0;
        std::string why;
        if (!ranged) {  // no ranges: a retry starts over
          done.fetch_sub(have);
          have = 0;
        }
        uint64_t got = FetchInto(cfg, curl_cfg, ranged, pt.src_off + have,
                                 pt.dst_off + have, pt.len - have, fd, &done,
                                 &code, &why);
        have += got;
        // A ranged part is done once all its bytes are written; a whole-body
        // GET once curl exits cleanly with the advertised length (if any).
        // -2 (the server ignored Range) wrote bytes from the wrong offset, so
        // the part is never complete then, and a retry would get the same.
        const bool complete =
            ranged ? have == pt.len && code != -2
                   : code == 0 && (pt.len == 0 || have == pt.len);
        if (complete) break;
        // -3: local write error, -2: server ignored Range. Neither is retried.
        if (code == -3 || code == -2 || attempt >= cfg.part_retries) {
          std::lock_guard<std::mutex> g(emu);
          if (first_error.empty()) {
            first_error = "bytes " + std::to_string(pt.src_off) + "+" +
                          std::to_string(pt.len) + " failed after " +
                          std::to_string(attempt + 1) + " tries: " +
                          (why.empty() ? "short read" : why);
          }
          failed.store(true);
          return;
        }
        retries.fetch_add(1);
        {
          std::lock_guard<std::mutex> g(emu);
          say("retry " + std::to_string(attempt + 1) + "/" +
              std::to_string(cfg.part_retries) + " of bytes " +
              std::to_string(pt.src_off + have) + "+" +
              std::to_string(pt.len - have) + ": " +
              (why.empty() ? "short read" : why));
        }
        std::this_thread::sleep_for(
            std::chrono::milliseconds(static_cast<int64_t>(cfg.backoff_ms) << std::min(attempt, 5)));
      }
    }
  };
  std::vector<std::thread> pool;
  for (size_t t = 0; t < nthreads; ++t) pool.emplace_back(worker);

  // Progress, from this thread, while the workers run.
  std::atomic<bool> finished{false};
  std::thread watcher([&]() {
    auto last = Clock::now();
    while (!finished.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      if (cfg.progress_s <= 0) continue;
      auto now = Clock::now();
      if (now - last < std::chrono::seconds(cfg.progress_s)) continue;
      last = now;
      double s = std::chrono::duration<double>(now - t0).count();
      uint64_t d = done.load();
      std::ostringstream m;
      m.setf(std::ios::fixed);
      m.precision(1);
      m << Mb(d);
      if (want > 0) m << " of " << Mb(want) << " (" << 100.0 * d / want << "%)";
      m << ", " << d / 1e6 / std::max(s, 1e-3) << " MB/s";
      std::lock_guard<std::mutex> g(emu);
      say(m.str());
    }
  });
  for (auto &t : pool) t.join();
  finished.store(true);
  watcher.join();

  r.retries = retries.load();
  r.bytes = done.load();
  r.seconds = std::chrono::duration<double>(Clock::now() - t0).count();
  if (failed.load()) {
    ::close(fd);
    ::unlink(part_path.c_str());
    r.error = first_error;
    return r;
  }
  if (::close(fd) != 0) {
    r.error = "close " + part_path + ": " + std::strerror(errno);
    ::unlink(part_path.c_str());
    return r;
  }
  if (pr.total >= 0 && r.bytes != want) {
    r.error = "got " + std::to_string(r.bytes) + " bytes, expected " +
              std::to_string(want);
    ::unlink(part_path.c_str());
    return r;
  }
  if (::rename(part_path.c_str(), req.dst_path.c_str()) != 0) {
    r.error = "rename to " + req.dst_path + ": " + std::strerror(errno);
    ::unlink(part_path.c_str());
    return r;
  }
  r.ok = true;
  return r;
}

}  // namespace clio::cae::core
