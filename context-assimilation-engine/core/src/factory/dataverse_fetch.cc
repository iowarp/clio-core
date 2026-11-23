/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 */

// Staging of Dataverse objects (see dataverse.h). The HTTP work is the http
// engine's (http_fetch.h: curl children, the same config file and CAE_HTTP_*
// knobs); this file adds the Dataverse listing, the zip directory walk and the
// inflation of the selected members. No runtime dependency, so cae_dataverse_tool
// and the assimilator share it.

#include <clio_cae/core/factory/dataverse.h>
#include <clio_cae/core/factory/http_auth.h>
#include <clio_cae/core/factory/http_fetch.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <zlib.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <thread>

namespace clio::cae::core {

namespace {

using Clock = std::chrono::steady_clock;

bool MkdirP(const std::string &dir) {
  if (dir.empty()) return true;
  struct stat st;
  if (::stat(dir.c_str(), &st) == 0) return S_ISDIR(st.st_mode);
  size_t s = dir.rfind('/');
  if (s != std::string::npos && s > 0 && !MkdirP(dir.substr(0, s))) return false;
  return ::mkdir(dir.c_str(), 0755) == 0 || errno == EEXIST;
}

std::string Dirname(const std::string &p) {
  size_t s = p.rfind('/');
  return s == std::string::npos ? "" : p.substr(0, s);
}

bool ReadFile(const std::string &path, std::string *out) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  std::ostringstream ss;
  ss << in.rdbuf();
  *out = ss.str();
  return true;
}

bool WriteFileAtomic(const std::string &path, const char *data, size_t n,
                     std::string *err) {
  if (!MkdirP(Dirname(path))) {
    *err = "cannot create directory for " + path;
    return false;
  }
  const std::string part = path + ".cae-part";
  {
    std::ofstream o(part, std::ios::binary | std::ios::trunc);
    if (!o || !o.write(data, static_cast<std::streamsize>(n))) {
      *err = "cannot write " + part;
      return false;
    }
  }
  if (std::rename(part.c_str(), path.c_str()) != 0) {
    *err = "cannot rename " + part + " to " + path;
    return false;
  }
  return true;
}

int64_t FileSize(const std::string &p) {
  struct stat st;
  return ::stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode)
             ? static_cast<int64_t>(st.st_size)
             : -1;
}

/** Shared state of one staging run. */
struct Run {
  DataverseSource src;
  std::string token;
  std::string stage_dir;
  std::string tmp_dir;
  HttpFetchConfig cfg;
  DataverseLog log;
  std::atomic<int> requests{0};
  std::atomic<uint64_t> fetched{0};
  std::atomic<int> tmp_seq{0};
  std::mutex log_mu;

  void Say(const std::string &m) {
    if (!log) return;
    std::lock_guard<std::mutex> g(log_mu);
    log(m);
  }

  bool Auth(const std::string &url, HttpAuth *a, std::string *err) {
    HttpAuthResult r = ResolveHttpAuth(url, "");
    if (!r.ok) {
      *err = r.error;
      return false;
    }
    *a = r.auth;
    // The API key is for the Dataverse host only, never for the storage
    // (S3) host its download redirects point at.
    if (!token.empty() && url.compare(0, src.base.size() + 1, src.base + "/") == 0)
      a->headers.push_back("X-Dataverse-key: " + token);
    return true;
  }

  /** Wait before retry `attempt` (1, 2, ...): CAE_HTTP_BACKOFF_MS doubled per
   *  attempt (2, 4, 8, ... s by default), at most 60 s. */
  void Backoff(int attempt) const {
    const int64_t ms = std::min<int64_t>(
        60000, static_cast<int64_t>(cfg.backoff_ms) << std::min(attempt, 6));
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
  }

  /**
   * Where the bytes of a data file are: the target of the access API's
   * redirect (a presigned storage URL), resolved with one request and reused
   * for every ranged read of that file, so a selection of many zip members
   * costs the Dataverse server one request per zip file rather than one per
   * member. Presigned URLs expire (an hour on Harvard Dataverse), so a link is
   * resolved again after kLinkTtl or when `refresh` is set. A file served by
   * the API itself (no redirect) is read from the API URL.
   */
  bool DataUrl(int64_t id, bool refresh, std::string *out, std::string *err) {
    static constexpr auto kLinkTtl = std::chrono::minutes(40);
    {
      std::lock_guard<std::mutex> g(link_mu);
      auto it = links.find(id);
      if (!refresh && it != links.end() &&
          Clock::now() - it->second.second < kLinkTtl) {
        *out = it->second.first;
        return true;
      }
    }
    const std::string api = DataverseFileUrl(src, id);
    for (int attempt = 0;; ++attempt) {
      int code = 0;
      std::string loc;
      if (!Resolve(api, &code, &loc, err)) return false;
      requests += 1;
      std::string url;
      if (code >= 300 && code < 400 && !loc.empty()) {
        url = loc;
      } else if (code == 200 || code == 206) {
        url = api;
      } else if ((code == 403 || code == 429 || code >= 500 || code == 0) &&
                 attempt < cfg.part_retries) {
        Say("data file " + std::to_string(id) + ": HTTP " +
            std::to_string(code) + ", retrying (attempt " +
            std::to_string(attempt + 1) + ")");
        Backoff(attempt + 1);
        continue;
      } else {
        *err = api + ": HTTP " + std::to_string(code);
        return false;
      }
      std::lock_guard<std::mutex> g(link_mu);
      links[id] = {url, Clock::now()};
      *out = url;
      return true;
    }
  }

  /** One GET of byte 0 without following redirects: status and Location. */
  bool Resolve(const std::string &url, int *code, std::string *loc,
               std::string *err) {
    HttpAuth a;
    if (!Auth(url, &a, err)) return false;
    std::string cfg_text = http_detail::CurlConfig(a, err);
    if (cfg_text.empty()) return false;
    // The URL and any credentials go to curl in a private config file, never
    // on its command line.
    const std::string path = tmp_dir + "/resolve." + std::to_string(::getpid()) +
                             "." + std::to_string(tmp_seq++);
    int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd < 0 || ::write(fd, cfg_text.data(), cfg_text.size()) !=
                      static_cast<ssize_t>(cfg_text.size())) {
      if (fd >= 0) ::close(fd);
      ::unlink(path.c_str());
      *err = "cannot write " + path;
      return false;
    }
    ::close(fd);
    const std::string cmd = cfg.curl +
                            " -sS -r 0-0 -o /dev/null --connect-timeout 30"
                            " -w '%{http_code} %{redirect_url}' -K '" +
                            path + "' 2>/dev/null";
    FILE *p = ::popen(cmd.c_str(), "r");
    std::string outp;
    if (p) {
      char buf[8192];
      size_t n;
      while ((n = std::fread(buf, 1, sizeof buf, p)) > 0) outp.append(buf, n);
      ::pclose(p);
    }
    ::unlink(path.c_str());
    *code = std::atoi(outp.c_str());
    size_t sp = outp.find(' ');
    *loc = sp == std::string::npos ? "" : outp.substr(sp + 1);
    return true;
  }

  /**
   * A 403 from storage is most often an expired presigned link. The probe
   * reports it as "HTTP 403"; a failed data GET only as curl's exit 22.
   */
  static bool LinkMayHaveExpired(const std::string &err) {
    return err.find("HTTP 403") != std::string::npos ||
           err.find("curl 22") != std::string::npos;
  }

  /** Bytes [off, off+len) of data file `id`, via its resolved link, retried. */
  bool GetData(int64_t id, uint64_t off, uint64_t len, std::string *out,
               std::string *err) {
    bool refresh = false;
    for (int attempt = 0;; ++attempt) {
      std::string url;
      if (!DataUrl(id, refresh, &url, err)) return false;
      if (Get(url, off, len, out, err)) return true;
      if (attempt >= cfg.part_retries) return false;
      refresh = LinkMayHaveExpired(*err);
      Say("data file " + std::to_string(id) + ": " + *err + "; retrying");
      Backoff(attempt + 1);
    }
  }

  std::mutex link_mu;
  std::map<int64_t, std::pair<std::string, Clock::time_point>> links;

  /** GET bytes [off, off+len) of url (len 0: everything) into *out. */
  bool Get(const std::string &url, uint64_t off, uint64_t len, std::string *out,
           std::string *err) {
    HttpFetchRequest req;
    if (!Auth(url, &req.auth, err)) return false;
    req.range_off = off;
    req.range_size = len;
    req.dst_path = tmp_dir + "/get." + std::to_string(::getpid()) + "." +
                   std::to_string(tmp_seq++);
    HttpFetchConfig c = cfg;
    c.overwrite = true;
    c.progress_s = 0;
    if (len > 0 && len <= c.part_bytes) c.streams = 1;
    HttpFetchResult r = HttpFetch(req, c, nullptr);
    requests += 1 + r.parts;  // the probe, then the GETs
    if (!r.ok) {
      ::unlink(req.dst_path.c_str());
      *err = url + ": " + r.error;
      return false;
    }
    fetched += r.bytes;
    bool ok = ReadFile(req.dst_path, out);
    ::unlink(req.dst_path.c_str());
    if (!ok) *err = "cannot read back " + req.dst_path;
    return ok;
  }

  /** Size of data file `id`, from a probe of its resolved link, retried. */
  int64_t Size(int64_t id, std::string *err) {
    bool refresh = false;
    for (int attempt = 0;; ++attempt) {
      std::string url;
      HttpAuth a;
      if (!DataUrl(id, refresh, &url, err) || !Auth(url, &a, err)) return -1;
      HttpProbe p = HttpProbeUrl(a, cfg);
      requests += 1;
      if (p.error.empty() && p.status > 0 && p.status < 400) {
        if (!p.ranges || p.total < 0) {
          *err = url + ": the server does not support ranged reads";
          return -1;
        }
        return p.total;
      }
      *err = url + ": probe " +
             (p.error.empty() ? "HTTP " + std::to_string(p.status) : p.error);
      if (attempt >= cfg.part_retries) return -1;
      refresh = LinkMayHaveExpired(*err);
      Say("data file " + std::to_string(id) + ": " + *err + "; retrying");
      Backoff(attempt + 1);
    }
  }
};

/** One object to stage. */
struct Job {
  DataverseObject obj;
  std::string url;
  ZipEntry entry;  ///< for a member
  uint64_t zip_size = 0;
};

bool Inflate(const std::string &raw, uint64_t usize, std::string *out,
             std::string *err) {
  out->assign(usize, '\0');
  z_stream zs{};
  if (inflateInit2(&zs, -MAX_WBITS) != Z_OK) {
    *err = "inflateInit2 failed";
    return false;
  }
  zs.next_in = reinterpret_cast<Bytef *>(const_cast<char *>(raw.data()));
  zs.avail_in = static_cast<uInt>(raw.size());
  zs.next_out = reinterpret_cast<Bytef *>(out->data());
  zs.avail_out = static_cast<uInt>(usize);
  int rc = inflate(&zs, Z_FINISH);
  const uint64_t got = zs.total_out;
  inflateEnd(&zs);
  if (rc != Z_STREAM_END || got != usize) {
    *err = "inflate failed (rc " + std::to_string(rc) + ", " +
           std::to_string(got) + " of " + std::to_string(usize) + " bytes)";
    return false;
  }
  return true;
}

/** Read a zip's directory and append a Job per selected member. */
bool ListZip(Run &run, const DataverseFile &f,
             const DataverseSelection &sel, std::vector<Job> *jobs,
             std::string *err) {
  const std::string url = DataverseFileUrl(run.src, f.id);
  int64_t total = f.size >= 0 ? f.size : run.Size(f.id, err);
  if (total < 0) return false;
  auto get = [&](uint64_t off, uint64_t len, std::string *out) {
    return run.GetData(f.id, off, len, out, err);
  };
  const uint64_t tail_len =
      std::min<uint64_t>(static_cast<uint64_t>(total), kZipTailBytes);
  std::string tail;
  if (!get(total - tail_len, tail_len, &tail)) return false;
  if (tail.size() != tail_len) {
    *err = f.path + ": short read of the zip's tail (size changed?)";
    return false;
  }
  ZipDirLocation loc;
  if (!ParseZipTail(tail, total, &loc, err)) {
    *err = f.path + ": " + *err;
    return false;
  }
  if (loc.need_zip64_record) {
    std::string rec;
    if (!get(loc.zip64_eocd_offset, 56, &rec) ||
        !ParseZip64Record(rec, &loc, err)) {
      *err = f.path + ": " + *err;
      return false;
    }
  }
  if (loc.cd_offset + loc.cd_size > static_cast<uint64_t>(total)) {
    *err = f.path + ": central directory lies past the end of the file";
    return false;
  }
  std::string cd;
  const uint64_t tail_off = total - tail_len;
  if (loc.cd_offset >= tail_off) {
    cd = tail.substr(loc.cd_offset - tail_off, loc.cd_size);
  } else if (!get(loc.cd_offset, loc.cd_size, &cd)) {
    return false;
  }
  std::vector<ZipEntry> entries;
  if (!ParseCentralDirectory(cd, &entries, err)) {
    *err = f.path + ": " + *err;
    return false;
  }
  size_t n = 0;
  for (const ZipEntry &e : entries) {
    if (e.IsDir() || !sel.Member(f.path, e.name)) continue;
    if (!SafeRelPath(e.name)) {
      *err = f.path + ": unsafe member name '" + e.name + "'";
      return false;
    }
    if (e.flags & 1) {
      *err = f.path + "/" + e.name + ": encrypted members are not supported";
      return false;
    }
    if (e.method != 0 && e.method != 8) {
      *err = f.path + "/" + e.name + ": compression method " +
             std::to_string(e.method) + " is not supported (stored/deflate)";
      return false;
    }
    Job j;
    j.url = url;
    j.entry = e;
    j.zip_size = static_cast<uint64_t>(total);
    j.obj.path = f.path + "/" + e.name;
    j.obj.local = run.stage_dir + "/" + j.obj.path;
    j.obj.size = e.usize;
    j.obj.file_id = f.id;
    j.obj.member = e.name;
    jobs->push_back(std::move(j));
    ++n;
  }
  run.Say(f.path + ": " + std::to_string(entries.size()) + " members, " +
          std::to_string(n) + " selected (directory " +
          std::to_string(loc.cd_size) + " B" + (loc.zip64 ? ", ZIP64" : "") +
          ")");
  return true;
}

bool StageMember(Run &run, Job &j, std::string *err) {
  const ZipEntry &e = j.entry;
  // The local header repeats the name and has its own extra field; ask for
  // the header, the data and some slack in one GET, and again if short.
  const uint64_t guess = 30 + e.name.size() + 1024 + e.csize;
  uint64_t len = std::min<uint64_t>(guess, j.zip_size - e.local_header_offset);
  std::string buf;
  if (!run.GetData(j.obj.file_id, e.local_header_offset, len, &buf, err))
    return false;
  uint64_t data_off = ZipLocalDataOffset(buf);
  if (data_off == 0) {
    *err = j.obj.path + ": no local header at offset " +
           std::to_string(e.local_header_offset);
    return false;
  }
  if (data_off + e.csize > buf.size()) {
    if (!run.GetData(j.obj.file_id, e.local_header_offset, data_off + e.csize,
                     &buf, err))
      return false;
  }
  std::string raw = buf.substr(data_off, e.csize);
  std::string data;
  if (e.method == 8) {
    if (!Inflate(raw, e.usize, &data, err)) {
      *err = j.obj.path + ": " + *err;
      return false;
    }
  } else {
    data = std::move(raw);
  }
  const uint32_t crc = static_cast<uint32_t>(
      crc32(0L, reinterpret_cast<const Bytef *>(data.data()),
            static_cast<uInt>(data.size())));
  if (crc != e.crc32 || data.size() != e.usize) {
    *err = j.obj.path + ": CRC-32 or size mismatch after extraction";
    return false;
  }
  return WriteFileAtomic(j.obj.local, data.data(), data.size(), err);
}

/**
 * A whole data file, fetched from its resolved link (not the API URL: curl
 * would carry the API key along the redirect to the storage host).
 */
bool StageWhole(Run &run, Job &j, std::string *err) {
  bool refresh = false;
  for (int attempt = 0;; ++attempt) {
    HttpFetchRequest req;
    std::string url;
    if (!run.DataUrl(j.obj.file_id, refresh, &url, err) ||
        !run.Auth(url, &req.auth, err))
      return false;
    req.dst_path = j.obj.local;
    HttpFetchResult r = HttpFetch(req, run.cfg, [&](const std::string &m) {
      run.Say(j.obj.path + ": " + m);
    });
    run.requests += 1 + r.parts;
    if (r.ok) {
      run.fetched += r.skipped ? 0 : r.bytes;
      j.obj.size = r.bytes;
      j.obj.skipped = r.skipped;
      return true;
    }
    *err = j.obj.path + ": " + r.error;
    if (attempt >= run.cfg.part_retries) return false;
    refresh = Run::LinkMayHaveExpired(r.error);
    run.Say(*err + "; retrying");
    run.Backoff(attempt + 1);
  }
}

/** Run fn over [0, n) on `jobs` threads; the first error stops new work. */
template <class Fn>
bool Parallel(size_t n, size_t jobs, Fn fn, std::string *err) {
  std::atomic<size_t> next{0};
  std::atomic<bool> failed{false};
  std::mutex mu;
  auto worker = [&] {
    for (;;) {
      if (failed) return;
      size_t i = next++;
      if (i >= n) return;
      std::string e;
      if (!fn(i, &e)) {
        std::lock_guard<std::mutex> g(mu);
        if (!failed.exchange(true)) *err = e;
        return;
      }
    }
  };
  std::vector<std::thread> ts;
  for (size_t t = 0; t < std::max<size_t>(1, std::min(jobs, n)); ++t)
    ts.emplace_back(worker);
  for (auto &t : ts) t.join();
  return !failed;
}

}  // namespace

std::string DataverseCacheDir(const DataverseSource &s) {
  std::string base;
  if (const char *c = std::getenv("CAE_DATAVERSE_CACHE"); c && *c) {
    base = c;
  } else {
    const char *h = std::getenv("HOME");
    base = std::string(h ? h : "/tmp") + "/.clio/cae_dataverse";
  }
  std::string id = s.pid;
  for (char &ch : id)
    if (ch == '/' || ch == ':') ch = '_';
  return base + "/" + id;
}

DataverseStageResult DataverseStage(const std::string &src,
                                    const std::vector<std::string> &include,
                                    const std::vector<std::string> &exclude,
                                    const std::string &stage_dir,
                                    const std::string &token, bool list_only,
                                    const DataverseLog &log) {
  DataverseStageResult res;
  const auto t0 = Clock::now();
  Run run;
  run.log = log;
  run.token = token;
  run.cfg = HttpFetchConfig::FromEnv();
  if (!ParseDataverseUrl(src, &run.src, &res.error)) return res;
  run.stage_dir = stage_dir;
  while (run.stage_dir.size() > 1 && run.stage_dir.back() == '/')
    run.stage_dir.pop_back();
  run.tmp_dir = run.stage_dir + "/.cae-tmp";
  if (!MkdirP(run.tmp_dir)) {
    res.error = "cannot create " + run.tmp_dir;
    return res;
  }

  // 1. The dataset's files.
  std::string body;
  for (int attempt = 0;; ++attempt) {
    if (run.Get(DataverseDatasetApiUrl(run.src), 0, 0, &body, &res.error)) break;
    if (attempt >= run.cfg.part_retries) return res;
    run.Say(res.error + "; retrying");
    run.Backoff(attempt + 1);
  }
  std::vector<DataverseFile> files;
  if (!ParseDataverseFiles(body, &files, &res.version, &res.error)) return res;
  res.dataset_files = files.size();
  run.Say(run.src.pid + " (" + res.version + "): " +
          std::to_string(files.size()) + " files");

  // 2. Selection; zip directories are read in parallel.
  DataverseSelection sel(include, exclude);
  std::vector<Job> jobs;
  std::vector<const DataverseFile *> zips;
  for (const auto &f : files) {
    if (sel.WholeFile(f.path)) {
      Job j;
      j.url = DataverseFileUrl(run.src, f.id);
      j.obj.path = f.path;
      j.obj.local = run.stage_dir + "/" + f.path;
      j.obj.size = f.size < 0 ? 0 : static_cast<uint64_t>(f.size);
      j.obj.file_id = f.id;
      jobs.push_back(std::move(j));
    }
    const bool is_zip =
        f.path.size() > 4 && f.path.compare(f.path.size() - 4, 4, ".zip") == 0;
    if (is_zip && sel.OpenZip(f.path)) zips.push_back(&f);
  }
  const char *jv = std::getenv("CAE_DATAVERSE_JOBS");
  const size_t njobs = jv ? std::max(1, std::min(64, std::atoi(jv))) : 8;
  std::vector<std::vector<Job>> per_zip(zips.size());
  if (!Parallel(
          zips.size(), njobs,
          [&](size_t i, std::string *e) {
            return ListZip(run, *zips[i], sel, &per_zip[i], e);
          },
          &res.error))
    return res;
  for (auto &v : per_zip)
    for (auto &j : v) jobs.push_back(std::move(j));
  std::sort(jobs.begin(), jobs.end(), [](const Job &a, const Job &b) {
    return a.obj.path < b.obj.path;
  });
  for (size_t i = 1; i < jobs.size(); ++i) {
    if (jobs[i].obj.path == jobs[i - 1].obj.path) {
      res.error = "two selected objects share the path " + jobs[i].obj.path;
      return res;
    }
  }
  run.Say(std::to_string(jobs.size()) + " objects selected from " +
          std::to_string(zips.size()) + " zip files and " +
          std::to_string(files.size()) + " dataset files");

  // 3. Stage.
  if (!list_only) {
    std::atomic<size_t> done{0};
    if (!Parallel(
            jobs.size(), njobs,
            [&](size_t i, std::string *e) {
              Job &j = jobs[i];
              if (j.obj.member.empty()) return StageWhole(run, j, e);
              if (!run.cfg.overwrite &&
                  FileSize(j.obj.local) == static_cast<int64_t>(j.obj.size)) {
                j.obj.skipped = true;
              } else if (!StageMember(run, j, e)) {
                return false;
              }
              size_t d = ++done;
              if (d % 100 == 0) {
                run.Say(std::to_string(d) + "/" + std::to_string(jobs.size()) +
                        " members staged");
              }
              return true;
            },
            &res.error))
      return res;
  }

  for (auto &j : jobs) {
    res.bytes += j.obj.size;
    res.objects.push_back(j.obj);
  }
  if (!list_only) {
    std::ostringstream m;
    m << "# path\tsize\tfile_id\tmember\n";
    for (const auto &o : res.objects)
      m << o.path << '\t' << o.size << '\t' << o.file_id << '\t' << o.member
        << '\n';
    const std::string text = m.str();
    if (!WriteFileAtomic(run.stage_dir + "/" + kDataverseManifest, text.data(),
                         text.size(), &res.error))
      return res;
  }
  ::rmdir(run.tmp_dir.c_str());
  res.requests = run.requests;
  res.fetched = run.fetched;
  res.seconds = std::chrono::duration<double>(Clock::now() - t0).count();
  res.ok = true;
  return res;
}

}  // namespace clio::cae::core
