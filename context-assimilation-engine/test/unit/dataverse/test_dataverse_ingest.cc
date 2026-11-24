/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 */

// dataverse:// and http(s) zip sources through a real runtime (ParseOmni),
// against the fake server of fake_dataverse.h on 127.0.0.1 and the real curl.
//
// Covers dataverse_file_assimilator.cc (staging to file:: and ingest into
// iowarp::, HDF5 members through the HDF5 assimilator and other members as
// binary, empty members skipped, the manifest blob, and its error paths) and
// the http(s) -> iowarp:: hand-off in http_file_assimilator.cc.

#include <hdf5.h>

#include <clio_runtime/clio_runtime.h>
#include <clio_cae/core/constants.h>
#include <clio_cae/core/core_client.h>
#include <clio_cae/core/factory/assimilation_ctx.h>
#include <clio_cte/core/core_client.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "fake_dataverse.h"
#include "simple_test.h"
#include "zip_builder.h"

namespace cae = clio::cae::core;
using fake_dataverse::FakeDataverse;
using zip_builder::MakeZip;

namespace {

std::string TmpDir() {
  static std::string d = [] {
    const char *t = std::getenv("TMPDIR");
    std::string tmpl = std::string(t && *t ? t : "/tmp") + "/cae_dvi_XXXXXX";
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

const double kData[4] = {1.5, -2.0, 3.25, 1e300};

/** A small HDF5 file: dataset /data (4 doubles, attribute units) and a root
 *  attribute title. */
std::string MakeHdf5() {
  const std::string path = TmpDir() + "/src.h5";
  hid_t f = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  hsize_t n = 4;
  hid_t sp = H5Screate_simple(1, &n, nullptr);
  hid_t ds = H5Dcreate2(f, "/data", H5T_IEEE_F64LE, sp, H5P_DEFAULT,
                        H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, kData);
  hid_t scalar = H5Screate(H5S_SCALAR);
  hid_t st = H5Tcopy(H5T_C_S1);
  H5Tset_size(st, 1);
  hid_t a = H5Acreate2(ds, "units", st, scalar, H5P_DEFAULT, H5P_DEFAULT);
  H5Awrite(a, st, "m");
  H5Aclose(a);
  a = H5Acreate2(f, "title", st, scalar, H5P_DEFAULT, H5P_DEFAULT);
  H5Awrite(a, st, "t");
  H5Aclose(a);
  H5Tclose(st);
  H5Sclose(scalar);
  H5Dclose(ds);
  H5Sclose(sp);
  H5Fclose(f);
  return Slurp(path);
}

const std::string kNotes = "notes about the release\n";
const std::string kReadme = "release readme\n";

/** The fake server with a Dataverse dataset and two web zips, and a runtime
 *  with a CAE pool, set up once. */
struct World {
  FakeDataverse *srv = nullptr;
  cae::Client *cae_client = nullptr;
  std::string base;

  static World &Get() {
    static World w = [] {
      World x;
      for (const char *v : {"http_proxy", "https_proxy", "HTTP_PROXY",
                            "HTTPS_PROXY", "ALL_PROXY", "all_proxy"}) {
        ::unsetenv(v);
      }
      ::setenv("no_proxy", "*", 1);
      ::setenv("NO_PROXY", "*", 1);
      ::setenv("CAE_HTTP_AUTH_FILE", (TmpDir() + "/no_such.ini").c_str(), 1);
      ::setenv("CAE_HTTP_BACKOFF_MS", "1", 1);
      ::setenv("CAE_HTTP_PART_RETRIES", "1", 1);
      ::setenv("CAE_HTTP_PROGRESS_S", "0", 1);
      ::setenv("CAE_DATAVERSE_CACHE", (TmpDir() + "/cache").c_str(), 1);

      x.srv = new FakeDataverse();
      x.base = "http://127.0.0.1:" + std::to_string(x.srv->Port());
      const std::string h5 = MakeHdf5();
      std::map<int64_t, std::string> b;
      b[901] = MakeZip({{"rel/", "", false},
                        {"rel/test.h5", h5, false},
                        {"rel/notes.txt", kNotes, true},
                        {"rel/empty.txt", "", false}},
                       false);
      b[902] = kReadme;
      b[903] = MakeZip({{"bad.h5", "not an HDF5 file", false}}, false);
      x.srv->AddDataset("doi:10.5072/FK2/ING",
                        {{901, "", "Z.zip", true}, {902, "", "README.txt", true}},
                        b);

      if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true)) {
        std::fprintf(stderr, "CLIO_INIT failed\n");
        std::exit(1);
      }
      clio::cte::core::CLIO_CTE_CLIENT_INIT();
      CLIO_CAE_CLIENT_INIT();
      x.cae_client = new cae::Client();
      cae::CreateParams params;
      auto create = x.cae_client->AsyncCreate(clio::run::PoolQuery::Local(),
                                              "test_cae_dv_pool",
                                              cae::kCaePoolId, params);
      create.Wait();
      return x;
    }();
    return w;
  }

  /** Run one transfer; its return code (0 on success). */
  int Ingest(const std::string &src, const std::string &dst,
             const std::vector<std::string> &include,
             const std::string &format = "hdf5") {
    cae::AssimilationCtx ctx;
    ctx.src = src;
    ctx.dst = dst;
    ctx.format = format;
    ctx.include_patterns = include;
    auto t = cae_client->AsyncParseOmni(std::vector<cae::AssimilationCtx>{ctx});
    t.Wait();
    return static_cast<int>(t->GetReturnCode());
  }
};

/** Blob `name` of tag `tag`, whole; empty when absent. */
std::string Blob(const std::string &tag, const std::string &name) {
  auto *cte = CLIO_CTE_CLIENT;
  auto tt = cte->AsyncGetOrCreateTag(tag);
  tt.Wait();
  auto sz = cte->AsyncGetBlobSize(tt->tag_id_, name);
  sz.Wait();
  const size_t n = sz->size_;
  if (n == 0) return std::string();
  auto buf = CLIO_IPC->AllocateBuffer(n);
  auto get = cte->AsyncGetBlob(tt->tag_id_, name, 0, n, 0,
                               buf.shm_.template Cast<void>());
  get.Wait();
  std::string out = get->GetReturnCode() == 0
                        ? std::string(static_cast<const char *>(buf.ptr_), n)
                        : std::string();
  CLIO_IPC->FreeBuffer(buf);
  return out;
}

std::string Doubles(const double *v, size_t n) {
  return std::string(reinterpret_cast<const char *>(v), n * sizeof(double));
}

}  // namespace

TEST_CASE("dataverse_ingest_to_cte", "[dataverse][ingest]") {
  auto &w = World::Get();
  const std::string src = "dataverse::" + w.base + "/doi:10.5072/FK2/ING";
  REQUIRE(w.Ingest(src, "iowarp::dvi", {"Z.zip/*", "README.txt"}) == 0);

  const std::string m = Blob("dvi", "manifest");
  INFO(m);
  REQUIRE(m.find("# path\tsize\tfile_id\tmember\ttag\n") != std::string::npos);
  REQUIRE(m.find("Z.zip/rel/test.h5\t") != std::string::npos);
  REQUIRE(m.find("README.txt\t" + std::to_string(kReadme.size()) +
                 "\t902\t\tdvi/README.txt\n") != std::string::npos);
  // HDF5 member: one tag per dataset, with its attributes; root attributes on
  // the file's tag.
  REQUIRE(Blob("dvi/Z.zip/rel/test.h5/data", "description") ==
          "tensor<float64, 4>");
  REQUIRE(Blob("dvi/Z.zip/rel/test.h5/data", "chunk_0") == Doubles(kData, 4));
  REQUIRE(Blob("dvi/Z.zip/rel/test.h5/data", "attributes") ==
          R"({"units":"m"})");
  REQUIRE(Blob("dvi/Z.zip/rel/test.h5", "attributes") == R"({"title":"t"})");
  // Other members and files: binary. The empty member is listed, not stored.
  REQUIRE(Blob("dvi/Z.zip/rel/notes.txt", "chunk_0") == kNotes);
  REQUIRE(Blob("dvi/README.txt", "chunk_0") == kReadme);
  REQUIRE(m.find("Z.zip/rel/empty.txt\t0\t") != std::string::npos);
  REQUIRE(Blob("dvi/Z.zip/rel/empty.txt", "chunk_0").empty());
}

TEST_CASE("dataverse_ingest_binary_format", "[dataverse][ingest]") {
  // Without format hdf5 an HDF5 member is ingested as a plain file.
  auto &w = World::Get();
  const std::string src = "dataverse::" + w.base + "/doi:10.5072/FK2/ING";
  REQUIRE(w.Ingest(src, "iowarp::dvb", {"Z.zip/rel/test.h5"}, "binary") == 0);
  REQUIRE(Blob("dvb/Z.zip/rel/test.h5", "chunk_0").substr(0, 8) ==
          "\x89HDF\r\n\x1a\n");
}

TEST_CASE("dataverse_stage_to_file", "[dataverse][ingest]") {
  auto &w = World::Get();
  const std::string dir = TmpDir() + "/staged";
  const std::string src = "dataverse::" + w.base + "/doi:10.5072/FK2/ING";
  REQUIRE(w.Ingest(src, "file::" + dir, {"Z.zip/rel/notes.txt"}) == 0);
  REQUIRE(Slurp(dir + "/Z.zip/rel/notes.txt") == kNotes);
}

TEST_CASE("web_zip_ingest_to_cte", "[http][ingest]") {
  // An http(s) source to iowarp:: goes through the zip stager.
  auto &w = World::Get();
  REQUIRE(w.Ingest(w.base + "/web/901/release.zip", "iowarp::web",
                   {"*.zip/rel/test.h5", "hdf5:*data"}) == 0);
  const std::string m = Blob("web", "manifest");
  INFO(m);
  REQUIRE(m.find("# " + w.base + "/web/901/release.zip (web)\n") == 0);
  REQUIRE(m.find("release.zip/rel/test.h5\t") != std::string::npos);
  REQUIRE(Blob("web/release.zip/rel/test.h5/data", "chunk_0") ==
          Doubles(kData, 4));

  // To file:: the URL is still downloaded whole.
  const std::string out = TmpDir() + "/readme_copy.txt";
  REQUIRE(w.Ingest(w.base + "/web/902/README.txt", "file::" + out, {},
                   "binary") == 0);
  REQUIRE(Slurp(out) == kReadme);
}

TEST_CASE("dataverse_ingest_errors", "[dataverse][ingest]") {
  auto &w = World::Get();
  const std::string src = "dataverse::" + w.base + "/doi:10.5072/FK2/ING";
  // Not a dataverse source.
  REQUIRE(w.Ingest("dataverse://host", "iowarp::e1", {}) != 0);
  // A destination that is neither file:: nor iowarp::.
  REQUIRE(w.Ingest(src, "s3::bucket/x", {}) != 0);
  // No such dataset.
  REQUIRE(w.Ingest("dataverse::" + w.base + "/doi:10.5072/FK2/NONE",
                   "iowarp::e2", {}) != 0);
  // A web file that is not there.
  REQUIRE(w.Ingest(w.base + "/web/999/gone.zip", "iowarp::e3", {"*.zip/*"}) !=
          0);
  // An HDF5 member the HDF5 assimilator cannot open.
  REQUIRE(w.Ingest(w.base + "/web/903/bad.zip", "iowarp::e4", {"*.zip/*"}) !=
          0);
}

SIMPLE_TEST_MAIN()
