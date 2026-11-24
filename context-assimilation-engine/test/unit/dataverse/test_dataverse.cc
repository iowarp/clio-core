/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 */

// Offline tests of the dataverse:// source (dataverse.h): the URL parser, the
// JSON reader on a Dataverse API reply, the dataset_filter rules, and the
// ZIP/ZIP64 directory parser on archives built here in memory with zlib.

#include <clio_cae/core/factory/dataverse.h>

#include <zlib.h>

#include <string>
#include <vector>

#include "simple_test.h"
#include "zip_builder.h"

namespace cae = clio::cae::core;

namespace {

using zip_builder::Member;
using zip_builder::MakeZip;

/** Extract `e` from the whole archive `z` the way the stager does. */
std::string Extract(const std::string &z, const cae::ZipEntry &e) {
  const std::string hdr = z.substr(e.local_header_offset, 30 + 256);
  const uint64_t off = cae::ZipLocalDataOffset(hdr);
  if (off == 0) return "<no local header>";
  const std::string raw = z.substr(e.local_header_offset + off, e.csize);
  if (e.method == 0) return raw;
  std::string out(e.usize, '\0');
  z_stream zs{};
  inflateInit2(&zs, -MAX_WBITS);
  zs.next_in = reinterpret_cast<Bytef *>(const_cast<char *>(raw.data()));
  zs.avail_in = static_cast<uInt>(raw.size());
  zs.next_out = reinterpret_cast<Bytef *>(out.data());
  zs.avail_out = static_cast<uInt>(out.size());
  inflate(&zs, Z_FINISH);
  inflateEnd(&zs);
  return out;
}

void CheckZip(bool zip64, const std::string &comment) {
  std::string big(100000, '\0');
  for (size_t i = 0; i < big.size(); ++i) big[i] = static_cast<char>(i * 7 % 13);
  std::vector<Member> ms = {
      {"Swing01/", "", false},
      {"Swing01/P01_Swing01_stream_data.hdf5", big, true},
      {"Swing01/FPV_RGB.mp4", "not really video", false},
  };
  const std::string z = MakeZip(ms, zip64, comment);
  // Hand the parser only the tail, as the stager does for a large file.
  const uint64_t tail_len = std::min<uint64_t>(z.size(), 22 + comment.size() + 100);
  cae::ZipDirLocation loc;
  std::string err;
  REQUIRE(cae::ParseZipTail(z.substr(z.size() - tail_len), z.size(), &loc, &err));
  REQUIRE(loc.zip64 == zip64);
  if (loc.need_zip64_record) {
    REQUIRE(cae::ParseZip64Record(z.substr(loc.zip64_eocd_offset, 56), &loc, &err));
  }
  REQUIRE(loc.entries == 3);
  std::vector<cae::ZipEntry> es;
  REQUIRE(cae::ParseCentralDirectory(z.substr(loc.cd_offset, loc.cd_size), &es, &err));
  REQUIRE(es.size() == 3);
  REQUIRE(es[0].IsDir());
  REQUIRE(es[1].name == "Swing01/P01_Swing01_stream_data.hdf5");
  REQUIRE(es[1].method == 8);
  REQUIRE(es[1].usize == big.size());
  REQUIRE(es[1].csize < big.size());
  REQUIRE(Extract(z, es[1]) == big);
  REQUIRE(es[2].method == 0);
  REQUIRE(Extract(z, es[2]) == "not really video");
}

}  // namespace

TEST_CASE("dataverse_url_forms", "[dataverse]") {
  cae::DataverseSource s;
  std::string err;
  REQUIRE(cae::ParseDataverseUrl(
      "dataverse://dataverse.harvard.edu/doi:10.7910/DVN/LCCLLW", &s, &err));
  REQUIRE(s.base == "https://dataverse.harvard.edu");
  REQUIRE(s.pid == "doi:10.7910/DVN/LCCLLW");
  REQUIRE(cae::DataverseDatasetApiUrl(s) ==
          "https://dataverse.harvard.edu/api/datasets/:persistentId/"
          "?persistentId=doi%3A10.7910%2FDVN%2FLCCLLW");
  REQUIRE(cae::DataverseFileUrl(s, 14305276) ==
          "https://dataverse.harvard.edu/api/access/datafile/14305276");
  REQUIRE(cae::ParseDataverseUrl("dataverse::http://localhost:8080/hdl:1902.1/1",
                                 &s, &err));
  REQUIRE(s.base == "http://localhost:8080");
  REQUIRE(s.pid == "hdl:1902.1/1");
  REQUIRE(!cae::ParseDataverseUrl("dataverse://host", &s, &err));
  REQUIRE(!cae::ParseDataverseUrl("dataverse://host/LCCLLW", &s, &err));
  REQUIRE(!cae::ParseDataverseUrl("ftp://host/doi:1/2", &s, &err));
  // A plain web URL is a single web file, not a Dataverse dataset.
  REQUIRE(cae::ParseDataverseUrl("https://host/doi:1/2", &s, &err));
  REQUIRE(s.url == "https://host/doi:1/2");
}

TEST_CASE("dataverse_json_reader", "[dataverse]") {
  cae::Json j;
  std::string err;
  REQUIRE(cae::ParseJson(
      R"({"a":[1,-2.5e3,true,false,null],"s":"x\"\\\/\né😀","o":{}})",
      &j, &err));
  REQUIRE(j.Get("a")->arr.size() == 5);
  REQUIRE(j.Get("a")->arr[1].num == -2500);
  REQUIRE(j.Str("s") == "x\"\\/\n\xC3\xA9\xF0\x9F\x98\x80");
  REQUIRE(!cae::ParseJson(R"({"a":1,})", &j, &err));
  REQUIRE(!cae::ParseJson(R"({"a":"unterminated)", &j, &err));
  REQUIRE(!cae::ParseJson(R"([1] 2)", &j, &err));
}

TEST_CASE("dataverse_file_list", "[dataverse]") {
  const std::string body = R"({"status":"OK","data":{"latestVersion":{
    "versionState":"RELEASED","versionNumber":4,"versionMinorNumber":0,
    "files":[
      {"label":"P01.zip","restricted":false,"dataFile":{"id":14305279,
        "filename":"P01.zip","contentType":"application/zip","filesize":1138835174}},
      {"label":"notes.txt","directoryLabel":"docs/extra","restricted":true,
        "dataFile":{"id":7,"filename":"notes.txt","filesize":12}}]}}})";
  std::vector<cae::DataverseFile> fs;
  std::string ver, err;
  REQUIRE(cae::ParseDataverseFiles(body, &fs, &ver, &err));
  REQUIRE(ver == "RELEASED v4.0");
  REQUIRE(fs.size() == 2);
  REQUIRE(fs[0].id == 14305279);
  REQUIRE(fs[0].size == 1138835174LL);
  REQUIRE(fs[1].path == "docs/extra/notes.txt");
  REQUIRE(fs[1].restricted);
  REQUIRE(!cae::ParseDataverseFiles(R"({"status":"ERROR","message":"no"})", &fs,
                                    &ver, &err));
  REQUIRE(err.find("no") != std::string::npos);
  std::vector<cae::DataverseFile> bad;
  REQUIRE(!cae::ParseDataverseFiles(
      R"({"status":"OK","data":{"latestVersion":{"files":[
        {"directoryLabel":"..","dataFile":{"id":1,"filename":"x"}}]}}})",
      &bad, &ver, &err));
}

TEST_CASE("dataverse_selection_rules", "[dataverse]") {
  cae::DataverseSelection all({}, {"*.mp4"});
  REQUIRE(all.WholeFile("P01.zip"));
  REQUIRE(!all.WholeFile("clip.mp4"));
  REQUIRE(!all.OpenZip("P01.zip"));  // no member pattern: zips stay closed

  cae::DataverseSelection sel(
      {"Documentation.zip/*", "P*.zip/*_stream_data.hdf5", "README.txt"},
      {"P24_3.zip/*", "*Swing13*"});
  REQUIRE(sel.WholeFile("README.txt"));
  REQUIRE(!sel.WholeFile("P01.zip"));
  REQUIRE(sel.OpenZip("P01.zip"));
  REQUIRE(sel.OpenZip("P17_2.zip"));
  REQUIRE(!sel.OpenZip("Q01.zip"));
  REQUIRE(sel.Member("P01.zip", "Swing01/P01_Swing01_stream_data.hdf5"));
  REQUIRE(!sel.Member("P01.zip", "Swing01/FPV_RGB.mp4"));
  REQUIRE(!sel.Member("P01.zip", "Swing13/P01_Swing13_stream_data.hdf5"));
  REQUIRE(!sel.Member("P24_3.zip", "Swing01/P24_Swing01_stream_data.hdf5"));
  REQUIRE(sel.Member("Documentation.zip", "Annotation Data.csv"));
  REQUIRE(!sel.OpenZip("P24_3.zip") ||
          !sel.Member("P24_3.zip", "Swing02/P24_Swing02_stream_data.hdf5"));

  REQUIRE(cae::SafeRelPath("P01.zip/Swing01/a.hdf5"));
  REQUIRE(!cae::SafeRelPath("../etc/passwd"));
  REQUIRE(!cae::SafeRelPath("/abs"));
  REQUIRE(!cae::SafeRelPath("a//b"));
  REQUIRE(!cae::SafeRelPath("a/./b"));
}

TEST_CASE("dataverse_zip_directory", "[dataverse]") {
  CheckZip(/*zip64=*/false, "");
  CheckZip(/*zip64=*/false, "an archive comment");
  CheckZip(/*zip64=*/true, "");
  cae::ZipDirLocation loc;
  std::string err;
  REQUIRE(!cae::ParseZipTail(std::string(100, 'x'), 100, &loc, &err));
  REQUIRE(cae::ZipLocalDataOffset("not a header") == 0);
}

SIMPLE_TEST_MAIN()
