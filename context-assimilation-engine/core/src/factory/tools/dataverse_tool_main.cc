/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 */

// cae_dataverse_tool: what DataverseFileAssimilator does before it ingests,
// without a runtime -- list a dataset, try OMNI dataset_filter patterns on it,
// and stage the selected files and zip members to a directory.
//
//   cae_dataverse_tool list  <src> [-i <include>]... [-x <exclude>]...
//   cae_dataverse_tool stage <src> <dir> [-i <include>]... [-x <exclude>]...
//
// <src> is dataverse://<host>/<persistent id>. CAE_DATAVERSE_TOKEN stands in
// for the OMNI src_token (sent as X-Dataverse-key).

#include <clio_cae/core/factory/dataverse.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace cae = clio::cae::core;

namespace {

int Usage(const char *a0) {
  std::fprintf(stderr,
               "usage: %s list  <dataverse://host/pid> [-i glob]... [-x glob]...\n"
               "       %s stage <dataverse://host/pid> <dir> [-i glob]... "
               "[-x glob]...\n",
               a0, a0);
  return 2;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 3) return Usage(argv[0]);
  const std::string cmd = argv[1];
  const std::string src = argv[2];
  int i = 3;
  std::string dir;
  if (cmd == "stage") {
    if (argc < 4) return Usage(argv[0]);
    dir = argv[3];
    i = 4;
  } else if (cmd != "list") {
    return Usage(argv[0]);
  }
  std::vector<std::string> inc, exc;
  for (; i < argc; ++i) {
    std::string a = argv[i];
    if ((a == "-i" || a == "-x") && i + 1 < argc) {
      (a == "-i" ? inc : exc).push_back(argv[++i]);
    } else {
      return Usage(argv[0]);
    }
  }
  cae::DataverseSource s;
  std::string err;
  if (!cae::ParseDataverseUrl(src, &s, &err)) {
    std::fprintf(stderr, "error: %s\n", err.c_str());
    return 2;
  }
  const bool list_only = cmd == "list";
  if (list_only) dir = cae::DataverseCacheDir(s);
  const char *tok = std::getenv("CAE_DATAVERSE_TOKEN");
  cae::DataverseStageResult r = cae::DataverseStage(
      src, inc, exc, dir, tok ? tok : "", list_only,
      [](const std::string &m) { std::fprintf(stderr, "%s\n", m.c_str()); });
  if (!r.ok) {
    std::fprintf(stderr, "error: %s\n", r.error.c_str());
    return 1;
  }
  for (const auto &o : r.objects) {
    std::printf("%12llu  %s%s\n", static_cast<unsigned long long>(o.size),
                o.path.c_str(), o.skipped ? "  (already staged)" : "");
  }
  std::fprintf(stderr,
               "%zu objects, %llu bytes; %d HTTP requests, %llu bytes "
               "received, %.1f s%s%s\n",
               r.objects.size(), static_cast<unsigned long long>(r.bytes),
               r.requests, static_cast<unsigned long long>(r.fetched),
               r.seconds, list_only ? "" : "; staged in ",
               list_only ? "" : dir.c_str());
  return 0;
}
