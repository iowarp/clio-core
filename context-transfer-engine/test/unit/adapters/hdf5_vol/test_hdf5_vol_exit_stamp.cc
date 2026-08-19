/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * "The application never closed its file" regression.
 *
 * HDF5 registers H5_term_library() with atexit, so a writer that does not call
 * H5Fclose has its files closed during process exit. The connector's close
 * path is what writes the coherence stamp, and if the runtime has already been
 * torn down by then the stamp is silently skipped. That is not a small loss:
 * the next open reads "stamp absent", cannot trust the tier, DELETES the tag
 * and re-creates it under a fresh id -- leaving every chunk the writer stored
 * keyed to a tag nothing asks about. Measured on a 1 GB LAMMPS run as a writer
 * storing under tag (0.1) and the reader probing (0.2) and missing all of it.
 *
 * Reads keep returning correct data throughout, because the native file is
 * authoritative -- which is precisely why this hid: the round trip passes
 * while the entire compressed tier is dead weight. So the assertion here is
 * not "the data is right", it is "the writer's chunks are still reachable".
 *
 * Two processes, because process exit is the thing under test and neither a
 * thread nor fork() reproduces it -- see exit_stamp_child.cc.
 */

#include "simple_test.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;

namespace {

/** Absolute directory holding this test binary (and the VOL connector). */
std::string BinDir() {
  fs::path self = fs::read_symlink("/proc/self/exe");
  return self.parent_path().string();
}

/** Run the helper with the connector wired up through the environment only. */
int RunChild(const std::string &mode, const fs::path &store,
             const fs::path &h5, bool restart) {
  const std::string bin_dir = BinDir();
  std::string cmd;
  cmd += "CLIO_SERVER_CONF=" + (store / "compose.yaml").string() + " ";
  cmd += "CLIO_WITH_RUNTIME=1 ";
  cmd += "HDF5_VOL_CONNECTOR=clio ";
  cmd += "HDF5_PLUGIN_PATH=" + bin_dir + " ";
  cmd += "CLIO_VOL_CHUNK_SIZE=4096 ";
  /* Route through the compressor pool, which is what pins the core pool the
     chunks land in. Without it the connector's CTE client binds to whatever
     default pool exists, the configured cte_core (and its metadata log) goes
     unused, and the reader's restart has nothing to replay. */
  cmd += "CLIO_VOL_COMPRESSOR_POOL=512.0 ";
  /* 0 disables the ambiguity granule: without it a create and a close inside
     the same 10 ms window are reported ambiguous and the tag is dropped for a
     reason unrelated to what this test is about. */
  cmd += "CLIO_VOL_STAMP_GRANULARITY_NS=0 ";
  if (restart) cmd += "CLIO_RESTART=1 ";
  cmd += "LD_LIBRARY_PATH=" + bin_dir + ":${LD_LIBRARY_PATH:-} ";
  cmd += bin_dir + "/exit_stamp_child " + mode + " " + h5.string();
  cmd += " >/dev/null 2>&1";
  int rc = std::system(cmd.c_str());
  return WIFEXITED(rc) ? WEXITSTATUS(rc) : -1;
}

}  // namespace

/* WHAT THIS DOES AND DOES NOT COVER -- read before trusting it.
 *
 * It covers the shape: a writer that never calls H5Fclose, and a separate
 * process that reopens under CLIO_RESTART=1 and finds the writer's chunks
 * still reachable. That is a real property and it is worth a guard.
 *
 * It does NOT reproduce the atexit ORDERING that motivated the fix. Measured:
 * with the H5close() in clio_flush_stamps_at_exit() disabled, this test still
 * passes, while a stock LAMMPS run against the same connector goes from
 * populated=1/28 decompressions to populated=0/0. Whether HDF5's handler runs
 * before or after the runtime's teardown depends on static-initialisation
 * order across the whole link, and this helper links a small fraction of what
 * LAMMPS does. Do not read a pass here as proof that ordering is still
 * correct -- the reproducer for that is the two-process LAMMPS run. */
TEST_CASE("HDF5 VOL Exit - A file never closed still round-trips a restart",
          "[hdf5_vol][exit]") {
  const fs::path store = fs::temp_directory_path() / "clio_vol_exit_stamp";
  fs::remove_all(store);
  fs::create_directories(store);
  const fs::path h5 = store / "noclose.h5";

  /* A file-backed tier plus a metadata log: the reader is a different process,
     so RAM would die with the writer and without the log nothing would
     remember which blobs the tier holds. */
  {
    std::ofstream f(store / "compose.yaml");
    f << "networking:\n  port: 9421\n"
         "runtime:\n  num_threads: 2\n  queue_depth: 256\n"
         "compose:\n"
         "  - mod_name: clio_bdev\n"
         "    pool_name: \"" << (store / "bdev.dat").string() << "\"\n"
         "    pool_query: local\n    pool_id: \"301.0\"\n"
         "    bdev_type: file\n"
         "    path: \"" << (store / "bdev.dat").string() << "\"\n"
         "    capacity: \"256MB\"\n"
         "  - mod_name: clio_cte_compressor\n"
         "    pool_name: cte_compressor\n    pool_query: local\n"
         "    pool_id: \"512.0\"\n    next_pool_id: \"513.0\"\n"
         "  - mod_name: clio_cte_core\n"
         "    pool_name: cte_core\n    pool_query: local\n"
         "    pool_id: \"513.0\"\n"
         "    storage:\n"
         "      - path: \"" << (store / "tier.dat").string() << "\"\n"
         "        bdev_type: \"file\"\n"
         "        capacity_limit: \"128MB\"\n"
         "        score: 1.0\n"
         "        persistence_level: \"temporary\"\n"
         "    performance:\n"
         "      metadata_log_path: \"" << (store / "meta_log").string() << "\"\n"
         "      transaction_log_capacity: \"8MB\"\n"
         "    dpe:\n      dpe_type: \"max_bw\"\n";
  }

  /* Phase 1: write and exit with the file still open. A non-zero code here is
     the writer failing outright -- an earlier version of this path segfaulted
     in Transport::Send when the close ran after the transports were gone. */
  REQUIRE(RunChild("write", store, h5, /*restart=*/false) == 0);

  /* Phase 2: a separate process, its own runtime, replaying the log. Exit
     code 8 is the regression: chunk_0 unreachable because the tag was
     dropped and re-created when the stamp turned out to be missing. */
  REQUIRE(RunChild("verify", store, h5, /*restart=*/true) == 0);

  if (std::getenv("CLIO_VOL_EXIT_TEST_KEEP") == nullptr) {
    fs::remove_all(store);
  }
}

SIMPLE_TEST_MAIN()
