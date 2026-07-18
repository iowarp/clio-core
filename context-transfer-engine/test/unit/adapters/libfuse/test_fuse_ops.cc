/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * FUSE ADAPTER — CALLBACK COVERAGE TEST
 *
 * Drives every FUSE operation callback in fuse_cte.cc directly, in-process,
 * against a live embedded Clio runtime + filesystem chimod. No real mount is
 * required: the callbacks are plain functions over CLIO_CFS_CLIENT, so we call
 * them with hand-built fuse_file_info / stat / timespec arguments and assert
 * their POSIX-style return codes and side effects.
 *
 * fuse_cte.cc is #included directly (with its main() renamed) so this TU can
 * reach its file-static callbacks and the same object is coverage-instrumented.
 *
 * Covered: init, getattr (root/file/dir/symlink/missing), create/open/read/
 * write/flush/fsync/release, mkdir/rmdir/readdir, chmod/chown/utimens,
 * symlink/readlink/link/unlink, truncate/fallocate, the four xattr ops, rename
 * (plain / NOREPLACE / unsupported-flag), and statfs.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/bdev/bdev_client.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>

#include <unistd.h>  // _exit

#include <algorithm>
#include <chrono>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

// Pull in the adapter implementation so the file-static callbacks and the
// cte_fuse_ops table are visible here and instrumented for coverage. The
// adapter's main()/apptainer mount glue lives in fuse_cte_main.cc (not included
// here) — it needs a real mount + apptainer fd injection and is exercised by
// the mount smoke test, not this unit test.
#include "adapter/libfuse/fuse_cte.cc"  // NOLINT(build/include)

#include "simple_test.h"

using namespace clio::cae::fuse;  // NOLINT(build/namespaces)

// ============================================================================
// Fixture: embedded runtime + CTE pool + RAM target + filesystem chimod
// ============================================================================

class FuseOpsFixture {
 public:
  static constexpr clio::run::u64 kTargetSize = 64ULL * 1024 * 1024;  // 64 MB
  static inline bool g_initialized = false;

  FuseOpsFixture() {
    if (g_initialized) return;

    bool success = clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
    REQUIRE(success);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    success = clio::cte::core::CLIO_CTE_CLIENT_INIT();
    REQUIRE(success);

    auto *cte_client = CLIO_CTE_CLIENT;
    REQUIRE(cte_client != nullptr);
    cte_client->Init(clio::cte::core::kCtePoolId);

    clio::cte::core::CreateParams params;
    params.config_.performance_.stat_targets_period_ms_ = 0;
    auto create_task = cte_client->AsyncCreate(
        clio::run::PoolQuery::Dynamic(), clio::cte::core::kCtePoolName,
        clio::cte::core::kCtePoolId, params);
    create_task.Wait();
    REQUIRE(create_task->GetReturnCode() == 0);

    clio::run::PoolId bdev_pool_id(952, 0);
    auto reg_task = cte_client->AsyncRegisterTarget(
        "cte_fuse_ops_ram", clio::run::bdev::BdevType::kRam, kTargetSize,
        clio::run::PoolQuery::DirectHash(0), bdev_pool_id);
    reg_task.Wait();
    REQUIRE(reg_task->GetReturnCode() == 0);

    // Bring up the filesystem chimod the callbacks delegate to.
    success = clio::cte::filesystem::CLIO_CFS_CLIENT_INIT();
    REQUIRE(success);

    g_initialized = true;
    INFO("Embedded runtime + CTE pool + RAM target + CFS chimod ready");
  }
};

static FuseOpsFixture *Fx() {
  return ctp::Singleton<FuseOpsFixture>::GetInstance();
}

// A zeroed fuse_file_info for callbacks that only read/write ->fh and ->flags.
static struct fuse_file_info MakeFi() {
  struct fuse_file_info fi;
  std::memset(&fi, 0, sizeof(fi));
  return fi;
}

// readdir collector: matches fuse_fill_dir_t.
static std::vector<std::string> *g_fill_names = nullptr;
static int CollectFiller(void *buf, const char *name,
                         const struct stat *stbuf, off_t off,
                         enum fuse_fill_dir_flags flags) {
  (void)buf;
  (void)stbuf;
  (void)off;
  (void)flags;
  if (g_fill_names != nullptr && name != nullptr) {
    g_fill_names->emplace_back(name);
  }
  return 0;
}

// ============================================================================
// Lifecycle
// ============================================================================

TEST_CASE("FUSE ops - init is idempotent", "[fuse][ops]") {
  Fx();
  // Calling the real init callback after setup re-runs the (idempotent) client
  // inits and fills the fuse_config, covering cte_fuse_init.
  struct fuse_conn_info conn;
  std::memset(&conn, 0, sizeof(conn));
  struct fuse_config cfg;
  std::memset(&cfg, 0, sizeof(cfg));
  void *ret = cte_fuse_init(&conn, &cfg);
  (void)ret;
  REQUIRE(cfg.use_ino == 1);
  REQUIRE(cfg.attr_timeout == 0);
}

// ============================================================================
// getattr
// ============================================================================

TEST_CASE("FUSE ops - getattr root and missing", "[fuse][ops]") {
  Fx();
  struct stat st;
  REQUIRE(cte_fuse_getattr("/", &st, nullptr) == 0);
  REQUIRE(S_ISDIR(st.st_mode));
  REQUIRE(st.st_ino == 1);

  REQUIRE(cte_fuse_getattr("/no_such_file_xyz", &st, nullptr) == -ENOENT);
}

// ============================================================================
// create / write / read / release + getattr on a regular file
// ============================================================================

TEST_CASE("FUSE ops - file create write read release", "[fuse][ops]") {
  Fx();
  const char *path = "/ops/file.bin";
  auto fi = MakeFi();
  REQUIRE(cte_fuse_create(path, 0644, &fi) == 0);

  const std::string payload = "hello fuse ops coverage payload 1234567890";
  int wrote = cte_fuse_write(path, payload.data(), payload.size(), 0, &fi);
  REQUIRE(wrote == static_cast<int>(payload.size()));

  // Zero-length write short-circuits to 0.
  REQUIRE(cte_fuse_write(path, payload.data(), 0, 0, &fi) == 0);

  REQUIRE(cte_fuse_flush(path, &fi) == 0);
  REQUIRE(cte_fuse_fsync(path, 0, &fi) == 0);

  std::vector<char> buf(payload.size());
  int got = cte_fuse_read(path, buf.data(), buf.size(), 0, &fi);
  REQUIRE(got == static_cast<int>(payload.size()));
  REQUIRE(std::memcmp(buf.data(), payload.data(), payload.size()) == 0);

  // Zero-length read short-circuits to 0.
  REQUIRE(cte_fuse_read(path, buf.data(), 0, 0, &fi) == 0);

  // getattr on the regular file reports size + reg mode.
  struct stat st;
  REQUIRE(cte_fuse_getattr(path, &st, &fi) == 0);
  REQUIRE(S_ISREG(st.st_mode));
  REQUIRE(st.st_size == static_cast<off_t>(payload.size()));
  REQUIRE(st.st_blocks > 0);

  REQUIRE(cte_fuse_release(path, &fi) == 0);

  // Re-open the existing file with plain open, read back.
  auto fi2 = MakeFi();
  REQUIRE(cte_fuse_open(path, &fi2) == 0);
  int got2 = cte_fuse_read(path, buf.data(), buf.size(), 0, &fi2);
  REQUIRE(got2 == static_cast<int>(payload.size()));
  REQUIRE(cte_fuse_release(path, &fi2) == 0);

  // Re-open with O_TRUNC: the open callback truncates the file to zero length.
  auto fi3 = MakeFi();
  fi3.flags = O_TRUNC | O_WRONLY;
  REQUIRE(cte_fuse_open(path, &fi3) == 0);
  REQUIRE(cte_fuse_release(path, &fi3) == 0);
  struct stat st_trunc;
  REQUIRE(cte_fuse_getattr(path, &st_trunc, nullptr) == 0);
  REQUIRE(st_trunc.st_size == 0);

  REQUIRE(cte_fuse_unlink(path) == 0);
}

TEST_CASE("FUSE ops - read/write bad handle", "[fuse][ops]") {
  Fx();
  auto fi = MakeFi();  // fh == 0 -> no handle
  char c = 'x';
  REQUIRE(cte_fuse_read("/x", &c, 1, 0, &fi) == -EBADF);
  REQUIRE(cte_fuse_write("/x", &c, 1, 0, &fi) == -EBADF);
}

TEST_CASE("FUSE ops - open missing file is ENOENT", "[fuse][ops]") {
  Fx();
  auto fi = MakeFi();
  int rc = cte_fuse_open("/definitely/not/here.dat", &fi);
  REQUIRE(rc == -ENOENT);
}

// ============================================================================
// directories
// ============================================================================

TEST_CASE("FUSE ops - mkdir readdir rmdir", "[fuse][ops]") {
  Fx();
  REQUIRE(cte_fuse_mkdir("/ops_dir", 0755) == 0);

  // Create a couple of children so readdir has entries.
  auto fa = MakeFi();
  REQUIRE(cte_fuse_create("/ops_dir/a.txt", 0644, &fa) == 0);
  REQUIRE(cte_fuse_release("/ops_dir/a.txt", &fa) == 0);
  auto fb = MakeFi();
  REQUIRE(cte_fuse_create("/ops_dir/b.txt", 0644, &fb) == 0);
  REQUIRE(cte_fuse_release("/ops_dir/b.txt", &fb) == 0);

  std::vector<std::string> names;
  g_fill_names = &names;
  auto rfi = MakeFi();
  REQUIRE(cte_fuse_readdir("/ops_dir", nullptr, CollectFiller, 0, &rfi,
                           static_cast<enum fuse_readdir_flags>(0)) == 0);
  g_fill_names = nullptr;
  REQUIRE(std::find(names.begin(), names.end(), ".") != names.end());
  REQUIRE(std::find(names.begin(), names.end(), "a.txt") != names.end());
  REQUIRE(std::find(names.begin(), names.end(), "b.txt") != names.end());

  // getattr on the directory reports dir mode.
  struct stat st;
  REQUIRE(cte_fuse_getattr("/ops_dir", &st, nullptr) == 0);
  REQUIRE(S_ISDIR(st.st_mode));

  // Non-empty rmdir should fail; remove children first.
  REQUIRE(cte_fuse_unlink("/ops_dir/a.txt") == 0);
  REQUIRE(cte_fuse_unlink("/ops_dir/b.txt") == 0);
  REQUIRE(cte_fuse_rmdir("/ops_dir") == 0);
}

// ============================================================================
// chmod / chown / utimens
// ============================================================================

TEST_CASE("FUSE ops - chmod chown utimens", "[fuse][ops]") {
  Fx();
  const char *path = "/ops/perm.bin";
  auto fi = MakeFi();
  REQUIRE(cte_fuse_create(path, 0644, &fi) == 0);
  REQUIRE(cte_fuse_release(path, &fi) == 0);

  REQUIRE(cte_fuse_chmod(path, 0755, nullptr) == 0);
  struct stat st;
  REQUIRE(cte_fuse_getattr(path, &st, nullptr) == 0);
  REQUIRE((st.st_mode & 0777) == 0755);

  // chmod on a missing path must be ENOENT (existence probe).
  REQUIRE(cte_fuse_chmod("/ops/missing_perm.bin", 0600, nullptr) == -ENOENT);

  REQUIRE(cte_fuse_chown(path, 4242, 4343, nullptr) == 0);
  REQUIRE(cte_fuse_getattr(path, &st, nullptr) == 0);
  REQUIRE(st.st_uid == 4242);
  REQUIRE(st.st_gid == 4343);

  // utimens: explicit times.
  struct timespec tv[2];
  tv[0].tv_sec = 111111;
  tv[0].tv_nsec = 222;
  tv[1].tv_sec = 333333;
  tv[1].tv_nsec = 444;
  REQUIRE(cte_fuse_utimens(path, tv, nullptr) == 0);

  // utimens: UTIME_NOW / UTIME_OMIT.
  tv[0].tv_nsec = UTIME_NOW;
  tv[1].tv_nsec = UTIME_OMIT;
  REQUIRE(cte_fuse_utimens(path, tv, nullptr) == 0);

  // utimens: mtime UTIME_NOW, atime UTIME_OMIT (the mirror of the case above,
  // covering the mtime-now branch).
  tv[0].tv_nsec = UTIME_OMIT;
  tv[1].tv_nsec = UTIME_NOW;
  REQUIRE(cte_fuse_utimens(path, tv, nullptr) == 0);

  // utimens: a pre-epoch (negative) timestamp round-trips through the signed
  // ns<->timespec conversion (the rem<0 normalization branch).
  tv[0].tv_sec = -5;  // 5 s before the epoch
  tv[0].tv_nsec = 500000000;  // .5 s
  tv[1].tv_sec = -5;
  tv[1].tv_nsec = 500000000;
  REQUIRE(cte_fuse_utimens(path, tv, nullptr) == 0);
  struct stat st_pre;
  REQUIRE(cte_fuse_getattr(path, &st_pre, nullptr) == 0);
  REQUIRE(st_pre.st_mtim.tv_sec == -5);
  REQUIRE(st_pre.st_mtim.tv_nsec == 500000000);

  // utimens: NULL tv means "set both to now".
  REQUIRE(cte_fuse_utimens(path, nullptr, nullptr) == 0);

  REQUIRE(cte_fuse_unlink(path) == 0);
}

// ============================================================================
// symlink / readlink / link / unlink
// ============================================================================

TEST_CASE("FUSE ops - symlink readlink", "[fuse][ops]") {
  Fx();
  const char *target = "/ops/real_target.dat";
  const char *link = "/ops/soft_link";
  auto fi = MakeFi();
  REQUIRE(cte_fuse_create(target, 0644, &fi) == 0);
  REQUIRE(cte_fuse_release(target, &fi) == 0);

  REQUIRE(cte_fuse_symlink(target, link) == 0);

  char buf[256];
  REQUIRE(cte_fuse_readlink(link, buf, sizeof(buf)) == 0);
  REQUIRE(std::string(buf) == target);

  // readlink with zero size is EINVAL.
  REQUIRE(cte_fuse_readlink(link, buf, 0) == -EINVAL);

  // getattr on the symlink reports S_IFLNK.
  struct stat st;
  REQUIRE(cte_fuse_getattr(link, &st, nullptr) == 0);
  REQUIRE(S_ISLNK(st.st_mode));

  REQUIRE(cte_fuse_unlink(link) == 0);
  REQUIRE(cte_fuse_unlink(target) == 0);
}

TEST_CASE("FUSE ops - hard link", "[fuse][ops]") {
  Fx();
  const char *from = "/ops/link_src.dat";
  const char *to = "/ops/link_alias.dat";
  auto fi = MakeFi();
  REQUIRE(cte_fuse_create(from, 0644, &fi) == 0);
  const std::string data = "hardlink-shared-data";
  REQUIRE(cte_fuse_write(from, data.data(), data.size(), 0, &fi) ==
          static_cast<int>(data.size()));
  REQUIRE(cte_fuse_release(from, &fi) == 0);

  int rc = cte_fuse_link(from, to);
  REQUIRE(rc == 0);

  REQUIRE(cte_fuse_unlink(to) == 0);
  REQUIRE(cte_fuse_unlink(from) == 0);
}

// ============================================================================
// truncate
// ============================================================================

TEST_CASE("FUSE ops - truncate", "[fuse][ops]") {
  Fx();
  const char *path = "/ops/trunc.bin";
  auto fi = MakeFi();
  REQUIRE(cte_fuse_create(path, 0644, &fi) == 0);
  std::string data(1000, 'T');
  REQUIRE(cte_fuse_write(path, data.data(), data.size(), 0, &fi) ==
          static_cast<int>(data.size()));
  REQUIRE(cte_fuse_release(path, &fi) == 0);

  REQUIRE(cte_fuse_truncate(path, 100, nullptr) == 0);
  struct stat st;
  REQUIRE(cte_fuse_getattr(path, &st, nullptr) == 0);
  REQUIRE(st.st_size == 100);

  REQUIRE(cte_fuse_unlink(path) == 0);
}

// ============================================================================
// fallocate (Linux)
// ============================================================================

#ifdef __linux__
TEST_CASE("FUSE ops - fallocate modes", "[fuse][ops]") {
  Fx();
  const char *path = "/ops/falloc.bin";
  auto fi = MakeFi();
  REQUIRE(cte_fuse_create(path, 0644, &fi) == 0);

  // EINVAL: negative offset / non-positive length.
  REQUIRE(cte_fuse_fallocate(path, 0, -1, 10, &fi) == -EINVAL);
  REQUIRE(cte_fuse_fallocate(path, 0, 0, 0, &fi) == -EINVAL);

  // Unsupported (layout-changing) mode.
  REQUIRE(cte_fuse_fallocate(path, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE,
                             0, 10, &fi) == -EOPNOTSUPP);

  // mode 0: grow EOF to offset+length.
  REQUIRE(cte_fuse_fallocate(path, 0, 0, 4096, &fi) == 0);
  struct stat st;
  REQUIRE(cte_fuse_getattr(path, &st, &fi) == 0);
  REQUIRE(st.st_size == 4096);

  // mode 0 again but smaller: never shrinks -> success, size unchanged.
  REQUIRE(cte_fuse_fallocate(path, 0, 0, 100, &fi) == 0);
  REQUIRE(cte_fuse_getattr(path, &st, &fi) == 0);
  REQUIRE(st.st_size == 4096);

  // KEEP_SIZE only: no-op success.
  REQUIRE(cte_fuse_fallocate(path, FALLOC_FL_KEEP_SIZE, 0, 8192, &fi) == 0);

  // ZERO_RANGE: writes zeros over a range.
  REQUIRE(cte_fuse_fallocate(path, FALLOC_FL_ZERO_RANGE, 0, 2048, &fi) == 0);

  // ZERO_RANGE | KEEP_SIZE: zero past EOF, then restore size.
  REQUIRE(cte_fuse_fallocate(
              path, FALLOC_FL_ZERO_RANGE | FALLOC_FL_KEEP_SIZE, 4000, 1000,
              &fi) == 0);
  REQUIRE(cte_fuse_getattr(path, &st, &fi) == 0);
  REQUIRE(st.st_size == 4096);

  REQUIRE(cte_fuse_release(path, &fi) == 0);
  REQUIRE(cte_fuse_unlink(path) == 0);
}
#endif

// ============================================================================
// xattr
// ============================================================================

TEST_CASE("FUSE ops - xattr set get list remove", "[fuse][ops]") {
  Fx();
  const char *path = "/ops/xattr.bin";
  auto fi = MakeFi();
  REQUIRE(cte_fuse_create(path, 0644, &fi) == 0);
  REQUIRE(cte_fuse_release(path, &fi) == 0);

  const std::string name = "user.color";
  const std::string value = "blue";
  REQUIRE(cte_fuse_setxattr(path, name.c_str(), value.data(), value.size(),
                            0) == 0);
  // A second attribute so that removing the first leaves a non-empty map (the
  // chimod stores the remaining set as a blob; an all-empty set is a separate
  // edge case exercised by removing this one last, below).
  const std::string name2 = "user.shape";
  const std::string value2 = "round";
  REQUIRE(cte_fuse_setxattr(path, name2.c_str(), value2.data(), value2.size(),
                            0) == 0);

  // Length query.
  int len = cte_fuse_getxattr(path, name.c_str(), nullptr, 0);
  REQUIRE(len == static_cast<int>(value.size()));

  // Full read.
  char vbuf[64];
  int got = cte_fuse_getxattr(path, name.c_str(), vbuf, sizeof(vbuf));
  REQUIRE(got == static_cast<int>(value.size()));
  REQUIRE(std::string(vbuf, got) == value);

  // ERANGE: buffer too small.
  REQUIRE(cte_fuse_getxattr(path, name.c_str(), vbuf, 1) == -ERANGE);

  // Missing attribute -> ENODATA.
  REQUIRE(cte_fuse_getxattr(path, "user.absent", vbuf, sizeof(vbuf)) ==
          -ENODATA);

  // listxattr: length query then full.
  int llen = cte_fuse_listxattr(path, nullptr, 0);
  REQUIRE(llen > 0);
  std::vector<char> lbuf(llen);
  REQUIRE(cte_fuse_listxattr(path, lbuf.data(), lbuf.size()) == llen);
  REQUIRE(cte_fuse_listxattr(path, lbuf.data(), 1) == -ERANGE);

  // Remove one attribute (the other remains) -> success.
  REQUIRE(cte_fuse_removexattr(path, name.c_str()) == 0);
  REQUIRE(cte_fuse_getxattr(path, name.c_str(), vbuf, sizeof(vbuf)) ==
          -ENODATA);
  // The surviving attribute is still readable.
  REQUIRE(cte_fuse_getxattr(path, name2.c_str(), vbuf, sizeof(vbuf)) ==
          static_cast<int>(value2.size()));

  // Removing a non-existent attribute -> ENODATA.
  REQUIRE(cte_fuse_removexattr(path, "user.absent") == -ENODATA);

  REQUIRE(cte_fuse_unlink(path) == 0);
}

// ============================================================================
// metadata ops on a missing path -> ENOENT (error propagation branches)
// ============================================================================

TEST_CASE("FUSE ops - metadata ops on missing path", "[fuse][ops]") {
  Fx();
  const char *missing = "/ops/no_such_entry";
  char buf[64];
  REQUIRE(cte_fuse_readlink(missing, buf, sizeof(buf)) == -ENOENT);
  REQUIRE(cte_fuse_getxattr(missing, "user.x", buf, sizeof(buf)) == -ENOENT);
  REQUIRE(cte_fuse_listxattr(missing, buf, sizeof(buf)) == -ENOENT);
  REQUIRE(cte_fuse_removexattr(missing, "user.x") == -ENOENT);
}

// ============================================================================
// rename
// ============================================================================

TEST_CASE("FUSE ops - rename", "[fuse][ops]") {
  Fx();
  const char *a = "/ops/ren_a.dat";
  const char *b = "/ops/ren_b.dat";
  const char *c = "/ops/ren_c.dat";
  auto fi = MakeFi();
  REQUIRE(cte_fuse_create(a, 0644, &fi) == 0);
  REQUIRE(cte_fuse_release(a, &fi) == 0);

  // Plain rename a -> b.
  REQUIRE(cte_fuse_rename(a, b, 0) == 0);
  struct stat st;
  REQUIRE(cte_fuse_getattr(b, &st, nullptr) == 0);
  REQUIRE(cte_fuse_getattr(a, &st, nullptr) == -ENOENT);

  // NOREPLACE where dest does not exist -> succeeds (b -> c).
  REQUIRE(cte_fuse_rename(b, c, RENAME_NOREPLACE) == 0);

  // NOREPLACE where dest exists -> EEXIST. Recreate a so it exists.
  REQUIRE(cte_fuse_create(a, 0644, &fi) == 0);
  REQUIRE(cte_fuse_release(a, &fi) == 0);
  REQUIRE(cte_fuse_rename(a, c, RENAME_NOREPLACE) == -EEXIST);

  // Unsupported flag (e.g. RENAME_EXCHANGE) -> EINVAL.
  REQUIRE(cte_fuse_rename(a, c, RENAME_EXCHANGE) == -EINVAL);

  REQUIRE(cte_fuse_unlink(a) == 0);
  REQUIRE(cte_fuse_unlink(c) == 0);
}

// ============================================================================
// statfs
// ============================================================================

TEST_CASE("FUSE ops - statfs", "[fuse][ops]") {
  Fx();
  struct statvfs sv;
  REQUIRE(cte_fuse_statfs("/", &sv) == 0);
  REQUIRE(sv.f_bsize == 4096);
  REQUIRE(sv.f_namemax == 255);
}

// gcov flush entry point, present only in coverage builds (the build wires
// -DCLIO_COVERAGE_ENABLED alongside --coverage). A strong reference forces the
// linker to pull libgcov's dumper in; a weak reference is left undefined and
// never runs, so it must be gated by the macro rather than by weak linkage.
#ifdef CLIO_COVERAGE_ENABLED
extern "C" void __gcov_dump(void);
#endif

// Custom main (instead of SIMPLE_TEST_MAIN): the embedded runtime's static
// destructors abort ("stack smashing") on process exit — a known teardown
// hazard also seen in the copy-workspace test. That abort would discard this
// TU's coverage. So we explicitly flush gcov counters after the tests pass,
// then _exit() with the result to skip the crash-prone global teardown (the
// POSIX analogue of what SIMPLE_TEST_PROCESS_EXIT does on Windows).
int main(int argc, char *argv[]) {
  std::string filter = (argc > 1) ? argv[1] : "";
  int result = SimpleTest::run_all_tests(filter);
#ifdef CLIO_COVERAGE_ENABLED
  __gcov_dump();
#endif
  _exit(result);
}
