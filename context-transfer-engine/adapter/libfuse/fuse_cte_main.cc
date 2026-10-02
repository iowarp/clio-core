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

// ============================================================================
// FUSE process entry / mount glue.
//
// This file holds nothing but the program entry point and the platform-
// specific machinery to bind the FUSE session to a mountpoint:
//   * the normal fuse_main() mount-and-serve path (Linux bare-metal, macOS
//     macFUSE, Windows WinFsp), and
//   * the Apptainer `--fusemount` fd-injection path (Linux only), which drives
//     the FUSE protocol on a pre-opened /dev/fuse fd via custom_io.
//
// None of it is unit-testable — it either blocks in the FUSE event loop or
// needs a real kernel mount / Apptainer user namespace — so it is deliberately
// separated from fuse_cte.cc (which holds the operation callbacks) and excluded
// from coverage. The callbacks it dispatches to are exercised directly by
// test_fuse_ops, and the end-to-end mount path by CI/fuse_mount_smoke.sh.
// ============================================================================

#include "fuse_cte.h"  // cte_fuse_ops (the operation table) + FUSE headers

#include <cerrno>   // errno
#include <climits>
#include <cstdio>   // snprintf, fprintf
#include <cstdlib>  // atoi, getenv
#include <cstring>  // strncmp, strerror
#include <string>
#include <vector>
#include <fcntl.h>

#ifndef _WIN32
#include <unistd.h>  // getuid, getgid, read
#if defined(__linux__)
#include <sys/prctl.h>  // PR_SET_PTRACER
#endif
#ifndef __APPLE__
#include <fuse3/fuse_lowlevel.h>  // fuse_session_custom_io, struct fuse_custom_io
#include <dlfcn.h>                // dlsym (resolve fuse_session_custom_io at runtime)
#include <sys/mount.h>            // mount syscall
#include <sys/uio.h>              // struct iovec, writev
#endif  // __APPLE__
#endif  // _WIN32

#ifndef _WIN32
// custom_io callbacks: when apptainer hands us an already-mounted
// /dev/fuse fd, we need to drive the FUSE protocol on that fd directly
// instead of having libfuse mount its own. Plain read/writev syscalls
// suffice; splice is optional.
// The FUSE_VERSION arm matches the guard in main(): below libfuse 3.14 the
// custom-io path is compiled out entirely, and defining these here anyway
// would leave two unused statics behind (-Wunused-function, fatal under
// -Werror).
#if !defined(__APPLE__) && FUSE_VERSION >= FUSE_MAKE_VERSION(3, 14)
static ssize_t cte_custom_writev(int fd, struct iovec *iov, int count,
                                 void * /*userdata*/) {
  return writev(fd, iov, count);
}
static ssize_t cte_custom_read(int fd, void *buf, size_t buf_len,
                               void * /*userdata*/) {
  return read(fd, buf, buf_len);
}
#endif  // !__APPLE__ && FUSE_VERSION >= 3.14
#endif  // _WIN32

/**
 * Whether the kernel should enforce permission bits (FUSE
 * default_permissions): mode, owner and group are checked against every
 * access like on ext4. CLIO_FUSE_PERMISSIONS=0 turns it off.
 * @return true unless disabled
 */
static bool EnforcePermissions() {
  const char *e = std::getenv("CLIO_FUSE_PERMISSIONS");
  return e == nullptr || std::strcmp(e, "0") != 0;
}

/**
 * argv for fuse_main, plus "-o default_permissions" when enforced.
 * @param argc argument count
 * @param argv arguments
 * @return null-terminated argument vector (size() - 1 arguments)
 */
static std::vector<char *> MountArgv(int argc, char *argv[]) {
  static char opt_flag[] = "-o";
  static char opt_perm[] = "default_permissions";
  std::vector<char *> v(argv, argv + argc);
  if (EnforcePermissions()) {
    v.push_back(opt_flag);
    v.push_back(opt_perm);
  }
  v.push_back(nullptr);
  return v;
}

/**
 * Default the runtime-connect wait (CLIO_WAIT_SERVER) to 10 minutes for the
 * mount unless the caller set it. A mount started with the runtime after a
 * reboot must outlast the runtime's recovery (WAL replay), which can take
 * longer than the generic 30 s client default; giving up leaves a dead mount.
 */
static void DefaultMountServerWait() {
  if (std::getenv("CLIO_WAIT_SERVER") == nullptr) {
#ifdef _WIN32
    _putenv_s("CLIO_WAIT_SERVER", "600");
#else
    setenv("CLIO_WAIT_SERVER", "600", 0);
#endif
  }
}

/**
 * Take the atime mount options (-o noatime / strictatime / relatime and
 * their negations) out of argv and carry the choice to the read path as
 * CLIO_FUSE_ATIME. FUSE leaves atime to the filesystem, so the kernel flag
 * would change nothing -- and libfuse rejects relatime/strictatime as
 * unknown, failing the mount. An explicit CLIO_FUSE_ATIME wins. The option
 * strings are rewritten in place (they only shrink); one left empty becomes
 * "rw".
 * @param argc argument count
 * @param argv arguments (modified)
 */
static void AtimeFromMountOptions(int argc, char *argv[]) {
  const char *mode = nullptr;
  for (int i = 1; i < argc; ++i) {
    char *opts = nullptr;
    if (std::strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
      opts = argv[++i];
    } else if (std::strncmp(argv[i], "-o", 2) == 0) {
      opts = argv[i] + 2;
    } else {
      continue;
    }
    std::string kept;
    std::string all(opts);
    size_t pos = 0;
    while (pos <= all.size()) {
      size_t end = all.find(',', pos);
      if (end == std::string::npos) end = all.size();
      const std::string tok = all.substr(pos, end - pos);
      pos = end + 1;
      if (tok == "noatime") { mode = "0"; continue; }
      if (tok == "strictatime") { mode = "strict"; continue; }
      if (tok == "relatime" || tok == "atime") { mode = "relatime"; continue; }
      if (tok == "norelatime" || tok == "nostrictatime" ||
          tok == "nodiratime" || tok == "diratime") {
        continue;
      }
      if (tok.empty()) continue;
      if (!kept.empty()) kept += ',';
      kept += tok;
    }
    if (kept.empty()) kept = "rw";
    std::memcpy(opts, kept.c_str(), kept.size() + 1);
  }
  if (mode == nullptr || std::getenv("CLIO_FUSE_ATIME") != nullptr) return;
#ifdef _WIN32
  _putenv_s("CLIO_FUSE_ATIME", mode);
#else
  setenv("CLIO_FUSE_ATIME", mode, 1);
#endif
}

/**
 * Let a debugger attach to this mount daemon when CLIO_ALLOW_PTRACE=1, as
 * clio_run does: hosts with kernel.yama.ptrace_scope=1 only allow tracing
 * descendants, and a hung FUSE request (one the runtime is not working on)
 * can only be located from this process's thread stacks. Opt-in only.
 */
static void MaybeAllowPtrace() {
#if defined(__linux__) && defined(PR_SET_PTRACER)
  const char *e = std::getenv("CLIO_ALLOW_PTRACE");
  if (e != nullptr && e[0] == '1') {
    prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0);
  }
#endif
}

int main(int argc, char *argv[]) {
  MaybeAllowPtrace();
  DefaultMountServerWait();
  AtimeFromMountOptions(argc, argv);
#if defined(_WIN32) || defined(__APPLE__)
  // Native Windows (WinFsp) and macOS (macFUSE): no Apptainer-style
  // /dev/fuse fd injection. fuse_main() parses argv (on Windows the
  // mountpoint is a drive letter like "Z:" or a host directory) and drives
  // the FUSE protocol. The callbacks and the entire CTE data path below them
  // are identical to the Linux build.
  cte_fuse_mark_session_live();
  return fuse_main(argc, argv, &cte_fuse_ops, nullptr);
#else
  // Apptainer's --fusemount opens /dev/fuse on the host, performs the
  // kernel mount, and passes the fd to the FUSE binary as the last
  // argv ("/dev/fd/<N>"). libfuse 3's high-level argv parser doesn't
  // recognize this token (it's a libfuse2-era convention) and aborts
  // with "fuse: invalid argument '/dev/fd/N'". apptainer also strips
  // the mountpoint from the argv since it has already mounted, so
  // there's no fuse_main path that works.
  //
  // libfuse 3.14 added fuse_session_custom_io() which lets us drive
  // the protocol on an existing fd. When we detect /dev/fd/<N> as the
  // last argv, take the custom-io path; otherwise fall back to the
  // normal fuse_main mount-and-serve flow (host bare-metal use).
  int prefd = -1;
  int new_argc = argc;
  if (argc >= 2 && std::strncmp(argv[argc - 1], "/dev/fd/", 8) == 0) {
    prefd = std::atoi(argv[argc - 1] + 8);
    if (prefd < 0) {
      prefd = -1;
    } else {
      new_argc = argc - 1;  // drop /dev/fd/N from argv
    }
  }

  if (prefd == -1) {
    cte_fuse_mark_session_live();
    std::vector<char *> mount_argv = MountArgv(argc, argv);
    return fuse_main(static_cast<int>(mount_argv.size()) - 1,
                     mount_argv.data(), &cte_fuse_ops, nullptr);
  }

#if FUSE_VERSION < FUSE_MAKE_VERSION(3, 14)
  // Built against libfuse older than 3.14, where `struct fuse_custom_io` is
  // not declared at all. The dlsym dance below handles a RUNTIME libfuse that
  // lacks fuse_session_custom_io, but it cannot help here: the struct is an
  // incomplete type, so the initializer below is a hard compile error rather
  // than a link error. The manylinux images ship libfuse 3.10.2, which is
  // exactly this case -- enabling the FUSE adapter for Linux wheels without
  // this guard fails the wheel build outright.
  //
  // Only the Apptainer fd-injection path is lost. Normal mounting still works
  // through the fuse_main() path above, which is what every non-Apptainer
  // caller (including the wheel's documented `clio_cte_fuse <mnt>` usage)
  // takes. The message matches the runtime-missing-symbol case below.
  (void)new_argc;
  std::fprintf(stderr,
               "clio_cte_fuse: this binary was built against libfuse %d.%d; "
               "--fusemount (/dev/fd/N) mode needs 3.14+ headers at build "
               "time.\n",
               FUSE_MAJOR_VERSION, FUSE_MINOR_VERSION);
  return 1;
#else

  // Apptainer 1.2.5 (unprivileged, no starter-suid) doesn't actually
  // call mount(2) when --fusemount kicks in for a user-supplied
  // binary — it only opens /dev/fuse and hands us the fd. We have to
  // bind that fd to a mountpoint ourselves (this requires CAP_SYS_ADMIN
  // in the current user_ns, which we have via apptainer's userns
  // mapping). The mountpoint isn't communicated to us through argv or
  // env by apptainer, so the caller MUST set CLIO_CTE_FUSE_MOUNTPOINT
  // before exec'ing the FUSE binary via --fusemount.
  const char *mountpoint = std::getenv("CLIO_CTE_FUSE_MOUNTPOINT");
  if (mountpoint == nullptr) {
    std::fprintf(stderr,
                 "clio_cte_fuse: got pre-opened fd %d but "
                 "CLIO_CTE_FUSE_MOUNTPOINT env var is not set\n", prefd);
    return 1;
  }
  char mount_opts[256];
  std::snprintf(mount_opts, sizeof(mount_opts),
                "fd=%d,rootmode=040000,user_id=%u,group_id=%u",
                prefd, (unsigned)getuid(), (unsigned)getgid());
  if (mount("nodev", mountpoint, "fuse", MS_NODEV | MS_NOSUID,
            mount_opts) != 0) {
    std::fprintf(stderr, "clio_cte_fuse: mount(\"%s\", fuse) failed: %s\n",
                 mountpoint, std::strerror(errno));
    return 1;
  }
  std::fprintf(stderr, "clio_cte_fuse: mounted FUSE at %s with fd=%d\n",
               mountpoint, prefd);

  struct fuse_args args = FUSE_ARGS_INIT(new_argc, argv);
  if (EnforcePermissions()) fuse_opt_add_arg(&args, "-odefault_permissions");
  struct fuse *fuse =
      (cte_fuse_mark_session_live(),
       fuse_new(&args, &cte_fuse_ops, sizeof(cte_fuse_ops), nullptr));
  if (!fuse) {
    fuse_opt_free_args(&args);
    return 1;
  }

  struct fuse_session *se = fuse_get_session(fuse);
  static const struct fuse_custom_io custom_io = {
      .writev = cte_custom_writev,
      .read = cte_custom_read,
      .splice_receive = nullptr,
      .splice_send = nullptr,
  };

  // Resolve fuse_session_custom_io at runtime via dlsym so this binary
  // links against system libfuse 3.10.5 (which has setuid
  // /usr/bin/fusermount3) without needing the 3.14+ symbol present at
  // link time. The symbol IS present at runtime when the caller uses
  // a newer libfuse runtime (e.g. apptainer --fusemount).
  using FuseSessionCustomIoFn = int (*)(struct fuse_session *,
                                        const struct fuse_custom_io *, int);
  auto fuse_session_custom_io_dyn = reinterpret_cast<FuseSessionCustomIoFn>(
      dlsym(RTLD_DEFAULT, "fuse_session_custom_io"));
  if (!fuse_session_custom_io_dyn) {
    std::fprintf(stderr,
                 "clio_cte_fuse: fuse_session_custom_io not available in "
                 "the loaded libfuse (need 3.14+); --fusemount mode "
                 "requires a newer libfuse runtime.\n");
    fuse_destroy(fuse);
    fuse_opt_free_args(&args);
    return 1;
  }
  if (fuse_session_custom_io_dyn(se, &custom_io, prefd) != 0) {
    fuse_destroy(fuse);
    fuse_opt_free_args(&args);
    return 1;
  }

  // Multi-threaded FUSE loop. Single-threaded `fuse_loop()` serializes
  // every fs op through one handler -- each call blocks on
  // AsyncPutBlob.Wait() before the next can run -- so 24 concurrent
  // ranks per node effectively run sequentially and a workload that
  // should finish in seconds hangs past the test timeout. fuse_loop_mt
  // spawns worker threads that handle ops in parallel; AsyncPutBlobs
  // from different ranks now overlap.
  //
  // The libfuse 3.16 headers we compile against rewrite `fuse_loop_mt`
  // to `fuse_loop_mt_32` (via macro), but the runtime libfuse 3.10.5
  // (system) only exports `fuse_loop_mt@@FUSE_3.2` (the unversioned
  // default), not `fuse_loop_mt_32`. Resolve the unversioned symbol
  // via dlsym -- same pattern this file already uses for
  // fuse_session_custom_io -- so the binary works against any
  // libfuse runtime >= 3.2.
  using FuseLoopMtFn = int (*)(struct fuse *, struct fuse_loop_config *);
  auto fuse_loop_mt_dyn = reinterpret_cast<FuseLoopMtFn>(
      dlsym(RTLD_DEFAULT, "fuse_loop_mt"));

  int ret;
  if (fuse_loop_mt_dyn != nullptr) {
    struct fuse_loop_config loop_cfg = {0};
    loop_cfg.clone_fd = 0;            // share /dev/fuse fd across workers
    loop_cfg.max_idle_threads = 32;   // headroom for 24 ranks/node
    ret = fuse_loop_mt_dyn(fuse, &loop_cfg);
  } else {
    std::fprintf(stderr,
                 "clio_cte_fuse: fuse_loop_mt not in runtime libfuse, "
                 "falling back to single-threaded loop.\n");
    ret = fuse_loop(fuse);
  }
  fuse_destroy(fuse);
  fuse_opt_free_args(&args);
  return ret;
#endif  // FUSE_VERSION < 3.14
#endif  // _WIN32 || __APPLE__
}
