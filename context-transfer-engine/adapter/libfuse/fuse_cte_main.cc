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
#include <fcntl.h>

#ifndef _WIN32
#include <unistd.h>  // getuid, getgid, read
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
#ifndef __APPLE__
static ssize_t cte_custom_writev(int fd, struct iovec *iov, int count,
                                 void * /*userdata*/) {
  return writev(fd, iov, count);
}
static ssize_t cte_custom_read(int fd, void *buf, size_t buf_len,
                               void * /*userdata*/) {
  return read(fd, buf, buf_len);
}
#endif  // __APPLE__
#endif  // _WIN32

int main(int argc, char *argv[]) {
#if defined(_WIN32) || defined(__APPLE__)
  // Native Windows (WinFsp) and macOS (macFUSE): no Apptainer-style
  // /dev/fuse fd injection. fuse_main() parses argv (on Windows the
  // mountpoint is a drive letter like "Z:" or a host directory) and drives
  // the FUSE protocol. The callbacks and the entire CTE data path below them
  // are identical to the Linux build.
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
    return fuse_main(argc, argv, &cte_fuse_ops, nullptr);
  }

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
  struct fuse *fuse =
      fuse_new(&args, &cte_fuse_ops, sizeof(cte_fuse_ops), nullptr);
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
#endif  // _WIN32 || __APPLE__
}
