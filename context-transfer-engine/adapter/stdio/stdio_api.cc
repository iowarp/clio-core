/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 *
 * STDIO interceptor. Tracked paths (those carrying the "clio::" marker) are
 * routed to the context-filesystem chimod via the shared CfsIo core. A clio::
 * stream is represented by an opaque heap token reinterpreted as FILE* (the
 * app only ever hands it back to the intercepted calls); the token maps to a
 * CfsIo descriptor. Everything else falls through to the real libc API.
 */

bool stdio_intercepted = true;

#include "stdio_api.h"

#include <dlfcn.h>
#include <fcntl.h>
#include <limits.h>

#include <cstdio>
#include <cstring>
#include <mutex>
#include <unordered_map>

#include "clio_ctp/util/logging.h"
#include "adapter/cfs/cfs_io.h"

namespace {

/** Opaque per-stream token; its address is the FILE* handed to the app. */
struct CfsStream {
  int fd = -1;
};

/** Maps clio:: FILE* tokens to their CfsIo descriptors. */
class StdioShim {
 public:
  /** Translate an fopen() mode string to open(2) flags. */
  static int ModeToFlags(const char *mode) {
    bool plus = (mode != nullptr && std::strchr(mode, '+') != nullptr);
    char c = (mode != nullptr) ? mode[0] : 'r';
    switch (c) {
      case 'w':
        return (plus ? O_RDWR : O_WRONLY) | O_CREAT | O_TRUNC;
      case 'a':
        return (plus ? O_RDWR : O_WRONLY) | O_CREAT | O_APPEND;
      case 'r':
      default:
        return plus ? O_RDWR : O_RDONLY;
    }
  }

  /** Wrap an already-open CfsIo descriptor in a FILE* token. */
  FILE *Wrap(int fd) {
    auto *s = new CfsStream();
    s->fd = fd;
    FILE *fp = reinterpret_cast<FILE *>(s);
    std::lock_guard<std::mutex> g(mu_);
    streams_[fp] = s;
    return fp;
  }

  /** -1 if fp is not a clio:: stream. */
  int FdOf(FILE *fp) {
    std::lock_guard<std::mutex> g(mu_);
    auto it = streams_.find(fp);
    return (it == streams_.end()) ? -1 : it->second->fd;
  }

  bool IsTracked(FILE *fp) { return FdOf(fp) >= 0; }

  /** Drop the token, returning its fd (or -1). */
  int Release(FILE *fp) {
    std::lock_guard<std::mutex> g(mu_);
    auto it = streams_.find(fp);
    if (it == streams_.end()) {
      return -1;
    }
    int fd = it->second->fd;
    delete it->second;
    streams_.erase(it);
    return fd;
  }

 private:
  std::mutex mu_;
  std::unordered_map<FILE *, CfsStream *> streams_;
};

StdioShim &Shim() {
  static StdioShim shim;
  return shim;
}

}  // namespace

extern "C" {

/**
 * STDIO
 */

FILE *CLIO_CTE_DECL(fopen)(const char *path, const char *mode) {
  auto real_api = CLIO_CTE_STDIO_API;
  if (clio::cae::CfsIo::IsPathTracked(path)) {
    HLOG(kDebug, "Intercepting fopen({}, {})", path, mode);
    int fd = CLIO_CTE_CFS->Open(path, StdioShim::ModeToFlags(mode), 0644);
    if (fd < 0) {
      return nullptr;
    }
    return Shim().Wrap(fd);
  }
  return real_api->fopen(path, mode);
}

FILE *CLIO_CTE_DECL(fopen64)(const char *path, const char *mode) {
  auto real_api = CLIO_CTE_STDIO_API;
  if (clio::cae::CfsIo::IsPathTracked(path)) {
    HLOG(kDebug, "Intercepting fopen64({}, {})", path, mode);
    int fd = CLIO_CTE_CFS->Open(path, StdioShim::ModeToFlags(mode), 0644);
    if (fd < 0) {
      return nullptr;
    }
    return Shim().Wrap(fd);
  }
  return real_api->fopen64(path, mode);
}

FILE *CLIO_CTE_DECL(fdopen)(int fd, const char *mode) {
  auto real_api = CLIO_CTE_STDIO_API;
  if (CLIO_CTE_CFS->IsFdTracked(fd)) {
    HLOG(kDebug, "Intercepting fdopen({}, {})", fd, mode);
    return Shim().Wrap(fd);
  }
  return real_api->fdopen(fd, mode);
}

FILE *CLIO_CTE_DECL(freopen)(const char *path, const char *mode, FILE *stream) {
  auto real_api = CLIO_CTE_STDIO_API;
  if (Shim().IsTracked(stream)) {
    HLOG(kDebug, "Intercepting freopen({}, {})", path, mode);
    int old_fd = Shim().Release(stream);
    if (old_fd >= 0) {
      CLIO_CTE_CFS->Close(old_fd);
    }
    if (!clio::cae::CfsIo::IsPathTracked(path)) {
      return nullptr;  // can't reopen a clio:: stream onto a kernel path
    }
    int fd = CLIO_CTE_CFS->Open(path, StdioShim::ModeToFlags(mode), 0644);
    if (fd < 0) {
      return nullptr;
    }
    return Shim().Wrap(fd);
  }
  return real_api->freopen(path, mode, stream);
}

FILE *CLIO_CTE_DECL(freopen64)(const char *path, const char *mode,
                               FILE *stream) {
  return CLIO_CTE_DECL(freopen)(path, mode, stream);
}

int CLIO_CTE_DECL(fflush)(FILE *fp) {
  auto real_api = CLIO_CTE_STDIO_API;
  int fd = Shim().FdOf(fp);
  if (fd >= 0) {
    HLOG(kDebug, "Intercepting fflush");
    return CLIO_CTE_CFS->Sync(fd);
  }
  return real_api->fflush(fp);
}

int CLIO_CTE_DECL(fclose)(FILE *fp) {
  auto real_api = CLIO_CTE_STDIO_API;
  int fd = Shim().Release(fp);
  if (fd >= 0) {
    HLOG(kDebug, "Intercepting fclose({})", (void *)fp);
    return CLIO_CTE_CFS->Close(fd);
  }
  return real_api->fclose(fp);
}

size_t CLIO_CTE_DECL(fwrite)(const void *ptr, size_t size, size_t nmemb,
                             FILE *fp) {
  auto real_api = CLIO_CTE_STDIO_API;
  int fd = Shim().FdOf(fp);
  if (fd >= 0) {
    HLOG(kDebug, "Intercepting fwrite size: {} nmemb: {}", size, nmemb);
    if (size == 0) {
      return 0;
    }
    ssize_t ret = CLIO_CTE_CFS->Write(fd, ptr, size * nmemb);
    return (ret > 0) ? static_cast<size_t>(ret) / size : 0;
  }
  return real_api->fwrite(ptr, size, nmemb, fp);
}

int CLIO_CTE_DECL(fputc)(int c, FILE *fp) {
  auto real_api = CLIO_CTE_STDIO_API;
  int fd = Shim().FdOf(fp);
  if (fd >= 0) {
    HLOG(kDebug, "Intercepting fputc({})", c);
    unsigned char ch = static_cast<unsigned char>(c);
    return (CLIO_CTE_CFS->Write(fd, &ch, 1) == 1) ? c : EOF;
  }
  return real_api->fputc(c, fp);
}

int CLIO_CTE_DECL(fgetpos)(FILE *fp, fpos_t *pos) {
  auto real_api = CLIO_CTE_STDIO_API;
  int fd = Shim().FdOf(fp);
  if (fd >= 0 && pos) {
    HLOG(kDebug, "Intercepting fgetpos");
    pos->__pos = CLIO_CTE_CFS->Tell(fd);
    return 0;
  }
  return real_api->fgetpos(fp, pos);
}

int CLIO_CTE_DECL(fgetpos64)(FILE *fp, fpos64_t *pos) {
  auto real_api = CLIO_CTE_STDIO_API;
  int fd = Shim().FdOf(fp);
  if (fd >= 0 && pos) {
    HLOG(kDebug, "Intercepting fgetpos64");
    pos->__pos = CLIO_CTE_CFS->Tell(fd);
    return 0;
  }
  return real_api->fgetpos64(fp, pos);
}

int CLIO_CTE_DECL(putc)(int c, FILE *fp) { return CLIO_CTE_DECL(fputc)(c, fp); }

int CLIO_CTE_DECL(putw)(int w, FILE *fp) {
  auto real_api = CLIO_CTE_STDIO_API;
  int fd = Shim().FdOf(fp);
  if (fd >= 0) {
    HLOG(kDebug, "Intercepting putw");
    return (CLIO_CTE_CFS->Write(fd, &w, sizeof(w)) ==
            static_cast<ssize_t>(sizeof(w)))
               ? 0
               : EOF;
  }
  return real_api->putw(w, fp);
}

int CLIO_CTE_DECL(fputs)(const char *s, FILE *stream) {
  auto real_api = CLIO_CTE_STDIO_API;
  int fd = Shim().FdOf(stream);
  if (fd >= 0) {
    HLOG(kDebug, "Intercepting fputs");
    ssize_t ret = CLIO_CTE_CFS->Write(fd, s, std::strlen(s));
    return (ret >= 0) ? static_cast<int>(ret) : EOF;
  }
  return real_api->fputs(s, stream);
}

size_t CLIO_CTE_DECL(fread)(void *ptr, size_t size, size_t nmemb,
                            FILE *stream) {
  auto real_api = CLIO_CTE_STDIO_API;
  int fd = Shim().FdOf(stream);
  if (fd >= 0) {
    HLOG(kDebug, "Intercepting fread size: {} nmemb: {}", size, nmemb);
    if (size == 0) {
      return 0;
    }
    ssize_t ret = CLIO_CTE_CFS->Read(fd, ptr, size * nmemb);
    return (ret > 0) ? static_cast<size_t>(ret) / size : 0;
  }
  return real_api->fread(ptr, size, nmemb, stream);
}

int CLIO_CTE_DECL(fgetc)(FILE *stream) {
  auto real_api = CLIO_CTE_STDIO_API;
  int fd = Shim().FdOf(stream);
  if (fd >= 0) {
    HLOG(kDebug, "Intercepting fgetc");
    unsigned char value = 0;
    return (CLIO_CTE_CFS->Read(fd, &value, 1) == 1) ? value : EOF;
  }
  return real_api->fgetc(stream);
}

int CLIO_CTE_DECL(getc)(FILE *stream) { return CLIO_CTE_DECL(fgetc)(stream); }

int CLIO_CTE_DECL(getw)(FILE *stream) {
  auto real_api = CLIO_CTE_STDIO_API;
  int fd = Shim().FdOf(stream);
  if (fd >= 0) {
    HLOG(kDebug, "Intercepting getw");
    int value = 0;
    return (CLIO_CTE_CFS->Read(fd, &value, sizeof(int)) ==
            static_cast<ssize_t>(sizeof(int)))
               ? value
               : EOF;
  }
  return real_api->getw(stream);
}

char *CLIO_CTE_DECL(fgets)(char *s, int size, FILE *stream) {
  auto real_api = CLIO_CTE_STDIO_API;
  int fd = Shim().FdOf(stream);
  if (fd >= 0) {
    HLOG(kDebug, "Intercepting fgets");
    if (size <= 0) {
      return nullptr;
    }
    size_t want = static_cast<size_t>(size - 1);
    ssize_t got = CLIO_CTE_CFS->Read(fd, s, want);
    if (got <= 0) {
      return nullptr;
    }
    // Stop at the first newline (fgets semantics), keep it, NUL-terminate.
    size_t end = static_cast<size_t>(got);
    for (size_t i = 0; i < static_cast<size_t>(got); ++i) {
      if (s[i] == '\n') {
        end = i + 1;
        break;
      }
    }
    // Rewind the stream to just past the consumed bytes.
    if (end < static_cast<size_t>(got)) {
      CLIO_CTE_CFS->Seek(fd, -(static_cast<off_t>(got) - static_cast<off_t>(end)),
                         SEEK_CUR);
    }
    s[end] = '\0';
    return s;
  }
  return real_api->fgets(s, size, stream);
}

void CLIO_CTE_DECL(rewind)(FILE *stream) {
  auto real_api = CLIO_CTE_STDIO_API;
  int fd = Shim().FdOf(stream);
  if (fd >= 0) {
    HLOG(kDebug, "Intercepting rewind");
    CLIO_CTE_CFS->Seek(fd, 0, SEEK_SET);
    return;
  }
  real_api->rewind(stream);
}

int CLIO_CTE_DECL(fseek)(FILE *stream, long offset, int whence) {
  auto real_api = CLIO_CTE_STDIO_API;
  int fd = Shim().FdOf(stream);
  if (fd >= 0) {
    HLOG(kDebug, "Intercepting fseek offset: {} whence: {}", offset, whence);
    return (CLIO_CTE_CFS->Seek(fd, offset, whence) >= 0) ? 0 : -1;
  }
  return real_api->fseek(stream, offset, whence);
}

int CLIO_CTE_DECL(fseeko)(FILE *stream, off_t offset, int whence) {
  auto real_api = CLIO_CTE_STDIO_API;
  int fd = Shim().FdOf(stream);
  if (fd >= 0) {
    HLOG(kDebug, "Intercepting fseeko offset: {} whence: {}", offset, whence);
    return (CLIO_CTE_CFS->Seek(fd, offset, whence) >= 0) ? 0 : -1;
  }
  return real_api->fseeko(stream, offset, whence);
}

int CLIO_CTE_DECL(fseeko64)(FILE *stream, off64_t offset, int whence) {
  auto real_api = CLIO_CTE_STDIO_API;
  int fd = Shim().FdOf(stream);
  if (fd >= 0) {
    HLOG(kDebug, "Intercepting fseeko64 offset: {} whence: {}", offset, whence);
    return (CLIO_CTE_CFS->Seek(fd, static_cast<off_t>(offset), whence) >= 0)
               ? 0
               : -1;
  }
  return real_api->fseeko64(stream, offset, whence);
}

int CLIO_CTE_DECL(fsetpos)(FILE *stream, const fpos_t *pos) {
  auto real_api = CLIO_CTE_STDIO_API;
  int fd = Shim().FdOf(stream);
  if (fd >= 0 && pos) {
    HLOG(kDebug, "Intercepting fsetpos");
    return (CLIO_CTE_CFS->Seek(fd, pos->__pos, SEEK_SET) >= 0) ? 0 : -1;
  }
  return real_api->fsetpos(stream, pos);
}

int CLIO_CTE_DECL(fsetpos64)(FILE *stream, const fpos64_t *pos) {
  auto real_api = CLIO_CTE_STDIO_API;
  int fd = Shim().FdOf(stream);
  if (fd >= 0 && pos) {
    HLOG(kDebug, "Intercepting fsetpos64");
    return (CLIO_CTE_CFS->Seek(fd, pos->__pos, SEEK_SET) >= 0) ? 0 : -1;
  }
  return real_api->fsetpos64(stream, pos);
}

long int CLIO_CTE_DECL(ftell)(FILE *fp) {
  auto real_api = CLIO_CTE_STDIO_API;
  int fd = Shim().FdOf(fp);
  if (fd >= 0) {
    HLOG(kDebug, "Intercepting ftell");
    return static_cast<long int>(CLIO_CTE_CFS->Tell(fd));
  }
  return real_api->ftell(fp);
}

// fileno() must report the underlying CTE descriptor for clio:: streams (our
// FILE* token is not a real libc FILE, so the real fileno would misbehave).
int CLIO_CTE_DECL(fileno)(FILE *stream) {
  int fd = Shim().FdOf(stream);
  if (fd >= 0) {
    return fd;
  }
  static int (*real_fileno)(FILE *) =
      reinterpret_cast<int (*)(FILE *)>(dlsym(RTLD_NEXT, "fileno"));
  return real_fileno ? real_fileno(stream) : -1;
}

}  // extern C
