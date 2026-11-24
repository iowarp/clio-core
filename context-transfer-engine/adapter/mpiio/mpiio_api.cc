/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 *
 * MPI-IO interceptor. Tracked paths (those carrying the "clio::" marker) are
 * routed to the context-filesystem chimod via the filesystem client; a clio::
 * MPI_File is an opaque heap token mapped to a client descriptor. Everything
 * else falls through to the real MPI-IO API.
 *
 * Single-rank (MPI_COMM_SELF) byte semantics: collective (_all), shared and
 * ordered variants behave like their independent counterparts here — the
 * chimod owns placement, so there is no separate collective-aggregation step
 * at this layer. Asynchronous (i*) ops complete synchronously and hand back
 * MPI_REQUEST_NULL.
 *
 * Errors: the descriptor layer defers writes (write-behind), so a failed
 * write usually surfaces at MPI_File_sync or MPI_File_close, as it would at
 * fsync/close. Every descriptor-layer failure is mapped from errno to an MPI
 * error class and returned (#1187).
 *
 * File views and derived datatypes: MPI_File_set_view flattens the filetype
 * into byte blocks, and the explicit-offset ops (*_at, *_at_all, i*_at) map
 * the data stream through it, so the non-contiguous views parallel HDF5 sets
 * for collective I/O read and write the right bytes. Derived memory
 * datatypes are scattered/gathered the same way. A type built with a
 * constructor this layer does not decode (subarray, darray, ...) is refused
 * with an error rather than transferred as if contiguous.
 *
 * Known limits of this layer (not errors): shared and ordered file pointers
 * are per-process; the file-pointer ops (read/write, _all, _shared,
 * _ordered) require the default view and contiguous memory datatypes.
 *
 * Preload order: when interposed with LD_PRELOAD, this library must come
 * before libmpi, or libmpi's own MPI_File_* win symbol lookup.
 */

bool mpiio_intercepted = true;

#include "mpiio_api.h"

#include <clio_cte/filesystem/filesystem_client.h>
#include <fcntl.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "clio_ctp/util/logging.h"
#include "clio_ctp/util/singleton.h"

namespace {

/** Byte blocks (offset, length) of a datatype, in type-map order. */
using Blocks = std::vector<std::pair<MPI_Aint, MPI_Aint>>;

/** A file view: displacement, etype size and one flattened filetype tile. */
struct FileView {
  MPI_Offset disp = 0;
  int etype_size = 1;
  Blocks tile{{0, 1}};       // byte blocks of one filetype instance
  MPI_Aint tile_size = 1;    // bytes of data per tile
  MPI_Aint tile_extent = 1;  // file bytes spanned per tile
  /** True for the default view: a byte stream starting at offset 0. */
  bool IsDefault() const {
    return disp == 0 && tile.size() == 1 && tile[0].first == 0 &&
           tile_size == tile_extent;
  }
  /** True when the view is one contiguous stream (possibly displaced). */
  bool IsContiguous() const {
    return tile.size() == 1 && tile[0].first == 0 && tile_size == tile_extent;
  }
};

struct MpiToken {
  int fd = -1;
  int amode = 0;
  int atomic = 0;
  FileView view;
};

/**
 * Append the byte blocks of one instance of datatype t placed at base;
 * adjacent blocks are merged.
 * @return false for a constructor this layer does not decode
 */
bool Flatten(MPI_Datatype t, MPI_Aint base, Blocks* out) {
  int ni, na, nd, comb;
  MPI_Type_get_envelope(t, &ni, &na, &nd, &comb);
  auto push = [out](MPI_Aint off, MPI_Aint len) {
    if (len <= 0) return;
    if (!out->empty() && out->back().first + out->back().second == off) {
      out->back().second += len;
    } else {
      out->emplace_back(off, len);
    }
  };
  if (comb == MPI_COMBINER_NAMED) {
    int sz = 0;
    MPI_Type_size(t, &sz);
    push(base, sz);
    return true;
  }
  std::vector<int> iv(ni);
  std::vector<MPI_Aint> av(na);
  std::vector<MPI_Datatype> dv(nd);
  MPI_Type_get_contents(t, ni, na, nd, iv.data(), av.data(), dv.data());
  auto extent = [](MPI_Datatype d) {
    MPI_Aint lb, ext;
    MPI_Type_get_extent(d, &lb, &ext);
    return ext;
  };
  // n consecutive instances of child d starting at off
  auto run = [&](MPI_Datatype d, MPI_Aint off, MPI_Aint n) {
    int ci, ca, cd, cc;
    MPI_Type_get_envelope(d, &ci, &ca, &cd, &cc);
    if (cc == MPI_COMBINER_NAMED) {
      int sz = 0;
      MPI_Type_size(d, &sz);
      push(off, n * sz);
      return true;
    }
    MPI_Aint e = extent(d);
    for (MPI_Aint j = 0; j < n; ++j) {
      if (!Flatten(d, off + j * e, out)) return false;
    }
    return true;
  };
  bool ok = true;
  switch (comb) {
    case MPI_COMBINER_DUP:
    case MPI_COMBINER_RESIZED:  // resizing moves bounds, not data
      ok = Flatten(dv[0], base, out);
      break;
    case MPI_COMBINER_CONTIGUOUS:
      ok = run(dv[0], base, iv[0]);
      break;
    case MPI_COMBINER_VECTOR:
      for (int i = 0; ok && i < iv[0]; ++i) {
        ok = run(dv[0], base + MPI_Aint(i) * iv[2] * extent(dv[0]), iv[1]);
      }
      break;
    case MPI_COMBINER_HVECTOR:
      for (int i = 0; ok && i < iv[0]; ++i) {
        ok = run(dv[0], base + MPI_Aint(i) * av[0], iv[1]);
      }
      break;
    case MPI_COMBINER_INDEXED:
      for (int i = 0; ok && i < iv[0]; ++i) {
        ok = run(dv[0], base + MPI_Aint(iv[1 + iv[0] + i]) * extent(dv[0]),
                 iv[1 + i]);
      }
      break;
    case MPI_COMBINER_HINDEXED:
      for (int i = 0; ok && i < iv[0]; ++i) {
        ok = run(dv[0], base + av[i], iv[1 + i]);
      }
      break;
    case MPI_COMBINER_INDEXED_BLOCK:
      for (int i = 0; ok && i < iv[0]; ++i) {
        ok = run(dv[0], base + MPI_Aint(iv[2 + i]) * extent(dv[0]), iv[1]);
      }
      break;
    case MPI_COMBINER_HINDEXED_BLOCK:
      for (int i = 0; ok && i < iv[0]; ++i) {
        ok = run(dv[0], base + av[i], iv[1]);
      }
      break;
    case MPI_COMBINER_STRUCT:
      for (int i = 0; ok && i < iv[0]; ++i) {
        ok = run(dv[i], base + av[i], iv[1 + i]);
      }
      break;
    default:
      ok = false;
  }
  for (auto& d : dv) {
    int a, b, c, k;
    MPI_Type_get_envelope(d, &a, &b, &c, &k);
    if (k != MPI_COMBINER_NAMED) MPI_Type_free(&d);
  }
  return ok;
}

/**
 * Flatten count instances of a memory datatype.
 * @return false for a constructor this layer does not decode
 */
bool FlattenMemory(int count, MPI_Datatype t, Blocks* out) {
  MPI_Aint lb, ext;
  MPI_Type_get_extent(t, &lb, &ext);
  for (int i = 0; i < count; ++i) {
    if (!Flatten(t, MPI_Aint(i) * ext, out)) return false;
  }
  return true;
}

/** True when a datatype's data is one run of bytes starting at offset 0. */
bool IsContiguous(MPI_Datatype t) {
  int ni, na, nd, comb;
  MPI_Type_get_envelope(t, &ni, &na, &nd, &comb);
  if (comb == MPI_COMBINER_NAMED) return true;
  Blocks b;
  if (!Flatten(t, 0, &b)) return false;
  MPI_Aint lb, ext;
  MPI_Type_get_extent(t, &lb, &ext);
  return b.size() == 1 && b[0].first == 0 && b[0].second == ext;
}

/** Maps clio:: MPI_File tokens to their filesystem-client descriptors. */
class MpiioShim {
 public:
  static int AmodeToFlags(int amode) {
    int flags = 0;
    if (amode & MPI_MODE_RDWR) {
      flags |= O_RDWR;
    } else if (amode & MPI_MODE_WRONLY) {
      flags |= O_WRONLY;
    } else {
      flags |= O_RDONLY;
    }
    if (amode & MPI_MODE_CREATE) flags |= O_CREAT;
    if (amode & MPI_MODE_APPEND) flags |= O_APPEND;
    if (amode & MPI_MODE_EXCL) flags |= O_EXCL;
    return flags;
  }

  MPI_File Wrap(int fd, int amode) {
    auto *t = new MpiToken();
    t->fd = fd;
    t->amode = amode;
    MPI_File fh = reinterpret_cast<MPI_File>(t);
    std::lock_guard<std::mutex> g(mu_);
    files_[fh] = t;
    return fh;
  }

  int FdOf(MPI_File fh) {
    std::lock_guard<std::mutex> g(mu_);
    auto it = files_.find(fh);
    return (it == files_.end()) ? -1 : it->second->fd;
  }

  bool IsTracked(MPI_File fh) { return FdOf(fh) >= 0; }

  /** The token behind a clio:: handle, or nullptr for any other handle. */
  MpiToken* TokenOf(MPI_File fh) {
    std::lock_guard<std::mutex> g(mu_);
    auto it = files_.find(fh);
    return (it == files_.end()) ? nullptr : it->second;
  }

  int Release(MPI_File fh) {
    std::lock_guard<std::mutex> g(mu_);
    auto it = files_.find(fh);
    if (it == files_.end()) {
      return -1;
    }
    int fd = it->second->fd;
    delete it->second;
    files_.erase(it);
    return fd;
  }

 private:
  std::mutex mu_;
  std::unordered_map<MPI_File, MpiToken *> files_;
};

MpiioShim &Shim() {
  static MpiioShim shim;
  return shim;
}

/** Byte count for (count, datatype). */
size_t Bytes(int count, MPI_Datatype datatype) {
  int tsize = 0;
  MPI_Type_size(datatype, &tsize);
  if (tsize <= 0) {
    tsize = 1;
  }
  return static_cast<size_t>(count) * static_cast<size_t>(tsize);
}

/**
 * Map a descriptor-layer errno to the closest MPI error class.
 * @param err errno from a failed descriptor-layer call
 * @return an MPI error class (never MPI_SUCCESS)
 */
int ErrnoToMpi(int err) {
  switch (err) {
    case ENOSPC:
    case EFBIG:
      return MPI_ERR_NO_SPACE;
#ifdef EDQUOT
    case EDQUOT:
      return MPI_ERR_QUOTA;
#endif
    case ENOENT:
      return MPI_ERR_NO_SUCH_FILE;
    case EEXIST:
      return MPI_ERR_FILE_EXISTS;
    case EACCES:
    case EPERM:
      return MPI_ERR_ACCESS;
    case EROFS:
      return MPI_ERR_READ_ONLY;
    case EBADF:
      return MPI_ERR_FILE;
    default:
      return MPI_ERR_IO;
  }
}

/**
 * Record how many bytes an operation transferred, so MPI_Get_count and
 * MPI_Get_elements report the real amount (short at EOF) instead of garbage.
 * @param status the caller's status (may be MPI_STATUS_IGNORE)
 * @param bytes bytes actually transferred
 */
void SetStatusBytes(MPI_Status *status, size_t bytes) {
  if (status == MPI_STATUS_IGNORE) {
    return;
  }
  // Count in MPI_BYTE, as ROMIO does: the status stores bytes, and
  // MPI_Get_count divides by the caller's datatype size.
  MPI_Status_set_elements_x(status, MPI_BYTE, static_cast<MPI_Count>(bytes));
  MPI_Status_set_cancelled(status, 0);
}

/**
 * Turn a descriptor-layer transfer result into an MPI return code and fill
 * the status.
 * @param n bytes transferred, or -1 with errno set
 * @param status the caller's status (may be MPI_STATUS_IGNORE)
 * @return MPI_SUCCESS, or the mapped error class
 */
int FinishXfer(clio::cte::filesystem::FsSsize n, MPI_Status *status) {
  if (n < 0) {
    int err = errno;
    SetStatusBytes(status, 0);
    return ErrnoToMpi(err);
  }
  SetStatusBytes(status, static_cast<size_t>(n));
  return MPI_SUCCESS;
}

/**
 * Refuse an op this layer cannot perform correctly, rather than move the
 * wrong bytes.
 * @param what what was refused
 * @return MPI_ERR_UNSUPPORTED_OPERATION
 */
int Unsupported(const char* what) {
  HLOG(kError, "MPI-IO adapter: unsupported on a clio:: file: {}", what);
  return MPI_ERR_UNSUPPORTED_OPERATION;
}

/** File-pointer ops need the default view and a contiguous memory type. */
int CheckSimple(MPI_File fh, MPI_Datatype datatype) {
  MpiToken* t = Shim().TokenOf(fh);
  if (t && !t->view.IsDefault()) {
    return Unsupported("file-pointer op under a non-default file view");
  }
  if (!IsContiguous(datatype)) {
    return Unsupported("file-pointer op with a non-contiguous datatype");
  }
  return MPI_SUCCESS;
}

/** Sequential read at the file pointer. */
int DoRead(MPI_File fh, int fd, void* buf, int count, MPI_Datatype datatype,
           MPI_Status* status) {
  int rc = CheckSimple(fh, datatype);
  if (rc != MPI_SUCCESS) return rc;
  return FinishXfer(CLIO_CFS_CLIENT->ReadFd(fd, buf, Bytes(count, datatype)),
                    status);
}
/** Sequential write at the file pointer. */
int DoWrite(MPI_File fh, int fd, const void* buf, int count,
            MPI_Datatype datatype, MPI_Status* status) {
  int rc = CheckSimple(fh, datatype);
  if (rc != MPI_SUCCESS) return rc;
  return FinishXfer(CLIO_CFS_CLIENT->WriteFd(fd, buf, Bytes(count, datatype)),
                    status);
}

/**
 * Map the data-stream range [pos, pos + n) through a file view to absolute
 * file segments (offset, length), in stream order.
 */
Blocks ViewSegments(const FileView& v, MPI_Aint pos, MPI_Aint n) {
  Blocks segs;
  if (n <= 0) return segs;
  if (v.IsContiguous()) {
    segs.emplace_back(v.disp + pos, n);
    return segs;
  }
  MPI_Aint tile = pos / v.tile_size;
  MPI_Aint skip = pos % v.tile_size;
  size_t bi = 0;
  while (skip >= v.tile[bi].second) {
    skip -= v.tile[bi].second;
    ++bi;
  }
  while (n > 0) {
    const auto& blk = v.tile[bi];
    MPI_Aint take = std::min(blk.second - skip, n);
    MPI_Aint off = v.disp + tile * v.tile_extent + blk.first + skip;
    if (!segs.empty() && segs.back().first + segs.back().second == off) {
      segs.back().second += take;
    } else {
      segs.emplace_back(off, take);
    }
    n -= take;
    skip = 0;
    if (++bi == v.tile.size()) {
      bi = 0;
      ++tile;
    }
  }
  return segs;
}

/** Copy between a packed buffer and a memory layout (gather or scatter). */
void CopyLayout(char* user, const Blocks& mem, char* packed, bool to_packed) {
  size_t q = 0;
  for (const auto& b : mem) {
    if (to_packed) {
      memcpy(packed + q, user + b.first, b.second);
    } else {
      memcpy(user + b.first, packed + q, b.second);
    }
    q += b.second;
  }
}

/**
 * Explicit-offset transfer honoring the file view and the memory datatype.
 * Reads of a non-contiguous view fetch the covering file span in one request
 * when it is dense enough (data sieving), else one request per segment.
 * Bytes past EOF read as zero and are not counted in the status.
 * @param off offset in etypes, relative to the view
 */
int XferAt(MPI_File fh, int fd, MPI_Offset off, void* buf, int count,
           MPI_Datatype datatype, MPI_Status* status, bool is_write) {
  constexpr MPI_Aint kSieveSlack = 4 << 20;
  MpiToken* t = Shim().TokenOf(fh);
  const FileView& v = t->view;
  const size_t n = Bytes(count, datatype);
  Blocks mem;
  const bool mem_contig = IsContiguous(datatype);
  if (!mem_contig && !FlattenMemory(count, datatype, &mem)) {
    return Unsupported("memory datatype constructor");
  }
  Blocks segs = ViewSegments(v, MPI_Aint(off) * v.etype_size, MPI_Aint(n));
  if (segs.empty()) {
    SetStatusBytes(status, 0);
    return MPI_SUCCESS;
  }
  // Fast path: one file segment, contiguous memory.
  if (mem_contig && segs.size() == 1) {
    auto r = is_write ? CLIO_CFS_CLIENT->PwriteFd(fd, buf, n, segs[0].first)
                      : CLIO_CFS_CLIENT->PreadFd(fd, buf, n, segs[0].first);
    return FinishXfer(r, status);
  }
  std::vector<char> packed(n);
  char* user = static_cast<char*>(buf);
  if (is_write) {
    if (mem_contig) {
      memcpy(packed.data(), user, n);
    } else {
      CopyLayout(user, mem, packed.data(), true);
    }
  }
  size_t done = 0;
  bool ascending = true;
  for (size_t i = 1; i < segs.size(); ++i) {
    if (segs[i].first < segs[i - 1].first + segs[i - 1].second) {
      ascending = false;
    }
  }
  const MPI_Aint lo = segs.front().first;
  const MPI_Aint hi = segs.back().first + segs.back().second;
  if (!is_write && ascending && hi - lo <= MPI_Aint(2 * n) + kSieveSlack) {
    std::vector<char> span(hi - lo);
    auto got = CLIO_CFS_CLIENT->PreadFd(fd, span.data(), span.size(), lo);
    if (got < 0) return FinishXfer(got, status);
    size_t q = 0;
    for (const auto& sg : segs) {
      MPI_Aint at = sg.first - lo;
      MPI_Aint avail =
          std::max<MPI_Aint>(0, std::min<MPI_Aint>(sg.second, got - at));
      memcpy(packed.data() + q, span.data() + at, avail);
      memset(packed.data() + q + avail, 0, sg.second - avail);
      q += sg.second;
      done += avail;
    }
  } else {
    size_t q = 0;
    for (const auto& sg : segs) {
      auto r = is_write ? CLIO_CFS_CLIENT->PwriteFd(fd, packed.data() + q,
                                                    sg.second, sg.first)
                        : CLIO_CFS_CLIENT->PreadFd(fd, packed.data() + q,
                                                   sg.second, sg.first);
      if (r < 0) return FinishXfer(r, status);
      done += r;
      if (!is_write && r < sg.second) {
        memset(packed.data() + q + r, 0, sg.second - r);
      }
      q += sg.second;
      if (is_write && r < sg.second) break;
    }
  }
  if (!is_write) {
    if (mem_contig) {
      memcpy(user, packed.data(), n);
    } else {
      CopyLayout(user, mem, packed.data(), false);
    }
  }
  return FinishXfer(static_cast<clio::cte::filesystem::FsSsize>(done), status);
}

/**
 * Complete a non-blocking op synchronously. There is no request to carry a
 * status, so only the return code reports the outcome.
 * @param rc the MPI return code of the synchronous op
 * @param request out: always MPI_REQUEST_NULL
 * @return rc
 */
int CompleteNow(int rc, MPI_Request *request) {
  *request = MPI_REQUEST_NULL;
  return rc;
}

}  // namespace

extern "C" {

/**
 * MPI lifecycle (pass-through; we only need CTE up before any file op).
 */
int CLIO_CTE_DECL(MPI_Init)(int *argc, char ***argv) {
  auto real_api = CLIO_CTE_MPIIO_API;
  return real_api->MPI_Init(argc, argv);
}

int CLIO_CTE_DECL(MPI_Finalize)(void) {
  auto real_api = CLIO_CTE_MPIIO_API;
  return real_api->MPI_Finalize();
}

int CLIO_CTE_DECL(MPI_Wait)(MPI_Request *req, MPI_Status *status) {
  auto real_api = CLIO_CTE_MPIIO_API;
  return real_api->MPI_Wait(req, status);
}

int CLIO_CTE_DECL(MPI_Waitall)(int count, MPI_Request *req,
                               MPI_Status *status) {
  auto real_api = CLIO_CTE_MPIIO_API;
  return real_api->MPI_Waitall(count, req, status);
}

/**
 * Metadata
 */
int CLIO_CTE_DECL(MPI_File_open)(MPI_Comm comm, const char *filename, int amode,
                                 MPI_Info info, MPI_File *fh) {
  auto real_api = CLIO_CTE_MPIIO_API;
  if (clio::cte::filesystem::Client::IsPathTracked(filename)) {
    HLOG(kDebug, "Intercept MPI_File_open {} amode {}", filename, amode);
    int fd = CLIO_CFS_CLIENT->OpenFd(filename, MpiioShim::AmodeToFlags(amode), 0644);
    if (fd < 0) {
      return ErrnoToMpi(errno);
    }
    *fh = Shim().Wrap(fd, amode);
    return MPI_SUCCESS;
  }
  return real_api->MPI_File_open(comm, filename, amode, info, fh);
}

int CLIO_CTE_DECL(MPI_File_close)(MPI_File *fh) {
  auto real_api = CLIO_CTE_MPIIO_API;
  int fd = Shim().Release(*fh);
  if (fd >= 0) {
    HLOG(kDebug, "Intercept MPI_File_close");
    // The handle is released either way, as close(2) releases the fd; a
    // non-success return reports a deferred write that failed.
    int rc = CLIO_CFS_CLIENT->CloseFd(fd);
    int err = errno;
    *fh = MPI_FILE_NULL;
    return (rc == 0) ? MPI_SUCCESS : ErrnoToMpi(err);
  }
  return real_api->MPI_File_close(fh);
}

int CLIO_CTE_DECL(MPI_File_seek)(MPI_File fh, MPI_Offset offset, int whence) {
  auto real_api = CLIO_CTE_MPIIO_API;
  int fd = Shim().FdOf(fh);
  if (fd >= 0) {
    int w = (whence == MPI_SEEK_SET)   ? SEEK_SET
            : (whence == MPI_SEEK_CUR) ? SEEK_CUR
                                       : SEEK_END;
    return (CLIO_CFS_CLIENT->SeekFd(fd, static_cast<off_t>(offset), w) >= 0)
               ? MPI_SUCCESS
               : ErrnoToMpi(errno);
  }
  return real_api->MPI_File_seek(fh, offset, whence);
}

int CLIO_CTE_DECL(MPI_File_seek_shared)(MPI_File fh, MPI_Offset offset,
                                        int whence) {
  if (Shim().IsTracked(fh)) {
    return CLIO_CTE_DECL(MPI_File_seek)(fh, offset, whence);
  }
  return CLIO_CTE_MPIIO_API->MPI_File_seek_shared(fh, offset, whence);
}

int CLIO_CTE_DECL(MPI_File_get_position)(MPI_File fh, MPI_Offset *offset) {
  auto real_api = CLIO_CTE_MPIIO_API;
  int fd = Shim().FdOf(fh);
  if (fd >= 0) {
    *offset = static_cast<MPI_Offset>(CLIO_CFS_CLIENT->TellFd(fd));
    return MPI_SUCCESS;
  }
  return real_api->MPI_File_get_position(fh, offset);
}

/**
 * Reads
 */
int CLIO_CTE_DECL(MPI_File_read)(MPI_File fh, void *buf, int count,
                                 MPI_Datatype datatype, MPI_Status *status) {
  int fd = Shim().FdOf(fh);
  if (fd >= 0) {
    return DoRead(fh, fd, buf, count, datatype, status);
  }
  return CLIO_CTE_MPIIO_API->MPI_File_read(fh, buf, count, datatype, status);
}
int CLIO_CTE_DECL(MPI_File_read_all)(MPI_File fh, void *buf, int count,
                                     MPI_Datatype datatype,
                                     MPI_Status *status) {
  int fd = Shim().FdOf(fh);
  if (fd >= 0) {
    return DoRead(fh, fd, buf, count, datatype, status);
  }
  return CLIO_CTE_MPIIO_API->MPI_File_read_all(fh, buf, count, datatype,
                                               status);
}
int CLIO_CTE_DECL(MPI_File_read_shared)(MPI_File fh, void *buf, int count,
                                        MPI_Datatype datatype,
                                        MPI_Status *status) {
  int fd = Shim().FdOf(fh);
  if (fd >= 0) {
    return DoRead(fh, fd, buf, count, datatype, status);
  }
  return CLIO_CTE_MPIIO_API->MPI_File_read_shared(fh, buf, count, datatype,
                                                  status);
}
int CLIO_CTE_DECL(MPI_File_read_ordered)(MPI_File fh, void *buf, int count,
                                         MPI_Datatype datatype,
                                         MPI_Status *status) {
  int fd = Shim().FdOf(fh);
  if (fd >= 0) {
    return DoRead(fh, fd, buf, count, datatype, status);
  }
  return CLIO_CTE_MPIIO_API->MPI_File_read_ordered(fh, buf, count, datatype,
                                                   status);
}
int CLIO_CTE_DECL(MPI_File_read_at)(MPI_File fh, MPI_Offset offset, void *buf,
                                    int count, MPI_Datatype datatype,
                                    MPI_Status *status) {
  int fd = Shim().FdOf(fh);
  if (fd >= 0) {
    return XferAt(fh, fd, offset, buf, count, datatype, status, false);
  }
  return CLIO_CTE_MPIIO_API->MPI_File_read_at(fh, offset, buf, count, datatype,
                                              status);
}
int CLIO_CTE_DECL(MPI_File_read_at_all)(MPI_File fh, MPI_Offset offset,
                                        void *buf, int count,
                                        MPI_Datatype datatype,
                                        MPI_Status *status) {
  int fd = Shim().FdOf(fh);
  if (fd >= 0) {
    return XferAt(fh, fd, offset, buf, count, datatype, status, false);
  }
  return CLIO_CTE_MPIIO_API->MPI_File_read_at_all(fh, offset, buf, count,
                                                  datatype, status);
}

/**
 * Writes
 */
int CLIO_CTE_DECL(MPI_File_write)(MPI_File fh, const void *buf, int count,
                                  MPI_Datatype datatype, MPI_Status *status) {
  int fd = Shim().FdOf(fh);
  if (fd >= 0) {
    return DoWrite(fh, fd, buf, count, datatype, status);
  }
  return CLIO_CTE_MPIIO_API->MPI_File_write(fh, buf, count, datatype, status);
}
int CLIO_CTE_DECL(MPI_File_write_all)(MPI_File fh, const void *buf, int count,
                                      MPI_Datatype datatype,
                                      MPI_Status *status) {
  int fd = Shim().FdOf(fh);
  if (fd >= 0) {
    return DoWrite(fh, fd, buf, count, datatype, status);
  }
  return CLIO_CTE_MPIIO_API->MPI_File_write_all(fh, buf, count, datatype,
                                                status);
}
int CLIO_CTE_DECL(MPI_File_write_shared)(MPI_File fh, const void *buf,
                                         int count, MPI_Datatype datatype,
                                         MPI_Status *status) {
  int fd = Shim().FdOf(fh);
  if (fd >= 0) {
    return DoWrite(fh, fd, buf, count, datatype, status);
  }
  return CLIO_CTE_MPIIO_API->MPI_File_write_shared(fh, buf, count, datatype,
                                                   status);
}
int CLIO_CTE_DECL(MPI_File_write_ordered)(MPI_File fh, const void *buf,
                                          int count, MPI_Datatype datatype,
                                          MPI_Status *status) {
  int fd = Shim().FdOf(fh);
  if (fd >= 0) {
    return DoWrite(fh, fd, buf, count, datatype, status);
  }
  return CLIO_CTE_MPIIO_API->MPI_File_write_ordered(fh, buf, count, datatype,
                                                    status);
}
int CLIO_CTE_DECL(MPI_File_write_at)(MPI_File fh, MPI_Offset offset,
                                     const void *buf, int count,
                                     MPI_Datatype datatype,
                                     MPI_Status *status) {
  int fd = Shim().FdOf(fh);
  if (fd >= 0) {
    return XferAt(fh, fd, offset, const_cast<void*>(buf), count, datatype,
                  status, true);
  }
  return CLIO_CTE_MPIIO_API->MPI_File_write_at(fh, offset, buf, count, datatype,
                                               status);
}
int CLIO_CTE_DECL(MPI_File_write_at_all)(MPI_File fh, MPI_Offset offset,
                                         const void *buf, int count,
                                         MPI_Datatype datatype,
                                         MPI_Status *status) {
  int fd = Shim().FdOf(fh);
  if (fd >= 0) {
    return XferAt(fh, fd, offset, const_cast<void*>(buf), count, datatype,
                  status, true);
  }
  return CLIO_CTE_MPIIO_API->MPI_File_write_at_all(fh, offset, buf, count,
                                                   datatype, status);
}

/**
 * Asynchronous variants — complete synchronously, hand back a null request.
 */
int CLIO_CTE_DECL(MPI_File_iread)(MPI_File fh, void *buf, int count,
                                  MPI_Datatype datatype, MPI_Request *request) {
  int fd = Shim().FdOf(fh);
  if (fd >= 0) {
    return CompleteNow(DoRead(fh, fd, buf, count, datatype, MPI_STATUS_IGNORE),
                       request);
  }
  return CLIO_CTE_MPIIO_API->MPI_File_iread(fh, buf, count, datatype, request);
}
int CLIO_CTE_DECL(MPI_File_iread_shared)(MPI_File fh, void *buf, int count,
                                         MPI_Datatype datatype,
                                         MPI_Request *request) {
  int fd = Shim().FdOf(fh);
  if (fd >= 0) {
    return CompleteNow(DoRead(fh, fd, buf, count, datatype, MPI_STATUS_IGNORE),
                       request);
  }
  return CLIO_CTE_MPIIO_API->MPI_File_iread_shared(fh, buf, count, datatype,
                                                   request);
}
int CLIO_CTE_DECL(MPI_File_iread_at)(MPI_File fh, MPI_Offset offset, void *buf,
                                     int count, MPI_Datatype datatype,
                                     MPI_Request *request) {
  int fd = Shim().FdOf(fh);
  if (fd >= 0) {
    return CompleteNow(
        XferAt(fh, fd, offset, buf, count, datatype, MPI_STATUS_IGNORE, false),
        request);
  }
  return CLIO_CTE_MPIIO_API->MPI_File_iread_at(fh, offset, buf, count, datatype,
                                               request);
}
int CLIO_CTE_DECL(MPI_File_iwrite)(MPI_File fh, const void *buf, int count,
                                   MPI_Datatype datatype,
                                   MPI_Request *request) {
  int fd = Shim().FdOf(fh);
  if (fd >= 0) {
    return CompleteNow(DoWrite(fh, fd, buf, count, datatype, MPI_STATUS_IGNORE),
                       request);
  }
  return CLIO_CTE_MPIIO_API->MPI_File_iwrite(fh, buf, count, datatype, request);
}
int CLIO_CTE_DECL(MPI_File_iwrite_shared)(MPI_File fh, const void *buf,
                                          int count, MPI_Datatype datatype,
                                          MPI_Request *request) {
  int fd = Shim().FdOf(fh);
  if (fd >= 0) {
    return CompleteNow(DoWrite(fh, fd, buf, count, datatype, MPI_STATUS_IGNORE),
                       request);
  }
  return CLIO_CTE_MPIIO_API->MPI_File_iwrite_shared(fh, buf, count, datatype,
                                                    request);
}
int CLIO_CTE_DECL(MPI_File_iwrite_at)(MPI_File fh, MPI_Offset offset,
                                      const void *buf, int count,
                                      MPI_Datatype datatype,
                                      MPI_Request *request) {
  int fd = Shim().FdOf(fh);
  if (fd >= 0) {
    return CompleteNow(XferAt(fh, fd, offset, const_cast<void*>(buf), count,
                              datatype, MPI_STATUS_IGNORE, true),
                       request);
  }
  return CLIO_CTE_MPIIO_API->MPI_File_iwrite_at(fh, offset, buf, count,
                                                datatype, request);
}

int CLIO_CTE_DECL(MPI_File_sync)(MPI_File fh) {
  int fd = Shim().FdOf(fh);
  if (fd >= 0) {
    // Writes are deferred (write-behind): wait for this file's writes and
    // report, once, a failure any of them latched -- fsync(2) semantics.
    return (CLIO_CFS_CLIENT->SyncFd(fd) == 0) ? MPI_SUCCESS
                                              : ErrnoToMpi(errno);
  }
  return CLIO_CTE_MPIIO_API->MPI_File_sync(fh);
}

/**
 * File metadata, info and views. Parallel HDF5's MPI-IO driver calls these on
 * every file it opens; a clio:: handle is not an MPI_File, so none of them
 * may reach the real MPI library.
 */
int CLIO_CTE_DECL(MPI_File_get_size)(MPI_File fh, MPI_Offset* size) {
  int fd = Shim().FdOf(fh);
  if (fd >= 0) {
    auto n = CLIO_CFS_CLIENT->SizeFd(fd);
    if (n < 0) return ErrnoToMpi(errno);
    *size = static_cast<MPI_Offset>(n);
    return MPI_SUCCESS;
  }
  return CLIO_CTE_MPIIO_API->MPI_File_get_size(fh, size);
}

int CLIO_CTE_DECL(MPI_File_set_size)(MPI_File fh, MPI_Offset size) {
  int fd = Shim().FdOf(fh);
  if (fd >= 0) {
    return (CLIO_CFS_CLIENT->FtruncateFd(fd, static_cast<off_t>(size)) == 0)
               ? MPI_SUCCESS
               : ErrnoToMpi(errno);
  }
  return CLIO_CTE_MPIIO_API->MPI_File_set_size(fh, size);
}

int CLIO_CTE_DECL(MPI_File_get_info)(MPI_File fh, MPI_Info* info_used) {
  if (Shim().IsTracked(fh)) {
    // No hints are in effect; the caller frees the returned object.
    return MPI_Info_create(info_used);
  }
  return CLIO_CTE_MPIIO_API->MPI_File_get_info(fh, info_used);
}

int CLIO_CTE_DECL(MPI_File_set_info)(MPI_File fh, MPI_Info info) {
  if (Shim().IsTracked(fh)) {
    return MPI_SUCCESS;  // hints are advisory; none apply to clio-fs
  }
  return CLIO_CTE_MPIIO_API->MPI_File_set_info(fh, info);
}

int CLIO_CTE_DECL(MPI_File_get_amode)(MPI_File fh, int* amode) {
  if (MpiToken* t = Shim().TokenOf(fh)) {
    *amode = t->amode;
    return MPI_SUCCESS;
  }
  return CLIO_CTE_MPIIO_API->MPI_File_get_amode(fh, amode);
}

int CLIO_CTE_DECL(MPI_File_set_atomicity)(MPI_File fh, int flag) {
  if (MpiToken* t = Shim().TokenOf(fh)) {
    // Recorded only: each transfer is already a single clio-fs request.
    t->atomic = flag;
    return MPI_SUCCESS;
  }
  return CLIO_CTE_MPIIO_API->MPI_File_set_atomicity(fh, flag);
}

int CLIO_CTE_DECL(MPI_File_get_atomicity)(MPI_File fh, int* flag) {
  if (MpiToken* t = Shim().TokenOf(fh)) {
    *flag = t->atomic;
    return MPI_SUCCESS;
  }
  return CLIO_CTE_MPIIO_API->MPI_File_get_atomicity(fh, flag);
}

int CLIO_CTE_DECL(MPI_File_set_view)(MPI_File fh, MPI_Offset disp,
                                     MPI_Datatype etype, MPI_Datatype filetype,
                                     const char* datarep, MPI_Info info) {
  if (MpiToken* t = Shim().TokenOf(fh)) {
    if (datarep && strcmp(datarep, "native") != 0) {
      return Unsupported("file view data representation other than native");
    }
    Blocks tile;
    if (!Flatten(filetype, 0, &tile)) {
      return Unsupported("file view filetype constructor");
    }
    MPI_Aint size = 0;
    for (const auto& b : tile) size += b.second;
    if (size <= 0) {
      return Unsupported("file view filetype with no data");
    }
    int esize = 0;
    MPI_Type_size(etype, &esize);
    MPI_Aint lb, ext;
    MPI_Type_get_extent(filetype, &lb, &ext);
    FileView& v = t->view;
    v.disp = disp;
    v.etype_size = (esize > 0) ? esize : 1;
    v.tile = std::move(tile);
    v.tile_size = size;
    v.tile_extent = (ext > 0) ? ext : size;
    return MPI_SUCCESS;
  }
  return CLIO_CTE_MPIIO_API->MPI_File_set_view(fh, disp, etype, filetype,
                                               datarep, info);
}

int CLIO_CTE_DECL(MPI_File_delete)(const char* filename, MPI_Info info) {
  if (clio::cte::filesystem::Client::IsPathTracked(filename)) {
    return (CLIO_CFS_CLIENT->RemovePath(filename) == 0) ? MPI_SUCCESS
                                                        : ErrnoToMpi(errno);
  }
  return CLIO_CTE_MPIIO_API->MPI_File_delete(filename, info);
}

}  // extern C
