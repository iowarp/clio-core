/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 *
 * MPI-IO interceptor. Tracked paths (those carrying the "clio::" marker) are
 * routed to the context-filesystem chimod via the shared CfsIo core; a clio::
 * MPI_File is an opaque heap token mapped to a CfsIo descriptor. Everything
 * else falls through to the real MPI-IO API.
 *
 * Single-rank (MPI_COMM_SELF) byte semantics: collective (_all), shared and
 * ordered variants behave like their independent counterparts here — the
 * chimod owns placement, so there is no separate collective-aggregation step
 * at this layer. Asynchronous (i*) ops complete synchronously and hand back
 * MPI_REQUEST_NULL.
 */

bool mpiio_intercepted = true;

#include "mpiio_api.h"

#include <fcntl.h>

#include <mutex>
#include <unordered_map>

#include "clio_ctp/util/logging.h"
#include "clio_ctp/util/singleton.h"
#include "adapter/cfs/cfs_io.h"

namespace {

struct MpiToken {
  int fd = -1;
};

/** Maps clio:: MPI_File tokens to their CfsIo descriptors. */
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

  MPI_File Wrap(int fd) {
    auto *t = new MpiToken();
    t->fd = fd;
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

/** Sequential read at the file pointer. */
int DoRead(int fd, void *buf, int count, MPI_Datatype datatype) {
  CLIO_CTE_CFS->Read(fd, buf, Bytes(count, datatype));
  return MPI_SUCCESS;
}
int DoReadAt(int fd, MPI_Offset off, void *buf, int count,
             MPI_Datatype datatype) {
  CLIO_CTE_CFS->Pread(fd, buf, Bytes(count, datatype),
                      static_cast<off_t>(off));
  return MPI_SUCCESS;
}
int DoWrite(int fd, const void *buf, int count, MPI_Datatype datatype) {
  CLIO_CTE_CFS->Write(fd, buf, Bytes(count, datatype));
  return MPI_SUCCESS;
}
int DoWriteAt(int fd, MPI_Offset off, const void *buf, int count,
              MPI_Datatype datatype) {
  CLIO_CTE_CFS->Pwrite(fd, buf, Bytes(count, datatype),
                       static_cast<off_t>(off));
  return MPI_SUCCESS;
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
  if (clio::cae::CfsIo::IsPathTracked(filename)) {
    HLOG(kDebug, "Intercept MPI_File_open {} amode {}", filename, amode);
    int fd = CLIO_CTE_CFS->Open(filename, MpiioShim::AmodeToFlags(amode), 0644);
    if (fd < 0) {
      return MPI_ERR_NO_SUCH_FILE;
    }
    *fh = Shim().Wrap(fd);
    return MPI_SUCCESS;
  }
  return real_api->MPI_File_open(comm, filename, amode, info, fh);
}

int CLIO_CTE_DECL(MPI_File_close)(MPI_File *fh) {
  auto real_api = CLIO_CTE_MPIIO_API;
  int fd = Shim().Release(*fh);
  if (fd >= 0) {
    HLOG(kDebug, "Intercept MPI_File_close");
    CLIO_CTE_CFS->Close(fd);
    *fh = MPI_FILE_NULL;
    return MPI_SUCCESS;
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
    return (CLIO_CTE_CFS->Seek(fd, static_cast<off_t>(offset), w) >= 0)
               ? MPI_SUCCESS
               : MPI_ERR_IO;
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
    *offset = static_cast<MPI_Offset>(CLIO_CTE_CFS->Tell(fd));
    return MPI_SUCCESS;
  }
  return real_api->MPI_File_get_position(fh, offset);
}

/**
 * Reads
 */
int CLIO_CTE_DECL(MPI_File_read)(MPI_File fh, void *buf, int count,
                                 MPI_Datatype datatype, MPI_Status *status) {
  (void)status;
  int fd = Shim().FdOf(fh);
  if (fd >= 0) {
    return DoRead(fd, buf, count, datatype);
  }
  return CLIO_CTE_MPIIO_API->MPI_File_read(fh, buf, count, datatype, status);
}
int CLIO_CTE_DECL(MPI_File_read_all)(MPI_File fh, void *buf, int count,
                                     MPI_Datatype datatype,
                                     MPI_Status *status) {
  (void)status;
  int fd = Shim().FdOf(fh);
  if (fd >= 0) {
    return DoRead(fd, buf, count, datatype);
  }
  return CLIO_CTE_MPIIO_API->MPI_File_read_all(fh, buf, count, datatype,
                                               status);
}
int CLIO_CTE_DECL(MPI_File_read_shared)(MPI_File fh, void *buf, int count,
                                        MPI_Datatype datatype,
                                        MPI_Status *status) {
  (void)status;
  int fd = Shim().FdOf(fh);
  if (fd >= 0) {
    return DoRead(fd, buf, count, datatype);
  }
  return CLIO_CTE_MPIIO_API->MPI_File_read_shared(fh, buf, count, datatype,
                                                  status);
}
int CLIO_CTE_DECL(MPI_File_read_ordered)(MPI_File fh, void *buf, int count,
                                         MPI_Datatype datatype,
                                         MPI_Status *status) {
  (void)status;
  int fd = Shim().FdOf(fh);
  if (fd >= 0) {
    return DoRead(fd, buf, count, datatype);
  }
  return CLIO_CTE_MPIIO_API->MPI_File_read_ordered(fh, buf, count, datatype,
                                                   status);
}
int CLIO_CTE_DECL(MPI_File_read_at)(MPI_File fh, MPI_Offset offset, void *buf,
                                    int count, MPI_Datatype datatype,
                                    MPI_Status *status) {
  (void)status;
  int fd = Shim().FdOf(fh);
  if (fd >= 0) {
    return DoReadAt(fd, offset, buf, count, datatype);
  }
  return CLIO_CTE_MPIIO_API->MPI_File_read_at(fh, offset, buf, count, datatype,
                                              status);
}
int CLIO_CTE_DECL(MPI_File_read_at_all)(MPI_File fh, MPI_Offset offset,
                                        void *buf, int count,
                                        MPI_Datatype datatype,
                                        MPI_Status *status) {
  (void)status;
  int fd = Shim().FdOf(fh);
  if (fd >= 0) {
    return DoReadAt(fd, offset, buf, count, datatype);
  }
  return CLIO_CTE_MPIIO_API->MPI_File_read_at_all(fh, offset, buf, count,
                                                  datatype, status);
}

/**
 * Writes
 */
int CLIO_CTE_DECL(MPI_File_write)(MPI_File fh, const void *buf, int count,
                                  MPI_Datatype datatype, MPI_Status *status) {
  (void)status;
  int fd = Shim().FdOf(fh);
  if (fd >= 0) {
    return DoWrite(fd, buf, count, datatype);
  }
  return CLIO_CTE_MPIIO_API->MPI_File_write(fh, buf, count, datatype, status);
}
int CLIO_CTE_DECL(MPI_File_write_all)(MPI_File fh, const void *buf, int count,
                                      MPI_Datatype datatype,
                                      MPI_Status *status) {
  (void)status;
  int fd = Shim().FdOf(fh);
  if (fd >= 0) {
    return DoWrite(fd, buf, count, datatype);
  }
  return CLIO_CTE_MPIIO_API->MPI_File_write_all(fh, buf, count, datatype,
                                                status);
}
int CLIO_CTE_DECL(MPI_File_write_shared)(MPI_File fh, const void *buf,
                                         int count, MPI_Datatype datatype,
                                         MPI_Status *status) {
  (void)status;
  int fd = Shim().FdOf(fh);
  if (fd >= 0) {
    return DoWrite(fd, buf, count, datatype);
  }
  return CLIO_CTE_MPIIO_API->MPI_File_write_shared(fh, buf, count, datatype,
                                                   status);
}
int CLIO_CTE_DECL(MPI_File_write_ordered)(MPI_File fh, const void *buf,
                                          int count, MPI_Datatype datatype,
                                          MPI_Status *status) {
  (void)status;
  int fd = Shim().FdOf(fh);
  if (fd >= 0) {
    return DoWrite(fd, buf, count, datatype);
  }
  return CLIO_CTE_MPIIO_API->MPI_File_write_ordered(fh, buf, count, datatype,
                                                    status);
}
int CLIO_CTE_DECL(MPI_File_write_at)(MPI_File fh, MPI_Offset offset,
                                     const void *buf, int count,
                                     MPI_Datatype datatype,
                                     MPI_Status *status) {
  (void)status;
  int fd = Shim().FdOf(fh);
  if (fd >= 0) {
    return DoWriteAt(fd, offset, buf, count, datatype);
  }
  return CLIO_CTE_MPIIO_API->MPI_File_write_at(fh, offset, buf, count, datatype,
                                               status);
}
int CLIO_CTE_DECL(MPI_File_write_at_all)(MPI_File fh, MPI_Offset offset,
                                         const void *buf, int count,
                                         MPI_Datatype datatype,
                                         MPI_Status *status) {
  (void)status;
  int fd = Shim().FdOf(fh);
  if (fd >= 0) {
    return DoWriteAt(fd, offset, buf, count, datatype);
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
    DoRead(fd, buf, count, datatype);
    *request = MPI_REQUEST_NULL;
    return MPI_SUCCESS;
  }
  return CLIO_CTE_MPIIO_API->MPI_File_iread(fh, buf, count, datatype, request);
}
int CLIO_CTE_DECL(MPI_File_iread_shared)(MPI_File fh, void *buf, int count,
                                         MPI_Datatype datatype,
                                         MPI_Request *request) {
  int fd = Shim().FdOf(fh);
  if (fd >= 0) {
    DoRead(fd, buf, count, datatype);
    *request = MPI_REQUEST_NULL;
    return MPI_SUCCESS;
  }
  return CLIO_CTE_MPIIO_API->MPI_File_iread_shared(fh, buf, count, datatype,
                                                   request);
}
int CLIO_CTE_DECL(MPI_File_iread_at)(MPI_File fh, MPI_Offset offset, void *buf,
                                     int count, MPI_Datatype datatype,
                                     MPI_Request *request) {
  int fd = Shim().FdOf(fh);
  if (fd >= 0) {
    DoReadAt(fd, offset, buf, count, datatype);
    *request = MPI_REQUEST_NULL;
    return MPI_SUCCESS;
  }
  return CLIO_CTE_MPIIO_API->MPI_File_iread_at(fh, offset, buf, count, datatype,
                                               request);
}
int CLIO_CTE_DECL(MPI_File_iwrite)(MPI_File fh, const void *buf, int count,
                                   MPI_Datatype datatype,
                                   MPI_Request *request) {
  int fd = Shim().FdOf(fh);
  if (fd >= 0) {
    DoWrite(fd, buf, count, datatype);
    *request = MPI_REQUEST_NULL;
    return MPI_SUCCESS;
  }
  return CLIO_CTE_MPIIO_API->MPI_File_iwrite(fh, buf, count, datatype, request);
}
int CLIO_CTE_DECL(MPI_File_iwrite_shared)(MPI_File fh, const void *buf,
                                          int count, MPI_Datatype datatype,
                                          MPI_Request *request) {
  int fd = Shim().FdOf(fh);
  if (fd >= 0) {
    DoWrite(fd, buf, count, datatype);
    *request = MPI_REQUEST_NULL;
    return MPI_SUCCESS;
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
    DoWriteAt(fd, offset, buf, count, datatype);
    *request = MPI_REQUEST_NULL;
    return MPI_SUCCESS;
  }
  return CLIO_CTE_MPIIO_API->MPI_File_iwrite_at(fh, offset, buf, count,
                                                datatype, request);
}

int CLIO_CTE_DECL(MPI_File_sync)(MPI_File fh) {
  int fd = Shim().FdOf(fh);
  if (fd >= 0) {
    return MPI_SUCCESS;  // writes are synchronous
  }
  return CLIO_CTE_MPIIO_API->MPI_File_sync(fh);
}

}  // extern C
