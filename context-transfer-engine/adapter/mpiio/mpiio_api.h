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

#ifndef CLIO_CTE_ADAPTER_MPIIO_H
#define CLIO_CTE_ADAPTER_MPIIO_H
#include <dlfcn.h>
#include <mpi.h>

#include <iostream>
#include <string>

#include "clio_ctp/util/logging.h"
#ifdef CLIO_CTE_MPICH
#include <mpio.h>
#endif
#include "clio_ctp/util/real_api.h"

#ifndef MPI_MODE_TRUNCATE
#define MPI_MODE_TRUNCATE 0
#endif

extern "C" {
typedef int (*MPI_Init_t)(int* argc, char*** argv);
typedef int (*MPI_Finalize_t)(void);
typedef int (*MPI_Wait_t)(MPI_Request* req, MPI_Status* status);
typedef int (*MPI_Waitall_t)(int count, MPI_Request* req, MPI_Status* status);
typedef int (*MPI_File_open_t)(MPI_Comm comm, const char* filename, int amode,
                               MPI_Info info, MPI_File* fh);
typedef int (*MPI_File_close_t)(MPI_File* fh);
typedef int (*MPI_File_seek_shared_t)(MPI_File fh, MPI_Offset offset,
                                      int whence);
typedef int (*MPI_File_seek_t)(MPI_File fh, MPI_Offset offset, int whence);
typedef int (*MPI_File_get_position_t)(MPI_File fh, MPI_Offset* offset);
typedef int (*MPI_File_read_all_t)(MPI_File fh, void* buf, int count,
                                   MPI_Datatype datatype, MPI_Status* status);
typedef int (*MPI_File_read_at_all_t)(MPI_File fh, MPI_Offset offset, void* buf,
                                      int count, MPI_Datatype datatype,
                                      MPI_Status* status);
typedef int (*MPI_File_read_at_t)(MPI_File fh, MPI_Offset offset, void* buf,
                                  int count, MPI_Datatype datatype,
                                  MPI_Status* status);
typedef int (*MPI_File_read_t)(MPI_File fh, void* buf, int count,
                               MPI_Datatype datatype, MPI_Status* status);
typedef int (*MPI_File_read_ordered_t)(MPI_File fh, void* buf, int count,
                                       MPI_Datatype datatype,
                                       MPI_Status* status);
typedef int (*MPI_File_read_shared_t)(MPI_File fh, void* buf, int count,
                                      MPI_Datatype datatype,
                                      MPI_Status* status);
typedef int (*MPI_File_write_all_t)(MPI_File fh, const void* buf, int count,
                                    MPI_Datatype datatype, MPI_Status* status);
typedef int (*MPI_File_write_at_all_t)(MPI_File fh, MPI_Offset offset,
                                       const void* buf, int count,
                                       MPI_Datatype datatype,
                                       MPI_Status* status);
typedef int (*MPI_File_write_at_t)(MPI_File fh, MPI_Offset offset,
                                   const void* buf, int count,
                                   MPI_Datatype datatype, MPI_Status* status);
typedef int (*MPI_File_write_t)(MPI_File fh, const void* buf, int count,
                                MPI_Datatype datatype, MPI_Status* status);
typedef int (*MPI_File_write_ordered_t)(MPI_File fh, const void* buf, int count,
                                        MPI_Datatype datatype,
                                        MPI_Status* status);
typedef int (*MPI_File_write_shared_t)(MPI_File fh, const void* buf, int count,
                                       MPI_Datatype datatype,
                                       MPI_Status* status);
typedef int (*MPI_File_iread_at_t)(MPI_File fh, MPI_Offset offset, void* buf,
                                   int count, MPI_Datatype datatype,
                                   MPI_Request* request);
typedef int (*MPI_File_iread_t)(MPI_File fh, void* buf, int count,
                                MPI_Datatype datatype, MPI_Request* request);
typedef int (*MPI_File_iread_shared_t)(MPI_File fh, void* buf, int count,
                                       MPI_Datatype datatype,
                                       MPI_Request* request);
typedef int (*MPI_File_iwrite_at_t)(MPI_File fh, MPI_Offset offset,
                                    const void* buf, int count,
                                    MPI_Datatype datatype,
                                    MPI_Request* request);
typedef int (*MPI_File_iwrite_t)(MPI_File fh, const void* buf, int count,
                                 MPI_Datatype datatype, MPI_Request* request);
typedef int (*MPI_File_iwrite_shared_t)(MPI_File fh, const void* buf, int count,
                                        MPI_Datatype datatype,
                                        MPI_Request* request);
typedef int (*MPI_File_sync_t)(MPI_File fh);
typedef int (*MPI_File_get_size_t)(MPI_File fh, MPI_Offset* size);
typedef int (*MPI_File_set_size_t)(MPI_File fh, MPI_Offset size);
typedef int (*MPI_File_get_info_t)(MPI_File fh, MPI_Info* info_used);
typedef int (*MPI_File_set_info_t)(MPI_File fh, MPI_Info info);
typedef int (*MPI_File_get_amode_t)(MPI_File fh, int* amode);
typedef int (*MPI_File_set_atomicity_t)(MPI_File fh, int flag);
typedef int (*MPI_File_get_atomicity_t)(MPI_File fh, int* flag);
typedef int (*MPI_File_set_view_t)(MPI_File fh, MPI_Offset disp,
                                   MPI_Datatype etype, MPI_Datatype filetype,
                                   const char* datarep, MPI_Info info);
typedef int (*MPI_File_delete_t)(const char* filename, MPI_Info info);
}

namespace clio::cae {

using ctp::RealApi;

/** Pointers to the real mpiio API */
class MpiioApi : public RealApi {
 public:
  /** MPI_Init */
  MPI_Init_t MPI_Init = nullptr;
  /** MPI_Finalize */
  MPI_Finalize_t MPI_Finalize = nullptr;
  /** MPI_Wait */
  MPI_Wait_t MPI_Wait = nullptr;
  /** MPI_Waitall */
  MPI_Waitall_t MPI_Waitall = nullptr;
  /** MPI_File_open */
  MPI_File_open_t MPI_File_open = nullptr;
  /** MPI_File_close */
  MPI_File_close_t MPI_File_close = nullptr;
  /** MPI_File_seek_shared */
  MPI_File_seek_shared_t MPI_File_seek_shared = nullptr;
  /** MPI_File_seek */
  MPI_File_seek_t MPI_File_seek = nullptr;
  /** MPI_File_get_position */
  MPI_File_get_position_t MPI_File_get_position = nullptr;
  /** MPI_File_read_all */
  MPI_File_read_all_t MPI_File_read_all = nullptr;
  /** MPI_File_read_at_all */
  MPI_File_read_at_all_t MPI_File_read_at_all = nullptr;
  /** MPI_File_read_at */
  MPI_File_read_at_t MPI_File_read_at = nullptr;
  /** MPI_File_read */
  MPI_File_read_t MPI_File_read = nullptr;
  /** MPI_File_read_ordered */
  MPI_File_read_ordered_t MPI_File_read_ordered = nullptr;
  /** MPI_File_read_shared */
  MPI_File_read_shared_t MPI_File_read_shared = nullptr;
  /** MPI_File_write_all */
  MPI_File_write_all_t MPI_File_write_all = nullptr;
  /** MPI_File_write_at_all */
  MPI_File_write_at_all_t MPI_File_write_at_all = nullptr;
  /** MPI_File_write_at */
  MPI_File_write_at_t MPI_File_write_at = nullptr;
  /** MPI_File_write */
  MPI_File_write_t MPI_File_write = nullptr;
  /** MPI_File_write_ordered */
  MPI_File_write_ordered_t MPI_File_write_ordered = nullptr;
  /** MPI_File_write_shared */
  MPI_File_write_shared_t MPI_File_write_shared = nullptr;
  /** MPI_File_iread_at */
  MPI_File_iread_at_t MPI_File_iread_at = nullptr;
  /** MPI_File_iread */
  MPI_File_iread_t MPI_File_iread = nullptr;
  /** MPI_File_iread_shared */
  MPI_File_iread_shared_t MPI_File_iread_shared = nullptr;
  /** MPI_File_iwrite_at */
  MPI_File_iwrite_at_t MPI_File_iwrite_at = nullptr;
  /** MPI_File_iwrite */
  MPI_File_iwrite_t MPI_File_iwrite = nullptr;
  /** MPI_File_iwrite_shared */
  MPI_File_iwrite_shared_t MPI_File_iwrite_shared = nullptr;
  /** MPI_File_sync */
  MPI_File_sync_t MPI_File_sync = nullptr;
  /** MPI_File_get_size */
  MPI_File_get_size_t MPI_File_get_size = nullptr;
  /** MPI_File_set_size */
  MPI_File_set_size_t MPI_File_set_size = nullptr;
  /** MPI_File_get_info */
  MPI_File_get_info_t MPI_File_get_info = nullptr;
  /** MPI_File_set_info */
  MPI_File_set_info_t MPI_File_set_info = nullptr;
  /** MPI_File_get_amode */
  MPI_File_get_amode_t MPI_File_get_amode = nullptr;
  /** MPI_File_set_atomicity */
  MPI_File_set_atomicity_t MPI_File_set_atomicity = nullptr;
  /** MPI_File_get_atomicity */
  MPI_File_get_atomicity_t MPI_File_get_atomicity = nullptr;
  /** MPI_File_set_view */
  MPI_File_set_view_t MPI_File_set_view = nullptr;
  /** MPI_File_delete */
  MPI_File_delete_t MPI_File_delete = nullptr;

  MpiioApi() : RealApi("MPI_Init", "mpiio_intercepted") {
    MPI_Init = (MPI_Init_t)dlsym(real_lib_, "MPI_Init");
    REQUIRE_API(MPI_Init)
    MPI_Finalize = (MPI_Finalize_t)dlsym(real_lib_, "MPI_Finalize");
    REQUIRE_API(MPI_Finalize)
    MPI_Wait = (MPI_Wait_t)dlsym(real_lib_, "MPI_Wait");
    REQUIRE_API(MPI_Wait)
    MPI_Waitall = (MPI_Waitall_t)dlsym(real_lib_, "MPI_Waitall");
    REQUIRE_API(MPI_Waitall)
    MPI_File_open = (MPI_File_open_t)dlsym(real_lib_, "MPI_File_open");
    REQUIRE_API(MPI_File_open)
    MPI_File_close = (MPI_File_close_t)dlsym(real_lib_, "MPI_File_close");
    REQUIRE_API(MPI_File_close)
    MPI_File_seek_shared =
        (MPI_File_seek_shared_t)dlsym(real_lib_, "MPI_File_seek_shared");
    REQUIRE_API(MPI_File_seek_shared)
    MPI_File_seek = (MPI_File_seek_t)dlsym(real_lib_, "MPI_File_seek");
    REQUIRE_API(MPI_File_seek)
    MPI_File_get_position =
        (MPI_File_get_position_t)dlsym(real_lib_, "MPI_File_get_position");
    REQUIRE_API(MPI_File_get_position)
    MPI_File_read_all =
        (MPI_File_read_all_t)dlsym(real_lib_, "MPI_File_read_all");
    REQUIRE_API(MPI_File_read_all)
    MPI_File_read_at_all =
        (MPI_File_read_at_all_t)dlsym(real_lib_, "MPI_File_read_at_all");
    REQUIRE_API(MPI_File_read_at_all)
    MPI_File_read_at = (MPI_File_read_at_t)dlsym(real_lib_, "MPI_File_read_at");
    REQUIRE_API(MPI_File_read_at)
    MPI_File_read = (MPI_File_read_t)dlsym(real_lib_, "MPI_File_read");
    REQUIRE_API(MPI_File_read)
    MPI_File_read_ordered =
        (MPI_File_read_ordered_t)dlsym(real_lib_, "MPI_File_read_ordered");
    REQUIRE_API(MPI_File_read_ordered)
    MPI_File_read_shared =
        (MPI_File_read_shared_t)dlsym(real_lib_, "MPI_File_read_shared");
    REQUIRE_API(MPI_File_read_shared)
    MPI_File_write_all =
        (MPI_File_write_all_t)dlsym(real_lib_, "MPI_File_write_all");
    REQUIRE_API(MPI_File_write_all)
    MPI_File_write_at_all =
        (MPI_File_write_at_all_t)dlsym(real_lib_, "MPI_File_write_at_all");
    REQUIRE_API(MPI_File_write_at_all)
    MPI_File_write_at =
        (MPI_File_write_at_t)dlsym(real_lib_, "MPI_File_write_at");
    REQUIRE_API(MPI_File_write_at)
    MPI_File_write = (MPI_File_write_t)dlsym(real_lib_, "MPI_File_write");
    REQUIRE_API(MPI_File_write)
    MPI_File_write_ordered =
        (MPI_File_write_ordered_t)dlsym(real_lib_, "MPI_File_write_ordered");
    REQUIRE_API(MPI_File_write_ordered)
    MPI_File_write_shared =
        (MPI_File_write_shared_t)dlsym(real_lib_, "MPI_File_write_shared");
    REQUIRE_API(MPI_File_write_shared)
    MPI_File_iread_at =
        (MPI_File_iread_at_t)dlsym(real_lib_, "MPI_File_iread_at");
    REQUIRE_API(MPI_File_iread_at)
    MPI_File_iread = (MPI_File_iread_t)dlsym(real_lib_, "MPI_File_iread");
    REQUIRE_API(MPI_File_iread)
    MPI_File_iread_shared =
        (MPI_File_iread_shared_t)dlsym(real_lib_, "MPI_File_iread_shared");
    REQUIRE_API(MPI_File_iread_shared)
    MPI_File_iwrite_at =
        (MPI_File_iwrite_at_t)dlsym(real_lib_, "MPI_File_iwrite_at");
    REQUIRE_API(MPI_File_iwrite_at)
    MPI_File_iwrite = (MPI_File_iwrite_t)dlsym(real_lib_, "MPI_File_iwrite");
    REQUIRE_API(MPI_File_iwrite)
    MPI_File_iwrite_shared =
        (MPI_File_iwrite_shared_t)dlsym(real_lib_, "MPI_File_iwrite_shared");
    REQUIRE_API(MPI_File_iwrite_shared)
    MPI_File_sync = (MPI_File_sync_t)dlsym(real_lib_, "MPI_File_sync");
    REQUIRE_API(MPI_File_sync)
    MPI_File_get_size =
        (MPI_File_get_size_t)dlsym(real_lib_, "MPI_File_get_size");
    REQUIRE_API(MPI_File_get_size)
    MPI_File_set_size =
        (MPI_File_set_size_t)dlsym(real_lib_, "MPI_File_set_size");
    REQUIRE_API(MPI_File_set_size)
    MPI_File_get_info =
        (MPI_File_get_info_t)dlsym(real_lib_, "MPI_File_get_info");
    REQUIRE_API(MPI_File_get_info)
    MPI_File_set_info =
        (MPI_File_set_info_t)dlsym(real_lib_, "MPI_File_set_info");
    REQUIRE_API(MPI_File_set_info)
    MPI_File_get_amode =
        (MPI_File_get_amode_t)dlsym(real_lib_, "MPI_File_get_amode");
    REQUIRE_API(MPI_File_get_amode)
    MPI_File_set_atomicity =
        (MPI_File_set_atomicity_t)dlsym(real_lib_, "MPI_File_set_atomicity");
    REQUIRE_API(MPI_File_set_atomicity)
    MPI_File_get_atomicity =
        (MPI_File_get_atomicity_t)dlsym(real_lib_, "MPI_File_get_atomicity");
    REQUIRE_API(MPI_File_get_atomicity)
    MPI_File_set_view =
        (MPI_File_set_view_t)dlsym(real_lib_, "MPI_File_set_view");
    REQUIRE_API(MPI_File_set_view)
    MPI_File_delete = (MPI_File_delete_t)dlsym(real_lib_, "MPI_File_delete");
    REQUIRE_API(MPI_File_delete)
  }
};
}  // namespace clio::cae

#include "clio_ctp/util/singleton.h"

/** Simplify access to the stateless MpiioFs Singleton */
#define CLIO_CTE_MPIIO_API \
  ctp::Singleton<::clio::cae::MpiioApi>::GetInstance()
#define CLIO_CTE_MPIIO_API_T clio::cae::MpiioApi*

#endif  // CLIO_CTE_ADAPTER_MPIIO_H
