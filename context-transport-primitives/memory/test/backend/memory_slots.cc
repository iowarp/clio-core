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

#include <mpi.h>

#include <iostream>

#include "basic_test.h"
#include "clio_ctp/memory/backend/posix_shm_mmap.h"

using ctp::ipc::PosixShmMmap;

TEST_CASE("MemorySlot") {
  int rank;
  char nonce = 8;
  std::string shm_url = "test_mem_backend";
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  CTP_ERROR_HANDLE_START()

  PosixShmMmap backend;
  if (rank == 0) {
    {
      std::cout << "Creating SHMEM (rank 0)" << std::endl;
      if (!backend.shm_init(ctp::ipc::MemoryBackendId::GetRoot(),
                            ctp::Unit<size_t>::Megabytes(1), shm_url)) {
        throw std::runtime_error("Couldn't create backend");
      }
      std::cout << "Backend data: " << (void *)backend.data_ << std::endl;
      std::cout << "Backend sz: " << backend.data_capacity_ << std::endl;
      memset(backend.data_, nonce, backend.data_capacity_);
      std::cout << "Wrote backend data" << std::endl;
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank != 0) {
    {
      std::cout << "Attaching SHMEM (rank 1)" << std::endl;
      backend.shm_attach(shm_url);
      char *ptr = backend.data_;
      REQUIRE(VerifyBuffer(ptr, backend.data_capacity_, nonce));
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    {
      std::cout << "Destroying shmem (rank 1)" << std::endl;
    }
  }

  CTP_ERROR_HANDLE_END()
}
