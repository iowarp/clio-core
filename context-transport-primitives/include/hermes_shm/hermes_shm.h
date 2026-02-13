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

#ifndef HSHM_SHM_INCLUDE_HSHM_SHM_HSHM_SHM_H_
#define HSHM_SHM_INCLUDE_HSHM_SHM_HSHM_SHM_H_

// Comprehensive include for all hermes_shm headers
// Since all headers now have proper compile-time guards, this is safe to
// include

// Core constants and macros
#include "constants/macros.h"

// Basic types (foundation dependencies)
#include "types/argpack.h"
#include "types/atomic.h"
#include "types/bitfield.h"
#include "types/hash.h"
#include "types/numbers.h"

// Utilities (low-level support)
#include "util/affinity.h"
#include "util/auto_trace.h"
#include "util/config_parse.h"
#include "util/error.h"
#include "util/errors.h"
#include "util/formatter.h"
#include "util/gpu_api.h"
#include "util/logging.h"
#include "util/random.h"
#include "util/real_api.h"
#include "util/singleton.h"
#include "util/timer.h"
#include "util/timer_mpi.h"
#include "util/timer_thread.h"
#include "util/type_switch.h"

// Compression utilities (guarded by HSHM_ENABLE_COMPRESS)
#include "compress/blosc.h"
#include "compress/brotli.h"
#include "compress/bzip2.h"
#include "compress/compress.h"
#include "compress/compress_factory.h"
#include "compress/lz4.h"
#include "compress/lzma.h"
#include "compress/lzo.h"
#include "compress/snappy.h"
#include "compress/zlib.h"
#include "compress/zstd.h"

// Encryption utilities (guarded by HSHM_ENABLE_ENCRYPT)
#include "encrypt/aes.h"
#include "encrypt/encrypt.h"

// Thread models and synchronization (guarded by respective HSHM_ENABLE_*
// macros)
#include "thread/lock.h"
#include "thread/lock/mutex.h"
#include "thread/lock/rwlock.h"
#include "thread/lock/spin_lock.h"
#include "thread/thread_model/argobots.h"
#include "thread/thread_model/cuda.h"
#include "thread/thread_model/pthread.h"
#include "thread/thread_model/rocm.h"
#include "thread/thread_model/std_thread.h"
#include "thread/thread_model/thread_model.h"
#include "thread/thread_model_manager.h"

// Memory management
// Allocators (memory allocation strategies)
#include "memory/allocator/allocator.h"
#include "memory/allocator/arena_allocator.h"
#include "memory/allocator/buddy_allocator.h"
#include "memory/allocator/heap.h"
#include "memory/allocator/mp_allocator.h"

// Memory backends (low-level memory management implementations)
#include "memory/backend/array_backend.h"
#include "memory/backend/gpu_malloc.h"
#include "memory/backend/gpu_shm_mmap.h"
#include "memory/backend/malloc_backend.h"
#include "memory/backend/memory_backend.h"
#include "memory/backend/posix_mmap.h"
#include "memory/backend/posix_shm_mmap.h"

// Data structures
// IPC data structures (inter-process communication containers)
#include "data_structures/ipc/algorithm.h"
#include "data_structures/ipc/multi_ring_buffer.h"
#include "data_structures/ipc/rb_tree_pre.h"
#include "data_structures/ipc/ring_buffer.h"
#include "data_structures/ipc/shm_container.h"
#include "data_structures/ipc/slist_pre.h"
#include "data_structures/ipc/vector.h"

// Private data structures (single-process containers)
#include "data_structures/priv/string.h"
#include "data_structures/priv/vector.h"

// Serialization support
#include "data_structures/serialization/local_serialize.h"
#include "data_structures/serialization/serialize_common.h"

// System introspection
#include "introspect/system_info.h"

// Solver functionality
#include "solver/nonlinear_least_squares.h"

// Lightbeam transport layer
#include "lightbeam/transport_factory_impl.h"

#endif  // HSHM_SHM_INCLUDE_HSHM_SHM_HSHM_SHM_H_