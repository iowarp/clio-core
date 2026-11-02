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

#ifndef CTP_SHM_INCLUDE_HSHM_SHM_HSHM_SHM_H_
#define CTP_SHM_INCLUDE_HSHM_SHM_HSHM_SHM_H_

// Comprehensive include for all clio_ctp headers
// Since all headers now have proper compile-time guards, this is safe to
// include

// Core constants and macros
#include "clio_ctp/constants/macros.h"

// Basic types (foundation dependencies)
#include "clio_ctp/types/argpack.h"
#include "clio_ctp/types/atomic.h"
#include "clio_ctp/types/bitfield.h"
#include "clio_ctp/types/hash.h"
#include "clio_ctp/types/numbers.h"

// Utilities (low-level support)
#include "clio_ctp/util/affinity.h"
#include "clio_ctp/util/auto_trace.h"
#include "clio_ctp/util/config_parse.h"
#include "clio_ctp/util/error.h"
#include "clio_ctp/util/errors.h"
#include "clio_ctp/util/formatter.h"
#include "clio_ctp/util/gpu_api.h"
#include "clio_ctp/util/logging.h"
#include "clio_ctp/util/random.h"
#include "clio_ctp/util/real_api.h"
#include "clio_ctp/util/singleton.h"
#include "clio_ctp/util/timer.h"
#include "clio_ctp/util/timer_mpi.h"
#include "clio_ctp/util/timer_thread.h"
#include "clio_ctp/util/type_switch.h"

// Compression utilities (guarded by CTP_ENABLE_COMPRESS)
#include "clio_ctp/compress/blosc.h"
#include "clio_ctp/compress/brotli.h"
#include "clio_ctp/compress/bzip2.h"
#include "clio_ctp/compress/compress.h"
#include "clio_ctp/compress/compress_factory.h"
#include "clio_ctp/compress/lz4.h"
#include "clio_ctp/compress/lzma.h"
#include "clio_ctp/compress/lzo.h"
#include "clio_ctp/compress/snappy.h"
#include "clio_ctp/compress/zlib.h"
#include "clio_ctp/compress/zstd.h"

// Encryption utilities (guarded by CTP_ENABLE_ENCRYPT)
#include "clio_ctp/encrypt/aes.h"
#include "clio_ctp/encrypt/encrypt.h"

// Thread models and synchronization (guarded by respective CTP_ENABLE_*
// macros)
#include "clio_ctp/thread/lock.h"
#include "clio_ctp/thread/lock/mutex.h"
#include "clio_ctp/thread/lock/rwlock.h"
#include "clio_ctp/thread/lock/spin_lock.h"
#include "clio_ctp/thread/thread_model/argobots.h"
#include "clio_ctp/thread/thread_model/cuda.h"
#include "clio_ctp/thread/thread_model/pthread.h"
#include "clio_ctp/thread/thread_model/rocm.h"
#include "clio_ctp/thread/thread_model/std_thread.h"
#include "clio_ctp/thread/thread_model/thread_model.h"
#include "clio_ctp/thread/thread_model_manager.h"

// Memory management
// Allocators (memory allocation strategies)
#include "clio_ctp/memory/allocator/allocator.h"
#include "clio_ctp/memory/allocator/arena_allocator.h"
#include "clio_ctp/memory/allocator/buddy_allocator.h"
#include "clio_ctp/memory/allocator/heap.h"
#include "clio_ctp/memory/allocator/mp_allocator.h"

// Memory backends (low-level memory management implementations)
#include "clio_ctp/memory/backend/array_backend.h"
#include "clio_ctp/memory/backend/gpu_malloc.h"
#include "clio_ctp/memory/backend/gpu_shm_mmap.h"
#include "clio_ctp/memory/backend/malloc_backend.h"
#include "clio_ctp/memory/backend/memory_backend.h"
#include "clio_ctp/memory/backend/posix_mmap.h"
#include "clio_ctp/memory/backend/posix_shm_mmap.h"

// Data structures
// IPC data structures (inter-process communication containers)
#include "clio_ctp/data_structures/ipc/algorithm.h"
#include "clio_ctp/data_structures/ipc/multi_ring_buffer.h"
#include "clio_ctp/data_structures/ipc/rb_tree_pre.h"
#include "clio_ctp/data_structures/ipc/ring_buffer.h"
#include "clio_ctp/data_structures/ipc/shm_container.h"
#include "clio_ctp/data_structures/ipc/slist_pre.h"
#include "clio_ctp/data_structures/ipc/vector.h"

// Private data structures (single-process containers)
#include "clio_ctp/data_structures/priv/array_vector.h"
#include "clio_ctp/data_structures/priv/fixed_string.h"
#include "clio_ctp/data_structures/priv/string.h"
#include "clio_ctp/data_structures/priv/vector.h"
#include "clio_ctp/data_structures/priv/unordered_map_ll.h"

// Serialization support
#include "clio_ctp/data_structures/serialization/local_serialize.h"
#include "clio_ctp/data_structures/serialization/serialize_common.h"

// System introspection
#include "clio_ctp/introspect/system_info.h"

// Solver functionality
#include "clio_ctp/solver/nonlinear_least_squares.h"

// Lightbeam transport layer (base types always available)
#include "clio_ctp/lightbeam/lightbeam.h"
// Concrete transports + factory (only when ctp::lightbeam is linked)
#include "clio_ctp/lightbeam/transport_factory_impl.h"

#endif  // CTP_SHM_INCLUDE_HSHM_SHM_HSHM_SHM_H_