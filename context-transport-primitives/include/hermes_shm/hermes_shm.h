/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Distributed under BSD 3-Clause license.                                   *
 * Copyright by The HDF Group.                                               *
 * Copyright by the Illinois Institute of Technology.                        *
 * All rights reserved.                                                      *
 *                                                                           *
 * This file is part of Hermes. The full Hermes copyright notice, including  *
 * terms governing use, modification, and redistribution, is contained in    *
 * the COPYING file, which can be found at the top directory. If you do not  *
 * have access to the file, you may request a copy from help@hdfgroup.org.   *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

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
#include "util/compress/blosc.h"
#include "util/compress/brotli.h"
#include "util/compress/bzip2.h"
#include "util/compress/compress.h"
#include "util/compress/compress_factory.h"
#include "util/compress/lz4.h"
#include "util/compress/lzma.h"
#include "util/compress/lzo.h"
#include "util/compress/snappy.h"
#include "util/compress/zlib.h"
#include "util/compress/zstd.h"

// Encryption utilities (guarded by HSHM_ENABLE_ENCRYPT)
#include "util/encrypt/aes.h"
#include "util/encrypt/encrypt.h"

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

// Lightbeam transport layer (guarded by respective HSHM_ENABLE_* macros)
#include "lightbeam/lightbeam.h"
#include "lightbeam/zmq_transport.h"

#endif  // HSHM_SHM_INCLUDE_HSHM_SHM_HSHM_SHM_H_