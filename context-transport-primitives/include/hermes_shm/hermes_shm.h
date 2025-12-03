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
#include "types/qtok.h"
#include "types/real_number.h"

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
#include "memory/allocator/allocator.h"
// Following allocator includes deleted during hard refactoring:
// #include "memory/allocator/allocator_factory.h"
// #include "memory/allocator/allocator_factory_.h"
// #include "memory/allocator/gpu_stack_allocator.h"
// #include "memory/allocator/heap.h"
// #include "memory/allocator/malloc_allocator.h"
// #include "memory/allocator/mp_page.h"
// #include "memory/allocator/page_allocator.h"
// #include "memory/allocator/scalable_page_allocator.h"
// #include "memory/allocator/stack_allocator.h"
// #include "memory/allocator/test_allocator.h"
// #include "memory/allocator/thread_local_allocator.h"
#include "memory/backend/array_backend.h"
#include "memory/backend/gpu_malloc.h"
#include "memory/backend/gpu_shm_mmap.h"
#include "memory/backend/malloc_backend.h"
#include "memory/backend/memory_backend.h"
// #include "memory/backend/memory_backend_factory.h"
#include "memory/backend/posix_mmap.h"
#include "memory/backend/posix_shm_mmap.h"
// #include "memory/memory.h"  // Merged into allocator.h

// Data structures (internal templates first, then containers)
// #include "data_structures/all.h"  // Deleted during hard refactoring
// Following internal includes deleted during hard refactoring:
// #include "data_structures/internal/shm_archive.h"
// #include "data_structures/internal/shm_container.h"
// #include "data_structures/internal/shm_internal.h"
// #include "data_structures/internal/shm_macros.h"
#include "data_structures/ipc/algorithm.h"
// Following includes deleted during hard refactoring:
// #include "data_structures/ipc/chararr.h"
// #include "data_structures/ipc/charwrap.h"
// #include "data_structures/ipc/dynamic_queue.h"
// #include "data_structures/ipc/functional.h"
// #include "data_structures/ipc/hash.h"
// #include "data_structures/ipc/key_set.h"
// #include "data_structures/ipc/lifo_list_queue.h"
// #include "data_structures/ipc/list.h"
// #include "data_structures/ipc/mpsc_lifo_list_queue.h"
// #include "data_structures/ipc/multi_ring_buffer.h"
// #include "data_structures/ipc/pair.h"
// #include "data_structures/ipc/ring_ptr_queue.h"
// #include "data_structures/ipc/ring_queue.h"
// #include "data_structures/ipc/ring_queue_flags.h"
// #include "data_structures/ipc/slist.h"
// #include "data_structures/ipc/split_ticket_queue.h"
// #include "data_structures/ipc/spsc_fifo_list_queue.h"
// #include "data_structures/ipc/string.h"
// #include "data_structures/ipc/string_common.h"
// #include "data_structures/ipc/stringstream.h"
// #include "data_structures/ipc/ticket_queue.h"
// #include "data_structures/ipc/tuple_base.h"
// #include "data_structures/ipc/unordered_map.h"
// #include "data_structures/ipc/vector.h"
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