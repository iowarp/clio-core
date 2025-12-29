#ifndef CHIMAERA_INCLUDE_CHIMAERA_COMUTEX_H_
#define CHIMAERA_INCLUDE_CHIMAERA_COMUTEX_H_

#include <hermes_shm/thread/lock/mutex.h>

namespace chi {

/**
 * CoMutex - Simple wrapper around hshm::Mutex
 */
using CoMutex = hshm::Mutex;

/**
 * ScopedCoMutex - RAII mutex wrapper with default owner
 * Wraps hshm::ScopedMutex with a default owner value of 0
 */
struct ScopedCoMutex {
  hshm::ScopedMutex scoped_mutex_;

  /** Acquire the mutex with default owner */
  explicit ScopedCoMutex(CoMutex& lock)
      : scoped_mutex_(lock, 0) {}

  /** Release handled by scoped_mutex_ destructor */
  ~ScopedCoMutex() = default;
};

}  // namespace chi

#endif  // CHIMAERA_INCLUDE_CHIMAERA_COMUTEX_H_
