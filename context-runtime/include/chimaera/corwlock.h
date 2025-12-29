#ifndef CHIMAERA_INCLUDE_CHIMAERA_CORWLOCK_H_
#define CHIMAERA_INCLUDE_CHIMAERA_CORWLOCK_H_

#include <hermes_shm/thread/lock/rwlock.h>

namespace chi {

/**
 * CoRwLock - Simple wrapper around hshm::RwLock
 */
using CoRwLock = hshm::RwLock;

/**
 * ScopedCoRwReadLock - RAII read lock wrapper with default owner
 * Wraps hshm::ScopedRwReadLock with a default owner value of 0
 */
struct ScopedCoRwReadLock {
  hshm::ScopedRwReadLock scoped_lock_;

  /** Acquire the read lock with default owner */
  explicit ScopedCoRwReadLock(CoRwLock& lock)
      : scoped_lock_(lock, 0) {}

  /** Release handled by scoped_lock_ destructor */
  ~ScopedCoRwReadLock() = default;
};

/**
 * ScopedCoRwWriteLock - RAII write lock wrapper with default owner
 * Wraps hshm::ScopedRwWriteLock with a default owner value of 0
 */
struct ScopedCoRwWriteLock {
  hshm::ScopedRwWriteLock scoped_lock_;

  /** Acquire the write lock with default owner */
  explicit ScopedCoRwWriteLock(CoRwLock& lock)
      : scoped_lock_(lock, 0) {}

  /** Release handled by scoped_lock_ destructor */
  ~ScopedCoRwWriteLock() = default;
};

}  // namespace chi

#endif  // CHIMAERA_INCLUDE_CHIMAERA_CORWLOCK_H_
