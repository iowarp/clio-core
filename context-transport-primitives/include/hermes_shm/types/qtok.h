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

#ifndef HSHM_SHM_INCLUDE_HSHM_SHM_DATA_STRUCTURES_IPC_QUEUE_H_
#define HSHM_SHM_INCLUDE_HSHM_SHM_DATA_STRUCTURES_IPC_QUEUE_H_

#include "numbers.h"

namespace hshm {

/** Represents the internal qtok type */
typedef hshm::size_t qtok_id;
typedef qtok_id _qtok_t;

/** Represents a ticket in the queue */
struct qtok_t {
  qtok_id id_;

  /** Default constructor */
  qtok_t() = default;

  /** Emplace constructor */
  HSHM_INLINE_CROSS_FUN explicit qtok_t(qtok_id id) : id_(id) {}

  /** Copy constructor */
  HSHM_INLINE_CROSS_FUN qtok_t(const qtok_t &other) : id_(other.id_) {}

  /** Copy assign */
  HSHM_INLINE_CROSS_FUN qtok_t &operator=(const qtok_t &other) {
    if (this != &other) {
      id_ = other.id_;
    }
    return *this;
  }

  /** Move constructor */
  HSHM_INLINE_CROSS_FUN qtok_t(qtok_t &&other) : id_(other.id_) {
    other.SetNull();
  }

  /** Move assign */
  HSHM_INLINE_CROSS_FUN qtok_t &operator=(qtok_t &&other) {
    if (this != &other) {
      id_ = other.id_;
      other.SetNull();
    }
    return *this;
  }

  /** Set to the null qtok */
  HSHM_INLINE_CROSS_FUN void SetNull() { id_ = qtok_t::GetNull().id_; }

  /** Get the null qtok */
  HSHM_INLINE_CROSS_FUN static qtok_t GetNull() {
    return qtok_t(std::numeric_limits<qtok_id>::max());
  }

  /** Check if null */
  HSHM_INLINE_CROSS_FUN bool IsNull() const { return id_ == GetNull().id_; }
};

}  // namespace hshm

#endif  // HSHM_SHM_INCLUDE_HSHM_SHM_DATA_STRUCTURES_IPC_QUEUE_H_
