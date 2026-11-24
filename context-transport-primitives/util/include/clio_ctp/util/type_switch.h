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

#ifndef CTP_SHM_INCLUDE_HSHM_SHM_UTIL_STATIC_SWITCH_H_
#define CTP_SHM_INCLUDE_HSHM_SHM_UTIL_STATIC_SWITCH_H_

#include <functional>

namespace ctp {

/** Ends the recurrence of the switch-case */
class EndTypeSwitch {};

/**
 * A compile-time switch-case statement used for choosing
 * a type based on another type
 *
 * @param T the type being checked (i.e., switch (T)
 * @param Default the default case
 * @param Case a case of the switch (i.e., case Case:)
 * @param Val the body of the case
 * */
template <typename T, typename Default, typename Case = EndTypeSwitch,
          typename Val = EndTypeSwitch, typename... Args>
struct type_switch {
  typedef typename std::conditional<
      std::is_same_v<T, Case>, Val,
      typename type_switch<T, Default, Args...>::type>::type type;
};

/** The default case */
template <typename T, typename Default>
struct type_switch<T, Default, EndTypeSwitch, EndTypeSwitch> {
  typedef Default type;
};

}  // namespace ctp

#endif  // CTP_SHM_INCLUDE_HSHM_SHM_UTIL_STATIC_SWITCH_H_
