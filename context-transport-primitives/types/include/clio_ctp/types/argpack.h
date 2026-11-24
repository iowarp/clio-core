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

#ifndef CTP_INCLUDE_HSHM_TYPES_ARGPACK_H_
#define CTP_INCLUDE_HSHM_TYPES_ARGPACK_H_

#include <utility>

#include "clio_ctp/constants/macros.h"
#include "numbers.h"
// #include <functional>

namespace ctp {

/** Type which indicates that a constructor takes ArgPacks as input */
struct PiecewiseConstruct {};

/** Type which ends template recurrence */
struct EndTemplateRecurrence {};

/** Recurrence used to create argument pack */
template <size_t idx, typename T = EndTemplateRecurrence, typename... Args>
struct ArgPackRecur {
  T arg_;                                /**< The element stored */
  ArgPackRecur<idx + 1, Args...> recur_; /**< Remaining args */

  /** Default constructor */
  CTP_INLINE_CROSS_FUN ArgPackRecur() {}

  /** Constructor. Rvalue reference. */
  CTP_INLINE_CROSS_FUN explicit ArgPackRecur(T arg, Args &&...args)
      : arg_(std::forward<T>(arg)), recur_(std::forward<Args>(args)...) {}

  /** Forward an rvalue reference (only if argpack) */
  template <size_t i>
  CTP_INLINE_CROSS_FUN constexpr decltype(auto) Forward() {
    if constexpr (i == idx) {
      return std::forward<T>(arg_);
    } else {
      return recur_.template Forward<i>();
    }
  }
};

/** Terminator of the ArgPack recurrence */
template <size_t idx>
struct ArgPackRecur<idx, EndTemplateRecurrence> {
  /** Default constructor */
  CTP_INLINE_CROSS_FUN ArgPackRecur() {}

  /** Forward an rvalue reference (only if argpack) */
  template <size_t i>
  CTP_INLINE_CROSS_FUN constexpr void Forward() {
    // TODO(llogan): fix assert
    STATIC_ASSERT(true, "(Forward) ArgPack index outside of range", void);
  }
};

/** Used to semantically pack arguments */
template <typename... Args>
struct ArgPack {
  /** Variable argument pack */
  ArgPackRecur<0, Args...> recur_;
  /** Size of the argpack */
  static constexpr const size_t size_ = sizeof...(Args);

  /** General Constructor. */
  CTP_INLINE_CROSS_FUN ArgPack(Args &&...args)  // NOLINT
      : recur_(std::forward<Args>(args)...) {}

  /** Get forward reference */
  template <size_t idx>
  CTP_INLINE_CROSS_FUN constexpr decltype(auto) Forward() {
    return recur_.template Forward<idx>();
  }

  /** Size */
  CTP_INLINE_CROSS_FUN constexpr static size_t Size() { return size_; }
};

/** Make an argpack */
template <typename... Args>
CTP_INLINE_CROSS_FUN constexpr ArgPack<Args &&...> make_argpack(
    Args &&...args) {
  return ArgPack<Args &&...>(std::forward<Args>(args)...);
}

/** Get the type + reference of the forward for \a pack pack at \a index i */
#define FORWARD_ARGPACK_FULL_TYPE(pack, i) decltype(pack.template Forward<i>())

/** Forward the param for \a pack pack at \a index i */
#define FORWARD_ARGPACK_PARAM(pack, i) \
  std::forward<FORWARD_ARGPACK_FULL_TYPE(pack, i)>(pack.template Forward<i>())

/** Forward an argpack */
#define FORWARD_ARGPACK(pack) std::forward<decltype(pack)>(pack)

/** Used to pass an argument pack to a function or class method */
class PassArgPack {
 public:
  /** Call function with ArgPack */
  template <typename ArgPackT, typename F>
  CTP_INLINE_CROSS_FUN constexpr static decltype(auto) Call(ArgPackT &&pack,
                                                             F &&f) {
    return _CallRecur<0, ArgPackT, F>(std::forward<ArgPackT>(pack),
                                      std::forward<F>(f));
  }

 private:
  /** Unpacks the ArgPack and passes it to the function */
  template <size_t i, typename ArgPackT, typename F, typename... CurArgs>
  CTP_INLINE_CROSS_FUN constexpr static decltype(auto) _CallRecur(
      ArgPackT &&pack, F &&f, CurArgs &&...args) {
    if constexpr (i < ArgPackT::Size()) {
      return _CallRecur<i + 1, ArgPackT, F>(
          std::forward<ArgPackT>(pack), std::forward<F>(f),
          std::forward<CurArgs>(args)..., FORWARD_ARGPACK_PARAM(pack, i));
    } else {
      if constexpr (std::is_void_v<decltype(f(
                        std::forward<CurArgs>(args)...))>) {
        f(std::forward<CurArgs>(args)...);
      } else {
        return f(std::forward<CurArgs>(args)...);
      }
    }
  }
};

/** Combine multiple argpacks into a single argpack */
class MergeArgPacks {
 public:
  /** Call function with ArgPack */
  template <typename... ArgPacks>
  CTP_INLINE_CROSS_FUN constexpr static decltype(auto) Merge(
      ArgPacks &&...packs) {
    return _MergePacksRecur<0>(make_argpack(std::forward<ArgPacks>(packs)...));
  }

 private:
  /** Unpacks the C++ parameter pack of ArgPacks */
  template <size_t cur_pack, typename ArgPacksT, typename... CurArgs>
  CTP_INLINE_CROSS_FUN constexpr static decltype(auto) _MergePacksRecur(
      ArgPacksT &&packs, CurArgs &&...args) {
    if constexpr (cur_pack < ArgPacksT::Size()) {
      return _MergeRecur<cur_pack, ArgPacksT, 0>(
          // End template parameters
          std::forward<ArgPacksT>(packs),
          FORWARD_ARGPACK_PARAM(packs, cur_pack),
          std::forward<CurArgs>(args)...);
    } else {
      return make_argpack(std::forward<CurArgs>(args)...);
    }
  }

  /** Unpacks the C++ parameter pack of ArgPacks */
  template <size_t cur_pack, typename ArgPacksT, size_t i, typename ArgPackT,
            typename... CurArgs>
  CTP_INLINE_CROSS_FUN constexpr static decltype(auto) _MergeRecur(
      ArgPacksT &&packs, ArgPackT &&pack, CurArgs &&...args) {
    if constexpr (i < ArgPackT::Size()) {
      return _MergeRecur<cur_pack, ArgPacksT, i + 1, ArgPackT>(
          std::forward<ArgPacksT>(packs), std::forward<ArgPackT>(pack),
          std::forward<CurArgs>(args)..., FORWARD_ARGPACK_PARAM(pack, i));
    } else {
      return _MergePacksRecur<cur_pack + 1, ArgPacksT>(
          std::forward<ArgPacksT>(packs), std::forward<CurArgs>(args)...);
    }
  }
};

/** Insert an argpack at the head of each pack in a set of ArgPacks */
class ProductArgPacks {
 public:
  /** The product function */
  template <typename ProductPackT, typename... ArgPacks>
  CTP_INLINE_CROSS_FUN constexpr static decltype(auto) Product(
      ProductPackT &&prod_pack, ArgPacks &&...packs) {
    return _ProductPacksRecur<0>(
        std::forward<ProductPackT>(prod_pack),
        make_argpack(std::forward<ArgPacks>(packs)...));
  }

 private:
  /** Prepend \a ArgPack prod_pack to every ArgPack in orig_packs */
  template <size_t cur_pack, typename ProductPackT, typename OrigPacksT,
            typename... NewPacks>
  CTP_INLINE_CROSS_FUN constexpr static decltype(auto) _ProductPacksRecur(
      ProductPackT &&prod_pack, OrigPacksT &&orig_packs, NewPacks &&...packs) {
    if constexpr (cur_pack < OrigPacksT::Size()) {
      return _ProductPacksRecur<cur_pack + 1>(
          std::forward<ProductPackT>(prod_pack),
          std::forward<OrigPacksT>(orig_packs),
          std::forward<NewPacks>(packs)...,
          std::forward<ProductPackT>(prod_pack),
          FORWARD_ARGPACK_PARAM(orig_packs, cur_pack));
    } else {
      return make_argpack(std::forward<NewPacks>(packs)...);
    }
  }
};

/** Used to emulate constexpr to lambda */
template <typename T, T Val>
struct MakeConstexpr {
  constexpr static T val_ = Val;
  CTP_INLINE_CROSS_FUN constexpr static T Get() { return val_; }
};

/** Apply a function over an entire TupleBase / tuple */
template <bool reverse>
class IterateArgpack {
 public:
  /** Apply a function to every element of a tuple */
  template <typename TupleT, typename F>
  CTP_INLINE_CROSS_FUN constexpr static void Apply(TupleT &&pack, F &&f) {
    _Apply<0, TupleT, F>(std::forward<TupleT>(pack), std::forward<F>(f));
  }

 private:
  /** Apply the function recursively */
  template <size_t i, typename TupleT, typename F>
  CTP_INLINE_CROSS_FUN constexpr static void _Apply(TupleT &&pack, F &&f) {
    if constexpr (i < TupleT::Size()) {
      if constexpr (reverse) {
        _Apply<i + 1, TupleT, F>(std::forward<TupleT>(pack),
                                 std::forward<F>(f));
        f(MakeConstexpr<size_t, i>(), pack.template Forward<i>());
      } else {
        f(MakeConstexpr<size_t, i>(), pack.template Forward<i>());
        _Apply<i + 1, TupleT, F>(std::forward<TupleT>(pack),
                                 std::forward<F>(f));
      }
    }
  }
};

/** Forward iterate over tuple and apply function  */
using ForwardIterateArgpack = IterateArgpack<false>;

/** Reverse iterate over tuple and apply function */
using ReverseIterateArgpack = IterateArgpack<true>;

}  // namespace ctp

#endif  // CTP_INCLUDE_HSHM_TYPES_ARGPACK_H_
