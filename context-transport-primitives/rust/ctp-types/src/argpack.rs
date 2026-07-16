// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).

//! Compile-time argument packs — port of `clio_ctp/types/argpack.h`.
//!
//! The C++ header builds a variadic `ArgPack<Args...>` out of a hand-rolled
//! recursion (`ArgPackRecur`) because C++ needs one. Rust already has the
//! thing that recursion was emulating: the tuple. So the port keeps the C++
//! *operations* (pass, merge, product, iterate) and drops the machinery —
//! an "argpack" here is a plain tuple `(A, B, C)`, and the operations are
//! traits implemented for tuples up to arity 12.
//!
//! # C++ → Rust name mapping
//!
//! | C++ (`clio_ctp/types/argpack.h`) | Rust (`ctp_types::argpack`) |
//! |---|---|
//! | `ctp::ArgPack<Args...>` | native tuple `(A, B, ...)`, described by trait [`ArgPack`] |
//! | `ctp::ArgPack<>` | `()` |
//! | `ctp::make_argpack(a, b)` | [`argpack!`]`(a, b)` |
//! | `pack.Size()`, `ArgPack::size_` | [`ArgPack::SIZE`] (const), [`ArgPack::size`] |
//! | `pack.Forward<I>()` | [`ArgPackGet::get`] / [`get_mut`](ArgPackGet::get_mut) / [`into_arg`](ArgPackGet::into_arg), or native `pack.0` |
//! | `ctp::PassArgPack::Call(pack, f)` | [`PassArgPack::call`]`(pack, f)`, i.e. [`CallArgs::call`] |
//! | `ctp::MergeArgPacks::Merge(a, b)` | [`MergeArgPacks::merge`]`(a, b)`, i.e. [`MergeArgPack::merge`] |
//! | `ctp::MergeArgPacks::Merge(a, b, c, ...)` | [`merge_argpacks!`]`(a, b, c, ...)` |
//! | `ctp::ProductArgPacks::Product(p, a, b)` | [`product_argpacks!`]`(p, a, b)` |
//! | `ctp::ForwardIterateArgpack::Apply(pack, f)` | [`ForEachArg::for_each`], [`ForEachArgMut::for_each_mut`] |
//! | `ctp::ReverseIterateArgpack::Apply(pack, f)` | [`ForEachArg::for_each_rev`], [`ForEachArgMut::for_each_mut_rev`] |
//! | `ctp::IterateArgpack<reverse>` | the `_rev` method pair (no `bool` type parameter) |
//! | generic lambda `[](auto i, auto &arg)` | an impl of [`VisitArg`] / [`VisitArgMut`] |
//! | `ctp::MakeConstexpr<T, Val>` | [`MakeConstexpr`]`<const VAL: usize>` (usize only) |
//! | `ctp::PiecewiseConstruct` | [`PiecewiseConstruct`] (marker; vestigial, see below) |
//! | `ctp::ArgPackRecur`, `ctp::EndTemplateRecurrence` | *(none — tuples are native)* |
//! | `FORWARD_ARGPACK`, `FORWARD_ARGPACK_PARAM`, `FORWARD_ARGPACK_FULL_TYPE` | *(none — no perfect forwarding)* |
//!
//! # Semantic divergences
//!
//! 1. **Arity is bounded.** C++ variadic templates are unbounded; Rust has no
//!    variadic generics, so impls are macro-generated for arities `0..=12`
//!    (matching std's own tuple trait impls). [`MergeArgPack`] is implemented
//!    for every left/right pair whose *combined* arity is `<= 12`. Exceeding
//!    these is a compile error, not silent truncation.
//! 2. **Generic lambdas become visitor traits.** C++ `Apply` takes one generic
//!    lambda and instantiates it per element type. A Rust closure is
//!    monomorphic and there is no stable `for<T> FnMut(&T)`, so callers
//!    implement [`VisitArg<T>`](VisitArg) — usually once, blanket, over the
//!    bound they actually need (`impl<T: Display> VisitArg<T> for MyFmt`).
//!    That reproduces the per-type instantiation the C++ lambda got for free.
//! 3. **The visitor index is a runtime `usize`**, not the compile-time
//!    `MakeConstexpr<size_t, i>` token C++ passes. Every real consumer
//!    (`util/formatter.h`, `serialization/{local,global}_serialize.h`) either
//!    ignores the index or uses it as a value, so nothing is lost.
//!    [`MakeConstexpr`] is still ported for API parity, but only over `usize`:
//!    Rust const generics accept integers/`bool`/`char`, not arbitrary `T`.
//! 4. **Out-of-range index is a compile error.** The C++ recursion terminator
//!    guards with `STATIC_ASSERT(true, ...)` — a tautology, so it never fires
//!    and `Forward<OOB>()` silently returns `void`. Rust simply has no
//!    `ArgPackGet<I>` impl past the end, so the mistake fails to compile. This
//!    is deliberately stricter than the C++.
//! 5. **No perfect forwarding / reference collapsing.** C++ `make_argpack`
//!    yields `ArgPack<Args&&...>`, a pack of *references* that never copies or
//!    moves its inputs (see the C++ `TestArgpackCopy` case). [`argpack!`]
//!    instead moves its arguments into the tuple by value. The "no copy"
//!    guarantee survives (a move is not a copy, and non-`Copy` types cannot be
//!    silently duplicated); callers wanting the C++ reference semantics pass
//!    `&x` / `&mut x` explicitly and the tuple holds `&T` / `&mut T`.
//! 6. **`Forward<I>()` has no exact analog.** In C++ it hands out an rvalue
//!    reference while leaving the pack alive. [`ArgPackGet::into_arg`]
//!    consumes the whole pack (the other elements are dropped); use
//!    [`get`](ArgPackGet::get) / [`get_mut`](ArgPackGet::get_mut) for
//!    non-consuming access.
//! 7. **`ProductArgPacks` clones.** The C++ `_ProductPacksRecur` forwards the
//!    *same* `prod_pack` rvalue into every output slot — an aliasing /
//!    double-move hazard that only survives because nothing ever moves from
//!    it. [`product_argpacks!`] requires `Clone` and clones per slot (and
//!    evaluates the product expression exactly once), which is the safe
//!    reading of that intent.
//! 8. **No void/non-void split.** `PassArgPack::_CallRecur` branches on
//!    `std::is_void_v` because C++ cannot `return f(...)` when `f` returns
//!    void. In Rust `()` is an ordinary value, so [`CallArgs::call`] is one
//!    path.
//! 9. **[`PiecewiseConstruct`] is vestigial.** It tags C++ constructor
//!    overloads that take argpacks; Rust has no constructor overloading and
//!    uses distinct named constructors instead. Ported as a ZST marker purely
//!    so the C++ name resolves during the migration.
//! 10. **One implementation covers `ArgPack` and `tuple`.** C++ applies
//!     `IterateArgpack` to both `ArgPack` and `TupleBase` (they both expose
//!     `Size()`/`Forward<i>()`); in Rust both collapse onto the tuple, so
//!     `data_structures/ipc/tuple_base.h` needs no separate iteration port.

// ---------------------------------------------------------------------------
// Markers
// ---------------------------------------------------------------------------

/// Indicates that a constructor takes argpacks as input (`ctp::PiecewiseConstruct`).
///
/// Vestigial in Rust — see divergence 9 in the module docs. Kept so ported
/// call sites can name the tag while the C++ and Rust trees coexist.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PiecewiseConstruct;

/// Emulates a constexpr value usable as a lambda argument (`ctp::MakeConstexpr<T, Val>`).
///
/// Restricted to `usize`, which is the only instantiation the C++ uses
/// (`MakeConstexpr<size_t, i>`); Rust const generics cannot be generic over an
/// arbitrary `T`. See divergence 3.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MakeConstexpr<const VAL: usize>;

impl<const VAL: usize> MakeConstexpr<VAL> {
    /// The wrapped value (`MakeConstexpr::val_`).
    pub const VAL: usize = VAL;

    /// Mirrors `MakeConstexpr::Get()`.
    #[inline]
    pub const fn get() -> usize {
        VAL
    }
}

// ---------------------------------------------------------------------------
// ArgPack: the tuple-as-argpack description
// ---------------------------------------------------------------------------

/// A pack of arguments (`ctp::ArgPack<Args...>`), implemented for tuples 0..=12.
pub trait ArgPack {
    /// Number of elements (`ArgPack::size_`).
    const SIZE: usize;

    /// Mirrors `ArgPack::Size()`.
    #[inline]
    fn size(&self) -> usize {
        Self::SIZE
    }
}

/// Indexed element access (`pack.Forward<I>()`).
///
/// Out-of-range `I` has no impl and therefore fails to compile — see
/// divergence 4.
pub trait ArgPackGet<const I: usize> {
    /// Type of the element at index `I`.
    type Item;

    /// Borrow the element at index `I`.
    fn get(&self) -> &Self::Item;

    /// Mutably borrow the element at index `I`.
    fn get_mut(&mut self) -> &mut Self::Item;

    /// Move the element at index `I` out, consuming the pack (divergence 6).
    fn into_arg(self) -> Self::Item;
}

// ---------------------------------------------------------------------------
// PassArgPack
// ---------------------------------------------------------------------------

/// Calls a function with the pack's elements as its arguments.
///
/// The trait behind [`PassArgPack::call`]; `Self` is the pack, `F` the callee.
pub trait CallArgs<F> {
    /// The callee's return type.
    type Output;

    /// Mirrors `PassArgPack::Call(pack, f)`.
    fn call(self, f: F) -> Self::Output;
}

/// Used to pass an argument pack to a function or method (`ctp::PassArgPack`).
#[derive(Debug, Default, Clone, Copy)]
pub struct PassArgPack;

impl PassArgPack {
    /// Mirrors `ctp::PassArgPack::Call`.
    #[inline]
    pub fn call<P, F>(pack: P, f: F) -> P::Output
    where
        P: CallArgs<F>,
    {
        pack.call(f)
    }
}

// ---------------------------------------------------------------------------
// MergeArgPacks
// ---------------------------------------------------------------------------

/// Concatenates two packs (`ctp::MergeArgPacks::Merge`).
///
/// Implemented for every pair whose combined arity is `<= 12` (divergence 1).
pub trait MergeArgPack<Rhs> {
    /// The concatenated pack.
    type Output;

    /// Concatenate `self` with `rhs`, preserving order.
    fn merge(self, rhs: Rhs) -> Self::Output;
}

/// Combine multiple argpacks into a single argpack (`ctp::MergeArgPacks`).
///
/// For more than two packs use [`merge_argpacks!`], which folds this pairwise.
#[derive(Debug, Default, Clone, Copy)]
pub struct MergeArgPacks;

impl MergeArgPacks {
    /// Mirrors `ctp::MergeArgPacks::Merge(a, b)`.
    #[inline]
    pub fn merge<A, B>(a: A, b: B) -> A::Output
    where
        A: MergeArgPack<B>,
    {
        a.merge(b)
    }
}

// ---------------------------------------------------------------------------
// IterateArgpack
// ---------------------------------------------------------------------------

/// Visits one element of a pack by shared reference.
///
/// The stable stand-in for the C++ generic lambda `[](auto i, auto &arg)`.
/// Implement it once per element type, or blanket over a bound:
///
/// ```
/// use ctp_types::argpack::VisitArg;
/// use std::fmt::Display;
///
/// struct Fmt(String);
/// impl<T: Display + ?Sized> VisitArg<T> for Fmt {
///     fn visit(&mut self, _index: usize, arg: &T) {
///         self.0 += &arg.to_string();
///     }
/// }
/// ```
pub trait VisitArg<T: ?Sized> {
    /// Mirrors one `f(MakeConstexpr<size_t, i>(), arg)` invocation.
    fn visit(&mut self, index: usize, arg: &T);
}

/// Visits one element of a pack by mutable reference.
///
/// Mirrors the C++ lambdas that take `auto &arg` and write through it — e.g.
/// `LocalDeserialize::operator()`.
pub trait VisitArgMut<T: ?Sized> {
    /// Mirrors one `f(MakeConstexpr<size_t, i>(), arg)` invocation.
    fn visit_mut(&mut self, index: usize, arg: &mut T);
}

/// Applies a visitor to every element of a pack (`ctp::IterateArgpack`).
pub trait ForEachArg<V> {
    /// Forward order, index `0` first (`ctp::ForwardIterateArgpack::Apply`).
    fn for_each(&self, visitor: &mut V);

    /// Reverse order, index `SIZE - 1` first (`ctp::ReverseIterateArgpack::Apply`).
    fn for_each_rev(&self, visitor: &mut V);
}

/// Applies a mutating visitor to every element of a pack.
pub trait ForEachArgMut<V> {
    /// Forward order, index `0` first.
    fn for_each_mut(&mut self, visitor: &mut V);

    /// Reverse order, index `SIZE - 1` first.
    fn for_each_mut_rev(&mut self, visitor: &mut V);
}

// ---------------------------------------------------------------------------
// Public macros
// ---------------------------------------------------------------------------

/// Makes an argpack (`ctp::make_argpack`).
///
/// `argpack!(a, b)` is `(a, b)`; `argpack!()` is `()`. Arguments are moved in
/// by value — pass `&x` for the C++ by-reference behavior (divergence 5).
///
/// ```
/// use ctp_types::{argpack, argpack::ArgPack};
/// assert_eq!(argpack!(1, 2.0, "three").size(), 3);
/// assert_eq!(argpack!().size(), 0);
/// ```
#[macro_export]
macro_rules! argpack {
    () => { () };
    ($($arg:expr),+ $(,)?) => { ($($arg,)+) };
}

/// Merges any number of argpacks into one (`ctp::MergeArgPacks::Merge`).
///
/// Folds [`MergeArgPack::merge`] left-to-right, so the combined arity must
/// stay `<= 12` at every step.
///
/// ```
/// use ctp_types::{argpack, merge_argpacks, argpack::ArgPack};
/// let merged = merge_argpacks!(argpack!(0), argpack!(1, 2), argpack!(3));
/// assert_eq!(merged, (0, 1, 2, 3));
/// assert_eq!(merged.size(), 4);
/// ```
#[macro_export]
macro_rules! merge_argpacks {
    () => { () };
    ($first:expr $(, $rest:expr)* $(,)?) => {{
        let acc = $first;
        $( let acc = $crate::argpack::MergeArgPack::merge(acc, $rest); )*
        acc
    }};
}

/// Inserts `prod` at the head of each pack (`ctp::ProductArgPacks::Product`).
///
/// `product_argpacks!(p, a, b)` is `(p.clone(), a, p, b)` — a flat pack of
/// `2 * N` elements alternating the product value and each original pack,
/// exactly like the C++. `prod` is evaluated once and cloned per slot
/// (divergence 7).
///
/// ```
/// use ctp_types::{argpack, product_argpacks, argpack::ArgPack};
/// let prod = product_argpacks!(0, argpack!(1, 2), argpack!(3.0, 4.0));
/// assert_eq!(prod.size(), 4);
/// assert_eq!(prod, (0, (1, 2), 0, (3.0, 4.0)));
/// ```
#[macro_export]
macro_rules! product_argpacks {
    ($prod:expr $(,)?) => { () };
    ($prod:expr, $($pack:expr),+ $(,)?) => {{
        let prod = $prod;
        ( $( ::core::clone::Clone::clone(&prod), $pack, )+ )
    }};
}

// ---------------------------------------------------------------------------
// Impl generators
// ---------------------------------------------------------------------------

/// Generates [`ArgPack`], [`ForEachArg`], [`ForEachArgMut`] and [`CallArgs`]
/// for one tuple arity. Takes the element list twice: forward, then reversed
/// (macro_rules cannot reverse a repetition itself).
macro_rules! impl_argpack {
    ($n:expr; [$($T:ident => $i:tt),+]; [$($RT:ident => $ri:tt),+]) => {
        impl<$($T),+> ArgPack for ($($T,)+) {
            const SIZE: usize = $n;
        }

        impl<V, $($T),+> ForEachArg<V> for ($($T,)+)
        where
            V: $(VisitArg<$T> +)+
        {
            #[inline]
            fn for_each(&self, visitor: &mut V) {
                $( visitor.visit($i, &self.$i); )+
            }

            #[inline]
            fn for_each_rev(&self, visitor: &mut V) {
                $( visitor.visit($ri, &self.$ri); )+
            }
        }

        impl<V, $($T),+> ForEachArgMut<V> for ($($T,)+)
        where
            V: $(VisitArgMut<$T> +)+
        {
            #[inline]
            fn for_each_mut(&mut self, visitor: &mut V) {
                $( visitor.visit_mut($i, &mut self.$i); )+
            }

            #[inline]
            fn for_each_mut_rev(&mut self, visitor: &mut V) {
                $( visitor.visit_mut($ri, &mut self.$ri); )+
            }
        }

        impl<F, R, $($T),+> CallArgs<F> for ($($T,)+)
        where
            F: FnOnce($($T),+) -> R,
        {
            type Output = R;

            #[inline]
            fn call(self, f: F) -> R {
                f($(self.$i),+)
            }
        }
    };
}

/// Generates one [`ArgPackGet`] impl: index `$i` of the tuple `[$($T),*]`.
macro_rules! impl_get {
    ($i:tt, $item:ident, [$($T:ident),+]) => {
        impl<$($T),+> ArgPackGet<$i> for ($($T,)+) {
            type Item = $item;

            #[inline]
            fn get(&self) -> &$item {
                &self.$i
            }

            #[inline]
            fn get_mut(&mut self) -> &mut $item {
                &mut self.$i
            }

            #[inline]
            fn into_arg(self) -> $item {
                self.$i
            }
        }
    };
}

/// Generates one [`MergeArgPack`] impl for a (left, right) arity pair.
macro_rules! impl_merge {
    ([$($A:ident => $ai:tt),*], [$($B:ident => $bi:tt),*]) => {
        impl<$($A,)* $($B,)*> MergeArgPack<($($B,)*)> for ($($A,)*) {
            type Output = ($($A,)* $($B,)*);

            #[inline]
            #[allow(clippy::unused_unit)]
            fn merge(self, rhs: ($($B,)*)) -> Self::Output {
                let _ = &rhs;
                ($(self.$ai,)* $(rhs.$bi,)*)
            }
        }
    };
}

// ---------------------------------------------------------------------------
// Arity-0 impls (the macros above need at least one element)
// ---------------------------------------------------------------------------

impl ArgPack for () {
    const SIZE: usize = 0;
}

impl<V> ForEachArg<V> for () {
    #[inline]
    fn for_each(&self, _visitor: &mut V) {}

    #[inline]
    fn for_each_rev(&self, _visitor: &mut V) {}
}

impl<V> ForEachArgMut<V> for () {
    #[inline]
    fn for_each_mut(&mut self, _visitor: &mut V) {}

    #[inline]
    fn for_each_mut_rev(&mut self, _visitor: &mut V) {}
}

impl<F, R> CallArgs<F> for ()
where
    F: FnOnce() -> R,
{
    type Output = R;

    #[inline]
    fn call(self, f: F) -> R {
        f()
    }
}

// ---------------------------------------------------------------------------
// Generated impls
// ---------------------------------------------------------------------------

// ---- ArgPack / ForEachArg / ForEachArgMut / CallArgs, arity 1..=12 ----
impl_argpack!(1; [A0 => 0]; [A0 => 0]);
impl_argpack!(2; [A0 => 0, A1 => 1]; [A1 => 1, A0 => 0]);
impl_argpack!(3; [A0 => 0, A1 => 1, A2 => 2]; [A2 => 2, A1 => 1, A0 => 0]);
impl_argpack!(4; [A0 => 0, A1 => 1, A2 => 2, A3 => 3]; [A3 => 3, A2 => 2, A1 => 1, A0 => 0]);
impl_argpack!(5; [A0 => 0, A1 => 1, A2 => 2, A3 => 3, A4 => 4]; [A4 => 4, A3 => 3, A2 => 2, A1 => 1, A0 => 0]);
impl_argpack!(6; [A0 => 0, A1 => 1, A2 => 2, A3 => 3, A4 => 4, A5 => 5]; [A5 => 5, A4 => 4, A3 => 3, A2 => 2, A1 => 1, A0 => 0]);
impl_argpack!(7; [A0 => 0, A1 => 1, A2 => 2, A3 => 3, A4 => 4, A5 => 5, A6 => 6]; [A6 => 6, A5 => 5, A4 => 4, A3 => 3, A2 => 2, A1 => 1, A0 => 0]);
impl_argpack!(8; [A0 => 0, A1 => 1, A2 => 2, A3 => 3, A4 => 4, A5 => 5, A6 => 6, A7 => 7]; [A7 => 7, A6 => 6, A5 => 5, A4 => 4, A3 => 3, A2 => 2, A1 => 1, A0 => 0]);
impl_argpack!(9; [A0 => 0, A1 => 1, A2 => 2, A3 => 3, A4 => 4, A5 => 5, A6 => 6, A7 => 7, A8 => 8]; [A8 => 8, A7 => 7, A6 => 6, A5 => 5, A4 => 4, A3 => 3, A2 => 2, A1 => 1, A0 => 0]);
impl_argpack!(10; [A0 => 0, A1 => 1, A2 => 2, A3 => 3, A4 => 4, A5 => 5, A6 => 6, A7 => 7, A8 => 8, A9 => 9]; [A9 => 9, A8 => 8, A7 => 7, A6 => 6, A5 => 5, A4 => 4, A3 => 3, A2 => 2, A1 => 1, A0 => 0]);
impl_argpack!(11; [A0 => 0, A1 => 1, A2 => 2, A3 => 3, A4 => 4, A5 => 5, A6 => 6, A7 => 7, A8 => 8, A9 => 9, A10 => 10]; [A10 => 10, A9 => 9, A8 => 8, A7 => 7, A6 => 6, A5 => 5, A4 => 4, A3 => 3, A2 => 2, A1 => 1, A0 => 0]);
impl_argpack!(12; [A0 => 0, A1 => 1, A2 => 2, A3 => 3, A4 => 4, A5 => 5, A6 => 6, A7 => 7, A8 => 8, A9 => 9, A10 => 10, A11 => 11]; [A11 => 11, A10 => 10, A9 => 9, A8 => 8, A7 => 7, A6 => 6, A5 => 5, A4 => 4, A3 => 3, A2 => 2, A1 => 1, A0 => 0]);

// ---- ArgPackGet: one impl per (arity, index) pair ----
impl_get!(0, A0, [A0]);
impl_get!(0, A0, [A0, A1]);
impl_get!(1, A1, [A0, A1]);
impl_get!(0, A0, [A0, A1, A2]);
impl_get!(1, A1, [A0, A1, A2]);
impl_get!(2, A2, [A0, A1, A2]);
impl_get!(0, A0, [A0, A1, A2, A3]);
impl_get!(1, A1, [A0, A1, A2, A3]);
impl_get!(2, A2, [A0, A1, A2, A3]);
impl_get!(3, A3, [A0, A1, A2, A3]);
impl_get!(0, A0, [A0, A1, A2, A3, A4]);
impl_get!(1, A1, [A0, A1, A2, A3, A4]);
impl_get!(2, A2, [A0, A1, A2, A3, A4]);
impl_get!(3, A3, [A0, A1, A2, A3, A4]);
impl_get!(4, A4, [A0, A1, A2, A3, A4]);
impl_get!(0, A0, [A0, A1, A2, A3, A4, A5]);
impl_get!(1, A1, [A0, A1, A2, A3, A4, A5]);
impl_get!(2, A2, [A0, A1, A2, A3, A4, A5]);
impl_get!(3, A3, [A0, A1, A2, A3, A4, A5]);
impl_get!(4, A4, [A0, A1, A2, A3, A4, A5]);
impl_get!(5, A5, [A0, A1, A2, A3, A4, A5]);
impl_get!(0, A0, [A0, A1, A2, A3, A4, A5, A6]);
impl_get!(1, A1, [A0, A1, A2, A3, A4, A5, A6]);
impl_get!(2, A2, [A0, A1, A2, A3, A4, A5, A6]);
impl_get!(3, A3, [A0, A1, A2, A3, A4, A5, A6]);
impl_get!(4, A4, [A0, A1, A2, A3, A4, A5, A6]);
impl_get!(5, A5, [A0, A1, A2, A3, A4, A5, A6]);
impl_get!(6, A6, [A0, A1, A2, A3, A4, A5, A6]);
impl_get!(0, A0, [A0, A1, A2, A3, A4, A5, A6, A7]);
impl_get!(1, A1, [A0, A1, A2, A3, A4, A5, A6, A7]);
impl_get!(2, A2, [A0, A1, A2, A3, A4, A5, A6, A7]);
impl_get!(3, A3, [A0, A1, A2, A3, A4, A5, A6, A7]);
impl_get!(4, A4, [A0, A1, A2, A3, A4, A5, A6, A7]);
impl_get!(5, A5, [A0, A1, A2, A3, A4, A5, A6, A7]);
impl_get!(6, A6, [A0, A1, A2, A3, A4, A5, A6, A7]);
impl_get!(7, A7, [A0, A1, A2, A3, A4, A5, A6, A7]);
impl_get!(0, A0, [A0, A1, A2, A3, A4, A5, A6, A7, A8]);
impl_get!(1, A1, [A0, A1, A2, A3, A4, A5, A6, A7, A8]);
impl_get!(2, A2, [A0, A1, A2, A3, A4, A5, A6, A7, A8]);
impl_get!(3, A3, [A0, A1, A2, A3, A4, A5, A6, A7, A8]);
impl_get!(4, A4, [A0, A1, A2, A3, A4, A5, A6, A7, A8]);
impl_get!(5, A5, [A0, A1, A2, A3, A4, A5, A6, A7, A8]);
impl_get!(6, A6, [A0, A1, A2, A3, A4, A5, A6, A7, A8]);
impl_get!(7, A7, [A0, A1, A2, A3, A4, A5, A6, A7, A8]);
impl_get!(8, A8, [A0, A1, A2, A3, A4, A5, A6, A7, A8]);
impl_get!(0, A0, [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9]);
impl_get!(1, A1, [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9]);
impl_get!(2, A2, [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9]);
impl_get!(3, A3, [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9]);
impl_get!(4, A4, [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9]);
impl_get!(5, A5, [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9]);
impl_get!(6, A6, [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9]);
impl_get!(7, A7, [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9]);
impl_get!(8, A8, [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9]);
impl_get!(9, A9, [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9]);
impl_get!(0, A0, [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10]);
impl_get!(1, A1, [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10]);
impl_get!(2, A2, [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10]);
impl_get!(3, A3, [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10]);
impl_get!(4, A4, [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10]);
impl_get!(5, A5, [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10]);
impl_get!(6, A6, [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10]);
impl_get!(7, A7, [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10]);
impl_get!(8, A8, [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10]);
impl_get!(9, A9, [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10]);
impl_get!(10, A10, [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10]);
impl_get!(0, A0, [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11]);
impl_get!(1, A1, [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11]);
impl_get!(2, A2, [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11]);
impl_get!(3, A3, [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11]);
impl_get!(4, A4, [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11]);
impl_get!(5, A5, [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11]);
impl_get!(6, A6, [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11]);
impl_get!(7, A7, [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11]);
impl_get!(8, A8, [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11]);
impl_get!(9, A9, [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11]);
impl_get!(10, A10, [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11]);
impl_get!(11, A11, [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11]);

// ---- MergeArgPack: every (n, m) with n + m <= 12 ----
impl_merge!([], []);
impl_merge!([], [B0 => 0]);
impl_merge!([], [B0 => 0, B1 => 1]);
impl_merge!([], [B0 => 0, B1 => 1, B2 => 2]);
impl_merge!([], [B0 => 0, B1 => 1, B2 => 2, B3 => 3]);
impl_merge!([], [B0 => 0, B1 => 1, B2 => 2, B3 => 3, B4 => 4]);
impl_merge!([], [B0 => 0, B1 => 1, B2 => 2, B3 => 3, B4 => 4, B5 => 5]);
impl_merge!([], [B0 => 0, B1 => 1, B2 => 2, B3 => 3, B4 => 4, B5 => 5, B6 => 6]);
impl_merge!([], [B0 => 0, B1 => 1, B2 => 2, B3 => 3, B4 => 4, B5 => 5, B6 => 6, B7 => 7]);
impl_merge!([], [B0 => 0, B1 => 1, B2 => 2, B3 => 3, B4 => 4, B5 => 5, B6 => 6, B7 => 7, B8 => 8]);
impl_merge!([], [B0 => 0, B1 => 1, B2 => 2, B3 => 3, B4 => 4, B5 => 5, B6 => 6, B7 => 7, B8 => 8, B9 => 9]);
impl_merge!([], [B0 => 0, B1 => 1, B2 => 2, B3 => 3, B4 => 4, B5 => 5, B6 => 6, B7 => 7, B8 => 8, B9 => 9, B10 => 10]);
impl_merge!([], [B0 => 0, B1 => 1, B2 => 2, B3 => 3, B4 => 4, B5 => 5, B6 => 6, B7 => 7, B8 => 8, B9 => 9, B10 => 10, B11 => 11]);
impl_merge!([A0 => 0], []);
impl_merge!([A0 => 0], [B0 => 0]);
impl_merge!([A0 => 0], [B0 => 0, B1 => 1]);
impl_merge!([A0 => 0], [B0 => 0, B1 => 1, B2 => 2]);
impl_merge!([A0 => 0], [B0 => 0, B1 => 1, B2 => 2, B3 => 3]);
impl_merge!([A0 => 0], [B0 => 0, B1 => 1, B2 => 2, B3 => 3, B4 => 4]);
impl_merge!([A0 => 0], [B0 => 0, B1 => 1, B2 => 2, B3 => 3, B4 => 4, B5 => 5]);
impl_merge!([A0 => 0], [B0 => 0, B1 => 1, B2 => 2, B3 => 3, B4 => 4, B5 => 5, B6 => 6]);
impl_merge!([A0 => 0], [B0 => 0, B1 => 1, B2 => 2, B3 => 3, B4 => 4, B5 => 5, B6 => 6, B7 => 7]);
impl_merge!([A0 => 0], [B0 => 0, B1 => 1, B2 => 2, B3 => 3, B4 => 4, B5 => 5, B6 => 6, B7 => 7, B8 => 8]);
impl_merge!([A0 => 0], [B0 => 0, B1 => 1, B2 => 2, B3 => 3, B4 => 4, B5 => 5, B6 => 6, B7 => 7, B8 => 8, B9 => 9]);
impl_merge!([A0 => 0], [B0 => 0, B1 => 1, B2 => 2, B3 => 3, B4 => 4, B5 => 5, B6 => 6, B7 => 7, B8 => 8, B9 => 9, B10 => 10]);
impl_merge!([A0 => 0, A1 => 1], []);
impl_merge!([A0 => 0, A1 => 1], [B0 => 0]);
impl_merge!([A0 => 0, A1 => 1], [B0 => 0, B1 => 1]);
impl_merge!([A0 => 0, A1 => 1], [B0 => 0, B1 => 1, B2 => 2]);
impl_merge!([A0 => 0, A1 => 1], [B0 => 0, B1 => 1, B2 => 2, B3 => 3]);
impl_merge!([A0 => 0, A1 => 1], [B0 => 0, B1 => 1, B2 => 2, B3 => 3, B4 => 4]);
impl_merge!([A0 => 0, A1 => 1], [B0 => 0, B1 => 1, B2 => 2, B3 => 3, B4 => 4, B5 => 5]);
impl_merge!([A0 => 0, A1 => 1], [B0 => 0, B1 => 1, B2 => 2, B3 => 3, B4 => 4, B5 => 5, B6 => 6]);
impl_merge!([A0 => 0, A1 => 1], [B0 => 0, B1 => 1, B2 => 2, B3 => 3, B4 => 4, B5 => 5, B6 => 6, B7 => 7]);
impl_merge!([A0 => 0, A1 => 1], [B0 => 0, B1 => 1, B2 => 2, B3 => 3, B4 => 4, B5 => 5, B6 => 6, B7 => 7, B8 => 8]);
impl_merge!([A0 => 0, A1 => 1], [B0 => 0, B1 => 1, B2 => 2, B3 => 3, B4 => 4, B5 => 5, B6 => 6, B7 => 7, B8 => 8, B9 => 9]);
impl_merge!([A0 => 0, A1 => 1, A2 => 2], []);
impl_merge!([A0 => 0, A1 => 1, A2 => 2], [B0 => 0]);
impl_merge!([A0 => 0, A1 => 1, A2 => 2], [B0 => 0, B1 => 1]);
impl_merge!([A0 => 0, A1 => 1, A2 => 2], [B0 => 0, B1 => 1, B2 => 2]);
impl_merge!([A0 => 0, A1 => 1, A2 => 2], [B0 => 0, B1 => 1, B2 => 2, B3 => 3]);
impl_merge!([A0 => 0, A1 => 1, A2 => 2], [B0 => 0, B1 => 1, B2 => 2, B3 => 3, B4 => 4]);
impl_merge!([A0 => 0, A1 => 1, A2 => 2], [B0 => 0, B1 => 1, B2 => 2, B3 => 3, B4 => 4, B5 => 5]);
impl_merge!([A0 => 0, A1 => 1, A2 => 2], [B0 => 0, B1 => 1, B2 => 2, B3 => 3, B4 => 4, B5 => 5, B6 => 6]);
impl_merge!([A0 => 0, A1 => 1, A2 => 2], [B0 => 0, B1 => 1, B2 => 2, B3 => 3, B4 => 4, B5 => 5, B6 => 6, B7 => 7]);
impl_merge!([A0 => 0, A1 => 1, A2 => 2], [B0 => 0, B1 => 1, B2 => 2, B3 => 3, B4 => 4, B5 => 5, B6 => 6, B7 => 7, B8 => 8]);
impl_merge!([A0 => 0, A1 => 1, A2 => 2, A3 => 3], []);
impl_merge!([A0 => 0, A1 => 1, A2 => 2, A3 => 3], [B0 => 0]);
impl_merge!([A0 => 0, A1 => 1, A2 => 2, A3 => 3], [B0 => 0, B1 => 1]);
impl_merge!([A0 => 0, A1 => 1, A2 => 2, A3 => 3], [B0 => 0, B1 => 1, B2 => 2]);
impl_merge!([A0 => 0, A1 => 1, A2 => 2, A3 => 3], [B0 => 0, B1 => 1, B2 => 2, B3 => 3]);
impl_merge!([A0 => 0, A1 => 1, A2 => 2, A3 => 3], [B0 => 0, B1 => 1, B2 => 2, B3 => 3, B4 => 4]);
impl_merge!([A0 => 0, A1 => 1, A2 => 2, A3 => 3], [B0 => 0, B1 => 1, B2 => 2, B3 => 3, B4 => 4, B5 => 5]);
impl_merge!([A0 => 0, A1 => 1, A2 => 2, A3 => 3], [B0 => 0, B1 => 1, B2 => 2, B3 => 3, B4 => 4, B5 => 5, B6 => 6]);
impl_merge!([A0 => 0, A1 => 1, A2 => 2, A3 => 3], [B0 => 0, B1 => 1, B2 => 2, B3 => 3, B4 => 4, B5 => 5, B6 => 6, B7 => 7]);
impl_merge!([A0 => 0, A1 => 1, A2 => 2, A3 => 3, A4 => 4], []);
impl_merge!([A0 => 0, A1 => 1, A2 => 2, A3 => 3, A4 => 4], [B0 => 0]);
impl_merge!([A0 => 0, A1 => 1, A2 => 2, A3 => 3, A4 => 4], [B0 => 0, B1 => 1]);
impl_merge!([A0 => 0, A1 => 1, A2 => 2, A3 => 3, A4 => 4], [B0 => 0, B1 => 1, B2 => 2]);
impl_merge!([A0 => 0, A1 => 1, A2 => 2, A3 => 3, A4 => 4], [B0 => 0, B1 => 1, B2 => 2, B3 => 3]);
impl_merge!([A0 => 0, A1 => 1, A2 => 2, A3 => 3, A4 => 4], [B0 => 0, B1 => 1, B2 => 2, B3 => 3, B4 => 4]);
impl_merge!([A0 => 0, A1 => 1, A2 => 2, A3 => 3, A4 => 4], [B0 => 0, B1 => 1, B2 => 2, B3 => 3, B4 => 4, B5 => 5]);
impl_merge!([A0 => 0, A1 => 1, A2 => 2, A3 => 3, A4 => 4], [B0 => 0, B1 => 1, B2 => 2, B3 => 3, B4 => 4, B5 => 5, B6 => 6]);
impl_merge!([A0 => 0, A1 => 1, A2 => 2, A3 => 3, A4 => 4, A5 => 5], []);
impl_merge!([A0 => 0, A1 => 1, A2 => 2, A3 => 3, A4 => 4, A5 => 5], [B0 => 0]);
impl_merge!([A0 => 0, A1 => 1, A2 => 2, A3 => 3, A4 => 4, A5 => 5], [B0 => 0, B1 => 1]);
impl_merge!([A0 => 0, A1 => 1, A2 => 2, A3 => 3, A4 => 4, A5 => 5], [B0 => 0, B1 => 1, B2 => 2]);
impl_merge!([A0 => 0, A1 => 1, A2 => 2, A3 => 3, A4 => 4, A5 => 5], [B0 => 0, B1 => 1, B2 => 2, B3 => 3]);
impl_merge!([A0 => 0, A1 => 1, A2 => 2, A3 => 3, A4 => 4, A5 => 5], [B0 => 0, B1 => 1, B2 => 2, B3 => 3, B4 => 4]);
impl_merge!([A0 => 0, A1 => 1, A2 => 2, A3 => 3, A4 => 4, A5 => 5], [B0 => 0, B1 => 1, B2 => 2, B3 => 3, B4 => 4, B5 => 5]);
impl_merge!([A0 => 0, A1 => 1, A2 => 2, A3 => 3, A4 => 4, A5 => 5, A6 => 6], []);
impl_merge!([A0 => 0, A1 => 1, A2 => 2, A3 => 3, A4 => 4, A5 => 5, A6 => 6], [B0 => 0]);
impl_merge!([A0 => 0, A1 => 1, A2 => 2, A3 => 3, A4 => 4, A5 => 5, A6 => 6], [B0 => 0, B1 => 1]);
impl_merge!([A0 => 0, A1 => 1, A2 => 2, A3 => 3, A4 => 4, A5 => 5, A6 => 6], [B0 => 0, B1 => 1, B2 => 2]);
impl_merge!([A0 => 0, A1 => 1, A2 => 2, A3 => 3, A4 => 4, A5 => 5, A6 => 6], [B0 => 0, B1 => 1, B2 => 2, B3 => 3]);
impl_merge!([A0 => 0, A1 => 1, A2 => 2, A3 => 3, A4 => 4, A5 => 5, A6 => 6], [B0 => 0, B1 => 1, B2 => 2, B3 => 3, B4 => 4]);
impl_merge!([A0 => 0, A1 => 1, A2 => 2, A3 => 3, A4 => 4, A5 => 5, A6 => 6, A7 => 7], []);
impl_merge!([A0 => 0, A1 => 1, A2 => 2, A3 => 3, A4 => 4, A5 => 5, A6 => 6, A7 => 7], [B0 => 0]);
impl_merge!([A0 => 0, A1 => 1, A2 => 2, A3 => 3, A4 => 4, A5 => 5, A6 => 6, A7 => 7], [B0 => 0, B1 => 1]);
impl_merge!([A0 => 0, A1 => 1, A2 => 2, A3 => 3, A4 => 4, A5 => 5, A6 => 6, A7 => 7], [B0 => 0, B1 => 1, B2 => 2]);
impl_merge!([A0 => 0, A1 => 1, A2 => 2, A3 => 3, A4 => 4, A5 => 5, A6 => 6, A7 => 7], [B0 => 0, B1 => 1, B2 => 2, B3 => 3]);
impl_merge!([A0 => 0, A1 => 1, A2 => 2, A3 => 3, A4 => 4, A5 => 5, A6 => 6, A7 => 7, A8 => 8], []);
impl_merge!([A0 => 0, A1 => 1, A2 => 2, A3 => 3, A4 => 4, A5 => 5, A6 => 6, A7 => 7, A8 => 8], [B0 => 0]);
impl_merge!([A0 => 0, A1 => 1, A2 => 2, A3 => 3, A4 => 4, A5 => 5, A6 => 6, A7 => 7, A8 => 8], [B0 => 0, B1 => 1]);
impl_merge!([A0 => 0, A1 => 1, A2 => 2, A3 => 3, A4 => 4, A5 => 5, A6 => 6, A7 => 7, A8 => 8], [B0 => 0, B1 => 1, B2 => 2]);
impl_merge!([A0 => 0, A1 => 1, A2 => 2, A3 => 3, A4 => 4, A5 => 5, A6 => 6, A7 => 7, A8 => 8, A9 => 9], []);
impl_merge!([A0 => 0, A1 => 1, A2 => 2, A3 => 3, A4 => 4, A5 => 5, A6 => 6, A7 => 7, A8 => 8, A9 => 9], [B0 => 0]);
impl_merge!([A0 => 0, A1 => 1, A2 => 2, A3 => 3, A4 => 4, A5 => 5, A6 => 6, A7 => 7, A8 => 8, A9 => 9], [B0 => 0, B1 => 1]);
impl_merge!([A0 => 0, A1 => 1, A2 => 2, A3 => 3, A4 => 4, A5 => 5, A6 => 6, A7 => 7, A8 => 8, A9 => 9, A10 => 10], []);
impl_merge!([A0 => 0, A1 => 1, A2 => 2, A3 => 3, A4 => 4, A5 => 5, A6 => 6, A7 => 7, A8 => 8, A9 => 9, A10 => 10], [B0 => 0]);
impl_merge!([A0 => 0, A1 => 1, A2 => 2, A3 => 3, A4 => 4, A5 => 5, A6 => 6, A7 => 7, A8 => 8, A9 => 9, A10 => 10, A11 => 11], []);

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    // `argpack!`, `merge_argpacks!` and `product_argpacks!` need no import
    // here: they are macro_rules defined earlier in this file, so textual
    // scope already covers this module.
    use super::*;
    use std::sync::Arc;

    // -- Visitors used across the tests ------------------------------------

    /// Mirrors `util/formatter.h`: one blanket impl over `Display` stands in
    /// for the C++ generic lambda `[](auto i, auto &arg) { ss << arg; }`.
    #[derive(Default)]
    struct Collect {
        seen: Vec<String>,
    }

    impl<T: std::fmt::Display + ?Sized> VisitArg<T> for Collect {
        fn visit(&mut self, index: usize, arg: &T) {
            self.seen.push(format!("{index}:{arg}"));
        }
    }

    /// Mirrors `LocalDeserialize::operator()`, whose lambda takes `auto &arg`
    /// and writes through it.
    #[derive(Default)]
    struct Doubler {
        visited: Vec<usize>,
    }

    impl VisitArgMut<i32> for Doubler {
        fn visit_mut(&mut self, index: usize, arg: &mut i32) {
            self.visited.push(index);
            *arg *= 2;
        }
    }

    impl VisitArgMut<f64> for Doubler {
        fn visit_mut(&mut self, index: usize, arg: &mut f64) {
            self.visited.push(index);
            *arg *= 2.0;
        }
    }

    // -- Size --------------------------------------------------------------

    #[test]
    fn size_of_empty_pack() {
        // C++: ctp::ArgPack<>() / PassArgPack::Call(ArgPack<>(), f)
        assert_eq!(<() as ArgPack>::SIZE, 0);
        assert_eq!(argpack!().size(), 0);
    }

    #[test]
    fn size_across_arities() {
        assert_eq!(argpack!(1).size(), 1);
        assert_eq!(argpack!(1, 2.0).size(), 2);
        assert_eq!(argpack!(1, 2.0, "3").size(), 3);
        assert_eq!(<(i32, f64, &str) as ArgPack>::SIZE, 3);
        // Trailing comma is accepted, and `argpack!(x)` is a 1-tuple `(x,)`.
        assert_eq!(argpack!(1, 2,).size(), 2);
    }

    #[test]
    fn size_at_max_arity() {
        let pack = argpack!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
        assert_eq!(pack.size(), 12);
        assert_eq!(
            <(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) as ArgPack>::SIZE,
            12
        );
    }

    // -- Forward / element access ------------------------------------------

    #[test]
    fn get_and_get_mut() {
        // C++: pack.Forward<i>()
        let mut pack = argpack!(10i32, 11i32, 12i32);
        assert_eq!(*ArgPackGet::<0>::get(&pack), 10);
        assert_eq!(*ArgPackGet::<1>::get(&pack), 11);
        assert_eq!(*ArgPackGet::<2>::get(&pack), 12);
        *ArgPackGet::<1>::get_mut(&mut pack) = 99;
        assert_eq!(pack, (10, 99, 12));
    }

    #[test]
    fn get_on_heterogeneous_pack() {
        let pack = argpack!(1i32, 2.5f64, "three");
        assert_eq!(*ArgPackGet::<0>::get(&pack), 1);
        assert_eq!(*ArgPackGet::<1>::get(&pack), 2.5);
        assert_eq!(*ArgPackGet::<2>::get(&pack), "three");
    }

    #[test]
    fn get_at_first_and_last_index_of_max_arity() {
        // Boundary: index 0 and index SIZE-1 of the largest supported pack.
        let pack = argpack!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 110);
        assert_eq!(*ArgPackGet::<0>::get(&pack), 0);
        assert_eq!(*ArgPackGet::<11>::get(&pack), 110);
        // `ArgPackGet::<12>::get(&pack)` does not compile: no impl past the
        // end. C++ silently returns void there (divergence 4).
    }

    #[test]
    fn into_arg_moves_element_out() {
        let pack = argpack!(String::from("a"), String::from("b"));
        let b: String = ArgPackGet::<1>::into_arg(pack);
        assert_eq!(b, "b");
    }

    // -- PassArgPack -------------------------------------------------------

    fn test_argpack0_pass() -> &'static str {
        "HERE0"
    }

    #[test]
    fn call_empty_pack() {
        // C++: ctp::PassArgPack::Call(ctp::ArgPack<>(), test_argpack0_pass);
        assert_eq!(PassArgPack::call((), test_argpack0_pass), "HERE0");
        assert_eq!(PassArgPack::call(argpack!(), test_argpack0_pass), "HERE0");
    }

    fn test_argpack3_pass(x: i32, y: f64, z: f32) {
        assert_eq!(x, 0);
        assert_eq!(y, 1.0);
        assert_eq!(z, 0.0);
    }

    #[test]
    fn call_three_arg_pack() {
        // C++: PassArgPack::Call(make_argpack(T1(0), T2(1), T3(0)),
        //                        test_argpack3_pass<T1, T2, T3>);
        PassArgPack::call(argpack!(0i32, 1.0f64, 0.0f32), test_argpack3_pass);
        // Same, via the trait method.
        argpack!(0i32, 1.0f64, 0.0f32).call(test_argpack3_pass);
    }

    #[test]
    fn call_returns_value_and_unit_alike() {
        // C++ needs an is_void_v branch here; Rust does not (divergence 8).
        let sum: i32 = PassArgPack::call(argpack!(1, 2, 3), |a: i32, b: i32, c: i32| a + b + c);
        assert_eq!(sum, 6);
        let unit: () = PassArgPack::call(argpack!(1), |_a: i32| {});
        assert_eq!(unit, ());
    }

    #[test]
    fn call_pack_holding_references() {
        // C++ make_argpack packs Args&&; the Rust analog is an explicit
        // reference element (divergence 5).
        let mut y = 1i32;
        PassArgPack::call(argpack!(0i32, &mut y, 2i32), |a: i32, b: &mut i32, c: i32| {
            *b += a + c;
        });
        assert_eq!(y, 3);
    }

    #[test]
    fn call_captures_environment() {
        let mut hits = 0;
        PassArgPack::call(argpack!(1, 2), |a: i32, b: i32| hits = a + b);
        assert_eq!(hits, 3);
    }

    #[test]
    fn call_at_max_arity() {
        let pack = argpack!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
        let total = PassArgPack::call(
            pack,
            |a: i32, b: i32, c: i32, d: i32, e: i32, f: i32, g: i32, h: i32, i: i32, j: i32, k: i32, l: i32| {
                a + b + c + d + e + f + g + h + i + j + k + l
            },
        );
        assert_eq!(total, 78);
    }

    // -- Move semantics ----------------------------------------------------

    #[test]
    fn pack_neither_copies_nor_clones() {
        // Rust analog of the C++ TestArgpackCopy case: an Arc's strong count
        // is the observable proof that nothing was duplicated on the way in
        // or on the way out.
        let arc = Arc::new(7u32);
        assert_eq!(Arc::strong_count(&arc), 1);

        let pack = argpack!(Arc::clone(&arc), 1u8);
        assert_eq!(Arc::strong_count(&arc), 2, "building the pack must not clone");

        let inner: Arc<u32> = ArgPackGet::<0>::into_arg(pack);
        assert_eq!(Arc::strong_count(&arc), 2, "into_arg must not clone");
        assert_eq!(*inner, 7);

        drop(inner);
        assert_eq!(Arc::strong_count(&arc), 1);
    }

    #[test]
    fn call_moves_non_copy_arguments_through() {
        let s = String::from("owned");
        let out = PassArgPack::call(argpack!(s), |x: String| x);
        assert_eq!(out, "owned");
    }

    // -- Iteration ---------------------------------------------------------

    #[test]
    fn for_each_empty_pack_visits_nothing() {
        let mut c = Collect::default();
        ForEachArg::<Collect>::for_each(&(), &mut c);
        ForEachArg::<Collect>::for_each_rev(&(), &mut c);
        assert!(c.seen.is_empty());
    }

    #[test]
    fn forward_iterate_visits_in_order_with_indices() {
        // C++: ForwardIterateArgpack::Apply(make_argpack(...), lambda)
        let mut c = Collect::default();
        argpack!(1i32, "two", 3.5f64).for_each(&mut c);
        assert_eq!(c.seen, ["0:1", "1:two", "2:3.5"]);
    }

    #[test]
    fn reverse_iterate_visits_in_reverse_order() {
        // C++: ReverseIterateArgpack::Apply(...) — recurses first, then applies.
        let mut c = Collect::default();
        argpack!(1i32, "two", 3.5f64).for_each_rev(&mut c);
        assert_eq!(c.seen, ["2:3.5", "1:two", "0:1"]);
    }

    #[test]
    fn iterate_single_element_pack() {
        let mut c = Collect::default();
        argpack!(42i32).for_each(&mut c);
        assert_eq!(c.seen, ["0:42"]);

        let mut r = Collect::default();
        argpack!(42i32).for_each_rev(&mut r);
        assert_eq!(r.seen, ["0:42"]);
    }

    #[test]
    fn iterate_pack_of_references() {
        // Mirrors formatter.h, which packs forwarded references and prints them.
        let a = 1i32;
        let b = String::from("b");
        let mut c = Collect::default();
        argpack!(&a, &b).for_each(&mut c);
        assert_eq!(c.seen, ["0:1", "1:b"]);
    }

    #[test]
    fn iterate_at_max_arity() {
        let mut c = Collect::default();
        argpack!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11).for_each(&mut c);
        assert_eq!(c.seen.len(), 12);
        assert_eq!(c.seen.first().unwrap(), "0:0");
        assert_eq!(c.seen.last().unwrap(), "11:11");
    }

    #[test]
    fn for_each_mut_writes_through() {
        // Mirrors LocalDeserialize: the lambda takes `auto &arg` and mutates.
        let mut pack = argpack!(1i32, 2.5f64, 3i32);
        let mut d = Doubler::default();
        pack.for_each_mut(&mut d);
        assert_eq!(pack, (2i32, 5.0f64, 6i32));
        assert_eq!(d.visited, [0, 1, 2]);
    }

    #[test]
    fn for_each_mut_rev_writes_through_in_reverse() {
        let mut pack = argpack!(1i32, 2.5f64);
        let mut d = Doubler::default();
        pack.for_each_mut_rev(&mut d);
        assert_eq!(pack, (2i32, 5.0f64));
        assert_eq!(d.visited, [1, 0]);
    }

    #[test]
    fn for_each_mut_on_empty_pack() {
        let mut d = Doubler::default();
        ForEachArgMut::<Doubler>::for_each_mut(&mut (), &mut d);
        ForEachArgMut::<Doubler>::for_each_mut_rev(&mut (), &mut d);
        assert!(d.visited.is_empty());
    }

    #[test]
    fn formatter_use_case() {
        // The shape util/formatter.h actually needs: pack size drives the
        // token count, then a forward iterate interleaves args with text.
        let args = argpack!(1i32, "b", 2.5f64);
        assert_eq!(args.size(), 3);

        let mut c = Collect::default();
        args.for_each(&mut c);
        let rendered: Vec<&str> = c.seen.iter().map(|s| s.split(':').nth(1).unwrap()).collect();
        assert_eq!(rendered.join(" | "), "1 | b | 2.5");
    }

    // -- MergeArgPacks -----------------------------------------------------

    #[test]
    fn merge_two_packs_sizes() {
        // C++: MergeArgPacks::Merge(make_argpack(T1(0)),
        //                           make_argpack(T2(1), T2(0))).Size() == 3
        let merged = MergeArgPacks::merge(argpack!(0i32), argpack!(1.0f64, 0.0f64));
        assert_eq!(merged.size(), 3);
        assert_eq!(merged, (0i32, 1.0f64, 0.0f64));
    }

    #[test]
    fn merge_preserves_order() {
        let merged = MergeArgPacks::merge(argpack!(1, 2), argpack!(3, 4));
        assert_eq!(merged, (1, 2, 3, 4));
    }

    #[test]
    fn merge_with_empty_packs() {
        assert_eq!(MergeArgPacks::merge((), ()), ());
        assert_eq!(MergeArgPacks::merge((), argpack!(1, 2)), (1, 2));
        assert_eq!(MergeArgPacks::merge(argpack!(1, 2), ()), (1, 2));
        assert_eq!(MergeArgPacks::merge((), argpack!(1)).size(), 1);
    }

    #[test]
    fn merged_pack_passed_to_function() {
        // C++: PassArgPack::Call(MergeArgPacks::Merge(make_argpack(T1(0)),
        //                                             make_argpack(T2(1), T3(0))),
        //                        test_argpack3_pass<T1, T2, T3>);
        PassArgPack::call(
            MergeArgPacks::merge(argpack!(0i32), argpack!(1.0f64, 0.0f32)),
            test_argpack3_pass,
        );
    }

    #[test]
    fn merge_argpacks_macro_folds_many() {
        // C++ Merge is variadic; the Rust macro folds the pairwise trait.
        let merged = merge_argpacks!(argpack!(0), argpack!(1, 2), argpack!(3), argpack!(4, 5));
        assert_eq!(merged, (0, 1, 2, 3, 4, 5));
        assert_eq!(merged.size(), 6);
    }

    #[test]
    fn merge_argpacks_macro_edge_cases() {
        let none: () = merge_argpacks!();
        assert_eq!(none.size(), 0);
        assert_eq!(merge_argpacks!(argpack!(1, 2)), (1, 2));
        assert_eq!(merge_argpacks!((), (), argpack!(9)), (9,));
    }

    #[test]
    fn merge_to_max_combined_arity() {
        // Boundary: 6 + 6 == 12 is supported; 13 would not compile.
        let merged = merge_argpacks!(argpack!(1, 2, 3, 4, 5, 6), argpack!(7, 8, 9, 10, 11, 12));
        assert_eq!(merged.size(), 12);
        assert_eq!(*ArgPackGet::<11>::get(&merged), 12);
    }

    #[test]
    fn merge_moves_without_cloning() {
        let arc = Arc::new(1u8);
        let merged = MergeArgPacks::merge(argpack!(Arc::clone(&arc)), argpack!(2u8));
        assert_eq!(Arc::strong_count(&arc), 2);
        assert_eq!(merged.size(), 2);
        drop(merged);
        assert_eq!(Arc::strong_count(&arc), 1);
    }

    // -- ProductArgPacks ---------------------------------------------------

    fn test_product1(b: i32, c: i32) {
        assert_eq!(b, 1);
        assert_eq!(c, 2);
    }

    fn test_product2(d: f64, e: f64) {
        assert_eq!(d, 3.0);
        assert_eq!(e, 4.0);
    }

    fn test_product(a: i32, pack1: (i32, i32), a2: i32, pack2: (f64, f64)) {
        assert_eq!(a, 0);
        assert_eq!(a2, 0);
        PassArgPack::call(pack1, test_product1);
        PassArgPack::call(pack2, test_product2);
    }

    #[test]
    fn product_size() {
        // C++: ProductArgPacks::Product(0, make_argpack(1, 2),
        //                               make_argpack<double, double>(3, 4)).Size() == 4
        let pack = product_argpacks!(0, argpack!(1, 2), argpack!(3.0f64, 4.0f64));
        assert_eq!(pack.size(), 4);
    }

    #[test]
    fn product_passed_to_function() {
        // C++: PassArgPack::Call(ProductArgPacks::Product(0, ...), test_product<...>);
        PassArgPack::call(
            product_argpacks!(0, argpack!(1, 2), argpack!(3.0f64, 4.0f64)),
            test_product,
        );
    }

    #[test]
    fn product_interleaves_head_with_each_pack() {
        let pack = product_argpacks!(7, argpack!(1), argpack!(2.0f64));
        assert_eq!(pack, (7, (1,), 7, (2.0,)));
    }

    #[test]
    fn product_with_one_pack_and_with_none() {
        assert_eq!(product_argpacks!(1, argpack!(2)).size(), 2);
        let empty: () = product_argpacks!(1);
        assert_eq!(empty.size(), 0);
    }

    #[test]
    fn product_evaluates_head_once() {
        // Divergence 7: the head is bound before cloning, so a side-effecting
        // expression runs exactly once no matter how many packs follow.
        let mut calls = 0;
        let mut head = || {
            calls += 1;
            5i32
        };
        let pack = product_argpacks!(head(), argpack!(1), argpack!(2), argpack!(3));
        assert_eq!(pack.size(), 6);
        assert_eq!(calls, 1);
    }

    // -- MakeConstexpr / markers -------------------------------------------

    #[test]
    fn make_constexpr_get() {
        // C++: MakeConstexpr<size_t, i>::Get()
        assert_eq!(MakeConstexpr::<0>::get(), 0);
        assert_eq!(MakeConstexpr::<7>::get(), 7);
        assert_eq!(MakeConstexpr::<{ usize::MAX }>::get(), usize::MAX);
        assert_eq!(MakeConstexpr::<3>::VAL, 3);
        // Usable in const context, like the C++ constexpr it ports.
        const V: usize = MakeConstexpr::<9>::get();
        assert_eq!(V, 9);
    }

    #[test]
    fn markers_are_zero_sized() {
        assert_eq!(std::mem::size_of::<PiecewiseConstruct>(), 0);
        assert_eq!(std::mem::size_of::<MakeConstexpr<4>>(), 0);
        assert_eq!(std::mem::size_of::<PassArgPack>(), 0);
        assert_eq!(std::mem::size_of::<MergeArgPacks>(), 0);
        // Generic so the Default impl is exercised without clippy flagging a
        // `Foo::default()` call on a unit struct.
        fn defaulted<T: Default>() -> T {
            T::default()
        }
        assert_eq!(defaulted::<PiecewiseConstruct>(), PiecewiseConstruct);
        assert_eq!(defaulted::<MakeConstexpr<4>>(), MakeConstexpr::<4>);
    }
}
