// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).
//
// Rust port of `clio_ctp/util/error.h` + `clio_ctp/util/errors.h`.
//
// C++ → Rust name mapping:
// - `ctp::Error`                      → `Error`
// - `ctp::Error::Error()`             → `Error::default()`
// - `ctp::Error::Error(const char *)` → `Error::new(&'static str)` (const fn)
// - `ctp::Error::format(args...)`     → `Error::format(&[&dyn Display])`
//   (Rust has no variadic templates; arguments are passed as a slice of
//   `&dyn Display` trait objects, applied in order exactly like the C++
//   `ctp::Formatter` interleaves them.)
// - `ctp::Error::what()`              → `Error::what()` (also the `Display`
//   impl and thus `std::error::Error`); like C++, an error that has not
//   been through `format` has an EMPTY message.
// - `ctp::Error::print()`             → `Error::print()` (stdout, like C++)
// - `CTP_ERROR_IS(err, check)`        → `Error::is(&self, &Error)` — the C++
//   macro calls a `get_code()` that does not actually exist on `ctp::Error`
//   (dead macro); the Rust method compares the format-string identity,
//   which is the evident intent. `Error::fmt_str()` exposes that string.
// - `ctp::Formatter::format`          → private `format_args` (same
//   placeholder algorithm: split on '{', skip two bytes, and if the number
//   of `{}` placeholders does not match the number of arguments the format
//   string is returned VERBATIM, exactly as in C++).
// - `errors.h` constants (`MEMORY_BACKEND_REPEATED`, `TOO_MANY_ALLOCATORS`,
//   `NOT_IMPLEMENTED`, `SHMEM_CREATE_FAILED`, `SHMEM_RESERVE_FAILED`,
//   `SHMEM_NOT_SUPPORTED`, `MEMORY_BACKEND_CREATE_FAILED`,
//   `MEMORY_BACKEND_NOT_FOUND`, `OUT_OF_MEMORY`, `INVALID_FREE`,
//   `DOUBLE_FREE`, `IPC_ARGS_NOT_SHM_COMPATIBLE`,
//   `UNORDERED_MAP_CANT_FIND`, `KEY_SET_OUT_OF_BOUNDS`,
//   `ARGPACK_INDEX_OUT_OF_BOUNDS`) → `pub const` items of the same names.
// - `CTP_THROW_ERROR(CODE, ...)` → idiomatic Rust: return
//   `Err(CODE.format(&[...]))`; no macro is provided (Rust has no
//   exceptions, and the GPU no-op variant is moot here).
//
// Semantic divergences (explicit):
// 1. A format string whose final byte is '{' (malformed placeholder at the
//    very end, e.g. `"a{"`): C++ computes an out-of-range trailing substring
//    and `std::string::substr` throws `std::out_of_range`; Rust saturates
//    and treats the trailing segment as empty (no panic).
// 2. If placeholder byte offsets fall inside a multi-byte UTF-8 sequence
//    (only possible with malformed placeholders such as `"{é"`), the C++
//    code emits raw split bytes; Rust replaces the broken sequence via
//    lossy UTF-8 conversion (U+FFFD) instead of producing invalid UTF-8.
// 3. `Error::is` replaces the C++ `CTP_ERROR_IS` macro whose `get_code()`
//    target never existed; it compares format strings by content.
// 4. The exception-handling macros (`CTP_ERROR_HANDLE_*`,
//    `CTP_THROW_ERROR`, `CTP_THROW_STD_ERROR`) are not ported: Rust uses
//    `Result<T, Error>` and `?` instead of exceptions.

use std::fmt::{self, Display, Write as _};

/// Splits `fmt` at every `'{'` byte, mirroring `ctp::Formatter::tokenize`.
///
/// Returns `(start, len)` byte ranges: the ranges BETWEEN placeholders.
/// A well-formed placeholder is `{}` (two bytes); like the C++ code, any
/// `'{'` unconditionally consumes the following byte as well.
fn tokenize(fmt: &[u8]) -> Vec<(usize, usize)> {
    let mut offsets: Vec<(usize, usize)> = vec![(0, fmt.len())];
    let mut i = 0;
    while i < fmt.len() {
        if fmt[i] == b'{' {
            // Close the prior substring at the '{'.
            let prior = offsets
                .last_mut()
                .expect("offsets is never empty: seeded with one entry");
            prior.1 = i - prior.0;
            // Skip "{}" (two bytes). Divergence 1: clamp instead of letting
            // the trailing range run out of bounds like C++ does.
            i += 2;
            let start = i.min(fmt.len());
            offsets.push((start, fmt.len() - start));
            continue;
        }
        i += 1;
    }
    offsets
}

/// Mirrors `ctp::Formatter::format`: interleaves `args` into the `{}`
/// placeholders of `fmt`. If the number of placeholders does not equal
/// `args.len()`, returns `fmt` unchanged (verbatim), exactly like C++.
fn format_args(fmt: &str, args: &[&dyn Display]) -> String {
    let bytes = fmt.as_bytes();
    let offsets = tokenize(bytes);
    if offsets.len() != args.len() + 1 {
        return fmt.to_string();
    }
    let mut out = String::new();
    for (idx, arg) in args.iter().enumerate() {
        let (start, len) = offsets[idx];
        out.push_str(&String::from_utf8_lossy(&bytes[start..start + len]));
        // Writing into a String cannot fail.
        let _ = write!(out, "{arg}");
    }
    let (start, len) = *offsets
        .last()
        .expect("offsets is never empty: seeded with one entry");
    if len > 0 {
        out.push_str(&String::from_utf8_lossy(&bytes[start..start + len]));
    }
    out
}

/// Port of `ctp::Error`: an error kind identified by a static format
/// string, plus the message produced once [`Error::format`] is applied.
///
/// Like the C++ class, the predefined constants carry an empty message
/// until `format` is called on them.
#[derive(Debug, Clone)]
pub struct Error {
    /// The compile-time format string (C++ `fmt_`).
    fmt: &'static str,
    /// The formatted message (C++ `msg_`); empty until `format` runs.
    msg: String,
}

impl Error {
    /// Port of `ctp::Error::Error(const char *fmt)`.
    #[must_use]
    pub const fn new(fmt: &'static str) -> Self {
        Self {
            fmt,
            msg: String::new(),
        }
    }

    /// Port of `ctp::Error::format(args...)`: returns a NEW `Error` of the
    /// same kind whose message is the format string with `args` substituted
    /// for the `{}` placeholders, in order. If the argument count does not
    /// match the placeholder count, the message is the format string
    /// verbatim (C++ `Formatter` behavior).
    #[must_use]
    pub fn format(&self, args: &[&dyn Display]) -> Self {
        Self {
            fmt: self.fmt,
            msg: format_args(self.fmt, args),
        }
    }

    /// Port of `ctp::Error::what()`: the formatted message. Empty if
    /// [`Error::format`] has not been applied.
    #[must_use]
    pub fn what(&self) -> &str {
        &self.msg
    }

    /// The static format string identifying this error kind (C++ `fmt_`).
    #[must_use]
    pub const fn fmt_str(&self) -> &'static str {
        self.fmt
    }

    /// Replacement for the C++ `CTP_ERROR_IS(err, check)` macro: true when
    /// both errors were created from the same format string (same kind),
    /// regardless of the formatted message.
    #[must_use]
    pub fn is(&self, other: &Self) -> bool {
        self.fmt == other.fmt
    }

    /// Port of `ctp::Error::print()`: writes the message to stdout.
    pub fn print(&self) {
        println!("{}", self.what());
    }
}

impl Default for Error {
    /// Port of the C++ default constructor (null format, empty message).
    fn default() -> Self {
        Self::new("")
    }
}

impl Display for Error {
    /// Same text as [`Error::what`] (and the C++ `what()`).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.msg)
    }
}

impl std::error::Error for Error {}

// ---------------------------------------------------------------------------
// Predefined errors — port of `clio_ctp/util/errors.h`.
// ---------------------------------------------------------------------------

pub const MEMORY_BACKEND_REPEATED: Error =
    Error::new("Attempted to register two backends with the same id");
pub const TOO_MANY_ALLOCATORS: Error = Error::new("Too many allocators");
pub const NOT_IMPLEMENTED: Error = Error::new("{} not implemented");

pub const SHMEM_CREATE_FAILED: Error = Error::new("Failed to allocate SHMEM");
pub const SHMEM_RESERVE_FAILED: Error = Error::new("Failed to reserve SHMEM");
pub const SHMEM_NOT_SUPPORTED: Error = Error::new("Attempting to deserialize a non-shm backend");
pub const MEMORY_BACKEND_CREATE_FAILED: Error = Error::new("Failed to load memory backend");
pub const MEMORY_BACKEND_NOT_FOUND: Error = Error::new("Failed to find the memory backend");
pub const OUT_OF_MEMORY: Error =
    Error::new("could not allocate memory of size {} from heap of size {}");
pub const INVALID_FREE: Error = Error::new("could not free memory");
pub const DOUBLE_FREE: Error = Error::new("Freeing the same memory twice: {}!");

pub const IPC_ARGS_NOT_SHM_COMPATIBLE: Error = Error::new("Args are not compatible with SHM");

pub const UNORDERED_MAP_CANT_FIND: Error = Error::new("Could not find key in unordered_map");
pub const KEY_SET_OUT_OF_BOUNDS: Error = Error::new("Too many keys in the key set");

pub const ARGPACK_INDEX_OUT_OF_BOUNDS: Error = Error::new("Argpack index out of bounds");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unformatted_error_has_empty_message() {
        // C++ parity: what() is msg_.c_str(), empty until format() runs.
        assert_eq!(NOT_IMPLEMENTED.what(), "");
        assert_eq!(NOT_IMPLEMENTED.to_string(), "");
        assert_eq!(Error::default().what(), "");
        assert_eq!(Error::default().fmt_str(), "");
    }

    #[test]
    fn format_single_placeholder() {
        let err = NOT_IMPLEMENTED.format(&[&"MyFunction"]);
        assert_eq!(err.what(), "MyFunction not implemented");
        assert_eq!(err.to_string(), "MyFunction not implemented");
    }

    #[test]
    fn format_two_placeholders_in_order() {
        let err = OUT_OF_MEMORY.format(&[&64usize, &32usize]);
        assert_eq!(
            err.what(),
            "could not allocate memory of size 64 from heap of size 32"
        );
    }

    #[test]
    fn format_no_placeholders_zero_args() {
        let err = TOO_MANY_ALLOCATORS.format(&[]);
        assert_eq!(err.what(), "Too many allocators");
    }

    #[test]
    fn format_arg_count_mismatch_returns_fmt_verbatim() {
        // Too few args: C++ Formatter returns the format string unchanged.
        let err = OUT_OF_MEMORY.format(&[&64usize]);
        assert_eq!(
            err.what(),
            "could not allocate memory of size {} from heap of size {}"
        );
        // Too many args.
        let err = NOT_IMPLEMENTED.format(&[&"a", &"b"]);
        assert_eq!(err.what(), "{} not implemented");
        // Args against a placeholder-free string.
        let err = INVALID_FREE.format(&[&1u8]);
        assert_eq!(err.what(), "could not free memory");
    }

    #[test]
    fn format_empty_format_string() {
        let empty = Error::new("");
        assert_eq!(empty.format(&[]).what(), "");
        // Mismatch on empty fmt returns the (empty) fmt verbatim.
        assert_eq!(empty.format(&[&1u32]).what(), "");
    }

    #[test]
    fn format_placeholder_only() {
        let e = Error::new("{}");
        assert_eq!(e.format(&[&42i32]).what(), "42");
    }

    #[test]
    fn format_placeholder_at_start_and_end() {
        assert_eq!(Error::new("{} tail").format(&[&"x"]).what(), "x tail");
        assert_eq!(Error::new("head {}").format(&[&"y"]).what(), "head y");
    }

    #[test]
    fn format_adjacent_placeholders() {
        let e = Error::new("{}{}");
        assert_eq!(e.format(&[&"a", &"b"]).what(), "ab");
    }

    #[test]
    fn format_trailing_placeholder_message_ends_with_arg() {
        let err = DOUBLE_FREE.format(&[&"0xdeadbeef"]);
        assert_eq!(err.what(), "Freeing the same memory twice: 0xdeadbeef!");
    }

    #[test]
    fn format_negative_and_boundary_numbers() {
        let e = Error::new("v={}");
        assert_eq!(e.format(&[&i64::MIN]).what(), "v=-9223372036854775808");
        assert_eq!(e.format(&[&u64::MAX]).what(), "v=18446744073709551615");
        assert_eq!(e.format(&[&0u8]).what(), "v=0");
    }

    #[test]
    fn format_non_ascii_fmt_and_args() {
        let e = Error::new("héllo {} wörld");
        assert_eq!(e.format(&[&"日本語"]).what(), "héllo 日本語 wörld");
    }

    #[test]
    fn format_malformed_trailing_open_brace() {
        // Divergence 1: C++ throws std::out_of_range here; Rust treats the
        // out-of-range trailing segment as empty.
        let e = Error::new("a{");
        assert_eq!(e.format(&[&"X"]).what(), "aX");
    }

    #[test]
    fn format_malformed_brace_with_content_matches_cpp_skip_two() {
        // C++ parity quirk: '{' unconditionally consumes the next byte, so
        // "{x}world" has ONE placeholder covering "{x" and the remainder is
        // "}world".
        let e = Error::new("{x}world");
        assert_eq!(e.format(&[&1i32]).what(), "1}world");
        // With a mismatched count it comes back verbatim.
        assert_eq!(e.format(&[]).what(), "{x}world");
    }

    #[test]
    fn format_does_not_mutate_original() {
        let base = NOT_IMPLEMENTED;
        let formatted = base.format(&[&"f"]);
        assert_eq!(base.what(), "");
        assert_eq!(formatted.what(), "f not implemented");
        // Kind identity is preserved through format().
        assert!(formatted.is(&NOT_IMPLEMENTED));
    }

    #[test]
    fn is_compares_error_kind_not_message() {
        let a = DOUBLE_FREE.format(&[&1u32]);
        let b = DOUBLE_FREE.format(&[&2u32]);
        assert!(a.is(&b));
        assert!(a.is(&DOUBLE_FREE));
        assert!(!a.is(&INVALID_FREE));
        assert!(!Error::default().is(&DOUBLE_FREE));
    }

    #[test]
    fn predefined_constants_have_expected_format_strings() {
        assert_eq!(
            MEMORY_BACKEND_REPEATED.fmt_str(),
            "Attempted to register two backends with the same id"
        );
        assert_eq!(SHMEM_CREATE_FAILED.fmt_str(), "Failed to allocate SHMEM");
        assert_eq!(SHMEM_RESERVE_FAILED.fmt_str(), "Failed to reserve SHMEM");
        assert_eq!(
            SHMEM_NOT_SUPPORTED.fmt_str(),
            "Attempting to deserialize a non-shm backend"
        );
        assert_eq!(
            MEMORY_BACKEND_CREATE_FAILED.fmt_str(),
            "Failed to load memory backend"
        );
        assert_eq!(
            MEMORY_BACKEND_NOT_FOUND.fmt_str(),
            "Failed to find the memory backend"
        );
        assert_eq!(
            IPC_ARGS_NOT_SHM_COMPATIBLE.fmt_str(),
            "Args are not compatible with SHM"
        );
        assert_eq!(
            UNORDERED_MAP_CANT_FIND.fmt_str(),
            "Could not find key in unordered_map"
        );
        assert_eq!(
            KEY_SET_OUT_OF_BOUNDS.fmt_str(),
            "Too many keys in the key set"
        );
        assert_eq!(
            ARGPACK_INDEX_OUT_OF_BOUNDS.fmt_str(),
            "Argpack index out of bounds"
        );
    }

    #[test]
    fn works_as_std_error_trait_object() {
        fn fails() -> Result<(), Box<dyn std::error::Error>> {
            Err(Box::new(NOT_IMPLEMENTED.format(&[&"feature"])))
        }
        let err = fails().unwrap_err();
        assert_eq!(err.to_string(), "feature not implemented");
    }

    #[test]
    fn error_is_clone_and_debug() {
        let a = OUT_OF_MEMORY.format(&[&1u8, &2u8]);
        let b = a.clone();
        assert_eq!(a.what(), b.what());
        assert!(format!("{a:?}").contains("Error"));
    }

    #[test]
    fn tokenize_counts_placeholders() {
        assert_eq!(tokenize(b"").len(), 1);
        assert_eq!(tokenize(b"no braces").len(), 1);
        assert_eq!(tokenize(b"{}").len(), 2);
        assert_eq!(tokenize(b"a{}b{}c").len(), 3);
        // Trailing '{' still opens a (clamped, empty) final segment.
        assert_eq!(tokenize(b"a{"), vec![(0, 1), (2, 0)]);
    }
}
