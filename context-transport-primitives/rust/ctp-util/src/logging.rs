// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).
//
//! Rust port of `clio_ctp/util/logging.h` — the HLOG logging facade.
//!
//! # C++ → Rust name mapping
//!
//! | C++ | Rust |
//! |---|---|
//! | `kDebug` / `kInfo` / `kSuccess` / `kWarning` / `kError` / `kFatal` | [`LogLevel::Debug`] / [`Info`](LogLevel::Info) / [`Success`](LogLevel::Success) / [`Warning`](LogLevel::Warning) / [`Error`](LogLevel::Error) / [`Fatal`](LogLevel::Fatal) (same integer codes 0..=5) |
//! | `ctp::Logger` | [`Logger`] |
//! | `CTP_LOG` (CrossSingleton) | [`Logger::global`] (process-wide `OnceLock`) |
//! | `Logger::GetLevelString` | [`level_string`] / [`LogLevel::as_str`] |
//! | `Logger::GetLevelColor` | [`level_color`] / [`LogLevel::color`] |
//! | `Logger::ShouldLog` | [`Logger::should_log`] |
//! | `Logger::Log<LOG_CODE>(...)` | [`Logger::log`] |
//! | `Logger::Print(...)` | [`Logger::print`] |
//! | `HLOG(LOG_CODE, ...)` | [`hlog!`](crate::hlog) |
//! | `HIPRINT(...)` | [`hiprint!`](crate::hiprint) |
//! | (no C++ equivalent) | level helpers [`hdebug!`](crate::hdebug), [`hinfo!`](crate::hinfo), [`hsuccess!`](crate::hsuccess), [`hwarning!`](crate::hwarning), [`herror!`](crate::herror), [`hfatal!`](crate::hfatal) |
//! | `CTP_LOG_LEVEL` env var | same env var name, parsed by [`parse_log_level`] |
//! | `CTP_LOG_OUT` env var | same env var name (log-file output, colors stripped) |
//!
//! Output format matches the C++ stderr line:
//! `{color}{file}:{line} {LEVEL} {tid} {function} {message}{reset}\n`
//! and the color-free file line `{file}:{line} {LEVEL} {tid} {function} {message}\n`.
//! All log output goes to stderr (never stdout), as in the C++ (stdout is
//! reserved for a program's actual data output). `Fatal` terminates the
//! process with exit code 1, as in the C++.
//!
//! # Semantic divergences from the C++
//!
//! - **No compile-time filtering.** The C++ `HLOG` compiles out messages below
//!   the `CTP_LOG_LEVEL` *macro* (default `kInfo`); Rust has no equivalent, so
//!   filtering is purely runtime. The default runtime threshold is `Info`
//!   (matching the C++ default), but unlike a default C++ build, setting the
//!   `CTP_LOG_LEVEL` env var to `debug`/`0` actually enables debug messages.
//! - **Level-name parsing is fully case-insensitive** (`Debug`, `dEbUg`, ...).
//!   The C++ only accepts all-lowercase or all-UPPERCASE names.
//! - **Numeric parsing follows `std::stoi` semantics** (leading whitespace,
//!   optional sign, trailing junk ignored: `" 3"`, `"3abc"`, `"3.5"` → 3).
//!   Out-of-`i32`-range input keeps the default in both implementations.
//! - **Thread id** is Rust's internal `std::thread::ThreadId` counter, not the
//!   OS tid returned by `SystemInfo::GetTid()`. It is still unique per live
//!   thread within the process.
//! - **Function name** (from [`function_name!`](crate::function_name)) is the
//!   fully-qualified Rust path (e.g. `my_crate::module::func`), not the bare
//!   `__func__` name.
//! - **`CTP_LOG_OUT` open failure** is silently ignored in both, but Rust uses
//!   `File::create` (truncate-on-open, like the C++ `fopen(..., "w")`).
//! - The C++ device-pass no-op variant of `HLOG` has no Rust counterpart
//!   (these crates are host-only).

use std::fs::File;
use std::io::Write;
use std::sync::{Mutex, OnceLock};

/// Log levels, with the same integer codes as the C++ `kDebug`..`kFatal`
/// macros. Lower values are more verbose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(i32)]
pub enum LogLevel {
    /// Low-priority debugging information (C++ `kDebug` = 0).
    Debug = 0,
    /// Useful information the user should know (C++ `kInfo` = 1).
    Info = 1,
    /// Operation completed successfully (C++ `kSuccess` = 2).
    Success = 2,
    /// Something might be wrong (C++ `kWarning` = 3).
    Warning = 3,
    /// A non-fatal error has occurred (C++ `kError` = 4).
    Error = 4,
    /// A fatal error has occurred; logging it exits the process (C++ `kFatal` = 5).
    Fatal = 5,
}

impl LogLevel {
    /// Integer code of this level (identical to the C++ macro values).
    pub const fn as_i32(self) -> i32 {
        self as i32
    }

    /// The level whose integer code is `code`, if any.
    pub const fn from_i32(code: i32) -> Option<LogLevel> {
        match code {
            0 => Some(LogLevel::Debug),
            1 => Some(LogLevel::Info),
            2 => Some(LogLevel::Success),
            3 => Some(LogLevel::Warning),
            4 => Some(LogLevel::Error),
            5 => Some(LogLevel::Fatal),
            _ => None,
        }
    }

    /// String name of this level (C++ `Logger::GetLevelString`).
    pub const fn as_str(self) -> &'static str {
        level_string(self.as_i32())
    }

    /// ANSI color escape for this level (C++ `Logger::GetLevelColor`).
    pub const fn color(self) -> &'static str {
        level_color(self.as_i32())
    }
}

/// String name for a raw level code (C++ `Logger::GetLevelString`).
/// Unknown codes yield `"UNKNOWN"`.
pub const fn level_string(level: i32) -> &'static str {
    match level {
        0 => "DEBUG",
        1 => "INFO",
        2 => "SUCCESS",
        3 => "WARNING",
        4 => "ERROR",
        5 => "FATAL",
        _ => "UNKNOWN",
    }
}

/// ANSI color escape for a raw level code (C++ `Logger::GetLevelColor`).
/// Unknown codes yield the reset sequence.
pub const fn level_color(level: i32) -> &'static str {
    match level {
        0 => "\x1b[90m",     // Dark grey
        1 => "\x1b[97m",     // White
        2 => "\x1b[32m",     // Green
        3 => "\x1b[33m",     // Yellow
        4 | 5 => "\x1b[31m", // Red
        _ => "\x1b[0m",      // Reset
    }
}

/// ANSI reset sequence appended after every colored stderr line.
const ANSI_RESET: &str = "\x1b[0m";

/// Parse a `CTP_LOG_LEVEL` env-var value into a level code.
///
/// Accepts the level names (case-insensitively — a superset of the C++,
/// which only matches all-lower/all-upper) and any integer that fits in
/// `i32`. Returns `None` for empty or unparsable input, in which case the
/// caller keeps its default (mirroring the C++ "keep default on parse
/// failure" behavior).
pub fn parse_log_level(value: &str) -> Option<i32> {
    if value.is_empty() {
        return None;
    }
    if value.eq_ignore_ascii_case("debug") {
        return Some(LogLevel::Debug.as_i32());
    }
    if value.eq_ignore_ascii_case("info") {
        return Some(LogLevel::Info.as_i32());
    }
    if value.eq_ignore_ascii_case("success") {
        return Some(LogLevel::Success.as_i32());
    }
    if value.eq_ignore_ascii_case("warning") {
        return Some(LogLevel::Warning.as_i32());
    }
    if value.eq_ignore_ascii_case("error") {
        return Some(LogLevel::Error.as_i32());
    }
    if value.eq_ignore_ascii_case("fatal") {
        return Some(LogLevel::Fatal.as_i32());
    }
    // Numeric fallback with C++ std::stoi semantics (parity fix from the
    // adversarial review): skip leading whitespace, accept an optional sign,
    // consume the leading digit run, and IGNORE trailing junk — so " 3",
    // "3 ", "3abc", and "3.5" all yield 3 exactly as a C++ deployment's env
    // value would. No digits at all (or overflow) → None (keep default,
    // matching stoi's thrown-and-caught path).
    let trimmed = value.trim_start();
    let (sign, digits_part) = match trimmed.strip_prefix('-') {
        Some(rest) => (-1i64, rest),
        None => (1i64, trimmed.strip_prefix('+').unwrap_or(trimmed)),
    };
    let digits: &str = &digits_part[..digits_part
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(digits_part.len())];
    if digits.is_empty() {
        return None;
    }
    digits
        .parse::<i64>()
        .ok()
        .map(|n| n * sign)
        .and_then(|n| i32::try_from(n).ok())
}

/// Current thread id as an integer. This is Rust's internal `ThreadId`
/// counter (unique per live thread in this process), NOT the OS tid the C++
/// `SystemInfo::GetTid()` returns — see the module-level divergence notes.
pub fn current_tid() -> u64 {
    // ThreadId formats as "ThreadId(N)"; extract N without unsafe or
    // unstable APIs. Fall back to 0 if the format ever changes.
    let repr = format!("{:?}", std::thread::current().id());
    let digits: String = repr.chars().filter(|c| c.is_ascii_digit()).collect();
    digits.parse::<u64>().unwrap_or(0)
}

/// Build the colored stderr line (C++ `Logger::Log` stderr format):
/// `{color}{file}:{line} {LEVEL} {tid} {func} {msg}{reset}\n`.
fn format_stderr_line(
    level: LogLevel,
    file: &str,
    func: &str,
    line: u32,
    tid: u64,
    msg: &str,
) -> String {
    format!(
        "{}{}:{} {} {} {} {}{}\n",
        level.color(),
        file,
        line,
        level.as_str(),
        tid,
        func,
        msg,
        ANSI_RESET
    )
}

/// Build the color-free log-file line (C++ `Logger::Log` file format):
/// `{file}:{line} {LEVEL} {tid} {func} {msg}\n`.
fn format_file_line(
    level: LogLevel,
    file: &str,
    func: &str,
    line: u32,
    tid: u64,
    msg: &str,
) -> String {
    format!(
        "{}:{} {} {} {} {}\n",
        file,
        line,
        level.as_str(),
        tid,
        func,
        msg
    )
}

/// Logging sink with runtime level filtering (C++ `ctp::Logger`).
///
/// Most code should use the [`hlog!`](crate::hlog) / level-helper macros,
/// which route through the process-wide [`Logger::global`] instance.
pub struct Logger {
    /// Runtime log level threshold (C++ `runtime_log_level_`). Levels with
    /// code `>=` this value are emitted.
    runtime_log_level: i32,
    /// Optional log-file sink (C++ `fout_`, from `CTP_LOG_OUT`).
    fout: Option<Mutex<File>>,
}

impl Logger {
    /// Default threshold, mirroring the C++ compile-time default
    /// `CTP_LOG_LEVEL = kInfo`.
    pub const DEFAULT_LEVEL: i32 = LogLevel::Info as i32;

    /// Build a logger from the environment, exactly like the C++ constructor:
    /// `CTP_LOG_LEVEL` overrides the threshold (default `Info`) and
    /// `CTP_LOG_OUT` names an optional log file (truncated on open; open
    /// failures are silently ignored).
    pub fn from_env() -> Logger {
        let mut level = Self::DEFAULT_LEVEL;
        if let Ok(value) = std::env::var("CTP_LOG_LEVEL") {
            if let Some(parsed) = parse_log_level(&value) {
                level = parsed;
            }
        }
        let fout = std::env::var("CTP_LOG_OUT")
            .ok()
            .filter(|path| !path.is_empty())
            .and_then(|path| File::create(path).ok())
            .map(Mutex::new);
        Logger {
            runtime_log_level: level,
            fout,
        }
    }

    /// Build a logger with an explicit threshold and optional file sink
    /// (embedding/test hook; the C++ singleton offers no equivalent).
    pub fn with_options(runtime_log_level: i32, fout: Option<File>) -> Logger {
        Logger {
            runtime_log_level,
            fout: fout.map(Mutex::new),
        }
    }

    /// Process-wide logger (C++ `CTP_LOG` singleton). Initialized from the
    /// environment on first use via `OnceLock` (never `thread_local`).
    pub fn global() -> &'static Logger {
        static LOGGER: OnceLock<Logger> = OnceLock::new();
        LOGGER.get_or_init(Logger::from_env)
    }

    /// Runtime threshold currently in effect.
    pub fn runtime_log_level(&self) -> i32 {
        self.runtime_log_level
    }

    /// Whether a message at `level` passes the runtime filter
    /// (C++ `Logger::ShouldLog`).
    pub fn should_log(&self, level: LogLevel) -> bool {
        level.as_i32() >= self.runtime_log_level
    }

    /// Emit one log record (C++ `Logger::Log<LOG_CODE>`): colored line to
    /// stderr (flushed), color-free line to the `CTP_LOG_OUT` file if open.
    /// `Fatal` terminates the process with exit code 1 after emitting.
    ///
    /// Prefer the [`hlog!`](crate::hlog) macro, which captures
    /// file/line/function automatically and skips formatting when the level
    /// is filtered out.
    pub fn log(
        &self,
        level: LogLevel,
        file: &str,
        func: &str,
        line: u32,
        args: std::fmt::Arguments<'_>,
    ) {
        if self.should_log(level) {
            let msg = args.to_string();
            let tid = current_tid();
            let out = format_stderr_line(level, file, func, line, tid, &msg);
            let stderr = std::io::stderr();
            let mut handle = stderr.lock();
            let _ = handle.write_all(out.as_bytes());
            let _ = handle.flush();
            drop(handle);
            if let Some(fout) = &self.fout {
                let file_out = format_file_line(level, file, func, line, tid, &msg);
                if let Ok(mut f) = fout.lock() {
                    let _ = f.write_all(file_out.as_bytes());
                    let _ = f.flush();
                }
            }
            // Fatal errors terminate the program (C++ exit(1)). As in the
            // C++, a Fatal message suppressed by the runtime filter (only
            // possible with a numeric threshold > 5) does NOT exit.
            if level == LogLevel::Fatal {
                std::process::exit(1);
            }
        }
    }

    /// Print a message plus newline to stderr and the optional log file,
    /// with no prefix or filtering (C++ `Logger::Print` / `HIPRINT`).
    pub fn print(&self, args: std::fmt::Arguments<'_>) {
        let out = format!("{args}\n");
        // Logs go to stderr, never stdout: stdout is reserved for a
        // program's actual data output.
        let stderr = std::io::stderr();
        let mut handle = stderr.lock();
        let _ = handle.write_all(out.as_bytes());
        let _ = handle.flush();
        drop(handle);
        if let Some(fout) = &self.fout {
            if let Ok(mut f) = fout.lock() {
                let _ = f.write_all(out.as_bytes());
            }
        }
    }
}

/// Fully-qualified name of the enclosing function (best-effort stand-in for
/// the C++ `__func__`; yields a Rust path like `crate::module::func`).
#[macro_export]
macro_rules! function_name {
    () => {{
        fn __ctp_fn_probe() {}
        let name = ::std::any::type_name_of_val(&__ctp_fn_probe);
        name.strip_suffix("::__ctp_fn_probe").unwrap_or(name)
    }};
}

/// Unified logging macro (C++ `HLOG(LOG_CODE, ...)`).
///
/// `hlog!(LogLevel::Warning, "took {} ms", elapsed)` emits
/// `file:line LEVEL tid function message` to stderr (and `CTP_LOG_OUT` if
/// set), subject to the runtime `CTP_LOG_LEVEL` filter. Formatting is
/// skipped entirely when the level is filtered out. `LogLevel::Fatal` exits
/// the process with code 1.
#[macro_export]
macro_rules! hlog {
    ($level:expr, $($arg:tt)*) => {{
        let __ctp_level: $crate::logging::LogLevel = $level;
        let __ctp_logger = $crate::logging::Logger::global();
        if __ctp_logger.should_log(__ctp_level) {
            __ctp_logger.log(
                __ctp_level,
                file!(),
                $crate::function_name!(),
                line!(),
                format_args!($($arg)*),
            );
        }
    }};
}

/// Print like `println!` but to stderr + optional log file (C++ `HIPRINT`).
#[macro_export]
macro_rules! hiprint {
    ($($arg:tt)*) => {
        $crate::logging::Logger::global().print(format_args!($($arg)*))
    };
}

/// `hlog!` at `LogLevel::Debug` (C++ `HLOG(kDebug, ...)`).
#[macro_export]
macro_rules! hdebug {
    ($($arg:tt)*) => { $crate::hlog!($crate::logging::LogLevel::Debug, $($arg)*) };
}

/// `hlog!` at `LogLevel::Info` (C++ `HLOG(kInfo, ...)`).
#[macro_export]
macro_rules! hinfo {
    ($($arg:tt)*) => { $crate::hlog!($crate::logging::LogLevel::Info, $($arg)*) };
}

/// `hlog!` at `LogLevel::Success` (C++ `HLOG(kSuccess, ...)`).
#[macro_export]
macro_rules! hsuccess {
    ($($arg:tt)*) => { $crate::hlog!($crate::logging::LogLevel::Success, $($arg)*) };
}

/// `hlog!` at `LogLevel::Warning` (C++ `HLOG(kWarning, ...)`).
#[macro_export]
macro_rules! hwarning {
    ($($arg:tt)*) => { $crate::hlog!($crate::logging::LogLevel::Warning, $($arg)*) };
}

/// `hlog!` at `LogLevel::Error` (C++ `HLOG(kError, ...)`).
#[macro_export]
macro_rules! herror {
    ($($arg:tt)*) => { $crate::hlog!($crate::logging::LogLevel::Error, $($arg)*) };
}

/// `hlog!` at `LogLevel::Fatal` (C++ `HLOG(kFatal, ...)`); exits the process
/// with code 1 after emitting.
#[macro_export]
macro_rules! hfatal {
    ($($arg:tt)*) => { $crate::hlog!($crate::logging::LogLevel::Fatal, $($arg)*) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    // ---- Level codes and names ----

    /// std::stoi parity for numeric CTP_LOG_LEVEL values (adversarial-review
    /// fix): leading whitespace, trailing junk, and fractional tails must
    /// all parse like a C++ deployment's env value would.
    #[test]
    fn parse_log_level_stoi_semantics() {
        assert_eq!(parse_log_level(" 3"), Some(3));
        assert_eq!(parse_log_level("3 "), Some(3));
        assert_eq!(parse_log_level("3abc"), Some(3));
        assert_eq!(parse_log_level("3.5"), Some(3));
        assert_eq!(parse_log_level("0x2"), Some(0)); // stoi stops at 'x'
        assert_eq!(parse_log_level("+2"), Some(2));
        assert_eq!(parse_log_level("-1"), Some(-1));
        assert_eq!(parse_log_level("abc"), None);
        assert_eq!(parse_log_level(""), None);
        assert_eq!(parse_log_level("99999999999999999999"), None); // overflow
    }

    #[test]
    fn level_codes_match_cpp_macros() {
        assert_eq!(LogLevel::Debug.as_i32(), 0);
        assert_eq!(LogLevel::Info.as_i32(), 1);
        assert_eq!(LogLevel::Success.as_i32(), 2);
        assert_eq!(LogLevel::Warning.as_i32(), 3);
        assert_eq!(LogLevel::Error.as_i32(), 4);
        assert_eq!(LogLevel::Fatal.as_i32(), 5);
    }

    #[test]
    fn from_i32_roundtrip_and_unknown() {
        for code in 0..=5 {
            let level = LogLevel::from_i32(code).unwrap();
            assert_eq!(level.as_i32(), code);
        }
        assert_eq!(LogLevel::from_i32(-1), None);
        assert_eq!(LogLevel::from_i32(6), None);
        assert_eq!(LogLevel::from_i32(i32::MAX), None);
        assert_eq!(LogLevel::from_i32(i32::MIN), None);
    }

    #[test]
    fn level_strings_match_cpp() {
        assert_eq!(level_string(0), "DEBUG");
        assert_eq!(level_string(1), "INFO");
        assert_eq!(level_string(2), "SUCCESS");
        assert_eq!(level_string(3), "WARNING");
        assert_eq!(level_string(4), "ERROR");
        assert_eq!(level_string(5), "FATAL");
        assert_eq!(level_string(-7), "UNKNOWN");
        assert_eq!(level_string(42), "UNKNOWN");
        assert_eq!(LogLevel::Warning.as_str(), "WARNING");
    }

    #[test]
    fn level_colors_match_cpp() {
        assert_eq!(level_color(0), "\x1b[90m");
        assert_eq!(level_color(1), "\x1b[97m");
        assert_eq!(level_color(2), "\x1b[32m");
        assert_eq!(level_color(3), "\x1b[33m");
        assert_eq!(level_color(4), "\x1b[31m");
        assert_eq!(level_color(5), "\x1b[31m");
        assert_eq!(level_color(99), "\x1b[0m"); // reset for unknown
        assert_eq!(LogLevel::Error.color(), "\x1b[31m");
    }

    // ---- Env-value parsing ----

    #[test]
    fn parse_empty_is_none() {
        assert_eq!(parse_log_level(""), None);
    }

    #[test]
    fn parse_names_lower_and_upper() {
        // The exact spellings the C++ accepts.
        assert_eq!(parse_log_level("debug"), Some(0));
        assert_eq!(parse_log_level("DEBUG"), Some(0));
        assert_eq!(parse_log_level("info"), Some(1));
        assert_eq!(parse_log_level("INFO"), Some(1));
        assert_eq!(parse_log_level("success"), Some(2));
        assert_eq!(parse_log_level("SUCCESS"), Some(2));
        assert_eq!(parse_log_level("warning"), Some(3));
        assert_eq!(parse_log_level("WARNING"), Some(3));
        assert_eq!(parse_log_level("error"), Some(4));
        assert_eq!(parse_log_level("ERROR"), Some(4));
        assert_eq!(parse_log_level("fatal"), Some(5));
        assert_eq!(parse_log_level("FATAL"), Some(5));
    }

    #[test]
    fn parse_names_mixed_case_divergence() {
        // Documented divergence: Rust accepts any case, a superset of C++.
        assert_eq!(parse_log_level("Debug"), Some(0));
        assert_eq!(parse_log_level("wArNiNg"), Some(3));
        assert_eq!(parse_log_level("Fatal"), Some(5));
    }

    #[test]
    fn parse_numeric_values() {
        for code in 0..=5 {
            assert_eq!(parse_log_level(&code.to_string()), Some(code));
        }
        // Arbitrary ints are allowed, as with the C++ stoi fallback.
        assert_eq!(parse_log_level("7"), Some(7));
        assert_eq!(parse_log_level("-1"), Some(-1));
        assert_eq!(parse_log_level(&i32::MAX.to_string()), Some(i32::MAX));
        assert_eq!(parse_log_level(&i32::MIN.to_string()), Some(i32::MIN));
    }

    #[test]
    fn parse_garbage_and_overflow_keep_default() {
        assert_eq!(parse_log_level("verbose"), None);
        assert_eq!(parse_log_level("debu"), None);
        assert_eq!(parse_log_level("debugx"), None);
        // Out of i32 range → None (C++: stoi throws, caught, default kept).
        assert_eq!(parse_log_level("99999999999999999999"), None);
        assert_eq!(parse_log_level("-99999999999999999999"), None);
        // stoi-leniency cases live in parse_log_level_stoi_semantics.
    }

    // ---- Runtime filtering ----

    #[test]
    fn should_log_boundaries() {
        let logger = Logger::with_options(LogLevel::Warning.as_i32(), None);
        assert!(!logger.should_log(LogLevel::Debug));
        assert!(!logger.should_log(LogLevel::Info));
        assert!(!logger.should_log(LogLevel::Success));
        assert!(logger.should_log(LogLevel::Warning)); // level == threshold
        assert!(logger.should_log(LogLevel::Error));
        assert!(logger.should_log(LogLevel::Fatal));
    }

    #[test]
    fn should_log_extreme_thresholds() {
        let allow_all = Logger::with_options(i32::MIN, None);
        assert!(allow_all.should_log(LogLevel::Debug));
        let deny_all = Logger::with_options(i32::MAX, None);
        assert!(!deny_all.should_log(LogLevel::Fatal));
        // ... though hlog!/log() still exits on Fatal regardless of filter.
    }

    #[test]
    fn default_level_is_info() {
        assert_eq!(Logger::DEFAULT_LEVEL, 1);
        let logger = Logger::with_options(Logger::DEFAULT_LEVEL, None);
        assert!(!logger.should_log(LogLevel::Debug));
        assert!(logger.should_log(LogLevel::Info));
    }

    // ---- Line formatting ----

    #[test]
    fn stderr_line_format_matches_cpp() {
        let line = format_stderr_line(
            LogLevel::Warning,
            "src/x.rs",
            "my::func",
            42,
            7,
            "hello world",
        );
        assert_eq!(
            line,
            "\x1b[33msrc/x.rs:42 WARNING 7 my::func hello world\x1b[0m\n"
        );
    }

    #[test]
    fn file_line_format_matches_cpp_and_has_no_color() {
        let line = format_file_line(LogLevel::Error, "a.rs", "f", 1, 99, "msg");
        assert_eq!(line, "a.rs:1 ERROR 99 f msg\n");
        assert!(!line.contains('\x1b'));
    }

    #[test]
    fn format_handles_empty_message() {
        let line = format_file_line(LogLevel::Info, "a.rs", "f", 0, 0, "");
        assert_eq!(line, "a.rs:0 INFO 0 f \n");
    }

    // ---- Thread id ----

    #[test]
    fn tid_is_nonzero_and_stable_within_thread() {
        let tid = current_tid();
        assert!(tid > 0);
        assert_eq!(tid, current_tid());
    }

    #[test]
    fn tid_differs_across_threads() {
        let main_tid = current_tid();
        let other_tid = std::thread::spawn(current_tid).join().unwrap();
        assert_ne!(main_tid, other_tid);
        assert!(other_tid > 0);
    }

    // ---- Logger sinks ----

    fn temp_log_path(name: &str) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "ctp_util_logging_test_{}_{}",
            std::process::id(),
            name
        ));
        path
    }

    #[test]
    fn log_writes_colorless_line_to_file() {
        let path = temp_log_path("log");
        let logger =
            Logger::with_options(LogLevel::Debug.as_i32(), Some(File::create(&path).unwrap()));
        logger.log(
            LogLevel::Error,
            "src/mod.rs",
            "crate::mod::f",
            123,
            format_args!("value = {}", 42),
        );
        // Filtered-out level must not reach the file.
        let filtered = Logger::with_options(LogLevel::Fatal.as_i32(), None);
        assert!(!filtered.should_log(LogLevel::Error));

        let mut contents = String::new();
        File::open(&path)
            .unwrap()
            .read_to_string(&mut contents)
            .unwrap();
        let _ = std::fs::remove_file(&path);
        let expected = format!(
            "src/mod.rs:123 ERROR {} crate::mod::f value = 42\n",
            current_tid()
        );
        assert_eq!(contents, expected);
    }

    #[test]
    fn log_below_threshold_writes_nothing_to_file() {
        let path = temp_log_path("filtered");
        let logger = Logger::with_options(
            LogLevel::Warning.as_i32(),
            Some(File::create(&path).unwrap()),
        );
        logger.log(LogLevel::Debug, "a.rs", "f", 1, format_args!("dropped"));
        logger.log(LogLevel::Info, "a.rs", "f", 2, format_args!("dropped"));
        let mut contents = String::new();
        File::open(&path)
            .unwrap()
            .read_to_string(&mut contents)
            .unwrap();
        let _ = std::fs::remove_file(&path);
        assert!(contents.is_empty());
    }

    #[test]
    fn print_writes_message_plus_newline_to_file() {
        let path = temp_log_path("print");
        let logger = Logger::with_options(
            LogLevel::Fatal.as_i32(), // print ignores the level filter
            Some(File::create(&path).unwrap()),
        );
        logger.print(format_args!("x = {}, y = {}", 1, "two"));
        logger.print(format_args!(""));
        let mut contents = String::new();
        File::open(&path)
            .unwrap()
            .read_to_string(&mut contents)
            .unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(contents, "x = 1, y = two\n\n");
    }

    // ---- Macros ----

    #[test]
    fn function_name_macro_yields_enclosing_fn_path() {
        let name = function_name!();
        assert!(
            name.ends_with("function_name_macro_yields_enclosing_fn_path"),
            "unexpected function name: {name}"
        );
        assert!(!name.ends_with("__ctp_fn_probe"));
    }

    #[test]
    fn hlog_and_helper_macros_compile_and_run() {
        // Smoke test: these route through the global (env-configured) logger
        // to stderr; correctness of the emitted format is covered by the
        // format_* tests above. Fatal is excluded (it exits the process).
        crate::hlog!(LogLevel::Info, "hlog test {} {}", 1, "two");
        crate::hdebug!("debug helper {}", 0);
        crate::hinfo!("info helper");
        crate::hsuccess!("success helper");
        crate::hwarning!("warning helper");
        crate::herror!("error helper");
        crate::hiprint!("hiprint {}", "test");
    }

    #[test]
    fn global_logger_is_a_singleton() {
        let a = Logger::global() as *const Logger;
        let b = Logger::global() as *const Logger;
        assert_eq!(a, b);
    }
}
