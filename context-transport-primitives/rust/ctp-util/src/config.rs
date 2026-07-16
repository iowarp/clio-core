// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).
//
//! Rust port of `clio_ctp/util/config_parse.h`.
//!
//! # C++ parity map
//!
//! | C++ name                              | Rust name                    |
//! |---------------------------------------|------------------------------|
//! | `ConfigParse::rm_char`                | [`rm_char`]                  |
//! | `ConfigParse::ParseHostNameString`    | [`parse_host_name_string`]   |
//! | `ConfigParse::ParseNumberSuffix`      | [`parse_number_suffix`]      |
//! | `ConfigParse::ParseNumber<double>`    | [`parse_number_f64`]         |
//! | `ConfigParse::ParseSize`              | [`parse_size`]               |
//! | `ConfigParse::ParseBandwidth`         | [`parse_bandwidth`]          |
//! | `ConfigParse::ParseLatency`           | [`parse_latency`]            |
//! | `ConfigParse::ExpandPath`             | [`expand_path`]              |
//! | `ConfigParse::ParseHostfile`          | [`parse_hostfile`]           |
//! | (none — Rust addition)                | [`format_size`]              |
//!
//! The size-suffix table matches `ctp::Unit<u64>` exactly (binary
//! multipliers): `k/K = 2^10`, `m/M = 2^20`, `g/G = 2^30`, `t/T = 2^40`,
//! `p/P = 2^50`; empty suffix = bytes; the literal string `"inf"` parses to
//! `u64::MAX`. Suffix matching looks only at the FIRST non-numeric,
//! non-whitespace, non-`.` character, so `"4KB"`, `"4kb"`, `"4k"`, and
//! `"4 K"` are all `4096`. CTE `capacity_limit` parsing depends on this
//! table — do not change it.
//!
//! # Semantic divergences from the C++ header
//!
//! 1. **Errors instead of process exit.** C++ `ParseSize` calls
//!    `HLOG(kFatal)`/`exit(1)` on an unknown suffix and `ParseLatency` logs
//!    fatal and returns 0. The Rust ports return
//!    `Err(`[`ConfigParseError`]`)` instead and never terminate the process.
//! 2. **Saturating float→int conversion.** Converting the parsed `double`
//!    to `u64` is undefined behavior in C++ for out-of-range values; Rust's
//!    `as` cast saturates (above `u64::MAX` → `u64::MAX`, negative → 0,
//!    NaN → 0). E.g. `parse_size("1000000p")` is `Ok(u64::MAX)` here.
//!    (Negative inputs like `"-1k"` never reach the cast: the leading `-`
//!    starts the suffix, which is an error in both implementations.)
//! 3. **`ParseNumber<T>` is monomorphized.** The header's template is only
//!    ever instantiated with `double` by the ported functions, so Rust
//!    provides [`parse_number_f64`] alone.
//! 4. **`ParseLatency` quirk kept verbatim.** The C++ implementation reuses
//!    the *binary byte* multipliers for time units (`u/U → 2^10`,
//!    `m/M → 2^20`, `s/S → 2^40` nanoseconds — not 10^3/10^6/10^9). This is
//!    ported exactly as written for parity.
//! 5. **Zero-padding of negative range values.** Host-range expansion pads
//!    with Rust's sign-aware `{:0w$}` (`-05`) where C++ `setfill('0')` puts
//!    the fill before the sign (`0-5`). Reachable only for pathological
//!    inputs (negative values inside `[...]` ranges).
//! 6. **`BaseConfig` / YAML helpers not ported.** `BaseConfig`,
//!    `ParseVector`, `ClearParseVector`, and `MsanUnpoisonYamlNode` depend
//!    on yaml-cpp / MSan; YAML config loading arrives with the serde phase
//!    of the migration (`MIGRATION.md`, phase 2) and no new dependencies are
//!    allowed here.
//! 7. **`ExpandPath`'s `${HOME}` special case** is implemented locally
//!    (USERPROFILE preferred over HOME on Windows, HOME elsewhere) rather
//!    than through `ctp::SystemInfo::GetHomeDir`, because `ctp-util` may not
//!    depend on `ctp-introspect`. The lookup order matches `GetHomeDir`.
//! 8. **`ParseHostfile` error logging** goes to stderr via `eprintln!`
//!    rather than `HLOG(kError)`; like C++, it returns an empty list when
//!    the file cannot be opened.

use std::fmt;

/// Error returned when a size/latency string has an unrecognized suffix.
///
/// C++ parity: `ConfigParse::ParseSize` exits the process and
/// `ConfigParse::ParseLatency` logs fatal on this condition (divergence 1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigParseError {
    message: String,
}

impl ConfigParseError {
    fn new(message: String) -> Self {
        Self { message }
    }

    /// Human-readable description of the parse failure.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for ConfigParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ConfigParseError {}

/// Remove every occurrence of `ch` from `s` (C++ `ConfigParse::rm_char`).
pub fn rm_char(s: &mut String, ch: char) {
    s.retain(|c| c != ch);
}

/// Return the unit suffix of a NUMBER text
/// (C++ `ConfigParse::ParseNumberSuffix`).
///
/// Skips ASCII digits, whitespace (space/tab/newline/carriage-return) and
/// `.` from the front; the suffix is everything from the first other
/// character onward (empty if the text is purely numeric).
pub fn parse_number_suffix(num_text: &str) -> &str {
    let bytes = num_text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'0'..=b'9' | b' ' | b'\t' | b'\n' | b'\r' | b'.' => i += 1,
            _ => break,
        }
    }
    &num_text[i..]
}

/// Parse the numeric part of a NUMBER text
/// (C++ `ConfigParse::ParseNumber<double>`).
///
/// - `"inf"` returns `f64::MAX` (C++ `std::numeric_limits<double>::max()`,
///   NOT infinity).
/// - Otherwise only the leading run of `[+-]?[0-9.]` is considered (this is
///   the header's macOS `"1p"` fix: unit letters like `p`/`e`/`x` never
///   reach the float parser).
/// - Mirroring `std::stringstream >> double`, accumulation stops at a
///   second `.`, and a text with no leading digits parses as `0.0`.
pub fn parse_number_f64(num_text: &str) -> f64 {
    if num_text == "inf" {
        return f64::MAX;
    }
    let bytes = num_text.as_bytes();
    let mut i = 0;
    if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
        i += 1;
    }
    while i < bytes.len() {
        match bytes[i] {
            b'0'..=b'9' | b'.' => i += 1,
            _ => break,
        }
    }
    let slice = &num_text[..i];
    // stringstream stops accumulating at the second decimal point.
    let mut end = slice.len();
    let mut seen_dot = false;
    for (j, c) in slice.bytes().enumerate() {
        if c == b'.' {
            if seen_dot {
                end = j;
                break;
            }
            seen_dot = true;
        }
    }
    let buf = &slice[..end];
    if !buf.bytes().any(|c| c.is_ascii_digit()) {
        // strtod on "", "+", "-", "." fails; C++11 stores 0 on failure.
        return 0.0;
    }
    buf.parse::<f64>().unwrap_or(0.0)
}

/// Saturating `f64` → `u64` (Rust `as` semantics; see divergence 2).
fn f64_to_u64_saturating(x: f64) -> u64 {
    x as u64
}

/// Convert SIZE text into a byte count (C++ `ConfigParse::ParseSize`).
///
/// Suffix table (first suffix character, case-insensitive, binary
/// multipliers — matches `ctp::Unit<u64>` exactly):
///
/// | suffix | multiplier |
/// |--------|------------|
/// | (none) | 1          |
/// | `k`/`K`| 2^10       |
/// | `m`/`M`| 2^20       |
/// | `g`/`G`| 2^30       |
/// | `t`/`T`| 2^40       |
/// | `p`/`P`| 2^50       |
///
/// `"inf"` yields `u64::MAX`. Any other suffix is an error (C++ exits the
/// process here; divergence 1).
pub fn parse_size(size_text: &str) -> Result<u64, ConfigParseError> {
    if size_text == "inf" {
        return Ok(u64::MAX);
    }
    let size = parse_number_f64(size_text);
    let suffix = parse_number_suffix(size_text);
    let mult: u64 = match suffix.as_bytes().first() {
        None => 1,
        Some(b'k') | Some(b'K') => 1 << 10,
        Some(b'm') | Some(b'M') => 1 << 20,
        Some(b'g') | Some(b'G') => 1 << 30,
        Some(b't') | Some(b'T') => 1 << 40,
        Some(b'p') | Some(b'P') => 1 << 50,
        Some(_) => {
            return Err(ConfigParseError::new(format!(
                "Could not parse the size: {size_text}"
            )));
        }
    };
    Ok(f64_to_u64_saturating(size * mult as f64))
}

/// Returns bandwidth in bytes/second (C++ `ConfigParse::ParseBandwidth`,
/// which forwards to `ParseSize`).
pub fn parse_bandwidth(size_text: &str) -> Result<u64, ConfigParseError> {
    parse_size(size_text)
}

/// Returns latency in nanoseconds (C++ `ConfigParse::ParseLatency`).
///
/// Ported verbatim, including the C++ quirk of reusing binary byte
/// multipliers for time units (divergence 4):
///
/// | suffix          | multiplier (ns) |
/// |-----------------|-----------------|
/// | (none), `n`/`N` | 1               |
/// | `u`/`U`         | 2^10            |
/// | `m`/`M`         | 2^20            |
/// | `s`/`S`         | 2^40            |
pub fn parse_latency(latency_text: &str) -> Result<u64, ConfigParseError> {
    let size = parse_number_f64(latency_text);
    let suffix = parse_number_suffix(latency_text);
    let mult: u64 = match suffix.as_bytes().first() {
        None | Some(b'n') | Some(b'N') => 1,
        Some(b'u') | Some(b'U') => 1 << 10,
        Some(b'm') | Some(b'M') => 1 << 20,
        Some(b's') | Some(b'S') => 1 << 40,
        Some(_) => {
            return Err(ConfigParseError::new(format!(
                "Could not parse the latency: {latency_text}"
            )));
        }
    };
    Ok(f64_to_u64_saturating(size * mult as f64))
}

/// Format a byte count back into a SIZE string that [`parse_size`] round-trips
/// (Rust addition; no C++ counterpart — see the parity map).
///
/// Picks the largest binary unit that divides `bytes` evenly (`4096` →
/// `"4KB"`); values that are not an exact multiple of any unit are emitted
/// as bare bytes (`1536` → `"1536"`); `u64::MAX` → `"inf"`. The invariant
/// `parse_size(&format_size(n)) == Ok(n)` holds for every `n`.
pub fn format_size(bytes: u64) -> String {
    if bytes == u64::MAX {
        return "inf".to_string();
    }
    const UNITS: [(u64, &str); 5] = [
        (1 << 50, "PB"),
        (1 << 40, "TB"),
        (1 << 30, "GB"),
        (1 << 20, "MB"),
        (1 << 10, "KB"),
    ];
    for (mult, suffix) in UNITS {
        if bytes >= mult && bytes.is_multiple_of(mult) {
            return format!("{}{}", bytes / mult, suffix);
        }
    }
    bytes.to_string()
}

/// Leading-integer parse mimicking `std::stringstream >> int` (C++11):
/// optional sign then digits; no digits → 0; overflow saturates to
/// `i32::MAX` / `i32::MIN`.
fn parse_leading_i32(s: &str) -> i32 {
    let bytes = s.as_bytes();
    let mut i = 0;
    let mut neg = false;
    if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
        neg = bytes[i] == b'-';
        i += 1;
    }
    let mut any = false;
    let mut acc: i64 = 0;
    let mut saturated = false;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        any = true;
        if !saturated {
            acc = acc * 10 + i64::from(bytes[i] - b'0');
            if acc > i64::from(i32::MAX) + 1 {
                saturated = true;
            }
        }
        i += 1;
    }
    if !any {
        return 0;
    }
    if neg {
        if saturated || -acc < i64::from(i32::MIN) {
            i32::MIN
        } else {
            (-acc) as i32
        }
    } else if saturated || acc > i64::from(i32::MAX) {
        i32::MAX
    } else {
        acc as i32
    }
}

/// Parse a hostfile string, appending expanded host names to `list`
/// (C++ `ConfigParse::ParseHostNameString`).
///
/// `[]` encloses a comma-separated list of ranges to expand; `;` separates
/// independent host name patterns. All whitespace is stripped first. A
/// range endpoint pair with equal digit widths (e.g. `00-09`) zero-pads the
/// expansion to that width.
///
/// Example: `"hello[00-09,10]-40g;hello2[11-13]-40g"` expands to
/// `hello00-40g` … `hello09-40g`, `hello10-40g`, `hello211-40g` …
/// `hello213-40g`.
///
/// C++ parity quirks preserved: an empty `;`-segment appends an empty
/// string (so `"a;"` yields `["a", ""]`), and a reversed range (`5-3`)
/// expands to nothing.
pub fn parse_host_name_string(hostname_set_str: &str, list: &mut Vec<String>) {
    // Remove all whitespace characters from the host name string.
    let mut cleaned = hostname_set_str.to_string();
    for ch in [' ', '\n', '\r', '\t'] {
        rm_char(&mut cleaned, ch);
    }
    if cleaned.is_empty() {
        return;
    }
    // Expand hostnames.
    for hostname in cleaned.split(';') {
        // Divide the hostname string into prefix, ranges, and suffix.
        let (prefix, ranges_str, suffix) = match (hostname.find('['), hostname.rfind(']')) {
            (Some(lb), Some(rb)) => {
                let prefix = &hostname[..lb];
                // C++ substr(lb + 1, rb - lb - 1): when rb < lb the
                // count underflows and clamps to "rest of string".
                let ranges_str = if rb > lb {
                    &hostname[lb + 1..rb]
                } else {
                    &hostname[lb + 1..]
                };
                let suffix = &hostname[rb + 1..];
                (prefix, ranges_str, suffix)
            }
            _ => {
                list.push(hostname.to_string());
                continue;
            }
        };

        // Parse the range list into tuples of (min, max, num_width).
        let mut ranges: Vec<(i32, i32, usize)> = Vec::new();
        for range_str in ranges_str.split(',') {
            // (Spaces were already removed globally above.)
            if let Some(dash) = range_str.find('-') {
                let min_str = &range_str[..dash];
                let max_str = &range_str[dash + 1..];
                let min = parse_leading_i32(min_str);
                let max = parse_leading_i32(max_str);
                // Zero-pad only when the endpoint widths agree.
                let num_width = if min_str.len() == max_str.len() {
                    min_str.len()
                } else {
                    0
                };
                ranges.push((min, max, num_width));
            } else if !range_str.is_empty() {
                let val = parse_leading_i32(range_str);
                ranges.push((val, val, range_str.len()));
            }
        }

        // Expand the host names by each range.
        for (min, max, num_width) in ranges {
            for i in min..=max {
                list.push(format!("{prefix}{i:0num_width$}{suffix}"));
            }
        }
    }
}

/// Home directory lookup matching `ctp::SystemInfo::GetHomeDir`
/// (divergence 7): USERPROFILE preferred over HOME on Windows, HOME on
/// POSIX; empty string when unset.
fn get_home_dir() -> String {
    #[cfg(windows)]
    {
        if let Ok(val) = std::env::var("USERPROFILE") {
            if !val.is_empty() {
                return val;
            }
        }
        std::env::var("HOME").unwrap_or_default()
    }
    #[cfg(not(windows))]
    {
        std::env::var("HOME").unwrap_or_default()
    }
}

/// Expand all `${VAR}` environment references in a path string
/// (C++ `ConfigParse::ExpandPath`).
///
/// `${HOME}` is special-cased through [`get_home_dir`]-equivalent logic so
/// POSIX-style configs work on Windows (USERPROFILE fallback). Unset
/// variables expand to the empty string; an unterminated `${` is left
/// as-is. Substituted text is not rescanned (no recursive expansion).
pub fn expand_path(path: &str) -> String {
    let mut path = path.to_string();
    let mut pos = 0;
    while let Some(start) = path[pos..].find("${").map(|off| off + pos) {
        let Some(end) = path[start + 2..].find('}').map(|off| off + start + 2) else {
            break;
        };
        let env_name = &path[start + 2..end];
        let env_val = if env_name == "HOME" {
            get_home_dir()
        } else {
            std::env::var(env_name).unwrap_or_default()
        };
        path.replace_range(start..=end, &env_val);
        pos = start + env_val.len();
    }
    path
}

/// Parse a hostfile: every line is fed through [`parse_host_name_string`]
/// (C++ `ConfigParse::ParseHostfile`). An unreadable file logs to stderr
/// and returns an empty list (divergence 8).
pub fn parse_hostfile(path: &str) -> Vec<String> {
    let mut hosts = Vec::new();
    match std::fs::read_to_string(path) {
        Ok(contents) => {
            for line in contents.lines() {
                parse_host_name_string(line, &mut hosts);
            }
        }
        Err(_) => {
            eprintln!("Could not open the hostfile: {path}");
        }
    }
    hosts
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------- rm_char ----------------

    #[test]
    fn rm_char_removes_all_occurrences() {
        let mut s = String::from("a b c b");
        rm_char(&mut s, 'b');
        assert_eq!(s, "a  c ");
        rm_char(&mut s, ' ');
        assert_eq!(s, "ac");
        let mut empty = String::new();
        rm_char(&mut empty, 'x');
        assert_eq!(empty, "");
    }

    // ---------------- parse_number_suffix ----------------

    #[test]
    fn suffix_basic() {
        assert_eq!(parse_number_suffix("4KB"), "KB");
        assert_eq!(parse_number_suffix("1.5GB"), "GB");
        assert_eq!(parse_number_suffix("0g"), "g");
        assert_eq!(parse_number_suffix("100"), "");
        assert_eq!(parse_number_suffix(""), "");
        assert_eq!(parse_number_suffix("abc"), "abc");
    }

    #[test]
    fn suffix_skips_whitespace_and_dots() {
        // Whitespace between number and unit is skipped.
        assert_eq!(parse_number_suffix("10 MB"), "MB");
        assert_eq!(parse_number_suffix(" 10\t.5\r\nk"), "k");
        // Sign characters are NOT skipped: suffix starts at the sign.
        assert_eq!(parse_number_suffix("-5k"), "-5k");
    }

    // ---------------- parse_number_f64 ----------------

    #[test]
    fn number_basic() {
        assert_eq!(parse_number_f64("4"), 4.0);
        assert_eq!(parse_number_f64("1.5"), 1.5);
        assert_eq!(parse_number_f64("+3"), 3.0);
        assert_eq!(parse_number_f64("-2.5"), -2.5);
        assert_eq!(parse_number_f64(".5"), 0.5);
        assert_eq!(parse_number_f64("1."), 1.0);
    }

    #[test]
    fn number_inf_is_f64_max_not_infinity() {
        assert_eq!(parse_number_f64("inf"), f64::MAX);
        assert!(parse_number_f64("inf").is_finite());
    }

    #[test]
    fn number_failures_yield_zero() {
        // C++11 stringstream stores 0 on extraction failure.
        assert_eq!(parse_number_f64(""), 0.0);
        assert_eq!(parse_number_f64("abc"), 0.0);
        assert_eq!(parse_number_f64("."), 0.0);
        assert_eq!(parse_number_f64("+"), 0.0);
        assert_eq!(parse_number_f64("+-3"), 0.0);
        // Leading whitespace breaks the numeric run (C++ parity).
        assert_eq!(parse_number_f64(" 10"), 0.0);
    }

    #[test]
    fn number_stops_at_unit_letters() {
        // The macOS "1p" regression the C++ header specifically guards.
        assert_eq!(parse_number_f64("1p"), 1.0);
        assert_eq!(parse_number_f64("1e5"), 1.0);
        assert_eq!(parse_number_f64("2x8"), 2.0);
    }

    #[test]
    fn number_second_dot_terminates() {
        // stringstream >> double on "1.2.3" extracts 1.2.
        assert_eq!(parse_number_f64("1.2.3"), 1.2);
    }

    // ---------------- parse_size ----------------

    #[test]
    fn size_bare_numbers_are_bytes() {
        assert_eq!(parse_size("0"), Ok(0));
        assert_eq!(parse_size("100"), Ok(100));
        assert_eq!(parse_size("1.5"), Ok(1)); // (u64)1.5 truncates
    }

    #[test]
    fn size_suffix_table_binary_multipliers() {
        assert_eq!(parse_size("1k"), Ok(1 << 10));
        assert_eq!(parse_size("1K"), Ok(1 << 10));
        assert_eq!(parse_size("1m"), Ok(1 << 20));
        assert_eq!(parse_size("1M"), Ok(1 << 20));
        assert_eq!(parse_size("1g"), Ok(1 << 30));
        assert_eq!(parse_size("1G"), Ok(1 << 30));
        assert_eq!(parse_size("1t"), Ok(1 << 40));
        assert_eq!(parse_size("1T"), Ok(1 << 40));
        assert_eq!(parse_size("1p"), Ok(1 << 50));
        assert_eq!(parse_size("1P"), Ok(1 << 50));
    }

    #[test]
    fn size_case_insensitive_and_multichar_suffix() {
        // Only the FIRST suffix character matters.
        assert_eq!(parse_size("4KB"), Ok(4096));
        assert_eq!(parse_size("4kb"), Ok(4096));
        assert_eq!(parse_size("4KiB"), Ok(4096));
        assert_eq!(parse_size("4Kilobytes"), Ok(4096));
        assert_eq!(parse_size("10 MB"), Ok(10 << 20));
    }

    #[test]
    fn size_fractional_and_zero() {
        assert_eq!(parse_size("1.5GB"), Ok(1_610_612_736));
        assert_eq!(parse_size("0g"), Ok(0));
        assert_eq!(parse_size("0.5k"), Ok(512));
        assert_eq!(parse_size("2.5m"), Ok(2_621_440));
    }

    #[test]
    fn size_inf() {
        assert_eq!(parse_size("inf"), Ok(u64::MAX));
    }

    #[test]
    fn size_unknown_suffix_is_error() {
        assert!(parse_size("4XB").is_err());
        assert!(parse_size("4 quarts").is_err());
        // "infinite" is not "inf": suffix starts at 'i'.
        assert!(parse_size("infinite").is_err());
        let err = parse_size("4XB").unwrap_err();
        assert!(err.message().contains("4XB"));
        assert_eq!(format!("{err}"), "Could not parse the size: 4XB");
    }

    #[test]
    fn size_empty_and_garbage_parse_to_zero_bytes() {
        // "" and "no digits" both parse the number as 0; "" has an empty
        // suffix so it is 0 bytes (C++ parity).
        assert_eq!(parse_size(""), Ok(0));
        assert_eq!(parse_size("   "), Ok(0));
        assert_eq!(parse_size("...."), Ok(0));
    }

    #[test]
    fn size_saturation() {
        // 10^6 PB overflows u64 -> saturates (C++ UB; divergence 2).
        assert_eq!(parse_size("1000000p"), Ok(u64::MAX));
        // Negative inputs never reach the cast: the '-' is not part of the
        // suffix-skip set, so the suffix starts at '-' and errors (C++
        // would HLOG(kFatal) + exit here).
        assert!(parse_size("-1").is_err());
        assert!(parse_size("-1k").is_err());
    }

    #[test]
    fn size_boundary_values() {
        // Largest exactly-representable interesting values.
        assert_eq!(parse_size("16383p"), Ok(16383u64 << 50));
        assert_eq!(parse_size("1023k"), Ok(1023 * 1024));
    }

    // ---------------- parse_bandwidth ----------------

    #[test]
    fn bandwidth_forwards_to_size() {
        assert_eq!(parse_bandwidth("100m"), parse_size("100m"));
        assert_eq!(parse_bandwidth("inf"), Ok(u64::MAX));
        assert!(parse_bandwidth("5z").is_err());
    }

    // ---------------- parse_latency ----------------

    #[test]
    fn latency_suffix_table_cpp_quirk() {
        // Verbatim C++ table (binary byte multipliers; divergence 4).
        assert_eq!(parse_latency("100"), Ok(100));
        assert_eq!(parse_latency("100n"), Ok(100));
        assert_eq!(parse_latency("100ns"), Ok(100));
        assert_eq!(parse_latency("100N"), Ok(100));
        assert_eq!(parse_latency("5u"), Ok(5 << 10));
        assert_eq!(parse_latency("5us"), Ok(5 << 10));
        assert_eq!(parse_latency("5U"), Ok(5 << 10));
        assert_eq!(parse_latency("2m"), Ok(2 << 20));
        assert_eq!(parse_latency("2ms"), Ok(2 << 20));
        assert_eq!(parse_latency("2M"), Ok(2 << 20));
        assert_eq!(parse_latency("1s"), Ok(1 << 40));
        assert_eq!(parse_latency("1S"), Ok(1 << 40));
    }

    #[test]
    fn latency_errors() {
        assert!(parse_latency("5x").is_err());
        // C++ ParseLatency has no "inf" fast path: 'i' is unknown.
        assert!(parse_latency("inf").is_err());
        let err = parse_latency("5x").unwrap_err();
        assert_eq!(format!("{err}"), "Could not parse the latency: 5x");
    }

    // ---------------- format_size ----------------

    #[test]
    fn format_size_exact_units() {
        assert_eq!(format_size(0), "0");
        assert_eq!(format_size(100), "100");
        assert_eq!(format_size(1 << 10), "1KB");
        assert_eq!(format_size(4096), "4KB");
        assert_eq!(format_size(10 << 20), "10MB");
        assert_eq!(format_size(1 << 30), "1GB");
        assert_eq!(format_size(3 << 40), "3TB");
        assert_eq!(format_size(2 << 50), "2PB");
        assert_eq!(format_size(u64::MAX), "inf");
        // Not an exact multiple of 1KB -> bare bytes.
        assert_eq!(format_size(1536), "1536");
    }

    #[test]
    fn format_size_round_trips_through_parse_size() {
        for n in [
            0u64,
            1,
            1023,
            1024,
            1536,
            4096,
            10 << 20,
            (1 << 30) + 1,
            3 << 40,
            2 << 50,
            u64::MAX,
        ] {
            assert_eq!(parse_size(&format_size(n)), Ok(n), "n = {n}");
        }
    }

    // ---------------- parse_host_name_string ----------------

    fn expand(s: &str) -> Vec<String> {
        let mut list = Vec::new();
        parse_host_name_string(s, &mut list);
        list
    }

    #[test]
    fn hosts_header_example() {
        // The exact example documented in the C++ header.
        let hosts = expand("hello[00-09,10]-40g;hello2[11-13]-40g");
        let mut expected: Vec<String> = (0..=9).map(|i| format!("hello{i:02}-40g")).collect();
        expected.push("hello10-40g".to_string());
        for i in 11..=13 {
            expected.push(format!("hello2{i}-40g"));
        }
        assert_eq!(hosts, expected);
    }

    #[test]
    fn hosts_plain_and_multiple() {
        assert_eq!(expand("node1"), vec!["node1"]);
        assert_eq!(expand("a;b"), vec!["a", "b"]);
        assert!(expand("").is_empty());
        assert!(expand("  \t\r\n").is_empty());
    }

    #[test]
    fn hosts_trailing_semicolon_appends_empty_cpp_parity() {
        // C++ getline loop pushes an empty hostname for "a;".
        assert_eq!(expand("a;"), vec!["a", ""]);
        assert_eq!(expand("a;;b"), vec!["a", "", "b"]);
    }

    #[test]
    fn hosts_zero_padding_rules() {
        // Equal endpoint widths -> pad to that width.
        assert_eq!(expand("h[08-10]"), vec!["h08", "h09", "h10"]);
        // Unequal widths -> no padding.
        assert_eq!(expand("h[8-10]"), vec!["h8", "h9", "h10"]);
        // Single value: width is the token length.
        assert_eq!(expand("h[05]"), vec!["h05"]);
        assert_eq!(expand("h[5]"), vec!["h5"]);
        // Number wider than the pad width prints in full.
        assert_eq!(expand("h[100-999]x")[0], "h100x");
    }

    #[test]
    fn hosts_whitespace_stripped_everywhere() {
        assert_eq!(expand("hello [1-2] x"), vec!["hello1x", "hello2x"]);
        assert_eq!(expand("h[1 - 2]"), vec!["h1", "h2"]);
        assert_eq!(expand(" a ; b "), vec!["a", "b"]);
    }

    #[test]
    fn hosts_range_edge_cases() {
        // Reversed range expands to nothing.
        assert!(expand("h[5-3]").is_empty());
        // Empty brackets expand to nothing.
        assert!(expand("h[]").is_empty());
        // Empty range entries are skipped; valid ones still expand.
        assert_eq!(expand("h[,2,]"), vec!["h2"]);
        // Single-element range n-n.
        assert_eq!(expand("h[7-7]"), vec!["h7"]);
        // Prefix/suffix around the brackets.
        assert_eq!(expand("[1-2]tail"), vec!["1tail", "2tail"]);
        assert_eq!(expand("head[1-2]"), vec!["head1", "head2"]);
    }

    #[test]
    fn hosts_lone_bracket_is_literal() {
        // Only ONE bracket present -> hostname is taken literally.
        assert_eq!(expand("h[1-3"), vec!["h[1-3"]);
        assert_eq!(expand("h1-3]"), vec!["h1-3]"]);
    }

    // ---------------- parse_leading_i32 ----------------

    #[test]
    fn leading_i32_semantics() {
        assert_eq!(parse_leading_i32("12"), 12);
        assert_eq!(parse_leading_i32("012"), 12);
        assert_eq!(parse_leading_i32("-3"), -3);
        assert_eq!(parse_leading_i32("+3"), 3);
        assert_eq!(parse_leading_i32("12ab"), 12);
        assert_eq!(parse_leading_i32("ab12"), 0);
        assert_eq!(parse_leading_i32(""), 0);
        // Overflow saturates like C++11 num_get.
        assert_eq!(parse_leading_i32("99999999999"), i32::MAX);
        assert_eq!(parse_leading_i32("-99999999999"), i32::MIN);
        assert_eq!(parse_leading_i32("2147483647"), i32::MAX);
        assert_eq!(parse_leading_i32("2147483648"), i32::MAX);
        assert_eq!(parse_leading_i32("-2147483648"), i32::MIN);
    }

    // ---------------- expand_path ----------------

    #[test]
    fn expand_path_home() {
        let expanded = expand_path("${HOME}/config.yaml");
        // On this machine HOME/USERPROFILE resolves to something non-empty.
        assert!(!expanded.contains("${"));
        assert!(expanded.ends_with("/config.yaml"));
        assert_ne!(expanded, "/config.yaml");
        assert_eq!(expanded, format!("{}/config.yaml", get_home_dir()));
    }

    #[test]
    fn expand_path_env_var_and_missing() {
        // Edition 2021: set_var is safe. Unique name avoids races with
        // other tests.
        std::env::set_var("CTP_RS_CONFIG_TEST_VAR_1", "val1");
        assert_eq!(expand_path("a/${CTP_RS_CONFIG_TEST_VAR_1}/b"), "a/val1/b");
        // Unset variables expand to the empty string.
        assert_eq!(
            expand_path("a/${CTP_RS_CONFIG_DEFINITELY_UNSET_XYZ}/b"),
            "a//b"
        );
    }

    #[test]
    fn expand_path_multiple_and_unterminated() {
        std::env::set_var("CTP_RS_CONFIG_TEST_VAR_A", "A");
        std::env::set_var("CTP_RS_CONFIG_TEST_VAR_B", "B");
        assert_eq!(
            expand_path("${CTP_RS_CONFIG_TEST_VAR_A}/${CTP_RS_CONFIG_TEST_VAR_B}"),
            "A/B"
        );
        // Unterminated ${ is left as-is.
        assert_eq!(expand_path("x/${NOPE"), "x/${NOPE");
        // Plain paths pass through untouched.
        assert_eq!(expand_path("/plain/path"), "/plain/path");
        assert_eq!(expand_path(""), "");
    }

    #[test]
    fn expand_path_no_recursive_expansion() {
        // A value containing "${...}" is not rescanned (pos advances past
        // the substituted text, matching the C++ pos += env_val.size()).
        std::env::set_var(
            "CTP_RS_CONFIG_TEST_VAR_REC",
            "${CTP_RS_CONFIG_TEST_VAR_REC}",
        );
        assert_eq!(
            expand_path("${CTP_RS_CONFIG_TEST_VAR_REC}"),
            "${CTP_RS_CONFIG_TEST_VAR_REC}"
        );
    }

    // ---------------- parse_hostfile ----------------

    #[test]
    fn hostfile_round_trip() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "ctp_rs_config_hostfile_test_{}.txt",
            std::process::id()
        ));
        std::fs::write(&path, "node[01-03]\n\nlogin1;login2\n").unwrap();
        let hosts = parse_hostfile(path.to_str().unwrap());
        std::fs::remove_file(&path).ok();
        assert_eq!(
            hosts,
            vec!["node01", "node02", "node03", "login1", "login2"]
        );
    }

    #[test]
    fn hostfile_missing_returns_empty() {
        let hosts = parse_hostfile("Z:/definitely/not/a/real/hostfile.txt");
        assert!(hosts.is_empty());
    }
}
