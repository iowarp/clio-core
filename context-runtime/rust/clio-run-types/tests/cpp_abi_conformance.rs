// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — context-runtime Rust adaptation (issue #756).
//
//! Conformance: the Rust POD types vs. the C++ headers that define them.
//!
//! TASK_ABI.md §5 says the port "owes a shared conformance test vector set",
//! because the task ABI is *behavioral* as well as structural. This is the
//! first installment: it reads the authoritative C++ headers out of the repo
//! and asserts the Rust side agrees, field for field and bit for bit.
//!
//! # Why this exists
//!
//! Every constant in `clio-run-types` was hand-transcribed from C++. Three
//! transcription bugs survived review and unit tests, because unit tests can
//! only assert that Rust equals *what the porter typed*:
//!
//! 1. `PoolQuery::default()` set `container_id = 0`; the C++ ctor uses
//!    `kInvalidContainerId`. `0` is a valid container, so `HasContainerId()`
//!    would have been true for every default query.
//! 2. `PoolQuery::default()` set `parallelism = 1`; the C++ ctor uses `32`.
//! 3. `TASK_DATA_OWNER`/`TASK_REMOTE` were packed at bits 1/2; the C++ has
//!    them at bits 2/3 (bit 1 is retired-but-reserved). Rust's `REMOTE`
//!    therefore *aliased* C++'s `TASK_DATA_OWNER` — a task crossing the FFI
//!    would have had a buffer it does not own freed under it.
//!
//! A test that compares Rust against the C++ *source of truth* catches that
//! whole class; a test that compares Rust against Rust cannot.
//!
//! # Scope and limits
//!
//! This parses header text, so it verifies **declared** facts: enum
//! discriminant order, macro bit numbers, field order/types, and ctor
//! initializers. It deliberately does NOT verify realized layout
//! (`sizeof`/`offsetof`/padding), which only a C++ compiler can answer —
//! that needs a compiled probe in the devcontainer (yaml-cpp and the rest of
//! the CTP stack are not available on a bare host). The `abi_layouts_are_frozen`
//! unit test pins the Rust side's sizes meanwhile. Closing that gap is
//! tracked as follow-up work in TASK_ABI.md.

use clio_run_types::*;
use std::collections::HashMap;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Header access + a very small C++ "parser" (enough for declarations)
// ---------------------------------------------------------------------------

/// Repo root, derived from this crate's location:
/// `<root>/context-runtime/rust/clio-run-types`.
fn repo_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for _ in 0..3 {
        p.pop();
    }
    p
}

fn read_header(rel: &str) -> String {
    let path = repo_root().join(rel);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "cannot read the authoritative C++ header {}: {e}.\n\
             This test compares the Rust port against the in-repo C++ headers; \
             it must run from a full checkout.",
            path.display()
        )
    });
    // Normalize CRLF: the repo checks out with native line endings, and macro
    // continuations are joined on a literal "\\\n" below.
    text.replace("\r\n", "\n")
}

/// Strip `//`-to-EOL and `/* ... */` comments. Good enough for the
/// declaration blocks we read (none contain string literals with comment
/// markers). Line-continuation backslashes are preserved.
fn strip_comments(src: &str) -> String {
    let b = src.as_bytes();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'/' {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
        } else if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'*' {
            i += 2;
            while i + 1 < b.len() && !(b[i] == b'*' && b[i + 1] == b'/') {
                i += 1;
            }
            i = (i + 2).min(b.len());
            out.push(' '); // a comment separates tokens
        } else {
            out.push(b[i] as char);
            i += 1;
        }
    }
    out
}

/// The text between `open` (searched after `start_marker`) and its matching
/// close brace, brace-depth aware.
fn block_after<'a>(src: &'a str, start_marker: &str) -> &'a str {
    let start = src
        .find(start_marker)
        .unwrap_or_else(|| panic!("marker {start_marker:?} not found in header"));
    let rest = &src[start..];
    let open = rest.find('{').expect("no `{` after marker");
    let bytes = rest.as_bytes();
    let mut depth = 0usize;
    for (i, &c) in bytes.iter().enumerate().skip(open) {
        match c {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &rest[open + 1..i];
                }
            }
            _ => {}
        }
    }
    panic!("unbalanced braces after {start_marker:?}");
}

// ---------------------------------------------------------------------------
// RoutingMode: discriminants are wire values, so ORDER is the contract
// ---------------------------------------------------------------------------

/// Enumerator names of `enum class RoutingMode`, in declaration order.
fn cpp_routing_mode_order() -> Vec<String> {
    let src = strip_comments(&read_header("context-runtime/include/clio_runtime/pool_query.h"));
    let body = block_after(&src, "enum class RoutingMode");
    body.split(',')
        .map(|e| e.trim())
        .filter(|e| !e.is_empty())
        .map(|e| {
            // Tolerate an explicit `= N` if one is ever added.
            assert!(
                !e.contains('='),
                "RoutingMode::{e} gained an explicit discriminant; this test \
                 assumes implicit 0..N numbering and must be updated"
            );
            e.to_string()
        })
        .collect()
}

#[test]
fn routing_mode_matches_cpp_declaration_order() {
    // C++ implicit enum numbering means declaration order IS the wire value.
    // These are serialized inside PoolQuery: reordering breaks in-flight and
    // persisted tasks, so a diff here is a deliberate-decision alarm.
    let cpp = cpp_routing_mode_order();
    let rust = [
        (RoutingMode::Local, "Local"),
        (RoutingMode::DirectId, "DirectId"),
        (RoutingMode::DirectHash, "DirectHash"),
        (RoutingMode::Range, "Range"),
        (RoutingMode::Broadcast, "Broadcast"),
        (RoutingMode::Physical, "Physical"),
        (RoutingMode::Dynamic, "Dynamic"),
        (RoutingMode::ToLocalCpu, "ToLocalCpu"),
        (RoutingMode::Null, "Null"),
        (RoutingMode::ManyToOne, "ManyToOne"),
        (RoutingMode::AllToOne, "AllToOne"),
    ];

    assert_eq!(
        cpp.len(),
        rust.len(),
        "RoutingMode arity drifted: C++ has {:?}, Rust has {} variants",
        cpp,
        rust.len()
    );
    for (i, (variant, name)) in rust.iter().enumerate() {
        assert_eq!(
            &cpp[i], name,
            "RoutingMode slot {i}: C++ declares {:?}, Rust has {name}",
            cpp[i]
        );
        assert_eq!(
            *variant as u32, i as u32,
            "RoutingMode::{name} must have discriminant {i} to match C++"
        );
    }
}

// ---------------------------------------------------------------------------
// TASK_* flags: bit NUMBERS, including the retired-but-reserved gaps
// ---------------------------------------------------------------------------

/// `TASK_* -> bit number`, parsed from the `BIT_OPT(u32, n)` macros.
fn cpp_task_flag_bits() -> HashMap<String, u32> {
    let src = strip_comments(&read_header("context-runtime/include/clio_runtime/types.h"));
    // Join macro line-continuations so multi-line #defines parse as one line.
    let joined = src.replace("\\\n", " ");
    let mut out = HashMap::new();
    for line in joined.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("#define TASK_") else {
            continue;
        };
        let mut it = rest.splitn(2, char::is_whitespace);
        let name = format!("TASK_{}", it.next().unwrap_or_default());
        let Some(body) = it.next() else { continue };
        let Some(args) = body.find("BIT_OPT(").map(|i| &body[i + 8..]) else {
            continue;
        };
        let args = &args[..args.find(')').expect("unterminated BIT_OPT")];
        let bit: u32 = args
            .rsplit(',')
            .next()
            .expect("BIT_OPT needs a bit argument")
            .trim()
            .parse()
            .unwrap_or_else(|e| panic!("{name}: non-literal BIT_OPT bit {args:?}: {e}"));
        out.insert(name, bit);
    }
    assert!(!out.is_empty(), "parsed no TASK_* macros from types.h");
    out
}

#[test]
fn bit_opt_still_means_one_shifted_left() {
    // The flag test below reads bit NUMBERS and shifts them itself, so it is
    // only valid while BIT_OPT is `1 << n`.
    let src = strip_comments(&read_header(
        "context-transport-primitives/include/clio_ctp/types/bitfield.h",
    ));
    let line = src
        .lines()
        .find(|l| l.trim_start().starts_with("#define BIT_OPT"))
        .expect("BIT_OPT macro not found in bitfield.h");
    let normalized: String = line.chars().filter(|c| !c.is_whitespace()).collect();
    assert_eq!(
        normalized, "#defineBIT_OPT(T,n)(((T)1)<<n)",
        "BIT_OPT changed shape; the task-flag conformance test assumes `1 << n`"
    );
}

#[test]
fn task_flags_match_cpp_bit_numbers() {
    let cpp = cpp_task_flag_bits();
    let rust = [
        ("TASK_PERIODIC", task_flags::PERIODIC),
        ("TASK_DATA_OWNER", task_flags::DATA_OWNER),
        ("TASK_REMOTE", task_flags::REMOTE),
        ("TASK_FIRE_AND_FORGET", task_flags::FIRE_AND_FORGET),
        ("TASK_BATCH_AGGREGATE", task_flags::BATCH_AGGREGATE),
        ("TASK_EXTERNAL_CLIENT", task_flags::EXTERNAL_CLIENT),
    ];

    for (name, value) in rust {
        let bit = cpp
            .get(name)
            .unwrap_or_else(|| panic!("{name} is no longer defined in the C++ types.h"));
        assert_eq!(
            value,
            1u32 << bit,
            "{name}: C++ puts it at bit {bit} ({:#x}), Rust has {value:#x}",
            1u32 << bit
        );
    }

    // Every live C++ flag must be ported: a flag that exists only on the C++
    // side is a silent hole the moment a Rust task round-trips through it.
    let ported: Vec<&str> = rust.iter().map(|(n, _)| *n).collect();
    let mut missing: Vec<&String> = cpp.keys().filter(|k| !ported.contains(&k.as_str())).collect();
    missing.sort();
    assert!(
        missing.is_empty(),
        "C++ defines TASK_* flags the Rust port is missing: {missing:?}"
    );
}

// ---------------------------------------------------------------------------
// Struct field order (the layout contract, as far as text can prove it)
// ---------------------------------------------------------------------------

/// `(type, name)` for each `type name_;` member in a declaration block.
fn cpp_fields(body: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for stmt in body.split(';') {
        let s = stmt.trim();
        if s.is_empty() || s.contains('(') || s.contains(')') || s.contains('{') {
            continue; // methods, ctors, macros — not data members
        }
        let toks: Vec<&str> = s.split_whitespace().collect();
        if toks.len() < 2 {
            continue;
        }
        if let Some(name) = toks.last() {
            if !name.ends_with('_') {
                continue; // project convention: data members end with `_`
            }
            out.push((toks[..toks.len() - 1].join(" "), name.to_string()));
        }
    }
    out
}

#[test]
fn pool_query_field_order_matches_cpp() {
    let src = strip_comments(&read_header("context-runtime/include/clio_runtime/pool_query.h"));
    let class_body = block_after(&src, "class PoolQuery");
    let private = class_body
        .split("private:")
        .nth(1)
        .expect("PoolQuery has no private section");
    let cpp = cpp_fields(private);

    let expected = [
        ("RoutingMode", "routing_mode_"),
        ("u32", "hash_value_"),
        ("ContainerId", "container_id_"),
        ("u32", "range_offset_"),
        ("u32", "range_count_"),
        ("u32", "node_id_"),
        ("u32", "ret_node_"),
        ("float", "net_timeout_"),
        ("float", "ttl_"),
        ("u32", "parallelism_"),
        ("u64", "batch_key_"),
        ("u64", "batch_for_ns_"),
    ];
    let actual: Vec<(&str, &str)> = cpp.iter().map(|(t, n)| (t.as_str(), n.as_str())).collect();
    assert_eq!(
        actual,
        expected.to_vec(),
        "PoolQuery's C++ fields drifted from the frozen order mirrored by the \
         Rust #[repr(C)] struct"
    );
}

#[test]
fn task_id_field_order_matches_cpp() {
    let src = strip_comments(&read_header("context-runtime/include/clio_runtime/types.h"));
    let cpp = cpp_fields(block_after(&src, "struct TaskId"));
    let expected = [
        ("u32", "pid_"),
        ("u32", "tid_"),
        ("u32", "major_"),
        ("u32", "replica_id_"),
        ("u32", "unique_"),
        ("u32", "node_id_"),
        ("size_t", "net_key_"),
    ];
    let actual: Vec<(&str, &str)> = cpp.iter().map(|(t, n)| (t.as_str(), n.as_str())).collect();
    assert_eq!(actual, expected.to_vec(), "TaskId's C++ layout drifted");
}

// ---------------------------------------------------------------------------
// Constructor defaults (a value contract, not a layout one)
// ---------------------------------------------------------------------------

/// `field_ -> initializer text` from a member-init list, e.g. the
/// `PoolQuery()` default ctor.
fn cpp_ctor_inits(src: &str, ctor_sig: &str) -> HashMap<String, String> {
    let at = src
        .find(ctor_sig)
        .unwrap_or_else(|| panic!("ctor {ctor_sig:?} not found"));
    let rest = &src[at + ctor_sig.len()..];
    let colon = rest.find(':').expect("ctor has no member-init list");
    let end = rest.find('{').expect("ctor has no body");
    assert!(colon < end, "ctor {ctor_sig:?} has no member-init list");

    let mut out = HashMap::new();
    for part in rest[colon + 1..end].split(',') {
        let p = part.trim();
        let Some(open) = p.find('(') else { continue };
        let Some(close) = p.rfind(')') else { continue };
        let name = p[..open].trim();
        if name.ends_with('_') {
            out.insert(name.to_string(), p[open + 1..close].trim().to_string());
        }
    }
    assert!(!out.is_empty(), "parsed no initializers from {ctor_sig:?}");
    out
}

#[test]
fn pool_query_default_ctor_matches_rust_default() {
    let src = strip_comments(&read_header("context-runtime/include/clio_runtime/pool_query.h"));
    let inits = cpp_ctor_inits(&src, "PoolQuery()");
    let d = PoolQuery::default();

    // The two non-zero defaults — both were wrong in the first port.
    assert_eq!(
        inits.get("container_id_").map(String::as_str),
        Some("kInvalidContainerId"),
        "C++ default container_id_ changed; Rust uses INVALID_CONTAINER_ID"
    );
    assert_eq!(d.container_id, INVALID_CONTAINER_ID);
    assert!(!d.has_container_id());

    assert_eq!(
        inits.get("parallelism_").map(String::as_str),
        Some("32"),
        "C++ default parallelism_ changed; Rust hardcodes 32"
    );
    assert_eq!(d.parallelism, 32);

    // ...and the rest of the init list, so any future addition is caught.
    assert_eq!(
        inits.get("routing_mode_").map(String::as_str),
        Some("RoutingMode::Local")
    );
    assert_eq!(d.routing_mode, RoutingMode::Local);
    for (field, expect, actual) in [
        ("net_timeout_", "-1.0f", d.net_timeout),
        ("ttl_", "-1.0f", d.ttl),
    ] {
        assert_eq!(inits.get(field).map(String::as_str), Some(expect), "{field}");
        assert_eq!(actual, -1.0, "{field}");
    }
    for (field, actual) in [
        ("hash_value_", d.hash_value),
        ("range_offset_", d.range_offset),
        ("range_count_", d.range_count),
        ("node_id_", d.node_id),
        ("ret_node_", d.ret_node),
    ] {
        assert_eq!(inits.get(field).map(String::as_str), Some("0"), "{field}");
        assert_eq!(actual, 0, "{field}");
    }
    for (field, actual) in [
        ("batch_key_", d.batch_key),
        ("batch_for_ns_", d.batch_for_ns),
    ] {
        assert_eq!(inits.get(field).map(String::as_str), Some("0"), "{field}");
        assert_eq!(actual, 0, "{field}");
    }
}

#[test]
fn invalid_container_id_matches_cpp() {
    let src = strip_comments(&read_header("context-runtime/include/clio_runtime/types.h"));
    let line = src
        .lines()
        .find(|l| l.contains("kInvalidContainerId"))
        .expect("kInvalidContainerId not found in types.h");
    let normalized: String = line.chars().filter(|c| !c.is_whitespace()).collect();
    // `static constexpr ContainerId kInvalidContainerId = ContainerId(-1);`
    assert!(
        normalized.contains("kInvalidContainerId=static_cast<ContainerId>(-1)"),
        "kInvalidContainerId is no longer (ContainerId)-1: {line:?}"
    );
    assert_eq!(INVALID_CONTAINER_ID, u32::MAX, "ContainerId is a u32 alias");

    // And ContainerId really is u32 — the cast above is only u32::MAX if so.
    let alias = src
        .lines()
        .find(|l| l.contains("using ContainerId"))
        .expect("ContainerId alias not found");
    let alias: String = alias.chars().filter(|c| !c.is_whitespace()).collect();
    assert_eq!(alias, "usingContainerId=u32;", "ContainerId's width changed");
}

// ---------------------------------------------------------------------------
// The routing predicate (behavioral ABI, per TASK_ABI.md §5)
// ---------------------------------------------------------------------------

#[test]
fn is_collective_mode_matches_cpp_body() {
    // ipc_manager.cc routes on IsCollectiveMode(), so Rust must agree on
    // exactly which modes it covers. The first port had Broadcast|Range|
    // AllToOne, which is neither a superset nor a subset of the C++.
    let src = strip_comments(&read_header("context-runtime/include/clio_runtime/pool_query.h"));
    let body = block_after(&src, "bool IsCollectiveMode()");
    let normalized: String = body.chars().filter(|c| !c.is_whitespace()).collect();
    assert_eq!(
        normalized,
        "returnrouting_mode_==RoutingMode::ManyToOne||routing_mode_==RoutingMode::AllToOne;",
        "IsCollectiveMode's C++ body changed; update is_collective_mode to match"
    );

    for mode in [
        RoutingMode::Local,
        RoutingMode::DirectId,
        RoutingMode::DirectHash,
        RoutingMode::Range,
        RoutingMode::Broadcast,
        RoutingMode::Physical,
        RoutingMode::Dynamic,
        RoutingMode::ToLocalCpu,
        RoutingMode::Null,
        RoutingMode::ManyToOne,
        RoutingMode::AllToOne,
    ] {
        let q = PoolQuery {
            routing_mode: mode,
            ..PoolQuery::default()
        };
        let expected = matches!(mode, RoutingMode::ManyToOne | RoutingMode::AllToOne);
        assert_eq!(
            q.is_collective_mode(),
            expected,
            "is_collective_mode disagrees with the C++ for {mode:?}"
        );
    }
}
