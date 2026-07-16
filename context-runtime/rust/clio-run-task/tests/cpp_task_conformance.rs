// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — context-runtime Rust adaptation (issue #756).
//
//! Conformance: `TaskBase`/`RunContext` vs. the C++ `task.h` they mirror.
//!
//! The port no longer shares an ABI with C++ (TASK_ABI.md §0 — everything is
//! Rust, bindings come later), so unlike `clio-run-types` this is **not** a
//! layout contract. It is a *structural* one, and it matters for as long as
//! both implementations exist: the C++ `Task` is the specification of what a
//! task is made of, and a member added there that never arrives here is a
//! feature this runtime silently lacks.
//!
//! So these tests fail when `task.h` grows, loses, or renames state. That is
//! a prompt to port the change deliberately, not a bug in the C++.
//!
//! What is checked: the `Task` data members (which are exhaustively marked
//! with the IN/OUT/TEMP annotation macros, so they parse precisely), the
//! `RunCtxFlag` values, and `TaskGroup`'s null sentinel.

use clio_run_task::{run_ctx_flags, TaskGroup};
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Header access + minimal parsing (see clio-run-types' conformance test)
// ---------------------------------------------------------------------------

fn read_header(rel: &str) -> String {
    let mut root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for _ in 0..3 {
        root.pop();
    }
    let path = root.join(rel);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()))
        .replace("\r\n", "\n")
}

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
            out.push(' ');
        } else {
            out.push(b[i] as char);
            i += 1;
        }
    }
    out
}

fn block_after<'a>(src: &'a str, marker: &str) -> &'a str {
    let start = src
        .find(marker)
        .unwrap_or_else(|| panic!("marker {marker:?} not found"));
    let rest = &src[start..];
    let open = rest.find('{').expect("no `{` after marker");
    let mut depth = 0usize;
    for (i, &c) in rest.as_bytes().iter().enumerate().skip(open) {
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
    panic!("unbalanced braces after {marker:?}");
}

fn task_h() -> String {
    strip_comments(&read_header("context-runtime/include/clio_runtime/task.h"))
}

// ---------------------------------------------------------------------------
// Task data members
// ---------------------------------------------------------------------------

/// `(annotation, name)` for each data member of `class Task`.
///
/// Data members are exactly the statements opening with an IN/OUT/TEMP/INOUT
/// macro, which is what makes this parseable at all: the class body is ~1900
/// lines of interleaved inline methods, and a naive "type name_;" scan trips
/// over assignments in their bodies.
fn cpp_task_members() -> Vec<(String, String)> {
    let src = task_h();
    let body = block_after(&src, "class Task {");
    let mut out = Vec::new();
    for stmt in body.split(';') {
        let toks: Vec<&str> = stmt.split_whitespace().collect();
        let [annotation, rest @ ..] = toks.as_slice() else {
            continue;
        };
        if !matches!(*annotation, "IN" | "OUT" | "INOUT" | "TEMP") {
            continue;
        }
        let Some(name) = rest.last() else { continue };
        // A declaration ends in the member name; anything with call syntax is
        // a method or an initializer, not a data member.
        if !name.ends_with('_') || stmt.contains('(') {
            continue;
        }
        out.push((annotation.to_string(), name.to_string()));
    }
    assert!(!out.is_empty(), "parsed no annotated members from class Task");
    out
}

#[test]
fn task_base_covers_every_cpp_task_member() {
    // Each C++ member and where it lives in the Rust port. A mismatch means
    // task.h changed: port the delta, then update this table.
    let expected = [
        // IN — client → runtime, serialized into the task archive.
        ("IN", "pool_id_"),      // TaskBase::pool_id
        ("IN", "task_id_"),      // TaskBase::task_id
        ("IN", "pool_query_"),   // TaskBase::pool_query
        ("IN", "method_"),       // TaskBase::method
        ("IN", "task_flags_"),   // TaskBase::task_flags
        ("IN", "period_ns_"),    // TaskBase::period_ns
        ("IN", "task_group_"),   // TaskBase::task_group
        // OUT — runtime → client.
        ("OUT", "return_code_"), // TaskBase::return_code
        ("OUT", "completer_"),   // TaskBase::completer
        // TEMP — process-local; never serialized, never copied.
        ("TEMP", "fut_"),         // TaskBase::fut
        ("TEMP", "is_new_data_"), // TaskBase::is_new_data
        ("TEMP", "run_ctx_"),     // TaskBase::run_ctx (private, Option<Box<_>>)
    ];

    let cpp = cpp_task_members();
    let actual: Vec<(&str, &str)> = cpp.iter().map(|(a, n)| (a.as_str(), n.as_str())).collect();
    assert_eq!(
        actual,
        expected.to_vec(),
        "class Task's members drifted from the Rust TaskBase"
    );
}

// ---------------------------------------------------------------------------
// RunContext flags
// ---------------------------------------------------------------------------

/// `RCTX_* -> bit number`, from `enum RunCtxFlag : u32 { RCTX_X = 1u << n }`.
fn cpp_run_ctx_flags() -> Vec<(String, u32)> {
    let src = task_h();
    let body = block_after(&src, "enum RunCtxFlag");
    let mut out = Vec::new();
    for entry in body.split(',') {
        let e = entry.trim();
        if e.is_empty() {
            continue;
        }
        let (name, value) = e.split_once('=').unwrap_or_else(|| {
            panic!("RunCtxFlag entry {e:?} has no explicit value; this test assumes `= 1u << n`")
        });
        let shift = value.split("<<").nth(1).unwrap_or_else(|| {
            panic!("RunCtxFlag {name} is not a `1u << n` literal: {value:?}")
        });
        out.push((
            name.trim().to_string(),
            shift.trim().parse().expect("non-literal shift"),
        ));
    }
    assert!(!out.is_empty(), "parsed no RunCtxFlag entries");
    out
}

#[test]
fn run_ctx_flags_match_cpp() {
    // These are per-execution and never serialized, so unlike TASK_* they are
    // free to renumber — but only in lockstep with the C++ while both exist.
    let expected = [
        ("RCTX_YIELDED", run_ctx_flags::YIELDED),
        ("RCTX_DID_WORK", run_ctx_flags::DID_WORK),
        ("RCTX_ROUTED", run_ctx_flags::ROUTED),
        ("RCTX_STARTED", run_ctx_flags::STARTED),
    ];
    let cpp = cpp_run_ctx_flags();
    assert_eq!(
        cpp.len(),
        expected.len(),
        "RunCtxFlag arity drifted: C++ has {cpp:?}"
    );
    for (i, (name, rust_value)) in expected.iter().enumerate() {
        let (cpp_name, cpp_bit) = &cpp[i];
        assert_eq!(cpp_name, name, "RunCtxFlag slot {i}");
        assert_eq!(
            *rust_value,
            1u32 << cpp_bit,
            "{name}: C++ has bit {cpp_bit}, Rust has {rust_value:#x}"
        );
    }
}

// ---------------------------------------------------------------------------
// TaskGroup
// ---------------------------------------------------------------------------

#[test]
fn task_group_null_sentinel_matches_cpp() {
    let src = task_h();
    let body = block_after(&src, "struct TaskGroup");
    let normalized: String = body.chars().filter(|c| !c.is_whitespace()).collect();

    // `int64_t id_{-1};` — the null group, and the reason 0 is a real group.
    assert!(
        normalized.contains("int64_tid_{-1}"),
        "TaskGroup's null sentinel is no longer -1: {body:?}"
    );
    assert!(
        normalized.contains("boolIsNull()const{returnid_==-1;}"),
        "TaskGroup::IsNull no longer compares against -1"
    );

    assert_eq!(TaskGroup::null().id(), -1);
    assert!(TaskGroup::null().is_null());
    assert!(TaskGroup::default().is_null());
    assert!(!TaskGroup::new(0).is_null());
}
