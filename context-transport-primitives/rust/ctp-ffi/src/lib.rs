// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).
//
// C ABI over the ctp-rs crates. Conventions (see rust/MIGRATION.md):
//  - every symbol is prefixed `ctp_rs_`
//  - strings cross as NUL-terminated UTF-8; Rust copies inputs immediately
//    and never retains caller pointers; returned strings are Rust-allocated
//    and must be released with ctp_rs_string_free
//  - no panic crosses the boundary: bodies are wrapped in catch_unwind and
//    report failure through return values
//
// The authoritative C header is include/ctp_rs.h (hand-maintained; reviewed
// like any public API).

#![deny(unsafe_op_in_unsafe_fn)]

use std::ffi::{c_char, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Copy a caller string; empty on NULL / non-UTF8 (lossy conversion).
///
/// # Safety
/// `s` must be NULL or a valid NUL-terminated string.
unsafe fn cstr_to_string(s: *const c_char) -> String {
    if s.is_null() {
        return String::new();
    }
    // SAFETY: caller contract — valid NUL-terminated pointer.
    unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned()
}

/// Return a Rust-allocated C string (caller frees with ctp_rs_string_free).
/// Interior NULs are replaced so the conversion cannot fail.
fn string_to_cstr(s: String) -> *mut c_char {
    let cleaned = s.replace('\0', " ");
    CString::new(cleaned)
        .expect("NULs removed above")
        .into_raw()
}

/// Run `f`, converting a panic into `default` — panics must not unwind
/// across the C boundary.
fn no_panic<T>(default: T, f: impl FnOnce() -> T) -> T {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or(default)
}

// ---------------------------------------------------------------------------
// Version / string memory
// ---------------------------------------------------------------------------

/// ABI version of this library (bump on breaking changes).
#[no_mangle]
pub extern "C" fn ctp_rs_abi_version() -> u32 {
    1
}

/// Free a string returned by any ctp_rs_* function. NULL is a no-op.
///
/// # Safety
/// `s` must be NULL or a pointer previously returned by this library.
#[no_mangle]
pub unsafe extern "C" fn ctp_rs_string_free(s: *mut c_char) {
    if !s.is_null() {
        // SAFETY: provenance guaranteed by the caller contract.
        drop(unsafe { CString::from_raw(s) });
    }
}

// ---------------------------------------------------------------------------
// Environment
// ---------------------------------------------------------------------------

/// # Safety
/// `name` must be NULL or a valid NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn ctp_rs_getenv(name: *const c_char, max_size: usize) -> *mut c_char {
    let name = unsafe { cstr_to_string(name) };
    no_panic(std::ptr::null_mut(), move || {
        string_to_cstr(ctp_introspect::getenv(&name, max_size))
    })
}

/// # Safety
/// `name`/`value` must be NULL or valid NUL-terminated strings.
#[no_mangle]
pub unsafe extern "C" fn ctp_rs_setenv(name: *const c_char, value: *const c_char, overwrite: i32) {
    let name = unsafe { cstr_to_string(name) };
    let value = unsafe { cstr_to_string(value) };
    no_panic((), move || {
        ctp_introspect::setenv(&name, &value, overwrite != 0)
    });
}

/// # Safety
/// `name` must be NULL or a valid NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn ctp_rs_unsetenv(name: *const c_char) {
    let name = unsafe { cstr_to_string(name) };
    no_panic((), move || ctp_introspect::unsetenv(&name));
}

// ---------------------------------------------------------------------------
// Host / process / thread introspection
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn ctp_rs_get_cpu_count() -> u64 {
    no_panic(1, || ctp_introspect::get_cpu_count() as u64)
}

#[no_mangle]
pub extern "C" fn ctp_rs_get_page_size() -> u64 {
    no_panic(4096, || ctp_introspect::get_page_size() as u64)
}

#[no_mangle]
pub extern "C" fn ctp_rs_get_pid() -> u32 {
    no_panic(0, ctp_introspect::get_pid)
}

#[no_mangle]
pub extern "C" fn ctp_rs_get_tid() -> u64 {
    no_panic(0, ctp_introspect::get_tid)
}

#[no_mangle]
pub extern "C" fn ctp_rs_get_ram_capacity() -> u64 {
    no_panic(0, || ctp_introspect::get_ram_capacity() as u64)
}

#[no_mangle]
pub extern "C" fn ctp_rs_get_hostname() -> *mut c_char {
    no_panic(std::ptr::null_mut(), || {
        string_to_cstr(ctp_introspect::get_hostname())
    })
}

#[no_mangle]
pub extern "C" fn ctp_rs_get_home_dir() -> *mut c_char {
    no_panic(std::ptr::null_mut(), || {
        string_to_cstr(ctp_introspect::get_home_dir())
    })
}

#[no_mangle]
pub extern "C" fn ctp_rs_is_process_alive(pid: u32) -> i32 {
    no_panic(0, move || ctp_introspect::is_process_alive(pid) as i32)
}

#[no_mangle]
pub extern "C" fn ctp_rs_sleep_for_us(us: u64) {
    no_panic((), move || ctp_introspect::sleep_for_us(us));
}

#[no_mangle]
pub extern "C" fn ctp_rs_yield_thread() {
    no_panic((), ctp_introspect::yield_thread);
}

#[no_mangle]
pub extern "C" fn ctp_rs_thread_cpu_time_ns() -> u64 {
    no_panic(0, ctp_introspect::thread_cpu_time_ns)
}

/// # Safety
/// `name` must be NULL or a valid NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn ctp_rs_set_current_thread_name(name: *const c_char) {
    let name = unsafe { cstr_to_string(name) };
    no_panic((), move || ctp_introspect::set_current_thread_name(&name));
}

// ---------------------------------------------------------------------------
// Child processes (opaque handle)
// ---------------------------------------------------------------------------

/// Spawn `exe` with `argc` arguments from `argv`. Returns an opaque handle
/// (free with ctp_rs_child_free) or NULL on failure.
///
/// # Safety
/// `exe` must be a valid NUL-terminated string; `argv` must point to `argc`
/// valid NUL-terminated strings (argv may be NULL when argc == 0).
#[no_mangle]
pub unsafe extern "C" fn ctp_rs_spawn_process(
    exe: *const c_char,
    argv: *const *const c_char,
    argc: usize,
) -> *mut ctp_introspect::SpawnedProcess {
    let exe = unsafe { cstr_to_string(exe) };
    let mut args = Vec::with_capacity(argc);
    for i in 0..argc {
        // SAFETY: caller contract — argv has argc valid entries.
        let p = unsafe { *argv.add(i) };
        args.push(unsafe { cstr_to_string(p) });
    }
    no_panic(std::ptr::null_mut(), move || {
        let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();
        match ctp_introspect::spawn_process(&exe, &arg_refs) {
            Ok(child) => Box::into_raw(Box::new(child)),
            Err(_) => std::ptr::null_mut(),
        }
    })
}

/// # Safety
/// `proc_` must be a live handle from ctp_rs_spawn_process.
#[no_mangle]
pub unsafe extern "C" fn ctp_rs_child_pid(proc_: *mut ctp_introspect::SpawnedProcess) -> u32 {
    if proc_.is_null() {
        return 0;
    }
    // SAFETY: caller contract — live handle.
    let child = unsafe { &*proc_ };
    no_panic(0, || child.pid())
}

/// # Safety
/// `proc_` must be a live handle from ctp_rs_spawn_process.
#[no_mangle]
pub unsafe extern "C" fn ctp_rs_is_child_running(
    proc_: *mut ctp_introspect::SpawnedProcess,
) -> i32 {
    if proc_.is_null() {
        return 0;
    }
    // SAFETY: caller contract — live, uniquely-held handle.
    let child = unsafe { &mut *proc_ };
    no_panic(0, || ctp_introspect::is_child_running(child) as i32)
}

/// # Safety
/// `proc_` must be a live handle from ctp_rs_spawn_process.
#[no_mangle]
pub unsafe extern "C" fn ctp_rs_terminate_child(
    proc_: *mut ctp_introspect::SpawnedProcess,
    grace_ms: u64,
) {
    if proc_.is_null() {
        return;
    }
    // SAFETY: caller contract — live, uniquely-held handle.
    let child = unsafe { &mut *proc_ };
    no_panic((), || ctp_introspect::terminate_child(child, grace_ms));
}

/// # Safety
/// `proc_` must be NULL or a handle from ctp_rs_spawn_process, not freed
/// before. The child is NOT terminated by freeing.
#[no_mangle]
pub unsafe extern "C" fn ctp_rs_child_free(proc_: *mut ctp_introspect::SpawnedProcess) {
    if !proc_.is_null() {
        // SAFETY: caller contract — exclusive ownership transferred here.
        drop(unsafe { Box::from_raw(proc_) });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ffi_env_and_string_roundtrip() {
        let name = CString::new("CTP_RS_FFI_TEST").unwrap();
        let value = CString::new("hello").unwrap();
        unsafe {
            ctp_rs_setenv(name.as_ptr(), value.as_ptr(), 1);
            let got = ctp_rs_getenv(name.as_ptr(), 1024);
            assert_eq!(CStr::from_ptr(got).to_str().unwrap(), "hello");
            ctp_rs_string_free(got);
            ctp_rs_unsetenv(name.as_ptr());
            let got = ctp_rs_getenv(name.as_ptr(), 1024);
            assert_eq!(CStr::from_ptr(got).to_str().unwrap(), "");
            ctp_rs_string_free(got);
        }
    }

    #[test]
    fn ffi_introspection_basics() {
        assert!(ctp_rs_get_cpu_count() >= 1);
        assert!(ctp_rs_get_page_size() >= 512);
        assert!(ctp_rs_get_pid() > 0);
        assert_eq!(ctp_rs_is_process_alive(ctp_rs_get_pid()), 1);
        let h = ctp_rs_get_hostname();
        unsafe { ctp_rs_string_free(h) };
        assert_eq!(ctp_rs_abi_version(), 1);
    }
}
