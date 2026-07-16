// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).
//
// Rust port of the core `ctp::SystemInfo` surface
// (clio_ctp/introspect/system_info.h): environment, host/process/thread
// introspection, child-process management, and aligned allocation.
//
// Shared-memory create/open/map is deliberately NOT here — it belongs to the
// `memory` module migration (phase 3, see rust/MIGRATION.md) where the SHM
// backends move as one unit.

#![deny(unsafe_op_in_unsafe_fn)]

use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

// ---------------------------------------------------------------------------
// Environment (Getenv / Setenv / Unsetenv)
// ---------------------------------------------------------------------------

/// `SystemInfo::Getenv` — empty string when unset (C++ parity), with the same
/// max-size guard (values longer than `max_size` are treated as unset).
pub fn getenv(name: &str, max_size: usize) -> String {
    match std::env::var(name) {
        Ok(v) if v.len() <= max_size => v,
        _ => String::new(),
    }
}

/// `SystemInfo::Setenv`. `overwrite == false` preserves an existing value.
pub fn setenv(name: &str, value: &str, overwrite: bool) {
    if !overwrite && std::env::var_os(name).is_some() {
        return;
    }
    std::env::set_var(name, value);
}

/// `SystemInfo::Unsetenv`.
pub fn unsetenv(name: &str) {
    std::env::remove_var(name);
}

// ---------------------------------------------------------------------------
// Host / process / thread introspection
// ---------------------------------------------------------------------------

/// `SystemInfo::GetCpuCount`.
pub fn get_cpu_count() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// `SystemInfo::GetPageSize`.
pub fn get_page_size() -> usize {
    #[cfg(unix)]
    {
        // SAFETY: sysconf with a valid name has no preconditions.
        let sz = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if sz > 0 {
            return sz as usize;
        }
        4096
    }
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::SystemInformation::{GetSystemInfo, SYSTEM_INFO};
        let mut info: SYSTEM_INFO = unsafe { std::mem::zeroed() };
        // SAFETY: GetSystemInfo writes into the provided struct.
        unsafe { GetSystemInfo(&mut info) };
        info.dwPageSize as usize
    }
}

/// `SystemInfo::GetPid`.
pub fn get_pid() -> u32 {
    std::process::id()
}

/// `SystemInfo::GetTid` — the OS thread id.
pub fn get_tid() -> u64 {
    #[cfg(target_os = "linux")]
    {
        // SAFETY: gettid has no preconditions.
        unsafe { libc::gettid() as u64 }
    }
    #[cfg(all(unix, not(target_os = "linux")))]
    {
        // macOS: pthread_threadid_np. SAFETY: writes into a valid u64.
        let mut tid: u64 = 0;
        unsafe { libc::pthread_threadid_np(0, &mut tid) };
        tid
    }
    #[cfg(windows)]
    {
        // SAFETY: no preconditions.
        unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() as u64 }
    }
}

/// `SystemInfo::GetRamCapacity` — total physical RAM in bytes (0 if unknown).
pub fn get_ram_capacity() -> usize {
    #[cfg(unix)]
    {
        // SAFETY: sysconf with valid names has no preconditions.
        let pages = unsafe { libc::sysconf(libc::_SC_PHYS_PAGES) };
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if pages > 0 && page > 0 {
            return (pages as usize).saturating_mul(page as usize);
        }
        0
    }
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
        let mut st: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
        st.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
        // SAFETY: struct length is set; API writes into it.
        if unsafe { GlobalMemoryStatusEx(&mut st) } != 0 {
            st.ullTotalPhys as usize
        } else {
            0
        }
    }
}

/// `SystemInfo::GetHostname` — best-effort, empty on failure (C++ parity).
pub fn get_hostname() -> String {
    #[cfg(unix)]
    {
        let mut buf = [0u8; 256];
        // SAFETY: buffer pointer/length are valid for the call.
        let rc = unsafe { libc::gethostname(buf.as_mut_ptr() as *mut libc::c_char, buf.len()) };
        if rc == 0 {
            let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
            return String::from_utf8_lossy(&buf[..end]).into_owned();
        }
        String::new()
    }
    #[cfg(windows)]
    {
        // COMPUTERNAME is maintained by the OS; matches Win32 GetComputerName
        // for the NetBIOS name, which is what the C++ side reports.
        std::env::var("COMPUTERNAME").unwrap_or_default()
    }
}

/// `SystemInfo::GetHomeDir` — HOME on POSIX, USERPROFILE on Windows; empty
/// string when neither is set (C++ parity).
pub fn get_home_dir() -> String {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Per-user clio tmp directory (GetMemfdDir / GetMemfdPath / EnsureMemfdDir)
// ---------------------------------------------------------------------------

fn current_user() -> String {
    std::env::var("USER")
        .or_else(|_| std::env::var("USERNAME"))
        .unwrap_or_else(|_| "unknown".to_string())
}

/// `SystemInfo::GetMemfdDir` — the per-user clio tmp directory.
pub fn get_memfd_dir() -> PathBuf {
    #[cfg(unix)]
    {
        PathBuf::from(format!("/tmp/clio_{}", current_user()))
    }
    #[cfg(windows)]
    {
        std::env::temp_dir().join(format!("clio_{}", current_user()))
    }
}

/// `SystemInfo::GetMemfdPath`.
pub fn get_memfd_path(name: &str) -> PathBuf {
    get_memfd_dir().join(name)
}

/// `SystemInfo::EnsureMemfdDir`.
pub fn ensure_memfd_dir() -> std::io::Result<()> {
    std::fs::create_dir_all(get_memfd_dir())
}

// ---------------------------------------------------------------------------
// Process liveness and child management (SpawnProcess / IsChildRunning /
// TerminateChild — the #721 SystemInfo surface)
// ---------------------------------------------------------------------------

/// `SystemInfo::IsProcessAlive` for an arbitrary (non-child) pid.
pub fn is_process_alive(pid: u32) -> bool {
    #[cfg(unix)]
    {
        // SAFETY: kill(pid, 0) probes existence without signaling.
        let rc = unsafe { libc::kill(pid as libc::pid_t, 0) };
        rc == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
    }
    #[cfg(windows)]
    {
        use windows_sys::Win32::Foundation::{CloseHandle, STILL_ACTIVE};
        use windows_sys::Win32::System::Threading::{
            GetExitCodeProcess, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
        };
        // SAFETY: handle is checked and closed on every path.
        unsafe {
            let h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
            if h.is_null() {
                return false;
            }
            let mut code: u32 = 0;
            let ok = GetExitCodeProcess(h, &mut code) != 0;
            CloseHandle(h);
            ok && code == STILL_ACTIVE as u32
        }
    }
}

/// Handle to a child spawned by [`spawn_process`] (mirrors
/// `SystemInfo::SpawnedProcess`).
pub struct SpawnedProcess {
    child: Child,
}

impl SpawnedProcess {
    pub fn pid(&self) -> u32 {
        self.child.id()
    }
}

/// `SystemInfo::SpawnProcess` — spawn `exe` with `args`, stdio inherited.
pub fn spawn_process(exe: &str, args: &[&str]) -> std::io::Result<SpawnedProcess> {
    let child = Command::new(exe).args(args).stdin(Stdio::null()).spawn()?;
    Ok(SpawnedProcess { child })
}

/// `SystemInfo::IsChildRunning` — non-blocking liveness check that also
/// reaps the child if it has exited (no zombie left behind).
pub fn is_child_running(proc_: &mut SpawnedProcess) -> bool {
    matches!(proc_.child.try_wait(), Ok(None))
}

/// `SystemInfo::TerminateChild` — graceful-then-forced termination.
/// POSIX: SIGTERM, wait up to `grace_ms`, then SIGKILL. Windows has no
/// graceful signal for arbitrary children, so this terminates directly.
pub fn terminate_child(proc_: &mut SpawnedProcess, grace_ms: u64) {
    if !is_child_running(proc_) {
        return;
    }
    #[cfg(unix)]
    {
        // SAFETY: signaling our own child's pid.
        unsafe { libc::kill(proc_.child.id() as libc::pid_t, libc::SIGTERM) };
        let deadline = std::time::Instant::now() + Duration::from_millis(grace_ms);
        while std::time::Instant::now() < deadline {
            if !is_child_running(proc_) {
                return;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }
    #[cfg(windows)]
    {
        let _ = grace_ms; // no graceful path on Windows
    }
    let _ = proc_.child.kill();
    let _ = proc_.child.wait();
}

// ---------------------------------------------------------------------------
// Thread utilities
// ---------------------------------------------------------------------------

/// `SystemInfo::YieldThread`.
pub fn yield_thread() {
    std::thread::yield_now();
}

/// `SystemInfo::SleepForUs`.
pub fn sleep_for_us(us: u64) {
    std::thread::sleep(Duration::from_micros(us));
}

/// `SystemInfo::SetCurrentThreadName` — best-effort, truncated to OS limits.
pub fn set_current_thread_name(name: &str) {
    #[cfg(target_os = "linux")]
    {
        // Linux limit: 15 chars + NUL.
        let trunc: String = name.chars().take(15).collect();
        if let Ok(c) = std::ffi::CString::new(trunc) {
            // SAFETY: valid NUL-terminated name for the current thread.
            unsafe { libc::pthread_setname_np(libc::pthread_self(), c.as_ptr()) };
        }
    }
    #[cfg(target_os = "macos")]
    {
        if let Ok(c) = std::ffi::CString::new(name) {
            // SAFETY: macOS variant names the calling thread only.
            unsafe { libc::pthread_setname_np(c.as_ptr()) };
        }
    }
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadDescription};
        let wide: Vec<u16> = name.encode_utf16().chain(std::iter::once(0)).collect();
        // SAFETY: valid wide string; current-thread pseudo handle.
        unsafe { SetThreadDescription(GetCurrentThread(), wide.as_ptr()) };
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
    {
        let _ = name;
    }
}

/// `SystemInfo::ThreadCpuTimeNs` — CPU time (user+kernel) of the calling
/// thread in nanoseconds; 0 if unsupported.
pub fn thread_cpu_time_ns() -> u64 {
    #[cfg(unix)]
    {
        let mut ts = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        // SAFETY: valid clock id and out-pointer.
        let rc = unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts) };
        if rc == 0 {
            return (ts.tv_sec as u64) * 1_000_000_000 + ts.tv_nsec as u64;
        }
        0
    }
    #[cfg(windows)]
    {
        use windows_sys::Win32::Foundation::FILETIME;
        use windows_sys::Win32::System::Threading::{GetCurrentThread, GetThreadTimes};
        let mut c: FILETIME = unsafe { std::mem::zeroed() };
        let mut e: FILETIME = unsafe { std::mem::zeroed() };
        let mut k: FILETIME = unsafe { std::mem::zeroed() };
        let mut u: FILETIME = unsafe { std::mem::zeroed() };
        // SAFETY: pseudo handle + valid out-pointers.
        let ok = unsafe { GetThreadTimes(GetCurrentThread(), &mut c, &mut e, &mut k, &mut u) };
        if ok != 0 {
            let ft = |t: &FILETIME| ((t.dwHighDateTime as u64) << 32) | t.dwLowDateTime as u64;
            // FILETIME is in 100 ns units.
            return (ft(&k) + ft(&u)) * 100;
        }
        0
    }
}

// ---------------------------------------------------------------------------
// Aligned allocation (AlignedAlloc / AlignedFree)
// ---------------------------------------------------------------------------

/// `SystemInfo::AlignedAlloc`. Must be freed with [`aligned_free`] using the
/// SAME alignment and size (Rust's allocator requires the layout to match —
/// this is stricter than the C++ contract and is enforced by the FFI shim).
pub fn aligned_alloc(alignment: usize, size: usize) -> *mut u8 {
    let Ok(layout) = std::alloc::Layout::from_size_align(size, alignment) else {
        return std::ptr::null_mut();
    };
    if size == 0 {
        return std::ptr::null_mut();
    }
    // SAFETY: layout is valid and non-zero-sized.
    unsafe { std::alloc::alloc(layout) }
}

/// `SystemInfo::AlignedFree` counterpart.
///
/// # Safety
/// `ptr` must have been returned by [`aligned_alloc`] with exactly this
/// `alignment` and `size`, and not freed before.
pub unsafe fn aligned_free(ptr: *mut u8, alignment: usize, size: usize) {
    if ptr.is_null() {
        return;
    }
    if let Ok(layout) = std::alloc::Layout::from_size_align(size, alignment) {
        // SAFETY: caller contract guarantees allocation provenance + layout.
        unsafe { std::alloc::dealloc(ptr, layout) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_roundtrip_and_overwrite_semantics() {
        let key = "CTP_RS_TEST_ENV_KEY";
        unsetenv(key);
        assert_eq!(getenv(key, 1024), "");
        setenv(key, "one", true);
        assert_eq!(getenv(key, 1024), "one");
        setenv(key, "two", false); // must NOT overwrite
        assert_eq!(getenv(key, 1024), "one");
        setenv(key, "two", true);
        assert_eq!(getenv(key, 1024), "two");
        // max_size guard: an over-long value reads as unset
        assert_eq!(getenv(key, 1), "");
        unsetenv(key);
        assert_eq!(getenv(key, 1024), "");
    }

    #[test]
    fn host_process_basics() {
        assert!(get_cpu_count() >= 1);
        assert!(get_page_size() >= 512);
        assert!(get_pid() > 0);
        assert!(get_tid() > 0);
        assert!(get_ram_capacity() > 0);
        assert!(!get_home_dir().is_empty());
        assert!(is_process_alive(get_pid()));
        assert!(!is_process_alive(u32::MAX - 1));
    }

    #[test]
    fn memfd_dir_paths() {
        let dir = get_memfd_dir();
        assert!(dir.to_string_lossy().contains("clio_"));
        assert_eq!(get_memfd_path("x.ipc"), dir.join("x.ipc"));
        ensure_memfd_dir().unwrap();
        assert!(dir.exists());
    }

    #[test]
    fn spawn_liveness_terminate() {
        // A child that would run for 30s unless terminated.
        #[cfg(windows)]
        let mut child = spawn_process("cmd", &["/C", "ping -n 30 127.0.0.1 > NUL"]).unwrap();
        #[cfg(unix)]
        let mut child = spawn_process("sleep", &["30"]).unwrap();

        assert!(child.pid() > 0);
        assert!(is_child_running(&mut child));
        assert!(is_process_alive(child.pid()));
        let t0 = std::time::Instant::now();
        terminate_child(&mut child, 2000);
        assert!(!is_child_running(&mut child));
        // Termination must not have burned the whole grace period.
        assert!(t0.elapsed() < Duration::from_secs(10));
    }

    #[test]
    fn thread_utils_and_aligned_alloc() {
        set_current_thread_name("ctp-rs-test");
        let before = thread_cpu_time_ns();
        let mut acc = 0u64;
        for i in 0..2_000_000u64 {
            acc = acc.wrapping_add(i);
        }
        std::hint::black_box(acc);
        assert!(thread_cpu_time_ns() >= before);

        let p = aligned_alloc(4096, 8192);
        assert!(!p.is_null());
        assert_eq!(p as usize % 4096, 0);
        unsafe { aligned_free(p, 4096, 8192) };
    }
}
