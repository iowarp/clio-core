// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).

//! Asynchronous file I/O — Rust port of `clio_ctp/io/`.
//!
//! Ports the submit/poll/complete abstraction defined by `io/async_io.h`
//! (`ctp::AsyncIO`), the backend selection of `io/async_io_factory.h`, the
//! errno/Win32 classification of `io/io_error.h`, and the *behaviour* of the
//! concrete backends (`iocp_io.cc`, `libaio_io.h`, `posix_aio.h`,
//! `iouring_io.h`) — all of which share one algorithm:
//!
//! * `Open` keeps **two** descriptors for the same path: a regular one and an
//!   `O_DIRECT` one (Windows: `FILE_FLAG_NO_BUFFERING`). The direct open is
//!   allowed to fail; everything then rides the regular descriptor.
//! * `Write`/`Read` pick the direct descriptor iff the buffer address **and**
//!   the size are both 4096-aligned, then submit and return an `IoToken`.
//! * `IsComplete(token, &result)` is a non-blocking poll: it looks in the
//!   completed map, harvests kernel completions, looks again; a completion is
//!   **consumed** (erased) by the poll that reports it.
//! * `Close` tears everything down and forgets all in-flight state; it is
//!   idempotent and the destructor calls it.
//!
//! # C++ → Rust name mapping
//!
//! | C++ (`clio_ctp/io/…`) | Rust (this module) |
//! |---|---|
//! | `ctp::IoToken` (`uint64_t`) | [`IoToken`] (`u64`) |
//! | `ctp::kInvalidIoToken` | [`INVALID_IO_TOKEN`] |
//! | `ctp::IoResult` | [`IoResult`] (`bytes_transferred: isize`, `error_code: i32`) |
//! | `mode_t` (`int` on Win32) | [`ModeT`] (`u32`) |
//! | `O_RDONLY`/`O_WRONLY`/`O_RDWR`/`O_APPEND`/`O_CREAT`/`O_TRUNC`/`O_DIRECT` | [`O_RDONLY`]/[`O_WRONLY`]/[`O_RDWR`]/[`O_APPEND`]/[`O_CREAT`]/[`O_TRUNC`]/[`O_DIRECT`] |
//! | `class ctp::AsyncIO` (abstract) | [`trait AsyncIo`](AsyncIo) |
//! | `AsyncIO::Open` | [`AsyncIo::open`] |
//! | `AsyncIO::GetFileSize` | [`AsyncIo::get_file_size`] |
//! | `AsyncIO::Truncate` | [`AsyncIo::truncate`] |
//! | `AsyncIO::Write` | [`AsyncIo::write`] (`unsafe`) |
//! | `AsyncIO::Read` | [`AsyncIo::read`] (`unsafe`) |
//! | `AsyncIO::IsComplete` | [`AsyncIo::is_complete`] |
//! | `AsyncIO::Close` | [`AsyncIo::close`] |
//! | `AsyncIO::GetEventFd` | [`AsyncIo::get_event_fd`] |
//! | `IocpAsyncIO` / `PosixAsyncIO` / `LinuxAioAsyncIO` / `IoUringAsyncIO` | [`ThreadPoolAsyncIo`] (one fallback, see D1) |
//! | `IocpAsyncIO::Impl::SelectHandle` / `*::SelectFd` | `ThreadPoolAsyncIo::select_file` (private) |
//! | `*::SubmitIO` | `ThreadPoolAsyncIo::submit` (private) |
//! | `*::HarvestCompletions` | implicit (workers publish; see D3) |
//! | `enum class AsyncIoBackend` | [`AsyncIoBackend`] |
//! | `AsyncIoFactory::Get` | [`AsyncIoFactory::get`] |
//! | `AsyncIoFactory::GetDefaultBackend` | [`AsyncIoFactory::default_backend`] |
//! | `enum class IoError` | [`IoError`] |
//! | `IoErrorName` | [`io_error_name`] (also [`Display`](std::fmt::Display)) |
//! | `IsTransient` | [`is_transient`] |
//! | `IsFatalDevice` | [`is_fatal_device`] |
//! | `ClassifyErrno` | [`classify_errno`] |
//! | `ClassifyLastErrno` | [`classify_last_errno`] (unix only, see D10) |
//! | `ClassifyWinError` | [`classify_win_error`] (see D10) |
//! | — (addition) | [`ThreadPoolAsyncIo::wait_complete`], [`write_owned`](ThreadPoolAsyncIo::write_owned), [`read_owned`](ThreadPoolAsyncIo::read_owned), [`try_take_read`](ThreadPoolAsyncIo::try_take_read), [`in_flight_count`](ThreadPoolAsyncIo::in_flight_count) |
//!
//! # Semantic divergences
//!
//! **D1 — Mechanism: thread pool, not IOCP/libaio.** The single concrete
//! backend here is [`ThreadPoolAsyncIo`]: submissions are queued and executed
//! by worker threads doing *positional* I/O (`pread`/`pwrite` on unix,
//! `seek_read`/`seek_write` on Windows). No IOCP completion port, no
//! `io_submit`, no `io_uring`, no `aio_read`. The observable API contract —
//! tokens, alignment-adaptive descriptor selection, poll-consumes-completion,
//! error codes, idempotent close — is preserved; the kernel mechanism is not.
//! Consequence: no kernel-level queue-depth backpressure and no zero-syscall
//! completion harvesting.
//!
//! **D2 — `io_depth` sizes the worker pool, not a kernel queue.** C++ passes it
//! to `io_setup`/`io_uring_queue_init` (IOCP and POSIX AIO *ignore* it). Here it
//! is clamped to `1..=`[`MAX_WORKERS`] worker threads. `io_depth == 0` yields one
//! worker and works, whereas `libaio`'s `io_setup(0, …)` fails `EINVAL` and makes
//! `Open` return false. Submission is never rejected for depth (matching IOCP,
//! diverging from `libaio`, whose `io_submit` can return `EAGAIN` at depth and
//! then yields `kInvalidIoToken`).
//!
//! **D3 — Completions are published, not harvested.** `IsComplete`'s
//! check → harvest → re-check dance exists because the kernel owns completions.
//! Workers here insert into the completed map themselves, so
//! [`is_complete`](AsyncIo::is_complete) is a single lookup. Externally
//! identical: still non-blocking, still consumes the completion, still returns
//! `false` for unknown/in-flight tokens.
//!
//! **D4 — `close()` quiesces; C++ cancels without waiting.** `IocpAsyncIO::Close`
//! calls `CancelIoEx` and closes the handles while ops may still touch caller
//! buffers; `LinuxAioAsyncIO::Close` calls `io_destroy` likewise. That is
//! tolerable in C++ and *unsound* in Rust: the raw-pointer API would let a
//! caller free a buffer a worker is still writing into. So [`close`](AsyncIo::close)
//! drops queued-but-unstarted jobs, **blocks until executing jobs finish**, and
//! only then releases the files. Tokens dropped from the queue never complete
//! (C++ likewise forgets them). `Drop` calls `close()` then joins the workers.
//!
//! **D5 — The "direct" descriptor is never unbuffered on Windows.** `libaio_io.h`
//! / `posix_aio.h` open it with `O_DIRECT`, and this port does the same on Linux
//! (`custom_flags(O_DIRECT)`), failing softly to a regular second handle exactly
//! as C++ does. On Windows, `iocp_io.cc` uses
//! `FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH`; this port opens a second
//! *plain* handle instead, degrading exactly the way `posix_aio.h` documents for
//! Darwin (`#define O_DIRECT 0`). Rationale: `SelectHandle` only checks buffer
//! and size alignment, never *offset* alignment, so the C++ Windows path faults
//! `ERROR_INVALID_PARAMETER` on an aligned buffer at an unaligned offset. Not
//! reproducing that bug is deliberate. The selection *rule* is ported verbatim,
//! so the direct handle is still chosen for 4096-aligned buffer+size.
//!
//! **D6 — `error_code` is platform-raw, as in C++.** Unix: `errno`
//! (matching `libaio`'s `-res` and `aio_error`). Windows: the Win32 code
//! `std::io` reports (matching `iocp_io.cc`, which stores `GetLastError()` on the
//! synchronous-failure path). So classify with [`classify_errno`] on unix and
//! [`classify_win_error`] on Windows. Unlike `iocp_io.cc`, this port never
//! reports a raw `NTSTATUS` (that file's harvest path stores
//! `OVERLAPPED::Internal`, an NTSTATUS, into the same `int` that otherwise holds
//! a Win32 code — a genuine C++ inconsistency, not carried over). A non-OS
//! `io::Error` (should not occur for positional I/O) maps to
//! [`FALLBACK_OS_ERROR`].
//!
//! **D7 — Immediate failures still return a live token (IOCP style).** A
//! negative `offset` cannot be submitted; instead of returning
//! `INVALID_IO_TOKEN`, the op is completed in place with `EINVAL` /
//! `bytes_transferred = -1` and a valid token is returned — mirroring
//! `IocpAsyncIO::Impl::SubmitIO`, which parks the `GetLastError()` value in
//! `completed[token]` and returns the token. `INVALID_IO_TOKEN` is returned only
//! when no descriptor is available (no `Open`, or post-`Close`), matching
//! `SelectHandle() == INVALID_HANDLE_VALUE`.
//!
//! **D8 — `get_event_fd()` always returns -1.** Matches `IocpAsyncIO` and
//! `PosixAsyncIO`. The `eventfd`/epoll integration of `libaio_io.h` and
//! `iouring_io.h` has no counterpart here and is **not ported**; use
//! [`wait_complete`](ThreadPoolAsyncIo::wait_complete) (a condvar wait, timeout in
//! **milliseconds** per MIGRATION.md) for blocking waits.
//!
//! **D9 — `&self`, not `&mut self`.** The C++ methods are non-`const` but every
//! backend is internally synchronized by its own `std::mutex`, so the honest
//! Rust signature is `&self` with interior mutability. `AsyncIo: Send + Sync`;
//! an engine can be shared through an `Arc` and submitted to concurrently, as
//! the C++ locking already implies. Mutex poisoning is swallowed
//! (`into_inner`) because a C++ `std::mutex` has no such state.
//!
//! **D10 — Classification helpers.** `ClassifyErrno`'s `#ifdef` ladder becomes
//! `cfg`-gated constants: unix values come from `libc`, Windows values are the
//! UCRT `<errno.h>` numbers. `EDQUOT` is absent from the UCRT, so — exactly like
//! the `#ifdef EDQUOT` guard — that branch does not exist on Windows. The
//! `EWOULDBLOCK`-vs-`EAGAIN` `#if` guard is unnecessary in Rust (an `if` ladder,
//! not `case` labels, so equal values are merely redundant). `ClassifyLastErrno`
//! is unix-only: `io::Error::last_os_error()` is `GetLastError()` (Win32) on
//! Windows, not the CRT `errno`, so exposing it there would silently misclassify;
//! [`classify_last_os_error`] is the portable spelling. `ClassifyWinError` is
//! `#if defined(_WIN32)` in C++ but is compiled unconditionally here — it is pure
//! table lookup needing no Win32 headers, and this keeps it unit-testable
//! everywhere.
//!
//! **D11 — Backends that C++ compiles out return `None`.** This crate has no
//! libaio/io_uring/NIXL bindings, so [`AsyncIoFactory::get`] yields `None` for
//! [`AsyncIoBackend::LinuxAio`], [`IoUring`](AsyncIoBackend::IoUring) and
//! [`Nixl`](AsyncIoBackend::Nixl) — precisely what the C++ factory's `default:`
//! arm returns when `CTP_ENABLE_LIBAIO`/`CTP_ENABLE_IO_URING`/`CTP_ENABLE_NIXL`
//! are off. Platform gating is ported verbatim: `Iocp` is Windows-only,
//! `PosixAio` is unix-only, and [`default_backend`](AsyncIoFactory::default_backend)
//! resolves to `Iocp` on Windows and `PosixAio` elsewhere.
//!
//! **D12 — `open()` flag handling goes through `OpenOptions`.** C++ hands
//! `base_flags` straight to `open(2)`/`CreateFileA`. Combinations that
//! `OpenOptions` rejects — `O_CREAT`/`O_TRUNC` without write access,
//! `O_APPEND` + `O_TRUNC`, an `O_ACCMODE` of 3 — make `open()` return `false`
//! where POSIX would succeed. `mode` is applied on unix
//! (`OpenOptionsExt::mode`) and ignored on Windows (as `iocp_io.cc` does:
//! `(void)mode`). Re-`open()`ing without `close()` drops the previous handles
//! rather than leaking the fds (a real leak in every C++ backend).
//!
//! **D13 — No 32-bit request truncation.** `iocp_io.cc` does
//! `static_cast<DWORD>(size)`, silently truncating requests ≥ 4 GiB. Sizes here
//! are `usize` end-to-end and are never truncated.

#![deny(unsafe_op_in_unsafe_fn)]

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// async_io.h — tokens, results, flags
// ---------------------------------------------------------------------------

/// Opaque token returned by a submit, used to check completion.
///
/// `ctp::IoToken`.
pub type IoToken = u64;

/// `ctp::kInvalidIoToken` — never handed out by a successful submit.
pub const INVALID_IO_TOKEN: IoToken = 0;

/// `mode_t`: file creation mode (e.g. `0o644`). See D12 for Windows.
pub type ModeT = u32;

/// Result of a completed operation (`ctp::IoResult`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IoResult {
    /// Bytes actually transferred (`-1` on error).
    pub bytes_transferred: isize,
    /// `0` on success, otherwise a raw OS error code (see D6).
    pub error_code: i32,
}

impl IoResult {
    /// A zero-filled result, matching a default-constructed C++ `IoResult`'s
    /// use as an out-parameter.
    pub const fn new() -> Self {
        IoResult { bytes_transferred: 0, error_code: 0 }
    }

    /// True when the op transferred without error.
    pub fn is_ok(&self) -> bool {
        self.error_code == 0 && self.bytes_transferred >= 0
    }

    /// Classification of [`IoResult::error_code`] on this platform (see D6).
    pub fn classify(&self) -> IoError {
        classify_last_os_error_value(self.error_code)
    }
}

impl Default for IoResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Alignment at which the direct descriptor becomes eligible (C++ hard-codes
/// `% 4096` in every `SelectFd`/`SelectHandle`).
pub const DIRECT_IO_ALIGN: usize = 4096;

/// Worker-thread ceiling; see D2.
pub const MAX_WORKERS: usize = 8;

/// Reported in [`IoResult::error_code`] when an `io::Error` carries no OS code
/// (see D6).
pub const FALLBACK_OS_ERROR: i32 = -1;

// Open flags. On Windows async_io.h hand-rolls the MSVC `_O_*` values rather
// than including <fcntl.h>; those exact values are reproduced here. On unix the
// platform's own values are used, since they differ (e.g. Linux `O_CREAT` is
// 0o100, not 0x100).
#[cfg(windows)]
pub const O_RDONLY: i32 = 0x0000;
#[cfg(windows)]
pub const O_WRONLY: i32 = 0x0001;
#[cfg(windows)]
pub const O_RDWR: i32 = 0x0002;
#[cfg(windows)]
pub const O_APPEND: i32 = 0x0008;
#[cfg(windows)]
pub const O_CREAT: i32 = 0x0100;
#[cfg(windows)]
pub const O_TRUNC: i32 = 0x0200;
/// Access-mode mask (`O_RDONLY | O_WRONLY | O_RDWR` bits).
#[cfg(windows)]
pub const O_ACCMODE: i32 = 0x0003;
/// Windows has no `O_DIRECT`; `posix_aio.h` defines it as 0 wherever it is
/// missing so the `& !O_DIRECT` / `| O_DIRECT` bit-ops still work. See D5.
#[cfg(windows)]
pub const O_DIRECT: i32 = 0;

#[cfg(unix)]
pub const O_RDONLY: i32 = libc::O_RDONLY;
#[cfg(unix)]
pub const O_WRONLY: i32 = libc::O_WRONLY;
#[cfg(unix)]
pub const O_RDWR: i32 = libc::O_RDWR;
#[cfg(unix)]
pub const O_APPEND: i32 = libc::O_APPEND;
#[cfg(unix)]
pub const O_CREAT: i32 = libc::O_CREAT;
#[cfg(unix)]
pub const O_TRUNC: i32 = libc::O_TRUNC;
/// Access-mode mask (`O_RDONLY | O_WRONLY | O_RDWR` bits).
#[cfg(unix)]
pub const O_ACCMODE: i32 = libc::O_ACCMODE;
/// `O_DIRECT` is a Linux extension; 0 elsewhere (`posix_aio.h`'s fallback).
#[cfg(any(target_os = "linux", target_os = "android"))]
pub const O_DIRECT: i32 = libc::O_DIRECT;
#[cfg(all(unix, not(any(target_os = "linux", target_os = "android"))))]
pub const O_DIRECT: i32 = 0;

// ---------------------------------------------------------------------------
// io_error.h
// ---------------------------------------------------------------------------

/// errno constants, `#ifdef`-ladder equivalent (see D10).
mod errno {
    #[cfg(unix)]
    pub use libc::{
        EACCES, EAGAIN, EBADF, EBUSY, EFAULT, EFBIG, EINTR, EINVAL, EIO, ENODEV, ENOENT, ENOSPC,
        ENXIO, EPERM, EROFS, ETIMEDOUT, EWOULDBLOCK,
    };

    // Windows UCRT <errno.h>. EDQUOT is not defined there — see D10.
    #[cfg(windows)]
    pub const EPERM: i32 = 1;
    #[cfg(windows)]
    pub const ENOENT: i32 = 2;
    #[cfg(windows)]
    pub const EINTR: i32 = 4;
    #[cfg(windows)]
    pub const EIO: i32 = 5;
    #[cfg(windows)]
    pub const ENXIO: i32 = 6;
    #[cfg(windows)]
    pub const EBADF: i32 = 9;
    #[cfg(windows)]
    pub const EAGAIN: i32 = 11;
    #[cfg(windows)]
    pub const EACCES: i32 = 13;
    #[cfg(windows)]
    pub const EFAULT: i32 = 14;
    #[cfg(windows)]
    pub const EBUSY: i32 = 16;
    #[cfg(windows)]
    pub const ENODEV: i32 = 19;
    #[cfg(windows)]
    pub const EINVAL: i32 = 22;
    #[cfg(windows)]
    pub const EFBIG: i32 = 27;
    #[cfg(windows)]
    pub const ENOSPC: i32 = 28;
    #[cfg(windows)]
    pub const EROFS: i32 = 30;
    #[cfg(windows)]
    pub const ETIMEDOUT: i32 = 138;
    #[cfg(windows)]
    pub const EWOULDBLOCK: i32 = 140;
}

/// Portable I/O error category (`enum class ctp::IoError : uint32_t`).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IoError {
    /// No error.
    Ok = 0,
    /// Retryable: `EAGAIN`/`EWOULDBLOCK`/`EINTR`/`EBUSY`.
    Transient = 1,
    /// Media/controller error: `EIO`.
    DeviceFault = 2,
    /// Device removed/absent: `ENODEV`/`ENXIO`/`ENOENT`.
    Disconnected = 3,
    /// Out of space/quota: `ENOSPC`/`EDQUOT`/`EFBIG`.
    NoSpace = 4,
    /// Access denied/read-only: `EACCES`/`EPERM`/`EROFS`.
    Permission = 5,
    /// Bad request/handle: `EINVAL`/`EBADF`/`EFAULT`.
    Invalid = 6,
    /// Operation timed out: `ETIMEDOUT`.
    Timeout = 7,
    /// Unrecognized error.
    Unknown = 8,
}

/// Human-readable name for logging (`ctp::IoErrorName`).
pub fn io_error_name(e: IoError) -> &'static str {
    match e {
        IoError::Ok => "Ok",
        IoError::Transient => "Transient",
        IoError::DeviceFault => "DeviceFault",
        IoError::Disconnected => "Disconnected",
        IoError::NoSpace => "NoSpace",
        IoError::Permission => "Permission",
        IoError::Invalid => "Invalid",
        IoError::Timeout => "Timeout",
        IoError::Unknown => "Unknown",
    }
}

impl fmt::Display for IoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(io_error_name(*self))
    }
}

/// True if the failure is worth retrying on the same device (`ctp::IsTransient`).
pub fn is_transient(e: IoError) -> bool {
    e == IoError::Transient
}

/// True if the failure means the device should be faulted — gone or broken
/// (`ctp::IsFatalDevice`).
pub fn is_fatal_device(e: IoError) -> bool {
    e == IoError::DeviceFault || e == IoError::Disconnected
}

/// Map a POSIX `errno` value to an [`IoError`]; `error == 0` yields
/// [`IoError::Ok`] (`ctp::ClassifyErrno`).
///
/// Specific categories are checked before the generic transient bucket, in the
/// same order as the C++ ladder.
pub fn classify_errno(error: i32) -> IoError {
    if error == 0 {
        return IoError::Ok;
    }
    if error == errno::EIO {
        return IoError::DeviceFault;
    }
    if error == errno::ENODEV {
        return IoError::Disconnected;
    }
    if error == errno::ENXIO {
        return IoError::Disconnected;
    }
    if error == errno::ENOENT {
        // backing file vanished
        return IoError::Disconnected;
    }
    if error == errno::ENOSPC {
        return IoError::NoSpace;
    }
    // `#ifdef EDQUOT` — absent from the Windows UCRT (D10).
    #[cfg(unix)]
    {
        if error == libc::EDQUOT {
            return IoError::NoSpace;
        }
    }
    if error == errno::EFBIG {
        return IoError::NoSpace;
    }
    if error == errno::EACCES {
        return IoError::Permission;
    }
    if error == errno::EPERM {
        return IoError::Permission;
    }
    if error == errno::EROFS {
        return IoError::Permission;
    }
    if error == errno::ETIMEDOUT {
        return IoError::Timeout;
    }
    if error == errno::EAGAIN {
        return IoError::Transient;
    }
    // Redundant where EWOULDBLOCK == EAGAIN (Linux); harmless in an if-ladder.
    if error == errno::EWOULDBLOCK {
        return IoError::Transient;
    }
    if error == errno::EINTR {
        return IoError::Transient;
    }
    if error == errno::EBUSY {
        return IoError::Transient;
    }
    if error == errno::EINVAL {
        return IoError::Invalid;
    }
    if error == errno::EBADF {
        return IoError::Invalid;
    }
    if error == errno::EFAULT {
        return IoError::Invalid;
    }
    IoError::Unknown
}

/// Classify the current value of `errno` (`ctp::ClassifyLastErrno`).
///
/// unix only: on Windows `io::Error::last_os_error()` reports `GetLastError()`,
/// not the CRT `errno`, so this spelling would silently misclassify there. Use
/// [`classify_last_os_error`] for portable code (D10).
#[cfg(unix)]
pub fn classify_last_errno() -> IoError {
    classify_errno(io::Error::last_os_error().raw_os_error().unwrap_or(0))
}

/// Map a Win32 error code (`GetLastError()`) to an [`IoError`]
/// (`ctp::ClassifyWinError`).
///
/// Compiled on every platform (D10); the numeric Win32 constants are inlined
/// exactly as in C++, where doing so avoids including `<windows.h>`.
pub fn classify_win_error(error: u32) -> IoError {
    match error {
        0 => IoError::Ok, // ERROR_SUCCESS
        21    // ERROR_NOT_READY
        | 170 // ERROR_BUSY
        => IoError::Transient,
        23     // ERROR_CRC
        | 1117 // ERROR_IO_DEVICE
        => IoError::DeviceFault,
        2      // ERROR_FILE_NOT_FOUND
        | 3    // ERROR_PATH_NOT_FOUND
        | 55   // ERROR_DEV_NOT_EXIST
        | 1167 // ERROR_DEVICE_NOT_CONNECTED
        => IoError::Disconnected,
        39    // ERROR_DISK_FULL
        | 112 // ERROR_DISK_FULL (alt)
        => IoError::NoSpace,
        5    // ERROR_ACCESS_DENIED
        | 19 // ERROR_WRITE_PROTECT
        => IoError::Permission,
        1460 => IoError::Timeout, // ERROR_TIMEOUT
        6    // ERROR_INVALID_HANDLE
        | 87 // ERROR_INVALID_PARAMETER
        => IoError::Invalid,
        _ => IoError::Unknown,
    }
}

/// Classify a raw OS error code the way *this* platform reports one: `errno` on
/// unix, Win32 on Windows. This is the right classifier for
/// [`IoResult::error_code`] (D6); it has no C++ counterpart.
pub fn classify_last_os_error_value(error: i32) -> IoError {
    #[cfg(windows)]
    {
        if error < 0 {
            return IoError::Unknown;
        }
        classify_win_error(error as u32)
    }
    #[cfg(not(windows))]
    {
        classify_errno(error)
    }
}

/// Classify the last OS error raised on this thread, per-platform (D10).
pub fn classify_last_os_error() -> IoError {
    classify_last_os_error_value(io::Error::last_os_error().raw_os_error().unwrap_or(0))
}

// ---------------------------------------------------------------------------
// async_io.h — the abstraction
// ---------------------------------------------------------------------------

/// Submit/poll/complete async file I/O (`class ctp::AsyncIO`).
///
/// Takes `&self` throughout and is `Send + Sync`: every C++ backend guards its
/// state with a `std::mutex`, so shared concurrent use is already the contract
/// (D9).
pub trait AsyncIo: Send + Sync {
    /// Open a file; `true` on success. The engine owns the descriptors
    /// internally and opens two: one with `O_DIRECT`, one without.
    ///
    /// * `path` — file path
    /// * `flags` — [`O_RDWR`], [`O_CREAT`], … ([`O_DIRECT`] is managed
    ///   internally and stripped from `flags`)
    /// * `mode` — file creation mode (e.g. `0o644`); ignored on Windows (D12)
    fn open(&self, path: &str, flags: i32, mode: ModeT) -> bool;

    /// File size, or `-1` on error.
    fn get_file_size(&self) -> isize;

    /// Truncate/extend the file; `true` on success.
    fn truncate(&self, size: usize) -> bool;

    /// Submit an async write. Automatically selects the `O_DIRECT` descriptor
    /// if buffer and size are aligned, otherwise the regular one.
    ///
    /// Returns an [`IoToken`] for tracking, or [`INVALID_IO_TOKEN`] on failure.
    ///
    /// # Safety
    ///
    /// `buffer` must point to at least `size` initialized bytes, and the region
    /// must stay allocated and untouched by the caller until
    /// [`is_complete`](AsyncIo::is_complete) reports this token complete, or
    /// until [`close`](AsyncIo::close) returns, whichever comes first. `buffer`
    /// may be null only when `size == 0`. Concurrent overlapping submissions on
    /// the same region are the caller's problem, exactly as in C++.
    unsafe fn write(&self, buffer: *mut u8, size: usize, offset: i64) -> IoToken;

    /// Submit an async read; same alignment-adaptive logic as
    /// [`write`](AsyncIo::write).
    ///
    /// # Safety
    ///
    /// `buffer` must point to at least `size` bytes of writable, allocated
    /// memory that stays valid and otherwise untouched until
    /// [`is_complete`](AsyncIo::is_complete) reports this token complete, or
    /// until [`close`](AsyncIo::close) returns, whichever comes first. `buffer`
    /// may be null only when `size == 0`.
    unsafe fn read(&self, buffer: *mut u8, size: usize, offset: i64) -> IoToken;

    /// Non-blocking check: is this operation complete? Fills `result` and
    /// consumes the completion when it returns `true`; `false` while still in
    /// progress (or for an unknown token).
    fn is_complete(&self, token: IoToken, result: &mut IoResult) -> bool;

    /// Close all descriptors and clean up. Idempotent; blocks until executing
    /// ops quiesce (D4).
    fn close(&self);

    /// The eventfd for epoll integration; always `-1` here (D8).
    fn get_event_fd(&self) -> i32;
}

// ---------------------------------------------------------------------------
// async_io_factory.h
// ---------------------------------------------------------------------------

/// `enum class ctp::AsyncIoBackend`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AsyncIoBackend {
    /// libaio (Linux) — not built in this crate (D11).
    LinuxAio,
    /// io_uring (Linux 5.1+) — not built in this crate (D11).
    IoUring,
    /// POSIX `aio_read`/`aio_write` — unix only.
    PosixAio,
    /// Windows I/O Completion Ports — Windows only.
    Iocp,
    /// NIXL (Network Interface eXtension Layer) — not built in this crate (D11).
    Nixl,
    /// Auto-select the best available for the platform.
    #[default]
    Default,
}

/// `class ctp::AsyncIoFactory`.
pub struct AsyncIoFactory;

impl AsyncIoFactory {
    /// `AsyncIoFactory::Get` — build a backend, or `None` if it is unavailable
    /// on this platform/build (the C++ `default:` arm returns `nullptr`).
    ///
    /// `io_depth` sizes the worker pool here rather than a kernel queue (D2).
    pub fn get(io_depth: u32, backend: AsyncIoBackend) -> Option<Box<dyn AsyncIo>> {
        let backend = if backend == AsyncIoBackend::Default {
            Self::default_backend()
        } else {
            backend
        };
        match backend {
            // `#if CTP_ENABLE_LIBAIO` / `#if CTP_ENABLE_IO_URING` /
            // `#if CTP_ENABLE_NIXL` — never enabled for the Rust crate (D11).
            AsyncIoBackend::LinuxAio | AsyncIoBackend::IoUring | AsyncIoBackend::Nixl => None,
            // `#if !defined(_WIN32)`
            AsyncIoBackend::PosixAio => {
                if cfg!(unix) {
                    Some(Box::new(ThreadPoolAsyncIo::new(io_depth)))
                } else {
                    None
                }
            }
            // `#ifdef _WIN32`
            AsyncIoBackend::Iocp => {
                if cfg!(windows) {
                    Some(Box::new(ThreadPoolAsyncIo::new(io_depth)))
                } else {
                    None
                }
            }
            AsyncIoBackend::Default => None,
        }
    }

    /// `AsyncIoFactory::GetDefaultBackend`. NIXL/io_uring/libaio are compiled
    /// out (D11), so this is `Iocp` on Windows and `PosixAio` elsewhere.
    pub fn default_backend() -> AsyncIoBackend {
        #[cfg(windows)]
        {
            AsyncIoBackend::Iocp
        }
        #[cfg(not(windows))]
        {
            AsyncIoBackend::PosixAio
        }
    }
}

// ---------------------------------------------------------------------------
// ThreadPoolAsyncIo — the concrete backend (D1)
// ---------------------------------------------------------------------------

/// A caller-owned raw buffer travelling to a worker thread.
struct RawBuf {
    ptr: *mut u8,
    len: usize,
}

// SAFETY: `RawBuf` is only ever constructed inside `AsyncIo::write`/`read`,
// whose safety contracts oblige the caller to keep the region allocated,
// untouched and valid across threads until the op completes or `close()`
// returns. `close()` (and therefore `Drop`) blocks until every executing job is
// finished, so no worker can be holding a `RawBuf` past the point where the
// caller is allowed to invalidate it. Exactly one thread ever accesses a given
// `RawBuf`: the job it belongs to is moved into the queue and popped by a single
// worker (`Send` is what that hand-off needs; `Sync` is deliberately not
// claimed).
unsafe impl Send for RawBuf {}

/// Either a caller-owned region (C++-parity API) or an engine-owned `Vec`
/// (safe API).
enum IoBuf {
    Raw(RawBuf),
    Owned(Vec<u8>),
}

impl IoBuf {
    /// Buffer address, for the 4096-alignment test in `select_file`.
    fn addr(&self) -> usize {
        match self {
            IoBuf::Raw(r) => r.ptr as usize,
            IoBuf::Owned(v) => v.as_ptr() as usize,
        }
    }

    fn len(&self) -> usize {
        match self {
            IoBuf::Raw(r) => r.len,
            IoBuf::Owned(v) => v.len(),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum JobKind {
    Read,
    Write,
}

/// One submitted operation (the C++ `iocb` / `aiocb` / `IoOp`).
struct Job {
    token: IoToken,
    kind: JobKind,
    buf: IoBuf,
    offset: i64,
    /// Descriptor chosen by `select_file` at submit time. Holding an `Arc`
    /// keeps it open for the duration of the op even if `close()` races.
    file: Arc<File>,
}

/// A harvested completion.
struct CompletedOp {
    result: IoResult,
    /// `Some` only for [`ThreadPoolAsyncIo::read_owned`] ops.
    buf: Option<Vec<u8>>,
}

struct State {
    /// `regular_fd_`
    regular: Option<Arc<File>>,
    /// `direct_fd_` (D5)
    direct: Option<Arc<File>>,
    queue: VecDeque<Job>,
    /// `in_flight_`
    in_flight: HashSet<IoToken>,
    /// `completed_`
    completed: HashMap<IoToken, CompletedOp>,
    /// Jobs popped by a worker but not yet finished; `close()` waits on this.
    active: usize,
    shutdown: bool,
}

struct Shared {
    state: Mutex<State>,
    /// Signals "a job was queued" / "shutdown".
    submit_cv: Condvar,
    /// Signals "a job finished" (drives both `wait_complete` and `close`).
    done_cv: Condvar,
}

impl Shared {
    /// A C++ `std::mutex` cannot be poisoned; recover the guard instead (D9).
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// Thread-pool [`AsyncIo`] backend: the portable stand-in for `IocpAsyncIO`,
/// `PosixAsyncIO`, `LinuxAioAsyncIO` and `IoUringAsyncIO` (D1).
///
/// Beyond the C++ surface it offers an *owned-buffer* API
/// ([`write_owned`](Self::write_owned) / [`read_owned`](Self::read_owned) /
/// [`try_take_read`](Self::try_take_read)) that needs no `unsafe` from callers,
/// and a blocking [`wait_complete`](Self::wait_complete).
pub struct ThreadPoolAsyncIo {
    shared: Arc<Shared>,
    /// `next_token_`, starting at 1 so 0 stays [`INVALID_IO_TOKEN`].
    next_token: AtomicU64,
    workers: Vec<JoinHandle<()>>,
}

impl ThreadPoolAsyncIo {
    /// Construct with the C++ `io_depth` parameter; see D2 for how it is used.
    pub fn new(io_depth: u32) -> Self {
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                regular: None,
                direct: None,
                queue: VecDeque::new(),
                in_flight: HashSet::new(),
                completed: HashMap::new(),
                active: 0,
                shutdown: false,
            }),
            submit_cv: Condvar::new(),
            done_cv: Condvar::new(),
        });
        // io_depth == 0 must still yield a usable engine (D2).
        let n_workers = (io_depth.max(1) as usize).min(MAX_WORKERS);
        let mut workers = Vec::with_capacity(n_workers);
        for _ in 0..n_workers {
            let shared = Arc::clone(&shared);
            workers.push(std::thread::spawn(move || worker_loop(&shared)));
        }
        ThreadPoolAsyncIo { shared, next_token: AtomicU64::new(1), workers }
    }

    /// Number of worker threads actually spawned (see D2).
    pub fn worker_count(&self) -> usize {
        self.workers.len()
    }

    /// Tokens submitted but not yet consumed by a completing poll (`in_flight_`).
    pub fn in_flight_count(&self) -> usize {
        self.shared.lock().in_flight.len()
    }

    /// `SelectFd`/`SelectHandle`: the direct descriptor iff it exists and both
    /// buffer address and size are [`DIRECT_IO_ALIGN`]-aligned; else the regular
    /// one. `None` means no descriptor at all (not open, or closed).
    fn select_file(st: &State, addr: usize, size: usize) -> Option<Arc<File>> {
        // C++ spells this `% 4096 == 0`; `is_multiple_of` is the same test.
        if st.direct.is_some()
            && addr.is_multiple_of(DIRECT_IO_ALIGN)
            && size.is_multiple_of(DIRECT_IO_ALIGN)
        {
            return st.direct.clone();
        }
        st.regular.clone()
    }

    /// `SubmitIO`.
    fn submit(&self, kind: JobKind, buf: IoBuf, offset: i64) -> IoToken {
        let addr = buf.addr();
        let size = buf.len();
        let mut st = self.shared.lock();
        // `SelectHandle() == INVALID_HANDLE_VALUE` → kInvalidIoToken (D7).
        let file = match Self::select_file(&st, addr, size) {
            Some(f) => f,
            None => return INVALID_IO_TOKEN,
        };
        let token = self.next_token.fetch_add(1, Ordering::Relaxed);
        st.in_flight.insert(token);
        st.queue.push_back(Job { token, kind, buf, offset, file });
        drop(st);
        self.shared.submit_cv.notify_one();
        token
    }

    /// Safe counterpart of [`AsyncIo::write`]: the engine takes ownership of
    /// `data` for the duration of the op and drops it on completion. Poll with
    /// [`AsyncIo::is_complete`] or [`wait_complete`](Self::wait_complete).
    pub fn write_owned(&self, data: Vec<u8>, offset: i64) -> IoToken {
        self.submit(JobKind::Write, IoBuf::Owned(data), offset)
    }

    /// Safe counterpart of [`AsyncIo::read`]: the engine allocates a
    /// `size`-byte buffer, reads into it, and hands it back from
    /// [`try_take_read`](Self::try_take_read).
    pub fn read_owned(&self, size: usize, offset: i64) -> IoToken {
        self.submit(JobKind::Read, IoBuf::Owned(vec![0u8; size]), offset)
    }

    /// Non-blocking poll for a [`read_owned`](Self::read_owned) token, returning
    /// the result and the engine-allocated buffer. Consumes the completion just
    /// like [`AsyncIo::is_complete`].
    ///
    /// The buffer keeps its submitted length; only the first
    /// `result.bytes_transferred` bytes are meaningful (C++ parity: the caller's
    /// buffer is likewise not resized by a short read). Returns `None` while the
    /// op is in flight, for an unknown token, or for a token that was not a
    /// `read_owned` (use [`AsyncIo::is_complete`] for those).
    pub fn try_take_read(&self, token: IoToken) -> Option<(IoResult, Vec<u8>)> {
        let mut st = self.shared.lock();
        // Peek first: only consume the completion if it carries a read buffer,
        // so a mistargeted call cannot silently swallow someone's result.
        st.completed.get(&token)?.buf.as_ref()?;
        let op = st.completed.remove(&token)?;
        st.in_flight.remove(&token);
        // `op.buf` was just checked to be Some.
        op.buf.map(|b| (op.result, b))
    }

    /// Block until `token` completes, up to `timeout_ms` **milliseconds**
    /// (MIGRATION.md: timings are ms). Consumes the completion and returns its
    /// result, or `None` on timeout / unknown token / a token dropped by
    /// [`close`](AsyncIo::close).
    ///
    /// An addition: the C++ API has no blocking wait, only `IsComplete` polling
    /// and (on libaio/io_uring) the eventfd this port does not carry (D8). A
    /// `read_owned` buffer reported through this call is dropped — use
    /// [`try_take_read`](Self::try_take_read) for those.
    pub fn wait_complete(&self, token: IoToken, timeout_ms: u64) -> Option<IoResult> {
        let deadline = Instant::now().checked_add(Duration::from_millis(timeout_ms));
        let mut st = self.shared.lock();
        loop {
            if let Some(op) = st.completed.remove(&token) {
                st.in_flight.remove(&token);
                return Some(op.result);
            }
            // Not completed and not in flight → nothing will ever arrive.
            if !st.in_flight.contains(&token) {
                return None;
            }
            let remaining = match deadline {
                // Saturating: an absurd timeout_ms overflows Instant; treat it
                // as "wait indefinitely" rather than as an instant timeout.
                None => Duration::from_secs(3600),
                Some(d) => match d.checked_duration_since(Instant::now()) {
                    Some(r) if !r.is_zero() => r,
                    _ => return None,
                },
            };
            let (guard, _timed_out) = self
                .shared
                .done_cv
                .wait_timeout(st, remaining)
                .unwrap_or_else(|e| e.into_inner());
            st = guard;
        }
    }
}

impl AsyncIo for ThreadPoolAsyncIo {
    fn open(&self, path: &str, flags: i32, mode: ModeT) -> bool {
        let mut st = self.shared.lock();
        // "Strip O_DIRECT from caller flags — we manage it internally".
        let base_flags = flags & !O_DIRECT;
        let regular = match open_file(path, base_flags, mode, false) {
            Ok(f) => f,
            Err(_) => return false,
        };
        // "If O_DIRECT fails, we'll just use regular_fd_ for everything".
        let direct = open_file(path, base_flags, mode, true).ok();
        st.regular = Some(Arc::new(regular));
        st.direct = direct.map(Arc::new);
        true
    }

    fn get_file_size(&self) -> isize {
        let st = self.shared.lock();
        // `regular_fd_ >= 0 ? regular_fd_ : direct_fd_`
        let file = match st.regular.as_ref().or(st.direct.as_ref()) {
            Some(f) => Arc::clone(f),
            None => return -1,
        };
        drop(st);
        match file.metadata() {
            Ok(m) => m.len() as isize,
            Err(_) => -1,
        }
    }

    fn truncate(&self, size: usize) -> bool {
        let st = self.shared.lock();
        let file = match st.regular.as_ref().or(st.direct.as_ref()) {
            Some(f) => Arc::clone(f),
            None => return false,
        };
        drop(st);
        file.set_len(size as u64).is_ok()
    }

    unsafe fn write(&self, buffer: *mut u8, size: usize, offset: i64) -> IoToken {
        self.submit(JobKind::Write, IoBuf::Raw(RawBuf { ptr: buffer, len: size }), offset)
    }

    unsafe fn read(&self, buffer: *mut u8, size: usize, offset: i64) -> IoToken {
        self.submit(JobKind::Read, IoBuf::Raw(RawBuf { ptr: buffer, len: size }), offset)
    }

    fn is_complete(&self, token: IoToken, result: &mut IoResult) -> bool {
        let mut st = self.shared.lock();
        // The C++ check → HarvestCompletions() → re-check collapses to one
        // lookup: workers publish completions themselves (D3).
        match st.completed.remove(&token) {
            Some(op) => {
                *result = op.result;
                st.in_flight.remove(&token);
                true
            }
            None => false,
        }
    }

    fn close(&self) {
        let mut st = self.shared.lock();
        // Queued-but-unstarted jobs never ran, so their buffers were never
        // touched; drop them. Their tokens simply never complete, as in C++.
        st.queue.clear();
        // Unlike CancelIoEx/io_destroy, wait for jobs already touching caller
        // buffers to finish before releasing anything (D4).
        while st.active > 0 {
            st = self.shared.done_cv.wait(st).unwrap_or_else(|e| e.into_inner());
        }
        st.regular = None;
        st.direct = None;
        st.in_flight.clear();
        st.completed.clear();
    }

    fn get_event_fd(&self) -> i32 {
        -1
    }
}

impl Drop for ThreadPoolAsyncIo {
    /// `~IocpAsyncIO()`/`~LinuxAioAsyncIO()` call `Close()`; this also retires
    /// the workers.
    fn drop(&mut self) {
        self.close();
        {
            let mut st = self.shared.lock();
            st.shutdown = true;
        }
        self.shared.submit_cv.notify_all();
        for h in self.workers.drain(..) {
            let _ = h.join();
        }
    }
}

/// `open(path, base_flags, mode)`, mapped onto `OpenOptions` (D12).
fn open_file(path: &str, base_flags: i32, mode: ModeT, direct: bool) -> io::Result<File> {
    let mut opts = OpenOptions::new();
    match base_flags & O_ACCMODE {
        O_RDONLY => {
            opts.read(true);
        }
        O_WRONLY => {
            opts.write(true);
        }
        O_RDWR => {
            opts.read(true).write(true);
        }
        // open(2) rejects an access mode of 3 with EINVAL.
        _ => return Err(io::Error::from_raw_os_error(errno::EINVAL)),
    }
    if base_flags & O_APPEND != 0 {
        opts.append(true);
    }
    if base_flags & O_CREAT != 0 {
        opts.create(true);
    }
    if base_flags & O_TRUNC != 0 {
        opts.truncate(true);
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(mode);
        // `open(path, base_flags | O_DIRECT, mode)`; O_DIRECT is 0 off Linux,
        // so the direct handle degrades to a second regular one (D5).
        if direct && O_DIRECT != 0 {
            opts.custom_flags(O_DIRECT);
        }
    }
    #[cfg(windows)]
    {
        // `(void)mode;` — POSIX mode bits aren't used on Windows, and the
        // direct handle is a second plain handle here (D5).
        let _ = mode;
        let _ = direct;
    }
    opts.open(path)
}

/// Pop the next job, or `None` once shut down. Increments `active` for the
/// caller.
fn next_job(shared: &Shared) -> Option<Job> {
    let mut st = shared.lock();
    loop {
        if st.shutdown {
            return None;
        }
        if let Some(job) = st.queue.pop_front() {
            st.active += 1;
            return Some(job);
        }
        st = shared.submit_cv.wait(st).unwrap_or_else(|e| e.into_inner());
    }
}

fn worker_loop(shared: &Shared) {
    while let Some(job) = next_job(shared) {
        let token = job.token;
        let (result, buf) = execute_job(job);
        let mut st = shared.lock();
        st.active -= 1;
        st.completed.insert(token, CompletedOp { result, buf });
        drop(st);
        // Wakes both `wait_complete` and a `close()` waiting to quiesce.
        shared.done_cv.notify_all();
    }
}

/// Perform one operation. Runs with no lock held.
fn execute_job(job: Job) -> (IoResult, Option<Vec<u8>>) {
    let Job { kind, mut buf, offset, file, .. } = job;
    // A zero-length raw buffer may legitimately be null; from_raw_parts_mut
    // demands non-null even for len 0, so hand out this local instead.
    let mut empty: [u8; 0] = [];

    let res: io::Result<usize> = if offset < 0 {
        // off_t is signed but pread/pwrite reject a negative offset; complete
        // in place with EINVAL rather than failing the submit (D7).
        Err(io::Error::from_raw_os_error(errno::EINVAL))
    } else {
        let off = offset as u64;
        match kind {
            JobKind::Read => {
                let slice: &mut [u8] = match &mut buf {
                    IoBuf::Raw(r) => {
                        if r.len == 0 || r.ptr.is_null() {
                            &mut empty
                        } else {
                            // SAFETY: `AsyncIo::read`'s contract guarantees
                            // `ptr` addresses `len` writable bytes that stay
                            // valid until this op completes; `close()` cannot
                            // return while this job is `active`, so the region
                            // is still live. u8 has alignment 1, and the
                            // null/zero-length cases are handled above, so the
                            // pointer is non-null and aligned. Nothing else
                            // aliases the slice: the caller is contractually
                            // hands-off and this job is owned by this thread.
                            unsafe { std::slice::from_raw_parts_mut(r.ptr, r.len) }
                        }
                    }
                    IoBuf::Owned(v) => v.as_mut_slice(),
                };
                pread(&file, slice, off)
            }
            JobKind::Write => {
                let slice: &[u8] = match &buf {
                    IoBuf::Raw(r) => {
                        if r.len == 0 || r.ptr.is_null() {
                            &empty
                        } else {
                            // SAFETY: as above, but read-only — `AsyncIo::write`
                            // guarantees `len` initialized bytes at `ptr`, live
                            // for the duration of this job and not concurrently
                            // mutated by the caller.
                            unsafe { std::slice::from_raw_parts(r.ptr as *const u8, r.len) }
                        }
                    }
                    IoBuf::Owned(v) => v.as_slice(),
                };
                pwrite(&file, slice, off)
            }
        }
    };

    // libaio: `res < 0 ? {-1, -res} : {res, 0}` — same shape, raw OS code (D6).
    let result = match res {
        Ok(n) => IoResult { bytes_transferred: n as isize, error_code: 0 },
        Err(e) => IoResult {
            bytes_transferred: -1,
            error_code: e.raw_os_error().unwrap_or(FALLBACK_OS_ERROR),
        },
    };
    let out = match (kind, buf) {
        (JobKind::Read, IoBuf::Owned(v)) => Some(v),
        _ => None,
    };
    (result, out)
}

/// One positional read; short reads (EOF) are reported, never retried — AIO
/// semantics.
#[cfg(unix)]
fn pread(file: &File, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    use std::os::unix::fs::FileExt;
    file.read_at(buf, offset)
}

#[cfg(windows)]
fn pread(file: &File, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    use std::os::windows::fs::FileExt;
    file.seek_read(buf, offset)
}

/// One positional write; short writes are reported, never retried.
#[cfg(unix)]
fn pwrite(file: &File, buf: &[u8], offset: u64) -> io::Result<usize> {
    use std::os::unix::fs::FileExt;
    file.write_at(buf, offset)
}

#[cfg(windows)]
fn pwrite(file: &File, buf: &[u8], offset: u64) -> io::Result<usize> {
    use std::os::windows::fs::FileExt;
    file.seek_write(buf, offset)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    const TIMEOUT_MS: u64 = 30_000;

    /// A unique temp path that removes itself. Declare it *before* the engine so
    /// the engine (and its open handles) drop first.
    struct TempPath(String);

    impl TempPath {
        fn new(tag: &str) -> Self {
            static COUNTER: AtomicU32 = AtomicU32::new(0);
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0);
            let mut p = std::env::temp_dir();
            p.push(format!("ctp_io_{}_{}_{}_{}.bin", tag, std::process::id(), n, nanos));
            TempPath(p.to_string_lossy().into_owned())
        }

        fn as_str(&self) -> &str {
            &self.0
        }
    }

    impl Drop for TempPath {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn new_open(tag: &str, depth: u32) -> (TempPath, ThreadPoolAsyncIo) {
        let path = TempPath::new(tag);
        let io = ThreadPoolAsyncIo::new(depth);
        assert!(io.open(path.as_str(), O_RDWR | O_CREAT | O_TRUNC, 0o644), "open failed");
        (path, io)
    }

    /// Poll `is_complete` to completion, as a C++ caller would.
    fn poll_to_completion(io: &dyn AsyncIo, token: IoToken) -> IoResult {
        let deadline = Instant::now() + Duration::from_millis(TIMEOUT_MS);
        let mut result = IoResult::new();
        while Instant::now() < deadline {
            if io.is_complete(token, &mut result) {
                return result;
            }
            std::thread::yield_now();
        }
        panic!("token {token} never completed");
    }

    // -- io_error.h ---------------------------------------------------------

    #[test]
    fn classify_errno_zero_is_ok() {
        assert_eq!(classify_errno(0), IoError::Ok);
    }

    #[test]
    fn classify_errno_categories() {
        assert_eq!(classify_errno(errno::EIO), IoError::DeviceFault);
        assert_eq!(classify_errno(errno::ENODEV), IoError::Disconnected);
        assert_eq!(classify_errno(errno::ENXIO), IoError::Disconnected);
        assert_eq!(classify_errno(errno::ENOENT), IoError::Disconnected);
        assert_eq!(classify_errno(errno::ENOSPC), IoError::NoSpace);
        assert_eq!(classify_errno(errno::EFBIG), IoError::NoSpace);
        assert_eq!(classify_errno(errno::EACCES), IoError::Permission);
        assert_eq!(classify_errno(errno::EPERM), IoError::Permission);
        assert_eq!(classify_errno(errno::EROFS), IoError::Permission);
        assert_eq!(classify_errno(errno::ETIMEDOUT), IoError::Timeout);
        assert_eq!(classify_errno(errno::EAGAIN), IoError::Transient);
        assert_eq!(classify_errno(errno::EWOULDBLOCK), IoError::Transient);
        assert_eq!(classify_errno(errno::EINTR), IoError::Transient);
        assert_eq!(classify_errno(errno::EBUSY), IoError::Transient);
        assert_eq!(classify_errno(errno::EINVAL), IoError::Invalid);
        assert_eq!(classify_errno(errno::EBADF), IoError::Invalid);
        assert_eq!(classify_errno(errno::EFAULT), IoError::Invalid);
    }

    #[cfg(unix)]
    #[test]
    fn classify_errno_edquot_is_nospace() {
        assert_eq!(classify_errno(libc::EDQUOT), IoError::NoSpace);
    }

    #[test]
    fn classify_errno_unknown_and_negative() {
        assert_eq!(classify_errno(999_999), IoError::Unknown);
        assert_eq!(classify_errno(-12345), IoError::Unknown);
        assert_eq!(classify_errno(i32::MAX), IoError::Unknown);
        assert_eq!(classify_errno(i32::MIN), IoError::Unknown);
    }

    #[test]
    fn classify_win_error_table() {
        assert_eq!(classify_win_error(0), IoError::Ok);
        assert_eq!(classify_win_error(21), IoError::Transient);
        assert_eq!(classify_win_error(170), IoError::Transient);
        assert_eq!(classify_win_error(23), IoError::DeviceFault);
        assert_eq!(classify_win_error(1117), IoError::DeviceFault);
        assert_eq!(classify_win_error(2), IoError::Disconnected);
        assert_eq!(classify_win_error(3), IoError::Disconnected);
        assert_eq!(classify_win_error(55), IoError::Disconnected);
        assert_eq!(classify_win_error(1167), IoError::Disconnected);
        assert_eq!(classify_win_error(39), IoError::NoSpace);
        assert_eq!(classify_win_error(112), IoError::NoSpace);
        assert_eq!(classify_win_error(5), IoError::Permission);
        assert_eq!(classify_win_error(19), IoError::Permission);
        assert_eq!(classify_win_error(1460), IoError::Timeout);
        assert_eq!(classify_win_error(6), IoError::Invalid);
        assert_eq!(classify_win_error(87), IoError::Invalid);
        assert_eq!(classify_win_error(4242), IoError::Unknown);
        assert_eq!(classify_win_error(u32::MAX), IoError::Unknown);
    }

    #[test]
    fn error_names_and_predicates() {
        assert_eq!(io_error_name(IoError::Ok), "Ok");
        assert_eq!(io_error_name(IoError::Transient), "Transient");
        assert_eq!(io_error_name(IoError::DeviceFault), "DeviceFault");
        assert_eq!(io_error_name(IoError::Disconnected), "Disconnected");
        assert_eq!(io_error_name(IoError::NoSpace), "NoSpace");
        assert_eq!(io_error_name(IoError::Permission), "Permission");
        assert_eq!(io_error_name(IoError::Invalid), "Invalid");
        assert_eq!(io_error_name(IoError::Timeout), "Timeout");
        assert_eq!(io_error_name(IoError::Unknown), "Unknown");
        assert_eq!(IoError::NoSpace.to_string(), "NoSpace");

        assert!(is_transient(IoError::Transient));
        assert!(!is_transient(IoError::Ok));
        assert!(!is_transient(IoError::DeviceFault));

        assert!(is_fatal_device(IoError::DeviceFault));
        assert!(is_fatal_device(IoError::Disconnected));
        assert!(!is_fatal_device(IoError::Transient));
        assert!(!is_fatal_device(IoError::Ok));
        assert!(!is_fatal_device(IoError::NoSpace));
    }

    #[test]
    fn io_error_discriminants_match_cpp() {
        assert_eq!(IoError::Ok as u32, 0);
        assert_eq!(IoError::Transient as u32, 1);
        assert_eq!(IoError::DeviceFault as u32, 2);
        assert_eq!(IoError::Disconnected as u32, 3);
        assert_eq!(IoError::NoSpace as u32, 4);
        assert_eq!(IoError::Permission as u32, 5);
        assert_eq!(IoError::Invalid as u32, 6);
        assert_eq!(IoError::Timeout as u32, 7);
        assert_eq!(IoError::Unknown as u32, 8);
    }

    // -- factory ------------------------------------------------------------

    #[test]
    fn factory_default_backend_is_platform_native() {
        let expected =
            if cfg!(windows) { AsyncIoBackend::Iocp } else { AsyncIoBackend::PosixAio };
        assert_eq!(AsyncIoFactory::default_backend(), expected);
        assert_eq!(AsyncIoBackend::default(), AsyncIoBackend::Default);
    }

    #[test]
    fn factory_builds_default_and_native_backend() {
        assert!(AsyncIoFactory::get(4, AsyncIoBackend::Default).is_some());
        let native = AsyncIoFactory::default_backend();
        assert!(AsyncIoFactory::get(4, native).is_some());
    }

    #[test]
    fn factory_returns_none_for_backends_not_built() {
        // Mirrors the C++ `default:` arm when CTP_ENABLE_* are off (D11).
        assert!(AsyncIoFactory::get(4, AsyncIoBackend::LinuxAio).is_none());
        assert!(AsyncIoFactory::get(4, AsyncIoBackend::IoUring).is_none());
        assert!(AsyncIoFactory::get(4, AsyncIoBackend::Nixl).is_none());
        // Platform gating: each is compiled out on the other OS.
        if cfg!(windows) {
            assert!(AsyncIoFactory::get(4, AsyncIoBackend::PosixAio).is_none());
        } else {
            assert!(AsyncIoFactory::get(4, AsyncIoBackend::Iocp).is_none());
        }
    }

    #[test]
    fn factory_backend_is_usable_end_to_end() {
        let path = TempPath::new("factory");
        let io = AsyncIoFactory::get(2, AsyncIoBackend::Default).expect("no backend");
        assert!(io.open(path.as_str(), O_RDWR | O_CREAT | O_TRUNC, 0o644));
        let mut data = *b"factory-made";
        // SAFETY: `data` outlives the poll loop below, which runs to completion
        // before `data` is dropped, and is not touched meanwhile.
        let t = unsafe { io.write(data.as_mut_ptr(), data.len(), 0) };
        assert_ne!(t, INVALID_IO_TOKEN);
        let r = poll_to_completion(io.as_ref(), t);
        assert_eq!(r.error_code, 0);
        assert_eq!(r.bytes_transferred, data.len() as isize);
        assert_eq!(io.get_event_fd(), -1);
        drop(io);
    }

    // -- worker sizing (D2) -------------------------------------------------

    #[test]
    fn io_depth_zero_still_yields_one_worker() {
        let io = ThreadPoolAsyncIo::new(0);
        assert_eq!(io.worker_count(), 1);
    }

    #[test]
    fn io_depth_is_clamped_to_max_workers() {
        assert_eq!(ThreadPoolAsyncIo::new(1).worker_count(), 1);
        assert_eq!(ThreadPoolAsyncIo::new(4).worker_count(), 4);
        assert_eq!(ThreadPoolAsyncIo::new(u32::MAX).worker_count(), MAX_WORKERS);
    }

    // -- tokens -------------------------------------------------------------

    #[test]
    fn tokens_start_at_one_and_increase() {
        let (_p, io) = new_open("tokens", 2);
        let a = io.write_owned(vec![1u8; 4], 0);
        let b = io.write_owned(vec![2u8; 4], 4);
        assert_eq!(a, 1);
        assert_eq!(b, 2);
        assert_ne!(a, INVALID_IO_TOKEN);
        assert!(io.wait_complete(a, TIMEOUT_MS).is_some());
        assert!(io.wait_complete(b, TIMEOUT_MS).is_some());
    }

    #[test]
    fn submit_without_open_is_invalid_token() {
        // `SelectHandle()` finds no handle → kInvalidIoToken (D7).
        let io = ThreadPoolAsyncIo::new(2);
        assert_eq!(io.write_owned(vec![0u8; 8], 0), INVALID_IO_TOKEN);
        assert_eq!(io.read_owned(8, 0), INVALID_IO_TOKEN);
        let mut buf = [0u8; 8];
        // SAFETY: the submit is rejected before the pointer is ever read, and
        // `buf` outlives the call regardless.
        let t = unsafe { io.read(buf.as_mut_ptr(), buf.len(), 0) };
        assert_eq!(t, INVALID_IO_TOKEN);
    }

    #[test]
    fn is_complete_on_unknown_token_is_false() {
        let (_p, io) = new_open("unknown", 2);
        let mut r = IoResult::new();
        assert!(!io.is_complete(INVALID_IO_TOKEN, &mut r));
        assert!(!io.is_complete(4242, &mut r));
        // The out-param is untouched when the poll says "not complete".
        assert_eq!(r, IoResult::new());
        assert!(io.wait_complete(4242, 0).is_none());
    }

    #[test]
    fn completion_is_consumed_by_the_poll_that_reports_it() {
        let (_p, io) = new_open("consume", 2);
        let t = io.write_owned(vec![7u8; 16], 0);
        let r = poll_to_completion(&io, t);
        assert_eq!(r.bytes_transferred, 16);
        // Second poll: the completion was erased, exactly as C++ does.
        let mut again = IoResult::new();
        assert!(!io.is_complete(t, &mut again));
        assert_eq!(io.in_flight_count(), 0);
    }

    // -- data path ----------------------------------------------------------

    #[test]
    fn write_then_read_roundtrip() {
        let (_p, io) = new_open("roundtrip", 4);
        let data: Vec<u8> = (0..=255u8).collect();
        let n = data.len();
        let w = io.write_owned(data.clone(), 0);
        let wr = io.wait_complete(w, TIMEOUT_MS).expect("write timed out");
        assert_eq!(wr.error_code, 0);
        assert_eq!(wr.bytes_transferred, n as isize);

        let r = io.read_owned(n, 0);
        let (rr, buf) = {
            let res = io.wait_complete_then_take(r);
            res.expect("read timed out")
        };
        assert_eq!(rr.error_code, 0);
        assert_eq!(rr.bytes_transferred, n as isize);
        assert_eq!(buf, data);
        assert_eq!(io.get_file_size(), n as isize);
    }

    #[test]
    fn positional_write_and_read_at_offset() {
        let (_p, io) = new_open("offsets", 4);
        let t = io.write_owned(b"world".to_vec(), 5);
        assert_eq!(io.wait_complete(t, TIMEOUT_MS).unwrap().bytes_transferred, 5);
        let t = io.write_owned(b"hello".to_vec(), 0);
        assert_eq!(io.wait_complete(t, TIMEOUT_MS).unwrap().bytes_transferred, 5);
        assert_eq!(io.get_file_size(), 10);

        let t = io.read_owned(5, 5);
        let (r, buf) = io.wait_complete_then_take(t).unwrap();
        assert_eq!(r.bytes_transferred, 5);
        assert_eq!(&buf, b"world");
    }

    #[test]
    fn raw_pointer_api_roundtrip() {
        let (_p, io) = new_open("rawptr", 2);
        let mut out = *b"raw-buffer-payload";
        // SAFETY: `out` lives until the end of this test and the poll below
        // runs to completion before it is dropped; nothing else touches it
        // while the op is in flight.
        let t = unsafe { io.write(out.as_mut_ptr(), out.len(), 0) };
        let r = poll_to_completion(&io, t);
        assert_eq!(r.bytes_transferred, out.len() as isize);

        let mut back = vec![0u8; out.len()];
        // SAFETY: as above — `back` outlives the completed poll.
        let t = unsafe { io.read(back.as_mut_ptr(), back.len(), 0) };
        let r = poll_to_completion(&io, t);
        assert_eq!(r.bytes_transferred, out.len() as isize);
        assert_eq!(back, out.to_vec());
    }

    #[test]
    fn zero_length_ops_succeed() {
        let (_p, io) = new_open("zerolen", 2);
        let t = io.write_owned(Vec::new(), 0);
        let r = io.wait_complete(t, TIMEOUT_MS).expect("zero write timed out");
        assert_eq!(r.bytes_transferred, 0);
        assert_eq!(r.error_code, 0);
        assert_eq!(io.get_file_size(), 0);

        let t = io.read_owned(0, 0);
        let (r, buf) = io.wait_complete_then_take(t).expect("zero read timed out");
        assert_eq!(r.bytes_transferred, 0);
        assert_eq!(r.error_code, 0);
        assert!(buf.is_empty());
    }

    #[test]
    fn zero_length_raw_op_with_null_pointer_is_safe() {
        let (_p, io) = new_open("nullzero", 2);
        // Contract: a null buffer is allowed exactly when size == 0.
        // SAFETY: size is 0, so the pointer is never dereferenced.
        let t = unsafe { io.write(std::ptr::null_mut(), 0, 0) };
        let r = poll_to_completion(&io, t);
        assert_eq!(r.bytes_transferred, 0);
        assert_eq!(r.error_code, 0);
    }

    #[test]
    fn read_past_eof_returns_zero_bytes() {
        let (_p, io) = new_open("eof", 2);
        let t = io.write_owned(vec![9u8; 8], 0);
        io.wait_complete(t, TIMEOUT_MS).unwrap();

        let t = io.read_owned(16, 64);
        let (r, _buf) = io.wait_complete_then_take(t).expect("eof read timed out");
        assert_eq!(r.error_code, 0);
        assert_eq!(r.bytes_transferred, 0);
    }

    #[test]
    fn short_read_at_eof_boundary() {
        let (_p, io) = new_open("shortread", 2);
        let t = io.write_owned(vec![3u8; 10], 0);
        io.wait_complete(t, TIMEOUT_MS).unwrap();

        // 8 bytes exist past offset 2; ask for 64.
        let t = io.read_owned(64, 2);
        let (r, buf) = io.wait_complete_then_take(t).expect("short read timed out");
        assert_eq!(r.error_code, 0);
        assert_eq!(r.bytes_transferred, 8);
        // The buffer keeps its submitted length; only the prefix is meaningful.
        assert_eq!(buf.len(), 64);
        assert_eq!(&buf[..8], &[3u8; 8]);
    }

    #[test]
    fn write_beyond_eof_extends_with_zeros() {
        let (_p, io) = new_open("extend", 2);
        let t = io.write_owned(vec![0xABu8; 4], 16);
        assert_eq!(io.wait_complete(t, TIMEOUT_MS).unwrap().bytes_transferred, 4);
        assert_eq!(io.get_file_size(), 20);

        let t = io.read_owned(20, 0);
        let (r, buf) = io.wait_complete_then_take(t).unwrap();
        assert_eq!(r.bytes_transferred, 20);
        assert_eq!(&buf[..16], &[0u8; 16]);
        assert_eq!(&buf[16..], &[0xABu8; 4]);
    }

    #[test]
    fn negative_offset_completes_with_einval() {
        // IOCP-style immediate failure: a live token carrying the error (D7).
        let (_p, io) = new_open("negoff", 2);
        let t = io.write_owned(vec![1u8; 4], -1);
        assert_ne!(t, INVALID_IO_TOKEN);
        let r = io.wait_complete(t, TIMEOUT_MS).expect("timed out");
        assert_eq!(r.bytes_transferred, -1);
        assert_eq!(r.error_code, errno::EINVAL);
        assert_eq!(classify_errno(r.error_code), IoError::Invalid);

        let t = io.read_owned(4, i64::MIN);
        let (r, _b) = io.wait_complete_then_take(t).expect("timed out");
        assert_eq!(r.bytes_transferred, -1);
        assert_eq!(r.error_code, errno::EINVAL);
    }

    // -- file metadata ------------------------------------------------------

    #[test]
    fn size_and_truncate_before_open_fail() {
        let io = ThreadPoolAsyncIo::new(2);
        assert_eq!(io.get_file_size(), -1);
        assert!(!io.truncate(4096));
    }

    #[test]
    fn truncate_extends_and_shrinks() {
        let (_p, io) = new_open("truncate", 2);
        assert_eq!(io.get_file_size(), 0);
        assert!(io.truncate(4096));
        assert_eq!(io.get_file_size(), 4096);
        assert!(io.truncate(7));
        assert_eq!(io.get_file_size(), 7);
        assert!(io.truncate(0));
        assert_eq!(io.get_file_size(), 0);
    }

    #[test]
    fn open_missing_file_without_create_fails() {
        let path = TempPath::new("missing");
        let io = ThreadPoolAsyncIo::new(2);
        assert!(!io.open(path.as_str(), O_RDWR, 0o644));
        assert_eq!(io.get_file_size(), -1);
    }

    #[test]
    fn open_readonly_reads_but_write_reports_error() {
        let path = TempPath::new("rdonly");
        {
            let (_p2, io) = (&path, ThreadPoolAsyncIo::new(2));
            assert!(io.open(path.as_str(), O_RDWR | O_CREAT | O_TRUNC, 0o644));
            let t = io.write_owned(b"seed".to_vec(), 0);
            io.wait_complete(t, TIMEOUT_MS).unwrap();
        }
        let io = ThreadPoolAsyncIo::new(2);
        assert!(io.open(path.as_str(), O_RDONLY, 0o644));
        let t = io.read_owned(4, 0);
        let (r, buf) = io.wait_complete_then_take(t).unwrap();
        assert_eq!(r.bytes_transferred, 4);
        assert_eq!(&buf, b"seed");

        // A write through a read-only descriptor fails; the token is still live
        // and reports the platform's error code.
        let t = io.write_owned(b"nope".to_vec(), 0);
        let r = io.wait_complete(t, TIMEOUT_MS).expect("timed out");
        assert_eq!(r.bytes_transferred, -1);
        assert_ne!(r.error_code, 0);
    }

    #[test]
    fn reopen_replaces_previous_handles() {
        let (_p1, io) = new_open("reopen1", 2);
        let t = io.write_owned(vec![1u8; 32], 0);
        io.wait_complete(t, TIMEOUT_MS).unwrap();
        assert_eq!(io.get_file_size(), 32);

        // D12: no fd leak, the old handles are dropped.
        let p2 = TempPath::new("reopen2");
        assert!(io.open(p2.as_str(), O_RDWR | O_CREAT | O_TRUNC, 0o644));
        assert_eq!(io.get_file_size(), 0);
    }

    // -- descriptor selection ----------------------------------------------

    #[test]
    fn select_file_prefers_direct_only_when_aligned() {
        let (_p, io) = new_open("select", 2);
        let st = io.shared.lock();
        // Both handles are open here (the direct open degrades to a plain
        // second handle where O_DIRECT is unavailable — D5).
        assert!(st.regular.is_some());
        let direct_open = st.direct.is_some();

        let aligned = ThreadPoolAsyncIo::select_file(&st, 4096 * 3, 4096 * 2);
        let unaligned_addr = ThreadPoolAsyncIo::select_file(&st, 4097, 4096);
        let unaligned_size = ThreadPoolAsyncIo::select_file(&st, 4096, 4095);

        if direct_open {
            let direct = st.direct.clone().unwrap();
            assert!(Arc::ptr_eq(&aligned.unwrap(), &direct), "aligned must pick direct");
        }
        let regular = st.regular.clone().unwrap();
        assert!(Arc::ptr_eq(&unaligned_addr.unwrap(), &regular));
        assert!(Arc::ptr_eq(&unaligned_size.unwrap(), &regular));
    }

    #[test]
    fn select_file_after_close_is_none() {
        let (_p, io) = new_open("selnone", 2);
        io.close();
        let st = io.shared.lock();
        assert!(ThreadPoolAsyncIo::select_file(&st, 0, 4096).is_none());
        assert!(ThreadPoolAsyncIo::select_file(&st, 1, 1).is_none());
    }

    #[test]
    fn aligned_buffer_roundtrip_through_direct_path() {
        // A 4096-aligned buffer of a 4096-multiple size takes the direct
        // descriptor; the data must still round-trip.
        let (_p, io) = new_open("direct", 2);
        let mut region: Vec<u64> = vec![0; (4096 * 2) / 8];
        for (i, v) in region.iter_mut().enumerate() {
            *v = i as u64;
        }
        let bytes = 4096 * 2;
        let base = region.as_mut_ptr() as *mut u8;
        if !(base as usize).is_multiple_of(DIRECT_IO_ALIGN) {
            // Allocation happened not to be page-aligned; the selection rule is
            // covered by `select_file_prefers_direct_only_when_aligned`.
            return;
        }
        // SAFETY: `region` owns `bytes` initialized bytes at `base`, outlives
        // both polls, and is untouched while the ops are in flight.
        let t = unsafe { io.write(base, bytes, 0) };
        let r = poll_to_completion(&io, t);
        assert_eq!(r.error_code, 0);
        assert_eq!(r.bytes_transferred, bytes as isize);

        let mut back: Vec<u64> = vec![0; (4096 * 2) / 8];
        let rbase = back.as_mut_ptr() as *mut u8;
        // SAFETY: as above.
        let t = unsafe { io.read(rbase, bytes, 0) };
        let r = poll_to_completion(&io, t);
        assert_eq!(r.error_code, 0);
        assert_eq!(r.bytes_transferred, bytes as isize);
        assert_eq!(back, region);
    }

    // -- close / drop -------------------------------------------------------

    #[test]
    fn close_is_idempotent_and_submits_fail_after() {
        let (_p, io) = new_open("close", 2);
        let t = io.write_owned(vec![1u8; 8], 0);
        io.wait_complete(t, TIMEOUT_MS).unwrap();
        io.close();
        io.close();
        io.close();
        assert_eq!(io.write_owned(vec![1u8; 8], 0), INVALID_IO_TOKEN);
        assert_eq!(io.get_file_size(), -1);
        assert!(!io.truncate(16));
        assert_eq!(io.in_flight_count(), 0);
    }

    #[test]
    fn close_forgets_in_flight_and_completed_state() {
        let (_p, io) = new_open("closeforget", 2);
        let mut tokens = Vec::new();
        for i in 0..32 {
            tokens.push(io.write_owned(vec![i as u8; 64], i * 64));
        }
        io.close();
        assert_eq!(io.in_flight_count(), 0);
        for t in tokens {
            let mut r = IoResult::new();
            assert!(!io.is_complete(t, &mut r), "close must forget every token");
            assert!(io.wait_complete(t, 0).is_none());
        }
    }

    #[test]
    fn close_quiesces_before_releasing_files() {
        // D4: after close() returns, no worker may still be executing a job.
        let (_p, io) = new_open("quiesce", 4);
        for i in 0..64 {
            io.write_owned(vec![0xEEu8; 1024], i * 1024);
        }
        io.close();
        let st = io.shared.lock();
        assert_eq!(st.active, 0);
        assert!(st.queue.is_empty());
        assert!(st.regular.is_none());
        assert!(st.direct.is_none());
    }

    #[test]
    fn reopen_after_close_works() {
        let (_p, io) = new_open("reopenclose", 2);
        io.close();
        let p2 = TempPath::new("reopenclose2");
        assert!(io.open(p2.as_str(), O_RDWR | O_CREAT | O_TRUNC, 0o644));
        let t = io.write_owned(b"again".to_vec(), 0);
        let r = io.wait_complete(t, TIMEOUT_MS).expect("timed out");
        assert_eq!(r.bytes_transferred, 5);
    }

    #[test]
    fn drop_with_ops_in_flight_does_not_hang() {
        let path = TempPath::new("dropinflight");
        {
            let io = ThreadPoolAsyncIo::new(4);
            assert!(io.open(path.as_str(), O_RDWR | O_CREAT | O_TRUNC, 0o644));
            for i in 0..128 {
                io.write_owned(vec![0x5Au8; 512], i * 512);
            }
            // Drop with submissions outstanding: close() drains, then the
            // workers are joined.
        }
        assert!(std::fs::metadata(path.as_str()).is_ok());
    }

    #[test]
    fn engine_with_no_open_drops_cleanly() {
        let io = ThreadPoolAsyncIo::new(8);
        assert_eq!(io.worker_count(), 8);
        drop(io);
    }

    // -- owned-buffer API ---------------------------------------------------

    #[test]
    fn try_take_read_returns_none_for_unknown_and_consumed_tokens() {
        let (_p, io) = new_open("takenone", 2);
        let t = io.write_owned(vec![1u8; 4], 0);
        io.wait_complete(t, TIMEOUT_MS).unwrap();
        // Already consumed, and a write never carries a buffer anyway.
        assert!(io.try_take_read(t).is_none());
        assert!(io.try_take_read(9999).is_none());
        assert!(io.try_take_read(INVALID_IO_TOKEN).is_none());
    }

    #[test]
    fn try_take_read_neither_yields_nor_consumes_a_write_completion() {
        let (_p, io) = new_open("takewrite", 2);
        let w = io.write_owned(vec![2u8; 4], 0);
        // Wait for the completion to be published *without* consuming it.
        let deadline = Instant::now() + Duration::from_millis(TIMEOUT_MS);
        while !io.shared.lock().completed.contains_key(&w) {
            assert!(Instant::now() < deadline, "write never completed");
            std::thread::yield_now();
        }
        // The completion is sitting there, but it carries no buffer.
        assert!(io.try_take_read(w).is_none());
        // ...and that peek must not have swallowed it.
        let mut r = IoResult::new();
        assert!(io.is_complete(w, &mut r), "try_take_read consumed a write completion");
        assert_eq!(r.bytes_transferred, 4);
        assert_eq!(r.error_code, 0);
    }

    #[test]
    fn try_take_read_on_an_unfinished_token_does_not_lose_it() {
        let (_p, io) = new_open("takeinflight", 1);
        let t = io.read_owned(8, 0);
        // The poll may land before or after the worker runs; either way the
        // token must still be redeemable.
        match io.try_take_read(t) {
            Some((r, buf)) => {
                assert_eq!(r.error_code, 0);
                assert_eq!(buf.len(), 8);
            }
            None => {
                let (r, buf) = io.wait_complete_then_take(t).expect("read timed out");
                assert_eq!(r.error_code, 0);
                assert_eq!(buf.len(), 8);
            }
        }
        assert_eq!(io.in_flight_count(), 0);
    }

    // -- concurrency --------------------------------------------------------

    #[test]
    fn concurrent_submits_from_many_threads_all_complete() {
        let path = TempPath::new("concurrent");
        let io = Arc::new(ThreadPoolAsyncIo::new(8));
        assert!(io.open(path.as_str(), O_RDWR | O_CREAT | O_TRUNC, 0o644));

        const THREADS: usize = 8;
        const PER_THREAD: usize = 32;
        const CHUNK: usize = 64;

        let mut handles = Vec::new();
        for t in 0..THREADS {
            let io = Arc::clone(&io);
            handles.push(std::thread::spawn(move || {
                for i in 0..PER_THREAD {
                    let idx = t * PER_THREAD + i;
                    let offset = (idx * CHUNK) as i64;
                    let token = io.write_owned(vec![idx as u8; CHUNK], offset);
                    assert_ne!(token, INVALID_IO_TOKEN);
                    let r = io.wait_complete(token, TIMEOUT_MS).expect("write timed out");
                    assert_eq!(r.error_code, 0);
                    assert_eq!(r.bytes_transferred, CHUNK as isize);

                    let token = io.read_owned(CHUNK, offset);
                    let (r, buf) = io.wait_complete_then_take(token).expect("read timed out");
                    assert_eq!(r.bytes_transferred, CHUNK as isize);
                    assert_eq!(buf, vec![idx as u8; CHUNK]);
                }
            }));
        }
        for h in handles {
            h.join().expect("worker thread panicked");
        }
        assert_eq!(io.in_flight_count(), 0);
        assert_eq!(io.get_file_size(), (THREADS * PER_THREAD * CHUNK) as isize);
    }

    #[test]
    fn tokens_are_unique_under_concurrent_submission() {
        let path = TempPath::new("uniqtokens");
        let io = Arc::new(ThreadPoolAsyncIo::new(4));
        assert!(io.open(path.as_str(), O_RDWR | O_CREAT | O_TRUNC, 0o644));

        const THREADS: usize = 4;
        const PER_THREAD: usize = 64;
        let mut handles = Vec::new();
        for t in 0..THREADS {
            let io = Arc::clone(&io);
            handles.push(std::thread::spawn(move || {
                let mut mine = Vec::with_capacity(PER_THREAD);
                for i in 0..PER_THREAD {
                    mine.push(io.write_owned(vec![0u8; 8], ((t * PER_THREAD + i) * 8) as i64));
                }
                mine
            }));
        }
        let mut all: Vec<IoToken> = Vec::new();
        for h in handles {
            all.extend(h.join().unwrap());
        }
        for t in &all {
            assert_ne!(*t, INVALID_IO_TOKEN);
            assert!(io.wait_complete(*t, TIMEOUT_MS).is_some());
        }
        let unique: HashSet<IoToken> = all.iter().copied().collect();
        assert_eq!(unique.len(), THREADS * PER_THREAD);
    }

    #[test]
    fn wait_complete_zero_timeout_is_a_poll() {
        let (_p, io) = new_open("zerotimeout", 2);
        // A zero-ms wait must not block; it may or may not observe completion.
        let t = io.write_owned(vec![1u8; 4], 0);
        let start = Instant::now();
        let early = io.wait_complete(t, 0);
        assert!(start.elapsed() < Duration::from_secs(2), "a 0 ms wait must not block");
        // Either way the token must settle: if the poll was too early, the
        // completion is still there to collect.
        if early.is_none() {
            let r = io.wait_complete(t, TIMEOUT_MS).expect("timed out");
            assert_eq!(r.bytes_transferred, 4);
        }
    }

    #[test]
    fn engine_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ThreadPoolAsyncIo>();
        assert_send_sync::<IoResult>();
        assert_send_sync::<IoError>();
    }

    // -- test helper --------------------------------------------------------

    impl ThreadPoolAsyncIo {
        /// Blocking [`try_take_read`](Self::try_take_read), for tests.
        fn wait_complete_then_take(&self, token: IoToken) -> Option<(IoResult, Vec<u8>)> {
            let deadline = Instant::now() + Duration::from_millis(TIMEOUT_MS);
            loop {
                if let Some(v) = self.try_take_read(token) {
                    return Some(v);
                }
                if Instant::now() >= deadline {
                    return None;
                }
                if !self.shared.lock().in_flight.contains(&token) {
                    return None;
                }
                std::thread::yield_now();
            }
        }
    }
}
