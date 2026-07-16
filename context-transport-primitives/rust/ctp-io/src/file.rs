// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).

//! Synchronous file abstraction for `ctp-io`.
//!
//! # Provenance: what this ports
//!
//! `include/clio_ctp/io/` contains **no standalone synchronous file class**.
//! It defines the `AsyncIO` interface (`async_io.h`), its backends
//! (`libaio_io.h`, `posix_aio.h`, `iouring_io.h`, `iocp_io.h`, `nixl_io.h`),
//! the backend factory, and the errno classifier (`io_error.h`). The
//! synchronous file surface is *implicit*: it is the set of blocking
//! operations every backend performs identically on its raw descriptor —
//! `open(path, flags, mode)`, `fstat().st_size`, `ftruncate`, `close`, and the
//! positioned `pread`/`pwrite` pair that `io_prep_pread`/`io_prep_pwrite`
//! (`libaio_io.h`) and the IOCP `OVERLAPPED` offset express asynchronously.
//! This module gives that implicit surface an explicit, RAII-owned Rust type,
//! preserving each operation's C++ semantics (including the `-1` / `false`
//! sentinels, mirrored below). Where the C++ has no counterpart at all
//! (`exists`), that is called out as an addition.
//!
//! The nearest C++ *type* is `ctp::File` (`introspect/system_info.h`) — a
//! `union { int posix_fd_; void *windows_fd_; }`, a raw non-owning handle with
//! no methods. [`File`] is the owning counterpart.
//!
//! # C++ → Rust name mapping
//!
//! | C++ (`clio_ctp`) | Rust (this module) | Notes |
//! |---|---|---|
//! | `union ctp::File { int posix_fd_; void *windows_fd_; }` (`introspect/system_info.h`) | [`File`] | Rust owns the handle (RAII); no raw-fd union is exposed. |
//! | `AsyncIO::Open(const std::string&, int flags, mode_t mode)` → `bool` | [`File::open`] → `io::Result<File>` | Same flag/mode contract; error detail replaces `false`. |
//! | `open(2)` flag block in `async_io.h` (`O_RDONLY` … `O_TRUNC`) | [`flags`] module | Values are per-platform; see divergence 1. |
//! | `O_DIRECT` fallback (`#define O_DIRECT 0`, `posix_aio.h`) | [`flags::O_DIRECT`] | 0 off Linux, exactly as the C++ falls back. |
//! | `mode_t` (`typedef int mode_t` under `_WIN32`, `async_io.h`) | `mode: u32` param of [`File::open`] | Honored on unix, ignored on Windows (divergence 5). |
//! | `AsyncIO::Close()` → `void` | [`File::close`] + `Drop` | Idempotent, infallible; matches the `fd = -1` guard. |
//! | `AsyncIO::GetFileSize() const` → `ssize_t` (`-1` on error) | [`File::size`] → `io::Result<u64>` | Sentinel-faithful form: [`File::get_file_size`] → `i64`. |
//! | `AsyncIO::Truncate(size_t)` → `bool` | [`File::truncate`] → `io::Result<()>` | `ftruncate` / `SetEndOfFile`. |
//! | `io_prep_pread(iocb, fd, buf, size, offset)` (`libaio_io.h`) | [`File::pread`] | Single-shot; short reads are not errors. |
//! | `io_prep_pwrite(iocb, fd, buf, size, offset)` (`libaio_io.h`) | [`File::pwrite`] | Single-shot; short writes are not errors. |
//! | *(no C++ counterpart)* | [`File::read`] / [`File::write`] | Cursor-relative I/O; the C++ backends are offset-only. |
//! | *(no C++ counterpart)* | [`File::pread_exact`] / [`File::pwrite_all`] | Looping convenience wrappers. |
//! | *(none in `io/`; cf. `SystemInfo::RemoveFile`)* | [`File::exists`] | Addition requested by the port task (divergence 8). |
//! | `IoError` / `ClassifyErrno` (`io_error.h`) | *not ported here* | Belongs to `async_io.rs`; see divergence 9. |
//!
//! # Semantic divergences from the C++
//!
//! 1. **`O_*` flag values are platform-specific — the C++ comment claiming a
//!    "Cross-platform: identical bit pattern" is wrong.** `async_io.h`
//!    hand-defines `O_APPEND = 0x0008` / `O_CREAT = 0x0100` / `O_TRUNC = 0x0200`
//!    under `_WIN32` (the MSVC `_O_*` values); on Linux the real `<fcntl.h>`
//!    values differ (`O_APPEND = 0o2000`, `O_CREAT = 0o100`). This module
//!    reproduces that split exactly — [`flags`] re-exports `libc`'s constants on
//!    unix and the hand-rolled MSVC values on Windows — so a flag word is only
//!    ever meaningful on the platform that produced it. Callers must use the
//!    [`flags`] constants, never hardcoded integers.
//! 2. **Errors are `io::Result`, not `bool` / `-1`.** The C++ collapses every
//!    failure to `false` / `-1` and leaves the cause in `errno`. Rust returns
//!    `io::Error`, whose `raw_os_error()` is the same errno the C++ classifier
//!    consumes. [`File::get_file_size`] retains the exact `-1` sentinel form.
//! 3. **`pread`/`pwrite` move the file cursor on Windows, not on unix.** unix
//!    uses `pread`/`pwrite` (`FileExt::read_at`/`write_at`), which never touch
//!    the cursor. Windows has no true positional read: `FileExt::seek_read` /
//!    `seek_write` pass the offset through `OVERLAPPED` — the *data* is
//!    correctly read/written at the requested offset (so concurrent positioned
//!    I/O is data-safe on both platforms), but the shared cursor is updated as a
//!    side effect. Mixing [`File::pread`]/[`File::pwrite`] with the
//!    cursor-relative [`File::read`]/[`File::write`] is therefore
//!    platform-dependent; pick one mode per handle. The C++ backends never mix
//!    (they are offset-only), so a faithful port of their usage cannot reach
//!    this divergence.
//! 4. **Positioned writes on an `O_APPEND` handle ignore the offset on
//!    Windows** (`FILE_APPEND_DATA` always writes at EOF). POSIX `pwrite` has
//!    the same quirk by specification (`O_APPEND` overrides the offset), so the
//!    platforms agree despite different mechanisms — noted because it surprises
//!    readers of `pwrite`.
//! 5. **`mode` is ignored on Windows.** It is applied via
//!    `OpenOptionsExt::mode` on unix (matching `open(path, flags, mode)`) and
//!    dropped on Windows, exactly as the C++ IOCP backend drops it — the
//!    `mode_t` there is a `typedef int` that no Win32 call consumes.
//! 6. **Flag bits beyond those mapped reach `open(2)` verbatim on unix and are
//!    dropped on Windows.** Any bit outside
//!    `O_ACCMODE | O_APPEND | O_CREAT | O_TRUNC` (e.g. `O_DIRECT`, `O_SYNC`,
//!    `O_NOATIME`) is forwarded through `OpenOptionsExt::custom_flags`. Windows
//!    has no such channel, so those bits are silently dropped there — as in the
//!    C++, whose `_O_*` set has no equivalents.
//! 7. **Two POSIX flag combinations need a non-atomic emulation pass.** Rust's
//!    `OpenOptions` rejects `create`/`truncate` without write access, and
//!    rejects `append + truncate` outright, whereas `open(2)` accepts
//!    `O_RDONLY|O_CREAT` (creates the file) and `O_WRONLY|O_APPEND|O_TRUNC`
//!    (truncates, then appends). Both are emulated with a preparatory open, so
//!    the observable end state matches the C++ — but the sequence is **two
//!    syscalls, not one**: a concurrent racer can observe the intermediate
//!    state. `O_RDONLY|O_CREAT` uses `create_new`, so it never clobbers — nor
//!    fails on — an existing write-protected file. `O_TRUNC` without write
//!    access stays an error (`InvalidInput`): POSIX leaves it unspecified and
//!    Linux does not truncate.
//! 8. **`exists` is an addition, not a port.** No `Exists` helper exists
//!    anywhere in `clio_ctp/io/` (the nearest relative is
//!    `SystemInfo::RemoveFile`). It follows symlinks and reports `false` for a
//!    path that exists but cannot be `stat`ed (e.g. permission denied on a
//!    parent directory) — it answers "is this path usable", not "is this name
//!    taken".
//! 9. **`io_error.h` (`IoError`, `ClassifyErrno`, `ClassifyWinError`) is not
//!    ported here.** It classifies `IoResult::error_code` for the async
//!    backends and belongs to `async_io.rs` (a sibling module owned by another
//!    port); duplicating it would collide. `io::Error::raw_os_error()` from any
//!    method here feeds `ClassifyErrno` unchanged.
//! 10. **Operating on a closed handle yields `InvalidInput`**, mirroring the
//!     C++ `fd < 0 → return -1 / false` guard. `InvalidInput` is deliberate:
//!     `EBADF` maps to `IoError::kInvalid` in `io_error.h`, so the category
//!     survives the port.
//! 11. **The dual-descriptor (`O_DIRECT` + buffered) trick is not reproduced.**
//!     `libaio_io.h` / `posix_aio.h` open the path *twice* and pick a descriptor
//!     per request based on buffer/size alignment. That is an `AsyncIO`-level
//!     policy over two handles, not a property of one file, so it belongs to the
//!     `async_io.rs` port; one [`File`] is one descriptor.
//! 12. **`close()` swallows errors**, like the C++ `void Close()`. Durability
//!     must be forced explicitly before dropping the handle (the C++ has no
//!     `fsync` wrapper either, so none is added).
//! 13. **Offsets are `u64` but capped at `i64::MAX`** (`InvalidInput` beyond).
//!     The C++ takes `off_t` — signed 64-bit — so offsets above `i64::MAX` are
//!     not expressible there at all, while Rust's `FileExt` takes `u64`. The cap
//!     restores the C++ domain and closes a Win32 trap: `seek_write` maps the
//!     offset onto `OVERLAPPED.Offset`/`OffsetHigh`, and the all-ones value
//!     (`u64::MAX`) is Win32's documented **"append at EOF"** sentinel — so an
//!     unclamped `pwrite(buf, u64::MAX)` silently appends on Windows while
//!     failing with `EINVAL` on Linux. Rejecting the whole out-of-domain range
//!     makes both platforms fail identically. (Verified: without the cap, the
//!     boundary test observed exactly this silent Windows append.)

use std::fs::{self, OpenOptions};
use std::io::{self, Read, Write};
use std::path::Path;

#[cfg(unix)]
use std::os::unix::fs::{FileExt, OpenOptionsExt};
#[cfg(windows)]
use std::os::windows::fs::FileExt;

/// `open(2)` flag constants.
///
/// Mirrors the flag block in `clio_ctp/io/async_io.h`, **including its
/// platform split**: on unix these are `libc`'s real `<fcntl.h>` values; on
/// Windows they are the MSVC `_O_*` values the C++ header hand-rolls. The two
/// sets are *not* interchangeable (divergence 1) — always use these constants
/// rather than literals.
pub mod flags {
    /// Open for reading only.
    pub const O_RDONLY: i32 = imp::O_RDONLY;
    /// Open for writing only.
    pub const O_WRONLY: i32 = imp::O_WRONLY;
    /// Open for reading and writing.
    pub const O_RDWR: i32 = imp::O_RDWR;
    /// Append on every write.
    pub const O_APPEND: i32 = imp::O_APPEND;
    /// Create the file if it does not exist.
    pub const O_CREAT: i32 = imp::O_CREAT;
    /// Truncate the file to length 0 on open.
    pub const O_TRUNC: i32 = imp::O_TRUNC;
    /// Mask selecting the access-mode bits of a flag word.
    pub const O_ACCMODE: i32 = imp::O_ACCMODE;

    /// Unbuffered I/O.
    ///
    /// Linux-only, exactly as in the C++: `posix_aio.h` does
    /// `#ifndef O_DIRECT` / `#define O_DIRECT 0` so that `flags & ~O_DIRECT`
    /// bit-ops still compile where the platform lacks it (Darwin uses
    /// `fcntl(F_NOCACHE)` post-open; Windows uses `FILE_FLAG_NO_BUFFERING`).
    /// A zero value makes `O_DIRECT` an inert bit that degrades to buffered
    /// I/O.
    pub const O_DIRECT: i32 = imp::O_DIRECT;

    #[cfg(unix)]
    mod imp {
        pub const O_RDONLY: i32 = libc::O_RDONLY;
        pub const O_WRONLY: i32 = libc::O_WRONLY;
        pub const O_RDWR: i32 = libc::O_RDWR;
        pub const O_APPEND: i32 = libc::O_APPEND;
        pub const O_CREAT: i32 = libc::O_CREAT;
        pub const O_TRUNC: i32 = libc::O_TRUNC;
        pub const O_ACCMODE: i32 = libc::O_ACCMODE;
        #[cfg(target_os = "linux")]
        pub const O_DIRECT: i32 = libc::O_DIRECT;
        #[cfg(not(target_os = "linux"))]
        pub const O_DIRECT: i32 = 0;
    }

    // Values hand-rolled by async_io.h under _WIN32 (the MSVC _O_* constants).
    #[cfg(windows)]
    mod imp {
        pub const O_RDONLY: i32 = 0x0000;
        pub const O_WRONLY: i32 = 0x0001;
        pub const O_RDWR: i32 = 0x0002;
        pub const O_APPEND: i32 = 0x0008;
        pub const O_CREAT: i32 = 0x0100;
        pub const O_TRUNC: i32 = 0x0200;
        pub const O_ACCMODE: i32 = 0x0003;
        pub const O_DIRECT: i32 = 0;
    }
}

/// Error for an operation on a closed handle (divergence 10).
fn closed_err() -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, "ctp::File: handle is closed")
}

fn invalid(msg: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, msg)
}

/// Reject offsets outside the C++ `off_t` domain (divergence 13).
///
/// Beyond `i64::MAX` the platforms disagree dangerously: Linux returns
/// `EINVAL`, while Windows reads `u64::MAX` as the `OVERLAPPED` "append at EOF"
/// sentinel and would *succeed* at the wrong place.
fn check_offset(offset: u64) -> io::Result<()> {
    if offset > i64::MAX as u64 {
        return Err(invalid("ctp::File: offset exceeds the off_t domain"));
    }
    Ok(())
}

/// An owned, synchronous file handle.
///
/// The RAII counterpart of the C++ `ctp::File` union
/// (`introspect/system_info.h`), carrying the blocking operations every
/// `AsyncIO` backend performs on its descriptor. Closing is idempotent and
/// happens automatically on drop, mirroring the backends'
/// `~AsyncIO() { Close(); }` plus `fd = -1` discipline.
///
/// Positioned I/O ([`pread`](File::pread) / [`pwrite`](File::pwrite)) takes
/// `&self` and is safe to call concurrently from multiple threads; see
/// divergence 3 for the Windows cursor caveat.
#[derive(Debug)]
pub struct File {
    inner: Option<fs::File>,
}

impl File {
    /// Open a file — the port of `AsyncIO::Open(path, flags, mode)`.
    ///
    /// `flags` is a bitwise-or of [`flags`] constants (platform-specific
    /// values — divergence 1). `mode` is the creation mode (e.g. `0o644`),
    /// applied only when the file is created and only on unix (divergence 5).
    ///
    /// Where the C++ returns `false`, this returns the underlying `io::Error`
    /// (divergence 2). See divergence 7 for the two flag combinations that
    /// require a non-atomic emulation pass.
    pub fn open(path: impl AsRef<Path>, flags: i32, mode: u32) -> io::Result<Self> {
        let path = path.as_ref();

        let (read, write) = match flags & flags::O_ACCMODE {
            a if a == flags::O_RDONLY => (true, false),
            a if a == flags::O_WRONLY => (false, true),
            a if a == flags::O_RDWR => (true, true),
            _ => return Err(invalid("ctp::File: invalid access mode in flags")),
        };
        let append = (flags & flags::O_APPEND) != 0;
        let creat = (flags & flags::O_CREAT) != 0;
        let trunc = (flags & flags::O_TRUNC) != 0;

        // O_TRUNC is meaningful only with write access: POSIX leaves
        // O_RDONLY|O_TRUNC unspecified and Linux does not truncate.
        if trunc && !write {
            return Err(invalid(
                "ctp::File: O_TRUNC requires write access (POSIX-unspecified combination)",
            ));
        }

        // Emulation pass 1 (divergence 7): open(2) creates for O_RDONLY|O_CREAT;
        // OpenOptions refuses `create` without write access. `create_new` leaves
        // an existing (possibly write-protected) file untouched.
        if creat && !write {
            let mut pre = OpenOptions::new();
            pre.write(true).create_new(true);
            #[cfg(unix)]
            pre.mode(mode);
            match pre.open(path) {
                Ok(_) => {}
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {}
                Err(e) => return Err(e),
            }
        }

        // Emulation pass 2 (divergence 7): OpenOptions rejects append+truncate,
        // while open(2) accepts O_APPEND|O_TRUNC (truncate first, then append).
        if append && trunc {
            let mut pre = OpenOptions::new();
            pre.write(true).truncate(true).create(creat);
            #[cfg(unix)]
            pre.mode(mode);
            pre.open(path)?;
        }

        let mut opts = OpenOptions::new();
        opts.read(read);
        if write {
            // O_APPEND without write access is inert in POSIX; ignore it rather
            // than let `append(true)` silently escalate to write access.
            if append {
                opts.append(true);
            } else {
                opts.write(true);
            }
            if creat {
                opts.create(true);
            }
            // Pass 2 already truncated; `append(true).truncate(true)` is rejected.
            if trunc && !append {
                opts.truncate(true);
            }
        }

        #[cfg(unix)]
        {
            opts.mode(mode);
            // Divergence 6: hand any unmapped bits (O_DIRECT, O_SYNC, ...)
            // straight to open(2). std masks O_ACCMODE out of custom flags.
            let mapped = flags::O_ACCMODE | flags::O_APPEND | flags::O_CREAT | flags::O_TRUNC;
            let extra = flags & !mapped;
            if extra != 0 {
                opts.custom_flags(extra);
            }
        }
        #[cfg(not(unix))]
        let _ = mode; // divergence 5

        Ok(Self {
            inner: Some(opts.open(path)?),
        })
    }

    /// True if `path` can be `stat`ed. Addition, not a port — see divergence 8.
    pub fn exists(path: impl AsRef<Path>) -> bool {
        fs::metadata(path).is_ok()
    }

    /// True while the handle is open (the C++ `fd >= 0` test).
    pub fn is_open(&self) -> bool {
        self.inner.is_some()
    }

    fn get(&self) -> io::Result<&fs::File> {
        self.inner.as_ref().ok_or_else(closed_err)
    }

    fn get_mut(&mut self) -> io::Result<&mut fs::File> {
        self.inner.as_mut().ok_or_else(closed_err)
    }

    /// Read at the current cursor, advancing it. Short reads are not errors;
    /// `Ok(0)` means EOF. No C++ counterpart (the backends are offset-only).
    pub fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.get_mut()?.read(buf)
    }

    /// Write at the current cursor, advancing it. Short writes are not errors.
    /// No C++ counterpart (the backends are offset-only).
    pub fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.get_mut()?.write(buf)
    }

    /// Positioned read — the blocking form of `io_prep_pread` (`libaio_io.h`).
    ///
    /// Single-shot: a short read is not an error, and `Ok(0)` means EOF (an
    /// offset at or past EOF reads nothing). `offset` must fit in `off_t`
    /// (divergence 13). See divergence 3 for the Windows cursor side effect.
    pub fn pread(&self, buf: &mut [u8], offset: u64) -> io::Result<usize> {
        let file = self.get()?;
        check_offset(offset)?;
        #[cfg(unix)]
        {
            file.read_at(buf, offset)
        }
        #[cfg(windows)]
        {
            file.seek_read(buf, offset)
        }
    }

    /// Positioned write — the blocking form of `io_prep_pwrite`
    /// (`libaio_io.h`).
    ///
    /// Single-shot: a short write is not an error. Writing past EOF extends the
    /// file, zero-filling the gap. `offset` must fit in `off_t`
    /// (divergence 13). See divergences 3 and 4 for the Windows cursor side
    /// effect and the `O_APPEND` offset override.
    pub fn pwrite(&self, buf: &[u8], offset: u64) -> io::Result<usize> {
        let file = self.get()?;
        check_offset(offset)?;
        #[cfg(unix)]
        {
            file.write_at(buf, offset)
        }
        #[cfg(windows)]
        {
            file.seek_write(buf, offset)
        }
    }

    /// Fill `buf` entirely from `offset`, looping over short reads and retrying
    /// on `EINTR`. Convenience addition; `UnexpectedEof` if EOF arrives first.
    pub fn pread_exact(&self, mut buf: &mut [u8], mut offset: u64) -> io::Result<()> {
        while !buf.is_empty() {
            match self.pread(buf, offset) {
                Ok(0) => {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "ctp::File: failed to fill whole buffer",
                    ))
                }
                Ok(n) => {
                    buf = &mut buf[n..];
                    offset = offset
                        .checked_add(n as u64)
                        .ok_or_else(|| invalid("ctp::File: offset overflow"))?;
                }
                Err(ref e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    /// Write all of `buf` at `offset`, looping over short writes and retrying
    /// on `EINTR`. Convenience addition; `WriteZero` if the file stops
    /// accepting bytes.
    pub fn pwrite_all(&self, mut buf: &[u8], mut offset: u64) -> io::Result<()> {
        while !buf.is_empty() {
            match self.pwrite(buf, offset) {
                Ok(0) => {
                    return Err(io::Error::new(
                        io::ErrorKind::WriteZero,
                        "ctp::File: failed to write whole buffer",
                    ))
                }
                Ok(n) => {
                    buf = &buf[n..];
                    offset = offset
                        .checked_add(n as u64)
                        .ok_or_else(|| invalid("ctp::File: offset overflow"))?;
                }
                Err(ref e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    /// File size in bytes — `AsyncIO::GetFileSize()` without the `-1` sentinel
    /// (`fstat().st_size`).
    pub fn size(&self) -> io::Result<u64> {
        Ok(self.get()?.metadata()?.len())
    }

    /// Sentinel-faithful `AsyncIO::GetFileSize()`: the size, or `-1` on any
    /// error (including a closed handle).
    ///
    /// A size exceeding `i64::MAX` also yields `-1`; the C++ would return the
    /// implementation-defined `ssize_t` narrowing of `st_size`, which cannot be
    /// reproduced without inviting a negative "success".
    pub fn get_file_size(&self) -> i64 {
        match self.size() {
            Ok(size) => i64::try_from(size).unwrap_or(-1),
            Err(_) => -1,
        }
    }

    /// Truncate or extend to `size` — `AsyncIO::Truncate(size)` (`ftruncate`).
    /// Extending zero-fills. Requires write access.
    pub fn truncate(&self, size: u64) -> io::Result<()> {
        self.get()?.set_len(size)
    }

    /// Close the handle — `AsyncIO::Close()`. Idempotent and infallible; errors
    /// are swallowed exactly as by the C++ `void Close()` (divergence 12).
    /// Called automatically on drop.
    pub fn close(&mut self) {
        self.inner = None;
    }
}

#[cfg(test)]
mod tests {
    use super::flags::*;
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Unique temp path that deletes itself on drop (no `tempfile` dep — the
    /// crate's Cargo.toml is fixed).
    struct TempPath(PathBuf);

    impl TempPath {
        fn new(tag: &str) -> Self {
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let mut p = std::env::temp_dir();
            p.push(format!(
                "ctp_io_file_{}_{}_{}.bin",
                tag,
                std::process::id(),
                n
            ));
            let _ = fs::remove_file(&p);
            Self(p)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempPath {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.0);
        }
    }

    fn create(path: &Path) -> File {
        File::open(path, O_RDWR | O_CREAT | O_TRUNC, 0o644).expect("create")
    }

    // -- flags -------------------------------------------------------------

    #[test]
    fn access_mode_bits_are_distinct_and_masked_by_accmode() {
        assert_eq!(O_RDONLY & O_ACCMODE, O_RDONLY);
        assert_eq!(O_WRONLY & O_ACCMODE, O_WRONLY);
        assert_eq!(O_RDWR & O_ACCMODE, O_RDWR);
        assert_ne!(O_RDONLY, O_WRONLY);
        assert_ne!(O_WRONLY, O_RDWR);
        // The modifiers must not collide with the access-mode bits, or the
        // O_ACCMODE mask in open() would misclassify them.
        for modifier in [O_APPEND, O_CREAT, O_TRUNC] {
            assert_eq!(
                modifier & O_ACCMODE,
                0,
                "modifier {modifier:#x} overlaps O_ACCMODE"
            );
        }
    }

    #[test]
    fn o_direct_is_zero_off_linux_like_the_cpp_fallback() {
        if cfg!(target_os = "linux") {
            assert_ne!(O_DIRECT, 0);
        } else {
            assert_eq!(O_DIRECT, 0);
        }
    }

    #[test]
    fn invalid_access_mode_is_rejected() {
        let tmp = TempPath::new("badmode");
        // O_ACCMODE (3) is not a valid access mode on either platform.
        let err = File::open(tmp.path(), O_ACCMODE | O_CREAT, 0o644).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        assert!(!File::exists(tmp.path()), "must not create on flag error");
    }

    // -- open / exists / close --------------------------------------------

    #[test]
    fn create_write_read_roundtrip() {
        let tmp = TempPath::new("roundtrip");
        let mut f = create(tmp.path());
        assert_eq!(f.write(b"hello world").unwrap(), 11);
        f.close();

        let f = File::open(tmp.path(), O_RDONLY, 0).unwrap();
        let mut buf = [0u8; 11];
        f.pread_exact(&mut buf, 0).unwrap();
        assert_eq!(&buf, b"hello world");
        assert_eq!(f.size().unwrap(), 11);
    }

    #[test]
    fn exists_tracks_creation_and_removal() {
        let tmp = TempPath::new("exists");
        assert!(!File::exists(tmp.path()));
        let f = create(tmp.path());
        assert!(File::exists(tmp.path()));
        drop(f);
        fs::remove_file(tmp.path()).unwrap();
        assert!(!File::exists(tmp.path()));
    }

    #[test]
    fn exists_is_false_for_empty_and_nonexistent_paths() {
        assert!(!File::exists(""));
        assert!(!File::exists("ctp_io_definitely_not_a_real_path_9f3a2b"));
    }

    #[test]
    fn open_missing_without_o_creat_fails() {
        let tmp = TempPath::new("missing");
        let err = File::open(tmp.path(), O_RDONLY, 0).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
        assert!(!File::exists(tmp.path()));
    }

    #[test]
    fn close_is_idempotent_and_closed_ops_report_invalid_input() {
        let tmp = TempPath::new("closed");
        let mut f = create(tmp.path());
        f.write(b"data").unwrap();
        assert!(f.is_open());

        f.close();
        f.close(); // idempotent, mirrors the C++ fd == -1 guard
        assert!(!f.is_open());

        assert_eq!(f.get_file_size(), -1); // sentinel-faithful
        assert_eq!(f.size().unwrap_err().kind(), io::ErrorKind::InvalidInput);
        assert_eq!(
            f.pread(&mut [0u8; 4], 0).unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
        assert_eq!(
            f.pwrite(b"x", 0).unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
        assert_eq!(
            f.read(&mut [0u8; 4]).unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
        assert_eq!(
            f.write(b"x").unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
        assert_eq!(
            f.truncate(0).unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
    }

    #[test]
    fn drop_closes_the_handle() {
        let tmp = TempPath::new("dropclose");
        {
            let mut f = create(tmp.path());
            f.write(b"abc").unwrap();
        } // dtor == C++ ~AsyncIO() { Close(); }
          // Deleting proves the handle is gone (on Windows an open handle blocks it).
        fs::remove_file(tmp.path()).unwrap();
        assert!(!File::exists(tmp.path()));
    }

    // -- flag semantics ----------------------------------------------------

    #[test]
    fn o_trunc_clears_existing_content() {
        let tmp = TempPath::new("trunc_flag");
        {
            let mut f = create(tmp.path());
            f.write(b"0123456789").unwrap();
            assert_eq!(f.size().unwrap(), 10);
        }
        let f = File::open(tmp.path(), O_WRONLY | O_TRUNC, 0o644).unwrap();
        assert_eq!(f.size().unwrap(), 0);
    }

    #[test]
    fn o_trunc_without_write_access_is_rejected() {
        let tmp = TempPath::new("trunc_ro");
        {
            let mut f = create(tmp.path());
            f.write(b"keepme").unwrap();
        }
        let err = File::open(tmp.path(), O_RDONLY | O_TRUNC, 0).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        // Divergence 7: the file must be left intact.
        let f = File::open(tmp.path(), O_RDONLY, 0).unwrap();
        assert_eq!(f.size().unwrap(), 6);
    }

    #[test]
    fn o_append_writes_go_to_end() {
        let tmp = TempPath::new("append");
        {
            let mut f = create(tmp.path());
            f.write(b"abc").unwrap();
        }
        {
            let mut f = File::open(tmp.path(), O_WRONLY | O_APPEND, 0o644).unwrap();
            f.write(b"def").unwrap();
            f.write(b"ghi").unwrap();
        }
        let f = File::open(tmp.path(), O_RDONLY, 0).unwrap();
        let mut buf = [0u8; 9];
        f.pread_exact(&mut buf, 0).unwrap();
        assert_eq!(&buf, b"abcdefghi");
    }

    #[test]
    fn o_append_with_o_trunc_truncates_then_appends() {
        // Divergence 7, emulation pass 2: OpenOptions rejects append+truncate;
        // open(2) accepts it. The end state must match the C++.
        let tmp = TempPath::new("append_trunc");
        {
            let mut f = create(tmp.path());
            f.write(b"old content").unwrap();
        }
        {
            let mut f = File::open(tmp.path(), O_WRONLY | O_APPEND | O_TRUNC, 0o644).unwrap();
            assert_eq!(f.size().unwrap(), 0, "O_TRUNC must have cleared the file");
            f.write(b"new").unwrap();
        }
        let f = File::open(tmp.path(), O_RDONLY, 0).unwrap();
        let mut buf = [0u8; 3];
        f.pread_exact(&mut buf, 0).unwrap();
        assert_eq!(&buf, b"new");
        assert_eq!(f.size().unwrap(), 3);
    }

    #[test]
    fn o_creat_with_o_rdonly_creates_the_file() {
        // Divergence 7, emulation pass 1: open(2) creates here; OpenOptions
        // alone refuses create without write access.
        let tmp = TempPath::new("creat_ro");
        assert!(!File::exists(tmp.path()));
        let f = File::open(tmp.path(), O_RDONLY | O_CREAT, 0o644).unwrap();
        assert!(File::exists(tmp.path()));
        assert_eq!(f.size().unwrap(), 0);
        // The handle is still read-only.
        assert!(f.pwrite(b"x", 0).is_err());
    }

    #[test]
    fn o_creat_with_o_rdonly_does_not_clobber_existing_content() {
        // The create_new-based emulation must never truncate an existing file.
        let tmp = TempPath::new("creat_ro_existing");
        {
            let mut f = create(tmp.path());
            f.write(b"precious").unwrap();
        }
        let f = File::open(tmp.path(), O_RDONLY | O_CREAT, 0o644).unwrap();
        let mut buf = [0u8; 8];
        f.pread_exact(&mut buf, 0).unwrap();
        assert_eq!(&buf, b"precious");
    }

    #[test]
    fn read_only_handle_rejects_writes_and_truncate() {
        let tmp = TempPath::new("ro");
        {
            let mut f = create(tmp.path());
            f.write(b"abcdef").unwrap();
        }
        let mut f = File::open(tmp.path(), O_RDONLY, 0).unwrap();
        assert!(f.pwrite(b"X", 0).is_err());
        assert!(f.write(b"X").is_err());
        assert!(f.truncate(0).is_err());
        // ...and the data is untouched.
        assert_eq!(f.size().unwrap(), 6);
    }

    #[test]
    fn write_only_handle_rejects_reads() {
        let tmp = TempPath::new("wo");
        {
            let mut f = create(tmp.path());
            f.write(b"abcdef").unwrap();
        }
        let f = File::open(tmp.path(), O_WRONLY, 0o644).unwrap();
        assert!(f.pread(&mut [0u8; 6], 0).is_err());
        // Size still works: it is an fstat, not a read.
        assert_eq!(f.get_file_size(), 6);
    }

    // -- empty / zero / EOF edges -----------------------------------------

    #[test]
    fn empty_file_has_zero_size_and_reads_eof() {
        let tmp = TempPath::new("empty");
        let f = create(tmp.path());
        assert_eq!(f.size().unwrap(), 0);
        assert_eq!(f.get_file_size(), 0);
        assert_eq!(f.pread(&mut [0u8; 8], 0).unwrap(), 0); // EOF, not an error
    }

    #[test]
    fn zero_length_buffers_transfer_nothing() {
        let tmp = TempPath::new("zerobuf");
        let mut f = create(tmp.path());
        assert_eq!(f.pwrite(&[], 0).unwrap(), 0);
        assert_eq!(f.pread(&mut [], 0).unwrap(), 0);
        assert_eq!(f.write(&[]).unwrap(), 0);
        assert_eq!(f.read(&mut []).unwrap(), 0);
        assert_eq!(f.size().unwrap(), 0);
        // Loops over an empty buffer are trivially complete, even at EOF.
        f.pread_exact(&mut [], 0).unwrap();
        f.pwrite_all(&[], 0).unwrap();
    }

    #[test]
    fn pread_past_eof_returns_zero_and_at_boundary_returns_short() {
        let tmp = TempPath::new("eof");
        let f = create(tmp.path());
        f.pwrite_all(b"0123456789", 0).unwrap();

        let mut buf = [0xAAu8; 8];
        // Straddling EOF: a short read of the 4 remaining bytes.
        assert_eq!(f.pread(&mut buf, 6).unwrap(), 4);
        assert_eq!(&buf[..4], b"6789");
        assert_eq!(buf[4], 0xAA, "bytes past EOF must be left untouched");
        // Exactly at EOF, and past it.
        assert_eq!(f.pread(&mut buf, 10).unwrap(), 0);
        assert_eq!(f.pread(&mut buf, 11).unwrap(), 0);
        assert_eq!(f.pread(&mut buf, 1_000_000).unwrap(), 0);
    }

    #[test]
    fn pread_exact_past_eof_is_unexpected_eof() {
        let tmp = TempPath::new("preadexact_eof");
        let f = create(tmp.path());
        f.pwrite_all(b"abc", 0).unwrap();
        let err = f.pread_exact(&mut [0u8; 4], 0).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn pwrite_past_eof_extends_and_zero_fills_the_gap() {
        let tmp = TempPath::new("hole");
        let f = create(tmp.path());
        f.pwrite_all(b"tail", 4096).unwrap();
        assert_eq!(f.size().unwrap(), 4100);

        let mut gap = [0xFFu8; 4096];
        f.pread_exact(&mut gap, 0).unwrap();
        assert!(gap.iter().all(|&b| b == 0), "gap must be zero-filled");

        let mut tail = [0u8; 4];
        f.pread_exact(&mut tail, 4096).unwrap();
        assert_eq!(&tail, b"tail");
    }

    #[test]
    fn pwrite_overwrites_in_place_without_extending() {
        let tmp = TempPath::new("overwrite");
        let f = create(tmp.path());
        f.pwrite_all(b"aaaaaaaa", 0).unwrap();
        f.pwrite_all(b"bb", 3).unwrap();
        let mut buf = [0u8; 8];
        f.pread_exact(&mut buf, 0).unwrap();
        assert_eq!(&buf, b"aaabbaaa");
        assert_eq!(f.size().unwrap(), 8);
    }

    // -- truncate ----------------------------------------------------------

    #[test]
    fn truncate_shrinks_extends_and_zeroes() {
        let tmp = TempPath::new("truncate");
        let f = create(tmp.path());
        f.pwrite_all(b"0123456789", 0).unwrap();

        f.truncate(4).unwrap();
        assert_eq!(f.size().unwrap(), 4);
        let mut buf = [0u8; 4];
        f.pread_exact(&mut buf, 0).unwrap();
        assert_eq!(&buf, b"0123");
        assert_eq!(f.pread(&mut [0u8; 4], 4).unwrap(), 0);

        // Extending zero-fills (ftruncate / SetEndOfFile semantics).
        f.truncate(12).unwrap();
        assert_eq!(f.size().unwrap(), 12);
        let mut ext = [0xFFu8; 8];
        f.pread_exact(&mut ext, 4).unwrap();
        assert!(ext.iter().all(|&b| b == 0));

        // To zero, and to the same size twice (idempotent).
        f.truncate(0).unwrap();
        assert_eq!(f.size().unwrap(), 0);
        f.truncate(0).unwrap();
        assert_eq!(f.get_file_size(), 0);
    }

    // -- boundaries / overflow --------------------------------------------

    #[test]
    fn offsets_beyond_the_off_t_domain_are_rejected() {
        // Divergence 13. u64::MAX is the dangerous one: it is Win32's
        // OVERLAPPED "append at EOF" sentinel, so an unclamped seek_write there
        // silently appends instead of failing (observed before the cap existed).
        let tmp = TempPath::new("bigoff");
        let f = create(tmp.path());
        f.pwrite_all(b"x", 0).unwrap();

        for offset in [i64::MAX as u64 + 1, u64::MAX] {
            assert_eq!(
                f.pread(&mut [0u8; 8], offset).unwrap_err().kind(),
                io::ErrorKind::InvalidInput,
                "pread at {offset:#x} must be rejected"
            );
            assert_eq!(
                f.pwrite(b"y", offset).unwrap_err().kind(),
                io::ErrorKind::InvalidInput,
                "pwrite at {offset:#x} must be rejected, not appended"
            );
        }
        assert_eq!(f.size().unwrap(), 1, "file must be unchanged");
    }

    #[test]
    fn the_largest_in_domain_offset_never_panics_or_fabricates_data() {
        // i64::MAX is inside off_t, so it reaches the OS. Platforms differ
        // (EFBIG/EINVAL vs. EOF); the contract is only that we neither panic nor
        // report bytes that cannot exist, and never corrupt the file.
        let tmp = TempPath::new("maxoff");
        let f = create(tmp.path());
        f.pwrite_all(b"x", 0).unwrap();

        if let Ok(n) = f.pread(&mut [0u8; 8], i64::MAX as u64) {
            assert_eq!(n, 0, "no data can exist at i64::MAX");
        }
        let _ = f.pwrite(b"y", i64::MAX as u64); // may fail (EFBIG); must not corrupt
        let mut first = [0u8; 1];
        f.pread_exact(&mut first, 0).unwrap();
        assert_eq!(&first, b"x", "byte 0 must survive");
    }

    #[test]
    fn looping_wrappers_reject_out_of_domain_offsets_rather_than_wrapping() {
        let tmp = TempPath::new("ovf");
        let f = create(tmp.path());
        f.pwrite_all(b"0123456789", 0).unwrap();
        // An offset where `offset + n` would wrap u64 is refused up front by the
        // off_t check; the `checked_add` inside the loops is defense in depth
        // (unreachable while the cap holds), never a wrap.
        assert_eq!(
            f.pread_exact(&mut [0u8; 4], u64::MAX - 1).unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
        assert_eq!(
            f.pwrite_all(b"abcd", u64::MAX - 1).unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
        assert_eq!(f.size().unwrap(), 10, "file must be unchanged");
    }

    #[test]
    fn get_file_size_never_reports_a_negative_size_for_a_healthy_file() {
        // i64::try_from guards the ssize_t narrowing (divergence 2). An >8 EiB
        // file cannot be created in a test, so assert the property that matters.
        let tmp = TempPath::new("sizesent");
        let f = create(tmp.path());
        f.pwrite_all(&[7u8; 1024], 0).unwrap();
        assert_eq!(f.get_file_size(), 1024);
        assert!(f.get_file_size() >= 0);
    }

    // -- cursor I/O --------------------------------------------------------

    #[test]
    fn sequential_reads_and_writes_advance_the_cursor() {
        let tmp = TempPath::new("cursor");
        let mut f = create(tmp.path());
        f.write(b"abc").unwrap();
        f.write(b"def").unwrap();
        assert_eq!(f.size().unwrap(), 6);
        f.close();

        let mut f = File::open(tmp.path(), O_RDONLY, 0).unwrap();
        let mut a = [0u8; 3];
        let mut b = [0u8; 3];
        assert_eq!(f.read(&mut a).unwrap(), 3);
        assert_eq!(f.read(&mut b).unwrap(), 3);
        assert_eq!(&a, b"abc");
        assert_eq!(&b, b"def");
        assert_eq!(f.read(&mut a).unwrap(), 0, "EOF");
    }

    // -- concurrency -------------------------------------------------------

    #[test]
    fn concurrent_positioned_io_on_disjoint_regions_is_consistent() {
        // pread/pwrite take &self; each thread owns one 4 KiB region. The data
        // must be correct on both platforms even though Windows shares a cursor
        // across the seek_read/seek_write calls (divergence 3).
        const THREADS: u64 = 8;
        const CHUNK: usize = 4096;

        let tmp = TempPath::new("concurrent");
        let f = create(tmp.path());
        f.truncate(THREADS * CHUNK as u64).unwrap();

        std::thread::scope(|scope| {
            for t in 0..THREADS {
                let f = &f;
                scope.spawn(move || {
                    let offset = t * CHUNK as u64;
                    let pattern = vec![t as u8 + 1; CHUNK];
                    for _ in 0..8 {
                        f.pwrite_all(&pattern, offset).unwrap();
                        let mut back = vec![0u8; CHUNK];
                        f.pread_exact(&mut back, offset).unwrap();
                        assert_eq!(back, pattern, "region {t} was corrupted");
                    }
                });
            }
        });

        assert_eq!(f.size().unwrap(), THREADS * CHUNK as u64);
        for t in 0..THREADS {
            let mut back = vec![0u8; CHUNK];
            f.pread_exact(&mut back, t * CHUNK as u64).unwrap();
            assert!(back.iter().all(|&b| b == t as u8 + 1), "region {t} wrong");
        }
    }

    #[test]
    fn concurrent_readers_of_the_same_region_agree() {
        const READERS: usize = 8;
        let tmp = TempPath::new("shared_read");
        let f = create(tmp.path());
        let data: Vec<u8> = (0..4096u32).map(|i| (i % 251) as u8).collect();
        f.pwrite_all(&data, 0).unwrap();

        std::thread::scope(|scope| {
            for _ in 0..READERS {
                let f = &f;
                let expect = &data;
                scope.spawn(move || {
                    for _ in 0..16 {
                        let mut back = vec![0u8; expect.len()];
                        f.pread_exact(&mut back, 0).unwrap();
                        assert_eq!(&back, expect);
                    }
                });
            }
        });
    }
}
