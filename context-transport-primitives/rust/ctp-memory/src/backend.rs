// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).
//
//! Shared-memory backend (C++ `PosixShmMmap` analog): named, cross-process
//! mappings via `shm_open`+`mmap` (POSIX) or `CreateFileMapping`+
//! `MapViewOfFile` (Windows). This is the one unsafe-heavy module; every
//! other layer works in offsets.

use std::ffi::c_void;
use std::io;

/// A named shared-memory segment mapped into this process.
///
/// Dropping unmaps the view; `destroy()` additionally removes the name
/// (POSIX `shm_unlink`; on Windows the object dies with its last handle).
pub struct SharedMemBackend {
    name: String,
    base: *mut c_void,
    size: usize,
    #[cfg(windows)]
    handle: windows_sys::Win32::Foundation::HANDLE,
    #[cfg(unix)]
    fd: i32,
}

// The mapping is process-wide state; base is stable for the lifetime of the
// object and all synchronization happens inside the segment (in-segment
// atomics per MEMORY_DESIGN.md pillar 3).
unsafe impl Send for SharedMemBackend {}
unsafe impl Sync for SharedMemBackend {}

impl SharedMemBackend {
    /// Create (or replace) a named segment of `size` bytes, zero-filled.
    pub fn create(name: &str, size: usize) -> io::Result<Self> {
        Self::open_impl(name, size, true)
    }

    /// Open an existing named segment created by any process (either
    /// language — the segment format, not the creator, is the contract).
    pub fn open(name: &str, size: usize) -> io::Result<Self> {
        Self::open_impl(name, size, false)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn size(&self) -> usize {
        self.size
    }

    /// Base address of the mapping in THIS process. Never stored inside the
    /// segment (pillar 3).
    pub fn base(&self) -> *mut u8 {
        self.base as *mut u8
    }

    #[cfg(unix)]
    fn open_impl(name: &str, size: usize, create: bool) -> io::Result<Self> {
        let cname = std::ffi::CString::new(format!("/{}", name.trim_start_matches('/')))
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "NUL in name"))?;
        // SAFETY: standard shm_open/ftruncate/mmap sequence; every return
        // value is checked and the fd is closed on failure paths.
        unsafe {
            let flags = if create {
                libc::O_CREAT | libc::O_RDWR
            } else {
                libc::O_RDWR
            };
            let fd = libc::shm_open(cname.as_ptr(), flags, 0o600);
            if fd < 0 {
                return Err(io::Error::last_os_error());
            }
            if create && libc::ftruncate(fd, size as libc::off_t) != 0 {
                let e = io::Error::last_os_error();
                libc::close(fd);
                return Err(e);
            }
            let base = libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            );
            if base == libc::MAP_FAILED {
                let e = io::Error::last_os_error();
                libc::close(fd);
                return Err(e);
            }
            Ok(Self {
                name: name.to_string(),
                base,
                size,
                fd,
            })
        }
    }

    #[cfg(windows)]
    fn open_impl(name: &str, size: usize, create: bool) -> io::Result<Self> {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MapViewOfFile, OpenFileMappingW, FILE_MAP_ALL_ACCESS,
            PAGE_READWRITE,
        };
        let wide: Vec<u16> = format!("Local\\{name}")
            .encode_utf16()
            .chain(std::iter::once(0))
            .collect();
        // SAFETY: standard file-mapping sequence; handles checked and
        // closed on failure paths.
        unsafe {
            let handle = if create {
                CreateFileMappingW(
                    windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE,
                    std::ptr::null(),
                    PAGE_READWRITE,
                    (size as u64 >> 32) as u32,
                    (size as u64 & 0xFFFF_FFFF) as u32,
                    wide.as_ptr(),
                )
            } else {
                OpenFileMappingW(FILE_MAP_ALL_ACCESS, 0, wide.as_ptr())
            };
            if handle.is_null() {
                return Err(io::Error::last_os_error());
            }
            let view = MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, size);
            if view.Value.is_null() {
                let e = io::Error::last_os_error();
                windows_sys::Win32::Foundation::CloseHandle(handle);
                return Err(e);
            }
            Ok(Self {
                name: name.to_string(),
                base: view.Value,
                size,
                handle,
            })
        }
    }

    /// Remove the segment's NAME from the system (POSIX shm_unlink). The
    /// mapping itself stays valid until dropped. On Windows the kernel
    /// object dies with its last handle, so this is a no-op.
    pub fn destroy(&self) {
        #[cfg(unix)]
        {
            if let Ok(cname) =
                std::ffi::CString::new(format!("/{}", self.name.trim_start_matches('/')))
            {
                // SAFETY: unlink of a name we created; errors ignored (may
                // already be unlinked), matching C++ DestroySharedMemory.
                unsafe { libc::shm_unlink(cname.as_ptr()) };
            }
        }
    }
}

impl Drop for SharedMemBackend {
    fn drop(&mut self) {
        // SAFETY: unmapping/closing exactly what open_impl mapped/opened.
        unsafe {
            #[cfg(unix)]
            {
                libc::munmap(self.base, self.size);
                libc::close(self.fd);
            }
            #[cfg(windows)]
            {
                use windows_sys::Win32::System::Memory::{
                    UnmapViewOfFile, MEMORY_MAPPED_VIEW_ADDRESS,
                };
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: self.base });
                windows_sys::Win32::Foundation::CloseHandle(self.handle);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_write_reopen_read() {
        let name = format!("ctp_rs_backend_test_{}", std::process::id());
        let seg = SharedMemBackend::create(&name, 4096).unwrap();
        // SAFETY: freshly mapped 4096-byte segment.
        unsafe {
            std::ptr::write_bytes(seg.base(), 0xAB, 128);
        }
        {
            let seg2 = SharedMemBackend::open(&name, 4096).unwrap();
            let b = unsafe { std::slice::from_raw_parts(seg2.base(), 128) };
            assert!(b.iter().all(|&x| x == 0xAB));
            // Write from the second mapping, visible through the first.
            unsafe { *seg2.base().add(200) = 0x5A };
        }
        assert_eq!(unsafe { *seg.base().add(200) }, 0x5A);
        seg.destroy();
    }
}
