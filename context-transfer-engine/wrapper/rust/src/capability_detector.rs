/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

//! eBPF Capability Detection Module
//!
//! This module provides runtime detection of eBPF capabilities and helps
//! choose the best interception method (eBPF or LD_PRELOAD) for the current system.
//!
//! # Example
//!
//! ```no_run
//! use clio_cte::capability_detector::{detect_best_mode, InterceptorMode};
//!
//! let mode = detect_best_mode();
//! match mode {
//!     InterceptorMode::Ebpf => println!("Using eBPF interception"),
//!     InterceptorMode::LdPreload => println!("Using LD_PRELOAD interception"),
//! }
//! ```

use std::fs;
use std::path::Path;

/// Interception mode for the profiler
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterceptorMode {
    /// Use eBPF-based interception (best performance)
    Ebpf,
    /// Use LD_PRELOAD-based interception (fallback)
    LdPreload,
}

impl std::fmt::Display for InterceptorMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InterceptorMode::Ebpf => write!(f, "eBPF"),
            InterceptorMode::LdPreload => write!(f, "LD_PRELOAD"),
        }
    }
}

/// Detailed information about eBPF capability status
#[derive(Debug, Clone)]
pub struct EbpfCapabilityInfo {
    /// Whether eBPF is fully supported
    pub is_supported: bool,
    /// Whether CAP_BPF capability is available
    pub has_cap_bpf: bool,
    /// Whether CAP_PERFMON capability is available
    pub has_cap_perfmon: bool,
    /// Kernel version (major, minor) if detectable
    pub kernel_version: Option<(u32, u32)>,
    /// Whether kernel version meets minimum requirement (5.8+)
    pub kernel_version_ok: bool,
    /// Whether /sys/fs/bpf is mounted and accessible
    pub bpf_fs_mounted: bool,
    /// Detailed reason if eBPF is not supported
    pub reason: Option<String>,
}

impl EbpfCapabilityInfo {
    /// Create a new capability info with all checks performed
    pub fn new() -> Self {
        let kernel_version = get_kernel_version();
        let kernel_version_ok = check_kernel_version(kernel_version);
        let has_cap_bpf = has_cap_bpf();
        let has_cap_perfmon = has_cap_perfmon();
        let bpf_fs_mounted = check_bpf_filesystem();

        let is_supported = kernel_version_ok && has_cap_bpf && has_cap_perfmon && bpf_fs_mounted;

        let reason = if !is_supported {
            let mut reasons = Vec::new();

            if !kernel_version_ok {
                reasons.push(format!(
                    "Kernel version {:?} is below minimum 5.8",
                    kernel_version
                ));
            }

            if !has_cap_bpf {
                reasons.push("Missing CAP_BPF capability".to_string());
            }

            if !has_cap_perfmon {
                reasons.push("Missing CAP_PERFMON capability".to_string());
            }

            if !bpf_fs_mounted {
                reasons.push("/sys/fs/bpf not mounted or not accessible".to_string());
            }

            Some(reasons.join("; "))
        } else {
            None
        };

        Self {
            is_supported,
            has_cap_bpf,
            has_cap_perfmon,
            kernel_version,
            kernel_version_ok,
            bpf_fs_mounted,
            reason,
        }
    }
}

impl Default for EbpfCapabilityInfo {
    fn default() -> Self {
        Self::new()
    }
}

/// Check if the process has CAP_BPF capability
///
/// This capability is required for loading eBPF programs and creating BPF maps.
///
/// # Returns
///
/// `true` if the process has CAP_BPF, `false` otherwise
pub fn has_cap_bpf() -> bool {
    #[cfg(target_os = "linux")]
    {
        use caps::{CapSet, Capability};

        // Check effective capabilities
        match caps::has_cap(None, CapSet::Effective, Capability::CAP_BPF) {
            Ok(has_cap) => has_cap,
            Err(e) => {
                eprintln!("Warning: Failed to check CAP_BPF: {}", e);
                false
            }
        }
    }

    #[cfg(not(target_os = "linux"))]
    {
        false
    }
}

/// Check if the process has CAP_PERFMON capability
///
/// This capability is required for eBPF performance monitoring features.
///
/// # Returns
///
/// `true` if the process has CAP_PERFMON, `false` otherwise
pub fn has_cap_perfmon() -> bool {
    #[cfg(target_os = "linux")]
    {
        use caps::{CapSet, Capability};

        // Check effective capabilities
        match caps::has_cap(None, CapSet::Effective, Capability::CAP_PERFMON) {
            Ok(has_cap) => has_cap,
            Err(e) => {
                eprintln!("Warning: Failed to check CAP_PERFMON: {}", e);
                false
            }
        }
    }

    #[cfg(not(target_os = "linux"))]
    {
        false
    }
}

/// Check if the process has both required eBPF capabilities
///
/// # Returns
///
/// `true` if both CAP_BPF and CAP_PERFMON are available, `false` otherwise
pub fn has_ebpf_capabilities() -> bool {
    has_cap_bpf() && has_cap_perfmon()
}

/// Get the kernel version by parsing /proc/sys/kernel/osrelease
///
/// # Returns
///
/// `Some((major, minor))` if kernel version can be parsed, `None` otherwise
///
/// # Example
///
/// ```no_run
/// if let Some((major, minor)) = clio_cte::capability_detector::get_kernel_version() {
///     println!("Kernel version: {}.{}", major, minor);
/// }
/// ```
pub fn get_kernel_version() -> Option<(u32, u32)> {
    // Try to read from /proc/sys/kernel/osrelease
    let osrelease = fs::read_to_string("/proc/sys/kernel/osrelease")
        .or_else(|_| fs::read_to_string("/proc/version"))
        .ok()?;

    // Parse version string (e.g., "5.15.0-91-generic" -> (5, 15))
    let parts: Vec<&str> = osrelease.trim().split('.').collect();
    if parts.len() >= 2 {
        let major = parts[0].parse::<u32>().ok()?;
        let minor = parts[1].parse::<u32>().ok()?;
        Some((major, minor))
    } else {
        None
    }
}

/// Check if the kernel version meets the minimum requirement for eBPF ring buffers
///
/// Kernel version 5.8+ is required for BPF ring buffers, which provide
/// the best performance for eBPF-based interception.
///
/// # Arguments
///
/// * `kernel_version` - Optional kernel version tuple (major, minor)
///
/// # Returns
///
/// `true` if kernel version is 5.8 or higher, `false` otherwise
pub fn check_kernel_version(kernel_version: Option<(u32, u32)>) -> bool {
    match kernel_version {
        Some((major, minor)) => {
            if major > 5 {
                true
            } else if major == 5 {
                minor >= 8
            } else {
                false
            }
        }
        None => false,
    }
}

/// Check if the BPF filesystem is mounted and accessible
///
/// The /sys/fs/bpf filesystem must be mounted for eBPF programs to
/// pin maps and programs.
///
/// # Returns
///
/// `true` if /sys/fs/bpf exists and is writable, `false` otherwise
pub fn check_bpf_filesystem() -> bool {
    let bpf_path = Path::new("/sys/fs/bpf");

    // Check if directory exists
    if !bpf_path.exists() {
        return false;
    }

    // Check if it's a directory
    if !bpf_path.is_dir() {
        return false;
    }

    // Check if we can write to it (try to create a temporary file)
    // This is a simple check - in practice, eBPF programs need proper permissions
    let test_path = bpf_path.join(".write_test");
    let can_write = fs::write(&test_path, "").is_ok();

    // Clean up test file if it was created
    let _ = fs::remove_file(&test_path);

    can_write
}

/// Detect the best interception mode based on system capabilities
///
/// This function performs a comprehensive check of eBPF capabilities:
/// - CAP_BPF and CAP_PERFMON capabilities
/// - Kernel version (requires 5.8+ for ring buffers)
/// - BPF filesystem mount status
///
/// # Returns
///
/// `InterceptorMode::Ebpf` if all eBPF requirements are met,
/// `InterceptorMode::LdPreload` otherwise
///
/// # Example
///
/// ```no_run
/// use clio_cte::capability_detector::{detect_best_mode, InterceptorMode};
///
/// let mode = detect_best_mode();
/// match mode {
///     InterceptorMode::Ebpf => {
///         // Initialize eBPF-based interceptor
///     }
///     InterceptorMode::LdPreload => {
///         // Fall back to LD_PRELOAD-based interceptor
///     }
/// }
/// ```
pub fn detect_best_mode() -> InterceptorMode {
    let info = EbpfCapabilityInfo::new();

    if info.is_supported {
        InterceptorMode::Ebpf
    } else {
        InterceptorMode::LdPreload
    }
}

/// Get detailed information about eBPF capability status
///
/// This function provides comprehensive information about why eBPF
/// is or isn't available on the current system.
///
/// # Returns
///
/// `EbpfCapabilityInfo` with detailed capability information
///
/// # Example
///
/// ```no_run
/// use clio_cte::capability_detector::get_ebpf_capability_info;
///
/// let info = get_ebpf_capability_info();
/// if !info.is_supported {
///     eprintln!("eBPF not available: {}", info.reason.unwrap_or_default());
/// }
/// ```
pub fn get_ebpf_capability_info() -> EbpfCapabilityInfo {
    EbpfCapabilityInfo::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_kernel_version() {
        // Just verify it doesn't panic
        let version = get_kernel_version();
        println!("Kernel version: {:?}", version);
    }

    #[test]
    fn test_check_kernel_version() {
        // Test version comparisons
        assert!(check_kernel_version(Some((5, 8))));
        assert!(check_kernel_version(Some((5, 9))));
        assert!(check_kernel_version(Some((5, 15))));
        assert!(check_kernel_version(Some((6, 0))));
        assert!(check_kernel_version(Some((6, 1))));

        assert!(!check_kernel_version(Some((5, 7))));
        assert!(!check_kernel_version(Some((4, 19))));
        assert!(!check_kernel_version(Some((3, 10))));
        assert!(!check_kernel_version(None));
    }

    #[test]
    fn test_capability_info_new() {
        let info = EbpfCapabilityInfo::new();

        // Verify all fields are populated
        println!("eBPF Capability Info:");
        println!("  is_supported: {}", info.is_supported);
        println!("  has_cap_bpf: {}", info.has_cap_bpf);
        println!("  has_cap_perfmon: {}", info.has_cap_perfmon);
        println!("  kernel_version: {:?}", info.kernel_version);
        println!("  kernel_version_ok: {}", info.kernel_version_ok);
        println!("  bpf_fs_mounted: {}", info.bpf_fs_mounted);
        println!("  reason: {:?}", info.reason);

        // Verify consistency
        if info.is_supported {
            assert!(info.reason.is_none());
            assert!(info.has_cap_bpf);
            assert!(info.has_cap_perfmon);
            assert!(info.kernel_version_ok);
            assert!(info.bpf_fs_mounted);
        }
    }

    #[test]
    fn test_detect_best_mode() {
        let mode = detect_best_mode();
        println!("Detected best mode: {}", mode);

        // Verify mode is valid
        match mode {
            InterceptorMode::Ebpf | InterceptorMode::LdPreload => {}
        }
    }

    #[test]
    fn test_interceptor_mode_display() {
        assert_eq!(format!("{}", InterceptorMode::Ebpf), "eBPF");
        assert_eq!(format!("{}", InterceptorMode::LdPreload), "LD_PRELOAD");
    }
}
