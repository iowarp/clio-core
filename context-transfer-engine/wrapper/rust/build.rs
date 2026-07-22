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

use std::env;
use std::path::{Path, PathBuf};

/// Parses a semicolon-separated list of library specifications.
/// Each item can be:
/// - A library name (e.g., "zmq")
/// - A full path to a library (e.g., "/usr/lib/libzmq.so.5")
/// - A colon-separated list for static linking (e.g., "zmq;stdc++;gcc_s")
fn parse_zmq_libs(libs_var: &str) -> Vec<String> {
    libs_var
        .split(';')
        .filter(|s| !s.is_empty())
        .map(|s| s.trim().to_string())
        .collect()
}

/// Parses a colon-separated list of library directories.
fn parse_zmq_lib_dirs(dirs_var: &str) -> Vec<String> {
    dirs_var
        .split(':')
        .filter(|s| !s.is_empty())
        .map(|s| s.trim().to_string())
        .collect()
}

/// Determines if a string is a path (contains / or \ or .)
fn is_library_path(spec: &str) -> bool {
    spec.contains('/') || spec.contains('\\') || spec.contains('.')
}

/// Links a library by name (extracts library name from path if needed).
fn link_library(lib_spec: &str) {
    if is_library_path(lib_spec) {
        // It's a path - we need to extract the library name
        let path = Path::new(lib_spec);
        let filename = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(lib_spec);

        // Remove common library prefixes and suffixes
        // libzmq.so.5 -> zmq
        // libzmq.so -> zmq
        // zmq.so -> zmq
        let lib_name = filename
            .strip_prefix("lib")
            .unwrap_or(filename)
            .split('.')
            .next()
            .unwrap_or(filename);

        // Add the directory containing the library to search path
        if let Some(parent) = path.parent() {
            if let Some(parent_str) = parent.to_str() {
                if !parent_str.is_empty() {
                    println!("cargo:rustc-link-search=native={}", parent_str);
                }
            }
        }

        println!("cargo:rustc-link-lib=dylib={}", lib_name);
    } else {
        // It's just a library name
        println!("cargo:rustc-link-lib=dylib={}", lib_spec);
    }
}

/// Attempts to link ZMQ from environment variables set by CMake.
/// Returns true if successful, false if fallback should be used.
fn try_link_zmq_from_cmake() -> bool {
    let zmq_libs = env::var("IOWARP_ZMQ_LIBS").unwrap_or_default();
    let zmq_lib_dirs = env::var("IOWARP_ZMQ_LIB_DIRS").unwrap_or_default();

    if zmq_libs.is_empty() {
        return false;
    }

    // Add library directories to search path
    for dir in parse_zmq_lib_dirs(&zmq_lib_dirs) {
        println!("cargo:rustc-link-search=native={}", dir);
    }

    // Link each library specification
    let libs = parse_zmq_libs(&zmq_libs);
    if libs.is_empty() {
        return false;
    }

    for lib in libs {
        link_library(&lib);
    }

    true
}

/// Attempts to link ZMQ using common library names for standalone cargo builds.
/// Tries common naming conventions as fallback.
fn link_zmq_fallback() {
    // Common ZMQ library names to try
    let fallback_names = vec!["zmq", "zmq:5"];

    // Use first candidate (simpler than loop that always breaks)
    if let Some(name) = fallback_names.first() {
        println!("cargo:rustc-link-lib=dylib={}", name);
    } else {
        panic!(
            "ZeroMQ (libzmq) not found. Please either:\n\
             1. Build with CMake which will set IOWARP_ZMQ_LIBS and IOWARP_ZMQ_LIB_DIRS\n\
             2. Install libzmq development package:\n\
                - Ubuntu/Debian: sudo apt-get install libzmq3-dev\n\
                - CentOS/RHEL: sudo yum install zeromq-devel\n\
                - macOS: brew install zeromq\n\
             3. Set environment variables manually:\n\
                export IOWARP_ZMQ_LIBS=zmq\n\
                export IOWARP_ZMQ_LIB_DIRS=/usr/local/lib\n\
             4. Set IOWARP_LIB_DIR to directory containing libzmq.so\n"
        );
    }
}

fn main() {
    use std::io::Write;
    // Debug: Print all environment variables at the start
    eprintln!("=== BUILD.RS START ===");
    eprintln!("Expected environment variables (set by CMake/Corrosion):");
    eprintln!("  IOWARP_INCLUDE_DIR: Primary include directory (clio_ctp_host)");
    eprintln!("  IOWARP_EXTRA_INCLUDES: Colon-separated extra includes (clio_runtime, CTE, etc.)");
    eprintln!("  IOWARP_LIB_DIR: Library directory for linking");
    eprintln!("  IOWARP_ZMQ_LIBS: ZeroMQ library specifications");
    eprintln!("  IOWARP_ZMQ_LIB_DIRS: ZeroMQ library directories");

    // Write all env vars to a file for debugging
    let mut file = std::fs::File::create("/tmp/cargo_env_vars.txt").expect("Failed to create file");
    for (key, value) in std::env::vars() {
        writeln!(file, "{} = {}", key, value).expect("Failed to write");
    }
    file.flush().expect("Failed to flush");
    eprintln!("Wrote env vars to /tmp/cargo_env_vars.txt");

    eprintln!("Path to env vars file: /tmp/cargo_env_vars.txt");

    // Get include and library paths from environment (set by CMake/Corrosion)
    // Fall back to defaults computed from repository layout for standalone cargo builds
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let manifest_path = PathBuf::from(&manifest_dir);
    
    // Helper to resolve paths robustly - always returns absolute paths
    let resolve_path = |rel_path: &str| -> String {
        // Get absolute path of manifest directory
        let abs_manifest = if manifest_path.is_absolute() {
            manifest_path.clone()
        } else {
            std::env::current_dir()
                .unwrap_or_else(|_| PathBuf::from("."))
                .join(&manifest_path)
        };
        
        // Join the relative path
        let joined = abs_manifest.join(rel_path);
        
        // Try to canonicalize (requires path to exist)
        if let Ok(canonical) = joined.canonicalize() {
            return canonical.to_string_lossy().to_string();
        }
        
        // If canonicalize fails, use dunce or similar to normalize
        // For now, just return the joined path - C++ compiler can handle .. in paths
        joined.to_string_lossy().to_string()
    };
    
    let include_dir = match std::env::var("IOWARP_INCLUDE_DIR") {
        Ok(val) => {
            eprintln!("DEBUG: IOWARP_INCLUDE_DIR = {} (from CMake)", val);
            val
        }
        Err(_) => {
            eprintln!("DEBUG: IOWARP_INCLUDE_DIR not set, computing from CARGO_MANIFEST_DIR");
            // From wrapper/rust, need ../../../context-transport-primitives/include
            let computed = resolve_path("../../../context-transport-primitives/include");
            eprintln!("DEBUG: Computed include_dir = {}", computed);
            computed
        }
    };
    
    let lib_dir = match std::env::var("IOWARP_LIB_DIR") {
        Ok(val) => {
            eprintln!("DEBUG: IOWARP_LIB_DIR = {} (from CMake)", val);
            val
        }
        Err(_) => {
            eprintln!("DEBUG: IOWARP_LIB_DIR not set, trying CMAKE_BINARY_DIR");
            // Try CMAKE_BINARY_DIR first
            if let Ok(cmake_bin_dir) = env::var("CMAKE_BINARY_DIR") {
                let lib_dir = format!("{}/bin", cmake_bin_dir);
                eprintln!("DEBUG: Using CMAKE_BINARY_DIR/bin = {}", lib_dir);
                lib_dir
            } else {
                // Fallback: compute from CARGO_MANIFEST_DIR using absolute path
                eprintln!("DEBUG: CMAKE_BINARY_DIR not set, computing from CARGO_MANIFEST_DIR");
                let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
                // From wrapper/rust, go up to clio-core root, then into build/bin
                // Use canonicalize to get absolute path
                let manifest_path = std::path::PathBuf::from(&manifest_dir);
                let build_dir = manifest_path.parent() // wrapper
                    .and_then(|p| p.parent()) // context-transfer-engine
                    .and_then(|p| p.parent()) // clio-core
                    .map(|p| p.join("build"))
                    .unwrap_or_else(|| manifest_path.join("../../build"));
                let lib_dir = build_dir.join("bin").to_string_lossy().to_string();
                eprintln!("DEBUG: Using computed lib_dir = {}", lib_dir);
                lib_dir
            }
        }
    };

    // Additional include paths for clio_runtime and other dependencies
    // Multiple paths separated by colons
    let extra_includes = match std::env::var("IOWARP_EXTRA_INCLUDES") {
        Ok(val) => {
            eprintln!("DEBUG: IOWARP_EXTRA_INCLUDES = {} (from CMake)", val);
            val
        }
        Err(_) => {
            eprintln!("DEBUG: IOWARP_EXTRA_INCLUDES not set, computing from repository layout");
            // From wrapper/rust: ../../../context-runtime/include and ../../core/include
            let runtime_include = resolve_path("../../../context-runtime/include");
            let cte_include = resolve_path("../../core/include");
            // Also add admin and bdev module includes
            let admin_include = resolve_path("../../../context-runtime/modules/admin/include");
            let bdev_include = resolve_path("../../../context-runtime/modules/bdev/include");
            eprintln!("DEBUG: Computed runtime_include = {}", runtime_include);
            eprintln!("DEBUG: Computed cte_include = {}", cte_include);
            eprintln!("DEBUG: Computed admin_include = {}", admin_include);
            eprintln!("DEBUG: Computed bdev_include = {}", bdev_include);
            format!("{}:{}:{}:{}", runtime_include, cte_include, admin_include, bdev_include)
        }
    };

    // Split extra_includes and add fallback paths for cereal if not present
    let mut include_paths: Vec<String> = extra_includes
        .split(':')
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect();

    // Check if cereal is already in the include paths
    let has_cereal = include_paths.iter().any(|path| {
        let cereal_path = format!("{}/cereal", path);
        std::path::Path::new(&cereal_path).exists()
    });

    // Add cereal fallback paths if not already present
    if !has_cereal {
        eprintln!("DEBUG: Cereal not found in IOWARP_EXTRA_INCLUDES, adding fallback paths");

        // Primary fallback: Check the build directory for CMake-fetched cereal
        let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
        let workspace_root = format!("{}/../../..", manifest_dir);

        let cereal_fallbacks = vec![
            // CMake FetchContent location
            format!("{}/build/_deps/cereal-src/include", workspace_root),
            // System locations
            "/usr/local/include".to_string(),
            "/usr/include".to_string(),
        ];

        for fallback in cereal_fallbacks {
            let cereal_path = format!("{}/cereal", fallback);
            if std::path::Path::new(&cereal_path).exists() {
                eprintln!("DEBUG: Found cereal at {}", fallback);
                include_paths.push(fallback);
            }
        }
    }

    // Build the CXX bridge and C++ shim
    let mut build = cxx_build::bridge("src/ffi.rs");
    build
        .file("shim/shim.cc")
        .std("c++20")
        // Coroutine support
        .flag("-fcoroutines")
        // Include paths
        .include(&include_dir)
        .include(".") // for shim/shim.h
        // CTP defines (required for thread model manager)
        .define("CTP_ENABLE_PTHREADS", "1")
        .define("CTP_ENABLE_OPENMP", "0")
        .define("CTP_ENABLE_WINDOWS_THREADS", "0")
        .define("CTP_DEFAULT_THREAD_MODEL", "ctp::thread::Pthread")
        .define("CTP_DEFAULT_THREAD_MODEL_GPU", "ctp::thread::StdThread")
        .define("CTP_LOG_LEVEL", "1")
        // HSHM defines (match CMake build)
        .define("HSHM_ENABLE_CEREAL", "1")
        .define("HSHM_ENABLE_ZMQ", "1")
        .define("HSHM_ENABLE_PTHREADS", "1")
        .define("HSHM_ENABLE_OPENMP", "0")
        .define("HSHM_ENABLE_WINDOWS_THREADS", "0")
        .define("HSHM_DEFAULT_THREAD_MODEL", "hshm::thread::Pthread")
        .define("HSHM_DEFAULT_THREAD_MODEL_GPU", "hshm::thread::StdThread")
        .define("HSHM_LOG_LEVEL", "0")
        // Suppress warnings from CTE/clio_runtime headers
        .flag("-Wno-unused-parameter")
        .flag("-Wno-unused-variable")
        .flag("-Wno-missing-field-initializers")
        .flag("-Wno-sign-compare")
        .flag("-Wno-reorder")
        .flag("-Wno-pedantic");

    // Add extra include directories
    eprintln!("DEBUG: Adding extra include directories:");
    for path in &include_paths {
        eprintln!("  - {}", path);
        build.include(path);
    }

    // Debug: Print all include paths being used
    eprintln!("DEBUG: Primary include directory: {}", include_dir);
    eprintln!("DEBUG: Library directory: {}", lib_dir);

    build.compile("cte_shim");

    // Library search paths
    println!("cargo:rustc-link-search=native={}", lib_dir);

    // Link to CTE and dependencies
    // Use actual target names, not legacy aliases:
    // - clio_run_cxx (not chimaera_cxx, which is just an alias)
    // - clio_ctp_host (not hermes_shm_host, which is the old name)
    println!("cargo:rustc-link-lib=dylib=clio_cte_core_client");
    println!("cargo:rustc-link-lib=dylib=clio_run_cxx");
    println!("cargo:rustc-link-lib=dylib=clio_ctp_host");

    // Dynamic ZMQ linking with CMake environment variables or fallback
    if !try_link_zmq_from_cmake() {
        link_zmq_fallback();
    }

    // RPATH for relocatable builds
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir);

    // Rebuild triggers
    println!("cargo:rerun-if-changed=shim/shim.h");
    println!("cargo:rerun-if-changed=shim/shim.cc");
    println!("cargo:rerun-if-changed=src/ffi.rs");
    println!("cargo:rerun-if-env-changed=IOWARP_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=IOWARP_LIB_DIR");
    println!("cargo:rerun-if-env-changed=IOWARP_EXTRA_INCLUDES");
    println!("cargo:rerun-if-env-changed=IOWARP_ZMQ_LIBS");
    println!("cargo:rerun-if-env-changed=IOWARP_ZMQ_LIB_DIRS");
}
