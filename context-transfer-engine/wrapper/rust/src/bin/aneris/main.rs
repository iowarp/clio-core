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
 *    contributors may be used to promote products derived from
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

//! Aneris Profiler - Combined Subprocess + Telemetry Capture
//!
//! This binary spawns a subprocess with CTE I/O interception and captures
//! telemetry in real-time, displaying results as they occur.
//!
//! Usage:
//!   LD_LIBRARY_PATH=~/clio-core/build/bin:$LD_LIBRARY_PATH \
//!     CHI_WITH_RUNTIME=1 \
//!     aneris-profiler [OPTIONS] <command> [args...]
//!
//! Example:
//!   aneris-profiler ior -t 1m -b 16m -s 16
//!
//! Options:
//!   --poll-interval-ms <ms>    Poll interval when data is active (default: 10)
//!   --idle-interval-ms <ms>    Poll interval when idle (default: 100)
//!   --realtime                 Show telemetry in real-time (default: summary at end)
//!   --help                     Show this help message

use std::env;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use clio_cte::sync::init;
use clio_cte::sync::Client;

/// Parse command-line arguments, separating profiler options from the command
fn parse_args() -> (Option<String>, Vec<String>, u64, u64, bool, bool) {
    let args: Vec<String> = env::args().collect();

    let mut command: Option<String> = None;
    let mut command_args: Vec<String> = Vec::new();
    let mut poll_interval_ms: u64 = 10; // Default: 10ms when active
    let mut idle_interval_ms: u64 = 100; // Default: 100ms when idle
    let mut realtime_telemetry = false;
    let mut no_telemetry = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                eprintln!("Aneris Profiler - Combined Subprocess + Telemetry Capture");
                eprintln!();
                eprintln!("Usage: {} [OPTIONS] <command> [args...]", args[0]);
                eprintln!();
                eprintln!("Options:");
                eprintln!(
                    "  --poll-interval-ms <ms>   Poll interval when data is active (default: 10)"
                );
                eprintln!("  --idle-interval-ms <ms>   Poll interval when idle (default: 100)");
                eprintln!("  --realtime                Show telemetry in real-time");
                eprintln!("  --no-telemetry            Disable telemetry, intercept only");
                eprintln!("  --help, -h                Show this help message");
                eprintln!();
                eprintln!("Example:");
                eprintln!("  {} ior -t 1m -b 16m -s 16", args[0]);
                std::process::exit(0);
            }
            "--poll-interval-ms" => {
                i += 1;
                if i < args.len() {
                    poll_interval_ms = args[i].parse().unwrap_or(10);
                }
            }
            "--idle-interval-ms" => {
                i += 1;
                if i < args.len() {
                    idle_interval_ms = args[i].parse().unwrap_or(100);
                }
            }
            "--realtime" => {
                realtime_telemetry = true;
            }
            "--no-telemetry" => {
                no_telemetry = true;
            }
            _ if command.is_none() => {
                // First non-option argument is the command
                if !args[i].starts_with("--") {
                    command = Some(args[i].clone());
                }
            }
            _ => {
                // Subsequent arguments are command arguments
                command_args.push(args[i].clone());
            }
        }
        i += 1;
    }

    (
        command,
        command_args,
        poll_interval_ms,
        idle_interval_ms,
        realtime_telemetry,
        no_telemetry,
    )
}

fn main() {
    // Parse command line
    let (
        executable_opt,
        exec_args,
        poll_interval_ms,
        idle_interval_ms,
        realtime_telemetry,
        no_telemetry,
    ) = parse_args();

    if executable_opt.is_none() {
        eprintln!("Error: No command specified");
        eprintln!("Usage: aneris-profiler [OPTIONS] <command> [args...]");
        eprintln!("Run 'aneris-profiler --help' for more information");
        std::process::exit(1);
    }

    let executable = executable_opt.unwrap();

    if no_telemetry {
        println!("=== Aneris Interceptor (No Telemetry) ===");
    } else if realtime_telemetry {
        println!("=== Aneris Profiler (Real-Time Mode) ===");
    } else {
        println!("=== Aneris Profiler ===");
    }
    println!("Executable: {}", executable);
    if !no_telemetry {
        println!("Poll interval (active): {}ms", poll_interval_ms);
        println!("Poll interval (idle): {}ms", idle_interval_ms);
    }
    println!("");

    // Initialize CTE
    println!("[1/2] Initializing CTE runtime...");
    if let Err(e) = init("") {
        eprintln!("Failed to initialize CTE: {}", e);
        std::process::exit(1);
    }
    println!("      ✓ CTE runtime initialized\n");

    // Get build directory with multiple fallback strategies
    let build_dir = env::var("IOWARP_BUILD_DIR")
        .or_else(|_| env::var("CMAKE_BINARY_DIR"))
        .unwrap_or_else(|_| {
            // Try to detect from current executable path
            env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|p| p.to_string_lossy().to_string()))
                .unwrap_or_else(|| "/tmp".to_string())
        });

    // The POSIX adapter is in bin/ not lib/
    // But build_dir might already be the bin directory if detected from current_exe
    let posix_adapter = if build_dir.ends_with("/bin") || build_dir.ends_with("/bin/") {
        format!("{}/libclio_cte_posix.so", build_dir)
    } else {
        format!("{}/bin/libclio_cte_posix.so", build_dir)
    };

    // Check adapter
    if !std::path::Path::new(&posix_adapter).exists() {
        eprintln!("[!] Warning: POSIX adapter not found at {}", posix_adapter);
        eprintln!("    I/O interception will not work.");
    } else {
        println!("[✓] POSIX adapter found");
    }

    // Give runtime time to initialize
    std::thread::sleep(Duration::from_millis(100));

    // Spawn subprocess with LD_PRELOAD
    if no_telemetry {
        println!("[2/2] Starting subprocess with I/O interception...");
    } else {
        println!("[2/3] Starting subprocess with I/O interception...");
    }
    let mut child = Command::new(&executable)
        .args(&exec_args)
        .env("LD_PRELOAD", &posix_adapter)
        .env_remove("CHI_WITH_RUNTIME") // Child should NOT start its own runtime
        .env(
            "LD_LIBRARY_PATH",
            format!(
                "{}:{}",
                if build_dir.ends_with("/bin") || build_dir.ends_with("/bin/") {
                    build_dir.clone()
                } else {
                    format!("{}/bin", build_dir)
                },
                env::var("LD_LIBRARY_PATH").unwrap_or_default()
            ),
        )
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("Failed to spawn subprocess");

    println!("      ✓ Subprocess started (PID: {})\n", child.id());

    if !no_telemetry {
        // Set up signal handler for graceful shutdown
        let running = Arc::new(AtomicBool::new(true));
        let running_clone = Arc::clone(&running);

        // NOTE: Signal handling would require additional crates like ctrlc
        // For now, we just poll until subprocess finishes

        // Spawn telemetry polling thread
        // Create NEW Client inside thread (Client is !Send, cannot move across threads)
        let poll_handle = thread::spawn(move || {
            // Create a new client for this thread (Client is !Send, must be created here)
            let mut client = match Client::new() {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("Telemetry polling thread: Failed to create client: {}", e);
                    return (0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
                }
            };

            let mut last_time: u64 = 0;
            let mut total_ops: u64 = 0;
            let mut total_bytes: u64 = 0;
            let mut write_bytes: u64 = 0;
            let mut read_bytes: u64 = 0;
            let mut entries_count: u64 = 0;

            while running_clone.load(Ordering::Relaxed) {
                // O(1) telemetry polling using timeout=0 to check availability
                // - Timeout code (1) means no data available, skip processing
                // - Success code (0) means data exists, process immediately
                // - Error code (2 or other) means runtime error
                //
                // Performance:
                // - Empty check is O(1) ~50 cycles vs polling O(1000 cycles)
                // - Avoids unnecessary poll calls when no data available
                // - Adaptive sleep: shorter when active, longer when idle
                match client.poll_telemetry(last_time, 0.0) {
                    Ok(entries) => {
                        if entries.is_empty() {
                            // No new data - use idle interval
                            thread::sleep(Duration::from_millis(idle_interval_ms));
                            continue;
                        }

                        entries_count += entries.len() as u64;

                        for entry in &entries {
                            if realtime_telemetry {
                                // Real-time output: each entry on its own line
                                println!(
                                    "[{:<20}] size={:<12} tag={}.{} logical={}",
                                    format!("{:?}", entry.op),
                                    entry.size,
                                    entry.tag_id.major,
                                    entry.tag_id.minor,
                                    entry.logical_time
                                );
                            }

                            last_time = entry.logical_time;
                            total_ops += 1;
                            total_bytes += entry.size;

                            // Track read/write separately
                            match entry.op {
                                clio_cte::ffi::CteOp::PutBlob => write_bytes += entry.size,
                                clio_cte::ffi::CteOp::GetBlob => read_bytes += entry.size,
                                _ => {}
                            }
                        }

                        // Data was found - use active interval
                        thread::sleep(Duration::from_millis(poll_interval_ms));
                    }
                    Err(clio_cte::CteError::Timeout) => {
                        // No data available - use idle interval
                        thread::sleep(Duration::from_millis(idle_interval_ms));
                    }
                    Err(e) => {
                        // Runtime error - continue with idle interval
                        eprintln!("Telemetry poll error: {}", e);
                        thread::sleep(Duration::from_millis(idle_interval_ms));
                    }
                }
            }

            // Return collected statistics
            (
                entries_count,
                total_ops,
                total_bytes,
                write_bytes,
                read_bytes,
                last_time,
            )
        });

        // Wait for subprocess
        let status = child.wait().expect("Failed to wait for subprocess");

        // Signal telemetry thread to stop
        running.store(false, Ordering::Relaxed);

        // Wait for telemetry thread to finish
        let (final_entries, final_ops, final_bytes, final_write, final_read, final_time) =
            poll_handle.join().expect("Telemetry thread panicked");

        // Give runtime time to catch final operations
        std::thread::sleep(Duration::from_millis(500));

        // Display summary
        println!("\n=== Telemetry Summary ===");

        if final_entries == 0 {
            println!("No telemetry entries captured.");
            println!(
                "This is normal if the subprocess completed before telemetry could be captured."
            );
        } else {
            println!("Captured {} telemetry entries\n", final_entries);

            // Display telemetry table
            println!(
                "{:<20} {:>12} {:>20}",
                "Operation", "Size (bytes)", "Tag ID"
            );
            println!("{}", "-".repeat(60));

            // Create client for final poll
            let client = match Client::new() {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("Failed to create client for final poll: {}", e);
                    std::process::exit(1);
                }
            };

            // Final poll with timeout to catch remaining entries
            match client.poll_telemetry(final_time, 5.0) {
                Ok(telemetry) => {
                    for entry in &telemetry {
                        println!(
                            "{:<20} {:>12} {:>20}",
                            format!("{:?}", entry.op),
                            entry.size,
                            format!("{}.{}", entry.tag_id.major, entry.tag_id.minor)
                        );
                    }
                }
                Err(_) => {}
            }

            // Summary statistics
            let avg_size = if final_ops > 0 {
                final_bytes / final_ops
            } else {
                0
            };

            println!("\n{}", "-".repeat(60));
            println!("=== Summary ===");
            println!("Total operations: {}", final_ops);
            println!(
                "Total data transferred: {} bytes ({} MB)",
                final_bytes,
                final_bytes / (1024 * 1024)
            );
            println!(
                "  - Writes: {} bytes ({} MB)",
                final_write,
                final_write / (1024 * 1024)
            );
            println!(
                "  - Reads: {} bytes ({} MB)",
                final_read,
                final_read / (1024 * 1024)
            );
            println!("Average size: {} bytes", avg_size);
        }

        println!("\nSubprocess exited with: {:?}", status.code());
    } else {
        // NO TELEMETRY: Just wait for subprocess
        let status = child.wait().expect("Failed to wait for subprocess");
        println!("\nSubprocess exited with: {:?}", status.code());
    }
}
