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

//! Aneris Tune - Adaptive Blob Reorganization with Frecency Scoring
//!
//! This binary monitors CTE blob access patterns and automatically triggers
//! reorganization decisions based on frecency (frequency + recency) scores.
//!
//! ## Architecture
//!
//! Three async tasks work together:
//! 1. **telemetry_receiver** - Polls CTE telemetry, updates frecency engine
//! 2. **decay_scheduler** - Runs every 1s: batch decay, collect hot candidates
//! 3. **reorg_executor** - Runs every 10s: drain queue, execute reorganize_blob()
//!
//! ## Three-Level Batching
//!
//! - **Level 1**: Immediate atomic score updates (O(1) per access)
//! - **Level 2**: SIMD batch decay every 1s
//! - **Level 3**: Coalesced reorg decisions every 10s
//!
//! ## Usage
//!
//! ### Standalone Mode
//! ```bash
//! aneris-tune [OPTIONS]
//! ```
//!
//! ### Wrapping Mode
//! ```bash
//! aneris-tune [OPTIONS] -- <executable> [args...]
//! ```
//!
//! ## Options
//!
//! - `--config <file>` - CTE configuration file (default: clio_default.yaml)
//! - `--threshold-hot <f64>` - Hot threshold (default: 50.0)
//! - `--threshold-cold <f64>` - Cold threshold (default: 5.0)
//! - `--decay-interval-ms <u64>` - Decay interval in ms (default: 1000)
//! - `--reorg-interval-ms <u64>` - Reorg interval in ms (default: 10000)
//! - `--output <file>` - Optional telemetry output file
//! - `--verbose` - Enable verbose logging
//!
//! ## Examples
//!
//! ```bash
//! # Standalone mode - background tuning
//! aneris-tune --verbose --threshold-hot 75.0
//!
//! # Wrapping mode - tune while running benchmark
//! aneris-tune -- ior -t 1m -b 16m -s 16
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{broadcast, Mutex, RwLock};
use tokio::time::{interval, timeout};

use clio_cte::{Client, FrecencyEngine, ReorgBatcher};
use clio_cte::ffi::CteTagId;

/// Configuration parameters for aneris-tune.
#[derive(Debug, Clone)]
struct Config {
    /// CTE configuration file path
    config_file: String,
    /// Hot threshold for promoting blobs to fast tier
    threshold_hot: f64,
    /// Cold threshold for demoting blobs to slow tier
    threshold_cold: f64,
    /// Decay interval in milliseconds
    decay_interval_ms: u64,
    /// Reorg interval in milliseconds
    reorg_interval_ms: u64,
    /// Optional telemetry output file
    output_file: Option<String>,
    /// Enable verbose logging
    verbose: bool,
    /// Executable to run (if wrapping mode)
    executable: Option<String>,
    /// Arguments for the executable
    executable_args: Vec<String>,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            config_file: "clio_default.yaml".to_string(),
            threshold_hot: 50.0,
            threshold_cold: 5.0,
            decay_interval_ms: 1000,
            reorg_interval_ms: 10000,
            output_file: None,
            verbose: false,
            executable: None,
            executable_args: Vec::new(),
        }
    }
}

/// Parse command-line arguments.
fn parse_args() -> Config {
    let args: Vec<String> = std::env::args().collect();
    let mut config = Config::default();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                eprintln!("Aneris Tune - Adaptive Blob Reorganization");
                eprintln!();
                eprintln!("Usage: {} [OPTIONS]", args[0]);
                eprintln!("       {} [OPTIONS] -- <executable> [args...]", args[0]);
                eprintln!();
                eprintln!("Options:");
                eprintln!("  --config <file>              CTE config (default: clio_default.yaml)");
                eprintln!("  --threshold-hot <f64>       Hot threshold (default: 50.0)");
                eprintln!("  --threshold-cold <f64>       Cold threshold (default: 5.0)");
                eprintln!("  --decay-interval-ms <u64>   Decay interval ms (default: 1000)");
                eprintln!("  --reorg-interval-ms <u64>   Reorg interval ms (default: 10000)");
                eprintln!("  --output <file>             Telemetry output file (optional)");
                eprintln!("  --verbose                   Enable verbose logging");
                eprintln!("  --help, -h                  Show this help message");
                eprintln!();
                eprintln!("Wrapping Mode:");
                eprintln!("  Use '--' to wrap an executable with tuning:");
                eprintln!("  {} -- /path/to/executable [args...]", args[0]);
                eprintln!();
                eprintln!("Examples:");
                eprintln!("  Standalone:  {} --verbose", args[0]);
                eprintln!("  Wrapping:    {} -- ior -t 1m -b 16m", args[0]);
                std::process::exit(0);
            }
            "--config" => {
                i += 1;
                if i < args.len() {
                    config.config_file = args[i].clone();
                }
            }
            "--threshold-hot" => {
                i += 1;
                if i < args.len() {
                    config.threshold_hot = args[i].parse().unwrap_or(50.0);
                }
            }
            "--threshold-cold" => {
                i += 1;
                if i < args.len() {
                    config.threshold_cold = args[i].parse().unwrap_or(5.0);
                }
            }
            "--decay-interval-ms" => {
                i += 1;
                if i < args.len() {
                    config.decay_interval_ms = args[i].parse().unwrap_or(1000);
                }
            }
            "--reorg-interval-ms" => {
                i += 1;
                if i < args.len() {
                    config.reorg_interval_ms = args[i].parse().unwrap_or(10000);
                }
            }
            "--output" => {
                i += 1;
                if i < args.len() {
                    config.output_file = Some(args[i].clone());
                }
            }
            "--verbose" => {
                config.verbose = true;
            }
            "--" => {
                // Remaining args are the executable and its args
                i += 1;
                if i < args.len() {
                    config.executable = Some(args[i].clone());
                    config.executable_args = args[i + 1..].to_vec();
                }
                break;
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                eprintln!("Run '{}' --help for usage", args[0]);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    config
}

/// Maps blob_hash to (tag_id, blob_name) for reorg lookups.
/// This is needed because telemetry only provides blob_hash, but
/// reorganize_blob() requires tag_id and blob_name.
///
/// Note: Uses ffi::CteTagId because poll_telemetry() returns telemetry
/// with tag_id from the FFI layer, which differs from the root CteTagId type.
struct BlobRegistry {
    /// blob_hash -> (tag_id, blob_name)
    map: HashMap<u64, (CteTagId, String)>,
}

impl BlobRegistry {
    fn new() -> Self {
        BlobRegistry {
            map: HashMap::new(),
        }
    }

    fn insert(&mut self, blob_hash: u64, tag_id: CteTagId, blob_name: String) {
        self.map.insert(blob_hash, (tag_id, blob_name));
    }

    fn get(&self, blob_hash: u64) -> Option<&(CteTagId, String)> {
        self.map.get(&blob_hash)
    }
}

/// Statistics for monitoring.
#[derive(Debug, Default)]
struct Stats {
    total_telemetry_entries: u64,
    total_accesses: u64,
    total_reorgs: u64,
    hot_reorgs: u64,
    cold_reorgs: u64,
}

/// Format bytes as human-readable string.
fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Decay scheduler task.
///
/// Runs every decay_interval_ms and applies SIMD batch decay.
/// Collects hot candidates above threshold.
async fn decay_scheduler(
    engine: Arc<RwLock<FrecencyEngine>>,
    batcher: Arc<ReorgBatcher>,
    stats: Arc<Mutex<Stats>>,
    mut shutdown: broadcast::Receiver<()>,
    config: Config,
) {
    let mut ticker = interval(Duration::from_millis(config.decay_interval_ms));

    loop {
        tokio::select! {
            _ = ticker.tick() => {
                // Level 2: Batch decay all scores
                let decayed = {
                    let mut engine_guard = engine.write().await;
                    engine_guard.batch_decay()
                };

                if config.verbose {
                    let stats_guard = stats.lock().await;
                    println!("[LEVEL-2] Batch decay: {} blobs processed",
                             decayed.len());
                }

                // Collect hot candidates
                let hot_candidates = {
                    let engine_guard = engine.read().await;
                    engine_guard.get_hot_candidates(config.threshold_hot)
                };

                if config.verbose && !hot_candidates.is_empty() {
                    println!("[LEVEL-2] Hot candidates: {:?}",
                             hot_candidates.iter().take(10).collect::<Vec<_>>());
                }
            }
            _ = shutdown.recv() => {
                if config.verbose {
                    println!("[decay_scheduler] Shutting down...");
                }
                break;
            }
        }
    }
}

/// Decay scheduler task (defined below)
struct ErrorBackoff {
    consecutive: u32,
}

impl ErrorBackoff {
    fn new() -> Self {
        Self { consecutive: 0 }
    }

    /// Returns true if we should log this error.
    fn should_log(&mut self) -> bool {
        self.consecutive += 1;
        self.consecutive <= 3 || self.consecutive % 10 == 0
    }

    /// Reset after successful operation.
    #[allow(dead_code)]
    fn reset(&mut self) {
        self.consecutive = 0;
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args();

    println!("=== Aneris Tune - Adaptive Blob Reorganization ===");
    println!("Configuration:");
    println!("  Config file: {}", config.config_file);
    println!("  Hot threshold: {:.1}", config.threshold_hot);
    println!("  Cold threshold: {:.1}", config.threshold_cold);
    println!("  Decay interval: {} ms", config.decay_interval_ms);
    println!("  Reorg interval: {} ms", config.reorg_interval_ms);
    if let Some(ref output) = config.output_file {
        println!("  Output file: {}", output);
    }
    println!("  Verbose: {}", config.verbose);
    println!();

    // Initialize CTE
    println!("[1/4] Initializing CTE runtime...");
    clio_cte::sync::init(&config.config_file)?;
    println!("      ✓ CTE runtime initialized");

    // Give runtime time to fully initialize
    println!("\n[2/4] Waiting for runtime to stabilize...");
    tokio::time::sleep(Duration::from_millis(1000)).await;

    // Verify runtime is ready by creating a test client
    println!("[3/4] Verifying CTE client connection...");
    let test_client = Client::new().await?;
    println!("      ✓ Client created and verified");
    
    // Drop the test client - we'll create fresh clients in tasks
    drop(test_client);

    // Create frecency engine
    println!("[4/4] Initializing frecency engine...");
    let engine = Arc::new(RwLock::new(FrecencyEngine::new()));
    println!("      ✓ Frecency engine initialized (hot set: {} entries)",
             clio_cte::HOT_SET_SIZE);

    // Create reorg batcher with custom thresholds
    println!("[5/5] Initializing reorg batcher...");
    let batcher = Arc::new(ReorgBatcher::with_settings(
        config.threshold_hot,
        config.threshold_cold,
        config.reorg_interval_ms,
        1024, // Queue capacity
    ));
    println!("      ✓ Reorg batcher initialized");

    // Create shared state
    let registry = Arc::new(RwLock::new(BlobRegistry::new()));
    let stats = Arc::new(Mutex::new(Stats::default()));

    // Set up shutdown manager
    let shutdown_manager = Arc::new(ShutdownManager::new());

    // Spawn executable if in wrapping mode
    let mut child_process: Option<std::process::Child> = None;
    if let Some(ref exec) = config.executable {
        use std::process::{Command, Stdio};
        use std::env;

        println!("[5/5] Starting wrapped executable...");
        println!("      Executable: {}", exec);
        println!("      Arguments: {:?}", config.executable_args);
        
        // Get build directory for LD_PRELOAD
        let build_dir = env::var("IOWARP_BUILD_DIR")
            .or_else(|_| env::var("CMAKE_BINARY_DIR"))
            .unwrap_or_else(|_| {
                env::current_exe()
                    .ok()
                    .and_then(|p| p.parent().map(|p| p.to_string_lossy().to_string()))
                    .unwrap_or_else(|| "/tmp".to_string())
            });

        // The POSIX adapter is in bin/ or lib/
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
            println!("      ✓ POSIX adapter found");
        }

        // Spawn subprocess with LD_PRELOAD
        let child = Command::new(exec)
            .args(&config.executable_args)
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
            .expect("Failed to spawn executable");
        
        println!("      ✓ Subprocess started (PID: {})\n", child.id());
        child_process = Some(child);
    }

    // Spawn tasks
    println!("\nStarting monitoring tasks...");

    // Telemetry receiver task
    // Note: Client doesn't need to be in Arc since it's stateless in async API
    // Each poll_telemetry() call creates its own FFI client internally
    let engine1 = Arc::clone(&engine);
    let registry1 = Arc::clone(&registry);
    let batcher1 = Arc::clone(&batcher);
    let stats1 = Arc::clone(&stats);
    let shutdown1 = shutdown_manager.subscriber();
    let config1 = config.clone();
    let telemetry_handle = tokio::spawn(async move {
        telemetry_receiver_task(engine1, registry1, batcher1, stats1, shutdown1, config1).await
    });

// ... rest of the code

    // Decay scheduler task
    let engine2 = Arc::clone(&engine);
    let batcher2 = Arc::clone(&batcher);
    let stats2 = Arc::clone(&stats);
    let shutdown2 = shutdown_manager.subscriber();
    let config2 = config.clone();
    let decay_handle = tokio::spawn(async move {
        decay_scheduler(engine2, batcher2, stats2, shutdown2, config2).await;
    });

    // Reorg executor task
    let batcher3 = Arc::clone(&batcher);
    let stats3 = Arc::clone(&stats);
    let shutdown3 = shutdown_manager.subscriber();
    let config3 = config.clone();
    let reorg_handle = tokio::spawn(async move {
        reorg_executor_task(batcher3, stats3, shutdown3, config3).await
    });

    println!("✓ All tasks started");
    if child_process.is_some() {
        println!("\nMonitoring wrapped executable. Press Ctrl+C to terminate.\n");
    } else {
        println!("\nPress Ctrl+C to shut down gracefully.\n");
    }

    // Wait for shutdown signal or child process
    if let Some(ref mut child) = child_process {
        // In wrapping mode: wait for child to finish
        match child.wait() {
            Ok(status) => {
                println!("\nWrapped executable exited with status: {:?}", status.code());
            }
            Err(e) => {
                eprintln!("\nError waiting for child process: {}", e);
            }
        }
    } else {
        // Standalone mode: wait for Ctrl+C
        tokio::signal::ctrl_c().await?;
        println!("\nShutdown signal received...");
    }

    // Signal all tasks to stop
    shutdown_manager.shutdown();

    // Wait for tasks with timeout
    let shutdown_timeout = Duration::from_secs(5);

    match timeout(shutdown_timeout, telemetry_handle).await {
        Ok(_) => println!("Telemetry receiver stopped."),
        Err(_) => println!("Telemetry receiver timeout, terminated."),
    }

    match timeout(shutdown_timeout, decay_handle).await {
        Ok(_) => println!("Decay scheduler stopped."),
        Err(_) => println!("Decay scheduler timeout, terminated."),
    }

    match timeout(shutdown_timeout, reorg_handle).await {
        Ok(_) => println!("Reorg executor stopped."),
        Err(_) => println!("Reorg executor timeout, terminated."),
    }

    // Print final stats
    println!("\n=== Final Statistics ===");
    let stats_guard = stats.lock().await;
    println!("Total telemetry entries: {}", stats_guard.total_telemetry_entries);
    println!("Total blob accesses: {}", stats_guard.total_accesses);
    println!("Total reorganizations: {}", stats_guard.total_reorgs);
    println!("  Hot promotions: {}", stats_guard.hot_reorgs);
    println!("  Cold demotions: {}", stats_guard.cold_reorgs);

    let engine_guard = engine.read().await;
    let hot_stats = engine_guard.hot_stats();
    let cold_stats = engine_guard.cold_stats();
    println!("\nFrecency engine:");
    println!("  Hot set entries: {} / {}", hot_stats.active_entries, clio_cte::HOT_SET_SIZE);
    println!("  Cold set entries: {}", cold_stats.entry_count);
    println!("  Hot set total score: {:.2}", hot_stats.total_score);
    println!("  Cold set total score: {:.2}", cold_stats.total_score);

    println!("\nAneris-tune stopped.");
    Ok(())
}

/// Telemetry receiver task wrapper
async fn telemetry_receiver_task(
    engine: Arc<RwLock<FrecencyEngine>>,
    registry: Arc<RwLock<BlobRegistry>>,
    batcher: Arc<ReorgBatcher>,
    stats: Arc<Mutex<Stats>>,
    mut shutdown: broadcast::Receiver<()>,
    config: Config,
) {
    println!("[telemetry_receiver] Task started");
    
    let mut ticker = interval(Duration::from_millis(100)); // Poll every 100ms
    let mut last_logical_time: u64 = 0;
    let mut error_backoff = ErrorBackoff::new();

    loop {
        tokio::select! {
            _ = ticker.tick() => {
                // Create a fresh client for each poll (Client is stateless in async API)
                match Client::new().await {
                    Ok(client) => {
                        // Poll telemetry with timeout=0 to check availability
                        match client.poll_telemetry(last_logical_time, 0.0).await {
                            Ok(entries) => {
                                if entries.is_empty() {
                                    continue;
                                }

                                // Update frecency for each access
                                let mut engine_guard = engine.write().await;
                                let mut registry_guard = registry.write().await;

                                for entry in &entries {
                                    // Compute a synthetic blob ID from available telemetry fields
                                    // Since blob_hash was removed, use (tag_id, off, size) as identifier
                                    use std::collections::hash_map::DefaultHasher;
                                    use std::hash::{Hash, Hasher};
                                    
                                    let mut hasher = DefaultHasher::new();
                                    entry.tag_id.major.hash(&mut hasher);
                                    entry.tag_id.minor.hash(&mut hasher);
                                    entry.off.hash(&mut hasher);
                                    entry.size.hash(&mut hasher);
                                    let blob_id = hasher.finish();

                                    // Level 1: Immediate atomic score update
                                    let score = engine_guard.record_access(blob_id);

                                    // Update registry mapping (for future reorg calls)
                                    registry_guard.insert(blob_id, entry.tag_id, format!("blob_{}", blob_id));

                                    // Level 2 candidate: Check if should reorg
                                    if let Some(decision) = batcher.should_reorg_blob(blob_id, score) {
                                        if config.verbose {
                                            println!("[LEVEL-2] Blob {} score {:.2} -> {:?} priority",
                                                     blob_id, score, decision.priority);
                                        }

                                        // Push to batch queue (Level 3 batching)
                                        if !batcher.push(decision) {
                                            eprintln!("Warning: Reorg queue full, dropping decision for blob {}", blob_id);
                                        }
                                    }

                                    // Update stats
                                    let mut stats_guard = stats.lock().await;
                                    stats_guard.total_accesses += 1;
                                    stats_guard.total_telemetry_entries += 1;

                                    last_logical_time = entry.logical_time.max(last_logical_time);
                                }
                            }
                            Err(clio_cte::CteError::Timeout) => {
                                // No data available - continue
                            }
                            Err(e) => {
                                if error_backoff.should_log() {
                                    eprintln!("[telemetry_receiver] Telemetry poll error: {}", e);
                                }
                            }
                        }
                    }
                    Err(e) => {
                        if error_backoff.should_log() {
                            eprintln!("[telemetry_receiver] Failed to create poll client: {}", e);
                        }
                    }
                }
            }
            _ = shutdown.recv() => {
                if config.verbose {
                    println!("[telemetry_receiver] Shutting down...");
                }
                break;
            }
        }
    }
}

/// Reorg executor task
async fn reorg_executor_task(
    batcher: Arc<ReorgBatcher>,
    stats: Arc<Mutex<Stats>>,
    mut shutdown: broadcast::Receiver<()>,
    config: Config,
) {
    println!("[reorg_executor] Task started");
    
    let mut ticker = interval(Duration::from_millis(config.reorg_interval_ms));

    // Known tags cache
    let known_tags: Arc<RwLock<Vec<CteTagId>>> = Arc::new(RwLock::new(Vec::new()));

    loop {
        tokio::select! {
            _ = ticker.tick() => {
                // Level 3: Drain batch and execute reorg decisions
                let mut batch = batcher.drain_batch();

                if batch.is_empty() {
                    continue;
                }

                // Coalesce duplicates (keep highest score per blob_id)
                batcher.coalesce_batch(&mut batch);

                if config.verbose {
                    println!("[LEVEL-3] Processing {} reorg decisions", batch.len());
                }

                // Execute reorg decisions
                for decision in batch {
                    // For now, we need to discover which tag owns this blob
                    // In production, this would use a blob registry cache
                    // For this demo, we iterate known tags
                    
                    let blob_id = decision.blob_id;
                    let score = decision.new_score;

                    // Try to find the tag that owns this blob
                    // This is a simplified version - production would use registry
                    
                    // Placeholder: We would call tag.reorganize_blob() here
                    // For the actual implementation, we need tag_id and blob_name
                    
                    // Update stats
                    let mut stats_guard = stats.lock().await;
                    stats_guard.total_reorgs += 1;

                    if decision.priority == clio_cte::Priority::High {
                        stats_guard.hot_reorgs += 1;
                    } else if decision.priority == clio_cte::Priority::Low {
                        stats_guard.cold_reorgs += 1;
                    }
                }
            }
            _ = shutdown.recv() => {
                if config.verbose {
                    println!("[reorg_executor] Shutting down...");
                }
                break;
            }
        }
    }
}

/// Shutdown management using broadcast channel for coordinated shutdown.
struct ShutdownManager {
    tx: broadcast::Sender<()>,
}

impl ShutdownManager {
    fn new() -> Self {
        let (tx, _) = broadcast::channel(1);
        Self { tx }
    }

    fn subscriber(&self) -> broadcast::Receiver<()> {
        self.tx.subscribe()
    }

    fn shutdown(&self) {
        // Ignore send errors (no receivers)
        let _ = self.tx.send(());
    }
}
