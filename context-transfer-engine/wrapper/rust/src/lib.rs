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

//! IOWarp Context Transfer Engine - Rust Bindings
//!
//! This crate provides Rust bindings to the IOWarp CTE (Context Transfer Engine),
//! enabling Rust programs to interface with CTE for blob storage, retrieval,
//! score adjustment, and telemetry.
//!
//! # Features
//!
//! - `async` (default): Async API using Tokio's `spawn_blocking`
//! - `sync`: Synchronous (blocking) API
//!
//! # Example - Async API
//!
//! ```no_run
//! use clio_cte::{Client, Tag};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Initialize and create client
//!     let client = Client::new().await?;
//!
//!     // Create or open a tag
//!     let tag = Tag::new("my_dataset").await?;
//!
//!     // Store data
//!     tag.put_blob("data.bin".to_string(), b"hello".to_vec(), 0, 1.0).await;
//!
//!     // Get telemetry
//!     let telemetry = client.poll_telemetry(0, 5.0).await?;
//!     for entry in telemetry {
//!         println!("Op: {:?}, Size: {}", entry.op, entry.size);
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! # Example - Sync API
//!
//! ```no_run
//! use clio_cte::sync::{init, Client, Tag};
//!
//! // Initialize CTE
//! init("").expect("CTE init failed");
//!
//! // Create client and tag
//! let client = Client::new().unwrap();
//! let tag = Tag::new("my_dataset");
//!
//! // Store data synchronously
//! tag.put_blob("data.bin", b"hello");
//! let data = tag.get_blob("data.bin", 5, 0);
//! ```

// Module declarations
pub mod capability_detector;
pub mod error;
pub mod ffi;
pub mod frecency_engine;
pub mod reorg_batch;
pub mod types;
pub mod tier_tracker;
pub mod zerocopy;

// Feature-gated API modules
#[cfg(feature = "async")]
pub mod r#async;

#[cfg(feature = "sync")]
pub mod sync;

// Re-export core types
pub use error::{CteError, CteResult};
pub use types::{
    BdevType, RuntimeMode, CteOp, CteTagId, CteTelemetry, PoolQuery, SteadyTime,
};

// Re-export tier tracking types
pub use tier_tracker::{
    TierMovementTracker,
    TierMovementEvent,
    BlobKey,
    CachedBlobState,
    RegistryEntry,
};

// Re-export frecency engine types
pub use frecency_engine::{
    FrecencyEngine,
    HotSet,
    HotSetStats,
    ColdSetStats,
    HOT_SET_SIZE,
    DECAY_FACTOR,
    DEFAULT_SCORE,
};

// Re-export reorg batch types
pub use reorg_batch::{
    ReorgBatcher,
    ReorgDecision,
    LockFreeQueue,
    Priority,
    THRESHOLD_HOT,
    THRESHOLD_COLD,
};

// Re-export API based on features
#[cfg(feature = "async")]
pub use r#async::{Client, Tag};

// When only sync feature is enabled (not async)
#[cfg(all(feature = "sync", not(feature = "async")))]
pub use sync::{Client, Tag};

// Keep existing ffi_c module for backward compatibility
// This provides C-ABI exports for calling from other languages
mod ffi_c;

// Unit tests module (no runtime required)
#[cfg(test)]
mod tests;

/// Version of the clio-cte-rs crate
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod async_tests {
    use super::*;

    #[tokio::test]
    #[cfg(feature = "async")]
    async fn test_client_new() {
        // This will fail if CTE is not initialized
        // Just verify it compiles
        let _ = Client::new().await;
    }

    #[tokio::test]
    #[cfg(feature = "async")]
    async fn test_tag_lifecycle() {
        // Note: Requires running CTE runtime
        // Set CHI_WITH_RUNTIME=1 before running tests
        
        // Skip if runtime not available
        if crate::sync::init("").is_err() {
            eprintln!("Skipping test: CTE runtime not available");
            return;
        }

        let tag = Tag::new("rust_test_tag").await.expect("Failed to create tag");
        let data = b"hello from rust test";

        // Put blob
        tag.put_blob("test_blob".to_string(), data.to_vec(), 0, 1.0).await.expect("put_blob failed");

        // Get blob size
        let size = tag.get_blob_size("test_blob").await.expect("get_blob_size failed");

        // Get blob
        let got = tag.get_blob("test_blob".to_string(), size, 0).await.expect("get_blob failed");
        assert_eq!(got, data);

        // Get blob score
        let score = tag.get_blob_score("test_blob").await.expect("get_blob_score failed");
        assert!((score - 1.0).abs() < 0.01);

        // Reorganize blob
        tag.reorganize_blob("test_blob".to_string(), 0.5).await.expect("reorganize failed");

        // Get new score
        let new_score = tag.get_blob_score("test_blob").await.expect("get_blob_score failed");
        assert!((new_score - 0.5).abs() < 0.01);
    }

    #[tokio::test]
    #[cfg(feature = "async")]
    async fn test_client_telemetry() {
        // Skip if runtime not available
        if crate::sync::init("").is_err() {
            eprintln!("Skipping test: CTE runtime not available");
            return;
        }

        let client = Client::new().await.expect("Failed to create client");

        // Get telemetry (may be empty if no operations)
        let telemetry = client.poll_telemetry(0, 5.0).await.expect("poll_telemetry failed");
        // Just verify it doesn't panic
        println!("Got {} telemetry entries", telemetry.len());
    }
}

#[cfg(test)]
mod sync_tests {
    use super::*;

    #[test]
    #[cfg(feature = "sync")]
    fn test_sync_api() {
        // Skip if runtime not available
        if crate::sync::init("").is_err() {
            eprintln!("Skipping test: CTE runtime not available");
            return;
        }

        let tag = sync::Tag::new("sync_test_tag");
        let data = b"sync test data";
        
        tag.put_blob("test_blob", data);

        let size = tag.get_blob_size("test_blob").expect("get_blob_size failed");
        assert_eq!(size, data.len() as u64);

        let got = tag.get_blob("test_blob", size, 0).expect("get_blob failed");
        assert_eq!(got, data);
    }
}
