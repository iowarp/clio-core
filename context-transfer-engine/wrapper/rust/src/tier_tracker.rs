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

//! High-performance tier movement tracker for real-time monitoring
//!
//! PERFORMANCE CHARACTERISTICS:
//! - O(1) telemetry polling (just reads)
//! - O(k) GetBlobInfo where k = blocks in dirty blobs
//! - HashMap for O(1) cache lookups
//! - Pre-allocated collections

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use crate::ffi::{BlobBlockInfo, BlobInfo, Client, CteOp, CteTagId as FfiCteTagId};
use crate::types::CteTagId;

use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

/// Unique blob identifier using hash - compact and hashable
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct BlobKey {
    pub tag_major: u32,
    pub tag_minor: u32,
    pub blob_hash: u64, // Use hash instead of name
}

impl BlobKey {
    /// Create from tag_id and blob hash
    #[inline]
    pub fn new(tag_id: &CteTagId, blob_hash: u64) -> Self {
        Self {
            tag_major: tag_id.major,
            tag_minor: tag_id.minor,
            blob_hash,
        }
    }
}

/// Hash → Name registry entry
#[derive(Debug, Clone)]
pub struct RegistryEntry {
    pub blob_name: String,
    pub last_seen: Instant,
}

/// Cached blob state for delta comparison
#[derive(Debug, Clone)]
pub struct CachedBlobState {
    pub info: BlobInfo,
    pub last_check: Instant,
}

/// Tier movement event - emitted when block changes tiers
#[derive(Debug, Clone)]
pub struct TierMovementEvent {
    pub blob_key: BlobKey,
    pub block_index: usize,
    pub from_pool: Option<u64>, // None = new block
    pub to_pool: u64,
    pub block_size: u64,
    pub timestamp: Instant,
    pub logical_time: u64,
}

/// High-performance tier movement tracker with hash registry
///
/// Uses event-driven detection: telemetry → dirty set → GetBlobInfo → delta
///
/// # Hash-Based Blob Identification
///
/// This tracker uses 64-bit FNV-1a hashes to identify blobs in telemetry,
/// avoiding the need to include blob names in the fixed-size CteTelemetry struct.
/// The hash registry maps (tag_major, tag_minor, blob_hash) → blob_name for O(1) lookups.
///
/// # Example
///
/// ```ignore
/// // After reorganizing a blob
/// tag.reorganize_blob("my_blob", 1.0);
/// tracker.mark_dirty_by_hash(&tag.id(), compute_blob_hash(&tag.id(), "my_blob"));
///
/// // Now poll will detect the movement
/// let events = tracker.poll_movements();
/// ```
pub struct TierMovementTracker {
    client: Client,
    /// Hash → blob_name registry: (tag_major, tag_minor, hash) → entry
    hash_registry: HashMap<(u32, u32, u64), RegistryEntry>,
    /// Track which tags have been populated
    populated_tags: HashSet<(u32, u32)>,
    /// Pool ID → tier name mapping
    tier_names: HashMap<u64, String>,
    /// Blob cache for delta comparison (by hash)
    blob_cache: HashMap<BlobKey, CachedBlobState>,
    /// Blobs that need checking (from ReorganizeBlob telemetry)
    dirty_hashes: HashSet<BlobKey>,
    /// Minimum time between checks for same blob (debounce)
    poll_interval: Duration,
    /// Last telemetry logical time
    last_telemetry_time: u64,
    /// Reusable buffer for blob info queries (avoids allocations)
    reuse_buffer: Vec<u8>,
}

impl TierMovementTracker {
    /// Create new tracker with default settings
    pub fn new(client: Client) -> Self {
        Self {
            client,
            hash_registry: HashMap::new(),
            populated_tags: HashSet::new(),
            tier_names: HashMap::new(),
            blob_cache: HashMap::new(),
            dirty_hashes: HashSet::new(),
            poll_interval: Duration::from_millis(100),
            last_telemetry_time: 0,
            reuse_buffer: Vec::with_capacity(1024),
        }
    }

    /// Set poll interval (default: 100ms)
    pub fn with_poll_interval(mut self, interval: Duration) -> Self {
        self.poll_interval = interval;
        self
    }

    /// Register tier name for pool ID
    pub fn register_tier(&mut self, pool_id: u64, name: &str) {
        self.tier_names.insert(pool_id, name.to_string());
    }

    /// Main polling function - returns tier movement events
    ///
    /// PERFORMANCE: Only queries GetBlobInfo for dirty blobs
    pub fn poll_movements(&mut self) -> Vec<TierMovementEvent> {
        let now = Instant::now();
        let mut events = Vec::new();

        // Step 1: Poll telemetry for logical time tracking (O(1) read)
        // Use 5 second timeout for telemetry polling
        // Note: Telemetry no longer contains blob_hash, so we rely on explicit
        // mark_dirty_by_hash() calls after reorganize operations.
        let telemetry = match self.client.poll_telemetry(self.last_telemetry_time, 5.0) {
            Ok(t) => t,
            Err(_) => return events, // Return empty events on error
        };

        for entry in &telemetry {
            // Update logical time tracking
            if entry.logical_time > self.last_telemetry_time {
                self.last_telemetry_time = entry.logical_time;
            }

            // Populate tags from PutBlob events for registry building
            if entry.op == CteOp::PutBlob {
                // Mark for population if needed
                self.populated_tags
                    .remove(&(entry.tag_id.major, entry.tag_id.minor));
            }
        }

        // Step 2: Check dirty hashes for tier movements
        // PERFORMANCE: Drain dirty set to avoid reallocation
        let dirty_list: Vec<_> = self.dirty_hashes.drain().collect();

        for blob_key in dirty_list {
            // Debounce: skip if checked recently
            if let Some(cached) = self.blob_cache.get(&blob_key) {
                if now.duration_since(cached.last_check) < self.poll_interval {
                    continue;
                }
            }

            // Resolve hash to name
            let tag_id_types = CteTagId {
                major: blob_key.tag_major,
                minor: blob_key.tag_minor,
            };
            let tag_id_ffi = FfiCteTagId {
                major: blob_key.tag_major,
                minor: blob_key.tag_minor,
            };

            let blob_name = match self.resolve_hash(&tag_id_types, blob_key.blob_hash) {
                Some(name) => name.to_string(),
                None => {
                    // Hash not in registry - try repopulating
                    self.populated_tags
                        .remove(&(blob_key.tag_major, blob_key.tag_minor));
                    self.populate_tag(&tag_id_types);

                    match self.resolve_hash(&tag_id_types, blob_key.blob_hash) {
                        Some(name) => name.to_string(),
                        None => {
                            eprintln!(
                                "Warning: Could not resolve blob hash {} for tag {},{}",
                                blob_key.blob_hash, blob_key.tag_major, blob_key.tag_minor
                            );
                            continue;
                        }
                    }
                }
            };

            // Query current blob info
            match self.client.get_blob_info(&tag_id_ffi, &blob_name) {
                Ok(blob_info) => {
                    // Detect movements by comparing with cache
                    if let Some(cached) = self.blob_cache.get(&blob_key) {
                        events.extend(Self::detect_movements(
                            &blob_key,
                            &cached.info,
                            &blob_info,
                            now,
                        ));
                    }

                    // Update cache
                    self.blob_cache.insert(
                        blob_key,
                        CachedBlobState {
                            info: blob_info,
                            last_check: now,
                        },
                    );
                }
                Err(_) => {
                    // Blob deleted - remove from cache
                    self.blob_cache.remove(&blob_key);
                }
            }
        }

        events
    }

    /// Populate hash registry for a tag
    pub fn populate_tag(&mut self, tag_id: &CteTagId) {
        let tag_key = (tag_id.major, tag_id.minor);

        // Check if already populated
        if self.populated_tags.contains(&tag_key) {
            return;
        }

        // Get all blobs in this tag (copy tag_id since from_id takes ownership)
        let tag_id_val = CteTagId {
            major: tag_id.major,
            minor: tag_id.minor,
        };
        if let Ok(tag) = std::panic::catch_unwind(|| crate::sync::Tag::from_id(tag_id_val)) {
            if let Ok(blobs) = std::panic::catch_unwind(|| tag.get_contained_blobs()) {
                for blob_name in blobs {
                    // Compute hash (same algorithm as C++)
                    let hash = Self::compute_hash(tag_id, &blob_name);

                    let registry_key = (tag_id.major, tag_id.minor, hash);
                    self.hash_registry.insert(
                        registry_key,
                        RegistryEntry {
                            blob_name,
                            last_seen: Instant::now(),
                        },
                    );
                }
            }
        }

        self.populated_tags.insert(tag_key);
    }

    /// Compute FNV-1a hash (must match C++ algorithm)
    fn compute_hash(tag_id: &CteTagId, blob_name: &str) -> u64 {
        const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
        const FNV_PRIME: u64 = 0x100000001b3;

        let mut hash = FNV_OFFSET_BASIS;

        // Hash tag_id (convert to bytes)
        let tag_bytes = tag_id.major.to_le_bytes();
        for byte in tag_bytes {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        let tag_bytes = tag_id.minor.to_le_bytes();
        for byte in tag_bytes {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }

        // Hash blob_name
        for byte in blob_name.bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }

        hash
    }

    /// Lookup blob name from hash (O(1))
    pub fn resolve_hash(&self, tag_id: &CteTagId, blob_hash: u64) -> Option<&str> {
        let key = (tag_id.major, tag_id.minor, blob_hash);
        self.hash_registry.get(&key).map(|e| e.blob_name.as_str())
    }

    /// Detect tier movements by comparing old vs new blob state
    ///
    /// PERFORMANCE: O(k) where k = number of blocks
    fn detect_movements(
        blob_key: &BlobKey,
        old_info: &BlobInfo,
        new_info: &BlobInfo,
        timestamp: Instant,
    ) -> Vec<TierMovementEvent> {
        let mut events = Vec::with_capacity(new_info.blocks.len());

        for (i, new_block) in new_info.blocks.iter().enumerate() {
            // Find matching old block by (size, offset) - unique identifier
            let old_block = old_info.blocks.iter().find(|b| {
                b.block_size == new_block.block_size && b.block_offset == new_block.block_offset
            });

            match old_block {
                Some(old) => {
                    if old.pool_id != new_block.pool_id {
                        // Tier movement detected!
                        events.push(TierMovementEvent {
                            blob_key: blob_key.clone(),
                            block_index: i,
                            from_pool: Some(old.pool_id),
                            to_pool: new_block.pool_id,
                            block_size: new_block.block_size,
                            timestamp,
                            logical_time: 0, // Set from telemetry if available
                        });
                    }
                }
                None => {
                    // New block created
                    events.push(TierMovementEvent {
                        blob_key: blob_key.clone(),
                        block_index: i,
                        from_pool: None,
                        to_pool: new_block.pool_id,
                        block_size: new_block.block_size,
                        timestamp,
                        logical_time: 0,
                    });
                }
            }
        }

        events
    }

    /// Get tier name for pool ID
    #[inline]
    pub fn get_tier_name(&self, pool_id: u64) -> Option<&str> {
        self.tier_names.get(&pool_id).map(|s| s.as_str())
    }

    /// Mark a blob as dirty by hash (to be checked on next poll)
    pub fn mark_dirty_by_hash(&mut self, tag_id: &CteTagId, blob_hash: u64) {
        self.dirty_hashes.insert(BlobKey::new(tag_id, blob_hash));
    }

    /// Clear cache (useful for testing or memory pressure)
    pub fn clear_cache(&mut self) {
        self.blob_cache.clear();
        self.dirty_hashes.clear();
        self.hash_registry.clear();
        self.populated_tags.clear();
    }

    /// Get cache stats
    pub fn cache_stats(&self) -> (usize, usize, usize) {
        (
            self.blob_cache.len(),
            self.dirty_hashes.len(),
            self.hash_registry.len(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blob_key_hash() {
        let key1 = BlobKey::new(&CteTagId { major: 1, minor: 2 }, 12345);
        let key2 = BlobKey::new(&CteTagId { major: 1, minor: 2 }, 12345);
        let key3 = BlobKey::new(&CteTagId { major: 1, minor: 3 }, 12345);

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_compute_hash() {
        let tag_id = CteTagId { major: 1, minor: 2 };
        let blob_name = "test_blob";

        // Compute hash twice - should be same
        let hash1 = TierMovementTracker::compute_hash(&tag_id, blob_name);
        let hash2 = TierMovementTracker::compute_hash(&tag_id, blob_name);

        assert_eq!(hash1, hash2);

        // Different blob name should give different hash
        let hash3 = TierMovementTracker::compute_hash(&tag_id, "other_blob");
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_detect_movements() {
        let blob_key = BlobKey::new(&CteTagId { major: 1, minor: 2 }, 12345);

        let old_info = BlobInfo {
            score: 0.5,
            total_size: 1024,
            blocks: vec![BlobBlockInfo {
                pool_id: 301,
                block_size: 1024,
                block_offset: 0,
            }],
        };

        let new_info = BlobInfo {
            score: 1.0,
            total_size: 1024,
            blocks: vec![BlobBlockInfo {
                pool_id: 302,
                block_size: 1024,
                block_offset: 0,
            }],
        };

        let events =
            TierMovementTracker::detect_movements(&blob_key, &old_info, &new_info, Instant::now());

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].from_pool, Some(301));
        assert_eq!(events[0].to_pool, 302);
    }
}
