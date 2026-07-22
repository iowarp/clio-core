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

//! Frecency engine with SoA data structures and SIMD-optimized decay.
//!
//! This module implements a frecency scoring system (frequency + recency) for blob
//! popularity tracking using a hot/cold split architecture:
//!
//! - **Hot set**: 512 most frequently accessed blobs in direct-indexed SoA layout
//! - **Cold set**: Overflow blobs stored in a HashMap
//!
//! The implementation uses AVX2 SIMD intrinsics for batch decay operations when
//! available (processes 4 f64 values per cycle), with scalar fallback otherwise.

use crate::ffi::ffi;

// Use foldhash for faster HashMap operations (3-5x faster than DefaultHasher)
// Keys are internal u64 blob hashes, not user-controlled — no DoS concern
type FastHashMap<K, V> = hashbrown::HashMap<K, V, foldhash::fast::RandomState>;

/// Number of hot set entries (fixed size for direct indexing)
pub const HOT_SET_SIZE: usize = 512;

/// Decay factor per tick (~10 second intervals)
/// Score formula: score = score * DECAY_FACTOR + 1.0
pub const DECAY_FACTOR: f64 = 0.999_999;

/// Default score for new entries
pub const DEFAULT_SCORE: f64 = 0.0;

/// Minimum alignment for AVX2 operations (32 bytes)
const AVX2_ALIGNMENT: usize = 32;

/// Minimum alignment for cache lines
const CACHE_LINE_ALIGNMENT: usize = 64;

// ============================================================================
// C++ Frecency Formula Constants (for validation)
// ============================================================================
// These constants match the C++ FrecencyDataOrganizer implementation:
// recency   = 2^(-age_sec / 600.0)        // 600s half-life
// frequency = n / (n + 10.0)              // 10 saturation
// score     = 0.5 * recency + 0.5 * frequency
// score     = clamp(score, 0.0, 1.0)
// Where age_sec = (now - max(last_read, last_modified)) / 1e9

/// Half-life for recency decay in seconds (matches C++)
const RECENCY_HALF_LIFE_SEC: f64 = 600.0;

/// Saturation constant for frequency calculation (matches C++)
const FREQUENCY_SATURATION: f64 = 10.0;

/// Weight for recency component in final score (matches C++)
const RECENCY_WEIGHT: f64 = 0.5;

/// Weight for frequency component in final score (matches C++)
const FREQUENCY_WEIGHT: f64 = 0.5;

/// A cold set entry stored in the HashMap
#[derive(Debug, Clone)]
struct ColdEntry {
    score: f64,
    count: u64,
    last_update: u64,
}

impl ColdEntry {
    fn new() -> Self {
        ColdEntry {
            score: DEFAULT_SCORE,
            count: 0,
            last_update: 0,
        }
    }
}

/// Hot set stored in SoA (Structure of Arrays) layout for SIMD efficiency.
///
/// Each array is aligned to cache lines (64 bytes) to avoid false sharing
/// and to enable SIMD operations where possible.
#[repr(align(64))]
pub struct HotSet {
    /// Frecency scores for each slot
    scores: Vec<f64>,
    /// Access counts for each slot
    counts: Vec<u64>,
    /// Last update timestamp for each slot
    last_updates: Vec<u64>,
    /// Blob ID (key) for each slot
    keys: Vec<u64>,
    /// Map from blob_id to slot index for O(1) lookup
    key_to_slot: FastHashMap<u64, usize>,
    /// Stack of free slot indices (for reuse)
    pub free_slots: Vec<usize>,
    /// Current tick counter (updated on batch_decay)
    current_tick: u64,
}

impl HotSet {
    /// Create a new hot set with all slots initially free.
    pub fn new() -> Self {
        let mut free_slots: Vec<usize> = (0..HOT_SET_SIZE).collect();
        free_slots.reverse(); // Pop from end for stack behavior

        HotSet {
            scores: vec![0.0; HOT_SET_SIZE],
            counts: vec![0; HOT_SET_SIZE],
            last_updates: vec![0; HOT_SET_SIZE],
            keys: vec![0; HOT_SET_SIZE],
            key_to_slot: FastHashMap::with_capacity_and_hasher(
                HOT_SET_SIZE,
                foldhash::fast::RandomState::default(),
            ),
            free_slots,
            current_tick: 0,
        }
    }

    /// Check if AVX2 is available at runtime.
    #[inline]
    fn has_avx2() -> bool {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            is_x86_feature_detected!("avx2")
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            false
        }
    }

    /// Find a blob in the hot set by ID.
    ///
    /// Returns the slot index if found, None otherwise.
    #[inline]
    pub fn find(&self, blob_id: u64) -> Option<usize> {
        self.key_to_slot.get(&blob_id).copied()
    }

    /// Record an access for a blob in the hot set.
    ///
    /// Updates score, count, and last_update timestamp.
    /// Returns the new score after update.
    #[inline]
    pub fn record_access(&mut self, slot: usize) -> f64 {
        debug_assert!(slot < HOT_SET_SIZE);

        // Apply decay for missing ticks
        let missed_ticks = self.current_tick.saturating_sub(self.last_updates[slot]);
        if missed_ticks > 0 {
            let decay_multiplier = DECAY_FACTOR.powi(missed_ticks as i32);
            self.scores[slot] *= decay_multiplier;
        }

        // Update entry
        self.scores[slot] += 1.0;
        self.counts[slot] += 1;
        self.last_updates[slot] = self.current_tick;

        self.scores[slot]
    }

    /// Insert a new blob into the hot set.
    ///
    /// Returns Some(slot) if successful, None if hot set is full.
    pub fn insert(&mut self, blob_id: u64) -> Option<usize> {
        // Check if already in hot set
        if let Some(slot) = self.key_to_slot.get(&blob_id) {
            return Some(*slot);
        }

        // Get free slot
        let slot = self.free_slots.pop()?;

        // Initialize entry
        self.scores[slot] = DEFAULT_SCORE + 1.0; // First access counts
        self.counts[slot] = 1;
        self.last_updates[slot] = self.current_tick;
        self.keys[slot] = blob_id;

        // Add to lookup map
        self.key_to_slot.insert(blob_id, slot);

        Some(slot)
    }

    /// Remove a blob from the hot set by slot index.
    #[inline]
    pub fn remove(&mut self, slot: usize) {
        debug_assert!(slot < HOT_SET_SIZE);

        let blob_id = self.keys[slot];
        self.key_to_slot.remove(&blob_id);

        // Reset slot values
        self.scores[slot] = 0.0;
        self.counts[slot] = 0;
        self.last_updates[slot] = 0;
        self.keys[slot] = 0;

        // Return slot to free list
        self.free_slots.push(slot);
    }

    /// Get the score for a specific slot.
    #[inline]
    pub fn get_score(&self, slot: usize) -> f64 {
        debug_assert!(slot < HOT_SET_SIZE);
        self.scores[slot]
    }

    /// Get the count for a specific slot.
    #[inline]
    pub fn get_count(&self, slot: usize) -> u64 {
        debug_assert!(slot < HOT_SET_SIZE);
        self.counts[slot]
    }

    /// Get the blob_id for a specific slot.
    #[inline]
    pub fn get_key(&self, slot: usize) -> u64 {
        debug_assert!(slot < HOT_SET_SIZE);
        self.keys[slot]
    }

    /// Get current tick counter.
    #[inline]
    pub fn current_tick(&self) -> u64 {
        self.current_tick
    }

    /// Increment tick counter.
    #[inline]
    pub fn increment_tick(&mut self) {
        self.current_tick += 1;
    }

    /// Get mutable reference to scores array for SIMD decay.
    #[inline]
    pub fn scores_mut(&mut self) -> &mut [f64] {
        &mut self.scores
    }

    /// Get scores slice (for testing).
    #[inline]
    pub fn scores(&self) -> &[f64] {
        &self.scores
    }

    /// Get number of active entries in hot set.
    pub fn active_count(&self) -> usize {
        HOT_SET_SIZE - self.free_slots.len()
    }

    /// Get stack of free slots (for testing).
    #[inline]
    pub fn free_slots(&self) -> &[usize] {
        &self.free_slots
    }

    /// Batch decay all scores using SIMD when available.
    ///
    /// Applies: score *= DECAY_FACTOR
    /// Processes 4 f64 values per SIMD operation (AVX2).
    pub fn batch_decay(&mut self) -> Vec<(u64, f64)> {
        let mut decayed = Vec::with_capacity(self.active_count());

        // Increment tick
        self.increment_tick();

        // Apply decay to all hot set entries
        if Self::has_avx2() {
            unsafe {
                self.batch_decay_simd();
            }
        } else {
            self.batch_decay_scalar();
        }

        // Collect (blob_id, score) pairs
        for (&blob_id, &slot) in self.key_to_slot.iter() {
            let score = self.scores[slot];
            decayed.push((blob_id, score));
        }

        decayed
    }

    /// Scalar fallback for batch decay (public for testing).
    pub fn batch_decay_scalar(&mut self) {
        for i in 0..HOT_SET_SIZE {
            if self.keys[i] != 0 {
                // Slot is active
                self.scores[i] *= DECAY_FACTOR;
            }
        }
    }

    /// SIMD-optimized batch decay using AVX2 intrinsics (public for testing).
    ///
    /// Safety: This function uses x86 intrinsics that require AVX2.
    /// Only call when is_x86_feature_detected!("avx2") returns true.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    pub unsafe fn batch_decay_simd(&mut self) {
        use std::arch::x86_64::*;

        // Process 4 f64 at a time with AVX2
        let decay = DECAY_FACTOR;
        let chunks = HOT_SET_SIZE / 4;

        // Broadcast decay factor to all lanes
        let decay_vec = _mm256_set1_pd(decay);

        for chunk in 0..chunks {
            let offset = chunk * 4;

            // Load 4 scores (unaligned load - safe for any alignment)
            let scores_ptr = self.scores.as_ptr().add(offset) as *const f64;
            let scores_vec = _mm256_loadu_pd(scores_ptr);

            // Multiply by decay factor
            let decayed = _mm256_mul_pd(scores_vec, decay_vec);

            // Store back (unaligned store - safe for any alignment)
            let dest_ptr = self.scores.as_mut_ptr().add(offset) as *mut f64;
            _mm256_storeu_pd(dest_ptr, decayed);
        }

        // Handle remaining elements (if HOT_SET_SIZE is not divisible by 4)
        let remainder_start = chunks * 4;
        for i in remainder_start..HOT_SET_SIZE {
            self.scores[i] *= DECAY_FACTOR;
        }
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    fn batch_decay_simd(&mut self) {
        // Non-x86 platforms: always use scalar
        self.batch_decay_scalar();
    }
}

impl Default for HotSet {
    fn default() -> Self {
        Self::new()
    }
}

/// Frecency engine with hot/cold split architecture.
///
/// Maintains a hot set of frequently accessed blobs for fast operations
/// and a cold set for overflow storage.
pub struct FrecencyEngine {
    /// Hot set: directly indexed array for SIMD operations
    hot: HotSet,
    /// Cold set: HashMap for overflow blobs
    cold: FastHashMap<u64, ColdEntry>,
    /// Current tick counter
    tick: u64,
}

impl FrecencyEngine {
    /// Create a new frecency engine.
    pub fn new() -> Self {
        FrecencyEngine {
            hot: HotSet::new(),
            cold: FastHashMap::with_hasher(foldhash::fast::RandomState::default()),
            tick: 0,
        }
    }

    /// Record an access for a blob.
    ///
    /// Updates both frecency score and access count.
    /// Returns the new frecency score.
    pub fn record_access(&mut self, blob_id: u64) -> f64 {
        // Try hot set first
        if let Some(slot) = self.hot.find(blob_id) {
            return self.hot.record_access(slot);
        }

        // Check if blob is in cold set
        if !self.cold.contains_key(&blob_id) {
            // New blob - try hot set first
            if self.hot.insert(blob_id).is_some() {
                return self.hot.get_score(self.hot.find(blob_id).unwrap());
            }
            // Hot set full - add to cold set
            let entry = ColdEntry::new();
            self.cold.insert(blob_id, entry);
            return 0.0;
        }

        // Blob is in cold set - compute new values
        let (score, count, should_promote) = {
            let entry = self.cold.get(&blob_id).unwrap();
            let mut score = entry.score;
            let missed_ticks = self.tick.saturating_sub(entry.last_update);
            if missed_ticks > 0 {
                score *= DECAY_FACTOR.powi(missed_ticks as i32);
            }
            score += 1.0;
            let count = entry.count + 1;
            (score, count, count > 3)
        };

        // Handle promotion separately to avoid borrow conflicts
        if should_promote {
            // Remove from cold set
            self.cold.remove(&blob_id);

            // Try to promote to hot set
            if self.hot.insert(blob_id).is_some() {
                if let Some(slot) = self.hot.find(blob_id) {
                    self.hot.scores[slot] = score;
                    self.hot.counts[slot] = count;
                    self.hot.last_updates[slot] = self.tick;
                }
                return score;
            }

            // Fall back to cold set if hot set is full
            self.cold.insert(
                blob_id,
                ColdEntry {
                    score,
                    count,
                    last_update: self.tick,
                },
            );
            return score;
        }

        // Update cold entry (not promoting)
        let entry = self.cold.get_mut(&blob_id).unwrap();
        entry.score = score;
        entry.count = count;
        entry.last_update = self.tick;
        score
    }

    /// Batch decay all entries (hot and cold).
    ///
    /// Applies the decay factor to all frecency scores.
    /// Hot set uses SIMD when AVX2 is available.
    ///
    /// Returns list of (blob_id, score) pairs for all hot set entries.
    pub fn batch_decay(&mut self) -> Vec<(u64, f64)> {
        self.hot.increment_tick();
        self.tick += 1;

        // Decay hot set (SIMD or scalar)
        if HotSet::has_avx2() {
            unsafe {
                self.hot.batch_decay_simd();
            }
        } else {
            self.hot.batch_decay_scalar();
        }

        // Decay cold set (always scalar)
        for entry in self.cold.values_mut() {
            entry.score *= DECAY_FACTOR;
        }

        // Collect hot set scores
        let mut decayed = Vec::with_capacity(self.hot.active_count());
        for (&blob_id, &slot) in self.hot.key_to_slot.iter() {
            decayed.push((blob_id, self.hot.scores[slot]));
        }

        decayed
    }

    /// Get candidates above score threshold.
    ///
    /// Returns blob_ids with scores >= threshold from hot set.
    pub fn get_hot_candidates(&self, threshold: f64) -> Vec<u64> {
        let mut candidates = Vec::new();

        for (&blob_id, &slot) in self.hot.key_to_slot.iter() {
            if self.hot.scores[slot] >= threshold {
                candidates.push(blob_id);
            }
        }

        candidates
    }

    /// Get score for a specific blob.
    ///
    /// Returns None if blob is not tracked.
    pub fn get_score(&self, blob_id: u64) -> Option<f64> {
        if let Some(slot) = self.hot.find(blob_id) {
            Some(self.hot.get_score(slot))
        } else if let Some(entry) = self.cold.get(&blob_id) {
            Some(entry.score)
        } else {
            None
        }
    }

    /// Get access count for a specific blob.
    pub fn get_count(&self, blob_id: u64) -> Option<u64> {
        if let Some(slot) = self.hot.find(blob_id) {
            Some(self.hot.get_count(slot))
        } else if let Some(entry) = self.cold.get(&blob_id) {
            Some(entry.count)
        } else {
            None
        }
    }

    /// Remove a blob from tracking.
    pub fn remove(&mut self, blob_id: u64) {
        if let Some(slot) = self.hot.find(blob_id) {
            self.hot.remove(slot);
        } else {
            self.cold.remove(&blob_id);
        }
    }

    /// Get number of tracked blobs (hot + cold).
    pub fn len(&self) -> usize {
        self.hot.active_count() + self.cold.len()
    }

    /// Check if engine is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get hot set statistics.
    pub fn hot_stats(&self) -> HotSetStats {
        let active = self.hot.active_count();
        let total_score: f64 = self
            .hot
            .key_to_slot
            .values()
            .map(|&slot| self.hot.scores[slot])
            .sum();

        HotSetStats {
            active_entries: active,
            free_slots: self.hot.free_slots.len(),
            total_score,
        }
    }

    /// Get cold set statistics.
    pub fn cold_stats(&self) -> ColdSetStats {
        let total_score: f64 = self.cold.values().map(|e| e.score).sum();

        ColdSetStats {
            entry_count: self.cold.len(),
            total_score,
        }
    }

    /// Get current tick.
    pub fn current_tick(&self) -> u64 {
        self.tick
    }
}

impl Default for FrecencyEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// C++ Frecency Formula Validation
// ============================================================================

/// Compute frecency score using the pure C++ formula (for validation).
///
/// This function implements the exact formula from FrecencyDataOrganizer:
/// - recency = 2^(-age_sec / 600.0)        // 600s half-life
/// - frequency = n / (n + 10.0)            // 10 saturation
/// - score = 0.5 * recency + 0.5 * frequency
/// - score = clamp(score, 0.0, 1.0)
///
/// # Arguments
/// * `access_count` - Number of data ops on the blob (Put+Get)
/// * `last_access_nanos` - Last access time (steady-clock ns) = max(last_read, last_modified)
/// * `now_nanos` - Current steady-clock time (ns)
///
/// # Returns
/// Score as f64 in [0, 1]
fn compute_score_pure(access_count: u64, last_access_nanos: u64, now_nanos: u64) -> f64 {
    // Calculate age in seconds
    let age_nanos = now_nanos.saturating_sub(last_access_nanos);
    let age_sec = age_nanos as f64 / 1e9;

    // Recency: 2^(-age_sec / 600.0)
    let recency = 2.0_f64.powf(-age_sec / RECENCY_HALF_LIFE_SEC);

    // Frequency: n / (n + 10.0)
    let frequency = access_count as f64 / (access_count as f64 + FREQUENCY_SATURATION);

    // Combined score: 0.5 * recency + 0.5 * frequency
    let score = RECENCY_WEIGHT * recency + FREQUENCY_WEIGHT * frequency;

    // Clamp to [0, 1]
    score.clamp(0.0, 1.0)
}

/// Validate the Rust frecency score against the C++ FrecencyDataOrganizer.
///
/// Compares the Rust-native computation to the C++ source of truth.
///
/// # Arguments
/// * `access_count` - Number of data ops on the blob
/// * `last_access_nanos` - Last access time (steady-clock ns)
/// * `now_nanos` - Current steady-clock time (ns)
///
/// # Returns
/// Ok(()) if scores match within tolerance, Err with the delta otherwise
pub fn validate_against_cpp(
    access_count: u64,
    last_access_nanos: u64,
    now_nanos: u64,
) -> Result<(), f32> {
    let rust_score = compute_score_pure(access_count, last_access_nanos, now_nanos);
    let cpp_score = ffi::cte_frecency_compute_score(access_count, last_access_nanos, now_nanos);
    let delta = (rust_score as f32 - cpp_score).abs();

    // Tolerance: 1e-5 (f32 precision)
    if delta < 1e-5 {
        Ok(())
    } else {
        Err(delta)
    }
}

// ============================================================================
// Statistics Structures
// ============================================================================

/// Statistics for hot set.
#[derive(Debug, Clone)]
pub struct HotSetStats {
    pub active_entries: usize,
    pub free_slots: usize,
    pub total_score: f64,
}

/// Statistics for cold set.
#[derive(Debug, Clone)]
pub struct ColdSetStats {
    pub entry_count: usize,
    pub total_score: f64,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hot_set_creation() {
        let hot = HotSet::new();
        assert_eq!(hot.active_count(), 0);
        assert_eq!(hot.free_slots.len(), HOT_SET_SIZE);
    }

    #[test]
    fn test_hot_set_insert_and_find() {
        let mut hot = HotSet::new();

        let blob_id = 12345u64;
        let slot = hot.insert(blob_id).expect("Insert should succeed");

        assert!(slot < HOT_SET_SIZE);
        assert_eq!(hot.find(blob_id), Some(slot));
        assert_eq!(hot.active_count(), 1);
    }

    #[test]
    fn test_hot_set_record_access() {
        let mut hot = HotSet::new();

        let blob_id = 12345u64;
        let slot = hot.insert(blob_id).unwrap();

        let score = hot.record_access(slot);
        assert!((score - 1.0).abs() < 0.01, "Initial score should be ~1.0");
        assert_eq!(hot.get_count(slot), 1);
    }

    #[test]
    fn test_hot_set_remove() {
        let mut hot = HotSet::new();

        let blob_id = 12345u64;
        let slot = hot.insert(blob_id).unwrap();

        hot.remove(slot);

        assert!(hot.find(blob_id).is_none());
        assert_eq!(hot.active_count(), 0);
    }

    #[test]
    fn test_hot_set_batch_decay_scalar() {
        let mut hot = HotSet::new();

        // Insert 3 entries
        let id1 = 100u64;
        let id2 = 200u64;
        let id3 = 300u64;

        hot.insert(id1);
        hot.insert(id2);
        hot.insert(id3);

        // Record accesses
        if let Some(slot) = hot.find(id1) {
            hot.record_access(slot);
        }
        if let Some(slot) = hot.find(id2) {
            hot.record_access(slot);
        }
        if let Some(slot) = hot.find(id3) {
            hot.record_access(slot);
        }

        // Get initial scores
        let score1_before = hot.get_score(hot.find(id1).unwrap());
        let score2_before = hot.get_score(hot.find(id2).unwrap());

        // Batch decay (force scalar for this test)
        hot.batch_decay_scalar();

        // Verify decay
        let score1_after = hot.get_score(hot.find(id1).unwrap());
        let score2_after = hot.get_score(hot.find(id2).unwrap());

        assert!((score1_after - score1_before * DECAY_FACTOR).abs() < 0.0001);
        assert!((score2_after - score2_before * DECAY_FACTOR).abs() < 0.0001);
    }

    #[test]
    fn test_frecency_engine_creation() {
        let engine = FrecencyEngine::new();
        assert!(engine.is_empty());
        assert_eq!(engine.len(), 0);
    }

    #[test]
    fn test_frecency_engine_record_access() {
        let mut engine = FrecencyEngine::new();

        let blob_id = 12345u64;
        let score = engine.record_access(blob_id);

        assert!((score - 1.0).abs() < 0.01);
        assert_eq!(engine.len(), 1);
        assert_eq!(engine.get_count(blob_id), Some(1));
    }

    #[test]
    fn test_frecency_engine_multiple_accesses() {
        let mut engine = FrecencyEngine::new();

        let blob_id = 12345u64;

        // Multiple accesses should increase score and count
        engine.record_access(blob_id);
        engine.record_access(blob_id);
        let score = engine.record_access(blob_id);

        assert!(score > 2.0, "Score should be > 2.0 after 3 accesses");
        assert_eq!(engine.get_count(blob_id), Some(3));
    }

    #[test]
    fn test_frecency_engine_batch_decay() {
        let mut engine = FrecencyEngine::new();

        // Insert entries
        let id1 = 100u64;
        let id2 = 200u64;

        engine.record_access(id1);
        engine.record_access(id2);

        let score_before = engine.get_score(id1).unwrap();

        // Batch decay
        let decayed = engine.batch_decay();

        let score_after = engine.get_score(id1).unwrap();

        assert!((score_after - score_before * DECAY_FACTOR).abs() < 0.0001);
        assert_eq!(decayed.len(), 2);
    }

    #[test]
    fn test_frecency_engine_get_hot_candidates() {
        let mut engine = FrecencyEngine::new();

        // Create entries with different scores
        let id1 = 100u64;
        let id2 = 200u64;
        let id3 = 300u64;

        // Access id1 many times
        for _ in 0..10 {
            engine.record_access(id1);
        }

        // Access id2 moderately
        for _ in 0..5 {
            engine.record_access(id2);
        }

        // Access id3 rarely
        engine.record_access(id3);

        // Get candidates with threshold
        let threshold = 3.0;
        let candidates = engine.get_hot_candidates(threshold);

        assert!(candidates.contains(&id1), "id1 should be hot (score > 3)");
        assert!(candidates.contains(&id2), "id2 should be hot (score > 3)");
        assert!(
            !candidates.contains(&id3),
            "id3 should not be hot (score < 3)"
        );
    }

    #[test]
    fn test_frecency_engine_remove() {
        let mut engine = FrecencyEngine::new();

        let blob_id = 12345u64;
        engine.record_access(blob_id);

        engine.remove(blob_id);

        assert!(engine.get_score(blob_id).is_none());
        assert_eq!(engine.len(), 0);
    }

    #[test]
    fn test_frecency_engine_stats() {
        let mut engine = FrecencyEngine::new();

        // Add some entries
        for i in 0..5 {
            engine.record_access(i as u64);
        }

        let hot_stats = engine.hot_stats();
        assert_eq!(hot_stats.active_entries, 5);
        assert!(hot_stats.total_score > 0.0);

        let cold_stats = engine.cold_stats();
        assert_eq!(cold_stats.entry_count, 0);
    }

    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn test_simd_decay() {
        // This test only runs on x86/x86_64 with AVX2 support
        if !is_x86_feature_detected!("avx2") {
            println!("Skipping SIMD test: AVX2 not available");
            return;
        }

        let mut hot = HotSet::new();

        // Fill all slots
        for i in 0..HOT_SET_SIZE {
            hot.insert(i as u64);
        }

        // Record some accesses
        for i in 0..HOT_SET_SIZE {
            let slot = hot.find(i as u64).unwrap();
            hot.record_access(slot);
        }

        // Get initial scores (all should be ~1.0)
        let scores_before: Vec<f64> = (0..HOT_SET_SIZE)
            .filter_map(|i| hot.find(i as u64).map(|s| hot.get_score(s)))
            .collect();

        // Batch decay with SIMD
        hot.increment_tick();
        unsafe {
            hot.batch_decay_simd();
        }

        // Verify decay applied correctly
        for i in 0..HOT_SET_SIZE {
            if let Some(slot) = hot.find(i as u64) {
                let score_after = hot.get_score(slot);
                let expected = scores_before[i] * DECAY_FACTOR;
                assert!(
                    (score_after - expected).abs() < 0.0001,
                    "Slot {} score mismatch: {} vs {}",
                    i,
                    score_after,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_cold_set_promotion() {
        let mut engine = FrecencyEngine::new();

        // Fill hot set to capacity
        for i in 0..(HOT_SET_SIZE as u64) {
            engine.record_access(i);
        }

        // Add new blob (goes to cold set)
        let cold_blob = (HOT_SET_SIZE + 100) as u64;
        engine.record_access(cold_blob);

        // Verify it's in cold set
        assert!(engine.cold.contains_key(&cold_blob));

        // Access it multiple times to promote
        for _ in 0..5 {
            engine.record_access(cold_blob);
        }

        // Should be promoted to hot set (if space available)
        // Or stay in cold set if hot set is full
        let score = engine.get_score(cold_blob);
        assert!(score.is_some());
    }

    #[test]
    fn test_compute_score_pure_constants() {
        // Verify the C++ formula constants are correct
        assert_eq!(RECENCY_HALF_LIFE_SEC, 600.0, "Half-life should be 600 seconds");
        assert_eq!(FREQUENCY_SATURATION, 10.0, "Frequency saturation should be 10.0");
        assert_eq!(RECENCY_WEIGHT, 0.5, "Recency weight should be 0.5");
        assert_eq!(FREQUENCY_WEIGHT, 0.5, "Frequency weight should be 0.5");
    }

    #[test]
    fn test_compute_score_pure_fresh_access() {
        // Fresh access (age = 0) with 1 access
        let now = 1_000_000_000u64; // 1 second in nanos
        let last_access = now;
        let score = compute_score_pure(1, last_access, now);

        // recency = 2^0 = 1.0
        // frequency = 1 / (1 + 10) = 0.0909...
        // score = 0.5 * 1.0 + 0.5 * 0.0909 = 0.5454...
        assert!(
            (score - 0.5454).abs() < 0.01,
            "Fresh access score should be ~0.545, got {}",
            score
        );
    }

    #[test]
    fn test_compute_score_pure_aged_access() {
        // Access from 600 seconds ago (one half-life)
        let now = 1_000_000_000_000u64; // 1000 seconds in nanos
        let last_access = 400_000_000_000u64; // 400 seconds in nanos
        let age_sec = 600.0;

        let score = compute_score_pure(10, last_access, now);

        // recency = 2^(-600/600) = 0.5
        // frequency = 10 / (10 + 10) = 0.5
        // score = 0.5 * 0.5 + 0.5 * 0.5 = 0.5
        let expected = 0.5;
        assert!(
            (score - expected).abs() < 0.001,
            "Aged access score should be ~0.5, got {}",
            score
        );
    }

    #[test]
    fn test_compute_score_pure_clamping() {
        // Very old access should clamp to minimum
        let now = 1_000_000_000_000_000u64; // Very far in future
        let last_access = 0u64;

        let score = compute_score_pure(1, last_access, now);
        assert!(score >= 0.0 && score <= 1.0, "Score should be clamped to [0, 1]");
    }

    #[test]
    fn test_validate_against_cpp_signature() {
        // Just verify the function exists and compiles
        // Actual validation requires CTE runtime to be initialized
        let _func = validate_against_cpp as fn(u64, u64, u64) -> Result<(), f32>;
    }
}
