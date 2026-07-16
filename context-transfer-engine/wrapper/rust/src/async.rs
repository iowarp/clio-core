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

//! Asynchronous CTE API (default feature)
//!
//! This module provides async/await wrappers around the blocking CTE FFI.
//! Uses `tokio::task::spawn_blocking` to bridge C++ blocking calls.
//!
//! # Example
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
//!     // Store data asynchronously
//!     tag.put_blob("data.bin".to_string(), b"hello".to_vec(), 0, 1.0).await?;
//!
//!     // Retrieve data asynchronously
//!     let data = tag.get_blob("data.bin".to_string(), 5, 0).await?;
//!     assert_eq!(data, b"hello");
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
//! # Thread Safety Guarantees
//!
//! This module uses `spawn_blocking` to ensure thread-safe access to C++ objects.
//! See the `SendableTag` and `SendableClient` wrappers for SAFETY documentation.

pub use crate::sync::frecency_compute_score;
pub use crate::sync::init;
pub use crate::sync::MAX_BLOB_SIZE;
pub use crate::types::{BdevType, CteOp, CteTagId, CteTelemetry, PoolQuery, RuntimeMode, SteadyTime};

use crate::error::{CteError, CteResult};
use crate::ffi::ffi;
use std::sync::Arc;

/// Wrapper to make UniquePtr<ffi::Tag> Send for use with spawn_blocking
///
/// This wrapper is necessary because `cxx::UniquePtr` does not implement `Send`
/// by default, which prevents it from being used across thread boundaries.
///
/// # Thread Safety
///
/// The wrapper ensures thread-safe access through the following mechanisms:
///
/// 1. **spawn_blocking Isolation**: The FFI call executes in a dedicated blocking
///    thread pool managed by tokio. This ensures the C++ code never runs on the
///    async executor's threads, preventing any potential interference with async
///    scheduling.
///
/// 2. **Per-Instance Isolation**: Each Tag instance is independent. Concurrent
///    operations should create multiple Tag instances (via `duplicate()`), each
///    running in its own `spawn_blocking` context. This matches the C++ design
///    where each Tag object is independent and thread-safe for its own instance.
///
/// 3. **C++ Thread-Safety Guarantees**: The underlying C++ `clio_cte::core::Tag`
///    class is designed for single-threaded use within each operation. All state
///    modifications are completed before returning from FFI calls, and the object
///    does not maintain internal threads or async state that could cause races.
///
/// 4. **No Interior Mutability**: The C++ Tag class does not use interior mutability
///    patterns that could cause data races. All mutations go through explicit FFI
///    calls.
///
/// # SAFETY
///
/// This implementation is safe because:
///
/// - Each Tag instance is used exclusively within `spawn_blocking` closures,
///   ensuring the C++ object is only accessed from one thread at a time.
///
/// - `spawn_blocking` guarantees that the closure runs on a dedicated thread pool
///   separate from the async executor, preventing async runtime interference.
///
/// - For concurrent operations, users create separate Tag instances via `duplicate()`,
///   each wrapped in its own `spawn_blocking` context. This eliminates shared state
///   at the Rust level entirely.
///
/// - The underlying C++ `Tag` object does not use callbacks, signals, or any
///   other mechanism that could cause re-entrancy or cross-thread access.
///
/// - The C++ object lifetime is managed by `UniquePtr`, which ensures proper
///   destruction when the Rust wrapper is dropped.
///
/// - No mutable static state exists in the C++ Tag implementation that could
///   cause cross-thread interference.
struct SendableTag(cxx::UniquePtr<ffi::Tag>);

// SAFETY: SendableTag is safe to send across threads because:
//
// 1. SPAWN_BLOCKING GUARANTEE: The UniquePtr<Tag> is created within a
//    spawn_blocking closure and used exclusively within that closure.
//    No concurrent access to the Tag object can occur.
//
// 2. SINGLE-THREADED ACCESS: Each Tag instance is accessed only from the
//    thread running the spawn_blocking closure. The UniquePtr is never
//    shared across threads for the same Tag instance.
//
// 3. C++ THREAD-SAFETY: The underlying clio_cte::core::Tag class is
//    designed for single-threaded use. It doesn't spawn threads, use atomics,
//    or have any internal concurrency. All state changes are completed
//    before the FFI call returns.
//
// 4. OWNERSHIP MODEL: UniquePtr ensures proper cleanup. The C++ destructor
//    runs exactly once when the UniquePtr is dropped, in the same thread
//    that created it (within the spawn_blocking closure).
//
// 5. NO SHARED STATE BETWEEN INSTANCES: For concurrent operations, users
//    create separate Tag instances via duplicate(). Each instance has its
//    own UniquePtr<Tag>, eliminating any possibility of shared mutable state.
//
// IMPORTANT: This Send impl is safe because each Tag instance's UniquePtr
// lives entirely within spawn_blocking closures, ensuring single-threaded
// access patterns required by the C++ implementation.
unsafe impl Send for SendableTag {}

// SAFETY: SendableTag is safe to synchronize across threads because:
// 1. Arc<SendableTag> requires SendableTag: Sync for Arc: Send
// 2. Each spawn_blocking closure owns its Arc clone (stack-allocated)
// 3. Concurrent closures access the same SendableTag via Arc, but each
//    closure's &SendableTag is scoped to that closure's stack
// 4. IMPORTANT: Users must use Tag::duplicate() for concurrent operations
//    to avoid concurrent access to the same C++ Tag object
unsafe impl Sync for SendableTag {}

/// Wrapper to make unique_ptr<ffi::Client> Send for use with spawn_blocking
///
/// This wrapper is necessary because `cxx::UniquePtr` does not implement `Send`
/// by default, which prevents it from being used across thread boundaries.
///
/// # Thread Safety
///
/// The wrapper ensures thread-safe access through the following mechanisms:
///
/// 1. **spawn_blocking Isolation**: The FFI call executes in a dedicated blocking
///    thread pool managed by tokio. Each FFI call runs in isolation.
///
/// 2. **Per-Call Client Creation**: Unlike Tag, Client objects are created fresh
///    for each FFI call within the spawn_blocking closure. This eliminates any
///    need for mutex synchronization since each call gets its own Client instance.
///
/// 3. **C++ Thread-Safety Guarantees**: The underlying C++ `clio_cte::core::Client`
///    class is stateless - it only communicates with the runtime. All state is
///    maintained in the runtime process, not in the Client object itself.
///
/// # SAFETY
///
/// This implementation is safe because:
///
/// - Each FFI call creates a temporary `ffi::client_new()` instance that lives
///   only within the spawn_blocking closure. No state is shared across calls.
///
/// - The C++ Client provides a stateless interface to the runtime. The only
///   shared state is in the runtime process, which uses its own synchronization.
///
/// - No mutable static state exists in the C++ Client implementation that could
///   cause cross-thread interference.
///
/// - The UniquePtr is created, used, and destroyed entirely within the
///   spawn_blocking closure, ensuring proper cleanup in the correct thread.
struct SendableClient(cxx::UniquePtr<ffi::Client>);

// SAFETY: SendableClient is safe to send across threads because:
//
// 1. PER-CALL INSTANCES: Each FFI call creates a fresh Client instance inside
//    spawn_blocking. No Client is shared across threads or calls.
//
// 2. SPAWN_BLOCKING ISOLATION: The closure runs on a dedicated blocking thread
//    pool, ensuring complete isolation from the async runtime and other blocking
//    tasks for the duration of the call.
//
// 3. C++ STATELESS DESIGN: The underlying clio_cte::core::Client class is
//    stateless. It only communicates with the CTE runtime via IPC. All shared
//    state is in the runtime, which has its own synchronization primitives.
//
// 4. IMMEDIATE CLEANUP: The UniquePtr<Client> is dropped at the end of the
//    spawn_blocking closure, ensuring proper C++ resource cleanup in the same
//    thread that created it.
//
// 5. NO CROSS-CALL STATE: Since each call gets a new Client, there's no
//    possibility of cross-thread state sharing or race conditions.
//
// The Send impl is needed to move the closure into spawn_blocking, but the
// actual access pattern (create, use, destroy) within the closure is safe.
unsafe impl Send for SendableClient {}

// SAFETY: SendableClient is safe to synchronize across threads because:
// 1. Client is stateless and created fresh per call
// 2. Each spawn_blocking closure owns its Client instance
// 3. No shared state between different Client instances
unsafe impl Sync for SendableClient {}

/// Async CTE client
///
/// Provides async methods for client-level operations.
/// Uses spawn_blocking to bridge C++ blocking calls.
pub struct Client {
    _marker: std::marker::PhantomData<()>,
}

impl Client {
    /// Create a new CTE client
    ///
    /// Prerequisites: CTE must be initialized via `init()` or `sync::init()` before calling this method.
    ///
    /// # Returns
    /// * `Ok(Client)` on success
    /// * `Err(CteError::InitFailed)` if CTE is not initialized
    ///
    /// # Example
    /// ```no_run
    /// use clio_cte::{init, Client};
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     // Initialize CTE first
    ///     init("")?;
    ///     
    ///     // Then create client
    ///     let client = Client::new().await?;
    ///     Ok(())
    /// }
    /// ```
    pub async fn new() -> CteResult<Self> {
        // Verify CTE is initialized (but don't call init() - user must initialize)
        // This check is thread-safe via OnceLock in sync::init
        crate::sync::init("")?;
        Ok(Self {
            _marker: std::marker::PhantomData,
        })
    }

    /// Poll telemetry log from CTE
    ///
    /// # Arguments
    /// * `min_time` - Minimum timestamp to fetch (0 for all)
    ///
    /// # Returns
    /// Vector of telemetry entries
    pub async fn poll_telemetry(&self, min_time: u64, timeout_sec: f32) -> CteResult<Vec<crate::ffi::CteTelemetry>> {
        tokio::task::spawn_blocking(move || {
            let client = SendableClient(ffi::client_new());
            let mut raw = Vec::new();
            let ret = ffi::client_poll_telemetry_raw(&client.0, min_time, timeout_sec, &mut raw);
            match ret {
                0 => Ok(crate::ffi::parse_telemetry(&raw)),
                1 => Err(crate::CteError::Timeout),
                2 => Err(crate::CteError::RuntimeError {
                    code: 1,
                    message: "Telemetry poll failed".to_string(),
                }),
                code => Err(crate::CteError::RuntimeError {
                    code: code as u32,
                    message: format!("Unknown return code: {}", code),
                }),
            }
        })
        .await
        .map_err(|e| CteError::FfiError {
            message: format!("Failed to poll telemetry: spawn_blocking error: {}", e),
        })?
    }

    /// Trigger the CTE's internal DynamicReorganize task (async).
    ///
    /// Delegates reorganization to the built-in data organizer (issue #738)
    /// instead of running a per-blob reorganization loop in Rust.
    ///
    /// # Arguments
    /// * `replica_id` - Which replica (0-based) to fire
    /// * `period_us` - Period in microseconds for the periodic task
    ///
    /// # Returns
    /// * `Ok(0)` on success
    /// * `Err(CteError::FfiError)` on FFI error
    pub async fn dynamic_reorganize(&self, replica_id: u32, period_us: f64) -> CteResult<i32> {
        tokio::task::spawn_blocking(move || {
            let rc = ffi::client_dynamic_reorganize(replica_id, period_us);
            if rc != 0 {
                return Err(CteError::FfiError {
                    message: format!(
                        "dynamic_reorganize failed with rc={}",
                        rc
                    ),
                });
            }
            Ok(rc)
        })
        .await
        .map_err(|e| CteError::FfiError {
            message: format!("dynamic_reorganize spawn_blocking error: {}", e),
        })?
    }

    /// Reorganize a blob (change placement score)
    ///
    /// # Arguments
    /// * `tag_id` - ID of the tag containing the blob
    /// * `name` - Blob name (must not be empty)
    /// * `score` - New placement score (0.0-1.0)
    ///
    /// # Returns
    /// * `Ok(())` on success
    /// * `Err(CteError::InvalidParameter)` if name is empty or score is out of range
    /// * `Err(CteError::RuntimeError)` on failure
    pub async fn reorganize_blob(
        &self,
        tag_id: CteTagId,
        name: String,
        score: f32,
    ) -> CteResult<()> {
        // Validate inputs
        if name.is_empty() {
            return Err(CteError::InvalidParameter {
                message: "Blob name cannot be empty".to_string(),
            });
        }
        if score < 0.0 || score > 1.0 || score.is_nan() {
            return Err(CteError::InvalidParameter {
                message: format!("Score must be between 0.0 and 1.0, got {}", score),
            });
        }

        tokio::task::spawn_blocking(move || {
            let client = SendableClient(ffi::client_new());
            let rc = ffi::client_reorganize_blob(
                &client.0,
                tag_id.major,
                tag_id.minor,
                &name,
                score,
            );
            if rc == 0 {
                Ok(())
            } else {
                Err(CteError::RuntimeError {
                    code: rc as u32,
                    message: format!(
                        "Failed to reorganize blob '{}' in tag {}.{} with score {}: error code {}",
                        name, tag_id.major, tag_id.minor, score, rc
                    ),
                })
            }
        })
        .await
        .map_err(|e| CteError::FfiError {
            message: format!("Reorganize blob spawn_blocking error: {}", e),
        })?
    }

    /// Delete a blob
    ///
    /// # Arguments
    /// * `tag_id` - ID of the tag containing the blob
    /// * `name` - Blob name (must not be empty)
    ///
    /// # Returns
    /// * `Ok(())` on success
    /// * `Err(CteError::InvalidParameter)` if name is empty
    /// * `Err(CteError::RuntimeError)` on failure
    pub async fn del_blob(&self, tag_id: CteTagId, name: String) -> CteResult<()> {
        // Validate inputs
        if name.is_empty() {
            return Err(CteError::InvalidParameter {
                message: "Blob name cannot be empty".to_string(),
            });
        }

        tokio::task::spawn_blocking(move || {
            let client = SendableClient(ffi::client_new());
            let rc = ffi::client_del_blob(
                &client.0,
                tag_id.major,
                tag_id.minor,
                &name,
            );
            if rc == 0 {
                Ok(())
            } else {
                Err(CteError::RuntimeError {
                    code: rc as u32,
                    message: format!(
                        "Failed to delete blob '{}' in tag {}.{}: error code {}",
                        name, tag_id.major, tag_id.minor, rc
                    ),
                })
            }
        })
        .await
        .map_err(|e| CteError::FfiError {
            message: format!("Delete blob spawn_blocking error: {}", e),
        })?
    }
}

/// Async tag wrapper
///
/// Provides async methods for tag/blob operations.
/// Uses `spawn_blocking` to bridge C++ blocking calls.
///
/// # Thread Safety
///
/// Each Tag instance is independent. For concurrent operations, create multiple
/// Tag instances using `duplicate()`. Each instance runs its FFI calls in a
/// dedicated blocking thread pool via `spawn_blocking`.
///
/// # Example (Concurrent Operations)
/// ```no_run
/// use clio_cte::Tag;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     // Create independent Tag instances for concurrent operations
///     let tag1 = Tag::new("my_dataset").await?;
///     let tag2 = tag1.duplicate().await?;  // Fresh instance for concurrent use
///
///     // Run operations concurrently
///     let handle1 = tokio::spawn(async move {
///         tag1.put_blob("data1.bin".to_string(), b"hello".to_vec(), 0, 1.0).await
///     });
///     let handle2 = tokio::spawn(async move {
///         tag2.get_blob("data2.bin".to_string(), 5, 0).await
///     });
///
///     handle1.await??;
///     handle2.await??;
///     Ok(())
/// }
/// ```
pub struct Tag {
    inner: Arc<SendableTag>,
    /// Tag name, if created via Tag::new(). None if created via Tag::from_id().
    name: Option<String>,
}

impl Tag {
    /// Create or get a tag by name
    ///
    /// # Arguments
    /// * `name` - Tag name (must not be empty)
    ///
    /// # Returns
    /// * `Ok(Tag)` on success
    /// * `Err(CteError::FfiError)` on spawn_blocking failure
    /// * `Err(CteError::InvalidParameter)` if name is empty
    ///
    /// # Example
    /// ```no_run
    /// use clio_cte::Tag;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let tag = Tag::new("my_dataset").await?;
    ///     Ok(())
    /// }
    /// ```
    pub async fn new(name: &str) -> CteResult<Self> {
        if name.is_empty() {
            return Err(CteError::InvalidParameter {
                message: "Tag name cannot be empty".to_string(),
            });
        }

        let name = name.to_string();
        let name_clone = name.clone();
        let sendable_tag = tokio::task::spawn_blocking(move || {
            SendableTag(ffi::tag_new(&name_clone))
        })
            .await
            .map_err(|e| CteError::FfiError {
                message: format!("Failed to create tag '{}': spawn_blocking error: {}", name, e),
            })?;

        Ok(Self {
            inner: Arc::new(sendable_tag),
            name: Some(name),
        })
    }

    /// Open an existing tag by ID
    ///
    /// # Arguments
    /// * `id` - Tag ID
    ///
    /// # Returns
    /// * `Ok(Tag)` on success
    /// * `Err(CteError::FfiError)` on spawn_blocking failure
    ///
    /// # Example
    /// ```no_run
    /// use clio_cte::{Tag, CteTagId};
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let id = CteTagId { major: 1, minor: 2 };
    ///     let tag = Tag::from_id(id).await?;
    ///     Ok(())
    /// }
    /// ```
    pub async fn from_id(id: CteTagId) -> CteResult<Self> {
        let (major, minor) = (id.major, id.minor);
        let sendable_tag = tokio::task::spawn_blocking(move || {
            SendableTag(ffi::tag_from_id(major, minor))
        })
            .await
            .map_err(|e| CteError::FfiError {
                message: format!(
                    "Failed to open tag {}.{}: spawn_blocking error: {}",
                    major, minor, e
                ),
            })?;

        Ok(Self {
            inner: Arc::new(sendable_tag),
            name: None,  // Created from ID, name unknown
        })
    }

    /// Get the tag ID
    ///
    /// # Returns
    /// The unique identifier for this tag.
    ///
    /// # Example
    /// ```no_run
    /// use clio_cte::Tag;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let tag = Tag::new("my_dataset").await?;
    ///     let id = tag.get_id().await?;
    ///     println!("Tag ID: major={}, minor={}", id.major, id.minor);
    ///     Ok(())
    /// }
    /// ```
    pub async fn get_id(&self) -> CteResult<CteTagId> {
        // Clone the Arc to move into spawn_blocking
        let inner = Arc::clone(&self.inner);
        
        tokio::task::spawn_blocking(move || {
            let tag_ref = inner.0.as_ref().unwrap();  // Use UniquePtr directly
            let major = ffi::tag_get_id_major(tag_ref);
            let minor = ffi::tag_get_id_minor(tag_ref);
            CteTagId { major, minor }
        })
            .await
            .map_err(|e| CteError::FfiError {
                message: format!("Get tag ID spawn_blocking error: {}", e),
            })
    }
    
    /// Create a fresh independent Tag instance with the same name
    ///
    /// Use this method to create multiple Tag instances for concurrent operations.
    /// Each instance is independent and can be used in separate async tasks.
    ///
    /// # Returns
    /// * `Ok(Tag)` - A fresh Tag instance with the same name
    /// * `Err(CteError::InvalidParameter)` if this Tag was created from ID (no name stored)
    ///
    /// # Example
    /// ```no_run
    /// use clio_cte::Tag;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let tag1 = Tag::new("my_dataset").await?;
    ///     let tag2 = tag1.duplicate().await?;  // Fresh instance for concurrent use
    ///
    ///     // Run operations concurrently
    ///     let h1 = tokio::spawn(async move {
    ///         tag1.put_blob("a.bin".to_string(), vec![1, 2, 3], 0, 1.0).await
    ///     });
    ///     let h2 = tokio::spawn(async move {
    ///         tag2.put_blob("b.bin".to_string(), vec![4, 5, 6], 0, 1.0).await
    ///     });
    ///
    ///     h1.await??;
    ///     h2.await??;
    ///     Ok(())
    /// }
    /// ```
    pub async fn duplicate(&self) -> CteResult<Self> {
        let name = self.name.clone().ok_or_else(|| {
            CteError::InvalidParameter {
                message: "Cannot duplicate a Tag created from ID (name unknown)".to_string(),
            }
        })?;
        
        Self::new(&name).await
    }

    /// Get the placement score of a blob
    ///
    /// # Arguments
    /// * `name` - Blob name (must not be empty)
    ///
    /// # Returns
    /// Score value (0.0-1.0)
    ///
    /// # Errors
    /// * `CteError::InvalidParameter` if name is empty
    /// * `CteError::FfiError` on spawn_blocking failure
    ///
    /// # Example
    /// ```no_run
    /// use clio_cte::Tag;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let tag = Tag::new("my_dataset").await?;
    ///     let score = tag.get_blob_score("data.bin").await?;
    ///     println!("Score: {}", score);
    ///     Ok(())
    /// }
    /// ```
    pub async fn get_blob_score(&self, name: &str) -> CteResult<f32> {
        if name.is_empty() {
            return Err(CteError::InvalidParameter {
                message: "Blob name cannot be empty".to_string(),
            });
        }

        let inner = Arc::clone(&self.inner);
        let name = name.to_string();
        
        tokio::task::spawn_blocking(move || {
            let tag_ref = inner.0.as_ref().unwrap();  // Use UniquePtr directly
            ffi::tag_get_blob_score(tag_ref, &name)
        })
            .await
            .map_err(|e| CteError::FfiError {
                message: format!("Get blob score spawn_blocking error: {}", e),
            })
    }

    /// Reorganize a blob (change placement score)
    ///
    /// # Arguments
    /// * `name` - Blob name (must not be empty)
    /// * `score` - New placement score (0.0-1.0)
    ///
    /// # Returns
    /// * `Ok(())` on success
    /// * `Err(CteError::InvalidParameter)` if name is empty or score out of range
    /// * `Err(CteError::RuntimeError)` on failure
    ///
    /// # Example
    /// ```no_run
    /// use clio_cte::Tag;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let tag = Tag::new("my_dataset").await?;
    ///     tag.reorganize_blob("data.bin".to_string(), 0.5).await?;
    ///     Ok(())
    /// }
    /// ```
    pub async fn reorganize_blob(&self, name: String, score: f32) -> CteResult<()> {
        if name.is_empty() {
            return Err(CteError::InvalidParameter {
                message: "Blob name cannot be empty".to_string(),
            });
        }
        if score < 0.0 || score > 1.0 || score.is_nan() {
            return Err(CteError::InvalidParameter {
                message: format!("Score must be between 0.0 and 1.0, got {}", score),
            });
        }

        let inner = Arc::clone(&self.inner);

        tokio::task::spawn_blocking(move || {
            let tag_ref = inner.0.as_ref().unwrap();  // Use UniquePtr directly
            let rc = ffi::tag_reorganize_blob(tag_ref, &name, score);
            if rc == 0 {
                Ok(())
            } else {
                let tag_id_major = ffi::tag_get_id_major(tag_ref);
                let tag_id_minor = ffi::tag_get_id_minor(tag_ref);
                Err(CteError::RuntimeError {
                    code: rc as u32,
                    message: format!(
                        "Failed to reorganize blob '{}' in tag {}.{} with score {}: error code {}",
                        name, tag_id_major, tag_id_minor, score, rc
                    ),
                })
            }
        })
            .await
            .map_err(|e| CteError::FfiError {
                message: format!("Reorganize blob spawn_blocking error: {}", e),
            })?
    }

    /// Write data into a blob
    ///
    /// # Arguments
    /// * `name` - Blob name (must not be empty)
    /// * `data` - Data buffer to write
    /// * `offset` - Offset within the blob to write to
    /// * `score` - Placement score (0.0-1.0)
    ///
    /// # Returns
    /// * `Ok(())` on success
    /// * `Err(CteError::InvalidParameter)` if name is empty, score out of range,
    ///   data exceeds MAX_BLOB_SIZE, or offset overflows
    /// * `Err(CteError::FfiError)` on spawn_blocking failure
    ///
    /// # Example
    /// ```no_run
    /// use clio_cte::Tag;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let tag = Tag::new("my_dataset").await?;
    ///     tag.put_blob("data.bin".to_string(), b"hello".to_vec(), 0, 1.0).await?;
    ///     Ok(())
    /// }
    /// ```
    pub async fn put_blob(&self, name: String, data: Vec<u8>, offset: u64, score: f32) -> CteResult<()> {
        if name.is_empty() {
            return Err(CteError::InvalidParameter {
                message: "Blob name cannot be empty".to_string(),
            });
        }
        if score < 0.0 || score > 1.0 || score.is_nan() {
            return Err(CteError::InvalidParameter {
                message: format!("Score must be between 0.0 and 1.0, got {}", score),
            });
        }

        // Check blob size limit
        let data_len = data.len() as u64;
        if data_len > MAX_BLOB_SIZE {
            return Err(CteError::InvalidParameter {
                message: format!(
                    "Data size {} exceeds maximum blob size {}",
                    data_len, MAX_BLOB_SIZE
                ),
            });
        }

        // Check for offset overflow
        let end_offset = offset.checked_add(data_len).ok_or_else(|| {
            CteError::InvalidParameter {
                message: format!(
                    "Offset {} + size {} would overflow u64",
                    offset, data_len
                ),
            }
        })?;

        if end_offset > MAX_BLOB_SIZE {
            return Err(CteError::InvalidParameter {
                message: format!(
                    "Total blob size {} exceeds maximum {}",
                    end_offset, MAX_BLOB_SIZE
                ),
            });
        }

        let inner = Arc::clone(&self.inner);

        tokio::task::spawn_blocking(move || {
            let tag_ref = inner.0.as_ref().unwrap();  // Use UniquePtr directly
            ffi::tag_put_blob(tag_ref, &name, &data, offset, score);
            Ok(())
        })
            .await
            .map_err(|e| CteError::FfiError {
                message: format!("Put blob spawn_blocking error: {}", e),
            })?
    }

    /// Zero-copy async put. The `ShmBuffer` is decomposed into Send primitives
    /// (handle + ptr + len) that move into a `spawn_blocking` closure; the closure
    /// does submit → wait → destroy → free, keeping the SHM region alive until the
    /// runtime completes the async put (wait-before-free).
    ///
    /// The caller must have already written the blob data into `buf` (via
    /// `buf.as_mut_slice()` / `as_mut_ptr()`) before calling this.
    ///
    /// # Arguments
    /// * `name` - Blob name (must not be empty)
    /// * `buf` - Caller-allocated SHM buffer containing the blob data
    /// * `offset` - Offset within the blob to write to
    /// * `score` - Placement score (0.0-1.0)
    ///
    /// # Returns
    /// * `Ok(())` on success
    /// * `Err(CteError::InvalidParameter)` if name is empty, score out of range,
    ///   or data exceeds MAX_BLOB_SIZE
    /// * `Err(CteError::ShmNullHandle)` if the buffer handle is null
    /// * `Err(CteError::FfiError)` on FFI failure
    ///
    /// # Example
    /// ```no_run
    /// use clio_cte::{Tag, zerocopy::ShmBuffer};
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let tag = Tag::new("my_dataset").await?;
    ///     let mut buf = ShmBuffer::alloc(1024)?;
    ///     buf.as_mut_slice().copy_from_slice(b"hello");
    ///     tag.put_blob_shm("data.bin".to_string(), buf, 0, 1.0).await?;
    ///     Ok(())
    /// }
    /// ```
    pub async fn put_blob_shm(
        &self,
        name: String,
        buf: crate::zerocopy::ShmBuffer,
        offset: u64,
        score: f32,
    ) -> CteResult<()> {
        // Validate inputs
        if name.is_empty() {
            return Err(CteError::InvalidParameter {
                message: "Blob name cannot be empty".to_string(),
            });
        }
        if score < 0.0 || score > 1.0 || score.is_nan() {
            return Err(CteError::InvalidParameter {
                message: format!("Score must be between 0.0 and 1.0, got {}", score),
            });
        }

        let size = buf.len() as u64;
        if size > MAX_BLOB_SIZE {
            return Err(CteError::InvalidParameter {
                message: format!(
                    "Data size {} exceeds maximum blob size {}",
                    size, MAX_BLOB_SIZE
                ),
            });
        }

        // Decompose the ShmBuffer into Send primitives (handle, ptr, len) so they
        // can move into spawn_blocking. into_parts nulls the ShmBuffer's handle so
        // its Drop is a no-op — the closure takes over freeing via cte_free_shm_buffer.
        let (handle, _ptr, _len) = buf.into_parts();
        if handle.is_null() {
            return Err(CteError::ShmNullHandle);
        }

        let inner = Arc::clone(&self.inner);

        tokio::task::spawn_blocking(move || {
            // Build a null-terminated C name
            let c_name = std::ffi::CString::new(name).map_err(|e| CteError::FfiError {
                message: format!("blob name has interior NUL: {}", e),
            })?;

            // Get the raw Tag pointer (SendableTag wraps UniquePtr<ffi::Tag>)
            let tag_ref = inner.0.as_ref().ok_or_else(|| CteError::FfiError {
                message: "Tag inner UniquePtr is null".to_string(),
            })?;
            let tag_ptr = std::ptr::addr_of!(*tag_ref) as *const std::ffi::c_void;

            // Submit
            let mut future: crate::ffi_c::CteFutureHandle = std::ptr::null_mut();
            let rc = unsafe {
                crate::ffi_c::cte_tag_async_put_shm(
                    tag_ptr,
                    c_name.as_ptr(),
                    offset,
                    size,
                    handle,
                    score,
                    &mut future,
                )
            };

            if rc != 0 {
                // Submit failed — free the SHM buffer (no future to wait on)
                unsafe { crate::ffi_c::cte_free_shm_buffer(handle) };
                return Err(CteError::FfiError {
                    message: format!("cte_tag_async_put_shm failed rc={}", rc),
                });
            }

            // WAIT before free — runtime is reading the SHM region asynchronously
            let mut wait_rc: i32 = 0;
            let wait_status = unsafe {
                crate::ffi_c::cte_future_wait(future, -1.0, &mut wait_rc)
            };

            unsafe { crate::ffi_c::cte_future_destroy(future) };

            // NOW free the SHM buffer (runtime has finished reading)
            unsafe { crate::ffi_c::cte_free_shm_buffer(handle) };

            if wait_status != 0 {
                return Err(CteError::FfiError {
                    message: format!(
                        "put_blob_shm wait failed (status={}, rc={})",
                        wait_status, wait_rc
                    ),
                });
            }

            if wait_rc != 0 {
                return Err(CteError::FfiError {
                    message: format!("put_blob_shm task failed rc={}", wait_rc),
                });
            }

            Ok(())
        })
        .await
        .map_err(|e| CteError::FfiError {
            message: format!("put_blob_shm spawn_blocking error: {}", e),
        })?
    }

    /// Zero-copy async get. Allocates an SHM output buffer, submits AsyncGetBlob
    /// so CTE writes the blob directly into SHM, waits, reads the bytes into a
    /// `Vec<u8>`, and frees the SHM. This is ONE copy (SHM -> Vec) — better than
    /// the old copying `get_blob` (2 copies). For true zero-copy async, call the
    /// sync `get_blob_shm` inside your own `spawn_blocking` and read `as_slice()`.
    ///
    /// LIFETIME (read-before-free): the closure does submit -> wait -> read ->
    /// destroy -> free. The SHM is not read or freed before the wait completes.
    ///
    /// # Arguments
    /// * `name` - Blob name (must not be empty)
    /// * `size` - Number of bytes to read
    /// * `offset` - Offset within the blob to read from
    ///
    /// # Returns
    /// * `Ok(Vec<u8>)` - Data read from blob (one copy from SHM)
    /// * `Err(CteError::InvalidParameter)` if name is empty, offset overflows,
    ///   or size exceeds MAX_BLOB_SIZE
    /// * `Err(CteError::ShmAllocFailed)` if SHM buffer allocation fails
    /// * `Err(CteError::ShmNullHandle)` if the buffer handle is null
    /// * `Err(CteError::FfiError)` on FFI failure or spawn_blocking error
    ///
    /// # Example
    /// ```no_run
    /// use clio_cte::Tag;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let tag = Tag::new("my_dataset").await?;
    ///     let data = tag.get_blob_shm("data.bin".to_string(), 1024, 0).await?;
    ///     assert_eq!(data.len(), 1024);
    ///     Ok(())
    /// }
    /// ```
    pub async fn get_blob_shm(
        &self,
        name: String,
        size: u64,
        offset: u64,
    ) -> CteResult<Vec<u8>> {
        // Validate inputs (mirror async get_blob validation)
        if name.is_empty() {
            return Err(CteError::InvalidParameter {
                message: "Blob name cannot be empty".to_string(),
            });
        }

        // Check for offset overflow
        let end_offset = offset.checked_add(size).ok_or_else(|| CteError::InvalidParameter {
            message: format!(
                "Offset {} + size {} would overflow u64",
                offset, size
            ),
        })?;

        // Check blob size limit
        if end_offset > MAX_BLOB_SIZE {
            return Err(CteError::InvalidParameter {
                message: format!(
                    "Total read size {} exceeds maximum {}",
                    end_offset, MAX_BLOB_SIZE
                ),
            });
        }

        // Allocate the output SHM buffer OUTSIDE the closure (ShmBuffer is !Send,
        // can't move into spawn_blocking). into_parts extracts the Send handle;
        // _ptr/_len are dropped here (not captured by the closure).
        let buf = crate::zerocopy::ShmBuffer::alloc(size as usize)?;
        let (handle, _ptr, _len) = buf.into_parts();
        if handle.is_null() {
            return Err(CteError::ShmNullHandle);
        }

        let inner = Arc::clone(&self.inner);

        tokio::task::spawn_blocking(move || {
            // Build a null-terminated C name
            let c_name = std::ffi::CString::new(name).map_err(|e| CteError::FfiError {
                message: format!("blob name has interior NUL: {}", e),
            })?;

            // Get the raw Tag pointer (SendableTag wraps UniquePtr<ffi::Tag>)
            let tag_ref = inner.0.as_ref().ok_or_else(|| CteError::FfiError {
                message: "Tag inner UniquePtr is null".to_string(),
            })?;
            let tag_ptr = std::ptr::addr_of!(*tag_ref) as *const std::ffi::c_void;

            // Submit
            let mut future: crate::ffi_c::CteFutureHandle = std::ptr::null_mut();
            let rc = unsafe {
                crate::ffi_c::cte_tag_async_get_shm(
                    tag_ptr,
                    c_name.as_ptr(),
                    offset,
                    size,
                    handle,
                    &mut future,
                )
            };

            if rc != 0 {
                // Submit failed — free the SHM buffer (no future to wait on)
                unsafe { crate::ffi_c::cte_free_shm_buffer(handle) };
                return Err(CteError::FfiError {
                    message: format!("cte_tag_async_get_shm failed rc={}", rc),
                });
            }

            // WAIT before read — CTE is writing into the SHM region right now
            let mut wait_rc: i32 = 0;
            let wait_status = unsafe {
                crate::ffi_c::cte_future_wait(future, -1.0, &mut wait_rc)
            };

            // Resolve the handle to a read pointer (created inside the closure —
            // safe to use here). Valid only after wait completes (CTE has written).
            let read_ptr = unsafe { crate::ffi_c::cte_shm_handle_to_ptr(handle) };

            // Destroy the future
            unsafe { crate::ffi_c::cte_future_destroy(future) };

            // Check wait/task status
            if wait_status != 0 {
                unsafe { crate::ffi_c::cte_free_shm_buffer(handle) };
                return Err(CteError::FfiError {
                    message: format!(
                        "get_blob_shm wait failed (status={}, rc={})",
                        wait_status, wait_rc
                    ),
                });
            }

            if wait_rc != 0 {
                unsafe { crate::ffi_c::cte_free_shm_buffer(handle) };
                return Err(CteError::FfiError {
                    message: format!("get_blob_shm task failed rc={}", wait_rc),
                });
            }

            // READ the bytes from SHM into a Vec (the one copy). Then free the SHM.
            let mut out = Vec::with_capacity(size as usize);
            if !read_ptr.is_null() {
                // SAFETY: read_ptr is valid for `size as usize` bytes (CTE wrote them,
                // wait confirmed). The SHM region is alive until we free it below.
                let slice = unsafe { std::slice::from_raw_parts(read_ptr, size as usize) };
                out.extend_from_slice(slice);
            }

            // NOW free the SHM buffer — runtime is done writing, we've read.
            unsafe { crate::ffi_c::cte_free_shm_buffer(handle) };

            Ok(out)
        })
        .await
        .map_err(|e| CteError::FfiError {
            message: format!("get_blob_shm spawn_blocking error: {}", e),
        })?
    }

    /// Read data from a blob
    ///
    /// # Arguments
    /// * `name` - Blob name (must not be empty)
    /// * `size` - Number of bytes to read
    /// * `offset` - Offset within the blob to read from
    ///
    /// # Returns
    /// The data read from the blob
    ///
    /// # Errors
    /// * `CteError::InvalidParameter` if name is empty
    /// * `CteError::FfiError` on spawn_blocking failure
    ///
    /// # Example
    /// ```no_run
    /// use clio_cte::Tag;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let tag = Tag::new("my_dataset").await?;
    ///     let data = tag.get_blob("data.bin".to_string(), 5, 0).await?;
    ///     assert_eq!(data, b"hello");
    ///     Ok(())
    /// }
    /// ```
    pub async fn get_blob(&self, name: String, size: u64, offset: u64) -> CteResult<Vec<u8>> {
        if name.is_empty() {
            return Err(CteError::InvalidParameter {
                message: "Blob name cannot be empty".to_string(),
            });
        }

        let inner = Arc::clone(&self.inner);

        tokio::task::spawn_blocking(move || {
            let tag_ref = inner.0.as_ref().unwrap();  // Use UniquePtr directly
            let mut out = Vec::new();
            ffi::tag_get_blob(tag_ref, &name, size, offset, &mut out);
            out
        })
            .await
            .map_err(|e| CteError::FfiError {
                message: format!("Get blob spawn_blocking error: {}", e),
            })
    }

    /// Get the size of a blob
    ///
    /// # Arguments
    /// * `name` - Blob name (must not be empty)
    ///
    /// # Returns
    /// Size of the blob in bytes
    ///
    /// # Errors
    /// * `CteError::InvalidParameter` if name is empty
    /// * `CteError::FfiError` on spawn_blocking failure
    ///
    /// # Example
    /// ```no_run
    /// use clio_cte::Tag;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let tag = Tag::new("my_dataset").await?;
    ///     let size = tag.get_blob_size("data.bin").await?;
    ///     println!("Blob size: {} bytes", size);
    ///     Ok(())
    /// }
    /// ```
    pub async fn get_blob_size(&self, name: &str) -> CteResult<u64> {
        if name.is_empty() {
            return Err(CteError::InvalidParameter {
                message: "Blob name cannot be empty".to_string(),
            });
        }

        let inner = Arc::clone(&self.inner);
        let name = name.to_string();

        tokio::task::spawn_blocking(move || {
            let tag_ref = inner.0.as_ref().unwrap();  // Use UniquePtr directly
            ffi::tag_get_blob_size(tag_ref, &name)
        })
            .await
            .map_err(|e| CteError::FfiError {
                message: format!("Get blob size spawn_blocking error: {}", e),
            })
    }

    /// List all blobs in this tag
    ///
    /// # Returns
    /// Vector of blob names
    ///
    /// # Errors
    /// * `CteError::FfiError` on spawn_blocking failure
    ///
    /// # Example
    /// ```no_run
    /// use clio_cte::Tag;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let tag = Tag::new("my_dataset").await?;
    ///     let blobs = tag.get_contained_blobs().await?;
    ///     for blob in blobs {
    ///         println!("Blob: {}", blob);
    ///     }
    ///     Ok(())
    /// }
    /// ```
    pub async fn get_contained_blobs(&self) -> CteResult<Vec<String>> {
        let inner = Arc::clone(&self.inner);

        tokio::task::spawn_blocking(move || {
            let tag_ref = inner.0.as_ref().unwrap();  // Use UniquePtr directly
            let mut out = Vec::new();
            ffi::tag_get_contained_blobs(tag_ref, &mut out);
            out
        })
            .await
            .map_err(|e| CteError::FfiError {
                message: format!("Get contained blobs spawn_blocking error: {}", e),
            })
    }
}

/// Shutdown the CTE runtime
///
/// This function should be called before program exit to properly
/// clean up CTE resources.
///
/// # Note
/// This uses the sync API's shutdown function internally, which must
/// be called from a blocking context.
pub async fn shutdown() -> CteResult<()> {
    tokio::task::spawn_blocking(move || {
        crate::sync::shutdown();
        Ok(())
    })
        .await
        .map_err(|e| CteError::FfiError {
            message: format!("Shutdown spawn_blocking error: {}", e),
        })?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_query_variants() {
        let local = PoolQuery::local();
        let dynamic = PoolQuery::dynamic(30.0);

        assert_eq!(local.net_timeout(), 0.0);
        assert_eq!(dynamic.net_timeout(), 30.0);
    }

    #[test]
    fn test_tag_validation() {
        // This test verifies input validation logic without actual FFI calls
        // Since Tag::new requires FFI, we test validation through async methods

        // Test empty name validation
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(async {
            // Validation happens before FFI call, so this should fail fast
            let tag_ptr = ffi::tag_new("test_tag");
            let tag = Tag {
                inner: Arc::new(SendableTag(tag_ptr)),
                name: Some("test_tag".to_string()),
            };
            
            // Test get_blob_score with empty name (should fail validation)
            tag.get_blob_score("").await
        });

        assert!(result.is_err());
        match result {
            Err(CteError::InvalidParameter { message }) => {
                assert!(message.contains("cannot be empty"));
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_score_validation() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(async {
            let tag_ptr = ffi::tag_new("test_tag");
            let tag = Tag {
                inner: Arc::new(SendableTag(tag_ptr)),
                name: Some("test_tag".to_string()),
            };
            
            // Test with invalid score (< 0)
            tag.reorganize_blob("test".to_string(), -1.0).await
        });

        assert!(result.is_err());
        match result {
            Err(CteError::InvalidParameter { message }) => {
                assert!(message.contains("Score must be between"));
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }
}
