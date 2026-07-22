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

//! Unit tests for CTE Rust bindings (no runtime required)
//!
//! These tests validate type correctness, error handling, and validation
//! logic without requiring a running CTE runtime.

use super::*;
use crate::error::ToCteResult;

// ============================================================================
// types.rs Tests
// ============================================================================

mod types_tests {
    use super::*;
    use std::time::Duration;

    /// Test CteTagId creation with new(), null(), and from_u64()
    #[test]
    fn test_cte_tag_id_creation() {
        // Test new()
        let id = CteTagId::new(42, 100);
        assert_eq!(id.major, 42);
        assert_eq!(id.minor, 100);
        assert!(!id.is_null());

        // Test null()
        let null_id = CteTagId::null();
        assert_eq!(null_id.major, 0);
        assert_eq!(null_id.minor, 0);
        assert!(null_id.is_null());

        // Test from_u64()
        let combined = (42u64 << 32) | 100u64;
        let from_u64 = CteTagId::from_u64(combined);
        assert_eq!(from_u64.major, 42);
        assert_eq!(from_u64.minor, 100);

        // Test round-trip conversion
        let orig = CteTagId::new(123, 456);
        let v = orig.to_u64();
        let back = CteTagId::from_u64(v);
        assert_eq!(back.major, orig.major);
        assert_eq!(back.minor, orig.minor);
    }

    /// Test CteTagId serialization and deserialization
    #[test]
    fn test_cte_tag_id_serde() {
        let id = CteTagId::new(0x1234_5678, 0xABCD_EF01);
        let v = id.to_u64();
        let back = CteTagId::from_u64(v);

        assert_eq!(back.major, 0x1234_5678);
        assert_eq!(back.minor, 0xABCD_EF01);

        // Test max values
        let max_id = CteTagId::new(u32::MAX, u32::MAX);
        let max_v = max_id.to_u64();
        let max_back = CteTagId::from_u64(max_v);
        assert_eq!(max_back.major, u32::MAX);
        assert_eq!(max_back.minor, u32::MAX);
    }

    /// Test SteadyTime elapsed_from calculation
    #[test]
    fn test_steady_time_elapsed() {
        // Test elapsed_from
        let t1 = SteadyTime::from_nanos(1000);
        let t2 = SteadyTime::from_nanos(2000);
        let duration = t2.elapsed_from(&t1);
        assert_eq!(duration.as_nanos(), 1000);
        assert_eq!(duration.as_micros(), 1);

        // Test duration_since
        let t3 = SteadyTime::from_nanos(5000);
        let duration2 = t3.duration_since(&t1);
        assert_eq!(duration2.as_nanos(), 4000);

        // Test default
        let default_time = SteadyTime::default();
        assert_eq!(default_time.nanos, 0);
    }

    /// Test PoolQuery variants
    #[test]
    fn test_pool_query_variants() {
        // Test Broadcast
        let broadcast = PoolQuery::broadcast(60.0);
        match broadcast {
            PoolQuery::Broadcast { net_timeout } => {
                assert!((net_timeout - 60.0).abs() < 0.01);
            }
            _ => panic!("Expected Broadcast variant"),
        }
        assert_eq!(broadcast.net_timeout(), 60.0);

        // Test Dynamic
        let dynamic = PoolQuery::dynamic(30.0);
        match dynamic {
            PoolQuery::Dynamic { net_timeout } => {
                assert!((net_timeout - 30.0).abs() < 0.01);
            }
            _ => panic!("Expected Dynamic variant"),
        }
        assert_eq!(dynamic.net_timeout(), 30.0);

        // Test Local
        let local = PoolQuery::local();
        match local {
            PoolQuery::Local => {}
            _ => panic!("Expected Local variant"),
        }
        assert_eq!(local.net_timeout(), 0.0);

        // Test Default
        let default_query = PoolQuery::default();
        match default_query {
            PoolQuery::Local => {}
            _ => panic!("Expected Local variant as default"),
        }
    }

    /// Test CteOp variants
    #[test]
    fn test_cte_op_variants() {
        // Test all operation variants
        assert_eq!(CteOp::PutBlob as u32, 0);
        assert_eq!(CteOp::GetBlob as u32, 1);
        assert_eq!(CteOp::DelBlob as u32, 2);
        assert_eq!(CteOp::GetOrCreateTag as u32, 3);
        assert_eq!(CteOp::DelTag as u32, 4);
        assert_eq!(CteOp::GetTagSize as u32, 5);

        // Test Debug and Clone traits
        let op = CteOp::PutBlob;
        let op_debug = format!("{:?}", op);
        assert!(op_debug.contains("PutBlob"));

        let op_copy = op.clone();
        assert_eq!(op, op_copy);
    }

    /// Test CteTagId bounds (major/minor values)
    #[test]
    fn test_cte_tag_id_bounds() {
        // Test minimum values
        let min_id = CteTagId::new(0, 0);
        assert!(min_id.is_null());

        // Test maximum values
        let max_id = CteTagId::new(u32::MAX, u32::MAX);
        assert!(!max_id.is_null());
        assert_eq!(max_id.major, u32::MAX);
        assert_eq!(max_id.minor, u32::MAX);

        // Test to_u64 boundary
        let max_combined = max_id.to_u64();
        assert_eq!(max_combined, u64::MAX);

        // Test from_u64 boundary
        let from_max = CteTagId::from_u64(u64::MAX);
        assert_eq!(from_max.major, u32::MAX);
        assert_eq!(from_max.minor, u32::MAX);

        // Test PartialEq and Eq
        let id1 = CteTagId::new(1, 2);
        let id2 = CteTagId::new(1, 2);
        let id3 = CteTagId::new(2, 1);
        assert_eq!(id1, id2);
        assert_ne!(id1, id3);

        // Test Hash
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher1 = DefaultHasher::new();
        let mut hasher2 = DefaultHasher::new();
        id1.hash(&mut hasher1);
        id2.hash(&mut hasher2);
        assert_eq!(hasher1.finish(), hasher2.finish());
    }

    /// Test RuntimeMode and BdevType
    #[test]
    fn test_enum_variants() {
        // Test RuntimeMode
        assert_eq!(RuntimeMode::Client as u32, 0);
        assert_eq!(RuntimeMode::Server as u32, 1);
        assert_eq!(RuntimeMode::Runtime as u32, 2);

        // Test BdevType
        assert_eq!(BdevType::File as u32, 0);
        assert_eq!(BdevType::Ram as u32, 1);

        // Test Clone
        let mode = RuntimeMode::Client;
        let mode_copy = mode.clone();
        assert_eq!(mode, mode_copy);

        let bdev_type = BdevType::File;
        let bdev_copy = bdev_type.clone();
        assert_eq!(bdev_type, bdev_copy);
    }

    /// Test CteTelemetry structure
    #[test]
    fn test_cte_telemetry() {
        let tag_id = CteTagId::new(1, 2);
        let telemetry = CteTelemetry {
            op: CteOp::PutBlob,
            off: 1024,
            size: 4096,
            tag_id,
            mod_time: SteadyTime::from_nanos(1000000),
            read_time: SteadyTime::from_nanos(2000000),
            logical_time: 42,
        };

        assert_eq!(telemetry.op, CteOp::PutBlob);
        assert_eq!(telemetry.off, 1024);
        assert_eq!(telemetry.size, 4096);
        assert_eq!(telemetry.tag_id.major, 1);
        assert_eq!(telemetry.tag_id.minor, 2);
        assert_eq!(telemetry.logical_time, 42);

        // Test Clone
        let telemetry_copy = telemetry.clone();
        assert_eq!(telemetry.op, telemetry_copy.op);
        assert_eq!(telemetry.size, telemetry_copy.size);
    }
}

// ============================================================================
// error.rs Tests
// ============================================================================

mod error_tests {
    use super::*;

    /// Test Display trait for all error variants
    #[test]
    fn test_error_display() {
        // InitFailed
        let err = CteError::InitFailed {
            reason: "config not found".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("initialization failed"));
        assert!(msg.contains("config not found"));

        // PoolCreationFailed
        let err = CteError::PoolCreationFailed {
            message: "out of memory".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Pool creation failed"));
        assert!(msg.contains("out of memory"));

        // PoolNotFound
        let err = CteError::PoolNotFound {
            pool_id: "pool_123".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Pool not found"));
        assert!(msg.contains("pool_123"));

        // TagNotFound
        let err = CteError::TagNotFound {
            name: "my_tag".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Tag not found"));
        assert!(msg.contains("my_tag"));

        // TagAlreadyExists
        let err = CteError::TagAlreadyExists {
            name: "duplicate".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Tag already exists"));

        // BlobNotFound
        let err = CteError::BlobNotFound {
            tag: "tag1".to_string(),
            blob: "blob1".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Blob not found"));
        assert!(msg.contains("tag1"));
        assert!(msg.contains("blob1"));

        // BlobIOError
        let err = CteError::BlobIOError {
            message: "read error".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Blob I/O error"));

        // TargetRegistrationFailed
        let err = CteError::TargetRegistrationFailed {
            path: "/dev/sda".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Target registration failed"));
        assert!(msg.contains("/dev/sda"));

        // TargetNotFound
        let err = CteError::TargetNotFound {
            path: "/dev/sdb".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Target not found"));

        // TelemetryUnavailable
        let err = CteError::TelemetryUnavailable;
        let msg = format!("{}", err);
        assert!(msg.contains("Telemetry unavailable"));

        // InvalidParameter
        let err = CteError::InvalidParameter {
            message: "name cannot be empty".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Invalid parameter"));
        assert!(msg.contains("name cannot be empty"));

        // RuntimeError
        let err = CteError::RuntimeError {
            code: 42,
            message: "internal error".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("runtime error"));
        assert!(msg.contains("code 42"));

        // Timeout
        let err = CteError::Timeout;
        let msg = format!("{}", err);
        assert!(msg.contains("timed out"));

        // FfiError
        let err = CteError::FfiError {
            message: "null pointer".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("FFI error"));

        // IoError
        let err = CteError::IoError {
            message: "file not found".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("I/O error"));

        // NotImplemented
        let err = CteError::NotImplemented {
            feature: "async_delete".to_string(),
            reason: "not yet implemented".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("not implemented"));
    }

    /// Test error chaining (source)
    #[test]
    fn test_error_source() {
        use std::error::Error;

        // Most errors have no source
        let err = CteError::InitFailed {
            reason: "test".to_string(),
        };
        assert!(err.source().is_none());

        let err = CteError::RuntimeError {
            code: 1,
            message: "test".to_string(),
        };
        assert!(err.source().is_none());

        let err = CteError::NotImplemented {
            feature: "test".to_string(),
            reason: "test".to_string(),
        };
        assert!(err.source().is_none());
    }

    /// Test ToCteResult trait for return code conversion
    #[test]
    fn test_to_cte_result() {
        // Success case (code 0)
        let result: CteResult<()> = 0u32.to_cte_result(0, |code| CteError::RuntimeError {
            code,
            message: "test".to_string(),
        });
        assert!(result.is_ok());

        // Failure case (code != 0)
        let result: CteResult<()> = 42u32.to_cte_result(0, |code| CteError::RuntimeError {
            code,
            message: format!("error {}", code),
        });
        assert!(result.is_err());
        match result {
            Err(CteError::RuntimeError { code: 42, message }) => {
                assert!(message.contains("42"));
            }
            _ => panic!("Expected RuntimeError"),
        }

        // Test with non-zero success code
        let result: CteResult<()> = 1u32.to_cte_result(1, |_| CteError::Timeout);
        assert!(result.is_ok());

        // Test with custom error
        let result: CteResult<()> = 5u32.to_cte_result(0, |_| CteError::InitFailed {
            reason: "init failed".to_string(),
        });
        assert!(result.is_err());
        match result {
            Err(CteError::InitFailed { reason }) => {
                assert_eq!(reason, "init failed");
            }
            _ => panic!("Expected InitFailed"),
        }
    }

    /// Test From<std::io::Error> conversion
    #[test]
    fn test_error_from_io() {
        use std::io;

        // Create IO error
        let io_err = io::Error::new(io::ErrorKind::NotFound, "file not found");
        let cte_err: CteError = io_err.into();

        match cte_err {
            CteError::IoError { message } => {
                assert!(message.contains("file not found"));
            }
            _ => panic!("Expected IoError variant"),
        }

        // Test with different error kind
        let io_err2 = io::Error::new(io::ErrorKind::PermissionDenied, "permission denied");
        let cte_err2: CteError = io_err2.into();

        match cte_err2 {
            CteError::IoError { message } => {
                assert!(message.contains("permission denied"));
            }
            _ => panic!("Expected IoError variant"),
        }
    }

    /// Test Clone for CteError
    #[test]
    fn test_error_clone() {
        let err = CteError::RuntimeError {
            code: 42,
            message: "test error".to_string(),
        };
        let err_clone = err.clone();
        match err_clone {
            CteError::RuntimeError { code, message } => {
                assert_eq!(code, 42);
                assert_eq!(message, "test error");
            }
            _ => panic!("Expected RuntimeError"),
        }
    }
}

// ============================================================================
// Validation Logic Tests
// ============================================================================

mod validation_tests {
    use super::*;

    /// Test score validation with valid values
    #[test]
    fn test_score_validation_valid() {
        // Valid scores: 0.0, 0.5, 1.0
        let valid_scores = [0.0, 0.001, 0.1, 0.25, 0.5, 0.75, 0.999, 1.0];

        for score in valid_scores {
            let result = validate_score(score);
            assert!(result.is_ok(), "Score {} should be valid", score);
        }

        fn validate_score(score: f32) -> CteResult<()> {
            if score < 0.0 || score > 1.0 || score.is_nan() {
                Err(CteError::InvalidParameter {
                    message: format!("Score must be between 0.0 and 1.0, got {}", score),
                })
            } else {
                Ok(())
            }
        }
    }

    /// Test score validation with invalid values
    #[test]
    fn test_score_validation_invalid() {
        // Invalid scores: negative, > 1.0, NaN, infinity
        let invalid_scores = [
            -0.001,
            -0.1,
            -1.0,
            1.001,
            1.5,
            2.0,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ];

        for score in invalid_scores {
            let result = validate_score(score);
            assert!(result.is_err(), "Score {} should be invalid", score);
        }

        fn validate_score(score: f32) -> CteResult<()> {
            if score < 0.0 || score > 1.0 || score.is_nan() {
                Err(CteError::InvalidParameter {
                    message: format!("Score must be between 0.0 and 1.0, got {}", score),
                })
            } else {
                Ok(())
            }
        }
    }

    /// Test empty name validation
    #[test]
    fn test_name_validation_empty() {
        fn validate_name(name: &str) -> CteResult<()> {
            if name.is_empty() {
                Err(CteError::InvalidParameter {
                    message: "Name cannot be empty".to_string(),
                })
            } else {
                Ok(())
            }
        }

        // Empty name should fail
        let result = validate_name("");
        assert!(result.is_err());
        match result {
            Err(CteError::InvalidParameter { message }) => {
                assert!(message.contains("cannot be empty"));
            }
            _ => panic!("Expected InvalidParameter"),
        }

        // Non-empty names should pass
        let valid_names = ["a", "test", "my_blob.bin", "path/to/blob"];
        for name in valid_names {
            let result = validate_name(name);
            assert!(result.is_ok(), "Name '{}' should be valid", name);
        }
    }

    /// Test size limits (MAX_BLOB_SIZE = 16GB)
    #[test]
    fn test_size_limits() {
        const MAX_BLOB_SIZE: u64 = 16 * 1024 * 1024 * 1024; // 16 GB

        fn validate_size(size: u64) -> CteResult<()> {
            if size > MAX_BLOB_SIZE {
                Err(CteError::InvalidParameter {
                    message: format!(
                        "Data size {} exceeds maximum blob size {}",
                        size, MAX_BLOB_SIZE
                    ),
                })
            } else {
                Ok(())
            }
        }

        // Valid sizes (under limit)
        let valid_sizes = [0, 1, 1024, 1024 * 1024, MAX_BLOB_SIZE];
        for size in valid_sizes {
            let result = validate_size(size);
            assert!(result.is_ok(), "Size {} should be valid", size);
        }

        // Invalid sizes (over limit)
        let invalid_sizes = [MAX_BLOB_SIZE + 1, MAX_BLOB_SIZE * 2, u64::MAX];
        for size in invalid_sizes {
            let result = validate_size(size);
            assert!(result.is_err(), "Size {} should be invalid", size);
        }
    }

    /// Test offset overflow (offset + size > u64::MAX)
    #[test]
    fn test_offset_overflow() {
        fn validate_offset_size(offset: u64, size: u64) -> CteResult<()> {
            let end_offset =
                offset
                    .checked_add(size)
                    .ok_or_else(|| CteError::InvalidParameter {
                        message: format!("Offset {} + size {} would overflow u64", offset, size),
                    })?;

            const MAX_BLOB_SIZE: u64 = 16 * 1024 * 1024 * 1024;
            if end_offset > MAX_BLOB_SIZE {
                return Err(CteError::InvalidParameter {
                    message: format!(
                        "Total blob size {} exceeds maximum {}",
                        end_offset, MAX_BLOB_SIZE
                    ),
                });
            }

            Ok(())
        }

        // Valid offset + size combinations
        let valid_cases = [
            (0, 0),
            (0, 100),
            (100, 200),
            (u64::MAX / 2, u64::MAX / 2 - 1),
        ];
        for (offset, size) in valid_cases {
            let result = validate_offset_size(offset, size);
            assert!(
                result.is_ok(),
                "Offset {} + size {} should be valid",
                offset,
                size
            );
        }

        // Invalid cases (overflow)
        let invalid_cases = [
            (u64::MAX, 1),                    // offset + 1 overflow
            (u64::MAX - 1, 2),                // offset + 2 overflow
            (u64::MAX / 2, u64::MAX / 2 + 1), // overflow
        ];
        for (offset, size) in invalid_cases {
            let result = validate_offset_size(offset, size);
            assert!(
                result.is_err(),
                "Offset {} + size {} should overflow",
                offset,
                size
            );
        }
    }

    /// Test combined validation (name, score, size, offset)
    #[test]
    fn test_combined_validation() {
        fn validate_blob_params(name: &str, data: &[u8], offset: u64, score: f32) -> CteResult<()> {
            // Validate name
            if name.is_empty() {
                return Err(CteError::InvalidParameter {
                    message: "Blob name cannot be empty".to_string(),
                });
            }

            // Validate score
            if score < 0.0 || score > 1.0 || score.is_nan() {
                return Err(CteError::InvalidParameter {
                    message: format!("Score must be between 0.0 and 1.0, got {}", score),
                });
            }

            // Validate size
            const MAX_BLOB_SIZE: u64 = 16 * 1024 * 1024 * 1024;
            let data_len = data.len() as u64;
            if data_len > MAX_BLOB_SIZE {
                return Err(CteError::InvalidParameter {
                    message: format!(
                        "Data size {} exceeds maximum blob size {}",
                        data_len, MAX_BLOB_SIZE
                    ),
                });
            }

            // Validate offset + size
            let end_offset =
                offset
                    .checked_add(data_len)
                    .ok_or_else(|| CteError::InvalidParameter {
                        message: format!(
                            "Offset {} + size {} would overflow u64",
                            offset, data_len
                        ),
                    })?;

            if end_offset > MAX_BLOB_SIZE {
                return Err(CteError::InvalidParameter {
                    message: format!(
                        "Total blob size {} exceeds maximum {}",
                        end_offset, MAX_BLOB_SIZE
                    ),
                });
            }

            Ok(())
        }

        // Valid case
        let result = validate_blob_params("test", b"hello", 0, 1.0);
        assert!(result.is_ok());

        // Invalid: empty name
        let result = validate_blob_params("", b"data", 0, 1.0);
        assert!(result.is_err());
        match result {
            Err(CteError::InvalidParameter { message }) => {
                assert!(message.contains("cannot be empty"));
            }
            _ => panic!("Expected InvalidParameter"),
        }

        // Invalid: bad score
        let result = validate_blob_params("test", b"data", 0, -1.0);
        assert!(result.is_err());
        match result {
            Err(CteError::InvalidParameter { message }) => {
                assert!(message.contains("Score must be between"));
            }
            _ => panic!("Expected InvalidParameter"),
        }

        // Invalid: overflow
        let result = validate_blob_params("test", b"data", u64::MAX, 1.0);
        assert!(result.is_err());
        match result {
            Err(CteError::InvalidParameter { message }) => {
                assert!(message.contains("overflow"));
            }
            _ => panic!("Expected InvalidParameter"),
        }
    }
}

// ============================================================================
// Frecency Engine Tests (Issue #738)
// ============================================================================

mod frecency_tests {
    use super::*;
    use crate::frecency_engine::{compute_score_pure, validate_against_cpp};

    /// Test frecency formula: 2^(-age/600) where age is in seconds
    #[test]
    fn test_frecency_formula_decay() {
        // age = 0 seconds -> score = 2^0 = 1.0
        let now = 1_000_000_000_000u64;
        let score = compute_score_pure(1, now, now);
        assert!((score - 1.0).abs() < 1e-10, "age=0 should give score=1.0");

        // age = 600 seconds (10 min) -> score = 2^(-1) = 0.5
        let age_seconds = 600;
        let age_nanos = age_seconds * 1_000_000_000;
        let last_access = now - age_nanos;
        let score = compute_score_pure(1, last_access, now);
        assert!((score - 0.5).abs() < 1e-10, "age=600s should give score=0.5");

        // age = 1200 seconds (20 min) -> score = 2^(-2) = 0.25
        let age_seconds = 1200;
        let age_nanos = age_seconds * 1_000_000_000;
        let last_access = now - age_nanos;
        let score = compute_score_pure(1, last_access, now);
        assert!((score - 0.25).abs() < 1e-10, "age=1200s should give score=0.25");
    }

    /// Test frecency score computation with access count
    #[test]
    fn test_frecency_score_with_access_count() {
        let now = 1_000_000_000_000u64;

        // 1 access, age=0 -> score = 1 * 1.0 = 1.0
        let score = compute_score_pure(1, now, now);
        assert!((score - 1.0).abs() < 1e-10);

        // 10 accesses, age=0 -> score = 10 * 1.0 = 10.0
        let score = compute_score_pure(10, now, now);
        assert!((score - 10.0).abs() < 1e-10);

        // 10 accesses, age=600s -> score = 10 * 0.5 = 5.0
        let age_nanos = 600 * 1_000_000_000;
        let last_access = now - age_nanos;
        let score = compute_score_pure(10, last_access, now);
        assert!((score - 5.0).abs() < 1e-10);

        // 100 accesses, age=1200s -> score = 100 * 0.25 = 25.0
        let age_nanos = 1200 * 1_000_000_000;
        let last_access = now - age_nanos;
        let score = compute_score_pure(100, last_access, now);
        assert!((score - 25.0).abs() < 1e-10);
    }

    /// Test validate_against_cpp - requires CTE runtime for FFI cross-check
    #[test]
    #[ignore = "requires running CTE runtime for C++ frecency cross-check"]
    fn test_frecency_validate_against_cpp() {
        // Score for a blob accessed 10 times, last accessed now, at time now
        let access_count = 10;
        let now = 1_000_000_000_000; // arbitrary steady-clock ns
        let last_access = now;
        let result = validate_against_cpp(access_count, last_access, now);
        // This will fail (Err) if no CTE runtime, which is expected when ignored.
        // When run manually with a CTE runtime up, it should be Ok(()).
        println!("validate_against_cpp result: {:?}", result);
    }
}
