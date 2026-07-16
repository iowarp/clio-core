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

//! Error types for CTE Rust bindings
//!
//! Provides idiomatic Rust error handling with detailed error variants
//! for all CTE operations.

use std::fmt;

/// Errors that can occur in CTE operations
///
/// This enum provides detailed error information for:
/// - Initialization failures
/// - Pool, tag, blob, and target operations
/// - FFI bridge errors
#[derive(Debug)]
pub enum CteError {
    /// Initialization failed
    InitFailed {
        reason: String,
    },

    /// Pool operations failed
    PoolCreationFailed {
        message: String,
    },
    PoolNotFound {
        pool_id: String,
    },

    /// Tag operations failed
    TagNotFound {
        name: String,
    },
    TagAlreadyExists {
        name: String,
    },

    /// Blob operations failed
    BlobNotFound {
        tag: String,
        blob: String,
    },
    BlobIOError {
        message: String,
    },

    /// Storage target operations failed
    TargetRegistrationFailed {
        path: String,
    },
    TargetNotFound {
        path: String,
    },

    /// Telemetry unavailable
    TelemetryUnavailable,

    /// Invalid parameter provided
    InvalidParameter {
        message: String,
    },

    /// C++ runtime returned error code
    RuntimeError {
        code: u32,
        message: String,
    },

    /// Operation timed out
    Timeout,

    /// FFI bridge error
    FfiError {
        message: String,
    },

    /// I/O error wrapper (stores error message since std::io::Error is not Clone)
    IoError {
        message: String,
    },

    /// Feature not yet implemented
    NotImplemented {
        feature: String,
        reason: String,
    },
}

impl Clone for CteError {
    fn clone(&self) -> Self {
        match self {
            CteError::InitFailed { reason } => CteError::InitFailed {
                reason: reason.clone(),
            },
            CteError::PoolCreationFailed { message } => CteError::PoolCreationFailed {
                message: message.clone(),
            },
            CteError::PoolNotFound { pool_id } => CteError::PoolNotFound {
                pool_id: pool_id.clone(),
            },
            CteError::TagNotFound { name } => CteError::TagNotFound { name: name.clone() },
            CteError::TagAlreadyExists { name } => {
                CteError::TagAlreadyExists { name: name.clone() }
            }
            CteError::BlobNotFound { tag, blob } => CteError::BlobNotFound {
                tag: tag.clone(),
                blob: blob.clone(),
            },
            CteError::BlobIOError { message } => CteError::BlobIOError {
                message: message.clone(),
            },
            CteError::TargetRegistrationFailed { path } => {
                CteError::TargetRegistrationFailed { path: path.clone() }
            }
            CteError::TargetNotFound { path } => CteError::TargetNotFound { path: path.clone() },
            CteError::TelemetryUnavailable => CteError::TelemetryUnavailable,
            CteError::InvalidParameter { message } => CteError::InvalidParameter {
                message: message.clone(),
            },
            CteError::RuntimeError { code, message } => CteError::RuntimeError {
                code: *code,
                message: message.clone(),
            },
            CteError::Timeout => CteError::Timeout,
            CteError::FfiError { message } => CteError::FfiError {
                message: message.clone(),
            },
            CteError::IoError { message } => CteError::IoError {
                message: message.clone(),
            },
            CteError::NotImplemented { feature, reason } => CteError::NotImplemented {
                feature: feature.clone(),
                reason: reason.clone(),
            },
        }
    }
}

impl fmt::Display for CteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CteError::InitFailed { reason } => {
                write!(f, "CTE initialization failed: {}", reason)
            }
            CteError::PoolCreationFailed { message } => {
                write!(f, "Pool creation failed: {}", message)
            }
            CteError::PoolNotFound { pool_id } => {
                write!(f, "Pool not found: {}", pool_id)
            }
            CteError::TagNotFound { name } => {
                write!(f, "Tag not found: {}", name)
            }
            CteError::TagAlreadyExists { name } => {
                write!(f, "Tag already exists: {}", name)
            }
            CteError::BlobNotFound { tag, blob } => {
                write!(f, "Blob not found: tag={}, blob={}", tag, blob)
            }
            CteError::BlobIOError { message } => {
                write!(f, "Blob I/O error: {}", message)
            }
            CteError::TargetRegistrationFailed { path } => {
                write!(f, "Target registration failed: {}", path)
            }
            CteError::TargetNotFound { path } => {
                write!(f, "Target not found: {}", path)
            }
            CteError::TelemetryUnavailable => {
                write!(f, "Telemetry unavailable")
            }
            CteError::InvalidParameter { message } => {
                write!(f, "Invalid parameter: {}", message)
            }
            CteError::RuntimeError { code, message } => {
                write!(f, "CTE runtime error (code {}): {}", code, message)
            }
            CteError::Timeout => {
                write!(f, "Operation timed out")
            }
            CteError::FfiError { message } => {
                write!(f, "FFI error: {}", message)
            }
            CteError::IoError { message } => {
                write!(f, "I/O error: {}", message)
            }
            CteError::NotImplemented { feature, reason } => {
                write!(f, "Feature not implemented: {} - {}", feature, reason)
            }
        }
    }
}

impl std::error::Error for CteError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            CteError::NotImplemented { .. } => None,
            _ => None,
        }
    }
}

impl From<std::io::Error> for CteError {
    fn from(err: std::io::Error) -> Self {
        CteError::IoError {
            message: err.to_string(),
        }
    }
}

/// Convenience type alias for CTE results
///
/// Use this for all CTE operations that can fail:
/// ```
/// fn do_something() -> CteResult<Output> {
///     Ok(output)
/// }
/// ```
pub type CteResult<T> = Result<T, CteError>;

/// Helper trait for converting C++ error codes
///
/// Provides ergonomic conversion from raw C++ return codes
/// to CteError variants.
pub(crate) trait ToCteResult {
    /// Convert to CteResult based on error code
    ///
    /// # Arguments
    /// * `success_code` - The value indicating success (typically 0)
    /// * `on_error` - Closure to generate error on failure
    fn to_cte_result<F>(self, success_code: u32, on_error: F) -> CteResult<()>
    where
        F: FnOnce(u32) -> CteError;
}

impl ToCteResult for u32 {
    fn to_cte_result<F>(self, success_code: u32, on_error: F) -> CteResult<()>
    where
        F: FnOnce(u32) -> CteError,
    {
        if self == success_code {
            Ok(())
        } else {
            Err(on_error(self))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = CteError::InitFailed {
            reason: "test".to_string(),
        };
        assert!(err.to_string().contains("initialization failed"));

        let err = CteError::TagNotFound {
            name: "mytag".to_string(),
        };
        assert!(err.to_string().contains("Tag not found"));
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let cte_err: CteError = io_err.into();

        match cte_err {
            CteError::IoError { message } => {
                assert!(message.contains("file not found"));
            }
            _ => panic!("Expected IoError variant"),
        }
    }

    #[test]
    fn test_to_cte_result() {
        // Success case
        let result: CteResult<()> = 0u32.to_cte_result(0, |_| CteError::RuntimeError {
            code: 1,
            message: "fail".to_string(),
        });
        assert!(result.is_ok());

        // Error case
        let result: CteResult<()> = 1u32.to_cte_result(0, |code| CteError::RuntimeError {
            code,
            message: format!("error {}", code),
        });
        assert!(result.is_err());
        match result {
            Err(CteError::RuntimeError { code: 1, .. }) => {}
            _ => panic!("Expected RuntimeError with code 1"),
        }
    }
}
