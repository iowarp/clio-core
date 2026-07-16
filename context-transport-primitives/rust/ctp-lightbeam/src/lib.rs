// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).
// STUB: implemented by the porting workflow.

pub mod transport;
pub mod shm_transport;
pub mod socket_transport;

/// Register every transport backend compiled into this crate with
/// [`transport::TransportFactory`].
///
/// The C++ has no equivalent because its backends are compiled into the
/// factory's `switch` behind `#if CTP_ENABLE_*`. This port resolves backends
/// through a runtime registry, so registration has to happen somewhere —
/// call this once at startup before asking the factory for anything.
///
/// Idempotent: re-registering replaces the entry with the same function.
pub fn register_builtin_transports() {
    socket_transport::register();
    shm_transport::register();
}
