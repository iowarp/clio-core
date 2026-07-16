// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).
//
//! Process-local allocator registry: `AllocatorId → mapped heap`.
//! Resolution turns the cross-process `(alloc_id, off)` currency into a raw
//! pointer valid in THIS process. `OnceLock<RwLock<..>>` (read-mostly); no
//! thread_local, per project rule. Unregistered ids resolve to `None`,
//! never UB.

use crate::ptr::{AllocatorId, ShmPtr, NULL_OFFSET};
use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

#[derive(Clone, Copy)]
struct Mapping {
    heap_base: usize,
    heap_size: u64,
}

fn table() -> &'static RwLock<HashMap<AllocatorId, Mapping>> {
    static TABLE: OnceLock<RwLock<HashMap<AllocatorId, Mapping>>> = OnceLock::new();
    TABLE.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Register an allocator's mapped heap. Called by allocator create/open;
/// re-registering an id replaces the mapping (restart semantics).
pub fn register_allocator(id: AllocatorId, heap_base: *mut u8, heap_size: u64) {
    table().write().expect("registry poisoned").insert(
        id,
        Mapping {
            heap_base: heap_base as usize,
            heap_size,
        },
    );
}

/// Remove an allocator (its heap is about to unmap).
pub fn unregister_allocator(id: AllocatorId) {
    table().write().expect("registry poisoned").remove(&id);
}

/// Resolve a `ShmPtr` to a raw pointer in this process, or `None` when the
/// allocator is unknown, the pointer is null, or the offset is out of range.
///
/// The returned pointer is valid while the allocator remains registered and
/// the pointee allocation live — the same contract C++ `ShmPtr` resolution
/// carries. Dereferencing is the caller's `unsafe`.
pub fn resolve<T>(ptr: ShmPtr<T>) -> Option<*mut T> {
    if ptr.is_null() || ptr.off == NULL_OFFSET {
        return None;
    }
    let map = table().read().expect("registry poisoned");
    let m = map.get(&ptr.alloc_id)?;
    let need = std::mem::size_of::<T>() as u64;
    if ptr.off.checked_add(need.max(1))? > m.heap_size {
        return None;
    }
    Some((m.heap_base + ptr.off as usize) as *mut T)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_unknown_and_out_of_range() {
        let id = AllocatorId::new(9999, 1);
        assert!(resolve(ShmPtr::<u64>::new(id, 0)).is_none()); // unregistered
        let mut buf = [0u8; 64];
        register_allocator(id, buf.as_mut_ptr(), 64);
        assert!(resolve(ShmPtr::<u64>::new(id, 0)).is_some());
        assert!(resolve(ShmPtr::<u64>::new(id, 60)).is_none()); // 8B past end
        assert!(resolve(ShmPtr::<u64>::null()).is_none());
        unregister_allocator(id);
        assert!(resolve(ShmPtr::<u64>::new(id, 0)).is_none());
    }
}
