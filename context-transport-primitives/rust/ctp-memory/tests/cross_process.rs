// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).
//
//! Cross-process shared-memory proof: a REAL child process maps the same
//! segment, resolves parent-written ShmPtrs, allocates from the shared
//! free list, and hands a pointer back through an in-segment mailbox.
//!
//! Mechanism: the test re-spawns its own test binary filtered to this one
//! test with CTP_MEM_XPROC_CHILD set; the child branch runs the other side.
//!
//! Mailbox protocol (all offsets heap-relative):
//!   The FIRST allocation in a fresh segment lands at offset 16
//!   (BLOCK_HEADER) — both sides rely on that determinism.
//!   mailbox[0]: parent → child   offset of parent's data block
//!   mailbox[1]: child  → parent  offset of child's response block (0 until set)

use ctp_memory::{AllocatorId, FreeListAllocator, OffsetPtr, SharedMemBackend};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

const MAILBOX_OFF: u64 = 16; // first allocation in a fresh segment
const SEG_SIZE: usize = 1 << 20;
const PARENT_PATTERN: u64 = 0x5045_4E54_5F4F_4B21; // "PENT_OK!"
const CHILD_PATTERN: u64 = 0xC0FF_EE00_C0FF_EE00;
const N: usize = 64;

fn seg_name() -> Option<String> {
    std::env::var("CTP_MEM_XPROC_CHILD").ok()
}

fn mailbox(alloc: &FreeListAllocator, slot: usize) -> &'static AtomicU64 {
    let p = ctp_memory::resolve(ctp_memory::ShmPtr::<u64>::new(
        alloc.id(),
        MAILBOX_OFF + (slot as u64) * 8,
    ))
    .expect("mailbox resolves");
    // SAFETY: live, 8-aligned u64 inside the registered segment.
    unsafe { ctp_memory::allocator::atomic_u64_at(p) }
}

fn child_main(name: &str) {
    let backend = SharedMemBackend::open(name, SEG_SIZE).expect("child opens segment");
    let alloc =
        FreeListAllocator::open(backend, AllocatorId::new(777, 1)).expect("child opens allocator");

    // Read the parent's data offset from mailbox[0] and verify contents.
    let data_off = mailbox(&alloc, 0).load(Ordering::Acquire);
    assert_ne!(data_off, 0, "parent must have published its block");
    let data =
        ctp_memory::resolve(ctp_memory::ShmPtr::<[u64; N]>::new(alloc.id(), data_off)).unwrap();
    // SAFETY: parent-owned live [u64; N] block, published via mailbox.
    let data = unsafe { &*data };
    for (i, v) in data.iter().enumerate() {
        assert_eq!(
            *v,
            PARENT_PATTERN.wrapping_add(i as u64),
            "parent data mismatch"
        );
    }

    // Allocate our own block FROM THE SHARED FREE LIST, fill, publish.
    let mine = alloc.alloc::<[u64; N]>().expect("child alloc");
    let raw = ctp_memory::resolve(mine).unwrap();
    // SAFETY: freshly allocated block owned by this child.
    unsafe {
        for i in 0..N {
            (*raw)[i] = CHILD_PATTERN.wrapping_add(i as u64);
        }
    }
    mailbox(&alloc, 1).store(mine.off, Ordering::Release);
    // Parent verifies + frees; exit without freeing (ownership handed off).
    std::mem::forget(alloc); // keep registration semantics trivial at exit
    std::process::exit(0);
}

#[test]
fn cross_process_shared_heap() {
    if let Some(name) = seg_name() {
        child_main(&name);
        return; // unreachable
    }

    // ---- Parent role ----
    let name = format!("ctp_rs_xproc_{}", std::process::id());
    let backend = SharedMemBackend::create(&name, SEG_SIZE).unwrap();
    let alloc = FreeListAllocator::create(backend, AllocatorId::new(777, 1));

    // First allocation = the two-slot mailbox at the deterministic offset.
    let mb = alloc.alloc_bytes(16).unwrap();
    assert_eq!(
        mb.off, MAILBOX_OFF,
        "mailbox determinism is part of the protocol"
    );
    mailbox(&alloc, 0).store(0, Ordering::Release);
    mailbox(&alloc, 1).store(0, Ordering::Release);

    // Parent data block, published through mailbox[0].
    let data = alloc.alloc::<[u64; N]>().unwrap();
    let raw = ctp_memory::resolve(data).unwrap();
    // SAFETY: freshly allocated block owned by the parent.
    unsafe {
        for i in 0..N {
            (*raw)[i] = PARENT_PATTERN.wrapping_add(i as u64);
        }
    }
    mailbox(&alloc, 0).store(data.off, Ordering::Release);

    // Spawn ourselves as the child role.
    let exe = std::env::current_exe().unwrap();
    let status = std::process::Command::new(exe)
        .args(["cross_process_shared_heap", "--exact", "--nocapture"])
        .env("CTP_MEM_XPROC_CHILD", &name)
        .status()
        .expect("spawn child");
    assert!(status.success(), "child process failed: {status:?}");

    // Child published its block before exiting; verify it.
    let deadline = Instant::now() + Duration::from_secs(10);
    let child_off = loop {
        let v = mailbox(&alloc, 1).load(Ordering::Acquire);
        if v != 0 {
            break v;
        }
        assert!(Instant::now() < deadline, "child never published");
        std::thread::sleep(Duration::from_millis(10));
    };
    assert_ne!(child_off, data.off, "child must have its own block");
    let child_ptr = ctp_memory::ShmPtr::<[u64; N]>::new(alloc.id(), child_off);
    let raw = ctp_memory::resolve(child_ptr).unwrap();
    // SAFETY: child-allocated live block, published via mailbox, child exited.
    let vals = unsafe { &*raw };
    for (i, v) in vals.iter().enumerate() {
        assert_eq!(
            *v,
            CHILD_PATTERN.wrapping_add(i as u64),
            "child data mismatch"
        );
    }

    // The parent (segment owner) reclaims the child's allocation — the
    // free-list metadata written by the child process must be coherent here.
    unsafe {
        alloc.free(child_ptr);
        alloc.free(data);
        alloc.free_bytes(OffsetPtr::new(MAILBOX_OFF));
    }
    // And the reclaimed space is reusable.
    assert!(alloc.alloc_bytes(512).is_some());

    alloc.backend().destroy();
}
