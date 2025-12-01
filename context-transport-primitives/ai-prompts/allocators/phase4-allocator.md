@CLAUDE.md 

# Reduce variables in Allocator and simplify Backend

Remove buffer_ and buffer_size_ from Allocator. We will use
accel_data_ and accel_data_size_. We should rename accel_data_ to
just data_ and accel_data_size_ to data_size_. Note that accel_id_
only applies to the data_ pointer, not the md_ pointer.

# MemoryBackend 

Augment the MemoryBackend class to include a variable called ``u64 root_offset_``. This is 0 by default.
This is used to represent the case where the backend is actually apart of a larger existing backend.
This is the case, for example, with sub allocators. Really the only time this should be non-zero.

Make it so MemoryBackendId has two variables:
```
MemoryBackendId {
    u32 major_;
    u32 minor_;
}
```

Major for example could represent pid, minor would be relative to a pid. This is for future use.
For now, assume user hardcodes the backend ids as constants.

# ArrayBackend

Make it so array backend uses malloc for md and sets md_size_ to the ArrayBackendHeader.

The region should be only for the data segment.

Augment ArrayBackend to take as input the offset in the case it is a sub allocator's backend.
It should be an optional parameter by default 0.

# Sub Allocators

I want to introduce the concept of SubAllocators. These are allocators that work in conjunction with the main allocator
for the backend. The OffsetPointer returned by a SubAllocator is always relative to the main backend.

AllocatorId should have the following fields:
```
struct AllocatorId {
    MemoryBackendId backend_id_;  // The backend this is attached to
    u64 sub_id_(0);  // The unique id of allocator on this backend. Main allocator always 0.
};
```

Expose the following method in the BaseAllocator class. Assume the AllocT has things like backend. 
CoreAllocT will inherit from Allocator always:
```
template<typename AllocT, typename ...Args>
AllocT *CreateSubAllocator(u64 sub_id, size_t size, Args&& ...args) {
    ArrayBackend backend;
    FullPtr<char> region = Allocate(size);
    backend.shm_init(region.ptr_, size, region.shm_.GetOffset());
    AllocatorId sub_alloc_id(backend_.id_, sub_id);
    AllocT sub_alloc;
    sub_alloc.shm_init(sub_alloc_id, backend, std::forward<Args>(args)...); 
}

template<typename AllocT>
void FreeSubAllocator(AllocT *alloc) {
    FreeOffset(alloc->backend.md_);
}
```

# Heap

Create a class called heap under context-transport-primitives/include/hermes_shm/memory/allocator. 

This is not an allocator in and of itself, but is a useful helper. 

```
template<bool ATOMIC>
class Heap {
    hipc::opt_atomic<ATOMIC> heap_(0);
    size_t max_size_;

    size_t Allocate(size_t size, size_t align = 8) {
        size = ...; // Align size to align bytes.
        size_t off = heap_.fetch_add(size);
        if (off + size > max_size_) {
            HSHM_THROW_ERROR(...);
        }
        return off;
    }
}
```

# ArenaAllocator

Add to context-transport-primitives/include/hermes_shm/memory/allocator/arena_allocator.h

Just grows upwards. FreeOffset, CreateTls, FreeTls, AlignedAllocate is unimplemented (but not erronous if it gets called).

Templated, takes as input ATOMIC. The arena may or may not be atomic.
* Allocate calls Allocate on the heap.
* The heap is stored in the shared memory header.

```
template<bool ATOMIC>
class ArenaAllocator {}
```

# Make Pointer better
@CLAUDE.md

Remove data_ and data_size_ from allocator. Use only backend.data_ and backend.size_
Backend should also have a function called Shift. 
Shift takes as input:
1. OffsetPointer shift (the offset from the beginning of data)
This will change both the size and offset.

FullPtr(CtxAllocator ctx_alloc, OffsetPointer ptr) and similar constructors should be updated to do:
ptr_ = ctx_alloc.backend.data_ + ctx_alloc.backend.offset_ + ptr;

Verify that unit tests still pass after this change.

Let's also make the following changes:
1. OffsetPointer -> OffsetPtr. Make OffsetPtr templated, with default void.
2. Pointer -> ShmPtr. Make ShmPtr templated, with default void.
3. Remove TypedPointer and replace all occurences with ShmPtr
