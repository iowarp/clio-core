@CLAUDE.md

Create this allocator and implement unit tests for it. the unit tests should include
multi-threaded cases. It should be comparable to malloc in terms of functionality and
generality.

This allocator is intended to be invoked by CPU only.
It will make use of HSHM_THREAD_MODEL->SetTls and GetTls a lot.
We will make a GPU-specific allocator later.

# Class / struct Overview for MultiProcessAllocator

```
class ThreadBlock : slist_node {
    int tid_;
    BuddyAllocator thread_;  // Can be private memory

    ThreadBlock(MemoryBackend backend) {
        // Shift memory backend by (char*)this + sizeof(ThreadBlock) - backend.data_.
        // Call shm_init for thread_ with this backend.
    }
}

class ProcessBlock : slist_node {
    int pid_;
    hshm::Mutex lock_;
    BuddyAllocatorHeader root_;  // Shared-memory
    pre::slist<ThreadBlock> thread_;

    ProcessBlock(const MemoryBackend &backend, void *region) {
        // 
        // Call root_.shm_init with region
    }

    ThreadBlock* AllocateThreadBlock(const MemoryBackend &backend, size_t region_size) {
        // Acquire lock_
        // Allocate region_size + sizeof(ThreadBlock) from root_
        // If that fails, return nullptr
        // Cast the region to ThreadBlock* and emplace into slist
        // 
    }
}


class MultiProcessAllocatorHeader {
    pre::slist<ProcessBlock> blocks_;
    BuddyAllocator root_;
    hshm::Mutex lock_;
}
```

# MultiProcessAllocator

## shm_init

### Implementation
1. Create the MultiProcessAllocatorHeader.
2. Initialize root_ with the remainder of the MemoryBackend.
3. Allocate and construct the first ProcesBlocks from the root_ allocator
4. Emplace into the blocks_ slist.
5. Allocate 

### Return Value
MemContext containing tid and pid of this process.

## shm_attach

### Parameters
1. process_unit_: Unit of process memory allocation. 1GB by default. If we run out of memory for the process,
it will allocate one large chunk of this unit size.
2. thread_unit_: Unit of thread allocation. 16MB by default. If we run out of space for the thread, it will allocate
one large chunk from the process allocator.

### 
Acquire the MultiProcessAllocatorHeader root_ allocator lock.
Allocate process_unit_ bytes from the root_ allocator.
Allocate a process ID from the ring_queue
Free that block into the sub-buddy allocator.


### Implementation
Allocate 





## shm_detach

Delete all memory associated with this process and free back 

## AllocateOffset

Check if the MemContext is valid. If not there are two cases:
1. If CPU: 
2. If GPU: 



## AlignedAllocateOffset

## FreeOffsetNoNullCheck

## Coalescing
