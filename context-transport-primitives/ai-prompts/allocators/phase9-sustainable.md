@CLAUDE.md

# Shm Backend update
I want context-transport-primitives/include/hermes_shm/memory/backend/posix_shm_mmap.h to support a mix of private and shared mapping.

I need a contiguous region where the first say 16KB of the region is private memory and the following size bytes are shared memory.
I don't mind if this requires multiple mmap calls, but it needs to be guaranteed correct.
Is this possible?

@CLAUDE.md

# General Backend Update

Each backend should have the first 16KB dedicated to some private memory for allocators
to leverage thread-local storage semantics better.

MemoryBackend should look like this:
data_: the shared part of the region (for posix shm mmap)

Every backend should support:
(data_ - kBachendPrivate) to get a region of valid private memory.

The kBackendPrivate should be in addition to any size parameter given for the data segment.

Create a global constant called kBackendPrivate = 4KB. Update the PosixShmMmap allocator to use this constant for the Mixed allocation.

@CLAUDE.md

# Improving allocator ease-of-use

We need to avoid passing the allocator so much. 

Let's make the Allocator classes themselves shared-memory compatible. 

## General Observation

Containers should be able to get the pointer to the allocator class as follows:
1. Upon construction, the container is initially passed the Allocator pointer
2. The container should store OffsetPtr<> this_ = (this - alloc)
3. Allocator *alloc = (this - this_);

This assumes that the Allocator is allocated on the Memory backend.
Instead of passing the MemoryBackend to the Allocator, 
we should be casting the MemoryBackend data_ pointer to an Allocator*.

## MemoryBackend
We should add the following new apis to the MemoryBackend:
1. AllocT* cast<AllocT>: this will simply return reinterpret_cast<AllocT>(data_);


## Allocator
Remove the following from the Allocator:
```
MemoryBackend backend_;
int accel_id_;
char *custom_header_;
```

Add the following:
```
size_t size_;  // The size of the memory backend.
```

Update ContainsPtr to use the size_ variable only.
```
ContainsPtr(OffsetPtr &off) {  return off < size_;  }
ContainsPtr(char *ptr) { (ptr - this) < size_; }
```


## BuddyAllocator

Remove the fields:
```
  size_t heap_begin_;           /**< Offset to heap beginning */
  size_t heap_current_;         /**< Current heap offset */
  size_t heap_end_;             /**< End of heap */
```

Do not let the following be pointers:
```
  pre::slist<false> *round_up_lists_;    /**< Free lists for sizes 32B - 16KB (round up) */
  pre::slist<false> *round_down_lists_;  /**< Free lists for sizes 16KB - 1MB (round down) */
```

Change them to this:
```
  pre::slist<false> round_up_lists_;    /**< Free lists for sizes 32B - 16KB (round up) */
  pre::slist<false> round_down_lists_;  /**< Free lists for sizes 16KB - 1MB (round down) */
```

## MultiProcessAllocator

Add a method called GetPrivate() that returns (this - kBackendPrivate).
this should be backend.data_.

Store the TLS keys inside (backend.data_ - kBackendPrivate).
```
struct MpPrivateHeader {
    ThreadLocalKey tls_key_;
};

MpPrivateHeader* GetPrivate() {
    return ((char*)this - kBackendPrivate)
}
```

## CtxAllocator

Let's remove CtxAllocator concept completely.
We will pass allocator pointers around.


@CLAUDE.md 
The current issue is that alloc_ must be the last entry of the shared memory in order to avoid corrupting class parameters. For pblock and tblock, this is an easy change.

However, the main block is different due to the custom header. Simply placing alloc_ at the end there is problematic.

How do we fix this:
1. Make custom header a part of the backend, not the allocator. I actually like this a lot. 

Each backend has a private header and a shared header, both 4KB long. 
Add a new method called GetSharedHeader to MemoryBackend. GetPrivateRegion should be renamed to GetPrivateHeader(). kBackendPrivate should be renamed to kBackendHeaderSize

GetPrivateRegion() should be GetSharedHeader() - kBackendHeaderSize. GetSharedHeader() should be data_ - kBackendHeaderSize().

Remove all logic in the allocators for considering custom_header_size_. Remove custom_header_size_ completely from allocators. We should rename GetCustomHeader in Allocator to GetSharedHeader().

We should add a new class variable to allocator called data_start_ (this is not custom_header_size_). This represents the start of data relative to this_. Technically, this is just the size of the allocator class: data_start_ = sizeof(AllocT).

GetAllocatorDataStart() should not depend on GetCustomHeader / GetSharedHeader anymore. Instead we should return (this) + data_start_

@CLAUDE.md

In the MemoryBackend, I want to add another variable called priv_header_off_.
This will store the difference between data_ and the beginning of the shared segment in the MemoryBackend.
In each MemoryBackend, we need to set this priv_header_off_. 

For example, for PosixShmMmap, we do a mixed allocation.
1. The very first kBackendHeaderSize bytes of the buffer returned is the private header
2. The next kBackendHeaderSize bytes are the shared header.
3. The next bytes can be the metadata.
4. And then data_ is set.
5. And then priv_header_off_ is data_ - (1).


For PosixMmap,
1. We mmap the buffer
2. The very first kBackendHeaderSize bytes of the buffer returned is the private header
3. The next kBackendHeaderSize bytes are the shared header. 
4. After md, the next kBackendHeader bytes are the private header and the next are the shared header.
5. After this is what gets stored in data_. 

In MemoryBackend:
```
GetPrivateHeader(): GetPrivateHeader(data_)
GetSharedHeader(): GetSharedHeader(data_)
GetPrivateHeader(char *data): (data - priv_header_off_)
GetSharedHeader(char *data): GetPrivateHeader<char>(data) + kBackendHeaderSize
```

In Allocator:
```
GetPrivateHeader(): backend_.GetPrivateHeader(GetBackendData());
GetSharedHeader(): backend_.GetSharedHeader(GetBackendData());
```
