@CLAUDE.md

Under test/unit add a subdirectory called allocator.

Add a new header file called allocator_test.h.

Implement a templated class. We are going to test the CtxAllocator apis.

```
template<typename AllocT>
class Test {
    hipc::CtxAllocator<AllocT> ctx_alloc_;
    Test(hipc::Allocator *alloc) {
        ctx_alloc_ = CtxAllocator<AllocT>(alloc);
    }
}
```

this class should test every API of the allocators. We should have at minimum the following tests:
1. Allocate and then free immediately in a loop. Same memory size
2. Allocate a bunch. Then free the bunch. Iteratively in a loop. Same memory size per alloc
3. Random allocation with random sizes between 0 and 1MB. Up to a total of 64MB or 5000 allocations.
After all allocations, free. Do this iteratively 16 times.
4. Multi-threaded. 8 threads calling the random allocation test. Use standard threads.

Then implement a source file called test_alloc.cc. Use catch2 to implement test cases.
Avoid TEST_CASE_METHOD and use TEST_CASE instead. 

Call the templated tester class for the MallocBackend and MallocAllocator only for now.
