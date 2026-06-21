// Stress test: verify _ProducerConsumerAllocator reuses freed memory.
//
// Pattern mirrors IpcManager::AllocateBuffer/FreeBuffer in CLIO_IPC_MODE=shm:
// repeat (Allocate -> Free) tightly. With reuse, the heap should never grow.
#include <catch2/catch_test_macros.hpp>
#include "allocator_test.h"
#include "clio_ctp/memory/backend/posix_mmap.h"
#include "clio_ctp/memory/backend/posix_shm_mmap.h"
#include "clio_ctp/memory/allocator/mp_allocator.h"
#include <pthread.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>


namespace {
struct ReuseResult {
  size_t completed;
  size_t fail_iter;  // first iteration that returned null, 0 if none
};

ReuseResult RunReuse(ctp::ipc::ProducerConsumerAllocator *alloc,
                     size_t alloc_size, size_t iterations) {
  ReuseResult r{0, 0};
  for (size_t i = 0; i < iterations; ++i) {
    auto ptr = alloc->Allocate<char>(alloc_size);
    if (ptr.IsNull()) {
      r.fail_iter = i;
      return r;
    }
    std::memset(ptr.ptr_, static_cast<unsigned char>(i & 0xFF), alloc_size);
    alloc->Free(ptr);
    ++r.completed;
  }
  return r;
}
}  // namespace

TEST_CASE("ProducerConsumerAllocator - Reuse: 4400B same-thread tight loop",
          "[ProducerConsumerAllocator][reuse]") {
  ctp::ipc::PosixMmap backend;
  // 64MB heap: way more than 4400B but far less than 1M*4400B (=4.2GB)
  size_t heap_size = 64 * 1024 * 1024;
  size_t header = sizeof(ctp::ipc::ProducerConsumerAllocator);
  REQUIRE(backend.shm_init(ctp::ipc::MemoryBackendId(99, 0), header + heap_size));
  auto *alloc = backend.MakeAlloc<ctp::ipc::ProducerConsumerAllocator>();
  REQUIRE(alloc != nullptr);

  auto r = RunReuse(alloc, 4400, 1'000'000);
  std::printf("[reuse-4400] completed=%zu fail_iter=%zu\n",
              r.completed, r.fail_iter);
  REQUIRE(r.fail_iter == 0);
  REQUIRE(r.completed == 1'000'000);

  alloc->shm_detach();
}

TEST_CASE("ProducerConsumerAllocator - Reuse: 4400B in worker thread",
          "[ProducerConsumerAllocator][reuse]") {
  ctp::ipc::PosixMmap backend;
  size_t heap_size = 64 * 1024 * 1024;
  size_t header = sizeof(ctp::ipc::ProducerConsumerAllocator);
  REQUIRE(backend.shm_init(ctp::ipc::MemoryBackendId(99, 1), header + heap_size));
  auto *alloc = backend.MakeAlloc<ctp::ipc::ProducerConsumerAllocator>();
  REQUIRE(alloc != nullptr);

  ReuseResult r{0, 0};
  std::thread t([&] { r = RunReuse(alloc, 4400, 1'000'000); });
  t.join();
  std::printf("[reuse-4400-thread] completed=%zu fail_iter=%zu\n",
              r.completed, r.fail_iter);
  REQUIRE(r.fail_iter == 0);
  REQUIRE(r.completed == 1'000'000);

  alloc->shm_detach();
}

// Simulates the IpcManager flow: producer process creates the allocator
// via shm_init, then a *different* process attaches via shm_attach. The
// producer then runs alloc/Free in a loop. With the current code, the
// runtime's shm_attach calls CreateTls(tblock_key_, ...) which overwrites
// the producer's pthread_key_t in shared memory, so producer's
// pthread_setspecific calls silently fail and EnsureTls allocates a new
// 2 MiB PcThreadBlock on every alloc.
TEST_CASE("ProducerConsumerAllocator - Reuse after cross-process attach",
          "[ProducerConsumerAllocator][reuse][crossproc]") {
  // Use a named SHM region so a child process can attach.
  std::string url = "test_mp_reuse_xproc";
  ::shm_unlink(url.c_str());

  ctp::ipc::PosixShmMmap backend;
  size_t heap_size = 64 * 1024 * 1024;
  size_t header = sizeof(ctp::ipc::ProducerConsumerAllocator);
  REQUIRE(backend.shm_init(ctp::ipc::MemoryBackendId(77, 0),
                            header + heap_size, url));
  auto *alloc = backend.MakeAlloc<ctp::ipc::ProducerConsumerAllocator>();
  REQUIRE(alloc != nullptr);

  // Fork a child to act as "the runtime": attach to the allocator and exit.
  // shm_attach internally calls _ProducerConsumerAllocator::shm_attach which
  // calls CreateTls(tblock_key_, ...) -> pthread_key_create -> overwrites
  // tblock_key_.pthread_key_ in shared memory with the *child's* pthread key.
  pid_t pid = fork();
  REQUIRE(pid >= 0);
  if (pid == 0) {
    // Child: allocate some pthread keys first to force a different key index
    // than the parent ended up with, simulating what a long-running runtime
    // process looks like (it has already allocated many keys for other
    // subsystems before attaching).
    pthread_key_t pad[8];
    for (auto &k : pad) pthread_key_create(&k, nullptr);
    ctp::ipc::PosixShmMmap child_backend;
    if (!child_backend.shm_attach(url)) _exit(2);
    if (child_backend.AttachAlloc<ctp::ipc::ProducerConsumerAllocator>() == nullptr)
      _exit(3);
    _exit(0);
  }
  int status = 0;
  REQUIRE(waitpid(pid, &status, 0) == pid);
  REQUIRE(WIFEXITED(status));
  REQUIRE(WEXITSTATUS(status) == 0);

  // Now run the same alloc/free loop as the in-process test. With reuse
  // working we should be able to do far more iterations than the heap can
  // hold at once.
  auto r = RunReuse(alloc, 4400, 1'000'000);
  std::printf("[reuse-4400-xproc] completed=%zu fail_iter=%zu\n",
              r.completed, r.fail_iter);
  REQUIRE(r.fail_iter == 0);
  REQUIRE(r.completed == 1'000'000);

  alloc->shm_detach();
  ::shm_unlink(url.c_str());
}
