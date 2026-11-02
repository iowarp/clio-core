/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * gpu2cpu ring BACKPRESSURE.
 *
 * The gpu2cpu lane is a ctp::ipc::multi_mpsc_ring_buffer, i.e. it is built with
 * RING_BUFFER_WAIT_FOR_SPACE. ring_buffer::Emplace() claims a slot with
 * `tail_.fetch_add_system(1)` and then, if the ring is full, SPINS on the device
 * re-reading `head_` until the CPU consumer frees a slot. So a GPU producer that
 * outruns the ring does not fail -- it PARKS ON THE DEVICE and waits for the CPU
 * to drain. The CPU side (ring_buffer::PopUnlocked) is what advances `head_`.
 *
 * Three cases here, all of which must pass:
 *
 *   1. "producer fills the ring exactly" -- submits depth-1 tasks, so no
 *      producer ever has to wait for space. Guards the normal path.
 *
 *   2. "producer OVERRUNS the ring" -- 3x depth, so producers must block in
 *      Emplace's wait-for-space spin and be drained by the CPU.
 *
 *   3. "producer OVERRUNS the ring hard" -- 16x depth.
 *
 * ---------------------------------------------------------------------------
 * The bug these cases pin (FIXED -- keep them green)
 * ---------------------------------------------------------------------------
 * Cases 2 and 3 used to deadlock. Emplace's wait-for-space spin re-read head_
 * with head_.load_device(), which is implemented as atomicAdd(&head_, 0) -- a
 * read-modify-WRITE, not a load. This ring lives in PINNED HOST MEMORY and its
 * consumer is the CPU, so a device-scope RMW against it is not coherent with
 * the host: each parked GPU producer wrote its stale cached head_ straight back
 * over the value PopUnlocked's head_.store_system(head + 1) had just stored.
 * The consumer's progress was erased as fast as it was made, head_ froze, and
 * the parked producers spun forever on a head_ that could never move.
 *
 * The signature was unmistakable -- the CPU would complete a pop or two and
 * then head_ would read back as its pre-park value:
 *
 *     head=1024  tail=4099  depth=1025
 *     entries the CPU sees READY = 1023
 *     tasks actually executed by the CPU = 2      <-- its stores were lost
 *
 * Fix: the gpu2cpu lane is now a RING_BUFFER_HOST_CONSUMER ring, so the spin
 * re-reads head_ with a pure volatile load (load_system) that cannot clobber.
 * See RING_BUFFER_HOST_CONSUMER in ring_buffer.h.
 *
 * FULL WRITE-UP: agents/paper-writing/traces/12-ring-backpressure-fix.md, and
 * GPU2CPU_RING_DEADLOCK.md next to this file (which records the refuted
 * theories -- do not re-derive them).
 *
 * Suffix `_gpu.cc` so NVCC/HIPCC compiles it, matching its sibling
 * test_gpu_kernel_stress_gpu.cc.
 */

#if (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL

#include "simple_test.h"
#include "test_gpu_kernel_stress_common.h"

#include <clio_ctp/util/gpu_api.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

using TaskT = clio::run::MOD_NAME::GpuSubmitTask;

namespace {

/**
 * How long the CPU consumer may go WITHOUT EXECUTING A SINGLE NEW TASK before
 * we call it wedged. This is a stall window, not a wall-clock budget: the
 * drain rate depends on the host (CI saw ~50k tasks/s on Frontera), and a
 * fixed total deadline turned a slow-but-healthy drain into a false "WEDGED".
 */
constexpr int kStallTimeoutMs = 10000;

/** Hard ceiling on one RunOverrun, well inside the ctest TIMEOUT of 180 s. */
constexpr int kTotalTimeoutMs = 120000;

/**
 * PRODUCER-ONLY submit kernel: pushes its task and returns immediately. It never
 * calls future.Wait(), so the ONLY thing that can park this kernel is the ring
 * being full -- which is exactly what we want to exercise.
 *
 * One block per task; thread 0 of each block does the Send (IpcGpu2Cpu::SendIn
 * is gated on threadIdx.x == 0).
 */
__global__ void Gpu2CpuOverrunKernel(clio::run::IpcManagerGpuInfo gpu_info,
                                     ctp::ipc::FullPtr<TaskT> *task_handles,
                                     clio::run::u32 num_tasks) {
  CLIO_GPU_INIT(gpu_info, /*ipc_ptr=*/nullptr);
  if (threadIdx.x != 0) return;
  clio::run::u32 slot = blockIdx.x;
  if (slot >= num_tasks) return;
  // Fire and forget -- no Wait(). See the note in ChiGpuKernelStress about using
  // g_ipc_manager_ptr rather than the CLIO_IPC macro here.
  (void)g_ipc_manager_ptr->Send(task_handles[slot]);
  (void)g_ipc_manager;
}

}  // namespace

#if !CTP_IS_DEVICE_PASS

namespace {

/**
 * Depth of the gpu2cpu ring a kernel Send() actually pushes into.
 *
 * When the device ring is live, IpcGpu2Cpu::SendIn pushes to
 * gpu_info.gpu2cpu_ring and never touches the lane, so the lane's depth says
 * nothing about when producers park. Sizing the overrun from the lane made the
 * "hard" case push 128x the live ring instead of 16x.
 *
 * @param info Kernel-facing GPU info for the device under test.
 * @return Slot count of the ring producers block on.
 */
size_t LiveRingDepth(const clio::run::IpcManagerGpuInfo &info) {
  if (info.gpu2cpu_ring != nullptr) return clio::run::kGpuRingCapacity;
  return info.gpu2cpu_queue->GetLane(0, 0).GetDepth();
}

/**
 * Placement-construct `num_tasks` tasks in a registered pinned backend.
 *
 * Slot i carries test_value = i, so the chimod must produce
 * result_value_ == i * 2 + gpu_id.
 *
 * @param alloc_id Allocator id of the registered backend.
 * @param base Host-visible base address of that backend.
 * @param gpu_id Device the tasks target.
 * @param num_tasks Number of tasks to place.
 * @return One FullPtr per task, in slot order.
 */
std::vector<ctp::ipc::FullPtr<TaskT>> PlaceTasks(ctp::ipc::AllocatorId alloc_id,
                                                 char *base,
                                                 clio::run::u32 gpu_id,
                                                 clio::run::u32 num_tasks) {
  using namespace chi_test_gpu_stress;
  std::vector<ctp::ipc::FullPtr<TaskT>> handles;
  handles.reserve(num_tasks);
  for (clio::run::u32 i = 0; i < num_tasks; ++i) {
    size_t task_off = static_cast<size_t>(i) * sizeof(TaskT);
    auto *task = new (base + task_off)
        TaskT(clio::run::CreateTaskId(), GetTestPoolId(),
              clio::run::PoolQuery::Dynamic(), gpu_id, /*test_value=*/i);
    task->fut_.task_size_ = static_cast<clio::run::u32>(sizeof(TaskT));
    ctp::ipc::FullPtr<TaskT> fp;
    fp.shm_.alloc_id_ = alloc_id;
    fp.shm_.off_ = task_off;
    fp.ptr_ = task;
    handles.push_back(fp);
  }
  return handles;
}

/**
 * Count the tasks the CPU worker has executed.
 *
 * @param handles Tasks placed by PlaceTasks.
 * @param gpu_id Device the tasks target.
 * @return Number of tasks whose result_value_ is already correct.
 */
clio::run::u32 CountExecuted(const std::vector<ctp::ipc::FullPtr<TaskT>> &handles,
                             clio::run::u32 gpu_id) {
  clio::run::u32 executed = 0;
  for (clio::run::u32 i = 0; i < handles.size(); ++i) {
    if (handles[i]->result_value_ == (i * 2u) + gpu_id) ++executed;
  }
  return executed;
}

/**
 * Poll until the producer kernel retired and every task was executed.
 *
 * Never blocks on Synchronize(): if the producer wedges against the CPU
 * consumer, a blocking Synchronize would hang the whole binary. A wedge is
 * declared only when no new task executes for kStallTimeoutMs.
 *
 * @param stream Stream the producer kernel was launched on.
 * @param handles Tasks placed by PlaceTasks.
 * @param gpu_id Device the tasks target.
 * @param kernel_done Out: whether the producer kernel retired.
 * @param executed Out: tasks executed when polling stopped.
 * @return true if the kernel retired and every task was executed.
 */
bool WaitForDrain(void *stream,
                  const std::vector<ctp::ipc::FullPtr<TaskT>> &handles,
                  clio::run::u32 gpu_id, bool *kernel_done,
                  clio::run::u32 *executed) {
  using Clock = std::chrono::steady_clock;
  const auto start = Clock::now();
  const auto hard_deadline = start + std::chrono::milliseconds(kTotalTimeoutMs);
  auto last_progress = start;
  *kernel_done = false;
  *executed = 0;
  while (Clock::now() < hard_deadline) {
    if (!*kernel_done) *kernel_done = ctp::GpuApi::StreamQuery(stream);
    clio::run::u32 now_executed = CountExecuted(handles, gpu_id);
    if (*kernel_done && now_executed == handles.size()) {
      *executed = now_executed;
      std::fprintf(stderr, "[BACKPRESSURE] drained %u tasks in %lld ms\n",
                   now_executed,
                   static_cast<long long>(
                       std::chrono::duration_cast<std::chrono::milliseconds>(
                           Clock::now() - start).count()));
      return true;
    }
    if (now_executed != *executed) {
      *executed = now_executed;
      last_progress = Clock::now();
    } else if (Clock::now() - last_progress >
               std::chrono::milliseconds(kStallTimeoutMs)) {
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  return false;
}

/**
 * Print the state of the ring producers were pushing into, for a wedge.
 *
 * For the live device ring that is the host consumer's tail and how many of
 * the next kGpuRingCapacity slots carry a published stamp; for the legacy
 * lane it is head/tail and the READY entries the CPU can see.
 *
 * @param info Kernel-facing GPU info for the device under test.
 * @param kernel_done Whether the producer kernel retired.
 * @param executed Tasks executed.
 * @param num_tasks Tasks submitted.
 */
void DumpWedge(const clio::run::IpcManagerGpuInfo &info, bool kernel_done,
               clio::run::u32 executed, clio::run::u32 num_tasks) {
  if (info.gpu2cpu_ring != nullptr) {
    auto *gpu_ipc = CLIO_CPU_IPC->GetGpuIpcManager();
    const auto &ring = gpu_ipc->per_gpu_devices_[info.gpu_id].ring;
    const unsigned long long tail = ring.tail;
    size_t n_ready = 0;
    for (unsigned long long s = tail; s < tail + clio::run::kGpuRingCapacity;
         ++s) {
      const unsigned int want =
          static_cast<unsigned int>(s / clio::run::kGpuRingCapacity) + 1u;
      if (ring.host_ready[s & clio::run::kGpuRingMask] == want) ++n_ready;
    }
    std::fprintf(stderr,
                 "[BACKPRESSURE] WEDGED (device ring): producer_retired=%d  "
                 "consumer tail=%llu capacity=%u  published_slots_ahead=%zu  "
                 "pending=%zu  tasks_executed=%u/%u\n",
                 (int)kernel_done, tail, clio::run::kGpuRingCapacity, n_ready,
                 ring.pending.size() - ring.pending_pos, executed, num_tasks);
    return;
  }
  auto &lane = info.gpu2cpu_queue->GetLane(0, 0);
  const size_t depth = lane.GetDepth();
  size_t n_ready = 0;
  for (size_t i = 0; i < depth; ++i) {
    clio::run::gpu::Future<clio::run::Task> pv;
    if (lane.Peek(lane.GetHead() + i, pv)) ++n_ready;
  }
  std::fprintf(stderr,
               "[BACKPRESSURE] WEDGED (lane): producer_retired=%d  ring "
               "head=%llu tail=%llu depth=%zu  entries_the_CPU_sees_READY=%zu  "
               "tasks_executed=%u/%u\n",
               (int)kernel_done, (unsigned long long)lane.GetHead(),
               (unsigned long long)lane.GetTail(), depth, n_ready, executed,
               num_tasks);
}

/**
 * Submit `num_tasks` tasks producer-only into the gpu2cpu ring and wait for the
 * CPU worker to drain them all.
 *
 * @param num_tasks Number of tasks (one kernel block each) to submit.
 * @return true if the producer kernel retired AND every task was executed.
 */
bool RunOverrun(clio::run::u32 num_tasks) {
  auto *ipc = CLIO_CPU_IPC;
  const clio::run::u32 gpu_id = 0;

  const size_t backend_bytes =
      static_cast<size_t>(num_tasks) * sizeof(TaskT) + 256;
  char *base = nullptr;
  ctp::ipc::AllocatorId alloc_id = ipc->AllocateAndRegisterGpuBackend(
      gpu_id, clio::run::gpu::IpcManager::MemKind::kPinnedHost, backend_bytes,
      &base);
  REQUIRE(!alloc_id.IsNull());
  REQUIRE(base != nullptr);
  auto handles = PlaceTasks(alloc_id, base, gpu_id, num_tasks);

  // BYTES, not a count: MallocHost passes its argument straight to
  // cudaMallocHost. Passing num_tasks alone under-allocated 24x, and the
  // hard-overrun subtest's handle staging wrote past the mapping -- the
  // SEGV this test "found" was its own.
  auto *task_handle_dev = ctp::GpuApi::MallocHost<ctp::ipc::FullPtr<TaskT>>(
      static_cast<size_t>(num_tasks) * sizeof(ctp::ipc::FullPtr<TaskT>));
  REQUIRE(task_handle_dev != nullptr);
  for (clio::run::u32 i = 0; i < num_tasks; ++i) task_handle_dev[i] = handles[i];

  clio::run::IpcManagerGpuInfo info =
      ipc->GetGpuIpcManager()->GetGpuInfo(gpu_id);
  REQUIRE(info.gpu2cpu_queue != nullptr);
  std::fprintf(stderr,
               "[BACKPRESSURE] submitting %u tasks producer-only into a "
               "depth-%zu %s ring\n",
               num_tasks, LiveRingDepth(info),
               info.gpu2cpu_ring != nullptr ? "device" : "lane");

  void *stream = ctp::GpuApi::CreateStream();
  REQUIRE(stream != nullptr);
  Gpu2CpuOverrunKernel<<<num_tasks, 32, 0, static_cast<cudaStream_t>(stream)>>>(
      info, task_handle_dev, num_tasks);

  bool kernel_done = false;
  clio::run::u32 executed = 0;
  const bool ok =
      WaitForDrain(stream, handles, gpu_id, &kernel_done, &executed);
  if (!ok) DumpWedge(info, kernel_done, executed, num_tasks);

  // Do NOT DestroyStream / FreeGpuBackend on the wedged path: the producer
  // kernel is still resident and still referencing both.
  if (ok) {
    ctp::GpuApi::DestroyStream(stream);
    ctp::GpuApi::FreeHost(reinterpret_cast<char *>(task_handle_dev));
    ipc->FreeGpuBackend(gpu_id, alloc_id);
  }
  return ok;
}

}  // namespace

TEST_CASE("gpu2cpu: producer fills the ring exactly", "[gpu2cpu][backpressure]") {
  using namespace chi_test_gpu_stress;
  EnsureInit();
  auto *ipc = CLIO_CPU_IPC;
  clio::run::IpcManagerGpuInfo info =
      ipc->GetGpuIpcManager()->GetGpuInfo(/*gpu_id=*/0);
  REQUIRE(info.gpu2cpu_queue != nullptr);

  // depth-1 submissions is the most that can be in flight without any producer
  // having to wait for space, so nothing should ever park in Emplace.
  const size_t depth = LiveRingDepth(info);
  REQUIRE(depth > 1);
  REQUIRE(RunOverrun(static_cast<clio::run::u32>(depth - 1)));
}

TEST_CASE("gpu2cpu: producer overruns the ring", "[gpu2cpu][backpressure]") {
  using namespace chi_test_gpu_stress;
  EnsureInit();

  auto *ipc = CLIO_CPU_IPC;
  clio::run::IpcManagerGpuInfo info =
      ipc->GetGpuIpcManager()->GetGpuInfo(/*gpu_id=*/0);
  REQUIRE(info.gpu2cpu_queue != nullptr);
  const size_t depth = LiveRingDepth(info);

  // 3x the ring depth: producers must block in Emplace at least twice over.
  REQUIRE(RunOverrun(static_cast<clio::run::u32>(depth * 3)));
}

TEST_CASE("gpu2cpu: producer overruns the ring hard", "[gpu2cpu][backpressure]") {
  using namespace chi_test_gpu_stress;
  EnsureInit();

  auto *ipc = CLIO_CPU_IPC;
  clio::run::IpcManagerGpuInfo info =
      ipc->GetGpuIpcManager()->GetGpuInfo(/*gpu_id=*/0);
  REQUIRE(info.gpu2cpu_queue != nullptr);
  const size_t depth = LiveRingDepth(info);

  // 16x the ring depth. This is far past anything gsbench asks for
  // (chunks * snaps == the number of gpu2cpu submissions; the arm that used to
  // hang was 128 chunks x 12 snaps = 1536). A blocking producer must be
  // drainable at any overrun ratio, so this is the real invariant.
  REQUIRE(RunOverrun(static_cast<clio::run::u32>(depth * 16)));
}

#endif  // !CTP_IS_DEVICE_PASS

SIMPLE_TEST_MAIN()

#else  // CUDA/ROCm not enabled (or SYCL is)

int main() { return 0; }

#endif
