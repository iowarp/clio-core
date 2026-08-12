/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * Cross-process CUDA IPC device-memory handle round trip regression test.
 *
 * AllocateAndRegisterGpuBackend's external-client branch used to send the
 * raw cudaMalloc'd pointer VALUE across to the runtime (instead of a real
 * cudaIpcMemHandle_t via cudaIpcGetMemHandle), and Runtime::RegisterMemory
 * on the receiving side never called cudaIpcOpenMemHandle to turn it back
 * into a pointer valid in ITS OWN process -- so a device-backed ShmPtr only
 * ever "worked" by accident within a single process.
 *
 * cudaIpcOpenMemHandle is documented as valid only across genuinely
 * separate processes (opening your own process's handle is not a
 * supported use), so this test spawns a REAL separate clio_run daemon
 * (RuntimeServer) and drives the real AllocateAndRegisterGpuBackend path
 * from a genuine external client. Verification is structural: the
 * RegisterMemoryTask round trip must return rc=0 / success_=true, meaning
 * the runtime's cudaIpcOpenMemHandle call on the real cross-process handle
 * did not error. This deliberately does not attempt byte-level readback
 * across the process boundary -- that would require either routing
 * through CTE's PutBlob (a separate, larger gap: its bulk transport
 * assumes host-readable memory, which a raw device pointer is not) or a
 * new purpose-built admin RPC, both out of scope for closing this
 * specific gap.
 */

#if (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL

#include "simple_test.h"
#include "runtime_server.h"

#include <clio_runtime/clio_runtime.h>
#include <clio_ctp/util/gpu_api.h>

namespace {
constexpr size_t kBytes = 4096;
constexpr char kPatternSeed = 0x5A;
}  // namespace

TEST_CASE("AllocateAndRegisterGpuBackend opens a real cudaIpcMemHandle "
          "across a genuine process boundary",
          "[gpu][ipc][cuda_ipc][register_memory][cross_process]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    return;
  }

  clio::run::test::RuntimeServer server;
  REQUIRE(server.Start());
  REQUIRE(server.WaitForReady());

  clio::run::test::SetEnvVar("CLIO_WITH_RUNTIME", "0");
  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false));

  // ---- Allocate real GPU device memory in THIS (client) process and fill
  // it with a known pattern -- proves the client side (GetIpcMemHandle) has
  // something real to hand off, even though this test only checks the
  // registration round trip's structural success. ----
  auto *ipc = CLIO_IPC;
  char *device_base = nullptr;
  auto alloc_id = ipc->AllocateAndRegisterGpuBackend(
      /*gpu_id=*/0, clio::run::gpu::IpcManager::MemKind::kDeviceMem, kBytes,
      &device_base);
  REQUIRE(!alloc_id.IsNull());
  REQUIRE(device_base != nullptr);

  std::vector<char> pattern(kBytes);
  for (size_t i = 0; i < kBytes; ++i) {
    pattern[i] = static_cast<char>((kPatternSeed + i) & 0xFF);
  }
  ctp::GpuApi::Memcpy(device_base, pattern.data(), kBytes);

  // AllocateAndRegisterGpuBackend already Wait()s on the RegisterMemoryTask
  // internally and only returns a non-null alloc_id when it succeeded, so
  // reaching here IS the structural proof: the runtime (a genuinely
  // separate process) received the real cudaIpcMemHandle_t and called
  // cudaIpcOpenMemHandle on it without erroring.

  ipc->FreeGpuBackend(/*gpu_id=*/0, alloc_id);
}

SIMPLE_TEST_MAIN()

#else

int main() { return 0; }

#endif  // (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL
