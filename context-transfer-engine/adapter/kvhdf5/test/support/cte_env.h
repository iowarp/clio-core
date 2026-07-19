#pragma once

// One-time CLIO/CTE bring-up shared by the GPU test binaries (clio_contract_tests
// and kvhdf5_e2e_tests). Both add test/support to their include dirs.
//
// Server start-up + CTE pool create + bdev registration is expensive and the
// CLIO server is process-global, so it must happen exactly once. A
// function-local static (SharedCteEnv) gives lazy, once-only construction the
// first time any test touches it; Catch2 runs all cases in one process so they
// share it. Bring-up failures throw, which Catch2 reports as a failed test.
//
// Host-only: guarded out of the nvcc device pass (it uses host clients + Catch2-
// reported throws). Include from the host side of a .cu integration test.

#if !CTP_IS_DEVICE_PASS

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/bdev/bdev_client.h>
#include <clio_runtime/singletons.h>
#include <clio_runtime/types.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>

#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <thread>

namespace kvhdf5::itest {

namespace cte = clio::cte::core;

struct ClioCteEnv {
    cte::TagId tag_id;

    ClioCteEnv() {
        using namespace std::chrono_literals;
        std::fprintf(stderr, "[init] bringing up CLIO server (once)\n");
        if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kServer))
            throw std::runtime_error("CLIO_INIT(kServer) failed");
        if (!cte::CLIO_CTE_CLIENT_INIT())
            throw std::runtime_error("CLIO_CTE_CLIENT_INIT failed");

        auto* cte_client = CLIO_CTE_CLIENT;
        if (cte_client == nullptr)
            throw std::runtime_error("CLIO_CTE_CLIENT is null");
        cte_client->Init(cte::kCtePoolId);

        cte::CreateParams params;
        auto create_task = cte_client->AsyncCreate(
            clio::run::PoolQuery::Dynamic(), cte::kCtePoolName, cte::kCtePoolId, params);
        create_task.Wait();
        if (create_task->GetReturnCode() != 0)
            throw std::runtime_error("CTE pool AsyncCreate failed");
        std::this_thread::sleep_for(50ms);

        // Register a kRam bdev target so PutBlob has somewhere to land.
        const clio::run::u64 kRamCapacity = 64ULL << 20;  // 64 MiB
        clio::run::PoolId bdev_pool_id(960, 0);
        clio::run::bdev::Client bdev_client(bdev_pool_id);
        auto bdev_create = bdev_client.AsyncCreate(
            clio::run::PoolQuery::Dynamic(), std::string("kvhdf5_itest_ram"),
            bdev_pool_id, clio::run::bdev::BdevType::kRam, kRamCapacity);
        bdev_create.Wait();
        if (bdev_create->GetReturnCode() != 0)
            throw std::runtime_error("bdev AsyncCreate failed");
        std::this_thread::sleep_for(50ms);

        auto reg_task = cte_client->AsyncRegisterTarget(
            "kvhdf5_itest_ram", clio::run::bdev::BdevType::kRam, kRamCapacity,
            clio::run::PoolQuery::Local(), bdev_pool_id);
        reg_task.Wait();
        if (reg_task->GetReturnCode() != 0)
            throw std::runtime_error("AsyncRegisterTarget failed");
        std::this_thread::sleep_for(50ms);

        auto tag_task = cte_client->AsyncGetOrCreateTag("kvhdf5_itest_tag");
        tag_task.Wait();
        if (tag_task->GetReturnCode() != 0)
            throw std::runtime_error("AsyncGetOrCreateTag failed");
        tag_id = tag_task->tag_id_;

        std::fprintf(stderr, "[init] ready (tag=(%u,%u))\n", tag_id.major_,
                     tag_id.minor_);
    }
};

/** Lazily construct (once) and return the shared CLIO/CTE environment. */
inline ClioCteEnv& SharedCteEnv() {
    static ClioCteEnv env;
    return env;
}

}  // namespace kvhdf5::itest

#endif  // !CTP_IS_DEVICE_PASS
