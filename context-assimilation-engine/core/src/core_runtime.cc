#include <wrp_cae/core/core_runtime.h>
#include <wrp_cae/core/factory/assimilation_ctx.h>
#include <wrp_cae/core/factory/assimilator_factory.h>
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include <sstream>
#include <vector>

// Include wrp_cte headers before opening namespace to avoid Method namespace collision
#include <wrp_cte/core/core_client.h>
#include <wrp_cte/core/core_tasks.h>

// Define ChiMod entry points using CHI_TASK_CC macro
CHI_TASK_CC(wrp_cae::core::Runtime)

namespace wrp_cae::core {

void Runtime::Create(hipc::FullPtr<CreateTask> task, chi::RunContext& ctx) {
  // Container is already initialized via Init() before Create is called
  // Do NOT call Init() here

  // Initialize CTE client using the CTE pool ID
  cte_client_ = std::make_shared<wrp_cte::core::Client>(wrp_cte::core::kCtePoolId);

  // Additional container-specific initialization logic here
  HILOG(kInfo, "Core container created and initialized for pool: {} (ID: {})",
        pool_name_, pool_id_);
}

void Runtime::ParseOmni(hipc::FullPtr<ParseOmniTask> task, chi::RunContext& ctx) {
  HILOG(kInfo, "ParseOmni called with {} bytes of serialized data",
        task->serialized_ctx_.size());

  // Deserialize the vector of AssimilationCtx
  std::vector<AssimilationCtx> assimilation_contexts;
  try {
    std::stringstream ss(task->serialized_ctx_.str());
    cereal::BinaryInputArchive ar(ss);
    ar(assimilation_contexts);
  } catch (const std::exception& e) {
    HELOG(kError, "ParseOmni: Failed to deserialize AssimilationCtx vector: {}", e.what());
    task->result_code_ = -1;
    task->error_message_ = e.what();
    task->num_tasks_scheduled_ = 0;
    return;
  }

  HILOG(kInfo, "ParseOmni: Processing {} assimilation contexts", assimilation_contexts.size());

  // Process each assimilation context
  chi::u32 tasks_scheduled = 0;
  AssimilatorFactory factory(cte_client_);

  for (size_t i = 0; i < assimilation_contexts.size(); ++i) {
    const auto& assimilation_ctx = assimilation_contexts[i];

    HILOG(kInfo, "ParseOmni: Processing context {}/{} - src: {}, dst: {}, format: {}",
          i + 1, assimilation_contexts.size(),
          assimilation_ctx.src, assimilation_ctx.dst, assimilation_ctx.format);

    // Get appropriate assimilator for this context
    auto assimilator = factory.Get(assimilation_ctx.src);

    if (!assimilator) {
      HELOG(kError, "ParseOmni: No assimilator found for source: {}", assimilation_ctx.src);
      task->result_code_ = -2;
      task->error_message_ = "No assimilator found for source: " + assimilation_ctx.src;
      task->num_tasks_scheduled_ = tasks_scheduled;
      return;
    }

    // Schedule the assimilation
    int result = assimilator->Schedule(assimilation_ctx);
    if (result != 0) {
      HELOG(kError, "ParseOmni: Assimilator failed for context {}/{} with error code: {}",
            i + 1, assimilation_contexts.size(), result);
      task->result_code_ = result;
      task->error_message_ = std::string("Assimilator failed");
      task->num_tasks_scheduled_ = tasks_scheduled;
      return;
    }

    tasks_scheduled++;
  }

  // Success
  task->result_code_ = 0;
  task->error_message_ = "";
  task->num_tasks_scheduled_ = tasks_scheduled;

  HILOG(kInfo, "ParseOmni: Successfully scheduled {} assimilations", tasks_scheduled);
}

}  // namespace wrp_cae::core
