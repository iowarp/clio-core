#ifndef WRP_CAE_CORE_RUNTIME_H_
#define WRP_CAE_CORE_RUNTIME_H_

#include <chimaera/chimaera.h>
#include <wrp_cae/core/core_tasks.h>
#include <wrp_cae/core/core_client.h>
#include <memory>

// Forward declaration for CTE client
namespace wrp_cte::core {
  class Client;
}

namespace wrp_cae::core {

class Runtime : public chi::Container {
 public:
  // CreateParams type used by CHI_TASK_CC macro for lib_name access
  using CreateParams = wrp_cae::core::CreateParams;

  Runtime() = default;
  ~Runtime() override = default;

  // Virtual methods implemented in autogen/core_lib_exec.cc
  void Run(chi::u32 method, hipc::FullPtr<chi::Task> task_ptr, chi::RunContext& rctx) override;
  void Del(chi::u32 method, hipc::FullPtr<chi::Task> task_ptr) override;
  chi::u64 GetWorkRemaining() const override;
  void SaveTask(chi::u32 method, chi::SaveTaskArchive& archive, hipc::FullPtr<chi::Task> task_ptr) override;
  void LoadTask(chi::u32 method, chi::LoadTaskArchive& archive, hipc::FullPtr<chi::Task>& task_ptr) override;
  void NewCopy(chi::u32 method, const hipc::FullPtr<chi::Task>& orig_task, hipc::FullPtr<chi::Task>& dup_task, bool deep) override;
  void Aggregate(chi::u32 method, hipc::FullPtr<chi::Task> origin_task, hipc::FullPtr<chi::Task> replica_task) override;

  /**
   * Initialize container with pool information (REQUIRED)
   * This is called by the framework before Create is called
   */
  void Init(const chi::PoolId& pool_id, const std::string& pool_name,
            chi::u32 container_id = 0) override;


  /**
   * Create the container (Method::kCreate)
   * This method creates queues and sets up container resources
   * NOTE: Container is already initialized via Init() before Create is called
   */
  void Create(hipc::FullPtr<CreateTask> task, chi::RunContext& ctx);

  /**
   * Destroy the container (Method::kDestroy)
   */
  void Destroy(hipc::FullPtr<chi::Task> task, chi::RunContext& ctx) {
    HILOG(kInfo, "Core container destroyed for pool: {} (ID: {})",
          pool_name_, pool_id_);
  }

  /**
   * ParseOmni - Parse OMNI YAML file and schedule assimilation tasks (Method::kParseOmni)
   */
  void ParseOmni(hipc::FullPtr<ParseOmniTask> task, chi::RunContext& ctx);

 private:
  Client client_;
  std::shared_ptr<wrp_cte::core::Client> cte_client_;
};

}  // namespace wrp_cae::core

#endif  // WRP_CAE_CORE_RUNTIME_H_
