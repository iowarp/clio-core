# Distributed Task Scheduling - Modified Specification

## Overview

This specification describes modifications to the Chimaera runtime to support distributed task scheduling across multiple nodes. The design focuses on integrating with existing infrastructure (Lightbeam transport, admin chimod) and uses static domain resolution without virtual functions.

## Key Design Principles

1. **No Virtual Functions**: Tasks operate in shared memory; virtual dispatch is incompatible
2. **Lightbeam Integration**: Leverage existing Lightbeam transport factory instead of custom networking
3. **Admin Chimod Extension**: Add networking methods to existing admin container
4. **Static Domain Resolution**: Node groups determined before networking layer
5. **Template-Based Serialization**: Use CRTP pattern for compile-time polymorphism

## Configuration Changes

Add hostfile support to Chimaera configuration:

```cpp
// In chimaera_config.h
struct ChimaeraConfig {
  // ... existing fields ...
  
  std::string hostfile_path;  // Path to hostfile (empty = single node)
  std::vector<std::string> node_list;  // Parsed list of nodes
  u32 node_rank;  // This node's rank in the cluster
  
  void ParseHostfile() {
    if (hostfile_path.empty()) {
      node_list.push_back("localhost");
      node_rank = 0;
      return;
    }
    
    std::string expanded_path = hshm::ConfigParse::ExpandPath(hostfile_path);
    node_list = hshm::ConfigParse::ParseHostfile(expanded_path);
    
    // Determine our rank based on hostname
    std::string my_hostname = hshm::SystemInfo::GetHostname();
    for (u32 i = 0; i < node_list.size(); ++i) {
      if (node_list[i] == my_hostname) {
        node_rank = i;
        break;
      }
    }
  }
};
```

## Task Serialization Without Virtual Functions

### Template-Based Serialization Pattern

Since virtual functions cannot be used, we employ a Curiously Recurring Template Pattern (CRTP) approach:

```cpp
// In chimaera/task.h

/**
 * CRTP base for tasks with serialization support
 * Derived classes implement SerializeIn/SerializeOut as regular methods
 */
template<typename Derived>
class SerializableTask : public Task {
public:
  explicit SerializableTask(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc)
      : Task(alloc) {}
      
  // Base serialization for common Task fields
  template<typename Archive>
  void BaseSerializeIn(Archive& ar) {
    ar(pool_id_, task_node_, pool_query_, method_, task_flags_, period_ns_);
  }
  
  template<typename Archive>
  void BaseSerializeOut(Archive& ar) {
    // Serialize output-only base fields if any
  }
  
  // Static dispatch to derived class
  template<typename Archive>
  void DoSerializeIn(Archive& ar) {
    BaseSerializeIn(ar);
    static_cast<Derived*>(this)->SerializeIn(ar);
  }
  
  template<typename Archive>
  void DoSerializeOut(Archive& ar) {
    BaseSerializeOut(ar);
    static_cast<Derived*>(this)->SerializeOut(ar);
  }
};
```

### Task Implementation Example

```cpp
// In admin/admin_tasks.h

struct NetworkForwardTask : public SerializableTask<NetworkForwardTask> {
  // Network-specific fields
  IN chi::u32 dest_node_rank_;     // Target node in cluster
  IN chi::u64 net_key_;            // Unique network identifier
  INOUT chi::priv::string task_data_;   // Serialized task data
  IN chi::u32 original_method_;    // Original task's method ID
  OUT chi::u32 result_code_;       // Execution result
  
  // SHM constructor
  explicit NetworkForwardTask(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc)
      : SerializableTask<NetworkForwardTask>(alloc),
        dest_node_rank_(0),
        net_key_(0),
        task_data_(alloc),
        original_method_(0),
        result_code_(0) {}
  
  // Emplace constructor
  explicit NetworkForwardTask(
      const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc,
      const chi::TaskNode &task_node,
      const chi::PoolId &pool_id,
      const chi::DomainQuery &pool_query,
      chi::u32 dest_node,
      chi::u64 net_key,
      const std::string &task_data,
      chi::u32 original_method)
      : SerializableTask<NetworkForwardTask>(alloc),
        dest_node_rank_(dest_node),
        net_key_(net_key),
        task_data_(alloc, task_data),
        original_method_(original_method),
        result_code_(0) {
    method_ = Method::kNetworkForward;
    task_node_ = task_node;
    pool_id_ = pool_id;
    pool_query_ = pool_query;
  }
  
  // Serialization methods (not virtual!)
  template<typename Archive>
  void SerializeIn(Archive& ar) {
    ar(dest_node_rank_, net_key_, task_data_, original_method_);
  }
  
  template<typename Archive>
  void SerializeOut(Archive& ar) {
    ar(result_code_);
  }
};
```

## Archive Types for Task Serialization

Four archive types handle different serialization scenarios without virtual dispatch:

```cpp
// In chimaera/archives.h

/**
 * Archive for serializing task inputs (sending side)
 */
class TaskOutputArchiveIN {
private:
  std::stringstream stream_;
  cereal::BinaryOutputArchive ar_;
  
public:
  TaskOutputArchiveIN() : ar_(stream_) {}
  
  // Serialize a task using static dispatch
  template<typename TaskType>
  void SerializeTask(TaskType* task) {
    // Use compile-time type information
    task->DoSerializeIn(ar_);
  }
  
  // Bulk transfer support
  void bulk(hipc::ShmPtr<> p, size_t size, u32 flags) {
    if (flags & CHI_WRITE) {
      // Serialize the data for transfer
      ar_.saveBinary(p.ToPtr(), size);
    } else if (flags & CHI_EXPOSE) {
      // Just serialize the pointer metadata
      ar_(p.off_, size, flags);
    }
  }
  
  std::string GetData() const { return stream_.str(); }
};

/**
 * Archive for deserializing task inputs (receiving side)
 */
class TaskInputArchiveIN {
private:
  std::stringstream stream_;
  cereal::BinaryInputArchive ar_;
  
public:
  explicit TaskInputArchiveIN(const std::string& data) 
      : stream_(data), ar_(stream_) {}
  
  // Deserialize with known type
  template<typename TaskType>
  hipc::FullPtr<TaskType> DeserializeTask(
      const hipc::CtxAllocator<CHI_MAIN_ALLOC_T>& alloc) {
    auto task = CHI_IPC->NewTask<TaskType>(chi::kMainSegment, alloc);
    task->DoSerializeIn(ar_);
    return task;
  }
  
  void bulk(hipc::ShmPtr<>& p, size_t& size, u32& flags) {
    if (flags & CHI_WRITE) {
      // Allocate and deserialize data
      p = CHI_IPC->AllocateBuffer(size);
      ar_.loadBinary(p.ToPtr(), size);
    } else {
      ar_(p.off_, size, flags);
    }
  }
};

// Similar implementations for TaskOutputArchiveOUT and TaskInputArchiveOUT
```

## Admin Chimod Networking Extensions

### New Method Constants

```cpp
// In admin/autogen/admin_methods.h

namespace chimaera::admin {

namespace Method {
  // ... existing methods ...
  
  // Networking methods
  GLOBAL_CONST chi::u32 kClientSendTaskIn = 20;
  GLOBAL_CONST chi::u32 kServerRecvTaskIn = 21;
  GLOBAL_CONST chi::u32 kServerSendTaskOut = 22;
  GLOBAL_CONST chi::u32 kClientRecvTaskOut = 23;
}

} // namespace chimaera::admin
```

### Container Implementation Updates

```cpp
// In admin/admin_runtime.h

class Container : public chi::Container {
private:
  // Networking state
  std::unique_ptr<lightbeam::Transport> transport_;
  std::unordered_map<u64, hipc::FullPtr<Task>> pending_tasks_;
  std::unordered_map<u32, TaskOutputArchiveIN> send_buffers_;
  
public:
  // ... existing methods ...
  
  /**
   * Client-side: Collect and send tasks to remote nodes
   * Called periodically to batch tasks for network transfer
   */
  void ClientSendTaskIn(hipc::FullPtr<ClientSendTaskInTask> task, 
                       chi::RunContext& ctx) {
    auto* worker = CHI_CUR_WORKER;
    auto* lane = CHI_CUR_LANE;
    size_t lane_size = lane->GetSize();
    
    // Group tasks by destination node
    std::unordered_map<u32, TaskOutputArchiveIN> archives;
    
    for (size_t i = 0; i < lane_size; ++i) {
      auto task_ptr = lane->Dequeue();
      if (task_ptr.IsNull()) break;
      
      auto* base_task = task_ptr.Cast<Task>().ptr_;
      
      // Extract destination from domain query (static resolution)
      u32 dest_node = base_task->pool_query_.GetTargetNode();
      
      // Assign unique network key
      base_task->net_key_ = reinterpret_cast<u64>(base_task);
      pending_tasks_[base_task->net_key_] = task_ptr;
      
      // Serialize based on method type (compile-time dispatch)
      SerializeTaskByMethod(archives[dest_node], base_task);
    }
    
    // Send batched tasks using Lightbeam
    for (auto& [dest_node, archive] : archives) {
      transport_->Send(dest_node, archive.GetData());
    }
  }
  
  /**
   * Helper to serialize tasks based on method ID
   * Uses switch-case for compile-time type resolution
   */
  void SerializeTaskByMethod(TaskOutputArchiveIN& ar, Task* task) {
    switch (task->method_) {
      case Method::kCreate: {
        auto* typed_task = static_cast<CreateTask*>(task);
        ar.SerializeTask(typed_task);
        break;
      }
      case Method::kCustom: {
        auto* typed_task = static_cast<CustomTask*>(task);
        ar.SerializeTask(typed_task);
        break;
      }
      // Add cases for all task types
      default:
        LOG(ERROR) << "Unknown task method: " << task->method_;
    }
  }
  
  /**
   * Server-side: Receive and schedule remote tasks
   * Periodic task that polls for incoming tasks
   */
  void ServerRecvTaskIn(hipc::FullPtr<ServerRecvTaskInTask> task,
                       chi::RunContext& ctx) {
    // Poll Lightbeam for incoming messages
    std::string data;
    u32 source_node;
    
    while (transport_->TryRecv(source_node, data)) {
      TaskInputArchiveIN archive(data);
      
      // Deserialize task count
      u32 num_tasks;
      archive.ar_ >> num_tasks;
      
      for (u32 i = 0; i < num_tasks; ++i) {
        // Read method ID to determine task type
        chi::u32 method;
        archive.ar_ >> method;
        
        // Deserialize and schedule based on method
        DeserializeAndSchedule(archive, method, source_node);
      }
    }
  }
  
  /**
   * Helper to deserialize and schedule tasks
   */
  void DeserializeAndSchedule(TaskInputArchiveIN& ar, 
                              chi::u32 method,
                              u32 source_node) {
    hipc::CtxAllocator<CHI_MAIN_ALLOC_T> alloc(CHI_IPC->GetAllocator());
    
    switch (method) {
      case Method::kCreate: {
        auto task = ar.DeserializeTask<CreateTask>(alloc);
        task->pool_query_.SetLocal();  // Execute locally
        CHI_IPC->Enqueue(task);
        break;
      }
      case Method::kCustom: {
        auto task = ar.DeserializeTask<CustomTask>(alloc);
        task->pool_query_.SetLocal();
        CHI_IPC->Enqueue(task);
        break;
      }
      // Add cases for all task types
    }
  }
  
  /**
   * Monitor method for ClientSendTaskIn
   */
  void MonitorClientSendTaskIn(chi::MonitorModeId mode,
                               hipc::FullPtr<ClientSendTaskInTask> task,
                               chi::RunContext& ctx) {
    switch (mode) {
      case chi::MonitorModeId::kLocalSchedule:
        // Route to networking queue
        if (auto* lane = GetLane(chi::kNetworking, 0)) {
          lane->Enqueue(task.shm_);
        }
        break;
    }
  }
  
  // Similar implementations for ServerSendTaskOut and ClientRecvTaskOut
};
```

## Lightbeam Transport Integration

```cpp
// In admin runtime initialization

void Container::Create(hipc::FullPtr<CreateTask> task, chi::RunContext& ctx) {
  chi::Container::Init(task->pool_id_, task->pool_name_.str());
  
  // Initialize queues
  CreateLocalQueue(chi::kLowLatency, 4);
  CreateLocalQueue(chi::kHighLatency, 2);
  CreateLocalQueue(chi::kNetworking, 1);  // Dedicated networking queue
  
  // Initialize Lightbeam transport
  auto& config = CHI_CONFIG;
  if (!config.node_list.empty() && config.node_list.size() > 1) {
    lightbeam::TransportConfig lb_config;
    lb_config.node_list = config.node_list;
    lb_config.node_rank = config.node_rank;
    
    transport_ = lightbeam::TransportFactory::Create("tcp", lb_config);
    
    // Schedule periodic networking tasks
    SchedulePeriodicTask<ServerRecvTaskInTask>(100000);  // 100ms
    SchedulePeriodicTask<ClientRecvTaskOutTask>(100000);
  }
}
```

## Worker Task Resolution Updates

```cpp
// In worker.cc

void Worker::ResolveTask(hipc::FullPtr<Task> task) {
  auto& pool_query = task->pool_query_;
  
  // Case 1: Dynamic domain resolution
  if (pool_query.IsDynamic()) {
    auto* container = pool_manager_->GetContainer(task->pool_id_);
    container->Monitor(chi::MonitorModeId::kGlobalSchedule, task->method_, task, *run_ctx_);
    // Fall through to check if now resolved
  }
  
  // Case 2: Remote task - forward to admin for networking
  if (!pool_query.IsLocal()) {
    // Get admin container
    auto admin_pool_id = pool_manager_->GetAdminPoolId();
    auto* admin_container = pool_manager_->GetContainer(admin_pool_id);
    
    // Create network forward task
    hipc::CtxAllocator<CHI_MAIN_ALLOC_T> alloc(CHI_IPC->GetAllocator());
    auto forward_task = CHI_IPC->NewTask<NetworkForwardTask>(
        chi::kMainSegment, alloc,
        task->task_node_,
        admin_pool_id,
        chi::DomainQuery::Local(),
        pool_query.GetTargetNode(),
        reinterpret_cast<u64>(task.ptr_),
        SerializeTaskToString(task),  // Helper function
        task->method_
    );
    
    // Route through admin's networking queue
    admin_container->Monitor(chi::MonitorModeId::kLocalSchedule, 
                           Method::kClientSendTaskIn, 
                           forward_task, 
                           *run_ctx_);
    return;
  }
  
  // Case 3: Local task - normal routing
  auto* container = pool_manager_->GetContainer(task->pool_id_);
  container->Monitor(chi::MonitorModeId::kLocalSchedule, 
                    task->method_, 
                    task, 
                    *run_ctx_);
}
```

## Static Domain Resolution

Domain resolution is determined before tasks reach the networking layer:

```cpp
// In chimaera/domain_query.h

class DomainQuery {
private:
  u32 target_node_;  // Target node rank (0xFFFFFFFF = local)
  u32 flags_;
  
public:
  // Check if task should execute locally
  bool IsLocal() const { 
    return target_node_ == 0xFFFFFFFF || 
           target_node_ == CHI_CONFIG.node_rank;
  }
  
  // Get target node for remote execution
  u32 GetTargetNode() const { return target_node_; }
  
  // Force local execution (used after receiving remote task)
  void SetLocal() { target_node_ = 0xFFFFFFFF; }
  
  // Set specific target node
  void SetTargetNode(u32 node) { target_node_ = node; }
  
  // Check if resolution is needed
  bool IsDynamic() const { return flags_ & kDynamicFlag; }
};
```

## Key Implementation Notes

### 1. No Virtual Functions
- All serialization uses templates and compile-time dispatch
- Method IDs drive switch-case statements for type resolution
- CRTP pattern enables base class serialization without virtuals

### 2. Lightbeam Integration
- Transport factory handles all network communication
- No custom socket programming needed
- Leverage existing message batching and reliability

### 3. Admin Chimod Pattern
- Follow MODULE_DEVELOPMENT_GUIDE patterns exactly
- Add new Method constants to namespace
- Implement Monitor methods with kLocalSchedule
- Use dedicated networking queue

### 4. Memory Management
- Tasks allocated in shared memory segments
- Network keys track tasks across nodes
- Bulk transfers use Lightbeam's zero-copy when possible

### 5. Error Handling
- Network failures don't crash runtime
- Tasks can timeout and be rescheduled
- Graceful degradation to single-node mode

## Testing Strategy

1. **Single Node**: Verify no regression in local execution
2. **Two Nodes**: Test basic task forwarding and results
3. **Multiple Nodes**: Validate load distribution
4. **Failure Cases**: Test node disconnection handling
5. **Performance**: Measure overhead of serialization

## Migration Path

1. Add configuration support for hostfile
2. Implement serialization methods in existing tasks
3. Add networking methods to admin chimod
4. Update worker resolution logic
5. Integrate Lightbeam transport
6. Test with increasing cluster sizes

This design maintains compatibility with existing code while adding distributed capabilities through careful extension of existing components rather than wholesale replacement.