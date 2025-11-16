Do the following:
1. Use the project-scaffolder agent to build an initial working skeleton of this specification, with all data structures and classes compiling.

# Chimaera

Chimaera is a distributed task execution framework. Tasks represent arbitrary C++ functions, similar to RPCs. However, Chimaera aims to implement dynamic load balancing and reorganization to reduce stress. Chimaera's fundamental abstraction are ChiPools and ChiContainers. A ChiPool represents a distributed system (e.g., key-value store), while a ChiContainer represents a subset of the global state (e.g., a bucket). These ChiPools can be communicate to form several I/O paths simultaneously. 

Use google c++ style guide for the implementation. Implement a draft of chimaera. Implement most code in the source files rather than headers. Ensure you document each function in the files you create. Do not make markdown files for this initially, just direct comments in C++. Use the namespace chi:: for all core chimaera types.

## CMake specifiction
Create CMake export targets so that external libraries can include chimaera and build their own chimods. Use RPATH and enable CMAKE_EXPORT_COMPILE_COMMANDS for building all chimaera objects. Ensure to find Hermes SHM (HSHM) and boost.

The root CMakeLists.txt should read environment variables from .env.cmake. This should be enabled/disabled using an option CHIMAERA_ENABLE_CMAKE_DOTENV. Make sure to always use this option when compiling this code. You will need it to find the packages for boost and hshm.

Struct cmakes into at least 5 sections: 
1. options 
2. compiler optimization. Have modes for debug and release. Debug should have no optimization (e.g., -O0).
3. find_package
4. source compilation. E.g., add_subdirectory, etc.
5. install code

At a high level, the project should have a src and include directory, and a CMakeLists.txt in the root of the project.

There should be a compiler macro called CHIMAERA_RUNTIME set to 1 for runtime code objects and 0 for client code objects.

## Pools and Domains

Pools represent a group of containers. Containers process tasks. Each container has a unique ID in the pool starting from 0. A SubDomain represents a named subset of containers in the pool. A SubDomainId represents a unique address of the container within the pool. A DomainId represents a unique address of the container in the entire system.  The following SubDomains should be provided: 
```cpp
/** Major identifier of subdomain */
typedef u32 SubDomainGroup;

/** Minor identifier of subdomain */
typedef u32 SubDomainMinor;

namespace SubDomain {
// Maps to an IP address of a node
static GLOBAL_CROSS_CONST SubDomainGroup kPhysicalNode = 0;
// Maps to a logical address global to the entire pool
static GLOBAL_CROSS_CONST SubDomainGroup kGlobal = 1
// Maps to a logical adress local to this node
static GLOBAL_CROSS_CONST SubDomainGroup kLocal = 2;
} // namespace SubDomain
// NOTE: we avoid using a class and static variables for SubDomain for GPU compatability. CUDA does not support static class variables.

struct SubDomainId {
  SubDomainGroup major_; /**< NodeSet, ContainerSet, ... */
  SubDomainMinor minor_; /**< NodeId, ContainerId, ... */
}

/** Represents a scoped domain */
struct DomainId {
  PoolId pool_id_;
  SubDomainId sub_id_;
}
```

A DomainQuery should be implemented that can be used for selecting basic regions of a domain. DomainQuery is not like a SQL query and should focus on being small in size and avoiding strings. DomainQuery has the following options:
1. LocalId(u32 id): Send task to container using its local address
2. GetGlobalId(u32 id): Send task to container using its global address 
3. LocalHash(u32 hash): Hash task to a container by taking modulo of the kLocal subdomain
4. GetGlobalHash(u32 hash): Hash task to a container by taking module of the kGlobal subdomain
5. GetGlobalBcast(): Replicates task to every node in the domain
5. GetDynamic(): Send this request to the container's Monitor method with MonitorMode kGlobalSchedule

Containers can internally create a set of concurrent queues for accepting requests. Queues have an ID. Lanes of these queues will be scheduled within the runtime when they have tasks to execute. The queues will be based on the multi_mpsc_queue data structure of hshm.

## The Base Task

Tasks are used to communicate with containers and pools. Tasks are like RPCs. They contain a DomainQuery to determine which pool and containers to send the task, they contain a method identifier, and any parameters to the method they should execute. There is a base task data structure that all specific tasks inherit from. At minimum, tasks look as follows:
```cpp
/** Decorator macros */
#define IN  // This is an input by the client
#define OUT  // This is output by the runtime
#define INOUT  // This is both an input and output
#define TEMP  // This is internally used by the runtime or client.

/** A container method to execute + parameters */
struct Task {
public:
  IN PoolId pool_id_;        /**< The unique ID of a pool */
  IN TaskNode task_node_;    /**< The unique ID of this task in the graph */
  IN DomainQuery pool_query_; /**< The nodes that the task should run on */
  IN MethodId method_;       /**< The method to call in the container */
  IN ibitfield task_flags_;  /**< Properties of the task */
  IN double period_ns_;      /**< The period of the task */ 

  Task(const hipc::CtxAllocator<AllocT> &alloc) {}

  void Wait();  // Wait for this task to complete
  void Wait(Task *subtask);  // Wait for a subtask to complete
  template <typename TaskT>
  HSHM_INLINE void Wait(std::vector<FullPtr<TaskT>> &subtasks);  // Wait for subtasks to complete
}
```

Tasks can have the following properties (task_flags_):
1. TASK_PERIODIC: This task will execute periodically. If this is not set, then the task is executed exactly once.
2. TASK_FIRE_AND_FORGET: This task has no return result and should be freed by the runtime upon completion.

TaskNode is the unique ID of a task in a task graph. I.e., if a task spawns a subtask, they should have the same major id, but different minors. Since tasks are stored in shared memory, they should never use virtual functions. 

An example task for compression is as follows: 
```cpp 
/** The CompressTask task */
struct CompressTask : public chi::Task {
  /** SHM default constructor */
  explicit CompressTask(
      const hipc::CtxAllocator<CHI_ALLOC_T> &alloc)
      : chi::Task(alloc) {}

  /** Emplace constructor */
  explicit CompressTask(
      const hipc::CtxAllocator<CHI_ALLOC_T> &alloc, const chi::TaskNode &task_node,
      const chi::PoolId &pool_id, const chi::DomainQuery &pool_query)
      : chi::Task(alloc) {
    // Initialize task
    task_node_ = task_node;
    pool_ = pool_id;
    method_ = Method::kCompress;
    task_flags_.SetBits(0);
    pool_query_ = pool_query;

    // Custom
  }
}; 
```

## The Runtime

The runtime implements an intelligent, multi-threaded task execution system. The runtime read the environment variable CHI_SERVER_CONF to see the server configuration yaml file, which stores all configurations for the runtime. There should be a Configration parser that inherits from Hermes SHM's BaseConfig.

Make a default configuration in the config directory. Turn this config into a C++ constant and place into a header file. Use LoadYaml to read the constant and get default values.

### Initialization

Create a new class called Chimaera with methods for unified initialization in include/chimaera/chimaera.h. Make a singleton using hshm for this class. Implement the CHIMAERA_INIT method in the created source file, which takes a ChimaeraMode enum (kClient, kServer/kRuntime) and an optional boolean for starting an embedded runtime.

### Configuration Manager
Make a singleton using hshm for this class. The configuration manager is responsible for parsing the chimaera server YAML file. A singleton should be made so that subsequent classes can access the config data. This class should inherit from the BaseConfig from hshm.

### IPC Manager
Make a singleton using hshm for this class. It implements a ClientInit and ServerInit method. The IPC manager should be different for client and runtime. The runtime should create shared memory segments, while clients load the segments.

For ServerInit, when the runtime initially starts, it must spawn a ZeroMQ server using the local loopback address. Use lightbeam from hshm for this. Clients can use this to detect a client on this node is executing and initially connect to the server. 

After this, shared memory backends and allocators over those backends are created. There should be three memory segments:
* main: allocates tasks shared by client and runtime
* client_data: allocates data shared by clients and runtime
* runtime_data: allocates data internally shared by runtime and clients

 The allocator used should be the following compiler macros:
 * ``CHI_MAIN_ALLOC_T``. The default value should be ``hipc::ThreadLocalAllocator``.  Another macro CHI_ALLOC_T that maps to this.
 * ``CHI_CDATA_ALLOC_T``. The default value should be ``hipc::ThreadLocalAllocator``.  
 * ``CHI_RDATA_ALLOC_T``. The default value should be ``hipc::ThreadLocalAllocator``.  

After this, a concurrent, priority queue named the process_queue is stored in the shared memory. This queue is for external processes to submit tasks to the runtime. The number of lanes (i.e., concurrency) is determined by the number of workers. There should be the following priorities: kLowLatency and kHighLatency. The queue lanes are implemented on top of multi_mpsc_queue from hshm. The queue should store a ``hipc::Pointer`` instead of a ``hipc::FullPtr``. This is because FullPtr stores both private and shared memory addresses, but the private address will not be correct at the runtime. The depth of the queue is configurable. It does not necessarily need to be a simple typedef.

The chimaera configuration should include an entry for specifying the hostfile. ``hshm::ConfigParse::ParseHostfile`` should be used to load the set of hosts. In the runtime, the IPC manager reads this hostfile. It attempts to spawn a ZeroMQ server for each ip address. On the first success, it stops trying. The offset in this list + 1 is the ID of this node.

The IPC manager should expose functions for allocating tasks and freeing them.
```cpp
class IpcManager {
  public:
    void ClientInit();
    void ServerInit();

    // Allocate task using main allocator
    template<typename TaskT, typename ...Args>
    hipc::FullPtr<TaskT> NewTask(const hipc::MemContext &ctx, Args &&...args) {
        return main_alloc_->NewObj<TaskT>(mctx, std::forward<Args>(args)...);
    }

    // Delete tasks using main allocator
    template<typename TaskT>
    void DelTask(const hipc::MemContext &ctx, hipc::FullPtr<TaskT> task);

    // Allocate task using cdata if CHIMAERA_RUNTIME not set, and rdata otherwise.
    FullPtr<char> AllocateBuffer();
}
```

### Module Manager
Make a singleton using hshm for this class. The module manager is responsible for loading modules for hshm. This class should be essentially empty for now. We will discuss details later.

### Pool Manager
Should maintain the set of ChiPools and ChiContainers on this node. A table should be stored mapping a ChiPool id to the ChiContainers it has on this node. Should be ways to get the chipool name from id quickly, etc. For now, typedef chicontainers to void. We will discuss chimod details later.

### Work Orchestrator
Make a work orchestrator class and singleton. It will spawn a configurable number of worker threads. There four types of worker threads:
1. Low latency: threads that execute only low-latency lanes. This includes lanes from the process queue. 
2. High latency: threads that execute only high-latency lanes.
3. Reinforcement: threads dedicated to the reinforcement of ChiMod performance models
4. Process Wreaper: detects when a process has died and frees its associated memory. For now, do not implement

Use ``HSHM_THREAD_MODEL->Spawn`` for spawning the threads.

When initially spawning the workers, the work orchestrator must also initially map the queues from the IPC Manager to each worker. It maps low-latency lanes to a subset of workers and then high-latency lanes to a different subset of workers. 

### Worker
Low-latency and high-latency workers iterate over a set of lanes and execute tasks from those lanes. Workers store an active lane queue and a cold lane queue. The active queue stores the set of lanes to iterate over. The cold queue stores lanes this worker is responsible for, but do not currently have activity.

When the worker executes a task, it must do the following:
1. Pop task from a lane
2. Resolve the domain query of the task. I.e., identify the exact set of nodes to distribute the task to. For now, this should assume all queries resolve to local.
3. Create a ``RunContext`` for the task, representing all state needed by the runtime for executing the task. This can include timers for periodic scheduling, boost fiber context, and anything else needed that shouldn't be stored in shared memory for the task. 
4. Allocate a stack space (64KB) and initiate ``boost::fiber``. This should be state apart of the ``RunContext``. For now, the function that executes the task should be empty. We will flesh out its details later.

## Utilities

Implement an executable to launch and stop the runtime: chimeara_start_runtime and chimaera_stop_runtime.
