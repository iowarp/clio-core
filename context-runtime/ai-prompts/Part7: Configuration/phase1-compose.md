@CLAUDE.md 

We will add a new field to the chimaera configuration called compose.
This will allow users to spawn a set of pools, each with its
own custom configuration.

## Example compose section
```
# Worker thread configuration
workers:
  sched_threads: 4           # Scheduler worker threads (for fast tasks with EstCpuTime < 50us)
  slow_threads: 4            # Slow worker threads (for long-running tasks with EstCpuTime >= 50us)

# Memory segment configuration  
memory:
  main_segment_size: 1073741824      # 1GB
  client_data_segment_size: 536870912 # 512MB
  runtime_data_segment_size: 536870912 # 512MB

# Network configuration
networking:
  port: 5555
  neighborhood_size: 32  # Maximum number of queries when splitting range queries
  
# Logging configuration
logging:
  level: "info"
  file: "/tmp/chimaera.log"

# Runtime configuration
runtime:
  stack_size: 65536  # 64KB per task
  queue_depth: 10000
  lane_map_policy: "round_robin"  # Options: map_by_pid_tid, round_robin (default), random
  heartbeat_interval: 1000  # milliseconds

# Modules to compose
compose:
- mod_name: chimaera_bdev  # Corresponds to chimod_lib_name
  pool_name: ram://test
  pool_query: dynamic  # Either dynamic or local
  pool_id: 200.0
  capacity: 2GB
```

## Configuration parser

Add the following new classes:
```
struct ComposeConfig {
    std::vector<ModuleConfig> pools_;
}

struct PoolConfig {
    std::string mod_name_;
    std::string pool_name_;
    PoolId pool_id_;
    PoolQuery pool_query_;
    std::string config_;  // remaining yaml data
}
```

The compose section will be parsed as a list of dictionaries.
We will need to extract the mod name, pool name, pool id,
and pool query. All remaining keys in the yaml should be
stored as one big string called config_. It can also store the entire
yaml dictionary in the config_ string, if that is easier.

For PoolId, expose a function called FromString to parse the pool string.

For PoolQuery, do the same. The query should be very simple: either a check
for local or dynamic. No other cases need to be considered for this.

## BaseCreateTask

Add to the template a new parameter called ``DO_COMPOSE=false``. 
This will indicate that this task is being called from compose
and does not do extensive error checking or expect custom outputs
from CreateTask

During the constructor, set a volatile variable named do_compose_ 
if this template is true.

During GetParams, deserialize a PoolConfig and then default construct
CreateTaskT. We will need to
update all CreateParams classes to expose a LoadConfig function.
LoadConfig will take as input the PoolConfig and then use yaml-cpp
to deserialize the yaml data for the specific library to pack its
CreateParams structure. This will need to be documented in 
@docs/MODULE_DEVELOPMENT_GUIDE.md.

During SetParams(), do nothing if do_compose_ is true.

Add a new typedef for BaseCreateTask called ComposeTask.

## Compose

The admin_client.h should expose a new method
called compose. This will take as input a ComposeConfig.
It will iterate over the ComposeConfig and create
the modules one-by-one in order synchronously.
It will iteratively create and schedule a ComposeTask.
Each ComposeTask will take as input a PoolConfig so that
GetParams can later deserialize the PoolConfig.

If a module has a nonzero return code, print that
the compose failed and break. For now there is
no need to reverse. We will generally assume the
composes are correct.

## Chimaera::ServerInit

Process the compose section of the configuration
as the last step of initializing the server using
the admin client's compose Compose method.

## chimaera_compose

Build a new utility script that takes as input the
compose script. Assume the runtime is already initialized
for now, and only use CHIMAERA_INIT(chi::ChimaeraMode::kClient, false)
to start a client connection. Load the compose script using the
existing code for configuration parsing (do not build another parser)
and then call CHI_ADMIN->Compose.

## Unit test

Use unit testing agent to build a simple test case for compose.
Add it as a new test file.

It should launch both runtime and client using CHIMAERA_INIT(chi::ChimaeraMode::kClient, true).
You should build an example correct chimaera configuration
for the bdev module.
You should load that configuration and then call CHI_ADMIN->Compose
with it.

