@CLAUDE.md Build a jarvis package for configuring the CTE. Build a repo called test/jarvis_iowarp.
Check @docs/jarvis/package_development_guide.md.

## wrp_cte

This will create the iowarp CTE configuration. This is a service type package. It should contain parameters for every part of the CTE configuration. It has empty start, stop, kill implementations.

It should build the configuration in the shared_dir. It should create a correct cte configuration and set the environment variable the CTE checks for configurations.

_configure_menu at a minimum has a parameter called devices: a list of (string, capacity, score). Capacity should support suffixes. 

_configure:
1. If devices is empty from the argument dict, identify the set of all common storage from the resource graph (@docs/jarvis/resource_graph.md)
2. Build the configuration based on the arg dict
3. Save to shared_dir
4. Update the environment variable with self.setenv

start: pass

stop: pass

kill: pass

clean:
Use the Rm node with PsshExec to destroy each device. 
Ensure that during configuration, if autodetecting devices from resource graph, we append cte_target.bin to the mount point so that the bdev creates a temporary file on the mount point.
