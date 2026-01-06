@CLAUDE.md Build a jarvis package for deploying this repo. Read @docs/jarvis/package_dev_guide.md
to see how. Create the jarvis repo in a new directory test/jarvis_iowarp. 

## wrp_runtime

A Service type package. Contains all parameters necessary to build the chimaera configuration.

The path to the generated chimaera configuration should be stored in the environment variable RuntimeInit and ClientInit check. Store in the shared directory.
Check to see what the real environment variables are. Check the config directory to see example configurations. Generate configurations during the _configure method.


Assume that chimaera has been installed. Do not require users to pass in specific file paths.
Place configurations in the shared_dir.

Use PsshExecInfo to launch the runtime on all nodes in the provided hostfile. Use env for the 
environment, not mod_env.
