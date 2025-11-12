@CLAUDE.md

I have the following repos under the directory ${IOWARP} on this system: 
1. cte-hermes-shm
2. iowarp-runtime
3. content-transfer-engine
4. content-assimilation-engine
5. context-exploration-interface

I want to bring them all together in this repo as follows:
1. Copy paste all 4 repos as subdirectories. Rename them as follows: 
  * cte-hermes-shm -> context-transport-primitives. 
  * iowarp-runtime -> runtime
  * content-transfer-engine -> context-transfer-engine
  * content-assimilation-engine -> context-assimilation-engine
  * context-exploration-interface -> context-exploration-engine
2. Create a unfied CLAUDE.md based on each of the sub-repo claude files.
In addition, let's copy the agents from context-transfer-engine into our
main directory.
3. Create a root CMakeLists.txt in this repo linking all of them together.
Its project should be something like iowarp-core. We should have options
for disabling each of the components. So options in the format:
WRP_CORE_ENABLE_RUNTIME
WRP_CORE_ENABLE_CTE
WRP_CORE_ENABLE_CAE
WRP_CORE_ENABLE_CEE
4. Use the cte-hermes-shm .devcontainer as the root devcontainer. Delete
all others. This does not need modification in any way.
5. Create a single docker subdirectory in the root. Copy the cte-hermes-shm
dockerfiles folder for this first. Make it so the shell scripts produce iowarp/core-build:latest
and iowarp/core:latest. Then look at the others to see if they have subdirectories in docker folder.
6. Ensure the correctness of all dockerfiles in the unit test directories in
each of the sub-repos. Ensure we do not use iowarp/iowarp:latest in the containers.
Instead we should use iowarp/core-build:latest.
7. Create unified github actions. Really the only action of interest is
the build docker action present in each of the repos. 
8. Build a unified gitignore based on the subdirectories
9. Ensure we add the proper submodules that the other repos added. Mainly nanobind.
10. Ensure that each subdirectory we have now created are no longer their own githubs.
11. Remove each subdirecotry .claude, .github. Unify the subdirectory .vscode directories.
Create a unified cpp lint and clangd. Remove .env and .env.cmake. Remove env.sh. Migrate
LICENSE to the root repo. Remove from each of the subdirectories afterward. Create unified
CMakePresets in the root directory and remove from subdirectories afterwords. 

We will ensure everything compiles later. 