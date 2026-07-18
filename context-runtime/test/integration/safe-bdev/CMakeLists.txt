# Safe-BDev interactive integration test.
#
# Launches (in one container) the clio runtime + safe-bdev + CTE + clio-fs
# (FUSE) + the context-visualizer dashboard, then -- in smoke mode -- scripts a
# member remove+replace and asserts recovery progressed.
#
# Requires: Docker with SYS_ADMIN + /dev/fuse, a build with
# -DCLIO_CTE_ENABLE_FUSE_ADAPTER=ON, and a container image carrying fuse3 +
# flask (iowarp/cte-xfstests). Registered only under CLIO_CORE_ENABLE_DOCKER_CI.

set(SAFE_BDEV_IT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_test(
    NAME cr_safe_bdev_integration_docker
    COMMAND ${CMAKE_COMMAND} -E env MODE=smoke ${SAFE_BDEV_IT_DIR}/run_tests.sh
    WORKING_DIRECTORY ${SAFE_BDEV_IT_DIR}
)

set_tests_properties(cr_safe_bdev_integration_docker PROPERTIES
    LABELS "integration;docker;safe-bdev;fuse"
    TIMEOUT 600
)

message(STATUS "Safe-BDev integration test configured (ctest -R cr_safe_bdev_integration_docker)")
message(STATUS "  Interactive run: ${SAFE_BDEV_IT_DIR}/run_tests.sh")
