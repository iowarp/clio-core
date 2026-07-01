# thallium-bench: raw libthallium async-RPC microbenchmark (issue #647).
#
# Built only when CLIO_CORE_ENABLE_THALLIUM=ON. Uses thallium directly (not the
# ctp::lbm transport) over the na+sm shared-memory transport to measure async
# RPC IOPS and latency.

add_executable(thallium_bench thallium_bench.cc)

# `thallium` is the imported target provided by find_package(thallium) in the
# root CMakeLists.txt (guarded by CLIO_CORE_ENABLE_THALLIUM).
target_link_libraries(thallium_bench PRIVATE thallium)

install(TARGETS thallium_bench DESTINATION bin)
