# Benchmark directory CMakeLists.txt
cmake_minimum_required(VERSION 3.20)

# Create clio_cte_bench executable
add_executable(clio_cte_bench
  clio_cte_bench.cc
)

# Link required libraries (MPI no longer required - uses worker threads instead)
target_link_libraries(clio_cte_bench
  clio::cte::core_client
  clio::run::cxx
)

# Set C++ standard
target_compile_features(clio_cte_bench PRIVATE cxx_std_17)

# Install the benchmark executable
install(TARGETS clio_cte_bench
  RUNTIME DESTINATION bin
)

# Create clio_cte_semantic_bench executable — ingests N keyword blobs under a
# single tag, then runs one broadcast SemanticSearch over them.
add_executable(clio_cte_semantic_bench
  clio_cte_semantic_bench.cc
)
target_link_libraries(clio_cte_semantic_bench
  clio::cte::core_client
  clio::run::cxx
)
target_compile_features(clio_cte_semantic_bench PRIVATE cxx_std_17)
install(TARGETS clio_cte_semantic_bench
  RUNTIME DESTINATION bin
)

# Create clio_cte_score_bench executable (MPI-based score/demotion benchmark)
# Only build if MPI is enabled
if(CLIO_CORE_ENABLE_MPI)
  add_executable(clio_cte_score_bench
    clio_cte_score_bench.cc
  )

  # Link required libraries including MPI
  target_link_libraries(clio_cte_score_bench
    clio::cte::core_client
    clio::run::cxx
    MPI::MPI_CXX
  )

  # Set C++ standard
  target_compile_features(clio_cte_score_bench PRIVATE cxx_std_17)

  # Install the score benchmark executable
  install(TARGETS clio_cte_score_bench
    RUNTIME DESTINATION bin
  )
else()
  message(STATUS "Skipping clio_cte_score_bench - MPI not enabled (set CLIO_CORE_ENABLE_MPI=ON)")
endif()

# Redis comparison benchmark — apples-to-apples mirror of clio_cte_bench
# (shares bench_common.h). Built only when CLIO_CORE_ENABLE_REDIS=ON, which
# locates hiredis and sets CLIO_REDIS_LIBS (see top-level CMakeLists.txt).
if(CLIO_CORE_ENABLE_REDIS)
  add_executable(clio_redis_bench
    clio_redis_bench.cc
  )
  target_link_libraries(clio_redis_bench
    ${CLIO_REDIS_LIBS}
    ctp::cxx
  )
  target_compile_features(clio_redis_bench PRIVATE cxx_std_17)
  install(TARGETS clio_redis_bench
    RUNTIME DESTINATION bin
  )
else()
  message(STATUS "Skipping clio_redis_bench - Redis not enabled "
                 "(set CLIO_CORE_ENABLE_REDIS=ON; requires hiredis)")
endif()

# NOTE: Compression benchmark (clio_cte_compress_bench) has been moved to
# context-transfer-engine/compressor/test/

# YCSB JNI binding (issue #862): thin shim behind site.ycsb.db.ClioClient.
# Built only when JNI headers are present so plain builds don't grow a Java
# dependency; the Java half lives in benchmark/ycsb/ClioClient.java and is
# compiled with javac against the YCSB core jar at benchmark time.
# find_path instead of find_package(JNI): FindJNI refuses headless JDKs
# (no AWT), and a loadable JNI library needs only jni.h — the JVM resolves
# its own symbols at System.loadLibrary time.
find_path(CLIO_JNI_INCLUDE_DIR jni.h
  HINTS $ENV{JAVA_HOME}/include /usr/lib/jvm/default-java/include)
# NOT WIN32: InitOnce() installs a no-op SIGUSR1 sigaction to survive the
# Linux SHM transport's tgkill wakeups — POSIX-signal machinery with no
# Windows counterpart (and no Windows SHM-wake path to survive). Windows
# runners ship a JDK, so without this gate every Windows build that finds
# jni.h dies compiling sigaction/SIGUSR1 (first seen when the packaging
# workflows ran against dev content in #899).
if(CLIO_JNI_INCLUDE_DIR AND NOT WIN32)
  message(STATUS "YCSB JNI binding enabled (jni.h at ${CLIO_JNI_INCLUDE_DIR})")
  add_library(clio_ycsb_jni SHARED
    ycsb/clio_ycsb_jni.cc
  )
  target_include_directories(clio_ycsb_jni PRIVATE
    ${CLIO_JNI_INCLUDE_DIR}
    # jni.h includes the per-platform jni_md.h from a subdirectory; listing
    # all three is harmless (nonexistent include dirs are ignored).
    ${CLIO_JNI_INCLUDE_DIR}/linux
    ${CLIO_JNI_INCLUDE_DIR}/darwin
    ${CLIO_JNI_INCLUDE_DIR}/win32)
  target_link_libraries(clio_ycsb_jni
    clio::cte::core_client
    clio::run::cxx
  )
  target_compile_features(clio_ycsb_jni PRIVATE cxx_std_17)
  install(TARGETS clio_ycsb_jni
    LIBRARY DESTINATION bin
  )
endif()
