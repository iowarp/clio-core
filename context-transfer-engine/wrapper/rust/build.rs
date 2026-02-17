fn main() {
    cxx_build::bridge("src/lib.rs")
        .file("shim/shim.cc")
        .std("c++20")
        // Include paths
        .include("/usr/local/include")
        .include("/home/iowarp/miniconda3/include") // yaml-cpp, cereal, etc.
        .include(".") // for "shim/shim.h"
        // Coroutine support
        .flag("-fcoroutines")
        // Suppress warnings from CTE/chimaera headers
        .flag("-Wno-unused-parameter")
        .flag("-Wno-unused-variable")
        .flag("-Wno-missing-field-initializers")
        .flag("-Wno-sign-compare")
        .flag("-Wno-reorder")
        .flag("-Wno-pedantic")
        // HSHM / chimaera defines (match CMake build)
        .define("HSHM_COMPILER_GNU", "1")
        .define("HSHM_COMPILER_MSVC", "0")
        .define("HSHM_DEBUG_LOCK", "0")
        .define("HSHM_DEFAULT_ALLOC_T", "hipc::ThreadLocalAllocator")
        .define("HSHM_DEFAULT_THREAD_MODEL", "hshm::thread::Pthread")
        .define("HSHM_DEFAULT_THREAD_MODEL_GPU", "hshm::thread::Cuda")
        .define("HSHM_ENABLE_CEREAL", "1")
        .define("HSHM_ENABLE_DLL_EXPORT", "1")
        .define("HSHM_ENABLE_DOXYGEN", "0")
        .define("HSHM_ENABLE_LIBFABRIC", "0")
        .define("HSHM_ENABLE_LIGHTBEAM", "1")
        .define("HSHM_ENABLE_OPENMP", "0")
        .define("HSHM_ENABLE_PROCFS_SYSINFO", "1")
        .define("HSHM_ENABLE_PTHREADS", "1")
        .define("HSHM_ENABLE_THALLIUM", "0")
        .define("HSHM_ENABLE_WINDOWS_SYSINFO", "0")
        .define("HSHM_ENABLE_WINDOWS_THREADS", "0")
        .define("HSHM_ENABLE_ZMQ", "1")
        .define("HSHM_LOG_LEVEL", "0")
        .compile("cte_shim");

    println!("cargo:rustc-link-search=native=/usr/local/lib");
    println!("cargo:rustc-link-search=native=/home/iowarp/miniconda3/lib");

    // Direct dependency
    println!("cargo:rustc-link-lib=dylib=wrp_cte_core_client");
    // Transitive deps (needed for test binary linking)
    println!("cargo:rustc-link-lib=dylib=chimaera_cxx");
    println!("cargo:rustc-link-lib=dylib=hermes_shm_host");
    println!("cargo:rustc-link-lib=dylib=zmq");

    println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/local/lib");
    println!("cargo:rustc-link-arg=-Wl,-rpath,/home/iowarp/miniconda3/lib");
    println!("cargo:rerun-if-changed=shim/shim.h");
    println!("cargo:rerun-if-changed=shim/shim.cc");
}
