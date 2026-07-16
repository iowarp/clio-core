// SPDX-License-Identifier: BSD-3-Clause
// Build script for ctp-gpu (issue #756).
//
// With `--features cuda`, link the CUDA driver API and NVRTC. Kernels are
// JIT-compiled by NVRTC at runtime, so no nvcc / host C++ compiler is
// needed here — only the toolkit's import libraries. Locations searched:
//   CUDA_PATH (Windows default install sets this) → lib/x64 | lib64
//   CUDA_HOME                                     → lib64
//   /usr/local/cuda                               → lib64 (+ stubs for
//                                                   driver-less build hosts)

fn main() {
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");

    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    let mut roots: Vec<String> = Vec::new();
    for var in ["CUDA_PATH", "CUDA_HOME"] {
        if let Ok(v) = std::env::var(var) {
            roots.push(v);
        }
    }
    roots.push("/usr/local/cuda".to_string());

    for root in &roots {
        for sub in ["lib/x64", "lib64", "lib64/stubs", "lib/stubs"] {
            let p = format!("{root}/{sub}");
            if std::path::Path::new(&p).exists() {
                println!("cargo:rustc-link-search=native={p}");
            }
        }
    }

    println!("cargo:rustc-link-lib=cuda"); // driver API (nvcuda.dll / libcuda.so)
    println!("cargo:rustc-link-lib=nvrtc");
}
