// SPDX-License-Identifier: BSD-3-Clause
// Build script for ctp-compress (issue #756).
//
// Links the system compression libraries for whichever codec features are
// enabled. Search order mirrors ctp-gpu's discovery style:
//   CTP_COMPRESS_LIB_DIR   (explicit override)
//   CONDA_PREFIX/lib       (the devcontainer installs several codecs there)
//   /usr/lib/x86_64-linux-gnu, /usr/local/lib, /usr/lib   (system)
//   vcpkg (Windows) via VCPKG_ROOT/installed/x64-windows/lib
//
// Nothing is linked unless a codec feature is on, so a default build needs
// no compression libraries present at all.

fn feature(name: &str) -> bool {
    std::env::var_os(format!(
        "CARGO_FEATURE_{}",
        name.to_uppercase().replace('-', "_")
    ))
    .is_some()
}

fn main() {
    for v in ["CTP_COMPRESS_LIB_DIR", "CONDA_PREFIX", "VCPKG_ROOT"] {
        println!("cargo:rerun-if-env-changed={v}");
    }

    let mut dirs: Vec<String> = Vec::new();
    if let Ok(d) = std::env::var("CTP_COMPRESS_LIB_DIR") {
        dirs.push(d);
    }
    if let Ok(p) = std::env::var("CONDA_PREFIX") {
        dirs.push(format!("{p}/lib"));
    }
    if let Ok(p) = std::env::var("VCPKG_ROOT") {
        dirs.push(format!("{p}/installed/x64-windows/lib"));
    }
    dirs.extend([
        "/usr/lib/x86_64-linux-gnu".to_string(),
        "/usr/local/lib".to_string(),
        "/usr/lib".to_string(),
    ]);
    for d in &dirs {
        if std::path::Path::new(d).exists() {
            println!("cargo:rustc-link-search=native={d}");
        }
    }

    // (feature, [libs to link])
    let table: &[(&str, &[&str])] = &[
        ("zstd", &["zstd"]),
        ("lz4", &["lz4"]),
        ("zlib", &["z"]),
        ("bzip2", &["bz2"]),
        ("snappy", &["snappy"]),
        ("lzma", &["lzma"]),
        ("brotli", &["brotlienc", "brotlidec", "brotlicommon"]),
        ("blosc2", &["blosc2"]),
        ("zfp", &["zfp"]),
        ("sz3", &["SZ3c"]),
        ("fpzip", &["fpzip"]),
    ];

    let mut need_cxx = false;
    for (feat, libs) in table {
        if feature(feat) {
            for l in *libs {
                println!("cargo:rustc-link-lib={l}");
            }
            // SZ3 / ZFP / FPZIP / Blosc2 pull in C++ runtime symbols.
            if matches!(*feat, "sz3" | "fpzip") {
                need_cxx = true;
            }
        }
    }
    if need_cxx && cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=stdc++");
    }
}
