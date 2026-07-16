// SPDX-License-Identifier: BSD-3-Clause
// Build script for ctp-coroutine (issue #756).
//
// The default (stackless async) backend is pure Rust — nothing to build.
// With `--features boost-fibers`, compile the Boost.Context shim and link
// libboost_context. Boost is located via, in order:
//   BOOST_INCLUDE_DIR / BOOST_LIB_DIR   (explicit split paths)
//   BOOST_ROOT                          (expects include/ and lib/ beneath)
//   system default include/lib paths    (e.g. the devcontainer's /usr)

fn main() {
    println!("cargo:rerun-if-changed=shim/boost_fiber_shim.cc");
    println!("cargo:rerun-if-env-changed=BOOST_ROOT");
    println!("cargo:rerun-if-env-changed=BOOST_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=BOOST_LIB_DIR");

    if std::env::var_os("CARGO_FEATURE_BOOST_FIBERS").is_none() {
        return;
    }

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .file("shim/boost_fiber_shim.cc");

    if let Ok(inc) = std::env::var("BOOST_INCLUDE_DIR") {
        build.include(inc);
    } else if let Ok(root) = std::env::var("BOOST_ROOT") {
        build.include(format!("{root}/include"));
    }

    build.compile("ctp_boost_fiber_shim");

    if let Ok(lib) = std::env::var("BOOST_LIB_DIR") {
        println!("cargo:rustc-link-search=native={lib}");
    } else if let Ok(root) = std::env::var("BOOST_ROOT") {
        println!("cargo:rustc-link-search=native={root}/lib");
    }
    println!("cargo:rustc-link-lib=boost_context");
}
