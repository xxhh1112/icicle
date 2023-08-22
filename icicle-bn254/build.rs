use std::env::var;
use cmake::Config;

fn main() {
    //TODO: check cargo features selected
    println!("cargo:rerun-if-env-changed=CXXFLAGS");
    println!("cargo:rerun-if-changed=../icicle-core");

    let cargo_dir = var("CARGO_MANIFEST_DIR").unwrap();
    let profile = var("PROFILE").unwrap();

    let target_output_dir = format!("{}/../target/{}", cargo_dir, profile);

    Config::new("../icicle-core/icicle")
                .define("BUILD_TESTS", "OFF")
                .define("CURVE", "254")
                .define("CURVE_NAME", "bn254")
                .define("LIBRARY_OUTPUT_DIRECTORY", target_output_dir)
                .build_target("subtract")
                .build();

    println!("cargo:rustc-link-lib=ingo_bn254");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=cudart");
}
