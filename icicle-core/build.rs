use std::env::var;

fn main() {
    let target_dir = var("CARGO_MANIFEST_DIR").unwrap();
    let profile = var("PROFILE").unwrap();

    let target_output_dir = format!("{}/../target/{}", target_dir, profile);
    println!("cargo:rustc-link-search={}", target_output_dir);
}
