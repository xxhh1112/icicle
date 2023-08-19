use std::env::var;
use std::fs;
use std::{env, path::Path, process::Command};
use cmake::Config;

fn main() {
    //TODO: check cargo features selected
    //TODO: can conflict/duplicate with make ?
    println!("cargo:rerun-if-env-changed=CXXFLAGS");
    println!("cargo:rerun-if-changed=./icicle");

    // let arch_type = env::var("ARCH_TYPE").unwrap_or(String::from("native"));
    let files = vec![
        "../icicle-core/icicle/primitives/projective.cu",
        "../icicle-core/icicle/appUtils/vec_subtract.cu",
    ];

    let mut object_files = vec![];

    let out_dir = var("OUT_DIR").unwrap();
    let profile = var("PROFILE").unwrap();

    let target_output_dir = format!("../target/{}", profile);

    let dst = Config::new("../icicle-core/icicle")
                // .define("BUILD_TESTS", "OFF")
                .define("CURVE", "254")
                .build_target("subtract")
                .build();
    println!("cargo:rustc-link-search=native={}", dst.display());

    for file in files {
        let path = Path::new(file);
        let obj_file = format!("{}/{:?}", out_dir, path.file_name().unwrap());

        // println!("Compiling {} to {}", file, obj_file);
        // let status = Command::new("nvcc")
        //     .arg("-DCURVE=254")
        //     .arg("-c") // Compile but don't link
        //     .arg(format!("-arch={}", &arch_type))
        //     .arg(file)
        //     .arg("-o")
        //     .arg(&obj_file)
        //     .status()
        //     .expect("Failed to execute nvcc command");

        if obj_file.as_str().contains("vec_subtract.cu") {
            // println!("here?");
            // Prefix symbols
            let output = Command::new("objcopy")
                // .arg("--prefix-symbol=bn254_")
                .arg("--redefine-sym")
                .arg("subtract=bn254_subtract")
                .arg(&obj_file)
                .output()
                .expect("Failed to execute objcopy command");

            if output.status.success() {
                println!("objcopy command executed successfully");
            } else {
                let stderr = String::from_utf8_lossy(&output.stderr);
                eprintln!("objcopy command failed:\n{}", stderr);
                panic!();
            }
        }

        println!("nvcc command executed successfully");
        object_files.push(obj_file.to_owned());
    }

    // Link object files into one static library
    let output_file = format!("{}/libingo_bn254.a", target_output_dir);
    let status = Command::new("ar")
        .arg("rcs")
        .arg(output_file)
        .args(object_files.iter().map(|p| p))
        .status()
        .expect("Failed to execute ar command");

    if status.success() {
        println!("ar command executed successfully");
    } else {
        eprintln!("ar command failed");
        std::process::exit(1);
    }

    // Remove object files
    for file in object_files {
        fs::remove_file(file).expect("Failed to remove object file");
    }

    println!("cargo:rustc-link-lib=ingo_bn254");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=cudart");
}
