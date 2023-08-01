use std::env::var;
use std::fs;
use std::{env, path::Path, process::Command};

fn main() {
    //TODO: check cargo features selected
    //TODO: can conflict/duplicate with make ?
    println!("cargo:rerun-if-env-changed=CXXFLAGS");
    println!("cargo:rerun-if-changed=../icicle-core");

    let arch_type = env::var("ARCH_TYPE").unwrap_or(String::from("native"));
    let files = vec![
        "../icicle-core/icicle/primitives/projective.cu",
        "../icicle-core/icicle/appUtils/smth.cu",
    ];

    let mut object_files = vec![];

    let out_dir = var("OUT_DIR").unwrap();
    let profile = var("PROFILE").unwrap();

    // TODO: what's with this weirdness?
    let target_output_dir = format!("../target/{}", profile);

    for file in files {
        let path = Path::new(file);
        let obj_file = format!("{}/{:?}", out_dir, path.file_name().unwrap());

        let status = Command::new("nvcc")
            .arg("-DCURVE=12381")
            .arg("-c") // Compile but don't link
            .arg(format!("-arch={}", &arch_type))
            .arg(file)
            .arg("-o")
            .arg(&obj_file)
            .status()
            .expect("Failed to execute nvcc command");

        if obj_file.as_str().contains("smth.cu") {
            // Prefix symbols
            let output = Command::new("objcopy")
                // .arg("--prefix-symbol=bls12_381_")
                .arg("--redefine-sym")
                .arg("do_smth=bls12_381_do_smth")
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

        if status.success() {
            println!("nvcc command executed successfully");
            object_files.push(obj_file.to_owned());
        } else {
            eprintln!("nvcc command failed");
            std::process::exit(1);
        }
    }

    // Link object files into one static library
    let output_file = format!("{}/libingo_bls12_381.a", target_output_dir);
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

    println!("cargo:rustc-link-lib=ingo_bls12_381");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=cudart");
}
