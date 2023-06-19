use std::env;

fn main() {
    //TODO: check cargo features selected
    //TODO: can conflict/duplicate with make ?
    println!("cargo:rerun-if-env-changed=CXXFLAGS");

    let arch_type = env::var("ARCH_TYPE").unwrap_or(String::from("native"));

    let mut arch = String::from("-arch=");
    arch.push_str(&arch_type);

    let mut nvcc = cc::Build::new();

    println!("Compiling icicle library using arch: {}", &arch);
    nvcc.define("BN254", None);
    nvcc.cuda(true);
    nvcc.debug(false);
    nvcc.flag(&arch);
    nvcc.files([
        "../icicle-core/icicle/primitives/projective.cu",
        "../icicle-core/icicle/appUtils/smth.cu",
    ]);
    nvcc.compile("ingo_bn254"); //TODO: extension??
}