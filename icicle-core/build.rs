use std::env;

fn main() {
    //TODO: check cargo features selected
    //TODO: can conflict/duplicate with make ?

    // println!("cargo:rerun-if-env-changed=CXXFLAGS");
    // println!("cargo:rerun-if-changed=./icicle");

    // let arch_type = env::var("ARCH_TYPE").unwrap_or(String::from("native"));

    // let mut arch = String::from("-arch=");
    // arch.push_str(&arch_type);

    // let mut nvcc = cc::Build::new();

    // println!("Compiling icicle library using arch: {}", &arch);

    // if cfg!(feature = "g2") {
    //     nvcc.define("G2_DEFINED", None);
    // }
    // if cfg!(feature = "bn254") {
    //     nvcc.define("BN254", None);
    // } else if cfg!(feature = "bls12_381") {
    //     nvcc.define("BLS12_381", None);
    // }
    // nvcc.cuda(true);
    // nvcc.debug(false);
    // nvcc.flag(&arch);
    // nvcc.files([
    //     "./icicle/appUtils/vector_manipulation/ve_mod_mult.cu",
    //     "./icicle/appUtils/ntt/lde.cu",
    //     "./icicle/appUtils/msm/msm.cu",
    //     "./icicle/appUtils/smth.cu",
    //     "./icicle/primitives/projective.cu",
    // ]);
    // nvcc.compile("ingo_icicle"); //TODO: extension??
}
