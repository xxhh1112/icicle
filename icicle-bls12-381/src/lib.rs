use icicle_core::tst::*;

mod bn12_381 {
    #[link(name = "ingo_bls12_381")]
    extern "C" {
        pub fn bls12_381_do_smth(scalar: *mut std::ffi::c_void) -> i32;
    }
}

pub fn do_smth_rust<T>(values: &mut [T]) -> i32 {
    let ret_code = unsafe { bn12_381::bls12_381_do_smth(values as *mut _ as *mut std::ffi::c_void) };
    ret_code
}

pub type ScalarField = FF<12>;
pub fn do_smth_381(values: &mut [ScalarField]) -> i32 {
    do_smth_rust(values)
}

#[test]
fn do_smth_test() {
    let mut values1 = [ScalarField::zero()];
    println!("Did smth 381: {:?}", do_smth_381(&mut values1[..]));
}
