use icicle_core::tst::*;

mod bn254 {
    #[link(name = "ingo_bn254")]
    extern "C" {
        pub fn bn254_do_smth(scalar: *mut std::ffi::c_void) -> i32;
    }
}

pub fn do_smth_rust<T>(values: &mut [T]) -> i32 {
    let ret_code = unsafe { bn254::bn254_do_smth(values as *mut _ as *mut std::ffi::c_void) };
    ret_code
}

pub type ScalarField = FF<8>;
pub fn do_smth_254(values: &mut [ScalarField]) -> i32 {
    do_smth_rust(values)
}

#[test]
fn do_smth_test() {
    let mut values1 = [ScalarField::zero()];
    println!("Did smth 254: {:?}", do_smth_254(&mut values1[..]));
}
