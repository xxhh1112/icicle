use icicle_core::tst::*;

extern "C" {
    fn do_smth1(scalar: *mut std::ffi::c_void) -> i32;
}

pub fn do_smth_rust<T>(values: &mut [T]) -> i32 {
    let ret_code = unsafe {
        do_smth1(
            values as *mut _ as *mut std::ffi::c_void,
        )
    };
    ret_code
}

pub type ScalarField = FF<12>;
pub fn do_smth_381(values: &mut [ScalarField]) -> i32 {
    do_smth_rust(values)
}