// #[cfg(any(feature = "bls12_381", feature = "bls12_377"))]
// const BASE_LIMBS_: usize = 12;
// #[cfg(feature = "bn254")]
// const BASE_LIMBS_: usize = 8;

#[derive(Debug, PartialEq, Copy, Clone)]
#[repr(C)]
pub struct FF<const NUM_LIMBS: usize> {
    pub s: [u32; NUM_LIMBS],
}

impl<const NUM_LIMBS: usize> FF<NUM_LIMBS> {
    pub fn zero() -> Self {
        FF {
            s: [0u32; NUM_LIMBS],
        }
    }
}

// pub type BaseField_ = FF<BASE_LIMBS_>;

// extern "C" {
//     fn do_smth(scalar: *mut std::ffi::c_void) -> i32;
// }

// pub fn do_smth_rust<T>(values: &mut [T]) -> i32 {
//     let ret_code = unsafe {
//         do_smth(
//             values as *mut _ as *mut std::ffi::c_void,
//         )
//     };
//     ret_code
// }
