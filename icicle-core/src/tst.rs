pub use paste::paste;


#[derive(Debug, PartialEq, Copy, Clone)]
#[repr(C)]
pub struct Limbs<const NUM_LIMBS: usize> {
    pub limbs: [u32; NUM_LIMBS],
}

impl<const NUM_LIMBS: usize> Limbs<NUM_LIMBS> {
    pub fn zero() -> Self {
        Limbs {
            limbs: [0u32; NUM_LIMBS],
        }
    }
}

pub trait Field<const NUM_LIMBS: usize> where Self: Sized {
    fn to_repr(&self) -> Limbs<NUM_LIMBS>;
    fn from_repr(repr: Limbs<NUM_LIMBS>) -> Self;

    fn zero() -> Self {
        Self::from_repr(Limbs::<NUM_LIMBS>::zero())
    }
}

#[macro_export]
macro_rules! impl_do_smth {
    ($curve:ident, $scalar_type:ident) => {
        paste! {
            extern "C" {
                fn [<$curve _do_smth>](scalar: *mut std::ffi::c_void) -> i32;
            }

            pub fn do_smth(values: &mut [$scalar_type]) -> i32 {
                let ret_code = unsafe {
                    [<$curve _do_smth>](
                        values as *mut _ as *mut std::ffi::c_void,
                    )
                };
                ret_code
            }
        }
    }
}

pub fn build_children() {
    
}
