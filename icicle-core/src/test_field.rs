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

    pub fn one() -> Self {
        let mut limbs = [0u32; NUM_LIMBS];
        limbs[0] = 1;
        Limbs { limbs }
    }
}

pub trait Field<const NUM_LIMBS: usize> where Self: Sized {
    fn to_repr(&self) -> Limbs<NUM_LIMBS>;
    fn from_repr(repr: Limbs<NUM_LIMBS>) -> Self;

    fn zero() -> Self {
        Self::from_repr(Limbs::<NUM_LIMBS>::zero())
    }

    fn one() -> Self {
        Self::from_repr(Limbs::<NUM_LIMBS>::one())
    }
}

#[macro_export]
macro_rules! impl_sub {
    ($curve:ident, $scalar_type:ident) => {
        paste! {
            extern "C" {
                fn [<$curve _subtract>](in1: *const $scalar_type, in2: *const $scalar_type, res: *mut $scalar_type, n_elements: usize) -> i32;
            }

            pub fn sub(in1: &[$scalar_type], in2: &[$scalar_type]) -> Vec<$scalar_type> {
                let mut res = vec![$scalar_type::zero(); in1.len()];
                unsafe {
                    [<$curve _subtract>](
                        in1 as *const _ as *const $scalar_type,
                        in2 as *const _ as *const $scalar_type,
                        &mut res[0] as *mut _ as *mut $scalar_type,
                        in1.len(),
                    )
                };
                res
            }
        }
    }
}
