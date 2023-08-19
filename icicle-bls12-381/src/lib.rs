use icicle_core::test_field::*;
use icicle_core::impl_sub;


const NUM_LIMBS: usize = 12;

#[derive(Debug, PartialEq, Copy, Clone)]
pub struct ScalarField {
    pub repr: Limbs<NUM_LIMBS>,
}

impl Field<NUM_LIMBS> for ScalarField {
    fn to_repr(&self) -> Limbs<NUM_LIMBS> {
        self.repr
    }

    fn from_repr(repr: Limbs<NUM_LIMBS>) -> Self {
        ScalarField { repr }
    }
}

impl_sub!(bls12_381, ScalarField);
