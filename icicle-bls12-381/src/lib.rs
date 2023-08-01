use icicle_core::tst::*;
use icicle_core::impl_do_smth;


const NUM_LIMBS: usize = 12;

pub struct ScalarField {
    pub repr: Limbs<NUM_LIMBS>,
}

impl Field<NUM_LIMBS> for ScalarField {
    fn to_repr(&self) -> Limbs<NUM_LIMBS> {
        self.repr
    }

    fn from_repr(repr: Limbs<NUM_LIMBS>) -> Self {
        ScalarField { repr: repr }
    }
}

impl_do_smth!(bls12_381, ScalarField);
