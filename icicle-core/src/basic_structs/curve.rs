use ark_ff::{Field as ArkField, PrimeField as ArkPrimeField};
use ark_ec::{AffineCurve as ArkAffineCurve, ProjectiveCurve as ArkProjectiveCurve};

use super::{
    scalar::FieldOps
};

pub trait Curve {
    type Base: FieldOps;
    type Scalar: FieldOps;
    type PointProjective; // Point<Self::Base, Self::Fq, Self::G1ArkAffine, Self::G1ArkProjective>;
    type PointAffine; // PointAffineNoInfinity<Self::Base, Self::Fq, Self::G1ArkAffine, Self::G1ArkProjective>;
    type Fq: ArkField;
    type Fr: ArkPrimeField;
    type G1ArkAffine: ArkAffineCurve;
    type G1ArkProjective: ArkProjectiveCurve;

    fn msm() {

    }

    fn ntt() {
        // uses Self::<type> for accessing Base and Scalar
    }
}
