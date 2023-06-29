use ark_bn254::{
    Fq as Fq_BN254,
    Fr as Fr_BN254,
    G1Affine as G1Affine_BN254,
    G1Projective as G1Projective_BN254,
    BigInteger256 as ArkBigInteger256
};
use icicle_core::{
    Curve,
    Base, Scalar
};

pub struct BN254;

impl Curve for BN254 {
    type Base = Base<ArkBigInteger256, 8>;
    type Scalar = Scalar<ArkBigInteger256, 8, Self::Fr>;
    type PointProjective = Point<Self::Base, Self::Fq, Self::G1ArkAffine, Self::G1ArkProjective>;
    type PointAffine = PointAffineNoInfinity<Self::Base, Self::Fq, Self::G1ArkAffine, Self::G1ArkProjective>;
    type Fq = Fq_BN254;
    type Fr = Fr_BN254;
    type G1ArkAffine = G1Affine_BN254;
    type G1ArkProjective = G1Projective_BN254;

    // ... implement rest of trait functions
}