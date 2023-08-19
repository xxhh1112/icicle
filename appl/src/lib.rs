use icicle_core::test_field::Field;
// use icicle_bls12_381::ScalarField as ScalarField1;
// use icicle_bls12_381::sub as sub1;
use icicle_bn254::ScalarField as ScalarField2;
use icicle_bn254::sub as sub2;


#[test]
fn it_works() {
    // let zero1 = [ScalarField1::zero()];
    // let one1 = [ScalarField1::one()];
    // println!("1 in bls12-381 base field: {:?}", sub1(&zero1[..], &sub1(&zero1[..], &one1[..])[..]));
    let zero2 = [ScalarField2::zero()];
    let one2 = [ScalarField2::one()];
    println!("1 in bn254 base field: {:?}", sub2(&zero2[..], &sub2(&zero2[..], &one2[..])[..]));
}
