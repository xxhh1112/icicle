use icicle_bls12_381::ScalarField as ScalarField1;
use icicle_bls12_381::do_smth_381;
use icicle_bn254::ScalarField as ScalarField2;
use icicle_bn254::do_smth_254;


#[test]
fn it_works() {
    let mut values1 = [ScalarField1::zero()];
    println!("Did smth 381: {:?}", do_smth_381(&mut values1[..]));
    let mut values2 = [ScalarField2::zero()];
    println!("Did smth 254: {:?}", do_smth_254(&mut values2[..]));
}
