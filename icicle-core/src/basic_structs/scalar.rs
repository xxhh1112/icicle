use std::mem::transmute;
use crate::utils::{u32_vec_to_u64_vec, u64_vec_to_u32_vec};
use std::marker::PhantomData;
use std::convert::TryInto;
use ark_ff::{BigInteger as ArkBigInteger, PrimeField as ArkPrimeField};

pub fn get_fixed_limbs<const NUM_LIMBS: usize>(val: &[u32]) -> [u32; NUM_LIMBS] {
    match val.len() {
        n if n < NUM_LIMBS => {
            let mut padded: [u32; NUM_LIMBS] = [0; NUM_LIMBS];
            padded[..val.len()].copy_from_slice(&val);
            padded
        }
        n if n == NUM_LIMBS => val.try_into().unwrap(),
        _ => panic!("slice has too many elements"),
    }
}

pub trait FieldOps {
    fn base_limbs() -> usize;
    fn zero() -> Self;
    fn from_limbs(value: &[u32]) -> Self;
    fn one() -> Self;
    fn to_bytes_le(&self) -> Vec<u8>;
    fn limbs(&self) -> &[u32];
}

pub trait ScalarOps {

}

#[derive(Debug, PartialEq, Clone, Copy)]
#[repr(C)]
pub struct Scalar<BigInteger, const NUM_LIMBS: usize, Fr> {
    pub value: [u32; NUM_LIMBS],
    pub big_int: PhantomData<BigInteger>,
    pub fr: PhantomData<Fr>
}

impl<BigInteger, const NUM_LIMBS: usize, Fr> FieldOps for Scalar<BigInteger, NUM_LIMBS, Fr> 
where 
    BigInteger: ArkBigInteger,
    Fr: ArkPrimeField
{

    fn base_limbs() -> usize {
        return NUM_LIMBS; 
    }

    fn zero() -> Self {
        Scalar {
            value: [0u32; NUM_LIMBS],
            big_int: PhantomData,
            fr: PhantomData
        }
    }

    fn from_limbs(value: &[u32]) -> Self {
        Self {
            value: get_fixed_limbs(value),
            big_int: PhantomData,
            fr: PhantomData
        }
    }

    fn one() -> Self {
        let mut s = [0u32; NUM_LIMBS];
        s[0] = 1;
        Scalar { 
            value: s,
            big_int: PhantomData,
            fr: PhantomData
        }
    }

    fn to_bytes_le(&self) -> Vec<u8> {
        self.value
            .iter()
            .map(|s| s.to_le_bytes().to_vec())
            .flatten()
            .collect::<Vec<_>>()
    }

    fn limbs(&self) -> &[u32] {
        &self.value
    }
}

impl<BigInteger, const NUM_LIMBS: usize, Fr> Scalar<BigInteger, NUM_LIMBS, Fr> 
where 
    BigInteger: ArkBigInteger,
    Fr: ArkPrimeField
{
    pub fn from_limbs_le(value: &[u32]) -> Scalar<BigInteger, NUM_LIMBS, Fr> {
        Self::from_limbs(value)
     }
 
    pub fn from_limbs_be(value: &[u32]) -> Scalar<BigInteger, NUM_LIMBS, Fr> {
         let mut value = value.to_vec();
         value.reverse();
         Self::from_limbs_le(&value)
     }
 
     // Additional Functions
     pub fn add(&self, other: Scalar<BigInteger, NUM_LIMBS, Fr>) -> Scalar<BigInteger, NUM_LIMBS, Fr>{  // overload + 
         return Scalar{
            value: [self.value[0] + other.value[0];NUM_LIMBS], 
            big_int: PhantomData,
            fr: PhantomData
         }; 
    }
    
    pub fn to_biginteger254(&self) -> BigInteger {
        // Need to change this to convert to BigUint first then to BigInteger
        // BigInteger doesn't have any conversion from vec to itself
        let internal = u32_vec_to_u64_vec(&self.limbs()).try_into().unwrap();
        BigInteger::from(internal)
    }

    pub fn to_ark(&self) -> BigInteger {
        BigInteger::new(u32_vec_to_u64_vec(&self.limbs()).try_into().unwrap())
    }

    pub fn from_biginteger256(ark: BigInteger) -> Self {
        Self{ value: u64_vec_to_u32_vec(&ark.0).try_into().unwrap(), big_int : PhantomData, fr: PhantomData}
    }

    pub fn to_biginteger256_transmute(&self) -> BigInteger {
        unsafe { transmute(*self) }
    }

    pub fn from_biginteger_transmute(v: BigInteger) -> Scalar<BigInteger, NUM_LIMBS, Fr> {
        Scalar{ value: unsafe{ transmute(v)}, big_int : PhantomData, fr: PhantomData }
    }

    pub fn to_ark_transmute(&self) -> Fr {
        unsafe { std::mem::transmute(*self) }
    }

    pub fn from_ark_transmute(v: &Fr) -> Scalar<BigInteger, NUM_LIMBS, Fr> {
        unsafe { std::mem::transmute_copy(v) }
    }

    pub fn to_ark_mod_p(&self) -> Fr {
        Fr::new(BigInteger::new(u32_vec_to_u64_vec(&self.limbs()).try_into().unwrap()))
    }

    pub fn to_ark_repr(&self) -> Fr {
        Fr::from_repr(BigInteger::new(u32_vec_to_u64_vec(&self.limbs()).try_into().unwrap())).unwrap()
    }

    pub fn from_ark(v: BigInteger) -> Scalar<BigInteger, NUM_LIMBS, Fr> {
        Self { value : u64_vec_to_u32_vec(&v.0).try_into().unwrap(), big_int: PhantomData, fr: PhantomData}
    }

}

// Base

pub struct Base<BigInteger, const NUM_LIMBS: usize> {
    pub value: [u32; NUM_LIMBS],
    pub big_int: PhantomData<BigInteger>
}

impl<BigInteger, const NUM_LIMBS: usize> FieldOps for Base<BigInteger, NUM_LIMBS> 
where 
    BigInteger: ArkBigInteger
{

    fn base_limbs() -> usize {
        return NUM_LIMBS; 
    }

    fn zero() -> Self {
        Base {
            value: [0u32; NUM_LIMBS],
            big_int: PhantomData
        }
    }

    fn from_limbs(value: &[u32]) -> Self {
        Self {
            value: get_fixed_limbs(value),
            big_int: PhantomData
        }
    }

    fn one() -> Self {
        let mut s = [0u32; NUM_LIMBS];
        s[0] = 1;
        Base { 
            value: s,
            big_int: PhantomData
        }
    }

    fn to_bytes_le(&self) -> Vec<u8> {
        self.value
            .iter()
            .map(|s| s.to_le_bytes().to_vec())
            .flatten()
            .collect::<Vec<_>>()
    }

    fn limbs(&self) -> &[u32] {
        &self.value
    }
}

impl<BigInteger, const NUM_LIMBS: usize> Base<BigInteger, NUM_LIMBS> 
where 
    BigInteger: ArkBigInteger
{
    pub fn to_ark(&self) -> BigInteger {
        BigInteger::new(u32_vec_to_u64_vec(&self.limbs()).try_into().unwrap())
    }

    pub fn from_ark(ark: BigInteger) -> Self {
        Self::from_limbs(&u64_vec_to_u32_vec(&ark.0))
    }
}