use std::marker::PhantomData;
use ark_ff::{PrimeField as ArkPrimeField};
use ark_ec::{AffineCurve as ArkAffineCurve, ProjectiveCurve as ArkProjectiveCurve};
use super::scalar::{self, FieldOps};
use rustacuda_core::DeviceCopy;
use rustacuda_derive::DeviceCopy;


#[derive(Debug, Clone, Copy, DeviceCopy)]
#[repr(C)]
pub struct Point<BaseField: scalar::FieldOps, Field: ArkPrimeField, Affine: ArkAffineCurve, Projective: ArkProjectiveCurve> {
    pub x: BaseField,
    pub y: BaseField,
    pub z: BaseField,
    pub ark_field: PhantomData<Field>,
    pub ark_affine: PhantomData<Affine>,
    pub ark_proj: PhantomData<Projective>
}

impl<BaseField, Field, Affine, Projective> Default for Point<BaseField, Field, Affine, Projective> 
where
    BaseField: DeviceCopy + scalar::FieldOps, 
    Field: ArkPrimeField,
    Affine: ArkAffineCurve,
    Projective: ArkProjectiveCurve
{
    fn default() -> Self {
        Point::zero()
    }
}

impl<BaseField, Field, Affine, Projective> Point<BaseField, Field, Affine, Projective> 
where
    BaseField: DeviceCopy + scalar::FieldOps, 
    Field: ArkPrimeField,
    Affine: ArkAffineCurve,
    Projective: ArkProjectiveCurve
{
    pub fn zero() -> Self {
        Point {
            x: BaseField::zero(),
            y: BaseField::one(),
            z: BaseField::zero(),
            ark_field: PhantomData,
            ark_affine: PhantomData,
            ark_proj: PhantomData
        }
    }

    pub fn infinity() -> Self {
        Self::zero()
    }
}

impl<BaseField, Field, Affine, Projective> Point<BaseField, Field, Affine, Projective>
where
    BaseField: DeviceCopy + scalar::FieldOps, 
    Field: ArkPrimeField,
    Affine: ArkAffineCurve,
    Projective: ArkProjectiveCurve
{
    pub fn from_limbs(x: &[u32], y: &[u32], z: &[u32]) -> Self {
        Point {
            x: BaseField::from_limbs(x),
            y: BaseField::from_limbs(y),
            z: BaseField::from_limbs(z),
            ark_field: PhantomData,
            ark_affine: PhantomData,
            ark_proj: PhantomData
        }
    }

    pub fn from_xy_limbs(value: &[u32]) -> Point<BaseField, Field, Affine, Projective> {
        let l = value.len();
        assert_eq!(l, 3 * BaseField::base_limbs(), "length must be 3 * {}", BaseField::base_limbs());
        Point {
            x: BaseField::from_limbs(value[..BaseField::base_limbs()].try_into().unwrap()),
            y: BaseField::from_limbs(value[BaseField::base_limbs()..BaseField::base_limbs() * 2].try_into().unwrap()),
            z: BaseField::from_limbs(value[BaseField::base_limbs() * 2..].try_into().unwrap()),
            ark_field: PhantomData,
            ark_affine: PhantomData,
            ark_proj: PhantomData
        }
    }

    pub fn to_xy_strip_z(&self) -> PointAffineNoInfinity<BaseField, Field, Affine, Projective> {
        PointAffineNoInfinity {
            x: self.x,
            y: self.y,
            f: PhantomData,
            ac: PhantomData,
            pc: PhantomData
        }
    }

    pub fn to_ark(&self) -> Projective {
        self.to_ark_affine().into_projective()
    }

    pub fn to_ark_affine(&self) -> Affine {
        //TODO: generic conversion
        use std::ops::Mul;
        let proj_x_field = Field::from_le_bytes_mod_order(&self.x.to_bytes_le());
        let proj_y_field = Field::from_le_bytes_mod_order(&self.y.to_bytes_le());
        let proj_z_field = Field::from_le_bytes_mod_order(&self.z.to_bytes_le());
        let inverse_z = proj_z_field.inverse().unwrap();
        let aff_x = proj_x_field.mul(inverse_z);
        let aff_y = proj_y_field.mul(inverse_z);
        Affine::new(aff_x, aff_y, false)
    }

    pub fn from_ark(ark: Projective) -> Point<BaseField, Field, Affine, Projective> {
        let z_inv = ark.z.inverse().unwrap();
        let z_invsq = z_inv * z_inv;
        let z_invq3 = z_invsq * z_inv;
        Point {
            x: BaseField::from_ark((ark.x * z_invsq).into_repr()),
            y: BaseField::from_ark((ark.y * z_invq3).into_repr()),
            z: BaseField::one(),
            ark_field: PhantomData,
            ark_affine: PhantomData,
            ark_proj: PhantomData
        }
    }

    pub fn to_affine(&self) -> PointAffineNoInfinity<BaseField, Field, Affine, Projective> {
        let ark_affine = self.to_ark_affine();
        PointAffineNoInfinity {
            x: BaseField::from_ark(ark_affine.x.into_repr()),
            y: BaseField::from_ark(ark_affine.y.into_repr()),
            f: PhantomData,
            ac: PhantomData,
            pc: PhantomData
        }
    }
}


// Start POINT AFFINE
#[derive(Debug, PartialEq, Clone, Copy, DeviceCopy)]
#[repr(C)]
pub struct PointAffineNoInfinity<BaseField, Field, Projective, Affine> {
    pub x: BaseField,
    pub y: BaseField,
    pub f: PhantomData<Field>,
    pub ac: PhantomData<Affine>,
    pub pc: PhantomData<Projective>
}

impl<BaseField, Field, Affine, Projective> Default for PointAffineNoInfinity<BaseField, Field, Affine, Projective> 
where
    BaseField: DeviceCopy + scalar::FieldOps, 
    Field: ArkPrimeField,
    Affine: ArkAffineCurve,
    Projective: ArkProjectiveCurve
{
    fn default() -> Self {
        PointAffineNoInfinity {
            x: BaseField::zero(),
            y: BaseField::zero(),
            f: PhantomData,
            ac: PhantomData,
            pc: PhantomData
        }
    }
}

impl<BaseField, Field, Affine, Projective> PointAffineNoInfinity<BaseField, Field, Affine, Projective> 
where
    BaseField: Copy + scalar::FieldOps, 
    Field: ArkPrimeField,
    Affine: ArkAffineCurve,
    Projective: ArkProjectiveCurve
{
    ///From u32 limbs x,y
    pub fn from_limbs(x: &[u32], y: &[u32]) -> Self {
        PointAffineNoInfinity {
            x: BaseField::from_limbs(x),
            y: BaseField::from_limbs(y),
            f: PhantomData,
            ac: PhantomData,
            pc: PhantomData
        }
    }

    pub fn limbs(&self) -> Vec<u32> {
        [self.x.limbs(), self.y.limbs()].concat()
    }

    pub fn to_projective(&self) -> Point<BaseField, Field, Affine, Projective> {
        Point {
            x: self.x,
            y: self.y,
            z: BaseField::one(),
            ark_field: PhantomData,
            ark_affine: PhantomData,
            ark_proj: PhantomData
        }
    }

    pub fn to_ark(&self) -> Affine {
        Affine::new(Field::new(self.x.to_ark()), Field::new(self.y.to_ark()), false)
    }

    pub fn to_ark_repr(&self) -> Affine {
        Affine::new(
            Field::from_repr(self.x.to_ark()).unwrap(),
            Field::from_repr(self.y.to_ark()).unwrap(),
            false,
        )
    }

    pub fn from_ark(p: &Affine) -> Self {
        PointAffineNoInfinity {
            x: BaseField::from_ark(p.x.into_repr()),
            y: BaseField::from_ark(p.y.into_repr()),
            f: PhantomData,
            ac: PhantomData,
            pc: PhantomData
        }
    }
}
