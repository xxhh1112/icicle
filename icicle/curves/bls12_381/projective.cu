#include "../../primitives/projective.cuh"
#include "curve_config.cuh"
#include <cuda.h>

extern "C" BLS12_381::projective_t random_projective_bls12_381() { return BLS12_381::projective_t::rand_host(); }

extern "C" BLS12_381::projective_t projective_zero_bls12_381() { return BLS12_381::projective_t::zero(); }

extern "C" bool projective_is_on_curve_bls12_381(BLS12_381::projective_t* point1)
{
  return BLS12_381::projective_t::is_on_curve(*point1);
}

extern "C" BLS12_381::affine_t projective_to_affine_bls12_381(BLS12_381::projective_t* point1)
{
  return BLS12_381::projective_t::to_affine(*point1);
}

extern "C" BLS12_381::projective_t projective_from_affine_bls12_381(BLS12_381::affine_t* point1)
{
  return BLS12_381::projective_t::from_affine(*point1);
}

extern "C" BLS12_381::scalar_field_t random_scalar_bls12_381() { return BLS12_381::scalar_field_t::rand_host(); }

extern "C" bool eq_bls12_381(BLS12_381::projective_t* point1, BLS12_381::projective_t* point2)
{
  return (*point1 == *point2) &&
         !((point1->x == BLS12_381::point_field_t::zero()) && (point1->y == BLS12_381::point_field_t::zero()) &&
           (point1->z == BLS12_381::point_field_t::zero())) &&
         !((point2->x == BLS12_381::point_field_t::zero()) && (point2->y == BLS12_381::point_field_t::zero()) &&
           (point2->z == BLS12_381::point_field_t::zero()));
}

#if defined(G2_DEFINED)
extern "C" bool eq_g2_bls12_381(BLS12_381::g2_projective_t* point1, BLS12_381::g2_projective_t* point2)
{
  return (*point1 == *point2) &&
         !((point1->x == BLS12_381::g2_point_field_t::zero()) && (point1->y == BLS12_381::g2_point_field_t::zero()) &&
           (point1->z == BLS12_381::g2_point_field_t::zero())) &&
         !((point2->x == BLS12_381::g2_point_field_t::zero()) && (point2->y == BLS12_381::g2_point_field_t::zero()) &&
           (point2->z == BLS12_381::g2_point_field_t::zero()));
}

extern "C" BLS12_381::g2_projective_t random_g2_projective_bls12_381()
{
  return BLS12_381::g2_projective_t::rand_host();
}

extern "C" BLS12_381::g2_affine_t g2_projective_to_affine_bls12_381(BLS12_381::g2_projective_t* point1)
{
  return BLS12_381::g2_projective_t::to_affine(*point1);
}

extern "C" BLS12_381::g2_projective_t g2_projective_from_affine_bls12_381(BLS12_381::g2_affine_t* point1)
{
  return BLS12_381::g2_projective_t::from_affine(*point1);
}

extern "C" bool g2_projective_is_on_curve_bls12_381(BLS12_381::g2_projective_t* point1)
{
  return BLS12_381::g2_projective_t::is_on_curve(*point1);
}

#endif
