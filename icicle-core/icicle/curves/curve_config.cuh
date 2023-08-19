#pragma once

#if defined(G2_DEFINED)
#include "../primitives/extension_field.cuh"
#else
#include "../primitives/field.cuh"
#endif
#include "../primitives/projective.cuh"

#if CURVE == 12381
#include "bls12_381.cuh"
#elif CURVE == 254
#include "bn254.cuh"
#endif

typedef Field<fp_config> scalar_field;
typedef Field<fq_config> point_field;
static constexpr point_field b = point_field{ weierstrass_b };
typedef Projective<point_field, scalar_field, b> proj;
typedef Affine<point_field> affine;
#if defined(G2_DEFINED)
typedef ExtensionField<fq_config> g2_point_field;
static constexpr g2_point_field b_g2 = g2_point_field{ point_field{ weierstrass_b_g2_re }, point_field{ weierstrass_b_g2_im }};
typedef Projective<g2_point_field, scalar_field, b_g2> g2_proj;
typedef Affine<g2_point_field> g2_affine;
#endif
