#pragma once

#include "../../primitives/field.cuh"
#include "../../primitives/projective.cuh"
#if defined(G2_DEFINED)
#include "../../primitives/extension_field.cuh"
#endif

#include "params.cuh"

namespace BLS12_381 {
  typedef Field<PARAMS_BLS12_381::fp_config> scalar_t;
  typedef Field<PARAMS_BLS12_381::fq_config> point_field_t;
  static constexpr point_field_t gen_x = point_field_t{PARAMS_BLS12_381::g1_gen_x};
  static constexpr point_field_t gen_y = point_field_t{PARAMS_BLS12_381::g1_gen_y};
  static constexpr point_field_t b = point_field_t{PARAMS_BLS12_381::weierstrass_b};
  typedef Projective<point_field_t, scalar_t, b, gen_x, gen_y> projective_t;
  typedef Affine<point_field_t> affine_t;
#if defined(G2_DEFINED)
  typedef ExtensionField<PARAMS_BLS12_381::fq_config> g2_point_field_t;
  static constexpr g2_point_field_t g2_gen_x =
    g2_point_field_t{point_field_t{PARAMS_BLS12_381::g2_gen_x_re}, point_field_t{PARAMS_BLS12_381::g2_gen_x_im}};
  static constexpr g2_point_field_t g2_gen_y =
    g2_point_field_t{point_field_t{PARAMS_BLS12_381::g2_gen_y_re}, point_field_t{PARAMS_BLS12_381::g2_gen_y_im}};
  static constexpr g2_point_field_t g2_b = g2_point_field_t{
    point_field_t{PARAMS_BLS12_381::weierstrass_b_g2_re}, point_field_t{PARAMS_BLS12_381::weierstrass_b_g2_im}};
  typedef Projective<g2_point_field_t, scalar_t, g2_b, g2_gen_x, g2_gen_y> g2_projective_t;
  typedef Affine<g2_point_field_t> g2_affine_t;
#endif
} // namespace BLS12_381
