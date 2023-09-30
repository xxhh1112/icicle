#pragma once

#include "constants/f251/constants_2.h"
// #include "constants/f251/constants_3.h"
// #include "constants/f251/constants_4.h"
// #include "constants/f251/constants_8.h"
#include <cassert>
#include <map>
#include <stdexcept>

uint32_t partial_rounds_number_from_arity(const uint32_t arity)
{
  switch (arity) {
  case 2:
    return 83;
  case 3:
    return 84;
  case 4:
    return 84;
  case 8:
    return 84;
  default:
    throw std::invalid_argument("unsupported arity");
  }
};

const uint32_t FULL_ROUNDS_DEFAULT = 4;

template <typename S>
S* load_round_constants(const uint32_t arity)
{
  S* constants;
  switch (arity) {
  case 2:
    constants = reinterpret_cast<S*>(&rc_2);
    break;
  // case 4:
  //   constants = constants_4;
  //   break;
  // case 8:
  //   constants = constants_8;
  //   break;
  // case 11:
  //   constants = constants_11;
  //   break;
  default:
    throw std::invalid_argument("unsupported arity");
  }
  return constants;
}

template <typename S>
S* load_mds_matrix(const uint32_t arity)
{
  S* constants;
  switch (arity) {
  case 2:
    constants = reinterpret_cast<S*>(&mds_2);
    break;
  // case 4:
  //   constants = constants_4;
  //   break;
  // case 8:
  //   constants = constants_8;
  //   break;
  // case 11:
  //   constants = constants_11;
  //   break;
  default:
    throw std::invalid_argument("unsupported arity");
  }
  return reinterpret_cast<S*>(constants);
}

template <typename S>
S* load_pre_mds_matrix(const uint32_t arity)
{
  S* constants;
  switch (arity) {
  case 2:
    constants = reinterpret_cast<S*>(&non_sparse_mds_2);
    break;
  // case 4:
  //   constants = constants_4;
  //   break;
  // case 8:
  //   constants = constants_8;
  //   break;
  // case 11:
  //   constants = constants_11;
  //   break;
  default:
    throw std::invalid_argument("unsupported arity");
  }
  return reinterpret_cast<S*>(constants);
}

template <typename S>
S* load_sparse_matrices(const uint32_t arity)
{
  S* constants;
  switch (arity) {
  case 2:
    constants = reinterpret_cast<S*>(&sparse_matrices_2);
    break;
  // case 4:
  //   constants = constants_4;
  //   break;
  // case 8:
  //   constants = constants_8;
  //   break;
  // case 11:
  //   constants = constants_11;
  //   break;
  default:
    throw std::invalid_argument("unsupported arity");
  }
  return reinterpret_cast<S*>(constants);
}