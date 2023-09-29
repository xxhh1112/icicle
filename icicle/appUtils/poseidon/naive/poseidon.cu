#include "poseidon.cuh"

template <typename S>
__global__ void
prepare_poseidon_states(S* states, size_t number_of_states, S domain_tag, const PoseidonConfiguration<S> config, bool aligned)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int state_number = idx / config.t;
  if (state_number >= number_of_states) { return; }
  int element_number = idx % config.t;

  S prepared_element;

  // Domain separation
  if (element_number == 0) {
    prepared_element = domain_tag;
  } else {
    if (aligned) {
        prepared_element = states[idx];
    } else {
        prepared_element = states[state_number * config.t + element_number - 1];
    }
  }

  if (!aligned) {
    __syncthreads();
  }

  // Store element in state
  states[idx] = prepared_element;
}

template <typename S>
__device__ __forceinline__ S sbox_cube(S element)
{
  S result = S::sqr(element);
  return result * element;
}

template <typename S>
__device__ S vecs_mul_matrix(S element, S* matrix, int element_number, int vec_number, int size, S* shared_states)
{
  shared_states[threadIdx.x] = element;
  __syncthreads();

  typename S::Wide element_wide = S::mul_wide(shared_states[vec_number * size], matrix[element_number]);
  for (int i = 1; i < size; i++) {
    element_wide = element_wide + S::mul_wide(shared_states[vec_number * size + i], matrix[i * size + element_number]);
  }
  __syncthreads();

  return S::reduce(element_wide);
}

template <typename S>
__device__ S round(
  S element,
  size_t rc_offset,
  bool is_full_round,
  int local_state_number,
  int element_number,
  S* shared_states,
  const PoseidonConfiguration<S> config)
{
  if (is_full_round || element_number == (config.t - 1)) {
    element = element + config.round_constants[rc_offset + element_number * is_full_round];
    element = sbox_cube(element);
  }

  // Multiply all the states by mds matrix
  return vecs_mul_matrix(element, config.mds_matrix, element_number, local_state_number, config.t, shared_states);
}

// Execute full rounds
template <typename S>
__global__ void rounds(
  S* states, size_t number_of_states, size_t rc_offset, bool is_full_rounds, const PoseidonConfiguration<S> config)
{
  extern __shared__ S shared_states[];

  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int state_number = idx / config.t;
  if (state_number >= number_of_states) { return; }
  int local_state_number = threadIdx.x / config.t;
  int element_number = idx % config.t;

  for (int i = 0; i < (is_full_rounds ? config.full_rounds_half : config.partial_rounds); i++) {
    states[idx] = round(states[idx], rc_offset, is_full_rounds,
                        local_state_number, element_number, shared_states, config);
    rc_offset += (is_full_rounds ? config.t : 1);
  }
}

// These function is just doing copy from the states to the output
template <typename S>
__global__ void get_hash_results(S* states, size_t number_of_states, S* out, int t)
{
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= number_of_states) { return; }

  out[idx] = states[idx * t + 1];
}

template <typename S>
__global__ void copy_recursive(S * state, size_t number_of_states, S * out, int t) {
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= number_of_states) {
      return;
  }

  state[(idx / (t - 1) * t) + (idx % (t - 1)) + 1] = out[idx];
}