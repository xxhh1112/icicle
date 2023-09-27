#include "poseidon.cuh"

#define FIRST_FULL_ROUNDS true
#define SECOND_FULL_ROUNDS false

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
__device__ __forceinline__ S sbox_alpha_five(S element)
{
  S result = S::sqr(element);
  result = S::sqr(result);
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
__device__ S full_round(
  S element,
  size_t rc_offset,
  int local_state_number,
  int element_number,
  bool multiply_by_mds,
  bool add_pre_round_constants,
  bool skip_rc,
  S* shared_states,
  const PoseidonConfiguration<S> config)
{
  if (add_pre_round_constants) {
    element = element + config.round_constants[rc_offset + element_number];
    rc_offset += config.t;
  }
  element = sbox_alpha_five(element);
  if (!skip_rc) {
    element = element + config.round_constants[rc_offset + element_number];
  }

  // Multiply all the states by mds matrix
  S* matrix = multiply_by_mds ? config.mds_matrix : config.non_sparse_matrix;
  return vecs_mul_matrix(element, matrix, element_number, local_state_number, config.t, shared_states);
}

// Execute full rounds
template <typename S>
__global__ void full_rounds(
  S* states, size_t number_of_states, size_t rc_offset, bool first_half, const PoseidonConfiguration<S> config)
{
  extern __shared__ S shared_states[];

  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int state_number = idx / config.t;
  if (state_number >= number_of_states) { return; }
  int local_state_number = threadIdx.x / config.t;
  int element_number = idx % config.t;

  bool add_pre_round_constants = first_half;
  for (int i = 0; i < config.full_rounds_half; i++) {
    states[idx] =
      full_round(states[idx], rc_offset, local_state_number,
                 element_number, !first_half || (i < (config.full_rounds_half - 1)),
                 add_pre_round_constants, !first_half && (i == config.full_rounds_half - 1), shared_states, config);
    rc_offset += config.t;

    if (add_pre_round_constants) {
      rc_offset += config.t;
      add_pre_round_constants = false;
    }
  }
}

template <typename S>
__device__ S partial_round(S* state, size_t rc_offset, int round_number, const PoseidonConfiguration<S> config)
{
  S element = state[0];
  element = sbox_alpha_five(element);
  element = element + config.round_constants[rc_offset];

  S* sparse_matrix = &config.sparse_matrices[(config.t * 2 - 1) * round_number];

  typename S::Wide state_0_wide = S::mul_wide(element, sparse_matrix[0]);
  for (int i = 1; i < config.t; i++) {
    state_0_wide = state_0_wide + S::mul_wide(state[i], sparse_matrix[i]);
  }
  state[0] = S::reduce(state_0_wide);

  for (int i = 1; i < config.t; i++) {
    state[i] = state[i] + (element * sparse_matrix[config.t + i - 1]);
  }
}

// Execute partial rounds
template <typename S>
__global__ void
partial_rounds(S* states, size_t number_of_states, size_t rc_offset, const PoseidonConfiguration<S> config)
{
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= number_of_states) { return; }

  S* state = &states[idx * config.t];

  for (int i = 0; i < config.partial_rounds; i++) {
    partial_round(state, rc_offset, i, config);
    rc_offset++;
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

template <typename S>
__host__ void Poseidon<S>::permute_many(S * states, size_t number_of_states, cudaStream_t stream) {
  size_t rc_offset = 0;
  
  // execute half full rounds
  full_rounds<<<
    kernel_params.number_of_full_blocks(number_of_states),
    kernel_params.number_of_threads,
    sizeof(S) * kernel_params.hashes_per_block * config.t,
    stream
  >>>(states, number_of_states, rc_offset, FIRST_FULL_ROUNDS, config);
  rc_offset += config.t * (config.full_rounds_half + 1);

#if !defined(__CUDA_ARCH__) && defined(DEBUG)
  cudaStreamSynchronize(stream);
  std::cout << "Full rounds 1. RCOFFSET: " << rc_offset << std::endl;
  print_buffer_from_cuda<S>(states, number_of_states * config.t, config.t);
#endif

  // execute partial rounds
  partial_rounds<<<
    kernel_params.number_of_singlehash_blocks(number_of_states),
    kernel_params.singlehash_block_size,
    0,
    stream
  >>>(states, number_of_states, rc_offset, config);
  rc_offset += config.partial_rounds;

#if !defined(__CUDA_ARCH__) && defined(DEBUG)
  cudaStreamSynchronize(stream);
  std::cout << "Partial rounds. RCOFFSET: " << rc_offset << std::endl;
  print_buffer_from_cuda<S>(states, number_of_states * config.t, config.t);
#endif

  // execute half full rounds
  full_rounds<<<
    kernel_params.number_of_full_blocks(number_of_states),
    kernel_params.number_of_threads,
    sizeof(S) * kernel_params.hashes_per_block * config.t,
    stream
  >>>(states, number_of_states, rc_offset, SECOND_FULL_ROUNDS, config);

#if !defined(__CUDA_ARCH__) && defined(DEBUG)
  cudaStreamSynchronize(stream);
  std::cout << "Full rounds 2. RCOFFSET: " << rc_offset << std::endl;
  print_buffer_from_cuda<S>(states, number_of_states * config.t, config.t);
#endif
}

// Compute the poseidon hash over a sequence of preimages
///
///=====================================================
/// # Arguments
/// * `states`  - a device pointer to the states memory. Expected to be of size `number_of_states * t` elements. States should contain the leaves values
/// * `number_of_states`  - number of preimages number_of_states. Each block is of size t
/// * `out` - a device pointer to the digests memory. Expected to be of size `sum(arity ^ (i)) for i in [0..height-1]`
/// * `hash_type`  - this will determine the domain_tag value
/// * `stream` - a cuda stream to run the kernels
/// * `aligned` - if set to `true`, the algorithm expects the states to contain leaves in an aligned form
/// * `loop_results` - if set to `true`, the resulting hash will be also copied into the states memory in aligned form.
///
/// Aligned form (for arity = 2):
/// [0, X1, X2, 0, X3, X4, ...]
///
/// Not aligned form (for arity = 2) (you will get this format
///                                   after copying leaves with cudaMemcpy2D):
/// [X1, X2, 0, X3, X4, 0]
/// Note: elements denoted by 0 doesn't need to be set to 0, the algorithm
/// will replace them with domain tags.
///
/// # Algorithm
/// The function will split large trees into many subtrees of size that will fit `STREAM_CHUNK_SIZE`.
/// The subtrees will be constructed in streams pool. Each stream will handle a subtree
/// After all subtrees are constructed - the function will combine the resulting sub-digests into the final top-tree
///======================================================
template <typename S>
__host__ void Poseidon<S>::poseidon_hash(S * states, size_t number_of_states, S * out, HashType hash_type, cudaStream_t stream, bool aligned, bool loop_results) {
  // Pick the domain_tag accordinaly
  S domain_tag;
  switch (hash_type) {
  case HashType::ConstInputLen:
    domain_tag = this->const_input_no_pad_domain_tag;
    break;

  case HashType::MerkleTree:
    domain_tag = this->tree_domain_tag;
  }

  // Domain separation and adding pre-round constants
  prepare_poseidon_states<<<
    kernel_params.number_of_full_blocks(number_of_states),
    kernel_params.number_of_threads,
    0,
    stream
  >>>(states, number_of_states, domain_tag, config, aligned);

#if !defined(__CUDA_ARCH__) && defined(DEBUG)
  cudaStreamSynchronize(stream);
  std::cout << "Domain separation: " << std::endl;
  print_buffer_from_cuda<S>(states, number_of_states * config.t, config.t);
#endif

  permute_many(states, number_of_states, stream);

    // get output
  get_hash_results<<<
    kernel_params.number_of_singlehash_blocks(number_of_states),
    kernel_params.singlehash_block_size,
    0,
    stream
  >>> (states, number_of_states, out, config.t);

  if (loop_results) {
    copy_recursive <<<
    kernel_params.number_of_singlehash_blocks(number_of_states),
    kernel_params.singlehash_block_size,
      0,
      stream
    >>> (states, number_of_states, out, config.t);
  }
}

template <typename S>
__host__ void Poseidon<S>::hash_blocks(const S * inp, size_t number_of_states, S * out, HashType hash_type, cudaStream_t stream) {
  S * states, * out_device;
  // allocate memory for {number_of_states} states of {t} scalars each
  if (cudaMallocAsync(&states, number_of_states * config.t * sizeof(S), stream) != cudaSuccess) {
      throw std::runtime_error("Failed memory allocation on the device");
  }
  if (cudaMallocAsync(&out_device, number_of_states * sizeof(S), stream) != cudaSuccess) {
      throw std::runtime_error("Failed memory allocation on the device");
  }

  // This is where the input matrix of size Arity x NumberOfBlocks is
  // padded and coppied to device in a T x NumberOfBlocks matrix
  cudaMemcpy2DAsync(states, config.t * sizeof(S),  // Device pointer and device pitch
                inp, (config.t - 1) * sizeof(S),    // Host pointer and pitch
                (config.t - 1) * sizeof(S), number_of_states, // Size of the source matrix (Arity x NumberOfBlocks)
                cudaMemcpyHostToDevice, stream);

  poseidon_hash(states, number_of_states, out_device, hash_type, stream, false, false);

  cudaFreeAsync(states, stream);
  cudaMemcpyAsync(out, out_device, number_of_states * sizeof(S), cudaMemcpyDeviceToHost, stream);
  cudaFreeAsync(out_device, stream);

#if !defined(__CUDA_ARCH__) && defined(POSEIDON_DEBUG)
  cudaDeviceReset();
#endif
}