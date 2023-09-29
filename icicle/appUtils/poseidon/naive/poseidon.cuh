#include "../poseidon.cuh"
#include "constants.cuh"

template <typename S>
class NaivePoseidon: public Poseidon<S>
{
public:
  PoseidonConfiguration<S> config;
  ParallelPoseidonConfiguration kernel_params;

  NaivePoseidon(const uint32_t arity, cudaStream_t stream) : Poseidon<S>(arity), kernel_params(arity + 1) {
    config.t = arity + 1;
    this->stream = stream;

    config.full_rounds_half = FULL_ROUNDS_DEFAULT;
    config.partial_rounds = partial_rounds_number_from_arity(arity);

    uint32_t round_constants_len = config.t * config.full_rounds_half * 2 + config.partial_rounds;
    uint32_t mds_matrix_len = config.t * config.t;

    S* round_constants = load_round_constants<S>(arity);
    S* mds_matrices = load_mds_matrices<S>(arity);

#if !defined(__CUDA_ARCH__) && defined(DEBUG)
    // for (int i = 0; i < round_constants_len; i++) {
    //   std::cout << round_constants[i] << std::endl;
    // }
    // for (int i = 0; i < mds_matrix_len; i++) {
    //   std::cout << mds_matrices[i] << std::endl;
    // }
    std::cout << "P: " << config.partial_rounds << " F: " << config.full_rounds_half << std::endl;
#endif

    // Create streams for copying constants
    cudaStream_t stream_copy_round_constants, stream_copy_mds_matrix;
    cudaStreamCreate(&stream_copy_round_constants);
    cudaStreamCreate(&stream_copy_mds_matrix);

    // Create events for copying constants
    cudaEvent_t event_copied_round_constants, event_copy_mds_matrix;
    cudaEventCreateWithFlags(&event_copied_round_constants, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&event_copy_mds_matrix, cudaEventDisableTiming);

    // Malloc memory for copying constants
    cudaMallocAsync(&config.round_constants, sizeof(S) * round_constants_len, stream_copy_round_constants);
    cudaMallocAsync(&config.mds_matrix, sizeof(S) * mds_matrix_len, stream_copy_mds_matrix);

    // Copy constants
    cudaMemcpyAsync(
      config.round_constants, round_constants, sizeof(S) * round_constants_len, cudaMemcpyHostToDevice,
      stream_copy_round_constants);
    cudaMemcpyAsync(
      config.mds_matrix, mds_matrices, sizeof(S) * mds_matrix_len, cudaMemcpyHostToDevice, stream_copy_mds_matrix);

    // Record finished copying event for streams
    cudaEventRecord(event_copied_round_constants, stream_copy_round_constants);
    cudaEventRecord(event_copy_mds_matrix, stream_copy_mds_matrix);

    // Main stream waits for copying to finish
    cudaStreamWaitEvent(stream, event_copied_round_constants);
    cudaStreamWaitEvent(stream, event_copy_mds_matrix);
  }

  ~NaivePoseidon()
  {
    cudaFreeAsync(config.round_constants, stream);
    cudaFreeAsync(config.mds_matrix, stream);
  }

  void prepare_states(S * states, size_t number_of_states, S domain_tag, bool aligned) override {
    prepare_poseidon_states<<<
      kernel_params.number_of_full_blocks(number_of_states),
      kernel_params.number_of_threads,
      0,
      stream
    >>>(states, number_of_states, domain_tag, config, aligned);

  #if !defined(__CUDA_ARCH__) && defined(DEBUG)
    cudaStreamSynchronize(stream);
    std::cout << "Prepare states:" << std::endl;
    print_buffer_from_cuda<S>(states, number_of_states * config.t, config.t);
  #endif
  }

  void process_results(S * states, size_t number_of_states, S * out, bool loop_results) override {
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

  void permute_many(S * states, size_t number_of_states, cudaStream_t stream) override {
    size_t rc_offset = 0;
    
    // execute half full rounds
    rounds<<<
      kernel_params.number_of_full_blocks(number_of_states),
      kernel_params.number_of_threads,
      sizeof(S) * kernel_params.hashes_per_block * config.t,
      stream
    >>>(states, number_of_states, rc_offset, true, config);
    rc_offset += config.t * config.full_rounds_half;

  #if !defined(__CUDA_ARCH__) && defined(DEBUG)
    cudaStreamSynchronize(stream);
    std::cout << "Full rounds 1. RCOFFSET: " << rc_offset << std::endl;
    print_buffer_from_cuda<S>(states, number_of_states * config.t, config.t);
  #endif

    // execute partial rounds
    rounds<<<
      kernel_params.number_of_full_blocks(number_of_states),
      kernel_params.number_of_threads,
      sizeof(S) * kernel_params.hashes_per_block * config.t,
      stream
    >>>(states, number_of_states, rc_offset, false, config);
    rc_offset += config.partial_rounds;

  #if !defined(__CUDA_ARCH__) && defined(DEBUG)
    cudaStreamSynchronize(stream);
    std::cout << "Partial rounds. RCOFFSET: " << rc_offset << std::endl;
    print_buffer_from_cuda<S>(states, number_of_states * config.t, config.t);
  #endif

    // execute half full rounds
    rounds<<<
      kernel_params.number_of_full_blocks(number_of_states),
      kernel_params.number_of_threads,
      sizeof(S) * kernel_params.hashes_per_block * config.t,
      stream
    >>>(states, number_of_states, rc_offset, true, config);

  #if !defined(__CUDA_ARCH__) && defined(DEBUG)
    cudaStreamSynchronize(stream);
    std::cout << "Full rounds 2. RCOFFSET: " << rc_offset << std::endl;
    print_buffer_from_cuda<S>(states, number_of_states * config.t, config.t);
  #endif
  }
  
private:
  cudaStream_t stream;
};