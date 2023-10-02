// msm_test.cu

#include "../curve_config.cuh"
#include "../msm.cu"

using namespace BLS12_381;



extern "C" int msm_wrapper(
  BLS12_381::projective_t* out,
  BLS12_381::affine_t points[],
  BLS12_381::scalar_t scalars[],
  size_t count,
  unsigned large_bucket_factor,
  size_t device_id = 0,
  cudaStream_t stream = 0,
  bool on_device = false
  ) // TODO: unify parameter types size_t/unsigned etc
{
  try {
    cudaError_t setDeviceErr = cudaSetDevice(device_id);
    if (setDeviceErr != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(setDeviceErr));
    }

    cudaStreamCreate(&stream);
    large_msm<BLS12_381::scalar_t, BLS12_381::projective_t, BLS12_381::affine_t>(
      scalars, points, count, out, on_device, false, large_bucket_factor, stream);
    cudaStreamSynchronize(stream);
    return CUDA_SUCCESS;
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what());
    return -1;
  }
}

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line)
{
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
    }
}


int main()
{
    printf("Starting benchmark...\n");
    unsigned batch_size = 1;
    unsigned msm_size = 65537; // FIXME: this produces CUDA 700 error with value of '65535+1'
    unsigned N = batch_size * msm_size;

    scalar_t* scalars = (scalar_t*)malloc(N * sizeof(scalar_t));
    affine_t* points = (affine_t*)malloc(N * sizeof(affine_t));

    projective_t out[batch_size * 2];

    scalar_t* scalars_d;
    affine_t* points_d;
    projective_t* out_d;

    cudaStream_t stream1;

    printf("MSM size = %d...\n", msm_size);

    printf("Host: Generating input scalars and points...\n");

    for (unsigned i = 0; i < N; i++) {
        points[i] = (i % msm_size < 10) ? projective_t::to_affine(projective_t::rand_host()) : points[i - 10];
        scalars[i] = scalar_t::rand_host();
    }

    printf("Host: Computing MSM...\n");
    CHECK_CUDA_ERROR(cudaMalloc(&scalars_d, sizeof(scalar_t) * msm_size));
    CHECK_CUDA_ERROR(cudaMalloc(&points_d, sizeof(affine_t) * msm_size));
    CHECK_CUDA_ERROR(cudaMalloc(&out_d, sizeof(projective_t)));
    CHECK_CUDA_ERROR(cudaMemcpy(scalars_d, scalars, sizeof(scalar_t) * msm_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(points_d, points, sizeof(affine_t) * msm_size, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();
    msm_wrapper(out_d, points_d, scalars_d, msm_size, 1, 0, stream1, true);
    cudaMemcpy(out, out_d, sizeof(projective_t), cudaMemcpyDeviceToHost);
    CHECK_LAST_CUDA_ERROR();
    std::cout << projective_t::to_affine(out[0]) << std::endl;
}
