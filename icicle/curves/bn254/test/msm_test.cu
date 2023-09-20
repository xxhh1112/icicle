#include "../../primitives/field.cuh"
#include "../../primitives/projective.cuh"
#include "../../utils/cuda_utils.cuh"
#include "../msm.cu"
#include "../curve_config.cuh"
#include <thread>
#include <cuda.h>

using namespace BN254;

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

int main() {
  unsigned batch_size = 1;
  unsigned msm_size = 12180757;
  unsigned N = batch_size * msm_size;

  scalar_t* scalars = (scalar_t*)malloc(N * sizeof(scalar_t));
  affine_t* points = (affine_t*)malloc(N * sizeof(affine_t));

  for (unsigned i = 0; i < N; i++) {
    points[i] = (i % msm_size < 10) ? projective_t::to_affine(projective_t::rand_host()) : points[i - 10];
    scalars[i] = scalar_t::rand_host();
  }
  std::cout << "finished generating" << std::endl;

  projective_t response1[batch_size * 2];
  
  cudaStream_t stream1;

  //auto beginmem = std::chrono::high_resolution_clock::now();
  //msm_cuda_bn254(response1, points, scalars, msm_size, 10, 0, stream1);  
  //msm_cuda_bn254(response2, points, scalars, msm_size, 10, 0, stream2);
  //auto endmem = std::chrono::high_resolution_clock::now();
  //auto elapsedmem = std::chrono::duration_cast<std::chrono::nanoseconds>(endmem - beginmem);
  //printf("Time taken with memcpy in sync: %.3f seconds.\n", elapsedmem.count() * 1e-9);
//
//
  //auto begin = std::chrono::high_resolution_clock::now();
  //msm_cuda_bn254(response1, points_d3, scalars_d3, msm_size, 10, 0, stream1);  
  //msm_cuda_bn254(response2, points_d3, scalars_d3, msm_size, 10, 0, stream2);
  //auto end = std::chrono::high_resolution_clock::now();
  //auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
  //printf("Time taken in sync: %.3f seconds.\n", elapsed.count() * 1e-9);

  auto beginmultithread = std::chrono::high_resolution_clock::now();

  std::thread thread1([&]() {
    scalar_t* scalars_d;
    affine_t* points_d;
    projective_t* large_res_d;

    CHECK_CUDA_ERROR(cudaMalloc(&scalars_d, sizeof(scalar_t) * msm_size));
    CHECK_CUDA_ERROR(cudaMalloc(&points_d, sizeof(affine_t) * msm_size));
    CHECK_CUDA_ERROR(cudaMalloc(&large_res_d, sizeof(projective_t)));
    CHECK_CUDA_ERROR(cudaMemcpy(scalars_d, scalars, sizeof(scalar_t) * msm_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(points_d, points, sizeof(affine_t) * msm_size, cudaMemcpyHostToDevice));

    auto begin = std::chrono::high_resolution_clock::now();
    msm_cuda_bn254(large_res_d, points_d, scalars_d, msm_size, 10, 0, stream1, true);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    printf("Thread 1 time taken: %.3f seconds.\n", elapsed.count() * 1e-9);
  });

/*
  std::thread thread2([&]() {

    scalar_t* scalars_d;
    affine_t* points_d;
    projective_t* large_res_d;

    cudaMalloc(&scalars_d, sizeof(scalar_t) * msm_size);
    cudaMalloc(&points_d, sizeof(affine_t) * msm_size);
    cudaMalloc(&large_res_d, sizeof(projective_t));
    cudaMemcpy(scalars_d, scalars, sizeof(scalar_t) * msm_size, cudaMemcpyHostToDevice);
    cudaMemcpy(points_d, points, sizeof(affine_t) * msm_size, cudaMemcpyHostToDevice);

    auto begin = std::chrono::high_resolution_clock::now();
    msm_cuda_bn254(response4, points_d, scalars_d, msm_size, 10, 1, stream4, false);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    printf("Thread 2 time taken: %.3f seconds.\n", elapsed.count() * 1e-9);
  });
*/

  thread1.join();

  auto endmultithread = std::chrono::high_resolution_clock::now();
  auto elapsedmultithread = std::chrono::duration_cast<std::chrono::nanoseconds>(endmultithread - beginmultithread);
  printf("Time taken in parrlell: %.3f seconds.\n", elapsedmultithread.count() * 1e-9);
}

