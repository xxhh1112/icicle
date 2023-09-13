#include "../../primitives/field.cuh"
#include "../../primitives/projective.cuh"
#include "../../utils/cuda_utils.cuh"
#include "../msm.cu"
#include "../curve_config.cuh"
#include <thread>

using namespace BN254;

int main() {
  unsigned batch_size = 1;
  //   unsigned msm_size = 1<<21;
  unsigned msm_size = 12180757;

  scalar_t* scalars = new scalar_t[msm_size];
  affine_t* points = new affine_t[msm_size];

  for (unsigned i = 0; i < msm_size; i++) {
    // scalars[i] = (i%msm_size < 10)? test_scalar::rand_host() : scalars[i-10];
    points[i] = (i % msm_size < 10) ? projective_t::to_affine(projective_t::rand_host()) : points[i - 10];
    scalars[i] = scalar_t::rand_host();
    // scalars[i] = i < N/2? test_scalar::rand_host() : test_scalar::one();
    // points[i] = test_projective::to_affine(test_projective::rand_host());
  }
  std::cout << "finished generating" << std::endl;

  projective_t response1[batch_size * 2];
  projective_t response2[batch_size * 2];
  scalar_t* scalars_d;
  affine_t* points_d;
  projective_t* large_res_d;

  cudaMalloc(&scalars_d, sizeof(scalar_t) * msm_size);
  cudaMalloc(&points_d, sizeof(affine_t) * msm_size);
  cudaMalloc(&large_res_d, sizeof(projective_t));
  cudaMemcpy(scalars_d, scalars, sizeof(scalar_t) * msm_size, cudaMemcpyHostToDevice);
  cudaMemcpy(points_d, points, sizeof(affine_t) * msm_size, cudaMemcpyHostToDevice);

  std::cout << "finished copying" << std::endl;
  
  cudaStream_t stream1;
  cudaStream_t stream2;

  // Create threads
  std::thread thread1([&]() {
    msm_cuda_bn254(response1, points, scalars, msm_size, 10, 0, stream1);

    std::cout << "Total runtime of both threads: " << response1->x.export_limbs() << " milliseconds" << std::endl;
  });
  std::thread thread2([&]() {
    msm_cuda_bn254(response2, points, scalars, msm_size, 10, 1, stream2);

    std::cout << "Total runtime of both threads: " << response2->x.export_limbs() << " milliseconds" << std::endl;
  });

  
  // Wait for both threads to finish
  thread1.join();
  thread2.join();
}

