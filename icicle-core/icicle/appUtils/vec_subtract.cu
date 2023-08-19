#include "../curves/curve_config.cuh"

#define MAX_THREADS_PER_BLOCK 256


template <typename T>
__global__ void vector_subtract_kernel(T *vec1, T *vec2, T *result, size_t n_elements)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n_elements)
    {
        result[tid] = vec1[tid] - vec2[tid];
    }
}

template <typename T>
int vector_subtract(T *vec1, T *vec2, T *result, size_t n_elements)
{
    // Set the grid and block dimensions
    int num_blocks = (int)ceil((float)n_elements / MAX_THREADS_PER_BLOCK);
    int threads_per_block = MAX_THREADS_PER_BLOCK;

    // Allocate memory on the device for the input vectors and the output vector
    T *d_vec1, *d_vec2, *d_result;
    cudaMalloc(&d_vec1, n_elements * sizeof(T));
    cudaMalloc(&d_vec2, n_elements * sizeof(T));
    cudaMalloc(&d_result, n_elements * sizeof(T));

    // Copy the input vectors from the host to the device
    cudaMemcpy(d_vec1, vec1, n_elements * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec2, vec2, n_elements * sizeof(T), cudaMemcpyHostToDevice);

    // Call the kernel to perform element-wise subtraction
    vector_subtract_kernel<<<num_blocks, threads_per_block>>>(d_vec1, d_vec2, d_result, n_elements);

    cudaMemcpy(result, d_result, n_elements * sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(d_vec1);
    cudaFree(d_vec2);
    cudaFree(d_result);

    return 0;
}

extern "C" int subtract(point_field *vec1, point_field *vec2, point_field *res, size_t n_elements)
{
    try
    {
        vector_subtract<point_field>(vec1, vec2, res, n_elements);
        return 0;
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}
