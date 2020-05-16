#include "cp.h"
#include <cuda_runtime.h>
#include "math.h"
#include <stdio.h>
#include <vector>
#include <numeric>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <iostream>

static inline int divup(int a, int b) {
    return (a + b - 1)/b;
}
static inline int roundup(int a, int b) {
    return divup(a, b) * b;
}

static inline void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK(x) check(x, #x)
#define BLOCK_SIZE 32

__global__ void mykernel(int ny, int nx, int nn_padding, float* norm_data, float* norm_data_transpose, float* result) {

    __shared__ float norm_data_shared [BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float norm_data_transpose_shared[BLOCK_SIZE][BLOCK_SIZE];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;

    for (int tileNUM = 0; tileNUM < gridDim.x; tileNUM++) {
        int j = tileNUM * BLOCK_SIZE + threadIdx.x;
        int i = tileNUM * BLOCK_SIZE + threadIdx.y;

        norm_data_shared[threadIdx.y][threadIdx.x] = norm_data[row * nn_padding + j];
        norm_data_transpose_shared[threadIdx.y][threadIdx.x] = norm_data_transpose[i * nn_padding + col];
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; k++) {

            sum += norm_data_shared[threadIdx.y][k] * norm_data_transpose_shared[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < ny && col < ny) result[row + col * ny] = sum;
}

__global__ void mykernel_transpose(int nn_padding, float* norm_data, float* norm_data_transpose) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= nn_padding || j >= nn_padding) return;
    norm_data_transpose[i + j * nn_padding] = norm_data[j + i * nn_padding];
}

void correlate(int ny, int nx, const float* data, float* result) {
    int nn_padding = 0;
    if (ny >= nx) nn_padding = roundup(ny, BLOCK_SIZE);
    else nn_padding = roundup(nx, BLOCK_SIZE);

    std::vector<float> interm_data(nn_padding * nn_padding);
    for (int row = 0; row < ny ; row++)
    {
        float sum = 0;
        for (int column = 0; column < nx; column ++)
        {
            sum += data[column + row * nx];
        }
        float mean = sum / nx;
        float square_sum = 0;
        for (int column = 0; column < nx; column ++)
        {
            float x = data[column + row * nx] - mean;
            interm_data[column + row * nn_padding] = x;
            square_sum += x * x;
        }
        square_sum = sqrt(square_sum);
        for(int column = 0; column < nx; column++) {
            interm_data[column + row * nn_padding] /= square_sum;
        }
        for (int column = nx; column < nn_padding; column++)
        {
            interm_data[column + row * nn_padding] = 0;
        }
    }
    for (int row = ny; row < nn_padding; row++)
    {
        for (int column = 0; column < nn_padding; column++)
        {
            interm_data[column + row * nn_padding] = 0;
        }
    }

    // Allocate memory & copy data to GPU
    float* dGPU = NULL;
    float* dGPU_transpose = NULL;
    float* rGPU = NULL;
    CHECK(cudaMalloc((void**)&dGPU, nn_padding * nn_padding * sizeof(float)));
    CHECK(cudaMalloc((void**)&dGPU_transpose, nn_padding * nn_padding * sizeof(float)));
    CHECK(cudaMalloc((void**)&rGPU, ny * ny * sizeof(float)));

    CHECK(cudaMemcpy(dGPU, interm_data.data(), nn_padding * nn_padding * sizeof(float), cudaMemcpyHostToDevice));

    // Run kernel for transpose
    {
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 dimGrid(divup(nn_padding, dimBlock.x), divup(nn_padding, dimBlock.y));
        mykernel_transpose<<<dimGrid, dimBlock>>>(nn_padding, dGPU, dGPU_transpose);
        CHECK(cudaGetLastError());
    }

    // Run kernel for matrix multiplication
    {
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 dimGrid(divup(nn_padding, dimBlock.x), divup(nn_padding, dimBlock.y));
        mykernel<<<dimGrid, dimBlock>>>(ny, nx, nn_padding, dGPU, dGPU_transpose, rGPU);
        CHECK(cudaGetLastError());
    }


    // Copy data back to CPU & release memory
    CHECK(cudaMemcpy(result, rGPU, ny * ny * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(dGPU));
    CHECK(cudaFree(dGPU_transpose));
    CHECK(cudaFree(rGPU));
}