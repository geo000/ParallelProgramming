#include "cp.h"
#include <cuda_runtime.h>
#include "math.h"
#include <stdio.h>
#include <vector>
#include <numeric>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>

static inline void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK(x) check(x, #x)

static inline int divup(int a, int b) {

    return (a + b - 1)/b;
}

__global__ void mykernel(int ny, int nx, float* norm_data, float* result) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= ny || j >= ny || j < i)
        return;
    float sumx = 0;
    for (int k = 0; k < nx; k++){
        sumx += norm_data[k + i * nx] * norm_data[k + j * nx];
    }
    result[j + i * ny] = sumx;
}

void correlate(int ny, int nx, const float* data, float* result) {

    float *interm_data= (float *)malloc(sizeof(float)*ny*nx);
    for (int row = 0; row < ny ; row ++)
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
            interm_data[column + row * nx] = x;
            square_sum += x * x;
        }
        square_sum = sqrt(square_sum);
        for(int column = 0; column < nx; column++) {
            interm_data[column + row * nx] /= square_sum;
        }
    }

    // Allocate memory & copy data to GPU

    float* dGPU = NULL;
    CHECK(cudaMalloc((void**)&dGPU, nx * ny * sizeof(float)));
    float* rGPU = NULL;
    CHECK(cudaMalloc((void**)&rGPU, ny * ny * sizeof(float)));
    CHECK(cudaMemcpy(dGPU, interm_data, nx * ny * sizeof(float), cudaMemcpyHostToDevice));

    // Run kernel

    dim3 dimBlock(16, 16);
    dim3 dimGrid(divup(ny, dimBlock.x), divup(ny, dimBlock.y));
    mykernel<<<dimGrid, dimBlock>>>(ny, nx, dGPU, rGPU);
    CHECK(cudaGetLastError());

    // Copy data back to CPU & release memory

    CHECK(cudaMemcpy(result, rGPU, ny * ny * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(dGPU));
    CHECK(cudaFree(rGPU));

    free(interm_data);
}