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

__global__ void mykernel(int ny, int nx, int nn, float* norm_data, float* norm_data_transpose, float* result) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= nn || j >= nn || j < i)
        return;
    float sumx = 0;
    for (int k = 0; k < nn; k++){
        sumx += norm_data_transpose[i + k * nn] * norm_data[k + j * nn];
    }
    if (i < ny && j < ny)
    {
        result[j + i * ny] = sumx;
    }

}

__global__ void mykernel_transpose(int ny, int nx, int nn, float* norm_data, float* norm_data_transpose) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= nn || j >= nn)
        return;
    norm_data_transpose[i + j * nn] = norm_data[j + i * nn];
}

void correlate(int ny, int nx, const float* data, float* result) {
    int nn = 0;
    if (ny >= nx)
    {
        nn = roundup(ny, 64);
    }
    else
    {
        nn = roundup(nx, 64);
    }

    float *interm_data= (float *)malloc(sizeof(float) * nn * nn);
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
            interm_data[column + row * nn] = x;
            square_sum += x * x;
        }
        square_sum = sqrt(square_sum);
        for(int column = 0; column < nx; column++) {
            interm_data[column + row * nn] /= square_sum;
        }
    }
    
    for (int row = ny; row < nn ; row ++)
    {
        for (int column = nx; column < nn; column ++)
        {
            interm_data[column + row * nn] = 0;
        }
    }

    // Allocate memory & copy data to GPU
    float* dGPU = NULL;
    float* dGPU_transpose = NULL;
    float* rGPU = NULL;
    CHECK(cudaMalloc((void**)&dGPU, nn * nn * sizeof(float)));
    CHECK(cudaMalloc((void**)&dGPU_transpose, nn * nn * sizeof(float)));
    CHECK(cudaMalloc((void**)&rGPU, ny * ny * sizeof(float)));

    CHECK(cudaMemcpy(dGPU, interm_data, nn * nn * sizeof(float), cudaMemcpyHostToDevice));

    // Run kernel for transpose
    {
        dim3 dimBlock(32, 32);
        dim3 dimGrid(divup(nn, dimBlock.x), divup(nn, dimBlock.y));
        mykernel_transpose<<<dimGrid, dimBlock>>>(ny, nx, nn, dGPU, dGPU_transpose);
        CHECK(cudaGetLastError());
    }

    // Run kernel for matrix multiplication
    {
        dim3 dimBlock(32, 32);
        dim3 dimGrid(divup(nn, dimBlock.x), divup(nn, dimBlock.y));
        mykernel<<<dimGrid, dimBlock>>>(ny, nx, nn, dGPU, dGPU_transpose, rGPU);
        CHECK(cudaGetLastError());
    }


    // Copy data back to CPU & release memory
    CHECK(cudaMemcpy(result, rGPU, ny * ny * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(dGPU));
    CHECK(cudaFree(dGPU_transpose));
    CHECK(cudaFree(rGPU));

    free(interm_data);
}