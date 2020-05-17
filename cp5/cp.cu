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

__global__ void mykernel(int ny, int nx, int nn_padding, int n, float* padded_norm_and_transpose, float* result) {
    if (blockIdx.x > blockIdx.y) return; 
    int ia = threadIdx.x;
    int ja = threadIdx.y;
    int ic = blockIdx.x;
    int jc = blockIdx.y;

    const float* transpose = padded_norm_and_transpose + nn_padding * nn_padding;

    float v[8][8];
    for (int ib = 0; ib < 8; ib++)
    {
        for (int jb = 0; jb < 8; jb++)
        {
            v[ib][jb] = 0;
        }
    }
    for (int k = 0; k < n; k++)
    {
        float x[8];
        float y[8];
        for (int ib = 0; ib < 8; ib++)
        {
            int i = ic * 64 + ib * 8 + ia;
            x[ib] = transpose[nn_padding * k + i];
        }
        for (int jb = 0; jb < 8; jb++)
        {
            int j = jc * 64 + jb * 8 + ja;
            y[jb] = transpose[nn_padding * k + j];
        }
        for (int ib = 0; ib < 8; ib++)
        {
            for (int jb = 0; jb < 8; jb++)
            {
                v[ib][jb] += x[ib] * y[jb];
            }
        }
    }
    for (int ib = 0; ib < 8; ib++)
    {
        for (int jb = 0; jb < 8; jb++)
        {
            int i = ic * 64 + ib * 8 + ia;
            int j = jc * 64 + jb * 8 + ja;
            if (i < ny && j < ny) result[j + i*ny] = v[ib][jb];
        }
    }
}

__global__ void mykernel_transpose(int nn_padding, int nx, int ny, float* padded_norm_and_transpose, float* norm_data_temp) {
    int ja = threadIdx.x;
    int i = blockIdx.y;
    float* transpose = padded_norm_and_transpose + nn_padding * nn_padding;

    for (int jb = 0; jb < nn_padding; jb += 64) 
    {
        int j = jb + ja;
        float v = (i < ny && j < nx) ? norm_data_temp[nx * i + j] : 0;
        padded_norm_and_transpose[nn_padding * i + j] = v;
        transpose[nn_padding * j + i] = v;
    }
}

void correlate(int ny, int nx, const float* data, float* result) {
    int nn_padding = 0, n = 0;
    if (ny >= nx)
    {
        nn_padding = roundup(ny, 64);
        n = ny;
    }
    else
    {
        nn_padding = roundup(nx, 64);
        n = nx;
    }

    std::vector<float> interm_data(nx * ny);
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
            interm_data[column + row * nx] = x;
            square_sum += x * x;
        }
        square_sum = sqrt(square_sum);
        for(int column = 0; column < nx; column++) {
            interm_data[column + row * nx] /= square_sum;
        }
    }

    const int input_size = nn_padding * nn_padding * sizeof(float);
    const int original_size = nx * ny * sizeof(float);
    const int output_size = ny * ny * sizeof(float);
    float* dGPU = NULL;
    float* dGPU_temp = NULL;
    float* rGPU = NULL;

    // Allocate memory & copy data to GPU
    CHECK(cudaMalloc((void**)&dGPU, 2 * input_size));
    CHECK(cudaMalloc((void**)&dGPU_temp, original_size));
    CHECK(cudaMalloc((void**)&rGPU, output_size));
    CHECK(cudaMemcpy(dGPU_temp, interm_data.data(), original_size, cudaMemcpyHostToDevice));

    // Run kernel for transpose and padding
    {
        dim3 dimBlock(64, 1);
        dim3 dimGrid(1, nn_padding);
        mykernel_transpose<<<dimGrid, dimBlock>>>(nn_padding, nx, ny, dGPU, dGPU_temp);
        CHECK(cudaGetLastError());
    }

    // Run kernel for matrix multiplication
    {
        dim3 dimBlock(8, 8);
        dim3 dimGrid(nn_padding / 64, nn_padding / 64);
        mykernel<<<dimGrid, dimBlock>>>(ny, nx, nn_padding, n, dGPU, rGPU);
        CHECK(cudaGetLastError());
    }

    // Copy data back to CPU & release memory
    CHECK(cudaMemcpy(result, rGPU, output_size, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(dGPU));
    CHECK(cudaFree(dGPU_temp));
    CHECK(cudaFree(rGPU));
}