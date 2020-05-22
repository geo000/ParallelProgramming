#include "is.h"
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
#include <algorithm> 

static inline int divup(int a, int b) {
    return (a + b - 1)/b;
}

static inline void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

__global__ void mykernel(int* coordinates, float* maximum, const float* simplified_sum_data, int nx, int ny) {
    int height = threadIdx.x + blockIdx.x * blockDim.x;
    int width = threadIdx.y + blockIdx.y * blockDim.y;
    if (height > ny || width > nx || height == 0 || width == 0) return;
    int nx_plus = nx+1;
    float allpixels = nx*ny;
    float all_colors = simplified_sum_data[nx + (nx_plus) * ny];
    float best_so_far = -1;
    int x0_inner = 0, y0_inner = 0, x1_inner = 0, y1_inner = 0;
    float pixel_x = height * width;
    float pixel_y = allpixels - pixel_x;
    pixel_x = 1 / pixel_x;
    pixel_y = 1 / pixel_y;

    for (int y0 = 0; y0 <= ny - height; y0++){
        for (int x0 = 0; x0 <= nx - width; x0++){
            int y1 = y0 + height;
            int x1 = x0 + width;
            float sum_of_colors_X = simplified_sum_data[x1 + (nx_plus) * y1] - simplified_sum_data[x0 + (nx_plus) * y1]
                           + simplified_sum_data[x0 + (nx_plus) * y0] - simplified_sum_data[x1 + (nx_plus) * y0];
            float sum_of_colors_Y = all_colors - sum_of_colors_X;
            float best = sum_of_colors_X * sum_of_colors_X * pixel_x + sum_of_colors_Y * sum_of_colors_Y * pixel_y;
            if (best > best_so_far)
            {
                best_so_far = best;
                y0_inner = y0;
                x0_inner = x0;
                y1_inner = y1;
                x1_inner = x1;
            }
        }
    }
    maximum[nx_plus* height + width] = best_so_far;
    coordinates[4 * (nx_plus * height + width) + 0] = y0_inner;
    coordinates[4 * (nx_plus * height + width) + 1] = x0_inner;
    coordinates[4 * (nx_plus * height + width) + 2] = y1_inner;
    coordinates[4 * (nx_plus * height + width) + 3] = x1_inner;
}

#define CHECK(x) check(x, #x)
Result segment(int ny, int nx, const float* data) {
    float allpixels = nx*ny;
    int nx_plus = nx+1;
    int ny_plus = ny+1;
    std::vector<int> coordinates(4 * ny_plus * nx_plus);
    std::vector<float> maximum(ny_plus * nx_plus);
    std::vector<float> simplified_sum_data((nx_plus) * (ny_plus));
    for (int y1 = 0; y1 < ny; y1++) {
        for (int x1 = 0; x1 < nx; x1++) {
            simplified_sum_data[(x1+1) + (nx_plus) * (y1+1)] = data[3 * (x1 + nx * y1)] + simplified_sum_data[x1 + (nx_plus) * (y1+1)] 
                                                         + simplified_sum_data[(x1+1) + (nx_plus) * y1] - simplified_sum_data[x1 + (nx_plus) * y1];      
        }
    }
    float all_colors = simplified_sum_data[nx + (nx_plus) * ny];

    // Allocate memory & copy data to GPU
    float* dGPU = NULL;
    CHECK(cudaMalloc((void**)&dGPU, nx_plus * ny_plus * sizeof(float)));
    int* rGPU1 = NULL;
    float* rGPU2 = NULL;
    CHECK(cudaMalloc((void**)&rGPU1, 4 * nx_plus * ny_plus * sizeof(int)));
    CHECK(cudaMalloc((void**)&rGPU2, nx_plus * ny_plus * sizeof(float)));
    CHECK(cudaMemcpy(dGPU, simplified_sum_data.data(), nx_plus * ny_plus * sizeof(float), cudaMemcpyHostToDevice));

    // Run kernel
    dim3 dimBlock(8, 8);
    dim3 dimGrid(divup(ny_plus, dimBlock.x), divup(nx_plus, dimBlock.y));
    mykernel<<<dimGrid, dimBlock>>>(rGPU1, rGPU2, dGPU, nx, ny);
    CHECK(cudaGetLastError());

    // Copy data back to CPU & release memory
    CHECK(cudaMemcpy(coordinates.data(), rGPU1, 4 * ny_plus * nx_plus * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(maximum.data(), rGPU2, ny_plus * nx_plus * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(dGPU));
    CHECK(cudaFree(rGPU1));
    CHECK(cudaFree(rGPU2));    

    int maxElementIndex = std::max_element(maximum.begin(),maximum.end()) - maximum.begin();
    int y0_ret = coordinates[4 * maxElementIndex + 0];
    int x0_ret = coordinates[4 * maxElementIndex + 1];
    int y1_ret = coordinates[4 * maxElementIndex + 2];
    int x1_ret = coordinates[4 * maxElementIndex + 3];  

    float pixel_x = (y1_ret - y0_ret) * (x1_ret - x0_ret) ;
    float pixel_y = allpixels - pixel_x;
    float sum_of_colors_X, sum_of_colors_Y;
    sum_of_colors_X = simplified_sum_data[x1_ret + (nx_plus) * y1_ret] - simplified_sum_data[x0_ret + (nx_plus) * y1_ret] 
                    - simplified_sum_data[x1_ret + (nx_plus) * y0_ret] + simplified_sum_data[x0_ret + (nx_plus) * y0_ret];
    sum_of_colors_Y = all_colors - sum_of_colors_X;
    sum_of_colors_Y /= pixel_y;
    sum_of_colors_X /= pixel_x;
    Result result {y0_ret, x0_ret, y1_ret, x1_ret, {sum_of_colors_Y, sum_of_colors_Y, sum_of_colors_Y}, {sum_of_colors_X, sum_of_colors_X, sum_of_colors_X} };
    return result;
}