#include "is.h"
#include "math.h"
#include <stdio.h>
#include <vector>
#include <numeric>
#include "vector.h"
#include <iostream>
#include <omp.h>

Result segment(int ny, int nx, const float* data) {
    double allpixels = nx*ny;
    double4_t* interm_data_vector = double4_alloc((nx+1) * (ny+1));
    double4_t* vectorized_data = double4_alloc(allpixels);
    double best_so_far = -1;
    int x0_ret = 0, y0_ret = 0, x1_ret = 0, y1_ret = 0;
    double4_t all_colors_vector;

    #pragma omp for schedule(dynamic)
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            for (int c = 0; c < 3; c++) {
                vectorized_data[x + nx * y][c] = data[c + 3 * (x + nx * y)];
            }
        }
    }
    for (int y = 0; y < ny+1; y++)
    {
        for (int x = 0; x < nx+1; x++) 
        {
            interm_data_vector[(nx+1) * y + x] = double4_0;
        }
    }
    for (int y1 = 0; y1 < ny; y1++) {
        for (int x1 = 0; x1 < nx; x1++) {
            interm_data_vector[(y1+1) * (nx+1) + (x1+1)] = vectorized_data[y1 * nx + x1] + interm_data_vector[(y1+1) * (nx+1) + x1] + interm_data_vector[y1 * (nx+1) + (x1+1)]
                                                               - interm_data_vector[y1 * (nx+1) + x1];      
        }
    }

    all_colors_vector = interm_data_vector[nx + (nx+1) * ny];
    #pragma omp parallel
    {
        double best_so_far_inner = -1;
        int x0_inner = 0, y0_inner = 0, x1_inner = 0, y1_inner = 0;
        double4_t best4, sum_of_colors_X, sum_of_colors_Y;
        #pragma omp for nowait
        for (int height = 1; height <= ny; height++)
        {
            for (int width = 1; width <= nx; width++)
            {
                double pixel_x = height * width;
                double pixel_y = allpixels - pixel_x;
                pixel_x = 1 / pixel_x;
                pixel_y = 1 / pixel_y;
                for (int y0 = 0; y0 <= ny - height; y0++)
                {
                    for (int x0 = 0; x0 <= nx - width; x0++)
                    {
                        int y1 = y0 + height;
                        int x1 = x0 + width;
                        double best = 0;
                        sum_of_colors_X = interm_data_vector[x1 + (nx+1) * y1] - interm_data_vector[x0 + (nx+1) * y1] 
                                        - interm_data_vector[x1 + (nx+1) * y0] + interm_data_vector[x0 + (nx+1) * y0];
                        sum_of_colors_Y = all_colors_vector - sum_of_colors_X;
                        best4 = sum_of_colors_X * sum_of_colors_X * pixel_x + sum_of_colors_Y * sum_of_colors_Y * pixel_y;
                        best = best4[0] + best4[1] + best4[2];
                        if (best > best_so_far_inner)
                        {
                            best_so_far_inner = best;
                            y0_inner = y0;
                            x0_inner = x0;
                            y1_inner = y1;
                            x1_inner = x1;
                        }
                    }
                }
            }
        }
        #pragma omp critical
        {
            if (best_so_far_inner > best_so_far)
            {
                best_so_far = best_so_far_inner;
                y0_ret = y0_inner;
                x0_ret = x0_inner;
                y1_ret = y1_inner;
                x1_ret = x1_inner;
            }
        }
    }
    double pixel_x = (y1_ret - y0_ret) * (x1_ret - x0_ret) ;
    double pixel_y = allpixels - pixel_x;
    double4_t sum_of_colors_X, sum_of_colors_Y;
    sum_of_colors_X = interm_data_vector[x1_ret + (nx+1) * y1_ret] - interm_data_vector[x0_ret + (nx+1) * y1_ret] 
                    - interm_data_vector[x1_ret + (nx+1) * y0_ret] + interm_data_vector[x0_ret + (nx+1) * y0_ret];
    sum_of_colors_Y = all_colors_vector - sum_of_colors_X;
    sum_of_colors_Y /= pixel_y;
    sum_of_colors_X /= pixel_x;
    Result result {y0_ret, x0_ret, y1_ret, x1_ret, {(float)sum_of_colors_Y[0], (float)sum_of_colors_Y[1], (float)sum_of_colors_Y[2]}, {(float)sum_of_colors_X[0], (float)sum_of_colors_X[1], (float)sum_of_colors_X[2]} };
    
    free(interm_data_vector);
    free(vectorized_data);

    return result;
}