#include "is.h"
#include "math.h"
#include <stdio.h>
#include <vector>
#include <numeric>
#include "vector.h"
#include <iostream>
#include <sys/time.h>
#include <omp.h>
Result segment(int ny, int nx, const float* data) {
    float allpixels = nx*ny;
    float best_so_far = -1;
    int x0_ret = 0, y0_ret = 0, x1_ret = 0, y1_ret = 0;
    std::vector<float> simplified_sum_data((nx+1) * (ny+1));
    for (int y1 = 0; y1 < ny; y1++) {
        for (int x1 = 0; x1 < nx; x1++) {
            simplified_sum_data[(x1+1) + (nx+1) * (y1+1)] = data[3 * (x1 + nx * y1)] + simplified_sum_data[x1 + (nx+1) * (y1+1)] 
                                                         + simplified_sum_data[(x1+1) + (nx+1) * y1] - simplified_sum_data[x1 + (nx+1) * y1];      
        }
    }
    float all_colors = simplified_sum_data[nx + (nx+1) * ny];
    #pragma omp parallel
    {
        float best_so_far_inner = -1;
        int x0_inner = 0, y0_inner = 0, x1_inner = 0, y1_inner = 0;
        for (int height = 1; height <= ny; height++)
        {
            #pragma omp for schedule(dynamic)
            for (int width = 1; width <= nx; width++)
            {
                float pixel_x = height * width;
                float pixel_y = allpixels - pixel_x;
                pixel_x = 1 / pixel_x;
                pixel_y = 1 / pixel_y;
                for (int y0 = 0; y0 <= ny - height; y0++)
                {
                    for (int x0 = 0; x0 <= nx - width; x0++)
                    {
                        int y1 = y0 + height;
                        int x1 = x0 + width;

                        float sum_of_colors_X = simplified_sum_data[x1 + (nx+1) * y1] - simplified_sum_data[x0 + (nx+1) * y1] 
                                        - simplified_sum_data[x1 + (nx+1) * y0] + simplified_sum_data[x0 + (nx+1) * y0];
                        float sum_of_colors_Y = all_colors - sum_of_colors_X;
                        float best = sum_of_colors_X * sum_of_colors_X * pixel_x + sum_of_colors_Y * sum_of_colors_Y * pixel_y;
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
    float pixel_x = (y1_ret - y0_ret) * (x1_ret - x0_ret) ;
    float pixel_y = allpixels - pixel_x;
    float sum_of_colors_X, sum_of_colors_Y;
    sum_of_colors_X = simplified_sum_data[x1_ret + (nx+1) * y1_ret] - simplified_sum_data[x0_ret + (nx+1) * y1_ret] 
                    - simplified_sum_data[x1_ret + (nx+1) * y0_ret] + simplified_sum_data[x0_ret + (nx+1) * y0_ret];
    sum_of_colors_Y = all_colors - sum_of_colors_X;
    sum_of_colors_Y /= pixel_y;
    sum_of_colors_X /= pixel_x;
    Result result {y0_ret, x0_ret, y1_ret, x1_ret, {sum_of_colors_Y, sum_of_colors_Y, sum_of_colors_Y}, {sum_of_colors_X, sum_of_colors_X, sum_of_colors_X} };
    return result;
}