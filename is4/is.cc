#include "is.h"
#include "math.h"
#include <stdio.h>
#include <vector>
#include <numeric>
#include "vector.h"
#include <sys/time.h>
#include <iostream>
#include <omp.h>

static double get_time() {
    struct timeval tm;
    gettimeofday(&tm, NULL);
    return static_cast<double>(tm.tv_sec)
        + static_cast<double>(tm.tv_usec) / 1E6;
}

Result segment(int ny, int nx, const float* data) {
    double4_t* interm_data_vector = double4_alloc((nx+1) * (ny+1));
    double best_so_far = -1;
    constexpr int szin = 3;
    int x0_ret = 0, y0_ret = 0, x1_ret = 0, y1_ret = 0;
    double4_t all_colors_vector, best4, sum_of_colors_X, sum_of_colors_Y;
    double allpixels = nx*ny;
    double t0 = get_time();
    for (int y1 = 0; y1 <= ny; y1++)
    {
        for (int x1 = 0; x1 <= nx; x1++)
        {
            interm_data_vector[x1 + (nx+1) * y1] = double4_0;
            for (int y = 0; y < y1; y++)
            {
                for (int x = 0; x < x1; x++) 
                {
                    for (int c = 0; c < szin; c++)
                    {
                        interm_data_vector[x1 + (nx+1) * y1][c] += data[c + 3 * ( x + nx * y )];
                    }
                }
            }
        }
    }
    double t1 = get_time();
    std::cout << "preprocess took " << t1 - t0 << " seconds" << std::endl;
    double t2 = get_time();
    all_colors_vector = interm_data_vector[nx + (nx+1) * ny];

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
                    if (best > best_so_far)
                    {
                        best_so_far = best;
                        y0_ret = y0;
                        x0_ret = x0;
                        y1_ret = y1;
                        x1_ret = x1;
                    }
                }
            }
        }
    }
    double t3 = get_time();
    std::cout << "rectangles took " << t3 - t2 << " seconds" << std::endl;
    double pixel_x = (y1_ret - y0_ret) * (x1_ret - x0_ret) ;
    double pixel_y = allpixels - pixel_x;
    sum_of_colors_X = interm_data_vector[x1_ret + (nx+1) * y1_ret] - interm_data_vector[x0_ret + (nx+1) * y1_ret] 
                    - interm_data_vector[x1_ret + (nx+1) * y0_ret] + interm_data_vector[x0_ret + (nx+1) * y0_ret];
    sum_of_colors_Y = all_colors_vector - sum_of_colors_X;
    sum_of_colors_Y /= pixel_y;
    sum_of_colors_X /= pixel_x;
    Result result {y0_ret, x0_ret, y1_ret, x1_ret, {(float)sum_of_colors_Y[0], (float)sum_of_colors_Y[1], (float)sum_of_colors_Y[2]}, {(float)sum_of_colors_X[0], (float)sum_of_colors_X[1], (float)sum_of_colors_X[2]} };
    free(interm_data_vector);
    return result;
}