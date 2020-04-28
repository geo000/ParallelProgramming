#include "is.h"
#include "math.h"
#include <stdio.h>
#include <vector>
#include <numeric>

Result segment(int ny, int nx, const float* data) {
    std::vector<double> interm_data((nx+1) * (ny+1) * 3);
    double best_so_far = 0;
    double return_data[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    for (int x1 = 0; x1 <= nx; x1++)
    {
        for (int y1 = 0; y1 <= ny; y1++)
        {
            for (int x = 0; x < x1; x++)
            {
                for (int y = 0; y < y1; y++) 
                {
                    for (int c = 0; c < 3; c++)
                    {
                        interm_data[c + 3 * x1 + 3 * (nx+1) * y1] += data[c + 3 * x + 3 * nx * y];
                    }
                }
            }
        }
    }

    for (int x0 = 0; x0 < nx; x0++)
    {
        for (int y0 = 0; y0 < ny; y0++)
        {
            for (int x1 = x0 + 1; x1 <= nx; x1++)
            {
                for (int y1 = y0 + 1; y1 <= ny; y1++)
                {
                    double pixel_x = (y1 - y0) * (x1 - x0);
                    double pixel_y = nx*ny - pixel_x;
                    double best = 0;
                    double sum_of_colors_X[3];
                    double sum_of_colors_Y[3];
                    for (int c = 0; c < 3; c++)
                    {
                        sum_of_colors_X[c] = interm_data[c + 3 * x1 + 3 * (nx+1) * y1] - interm_data[c + 3 * x0 + 3 * (nx+1) * y1] 
                        - interm_data[c + 3 * x1 + 3 * (nx+1) * y0] + interm_data[c + 3 * x0 + 3 * (nx+1) * y0];
                        sum_of_colors_Y[c] = interm_data[c + 3 * nx + 3 * (nx+1) * ny] - sum_of_colors_X[c];
                        best += ( (sum_of_colors_X[c] * sum_of_colors_X[c]) / pixel_x ) + ( (sum_of_colors_Y[c] * sum_of_colors_Y[c]) / pixel_y );
                    }
                    if (best > best_so_far)
                    {
                        best_so_far = best;
                        return_data[0] = y0;
                        return_data[1] = x0;
                        return_data[2] = y1;
                        return_data[3] = x1;
                        return_data[4] = sum_of_colors_Y[0] / pixel_y;
                        return_data[5] = sum_of_colors_Y[1] / pixel_y;
                        return_data[6] = sum_of_colors_Y[2] / pixel_y;
                        return_data[7] = sum_of_colors_X[0] / pixel_x;
                        return_data[8] = sum_of_colors_X[1] / pixel_x;
                        return_data[9] = sum_of_colors_X[2] / pixel_x;
                    }
                }
            }
        }
    }
    Result result { (int)return_data[0], (int)return_data[1], (int)return_data[2], (int)return_data[3], {(float)return_data[4], (float)return_data[5], (float)return_data[6]}, {(float)return_data[7], (float)return_data[8], (float)return_data[9]} };
    return result;
}