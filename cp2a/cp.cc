#include "cp.h"
#include "math.h"
#include <stdio.h>
#include <vector>
#include <numeric>

void correlate(int ny, int nx, const float* data, float* result) {
    int row, column, i, j, k1, k2, na;
    constexpr int nb = 4;
    na = nx % nb;
    double x, mean, sum, square_sum;
    std::vector<double> interm_data(nx * ny);
    for (row = 0; row < ny ; row ++)
    {
        sum = 0;
        for (column = 0; column < nx; column ++)
        {
            sum += data[column + row * nx];
        }
        mean = sum / nx;
        square_sum = 0;
        for (column = 0; column < nx; column ++)
        {
            x = data[column + row * nx] - mean;
            interm_data[column + row * nx] = x;
            square_sum += x * x;
        }
        square_sum = sqrt(square_sum);
        for(column = 0; column < nx; column++) {
            interm_data[column + row * nx] /= square_sum;
        }
    }

    for (i = 0; i < ny ; i++)
    {
        for (j = i; j < ny; j++) 
        {
            double sum_dat[nb] = {0};
            x = 0;
            for (k1 = 0; k1 < (nx / nb); k1++)
            {
                for (k2 = 0; k2 < nb; k2++)
                {
                    sum_dat[k2] += interm_data[4*k1+k2 + i * nx] * interm_data[4*k1+k2 + j * nx];
                }
            }
            for (k1 = 0; k1 < na; k1++)
            {
                x += interm_data[(nx-na)+k1 + i * nx] * interm_data[(nx-na)+k1 + j * nx];
            }
            for (k2 = 0; k2 < nb; k2++) {
                x += sum_dat[k2];
            }
            result[j + i * ny] = (float)x;
        }
    }
}

