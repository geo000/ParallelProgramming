#include "cp.h"
#include "math.h"
#include <stdio.h>
#include <vector>
#include <numeric>

void correlate(int ny, int nx, const float* data, float* result) {
    int row, column, i, j, k;
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
        square_sum =  sqrt(square_sum);
        for(column = 0; column < nx; column++) {
            interm_data[column + row * nx] /= square_sum;
        }
    }
    for (i = 0; i < ny ; i++)
    {
        for (j = i; j < ny; j++) 
        {
            x = 0;
            for (k = 0; k < nx; k++)
            {
                x +=  interm_data[k + i * nx] * interm_data[k + j * nx];
            }
            result[j + i * ny] = (float)x;
        }
    }
}