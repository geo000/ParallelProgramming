#include "cp.h"
#include "math.h"
#include <stdio.h>
#include <vector>
#include <numeric>

void correlate(int ny, int nx, const float* data, float* result) {
    std::vector<double> interm_data(nx * ny);
    for (int row = 0; row < ny ; row ++)
    {
        double sum = 0;
        for (int column = 0; column < nx; column ++)
        {
            sum += data[column + row * nx];
        }
        double mean = sum / nx;
        double square_sum = 0;
        for (int column = 0; column < nx; column ++)
        {
            double x = data[column + row * nx] - mean;
            interm_data[column + row * nx] = x;
            square_sum += x * x;
        }
        square_sum = sqrt(square_sum);
        for(int column = 0; column < nx; column++) {
            interm_data[column + row * nx] /= square_sum;
        }
    }
    for (int i = 0; i < ny ; i++)
    {
        for (int j = i; j < ny; j++) 
        {
            double sumx = 0;
            for (int k = 0; k < nx; k++)
            {
                sumx += interm_data[k + i * nx] * interm_data[k + j * nx];
            }
            result[j + i * ny] = (float) sumx;
        }
    }
}