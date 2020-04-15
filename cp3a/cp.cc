#include "cp.h"
#include "math.h"
#include <stdio.h>
#include <vector>
#include <vector.h>
#include <numeric>

void correlate(int ny, int nx, const float* data, float* result) {
    std::vector<double> interm_data(nx * ny);
    constexpr int nb = 4;
    int vector_count = nx / nb;
    int values_rest = nx % nb;
    double4_t* paralell_vectors = double4_alloc(nx * ny);

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

        for (int count = 0; count < vector_count; count++)
        {
            for (int i = 0; i < nb; i++)
            {
                int column = count * nb + i;
                paralell_vectors[vector_count*row + count + row][i] = interm_data[column + row * nx];
            }
        }
        for (int j = 0; j < values_rest; j++)
        {
            int column = vector_count * nb + j;
            paralell_vectors[(vector_count+1)*row + vector_count][j] = interm_data[column + row * nx];
        }
    }
    #pragma omp parallel for schedule(static, 1)
    for (int row1 = 0; row1 < ny; row1++)
    {
        for (int row2 = row1; row2 < ny; row2++)
            {
                double sumsum = 0;
                double4_t summa = {0.0, 0.0, 0.0, 0.0};
                for (int count = 0; count < vector_count; count++)
                {
                    summa += paralell_vectors[vector_count*row1 + count + row1] * paralell_vectors[vector_count*row2 + count + row2];
                }
                for (int i = 0; i < nb; i++)
                {
                    sumsum += summa[i];
                }
                for (int j = 0; j < values_rest; j++)
                {
                    sumsum += paralell_vectors[(vector_count+1)*row1 + vector_count][j] * paralell_vectors[(vector_count+1)*row2 + vector_count][j];
                }
                result[row2 + row1 * ny] = (float) sumsum;
            }
    }
std::free(paralell_vectors);
}