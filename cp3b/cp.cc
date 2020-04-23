#include "cp.h"
#include "math.h"
#include <stdio.h>
#include <vector>
#include <vector.h>
#include <numeric>

void correlate(int ny, int nx, const float* data, float* result) {
    constexpr int nb = 8;
    int vector_count = nx / nb;
    int values_rest = nx % nb;
    int nc = (ny + 3) / 4;
    int ncd = nc * 4; // rows after padding (blocks of 4)
    float8_t* paralell_vectors = float8_alloc(nx * ncd);
    #pragma omp parallel for schedule(static, 1)
    for (int row = 0; row < ny ; row ++)
    {
        float8_t sum = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        float8_t square_sum = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        float sumsum = 0;
        float square_sumsum = 0;
        for (int count = 0; count <= vector_count; count++)
        {
            for (int i = 0; i < nb; i++)
            {
                int column = count * nb + i;
                if ((count == vector_count) && (i>= values_rest))
                {
                    paralell_vectors[vector_count*row + count + row][i] = 0;    // padding with 0s
                }
                else{
                    paralell_vectors[vector_count*row + count + row][i] = data[column + row * nx]; // parallelize input data 
                }
            }
            float8_t x = paralell_vectors[vector_count*row + count + row];
            sum += x;
        }
        for (int i = 0; i < nb; i++)
        {
            sumsum += sum[i];   // calculate sum
        }
        float mean = sumsum / nx; // calculate mean

        for (int count = 0; count <= vector_count; count++)
        {
            paralell_vectors[vector_count*row + count + row] -= mean;
            for (int i = 0; i < nb; i++)
            {
                if ((count == vector_count) && (i>= values_rest))
                {
                    paralell_vectors[vector_count*row + count + row][i] = 0; // padding with 0s again
                }
            }
            square_sum += paralell_vectors[vector_count*row + count + row] * paralell_vectors[vector_count*row + count + row]; 
        }
        for (int i = 0; i < nb; i++)
        {
            square_sumsum += square_sum[i]; // calculating square sum
        }
        square_sumsum = sqrt(square_sumsum);
        for (int count = 0; count <= vector_count; count++)
        {
            paralell_vectors[vector_count*row + count + row] /= square_sumsum; // final normalization
        }
    }
    #pragma omp parallel for schedule(static, 1)
    for (int row = ny; row < ncd ; row ++)
    {
        for (int count = 0; count <= vector_count; count++)
        {
            for (int i = 0; i < nb; i++)
            {
                paralell_vectors[vector_count*row + count + row][i] = 0; // padding rows with 0s to have rows%4 = 0
            }
        }
    }

    #pragma omp parallel for schedule(static, 1)
    for (int row1 = 0; row1 < ncd; row1+=4)
    {
        float8_t sor[4], oszlop[4];
        for (int row2 = row1; row2 < ncd; row2+=4)
            {
                float8_t summa1 = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                float8_t summa2 = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                float8_t summa3 = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                float8_t summa4 = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                float8_t summa5 = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                float8_t summa6 = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                float8_t summa7 = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                float8_t summa8 = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                float8_t summa9 = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                float8_t summa10 = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                float8_t summa11 = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                float8_t summa12 = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                float8_t summa13 = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                float8_t summa14 = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                float8_t summa15 = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                float8_t summa16 = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                float sumsum1 = 0;
                float sumsum2 = 0;
                float sumsum3 = 0;
                float sumsum4 = 0;
                float sumsum5 = 0;
                float sumsum6 = 0;
                float sumsum7 = 0;
                float sumsum8 = 0;
                float sumsum9 = 0;
                float sumsum10 = 0;
                float sumsum11 = 0;
                float sumsum12 = 0;
                float sumsum13 = 0;
                float sumsum14 = 0;
                float sumsum15 = 0;
                float sumsum16 = 0;
                for (int count = 0; count <= vector_count; count++)
                {
                    for (int i = 0; i < 4; i++)
                    {
                        sor[i] = paralell_vectors[vector_count*(row1 + i) + count + row1 + i];
                        oszlop[i] = paralell_vectors[vector_count*(row2 + i) + count + row2 + i];
                    }
                    summa1 += sor[0] * oszlop[0];
                    summa2 += sor[1] * oszlop[1];
                    summa3 += sor[2] * oszlop[2];
                    summa4 += sor[3] * oszlop[3];
                    summa5 += sor[0] * oszlop[1];
                    summa6 += sor[0] * oszlop[2];
                    summa7 += sor[0] * oszlop[3];
                    summa8 += sor[1] * oszlop[0];
                    summa9 += sor[1] * oszlop[2];
                    summa10 += sor[1] * oszlop[3];
                    summa11 += sor[2] * oszlop[0];
                    summa12 += sor[2] * oszlop[1];
                    summa13 += sor[2] * oszlop[3];
                    summa14 += sor[3] * oszlop[0];
                    summa15 += sor[3] * oszlop[1];
                    summa16 += sor[3] * oszlop[2];
                }
                for (int i = 0; i < nb; i++)
                {
                    sumsum1 += summa1[i];
                    sumsum2 += summa2[i];
                    sumsum3 += summa3[i];
                    sumsum4 += summa4[i];
                    sumsum5 += summa5[i];
                    sumsum6 += summa6[i];
                    sumsum7 += summa7[i];
                    sumsum8 += summa8[i];
                    sumsum9 += summa9[i];
                    sumsum10 += summa10[i];
                    sumsum11 += summa11[i];
                    sumsum12 += summa12[i];
                    sumsum13 += summa13[i];
                    sumsum14 += summa14[i];
                    sumsum15 += summa15[i];
                    sumsum16 += summa16[i];
                }
                if (row1 < ny)
                {
                    if (row2 < ny)
                    {
                        result[row2 + row1 * ny] = sumsum1;
                    }
                    if (row2+1 < ny)
                    {
                        result[row2+1 + row1 * ny] = sumsum5;
                    }
                    if (row2+2 < ny)
                    {
                        result[row2+2 + row1 * ny] = sumsum6;
                    }
                    if (row2+3 < ny)
                    {
                        result[row2+3 + row1 * ny] = sumsum7;
                    }
                }
                if (row1+1 < ny)
                {
                    if (row2 < ny)
                    {
                        result[row2 + (row1+1) * ny] = sumsum8;
                    }
                    if (row2+1 < ny)
                    {
                        result[row2+1 + (row1+1) * ny] = sumsum2;
                    }
                    if (row2+2 < ny)
                    {
                        result[row2+2 + (row1+1) * ny] = sumsum9;
                    }
                    if (row2+3 < ny)
                    {
                        result[row2+3 + (row1+1) * ny] = sumsum10;
                    }
                }
                if (row1+2 < ny)
                {
                    if (row2 < ny)
                    {
                        result[row2 + (row1+2) * ny] = sumsum11;
                    }
                    if (row2+1 < ny)
                    {
                        result[row2+1 + (row1+2) * ny] = sumsum12;
                    }
                    if (row2+2 < ny)
                    {
                        result[row2+2 + (row1+2) * ny] = sumsum3;
                    }
                    if (row2+3 < ny)
                    {
                        result[row2+3 + (row1+2) * ny] = sumsum13;
                    }
                }
                if (row1+3 < ny)
                {
                    if (row2 < ny)
                    {
                        result[row2 + (row1+3) * ny] = sumsum14;
                    }
                    if (row2+1 < ny)
                    {
                        result[row2+1 + (row1+3) * ny] = sumsum15;
                    }
                    if (row2+2 < ny)
                    {
                        result[row2+2 + (row1+3) * ny] = sumsum16;
                    }
                    if (row2+3 < ny)
                    {
                        result[row2+3 + (row1+3) * ny] = sumsum4;
                    }
                }
            }
    }
std::free(paralell_vectors);
}

