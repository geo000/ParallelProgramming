#include "cp.h"
#include "math.h"
#include <stdio.h>
#include <vector>
#include <vector.h>
#include <numeric>

void correlate(int ny, int nx, const float* data, float* result) {
    constexpr int nb = 4;
    int vector_count = nx / nb;
    int values_rest = nx % nb;
    int nc = (ny + 3) / 4;
    int ncd = nc * 4; // rows after padding (blocks of 4)
    double4_t* paralell_vectors = double4_alloc(nx * ncd);
    for (int row = 0; row < ny ; row ++)
    {
        double4_t sum = {0.0, 0.0, 0.0, 0.0};
        double4_t square_sum = {0.0, 0.0, 0.0, 0.0};
        double sumsum = 0;
        double square_sumsum = 0;
        for (int count = 0; count <= vector_count; count++)
        {
            for (int i = 0; i < nb; i++)
            {
                int column = count * nb + i;
                if ((count == vector_count) && (i>= values_rest))
                {
                    paralell_vectors[vector_count*row + count + row][i] = 0;
                }
                else{
                    paralell_vectors[vector_count*row + count + row][i] = data[column + row * nx];
                }
            }
            double4_t x = paralell_vectors[vector_count*row + count + row];
            sum += x;
            square_sum += x * x;
        }
        for (int i = 0; i < nb; i++)
        {
            sumsum += sum[i];
            square_sumsum += square_sum[i];
        }
        double mean = sumsum / nx;
        double mult = 1 / std::sqrt((square_sumsum - sumsum*mean));

        for (int count = 0; count <= vector_count; count++)
        {
            paralell_vectors[vector_count*row + count + row] -= mean;
            for (int i = 0; i < nb; i++)
            {
                
                if ((count == vector_count) && (i>= values_rest))
                {
                    paralell_vectors[vector_count*row + count + row][i] = 0;
                }
            }
            paralell_vectors[vector_count*row + count + row] *= mult;
        }
    }

    for (int row = ny; row < ncd ; row ++)
    {
        for (int count = 0; count <= vector_count; count++)
        {
            for (int i = 0; i < nb; i++)
            {
                paralell_vectors[vector_count*row + count + row][i] = 0;
            }
        }
    }



    #pragma omp parallel for schedule(static, 1)
    for (int row1 = 0; row1 < ncd; row1+=4)
    {
        for (int row2 = row1; row2 < ncd; row2+=4)
            {
                double sumsum1 = 0;
                double4_t summa1 = {0.0, 0.0, 0.0, 0.0};
                double sumsum2 = 0;
                double4_t summa2 = {0.0, 0.0, 0.0, 0.0};
                double sumsum3 = 0;
                double4_t summa3 = {0.0, 0.0, 0.0, 0.0};
                double sumsum4 = 0;
                double4_t summa4 = {0.0, 0.0, 0.0, 0.0};
                double sumsum5 = 0;
                double4_t summa5 = {0.0, 0.0, 0.0, 0.0};
                double sumsum6 = 0;
                double4_t summa6 = {0.0, 0.0, 0.0, 0.0};
                double sumsum7 = 0;
                double4_t summa7 = {0.0, 0.0, 0.0, 0.0};
                double sumsum8 = 0;
                double4_t summa8 = {0.0, 0.0, 0.0, 0.0};
                double sumsum9 = 0;
                double4_t summa9 = {0.0, 0.0, 0.0, 0.0};
                double sumsum10 = 0;
                double4_t summa10 = {0.0, 0.0, 0.0, 0.0};
                double sumsum11 = 0;
                double4_t summa11 = {0.0, 0.0, 0.0, 0.0};
                double sumsum12 = 0;
                double4_t summa12 = {0.0, 0.0, 0.0, 0.0};
                double sumsum13 = 0;
                double4_t summa13 = {0.0, 0.0, 0.0, 0.0};
                double sumsum14 = 0;
                double4_t summa14 = {0.0, 0.0, 0.0, 0.0};
                double sumsum15 = 0;
                double4_t summa15 = {0.0, 0.0, 0.0, 0.0};
                double sumsum16 = 0;
                double4_t summa16 = {0.0, 0.0, 0.0, 0.0};
                for (int count = 0; count <= vector_count; count++)
                {
                    double4_t sor1 = paralell_vectors[vector_count*(row1) + count + row1];
                    double4_t sor2 = paralell_vectors[vector_count*(row1+1) + count + row1+1];
                    double4_t sor3 = paralell_vectors[vector_count*(row1+2)  + count + row1+2];
                    double4_t sor4 = paralell_vectors[vector_count*(row1+3)  + count + row1+3];
                    double4_t oszlop1 = paralell_vectors[vector_count*(row2) + count + row2];
                    double4_t oszlop2 = paralell_vectors[vector_count*(row2+1) + count + row2+1];
                    double4_t oszlop3 = paralell_vectors[vector_count*(row2+2) + count + row2+2];
                    double4_t oszlop4 = paralell_vectors[vector_count*(row2+3) + count + row2+3];
                    summa1 += sor1 * oszlop1;
                    summa2 += sor2 * oszlop2;
                    summa3 += sor3 * oszlop3;
                    summa4 += sor4 * oszlop4;

                    summa5 += sor1 * oszlop2;
                    summa6 += sor1 * oszlop3;
                    summa7 += sor1 * oszlop4;
                    summa8 += sor2 * oszlop1;

                    summa9 += sor2 * oszlop3;
                    summa10 += sor2 * oszlop4;
                    summa11 += sor3 * oszlop1;
                    summa12 += sor3 * oszlop2;

                    summa13 += sor3 * oszlop4;
                    summa14 += sor4 * oszlop1;
                    summa15 += sor4 * oszlop2;
                    summa16 += sor4 * oszlop3;
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
                        result[row2 + row1 * ny] = (float) sumsum1;
                    }
                    if (row2+1 < ny)
                    {
                        result[row2+1 + row1 * ny] = (float) sumsum5;
                    }
                    if (row2+2 < ny)
                    {
                        result[row2+2 + row1 * ny] = (float) sumsum6;
                    }
                    if (row2+3 < ny)
                    {
                        result[row2+3 + row1 * ny] = (float) sumsum7;
                    }
                }
                if (row1+1 < ny)
                {
                    if (row2 < ny)
                    {
                        result[row2 + (row1+1) * ny] = (float) sumsum8;
                    }
                    if (row2+1 < ny)
                    {
                        result[row2+1 + (row1+1) * ny] = (float) sumsum2;
                    }
                    if (row2+2 < ny)
                    {
                        result[row2+2 + (row1+1) * ny] = (float) sumsum9;
                    }
                    if (row2+3 < ny)
                    {
                        result[row2+3 + (row1+1) * ny] = (float) sumsum10;
                    }
                }
                if (row1+2 < ny)
                {
                    if (row2 < ny)
                    {
                        result[row2 + (row1+2) * ny] = (float) sumsum11;
                    }
                    if (row2+1 < ny)
                    {
                        result[row2+1 + (row1+2) * ny] = (float) sumsum12;
                    }
                    if (row2+2 < ny)
                    {
                        result[row2+2 + (row1+2) * ny] = (float) sumsum3;
                    }
                    if (row2+3 < ny)
                    {
                        result[row2+3 + (row1+2) * ny] = (float) sumsum13;
                    }
                }
                if (row1+3 < ny)
                {
                    if (row2 < ny)
                    {
                        result[row2 + (row1+3) * ny] = (float) sumsum14;
                    }
                    if (row2+1 < ny)
                    {
                        result[row2+1 + (row1+3) * ny] = (float) sumsum15;
                    }
                    if (row2+2 < ny)
                    {
                        result[row2+2 + (row1+3) * ny] = (float) sumsum16;
                    }
                    if (row2+3 < ny)
                    {
                        result[row2+3 + (row1+3) * ny] = (float) sumsum4;
                    }
                }
            }
    }



std::free(paralell_vectors);
}

