#include "mf.h"
#include <stdio.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>
#include <numeric>

void mf(int ny, int nx, int hy, int hx, const float* in, float* out) {

    #pragma omp parallel for schedule(static, 1)
    for (int row = 0; row < ny; row++)
    {
        for (int column = 0; column < nx; column++)
        {        
            int jmin = row - hy;
            if ((row - hy) <  0) jmin = 0;
            int imin = column - hx;
            if ((column - hx) <  0) imin = 0;
            int jmax = row + hy;
            if ((row + hy) >=  ny) jmax = ny - 1;
            int imax = column + hx;
            if ((column + hx) >=  nx) imax = nx - 1;

            std::vector<float> interm_data(((imax - imin) + 1) * ((jmax - jmin + 1)));
            interm_data.clear();
            for (int j = jmin; j <= jmax; j++)
            {
                for (int i = imin; i <= imax; i++)
                {
                    interm_data.push_back(in[i + nx * j]);
                }
            }
            int size = interm_data.size();
            int middle_num = size / 2;
            std::nth_element(interm_data.begin(), interm_data.begin() + middle_num, interm_data.end());
            float median = interm_data[middle_num];
            if(size % 2 == 1)
            {
                out[column + nx * row] = median;
            }
            else
            {
                std::nth_element(interm_data.begin(), interm_data.begin() + middle_num - 1, interm_data.end());
                out[column + nx * row] = 0.5 * (median + interm_data[middle_num - 1]);
            }
        }
    }
}