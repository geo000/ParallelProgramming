#include "so.h"
#include <algorithm>
#include "math.h"
#include <stdio.h>
#include <vector>
#include <numeric>
#include <omp.h>
#include<iostream>
using namespace std;

int nearestpwr2 (int x)
{
    x = x | (x >> 1);
    x = x | (x >> 2);
    x = x | (x >> 4);
    x = x | (x >> 8);
    x = x | (x >> 16);
    return x - (x >> 1);
}

void print(data_t* data, int n)
{
	for(int i = 0; i < n; i++)
    {
		cout << data[i] << " ";
	    cout << endl;
    }
}
void mergeSort(data_t* data, int input_size) 
{
    int max_thread = omp_get_max_threads();
    int round_down = nearestpwr2(max_thread);
    if (round_down < 2)
    {
        sort(data, data + input_size);
        return;
    }

    int block_size = input_size / round_down;
    if(input_size % round_down) block_size++;

    #pragma omp parallel num_threads(round_down)
    {
        int thread_num = omp_get_thread_num();
        int thread_num_next = thread_num + 1;
        sort(data + min(thread_num * block_size, input_size), data + min(thread_num_next * block_size, input_size));
    }
    round_down >>= 1;
    do
    {   
        #pragma omp parallel num_threads(round_down)
        {
            int thread_num = omp_get_thread_num() << 1;
            int thread_num_middle = thread_num + 1;
            int thread_num_next = thread_num + 2;
            inplace_merge(data + thread_num * block_size, data + min(thread_num_middle * block_size, input_size),
                          data + min(thread_num_next * block_size, input_size));
        }
    block_size <<= 1;
    round_down >>= 1;
    }while(round_down > 0);
}

void psort(int n, data_t* data) 
{
    mergeSort(data, n);
}
