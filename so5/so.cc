#include "so.h"
#include <algorithm>
#include <vector>
#include <omp.h>
#include <stdio.h>
#include<iostream> 

void quicksort(data_t* first, data_t* last, int max_thread){

    max_thread = max_thread / 2;

    if (max_thread < 2 || (last-first) < 3)
    {
        std::sort(first, last+1);
        return;
    }

    data_t pivot = *first;

    // elements < pivot : *it1 = [data + first, it] it is the pivot
    auto it = std::partition(first, last + 1, [pivot](auto element){ return element < pivot; });

    // elements = pivot : *it1 = [it, it2] it2 is > pivot
    auto it2 = std::partition(it, last + 1, [pivot](auto element){ return element == pivot; });
    // elements > pivot : *it1 = [it2, data + last]


    #pragma omp task
    quicksort(first, it, max_thread);
    #pragma omp task
    quicksort(it2, last, max_thread);
}

void psort(int n, data_t* data) {
    int max_thread = omp_get_max_threads() * 2;
    #pragma omp parallel
    #pragma omp single
    quicksort(data + 0, data + n-1, max_thread);
}
