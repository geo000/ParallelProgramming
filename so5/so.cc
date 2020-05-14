#include "so.h"
#include <algorithm>
#include <vector>
#include <omp.h>
#include <stdio.h>
#include<iostream> 
using namespace std;
void quicksort(data_t* first, data_t* last, int max_thread){

    // if number of threads < 2 or number of elements <= 2
    if (max_thread < 2 || (last-first) < 2)
    {
        sort(first, last+1);
        return;
    }
    data_t pivot = *(last-(last-first)/2);
    // *(last-(last-first)/2); // time = 2.84
    //*(first + rand() % (last - first)); // time = 4.8s
    //max(min(*first,*(first + (last-first)/2)), min(max(*first,*(first + (last-first)/2)),*last)); // time = 3.78s

    // elements < pivot : [first, it] 'it' is the pivot
    data_t* it = partition(first, last + 1, [pivot](data_t element){ return element < pivot; });
    // elements = pivot : [it, it2] it2 is > pivot
    data_t* it2 = partition(it, last + 1, [pivot](data_t element){ return element == pivot; });
    // elements > pivot :  [it2, last]
    #pragma omp task
    quicksort(first, it, max_thread / 2);
    #pragma omp task
    quicksort(it2, last, max_thread / 2);
}

void psort(int n, data_t* data) {
    int max_thread = omp_get_max_threads();
    #pragma omp parallel
    #pragma omp single
    quicksort(data + 0, data + n-1, max_thread);
}