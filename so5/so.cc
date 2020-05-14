#include "so.h"
#include <algorithm>
#include <vector>
#include <omp.h>
#include <stdio.h>
#include<iostream> 
using namespace std;
data_t middleOfThree(data_t a, data_t b, data_t c) 
{ 
    // x is positive if a is greater than b.  
    // x is negative if b is greater than a. 
    data_t x = a - b; 
  
    data_t y = b - c;  // Similar to x 
    data_t z = a - c;  // Similar to x and y. 
  
    // Checking if b is middle (x and y both 
    // are positive) 
    if (x * y > 0) 
        return b; 
  
    // Checking if c is middle (x and z both 
    // are positive) 
    else if (x * z > 0) 
        return c; 
    else
        return a; 
} 

void quicksort(data_t* first, data_t* last, int max_thread){

    // if number of threads < 2 or number of elements <= 1
    if (max_thread < 2 || (last-first) < 1)
    {
        sort(first, last+1);
        return;
    }
    data_t pivot = middleOfThree(*first, *(last-(last-first)/2), *last);


    /*
        PIVOT CHOOSING METHODS
        <-------------------->
            1. time = 2.84s - TRIAL AND ERROR#1 -->  *(last-(last-first)/2);
            2. time = 2.99s - MEDIAN            -->  middleOfThree(*first, *(last-(last-first)/2), *last);
            3. time = 4.8s  - RANDOM            --> *(first + rand() % (last - first));
        <-------------------->
    */


    // elements < pivot : [first, it] 'it' is the pivot
    data_t* it = partition(first, last + 1, [pivot](data_t element){ return element < pivot; });
    // elements == pivot : [it, it2] it2 is > pivot
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