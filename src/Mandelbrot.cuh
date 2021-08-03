#pragma once
#include <complex>
#include <cuda_runtime.h>

#define MAX_ITERATIONS 10000

__device__ __host__ inline int getIterations(double x0, double y0)
{
    double x = 0.0;
    double y = 0.0;

    int iterations = 0;
    while (iterations < MAX_ITERATIONS && x * x + y * y <= 4)
    {
        double xTemp = x * x - y * y + x0;
        y = 2 * x * y + y0;
        x = xTemp;
        iterations++;
    }
    return iterations;
}


