#pragma once
#include <cuda_runtime.h>
#include "Constants.h"
__device__ __host__ void inline doZoom(long long x, long long y, double Scale_, double XCenter_, double YCenter_, double &xFractal, double &yFractal)
{
    xFractal = (x - WIDTH / 2) * Scale_ + XCenter_;
    yFractal = (y - HEIGHT / 2) * Scale_ + YCenter_;
    
}