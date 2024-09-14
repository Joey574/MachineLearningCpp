#pragma once

#include <cuda_runtime.h>
#include <vector>

__global__ void forward_prop(float* weights, float* results, std::vector<int> dims);

__device__ void dot_prod(float* mat_a, float* mat_b, float* results, int a_r, int a_c, int b_c);

__global__ void element_add(float* mat_a, float* mat_b, float* results, int r, int c);