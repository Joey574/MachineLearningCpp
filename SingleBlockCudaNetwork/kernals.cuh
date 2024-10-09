#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void dot_prod(float* a, float* b, float* c, size_t a_r, size_t a_c, size_t b_r, size_t b_c, bool clear);
__global__ void dot_prod_t_a(float* a, float* b, float* c, size_t n, bool clear);
__global__ void dot_prod_t_b(float* a, float* b, float* c, size_t n, bool clear);