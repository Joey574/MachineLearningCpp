#include <cuda_runtime.h>
#include <vector>

__global__ void forward_prop(float* weights, float* results, std::vector<int> dims);