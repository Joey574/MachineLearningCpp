#include "cuda_network.cuh"

__global__ void forward_prop(float* weights, float* results, std::vector<int> dims) {
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	// 
}