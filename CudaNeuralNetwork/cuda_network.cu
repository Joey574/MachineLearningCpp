#include "cuda_network.cuh"

__global__ void forward_prop(float* weights, float* results, std::vector<int> dims) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;


}

__device__ void dot_prod(const float* mat_a, const float* mat_b, float* results, int a_r, int a_c, int b_c) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;


}

__global__ void element_add(float* mat_a, float* mat_b, float* results, int r, int c) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;

	int i = idx * c + idy;

	if (i < r * c) {
		results[i] = mat_a[i] + mat_b[i];
	}
}

