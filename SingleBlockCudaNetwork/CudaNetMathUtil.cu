#include "CudaKernals.cuh"

__global__ void horizontal_add(float* a, float* b, size_t a_r, size_t a_c) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < a_r) {
		for (size_t j = 0; j < a_c; j++) {
			a[i * a_r + j] += b[i];
		}
	}
}
__global__ void horizontal_sum(float* a, float* b, size_t a_r, size_t a_c) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < a_r) {
		b[i] = a[i * a_c];

		for (size_t j = 1; j < a_c; j++) {
			b[i] += a[i * a_c + j];
		}
	}
}

__global__ void update_weights(float* weight, float* d_weight, float lr, size_t n) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < n) {
		weight[i] -= d_weight[i] * lr;
	}
}
__global__ void update_bias(float* bias, float* d_bias, float lr, size_t n) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < n) {
		bias[i] -= d_bias[i] * lr;
	}
}