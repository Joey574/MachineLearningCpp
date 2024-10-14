#include "CudaKernals.cuh"

// activations
__global__ void leaky_relu(float* x, float* y, size_t r, size_t c) {
	int i = blockDim.y * blockIdx.y + threadIdx.y;
	int j = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < r && j < c) {
		y[i] = x[i] > 0.0f ? x[i] : x[i] * 0.1f;
	}
}

// derivatives
__global__ void leaky_relu_derivative(float* x, float* y, size_t r, size_t c) {
	int i = blockDim.y * blockIdx.y + threadIdx.y;
	int j = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < r && j < c) {
		y[i] = x[i] > 0.0f ? y[i] : y[i] * 0.1f;
	}
}