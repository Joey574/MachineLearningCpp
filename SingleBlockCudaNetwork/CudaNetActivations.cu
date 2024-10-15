#include "CudaKernals.cuh"

// activations
__global__ void leaky_relu(float* x, float* y, size_t r, size_t c) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	if (i < r && j < c) {
		y[i * c + j] = x[i* c + j] > 0.0f ? x[i * c + j] : x[i * c + j] * 0.1f;
	}
}

// derivatives
__global__ void leaky_relu_derivative(float* x, float* y, size_t r, size_t c) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	if (i < r && j < c) {
		y[i * c + j] = x[i * c + j] > 0.0f ? y[i * c + j] : y[i * c + j] * 0.1f;
	}
}