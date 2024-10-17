#include "CudaKernals.cuh"

// activations
__global__ void leaky_relu(float* x, float* y, size_t rows, size_t columns) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	int r = i / (int)columns;
	int c = i % (int)columns;

	if (r < rows) {
		y[r * columns + c] = x[r * columns + c] > 0.0f ? x[r * columns + c] : x[r * columns + c] * 0.1f;
	}
}

// derivatives
__global__ void leaky_relu_derivative(float* x, float* y, size_t rows, size_t columns) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	int r = i / (int)columns;
	int c = i % (int)columns;

	if (r < rows) {
		y[r * columns + c] = x[r * columns + c] > 0.0f ? y[r * columns + c] : y[r * columns + c] * 0.1f;
	}
}