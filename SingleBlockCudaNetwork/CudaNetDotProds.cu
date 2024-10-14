#include "CudaKernals.cuh"

__global__ void dot_prod(float* a, float* b, float* c, size_t a_r, size_t a_c, size_t b_r, size_t b_c) {
	int i = blockDim.y * blockIdx.y + threadIdx.y;
	int j = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < a_r && j < b_c) {
		float sum = 0.0f;

		for (int k = 0; k < b_c; k++) {
			sum += a[i * a_c + k] * b[k * b_c + j];
		}

		c[i * b_c + j] = sum;
	}
}
__global__ void dot_prod_t_a(float* a, float* b, float* c, size_t a_r, size_t a_c, size_t b_r, size_t b_c) {
	int i = blockDim.y * blockIdx.y + threadIdx.y;
	int j = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < a_c && j < b_c) {
		float sum = 0.0f;

		for (int k = 0; k < b_c; k++) {
			sum += a[k * a_c + i] * b[k * b_c + j];
		}

		c[i * b_c + j] = sum;
	}
}
__global__ void dot_prod_t_b(float* a, float* b, float* c, size_t a_r, size_t a_c, size_t b_r, size_t b_c) {
	int i = blockDim.y * blockIdx.y + threadIdx.y;
	int j = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < a_r && j < b_c) {
		float sum = 0.0f;

		for (int k = 0; k < b_r; k++) {
			sum += a[i * a_c + k] * b[i * b_c + k];
		}

		c[i * b_r + j] = sum;
	}
}