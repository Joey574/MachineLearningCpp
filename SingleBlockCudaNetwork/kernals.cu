#include "kernals.cuh"

__global__ void dot_prod(float* a, float* b, float* c, size_t a_r, size_t a_c, size_t b_r, size_t b_c, bool clear) {
	int i = blockDim.y * blockIdx.y + threadIdx.y;
	int j = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < a_r && j < b_c) {
		float sum = 0.0f;

		for (int k = 0; k < n; k++) {
			sum += a[i * a_c + k] * b[k * b_c + j];
		}

		c[i * n + j] = sum;
	}
}