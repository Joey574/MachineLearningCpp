#include "CudaKernals.cuh"
#include <stdio.h>

__global__ void dot_prod(float* a, float* b, float* c, size_t a_r, size_t a_c, size_t b_r, size_t b_c) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	int row = i / (int)a_c;
	int col = i % (int)a_c;

	if (row < a_r) {
		float sum = 0.0f;

		for (int k = 0; k < b_r; k++) {
			sum += a[row * a_c + k] * b[k * b_c + col];
		}

		c[row * b_c + col] = sum;
	}
}
__global__ void dot_prod_t_a(float* a, float* b, float* c, size_t a_r, size_t a_c, size_t b_r, size_t b_c) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	int row = i / (int)a_r;
	int col = i % (int)a_r;

	if (row < a_c) {
		float sum = 0.0f;

		for (int k = 0; k < b_r; k++) {
			sum += a[k * a_c + row] * b[k * b_c + col];
		}

		c[row * b_c + col] = sum;
	}
}
__global__ void dot_prod_t_b(float* a, float* b, float* c, size_t a_r, size_t a_c, size_t b_r, size_t b_c) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	int row = i / (int)a_c;
	int col = i % (int)a_c;

	if (row < a_r) {
		float sum = 0.0f;

		for (int k = 0; k < b_c; k++) {
			//sum += a[row * a_c + k] * b[col * b_c + k];
			sum += a[row * a_c + k];
		}

		c[row * b_r + col] = sum;
		
		//c[row * b_r + col] = 0.0f;
		//a[row * a_c + b_c] = 1.0f;
		//b[col * b_c + b_c - 1] = 1.0f;
	}
}