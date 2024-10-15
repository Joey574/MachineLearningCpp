#include "CudaKernals.cuh"

// loss
__global__ void one_hot_loss(float* pred, float* loss, float* y, size_t rows, size_t columns) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < columns) {
		for (int j = 0; j < rows; j++) {
			loss[j * columns + i] = pred[j * columns + i];
		}
		loss[(int)y[i] * columns + i]--;
	}
}
__global__ void mae_loss(float* pred, float* loss, float* y, size_t rows, size_t columns) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < columns) {
		for (int j = 0; j < rows; j++) {
			loss[j * columns + i] = pred[j * columns + i] - y[j * columns + i];
		}
	}
}

// score
__global__ void accuracy_score(float* pred, float* y, int* correct, size_t rows, size_t columns) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < columns) {
		int max_idx = 0;
		for (int j = 1; j < rows; j++) {
			if (pred[j * columns + i] > pred[max_idx * columns + i]) {
				max_idx = j;
			}
		}

		if (max_idx == (int)y[i]) {
			atomicAdd(correct, 1);
		}
	}
}