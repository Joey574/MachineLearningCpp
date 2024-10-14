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