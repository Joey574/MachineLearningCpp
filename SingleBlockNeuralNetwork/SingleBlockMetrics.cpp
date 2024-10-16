#include "SingleBlockNeuralNetwork.h"

// loss
void NeuralNetwork::mae_loss(float* x, float* y, float* c, size_t rows, size_t columns) {
	for (size_t i = 0; i < rows * columns; i++) {
		c[i] = x[i] - y[i];
	}
}
void NeuralNetwork::one_hot_loss(float* x, float* y, float* c, size_t rows, size_t columns) {
	for (size_t i = 0; i < rows * columns; i++) {
		c[i] = x[i];
	}
	for (size_t i = 0; i < columns; i++) {
		c[(int)y[i] * columns + i]--;
	}
}

// score
float NeuralNetwork::mae_score(float* x, float* y, size_t rows, size_t columns) {
	float err = 0.0f;

	for (size_t i = 0; i < rows * columns; i++) {
		err += std::abs(x[i] - y[i]);
	}

	return err / (float)columns;
}
float NeuralNetwork::accuracy_score(float* x, float* y, size_t rows, size_t columns) {
	size_t correct = 0;

	for (size_t i = 0; i < columns; i++) {

		size_t m_idx = 0;
		for (size_t j = 1; j < rows; j++) {
			if (x[j * columns + i] > x[m_idx * columns + i]) {
				m_idx = j;
			}
		}

		if (m_idx == y[i]) {
			correct++;
		}
	}

	return correct / (float)columns * 100.0f;
}