#include "SingleBlockNeuralNetwork.h"

// loss
void NeuralNetwork::mae_loss(float* x, float* y, float* c, size_t x_r, size_t x_c) {
	for (size_t i = 0; i < x_r * x_c; i++) {
		c[i] = x[i] - y[i];
	}
}
void NeuralNetwork::one_hot_loss(float* x, float* y, float* c, size_t x_r, size_t x_c) {
	for (size_t i = 0; i < x_r * x_c; i++) {
		c[i] = x[i];
	}
	for (size_t i = 0; i < x_c; i++) {
		c[(int)y[i] * x_c + i]--;
	}
}

// score
float NeuralNetwork::mae_score(float* x, float* y, size_t x_r, size_t x_c) {
	float err = 0.0f;

	for (size_t i = 0; i < x_r * x_c; i++) {
		err += std::abs(x[i] - y[i]);
	}

	return err / (float)x_c;
}
float NeuralNetwork::accuracy_score(float* x, float* y, size_t x_r, size_t x_c) {
	size_t correct = 0;

	for (size_t i = 0; i < x_c; i++) {

		size_t m_idx = 0;
		for (size_t j = 1; j < x_r; j++) {
			if (x[j * x_c + i] > x[m_idx * x_c + i]) {
				m_idx = j;
			}
		}

		if (m_idx == y[i]) {
			correct++;
		}
	}

	return correct / (float)x_c * 100.0f;
}