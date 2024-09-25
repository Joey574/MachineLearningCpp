#include "SingleBlockNeuralNetwork.h"

void NeuralNetwork::relu(float* x, float* y, size_t size) {
	#pragma omp parallel for
	for (size_t i = 0; i < size; i++) {
		y[i] = x[i] > 0.0f ? x[i] : 0.0f;
	}
}
void NeuralNetwork::leaky_relu(float* x, float* y, size_t size) {
	#pragma omp parallel for
	for (size_t i = 0; i < size; i++) {
		y[i] = x[i] > 0.0f ? x[i] : (0.1f * x[i]);
	}
}
void NeuralNetwork::elu(float* x, float* y, size_t size) {
	#pragma omp parallel for
	for (size_t i = 0; i < size; i++) {
		y[i] = x[i] > 0.0f ? x[i] : (std::exp(x[i]) - 1.0f);
	}
}
void NeuralNetwork::sigmoid(float* x, float* y, size_t size) {
	#pragma omp parallel for
	for (size_t i = 0; i < size; i++) {
		y[i] = 1.0f / (std::exp((-x[i])) + 1.0f);
	}
}


void NeuralNetwork::relu_derivative(float* x, float* y, size_t size) {
	#pragma omp parallel for
	for (size_t i = 0; i < size; i++) {
		y[i] = x[i] > 0.0f ? y[i] : 0.0f;
	}
}
void NeuralNetwork::leaky_relu_derivative(float* x, float* y, size_t size) {
	#pragma omp parallel for
	for (size_t i = 0; i < size; i++) {
		y[i] = x[i] > 0.0f ? y[i] : (y[i] * 0.1f * x[i]);
	}
}
void NeuralNetwork::elu_derivative(float* x, float* y, size_t size) {
	#pragma omp parallel for
	for (size_t i = 0; i < size; i++) {
		y[i] = x[i] > 0.0f ? y[i] : (y[i] * std::exp(x[i]));
	}
}
void NeuralNetwork::sigmoid_derivative(float* x, float* y, size_t size) {
	#pragma omp parallel for
	for (size_t i = 0; i < size; i++) {
		y[i] *= (1.0f / (std::exp((-x[i])) + 1.0f)) * (x[i] - (1.0f / (std::exp((-x[i])) + 1.0f)));
	}
}