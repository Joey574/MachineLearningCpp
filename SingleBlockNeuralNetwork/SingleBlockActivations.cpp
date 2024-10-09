#include "SingleBlockNeuralNetwork.h"

void NeuralNetwork::relu(float* x, float* y, size_t size) {
	#pragma omp parallel for
	for (size_t i = 0; i < size; i++) {
		y[i] = x[i] > 0.0f ? x[i] : 0.0f;
	}
}
void NeuralNetwork::leaky_relu(float* x, float* y, size_t size) {
	#pragma omp parallel for
	for (size_t i = 0; i <= size - 8; i += 8) {
		_mm256_store_ps(&y[i],
			_mm256_max_ps(
				_mm256_load_ps(&x[i]),
				_mm256_mul_ps(
					_mm256_load_ps(&x[i]),
					_mm256_set1_ps(0.1f)
				)));
	}

	for (size_t i = size - (size % 8); i < size; i++) {
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
	for (size_t i = 0; i <= size - 8; i += 8) {
		_mm256_store_ps(&y[i],
			_mm256_div_ps(
				_mm256_set1_ps(1.0f),
				_mm256_add_ps(
					_mm256_exp_ps(
						_mm256_mul_ps(
							_mm256_load_ps(&x[i]),
							_mm256_set1_ps(-1.0f))),
					_mm256_set1_ps(1.0f)
				)));
	}

	for (size_t i = size - (size % 8); i < size; i++) {
		y[i] = 1.0f / (std::exp(-x[i]) + 1.0f);
	}
}
void NeuralNetwork::softmax(float* x, float* y, size_t size) {
	#pragma omp parallel for
	for (size_t i = 0; i < size / m_dimensions.back(); i++) {

		float max = *std::max_element(&x[i * m_dimensions.back()], &x[i * m_dimensions.back() + m_dimensions.back()]);
		float sum = 0;

		for (size_t j = 0; j < m_dimensions.back(); j++) {
			sum += std::exp(x[i * m_dimensions.back() + j] - max);
		}

		float log_sum = max + std::log(sum);

		for (size_t j = 0; j < m_dimensions.back(); j++) {
			y[i * m_dimensions.back() + j] = std::exp(x[i * m_dimensions.back() + j] - log_sum);
		}
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
		y[i] = x[i] > 0.0f ? y[i] : (y[i] * 0.1f);
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
		y[i] *= (1.0f / (std::exp((-x[i])) + 1.0f)) * (1.0f - (1.0f / (std::exp((-x[i])) + 1.0f)));
	}
}