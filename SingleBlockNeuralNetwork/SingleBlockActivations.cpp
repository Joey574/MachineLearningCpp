#include "SingleBlockNeuralNetwork.h"

void NeuralNetwork::relu(float* __restrict x, float* __restrict y, size_t size) {
	#pragma omp parallel for
	for (size_t i = 0; i <= size - 8; i += 8) {
		const __m256 _x = _mm256_loadu_ps(&x[i]);

		_mm256_storeu_ps(&y[i], _mm256_max_ps(_x, _mm256_setzero_ps()));
	}

	for (size_t i = size - (size % 8); i < size; i++) {
		y[i] = x[i] > 0.0f ? x[i] : (0.0f);
	}
}
void NeuralNetwork::leaky_relu(float* __restrict x, float* __restrict y, size_t size) {
	const __m256 _cof = _mm256_set1_ps(0.1f);
	const __m256 _zero = _mm256_setzero_ps();

	#pragma omp parallel for
	for (size_t i = 0; i <= size - 8; i += 8) {
		const __m256 _x = _mm256_loadu_ps(&x[i]);
		const __m256 _x2 = _mm256_mul_ps(_x, _cof);

		const __m256 _res = _mm256_max_ps(_x2, _x);

		_mm256_storeu_ps(&y[i], _res);
	}

	for (size_t i = size - (size % 8); i < size; i++) {
		y[i] = x[i] > 0.0f ? x[i] : (0.1f * x[i]);
	}
}
void NeuralNetwork::elu(float* __restrict x, float* __restrict y, size_t size) {
	#pragma omp parallel for
	for (size_t i = 0; i < size; i++) {
		y[i] = x[i] > 0.0f ? x[i] : (std::exp(x[i]) - 1.0f);
	}
}
void NeuralNetwork::sigmoid(float* __restrict x, float* __restrict y, size_t size) {
	const __m256 _one = _mm256_set1_ps(1.0f);
	const __m256 _zero = _mm256_setzero_ps();

	#pragma omp parallel for
	for (size_t i = 0; i <= size - 8; i += 8) {
		const __m256 _x = _mm256_loadu_ps(&x[i]);
		const __m256 _nx = _mm256_sub_ps(_zero, _x);

		const __m256 _ex = _mm256_exp_ps(_nx);

		const __m256 _x2 = _mm256_add_ps(_ex, _one);
		const __m256 _res = _mm256_rcp_ps(_x2);

		_mm256_storeu_ps(&y[i], _res);
	}


	for (size_t i = size - (size % 8); i < size; i++) {
		y[i] = 1.0f / (1.0f + std::exp(-x[i]));
	}
}
void NeuralNetwork::softmax(float* __restrict x, float* __restrict y, size_t size) {

	size_t n = size / m_dimensions.back();

	#pragma omp parallel for
	for (size_t i = 0; i < size / m_dimensions.back(); i++) {

		float max = x[i];
		for (size_t j = 1; j < m_dimensions.back(); j++) {
			if (x[j * n + i] > max) {
				max = x[j * n + i];
			}
		}

		float sum = 0;
		for (size_t j = 0; j < m_dimensions.back(); j++) {
			sum += std::exp(x[j * n + i] - max);
		}

		float log_sum = max + std::log(sum);

		for (size_t j = 0; j < m_dimensions.back(); j++) {
			y[j * n + i] = std::exp(x[j * n + i] - log_sum);
		}
	}
}

void NeuralNetwork::relu_derivative(float* __restrict x, float* __restrict y, size_t size) {
	#pragma omp parallel for
	for (size_t i = 0; i < size; i++) {
		y[i] = x[i] > 0.0f ? y[i] : 0.0f;
	}
}
void NeuralNetwork::leaky_relu_derivative(float* __restrict x, float* __restrict y, size_t size) {

	// compiler seems to have found better way to do this, need to look into that
	/*#pragma omp parallel for
	for (size_t i = 0; i <= size - 8; i += 8) {
		const __m256 _x = _mm256_loadups(&x[i]);
		const __m256 _y = _mm256_loadups(&y[i]);

		const __m256 mask = _mm256_cmp_ps(_x, _mm256_setzero_ps(), _CMP_LT_OQ);

		const __m256 deriv = _mm256_blendv_ps(_mm256_set1_ps(1.0f), _mm256_set1_ps(0.1f), mask);

		_mm256_storeu_ps(&y[i], _mm256_mul_ps(_y, deriv));
	}

	for (size_t i = size - (size % 8); i < size; i++) {
		y[i] = x[i] > 0.0f ? y[i] : (y[i] * 0.1f);
	}*/

	#pragma omp parallel for
	for (size_t i = 0; i < size; i++) {
		y[i] = x[i] > 0.0f ? y[i] : (y[i] * 0.1f);
	}
}
void NeuralNetwork::elu_derivative(float* __restrict x, float* __restrict y, size_t size) {
	#pragma omp parallel for
	for (size_t i = 0; i < size; i++) {
		y[i] = x[i] > 0.0f ? y[i] : (y[i] * std::exp(x[i]));
	}
}
void NeuralNetwork::sigmoid_derivative(float* __restrict x, float* __restrict y, size_t size) {
	#pragma omp parallel for
	for (size_t i = 0; i < size; i++) {
		y[i] *= (1.0f / (std::exp((-x[i])) + 1.0f)) * (1.0f - (1.0f / (std::exp((-x[i])) + 1.0f)));
	}
}