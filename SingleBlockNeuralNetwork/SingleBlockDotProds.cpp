#include "SingleBlockNeuralNetwork.h"

void NeuralNetwork::dot_prod(float* a, float* b, float* c, size_t a_r, size_t a_c, size_t b_r, size_t b_c, bool clear) {
	#pragma omp parallel for
	for (size_t i = 0; i < a_r; i++) {

		// first j loop to clear existing c values
		if (clear) {
			__m256 scalar = _mm256_set1_ps(a[i * a_c + 0]);
			size_t k = 0;
			for (; k + 8 <= b_c; k += 8) {
				_mm256_store_ps(&c[i * b_c + k],
					_mm256_mul_ps(
						scalar,
						_mm256_load_ps(&b[0 * b_c + k])
					));
			}

			for (; k < b_c; k++) {
				c[i * b_c + k] = a[i * a_c + 0] * b[0 * b_c + k];
			}
		}

		for (size_t j = clear ? 1 : 0; j < b_r; j++) {
			__m256 scalar = _mm256_set1_ps(a[i * a_c + j]);
			
			size_t k = 0;
			for (; k + 8 <= b_c; k += 8) {
				_mm256_store_ps(&c[i * b_c + k],
					_mm256_fmadd_ps(
						scalar,
						_mm256_load_ps(&b[j * b_c + k]),
						_mm256_load_ps(&c[i * b_c + k])));
			}

			for (; k < b_c; k++) {
				c[i * b_c + k] += a[i * a_c + j] * b[j * b_c + k];
			}
		}
	}
}
void NeuralNetwork::dot_prod_t_a(float* a, float* b, float* c, size_t a_r, size_t a_c, size_t b_r, size_t b_c, bool clear) {
	#pragma omp parallel for
	for (size_t i = 0; i < a_c; i++) {

		// first j loop to clear existing c values
		if (clear) {
			__m256 scalar = _mm256_set1_ps(a[0 * a_c + i]);
			size_t k = 0;
			for (; k + 8 <= b_c; k += 8) {
				_mm256_store_ps(&c[i * b_c + k],
					_mm256_mul_ps(
						scalar,
						_mm256_load_ps(&b[0 * b_c + k])
					));
			}

			for (; k < b_c; k++) {
				c[i * b_c + k] = a[0 * a_c + i] * b[0 * b_c + k];
			}
		}

		for (size_t j = clear ? 1 : 0; j < b_r; j++) {
			__m256 scalar = _mm256_set1_ps(a[j * a_c + i]);

			size_t k = 0;
			for (; k + 8 <= b_c; k += 8) {
				_mm256_store_ps(&c[i * b_c + k],
					_mm256_fmadd_ps(
						scalar,
						_mm256_load_ps(&b[j * b_c + k]),
						_mm256_load_ps(&c[i * b_c + k])));
			}

			for (; k < b_c; k++) {
				c[i * b_c + k] += a[j * a_c + i] * b[j * b_c + k];
			}
		}
	}
}
void NeuralNetwork::dot_prod_t_b(float* a, float* b, float* c, size_t a_r, size_t a_c, size_t b_r, size_t b_c, bool clear) {
	#pragma omp parallel for
	for (size_t i = 0; i < a_r; i++) {
		for (size_t k = 0; k < b_r; k++) {

			if (clear) {
				c[i * b_r + k] = a[i * a_c + 0] * b[k * b_c + 0];
			}

			__m256 sum = _mm256_setzero_ps();
			size_t j = clear ? 1 : 0;
			for (; j + 8 <= b_c; j += 8) {

				sum = _mm256_fmadd_ps(
					_mm256_load_ps(&a[i * a_c + j]),
					_mm256_load_ps(&b[k * b_c + j]),
					sum);
			}

			float temp[8];

			_mm256_store_ps(temp, sum);

			c[i * b_r + k] +=
				temp[0] +
				temp[1] +
				temp[2] +
				temp[3] +
				temp[4] +
				temp[5] +
				temp[6] +
				temp[7];

			for (; j < b_c; j++) {
				c[i * b_r + k] += a[i * a_c + j] * b[k * b_c + j];
			}
		}
	}
}