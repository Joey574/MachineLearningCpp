#include "SingleBlockNeuralNetwork.h"

void NeuralNetwork::dot_prod(float* __restrict a, float* __restrict b, float* __restrict c, size_t a_r, size_t a_c, size_t b_r, size_t b_c, bool clear) {
	#if LOGDP
		printf("[ %zu x %zu ] * [ %zu x %zu ] = [ %zu x %zu ]\n", a_r, a_c, b_r, b_c, a_r, b_c);
	#endif

	#pragma omp parallel for
	for (size_t i = 0; i < a_r; i++) {

		// first j loop to clear existing c values
		if (clear) {
			const __m256 _a = _mm256_set1_ps(a[i * a_c + 0]);

			size_t k = 0;
			for (; k + 8 <= b_c; k += 8) {
				const __m256 _b = _mm256_loadu_ps(&b[0 * b_c + k]);
				const __m256 _c = _mm256_mul_ps(_a, _b);

				_mm256_storeu_ps(&c[i * b_c + k], _c);
			}

			for (; k < b_c; k++) {
				c[i * b_c + k] = a[i * a_c + 0] * b[0 * b_c + k];
			}
		}

		for (size_t j = clear ? 1 : 0; j < b_r; j++) {
			const __m256 _a = _mm256_set1_ps(a[i * a_c + j]);
			
			size_t k = 0;
			for (; k + 8 <= b_c; k += 8) {
				const __m256 _b = _mm256_loadu_ps(&b[j * b_c + k]);
				const __m256 _c = _mm256_loadu_ps(&c[i * b_c + k]);
				const __m256 _res = _mm256_fmadd_ps(_a, _b, _c);

				_mm256_storeu_ps(&c[i * b_c + k], _res);
			}

			for (; k < b_c; k++) {
				c[i * b_c + k] += a[i * a_c + j] * b[j * b_c + k];
			}
		}
	}
}
void NeuralNetwork::dot_prod_t_a(float* __restrict a, float* __restrict b, float* __restrict c, size_t a_r, size_t a_c, size_t b_r, size_t b_c, bool clear) {
	#if LOGDP
		printf("[ %zu x %zu ].T * [ %zu x %zu ] = [ %zu x %zu ]\n", a_r, a_c, b_r, b_c, a_c, b_c);
	#endif

	#pragma omp parallel for
	for (size_t i = 0; i < a_c; i++) {

		// first j loop to clear existing c values
		if (clear) {
			const __m256 _a_t = _mm256_set1_ps(a[0 * a_c + i]);

			size_t k = 0;
			for (; k + 8 <= b_c; k += 8) {
				const __m256 _b = _mm256_loadu_ps(&b[0 * b_c + k]);
				const __m256 _c = _mm256_mul_ps(_a_t, _b);

				_mm256_storeu_ps(&c[i * b_c + k], _c);
			}

			for (; k < b_c; k++) {
				c[i * b_c + k] = a[0 * a_c + i] * b[0 * b_c + k];
			}
		}

		for (size_t j = clear ? 1 : 0; j < b_r; j++) {
			const __m256 _a_t = _mm256_set1_ps(a[j * a_c + i]);

			size_t k = 0;
			for (; k + 8 <= b_c; k += 8) {
				const __m256 _b = _mm256_loadu_ps(&b[j * b_c + k]);
				const __m256 _c = _mm256_loadu_ps(&c[i * b_c + k]);
				const __m256 _res = _mm256_fmadd_ps(_a_t, _b, _c);

				_mm256_storeu_ps(&c[i * b_c + k], _res);
			}

			for (; k < b_c; k++) {
				c[i * b_c + k] += a[j * a_c + i] * b[j * b_c + k];
			}
		}
	}

}
void NeuralNetwork::dot_prod_t_b(float* __restrict a, float* __restrict b, float* __restrict c, size_t a_r, size_t a_c, size_t b_r, size_t b_c, bool clear) {
	#if LOGDP
		printf("[ %zu x %zu ] * [ %zu x %zu ].T = [ %zu x %zu ]\n", a_r, a_c, b_r, b_c, a_r, b_r);
	#endif

	#pragma omp parallel for
	for (size_t i = 0; i < a_r; i++) {
		for (size_t k = 0; k < b_r; k++) {
			size_t j = clear ? 1 : 0;

			if (clear) {
				c[i * b_r + k] = a[i * a_c + 0] * b[k * b_c + 0];
			}

			__m256 sum = _mm256_setzero_ps();
			for (; j + 8 <= b_c; j += 8) {
				const __m256 _a = _mm256_loadu_ps(&a[i * a_c + j]);
				const __m256 _b = _mm256_loadu_ps(&b[k * b_c + j]);

				sum = _mm256_fmadd_ps(_a, _b, sum);
			}

			// sum values into one location
			const __m128 hi_four = _mm256_extractf128_ps(sum, 1);
			const __m128 lo_four = _mm256_extractf128_ps(sum, 0);
			const __m128 sum_four = _mm_add_ps(lo_four, hi_four);

			const __m128 lo_dual = sum_four;
			const __m128 hi_dual = _mm_movehl_ps(lo_dual, sum_four);
			const __m128 sum_dual = _mm_add_ps(lo_dual, hi_dual);

			const __m128 lo = sum_dual;
			const __m128 hi = _mm_shuffle_ps(sum_dual, sum_dual, 0x1);
			const __m128 fsum = _mm_add_ss(lo, hi);

			c[i * b_r + k] += _mm_cvtss_f32(fsum);
			
			for (; j < b_c; j++) {
				c[i * b_r + k] += a[i * a_c + j] * b[k * b_c + j];
			}
		}
	}
}
