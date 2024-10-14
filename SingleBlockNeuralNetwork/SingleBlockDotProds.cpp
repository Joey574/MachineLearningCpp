#include "SingleBlockNeuralNetwork.h"

void NeuralNetwork::dot_prod(float* a, float* b, float* c, size_t a_r, size_t a_c, size_t b_r, size_t b_c, bool clear) {
	#pragma omp parallel for
	for (size_t i = 0; i < a_r; i++) {

		// first j loop to clear existing c values
		if (clear) {
			const __m256 _scalar = _mm256_set1_ps(a[i * a_c + 0]);

			size_t k = 0;
			for (; k + 8 <= b_c; k += 8) {
				const __m256 _b = _mm256_load_ps(&b[0 * b_c + k]);
				const __m256 _c = _mm256_mul_ps(_scalar, _b);

				_mm256_store_ps(&c[i * b_c + k], _c);
			}

			for (; k < b_c; k++) {
				c[i * b_c + k] = a[i * a_c + 0] * b[0 * b_c + k];
			}
		}

		for (size_t j = clear ? 1 : 0; j < b_r; j++) {
			const __m256 _scalar = _mm256_set1_ps(a[i * a_c + j]);
			
			size_t k = 0;
			for (; k + 8 <= b_c; k += 8) {
				const __m256 _b = _mm256_load_ps(&b[j * b_c + k]);
				const __m256 _c = _mm256_load_ps(&c[i * b_c + k]);
				const __m256 _res = _mm256_fmadd_ps(_scalar, _b, _c);

				_mm256_store_ps(&c[i * b_c + k], _res);
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
			const __m256 _scalar = _mm256_set1_ps(a[0 * a_c + i]);

			size_t k = 0;
			for (; k + 8 <= b_c; k += 8) {
				const __m256 _b = _mm256_load_ps(&b[0 * b_c + k]);
				const __m256 _c = _mm256_mul_ps(_scalar, _b);

				_mm256_store_ps(&c[i * b_c + k], _c);
			}

			for (; k < b_c; k++) {
				c[i * b_c + k] = a[0 * a_c + i] * b[0 * b_c + k];
			}
		}

		for (size_t j = clear ? 1 : 0; j < b_r; j++) {
			const __m256 _scalar = _mm256_set1_ps(a[j * a_c + i]);

			size_t k = 0;
			for (; k + 8 <= b_c; k += 8) {
				const __m256 _b = _mm256_load_ps(&b[j * b_c + k]);
				const __m256 _c = _mm256_load_ps(&c[i * b_c + k]);
				const __m256 _res = _mm256_fmadd_ps(_scalar, _b, _c);

				_mm256_store_ps(&c[i * b_c + k], _res);
			}

			for (; k < b_c; k++) {
				c[i * b_c + k] += a[j * a_c + i] * b[j * b_c + k];
			}
		}
	}
}
void NeuralNetwork::dot_prod_t_b(float* a, float* b, float* c, size_t a_r, size_t a_c, size_t b_r, size_t b_c, bool clear) {

	//float* b_t = (float*)malloc(b_r * b_c * sizeof(float));

	//// transpose b into b_t
	//#pragma omp parallel for
	//for (size_t c = 0; c < b_c; c++) {
	//	for (size_t r = 0; r < b_r; r++) {
	//		b_t[c * b_r + r] = b[r * b_c + c];
	//	}
	//}

	//size_t t = b_r;
	//b_r = b_c;
	//b_c = t;

	//// normal dot_prod with b_t
	//#pragma omp parallel for
	//for (size_t i = 0; i < a_r; i++) {

	//	// first j loop to clear existing c values
	//	if (clear) {
	//		const __m256 _scalar = _mm256_set1_ps(a[i * a_c + 0]);

	//		size_t k = 0;
	//		for (; k + 8 <= b_c; k += 8) {
	//			const __m256 _b = _mm256_load_ps(&b_t[0 * b_c + k]);
	//			const __m256 _c = _mm256_mul_ps(_scalar, _b);

	//			_mm256_store_ps(&c[i * b_c + k], _c);
	//		}

	//		for (; k < b_c; k++) {
	//			c[i * b_c + k] = a[i * a_c + 0] * b_t[0 * b_c + k];
	//		}
	//	}

	//	for (size_t j = clear ? 1 : 0; j < b_r; j++) {
	//		const __m256 _scalar = _mm256_set1_ps(a[i * a_c + j]);

	//		size_t k = 0;
	//		for (; k + 8 <= b_c; k += 8) {
	//			const __m256 _b = _mm256_load_ps(&b_t[j * b_c + k]);
	//			const __m256 _c = _mm256_load_ps(&c[i * b_c + k]);
	//			const __m256 _res = _mm256_fmadd_ps(_scalar, _b, _c);

	//			_mm256_store_ps(&c[i * b_c + k], _res);
	//		}

	//		for (; k < b_c; k++) {
	//			c[i * b_c + k] += a[i * a_c + j] * b_t[j * b_c + k];
	//		}
	//	}
	//}

	//// free transposed matrix
	//free(b_t);


	#pragma omp parallel for
	for (size_t i = 0; i < a_r; i++) {
		for (size_t k = 0; k < b_r; k++) {

			if (clear) {
				c[i * b_r + k] = a[i * a_c + 0] * b[k * b_c + 0];
			}

			__m256 sum = _mm256_setzero_ps();
			size_t j = clear ? 1 : 0;
			for (; j + 16 <= b_c; j += 16) {
				const __m256 _a1 = _mm256_load_ps(&a[i * a_c + j]);
				const __m256 _a2 = _mm256_load_ps(&a[i * a_c + j + 8]);

				const __m256 _b1 = _mm256_load_ps(&b[k * b_c + j]);
				const __m256 _b2 = _mm256_load_ps(&b[k * b_c + j + 8]);

				sum = _mm256_fmadd_ps(_a1, _b1, sum);
				sum = _mm256_fmadd_ps(_a2, _b2, sum);
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