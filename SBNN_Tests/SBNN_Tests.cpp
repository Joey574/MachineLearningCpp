#include <iostream>

#include "Matrix.h"

// copy pasted dot prod functions here :)
void dot_prod(float* a, float* b, float* c, size_t a_r, size_t a_c, size_t b_r, size_t b_c, bool clear) {
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
void dot_prod_t_a(float* a, float* b, float* c, size_t a_r, size_t a_c, size_t b_r, size_t b_c, bool clear) {
	#pragma omp parallel for
	for (size_t i = 0; i < a_r; i++) {

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
void dot_prod_t_b(float* a, float* b, float* c, size_t a_r, size_t a_c, size_t b_r, size_t b_c, bool clear) {
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

void test(Matrix a, Matrix b, Matrix c);

int main()
{
	Matrix dot_a = Matrix({
	   {1, 2, 3},
	   {4, 5, 6}
		});

	Matrix dot_b = Matrix({
		{7, 8},
		{9, 10},
		{11, 12}
		});

	Matrix check_a = Matrix({
		{58, 64},
		{139, 154}
		});

	/*test(dot_a, dot_b, check_a);

	dot_a = Matrix({
	   {3, 4, 2}
		});

	dot_b = Matrix({
		{13, 9, 7, 15},
		{8, 7, 4, 6},
		{6, 4, 0, 3}
		});

	check_a = Matrix({
		{83, 63, 37, 75}
		});

	test(dot_a, dot_b, check_a);*/

	dot_a = Matrix({
	{-2.5f, 0},
	{-2.0f, 0},
	{-1.5f, 0},
	{-1.0f, 0},
	{-0.5f, 0},
	{0.0f, 0},
	{0.5f, 0},
	{1.0f, 0},
	{-2.5f, 0},
	{-2.0f, 0},
	{-1.5f, 0},
	{-1.0f, 0},
	{-0.5f, 0},
	{0.0f, 0},
	{0.5f, 0},
	{1.0f, 0},
	{-0.5f, 0},
	{0.0f, 0},
	{0.5f, 0},
	{1.0f, 0},
		});

	dot_b = Matrix({
		{1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4},
		{2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5},
		});

	check_a = Matrix({
		{-2.5f, -5.0f, -7.5f, -10.0f, -12.5f, -15.0f, -17.5f, -20.0f, -2.5f, -5.0f, -7.5f, -10.0f, -12.5f, -15.0f, -17.5f, -20.0f, -2.5f, -5.0f, -7.5f, -10.0f},
		{-2.0f, -4.0f, -6.0f, -8.0f, -10.0f, -12.0f, -14.0f, -16.0f, -2.0f, -4.0f, -6.0f, -8.0f, -10.0f, -12.0f, -14.0f, -16.0f, -2.0f, -4.0f, -6.0f, -8.0f},
		{-1.5f, -3.0f, -4.5f, -6.0f, -7.5f, -9.0f, -10.5f, -12.0f, -1.5f, -3.0f, -4.5f, -6.0f, -7.5f, -9.0f, -10.5, -12.0f, -1.5f, -3.0f, -4.5f, -6.0f},
		{-1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f, -7.0f, -8.0f, -1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f, -7.0f, -8.0f, -1.0f, -2.0f, -3.0f, -4.0f},
		{-0.5f, -1.0f, -1.5f, -2.0f, -2.5f, -3.0f, -3.5f, -4.0f, -0.5f, -1.0f, -1.5f, -2.0f, -2.5f, -3.0f, -3.5f, -4.0f, -0.5f, -1.0f, -1.5f, -2.0f},
		{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
		{0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f, 0.5f, 1.0f, 1.5f, 2.0f},
		{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 1.0f, 2.0f, 3.0f, 4.0f},
		{-2.5f, -5.0f, -7.5f, -10.0f, -12.5f, -15.0f, -17.5f, -20.0f, -2.5f, -5.0f, -7.5f, -10.0f, -12.5f, -15.0f, -17.5f, -20.0f, -2.5f, -5.0f, -7.5f, -10.0f},
		{-2.0f, -4.0f, -6.0f, -8.0f, -10.0f, -12.0f, -14.0f, -16.0f, -2.0f, -4.0f, -6.0f, -8.0f, -10.0f, -12.0f, -14.0f, -16.0f, -2.0f, -4.0f, -6.0f, -8.0f},
		{-1.5f, -3.0f, -4.5f, -6.0f, -7.5f, -9.0f, -10.5f, -12.0f, -1.5f, -3.0f, -4.5f, -6.0f, -7.5f, -9.0f, -10.5, -12.0f, -1.5f, -3.0f, -4.5f, -6.0f},
		{-1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f, -7.0f, -8.0f, -1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f, -7.0f, -8.0f, -1.0f, -2.0f, -3.0f, -4.0f},
		{-0.5f, -1.0f, -1.5f, -2.0f, -2.5f, -3.0f, -3.5f, -4.0f, -0.5f, -1.0f, -1.5f, -2.0f, -2.5f, -3.0f, -3.5f, -4.0f, -0.5f, -1.0f, -1.5f, -2.0f},
		{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
		{0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f, 0.5f, 1.0f, 1.5f, 2.0f},
		{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 1.0f, 2.0f, 3.0f, 4.0f},
		{-0.5f, -1.0f, -1.5f, -2.0f, -2.5f, -3.0f, -3.5f, -4.0f, -0.5f, -1.0f, -1.5f, -2.0f, -2.5f, -3.0f, -3.5f, -4.0f, -0.5f, -1.0f, -1.5f, -2.0f},
		{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
		{0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f, 0.5f, 1.0f, 1.5f, 2.0f},
		{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 1.0f, 2.0f, 3.0f, 4.0f}
		});

	test(dot_a, dot_b, check_a);

}


void test(Matrix a, Matrix b, Matrix c) {

	Matrix temp(c.RowCount, c.ColumnCount, 1.0f);

	std::cout << "Expected:\n" << c.ToString() << "\n";

	std::cout << "Base Case:\n" << a.dot_product(b).ToString() << "\n";

	dot_prod(a.matrix, b.matrix, temp.matrix, a.RowCount, a.ColumnCount, b.RowCount, b.ColumnCount, true);
	std::cout << "Base Check:\n" << temp.ToString() << "\n";

	a = a.Transpose();
	std::cout << "a.T Case:\n" << a.Transpose().dot_product(b).ToString() << "\n";

	dot_prod_t_a(a.matrix, b.matrix, temp.matrix, a.RowCount, a.ColumnCount, b.RowCount, b.ColumnCount, true);
	std::cout << "a.T Check:\n" << temp.ToString() << "\n";

	// transpose a back to normal
	a = a.Transpose();

	b = b.Transpose();
	std::cout << "b.T Case:\n" << a.dot_product(b.Transpose()).ToString() << "\n";

	dot_prod_t_b(a.matrix, b.matrix, temp.matrix, a.RowCount, a.ColumnCount, b.RowCount, b.ColumnCount, true);
	std::cout << "b.T Check:\n" << temp.ToString() << "\n";
}