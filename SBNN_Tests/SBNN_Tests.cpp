#include <iostream>

#include "Matrix.h"

// copy pasted dot prod functions here :)
void dot_prod(float* a, float* b, float* c, size_t a_r, size_t a_c, size_t b_r, size_t b_c, bool clear) {
	//std::cout << "a:\n\trows: " << a_r << "\n\tcols: " << a_c << "\nb:\n\trows: " << b_r << "\n\tcols: " << b_c << "\nclear: " << clear << "\n";

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

				if (i * b_c + k > a_r * b_c) {
					std::cout << "size error indexing into c\n";
				}
				if (i * a_c + j > a_c * a_r) {
					std::cout << "size error indexing into a\n";
				}
				if (j * b_c + k > b_r * b_c) {
					std::cout << "size error indexing into b\n";
				}

				c[i * b_c + k] += a[i * a_c + j] * b[j * b_c + k];
			}
		}
	}
}
void dot_prod_t_a(float* a, float* b, float* c, size_t a_r, size_t a_c, size_t b_r, size_t b_c, bool clear) {
	//std::cout << "a:\n\trows: " << a_r << "\n\tcols: " << a_c << "\nb:\n\trows: " << b_r << "\n\tcols: " << b_c << "\nclear: " << clear << "\n";

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

				if (i * b_c + k > a_c * b_c) {
					std::cout << "size error indexing into c\n";
				}
				if (j * a_c + i > a_c * a_r) {
					std::cout << "size error indexing into a\n";
				}
				if (j * b_c + k > b_r * b_c) {
					std::cout << "size error indexing into b\n";
				}

				c[i * b_c + k] += a[j * a_c + i] * b[j * b_c + k];
			}
		}
	}
}
void dot_prod_t_b(float* a, float* b, float* c, size_t a_r, size_t a_c, size_t b_r, size_t b_c, bool clear) {
	//std::cout << "a:\n\trows: " << a_r << "\n\tcols: " << a_c << "\nb:\n\trows: " << b_r << "\n\tcols: " << b_c << "\nclear: " << clear << "\n";

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

				if (i * b_r + k > a_r * b_r) {
					std::cout << "size error indexing into c\n";
				}
				if (i * a_c + j > a_c * a_r) {
					std::cout << "size error indexing into a\n";
				}
				if (k * b_c + j > b_c * b_r) {
					std::cout << "size error indexing into b\n";
				}

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

	test(dot_a, dot_b, check_a);

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

	test(dot_a, dot_b, check_a);

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

	dot_a = Matrix({
		{0.5f, 1.5f, 6.0f, 3.5f, 7.0f, 2.0f, 6.0f, 8.0f, 5.5f, 4.25f},
		{2.5f, 7.0f, 2.0f, 6.5f, 2.0f, 8.0f, 9.0f, 1.0f, 0.0f, 7.25f},
		{1.0f, 5.0f, 7.0f, 3.0f, 4.0f, 9.0f, 0.0f, 1.5f, 2.5f, 7.25f},
		{1.25f, 6.0f, 2.5f, 7.5f, 4.5f, 3.5f, 1.5f, 8.5f, 5.5f, 3.5f},
		{6.5f, 5.5f, 2.5f, 2.25f, 5.5f, 3.5f, 7.0f, 9.0f, 1.0f, 4.0f},
		{1.0f, 7.0f, 5.0f, 3.0f, 9.0f, 4.0f, 6.0f, 1.0f, 9.25f, 2.0f},
		{5.0f, 3.0f, 7.0f, 8.0f, 1.0f, 6.0f, 2.0f, 4.0f, 7.5f, 3.25f},
		{4.0f, 2.0f, 7.0f, 9.25f, 2.0f, 1.0f, 6.5f, 8.5f, 2.5f, 1.0f},
		{1.0f, 7.0f, 5.5f, 6.5f, 7.0f, 2.0f, 4.25f, 2.0f, 9.0f, 1.0f},
		{7.0f, 1.0f, 5.0f, 6.25f, 1.0f, 9.0f, 6.0f, 2.0f, 5.5f, 7.5f},
		});

	dot_b = Matrix({
		{6.0f, 2.0f, 2.0f, 9.5f, 2.25f, 8.0f, 1.0f, 5.5f, 2.5f, 4.0f},
		{1.5f, 2.0f, 2.0f, 7.0f, 5.5f, 3.25f, 1.0f, 8.0f, 9.0f, 1.0f},
		{6.25f, 6.0f, 2.0f, 1.0f, 8.0f, 1.0f, 2.0f, 5.0f, 9.0f, 2.5f},
		{1.5f, 6.5f, 7.0f, 4.0f, 5.5f, 8.25f, 1.5f, 8.0f, 7.0f, 2.0f},
		{1.0f, 6.0f, 2.5f, 7.25f, 8.0f, 2.0f, 5.0f, 3.5f, 1.0f, 6.5f},
		{1.0f, 5.5f, 4.0f, 7.25f, 3.5f, 5.5f, 2.5f, 8.5f, 5.0f, 4.0f},
		{1.5f, 2.0f, 7.0f, 2.0f, 5.5f, 8.25f, 2.5f, 2.0f, 1.0f, 6.5f},
		{9.0f, 2.25f, 2.0f, 7.5f, 7.0f, 1.0f, 4.5f, 1.0f, 6.0f, 9.0f},
		{6.5f, 2.0f, 7.0f, 6.25f, 2.5f, 4.5f, 5.0f, 8.0f, 1.0f, 4.0f,},
		{7.0f, 2.0f, 6.5f, 8.0f, 1.5f, 7.0f, 6.25f, 1.0f, 7.0f, 4.5f},
		});

	check_a = Matrix({
		{203.5f, 165.25f, 190.125f, 240.875f, 248.75f, 180.75f, 164.3125f, 182.5f, 199.5f, 231.125f, },
		{131.0f, 164.0f, 217.625f, 256.75f, 207.25f, 272.375f, 125.5625f, 233.0f, 240.5f, 180.125f, },
		{155.25f, 169.875f, 160.625f, 242.625f, 193.375f, 177.0f, 131.5625f, 223.75f, 242.75f, 150.625f },
		{190.375f, 164.625f, 186.0f, 273.5f, 232.0625f, 192.25f, 146.125f, 231.875f, 236.625f, 199.5f },
		{201.25f, 150.125f, 172.5f, 296.75f, 243.5f, 220.4375f, 144.625f, 194.25f, 217.0f, 240.5f },
		{157.375f, 178.25f, 207.25f, 263.0625f, 249.375f, 206.625f, 155.75f, 265.0f, 195.75f, 198.0f },
		{207.75f, 183.5f, 208.125f, 265.125f, 219.375f, 234.75f, 134.8125f, 276.25f, 245.75f, 180.625f },
		{197.125f, 170.75f, 186.25f, 218.125f, 249.375f, 211.6875f, 119.625f, 205.0f, 229.75f, 204.25f },
		{159.5f, 177.25f, 201.25f, 243.0f, 244.875f, 199.4375f, 139.625f, 266.0f, 209.75f, 177.375f },
		{209.375f, 184.625f, 241.5f, 297.375f, 207.125f, 296.0625f, 153.25f, 267.0f, 237.25f, 209.25f },
		});

	test(dot_a, dot_b, check_a);

}


void test(Matrix a, Matrix b, Matrix c) {

	Matrix temp(c.RowCount, c.ColumnCount, 1.0f);

	std::cout << "Base Case: " <<  (a.dot_product(b) == c) << "\n";

	dot_prod(a.matrix, b.matrix, temp.matrix, a.RowCount, a.ColumnCount, b.RowCount, b.ColumnCount, true);
	std::cout << "Base Check: " << (temp == c) << "\n";

	a = a.Transpose();
	std::cout << "a.T Case: " << (a.Transpose().dot_product(b) == c) << "\n";

	dot_prod_t_a(a.matrix, b.matrix, temp.matrix, a.RowCount, a.ColumnCount, b.RowCount, b.ColumnCount, true);
	std::cout << "a.T Check: " << (temp == c) << "\n";

	// transpose a back to normal
	a = a.Transpose();

	b = b.Transpose();
	std::cout << "b.T Case: " << (a.dot_product(b.Transpose()) == c) << "\n";

	dot_prod_t_b(a.matrix, b.matrix, temp.matrix, a.RowCount, a.ColumnCount, b.RowCount, b.ColumnCount, true);
	std::cout << "b.T Check: " << (temp == c) << "\n";
}