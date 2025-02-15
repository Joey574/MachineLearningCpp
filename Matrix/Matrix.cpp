#include "Matrix.h"

#define BLOCK_SIZE 64
alignas(64) float LOCAL_A[BLOCK_SIZE * BLOCK_SIZE];
alignas(64) float LOCAL_B[BLOCK_SIZE * BLOCK_SIZE];
alignas(64) float LOCAL_C[BLOCK_SIZE * BLOCK_SIZE];
#pragma omp threadprivate(LOCAL_A, LOCAL_B, LOCAL_C)

// Constructors
Matrix::Matrix(size_t rows, size_t columns) : RowCount(rows), ColumnCount(columns), matrix(c_init()) {}
Matrix::Matrix(size_t rows, size_t columns, float value) : RowCount(rows), ColumnCount(columns), matrix(m_init()) {
	std::fill(matrix, matrix + (RowCount * ColumnCount), value);
}
Matrix::Matrix(size_t rows, size_t columns, init initType) : RowCount(rows), ColumnCount(columns), matrix(m_init()) {
	float lowerRand = -0.5f;
	float upperRand = 0.5f;

	std::random_device rd;
	std::mt19937 gen(rd());

	switch (initType) {
	case Matrix::init::Xavier: {
		lowerRand = -(1.0f / std::sqrt(RowCount));
		upperRand = 1.0f / std::sqrt(RowCount);;

		std::uniform_real_distribution<float> dist_x(lowerRand, upperRand);

		for (int i = 0; i < RowCount * ColumnCount; i++) {
			matrix[i] = dist_x(gen);
		}
		break;
	}
	case Matrix::init::He: {
		std::normal_distribution<float> dist_h(0.0, std::sqrt(2.0f / RowCount));

		for (int i = 0; i < RowCount * ColumnCount; i++) {
			matrix[i] = dist_h(gen);
		}
		break;
	}
	case Matrix::init::Normalize: {
		std::uniform_real_distribution<float> dist_n(lowerRand, upperRand);

		for (int i = 0; i < RowCount * ColumnCount; i++) {
			matrix[i] = dist_n(gen) * std::sqrt(1.0f / ColumnCount);
		}
		break;
	}
	case Matrix::init::Random: {
		std::uniform_real_distribution<float> dist_r(lowerRand, upperRand);

		for (int i = 0; i < RowCount * ColumnCount; i++) {
			matrix[i] = dist_r(gen);
		}
		break;
	}
	}
}
Matrix::Matrix(const std::vector<std::vector<float>>& matrix) : RowCount(matrix.size()), ColumnCount(matrix[0].size()), matrix(m_init()) {
	for (int r = 0; r < RowCount; r++) {
		for (int c = 0; c < ColumnCount; c++) {
			this->matrix[r * ColumnCount + c] = matrix[r][c];
		}
	}
}
Matrix::Matrix(const float* matrix, size_t rows, size_t columns) : RowCount(rows), ColumnCount(columns), matrix(m_init()) {
std:memcpy(this->matrix, matrix, rows * columns * sizeof(float));
}
Matrix::Matrix(const Matrix& other) : RowCount(other.RowCount), ColumnCount(other.ColumnCount), matrix(m_init()) {
	std::memcpy(matrix, other.matrix, RowCount * ColumnCount * sizeof(float));
}

// Util
void Matrix::SetColumn(int index, const std::vector<float>& column) {
	for (int i = 0; i < RowCount; i++) {
		matrix[i * ColumnCount + index] = column[i];
	}
}
void Matrix::SetColumn(int index, const std::vector<int>& column) {
	for (int i = 0; i < RowCount; i++) {
		matrix[i * ColumnCount + index] = column[i];
	}
}

void Matrix::SetRow(int index, const std::vector<float>& row) {
	std::memcpy(matrix + (index * ColumnCount), row.data(), ColumnCount * sizeof(float));
}
void Matrix::SetRow(int index, const std::vector<int>& row) {
	for (int i = 0; i < ColumnCount; i++) {
		matrix[index * ColumnCount + i] = row[i];
	}
}

void Matrix::add_row(float* row) {
	RowCount++;
	matrix = (float*)realloc(matrix, RowCount * ColumnCount * sizeof(float));
	std::memcpy(matrix + ((RowCount - 1) * ColumnCount), row, ColumnCount * sizeof(float));
}
void Matrix::add_row(const std::vector<float>& row) {
	RowCount++;
	matrix = (float*)realloc(matrix, RowCount * ColumnCount * sizeof(float));
	std::memcpy(matrix + ((RowCount - 1) * ColumnCount), row.data(), ColumnCount * sizeof(float));
}
void Matrix::add_row(const std::vector<int>& row) {
	RowCount++;
	matrix = (float*)realloc(matrix, RowCount * ColumnCount * sizeof(float));

	for (int i = 0; i < ColumnCount; i++) {
		matrix[(RowCount - 1) * ColumnCount + i] = row[i];
	}
}

void Matrix::Insert(int startRow, Matrix element) {
	for (int i = 0; i < element.RowCount; i++) {
		this->SetRow(i + startRow, element.Row(i));
	}
}

Matrix Matrix::SegmentR(int startRow, int endRow) const {
	Matrix a = Matrix(endRow - startRow, ColumnCount);

	int r_idx = 0;
	for (int r = startRow; r_idx < a.RowCount; r++, r_idx++) {
		for (int c = 0; c < a.ColumnCount; c++) {
			a.matrix[r_idx * ColumnCount + c] = matrix[r * ColumnCount + c];
		}
	}

	return a;
}
Matrix Matrix::SegmentR(int startRow) const {
	Matrix a = Matrix(RowCount - startRow, ColumnCount);

	int r_idx = 0;
	for (int r = startRow; r_idx < a.RowCount; r++, r_idx++) {
		for (int c = 0; c < a.ColumnCount; c++) {
			a.matrix[r_idx * ColumnCount + c] = matrix[r * ColumnCount + c];
		}
	}

	return a;
}

Matrix Matrix::SegmentC(int startColumn, int endColumn) const {
	Matrix a = Matrix(RowCount, endColumn - startColumn);

	for (int r = 0; r < a.RowCount; r++) {

		int c_idx = 0;
		for (int c = startColumn; c_idx < a.ColumnCount; c++, c_idx++) {
			a.matrix[r * a.ColumnCount + c_idx] = matrix[r * ColumnCount + c];
		}
	}

	return a;
}
Matrix Matrix::SegmentC(int startColumn) const {
	Matrix a = Matrix(RowCount, ColumnCount - startColumn);

	for (int r = 0; r < a.RowCount; r++) {

		int c_idx = 0;
		for (int c = startColumn; c_idx < a.ColumnCount; c++, c_idx++) {
			a.matrix[r * a.ColumnCount + c_idx] = matrix[r * ColumnCount + c];
		}
	}

	return a;
}

std::vector<float> Matrix::ColumnSums() const {
	std::vector<float> sums(ColumnCount);

	for (int r = 0; r < RowCount; r++) {

		int c = 0;
		for (; c + 8 <= ColumnCount; c += 8) {
			_mm256_store_ps(&sums[c],
				_mm256_add_ps(
					_mm256_load_ps(&matrix[r * ColumnCount + c]),
					_mm256_load_ps(&sums[c])
				));
		}

		for (; c < ColumnCount; c++) {
			sums[c] += matrix[r * ColumnCount + c];
		}

		/*for (int c = 0; c < ColumnCount; c++) {
			sums[c] += matrix[r * ColumnCount + c];
		}*/
	}

	return sums;
}
std::vector<float> Matrix::RowSums() const {
	std::vector<float> sums(RowCount);

	for (int r = 0; r < RowCount; r++) {
		for (int c = 0; c < ColumnCount; c++) {
			sums[r] += matrix[r * ColumnCount + c];
		}
	}
	return sums;
}

std::vector<float> Matrix::Column(int index) const {
	std::vector<float> column(RowCount);

	for (int i = 0; i < RowCount; i++) {
		column[i] = matrix[i * ColumnCount + index];
	}

	return column;
}
std::vector<float> Matrix::Row(int index) const {
	std::vector<float> row(ColumnCount);

	std::memcpy(row.data(), matrix + (index * ColumnCount), ColumnCount * sizeof(float));

	return row;
}

// "Advanced" math
Matrix Matrix::extract_features(int fourier, int taylor, int chebyshev, int legendre, int laguerre, float lowerNormal, float upperNormal) const {
	// Normalize
	Matrix mat = *this;
	Matrix taylorNormal;
	Matrix fourierNormal;
	Matrix chebyshevNormal;

	if (fourier) { fourierNormal = mat.normalized(-M_PI, M_PI); }
	if (taylor) { taylorNormal = mat.normalized(0.0f, 1.0f); }
	if (chebyshev + legendre + laguerre) { chebyshevNormal = mat.normalized(-1.0f, 1.0f); }

	// Compute Fourier Series
	if (fourier) {
		for (int f = 0; f < fourier; f++) {
			mat = mat.Combine(fourierNormal.fourier_series(f + 1));
		}
	}

	// Compute Taylor Series
	if (taylor) {
		for (int t = 0; t < taylor; t++) {
			mat = mat.Combine(taylorNormal.taylor_series(t + 1));
		}
	}

	// Compute Chebyshev Series
	if (chebyshev) {
		for (int c = 0; c < chebyshev; c++) {
			mat = mat.Combine(chebyshevNormal.chebyshev_series(c + 1));
		}
	}

	// Compute Legendre Series
	if (legendre) {
		for (int l = 0; l < legendre; l++) {
			mat = mat.Combine(chebyshevNormal.legendre_series(l + 1));
		}
	}

	// Compute Laguerre Series
	if (laguerre) {
		for (int l = 0; l < laguerre; l++) {
			mat = mat.Combine(chebyshevNormal.laguerre_series(l + 1));
		}
	}

	mat = mat.normalized(lowerNormal, upperNormal);

	return mat;
}

Matrix Matrix::normalized(float lowerRange, float upperRange) const noexcept {

	Matrix normal = *this;

	float min = *std::min_element(matrix, matrix + RowCount * ColumnCount);
	float max = *std::max_element(matrix, matrix + RowCount * ColumnCount);

	for (int r = 0; r < RowCount; r++) {
		std::vector<float> vec = this->Row(r);

		for (int i = 0; i < vec.size(); i++) {
			vec[i] = lowerRange + ((vec[i] - min) / (max - min) * (upperRange - lowerRange));
		}

		normal.SetRow(r, vec);
	}
	return normal;
}

Matrix Matrix::fourier_series(int n) const noexcept {
	return this->Multiply(n).Sin().Combine(this->Multiply(n).Cos());
}
Matrix Matrix::taylor_series(int n) const noexcept {
	return this->Pow(n);
}
Matrix Matrix::chebyshev_series(int n) const noexcept {
	return this->Acos().Multiply(n).Cos();
}
Matrix Matrix::legendre_series(int n) const noexcept {
	return (this->Pow(2) - 1).Pow(n);
}
Matrix Matrix::laguerre_series(int n) const noexcept {
	return this->Pow(n).Multiply(this->Negative().Exp());
}

Matrix Matrix::dot_product(const Matrix& element) const {
	Matrix mat(RowCount, element.ColumnCount);

	// error handling -> for losers
	if (ColumnCount != element.RowCount) {
		std::cout << "size mismatch\n";
	}

	#pragma omp parallel for
	for (int r = 0; r < RowCount; r++) {
		for (int k = 0; k < element.RowCount; k++) {
			__m256 scalar = _mm256_set1_ps(matrix[r * ColumnCount + k]);

			int c = 0;
			for (; c + 16 <= element.ColumnCount; c += 8) {

				_mm256_store_ps(&mat.matrix[r * element.ColumnCount + c], 
					_mm256_fmadd_ps(_mm256_load_ps(
						&element.matrix[k * element.ColumnCount + c]), 
						scalar,
						_mm256_load_ps(&mat.matrix[r * element.ColumnCount + c])));

				c += 8;
				_mm256_store_ps(&mat.matrix[r * element.ColumnCount + c],
					_mm256_fmadd_ps(_mm256_load_ps(
						&element.matrix[k * element.ColumnCount + c]),
						scalar,
						_mm256_load_ps(&mat.matrix[r * element.ColumnCount + c])));
			}

			for (; c < element.ColumnCount; c++) {
				mat.matrix[r * element.ColumnCount + c] += matrix[r * ColumnCount + k] * element.matrix[k * element.ColumnCount + c];
			}
		}
	}

	return mat;
}

std::vector<float> Matrix::log_sum_exp() const noexcept {

	std::vector<float> logSum(ColumnCount);

	for (int c = 0; c < ColumnCount; c++) {

		std::vector<float> col = Column(c);

		float max = *std::max_element(col.begin(), col.end());
		float sum = 0;

		for (int i = 0; i < col.size(); i++) {
			sum += std::exp(col[i] - max);
		}
		logSum[c] = max + std::log(sum);
	}
	return logSum;
}

// Basic Math
Matrix Matrix::Negative() const {
	return single_float_operation(&Matrix::SIMDMul, &Matrix::RemainderMul, -1);
}
Matrix Matrix::Abs() const {
	return single_float_operation(&Matrix::SIMDAbs, &Matrix::RemainderAbs, 0);
}

Matrix Matrix::Add(float scalar) const {
	return single_float_operation(&Matrix::SIMDAdd, &Matrix::RemainderAdd, scalar);
}
Matrix Matrix::Add(const std::vector<float>& scalar) const {
	return vector_float_operation(&Matrix::SIMDAdd, &Matrix::RemainderAdd, scalar);
}
Matrix Matrix::Add(const Matrix& element) const {
	return matrix_float_operation(&Matrix::SIMDAdd, &Matrix::RemainderAdd, element);
}

void Matrix::Add(float scalar, Matrix& store) const {
	single_float_operation_store(&Matrix::SIMDAdd, &Matrix::RemainderAdd, scalar, store);
}
void Matrix::Add(const std::vector<float>& scalar, Matrix& store) const {
	vector_float_operation_store(&Matrix::SIMDAdd, &Matrix::RemainderAdd, scalar, store);
}
void Matrix::Add(const Matrix& element, Matrix& store) const {
	matrix_float_operation_store(&Matrix::SIMDAdd, &Matrix::RemainderAdd, element, store);
}

Matrix Matrix::Subtract(float scalar) const {
	return single_float_operation(&Matrix::SIMDSub, &Matrix::RemainderSub, scalar);
}
Matrix Matrix::Subtract(const std::vector<float>& scalar) const {
	return vector_float_operation(&Matrix::SIMDSub, &Matrix::RemainderSub, scalar);
}
Matrix Matrix::Subtract(const Matrix& element) const {
	return matrix_float_operation(&Matrix::SIMDSub, &Matrix::RemainderSub, element);
}

void Matrix::Subtract(float scalar, Matrix& store) const {
	single_float_operation_store(&Matrix::SIMDSub, &Matrix::RemainderSub, scalar, store);
}
void Matrix::Subtract(const std::vector<float>& scalar, Matrix& store) const {
	vector_float_operation_store(&Matrix::SIMDSub, &Matrix::RemainderSub, scalar, store);
}
void Matrix::Subtract(const Matrix& element, Matrix& store) const {
	matrix_float_operation_store(&Matrix::SIMDSub, &Matrix::RemainderSub, element, store);
}

Matrix Matrix::Multiply(float scalar) const {
	return single_float_operation(&Matrix::SIMDMul, &Matrix::RemainderMul, scalar);
}
Matrix Matrix::Multiply(const std::vector<float>& scalar) const {
	return vector_float_operation(&Matrix::SIMDMul, &Matrix::RemainderMul, scalar);
}
Matrix Matrix::Multiply(const Matrix& element) const {
	return matrix_float_operation(&Matrix::SIMDMul, &Matrix::RemainderMul, element);
}

void Matrix::Multiply(float scalar, Matrix& store) const {
	single_float_operation_store(&Matrix::SIMDMul, &Matrix::RemainderMul, scalar, store);
}
void Matrix::Multiply(const std::vector<float>& scalar, Matrix& store) const {
	vector_float_operation_store(&Matrix::SIMDMul, &Matrix::RemainderMul, scalar, store);
}
void Matrix::Multiply(const Matrix& element, Matrix& store) const {
	matrix_float_operation_store(&Matrix::SIMDMul, &Matrix::RemainderMul, element, store);
}

Matrix Matrix::Divide(float scalar) const {
	return single_float_operation(&Matrix::SIMDDiv, &Matrix::RemainderDiv, scalar);
}
Matrix Matrix::Divide(const std::vector<float>& scalar) const {
	return vector_float_operation(&Matrix::SIMDDiv, &Matrix::RemainderDiv, scalar);
}
Matrix Matrix::Divide(const Matrix& element) const {
	return matrix_float_operation(&Matrix::SIMDDiv, &Matrix::RemainderDiv, element);
}

void Matrix::Divide(float scalar, Matrix& store) const {
	single_float_operation_store(&Matrix::SIMDDiv, &Matrix::RemainderDiv, scalar, store);
}
void Matrix::Divide(const std::vector<float>& scalar, Matrix& store) const {
	vector_float_operation_store(&Matrix::SIMDDiv, &Matrix::RemainderDiv, scalar, store);
}
void Matrix::Divide(const Matrix& element, Matrix& store) const {
	matrix_float_operation_store(&Matrix::SIMDDiv, &Matrix::RemainderDiv, element, store);
}

 Matrix Matrix::Pow(float scalar) const {
	return single_float_operation(&Matrix::SIMDPow, &Matrix::RemainderPow, scalar);
}
 Matrix Matrix::Pow(const std::vector<float>& scalar) const {
	return vector_float_operation(&Matrix::SIMDPow, &Matrix::RemainderPow, scalar);
}
 Matrix Matrix::Pow(const Matrix& element) const {
	return matrix_float_operation(&Matrix::SIMDPow, &Matrix::RemainderPow, element);
}

 Matrix Matrix::Exp(float base) const {
	return single_float_operation(&Matrix::SIMDExp, &Matrix::RemainderExp, base);
}
 Matrix Matrix::Exp(const std::vector<float>& base) const {
	return vector_float_operation(&Matrix::SIMDExp, &Matrix::RemainderExp, base);
}
 Matrix Matrix::Exp(const Matrix& base) const {
	return matrix_float_operation(&Matrix::SIMDExp, &Matrix::RemainderExp, base);
}

 Matrix Matrix::Log() const {
	return single_float_operation(&Matrix::SIMDLog, &Matrix::RemainderLog, 0);
}

// Trig
Matrix Matrix::Cos() const {
	return this->single_float_operation(&Matrix::SIMDCos, &Matrix::RemainderCos, 0);
}
Matrix Matrix::Sin() const {
	return this->single_float_operation(&Matrix::SIMDSin, &Matrix::RemainderSin, 0);
}
Matrix Matrix::Acos() const {
	return this->single_float_operation(&Matrix::SIMDAcos, &Matrix::RemainderAcos, 0);
}
Matrix Matrix::Asin() const {
	return this->single_float_operation(&Matrix::SIMDAsin, &Matrix::RemainderAsin, 0);
}

// Activation Functions
Matrix Matrix::Sigmoid() const {
	Matrix one = Matrix(RowCount, ColumnCount, 1);
	return one / (this->Negative().Exp() + 1);
}
Matrix Matrix::ReLU() const {
	return single_float_operation(&Matrix::SIMDMax, &Matrix::RemainderMax, 0);
}
Matrix Matrix::LeakyReLU() const {
	return matrix_float_operation(&Matrix::SIMDMax, &Matrix::RemainderMax, this->Multiply(0.1f));
}
Matrix Matrix::ELU() const {
	Matrix a = *this;

	#pragma omp parallel for
	for (int r = 0; r < RowCount; r++) {
		for (int c = 0; c < ColumnCount; c++) {
			a(r, c) = matrix[r * ColumnCount + c] < 0.0f ? (std::exp(matrix[r * ColumnCount + c]) - 1) : matrix[r * ColumnCount + c];
		}
	}
	return a;
}
Matrix Matrix::Tanh() const {
	Matrix a = this->Exp();
	Matrix b = this->Negative().Exp();

	return (a - b) / (a + b);
}
Matrix Matrix::Softplus() const {
	return (this->Exp() + 1).Log();
}
Matrix Matrix::SiLU() const {
	return this->Divide((this->Negative().Exp() + 1.0f));
}

Matrix Matrix::SoftMax() const {
	return this->Subtract(this->log_sum_exp()).Exp();
}

// Activation Derivatives
Matrix Matrix::SigmoidDerivative() const {
	Matrix a = *this;
	return this->Sigmoid() * (a - this->Sigmoid());
}
Matrix Matrix::ReLUDerivative() const {
	Matrix a = *this;

	for (int r = 0; r < this->RowCount; r++) {
		for (int c = 0; c < this->ColumnCount; c++) {
			a(r, c) = matrix[r * ColumnCount + c] > 0.0f ? 1.0f : 0.0f;
		}
	}
	return a;
}
Matrix Matrix::LeakyReLUDerivative() const {
	Matrix deriv = *this;

	#pragma omp parallel for
	for (int c = 0; c < ColumnCount; c++) {
		for (int r = 0; r < RowCount; r++) {
			deriv(r, c) = deriv(r, c) > 0.0f ? 1.0f : 0.1f;
		}
	}
	return deriv;
}
Matrix Matrix::ELUDerivative() const {
	Matrix a = *this;

	#pragma omp parallel for
	for (int r = 0; r < this->RowCount; r++) {
		for (int c = 0; c < this->ColumnCount; c++) {
			a(r, c) = matrix[r * ColumnCount + c] > 0.0f ? 1.0f : std::exp(matrix[r * ColumnCount + c]);
		}
	}
	return a;
}
Matrix Matrix::TanhDerivative() const {
	Matrix one = Matrix(this->RowCount, this->ColumnCount, 1);
	return one - this->Tanh().Pow(2);
}
Matrix Matrix::SoftplusDerivative() const {
	Matrix one = Matrix(this->RowCount, this->ColumnCount, 1);
	return one / (this->Negative().Exp() + 1);
}
Matrix Matrix::SiLUDerivative() const {
	return (this->Negative().Exp() + (this->Multiply(this->Negative().Exp()) + 1) / (this->Negative().Exp() + 1).Pow(2));
}

// SIMD Implementation new object
Matrix Matrix::single_float_operation(__m256 (Matrix::* operation)(__m256 opOne, __m256 opTwo) const noexcept,
	float (Matrix::* remainderOperation)(float a, float b) const noexcept, float scalar) const {
	Matrix mat = *this;

	__m256 _scalar = _mm256_set1_ps(scalar);

	#pragma omp parallel for
	for (int i = 0; i < RowCount * ColumnCount; i += 8) {
		_mm256_store_ps(&mat.matrix[i], (this->*operation)(_mm256_load_ps(&mat.matrix[i]), _scalar));
	}

	return mat;
}

Matrix Matrix::vector_float_operation(__m256 (Matrix::* operation)(__m256 opOne, __m256 opTwo) const noexcept,
	float (Matrix::* remainderOperation)(float a, float b) const noexcept, const std::vector<float>& scalar) const {

	Matrix mat = *this;

	if (scalar.size() == ColumnCount) {
		#pragma omp parallel for
		for (int r = 0; r < RowCount; r++) {

			int c = 0;
			for (; c + 8 < ColumnCount; c += 8) {
				_mm256_store_ps(&mat.matrix[r * ColumnCount + c], (this->*operation)(_mm256_load_ps(&mat.matrix[r * ColumnCount + c]), _mm256_load_ps(&scalar[c])));
			}

			for (; c < ColumnCount; c++) {
				mat.matrix[r * ColumnCount + c] = (this->*remainderOperation)(mat.matrix[r * ColumnCount + c], scalar[c]);
			}
		}
	} else if (scalar.size() == RowCount) {
		#pragma omp parallel for
		for (int r = 0; r < RowCount; r++) {
			__m256 _scalar = _mm256_set1_ps(scalar[r]);

			int c = 0;
			for (; c + 8 < ColumnCount; c += 8) {
				_mm256_store_ps(&mat.matrix[r * ColumnCount + c], (this->*operation)(_mm256_load_ps(&mat.matrix[r * ColumnCount + c]), _scalar));
			}

			for (; c < ColumnCount; c++) {
				mat.matrix[r * ColumnCount + c] = (this->*remainderOperation)(mat.matrix[r * ColumnCount + c], scalar[r]);
			}
		}
	}

	return mat;
}

Matrix Matrix::matrix_float_operation(__m256 (Matrix::* operation)(__m256 opOne, __m256 opTwo) const noexcept,
	float (Matrix::* remainderOperation)(float a, float b) const noexcept, const Matrix& element) const {
	Matrix mat = element;

	#pragma omp parallel for
	for (int i = 0; i < RowCount * ColumnCount; i += 8) {
		_mm256_store_ps(&mat.matrix[i], (this->*operation)(_mm256_load_ps(&matrix[i]), _mm256_load_ps(&mat.matrix[i])));
	}

	return mat;
}


// SIMD Implementation in place
void Matrix::single_float_operation_in_place(__m256 (Matrix::* operation)(__m256 opOne, __m256 opTwo) const,
	float (Matrix::* remainderOperation)(float a, float b) const, float scalar) {

	__m256 _scalar = _mm256_set1_ps(scalar);

	#pragma omp parallel for
	for (int i = 0; i < RowCount * ColumnCount; i += 8) {
		_mm256_store_ps(&matrix[i], (this->*operation)(_mm256_load_ps(&matrix[i]), _scalar));
	}
}

void Matrix::vector_float_operation_in_place(__m256 (Matrix::* operation)(__m256 opOne, __m256 opTwo) const,
	float (Matrix::* remainderOperation)(float a, float b) const, const std::vector<float>& scalar) {

	if (scalar.size() == ColumnCount) {

		#pragma omp parallel for
		for (int r = 0; r < RowCount; r++) {

			int c = 0;
			for (; c + 8 < ColumnCount; c += 8) {
				_mm256_store_ps(&matrix[r * ColumnCount + c], (this->*operation)(_mm256_load_ps(&matrix[r * ColumnCount + c]), _mm256_load_ps(&scalar[c])));
			}

			for (; c < ColumnCount; c++) {
				matrix[r * ColumnCount + c] = (this->*remainderOperation)(matrix[r * ColumnCount + c], scalar[c]);
			}
		}
	} else if (scalar.size() == RowCount) {

		#pragma omp parallel for
		for (int r = 0; r < RowCount; r++) {
			__m256 loaded_b = _mm256_set1_ps(scalar[r]);

			int c = 0;
			for (; c + 8 < ColumnCount; c += 8) {
				_mm256_store_ps(&matrix[r * ColumnCount + c], (this->*operation)(_mm256_load_ps(&matrix[r * ColumnCount + c]), loaded_b));
			}

			for (; c < ColumnCount; c++) {
				matrix[r * ColumnCount + c] = (this->*remainderOperation)(matrix[r * ColumnCount + c], scalar[r]);
			}
		}
	}

}

void Matrix::matrix_float_operation_in_place(__m256 (Matrix::* operation)(__m256 opOne, __m256 opTwo) const,
	float (Matrix::* remainderOperation)(float a, float b) const, const Matrix& element) {

	#pragma omp parallel for
	for (int i = 0; i < RowCount * ColumnCount; i += 8) {
		_mm256_store_ps(&matrix[i], (this->*operation)(_mm256_load_ps(&matrix[i]), _mm256_load_ps(&element.matrix[i])));
	}
}


// SIMD Implementation store
void Matrix::single_float_operation_store(__m256(Matrix::* operation)(__m256, __m256) const noexcept,
	float(Matrix::* remainderOperation)(float a, float b) const noexcept, float scalar, Matrix& store) const {

	__m256 _scalar = _mm256_set1_ps(scalar);

	#pragma omp parallel for
	for (int i = 0; i < RowCount * ColumnCount; i += 8) {
		_mm256_store_ps(&store.matrix[i], (this->*operation)(_mm256_load_ps(&matrix[i]), _scalar));
	}
}

void Matrix::vector_float_operation_store(__m256(Matrix::* operation)(__m256, __m256) const noexcept,
	float(Matrix::* remainderOperation)(float a, float b) const noexcept, const std::vector<float>& scalar, Matrix& store) const {

	if (scalar.size() == ColumnCount) {

		#pragma omp parallel for
		for (int r = 0; r < RowCount; r++) {

			int c = 0;
			for (; c + 8 < ColumnCount; c += 8) {
				_mm256_store_ps(&store.matrix[r * ColumnCount + c], (this->*operation)(_mm256_load_ps(&matrix[r * ColumnCount + c]), _mm256_load_ps(&scalar[c])));
			}

			for (; c < ColumnCount; c++) {
				store.matrix[r * ColumnCount + c] = (this->*remainderOperation)(matrix[r * ColumnCount + c], scalar[c]);
			}
		}
	}
	else if (scalar.size() == RowCount) {

		#pragma omp parallel for
		for (int r = 0; r < RowCount; r++) {
			__m256 loaded_b = _mm256_set1_ps(scalar[r]);

			int c = 0;
			for (; c + 8 < ColumnCount; c += 8) {
				_mm256_store_ps(&store.matrix[r * ColumnCount + c], (this->*operation)(_mm256_load_ps(&matrix[r * ColumnCount + c]), loaded_b));
			}

			for (; c < ColumnCount; c++) {
				store.matrix[r * ColumnCount + c] = (this->*remainderOperation)(matrix[r * ColumnCount + c], scalar[r]);
			}
		}
	}

}

void Matrix::matrix_float_operation_store(__m256(Matrix::* operation)(__m256, __m256) const noexcept,
	float(Matrix::* remainderOperation)(float a, float b) const noexcept, const Matrix& element, Matrix& store) const {

	#pragma omp parallel for
	for (int i = 0; i < RowCount * ColumnCount; i += 8) {
		_mm256_store_ps(&store.matrix[i], (this->*operation)(_mm256_load_ps(&matrix[i]), _mm256_load_ps(&element.matrix[i])));
	}
}


// SIMD Operations
inline __m256 Matrix::SIMDAdd(__m256 opOne, __m256 opTwo) const noexcept {
	return _mm256_add_ps(opOne, opTwo);
}
inline __m256 Matrix::SIMDSub(__m256 opOne, __m256 opTwo) const noexcept {
	return _mm256_sub_ps(opOne, opTwo);
}
inline __m256 Matrix::SIMDMul(__m256 opOne, __m256 opTwo) const noexcept {
	return _mm256_mul_ps(opOne, opTwo);
}
inline __m256 Matrix::SIMDDiv(__m256 opOne, __m256 opTwo) const noexcept {
	return _mm256_div_ps(opOne, opTwo);
}
inline __m256 Matrix::SIMDPow(__m256 opOne, __m256 opTwo) const noexcept {
	return _mm256_pow_ps(opOne, opTwo);
}
inline __m256 Matrix::SIMDExp(__m256 opOne, __m256 opTwo) const noexcept {
	return _mm256_pow_ps(opTwo, opOne);
}
inline __m256 Matrix::SIMDLog(__m256 opOne, __m256 opTwo) const noexcept {
	return _mm256_log_ps(opOne);
}
inline __m256 Matrix::SIMDMax(__m256 opOne, __m256 opTwo) const noexcept {
	return _mm256_max_ps(opOne, opTwo);
}
inline __m256 Matrix::SIMDAbs(__m256 opOne, __m256 opTwo) const noexcept {
	__m256 mask = _mm256_castsi256_ps(_mm256_srli_epi32(_mm256_set1_epi32(-1), 1));
	__m256 result = _mm256_and_ps(opOne, mask);
	return result;
}

// SIMD Trig
inline __m256 Matrix::SIMDSin(__m256 opOne, __m256 opTwo) const noexcept {
	return _mm256_sin_ps(opOne);
}
inline __m256 Matrix::SIMDCos(__m256 opOne, __m256 opTwo) const noexcept {
	return _mm256_cos_ps(opOne);
}
inline __m256 Matrix::SIMDSec(__m256 opOne, __m256 opTwo) const noexcept {
	return _mm256_div_ps(_mm256_set1_ps(1.0f), _mm256_cos_ps(opOne));
}
inline __m256 Matrix::SIMDCsc(__m256 opOne, __m256 opTwo) const noexcept {
	return _mm256_div_ps(_mm256_set1_ps(1.0f), _mm256_sin_ps(opOne));
}
inline __m256 Matrix::SIMDAcos(__m256 opOne, __m256 opTwo) const noexcept {
	return _mm256_acos_ps(opOne);
}
inline __m256 Matrix::SIMDAsin(__m256 opOne, __m256 opTwo) const noexcept {
	return _mm256_asin_ps(opOne);
}

// Remainder Operations
inline float Matrix::RemainderAdd(float a, float b) const noexcept {
	return a + b;
}
inline float Matrix::RemainderSub(float a, float b) const noexcept {
	return a - b;
}
inline float Matrix::RemainderMul(float a, float b) const noexcept {
	return a * b;
}
inline float Matrix::RemainderDiv(float a, float b) const noexcept {
	return a / b;
}
inline float Matrix::RemainderPow(float a, float b) const noexcept {
	return std::pow(a, b);
}
inline float Matrix::RemainderExp(float a, float b) const noexcept {
	return std::pow(b, a);
}
inline float Matrix::RemainderLog(float a, float b) const noexcept {
	return std::log(a);
}
inline float Matrix::RemainderMax(float a, float b) const noexcept {
	return std::max(a, b);
}
inline float Matrix::RemainderAbs(float a, float b) const noexcept {
	return std::abs(a);
}

// Remainder Trig
inline float Matrix::RemainderSin(float a, float b) const noexcept {
	return std::sin(a);
}
inline float Matrix::RemainderCos(float a, float b) const noexcept {
	return std::cos(a);
}
inline float Matrix::RemainderSec(float a, float b) const noexcept {
	return 1.0f / std::cos(a);
}
inline float Matrix::RemainderCsc(float a, float b) const noexcept {
	return 1.0f / std::sin(a);
}
inline float Matrix::RemainderAcos(float a, float b) const noexcept {
	return std::acos(a);
}
inline float Matrix::RemainderAsin(float a, float b) const noexcept {
	return std::asin(a);
}

// MISC
std::string Matrix::ToString() const {
	std::string out = "";

	for (int r = 0; r < RowCount; r++) {
		for (int c = 0; c < ColumnCount; c++) {
			out += std::to_string(matrix[r * ColumnCount + c]) + " ";
		}
		out += "\n";
	}

	return out;
}
std::string Matrix::Size() const {
	return std::to_string(RowCount).append(" :: ").append(std::to_string(ColumnCount)).append("\n");
}

Matrix Matrix::Combine(Matrix element) {
	Matrix a(RowCount + element.RowCount, ColumnCount);

	std::memcpy(a.matrix, matrix, RowCount * ColumnCount * sizeof(float));
	std::memcpy(a.matrix + (RowCount * ColumnCount), element.matrix, element.RowCount * element.ColumnCount * sizeof(float));

	return a;
}

Matrix Matrix::Join(Matrix element) {
	Matrix a(RowCount, ColumnCount + element.ColumnCount);

	for (int r = 0; r < RowCount; r++) {
		for (int c = 0; c < ColumnCount + element.ColumnCount; c++) {
			a.matrix[r * ColumnCount + c] = (c < ColumnCount ? (matrix[r * ColumnCount + c]) : (element(r, c - ColumnCount)));
		}
	}

	return a;
}

Matrix Matrix::Transpose() const {
	Matrix a(ColumnCount, RowCount);

	/*#pragma omp parallel for
	for (int c = 0; c < ColumnCount; c++) {
		for (int r = 0; r < RowCount; r++) {
			a.matrix[c * a.ColumnCount + r] = matrix[r * ColumnCount + c];
		}
	}*/

	/*#pragma omp parallel for
	for (int r = 0; r < RowCount; r++) {
		for (int c = 0; c < ColumnCount; c++) {
			a.matrix[c * a.ColumnCount + r] = matrix[r * ColumnCount + c];
		}
	}*/

	#pragma omp parallel for
	for (int c = 0; c < ColumnCount; c++) {

		int r = 0;
		for (; r + 8 <= RowCount; r += 8) {
			_mm256_store_ps(&a.matrix[c * a.ColumnCount + r], {
				matrix[r * ColumnCount + c],
				matrix[(r + 1) * ColumnCount + c],
				matrix[(r + 2) * ColumnCount + c],
				matrix[(r + 3) * ColumnCount + c],
				matrix[(r + 4) * ColumnCount + c],
				matrix[(r + 5) * ColumnCount + c],
				matrix[(r + 6) * ColumnCount + c],
				matrix[(r + 7) * ColumnCount + c]
				});
		}

		for (; r < RowCount; r++) {
			a.matrix[c * a.ColumnCount + r] = matrix[r * ColumnCount + c];
		}
	}

	return a;
}

bool Matrix::contains_nan() const {
	return std::any_of(matrix, (matrix + (RowCount * ColumnCount)), [](float value) {
		return std::isnan(value);
		});
}