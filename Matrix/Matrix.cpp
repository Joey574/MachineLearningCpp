#include "Matrix.h"
#include <iostream>

// Constructors
Matrix::Matrix(size_t rows, size_t columns) : RowCount(rows), ColumnCount(columns) {
	this->matrix = (float*)calloc(RowCount * ColumnCount, sizeof(float));
}
Matrix::Matrix(size_t rows, size_t columns, float value) : RowCount(rows), ColumnCount(columns), matrix((float*)malloc(rows * columns * sizeof(float))) {
	std::fill(matrix, matrix + (RowCount * ColumnCount), value);
}
Matrix::Matrix(size_t rows, size_t columns, init initType) : RowCount(rows), ColumnCount(columns), matrix((float*)malloc(rows * columns * sizeof(float))) {
	float lowerRand = -0.5f;
	float upperRand = 0.5f;

	std::random_device rd;
	std::mt19937 gen(rd());

	switch (initType) {
	case Matrix::init::Xavier: {
		lowerRand = -(1.0f / std::sqrt(RowCount));
		upperRand = 1.0f / std::sqrt(RowCount);;

		std::uniform_real_distribution<float> dist_x(lowerRand, upperRand);

		for (int r = 0; r < RowCount; r++) {
			for (int c = 0; c < ColumnCount; c++) {
				matrix[r * ColumnCount + c] = dist_x(gen);
			}
		}
		break;
	}
	case Matrix::init::He: {
		std::normal_distribution<float> dist_h(0.0, std::sqrt(2.0f / RowCount));

		for (int r = 0; r < RowCount; r++) {
			for (int c = 0; c < ColumnCount; c++) {
				matrix[r * ColumnCount + c] = dist_h(gen);
			}
		}
		break;
	}
	case Matrix::init::Normalize: {
		std::uniform_real_distribution<float> dist_n(lowerRand, upperRand);

		for (int r = 0; r < RowCount; r++) {
			for (int c = 0; c < ColumnCount; c++) {
				matrix[r * ColumnCount + c] = dist_n(gen) * std::sqrt(1.0f / ColumnCount);
			}
		}
		break;
	}
	case Matrix::init::Random: {
		std::uniform_real_distribution<float> dist_r(lowerRand, upperRand);

		for (int r = 0; r < RowCount; r++) {
			for (int c = 0; c < ColumnCount; c++) {
				matrix[r * ColumnCount + c] = dist_r(gen);
			}
		}
		break;
	}
	}
}
Matrix::Matrix(const std::vector<std::vector<float>>& matrix) : RowCount(matrix.size()), ColumnCount(matrix[0].size()) {
	this->matrix = (float*)malloc(RowCount * ColumnCount * sizeof(float));

	for (int r = 0; r < RowCount; r++) {
		for (int c = 0; c < ColumnCount; c++) {
			this->matrix[r * ColumnCount + c] = matrix[r][c];
		}
	}
}
Matrix::Matrix(const float* matrix, size_t rows, size_t columns) : RowCount(rows), ColumnCount(columns), matrix((float*)malloc(rows* columns * sizeof(float))) {
std:memcpy(this->matrix, matrix, rows * columns * sizeof(float));
}
Matrix::Matrix(const Matrix& other) : RowCount(other.RowCount), ColumnCount(other.ColumnCount), matrix((float*)malloc(RowCount * ColumnCount * sizeof(float))) {
	std::memcpy(matrix, other.matrix, RowCount * ColumnCount * sizeof(float));
}
// Util

void Matrix::SetColumn(int index, const std::vector<float>& vector) {
	for (int i = 0; i < RowCount; i++) {
		matrix[i * ColumnCount + index] = vector[i];
	}
}
void Matrix::SetColumn(int index, const std::vector<int>& vector) {
	for (int i = 0; i < RowCount; i++) {
		matrix[i * ColumnCount + index] = vector[i];
	}
}

void Matrix::SetRow(int index, const std::vector<float>& vector) {
	std::memcpy(matrix + (index * ColumnCount), vector.data(), ColumnCount * sizeof(float));
}
void Matrix::SetRow(int index, const std::vector<int>& vector) {
	for (int i = 0; i < ColumnCount; i++) {
		matrix[index * ColumnCount + i] = vector[i];
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
		for (int c = 0; c < ColumnCount; c++) {
			sums[c] += matrix[r * ColumnCount + c];
		}
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

	if (ColumnCount != element.RowCount) {
		std::cout << "size mismatch\n";
	}

	/*for (int i = 0; i < RowCount; i++) {
		mat.SetRow(i, element.Multiply(this->Row(i)).ColumnSums());
	}

	return mat;*/


	for (int c = 0; c < element.ColumnCount; c++) {

		std::vector<float> column = element.Column(c);

		for (int r = 0; r < RowCount; r++) {
			__m256 sum = _mm256_setzero_ps();


			int k = 0;
			for (; k + 8 <= ColumnCount; k += 8) {
				__m256 loaded_a = _mm256_load_ps(&matrix[r * ColumnCount + k]);
				__m256 loaded_b = _mm256_load_ps(&column[k]);

				sum = _mm256_fmadd_ps(loaded_a, loaded_b, sum);
			}

			float result[4];

			__m128 _a = _mm256_extractf128_ps(sum, 0);
			__m128 _b = _mm256_extractf128_ps(sum, 1);

			_a = _mm_hadd_ps(_a, _a);
			_b = _mm_hadd_ps(_b, _b);

			_a = _mm_hadd_ps(_a, _b);

			_mm_store_ps(result, _a);

			for (; k < ColumnCount; k++) {
				result[0] += matrix[r * ColumnCount + k] * element.matrix[k * element.ColumnCount + c];
			}

			mat.matrix[r * mat.ColumnCount + c] = result[0];
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
	return SingleFloatOperation(&Matrix::SIMDMul, &Matrix::RemainderMul, -1);
}
Matrix Matrix::Abs() const {
	return SingleFloatOperation(&Matrix::SIMDAbs, &Matrix::RemainderAbs, 0);
}

Matrix Matrix::Add(float scalar) const {
	return SingleFloatOperation(&Matrix::SIMDAdd, &Matrix::RemainderAdd, scalar);
}
Matrix Matrix::Add(const std::vector<float>& scalar) const {
	return VectorFloatOperation(&Matrix::SIMDAdd, &Matrix::RemainderAdd, scalar);
}
Matrix Matrix::Add(const Matrix& element) const {
	return MatrixFloatOperation(&Matrix::SIMDAdd, &Matrix::RemainderAdd, element);
}

Matrix Matrix::Subtract(float scalar) const {
	return SingleFloatOperation(&Matrix::SIMDSub, &Matrix::RemainderSub, scalar);
}
Matrix Matrix::Subtract(const std::vector<float>& scalar) const {
	return VectorFloatOperation(&Matrix::SIMDSub, &Matrix::RemainderSub, scalar);
}
Matrix Matrix::Subtract(const Matrix& element) const {
	return MatrixFloatOperation(&Matrix::SIMDSub, &Matrix::RemainderSub, element);
}

Matrix Matrix::Multiply(float scalar) const {
	return SingleFloatOperation(&Matrix::SIMDMul, &Matrix::RemainderMul, scalar);
}
Matrix Matrix::Multiply(const std::vector<float>& scalar) const {
	return VectorFloatOperation(&Matrix::SIMDMul, &Matrix::RemainderMul, scalar);
}
Matrix Matrix::Multiply(const Matrix& element) const {
	return MatrixFloatOperation(&Matrix::SIMDMul, &Matrix::RemainderMul, element);
}

Matrix Matrix::Divide(float scalar) const {
	return SingleFloatOperation(&Matrix::SIMDDiv, &Matrix::RemainderDiv, scalar);
}
Matrix Matrix::Divide(const std::vector<float>& scalar) const {
	return VectorFloatOperation(&Matrix::SIMDDiv, &Matrix::RemainderDiv, scalar);
}
Matrix Matrix::Divide(const Matrix& element) const {
	return MatrixFloatOperation(&Matrix::SIMDDiv, &Matrix::RemainderDiv, element);
}

Matrix Matrix::Pow(float scalar) const {
	return SingleFloatOperation(&Matrix::SIMDPow, &Matrix::RemainderPow, scalar);
}
Matrix Matrix::Pow(const std::vector<float>& scalar) const {
	return VectorFloatOperation(&Matrix::SIMDPow, &Matrix::RemainderPow, scalar);
}
Matrix Matrix::Pow(const Matrix& element) const {
	return MatrixFloatOperation(&Matrix::SIMDPow, &Matrix::RemainderPow, element);
}

Matrix Matrix::Exp(float base) const {
	return SingleFloatOperation(&Matrix::SIMDExp, &Matrix::RemainderExp, base);
}
Matrix Matrix::Exp(const std::vector<float>& base) const {
	return VectorFloatOperation(&Matrix::SIMDExp, &Matrix::RemainderExp, base);
}
Matrix Matrix::Exp(const Matrix& base) const {
	return MatrixFloatOperation(&Matrix::SIMDExp, &Matrix::RemainderExp, base);
}

Matrix Matrix::Log() const {
	return SingleFloatOperation(&Matrix::SIMDLog, &Matrix::RemainderLog, 0);
}

// Trig
Matrix Matrix::Cos() const {
	return this->SingleFloatOperation(&Matrix::SIMDCos, &Matrix::RemainderCos, 0);
}
Matrix Matrix::Sin() const {
	return this->SingleFloatOperation(&Matrix::SIMDSin, &Matrix::RemainderSin, 0);
}
Matrix Matrix::Acos() const {
	return this->SingleFloatOperation(&Matrix::SIMDAcos, &Matrix::RemainderAcos, 0);
}
Matrix Matrix::Asin() const {
	return this->SingleFloatOperation(&Matrix::SIMDAsin, &Matrix::RemainderAsin, 0);
}

// Activation Functions
Matrix Matrix::Sigmoid() const {
	Matrix one = Matrix(RowCount, ColumnCount, 1);
	return one / (this->Negative().Exp() + 1);
}
Matrix Matrix::ReLU() const {
	return SingleFloatOperation(&Matrix::SIMDMax, &Matrix::RemainderMax, 0);
}
Matrix Matrix::LeakyReLU(float alpha) const {
	return MatrixFloatOperation(&Matrix::SIMDMax, &Matrix::RemainderMax, this->Multiply(alpha));
}
Matrix Matrix::_LeakyReLU() const {
	return LeakyReLU();
}
Matrix Matrix::ELU(float alpha) const {
	Matrix a = *this;
	for (int r = 0; r < RowCount; r++) {
		for (int c = 0; c < ColumnCount; c++) {
			a(r, c) = matrix[r * ColumnCount + c] < 0.0f ? alpha * (std::exp(matrix[r * ColumnCount + c]) - 1) : matrix[r * ColumnCount + c];
		}
	}
	return a;
}
Matrix Matrix::_ELU() const {
	return ELU();
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
Matrix Matrix::LeakyReLUDerivative(float alpha) const {
	Matrix deriv = *this;
	for (int c = 0; c < ColumnCount; c++) {
		for (int r = 0; r < RowCount; r++) {
			deriv(r, c) = deriv(r, c) > 0.0f ? 1.0f : alpha;
		}
	}
	return deriv;
}
Matrix Matrix::_LeakyReLUDerivative() const {
	return LeakyReLUDerivative();
}
Matrix Matrix::ELUDerivative(float alpha) const {
	Matrix a = *this;

	for (int r = 0; r < this->RowCount; r++) {
		for (int c = 0; c < this->ColumnCount; c++) {
			a(r, c) = matrix[r * ColumnCount + c] > 0.0f ? 1.0f : alpha * std::exp(matrix[r * ColumnCount + c]);
		}
	}
	return a;
}
Matrix Matrix::_ELUDerivative() const {
	return ELUDerivative();
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
Matrix Matrix::SingleFloatOperation(__m256 (Matrix::* operation)(__m256 opOne, __m256 opTwo) const noexcept,
	float (Matrix::* remainderOperation)(float a, float b) const noexcept, float scalar) const {
	Matrix mat = *this;

	__m256 _scalar = _mm256_set1_ps(scalar);

	int i = 0;
	for (; i + 8 <= (mat.RowCount * mat.ColumnCount); i += 8) {
		__m256 loaded_a = _mm256_load_ps(&mat.matrix[i]);
		loaded_a = (this->*operation)(loaded_a, _scalar);
		_mm256_store_ps(&mat.matrix[i], loaded_a);
	}

	for (; i < (mat.RowCount * mat.ColumnCount); i++) {
		mat.matrix[i] = (this->*remainderOperation)(mat.matrix[i], scalar);
	}

	return mat;
}

Matrix Matrix::VectorFloatOperation(__m256 (Matrix::* operation)(__m256 opOne, __m256 opTwo) const noexcept,
	float (Matrix::* remainderOperation)(float a, float b) const noexcept, const std::vector<float>& scalar) const {

	Matrix mat = *this;
	const int alignedN = mat.ColumnCount - (mat.ColumnCount % 8);

	if (scalar.size() == ColumnCount) {
		for (int r = 0; r < mat.RowCount; r++) {

			for (int i = 0; i < alignedN; i += 8) {
				__m256 loaded_a = _mm256_load_ps(&mat(r, i));
				__m256 loaded_b = _mm256_load_ps(&scalar[i]);

				_mm256_store_ps(&mat(r, i), (this->*operation)(loaded_a, loaded_b));
			}

			for (int i = alignedN; i < mat.ColumnCount; i++) {
				mat(r, i) = (this->*remainderOperation)(mat(r, i), scalar[i]);
			}
		}
	}
	else if (scalar.size() == RowCount) {
		for (int r = 0; r < mat.RowCount; r++) {

			__m256 loaded_b = _mm256_set1_ps(scalar[r]);

			for (int i = 0; i < alignedN; i += 8) {
				__m256 loaded_a = _mm256_load_ps(&mat(r, i));

				_mm256_store_ps(&mat(r, i), (this->*operation)(loaded_a, loaded_b));
			}

			for (int i = alignedN; i < mat.ColumnCount; i++) {
				mat(r, i) = (this->*remainderOperation)(mat(r, i), scalar[r]);
			}
		}
	}

	return mat;
}

Matrix Matrix::MatrixFloatOperation(__m256 (Matrix::* operation)(__m256 opOne, __m256 opTwo) const noexcept,
	float (Matrix::* remainderOperation)(float a, float b) const noexcept, const Matrix& element) const {
	Matrix mat = element;

	int i = 0;
	for (; i + 8 <= (mat.RowCount * mat.ColumnCount); i += 8) {
		__m256 loaded_a = _mm256_load_ps(&matrix[i]);
		__m256 loaded_b = _mm256_load_ps(&mat.matrix[i]);

		loaded_a = (this->*operation)(loaded_a, loaded_b);
		_mm256_store_ps(&mat.matrix[i], loaded_a);
	}

	for (; i < (mat.RowCount * mat.ColumnCount); i++) {
		mat.matrix[i] = (this->*remainderOperation)(matrix[i], mat.matrix[i]);
	}

	return mat;
}


// SIMD Implementation in place
Matrix Matrix::single_float_operation_in_place(__m256 (Matrix::* operation)(__m256 opOne, __m256 opTwo) const,
	float (Matrix::* remainderOperation)(float a, float b) const, float scalar) {

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
__m256 Matrix::SIMDSin(__m256 opOne, __m256 opTwo) const noexcept {
	return _mm256_sin_ps(opOne);
}
__m256 Matrix::SIMDCos(__m256 opOne, __m256 opTwo) const noexcept {
	return _mm256_cos_ps(opOne);
}
__m256 Matrix::SIMDSec(__m256 opOne, __m256 opTwo) const noexcept {
	return _mm256_div_ps(_mm256_set1_ps(1.0f), _mm256_cos_ps(opOne));
}
__m256 Matrix::SIMDCsc(__m256 opOne, __m256 opTwo) const noexcept {
	return _mm256_div_ps(_mm256_set1_ps(1.0f), _mm256_sin_ps(opOne));
}
__m256 Matrix::SIMDAcos(__m256 opOne, __m256 opTwo) const noexcept {
	return _mm256_acos_ps(opOne);
}
__m256 Matrix::SIMDAsin(__m256 opOne, __m256 opTwo) const noexcept {
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
float Matrix::RemainderSin(float a, float b) const noexcept {
	return std::sin(a);
}
float Matrix::RemainderCos(float a, float b) const noexcept {
	return std::cos(a);
}
float Matrix::RemainderSec(float a, float b) const noexcept {
	return 1.0f / std::cos(a);
}
float Matrix::RemainderCsc(float a, float b) const noexcept {
	return 1.0f / std::sin(a);
}
float Matrix::RemainderAcos(float a, float b) const noexcept {
	return std::acos(a);
}
float Matrix::RemainderAsin(float a, float b) const noexcept {
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
	float* a = (float*)malloc((RowCount + element.RowCount) * ColumnCount * sizeof(float));

	std::memcpy(a, matrix, RowCount * ColumnCount * sizeof(float));
	std::memcpy(a + (RowCount * ColumnCount), element.matrix, element.RowCount * element.ColumnCount * sizeof(float));

	Matrix total = Matrix(a, RowCount + element.RowCount, ColumnCount);
	free(a);

	return total;
}

Matrix Matrix::Join(Matrix element) {
	float* a = (float*)malloc(RowCount * (ColumnCount + element.ColumnCount) * sizeof(float));

	for (int r = 0; r < RowCount; r++) {
		for (int c = 0; c < ColumnCount + element.ColumnCount; c++) {
			a[r * ColumnCount + c] = (c < ColumnCount ? (matrix[r * ColumnCount + c]) : (element(r, c - ColumnCount)));
		}
	}

	Matrix total = Matrix(a, RowCount, ColumnCount + element.ColumnCount);
	free(a);

	return total;
}

Matrix Matrix::Transpose() const {
	float* a = (float*)malloc(RowCount * ColumnCount * sizeof(float));

	for (int r = 0; r < RowCount; r++) {
		for (int c = 0; c < ColumnCount; c++) {
			a[c * RowCount + r] = matrix[r * ColumnCount + c];
		}
	}

	Matrix transpose(a, ColumnCount, RowCount);
	free(a);

	return transpose;
}

bool Matrix::contains_nan() const {
	return std::any_of(matrix, (matrix + (RowCount * ColumnCount)), [](float value) {
		return std::isnan(value);
		});
}