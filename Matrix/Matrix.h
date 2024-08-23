#pragma once

#define _USE_MATH_DEFINES
#include <vector>
#include <algorithm>
#include <execution>
#include <cmath>
#include <random>
#include <immintrin.h> 
#include <string>
#include <cmath>
#include <utility>

class Matrix
{
public:

	static enum init
	{
		Random, Normalize, Xavier, He
	};

	Matrix() : RowCount(0), ColumnCount(0), matrix(nullptr) {}
	Matrix(size_t rows, size_t columns);
	Matrix(size_t rows, size_t columns, init initType);
	Matrix(size_t rows, size_t columns, float value);
	Matrix(const std::vector<std::vector<float>>& matrix);
	Matrix(const float* matrix, size_t rows, size_t columns);
	Matrix(const Matrix& other);

	std::vector<float> Column(int index) const;
	std::vector<float> Row(int index) const;

	void SetColumn(int index, const std::vector<float>& column);
	void SetColumn(int index, const std::vector<int>& column);
	void SetRow(int index, const std::vector<float>& row);
	void SetRow(int index, const std::vector<int>& row);

	Matrix SegmentR(int startRow, int endRow) const;
	Matrix SegmentR(int startRow) const;

	Matrix SegmentC(int startColumn, int endColumn) const;
	Matrix SegmentC(int startColumn) const;

	std::vector<float> ColumnSums() const;
	std::vector<float> RowSums() const;

	// "Advanced" Math

	/// <summary>
	/// Takes input and generates various series and combines them with the original matrix, such that # of columns remains the same,
	/// all values are normalized to the supplied ranges at the end
	/// </summary>
	/// <param name="fourier"></param>
	/// <param name="taylor"></param>
	/// <param name="chebyshev"></param>
	/// <param name="legendre"></param>
	/// <param name="laguerre"></param>
	/// <param name="lowerNormal"></param>
	/// <param name="upperNormal"></param>
	/// <returns></returns>
	Matrix extract_features(int fourier, int taylor, int chebyshev, int legendre, int laguerre, float lowerNormal, float upperNormal) const;

	/// <summary>
	/// Normalizes values by the global min and max
	/// </summary>
	/// <param name="lowerRange"></param>
	/// <param name="upperRange"></param>
	/// <returns></returns>
	Matrix normalized(float lowerRange, float upperRange) const noexcept;

	/// <summary>
	/// Computes sin(nx) and cos(nx)
	/// </summary>
	/// <param name="n"></param>
	/// <returns></returns>
	Matrix fourier_series(int n) const noexcept;
	/// <summary>
	/// Computes x^n
	/// </summary>
	/// <param name="n"></param>
	/// <returns></returns>
	Matrix taylor_series(int n) const noexcept;
	/// <summary>
	/// Computes cos(n*acos(x))
	/// </summary>
	/// <param name="n"></param>
	/// <returns></returns>
	Matrix chebyshev_series(int n) const noexcept;
	/// <summary>
	/// Computes (x^2 - 1)^n
	/// </summary>
	/// <param name="n"></param>
	/// <returns></returns>
	Matrix legendre_series(int n) const noexcept;
	/// <summary>
	/// Computes x^n * e^-x
	/// </summary>
	/// <param name="n"></param>
	/// <returns></returns>
	Matrix laguerre_series(int n) const noexcept;

	/// <summary>
	/// Computes the dot product between two matrices, columncount of host must match rowcount of "element", returned matrix is host rowcount by element columncount
	/// </summary>
	/// <param name="element"></param>
	/// <returns></returns>
	Matrix dot_product(const Matrix& element) const;

	/// <summary>
	/// Log sum exp trick on each column, 
	/// </summary>
	/// <returns></returns>
	std::vector<float> log_sum_exp() const noexcept;

	// Basic Math
	Matrix Negative() const;
	Matrix Abs() const;

	Matrix Add(float scalar) const;
	Matrix Add(const std::vector<float>& scalar) const;
	Matrix Add(const Matrix& element) const;

	Matrix Subtract(float scalar) const;
	Matrix Subtract(const std::vector<float>& scalar) const;
	Matrix Subtract(const Matrix& element) const;

	Matrix Multiply(float scalar) const;
	Matrix Multiply(const std::vector<float>& scalar) const;
	Matrix Multiply(const Matrix& element) const;

	Matrix Divide(float scalar) const;
	Matrix Divide(const std::vector<float>& scalar) const;
	Matrix Divide(const Matrix& element) const;

	Matrix Pow(float scalar) const;
	Matrix Pow(const std::vector<float>& scalar) const;
	Matrix Pow(const Matrix& element) const;

	Matrix Exp(float base = std::exp(1.0)) const;
	Matrix Exp(const std::vector<float>& base) const;
	Matrix Exp(const Matrix& base) const;

	Matrix Log() const;

	// Trig
	Matrix Cos() const;
	Matrix Sin() const;
	Matrix Acos() const;
	Matrix Asin() const;

	// Activation Functions
	Matrix Sigmoid() const;
	Matrix ReLU() const;
	Matrix LeakyReLU(float alpha = 0.1f) const;
	Matrix _LeakyReLU() const;
	Matrix ELU(float alpha = 1.0f) const;
	Matrix _ELU() const;
	Matrix Tanh() const;
	Matrix Softplus() const;
	Matrix SiLU() const;

	Matrix SoftMax() const;

	// Activation Derivatives
	Matrix SigmoidDerivative() const;
	Matrix ReLUDerivative() const;
	Matrix LeakyReLUDerivative(float alpha = 0.1f) const;
	Matrix _LeakyReLUDerivative() const;
	Matrix ELUDerivative(float alpha = 1.0f) const;
	Matrix _ELUDerivative() const;
	Matrix TanhDerivative() const;
	Matrix SoftplusDerivative() const;
	Matrix SiLUDerivative() const;

	Matrix Transpose() const;

	Matrix Combine(Matrix element);
	Matrix Join(Matrix element);

	void Insert(int startRow, Matrix element);

	std::string ToString() const;
	std::string Size() const;

	bool contains_nan() const;

	size_t ColumnCount;
	size_t RowCount;

	inline const float& operator() (int row, int column) const {
		return matrix[row * ColumnCount + column];
	}
	inline float& operator() (int row, int column) {
		return matrix[row * ColumnCount + column];
	}

	inline Matrix operator + (float scalar) const noexcept {
		return this->Add(scalar);
	}
	inline Matrix operator + (const std::vector<float>& scalar) const noexcept {
		return this->Add(scalar);
	}
	inline Matrix operator + (const Matrix& element) const noexcept {
		return this->Add(element);
	}

	inline Matrix operator - (float scalar) const noexcept {
		return this->Subtract(scalar);
	}
	inline Matrix operator - (const std::vector<float>& scalar) const noexcept {
		return this->Subtract(scalar);
	}
	inline Matrix operator - (const Matrix& element) const noexcept {
		return this->Subtract(element);
	}

	inline Matrix operator * (float scalar) const noexcept {
		return this->Multiply(scalar);
	}
	inline Matrix operator * (const std::vector<float>& scalar) const noexcept {
		return this->Multiply(scalar);
	}
	inline Matrix operator * (const Matrix& element) const noexcept {
		return this->Multiply(element);
	}

	inline Matrix operator / (float scalar) const noexcept {
		return this->Divide(scalar);
	}
	inline Matrix operator / (const std::vector<float>& scalar) const noexcept {
		return this->Divide(scalar);
	}
	inline Matrix operator / (const Matrix& element) const noexcept {
		return this->Divide(element);
	}


	inline Matrix& operator += (float scalar) noexcept {
		*this = this->Add(scalar);
		return *this;
	}
	inline Matrix& operator += (const std::vector<float>& scalar) noexcept {
		*this = this->Add(scalar);
		return *this;
	}
	inline Matrix& operator += (const Matrix& element) noexcept {
		*this = this->Add(element);
		return *this;
	}

	inline Matrix& operator -= (float scalar) noexcept {
		*this = this->Subtract(scalar);
		return *this;
	}
	inline Matrix& operator -= (const std::vector<float>& scalar) noexcept {
		*this = this->Subtract(scalar);
		return *this;
	}
	inline Matrix& operator -= (const Matrix& element) noexcept {
		*this = this->Subtract(element);
		return *this;
	}

	inline Matrix& operator *= (float scalar) noexcept {
		*this = this->Multiply(scalar);
		return *this;
	}
	inline Matrix& operator *= (const std::vector<float>& scalar) noexcept {
		*this = this->Multiply(scalar);
		return *this;
	}
	inline Matrix& operator *= (const Matrix& element) noexcept {
		*this = this->Multiply(element);
		return *this;
	}

	inline Matrix& operator /= (float scalar) noexcept {
		*this = this->Divide(scalar);
		return *this;
	}
	inline Matrix& operator /= (const std::vector<float>& scalar) noexcept {
		*this = this->Divide(scalar);
		return *this;
	}
	inline Matrix& operator /= (const Matrix& element) noexcept {
		*this = this->Divide(element);
		return *this;
	}

	inline Matrix& operator = (const Matrix& other) noexcept {
		RowCount = other.RowCount;
		ColumnCount = other.ColumnCount;

		if (matrix) {
			free(matrix);
		}

		matrix = (float*)malloc(RowCount * ColumnCount * sizeof(float));

		std::memcpy(matrix, other.matrix, RowCount * ColumnCount * sizeof(float));
		return *this;
	}

	float* matrix;

	~Matrix() {
		if (matrix != nullptr) {
			free(matrix);
		}
	}


private:

	Matrix SingleFloatOperation(__m256 (Matrix::* operation)(__m256 opOne, __m256 opTwo) const noexcept,
		float (Matrix::* remainderOperation)(float a, float b) const noexcept, float scalar) const;
	Matrix VectorFloatOperation(__m256 (Matrix::* operation)(__m256 opOne, __m256 opTwo) const noexcept,
		float (Matrix::* remainderOperation)(float a, float b) const noexcept, const std::vector<float>& scalar) const;
	Matrix MatrixFloatOperation(__m256 (Matrix::* operation) (__m256 opOne, __m256 opTwo) const noexcept,
		float (Matrix::* remainderOperation)(float a, float b) const noexcept, const Matrix& element) const;

	Matrix single_float_operation_in_place(__m256 (Matrix::* operation)(__m256 opOne, __m256 opTwo) const,
		float (Matrix::* remainderOperation)(float a, float b) const, float scalar);
	Matrix vector_float_operation_in_place(__m256 (Matrix::* operation)(__m256 opOne, __m256 opTwo) const,
		float (Matrix::* remainderOperation)(float a, float b) const, const std::vector<float>& scalar);
	Matrix matrix_float_operation_in_place(__m256 (Matrix::* operation)(__m256 opOne, __m256 opTwo) const,
		float (Matrix::* remainderOperation)(float a, float b) const, const Matrix& element);

	inline __m256 SIMDAdd(__m256 opOne, __m256 opTwo) const noexcept;
	inline __m256 SIMDSub(__m256 opOne, __m256 opTwo) const noexcept;
	inline __m256 SIMDMul(__m256 opOne, __m256 opTwo) const noexcept;
	inline __m256 SIMDDiv(__m256 opOne, __m256 opTwo) const noexcept;
	inline __m256 SIMDPow(__m256 opOne, __m256 opTwo) const noexcept;
	inline __m256 SIMDExp(__m256 opOne, __m256 opTwo) const noexcept;
	inline __m256 SIMDLog(__m256 opOne, __m256 opTwo) const noexcept;
	inline __m256 SIMDMax(__m256 opOne, __m256 opTwo) const noexcept;
	inline __m256 SIMDAbs(__m256 opOne, __m256 opTwo) const noexcept;

	// SIMD Trig
	__m256 SIMDSin(__m256 opOne, __m256 opTwo) const noexcept;
	__m256 SIMDCos(__m256 opOne, __m256 opTwo) const noexcept;
	__m256 SIMDSec(__m256 opOne, __m256 opTwo) const noexcept;
	__m256 SIMDCsc(__m256 opOne, __m256 opTwo) const noexcept;
	__m256 SIMDAcos(__m256 opOne, __m256 opTwo) const noexcept;
	__m256 SIMDAsin(__m256 opOne, __m256 opTwo) const noexcept;

	inline float RemainderAdd(float a, float b) const noexcept;
	inline float RemainderSub(float a, float b) const noexcept;
	inline float RemainderMul(float a, float b) const noexcept;
	inline float RemainderDiv(float a, float b) const noexcept;
	inline float RemainderPow(float a, float b) const noexcept;
	inline float RemainderExp(float a, float b) const noexcept;
	inline float RemainderLog(float a, float b) const noexcept;
	inline float RemainderMax(float a, float b) const noexcept;
	inline float RemainderAbs(float a, float b) const noexcept;

	// SIMD Trig
	float RemainderSin(float a, float b) const noexcept;
	float RemainderCos(float a, float b) const noexcept;
	float RemainderSec(float a, float b) const noexcept;
	float RemainderCsc(float a, float b) const noexcept;
	float RemainderAcos(float a, float b) const noexcept;
	float RemainderAsin(float a, float b) const noexcept;
};