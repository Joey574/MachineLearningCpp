#pragma once

#define _USE_MATH_DEFINES
#include <vector>
#include <algorithm>
#include <execution>
#include <cmath>
#include <random>
#include <immintrin.h> 
#include <string>
#include <utility>
#include <iostream>

class Matrix
{
public:

	static enum init
	{
		Random, Normalize, Xavier, He
	};

	Matrix() : RowCount(0), ColumnCount(0), matrix(nullptr) {}
	Matrix(size_t, size_t);
	Matrix(size_t, size_t, init);
	Matrix(size_t, size_t, float);
	Matrix(const std::vector<std::vector<float>>&);
	Matrix(const float*, size_t, size_t);
	Matrix(const Matrix&);

	std::vector<float> Column(int) const;
	std::vector<float> Row(int) const;

	void SetColumn(int, const std::vector<float>&);
	void SetColumn(int, const std::vector<int>&);
	void SetRow(int, const std::vector<float>&);
	void SetRow(int, const std::vector<int>&);

	void add_row(float* row);
	void add_row(const std::vector<float>&);
	void add_row(const std::vector<int>&);

	Matrix SegmentR(int, int) const;
	Matrix SegmentR(int) const;

	Matrix SegmentC(int, int) const;
	Matrix SegmentC(int) const;

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
	Matrix extract_features(int, int, int, int, int, float, float) const;

	/// <summary>
	/// Normalizes values by the global min and max
	/// </summary>
	/// <param name="lowerRange"></param>
	/// <param name="upperRange"></param>
	/// <returns></returns>
	Matrix normalized(float, float) const noexcept;

	/// <summary>
	/// Computes sin(nx) and cos(nx)
	/// </summary>
	/// <param name="n"></param>
	/// <returns></returns>
	Matrix fourier_series(int) const noexcept;
	/// <summary>
	/// Computes x^n
	/// </summary>
	/// <param name="n"></param>
	/// <returns></returns>
	Matrix taylor_series(int) const noexcept;
	/// <summary>
	/// Computes cos(n*acos(x))
	/// </summary>
	/// <param name="n"></param>
	/// <returns></returns>
	Matrix chebyshev_series(int) const noexcept;
	/// <summary>
	/// Computes (x^2 - 1)^n
	/// </summary>
	/// <param name="n"></param>
	/// <returns></returns>
	Matrix legendre_series(int) const noexcept;
	/// <summary>
	/// Computes x^n * e^-x
	/// </summary>
	/// <param name="n"></param>
	/// <returns></returns>
	Matrix laguerre_series(int) const noexcept;

	/// <summary>
	/// Computes the dot product between two matrices, columncount of host must match rowcount of "element", returned matrix is host rowcount by element columncount
	/// </summary>
	/// <param name="element"></param>
	/// <returns></returns>
	Matrix dot_product(const Matrix&) const;

	/// <summary>
	/// Log sum exp trick on each column, 
	/// </summary>
	/// <returns></returns>
	std::vector<float> log_sum_exp() const noexcept;

	// Basic Math
	Matrix Negative() const;
	Matrix Abs() const;

	Matrix Add(float) const;
	Matrix Add(const std::vector<float>&) const;
	Matrix Add(const Matrix&) const;

	void Add(float, Matrix&) const;
	void Add(const std::vector<float>&, Matrix&) const;
	void Add(const Matrix&, Matrix&) const;

	Matrix Subtract(float) const;
	Matrix Subtract(const std::vector<float>&) const;
	Matrix Subtract(const Matrix&) const;

	void Subtract(float, Matrix&) const;
	void Subtract(const std::vector<float>&, Matrix&) const;
	void Subtract(const Matrix&, Matrix&) const;

	Matrix Multiply(float) const;
	Matrix Multiply(const std::vector<float>&) const;
	Matrix Multiply(const Matrix&) const;

	void Multiply(float, Matrix&) const;
	void Multiply(const std::vector<float>&, Matrix&) const;
	void Multiply(const Matrix&, Matrix&) const;

	Matrix Divide(float) const;
	Matrix Divide(const std::vector<float>&) const;
	Matrix Divide(const Matrix&) const;

	void Divide(float, Matrix&) const;
	void Divide(const std::vector<float>&, Matrix&) const;
	void Divide(const Matrix&, Matrix&) const;

	Matrix Pow(float) const;
	Matrix Pow(const std::vector<float>&) const;
	Matrix Pow(const Matrix&) const;

	Matrix Exp(float = std::exp(1.0)) const;
	Matrix Exp(const std::vector<float>&) const;
	Matrix Exp(const Matrix&) const;

	Matrix Log() const;

	// Trig
	Matrix Cos() const;
	Matrix Sin() const;
	Matrix Acos() const;
	Matrix Asin() const;

	// Activation Functions
	Matrix Sigmoid() const;
	Matrix ReLU() const;
	Matrix LeakyReLU() const;
	Matrix ELU() const;
	Matrix Tanh() const;
	Matrix Softplus() const;
	Matrix SiLU() const;

	Matrix SoftMax() const;

	// Activation Derivatives
	Matrix SigmoidDerivative() const;
	Matrix ReLUDerivative() const;
	Matrix LeakyReLUDerivative() const;
	Matrix ELUDerivative() const;
	Matrix TanhDerivative() const;
	Matrix SoftplusDerivative() const;
	Matrix SiLUDerivative() const;

	Matrix Transpose() const;

	Matrix Combine(Matrix);
	Matrix Join(Matrix);

	void Insert(int, Matrix);

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
		return single_float_operation(&Matrix::SIMDAdd, &Matrix::RemainderAdd, scalar);
	}
	inline Matrix operator + (const std::vector<float>& scalar) const noexcept {
		return vector_float_operation(&Matrix::SIMDAdd, &Matrix::RemainderAdd, scalar);
	}
	inline Matrix operator + (const Matrix& element) const noexcept {
		return matrix_float_operation(&Matrix::SIMDAdd, &Matrix::RemainderAdd, element);
	}

	inline Matrix operator - (float scalar) const noexcept {
		return single_float_operation(&Matrix::SIMDSub, &Matrix::RemainderSub, scalar);
	}
	inline Matrix operator - (const std::vector<float>& scalar) const noexcept {
		return vector_float_operation(&Matrix::SIMDSub, &Matrix::RemainderSub, scalar);
	}
	inline Matrix operator - (const Matrix& element) const noexcept {
		return matrix_float_operation(&Matrix::SIMDSub, &Matrix::RemainderSub, element);
	}

	inline Matrix operator * (float scalar) const noexcept {
		return single_float_operation(&Matrix::SIMDMul, &Matrix::RemainderMul, scalar);
	}
	inline Matrix operator * (const std::vector<float>& scalar) const noexcept {
		return vector_float_operation(&Matrix::SIMDMul, &Matrix::RemainderMul, scalar);
	}
	inline Matrix operator * (const Matrix& element) const noexcept {
		return matrix_float_operation(&Matrix::SIMDMul, &Matrix::RemainderMul, element);
	}

	inline Matrix operator / (float scalar) const noexcept {
		return single_float_operation(&Matrix::SIMDDiv, &Matrix::RemainderDiv, scalar);
	}
	inline Matrix operator / (const std::vector<float>& scalar) const noexcept {
		return vector_float_operation(&Matrix::SIMDDiv, &Matrix::RemainderDiv, scalar);
	}
	inline Matrix operator / (const Matrix& element) const noexcept {
		return matrix_float_operation(&Matrix::SIMDDiv, &Matrix::RemainderDiv, element);
	}


	inline void operator += (float scalar) noexcept {
		single_float_operation_in_place(&Matrix::SIMDAdd, &Matrix::RemainderAdd, scalar);
	}
	inline void operator += (const std::vector<float>& scalar) noexcept {
		vector_float_operation_in_place(&Matrix::SIMDAdd, &Matrix::RemainderAdd, scalar);
	}
	inline void operator += (const Matrix& element) noexcept {
		matrix_float_operation_in_place(&Matrix::SIMDAdd, &Matrix::RemainderAdd, element);
	}

	inline void operator -= (float scalar) noexcept {
		single_float_operation_in_place(&Matrix::SIMDSub, &Matrix::RemainderSub, scalar);
	}
	inline void operator -= (const std::vector<float>& scalar) noexcept {
		vector_float_operation_in_place(&Matrix::SIMDSub, &Matrix::RemainderSub, scalar);
	}
	inline void operator -= (const Matrix& element) noexcept {
		matrix_float_operation_in_place(&Matrix::SIMDSub, &Matrix::RemainderSub, element);
	}

	inline void operator *= (float scalar) noexcept {
		single_float_operation_in_place(&Matrix::SIMDMul, &Matrix::RemainderMul, scalar);
	}
	inline void operator *= (const std::vector<float>& scalar) noexcept {
		vector_float_operation_in_place(&Matrix::SIMDMul, &Matrix::RemainderMul, scalar);
	}
	inline void operator *= (const Matrix& element) noexcept {
		matrix_float_operation_in_place(&Matrix::SIMDMul, &Matrix::RemainderMul, element);
	}

	inline void operator /= (float scalar) noexcept {
		single_float_operation_in_place(&Matrix::SIMDDiv, &Matrix::RemainderDiv, scalar);
	}
	inline void operator /= (const std::vector<float>& scalar) noexcept {
		vector_float_operation_in_place(&Matrix::SIMDDiv, &Matrix::RemainderDiv, scalar);
	}
	inline void operator /= (const Matrix& element) noexcept {
		matrix_float_operation_in_place(&Matrix::SIMDDiv, &Matrix::RemainderDiv, element);
	}

	inline Matrix& operator = (const Matrix& other) {

		RowCount = other.RowCount;
		ColumnCount = other.ColumnCount;

		free(matrix);
		matrix = m_init();

		std::memcpy(matrix, other.matrix, RowCount * ColumnCount * sizeof(float));
		return *this;
	}

	inline bool operator == (const Matrix& other) {
		if (RowCount != other.RowCount || ColumnCount != other.ColumnCount) {
			return false;
		}

		for (int i = 0; i < RowCount * ColumnCount; i++) {
			if (matrix[i] != other.matrix[i]) {
				return false;
			}
		}

		return true;
	}

	float* matrix;

	~Matrix() {
		if (matrix) { free(matrix); }
	}


private:

	Matrix single_float_operation(__m256 (Matrix::* operation)(__m256 opOne, __m256 opTwo) const noexcept,
		float (Matrix::* remainderOperation)(float a, float b) const noexcept, float scalar) const;
	Matrix vector_float_operation(__m256 (Matrix::* operation)(__m256 opOne, __m256 opTwo) const noexcept,
		float (Matrix::* remainderOperation)(float a, float b) const noexcept, const std::vector<float>& scalar) const;
	Matrix matrix_float_operation(__m256 (Matrix::* operation) (__m256 opOne, __m256 opTwo) const noexcept,
		float (Matrix::* remainderOperation)(float a, float b) const noexcept, const Matrix& element) const;

	void single_float_operation_in_place(__m256 (Matrix::* operation)(__m256 opOne, __m256 opTwo) const,
		float (Matrix::* remainderOperation)(float a, float b) const, float scalar);
	void vector_float_operation_in_place(__m256 (Matrix::* operation)(__m256 opOne, __m256 opTwo) const,
		float (Matrix::* remainderOperation)(float a, float b) const, const std::vector<float>& scalar);
	void matrix_float_operation_in_place(__m256 (Matrix::* operation)(__m256 opOne, __m256 opTwo) const,
		float (Matrix::* remainderOperation)(float a, float b) const, const Matrix& element);

	void single_float_operation_store(__m256(Matrix::* operation)(__m256, __m256) const noexcept,
		float(Matrix::* remainderOperation)(float a, float b) const noexcept, float scalar, Matrix& store) const;
	void vector_float_operation_store(__m256(Matrix::* operation)(__m256, __m256) const noexcept,
		float(Matrix::* remainderOperation)(float a, float b) const noexcept, const std::vector<float>& scalar, Matrix& store) const;
	void matrix_float_operation_store(__m256(Matrix::* operation)(__m256, __m256) const noexcept,
		float(Matrix::* remainderOperation)(float a, float b) const noexcept, const Matrix& element, Matrix& store) const;

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
	inline __m256 SIMDSin(__m256 opOne, __m256 opTwo) const noexcept;
	inline __m256 SIMDCos(__m256 opOne, __m256 opTwo) const noexcept;
	inline __m256 SIMDSec(__m256 opOne, __m256 opTwo) const noexcept;
	inline __m256 SIMDCsc(__m256 opOne, __m256 opTwo) const noexcept;
	inline __m256 SIMDAcos(__m256 opOne, __m256 opTwo) const noexcept;
	inline __m256 SIMDAsin(__m256 opOne, __m256 opTwo) const noexcept;

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
	inline float RemainderSin(float a, float b) const noexcept;
	inline float RemainderCos(float a, float b) const noexcept;
	inline float RemainderSec(float a, float b) const noexcept;
	inline float RemainderCsc(float a, float b) const noexcept;
	inline float RemainderAcos(float a, float b) const noexcept;
	inline float RemainderAsin(float a, float b) const noexcept;

	inline float* m_init() {
		int pad = RowCount * ColumnCount % 8 == 0 ? 0 : 8 - (RowCount * ColumnCount % 8);
		return (float*)malloc((RowCount * ColumnCount + pad) * sizeof(float));
	}

	inline float* c_init() {
		int pad = RowCount * ColumnCount % 8 == 0 ? 0 : 8 - (RowCount * ColumnCount % 8);
		return (float*)calloc((RowCount * ColumnCount + pad), sizeof(float));
	}
};