#pragma once
#include <vector>
#include <algorithm>
#include <execution>
#include <cmath>
#include <random>
#include <immintrin.h> 
#include <string>
#include <utility>

class Biases
{
public:

	Biases() : dimensions({0}), bias(nullptr) {}
	Biases(const Biases& other) : dimensions(other.dimensions), bias((float*)std::malloc(other.size() * sizeof(float))) {
		std::memcpy(bias, other.bias, other.size() * sizeof(float));
	}

	Biases(std::vector<int> dimensions) : dimensions(dimensions) {

		// offset for nn, input doesn't have any biases
		this->dimensions[0] = 0;
		for (int i = 1; i < dimensions.size(); i++) {
			this->dimensions[i] = this->dimensions[i] + this->dimensions[i - 1];
		}

		bias = (float*)calloc(this->dimensions.back(), sizeof(float));
	}

	inline std::vector<float> const operator [] (int i) const {
		return std::vector<float>(bias + dimensions[i], bias + dimensions[i + 1]);
	}

	inline Biases& operator = (const Biases& other) noexcept {
		free(bias);
		bias = (float*)malloc(other.size() * sizeof(float));
		std::memcpy(bias, other.bias, other.size() * sizeof(float));

		this->dimensions = other.dimensions;

		return *this;
	}

	inline int size() const {
		return  dimensions.back();
	}

	inline void assign(int i, const std::vector<float> &vec) {
		std::memcpy(&bias[dimensions[i]], vec.data(), (dimensions[i + 1] - dimensions[i]) * sizeof(float));
	}

	void update(const Biases& deriv, float mult) {

		__m256 _scalar = _mm256_set1_ps(mult);

		int i = 0;
		for (; i + 8 < this->size(); i += 8) {

			_mm256_store_ps(&bias[i], _mm256_sub_ps(_mm256_load_ps(&bias[i]), _mm256_mul_ps(_mm256_load_ps(&deriv.bias[i]), _scalar)));
		}

		for (; i < this->size(); i++) {
			bias[i] -= deriv.bias[i] * mult;
		}
	}

	~Biases() {
		free(bias);
	}

	float* bias;
private:
	std::vector<int> dimensions;

};