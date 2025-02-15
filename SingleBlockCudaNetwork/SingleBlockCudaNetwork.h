#pragma once
#include <vector>
#include <iostream>
#include <chrono>
#include <random>
#include <string>

#include "CudaKernals.cuh"

class CudaNetwork {
public:

	static enum class weight_init {
		he, normalize, xavier
	};

	struct matrix {
		size_t rows;
		size_t cols;
		std::vector<float> mat;
	};

	void define(std::vector<size_t> dimensions);

	void compile(weight_init init);

	void fit(
		matrix x_train,
		matrix y_train,
		matrix x_test,
		matrix y_test,
		size_t batch_size, 
		size_t epochs,
		float learning_rate,
		bool shuffle, 
		int validation_freq
	);


private:
	std::vector<size_t> m_dimensions;

	float* m_network;
	float* m_batch_data;
	float* m_test_data;

	float* m_bias;

	float* m_activation;

	float* m_d_total;
	float* m_d_weights;
	float* m_d_bias;

	float* m_test_activation;

	size_t m_network_size;
	size_t m_weights_size;
	size_t m_bias_size;

	size_t m_batch_data_size;
	size_t m_batch_activation_size;

	size_t m_test_activation_size;

	// mem init
	void initialize_train_data(
		float** d_x_train, float** d_y_train, float** d_x_test, float** d_y_test,
		matrix h_x_train, matrix h_y_train, matrix h_x_test, matrix h_y_test
	);
	void initialize_batch_data(size_t batch_size);
	void initialize_test_data(size_t test_size);

	void forward_prop(float* x_data, float* result_data, size_t activation_size, size_t num_elements);
	void back_prop(float* x_data, float* y_data, float learning_rate, size_t num_elements);

	std::string test_network(float* x, float* y, size_t test_size);
	std::string verbose(float* d_x_test, float* d_y_test, size_t test_samples, size_t epoch, int validation_freq, std::chrono::steady_clock::time_point start_time);
	std::string clean_time(double time);
};