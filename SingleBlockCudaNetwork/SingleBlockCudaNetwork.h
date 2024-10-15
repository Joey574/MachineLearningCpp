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

	void define(std::vector<size_t> dimensions);

	void compile(weight_init init);

	void fit(
		float* x_train, 
		float* y_train, 
		float* x_valid, 
		float* y_valid, 
		size_t train_samples, 
		size_t test_samples, 
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
	void initialize_batch_data(size_t batch_size);
	void initialize_test_data(size_t test_size);

	void forward_prop(float* x_data, float* result_data, size_t activation_size, size_t num_elements);
	void back_prop(float* x_data, float* y_data, float learning_rate, size_t num_elements);

	std::string test_network(float* x, float* y, size_t test_size);
	std::string clean_time(double time);
};