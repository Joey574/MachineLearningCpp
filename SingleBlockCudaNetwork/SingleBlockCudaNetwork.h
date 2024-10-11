#pragma once
#include <vector>
#include <iostream>
#include <chrono>
#include <string>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

class CudaNetwork {
public:

	static enum class weight_init {
		he, normalize, xavier
	};

	void define(std::vector<int> dimensions);

	void compile(weight_init init);

	void fit(
		float* x_train,
		float* y_train,
		float* x_valid,
		float* y_valid,
		int num_elements,
		int batch_size,
		int epochs,
		float learning_rate,
		bool shuffle,
		int validation_freq	
	);


private:

	std::vector<int> m_dimensions;

	float* m_network;
	float* m_batch_data;
	float* m_test_data;

	float* m_bias;

	float* m_activation;

	float* m_d_total;
	float* m_d_weights;
	float* m_d_bias;

	size_t m_weights_size;
	size_t m_bias_size;
	size_t m_batch_activation_size;


	__global__ void dot_prod(float* a, float* b, float* c, size_t a_r, size_t a_c, size_t b_r, size_t b_c);
	__global__ void dot_prod_t_a(float* a, float* b, float* c, size_t a_r, size_t a_c, size_t b_r, size_t b_c);
	__global__ void dot_prod_t_b(float* a, float* b, float* c, size_t a_r, size_t a_c, size_t b_r, size_t b_c);

	__global__ void horizontal_add(float* a, float* b, size_t a_r, size_t a_c);
	__global__ void horizontal_sum(float* a, float* b, size_t a_r, size_t a_c);

	__global__ void update_weights(float* a, float* b, float lr, size_t n);
	__global__ void update_bias(float* a, float* b, float lr, size_t n);


	__global__ void log_loss(float* a, float* b, float* y, size_t a_r, size_t a_c);


	// activation functions
	__global__ void leaky_relu(float* x, float* y, size_t r, size_t c);

	// derivative functions
	__global__ void leaky_relu_derivative(float* x, float* y, size_t r, size_t c);


	void forward_prop(float* x_data, float* result_data, int activation_size, int num_elements);
	void back_prop(float* x_data, float* y_data, float learning_rate, int num_elements);

	std::string test_network(float* x, float* y, int test_size);

	std::string clean_time(double time);
};