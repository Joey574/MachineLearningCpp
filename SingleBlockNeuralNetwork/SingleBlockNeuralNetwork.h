#pragma once

#include <vector>
#include <random>
#include <chrono>
#include <numeric>

#include "Matrix.h"

class NeuralNetwork {
public:

	static enum class weight_init {
		he, normalize, xavier
	};
	static enum class loss_metric {
		mse
	};
	static enum class activation_functions {
		relu, leaky_relu, elu, sigmoid
	};

	struct history {
		std::chrono::duration<double, std::milli> train_time;
		std::chrono::duration<double, std::milli> epoch_time;
		std::vector<float> metric_history;
		std::vector<float> loss_history;
	};

	void define(
		std::vector<int> dimensions,
		std::vector<activation_functions> activations
	);

	void compile(
		loss_metric loss,
		loss_metric metrics,
		weight_init weight_initialization
	);

	history fit(
		Matrix& x_train,
		Matrix& y_train,
		Matrix& x_valid,
		Matrix& y_valid,
		int batch_size,
		int epochs,
		float learning_rate,
		bool shuffle,
		int validation_freq
	);

	~NeuralNetwork() {
		if (m_network) { free(m_network); }
		if (m_batch_data) { free(m_batch_data); }
		if (m_test_data) { free(m_test_data); }
	}

private:

	struct activation_data {
		activation_functions type;
		void (NeuralNetwork::* activation)(float*, float*, size_t);
		void (NeuralNetwork::* derivative)(float*, float*, size_t);
	};

	/* Memory Structure for network
	* 
	* M_NETWORK -> contains weights and biases
	* w0 -> w1 ... wn
	* b0 -> b1 ... bn
	* 
	* m_biases := pointer to b0
	* 
	* m_weights_size := size of weights
	* m_biases_size := size of biases
	* 
	* m_network_size := total size of network
	* 
	* 
	* M_BATCH_DATA -> contains activation and derivatives
	* t0 -> t1 ... tn
	* a0 -> a1 ... an
	* 
	* dt0 -> dt1 ... dtn
	* dw0 -> dw1 ... dwn
	* db0 -> db1 ... dbn
	* 
	* m_activation := pointer to a0
	* 
	* m_deriv_t := pointer to dt0
	* m_deriv_w := pointer to dw0
	* m_deriv_b := pointer to db0
	* 
	* m_batch_activation_size := size of t0 t1 ... (a0 a1 ... , are of the same size) (dt0 dt1 ... , are of the same size)
	* 
	* m_batch_data_size := total size of batch_data
	* 
	* M_TEST_DATA -> contains activation for test data
	* t0 -> t1 ... tn
	* a0 -> a1 ... an
	* 
	* m_test_activation := pointer to a0 in m_test_data
	* 
	* m_test_activation_size := size of t0 t1 ... in m_test_data, a0 a1 ... is of equal size
	*/

	// pointer to start of weights and biases
	float* m_network;

	float* m_biases;

	int m_network_size;
	int m_weights_size;
	int m_biases_size;

	// pointer to start of batch data -> results and derivs
	float* m_batch_data;

	float* m_activation;

	float* m_deriv_t;
	float* m_deriv_w;
	float* m_deriv_b;

	int m_batch_data_size;
	int m_batch_activation_size;

	// pointer to start of test data results
	float* m_test_data;

	float* m_test_activation;

	int m_test_activation_size;

	// misc
	std::vector<int> m_dimensions;
	std::vector<activation_data> m_activation_data;


	void forward_prop(
		float* x_data,
		float* result_data,
		int activation_size,
		int num_elements
	);

	void back_prop(
		float* x_data,
		float* y_data,
		float learning_rate,
		int num_elements
	);

	std::string test_network(
		float* x,
		float* y,
		int test_size
	);

	void dot_prod(float* a, float* b, float* c, size_t a_r, size_t a_c, size_t b_r, size_t b_c, bool clear);
	void dot_prod_t_a(float* a, float* b, float* c, size_t a_r, size_t a_c, size_t b_r, size_t b_c, bool clear);
	void dot_prod_t_b(float* a, float* b, float* c, size_t a_r, size_t a_c, size_t b_r, size_t b_c, bool clear);

	void initialize_batch_data(int batch_size);
	void initialize_test_data(int test_size);

	// activation functions
	void relu(float* x, float* y, size_t size);
	void leaky_relu(float* x, float* y, size_t size);
	void elu(float* x, float* y, size_t size);
	void sigmoid(float* x, float* y, size_t size);

	// activation derivatives
	void relu_derivative(float* x, float* y, size_t size);
	void leaky_relu_derivative(float* x, float* y, size_t size);
	void elu_derivative(float* x, float* y, size_t size);
	void sigmoid_derivative(float* x, float* y, size_t size);

	std::string clean_time(double time);
};