#pragma once

#include <vector>
#include <random>
#include <chrono>
#include <numeric>
#include <fstream>

#include "Matrix.h"

class NeuralNetwork {
public:

	static enum class weight_init {
		he, normalize, xavier
	};
	static enum class loss_metric {
		mae, accuracy, one_hot
	};
	static enum class activation_functions {
		relu, leaky_relu, elu, sigmoid, softmax
	};

	struct history {
		std::chrono::duration<double, std::milli> train_time;
		std::chrono::duration<double, std::milli> epoch_time;
		std::vector<double> metric_history;
	};

	void define(
		std::vector<size_t> dimensions,
		std::vector<activation_functions> activations
	);

	void compile(
		loss_metric loss,
		loss_metric metrics,
		weight_init weight_initialization
	);

	history fit(
		Matrix x_train,
		Matrix y_train,
		Matrix x_valid,
		Matrix y_valid,
		size_t batch_size,
		size_t epochs,
		float learning_rate,
		bool shuffle,
		int validation_freq,
		float validation_split
	);

	std::vector<float> predict(const Matrix& x);

	std::string summary();

	void serialize(std::string filepath);
	void deserialize(std::string filepath);

	~NeuralNetwork() {
		//_aligned_free(m_network); m_network = nullptr;
		//if (m_batch_data) { _aligned_free(m_batch_data); m_batch_data = nullptr; }
		//if (m_test_data) { _aligned_free(m_test_data); m_test_data = nullptr; }
	}

private:

	struct activation_data {
		activation_functions type;
		void (NeuralNetwork::* activation)(float*, float*, size_t);
		void (NeuralNetwork::* derivative)(float*, float*, size_t);
	};

	/* Memory Layout


	 _____|m_network|_____ 
	|					  |
	|		weights		  |  <- m_weights_size
	|					  |
	|------|m_biases|-----|
	|					  |
	|		 biases		  |  <- m_bias_size
	|					  |
	 ---------------------

	 m_network_size := m_weights_size + m_bias_size



	 ____|m_batch_data|____
	|					   |
	|		  total		   |  <- m_batch_activation_size
	|					   |
	|----|m_activation|----|
	|					   |
	|	   activation	   |  <- m_batch_activation_size
	|					   |
	|------|m_d_total|-----|
	|					   |
	|		d_total		   |  <- m_batch_activation_size
	|					   |
	|-----|m_d_weights|----|
	|					   |
	|	   d_weights	   |  <- m_weights_size
	|					   |
	|-----|m_d_biases|-----|
	|					   |
	|	    d_biases	   |  <- m_bias_size
	|					   |
	 ----------------------

	 m_batch_data_size := (3 * m_batch_activation_size) + m_network_size



	 _____|m_test_data|_____
	|					    |
	|		  total		    |  <- m_test_activation_size
	|					    |
	|--|m_test_activation|--|
	|					    |
	|	   activation	    |  <- m_test_activation_size
	|					    |
	 -----------------------


	*/

	// pointers
	float* m_network;
	float* m_batch_data;
	float* m_test_data;

	float* m_biases;

	float* m_activation;

	float* m_d_total;
	float* m_d_weights;
	float* m_d_biases;

	float* m_test_activation;

	// size for various pointers
	size_t m_network_size;
	size_t m_weights_size;
	size_t m_biases_size;

	size_t m_batch_data_size;
	size_t m_batch_activation_size;

	size_t m_test_activation_size;


	bool loaded = false;

	// misc
	std::vector<size_t> m_dimensions;
	std::vector<activation_data> m_activation_data;

	void (NeuralNetwork::* m_loss)(float*, float*, float*, size_t, size_t);
	float (NeuralNetwork::* m_metric)(float* a, float* b, size_t a_r, size_t a_c);

	void data_preprocess(Matrix& x_train, Matrix& y_train, Matrix& x_valid, Matrix& y_valid, float validation_split, bool shuffle);

	void forward_prop(
		float* x_data,
		float* result_data,
		size_t activation_size,
		size_t num_elements
	);

	void back_prop(
		float* x_data,
		float* y_data,
		float learning_rate,
		size_t num_elements
	);

	std::string test_network(
		float* x,
		float* y,
		size_t test_size,
		history& h
	);


	// dot prods
	void dot_prod(float* a, float* b, float* c, size_t a_r, size_t a_c, size_t b_r, size_t b_c, bool clear);
	void dot_prod_t_a(float* a, float* b, float* c, size_t a_r, size_t a_c, size_t b_r, size_t b_c, bool clear);
	void dot_prod_t_b(float* a, float* b, float* c, size_t a_r, size_t a_c, size_t b_r, size_t b_c, bool clear);

	// mem init
	void initialize_batch_data(size_t batch_size);
	void initialize_test_data(size_t test_size);

	// activation functions
	void relu(float* x, float* y, size_t size);
	void leaky_relu(float* x, float* y, size_t size);
	void elu(float* x, float* y, size_t size);
	void sigmoid(float* x, float* y, size_t size);
	void softmax(float* x, float* y, size_t size);

	// activation derivatives
	void relu_derivative(float* x, float* y, size_t size);
	void leaky_relu_derivative(float* x, float* y, size_t size);
	void elu_derivative(float* x, float* y, size_t size);
	void sigmoid_derivative(float* x, float* y, size_t size);

	// score
	float mae_score(float* x, float* y, size_t rows, size_t columns);
	float accuracy_score(float* x, float* y, size_t rows, size_t columns);

	// loss
	void mae_loss(float* x, float* y, float* c, size_t rows, size_t columns);
	void one_hot_loss(float* x, float* y, float* c, size_t rows, size_t columns);

	std::string clean_time(double time);
};