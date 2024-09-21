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

	struct history {
		std::chrono::duration<double, std::milli> train_time;
		std::chrono::duration<double, std::milli> epoch_time;
		std::vector<float> metric_history;
		std::vector<float> loss_history;
	};

	void define(
		std::vector<int> dimensions
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
		int batch_size,
		int epochs,
		float learning_rate,
		bool shuffle,
		int validation_freq
	);

	~NeuralNetwork() {
		if (m_network) { free(m_network); }
	}

private:

	/* Memory Structure for network
	* 
	* m_network -> contains weights and biases
	* w0 -> w1 -> ... wn
	* b0 -> b1 -> ... bn
	* 
	* m_biases := pointer to b0
	* 
	* m_weights_size := size of weights
	* m_biases_size := size of biases
	* 
	* m_network_size := total size of network
	* 
	* 
	* m_batch_data
	* t0 -> t1 := ... tn
	* a0 -> a1 := ... an
	* 
	* dt0 -> dt1 := ... dtn
	* dw0 -> dw1 := .. dwn
	* db0 -> db1 := ... dbn
	* 
	* m_activation := pointer to a0
	* 
	* m_deriv_t := pointer to dt0
	* m_deriv_w := pointer to dw0
	* m_deriv_b := pointer to db0
	* 
	* m_r_total_size := size of t0 t1 ... (a0 a1 ... , are of the same size) (dt0 dt1 ... , are of the same size)
	* 
	* m_batch_data_size := total size of batch_data
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
	int m_r_total_size;

	std::vector<int> m_dimensions;


	void forward_prop(
		float* x_data,
		float learning_rate,
		int batch_size
	);

	void back_prop(
		float* x_data,
		float* y_data,
		float learning_rate,
		int batch_size
	);

	std::string test_network(
		float* x,
		float* y
	);

	void initialize_batch_data(int batch_size);

	std::string clean_time(double time);

};