#pragma once

#include <vector>
#include <random>
#include <chrono>

#include "Matrix.h"

class NeuralNetwork {
public:

	static enum class weight_init {
		he, normalize, xavier
	};

	static enum class loss_metric {
		mse
	};

	void define(
		std::vector<int> dimensions
	);

	void compile(
		loss_metric loss,
		loss_metric metrics,
		weight_init weight_initialization
	);

	void fit(
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
		free(m_network);
		free(m_batch_data);
	}

private:

	// pointer to start of weights and biases
    float* m_network;
	int m_network_size;

	// pointer to start of batch data -> results and derivs
	float* m_batch_data;
	int m_batch_data_size;


	std::vector<int> m_dimensions;


	void forward_prop(
		float* x_data,
		float* y_data, 
		float learning_rate
	);

	void back_prop();

};