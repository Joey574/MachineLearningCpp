#pragma once
#include <vector>
#include <chrono>

#include "StandaloneNeuralNetwork.h"
#include "Matrix.h"

class StandaloneNeuralNetwork
{
public:

	// Network Enums
	static enum class loss_metric {
		none, mse, mae, cross_entropy, accuracy
	};
	static enum class activation_function {
		Sigmoid, ReLU, leakyReLU, ELU, tanH, Softplus, SiLU, Softmax
	};
	static enum class optimization_technique {
		none
	};

	// Network Structs
	struct training_history {
		std::chrono::duration<double, std::milli> train_time;
		std::chrono::duration<double, std::milli> epoch_time;
		std::vector<float> metric_history;
		std::vector<float> loss_history;
	};


	void Define(
		std::vector<int> dimensions,
		std::vector<activation_function> activation_functions
	);

	void Compile(
		loss_metric loss = loss_metric::none,
		loss_metric metrics = loss_metric::none,
		optimization_technique optimizer = optimization_technique::none
	);

	training_history Fit(
		Matrix x_train,
		Matrix y_train,
		Matrix x_valid,
		Matrix y_valid,
		int batch_size,
		int epochs,
		float learning_rate,
		float weight_decay = 0.0f,
		float validation_split = 0.0f,
		bool shuffle = true,
		int validation_freq = 1
	);


	~StandaloneNeuralNetwork() {
		free(_net);
		free(_results);
		free(_derivatives);
	}

private:
	float* _net;
	float* _results;
	float* _derivatives;

	int _net_size;
	int _results_size;
	int _derivatives_size;

	std::vector<int> _dimensions;


	void forward_propogate();
	void backward_propogate();
};
