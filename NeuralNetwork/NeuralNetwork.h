#pragma once
#include <unordered_set>
#include <iostream>
#include <fstream>

#include "Matrix.h"
#include "Biases.h"

class NeuralNetwork
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
		std::unordered_set<int> res_net,
		std::unordered_set<int> batch_normalization,
		std::vector<Matrix(Matrix::*)() const> activation_functions,
		std::vector<Matrix(Matrix::*)() const> derivative_functions
	);

	void Compile(
		loss_metric loss = loss_metric::none,
		loss_metric metrics = loss_metric::none,
		optimization_technique optimizer = optimization_technique::none,
		Matrix::init weight_initialization = Matrix::init::Random
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

	std::tuple<Matrix, Matrix> Shuffle(Matrix x, Matrix y);

	std::string Evaluate(Matrix x_test, Matrix y_test);

	Matrix Predict(Matrix x_test);

	void save(std::string filename);

	void load(std::string filename);

	std::string summary();

private:

	// Network Structs
	struct network_structure {
		std::vector<Matrix> weights;
		Biases biases;
	};
	struct result_matrices {
		std::vector<Matrix> total;
		std::vector<Matrix> activation;
	};
	struct derivative_matrices {
		std::vector<Matrix> d_total;
		std::vector<Matrix> d_weights;
		Biases d_biases;
	};

	// Other structs
	struct metric_data {
		std::string name;
		loss_metric type;
		Matrix(NeuralNetwork::* derivative)(Matrix final_activation, Matrix labels);
		float(NeuralNetwork::* total)(Matrix final_activation, Matrix labels);
	};

	struct activation_function_data {
		activation_function type;
		Matrix(Matrix::* activation)() const;
		Matrix(Matrix::* derivative)() const;
	};

	// Function Pointers
	std::vector<Matrix(Matrix::*)() const>  _activation_functions;
	std::vector<Matrix(Matrix::*)() const>  _derivative_functions;

	// Cost functions / metrics
	metric_data _loss;
	metric_data _metric;

	// Network
	network_structure _network;

	std::vector<int> _dimensions;

	std::unordered_set<int> _res_net;
	std::unordered_set<int> _batch_norm;

	// Function Prototypes
	void initialize_result_matrices(int batch_size, result_matrices &results, derivative_matrices &derivs);
	std::tuple<Matrix, Matrix, Matrix, Matrix> data_preprocessing(Matrix x_train, Matrix y_train, Matrix x_valid, Matrix y_valid, bool shuffle, float validation_split);

	result_matrices forward_propogate(Matrix x, network_structure net, result_matrices results);
	network_structure backward_propogate(Matrix x, Matrix y, float learning_rate, float weight_decay, network_structure net, result_matrices results, derivative_matrices deriv);
	std::string test_network(Matrix x, Matrix y, network_structure net);

	void intermediate_history(training_history& history);
	void final_history(training_history& history, std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end, int epochs);

	metric_data compile_metric_data(loss_metric type);
	activation_function_data compile_activation_function(activation_function type);

	Matrix mse_loss(Matrix final_activation, Matrix labels);
	float mse_total(Matrix final_activation, Matrix labels);

	Matrix mae_loss(Matrix final_activation, Matrix labels);
	float mae_total(Matrix final_activation, Matrix labels);

	Matrix cross_entropy(Matrix final_activation, Matrix labels);
	float accuracy(Matrix final_activation, Matrix labels);

	std::string clean_time(double time);
};