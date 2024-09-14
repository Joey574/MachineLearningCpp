#include "NeuralNetwork.h"
#include "NeuralNetworkUtils.cpp"
#include "NeuralNetworkLoss.cpp"

void NeuralNetwork::Define(std::vector<int> dimensions,	std::vector<activations> activation_type) {

	this->_dimensions = dimensions;

	if (activation_type.size() != dimensions.size() - 1) {
		std::cerr << "activation functions passed do not match dimensions\n";
	}
	
	_activation_functions = std::vector<activation_data>(activation_type.size());
	for (int i = 0; i < activation_type.size(); i++) {
		_activation_functions[i] = compile_activation_function(activation_type[i]);
	}

	std::cout << this->summary();
}
void NeuralNetwork::Compile(loss_metric loss, loss_metric metric, optimization_technique optimizer, Matrix::init weight_initialization) {

	// Initialization of network matrices
	for (int i = 0; i < _dimensions.size() - 1; i++) {
		_network.weights.emplace_back(_dimensions[i + 1], _dimensions[i], weight_initialization);
	}
	_network.biases = Biases(_dimensions);

	_loss = compile_metric_data(loss);
	_metric = compile_metric_data(metric);

	std::cout << "Status: network_compiled\n";
}
Matrix NeuralNetwork::Predict(Matrix x_test) {

	result_matrices test_results;

	// Initialize test result matrices
	test_results.total = std::vector<Matrix>(_network.weights.size());
	test_results.activation = std::vector<Matrix>(_network.weights.size());

	x_test = x_test.Transpose();
	test_results = forward_propogate(x_test, _network, test_results);

	return test_results.activation.back();
}
std::string NeuralNetwork::Evaluate(Matrix x_test, Matrix y_test) {
	return test_network(x_test, y_test, _network);
}

std::string NeuralNetwork::test_network(Matrix x, Matrix y, network_structure net) {

	std::string out;

	result_matrices test_results;

	// Initialize test result matrices
	test_results.total = std::vector<Matrix>(net.weights.size());
	test_results.activation = std::vector<Matrix>(net.weights.size());

	x = x.Transpose();
	test_results = forward_propogate(x, net, test_results);

	out = _metric.name + (": ");

	out.append(std::to_string((this->*_metric.total)(test_results.activation.back(), y.Transpose())));

	return out;
}

NeuralNetwork::training_history NeuralNetwork::Fit(Matrix x_train, Matrix y_train, Matrix x_valid, Matrix y_valid, int batch_size, int epochs, float learning_rate, float weight_decay, float validation_split, bool shuffle, int validation_freq) {
	std::cout << "Status: network_training\n";

	auto start_time = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> time;

	result_matrices current_results;
	derivative_matrices current_derivs;

	training_history history;

	// If validation data is provided, don't split training data
	if (x_valid.matrix && y_valid.matrix) {
		validation_split = 0.0f;
	}

	initialize_result_matrices(batch_size, current_results, current_derivs);

	std::tie(x_train, y_train, x_valid, y_valid) = data_preprocessing(x_train, y_train, x_valid, y_valid, shuffle, validation_split);

	const int iterations = x_train.RowCount / batch_size;
	for (int e = 0; e < epochs; e++) {

		auto epoch_start_time = std::chrono::high_resolution_clock::now();

		for (int i = 0; i < iterations; i++) {

			Matrix x = x_train.SegmentR((i * batch_size), (batch_size + (i * batch_size))).Transpose();
			Matrix y = y_train.SegmentR((i * batch_size), (batch_size + (i * batch_size)));

			current_results = forward_propogate(x, _network, current_results);
			_network = backward_propogate(x, y, learning_rate, weight_decay, _network, current_results, current_derivs);
		}

		// Test network every n epochs
		std::string out;
		if (e % validation_freq == 0) {
			std::string score = test_network(x_valid, y_valid, _network);

			time = std::chrono::high_resolution_clock::now() - epoch_start_time;
			std::cout << "Epoch: " << e << " Time: " << clean_time(time.count()) << " " << score << "\n";
		}
		else {
			time = std::chrono::high_resolution_clock::now() - epoch_start_time;
			std::cout << "Epoch: " << e << " Time: " << clean_time(time.count()) << "\n";
		}
		intermediate_history(history);
	}
	auto end_time = std::chrono::high_resolution_clock::now();

	final_history(history, start_time, end_time, epochs);

	std::cout << "Status: training_complete\n";

	return history;
}

NeuralNetwork::result_matrices NeuralNetwork::forward_propogate(Matrix x, network_structure net, result_matrices results) {
	for (int i = 0; i < results.total.size(); i++) {
		results.total[i] = net.weights[i].dot_product_add(((i == 0) ? x : results.activation[i - 1]), net.biases[i]);
		results.activation[i] = (results.total[i].*_activation_functions[i].activation)();
	}
	return results;
}
NeuralNetwork::network_structure  NeuralNetwork::backward_propogate(Matrix x, Matrix y, float learning_rate, float weight_decay, network_structure net, result_matrices results, derivative_matrices deriv) {
	
	const float l2_reg = weight_decay ? (1.0f - learning_rate * (weight_decay / x.ColumnCount)) : 1.0f;
	const float s_factor = std::sqrt(learning_rate) / std::sqrt(float(x.ColumnCount));

	// Compute loss
	deriv.d_total.back() = (this->*_loss.derivative)(results.activation.back(), y.Transpose());

	for (int i = deriv.d_total.size() - 1; i > 0; i--) {
		deriv.d_total[i - 1] = net.weights[i].Transpose().dot_product(deriv.d_total[i]) * (results.total[i - 1].*_activation_functions[i - 1].derivative)();
	}

	for (int i = 0; i < deriv.d_weights.size(); i++) {
		deriv.d_weights[i] = deriv.d_total[i].dot_product(i == 0 ? x.Transpose() : results.activation[i - 1].Transpose()) * s_factor;
		deriv.d_biases.assign(i, deriv.d_total[i].Multiply(s_factor).RowSums());
	}

	// net.weights[i].matrix[j] := (net.weights[i].matrix[j] * l2_reg) - (deriv.d_weights[i].matrix[j] * s_factor)
	#pragma omp parallel for
	for (int i = 0; i < net.weights.size(); i++) {

		int j = 0;
		for (; j + 16 <= net.weights[i].RowCount * net.weights[i].ColumnCount; j += 8) {
			_mm256_store_ps(&net.weights[i].matrix[j],
				_mm256_fmsub_ps(
					_mm256_load_ps(&net.weights[i].matrix[j]),
					_mm256_set1_ps(l2_reg),
					_mm256_mul_ps(_mm256_load_ps(&deriv.d_weights[i].matrix[j]), _mm256_set1_ps(s_factor))));

			j += 8;
			_mm256_store_ps(&net.weights[i].matrix[j],
				_mm256_fmsub_ps(
					_mm256_load_ps(&net.weights[i].matrix[j]),
					_mm256_set1_ps(l2_reg),
					_mm256_mul_ps(_mm256_load_ps(&deriv.d_weights[i].matrix[j]), _mm256_set1_ps(s_factor))));
		}

		for (; j < net.weights[i].RowCount * net.weights[i].ColumnCount; j++) {\
			net.weights[i].matrix[j] = (net.weights[i].matrix[j] * l2_reg) - (deriv.d_weights[i].matrix[j] * s_factor);
		}
	}
	net.biases.update(deriv.d_biases, s_factor);

	return net;
}

void NeuralNetwork::base_weight_update() {

}
void NeuralNetwork::l1_weight_update() {

}
void NeuralNetwork::l2_weight_update() {

}
void NeuralNetwork::momentum_weight_update() {

}