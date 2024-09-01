#include "NeuralNetwork.h"

void NeuralNetwork::Define(std::vector<int> dimensions, std::unordered_set<int> res_net, std::unordered_set<int> batch_normalization,
	std::vector<Matrix(Matrix::*)() const> activation_functions, std::vector<Matrix(Matrix::*)() const> derivative_functions) {

	this->_res_net = res_net;
	this->_batch_norm = batch_normalization;
	this->_dimensions = dimensions;

	if (activation_functions.size() != dimensions.size() - 1 || derivative_functions.size() != dimensions.size() - 2) {
		std::cout << "activation functions passed do not match required size\n";
	}
	
	this->_activation_functions = activation_functions;
	this->_derivative_functions = derivative_functions;

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
			std::cout << "Epoch: " << e << " Time: " << clean_time(time.count()) << " " << score << std::endl;
		}
		else {
			time = std::chrono::high_resolution_clock::now() - epoch_start_time;
			std::cout << "Epoch: " << e << " Time: " << clean_time(time.count()) << std::endl;
		}
		intermediate_history(history);
	}
	auto end_time = std::chrono::high_resolution_clock::now();

	final_history(history, start_time, end_time, epochs);

	std::cout << "Status: training_complete\n";

	return history;
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

void NeuralNetwork::initialize_result_matrices(int batch_size, result_matrices &results, derivative_matrices &derivs) {
	results.total = std::vector<Matrix>(_network.weights.size());
	results.activation = std::vector<Matrix>(_network.weights.size());

	derivs.d_total = std::vector<Matrix>(_network.weights.size());
	derivs.d_weights = std::vector<Matrix>(_network.weights.size());
	derivs.d_biases = Biases(_dimensions);
}

std::tuple<Matrix, Matrix, Matrix, Matrix> NeuralNetwork::data_preprocessing(Matrix x_train, Matrix y_train, Matrix x_valid, Matrix y_valid, bool shuffle, float validation_split) {
	Matrix x;
	Matrix y;

	if (shuffle) {
		std::tie(x_train, y_train) = Shuffle(x_train, y_train);
	}

	if (validation_split > 0.0f) {
		int elements = (float)x_train.RowCount * validation_split;

		x_valid = x_train.SegmentR(x_train.RowCount - elements);
		y_valid = y_train.SegmentR(x_train.RowCount - elements);

		x = x_train.SegmentR(0, x_train.RowCount - elements);
		y = y_train.SegmentR(0, x_train.RowCount - elements);
	} else {
		x = x_train;
		y = y_train;
	}
	return std::make_tuple(x, y, x_valid, y_valid);
}


NeuralNetwork::result_matrices NeuralNetwork::forward_propogate(Matrix x, network_structure net, result_matrices results) {
	for (int i = 0; i < results.total.size(); i++) {
		results.total[i] = net.weights[i].dot_product_add(((i == 0) ? x : results.activation[i - 1]), net.biases[i]);
		results.activation[i] = (results.total[i].*_activation_functions[i])();
	}
	return results;
}
NeuralNetwork::network_structure  NeuralNetwork::backward_propogate(Matrix x, Matrix y, float learning_rate, float weight_decay, network_structure net, result_matrices results, derivative_matrices deriv) {
	
	const float s_batch = std::sqrt((float)x.ColumnCount);
	const float factor = weight_decay ? (1.0f - learning_rate * (weight_decay / s_batch)) : 1.0f;

	// Compute loss
	deriv.d_total[deriv.d_total.size() - 1] = (this->*_loss.derivative)(results.activation.back(), y.Transpose());

	for (int i = deriv.d_total.size() - 1; i > 0; i--) {
		deriv.d_total[i - 1] = net.weights[i].Transpose().dot_product(deriv.d_total[i]) * (results.total[i - 1].*_derivative_functions[i - 1])();
	}

	for (int i = 0; i < deriv.d_weights.size(); i++) {
		//deriv.d_weights[i] = deriv.d_total[i].dot_product(i == 0 ? x.Transpose() : results.activation[i - 1].Transpose()) * ((1.0f / s_batch) * factor);
		deriv.d_weights[i] = deriv.d_total[i].dot_product(i == 0 ? x.Transpose() : results.activation[i - 1].Transpose()) * ((1.0f / s_batch));
		deriv.d_biases.assign(i, deriv.d_total[i].Multiply(1.0f / s_batch).RowSums());
	}

	for (int i = 0; i < net.weights.size(); i++) {
		//net.weights[i] -= deriv.d_weights[i].Multiply((learning_rate / s_batch) * factor);
		net.weights[i] = net.weights[i] * (1.0f - learning_rate * (weight_decay / x.ColumnCount)) - deriv.d_weights[i].Multiply(learning_rate / s_batch);
	}
	net.biases.update(deriv.d_biases, learning_rate / s_batch);

	return net;
}

NeuralNetwork::result_matrices NeuralNetwork::forward_propogate_optimized(Matrix x, network_structure net, result_matrices results) {
	for (int i = 0; i < results.total.size(); i++) {

		int x = 0;
		for (; x + 8 < results.total[x].RowCount * results.total[x].ColumnCount; x += 8) {
			_mm256_store_ps(&results.total[i].matrix[x], _mm256_set1_ps(net.biases[i][x / results.total[i].ColumnCount]));
		}
		for (; x < results.total[x].RowCount * results.total[x].ColumnCount; x++) {
			results.total[i].matrix[x] = net.biases[i][x / results.total[i].ColumnCount];
		}


	}
}
NeuralNetwork::network_structure NeuralNetwork::backward_propogate_optimized(Matrix x, Matrix y, float learning_rate, float weight_decay, network_structure net, result_matrices results, derivative_matrices deriv) {

}


void NeuralNetwork::intermediate_history(NeuralNetwork::training_history& history) {

}

void NeuralNetwork::final_history(NeuralNetwork::training_history& history, std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end, int epochs) {
	history.epoch_time = (end - start) / epochs;
	history.train_time = end - start;
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


std::tuple<Matrix, Matrix> NeuralNetwork::Shuffle(Matrix x, Matrix y) {
	for (int k = 0; k < x.RowCount; k++) {

		int r = k + rand() % (x.RowCount - k);

		std::vector<float> tempX = x.Row(k);
		std::vector<float> tempY = y.Row(k);

		x.SetRow(k, x.Row(r));
		y.SetRow(k, y.Row(r));

		x.SetRow(r, tempX);
		y.SetRow(r, tempY);
	}

	return std::make_tuple(x, y);
}

void NeuralNetwork::save(std::string filename) {
	std::ofstream file_writer(filename, std::ios::binary);

	// write size of network_dimensions
	int dims = _dimensions.size();
	file_writer.write(reinterpret_cast<const char*>(&dims), sizeof(int));

	// Write loss and metric enums
	file_writer.write(reinterpret_cast<const char*>(&_loss.type), sizeof(loss_metric));
	file_writer.write(reinterpret_cast<const char*>(&_metric.type), sizeof(loss_metric));

	// write network dimensions
	file_writer.write(reinterpret_cast<const char*>(_dimensions.data()), dims * sizeof(int));

	// write weights
	for (const auto &weight : _network.weights) {
		file_writer.write(reinterpret_cast<const char*>(weight.matrix), weight.RowCount * weight.ColumnCount * sizeof(float));
	}

	// write biases
	file_writer.write(reinterpret_cast<const char*>(_network.biases.bias), _network.biases.size() * sizeof(float));
}
void NeuralNetwork::load(std::string filename) {
	std::ifstream file_reader(filename, std::ios::binary);

	if (!file_reader.is_open()) {
		std::cout << "file not found\n";
	}

	// read dims size
	int dims;
	file_reader.read(reinterpret_cast<char*>(&dims), sizeof(int));
	_dimensions = std::vector<int>(dims);

	// read loss and metric data
	loss_metric loss;
	loss_metric metric;

	file_reader.read(reinterpret_cast<char*>(&loss), sizeof(loss_metric));
	file_reader.read(reinterpret_cast<char*>(&metric), sizeof(loss_metric));

	_loss = compile_metric_data(loss);
	_metric = compile_metric_data(metric);

	// read network dimensions
	file_reader.read(reinterpret_cast<char*>(_dimensions.data()), dims * sizeof(int));

	// read weights
	_network.weights = std::vector<Matrix>(dims - 1);
	for (int i = 0; i < _dimensions.size() - 1; i++) {
		_network.weights[i] = Matrix(_dimensions[i + 1], _dimensions[i]);
		file_reader.read(reinterpret_cast<char*>(&_network.weights[i].matrix[0]), _dimensions[i] * _dimensions[i + 1] * sizeof(float));
	}

	// read biases
	_network.biases = Biases(_dimensions);
	file_reader.read(reinterpret_cast<char*>(&_network.biases.bias[0]), std::accumulate(_dimensions.begin() + 1, _dimensions.end(), 0) * sizeof(float));
}

std::string NeuralNetwork::summary() {
	std::string summary = "Network Summary {\ndims = ";
	for (int i = 0; i < _dimensions.size() - 1; i++) {
		summary.append(std::to_string(_dimensions[i])).append("_");
	} summary.append(std::to_string(_dimensions.back()));

	summary.append("\n}\n\n");
	return summary;
}

NeuralNetwork::metric_data NeuralNetwork::compile_metric_data(loss_metric type) {
	metric_data metric;
	metric.type = type;
	switch (type) {
	case loss_metric::mse:
		metric.name = "mse";
		metric.derivative = &NeuralNetwork::mse_loss;
		metric.total = &NeuralNetwork::mse_total;
		break;
	case loss_metric::mae:
		metric.name = "mae";
		metric.derivative = &NeuralNetwork::mae_loss;
		metric.total = &NeuralNetwork::mae_total;
		break;
	case loss_metric::cross_entropy:
		metric.name = "cross_entropy";
		metric.derivative = &NeuralNetwork::cross_entropy;
		metric.total = &NeuralNetwork::accuracy;
		break;
	case loss_metric::accuracy:
		metric.name = "accuracy";
		metric.derivative = &NeuralNetwork::cross_entropy;
		metric.total = &NeuralNetwork::accuracy;
		break;
	}
	return metric;
}
NeuralNetwork::activation_function_data NeuralNetwork::compile_activation_function(activation_function type) {
	activation_function_data function;
	function.type = type;
	switch (type) {
	case activation_function::Sigmoid:
		function.activation = &Matrix::Sigmoid;
		function.derivative = &Matrix::SigmoidDerivative;
		break;
	case activation_function::ReLU:
		function.activation = &Matrix::ReLU;
		function.derivative = &Matrix::ReLUDerivative;
		break;
	case activation_function::leakyReLU:
		function.activation = &Matrix::_LeakyReLU;
		function.derivative = &Matrix::_LeakyReLUDerivative;
		break;
	case activation_function::ELU:
		function.activation = &Matrix::_ELU;
		function.derivative = &Matrix::_ELUDerivative;
		break;
	case activation_function::tanH:
		function.activation = &Matrix::Tanh;
		function.derivative = &Matrix::TanhDerivative;
		break;
	case activation_function::Softplus:
		function.activation = &Matrix::Softplus;
		function.derivative = &Matrix::SoftplusDerivative;
		break;
	case activation_function::SiLU:
		function.activation = &Matrix::SiLU;
		function.derivative = &Matrix::SiLUDerivative;
		break;
	case activation_function::Softmax:
		function.activation = &Matrix::SoftMax;
		function.derivative = &Matrix::SigmoidDerivative;
		break;
	}
	return function;
}

Matrix NeuralNetwork::mae_loss(Matrix final_activation, Matrix labels) {
	return (final_activation - labels);
}
float NeuralNetwork::mae_total(Matrix final_activation, Matrix labels) {
	return  std::abs(std::accumulate(final_activation.matrix, final_activation.matrix + (final_activation.RowCount * final_activation.ColumnCount), 0) -
		std::accumulate(labels.matrix, labels.matrix + (labels.RowCount * labels.ColumnCount), 0)) / labels.ColumnCount;
}
Matrix NeuralNetwork::mse_loss(Matrix final_activation, Matrix labels) {
	return (final_activation - labels);
}
float NeuralNetwork::mse_total(Matrix final_activation, Matrix labels) {
	return  std::abs(std::pow(std::accumulate(final_activation.matrix, final_activation.matrix + (final_activation.RowCount * final_activation.ColumnCount), 0) -
		std::accumulate(labels.matrix, labels.matrix + (labels.RowCount * labels.ColumnCount), 0), 2)) / labels.ColumnCount;
}

Matrix NeuralNetwork::cross_entropy(Matrix final_activation, Matrix labels) {
	for (int c = 0; c < final_activation.ColumnCount; c++) {
		final_activation(labels(0, c), c)--;
	}
	return final_activation;
}
float NeuralNetwork::accuracy(Matrix final_activation, Matrix labels) {
	int correct = 0;
	for (int c = 0; c < final_activation.ColumnCount; c++) {
		std::vector<float> col = final_activation.Column(c);
		int max_idx = std::distance(col.begin(), std::max_element(col.begin(), col.end()));

		correct = max_idx == labels(0, c) ? correct + 1 : correct;
	}
	return (float)correct / (float)final_activation.ColumnCount * 100.0f;
}

std::string NeuralNetwork::clean_time(double time) {
	const double hour = 3600000.00;
	const double minute = 60000.00;
	const double second = 1000.00;
	std::string out;

	if (time / hour > 1.00) {
		out = std::to_string(time / hour).append(" hours");
	}
	else if (time / minute > 1.00) {
		out = std::to_string(time / minute).append(" minutes");
	}
	else if (time / second > 1.00) {
		out = std::to_string(time / second).append(" seconds");
	}
	else {
		out = std::to_string(time).append(" ms");
	}
	return out;
}