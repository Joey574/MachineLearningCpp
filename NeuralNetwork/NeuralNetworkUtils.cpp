#include "NeuralNetwork.h"

void NeuralNetwork::intermediate_history(NeuralNetwork::training_history& history) {

}
void NeuralNetwork::final_history(NeuralNetwork::training_history& history, std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end, int epochs) {
	history.epoch_time = (end - start) / epochs;
	history.train_time = end - start;
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
NeuralNetwork::activation_data NeuralNetwork::compile_activation_function(activations type) {
	activation_data function;
	function.type = type;
	switch (type) {
	case activations::Sigmoid:
		function.activation = &Matrix::Sigmoid;
		function.derivative = &Matrix::SigmoidDerivative;
		break;
	case activations::ReLU:
		function.activation = &Matrix::ReLU;
		function.derivative = &Matrix::ReLUDerivative;
		break;
	case activations::leakyReLU:
		function.activation = &Matrix::LeakyReLU;
		function.derivative = &Matrix::LeakyReLUDerivative;
		break;
	case activations::ELU:
		function.activation = &Matrix::ELU;
		function.derivative = &Matrix::ELUDerivative;
		break;
	case activations::tanH:
		function.activation = &Matrix::Tanh;
		function.derivative = &Matrix::TanhDerivative;
		break;
	case activations::Softplus:
		function.activation = &Matrix::Softplus;
		function.derivative = &Matrix::SoftplusDerivative;
		break;
	case activations::SiLU:
		function.activation = &Matrix::SiLU;
		function.derivative = &Matrix::SiLUDerivative;
		break;
	case activations::Softmax:
		function.activation = &Matrix::SoftMax;
		function.derivative = &Matrix::SigmoidDerivative;
		break;
	}
	return function;
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
	}
	else {
		x = x_train;
		y = y_train;
	}
	return std::make_tuple(x, y, x_valid, y_valid);
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


void NeuralNetwork::initialize_result_matrices(int batch_size, result_matrices& results, derivative_matrices& derivs) {
	results.total = std::vector<Matrix>(_network.weights.size());
	results.activation = std::vector<Matrix>(_network.weights.size());

	derivs.d_total = std::vector<Matrix>(_network.weights.size());
	derivs.d_weights = std::vector<Matrix>(_network.weights.size());
	derivs.d_biases = Biases(_dimensions);
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
	for (const auto& weight : _network.weights) {
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
