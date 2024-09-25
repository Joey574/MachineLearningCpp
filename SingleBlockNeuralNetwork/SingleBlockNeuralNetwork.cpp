#include "SingleBlockNeuralNetwork.h"

#include "SingleBlockActivations.cpp"
#include "SingleBlockDotProds.cpp"

void NeuralNetwork::define(std::vector<int> dimensions, std::vector<activation_functions> activations) {

	m_network_size = 0;
	m_weights_size = 0;
	m_biases_size = 0;

	for (int i = 0; i < dimensions.size() - 1; i++) {

		// size for weights
		m_network_size += (dimensions[i] * dimensions[i + 1]);

		// size for biases
		m_network_size += dimensions[i + 1];

		m_weights_size += (dimensions[i] * dimensions[i + 1]);
		m_biases_size += dimensions[i + 1];
	}

	// allocate memory for network
	m_network = (float*)malloc(m_network_size * sizeof(float));
	m_biases = &m_network[m_weights_size];

	m_dimensions = dimensions;
	m_activation_data = std::vector<activation_data>(activations.size());

	for (size_t i = 0; i < activations.size(); i++) {
		m_activation_data[i].type = activations[i];
	}
}

void NeuralNetwork::compile(loss_metric loss, loss_metric metrics, weight_init weight_initialization) {

	switch (loss) {

	}

	switch (metrics) {

	}

	// assign pointers to activation functions and respective derivatives
	for (size_t i = 0; i < m_activation_data.size(); i++) {
		switch (m_activation_data[i].type) {
		case activation_functions::relu:
			m_activation_data[i].activation = &NeuralNetwork::relu;
			m_activation_data[i].derivative = &NeuralNetwork::relu_derivative;
			break;
		case activation_functions::leaky_relu:
			m_activation_data[i].activation = &NeuralNetwork::leaky_relu;
			m_activation_data[i].derivative = &NeuralNetwork::leaky_relu_derivative;
			break;
		case activation_functions::elu:
			m_activation_data[i].activation = &NeuralNetwork::elu;
			m_activation_data[i].derivative = &NeuralNetwork::elu_derivative;
			break;
		case activation_functions::sigmoid:
			m_activation_data[i].activation = &NeuralNetwork::sigmoid;
			m_activation_data[i].derivative = &NeuralNetwork::sigmoid_derivative;
			break;
		}
	}

	float lower_rand;
	float upper_rand;

	int idx = 0;

	std::random_device rd;
	std::mt19937 gen(rd());

	// assign weight values based on init type
	switch (weight_initialization) {

		

	case weight_init::xavier: {
		for (size_t i = 0; i < m_dimensions.size() - 1; i++) {
			lower_rand = -(1.0f / std::sqrt(m_dimensions[i + 1]));
			upper_rand = 1.0f / std::sqrt(m_dimensions[i + 1]);

			std::uniform_real_distribution<float> dist_x(lower_rand, upper_rand);

			for (size_t j = 0; j < m_dimensions[i] * m_dimensions[i + 1]; j++, idx++) {
				m_network[idx] = dist_x(gen);
			}
		}
		break;
	}
	case weight_init::he: {
		lower_rand = 0.0f;

		for (size_t i = 0; i < m_dimensions.size() - 1; i++) {
			upper_rand = std::sqrt(2.0f / m_dimensions[i + 1]);

			std::normal_distribution<float> dist_h(lower_rand, upper_rand);

			for (size_t j = 0; j < m_dimensions[i] * m_dimensions[i + 1]; j++, idx++) {
				m_network[idx] = dist_h(gen);
			}
		}
		break;
	}
	case weight_init::normalize: {
		lower_rand = -0.5f;
		upper_rand = 0.5f;

		for (size_t i = 0; i < m_dimensions.size() - 1; i++) {
			std::uniform_real_distribution<float> dist_n(lower_rand, upper_rand);

			for (size_t j = 0; j < m_dimensions[i] * m_dimensions[i + 1]; j++, idx++) {
				m_network[idx] = dist_n(gen) * std::sqrt(1.0f / m_dimensions[i + 1]);
			}
		}
		break;
	}
	}
}

NeuralNetwork::history NeuralNetwork::fit(Matrix& x_train, Matrix& y_train, Matrix& x_valid, Matrix& y_valid, int batch_size, int epochs, float learning_rate, bool shuffle, int validation_freq) {

	std::cout << "Status: network_training\n";

	history h;

	auto start_time = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> time;

	const int iterations = x_train.RowCount / batch_size;

	initialize_batch_data(batch_size);
	initialize_test_data(x_valid.RowCount);

	// train network
	for (int e = 0; e < epochs; e++) {

		auto epoch_start_time = std::chrono::high_resolution_clock::now();

		for (int i = 0; i < iterations - 1; i++) {

			// adjust pointer to start of data
			float* x = &x_train.matrix[(i * batch_size) * x_train.ColumnCount];

			forward_prop(x, m_batch_data, m_batch_activation_size, batch_size);

			// stupid fix in meantime to prevent dereferencing
			x = &x_train.matrix[(i * batch_size) * x_train.ColumnCount];

			float* y = &y_train.matrix[(i * batch_size) * y_train.ColumnCount];
			
			back_prop(x, y, learning_rate, batch_size);
		}

		if (e % validation_freq == 0) {
			std::string score = test_network(x_valid.matrix, y_valid.matrix, x_valid.RowCount);

			time = std::chrono::high_resolution_clock::now() - epoch_start_time;
			std::cout << "Epoch: " << e << " Time: " << clean_time(time.count()) << " " << score << "\n";
		} else {
			time = std::chrono::high_resolution_clock::now() - epoch_start_time;
			std::cout << "Epoch: " << e << " Time: " << clean_time(time.count()) << "\n";
		}
	}

	free(m_batch_data); m_batch_data = nullptr;
	free(m_test_data); m_test_data = nullptr;

	auto end_time = std::chrono::high_resolution_clock::now();

	std::cout << "Status: training_complete\n";

	time = end_time - start_time;

	std::cout << "average epoch: " << clean_time(time.count() / (double)epochs);

	return h;
}

void NeuralNetwork::initialize_batch_data(int batch_size) {

	m_batch_activation_size = 0;

	// size for dw and db
	m_batch_data_size = m_network_size;
	for (int i = 1; i < m_dimensions.size(); i++) {
		// size for activation, total, and deriv_t
		m_batch_data_size += 3 * (m_dimensions[i] * batch_size);

		m_batch_activation_size += m_dimensions[i] * batch_size;
	}

	// allocate memory for m_batch_data
	m_batch_data = (float*)malloc(m_batch_data_size * sizeof(float));

	m_activation = &m_batch_data[m_batch_activation_size];
	m_deriv_t = &m_activation[m_batch_activation_size];
	m_deriv_w = &m_deriv_t[m_batch_activation_size];
	m_deriv_b = &m_deriv_w[m_batch_activation_size];
 
}
void NeuralNetwork::initialize_test_data(int test_size) {
	int size = 0;
	for (int i = 1; i < m_dimensions.size(); i++) {
		// size for activation and total
		size += 2 * (m_dimensions[i] * test_size);

		m_test_activation_size += m_dimensions[i] * test_size;
	}

	m_test_data = (float*)malloc(size * sizeof(float));
	m_test_activation = &m_test_data[m_test_activation_size];
}

std::string NeuralNetwork::test_network(float* x, float* y, int test_size) {

	forward_prop(x, m_test_data, m_test_activation_size, test_size);

	// compute metric data hardcoded to accuracy for now
	float* last_activation = &m_test_activation[m_test_activation_size - (m_dimensions.back() * test_size)];

	size_t correct = 0;
	for (size_t i = 0; i < test_size; i++) {

		int max_idx = 0;

		// find max value idx in loop
		for (size_t j = 1; j < m_dimensions.back(); j++) {
			if (last_activation[i * m_dimensions.back() + j] > last_activation[i * m_dimensions.back() + max_idx]) {
				max_idx = j;
			}
		}

		if (max_idx == y[i]) {
			correct++;
		}
	}

	return "accuracy: " + std::to_string(((float)correct / test_size) * 100.0f);
}

void NeuralNetwork::forward_prop(float* x_data, float* result_data, int activation_size, int num_elements) {

	int weight_idx = 0;
	int bias_idx = 0;

	int input_idx = 0;
	int output_idx = 0;

	for (int i = 0; i < m_dimensions.size() - 1; i++) {

		// initialize pointers
		float* weights_start = &m_network[weight_idx];
		float* bias_start = &m_biases[bias_idx];

		float* output_start = &result_data[output_idx];

		float* input_start = i == 0 ? &x_data[0] : &result_data[input_idx + activation_size];

		// -> initialize memory to bias values, prevents having to clear existing values
		#pragma omp parallel for
		for (size_t r = 0; r < m_dimensions[i + 1]; r++) {
			std::fill(&output_start[r * num_elements], &output_start[r * num_elements + num_elements], bias_start[r]);
		}

		// -> compute dot prod with weight and input
		i == 0 ? 
			dot_prod_t_b(weights_start, input_start, output_start, m_dimensions[i + 1], m_dimensions[i], num_elements, m_dimensions[i], false) :
			dot_prod(weights_start, input_start, output_start, m_dimensions[i + 1], m_dimensions[i], m_dimensions[i], num_elements, false);

		// -> compute activation
		(this->*m_activation_data[i].activation)(output_start, &output_start[activation_size], m_dimensions[i + 1] * num_elements);

		// update pointers
		weight_idx += m_dimensions[i] * m_dimensions[i + 1];
		bias_idx += m_dimensions[i + 1];

		output_idx += m_dimensions[i + 1] * num_elements;

		input_idx += i == 0 ? 0 : (m_dimensions[i] * num_elements);
	}
}
void NeuralNetwork::back_prop(float* x_data, float* y_data, float learning_rate, int num_elements) {

	const float s_factor = std::sqrt(learning_rate) / std::sqrt(float(num_elements));
	const __m256 _s_factor = _mm256_set1_ps(s_factor);

	// -> compute loss
	// hardcoded to log loss at the moment
	float* last_activation = &m_activation[m_batch_activation_size - (m_dimensions.back() * num_elements)];
	float* last_d_t = &m_deriv_t[m_batch_activation_size - (m_dimensions.back() * num_elements)];

	for (size_t i = 0; i < num_elements; i++) {
		for (size_t j = 0; j < m_dimensions.back(); j++) {
			last_d_t[i * m_dimensions.back() + j] = last_activation[i * m_dimensions.back() + j];
		}
		last_d_t[i * m_dimensions.back() + (int)y_data[i]]--;
	}

	int weight_idx = m_weights_size - (m_dimensions.back() * m_dimensions[m_dimensions.size() - 2]);
	int d_total_idx = m_batch_activation_size - (m_dimensions.back() * num_elements); // -> initialize to last element of dt

	for (size_t i = m_dimensions.size() - 2; i > 0; i--) {

		float* weight = &m_network[weight_idx];

		float* cur_d_total = &m_deriv_t[d_total_idx];
		float* prev_d_total = &m_deriv_t[d_total_idx - (m_dimensions[i] * num_elements)]; // sub offset to previous element of dt

		// -> compute d_total
		// d_total[i - 1] := weight[i].T.dot(d_total[i]) * total[i - 1].activ_derivative
		dot_prod_t_a(weight, cur_d_total, prev_d_total, m_dimensions[i + 1], m_dimensions[i], m_dimensions[i + 1], num_elements, true);

		float* prev_activation = &m_batch_data[d_total_idx - (m_dimensions[i] * num_elements)]; // sub offset to previous element of result total

		// mult by activation derivative
		(this->*m_activation_data[i - 1].derivative)(prev_activation, prev_d_total, m_dimensions[i] * num_elements);
		
		d_total_idx -= m_dimensions[i] * num_elements; // -> move back to previous dt
		weight_idx -= m_dimensions[i] * m_dimensions[i - 1]; // -> move back to previous weight
	}


	int activation_idx = 0;
	int d_weight_idx = 0;
	int d_bias_idx = 0;

	d_total_idx = 0;

	for (size_t i = 0; i < m_dimensions.size() - 1; i++) {

		float* prev_activ = i == 0 ? &x_data[0] : &m_activation[activation_idx];

		float* d_t = &m_deriv_t[d_total_idx];
		float* d_w = &m_deriv_w[d_weight_idx];

		// -> compute d_weights
		// in the casse of x since we don't transpose it in 'fit' we don't transpose it here, hence the ternary operator
		// d_weights[i] := d_total[i].dot(x.T || activation[i - 1].T) * s_factor
		i == 0 ?
			dot_prod(d_t, prev_activ, d_w, m_dimensions[i + 1], num_elements, num_elements, m_dimensions[i], true) :
			dot_prod_t_b(d_t, prev_activ, d_w, m_dimensions[i + 1], num_elements, m_dimensions[i], num_elements, true);

		// multiply by s_factor
		//#pragma omp parallel for
		for (size_t k = 0; k < m_dimensions[i] * m_dimensions[i + 1]; k++) {

			//if (std::_Is_nan(d_w[k])) { std::cout << "d_w[" << k << "]: is_nan (s_factor mult) i == " << i << "\n"; }

			d_w[k] *= s_factor;
		}

		float* d_bias = &m_deriv_b[d_bias_idx];

		// -> compute b_biases
		// d_biases[i] := (d_total[i] * s_factor).row_sums
		#pragma omp parallel for
		for (size_t j = 0; j < m_dimensions[i + 1]; j++) {

			// set value, overwrite old data
			d_bias[j] = d_t[0] * s_factor;

			// rest of loop
			for (size_t k = 1; k < num_elements; k++) {
				d_bias[j] += d_t[k] * s_factor;
			}
		}

		d_bias_idx += m_dimensions[i + 1];
		d_total_idx += m_dimensions[i + 1] * num_elements;
		d_weight_idx += m_dimensions[i] * m_dimensions[i + 1];
		activation_idx += i == 0 ? 0 : (m_dimensions[i] * num_elements);
	}

	//size_t i = 0;

	//// update weights
	//for (; i + 8 <= m_weights_size; i += 8) {
	//	_mm256_store_ps(&m_network[i],
	//		_mm256_fnmadd_ps(
	//			_s_factor,
	//			_mm256_load_ps(&m_deriv_w[i]),
	//			_mm256_load_ps(&m_network[i])
	//		));
	//}

	//for (; i < m_weights_size; i++) {
	//	m_network[i] -= m_deriv_w[i] * s_factor;
	//}

	//i = 0;
	//// update biases
	//for (; i + 8 <= m_biases_size; i += 8) {
	//	_mm256_store_ps(&m_biases[i],
	//		_mm256_fnmadd_ps(
	//			_mm256_load_ps(&m_deriv_b[i]),
	//			_s_factor,
	//			_mm256_load_ps(&m_biases[i])
	//		));
	//}

	//for (; i < m_biases_size; i ++) {
	//	m_biases[i] -= m_deriv_b[i] * s_factor;
	//}

	#pragma omp parallel for
	for (size_t i = 0; i < m_weights_size; i++) {

		//if (std::_Is_nan(m_deriv_w[i])) { std::cout << "d_w[" + std::to_string(i).append("]: is_nan\n"); }
		//if (std::_Is_nan(m_network[i])) { std::cout << "w[" + std::to_string(i).append("]: is_nan\n"); }

		m_network[i] -= m_deriv_w[i] * s_factor;
		//std::cout << (i % 1000 == 0 ? "n_w[" + std::to_string(i).append("]: ").append(std::to_string(m_network[i])).append(" :: d_w[").append(std::to_string(i)).append("]: ").append(std::to_string(m_deriv_w[i])).append(" * ").append(std::to_string(s_factor)).append("\n") : "");
	}

	#pragma omp parallel for 
	for (size_t i = 0; i < m_biases_size; i ++) {
		m_biases[i] -= m_deriv_b[i] * s_factor;
	}
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