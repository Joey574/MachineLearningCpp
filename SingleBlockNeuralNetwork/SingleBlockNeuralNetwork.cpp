#include "SingleBlockNeuralNetwork.h"
#include <cstdint>

void NeuralNetwork::define(std::vector<int> dimensions) {

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
}

void NeuralNetwork::compile(loss_metric loss, loss_metric metrics, weight_init weight_initialization) {

	switch (loss) {

	}

	switch (metrics) {

	}

	float lower_rand;
	float upper_rand;

	int idx = 0;

	std::random_device rd;
	std::mt19937 gen(rd());

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

NeuralNetwork::history NeuralNetwork::fit(Matrix x_train, Matrix y_train, Matrix x_valid, Matrix y_valid, int batch_size, int epochs, float learning_rate, bool shuffle, int validation_freq) {

	std::cout << "Status: network_training\n";

	history h;

	auto start_time = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> time;

	const int iterations = x_train.RowCount / batch_size;

	initialize_batch_data(batch_size);

	// train network
	for (int e = 0; e < epochs; e++) {

		auto epoch_start_time = std::chrono::high_resolution_clock::now();

		for (int i = 0; i < iterations; i++) {

			// adjust pointer to start of data
			float* x = &x_train.matrix[(i * batch_size) * x_train.ColumnCount];
			float* y = &y_train.matrix[(i * batch_size) * y_train.ColumnCount];

			forward_prop(x, learning_rate, batch_size);
			back_prop(x, y, learning_rate, batch_size);
		}

		if (e % validation_freq == 0) {
			std::string score = test_network(x_valid.matrix, y_valid.matrix);

			time = std::chrono::high_resolution_clock::now() - epoch_start_time;
			std::cout << "Epoch: " << e << " Time: " << clean_time(time.count()) << " " << score << "\n";
		} else {
			time = std::chrono::high_resolution_clock::now() - epoch_start_time;
			std::cout << "Epoch: " << e << " Time: " << clean_time(time.count()) << "\n";
		}

	}
	auto end_time = std::chrono::high_resolution_clock::now();

	free(m_batch_data);

	std::cout << "Status: training_complete\n";

	return h;
}

void NeuralNetwork::initialize_batch_data(int batch_size) {

	m_r_total_size = 0;

	// size for dw and db
	m_batch_data_size = m_network_size;
	for (int i = 1; i < m_dimensions.size(); i++) {
		// size for activation, total, and deriv_t
		m_batch_data_size += 3 * (m_dimensions[i] * batch_size);

		m_r_total_size += m_dimensions[i] * batch_size;
	}

	// allocate memory for m_batch_data
	m_batch_data = (float*)malloc(m_batch_data_size * sizeof(float));

	m_activation = &m_batch_data[m_r_total_size];
	m_deriv_t = &m_activation[m_r_total_size];
	m_deriv_w = &m_deriv_t[m_r_total_size];
	m_deriv_b = &m_deriv_w[m_weights_size];
 
}

std::string NeuralNetwork::test_network(float* x, float* y) {
	return "";
}

void NeuralNetwork::forward_prop(float* x_data, float learning_rate, int batch_size) {

	int weight_idx = 0;
	int bias_idx = 0;

	int input_idx = 0;
	int output_idx = 0;

	for (int i = 0; i < m_dimensions.size() - 1; i++) {

		float* weights_start = &m_network[weight_idx];
		float* bias_start = &m_biases[bias_idx];

		float* output_start = &m_batch_data[output_idx];

		float* input_start = i == 0 ? x_data : &m_batch_data[input_idx + m_r_total_size];

		// -> initialize memory to bias values, prevents having to clear later
		for (size_t r = 0; r < m_dimensions[i + 1]; r++) {
			for (size_t c = 0; c < batch_size; c++) {
				output_start[r * batch_size + c] = bias_start[r];
			}
		}

		// -> compute dot prod with weight and input
		#pragma omp parallel for
		for (size_t r = 0; r < m_dimensions[i + 1]; r++) {
			for (size_t k = 0; k < m_dimensions[i]; k++) {
				__m256 scalar = _mm256_set1_ps(weights_start[r * m_dimensions[i] + k]);

				size_t c = 0;
				for (; c + 8 <= batch_size; c += 8) {

					_mm256_store_ps(&output_start[r * batch_size + c],
						_mm256_fmadd_ps(_mm256_load_ps(
							&input_start[k * batch_size + c]),
							scalar,
							_mm256_load_ps(&output_start[r * batch_size + c])));
				}

				for (; c < batch_size; c++) {
					output_start[r * batch_size + c] += weights_start[r * m_dimensions[i] + k] * input_start[k * batch_size + c];
				}
			}
		}

		// -> compute activation hardcoded to leaky_relu at the moement
		#pragma omp parallel for
		for (size_t r = 0; r < m_dimensions[i + 1]; r++) {
			for (size_t c = 0; c < batch_size; c++) {
				output_start[m_r_total_size + (r * batch_size + c)] = output_start[m_r_total_size + (r * batch_size + c)] > 0.0f ? output_start[m_r_total_size + (r * batch_size + c)] : 0.1f * output_start[m_r_total_size + (r * batch_size + c)];
			}
		}

		weight_idx += m_dimensions[i] * m_dimensions[i + 1];
		bias_idx += m_dimensions[i + 1];

		output_idx += m_dimensions[i + 1] * batch_size;

		input_idx += i == 0 ? 0 : m_dimensions[i] * batch_size;
	}
}

void NeuralNetwork::back_prop(float* x_data, float* y_data,	float learning_rate, int batch_size) {

	const float s_factor = std::sqrt(learning_rate) / std::sqrt(float(batch_size));
	const __m256 _s_factor = _mm256_set1_ps(s_factor);

	// compute loss

	// 784 128 128 10
		// weight shape:
		// w0: 128 x 784
		// w1: 128 x 128
		// w2: 10 x 128

		// input shape:
		// x0: 784 x 320
		// x1: 128 x 320 = a0
		// x2: 128 x 320 = a1

		// output shape:
		// t0: 128 x 320
		// t1: 128 x 320
		// t2: 10 x 320

		// a0: 128 x 320
		// a1: 128 x 320
		// a2: 10 x 320

	// d_total[i] := weight.T.dot(d_total[i + 1]) * total[i].activ_derivative
	for (size_t i = m_dimensions.size() - 2; i > 0; i--) {

		float* weight = &m_network[0];

		float* prev_d_total = &m_batch_data[0];
		float* cur_d_total = &m_batch_data[0];

		float* prev_activation = &m_batch_data[0];

		// deriv.d_total[i - 1] = net.weights[i].Transpose().dot_product(deriv.d_total[i]) * (results.total[i - 1].*_activation_functions[i - 1].derivative)();
		for (size_t j = 0; j < m_dimensions[i]; j++) { // -> rowc
			for (size_t k = 0; k < m_dimensions[i]; k++) { // ele.rowc

				__m256 _scalar;

				size_t l = 0;
				for (; l + 8 <= batch_size; l += 8) {
					
				}

				for (; l < batch_size; l++) {

				}
			}
		}
	}

	// d_weights[i] := d_total[i].dot(x.T || activation[i - 1].T) * s_factor
	// d_biases[i] := (d_total[i] * s_factor).row_sums


	size_t i = 0;

	// update weights
	for (; i + 8 <= m_weights_size; i += 8) {
		_mm256_store_ps(&m_network[i],
			_mm256_fnmadd_ps(
				_s_factor,
				_mm256_load_ps(&m_deriv_w[i]),
				_mm256_load_ps(&m_network[i])
			));
	}

	for (; i < m_weights_size; i++) {
		m_network[i] -= m_deriv_w[i] * s_factor;
	}

	i = 0;

	// update biases
	for (; i + 8 <= m_biases_size; i += 8) {
		_mm256_store_ps(&m_biases[i],
			_mm256_fnmadd_ps(
				_mm256_load_ps(&m_deriv_b[i]),
				_s_factor,
				_mm256_load_ps(&m_biases[i])
			));
	}

	for (; i < m_biases_size; i ++) {
		m_biases[i] -= m_deriv_b[i] * s_factor;
	}

	/*#pragma omp parallel for
	for (size_t i = 0; i < m_weights_size; i++) {
		m_network[i] -= m_deriv_w[i] * s_factor;
	}

	#pragma omp parallel for 
	for (size_t i = 0; i < m_biases_size; i ++) {
		m_biases[i] -= m_deriv_b[i] * s_factor;
	}*/
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