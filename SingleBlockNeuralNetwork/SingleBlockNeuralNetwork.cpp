	#include "SingleBlockNeuralNetwork.h"

	void NeuralNetwork::define(std::vector<size_t> dimensions, std::vector<activation_functions> activations) {

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
		m_network = (float*)_aligned_malloc(m_network_size * sizeof(float), 64);


		m_dimensions = dimensions;
		m_activation_data = std::vector<activation_data>(activations.size());

		for (size_t i = 0; i < activations.size(); i++) {
			m_activation_data[i].type = activations[i];
		}
	}
	void NeuralNetwork::compile(loss_metric loss, loss_metric metrics, weight_init weight_initialization) {

		// assign pointers to loss functions and metrics
		switch (loss) {
		case loss_metric::mae:
			m_loss = &NeuralNetwork::mae_loss;
			break;
		case loss_metric::one_hot:
			m_loss = &NeuralNetwork::one_hot_loss;
			break;
		default:
			std::cout << "ERROR: LOSS NOT FOUND\n";
			break;
		}

		switch (metrics) {
		case loss_metric::mae:
			m_metric = &NeuralNetwork::mae_score;
			break;
		case loss_metric::accuracy:
			m_metric = &NeuralNetwork::accuracy_score;
			break;
		default:
			std::cout << "ERROR: METRIC NOT FOUND\n";
			break;
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
			case activation_functions::softmax:
				m_activation_data[i].activation = &NeuralNetwork::softmax;
				m_activation_data[i].derivative = &NeuralNetwork::sigmoid_derivative;
			default:
				std::cout << "ERROR ACTIVATION FUNCTION NOT FOUND\n";
				break;
			}
		}

		float lower_rand;
		float upper_rand;

		int idx = 0;

		std::random_device rd;
		std::default_random_engine gen(rd());

		// assign weight values based on init type
		if (!loaded) {
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
	
		std::cout << this->summary();
	}

	NeuralNetwork::history NeuralNetwork::fit(Matrix x_train, Matrix y_train, Matrix x_valid, Matrix y_valid, size_t batch_size, size_t epochs, float learning_rate, bool shuffle, int validation_freq, float validation_split) {

		m_biases = &m_network[m_weights_size];

		std::cout << "Status: network_training\n";

		history h;
		auto start_time = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> time;

		// do first
		data_preprocess(x_train, y_train, x_valid, y_valid, validation_split, shuffle);

		const int iterations = x_train.RowCount / batch_size;
		initialize_batch_data(batch_size);
		initialize_test_data(x_valid.RowCount);

		// train network
		for (int e = 0; e < epochs; e++) {

			auto epoch_start_time = std::chrono::high_resolution_clock::now();

			for (int i = 0; i < iterations - 1; i++) {

				// adjust pointer to start of data
				float* x = &x_train.matrix[(i * batch_size) * x_train.ColumnCount];
				float* y = &y_train.matrix[(i * batch_size) * y_train.ColumnCount];

				forward_prop(x, m_batch_data, m_batch_activation_size, batch_size);

				// stupid fix in meantime to prevent dereferencing
				x = &x_train.matrix[(i * batch_size) * x_train.ColumnCount];
			
				back_prop(x, y, learning_rate, batch_size);
			}

			std::string tmp = "Epoch: " + std::to_string(e).append(" Time: "); int tmp_len = tmp.length();
			if (e % validation_freq == 0) {
				tmp.append(test_network(x_valid.matrix, y_valid.matrix, x_valid.RowCount, h));
			}
			time = std::chrono::high_resolution_clock::now() - epoch_start_time;
			std::cout << tmp.insert(tmp_len, clean_time(time.count()).append(" ")).append("\n");

		}
		time = std::chrono::high_resolution_clock::now() - start_time;

		_aligned_free(m_batch_data); m_batch_data = nullptr;
		_aligned_free(m_test_data); m_test_data = nullptr;

		h.epoch_time = (time / (double)epochs);
		h.train_time = time;

		std::cout << "Status: training_complete\n";

		return h;
	}

	void NeuralNetwork::data_preprocess(Matrix& x_train, Matrix& y_train, Matrix& x_valid, Matrix& y_valid, float validation_split, bool shuffle) {
		if (shuffle) {
			for (int k = 0; k < x_train.RowCount; k++) {

				int r = k + rand() % (x_train.RowCount - k);

				std::vector<float> tempX = x_train.Row(k);
				std::vector<float> tempY = y_train.Row(k);

				x_train.SetRow(k, x_train.Row(r));
				y_train.SetRow(k, y_train.Row(r));

				x_train.SetRow(r, tempX);
				y_train.SetRow(r, tempY);
			}
		}

		if (validation_split > 0.0f && x_valid.RowCount == 0 && y_valid.RowCount == 0) {

			int elements = (float)x_train.RowCount * validation_split;

			x_valid = x_train.SegmentR(x_train.RowCount - elements);
			y_valid = y_train.SegmentR(y_train.RowCount - elements);

			x_train = x_train.SegmentR(0, x_train.RowCount - elements);
			y_train = y_train.SegmentR(0, y_train.RowCount - elements);

		}
	}

	void NeuralNetwork::initialize_batch_data(size_t batch_size) {
		m_batch_activation_size = 0;

		for (int i = 1; i < m_dimensions.size(); i++) {
			m_batch_activation_size += m_dimensions[i] * batch_size;
		}
		m_batch_data_size = (3 * m_batch_activation_size) + m_network_size;

		// allocate memory for m_batch_data
		m_batch_data = (float*)_aligned_malloc(m_batch_data_size * sizeof(float), 64);

		m_activation = &m_batch_data[m_batch_activation_size];

		m_d_total = &m_activation[m_batch_activation_size];
		m_d_weights = &m_d_total[m_batch_activation_size];
		m_d_biases = &m_d_weights[m_weights_size];
	}
	void NeuralNetwork::initialize_test_data(size_t test_size) {
		m_test_activation_size = 0;

		for (int i = 1; i < m_dimensions.size(); i++) {
			m_test_activation_size += m_dimensions[i] * test_size;
		}

		m_test_data = (float*)_aligned_malloc(m_test_activation_size * 2 * sizeof(float), 64);

		m_test_activation = &m_test_data[m_test_activation_size];
	}

	std::string NeuralNetwork::test_network(float* x, float* y, size_t test_size, history& h) {

		forward_prop(x, m_test_data, m_test_activation_size, test_size);

		float* last_activation = &m_test_activation[m_test_activation_size - (m_dimensions.back() * test_size)];

		// compute metric data 
		float score = (this->*m_metric)(last_activation, y, m_dimensions.back(), test_size);

		h.metric_history.push_back(score);
		return "score: " + std::to_string(score);
	}

	void NeuralNetwork::forward_prop(float* x_data, float* result_data, size_t activation_size, size_t num_elements) {

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

			// initialize memory to bias values, prevents having to clear existing values
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
	void NeuralNetwork::back_prop(float* x_data, float* y_data, float learning_rate, size_t num_elements) {

		const float factor = learning_rate / (float)num_elements;
		const __m256 _factor = _mm256_set1_ps(factor);

		// d_total[i - 1] := weight[i].T.dot(d_total[i]) * total[i - 1].activ_derivative
		// d_weights[i] := d_total[i].dot(x || activation[i - 1].T)
		// d_biases[i] := d_total[i].row_sums

		float* last_activation = &m_activation[m_batch_activation_size - (m_dimensions.back() * num_elements)];
		float* last_d_total = &m_d_total[m_batch_activation_size - (m_dimensions.back() * num_elements)];

		// -> compute loss
		(this->*m_loss)(last_activation, y_data, last_d_total, m_dimensions.back(), num_elements);

		int weight_idx = m_weights_size - (m_dimensions.back() * m_dimensions[m_dimensions.size() - 2]); // -> initialize to last weight
		int d_total_idx = m_batch_activation_size - (m_dimensions.back() * num_elements); // -> initialize to last element of dt

		// -> compute d_total
		for (size_t i = m_dimensions.size() - 2; i > 0; i--) {

			float* weight = &m_network[weight_idx];
			float* prev_total = &m_batch_data[d_total_idx - (m_dimensions[i] * num_elements)];

			float* cur_d_total = &m_d_total[d_total_idx];
			float* prev_d_total = &m_d_total[d_total_idx - (m_dimensions[i] * num_elements)];

			dot_prod_t_a(weight, cur_d_total, prev_d_total, m_dimensions[i + 1], m_dimensions[i], m_dimensions[i + 1], num_elements, true);

			// mult by activation derivative
			(this->*m_activation_data[i - 1].derivative)(prev_total, prev_d_total, m_dimensions[i] * num_elements);
		
			d_total_idx -= m_dimensions[i] * num_elements;
			weight_idx -= m_dimensions[i] * m_dimensions[i - 1];
		}

		int activation_idx = 0;
		int d_weight_idx = 0;
		int d_bias_idx = 0;

		d_total_idx = 0;

		// -> compute d_weights
		for (size_t i = 0; i < m_dimensions.size() - 1; i++) {

			float* prev_activ = i == 0 ? &x_data[0] : &m_activation[activation_idx];

			float* d_total = &m_d_total[d_total_idx];
			float* d_weights = &m_d_weights[d_weight_idx];
			float* d_bias = &m_d_biases[d_bias_idx];

			i == 0 ?
				dot_prod(d_total, prev_activ, d_weights, m_dimensions[i + 1], num_elements, num_elements, m_dimensions[i], true) :
				dot_prod_t_b(d_total, prev_activ, d_weights, m_dimensions[i + 1], num_elements, m_dimensions[i], num_elements, true);

			// -> compute d_biases
			#pragma omp parallel for
			for (size_t j = 0; j < m_dimensions[i + 1]; j++) {
				__m256 sum = _mm256_setzero_ps();

				size_t k = 0;
				for (; k <= num_elements - 8; k += 8) {
					sum = _mm256_add_ps(sum, _mm256_load_ps(&d_total[j * num_elements + k]));
				}

				float t[8];
				_mm256_store_ps(t, sum);
				d_bias[j] = t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];

				for (; k < num_elements; k++) {
					d_bias[j] += d_total[j * num_elements + k];
				}
			}

			d_bias_idx += m_dimensions[i + 1];
			d_total_idx += m_dimensions[i + 1] * num_elements;
			d_weight_idx += m_dimensions[i] * m_dimensions[i + 1];
			activation_idx += i == 0 ? 0 : (m_dimensions[i] * num_elements);
		}


		// update weights
		#pragma omp parallel for
		for (size_t i = 0; i <= m_weights_size - 8; i += 8) {
			_mm256_store_ps(&m_network[i], 
				_mm256_fnmadd_ps(
					_mm256_load_ps(&m_d_weights[i]),
					_factor,
					_mm256_load_ps(&m_network[i])
				));
		}

		for (size_t i = m_weights_size - (m_weights_size % 8); i < m_weights_size; i++) {
			m_network[i] -= m_d_weights[i] * factor;
		}


		// update biases
		#pragma omp parallel for
		for (size_t i = 0; i <= m_biases_size - 8; i += 8) {
			_mm256_store_ps(&m_biases[i],
				_mm256_fnmadd_ps(
					_mm256_load_ps(&m_d_biases[i]),
					_factor,
					_mm256_load_ps(&m_biases[i])
				));
		}

		for (size_t i = m_biases_size - (m_biases_size % 8); i < m_biases_size; i ++) {
			m_biases[i] -= m_d_biases[i] * factor;
		}
	}

	std::vector<float> NeuralNetwork::predict(const Matrix& x) {

		int activation_size = 0;
		int malloc_size = 0;

		for (size_t i = 1; i < m_dimensions.size(); i++) {
			activation_size += m_dimensions[i] * x.RowCount;
			malloc_size += 2 * m_dimensions[i] * x.RowCount;
		}
		float* results = (float*)_aligned_malloc(malloc_size * sizeof(float), 64);

		forward_prop(x.matrix, results, activation_size, x.RowCount);

		std::vector<float> predictions(&results[malloc_size - (m_dimensions.back() * x.RowCount)], &results[malloc_size]);

		_aligned_free(results);

		return predictions;
	}