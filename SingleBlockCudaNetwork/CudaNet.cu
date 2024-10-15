#include "SingleBlockCudaNetwork.h"

void CudaNetwork::initialize_batch_data(size_t batch_size) {
	m_batch_activation_size = 0;

	m_batch_data_size = m_network_size;

	for (size_t i = 0; i < m_dimensions.size(); i++) {
		m_batch_data_size += 3 * (m_dimensions[i] * batch_size);
		m_batch_activation_size += m_dimensions[i] * batch_size;
	}

	cudaMalloc(&m_batch_data, m_batch_data_size * sizeof(float));

	m_activation = &m_batch_data[m_batch_activation_size];

	m_d_total = &m_activation[m_batch_activation_size];
	m_d_weights = &m_d_total[m_batch_activation_size];
	m_d_bias = &m_d_weights[m_weights_size];

	m_bias = &m_network[m_weights_size];
}
void CudaNetwork::initialize_test_data(size_t test_size) {
	size_t size = 0;

	m_test_activation_size = 0;

	for (size_t i = 1; i < m_dimensions.size(); i++) {
		size += 2 * (m_dimensions[i] * test_size);

		m_test_activation_size += m_dimensions[i] * test_size;
	}

	cudaMalloc(&m_test_data, size * sizeof(float));
	m_test_activation = &m_test_data[m_test_activation_size];
}

void CudaNetwork::define(std::vector<size_t> dimensions) {
	this->m_dimensions = dimensions;

	m_weights_size = 0;
	m_bias_size = 0;
	for (size_t i = 0; i < dimensions.size() - 1; i++) {
		m_weights_size += dimensions[i] * dimensions[i + 1];
		m_bias_size += dimensions[i + 1];
	}
	m_network_size = m_weights_size + m_bias_size;
}
void CudaNetwork::compile(CudaNetwork::weight_init init) {

	float* net = (float*)calloc(m_network_size, sizeof(float));

	std::random_device rd;
	std::default_random_engine gen(rd());

	int idx = 0;
	switch (init) {
	case weight_init::xavier: {
		for (size_t i = 0; i < m_dimensions.size() - 1; i++) {
			float lower_rand = -(1.0f / std::sqrt(m_dimensions[i + 1]));
			float upper_rand = 1.0f / std::sqrt(m_dimensions[i + 1]);

			std::uniform_real_distribution<float> dist_x(lower_rand, upper_rand);

			for (size_t j = 0; j < m_dimensions[i] * m_dimensions[i + 1]; j++, idx++) {
				net[idx] = dist_x(gen);
			}
		}
		break;
	}
	case weight_init::he: {
		float lower_rand = 0.0f;

		for (size_t i = 0; i < m_dimensions.size() - 1; i++) {
			float upper_rand = std::sqrt(2.0f / m_dimensions[i + 1]);

			std::normal_distribution<float> dist_h(lower_rand, upper_rand);

			for (size_t j = 0; j < m_dimensions[i] * m_dimensions[i + 1]; j++, idx++) {
				net[idx] = dist_h(gen);
			}
		}
		break;
	}
	case weight_init::normalize: {
		float lower_rand = -0.5f;
		float upper_rand = 0.5f;

		for (size_t i = 0; i < m_dimensions.size() - 1; i++) {
			std::uniform_real_distribution<float> dist_n(lower_rand, upper_rand);

			for (size_t j = 0; j < m_dimensions[i] * m_dimensions[i + 1]; j++, idx++) {
				net[idx] = dist_n(gen) * std::sqrt(1.0f / m_dimensions[i + 1]);
			}
		}
		break;
	}
	}

	cudaMalloc(&m_network, m_network_size * sizeof(float));
	cudaMemcpy(m_network, net, m_network_size * sizeof(float), cudaMemcpyHostToDevice);

	free(net);

	std::cout << "net compiled\n";
}
 
void CudaNetwork::fit(float* x_train, float* y_train, float* x_valid, float* y_valid, size_t train_samples, size_t test_samples, size_t batch_size, size_t epochs, float learning_rate, bool shuffle, int validation_freq) {
	std::cout << "Status: network_training\n";

	auto start_time = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> time;

	const size_t iterations = train_samples / batch_size;

	float* d_x_train;
	float* d_y_train;

	float* d_x_valid;
	float* d_y_valid;

	initialize_batch_data(batch_size);
	initialize_test_data(test_samples);

	// initialize training data on the gpu
	cudaMalloc(&d_x_train, train_samples * m_dimensions[0] * sizeof(float));
	cudaMalloc(&d_y_train, train_samples * sizeof(float));

	cudaMalloc(&d_x_valid, test_samples * m_dimensions[0] * sizeof(float));
	cudaMalloc(&d_y_valid, test_samples * sizeof(float));


	cudaMemcpy(d_x_train, x_train, train_samples * m_dimensions[0] * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y_train, y_train, train_samples * sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(d_x_valid, x_valid, test_samples * m_dimensions[0] * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y_valid, y_valid, test_samples * sizeof(float), cudaMemcpyHostToDevice);


	for (size_t e = 0; e < epochs; e++) {
		auto epoch_start_time = std::chrono::high_resolution_clock::now();

		for (size_t i = 0; i < iterations; i++) {

			float* x = &d_x_train[(i * batch_size) * m_dimensions[0]];
			float* y = &d_y_train[(i * batch_size) * m_dimensions[0]];

			forward_prop(x, m_batch_data, m_batch_activation_size, batch_size);
			back_prop(x, y, learning_rate, batch_size);
		}

		std::string tmp = "Epoch: " + std::to_string(e).append(" Time: "); int tmp_len = tmp.length();
		if (e % validation_freq == 0) {
			tmp.append(test_network(d_x_valid, d_y_valid, test_samples));
		}
		time = std::chrono::high_resolution_clock::now() - epoch_start_time;
		std::cout << tmp.insert(tmp_len, clean_time(time.count()).append(" ")).append("\n");
	}
	time = std::chrono::high_resolution_clock::now() - start_time;


	cudaFree(m_batch_data);
	cudaFree(m_test_data);

	cudaFree(d_x_train);
	cudaFree(d_y_train);
	cudaFree(d_x_valid);
	cudaFree(d_y_train);

	std::cout << "Status: training_complete\n";
}

void CudaNetwork::forward_prop(float* x_data, float* result_data, size_t activation_size, size_t num_elements) {

	int weight_idx = 0;
	int bias_idx = 0;

	int input_idx = 0;
	int output_idx = 0;

	for (size_t i = 0; i < m_dimensions.size() - 1; i++) {

		dim3 w_grid = (ceil(m_dimensions[i + 1] / 8), ceil(num_elements / 8), 1);
		dim3 b_grid = (ceil(m_dimensions[i + 1] / 8), 1, 1);

		float* weights = &m_network[weight_idx];
		float* bias = &m_bias[bias_idx];

		float* input = i == 0 ? &x_data[0] : &result_data[input_idx + activation_size];
		float* output = &result_data[output_idx];


		i == 0 ?
			dot_prod_t_b << < w_grid, (8, 8) >> > (weights, input, output, m_dimensions[i + 1], m_dimensions[i], num_elements, m_dimensions[i]) :
			dot_prod << < w_grid, (8, 8) >> > (weights, input, output, m_dimensions[i + 1], m_dimensions[i], m_dimensions[i], num_elements);
		cudaDeviceSynchronize();

		// add bias
		horizontal_add << <ceil(m_dimensions[i + 1] / 8), 8 >> > (output, bias, m_dimensions[i + 1], num_elements);
		cudaDeviceSynchronize();

		// activation
		leaky_relu << <w_grid, (8, 8) >> > (output, &output[activation_size], m_dimensions[i + 1], num_elements);
		cudaDeviceSynchronize();

		weight_idx += m_dimensions[i] * m_dimensions[i + 1];
		bias_idx += m_dimensions[i + 1];

		input_idx += i == 0 ? 0 : m_dimensions[i] * num_elements;
		output_idx += m_dimensions[i + 1] * num_elements;
	}
}
void CudaNetwork::back_prop(float* x_data, float* y_data, float learning_rate, size_t num_elements) {

	const float factor = learning_rate / (float)num_elements;


	// -> compute loss
	{
		float* last_d_total = &m_d_total[m_batch_activation_size - (m_dimensions.back() * num_elements)];
		float* last_activation = &m_activation[m_batch_activation_size - (m_dimensions.back() * num_elements)];

		one_hot_loss <<< ceil(num_elements / 8), 8 >> >(last_d_total, last_activation, y_data, m_dimensions.back(), num_elements);
		cudaDeviceSynchronize();
	}


	// -> compute d_total
	{
		int weight_idx = m_weights_size - (m_dimensions.back() * m_dimensions[m_dimensions.size() - 2]);
		int d_total_idx = m_batch_activation_size - (m_dimensions.back() * num_elements);

		for (size_t i = m_dimensions.size() - 2; i > 0; i--) {

			float* weight = &m_network[weight_idx];
			float* prev_total = &m_batch_data[d_total_idx - (m_dimensions[i] * num_elements)];

			float* cur_d_total = &m_d_total[d_total_idx];
			float* prev_d_total = &m_d_total[d_total_idx - (m_dimensions[i] * num_elements)];

			dot_prod_t_a << <(ceil(m_dimensions[i + 1] / 8), ceil(num_elements / 8)), (8, 8) >> > (weight, cur_d_total, prev_d_total, m_dimensions[i + 1], m_dimensions[i], m_dimensions[i + 1], num_elements);
			cudaDeviceSynchronize();

			// multiply by activation function derivative
			leaky_relu_derivative << <(ceil(m_dimensions[i + 1] / 8), ceil(num_elements / 8)), (8, 8) >> > (prev_total, prev_d_total, m_dimensions[i], num_elements);
			cudaDeviceSynchronize();

			d_total_idx -= m_dimensions[i] * num_elements;
			weight_idx -= m_dimensions[i] * m_dimensions[i - 1];
		}
	}

	
	// -> compute d_weights and d_biases
	{
		int activation_idx = 0;

		int d_total_idx = 0;
		int d_weights_idx = 0;
		int d_bias_idx = 0;

		for (size_t i = 0; i < m_dimensions.size() - 1; i++) {
			dim3 w_grid(ceil(m_dimensions[i + 1] / 8), ceil(m_dimensions[i] / 8), 1);

			float* prev_activation = i == 0 ? &x_data[0] : &m_activation[activation_idx];

			float* d_total = &m_d_total[d_total_idx];
			float* d_weights = &m_d_weights[d_weights_idx];
			float* d_bias = &m_d_bias[d_bias_idx];

			// d_weights
			i == 0 ?
				dot_prod << < w_grid, (8, 8) >> > (d_total, prev_activation, d_weights, m_dimensions[i + 1], num_elements, num_elements, m_dimensions[i]) :
				dot_prod_t_b << < w_grid, (8, 8) >> > (d_total, prev_activation, d_weights, m_dimensions[i + 1], num_elements, m_dimensions[i], num_elements);
			cudaDeviceSynchronize();

			// d_biases
			horizontal_sum << < ceil(m_dimensions[i + 1] / 8), 8 >> > (d_total, d_bias, m_dimensions[i + 1], num_elements);
			cudaDeviceSynchronize();

			d_bias_idx += m_dimensions[i + 1];
			d_total_idx += m_dimensions[i + 1] * num_elements;
			d_weights_idx += m_dimensions[i] * m_dimensions[i + 1];
			activation_idx += i == 0 ? 0 : (m_dimensions[i] * num_elements);
		}
	}

	// update weights and biases
	{
		update_weights << < ceil(m_weights_size / 8), 8 >> > (m_network, m_d_weights, factor, m_weights_size);
		update_bias << < ceil(m_bias_size / 8), 8 >> > (m_bias, m_d_bias, factor, m_bias_size);
	}
}

std::string CudaNetwork::test_network(float* x, float* y, size_t test_size) {

	forward_prop(x, m_test_data, m_test_activation_size, test_size);

	int* d_correct;
	int correct;

	cudaMalloc(&d_correct, sizeof(int));

	accuracy_score << <(ceil(test_size / 8)), 8 >> > (&m_test_activation[m_test_activation_size - (m_dimensions.back() * test_size)], y, d_correct, m_dimensions.back(), test_size);
	cudaDeviceSynchronize();

	cudaMemcpy(&correct, d_correct, sizeof(int), cudaMemcpyDeviceToHost);

	return "score: " + std::to_string((float)correct / (float)test_size * 100.0f);
}

std::string CudaNetwork::clean_time(double time) {
	const double hour = 3600000.00;
	const double minute = 60000.00;
	const double second = 1000.00;
	std::string out;

	if (time / hour > 1.00) {
		out = std::to_string(time / hour).append(" hours");
	} else if (time / minute > 1.00) {
		out = std::to_string(time / minute).append(" minutes");
	} else if (time / second > 1.00) {
		out = std::to_string(time / second).append(" seconds");
	} else {
		out = std::to_string(time).append("(ms)");
	}

	return out;
}