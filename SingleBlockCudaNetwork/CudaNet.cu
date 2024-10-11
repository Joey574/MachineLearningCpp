#include "SingleBlockCudaNetwork.h"

#include "CudaNetActivations.cu"
#include "CudaNetDotProds.cu"

__global__ void CudaNetwork::horizontal_add(float* a, float* b, size_t a_r, size_t a_c) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < a_r) {
		for (size_t j = 0; j < a_c; j++) {
			a[i * a_r + j] += b[i];
		}
	}
}
__global__ void CudaNetwork::horizontal_sum(float* a, float* b, size_t a_r, size_t a_c) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < a_r) {
		b[i] = a[i * a_c];

		for (size_t j = 1; j < a_c; j++) {
			b[i] += a[i * a_c + j];
		}
	}
}

__global__ void CudaNetwork::update_weights(float* a, float* b, float lr, size_t n) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < n) {
		b[i] -= a[i] * lr;
	}
}
__global__ void CudaNetwork::update_bias(float* a, float* b, float lr, size_t n) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < n) {
		b[i] -= a[i] * lr;
	}
}

__global__ void CudaNetwork::log_loss(float* a, float* b, float* y, size_t a_r, size_t a_c) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < a_c) {
		for (int j = 0; j < a_r; j++) {
			b[i * a_r + j] = a[i * a_r + j];
		}
		b[i * a_r + (int)y[i]]--;
	}
}

void CudaNetwork::fit(float* x_train, float* y_train, float* x_valid, float* y_valid, int num_elements, int batch_size, int epochs, float learning_rate, bool shuffle, int validation_freq) {
	std::cout << "Status: network_training\n";

	auto start_time = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> time;

	const int iterations = num_elements / batch_size;

	float* d_x_train;
	float* d_y_train;

	float* d_x_valid;
	float* d_y_valid;




	for (int e = 0; e < epochs; e++) {
		auto epoch_start_time = std::chrono::high_resolution_clock::now();

		for (int i = 0; i < iterations; i++) {

			float* x = &d_x_train[(i * batch_size)*];



		}

		std::string tmp = "Epoch: " + std::to_string(e).append("Time: "); int tmp_len = tmp.length();
		if (e % validation_freq == 0) {
			tmp.append(test_network(nullptr, nullptr, 0));
		}
		time = std::chrono::high_resolution_clock::now() - epoch_start_time;
		std::cout << tmp.insert(tmp_len - 1, clean_time(time.count()).append("\n"));
	}

	time = std::chrono::high_resolution_clock::now() - start_time;
	std::cout << "Status: training_complete\nAverage Epoch: " << clean_time(time.count() / (double)epochs);


}

void CudaNetwork::forward_prop(float* x_data, float* result_data, int activation_size, int num_elements) {

	const dim3 w_block(8, 8, 1);
	const dim3 b_block(8, 1, 1);
	dim3 w_grid;
	dim3 b_grid;

	int weight_idx = 0;
	int bias_idx = 0;

	int input_idx = 0;
	int output_idx = 0;

	for (size_t i = 0; i < m_dimensions.size() - 1; i++) {

		w_grid = (ceil(m_dimensions[i + 1] / 8), ceil(num_elements / 8), 1);
		b_grid = (ceil(m_dimensions[i + 1] / 8), 1, 1);

		float* weights = &m_network[weight_idx];
		float* bias = &m_bias[bias_idx];

		float* input = i == 0 ? &x_data[0] : &result_data[input_idx + activation_size];
		float* output = &result_data[output_idx];


		i == 0 ?
			dot_prod_t_b << < w_grid, w_block >> > (weights, input, output, m_dimensions[i + 1], m_dimensions[i], num_elements, m_dimensions[i]) :
			dot_prod << < w_grid, w_block >> > (weights, input, output, m_dimensions[i + 1], m_dimensions[i], m_dimensions[i], num_elements);

		// add bias
		horizontal_add << <b_grid, b_block >> > (output, bias, m_dimensions[i + 1], num_elements);

		// activation
		leaky_relu << <w_grid, w_block >> > (output, &output[activation_size], m_dimensions[i + 1], num_elements);

		weight_idx += m_dimensions[i] * m_dimensions[i + 1];
		bias_idx += m_dimensions[i + 1];

		input_idx += i == 0 ? 0 : m_dimensions[i] * num_elements;
		output_idx += m_dimensions[i + 1] * num_elements;
	}
}
void CudaNetwork::back_prop(float* x_data, float* y_data, float learning_rate, int num_elements) {

	// -> compute loss
	{
		dim3 block(8, 1, 1);
		dim3 grid(ceil(num_elements / 8), 1, 1);

		float* last_d_total = &m_d_total[m_batch_activation_size - (m_dimensions.back() * num_elements)];
		float* last_activation = &m_activation[m_batch_activation_size - (m_dimensions.back() * num_elements)];

		log_loss(last_d_total, last_activation, y_data, m_dimensions.back(), num_elements);
	}

	// -> compute d_total
	{
		dim3 block(8, 8, 1);

		int weight_idx = m_weights_size - (m_dimensions.back() * m_dimensions[m_dimensions.size() - 2]);
		int d_total_idx = m_batch_activation_size - (m_dimensions.back() * num_elements);

		for (size_t i = m_dimensions.size() - 2; i > 0; i--) {
			dim3 grid(ceil(m_dimensions[i + 1] / 8), ceil(num_elements / 8), 1);

			float* weight = &m_network[weight_idx];
			float* prev_total = &m_batch_data[d_total_idx - (m_dimensions[i] * num_elements)];

			float* cur_d_total = &m_d_total[d_total_idx];
			float* prev_d_total = &m_d_total[d_total_idx - (m_dimensions[i] * num_elements)];


			dot_prod_t_a << <grid, block >> > (weight, cur_d_total, prev_d_total, m_dimensions[i + 1], m_dimensions[i], m_dimensions[i + 1], num_elements);

			// add biases

			// multiply by activation function derivative
			leaky_relu_derivative << <grid, block >> > (prev_total, prev_d_total, m_dimensions[i], num_elements);

			d_total_idx -= m_dimensions[i] * num_elements;
			weight_idx -= m_dimensions[i] * m_dimensions[i - 1];
		}
	}

	
	// -> compute d_weights and d_biases
	{
		dim3 w_block(8, 8, 1);
		dim3 b_block(8, 1, 1);

		int activation_idx = 0;

		int d_total_idx = 0;
		int d_weights_idx = 0;
		int d_bias_idx = 0;


		for (size_t i = 0; i < m_dimensions.size() - 1; i++) {
			dim3 w_grid(ceil(m_dimensions[i + 1] / 8), ceil(m_dimensions[i] / 8), 1);
			dim3 b_grid(ceil(m_dimensions[i + 1] / 8), 1, 1);

			float* prev_activation = i == 0 ? &x_data[0] : &m_activation[activation_idx];

			float* d_total = &m_d_total[d_total_idx];
			float* d_weights = &m_d_weights[d_weights_idx];
			float* d_bias = &m_d_bias[d_bias_idx];

			// d_weights
			i == 0 ?
				dot_prod << < w_grid, w_block >> > (d_total, prev_activation, d_weights, m_dimensions[i + 1], num_elements, num_elements, m_dimensions[i]) :
				dot_prod_t_b << < w_grid, w_block >> > (d_total, prev_activation, d_weights, m_dimensions[i + 1], num_elements, m_dimensions[i], num_elements);

			// d_biases
			horizontal_sum << < b_grid, b_block >> > (d_total, d_bias, m_dimensions[i + 1], num_elements);

			d_bias_idx += m_dimensions[i + 1];
			d_total_idx += m_dimensions[i + 1] * num_elements;
			d_weights_idx += m_dimensions[i] * m_dimensions[i + 1];
			activation_idx += i == 0 ? 0 : (m_dimensions[i] * num_elements);
		}
	}

	// update weights and biases
	{
		dim3 block(8, 1, 1);
		dim3 grid(ceil(m_weights_size / 8), 1, 1);

		update_weights << < grid, block >> > (m_d_weights, m_network, m_weights_size);

		grid = (ceil(m_bias_size / 8), 1, 1);
		update_bias << < grid, block >> > (m_d_bias, m_bias, m_bias_size);
	}
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