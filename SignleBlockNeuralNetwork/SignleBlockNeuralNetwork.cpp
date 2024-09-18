#include "SingleBlockNeuralNetwork.h"
#include <cstdint>
/*

	MEMORY STRUCTURE OF "m_network":

	w0 -> w1 -> w2 ... -> wn ->
	b0 -> b1 -> b2 ... -> bn


	MEMORY STRUCTURE OF "m_batch_data":

	t0 -> t1 -> t2 ... -> tn ->
	a0 -> a1 -> a2 ... -> an ->
	d_t0 -> d_t1 -> d_t2 ... d_tn ->
	d_w0 -> d_w1 -> d_w2 ... d_wn ->
	d_b0 -> b_b1 -> d_b2 ... d_bn

*/

void NeuralNetwork::define(std::vector<int> dimensions) {

	m_network_size = 0;

	for (int i = 0; i < dimensions.size() - 1; i++) {

		// size for weights
		m_network_size += (dimensions[i] * dimensions[i + 1]);

		// size for biases
		m_network_size += dimensions[i + 1];
	}

	// allocate memory for network
	m_network = (float*)malloc(m_network_size * sizeof(float));

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

void NeuralNetwork::fit(Matrix x_train,	Matrix y_train,	Matrix x_valid,	Matrix y_valid,	int batch_size,	int epochs,	float learning_rate, bool shuffle, int validation_freq) {

	std::cout << "Status: network_training\n";

	auto start_time = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> time;

	const int iterations = x_train.RowCount / batch_size;

	// size for d_weights and d_biases
	m_batch_data_size = m_network_size;

	for (int i = 1; i < m_dimensions.size(); i++) {

		// size of d_total + res_total + res_activ
		m_batch_data_size += 3 * (m_dimensions[i] * batch_size);
	}

	// allocate memory for m_batch_data
	m_batch_data = (float*)malloc(m_batch_data_size * sizeof(float));

	for (int e = 0; e < epochs; e++) {

		for (int i = 0; i < iterations; i++) {

			// adjust pointer to start of data
			float* x = &x_train.matrix[(i * batch_size) * x_train.ColumnCount];
			float* y = &y_train.matrix[(i * batch_size) * y_train.ColumnCount];

			forward_prop(x, y, learning_rate);
			back_prop();
		}
		
	}

}

void NeuralNetwork::forward_prop(float* x_data,	float* y_data, float learning_rate) {

	//results.total[i] = net.weights[i].dot_product((i == 0) ? x : results.activation[i - 1]) + net.biases[i];
	//results.activation[i] = (results.total[i].*_activation_functions[i].activation)();

	// total := weights.dot(x || activation[i - 1]) + biases
	// activation := results.activation

	for (int i = 0; i < m_dimensions.size() - 1; i++) {

		/*#pragma omp parallel for
		for (int r = 0; r < RowCount; r++) {
			for (size_t k = 0; k < element.RowCount; k++) {
				__m256 scalar = _mm256_set1_ps(matrix[r * ColumnCount + k]);

				size_t c = 0;
				for (; c + 8 <= element.ColumnCount; c += 8) {

					_mm256_store_ps(&mat.matrix[r * element.ColumnCount + c],
						_mm256_fmadd_ps(_mm256_load_ps(
							&element.matrix[k * element.ColumnCount + c]),
							scalar,
							_mm256_load_ps(&mat.matrix[r * element.ColumnCount + c])));
				}

				for (; c < element.ColumnCount; c++) {
					mat.matrix[r * element.ColumnCount + c] += matrix[r * ColumnCount + k] * element.matrix[k * element.ColumnCount + c];
				}
			}
		}*/

	}
}

int main() {
	NeuralNetwork model;

	model.define({ 784, 128, 128, 10 });

	model.compile(NeuralNetwork::loss_metric::mse, NeuralNetwork::loss_metric::mse, NeuralNetwork::weight_init::he);
}