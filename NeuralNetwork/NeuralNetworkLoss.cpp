#include "NeuralNetwork.h"

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
