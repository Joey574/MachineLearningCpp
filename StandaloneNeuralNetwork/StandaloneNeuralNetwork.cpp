#include "StandaloneNeuralNetwork.h"

void StandaloneNeuralNetwork::Define(std::vector<int> dimensions, std::vector<activation_function> activation_functions) {

	this->_dimensions = dimensions;

	// probably right
	int size = 0;
	for (int i = 0; i < dimensions.size() - 1; i++) {
		size += dimensions[i] * dimensions[i + 1];
	}
	size += std::accumulate(dimensions.begin() + 1, dimensions.end(), 0);

	this->_net_size = size;

	_net = (float*)malloc(size * sizeof(float));
}

void StandaloneNeuralNetwork::Compile(loss_metric loss = loss_metric::none,	loss_metric metrics = loss_metric::none, 
	optimization_technique optimizer = optimization_technique::none) {

}

void StandaloneNeuralNetwork::forward_propogate() {
	/*for (int i = 0; i < results.total.size(); i++) {
		results.total[i] = net.weights[i].dot_product_add(((i == 0) ? x : results.activation[i - 1]), net.biases[i]);
		results.activation[i] = (results.total[i].*_activation_functions[i])();
	}
	return results;*/

	/*
	* 
	* 784, 128, 128, 10
	* walking dot prod -> immediate compute of activation
	* 
	* w1 0 -> (784 * 128) 100,352
	* w2 (784 * 128) 100,352 -> (784 * 128) + (128 * 128) 116,736
	* 
	* 
	* 
	*/

}

void StandaloneNeuralNetwork::backward_propogate() {

}