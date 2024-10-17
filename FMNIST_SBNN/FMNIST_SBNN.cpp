#include <iostream>
#include <Windows.h>

#include "Matrix.h"
#include "SingleBlockNeuralNetwork.h"
#include "FMNIST.cpp"

int main()
{
	system("CLS");
	SetPriorityClass(GetStdHandle, REALTIME_PRIORITY_CLASS);
	srand(time(0));

	// Model definitions
	std::vector<size_t> dims = { 784, 256, 256, 10 };
	std::vector<NeuralNetwork::activation_functions> act = {
		NeuralNetwork::activation_functions::leaky_relu,
		NeuralNetwork::activation_functions::leaky_relu,
		NeuralNetwork::activation_functions::sigmoid
	};

	// model information
	Matrix x_train;
	Matrix y_train;
	Matrix x_test;
	Matrix y_test;
	size_t batch_size = 320;
	size_t epochs = 50;
	float learning_rate = 0.01f;
	bool shuffle = true;
	int validation_freq = 1;

	// data information
	float lower_norm = 0.0f;
	float upper_norm = 1.0f;

	int fourier = 0;
	int taylor = 0;
	int chebyshev = 0;
	int legendre = 0;
	int laguarre = 0;

	std::tie(x_train, y_train, x_test, y_test) = FMNIST::load_data(fourier, taylor, chebyshev, legendre, laguarre, lower_norm, upper_norm);

	NeuralNetwork model;

	model.define(dims, act);
	model.compile(NeuralNetwork::loss_metric::one_hot, NeuralNetwork::loss_metric::accuracy, NeuralNetwork::weight_init::he);

	model.fit(
		x_train,
		y_train,
		x_test,
		y_test,
		batch_size,
		epochs,
		learning_rate,
		shuffle,
		validation_freq,
		0.0f
	);
}