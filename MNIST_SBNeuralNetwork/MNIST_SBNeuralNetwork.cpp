#include <iostream>
#include <Windows.h>

#include "SingleBlockNeuralNetwork.h"
#include "MNIST.cpp"

int main()
{
	system("CLS");
	SetPriorityClass(GetStdHandle, REALTIME_PRIORITY_CLASS);
	srand(time(0));

	// Model definitions
	std::vector<int> dims = { 784, 128, 128, 10 };

	// Model fit information
	Matrix x;
	Matrix y;
	Matrix x_test;
	Matrix y_test;
	int batch_size = 320;
	int epochs = 50;
	float learning_rate = 0.1f;
	float weight_decay = 0.01f;
	bool shuffle = true;
	int validation_freq = 1;

	// Feature engineering and dataset processing
	MNIST mnist;

	int fourier = 0;
	int taylor = 0;
	int chebyshev = 0;
	int legendre = 0;
	int laguarre = 0;

	float lower_norm = 0.0f;
	float upper_norm = 1.0f;

	std::tie(x, y, x_test, y_test) = mnist.load_data(fourier, taylor, chebyshev, legendre, laguarre, lower_norm, upper_norm);
	dims[0] = x.ColumnCount;

	NeuralNetwork model;

	model.define(dims);

	model.compile(NeuralNetwork::loss_metric::mse, NeuralNetwork::loss_metric::mse, NeuralNetwork::weight_init::he);

	model.fit(
		x,
		y,
		x_test,
		y_test,
		batch_size,
		epochs,
		learning_rate,
		shuffle,
		validation_freq
	);
}