#include <iostream>
#include <Windows.h>

#include "NeuralNetwork.h"
#include "MNIST.cpp"

int main()
{
	system("CLS");
	SetPriorityClass(GetStdHandle, REALTIME_PRIORITY_CLASS);
	srand(time(0));

	// Model definitions
	std::vector<int> dims = { 784, 128, 128, 128, 10 };

	// Model compilation parameters
	NeuralNetwork::loss_metric loss = NeuralNetwork::loss_metric::cross_entropy;
	NeuralNetwork::loss_metric metrics = NeuralNetwork::loss_metric::accuracy;
	NeuralNetwork::optimization_technique optimizer = NeuralNetwork::optimization_technique::none;
	Matrix::init weight_init = Matrix::init::He;

	// Model fit information
	Matrix x;
	Matrix y;
	Matrix x_test;
	Matrix y_test;
	int batch_size = 320;
	int epochs = 20;
	float learning_rate = 0.1f;
	float weight_decay = 0.0f;
	float validation_split = 0.0f;
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

	// Define the model
	NeuralNetwork model;

	model.Define(
		dims,
		{ NeuralNetwork::activations::leakyReLU, NeuralNetwork::activations::leakyReLU, NeuralNetwork::activations::leakyReLU, NeuralNetwork::activations::Softmax }
	);

	// Compile the model
	model.Compile(
		loss,
		metrics,
		optimizer,
		weight_init
	);

	// Fit model to training data
	auto history = model.Fit(
		x,
		y,
		x_test,
		y_test,
		batch_size,
		epochs,
		learning_rate,
		weight_decay,
		validation_split,
		shuffle,
		validation_freq
	);

	std::cout << "Training time: " << history.train_time.count() << "\n";
	std::cout << "Epoch time: " << history.epoch_time.count() << "\n";

	std::cout << "train_data: " << model.Evaluate(x, y) << "\n";
	std::cout << "test_data: " << model.Evaluate(x_test, y_test) << "\n";

	//model.save("Networks/784_4.txt");
}