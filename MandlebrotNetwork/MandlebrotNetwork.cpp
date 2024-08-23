#include <iostream>

#include "NeuralNetwork.h"
#include "Mandlebrot.cpp"

int main()
{
	// Model definitions
	std::vector<int> dims = { 2, 32, 1 };
	std::unordered_set<int> res = {  };
	std::unordered_set<int> batch_norm = {  };

	// Model compilation parameters
	NeuralNetwork::loss_metrics loss = NeuralNetwork::loss_metrics::mae;
	NeuralNetwork::loss_metrics metrics = NeuralNetwork::loss_metrics::mae;
	NeuralNetwork::optimization_technique optimizer = NeuralNetwork::optimization_technique::none;
	Matrix::init weight_init = Matrix::init::He;

	// Model fit information
	Matrix x;
	Matrix y;
	int batch_size = 500;
	int epochs = 20;
	float learning_rate = 0.1f;
	float validation_split = 0.05f;
	bool shuffle = true;
	int validation_freq = 5;

	// Feature engineering and dataset processing
	Mandlebrot mandlebrot;

	int fourier = 64;
	int taylor = 0;
	int chebyshev = 0;
	int legendre = 0;
	int laguarre = 0;

	float lower_norm = 0.0f;
	float upper_norm = 1.0f;

	std::tie(x, y) = mandlebrot.make_dataset(100000, 50, fourier, taylor, chebyshev, legendre, laguarre, lower_norm, upper_norm);
	dims[0] = x.RowCount;

	x = x.Transpose();

	std::cout << "X: " << x.Size();
	std::cout << "Y: " << y.Size();

	// Define the model
	NeuralNetwork model;

	model.Define(
		dims,
		res,
		batch_norm,
		&Matrix::_ELU,
		&Matrix::_ELUDerivative,
		&Matrix::Sigmoid
	);

	// Compile the model
	model.Compile(
		loss,
		metrics,
		optimizer,
		weight_init
	);

	// Fit model to training data
	model.Fit(
		x,
		y,
		batch_size,
		epochs,
		learning_rate,
		validation_split,
		shuffle,
		validation_freq
	);
}