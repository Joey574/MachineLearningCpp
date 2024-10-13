#include <iostream>
#include <Windows.h>

#include "SingleBlockNeuralNetwork.h"
#include "MNIST.cpp"

std::string clean_time(double time) {
	const double hour = 3600000.00;
	const double minute = 60000.00;
	const double second = 1000.00;
	std::string out;

	if (time / hour > 1.00) {
		out = std::to_string(time / hour).append(" hours");
	}
	else if (time / minute > 1.00) {
		out = std::to_string(time / minute).append(" minutes");
	}
	else if (time / second > 1.00) {
		out = std::to_string(time / second).append(" seconds");
	}
	else {
		out = std::to_string(time).append("(ms)");
	}
	return out;
}


int main()
{
	system("CLS");
	SetPriorityClass(GetStdHandle, REALTIME_PRIORITY_CLASS);
	srand(time(0));

	// Model definitions
	std::vector<int> dims = { 784, 128, 128, 10 };
	std::vector<NeuralNetwork::activation_functions> act = {
		NeuralNetwork::activation_functions::leaky_relu,
		NeuralNetwork::activation_functions::leaky_relu,
		NeuralNetwork::activation_functions::sigmoid
	};

	// Model fit information
	Matrix x;
	Matrix y;
	Matrix x_test;
	Matrix y_test;
	int batch_size = 320;
	int epochs = 15;
	float learning_rate = 0.05f;
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

	model.define(dims, act);
	model.compile(NeuralNetwork::loss_metric::one_hot, NeuralNetwork::loss_metric::accuracy, NeuralNetwork::weight_init::he);

	NeuralNetwork::history h = model.fit(
		x,
		y,
		x_test,
		y_test,
		batch_size,
		epochs,
		learning_rate,
		shuffle,
		validation_freq,
		0.0f
	);

	std::cout << "\nTotal training time: " << clean_time(h.train_time.count()) << "\n";
	std::cout << "Average epoch time: " << clean_time(h.epoch_time.count()) << "\n";

	double min_score = *std::min_element(h.metric_history.begin(), h.metric_history.end());
	double max_score = *std::max_element(h.metric_history.begin(), h.metric_history.end());

	std::cout << "\nMin score: " << min_score << "\nMax score: " << max_score << "\n";

}