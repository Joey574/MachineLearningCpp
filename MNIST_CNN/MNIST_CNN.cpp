#include <iostream>
#include <Windows.h>

#include "../SingleBlockCudaNetwork/SingleBlockCudaNetwork.h"
#include "Matrix.h"
#include "../MNIST/MNIST.cpp"

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
	std::vector<size_t> dims = { 784, 512, 512, 512, 10 };

	// Model fit information
	Matrix x;
	Matrix y;
	Matrix x_t;
	Matrix y_t;
	size_t batch_size = 320;
	size_t epochs = 20;
	float learning_rate = 0.01f;
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

	std::tie(x, y, x_t, y_t) = mnist.load_data(fourier, taylor, chebyshev, legendre, laguarre, lower_norm, upper_norm);
	dims[0] = x.ColumnCount;

	// construct data as CudaNet matrices
	CudaNetwork::matrix x_train; x_train.rows = x.RowCount; x_train.cols = x.ColumnCount; x_train.mat = std::vector<float>(x.matrix, x.matrix + (x.RowCount * x.ColumnCount));
	CudaNetwork::matrix y_train; y_train.rows = y.RowCount; y_train.cols = y.ColumnCount; y_train.mat = std::vector<float>(y.matrix, y.matrix + (y.RowCount * y.ColumnCount));

	CudaNetwork::matrix x_test; x_test.rows = x_t.RowCount; x_test.cols = x_t.ColumnCount; x_test.mat = std::vector<float>(x_t.matrix, x_t.matrix + (x_t.RowCount * x_t.ColumnCount));
	CudaNetwork::matrix y_test; y_test.rows = y_t.RowCount; y_test.cols = y_t.ColumnCount; y_test.mat = std::vector<float>(y_t.matrix, y_t.matrix + (y_t.RowCount * y_t.ColumnCount));

	CudaNetwork model;

	model.define(dims);
	model.compile(CudaNetwork::weight_init::he);

	model.fit(
		x_train,
		y_train,
		x_test,
		y_test,
		batch_size,
		epochs,
		learning_rate,
		shuffle,
		validation_freq
	);

	/*std::cout << "\nTotal training time: " << clean_time(h.train_time.count()) << "\n";
	std::cout << "Average epoch time: " << clean_time(h.epoch_time.count()) << "\n";

	double min_score = *std::min_element(h.metric_history.begin(), h.metric_history.end());
	double max_score = *std::max_element(h.metric_history.begin(), h.metric_history.end());

	std::cout << "\nMin score: " << min_score << "\nMax score: " << max_score << "\n";*/

}