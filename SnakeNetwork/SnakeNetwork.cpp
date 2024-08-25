#include <iostream>
#include <conio.h>
#include <Windows.h>

#include "NeuralNetwork.h"
#include "Snake.cpp"

int main()
{
	srand(time(0));

	int width = 20; int height = 10;
	Snake snake(width, height);

	// Model definitions
	std::vector<int> dims = { (width * height), 32, 32, 4 };
	std::unordered_set<int> res = {  };
	std::unordered_set<int> batch_norm = {  };

	// Model compilation parameters
	NeuralNetwork::loss_metrics loss = NeuralNetwork::loss_metrics::one_hot;
	NeuralNetwork::loss_metrics metrics = NeuralNetwork::loss_metrics::accuracy;
	NeuralNetwork::optimization_technique optimizer = NeuralNetwork::optimization_technique::none;
	Matrix::init weight_init = Matrix::init::He;

	// Model fit information
	Matrix x(0, width * height);
	Matrix y(0, 1);
	int batch_size = 50;
	int games_to_play = 10;
	int epochs = 50;
	float learning_rate = 0.1f;
	float validation_split = 0.0f;
	bool shuffle = true;
	int validation_freq = 1;

	int fourier = 0;
	int taylor = 8;
	int chebyshev = 0;
	int legendre = 0;
	int laguarre = 0;

	float lower_norm = 0.0f;
	float upper_norm = 1.0f;

	// Define the model
	NeuralNetwork model;

	model.Define(
		dims,
		res,
		batch_norm,
		&Matrix::_ELU,
		&Matrix::_ELUDerivative,
		&Matrix::SoftMax
	);

	// Compile the model
	model.Compile(
		loss,
		metrics,
		optimizer,
		weight_init
	);

	const std::chrono::milliseconds frameDuration(1000 / 5);

	for (int i = 0; i < games_to_play; i++) {

		while (true) {
			auto start = std::chrono::high_resolution_clock::now();

			Matrix state(1, width * height);
			std::memcpy(state.matrix, snake.gameboard.matrix, width * height * sizeof(float));

			Matrix prediction = model.Predict(state);
			int max_idx = std::distance(prediction.matrix, std::max_element(prediction.matrix, prediction.matrix + 4)) + 1;

			x.add_row(snake.gameboard.matrix);
			y.add_row(std::vector<int> {max_idx});

			if (!snake.move_snake(max_idx)) {
				snake.reset(width, height);
				break;
			}

			snake.draw_gameboard();

			auto elapsed = std::chrono::high_resolution_clock::now() - start;
			auto sleepTime = frameDuration - elapsed;

			if (sleepTime > std::chrono::milliseconds(0)) {
				std::this_thread::sleep_for(sleepTime);
			}
		}

	}

	

}