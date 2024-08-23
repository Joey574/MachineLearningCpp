#include <iostream>
#include <atlimage.h>

#include "NeuralNetwork.h"
#include "Mandlebrot.cpp"

// Prototypes
void make_bmp(std::string filename, int width, int height, float confidence_threshold, NeuralNetwork model, Matrix image_data);

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

	int width = 160;
	int height = 90;

	// Temp dataset just to get dimensions
	std::tie(x, y) = mandlebrot.make_dataset(1, 1, fourier, taylor, chebyshev, legendre, laguarre, lower_norm, upper_norm);
	dims[0] = x.ColumnCount;

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

	for (int i = 0; i < 10; i++) {

		// Actual dataset, create new one each training session
		std::tie(x, y) = mandlebrot.make_dataset(200000, 250, fourier, taylor, chebyshev, legendre, laguarre, lower_norm, upper_norm);

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

		// Create bmp image of model predictions
		make_bmp("test_" + std::to_string(i).append(".bmp"), width, height, 0.95f, model, mandlebrot.create_image_features(width, height, fourier, taylor, chebyshev, legendre, laguarre, lower_norm, upper_norm));
	}
}

void make_bmp(std::string filename, int width, int height, float confidence_threshold, NeuralNetwork model, Matrix image_data) {

	// Convert std::string to wstring with black magic
	int wideStrLength = MultiByteToWideChar(CP_UTF8, 0, filename.c_str(), -1, nullptr, 0);
	std::wstring wideStr(wideStrLength, L'\0');
	MultiByteToWideChar(CP_UTF8, 0, filename.c_str(), -1, &wideStr[0], wideStrLength);

	CImage image;
	image.Create(width, height, 24);

	Matrix pixel_data = model.Predict(image_data);

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {

			float color = pixel_data(0, (y * width) + x) * 255.0f;
			float other = pixel_data(0, (y * width) + x) > confidence_threshold ? 255.0f : 0;

			image.SetPixel(x, y, RGB(color, other, other));
		}
	}

	image.Save(wideStr.c_str(), Gdiplus::ImageFormatBMP);
	image.Destroy();
}