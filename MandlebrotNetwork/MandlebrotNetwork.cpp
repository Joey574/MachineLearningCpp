#include <iostream>
#include <atlimage.h>

#include "NeuralNetwork.h"
#include "Mandlebrot.cpp"

// Prototypes
void make_bmp(std::string filename, int width, int height, float confidence_threshold, NeuralNetwork model, Matrix image_data);

int main()
{
	srand(time(0));

	// Model definitions
	std::vector<int> dims = { 2, 128, 128, 128, 1 };
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
	int epochs = 5;
	float learning_rate = 1.0f;
	float validation_split = 0.1f;
	bool shuffle = true;
	int validation_freq = 1;

	// Feature engineering and dataset processing
	Mandlebrot mandlebrot;

	int fourier = 86;
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
		{ &Matrix::_ELU, &Matrix::_ELU, &Matrix::_ELU, &Matrix::Sigmoid},
		{ &Matrix::_ELUDerivative, &Matrix::_ELUDerivative, &Matrix::_ELUDerivative }
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
			(Matrix()),
			(Matrix()),
			batch_size,
			epochs,
			learning_rate,
			validation_split,
			shuffle,
			validation_freq
		);

		// Create bmp image of model predictions
		make_bmp("NetworkImages/test_" + std::to_string(i).append(".bmp"), width, height, 0.95f, model, mandlebrot.create_image_features(width, height, fourier, taylor, chebyshev, legendre, laguarre, lower_norm, upper_norm));
	}

	int f_width = 800;
	int f_height = 450;

	make_bmp("NetworkImages/final.bmp", f_width, f_height, 0.95f, model, mandlebrot.create_image_features(f_width, f_height, fourier, taylor, chebyshev, legendre, laguarre, lower_norm, upper_norm));
}

void make_bmp(std::string filename, int width, int height, float confidence_threshold, NeuralNetwork model, Matrix image_data) {

	auto start = std::chrono::high_resolution_clock::now();

	// Convert std::string to wstring with black magic
	int wideStrLength = MultiByteToWideChar(CP_UTF8, 0, filename.c_str(), -1, nullptr, 0);
	std::wstring wideStr(wideStrLength, L'\0');
	MultiByteToWideChar(CP_UTF8, 0, filename.c_str(), -1, &wideStr[0], wideStrLength);

	CImage image;
	image.Create(width, height, 24);

	for (int y = 0; y < height; y++) {
		Matrix pixel_data = model.Predict(image_data.SegmentR((y * width), (y * width) + width));

		for (int x = 0; x < width; x++) {

			std::vector<float> color = Mandlebrot::gradient(x, y, pixel_data(0, x), confidence_threshold, Mandlebrot::gradient_type::diagonal);

			image.SetPixel(x, y, RGB(color[0], color[1], color[2]));
		}
	}
	
	image.Save(wideStr.c_str(), Gdiplus::ImageFormatBMP);
	image.Destroy();

	std::chrono::duration<double, std::milli> time = std::chrono::high_resolution_clock::now() - start;

	std::cout << "image_made: " << (time.count() / 1000.0) << " seconds\n";
}