#include <iostream>
#include <Windows.h>

#include "SingleBlockNeuralNetwork.h"
#include "Mandlebrot.cpp"

void make_bmp(std::string filename, int width, int height, float confidence_threshold, NeuralNetwork& model, const Matrix& image_data) {

	auto start = std::chrono::high_resolution_clock::now();

	// Convert std::string to wstring with black magic
	int wideStrLength = MultiByteToWideChar(CP_UTF8, 0, filename.c_str(), -1, nullptr, 0);
	std::wstring wideStr(wideStrLength, L'\0');
	MultiByteToWideChar(CP_UTF8, 0, filename.c_str(), -1, &wideStr[0], wideStrLength);

	CImage image;
	image.Create(width, height, 24);

	for (int y = 0; y < height; y++) {

		std::vector<float> pixel_data = model.predict(image_data.SegmentR((y * width), (y * width) + width));

		Mandlebrot::fill_gradient(y, pixel_data, (BYTE*)image.GetPixelAddress(0, y), confidence_threshold, Mandlebrot::gradient_type::red_secondary);
	}

	image.Save(wideStr.c_str(), Gdiplus::ImageFormatBMP);
	image.Destroy();

	std::chrono::duration<double, std::milli> time = std::chrono::high_resolution_clock::now() - start;

	std::cout << "image_made: " << (time.count() / 1000.0) << " seconds\n";
}

int main()
{
	system("CLS");
	SetPriorityClass(GetStdHandle, REALTIME_PRIORITY_CLASS);
	srand(time(0));

	// Model definitions
	std::vector<size_t> dims = { 2, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1 };
	std::vector<NeuralNetwork::activation_functions> act = {
		NeuralNetwork::activation_functions::leaky_relu,
		NeuralNetwork::activation_functions::leaky_relu,
		NeuralNetwork::activation_functions::leaky_relu,
		NeuralNetwork::activation_functions::leaky_relu,
		NeuralNetwork::activation_functions::leaky_relu,
		NeuralNetwork::activation_functions::leaky_relu,
		NeuralNetwork::activation_functions::leaky_relu,
		NeuralNetwork::activation_functions::leaky_relu,
		NeuralNetwork::activation_functions::leaky_relu,
		NeuralNetwork::activation_functions::sigmoid
	};

	// Model fit information
	Matrix x;
	Matrix y;
	size_t batch_size = 640;
	size_t epochs = 5;
	float learning_rate = 0.0005f;
	bool shuffle = true;
	int validation_freq = 1;
	float validation_split = 0.1f;

	// Feature engineering and dataset processing
	Mandlebrot mandlebrot;

	int fourier = 128;
	int taylor = 0;
	int chebyshev = 0;
	int legendre = 0;
	int laguarre = 0;

	float lower_norm = 0.0f;
	float upper_norm = 1.0f;

	// sample dataset
	std::tie(x, y) = mandlebrot.make_dataset(1, 1, fourier, taylor, chebyshev, legendre, laguarre, lower_norm, upper_norm);
	dims[0] = x.ColumnCount;

	NeuralNetwork model;

	model.define(dims, act);
	model.deserialize("128_1024_9_network.txt");
	model.compile(NeuralNetwork::loss_metric::mae, NeuralNetwork::loss_metric::mae, NeuralNetwork::weight_init::he);

	/* 16:9 resolutions
	
	160 x 90
	320 x 180
	960 x 540
	1280 x 720
	1920 x 1080
	
	*/

	int width = 960;
	int height = 540;

	const Matrix image_features = mandlebrot.create_image_features(width, height, fourier, taylor, chebyshev, legendre, laguarre, lower_norm, upper_norm);


	for (size_t i = 0; i < 500; i++) {
		std::tie(x, y) = mandlebrot.make_dataset(100000, 500, fourier, taylor, chebyshev, legendre, laguarre, lower_norm, upper_norm);

		model.fit(
			x,
			y,
			*(new Matrix()),
			*(new Matrix()),
			batch_size,
			epochs,
			learning_rate,
			shuffle,
			validation_freq,
			validation_split
		);

		make_bmp("NetworkImages/4_image_" + std::to_string(i).append(".bmp"), width, height, 0.95f, model, image_features);

		model.serialize("128_1024_9_network.txt");
	}
	

	int f_width = 1920;
	int f_height = 1080;

	//make_bmp("NetworkImages/image_final.bmp", f_width, f_height, 0.95f, model, mandlebrot.create_image_features(f_width, f_height, fourier, taylor, chebyshev, legendre, laguarre, lower_norm, upper_norm));


}