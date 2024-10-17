#include <tuple>
#include <fstream>
#include <iostream>

#include "Matrix.h"

static class MNIST {
public:

	/// <summary>
	/// Returns a dataset such that each row is another entry in the MNIST dataset
	/// </summary>
	/// <returns></returns>
	static std::tuple<Matrix, Matrix, Matrix, Matrix> load_data(int fourier, int taylor, int chebyshev, int legendre, int laguarre, float lower_norm, float upper_norm) {
		// Train Data
		std::string trainingImages = "..\\Dependencies\\datasets\\mnist\\Training Data\\train-images.idx3-ubyte";
		std::string trainingLabels = "..\\Dependencies\\datasets\\mnist\\Training Data\\train-labels.idx1-ubyte";

		std::ifstream trainingFR = std::ifstream(trainingImages, std::ios::binary);
		std::ifstream trainingLabelsFR = std::ifstream(trainingLabels, std::ios::binary);

		// Discard
		int magicNum = read_big_int(&trainingLabelsFR);
		int imageNum = read_big_int(&trainingLabelsFR);
		magicNum = read_big_int(&trainingFR);

		// Read the important things
		imageNum = read_big_int(&trainingFR);
		int width = read_big_int(&trainingFR);
		int height = read_big_int(&trainingFR);

		Matrix data(imageNum, width * height);
		Matrix labels(imageNum, 1);

		for (int i = 0; i < imageNum; i++) {

			std::vector<uint8_t> byteData((width * height));
			trainingFR.read(reinterpret_cast<char*>(byteData.data()), byteData.size());
			std::vector<int> intData(byteData.begin(), byteData.end());

			data.SetRow(i, intData);

			char byte;
			trainingLabelsFR.read(&byte, 1);
			int label = static_cast<int>(static_cast<unsigned char>(byte));

			labels(i, 0) = label;
		}

		trainingFR.close();
		trainingLabelsFR.close();

		// Test Data
		std::string testingImages = "..\\Dependencies\\datasets\\mnist\\Testing Data\\t10k-images.idx3-ubyte";
		std::string testingLabels = "..\\Dependencies\\datasets\\mnist\\Testing Data\\t10k-labels.idx1-ubyte";

		std::ifstream testingFR = std::ifstream(testingImages, std::ios::binary);
		std::ifstream testingLabelFR = std::ifstream(testingLabels, std::ios::binary);

		if (testingFR.is_open() && testingLabelFR.is_open()) {
			std::cout << "loading training data\n";
		} else {
			std::cout << "file not found\n";
		}

		// Discard
		magicNum = read_big_int(&testingLabelFR);
		imageNum = read_big_int(&testingLabelFR);
		magicNum = read_big_int(&testingFR);

		// Read the important things
		imageNum = read_big_int(&testingFR);
		width = read_big_int(&testingFR);
		height = read_big_int(&testingFR);

		Matrix test_data(imageNum, width * height);
		Matrix test_labels(imageNum, 1);

		for (int i = 0; i < imageNum; i++) {

			std::vector<uint8_t> byteData((width * height));
			testingFR.read(reinterpret_cast<char*>(byteData.data()), byteData.size());
			std::vector<int> intData(byteData.begin(), byteData.end());

			test_data.SetRow(i, intData);

			char byte;
			testingLabelFR.read(&byte, 1);
			test_labels(i, 0) = static_cast<int>(static_cast<unsigned char>(byte));
		}
		testingFR.close();
		testingLabelFR.close();

		data = data.Transpose().extract_features(fourier, taylor, chebyshev, legendre,
			laguarre, lower_norm, upper_norm).Transpose();

		test_data = test_data.Transpose().extract_features(fourier, taylor, chebyshev, legendre,
			laguarre, lower_norm, upper_norm).Transpose();

		return std::make_tuple(data, labels, test_data, test_labels);
	}

	private:

	static int read_big_int(std::ifstream* fr) {

		int littleInt;
		fr->read(reinterpret_cast<char*>(&littleInt), sizeof(int));

		unsigned char* bytes = reinterpret_cast<unsigned char*>(&littleInt);
		std::swap(bytes[0], bytes[3]);
		std::swap(bytes[1], bytes[2]);

		return littleInt;
	}
	

};