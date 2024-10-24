#include <iostream>
#include <Windows.h>

#include "SingleBlockNeuralNetwork.h"

// datasets
#include "MNIST.cpp"
#include "FMNIST.cpp"
#include "Mandlebrot.cpp"

void menu(NeuralNetwork& model);
void load_network(NeuralNetwork& model);
void initialize_network(NeuralNetwork& model);

bool is_num(const std::string& str) {
	for (size_t i = 0; i < str.size(); i++) {
		if (str[i] < '0' || str[i] > '9') {
			return false;
		}
	}
	return true;
}
size_t to_num(const std::string& str) {
	size_t dims = 0;
	for (size_t i = 0; i < str.size(); i++) {
		dims += (str[i] - '0') * powf(10, str.size() - i - 1);
	}
	return dims;
}

int main()
{
	system("CLS");
	SetPriorityClass(GetStdHandle, REALTIME_PRIORITY_CLASS);
	srand(time(0));

	NeuralNetwork model;

	menu(model);
}

void menu(NeuralNetwork& model) {
	std::string input;

	L_MENU:
	system("CLS");
	std::cout << "Select an option:\n1: Load existing network\n2: Create new network\nInput: ";
	std::cin >> input;

	switch (input[0]) {
	case '1':
		load_network(model);
		break;
	case '2':
		initialize_network(model);
		break;
	default:
		goto L_MENU;
	}
}

void load_network(NeuralNetwork& model) {
	std::string input;


}
void initialize_network(NeuralNetwork& model) {
	std::string input;

	L1:
	system("CLS");
	std::cout << "Number of hidden layers?\nInput: ";
	std::cin >> input;

	if (!is_num(input)) {
		goto L1;
	}
	
	size_t dims = to_num(input);
	std::vector<size_t> dimensions(dims + 2, 0);

	for (size_t i = 0; i < dims; i++) {
		L2:

		system("CLS");
		std::cout << "Number of nodes in layer " + std::to_string(i + 1).append("?\nInput: ");
		std::cin >> input;

		if (!is_num(input)) {
			goto L2;
		}

		dimensions[i + 1] = to_num(input);
	}

	NeuralNetwork::loss_metric loss;
	NeuralNetwork::loss_metric metric;

	NeuralNetwork::weight_init weight;
	std::vector<NeuralNetwork::activation_functions> activations;


}