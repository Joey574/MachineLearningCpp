#include <iostream>
#include <Windows.h>

#include "SingleBlockNeuralNetwork.h"

// datasets
#include "MNIST.cpp"
#include "FMNIST.cpp"
#include "Mandlebrot.cpp"

void menu();
void load_network();
void initialize_network(network_init& init);

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

struct network_init {
	std::vector<size_t> dims;
	std::vector<NeuralNetwork::activation_functions> activation;
	NeuralNetwork::loss_metric loss;
	NeuralNetwork::loss_metric metric;
};

int main()
{
	system("CLS");
	SetPriorityClass(GetStdHandle, REALTIME_PRIORITY_CLASS);
	srand(time(0));


	menu();
}

void menu(NeuralNetwork& model) {
	std::string input;

	network_init init;

	L_MENU:
	system("CLS");
	std::cout << "Select an option:\n1: Load existing network\n2: Create new network\nInput: ";
	std::cin >> input;

	switch (input[0]) {
	case '1':
		load_network(model);
		break;
	case '2':
		initialize_network(init);
		break;
	default:
		goto L_MENU;
	}
}

void load_network(NeuralNetwork& model) {
	std::string input;


}
void initialize_network(network_init& init) {
	std::string input;

	NeuralNetwork::loss_metric loss;
	NeuralNetwork::loss_metric metric;
	NeuralNetwork::weight_init weight_init;

	L1:
	system("CLS");
	std::cout << "Number of hidden layers?\nInput: ";
	std::cin >> input;

	if (!is_num(input)) {
		goto L1;
	}
	
	size_t dims = to_num(input);

	std::vector<size_t> dimensions(dims + 2, 0);
	std::vector<NeuralNetwork::activation_functions> activations(dims + 1);

	
	// nodes in hidden layer
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

	// hidden layer activation
	for (size_t i = 0; i < dims; i++) {

		L3:
		system("CLS");
		std::cout << "Activation function for layer " + std::to_string(i + 1).append("?\n1: relu\n2: leaky_relu\n3: elu\n4: sigmoid\nInput: ");
		std::cin >> input;

		if (!is_num(input)) {
			goto L3;
		}

		switch (to_num(input)) {
		case 1:
			activations[i] = NeuralNetwork::activation_functions::relu;
			break;
		case 2:
			activations[i] = NeuralNetwork::activation_functions::leaky_relu;
			break;
		case 3:
			activations[i] = NeuralNetwork::activation_functions::elu;
			break;
		case 4:
			activations[i] = NeuralNetwork::activation_functions::sigmoid;
			break;
		default:
			goto L3;
		}
	}

	L4:
	system("CLS");
	std::cout << "Activation function for output layer?\n1: relu\n2: leaky_relu\n3: elu\n4: sigmoid\nInput: ";
	std::cin >> input;

	if (!is_num(input) || to_num(input) > 4) {
		goto L4;
	}

	switch (to_num(input)) {
	case 1:
		activations[dims] = NeuralNetwork::activation_functions::relu;
		break;
	case 2:
		activations[dims] = NeuralNetwork::activation_functions::leaky_relu;
		break;
	case 3:
		activations[dims] = NeuralNetwork::activation_functions::elu;
		break;
	case 4:
		activations[dims] = NeuralNetwork::activation_functions::sigmoid;
		break;
	default:
		goto L4;
	}


	L5:
	system("CLS");
	std::cout << "Loss function?\n1: mae\n2: one_hot\nInput: ";
	std::cin >> input;

	if (!is_num(input)) {
		goto L5;
	}

	switch (to_num(input)) {
	case 1:
		loss = NeuralNetwork::loss_metric::mae;
		break;
	case 2:
		loss = NeuralNetwork::loss_metric::one_hot;
		break;
	default:
		goto L5;
	}


	L6:
	system("CLS");
	std::cout << "Score function?\n1: mae\n2: accuracy\nInput: ";
	std::cin >> input;

	if (!is_num(input)) {
		goto L6;
	}

	switch (to_num(input)) {
	case 1:
		metric = NeuralNetwork::loss_metric::mae;
		break;
	case 2:
		metric = NeuralNetwork::loss_metric::accuracy;
		break;
	default:
		goto L6;
	}


	L7:
	system("CLS");
	std::cout << "Weight initialization\n1: He\n2: Normalize\n3: Xavier\nInput: ";
	std::cin >> input;

	if (!is_num(input)) {
		goto L7;
	}

	switch (to_num(input)) {
	case 1:
		weight_init = NeuralNetwork::weight_init::he;
		break;
	case 2:
		weight_init = NeuralNetwork::weight_init::normalize;
		break;
	case 3:
		weight_init = NeuralNetwork::weight_init::xavier;
		break;
	default:
		goto L7;
	}
	system("CLS");

	init.dims = dimensions;
	init.activation = activations;
	init.loss = loss;
	init.metric = metric;
}